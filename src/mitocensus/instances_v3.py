"""Post-Stage-C2 conservative panoptic rescue.

This module is a new, explicitly selection-exposed development iteration.  It
does not modify the frozen Stage-C or Stage-C2 implementations.  The rescue is
label-free at inference: it inspects only candidate semantic support and the
two specialist outputs.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .instances import connected_components, reconstruct_instances, sequential_labels


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
        or mask[:, :, 0].any()
        or mask[:, :, -1].any()
    )


def eligible_labels(
    labels: np.ndarray,
    *,
    minimum_voxels: int = 500,
    exclude_border: bool = True,
) -> np.ndarray:
    """Return sequential labels that satisfy the public evaluation geometry rule."""

    source = np.asarray(labels)
    output = np.zeros(source.shape, dtype=np.uint32)
    next_id = 1
    for label_id in np.unique(source):
        if label_id == 0:
            continue
        component = source == label_id
        if int(component.sum()) < int(minimum_voxels):
            continue
        if exclude_border and _touches_border(component):
            continue
        output[component] = next_id
        next_id += 1
    return output


def _ellipsoid(radius_nm: float, spacing_nm_zyx: tuple[float, float, float]) -> np.ndarray:
    radius = float(radius_nm)
    if radius <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    spacing = np.asarray(spacing_nm_zyx, dtype=float)
    half = np.maximum(1, np.ceil(radius / spacing).astype(int))
    grids = np.ogrid[tuple(slice(-value, value + 1) for value in half)]
    squared = sum((grid * step / radius) ** 2 for grid, step in zip(grids, spacing, strict=True))
    return np.asarray(squared <= 1.0, dtype=bool)


def _extend_with_nearby_primary(
    candidate: np.ndarray,
    mitonet: np.ndarray,
    spacing_nm_zyx: tuple[float, float, float],
    maximum_distance_nm: float,
) -> np.ndarray:
    if maximum_distance_nm <= 0 or not candidate.any():
        return candidate.copy()
    distance = ndi.distance_transform_edt(~candidate, sampling=spacing_nm_zyx)
    return candidate | (mitonet & (distance <= float(maximum_distance_nm)))


def _exact_omniem_rescue(
    candidate: np.ndarray,
    omniem: np.ndarray,
    *,
    minimum_voxels: int,
    minimum_candidate_overlap_fraction: float,
    maximum_objects: int,
) -> np.ndarray:
    components = eligible_labels(
        connected_components(omniem),
        minimum_voxels=minimum_voxels,
        exclude_border=True,
    )
    accepted: list[tuple[int, int]] = []
    for component_id in range(1, int(components.max(initial=0)) + 1):
        component = components == component_id
        size = int(component.sum())
        overlap = float(np.logical_and(component, candidate).sum() / max(1, size))
        if overlap >= float(minimum_candidate_overlap_fraction):
            accepted.append((size, component_id))
    accepted.sort(reverse=True)
    output = np.zeros(candidate.shape, dtype=np.uint32)
    for new_id, (_, component_id) in enumerate(accepted[: int(maximum_objects)], start=1):
        output[components == component_id] = new_id
    return output


def reconstruct_with_policy_v3(
    candidate_semantic: np.ndarray,
    primary_instances: np.ndarray,
    mitonet_semantic: np.ndarray,
    omniem_semantic: np.ndarray,
    spacing_nm_zyx: tuple[float, float, float],
    policy: dict,
) -> np.ndarray:
    """Build instances with a zero-primary rescue and bounded shape extension."""

    candidate = np.asarray(candidate_semantic, dtype=bool)
    primary = np.asarray(primary_instances)
    mitonet = np.asarray(mitonet_semantic, dtype=bool)
    omniem = np.asarray(omniem_semantic, dtype=bool)
    if not (candidate.shape == primary.shape == mitonet.shape == omniem.shape):
        raise ValueError("Stage-C3 input geometry mismatch")
    spacing = tuple(float(value) for value in spacing_nm_zyx)
    minimum_voxels = int(policy.get("minimum_native_voxels", 500))
    primary_eligible = eligible_labels(
        primary,
        minimum_voxels=minimum_voxels,
        exclude_border=True,
    )
    if int(primary_eligible.max(initial=0)) == 0:
        return _exact_omniem_rescue(
            candidate,
            omniem,
            minimum_voxels=minimum_voxels,
            minimum_candidate_overlap_fraction=float(
                policy["zero_primary_minimum_candidate_overlap_fraction"]
            ),
            maximum_objects=int(policy.get("zero_primary_maximum_rescue_objects", 4)),
        )

    support = _extend_with_nearby_primary(
        candidate,
        mitonet,
        spacing,
        float(policy["primary_extension_nm"]),
    )
    closing_radius = float(policy.get("closing_radius_nm", 0.0))
    if closing_radius > 0:
        support = ndi.binary_closing(
            support,
            structure=_ellipsoid(closing_radius, spacing),
            iterations=1,
        )
    return sequential_labels(reconstruct_instances(support, primary, spacing))

