"""Selection-exposed Stage-C2 instance reconstruction policies.

This module is intentionally separate from ``instances.py``, which belongs to
the frozen Stage-C whole-volume execution.  Stage-C2 was triggered after the
first completed volume exposed count inflation; it never rewrites Stage-C.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .instances import connected_components, sequential_labels


def _component_overlap_fraction(component: np.ndarray, mask: np.ndarray) -> float:
    denominator = int(component.sum())
    return 0.0 if denominator == 0 else float(np.logical_and(component, mask).sum() / denominator)


def reconstruct_with_policy(
    candidate_semantic: np.ndarray,
    primary_instances: np.ndarray,
    mitonet_semantic: np.ndarray,
    omniem_semantic: np.ndarray,
    spacing_nm_zyx: tuple[float, float, float],
    policy: dict,
) -> np.ndarray:
    candidate = np.asarray(candidate_semantic, dtype=bool)
    primary = np.asarray(primary_instances)
    mitonet = np.asarray(mitonet_semantic, dtype=bool)
    omniem = np.asarray(omniem_semantic, dtype=bool)
    if not (candidate.shape == primary.shape == mitonet.shape == omniem.shape):
        raise ValueError("Stage-C2 input geometry mismatch")

    support = candidate.copy()
    radius = policy.get("maximum_distance_to_mitonet_nm")
    if radius is not None:
        distance = ndi.distance_transform_edt(~mitonet, sampling=spacing_nm_zyx)
        support &= distance <= float(radius)

    components = connected_components(support)
    anchored_ids = set(int(value) for value in np.unique(components[primary > 0]) if value != 0)
    keep = np.zeros(support.shape, dtype=bool)
    accepted_new_ids: list[int] = []
    for component_id in range(1, int(components.max(initial=0)) + 1):
        component = components == component_id
        if component_id in anchored_ids:
            keep |= component
            continue
        size = int(component.sum())
        rule = policy["new_component_rule"]
        accept = False
        if rule == "all":
            accept = size >= int(policy.get("minimum_new_component_voxels", 1))
        elif rule == "none":
            accept = False
        elif rule == "omniem_consensus":
            accept = (
                size >= int(policy["minimum_new_component_voxels"])
                and _component_overlap_fraction(component, omniem) >= float(policy["minimum_omniem_overlap_fraction"])
            )
        elif rule == "large_only":
            accept = size >= int(policy["minimum_new_component_voxels"])
        else:
            raise ValueError(f"unknown new-component rule: {rule}")
        if accept:
            keep |= component
            accepted_new_ids.append(component_id)

    markers = sequential_labels(np.where(keep, primary, 0))
    next_id = int(markers.max(initial=0)) + 1
    for component_id in accepted_new_ids:
        component = components == component_id
        distance = ndi.distance_transform_edt(component, sampling=spacing_nm_zyx)
        markers[np.unravel_index(int(np.argmax(distance)), distance.shape)] = next_id
        next_id += 1
    if not markers.any():
        return np.zeros(keep.shape, dtype=np.uint32)
    distance = ndi.distance_transform_edt(keep, sampling=spacing_nm_zyx)
    return sequential_labels(watershed(-distance, markers=markers, mask=keep))
