from __future__ import annotations

import cc3d
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def sequential_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    unique = unique[unique != 0]
    if unique.size == 0:
        return np.zeros(labels.shape, dtype=np.uint32)
    if unique[-1] <= 10_000_000:
        lookup = np.zeros(int(unique[-1]) + 1, dtype=np.uint32)
        lookup[unique] = np.arange(1, unique.size + 1, dtype=np.uint32)
        return lookup[labels]
    output = np.zeros(labels.shape, dtype=np.uint32)
    for new_id, old_id in enumerate(unique, start=1):
        output[labels == old_id] = new_id
    return output


def connected_components(foreground: np.ndarray) -> np.ndarray:
    return cc3d.connected_components(
        np.asarray(foreground, dtype=np.uint8),
        connectivity=26,
        out_dtype=np.uint32,
    )


def reconstruct_instances(
    fused_semantic: np.ndarray,
    primary_instances: np.ndarray,
    spacing_nm_zyx: tuple[float, float, float],
) -> np.ndarray:
    """Use primary panoptic seeds inside a fused semantic support mask.

    Added components without a surviving primary seed receive one physical-
    distance maximum. Watershed distances are computed in nanometres so that
    anisotropic ssSEM crops do not treat section depth as an in-plane pixel.
    """

    foreground = np.asarray(fused_semantic, dtype=bool)
    primary = np.asarray(primary_instances)
    if foreground.shape != primary.shape:
        raise ValueError(
            f"semantic/instance geometry mismatch: {foreground.shape} vs {primary.shape}"
        )
    markers = sequential_labels(np.where(foreground, primary, 0))
    components = connected_components(foreground)
    n_components = int(components.max(initial=0))
    marker_components = np.unique(components[markers > 0])
    marker_components = set(int(value) for value in marker_components if value != 0)
    next_id = int(markers.max(initial=0)) + 1
    spacing = tuple(float(value) for value in spacing_nm_zyx)
    for component_id in range(1, n_components + 1):
        if component_id in marker_components:
            continue
        component = components == component_id
        if not component.any():
            continue
        distance = ndi.distance_transform_edt(component, sampling=spacing)
        seed_index = np.unravel_index(int(np.argmax(distance)), distance.shape)
        markers[seed_index] = next_id
        next_id += 1
    if not np.any(markers):
        return np.zeros(foreground.shape, dtype=np.uint32)
    distance = ndi.distance_transform_edt(foreground, sampling=spacing)
    return sequential_labels(watershed(-distance, markers=markers, mask=foreground))
