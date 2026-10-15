"""Adaptive contraction guard for post-Stage-C3 panoptic development."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .instances_v3 import eligible_labels, reconstruct_with_policy_v3


def reconstruct_with_policy_v4(
    candidate_semantic: np.ndarray,
    primary_instances: np.ndarray,
    mitonet_semantic: np.ndarray,
    omniem_semantic: np.ndarray,
    spacing_nm_zyx: tuple[float, float, float],
    policy: dict,
) -> np.ndarray:
    """Guard against global candidate contraction without a domain identifier.

    Zero-primary cases retain the Stage-C3 exact-OmniEM rescue.  Otherwise, a
    physically bounded amount of trusted primary support is restored only when
    the candidate foreground fraction contracts beyond the frozen ratio.
    """

    candidate = np.asarray(candidate_semantic, dtype=bool)
    primary = np.asarray(primary_instances)
    mitonet = np.asarray(mitonet_semantic, dtype=bool)
    omniem = np.asarray(omniem_semantic, dtype=bool)
    if not (candidate.shape == primary.shape == mitonet.shape == omniem.shape):
        raise ValueError("Stage-C4 input geometry mismatch")
    minimum_voxels = int(policy.get("minimum_native_voxels", 500))
    primary_eligible = eligible_labels(primary, minimum_voxels=minimum_voxels, exclude_border=True)
    base_policy = {
        "minimum_native_voxels": minimum_voxels,
        "zero_primary_minimum_candidate_overlap_fraction": float(
            policy["zero_primary_minimum_candidate_overlap_fraction"]
        ),
        "zero_primary_maximum_rescue_objects": int(
            policy.get("zero_primary_maximum_rescue_objects", 4)
        ),
        "primary_extension_nm": 0.0,
        "closing_radius_nm": 0.0,
    }
    if int(primary_eligible.max(initial=0)) == 0:
        return reconstruct_with_policy_v3(
            candidate, primary, mitonet, omniem, spacing_nm_zyx, base_policy
        )

    foreground_ratio = float(candidate.sum() / max(1, int(mitonet.sum())))
    support = candidate.copy()
    if foreground_ratio < float(policy["contraction_trigger_ratio"]):
        source_name = policy["extension_source"]
        source = mitonet if source_name == "mitonet_semantic" else primary > 0
        distance_nm = policy["primary_extension_nm"]
        if distance_nm == "full":
            support |= source
        else:
            distance = ndi.distance_transform_edt(~candidate, sampling=spacing_nm_zyx)
            support |= source & (distance <= float(distance_nm))
    return reconstruct_with_policy_v3(
        support, primary, mitonet, omniem, spacing_nm_zyx, base_policy
    )

