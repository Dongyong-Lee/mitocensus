from __future__ import annotations

import numpy as np

from mitocensus.instances_v4 import reconstruct_with_policy_v4


def policy(**updates) -> dict:
    value = {
        "minimum_native_voxels": 8,
        "zero_primary_minimum_candidate_overlap_fraction": 0.5,
        "zero_primary_maximum_rescue_objects": 4,
        "contraction_trigger_ratio": 0.95,
        "extension_source": "mitonet_semantic",
        "primary_extension_nm": 16.0,
    }
    value.update(updates)
    return value


def arrays():
    shape = (12, 24, 24)
    candidate = np.zeros(shape, dtype=bool)
    candidate[3:9, 8:14, 8:14] = True
    mitonet = candidate.copy()
    mitonet[3:9, 14:16, 8:14] = True
    primary = np.zeros(shape, dtype=np.uint32)
    primary[4:8, 9:13, 9:13] = 1
    omniem = np.zeros(shape, dtype=bool)
    return candidate, primary, mitonet, omniem


def test_contraction_guard_activates_only_below_frozen_ratio() -> None:
    candidate, primary, mitonet, omniem = arrays()
    guarded = reconstruct_with_policy_v4(
        candidate,
        primary,
        mitonet,
        omniem,
        (30.0, 16.0, 16.0),
        policy(contraction_trigger_ratio=0.99, primary_extension_nm=32.0),
    )
    unguarded = reconstruct_with_policy_v4(
        candidate,
        primary,
        mitonet,
        omniem,
        (30.0, 16.0, 16.0),
        policy(contraction_trigger_ratio=0.50, primary_extension_nm=32.0),
    )
    assert int((guarded > 0).sum()) > int((unguarded > 0).sum())


def test_full_extension_is_still_limited_to_selected_primary_source() -> None:
    candidate, primary, mitonet, omniem = arrays()
    mitonet[2:5, 18:21, 18:21] = True
    result = reconstruct_with_policy_v4(
        candidate,
        primary,
        mitonet,
        omniem,
        (30.0, 16.0, 16.0),
        policy(contraction_trigger_ratio=1.1, primary_extension_nm="full"),
    )
    assert np.logical_and(result > 0, mitonet).any()
    assert not result[:, :4, :4].any()

