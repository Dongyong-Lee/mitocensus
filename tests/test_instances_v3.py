from __future__ import annotations

import numpy as np

from mitocensus.instances_v3 import eligible_labels, reconstruct_with_policy_v3


def policy(**updates) -> dict:
    value = {
        "minimum_native_voxels": 8,
        "zero_primary_minimum_candidate_overlap_fraction": 0.5,
        "zero_primary_maximum_rescue_objects": 4,
        "primary_extension_nm": 0.0,
        "closing_radius_nm": 0.0,
    }
    value.update(updates)
    return value


def test_zero_primary_uses_only_candidate_supported_interior_omniem_objects() -> None:
    shape = (12, 20, 20)
    candidate = np.zeros(shape, dtype=bool)
    omniem = np.zeros(shape, dtype=bool)
    candidate[3:7, 5:9, 5:9] = True
    omniem[3:7, 5:9, 5:9] = True
    omniem[4:8, 12:16, 12:16] = True
    result = reconstruct_with_policy_v3(
        candidate,
        np.zeros(shape, dtype=np.uint32),
        np.zeros(shape, dtype=bool),
        omniem,
        (30.0, 16.0, 16.0),
        policy(),
    )
    assert int(result.max()) == 1
    assert np.array_equal(result > 0, omniem & candidate)


def test_border_objects_do_not_trigger_primary_eligibility() -> None:
    labels = np.zeros((8, 12, 12), dtype=np.uint32)
    labels[:, 2:5, 2:5] = 1
    labels[2:6, 6:10, 6:10] = 2
    eligible = eligible_labels(labels, minimum_voxels=8, exclude_border=True)
    assert int(eligible.max()) == 1
    assert not eligible[:, 2:5, 2:5].any()


def test_primary_extension_is_bounded_by_physical_distance() -> None:
    shape = (12, 24, 24)
    candidate = np.zeros(shape, dtype=bool)
    candidate[3:9, 8:14, 8:14] = True
    mitonet = candidate.copy()
    mitonet[3:9, 14:16, 8:14] = True
    primary = np.zeros(shape, dtype=np.uint32)
    primary[4:8, 9:13, 9:13] = 1
    no_extension = reconstruct_with_policy_v3(
        candidate,
        primary,
        mitonet,
        np.zeros(shape, dtype=bool),
        (30.0, 16.0, 16.0),
        policy(primary_extension_nm=0.0),
    )
    extended = reconstruct_with_policy_v3(
        candidate,
        primary,
        mitonet,
        np.zeros(shape, dtype=bool),
        (30.0, 16.0, 16.0),
        policy(primary_extension_nm=32.0),
    )
    assert int((extended > 0).sum()) > int((no_extension > 0).sum())
    assert not extended[:, 16:, :].any()

