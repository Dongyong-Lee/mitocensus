import numpy as np

from mitocensus.instances_v2 import reconstruct_with_policy


def fixture_arrays():
    candidate = np.zeros((5, 20, 20), dtype=bool)
    candidate[1:4, 2:8, 2:8] = True
    candidate[1:4, 12:18, 12:18] = True
    primary = np.zeros(candidate.shape, dtype=np.uint32)
    primary[2, 4:6, 4:6] = 7
    mitonet = primary > 0
    omniem = np.zeros(candidate.shape, dtype=bool)
    omniem[1:4, 12:18, 12:18] = True
    return candidate, primary, mitonet, omniem


def test_anchor_only_drops_unanchored_component():
    candidate, primary, mitonet, omniem = fixture_arrays()
    output = reconstruct_with_policy(
        candidate,
        primary,
        mitonet,
        omniem,
        (30.0, 16.0, 16.0),
        {"new_component_rule": "none"},
    )
    assert output[:, 2:8, 2:8].any()
    assert not output[:, 12:18, 12:18].any()
    assert len(np.unique(output[output > 0])) == 1


def test_consensus_rule_retains_supported_new_component():
    candidate, primary, mitonet, omniem = fixture_arrays()
    output = reconstruct_with_policy(
        candidate,
        primary,
        mitonet,
        omniem,
        (30.0, 16.0, 16.0),
        {
            "new_component_rule": "omniem_consensus",
            "minimum_new_component_voxels": 50,
            "minimum_omniem_overlap_fraction": 0.75,
        },
    )
    assert output[:, 2:8, 2:8].any()
    assert output[:, 12:18, 12:18].any()
    assert len(np.unique(output[output > 0])) == 2


def test_consensus_rule_rejects_below_size_cutoff():
    candidate, primary, mitonet, omniem = fixture_arrays()
    output = reconstruct_with_policy(
        candidate,
        primary,
        mitonet,
        omniem,
        (30.0, 16.0, 16.0),
        {
            "new_component_rule": "omniem_consensus",
            "minimum_new_component_voxels": 200,
            "minimum_omniem_overlap_fraction": 0.25,
        },
    )
    assert not output[:, 12:18, 12:18].any()
