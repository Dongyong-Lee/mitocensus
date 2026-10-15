from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mitocensus import __version__
from mitocensus.cli import build_parser, read_policy
from mitocensus.instances_v4 import reconstruct_with_policy_v4

PROJECT = Path(__file__).resolve().parents[1]


def selected_policy() -> dict:
    return {
        "contraction_trigger_ratio": 0.95,
        "extension_source": "primary_instances",
        "minimum_native_voxels": 8,
        "primary_extension_nm": 24,
        "zero_primary_maximum_rescue_objects": 4,
        "zero_primary_minimum_candidate_overlap_fraction": 0.5,
    }


def test_read_policy_accepts_selected_model_contract(tmp_path: Path) -> None:
    contract = tmp_path / "selected_model_contract.json"
    contract.write_text(
        json.dumps({"development_summary": {"policy": selected_policy()}})
    )
    assert read_policy(contract) == selected_policy()


def test_read_policy_rejects_legacy_incomplete_policy(tmp_path: Path) -> None:
    contract = tmp_path / "legacy_policy.json"
    contract.write_text(json.dumps({"new_component_rule": "none"}))
    try:
        read_policy(contract)
    except ValueError as error:
        assert "selected v4 reconstruction policy" in str(error)
    else:
        raise AssertionError("legacy Stage-C2 policy was not rejected")


def test_selected_policy_runs_with_v4_reconstruction() -> None:
    shape = (12, 24, 24)
    candidate = np.zeros(shape, dtype=bool)
    candidate[3:9, 8:14, 8:14] = True
    primary = np.zeros(shape, dtype=np.uint32)
    primary[4:8, 9:13, 9:13] = 1
    mitonet = candidate.copy()
    mitonet[3:9, 14:16, 8:14] = True
    omniem = np.zeros(shape, dtype=bool)
    result = reconstruct_with_policy_v4(
        candidate,
        primary,
        mitonet,
        omniem,
        (30.0, 16.0, 16.0),
        selected_policy(),
    )
    assert result.shape == shape
    assert result.dtype == np.uint32
    assert int(result.max()) == 1


def test_version_and_command_name_match_the_metadata() -> None:
    assert f'version = "{__version__}"' in (PROJECT / "pyproject.toml").read_text()
    assert f"version: {__version__}" in (PROJECT / "CITATION.cff").read_text()
    assert build_parser().prog == "mitocensus"
