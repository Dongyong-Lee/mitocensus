from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import tifffile

from mitocensus.biological_analysis import (
    FUNCTION_NAMES,
    BiologicalAnalysisConfig,
    _read_csv,
    analyze_biological_sample,
    classify_objects,
    compare_biological_cohort,
)
from mitocensus.cli import build_parser


def synthetic_volume(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    shape = (9, 48, 48)
    raw = np.full(shape, 190, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.indices(shape)

    first = (
        ((z - 4) / 2.4) ** 2
        + ((y - 14) / (4.5 * scale)) ** 2
        + ((x - 13) / (3.5 * scale)) ** 2
        <= 1
    )
    second = (
        ((z - 4) / 2.0) ** 2
        + ((y - 31) / 3.5) ** 2
        + ((x - 31) / 7.0) ** 2
        <= 1
    )
    third = (
        ((z - 4) / 1.8) ** 2
        + ((y - 24) / 2.8) ** 2
        + ((x - 20) / 2.8) ** 2
        <= 1
    )
    labels[first] = 1
    labels[second] = 2
    labels[third] = 3
    raw[first] = 92
    raw[second] = 108
    raw[third] = 125
    # Dark internal stripes provide a deterministic line-filter texture.
    raw[(labels > 0) & ((x % 4) == 0)] = 45
    return raw, labels


def test_all_thirteen_functions_and_outputs_are_emitted(tmp_path: Path) -> None:
    raw, labels = synthetic_volume()
    labels[0:2, 0:3, 0:3] = 4  # retained for amount, excluded from whole-object traits
    labels[0, 10, 10] = 5  # one-voxel debris must not enter any endpoint
    output = tmp_path / "analysis"
    summary = analyze_biological_sample(
        raw,
        labels,
        (24.0, 12.0, 12.0),
        output,
        sample_id="sample-1",
        group="control",
        config=BiologicalAnalysisConfig(min_voxels=5, min_axial_extent_nm=None, split_screen_min_iou=None),
        write_report=False,
    )
    assert tuple(summary["functions"]) == FUNCTION_NAMES
    amount = summary["functions"]["mitochondrial_amount"]
    assert amount["object_count"] == 4
    assert amount["qc_passing_object_count"] == 3
    assert amount["border_truncated_object_count"] == 1
    assert np.isclose(
        amount["mitochondrial_fraction"],
        np.isin(labels, (1, 2, 3, 4)).mean(),
    )
    assert summary["functions"]["disease_or_treatment_response"]["status"].startswith(
        "requires_"
    )
    assert np.isfinite(
        summary["functions"]["spatial_organization"][
            "median_to_csr_nearest_neighbor_ratio"
        ]
    )
    assert set(path.name for path in output.iterdir()) == {
        "contacts.csv",
        "manual_review_instances.tif",
        "objects.csv",
        "reference_profile.json",
        "sample_summary.json",
    }
    assert (output / "contacts.csv").read_text().startswith("object_id_a,")
    with (output / "objects.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {
        "shape_class",
        "line_filter_fraction",
        "matrix_electron_density",
        "composite_review_score",
        "rare_morphology_candidate",
        "moas_like_candidate",
        "nanotunnel_like_candidate",
        "junction_like_candidate",
        "small_dense_pattern",
        "swollen_weak_boundary_pattern",
        "double_membrane_candidate",
    }
    assert required_columns.issubset(rows[0])
    border_row = next(row for row in rows if row["object_id"] == "4")
    assert border_row["morphology_evaluable"] == "False"
    assert border_row["shape_class"] == "not_evaluable_border_truncated"
    assert border_row["manual_review_recommended"] == "False"
    summary_text = (output / "sample_summary.json").read_text()
    assert "NaN" not in summary_text
    assert json.loads(summary_text)["functions"]["mitochondrion_pairs"][
        "median_gap_nm"
    ] is None


def test_two_dimensional_tem_is_supported(tmp_path: Path) -> None:
    y, x = np.indices((64, 64))
    labels = np.zeros((64, 64), dtype=np.uint32)
    labels[((y - 22) / 7) ** 2 + ((x - 20) / 12) ** 2 <= 1] = 1001
    labels[((y - 43) / 6) ** 2 + ((x - 42) / 6) ** 2 <= 1] = 900000
    labels[5:8, 54:57] = 900000
    raw = np.full(labels.shape, 180, dtype=np.uint8)
    raw[labels > 0] = 90
    raw[(labels > 0) & (x % 5 == 0)] = 40
    summary = analyze_biological_sample(
        raw,
        labels,
        (8.0, 8.0),
        tmp_path / "analysis-2d",
        sample_id="tem-2d",
        config=BiologicalAnalysisConfig(min_voxels=5),
        write_report=False,
    )
    assert summary["dimensionality"] == 2
    amount = summary["functions"]["mitochondrial_amount"]
    assert amount["roi_measure_unit"] == "um2"
    assert amount["object_density_unit"] == "per_100_um2"
    with (tmp_path / "analysis-2d/objects.csv").open(newline="") as handle:
        object_rows = {int(row["object_id"]): row for row in csv.DictReader(handle)}
    object_ids = set(object_rows)
    assert object_ids == {1001, 900000}
    assert object_rows[900000]["connected_component_count"] == "2"
    assert object_rows[900000]["morphology_evaluable"] == "False"
    assert (
        object_rows[900000]["shape_class"]
        == "not_evaluable_disconnected_label"
    )

    roi = x >= 20
    roi_output = tmp_path / "analysis-2d-roi"
    analyze_biological_sample(
        raw,
        labels,
        (8.0, 8.0),
        roi_output,
        sample_id="tem-2d-roi",
        roi=roi,
        config=BiologicalAnalysisConfig(min_voxels=5),
        write_report=False,
    )
    with (roi_output / "objects.csv").open(newline="") as handle:
        roi_rows = {int(row["object_id"]): row for row in csv.DictReader(handle)}
    assert roi_rows[1001]["roi_boundary_touching"] == "True"
    assert roi_rows[1001]["morphology_evaluable"] == "False"


def test_rare_topology_moas_and_nanotunnel_candidates_are_operational(
    tmp_path: Path,
) -> None:
    shape = (25, 96, 96)
    z, y, x = np.indices(shape)
    labels = np.zeros(shape, dtype=np.uint16)
    for center_x in (20, 40, 60):
        labels[(z - 12) ** 2 + (y - 25) ** 2 + (x - center_x) ** 2 <= 6**2] = 1
    labels[
        (abs(z - 12) <= 1)
        & (abs(y - 25) <= 1)
        & (x >= 20)
        & (x <= 60)
    ] = 1
    for center_x in (15, 78):
        labels[(z - 12) ** 2 + (y - 65) ** 2 + (x - center_x) ** 2 <= 5**2] = 2
    labels[
        (abs(z - 12) <= 1)
        & (abs(y - 65) <= 1)
        & (x >= 15)
        & (x <= 78)
    ] = 2
    labels[
        (np.sqrt((x - 75) ** 2 + (y - 25) ** 2) - 12) ** 2
        + (z - 12) ** 2
        <= 3**2
    ] = 3
    raw = np.full(shape, 180, dtype=np.uint8)
    raw[labels > 0] = 85
    raw[(labels > 0) & (x % 5 == 0)] = 40
    output = tmp_path / "rare"
    analyze_biological_sample(
        raw,
        labels,
        (20.0, 10.0, 10.0),
        output,
        sample_id="rare",
        config=BiologicalAnalysisConfig(min_voxels=5),
        write_report=False,
    )
    with (output / "objects.csv").open(newline="") as handle:
        rows = {int(row["object_id"]): row for row in csv.DictReader(handle)}
    assert rows[1]["moas_like_candidate"] == "True"
    assert rows[2]["nanotunnel_like_candidate"] == "True"
    assert rows[3]["ring_or_fenestrated_candidate"] == "True"


def test_close_pairs_and_junction_like_candidate_are_operational(
    tmp_path: Path,
) -> None:
    shape = (15, 64, 64)
    z, y, x = np.indices(shape)
    labels = np.zeros(shape, dtype=np.uint16)
    labels[((z - 7) / 2) ** 2 + ((y - 27) / 3) ** 2 + ((x - 32) / 12) ** 2 <= 1] = 1
    labels[((z - 7) / 2) ** 2 + ((y - 35) / 3) ** 2 + ((x - 32) / 12) ** 2 <= 1] = 2
    raw = np.full(shape, 180, dtype=np.uint8)
    raw[labels > 0] = 80
    raw[(labels == 0) & (y >= 30) & (y <= 32) & (abs(x - 32) < 12)] = 40
    output = tmp_path / "contact"
    summary = analyze_biological_sample(
        raw,
        labels,
        (20.0, 10.0, 10.0),
        output,
        sample_id="contact",
        config=BiologicalAnalysisConfig(min_voxels=5, contact_distance_nm=40),
        write_report=False,
    )
    result = summary["functions"]["mitochondrion_pairs"]
    assert result["close_pair_count"] == 1
    assert result["junction_like_pair_count"] == 1
    assert result["median_gap_nm"] == 10.0
    assert result["mitochondria_with_close_pair_fraction"] == 1.0


def test_cell_state_candidate_rules_retain_non_diagnostic_labels() -> None:
    profile = {
        "source": "test-control",
        "features": {
            "physical_size": {"median": 10.0, "scale": 1.0, "q95": 12.0},
            "roundness": {"median": 0.5, "scale": 0.1},
            "elongation": {"median": 2.0, "scale": 0.5},
            "skeleton_length_um_proxy": {"median": 1.0, "scale": 0.2},
            "line_filter_fraction": {"median": 0.5, "scale": 0.1},
            "line_filter_spacing_nm": {"median": 20.0, "scale": 5.0},
            "line_filter_coherence": {"median": 0.7, "scale": 0.1},
            "matrix_electron_density": {"median": 0.5, "scale": 0.1},
            "matrix_heterogeneity": {"median": 0.2, "scale": 0.05},
            "boundary_electron_density": {"median": 0.5, "scale": 0.1},
            "weak_boundary_fraction": {"median": 0.2, "scale": 0.1},
            "perimito_double_membrane_score": {"median": 0.2, "scale": 0.1},
        },
    }
    common = {
        "roundness": 0.8,
        "elongation": 1.0,
        "skeleton_length_um_proxy": 0.5,
        "branch_points": 0,
        "line_filter_fraction": 0.1,
        "line_filter_spacing_nm": 35.0,
        "line_filter_coherence": 0.2,
        "matrix_heterogeneity": 0.4,
        "boundary_electron_density": 0.9,
        "weak_boundary_fraction": 0.8,
        "perimito_double_membrane_score": 0.5,
        "neck_ratio": 1.0,
        "lobe_count": 1,
        "neck_radius_q10_nm": 200.0,
        "topological_hole_candidate": False,
        "junction_like_candidate": False,
    }
    small_dense_like = {
        **common,
        "physical_size": 8.0,
        "matrix_electron_density": 0.7,
    }
    swollen_weak_boundary_like = {
        **common,
        "physical_size": 13.0,
        "matrix_electron_density": 0.3,
    }
    rows = [small_dense_like, swollen_weak_boundary_like]
    classify_objects(rows, profile)
    assert rows[0]["small_dense_pattern"] is True
    assert rows[0]["double_membrane_candidate"] is True
    assert rows[1]["swollen_weak_boundary_pattern"] is True
    assert rows[1]["stress_associated_pattern"] is True
    assert "diagnosis" not in rows[0]


def test_cohort_comparison_uses_biological_units(tmp_path: Path) -> None:
    records = []
    for sample_id, group, scale in (
        ("c1", "control", 1.0),
        ("c1-field2", "control", 1.2),
        ("c2", "control", 1.0),
        ("t1", "treated", 1.6),
        ("t2", "treated", 1.6),
    ):
        raw, labels = synthetic_volume(scale)
        analysis_dir = tmp_path / sample_id
        analyze_biological_sample(
            raw,
            labels,
            (24.0, 12.0, 12.0),
            analysis_dir,
            sample_id=sample_id,
            group=group,
            config=BiologicalAnalysisConfig(min_voxels=5),
            write_report=False,
        )
        records.append(
            {
                "sample_id": sample_id,
                "group": group,
                "biological_unit": "c1" if sample_id == "c1-field2" else sample_id,
                "analysis_dir": sample_id,
            }
        )
    manifest = tmp_path / "cohort.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    output = tmp_path / "response"
    result = compare_biological_cohort(
        manifest, "control", output, bootstrap_iterations=50
    )
    assert result["function_8_disease_or_treatment_response"]["status"] == "computed"
    assert result["n_samples"] == 5
    assert result["n_biological_units"] == 4
    assert result["groups"] == {"control": 2, "treated": 2}
    with (output / "sample_endpoints.csv").open(newline="") as handle:
        unit_rows = list(csv.DictReader(handle))
    assert len(unit_rows) == 4
    assert next(row for row in unit_rows if row["biological_unit"] == "c1")[
        "n_technical_samples"
    ] == "2"
    assert (output / "control_calibrated_objects.csv").is_file()
    calibrated = json.loads(
        (output / "control_calibrated_sample_summaries.json").read_text()
    )
    assert len(calibrated["samples"]) == 5
    assert calibrated["calibration_source"] == "biological_unit_balanced_control_group"
    with (output / "group_response.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    endpoints = {row["endpoint"] for row in rows}
    assert "moas_like_fraction" in endpoints
    assert "close_pairs_per_mitochondrion" in endpoints
    assert "stress_associated_fraction" in endpoints
    median_size = next(row for row in rows if row["endpoint"] == "median_size")
    assert int(median_size["n_control_biological_units"]) == 2
    assert int(median_size["n_comparison_biological_units"]) == 2
    assert median_size["inferential_p_value"] == "not_reported"
    assert json.loads((output / "disease_treatment_response.json").read_text())[
        "independent_unit"
    ] == "biological_unit from the cohort manifest"
    assert "NaN" not in (output / "disease_treatment_response.json").read_text()

    conflicting = [dict(row) for row in records]
    next(row for row in conflicting if row["sample_id"] == "t1")[
        "biological_unit"
    ] = "c1"
    conflicting_manifest = tmp_path / "cohort-conflicting.csv"
    with conflicting_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=conflicting[0])
        writer.writeheader()
        writer.writerows(conflicting)
    with pytest.raises(ValueError, match="occurs in multiple groups"):
        compare_biological_cohort(
            conflicting_manifest,
            "control",
            tmp_path / "conflicting-response",
            bootstrap_iterations=10,
        )

    confounded = [
        {
            **row,
            "acquisition_batch": (
                "batch-control" if row["group"] == "control" else "batch-treated"
            ),
        }
        for row in records
    ]
    confounded_manifest = tmp_path / "cohort-batch-confounded.csv"
    with confounded_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=confounded[0])
        writer.writeheader()
        writer.writerows(confounded)
    with pytest.raises(ValueError, match="completely confounded"):
        compare_biological_cohort(
            confounded_manifest,
            "control",
            tmp_path / "batch-confounded-response",
            bootstrap_iterations=10,
        )

    cli_output = tmp_path / "response-cli"
    cli_args = build_parser().parse_args(
        [
            "compare-biological",
            "--manifest",
            str(manifest),
            "--control-group",
            "control",
            "--bootstrap-iterations",
            "10",
            "--no-report",
            "--output",
            str(cli_output),
        ]
    )
    cli_args.function(cli_args)
    provenance = json.loads((cli_output / "cohort_manifest.json").read_text())
    assert provenance["n_biological_units"] == 4
    assert len(provenance["analysis_inputs"]) == 5
    assert not (cli_output / "cohort_response_report.png").exists()
    assert all(
        len(record["artifacts"]["objects.csv"]["sha256"]) == 64
        for record in provenance["analysis_inputs"]
    )


def test_cli_exposes_biological_commands() -> None:
    parser = build_parser()
    analyze = parser.parse_args(
        [
            "analyze-biological",
            "--raw",
            "raw.tif",
            "--instances",
            "instances.tif",
            "--spacing",
            "8,8",
            "--sample-id",
            "sample",
            "--output",
            "result",
        ]
    )
    assert analyze.command == "analyze-biological"
    compare = parser.parse_args(
        [
            "compare-biological",
            "--manifest",
            "cohort.csv",
            "--control-group",
            "control",
            "--output",
            "response",
        ]
    )
    assert compare.command == "compare-biological"


def test_biological_cli_writes_hash_manifest(tmp_path: Path) -> None:
    raw, labels = synthetic_volume()
    raw_path = tmp_path / "raw.tif"
    labels_path = tmp_path / "instances.tif"
    tifffile.imwrite(raw_path, raw)
    tifffile.imwrite(labels_path, labels)
    output = tmp_path / "cli-output"
    args = build_parser().parse_args(
        [
            "analyze-biological",
            "--raw",
            str(raw_path),
            "--instances",
            str(labels_path),
            "--spacing",
            "24,12,12",
            "--sample-id",
            "cli-sample",
            "--min-voxels",
            "5",
            "--no-report",
            "--output",
            str(output),
        ]
    )
    args.function(args)
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert len(manifest["functions"]) == 13
    assert len(manifest["inputs"]["raw"]["sha256"]) == 64
    assert "sample_summary.json" in manifest["outputs"]


def _qc2_volume() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic stack: one thick object, one single-section sliver and one profile split by an empty section."""
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.4, 0.6, size=(16, 48, 48)).astype(np.float32)
    labels = np.zeros(raw.shape, dtype=np.uint16)
    labels[2:9, 4:14, 4:14] = 1          # thick, QC-passing
    labels[5, 20:30, 20:24] = 2          # single-section sliver
    labels[3:7, 32:42, 32:42] = 3        # profile ...
    labels[8:12, 32:42, 32:42] = 4       # ... continuing after an unlabelled section (z = 7)
    raw[labels > 0] = 0.25
    return raw, labels


def test_v031_settings_keep_v1_behaviour(tmp_path: Path) -> None:
    raw, labels = _qc2_volume()
    config = BiologicalAnalysisConfig(min_axial_extent_nm=None, split_screen_min_iou=None)
    summary = analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "v1", sample_id="v1",
                                        config=config, write_report=False)
    rows = {int(float(r["object_id"])): r for r in csv.DictReader((tmp_path / "v1/objects.csv").open())}
    assert all(rows[i]["morphology_evaluable"] == "True" for i in (1, 2, 3, 4))
    assert summary["quality_control_v2"]["enabled"] is False
    assert rows[2]["axial_sections"] == "1"


def test_default_extent_check_and_split_flag(tmp_path: Path) -> None:
    raw, labels = _qc2_volume()
    summary = analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "v4", sample_id="v4",
                                        write_report=False, write_visual_record=False)
    rows = {int(float(r["object_id"])): r for r in csv.DictReader((tmp_path / "v4/objects.csv").open())}
    assert rows[1]["morphology_evaluable"] == "True"
    assert rows[2]["morphology_evaluable"] == "False"
    assert "axial_extent_below_minimum" in rows[2]["non_evaluable_reason"]
    assert rows[2]["shape_class"] == "not_evaluable_thin_profile"
    for i in (3, 4):  # flagged for review, still QC-passing
        assert rows[i]["split_label_candidate"] == "True"
        assert rows[i]["morphology_evaluable"] == "True"
        assert rows[i]["manual_review_recommended"] == "True"
        assert "split_label_candidate" not in rows[i]["non_evaluable_reason"]
    qc = summary["quality_control_v2"]
    assert qc["enabled"] is True and qc["split_screen_action"] == "flag"
    assert qc["split_label_groups"] == [[3, 4]] and qc["split_label_flagged_objects"] == 2
    assert [(link["object_id"], link["other_id"], link["section"], link["other_section"], link["iou"])
            for link in qc["split_label_links"]] == [(3, 4, 6, 8, 1.0), (4, 3, 8, 6, 1.0)]
    assert qc["objects_below_min_axial_extent"] == 1 and qc["min_axial_extent_nm"] == 90.0


def test_extent_check_is_physical(tmp_path: Path) -> None:
    raw, labels = _qc2_volume()
    # object 1 spans 7 sections: 7 x 12 nm = 84 nm < 90 nm on a finer grid, 7 x 30 nm passes
    fine = analyze_biological_sample(raw, labels, (12.0, 8.0, 8.0), tmp_path / "fine", sample_id="f",
                                     write_report=False, write_visual_record=False)
    rows = {int(float(r["object_id"])): r for r in csv.DictReader((tmp_path / "fine/objects.csv").open())}
    assert "axial_extent_below_minimum" in rows[1]["non_evaluable_reason"]
    assert fine["quality_control_v2"]["objects_below_min_axial_extent"] == 4


def test_qc_v2_excludes_slivers_and_flags_split_labels(tmp_path: Path) -> None:
    raw, labels = _qc2_volume()
    config = BiologicalAnalysisConfig(min_axial_sections=3, split_screen_min_iou=0.4, split_screen_action="exclude",
                                      min_axial_extent_nm=None)
    summary = analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "v2", sample_id="v2",
                                        config=config, write_report=False)
    rows = {int(float(r["object_id"])): r for r in csv.DictReader((tmp_path / "v2/objects.csv").open())}
    assert rows[1]["morphology_evaluable"] == "True"
    assert rows[2]["morphology_evaluable"] == "False"
    assert "axial_sections_below_minimum" in rows[2]["non_evaluable_reason"]
    assert rows[2]["shape_class"] == "not_evaluable_thin_profile"
    assert rows[3]["shape_class"] == "not_evaluable_split_label_candidate"
    for i in (3, 4):
        assert rows[i]["split_label_candidate"] == "True"
        assert rows[i]["morphology_evaluable"] == "False"
        assert "split_label_candidate" in rows[i]["non_evaluable_reason"]
    qc = summary["quality_control_v2"]
    assert qc["enabled"] is True and qc["split_label_groups"] == [[3, 4]]
    assert qc["objects_below_min_axial_sections"] == 1


def test_edge_correction_weights_are_opt_in_and_match_formula(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    raw = rng.uniform(0.4, 0.6, size=(20, 40, 40)).astype(np.float32)
    labels = np.zeros(raw.shape, dtype=np.uint16)
    labels[4:9, 5:10, 5:10] = 1          # 5 x 5 x 5, border-free
    labels[3:15, 15:35, 20:26] = 2       # 12 x 20 x 6, border-free
    labels[0:6, 30:38, 2:8] = 3          # touches the z = 0 face
    raw[labels > 0] = 0.25
    default = analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "off", sample_id="off",
                                        write_report=False)
    header = next(csv.reader((tmp_path / "off/objects.csv").open()))
    assert "edge_correction_weight" not in header and "edge_correction" not in default
    summary = analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "on", sample_id="on",
                                        config=BiologicalAnalysisConfig(edge_correction_weights=True),
                                        write_report=False)
    rows = {int(float(r["object_id"])): r for r in csv.DictReader((tmp_path / "on/objects.csv").open())}
    expected_1 = (20 / 14) * (40 / 34) * (40 / 34)
    expected_2 = (20 / 7) * (40 / 19) * (40 / 33)
    assert abs(float(rows[1]["edge_correction_weight"]) - expected_1) < 1e-9
    assert abs(float(rows[2]["edge_correction_weight"]) - expected_2) < 1e-9
    assert rows[3]["edge_correction_weight"] in ("", "nan", "NaN")
    edge = summary["edge_correction"]
    assert edge["region_shape_voxels"] == [20, 40, 40] and edge["region_is_box"] is True
    assert edge["n_weighted_objects"] == 2
    sizes = {i: float(rows[i]["physical_size"]) for i in (1, 2)}
    weighted = (expected_1 * sizes[1] + expected_2 * sizes[2]) / (expected_1 + expected_2)
    assert abs(edge["weighted_mean_physical_size"] - weighted) < 1e-12
    assert edge["weighted_mean_physical_size"] > edge["unweighted_mean_physical_size"]


def test_cli_passes_qc_v2_and_edge_options(tmp_path: Path) -> None:
    raw, labels = _qc2_volume()
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "labels.npy", labels)
    args = build_parser().parse_args([
        "analyze-biological", "--raw", str(tmp_path / "raw.npy"), "--instances", str(tmp_path / "labels.npy"),
        "--spacing", "30,8,8", "--sample-id", "cli-qc2", "--output", str(tmp_path / "out"),
        "--min-axial-sections", "3", "--split-screen-min-iou", "0.4", "--edge-correction-weights", "--no-report",
    ])
    args.function(args)
    summary = json.loads((tmp_path / "out/sample_summary.json").read_text())
    assert summary["quality_control_v2"]["enabled"] is True
    assert summary["quality_control_v2"]["split_label_groups"] == [[3, 4]]
    assert summary["edge_correction"]["enabled"] is True
    assert summary["configuration"]["min_axial_sections"] == 3



def test_split_screen_links_a_qc_passing_label_to_a_thin_continuation(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    raw = rng.uniform(0.4, 0.6, size=(16, 40, 40)).astype(np.float32)
    labels = np.zeros(raw.shape, dtype=np.uint16)
    labels[3, 10:20, 10:20] = 1          # one-section sliver ...
    labels[6:10, 10:20, 10:20] = 2       # ... continued, after two unlabelled sections, by a 4-section label
    labels[4:12, 26:34, 26:34] = 3       # unrelated object
    raw[labels > 0] = 0.25
    config = BiologicalAnalysisConfig(min_axial_sections=3, split_screen_min_iou=0.4, split_screen_action="exclude",
                                      min_axial_extent_nm=None)
    summary = analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "out", sample_id="s",
                                        config=config, write_report=False)
    rows = {int(float(r["object_id"])): r for r in csv.DictReader((tmp_path / "out/objects.csv").open())}
    assert summary["quality_control_v2"]["split_label_groups"] == [[1, 2]]
    assert rows[2]["split_label_candidate"] == "True" and rows[2]["morphology_evaluable"] == "False"
    assert rows[1]["shape_class"] == "not_evaluable_thin_profile"
    assert rows[2]["shape_class"] == "not_evaluable_split_label_candidate"
    assert rows[3]["morphology_evaluable"] == "True"


def test_outputs_name_image_criteria_not_mechanisms(tmp_path: Path) -> None:
    raw, labels = synthetic_volume()
    analyze_biological_sample(
        raw, labels, (24.0, 12.0, 12.0), tmp_path / "s1", sample_id="s1", group="control",
        config=BiologicalAnalysisConfig(min_voxels=5), write_report=False,
    )
    text = "".join((tmp_path / "s1" / name).read_text() for name in ("sample_summary.json", "reference_profile.json", "objects.csv"))
    for word in ("ferroptosis", "apoptosis", "mitophagy", "cristae"):
        assert word not in text.lower()


def test_empty_numeric_fields_are_missing_values(tmp_path: Path) -> None:
    """compare-biological once failed on a QC-passing object without a neck radius (empty CSV field)."""
    path = tmp_path / "objects.csv"
    path.write_text("sample_id,object_id,morphology_evaluable,elongation,neck_radius_q10_nm\ns,1,True,6.0,\n")
    row = _read_csv(path)[0]
    assert math.isnan(row["neck_radius_q10_nm"])
    rows = [dict(row, branch_points=0, end_points=2, lobe_count=1, roundness=0.3, physical_size=0.01,
                 skeleton_length_um_proxy=1.0, topological_hole_candidate=False, neck_ratio=math.nan)]
    classify_objects(rows, {"features": {}})
    assert rows[0]["nanotunnel_like_candidate"] is False


def test_records_with_another_schema_are_rejected(tmp_path: Path) -> None:
    records = []
    for sample_id, group, scale in (("c1", "control", 1.0), ("t1", "treated", 1.6)):
        raw, labels = synthetic_volume(scale)
        analyze_biological_sample(
            raw, labels, (24.0, 12.0, 12.0), tmp_path / sample_id, sample_id=sample_id, group=group,
            config=BiologicalAnalysisConfig(min_voxels=5), write_report=False,
        )
        records.append({"sample_id": sample_id, "group": group, "biological_unit": sample_id, "analysis_dir": sample_id})
    profile = json.loads((tmp_path / "c1/reference_profile.json").read_text())
    profile["schema_version"] = "other.biological-reference-profile.v1"
    with pytest.raises(ValueError, match="expected 'mitocensus.biological-reference-profile.v1'"):
        analyze_biological_sample(
            raw, labels, (24.0, 12.0, 12.0), tmp_path / "with_profile", sample_id="t2", group="treated",
            reference_profile=profile, config=BiologicalAnalysisConfig(min_voxels=5), write_report=False,
        )
    summary_path = tmp_path / "c1/sample_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["schema_version"] = "other.biological-analysis.v1"
    summary_path.write_text(json.dumps(summary))
    manifest = tmp_path / "cohort.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError, match="unsupported sample summary schema 'other.biological-analysis.v1'"):
        compare_biological_cohort(manifest, "control", tmp_path / "response", bootstrap_iterations=10, write_report=False)


# --- 0.3.1 measurement geometry -------------------------------------------------------------------------------


def _rows(output: Path) -> dict[int, dict]:
    with (output / "objects.csv").open(newline="") as handle:
        return {int(row["object_id"]): row for row in csv.DictReader(handle)}


def test_small_pieces_do_not_make_a_label_disconnected(tmp_path: Path) -> None:
    shape = (20, 48, 48)
    z, y, x = np.indices(shape)
    labels = np.zeros(shape, dtype=np.uint16)
    labels[((z - 10) / 4) ** 2 + ((y - 24) / 6) ** 2 + ((x - 24) / 6) ** 2 <= 1] = 5
    labels[3, 3:5, 40] = 5  # two stray voxels of the same label, far from the object
    labels[((z - 10) / 3) ** 2 + ((y - 10) / 4) ** 2 + ((x - 10) / 4) ** 2 <= 1] = 6
    labels[((z - 10) / 3) ** 2 + ((y - 38) / 4) ** 2 + ((x - 38) / 4) ** 2 <= 1] = 6  # a second large piece
    raw = np.full(shape, 180, dtype=np.uint8)
    raw[labels > 0] = 90
    analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "pieces", sample_id="pieces",
                              config=BiologicalAnalysisConfig(), write_report=False)
    rows = _rows(tmp_path / "pieces")
    assert rows[5]["connected_component_count"] == "1"
    assert rows[5]["ignored_fragment_voxels"] == "2"
    assert rows[5]["morphology_evaluable"] == "True"
    assert rows[6]["connected_component_count"] == "2"
    assert "disconnected_instance_label" in rows[6]["non_evaluable_reason"]


def test_skeleton_of_an_anisotropic_tube_has_no_branch_points() -> None:
    from mitocensus.biological_analysis import _skeleton_features

    rng = np.random.default_rng(7)
    shape = (36, 48, 48)
    z, y, x = np.indices(shape)
    radius = 12 + rng.integers(-1, 2, size=(shape[0], 1, 1))  # a bumpy wall, as in segmented sections
    mask = ((y - 24) ** 2 + (x - 24) ** 2 <= radius**2) & (z >= 3) & (z < 33)
    features = _skeleton_features(mask, (30.0, 8.0, 8.0))
    assert features["branch_points"] == 0
    assert features["end_points"] == 2
    assert 0.6 < features["skeleton_length_um_proxy"] < 0.95  # 900 nm tube


def test_skeleton_counts_a_real_branch() -> None:
    from mitocensus.biological_analysis import _skeleton_features

    shape = (40, 120, 120)
    z, y, x = np.indices(shape)
    mask = np.zeros(shape, dtype=bool)
    centre = np.array([20.0, 60.0, 60.0])
    for angle in (0.0, 2.0943951, 4.1887902):  # three arms of 50 voxels, 120 degrees apart
        direction = np.array([0.0, math.sin(angle), math.cos(angle)])
        for t in np.linspace(0, 50, 101):
            point = centre + t * direction
            mask |= (z - point[0]) ** 2 + (y - point[1]) ** 2 + (x - point[2]) ** 2 <= 5**2
    features = _skeleton_features(mask, (8.0, 8.0, 8.0))
    assert features["branch_points"] == 1
    assert features["end_points"] == 3


def test_line_filter_ignores_the_boundary_band(tmp_path: Path) -> None:
    shape = (12, 64, 64)
    z, y, x = np.indices(shape)
    distance = np.sqrt((y - 32) ** 2 + (x - 32) ** 2)
    inside = (distance <= 20) & (z >= 2) & (z < 10)
    labels = np.zeros(shape, dtype=np.uint16)
    labels[inside] = 1
    raw = np.full(shape, 200, dtype=np.uint8)
    raw[inside] = 150
    raw[inside & (distance >= 18)] = 40  # a dark 16-nm membrane band inside the boundary, no internal lines
    analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "band", sample_id="band",
                              config=BiologicalAnalysisConfig(), write_report=False)
    band = float(_rows(tmp_path / "band")[1]["line_filter_fraction"])
    raw_lines = raw.copy()
    raw_lines[inside & (distance < 12) & (np.abs(x - 32) % 6 == 0)] = 40  # internal crista-like lines
    analyze_biological_sample(raw_lines, labels, (30.0, 8.0, 8.0), tmp_path / "lines", sample_id="lines",
                              config=BiologicalAnalysisConfig(), write_report=False)
    lines = float(_rows(tmp_path / "lines")[1]["line_filter_fraction"])
    assert band < 0.01
    assert lines > 0.05


def test_boundary_measures_use_the_in_plane_rim() -> None:
    from mitocensus.biological_analysis import _in_plane_rim

    mask = np.zeros((7, 12, 12), dtype=bool)
    mask[1:6, 2:10, 2:10] = True
    rim = _in_plane_rim(mask)
    assert not rim[1, 4:8, 4:8].any()  # the inside of the first section is a face, not the rim
    assert rim[3, 2, 5] and rim[3, 5, 9]
    assert int(rim.sum()) == 5 * 28
