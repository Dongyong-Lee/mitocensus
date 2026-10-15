#!/usr/bin/env python3
"""End-to-end verifier for all thirteen MitoCensus object-analysis functions."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from mitocensus.biological_analysis import (
    FUNCTION_NAMES,
    BiologicalAnalysisConfig,
    analyze_biological_sample,
    compare_biological_cohort,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def synthetic_sample(scale: float) -> tuple[np.ndarray, np.ndarray]:
    shape = (9, 48, 48)
    raw = np.full(shape, 188, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.uint16)
    z, y, x = np.indices(shape)
    first = (
        ((z - 4) / 2.4) ** 2
        + ((y - 15) / (4.5 * scale)) ** 2
        + ((x - 14) / (3.5 * scale)) ** 2
        <= 1
    )
    second = (
        ((z - 4) / 2.0) ** 2
        + ((y - 32) / 4.0) ** 2
        + ((x - 33) / 7.0) ** 2
        <= 1
    )
    labels[first] = 1
    labels[second] = 2
    raw[first] = 95
    raw[second] = 110
    raw[(labels > 0) & ((x % 4) == 0)] = 42
    return raw, labels


def main() -> None:
    root = project_root()
    minimal = root / "examples/minimal"
    raw = tifffile.imread(minimal / "input/raw_em.tif")
    instances = tifffile.imread(
        minimal / "expected_output/refined_instances.tif"
    )
    with tempfile.TemporaryDirectory(prefix="mitocensus_analysis_verify_") as directory:
        temporary = Path(directory)
        single = analyze_biological_sample(
            raw,
            instances,
            (30.0, 16.0, 16.0),
            temporary / "minimal",
            sample_id="minimal",
            group="control",
            config=BiologicalAnalysisConfig(min_voxels=5),
            write_report=True,
        )
        if tuple(single["functions"]) != FUNCTION_NAMES:
            raise SystemExit("single-sample output does not cover all 13 functions")
        expected_single = {
            "objects.csv",
            "contacts.csv",
            "sample_summary.json",
            "reference_profile.json",
            "manual_review_instances.tif",
            "biological_report.png",
        }
        actual_single = {path.name for path in (temporary / "minimal").iterdir()}
        if actual_single != expected_single:
            raise SystemExit(
                f"single-sample artifact mismatch: {sorted(actual_single ^ expected_single)}"
            )

        manifest_rows = []
        for sample_id, group, scale in (
            ("control-01", "control", 1.0),
            ("control-02", "control", 1.0),
            ("treated-01", "treated", 1.6),
            ("treated-02", "treated", 1.6),
        ):
            sample_raw, sample_instances = synthetic_sample(scale)
            analysis_dir = temporary / sample_id
            analyze_biological_sample(
                sample_raw,
                sample_instances,
                (24.0, 12.0, 12.0),
                analysis_dir,
                sample_id=sample_id,
                group=group,
                config=BiologicalAnalysisConfig(min_voxels=5),
                write_report=False,
            )
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "group": group,
                    "biological_unit": sample_id,
                    "analysis_dir": analysis_dir.name,
                }
            )
        manifest = temporary / "cohort.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_rows[0])
            writer.writeheader()
            writer.writerows(manifest_rows)
        response = compare_biological_cohort(
            manifest,
            "control",
            temporary / "response",
            bootstrap_iterations=100,
        )
        if response["function_8_disease_or_treatment_response"]["status"] != "computed":
            raise SystemExit("function 8 was not computed")
        group_rows = list(
            csv.DictReader((temporary / "response/group_response.csv").open(newline=""))
        )
        if not group_rows or any(
            row["inferential_p_value"] != "not_reported" for row in group_rows
        ):
            raise SystemExit("cohort response contract failed")
        result = {
            "status": "PASS",
            "single_sample_functions": len(single["functions"]),
            "single_sample_objects": single["functions"]["mitochondrial_amount"][
                "object_count"
            ],
            "cohort_samples": response["n_samples"],
            "cohort_endpoint_comparisons": response[
                "function_8_disease_or_treatment_response"
            ]["n_endpoint_comparisons"],
            "claim_boundary_present": all(
                value in single["claim_boundaries"]
                for value in ("shape_classes", "line_filter", "matrix", "cell_state")
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        print("BIOLOGICAL_ANALYSIS_PASS")


if __name__ == "__main__":
    main()

