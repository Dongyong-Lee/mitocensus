from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from mitocensus.biological_analysis import analyze_biological_sample
from mitocensus.cli import build_parser

pytest.importorskip("matplotlib")


def _volume() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    raw = rng.uniform(0.45, 0.6, size=(16, 48, 48)).astype(np.float32)
    labels = np.zeros(raw.shape, dtype=np.uint16)
    z, y, x = np.indices(raw.shape)
    labels[((z - 7) / 4.5) ** 2 + ((y - 12) / 5) ** 2 + ((x - 14) / 9) ** 2 <= 1] = 1   # QC-passing
    labels[5, 30:36, 6:20] = 2                                                          # one-section label
    labels[3:7, 30:40, 30:40] = 3                                                       # possible split ...
    labels[8:12, 30:40, 30:40] = 4                                                      # ... across one section
    raw[labels > 0] = 0.3
    raw[(labels > 0) & (x % 3 == 0)] = 0.15
    return raw, labels


def test_visual_record_is_written_and_matches_the_table(tmp_path: Path) -> None:
    raw, labels = _volume()
    analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "a", sample_id="a", write_report=False,
                              write_visual_record=True)
    record = json.loads((tmp_path / "a/visual_record/visual_record.json").read_text())
    assert (tmp_path / "a/visual_record/index.html").stat().st_size > 1000
    assert record["consistency"]["objects_checked"] >= 2
    assert record["consistency"]["max_abs_feature_difference"] == 0.0
    assert record["consistency"]["all_skeleton_voxel_counts_match"] is True
    assert record["objects_shown"]["thin_labels"] == [2]
    assert set(record["objects_shown"]["split_flags"]) == {3, 4}
    for name in record["files"]["objects"] + ["overview.png", "thin_labels.png", "split_flags.png"]:
        assert (tmp_path / "a/visual_record" / name).exists(), name


def test_visual_record_follows_write_report_by_default(tmp_path: Path) -> None:
    raw, labels = _volume()
    analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "b", sample_id="b", write_report=False)
    assert not (tmp_path / "b/visual_record").exists()


def test_visual_record_command_rewrites_from_outputs(tmp_path: Path) -> None:
    raw, labels = _volume()
    tifffile.imwrite(tmp_path / "raw.tif", raw)
    tifffile.imwrite(tmp_path / "labels.tif", labels)
    analyze_biological_sample(raw, labels, (30.0, 8.0, 8.0), tmp_path / "c", sample_id="c", write_report=False)
    args = build_parser().parse_args(["visual-record", "--analysis", str(tmp_path / "c"), "--raw",
                                      str(tmp_path / "raw.tif"), "--instances", str(tmp_path / "labels.tif")])
    args.function(args)
    record = json.loads((tmp_path / "c/visual_record/visual_record.json").read_text())
    assert record["consistency"]["max_abs_feature_difference"] == 0.0
    # a summary without the recorded pairs (written before 0.4.0): the same pair is found again
    summary_path = tmp_path / "c/sample_summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["quality_control_v2"]["split_label_links"]
    summary_path.write_text(json.dumps(summary))
    args.function(args)
    record = json.loads((tmp_path / "c/visual_record/visual_record.json").read_text())
    assert set(record["objects_shown"]["split_flags"]) == {3, 4}


def test_cli_defaults_and_none() -> None:
    parser = build_parser()
    base = ["analyze-biological", "--raw", "r.tif", "--instances", "i.tif", "--spacing", "30,8,8", "--sample-id", "s",
            "--output", "o"]
    args = parser.parse_args(base)
    assert args.min_axial_extent_nm == 90.0 and args.split_screen_min_iou == 0.4 and args.split_screen_action == "flag"
    args = parser.parse_args(base + ["--min-axial-extent-nm", "none", "--split-screen-min-iou", "none",
                                     "--split-screen-action", "exclude"])
    assert args.min_axial_extent_nm is None and args.split_screen_min_iou is None
    assert args.split_screen_action == "exclude"


def test_measurement_cli_does_not_import_pytorch() -> None:
    code = "import sys, mitocensus.cli, mitocensus.visual_record; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
