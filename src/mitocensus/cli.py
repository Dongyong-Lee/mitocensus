"""MitoCensus command line: object-level measurement and quality control (analyze-biological), cohort comparison
by biological unit (compare-biological) and the optional refinement module (infer-from-priors)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from .biological_analysis import (
    BiologicalAnalysisConfig,
    analyze_biological_sample,
    compare_biological_cohort,
)
from .instances_v4 import reconstruct_with_policy_v4
from .stage_c import load_checkpoint, sha256
from .whole_volume import candidate_probability_slice, normalize_volume


def read_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    else:
        value = tifffile.imread(path)
    value = np.squeeze(value)
    if value.ndim != 3:
        raise ValueError(f"expected one 3-D ZYX volume: {path}: {value.shape}")
    return value


def save_volume(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, value, photometric="minisblack", compression="zlib", compressionargs={"level": 1}, metadata={"axes": "ZYX"})


def parse_spacing(value: str) -> tuple[float, float, float]:
    spacing = tuple(float(item) for item in value.split(","))
    if len(spacing) != 3 or any(item <= 0 for item in spacing):
        raise argparse.ArgumentTypeError("spacing must be positive Z,Y,X nanometres")
    return spacing


def parse_analysis_spacing(value: str) -> tuple[float, ...]:
    spacing = tuple(float(item) for item in value.split(","))
    if len(spacing) not in (2, 3) or any(item <= 0 for item in spacing):
        raise argparse.ArgumentTypeError(
            "analysis spacing must be positive Y,X or Z,Y,X nanometres"
        )
    return spacing


def read_analysis_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    else:
        value = tifffile.imread(path)
    value = np.squeeze(value)
    if value.ndim not in (2, 3):
        raise ValueError(f"expected one 2-D or 3-D image: {path}: {value.shape}")
    return value


def read_policy(path: Path | None) -> dict | None:
    if path is None:
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("reconstruction policy JSON must contain an object")
    candidates = (
        value.get("reconstruction_policy"),
        value.get("policy"),
        value.get("development_summary", {}).get("policy")
        if isinstance(value.get("development_summary"), dict)
        else None,
    )
    policy = next((item for item in candidates if isinstance(item, dict)), value)
    required = {
        "contraction_trigger_ratio",
        "extension_source",
        "minimum_native_voxels",
        "primary_extension_nm",
        "zero_primary_minimum_candidate_overlap_fraction",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(
            "selected v4 reconstruction policy is missing fields: "
            + ", ".join(missing)
        )
    return policy


def command_infer(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    raw_source = read_volume(args.raw)
    raw, normalization = normalize_volume(raw_source)
    mitonet = read_volume(args.mitonet_probability).astype(np.float32)
    omniem = read_volume(args.omniem_probability).astype(np.float32)
    if not (raw.shape == mitonet.shape == omniem.shape):
        raise ValueError(f"raw/prior geometry mismatch: {raw.shape}, {mitonet.shape}, {omniem.shape}")
    if not (np.isfinite(mitonet).all() and np.isfinite(omniem).all()):
        raise ValueError("probability priors contain non-finite values")
    if mitonet.min() < 0 or mitonet.max() > 1 or omniem.min() < 0 or omniem.max() > 1:
        raise ValueError("probability priors must be in [0,1]")
    acquisition_class = int(args.spacing[0] > 1.5 * max(args.spacing[1:]))
    probability = np.empty(raw.shape, dtype=np.float32)
    gate_sum = 0.0
    half = int(config["raw_context"]) // 2
    for z in range(raw.shape[0]):
        indices = np.clip(np.arange(z - half, z + half + 1), 0, raw.shape[0] - 1)
        value, gate = candidate_probability_slice(
            model,
            config,
            raw[indices],
            mitonet[z],
            omniem[z],
            acquisition_class,
            device,
            batch_size=args.batch_size,
        )
        probability[z] = value
        gate_sum += gate
    semantic = probability >= args.threshold
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    probability_path = output / "refined_probability.tif"
    semantic_path = output / "refined_semantic.tif"
    save_volume(probability_path, probability.astype(np.float32))
    save_volume(semantic_path, semantic.astype(np.uint8))
    instance_path = None
    policy = read_policy(args.policy_json)
    if args.mitonet_instances is not None:
        if policy is None:
            raise ValueError("--policy-json is required with --mitonet-instances")
        primary = read_volume(args.mitonet_instances)
        if primary.shape != raw.shape:
            raise ValueError("MitoNet instance geometry differs from raw")
        instances = reconstruct_with_policy_v4(
            semantic,
            primary,
            mitonet >= args.threshold,
            omniem >= args.threshold,
            args.spacing,
            policy,
        )
        instance_path = output / "refined_instances.tif"
        dtype = np.uint16 if instances.max(initial=0) <= np.iinfo(np.uint16).max else np.uint32
        save_volume(instance_path, instances.astype(dtype))
    manifest = {
        "schema_version": "mitocensus.inference-manifest.v1",
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": config,
        "shape_zyx": list(raw.shape),
        "spacing_nm_zyx": list(args.spacing),
        "normalization": normalization,
        "threshold": args.threshold,
        "mean_prior_gate_to_mitonet": gate_sum / raw.shape[0],
        "inputs": {
            "raw_sha256": sha256(args.raw),
            "mitonet_probability_sha256": sha256(args.mitonet_probability),
            "omniem_probability_sha256": sha256(args.omniem_probability),
            "mitonet_instances_sha256": None if args.mitonet_instances is None else sha256(args.mitonet_instances),
        },
        "outputs": {
            "probability_sha256": sha256(probability_path),
            "semantic_sha256": sha256(semantic_path),
            "instances_sha256": None if instance_path is None else sha256(instance_path),
        },
        "reconstruction_policy": policy,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def command_analyze_biological(args: argparse.Namespace) -> None:
    raw = read_analysis_image(args.raw)
    instances = read_analysis_image(args.instances)
    roi = None if args.roi is None else read_analysis_image(args.roi).astype(bool)
    profile = (
        None
        if args.reference_profile is None
        else json.loads(args.reference_profile.read_text())
    )
    config = BiologicalAnalysisConfig(
        min_voxels=args.min_voxels,
        contact_distance_nm=args.contact_distance_nm,
        shell_width_nm=args.shell_width_nm,
        maximum_objects=args.maximum_objects,
        min_axial_sections=args.min_axial_sections,
        split_screen_min_iou=args.split_screen_min_iou,
        split_screen_max_gap_sections=args.split_screen_max_gap_sections,
        edge_correction_weights=args.edge_correction_weights,
    )
    summary = analyze_biological_sample(
        raw,
        instances,
        args.spacing,
        args.output,
        sample_id=args.sample_id,
        group=args.group,
        roi=roi,
        reference_profile=profile,
        config=config,
        write_report=not args.no_report,
    )
    output_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": "mitocensus.biological-analysis-manifest.v1",
        "sample_id": args.sample_id,
        "group": args.group,
        "shape": list(raw.shape),
        "spacing_nm": list(args.spacing),
        "functions": list(summary["functions"]),
        "inputs": {
            "raw": {"path": str(args.raw), "sha256": sha256(args.raw)},
            "instances": {
                "path": str(args.instances),
                "sha256": sha256(args.instances),
            },
            "roi": None
            if args.roi is None
            else {"path": str(args.roi), "sha256": sha256(args.roi)},
            "reference_profile": None
            if args.reference_profile is None
            else {
                "path": str(args.reference_profile),
                "sha256": sha256(args.reference_profile),
            },
        },
        "outputs": output_files,
        "claim_boundaries": summary["claim_boundaries"],
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def command_compare_biological(args: argparse.Namespace) -> None:
    result = compare_biological_cohort(
        args.manifest,
        args.control_group,
        args.output,
        bootstrap_iterations=args.bootstrap_iterations,
        write_report=not args.no_report,
    )
    output_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    analysis_inputs = []
    with args.manifest.open(newline="") as handle:
        for record in csv.DictReader(handle):
            analysis_dir = Path(record["analysis_dir"])
            if not analysis_dir.is_absolute():
                analysis_dir = args.manifest.parent / analysis_dir
            artifact_hashes = {}
            for filename in (
                "objects.csv",
                "contacts.csv",
                "sample_summary.json",
                "reference_profile.json",
            ):
                path = analysis_dir / filename
                artifact_hashes[filename] = {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
            analysis_inputs.append(
                {
                    "sample_id": record["sample_id"],
                    "group": record["group"],
                    "biological_unit": record["biological_unit"],
                    "acquisition_batch": record.get("acquisition_batch") or None,
                    "artifacts": artifact_hashes,
                }
            )
    manifest = {
        "schema_version": "mitocensus.biological-cohort-manifest.v1",
        "control_group": args.control_group,
        "cohort_manifest": {
            "path": str(args.manifest),
            "sha256": sha256(args.manifest),
        },
        "n_samples": result["n_samples"],
        "n_biological_units": result["n_biological_units"],
        "dimensionality": result["dimensionality"],
        "analysis_inputs": analysis_inputs,
        "outputs": output_files,
        "claim_boundary": result["claim_boundary"],
    }
    (args.output / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mitocensus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer-from-priors", help="optional refinement module: run a frozen refinement checkpoint on two aligned probability priors")
    infer.add_argument("--raw", type=Path, required=True, help="3-D ZYX TIFF or NPY raw EM volume")
    infer.add_argument("--mitonet-probability", type=Path, required=True)
    infer.add_argument("--omniem-probability", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--spacing", type=parse_spacing, required=True, metavar="Z,Y,X")
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--threshold", type=float, default=0.5)
    infer.add_argument("--batch-size", type=int, default=8)
    infer.add_argument("--mitonet-instances", type=Path)
    infer.add_argument("--policy-json", type=Path)
    infer.add_argument("--cpu", action="store_true")
    infer.set_defaults(function=command_infer)

    analyze = subparsers.add_parser(
        "analyze-biological",
        help="measure and quality-control every object of an aligned instance mask (any segmenter)",
    )
    analyze.add_argument("--raw", type=Path, required=True, help="2-D or 3-D raw TEM/vEM image")
    analyze.add_argument("--instances", type=Path, required=True, help="aligned integer instance labels")
    analyze.add_argument(
        "--spacing",
        type=parse_analysis_spacing,
        required=True,
        metavar="Y,X|Z,Y,X",
    )
    analyze.add_argument("--sample-id", required=True)
    analyze.add_argument("--group")
    analyze.add_argument("--roi", type=Path, help="optional aligned binary cell/tissue ROI")
    analyze.add_argument(
        "--reference-profile",
        type=Path,
        help="optional control-derived reference_profile.json",
    )
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--min-voxels", type=int, default=20)
    analyze.add_argument("--contact-distance-nm", type=float, default=40.0)
    analyze.add_argument("--shell-width-nm", type=float, default=40.0)
    analyze.add_argument("--maximum-objects", type=int)
    analyze.add_argument(
        "--min-axial-sections",
        type=int,
        help="QC v2: exclude objects present in fewer serial sections from whole-object morphometry (e.g. 3)",
    )
    analyze.add_argument(
        "--split-screen-min-iou",
        type=float,
        help="QC v2: flag border-free single-component labels whose end-section footprint continues in another label (e.g. 0.4)",
    )
    analyze.add_argument("--split-screen-max-gap-sections", type=int, default=3)
    analyze.add_argument(
        "--edge-correction-weights",
        action="store_true",
        help="report Miles-Lantuejoul weights for border-free objects and a weighted size summary",
    )
    analyze.add_argument("--no-report", action="store_true")
    analyze.set_defaults(function=command_analyze_biological)

    compare = subparsers.add_parser(
        "compare-biological",
        help="compare biological-unit analysis outputs against a control group",
    )
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--control-group", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--bootstrap-iterations", type=int, default=2000)
    compare.add_argument("--no-report", action="store_true")
    compare.set_defaults(function=command_compare_biological)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "infer-from-priors" and not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    args.function(args)


if __name__ == "__main__":
    main()
