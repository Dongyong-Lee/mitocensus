from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from skimage.transform import resize
from torch.nn import functional as F

from .metrics import boundary_counts, semantic_counts


TARGET_INSTANCE_SPACING_NM_ZYX = (30.0, 16.0, 16.0)


def load_nifti_zyx(
    path: Path,
    expected_xyz: tuple[int, int, int],
    expected_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, str]:
    source = np.squeeze(np.asanyarray(nib.load(str(path)).dataobj))
    if source.shape == expected_xyz and source.shape != expected_zyx:
        return np.transpose(source, (2, 1, 0)), "xyz_transposed_to_zyx"
    if source.shape == expected_zyx and source.shape != expected_xyz:
        return source, "already_zyx"
    if source.shape == expected_xyz == expected_zyx:
        raise ValueError(f"ambiguous cubic NIfTI axis order: {path}")
    raise ValueError(
        f"unexpected NIfTI shape {source.shape}; expected XYZ {expected_xyz} or ZYX {expected_zyx}: {path}"
    )


def normalization_limits(array: np.ndarray) -> tuple[float, float]:
    sample = array[
        :: max(1, array.shape[0] // 64),
        ::8,
        ::8,
    ].astype(np.float32, copy=False)
    low, high = np.percentile(sample, (0.5, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError((low, high))
    return float(low), float(high)


def normalize_volume(array: np.ndarray) -> tuple[np.ndarray, dict]:
    low, high = normalization_limits(array)
    output = np.empty(array.shape, dtype=np.uint8)
    scale = 255.0 / (high - low)
    for start in range(0, array.shape[0], 16):
        block = (array[start : start + 16].astype(np.float32) - low) * scale
        output[start : start + 16] = np.clip(block, 0, 255).astype(np.uint8)
    return output, {"percentile_low": low, "percentile_high": high}


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def pad_spatial(array: np.ndarray, tile: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = array.shape[-2:]
    pad_h, pad_w = max(0, tile - height), max(0, tile - width)
    if pad_h or pad_w:
        pad = [(0, 0)] * array.ndim
        pad[-2] = (0, pad_h)
        pad[-1] = (0, pad_w)
        array = np.pad(array, pad, mode="reflect")
    return array, (height, width)


def pad_to_factor(tensor: torch.Tensor, factor: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = tensor.shape[-2:]
    pad_h, pad_w = (-height) % factor, (-width) % factor
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect"), (height, width)


def load_mitonet_probability_model(checkpoint: Path, device: torch.device):
    return torch.jit.load(str(checkpoint), map_location=device).to(device).eval()


def mitonet_probability_slice(model, raw: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(raw[None, None].astype(np.float32) / 255.0)
    tensor = ((tensor - 0.57571) / 0.12765).to(device)
    padded, shape = pad_to_factor(tensor)
    with torch.inference_mode():
        probability = torch.sigmoid(model(padded)["sem_logits"])[..., : shape[0], : shape[1]]
    return probability[0, 0].cpu().numpy().astype(np.float32)


def load_omniem(
    runtime: Path,
    root: Path,
    device: torch.device,
):
    # runtime: a local copy of the OmniEM source; root: the official OmniEM release (configs/, weights/)
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from omniem import OmniEM

    config = root / "configs/model_mito-seg-ViT-L-2D.yaml"
    backbone = root / "weights/backbone_emdino_v1.pt"
    head = root / "weights/head_mito-seg-ViT-L-2D.pt"
    model = OmniEM.load(config, backbone=backbone, head=head, device=device).eval()
    return model, {"config": config, "backbone": backbone, "head": head}


def omniem_probability_slice(
    model,
    raw: np.ndarray,
    device: torch.device,
    *,
    tile: int = 224,
    overlap: int = 56,
    batch_size: int = 16,
) -> np.ndarray:
    raw_padded, original = pad_spatial(raw[None], tile)
    height, width = raw_padded.shape[-2:]
    y_starts = tile_starts(height, tile, overlap)
    x_starts = tile_starts(width, tile, overlap)
    axis = np.hanning(tile).astype(np.float32)
    weight = np.maximum(np.outer(axis, axis), 0.05)
    summed = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    patches: list[np.ndarray] = []
    coordinates: list[tuple[int, int]] = []

    def flush() -> None:
        if not patches:
            return
        tensor = torch.from_numpy(np.stack(patches).astype(np.float32) / 255.0).to(device)
        with torch.inference_mode():
            logits = model.run(
                tensor,
                axes="byx",
                norm=None,
                conform="strict",
                return_logits=True,
            )
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        for probability, (y, x) in zip(probabilities, coordinates, strict=True):
            summed[y : y + tile, x : x + tile] += probability * weight
            weights[y : y + tile, x : x + tile] += weight
        patches.clear()
        coordinates.clear()

    for y in y_starts:
        for x in x_starts:
            patches.append(raw_padded[0, y : y + tile, x : x + tile])
            coordinates.append((y, x))
            if len(patches) == batch_size:
                flush()
    flush()
    result = summed / np.maximum(weights, 1e-6)
    return result[: original[0], : original[1]]


def candidate_probability_slice(
    model,
    config: dict,
    raw_context: np.ndarray,
    mitonet_probability: np.ndarray,
    omniem_probability: np.ndarray,
    acquisition_class: int,
    device: torch.device,
    *,
    tile: int = 256,
    overlap: int = 64,
    batch_size: int = 8,
) -> tuple[np.ndarray, float]:
    raw_padded, original = pad_spatial(raw_context, tile)
    mitonet_padded, _ = pad_spatial(mitonet_probability[None], tile)
    omniem_padded, _ = pad_spatial(omniem_probability[None], tile)
    height, width = raw_padded.shape[-2:]
    y_starts = tile_starts(height, tile, overlap)
    x_starts = tile_starts(width, tile, overlap)
    axis = np.hanning(tile).astype(np.float32)
    weight = np.maximum(np.outer(axis, axis), 0.05)
    summed = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    gate_sum = 0.0
    gate_weight = 0.0
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    coordinates: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal gate_sum, gate_weight
        if not batches:
            return
        raw_tensor = torch.from_numpy(
            np.stack([value[0] for value in batches]).astype(np.float32) / 255.0
        ).to(device)
        mitonet_tensor = torch.from_numpy(
            np.stack([value[1] for value in batches]).astype(np.float32)[:, None]
        ).to(device)
        omniem_tensor = torch.from_numpy(
            np.stack([value[2] for value in batches]).astype(np.float32)[:, None]
        ).to(device)
        acquisition = torch.full(
            (len(batches),), int(acquisition_class), dtype=torch.long, device=device
        )
        with torch.inference_mode():
            output = model(raw_tensor, mitonet_tensor, omniem_tensor, acquisition)
            probabilities = torch.sigmoid(output["region"])[:, 0].cpu().numpy()
            gates = output["prior_gate"][:, 0].cpu().numpy()
        for probability, gate, (y, x) in zip(probabilities, gates, coordinates, strict=True):
            summed[y : y + tile, x : x + tile] += probability * weight
            weights[y : y + tile, x : x + tile] += weight
            gate_sum += float((gate * weight).sum())
            gate_weight += float(weight.sum())
        batches.clear()
        coordinates.clear()

    for y in y_starts:
        for x in x_starts:
            batches.append(
                (
                    raw_padded[:, y : y + tile, x : x + tile],
                    mitonet_padded[0, y : y + tile, x : x + tile],
                    omniem_padded[0, y : y + tile, x : x + tile],
                )
            )
            coordinates.append((y, x))
            if len(batches) == batch_size:
                flush()
    flush()
    probability = summed / np.maximum(weights, 1e-6)
    return probability[: original[0], : original[1]], gate_sum / max(1e-6, gate_weight)


def blank_counts() -> dict[str, int]:
    return defaultdict(int)


def update_counts(counts: dict, prediction: np.ndarray, reference: np.ndarray) -> None:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    tp, fp, fn = semantic_counts(prediction, reference)
    bpm, bpt, brm, brt = boundary_counts(prediction, reference, tolerance=2)
    counts["tp"] += tp
    counts["fp"] += fp
    counts["fn"] += fn
    counts["pred_fg"] += int(prediction.sum())
    counts["ref_fg"] += int(reference.sum())
    counts["n"] += int(reference.size)
    counts["boundary_pred_match"] += bpm
    counts["boundary_pred_total"] += bpt
    counts["boundary_ref_match"] += brm
    counts["boundary_ref_total"] += brt


def summarize_counts(counts: dict) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    denominator = 2 * tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2 * tp / denominator if denominator else 1.0
    boundary_precision = counts["boundary_pred_match"] / max(1, counts["boundary_pred_total"])
    boundary_recall = counts["boundary_ref_match"] / max(1, counts["boundary_ref_total"])
    boundary_f1 = (
        2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall
        else 0.0
    )
    return {
        "semantic_dice": float(dice),
        "semantic_precision": float(precision),
        "semantic_recall": float(recall),
        "boundary_f1_tolerance2_px": float(boundary_f1),
        "absolute_foreground_fraction_error": float(
            abs(counts["pred_fg"] - counts["ref_fg"]) / max(1, counts["n"])
        ),
        "predicted_foreground_fraction": float(counts["pred_fg"] / max(1, counts["n"])),
        "reference_foreground_fraction": float(counts["ref_fg"] / max(1, counts["n"])),
    }


def resize_labels_nearest(labels: np.ndarray, output_shape: tuple[int, int, int]) -> np.ndarray:
    if labels.shape == output_shape:
        return labels.copy()
    indices = []
    for input_n, output_n in zip(labels.shape, output_shape):
        coordinates = (np.arange(output_n, dtype=np.float64) + 0.5) * input_n / output_n - 0.5
        indices.append(np.clip(np.rint(coordinates).astype(np.int64), 0, input_n - 1))
    return labels[np.ix_(*indices)]


def mitonet_panoptic_instances(
    raw_native: np.ndarray,
    spacing_nm_zyx: tuple[float, float, float],
    headless_model,
    headless_module,
) -> tuple[np.ndarray, dict]:
    native_shape = tuple(int(value) for value in raw_native.shape)
    common_shape = tuple(
        max(1, int(round(size * spacing / target)))
        for size, spacing, target in zip(
            native_shape, spacing_nm_zyx, TARGET_INSTANCE_SPACING_NM_ZYX
        )
    )
    common = resize(
        raw_native,
        common_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    common = np.clip(np.rint(common), 0, 255).astype(np.uint8)
    trackers = headless_module.infer_on_axis(
        headless_model,
        common,
        "xy",
        min_size=50,
        min_extent=2,
        confidence_thr=0.3,
        nms_threshold=0.1,
    )
    labels = headless_module.stack_label_volume(
        trackers, common.shape, min_size=50, min_extent=2
    )
    native = resize_labels_nearest(labels, native_shape).astype(np.uint32, copy=False)
    return native, {
        "common_shape_zyx": list(common_shape),
        "common_spacing_nm_zyx": list(TARGET_INSTANCE_SPACING_NM_ZYX),
        "common_instances": int(labels.max(initial=0)),
        "native_instances": int(native.max(initial=0)),
    }


def load_headless_mitonet(hub: Path):
    path = hub / "datasets/Mitochondria/mito_phenomics/scripts/mitonet_headless.py"
    spec = importlib.util.spec_from_file_location("mitocensus_mitonet_headless", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_mitonet("cpu"), module, path


def load_public_instance_evaluator(hub: Path):
    path = hub / "datasets/Mitochondria/mito_phenomics/scripts/20_mitoem2_evaluate_saved_predictions.py"
    spec = importlib.util.spec_from_file_location("mitocensus_instance_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path
