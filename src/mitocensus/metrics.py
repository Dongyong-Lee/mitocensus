from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage as ndi


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def semantic_counts(prediction: np.ndarray, reference: np.ndarray) -> tuple[int, int, int]:
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    return (
        int(np.logical_and(prediction, reference).sum()),
        int(np.logical_and(prediction, ~reference).sum()),
        int(np.logical_and(~prediction, reference).sum()),
    )


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    return mask ^ ndi.binary_erosion(mask, structure=structure, border_value=0)


def boundary_counts(prediction: np.ndarray, reference: np.ndarray, tolerance: int = 2) -> tuple[int, int, int, int]:
    pred_boundary = boundary_mask(prediction)
    ref_boundary = boundary_mask(reference)
    pred_match = np.logical_and(
        pred_boundary, ndi.binary_dilation(ref_boundary, iterations=tolerance)
    ).sum()
    ref_match = np.logical_and(
        ref_boundary, ndi.binary_dilation(pred_boundary, iterations=tolerance)
    ).sum()
    return int(pred_match), int(pred_boundary.sum()), int(ref_match), int(ref_boundary.sum())


def summarize_counts(values: dict[str, float]) -> dict[str, float]:
    tp, fp, fn = values["tp"], values["fp"], values["fn"]
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    boundary_precision = safe_divide(values["boundary_pred_match"], values["boundary_pred_total"])
    boundary_recall = safe_divide(values["boundary_ref_match"], values["boundary_ref_total"])
    boundary_f1 = safe_divide(
        2 * boundary_precision * boundary_recall, boundary_precision + boundary_recall
    )
    foreground_error = safe_divide(abs((tp + fp) - (tp + fn)), tp + fn)
    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "boundary_f1_tolerance2_px": boundary_f1,
        "absolute_foreground_error": foreground_error,
    }


def aggregate_patch_predictions(
    probabilities: np.ndarray,
    references: np.ndarray,
    volumes: list[str],
    threshold: float,
) -> tuple[list[dict], dict]:
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for probability, reference, volume in zip(probabilities, references, volumes):
        prediction = np.asarray(probability) >= threshold
        reference = np.asarray(reference, dtype=bool)
        tp, fp, fn = semantic_counts(prediction, reference)
        bpm, bpt, brm, brt = boundary_counts(prediction, reference, tolerance=2)
        current = grouped[volume]
        current["tp"] += tp
        current["fp"] += fp
        current["fn"] += fn
        current["boundary_pred_match"] += bpm
        current["boundary_pred_total"] += bpt
        current["boundary_ref_match"] += brm
        current["boundary_ref_total"] += brt
        current["n_patches"] += 1
    volume_rows = []
    for volume, counts in sorted(grouped.items()):
        volume_rows.append({"volume": volume, **summarize_counts(counts), "n_patches": int(counts["n_patches"])})
    summary = {
        "n_volumes": len(volume_rows),
        "volume_macro_dice": float(np.mean([row["dice"] for row in volume_rows])),
        "volume_macro_boundary_f1": float(np.mean([row["boundary_f1_tolerance2_px"] for row in volume_rows])),
        "volume_macro_absolute_foreground_error": float(
            np.mean([row["absolute_foreground_error"] for row in volume_rows])
        ),
        "worst_volume_dice": float(min(row["dice"] for row in volume_rows)),
    }
    return volume_rows, summary
