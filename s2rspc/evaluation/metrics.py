"""Binary gauge-line segmentation metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


def binary_counts(target: np.ndarray, prediction: np.ndarray) -> Tuple[int, int, int, int]:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if target.shape != prediction.shape:
        raise ValueError(f"Target and prediction shapes differ: {target.shape} vs {prediction.shape}")
    tp = int(np.sum(target & prediction))
    fp = int(np.sum(~target & prediction))
    fn = int(np.sum(target & ~prediction))
    tn = int(np.sum(~target & ~prediction))
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int = 0) -> Dict[str, float | int]:
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float | int]:
    return metrics_from_counts(*binary_counts(target, prediction))


def select_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    grid: Iterable[float],
) -> Tuple[float, Dict[str, float | int]]:
    """Select by IoU, then F1, then the lower threshold, using source validation only."""
    best = None
    for threshold in grid:
        result = binary_metrics(target, np.asarray(probability) >= float(threshold))
        key = (float(result["iou"]), float(result["f1"]), -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), result)
    if best is None:
        raise ValueError("Threshold grid is empty")
    return best[1], best[2]


def select_macro_threshold(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    grid: Iterable[float],
) -> Tuple[float, Dict[str, float]]:
    """Select a source-only threshold by mean per-cloud foreground IoU."""
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("Probabilities and targets must contain the same nonzero number of clouds")
    best = None
    for threshold in grid:
        results = [
            binary_metrics(target, np.asarray(probability) >= float(threshold))
            for probability, target in zip(probabilities, targets)
        ]
        summary = {
            "miou": float(np.mean([float(result["iou"]) for result in results])),
            "mean_precision": float(
                np.mean([float(result["precision"]) for result in results])
            ),
            "mean_recall": float(
                np.mean([float(result["recall"]) for result in results])
            ),
            "mean_f1": float(np.mean([float(result["f1"]) for result in results])),
        }
        key = (summary["miou"], summary["mean_f1"], -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), summary)
    if best is None:
        raise ValueError("Threshold grid is empty")
    return best[1], best[2]
