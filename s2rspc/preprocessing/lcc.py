"""Synthetic-calibrated local color-contrast candidate preservation."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32).copy()
    if rgb.size and float(np.nanmax(rgb)) > 1.5:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def local_color_contrast(
    xyz: np.ndarray,
    rgb: np.ndarray,
    neighbors: int = 256,
    chunk_size: int = 50000,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb01 = normalize_rgb(rgb)
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.float32)
    k = min(max(1, int(neighbors)), len(xyz))
    tree = cKDTree(xyz)
    scores = np.empty(len(xyz), dtype=np.float32)
    for start in range(0, len(xyz), int(chunk_size)):
        end = min(start + int(chunk_size), len(xyz))
        try:
            _, index = tree.query(xyz[start:end], k=k, workers=-1)
        except TypeError:
            _, index = tree.query(xyz[start:end], k=k)
        if k == 1:
            index = index.reshape(-1, 1)
        neighbor_mean = rgb01[index].mean(axis=1)
        scores[start:end] = np.linalg.norm(rgb01[start:end] - neighbor_mean, axis=1)
    return scores


def preserve_lcc_candidates(
    raw: Dict[str, np.ndarray],
    threshold: float,
    neighbors: int = 256,
    chunk_size: int = 50000,
) -> Dict[str, np.ndarray]:
    xyz = np.asarray(raw["xyz"], dtype=np.float32)
    rgb = np.asarray(raw["rgb"], dtype=np.float32)
    label = np.asarray(raw["label"], dtype=np.int64)
    scores = local_color_contrast(xyz, rgb, neighbors, chunk_size)
    keep = scores >= float(threshold)
    return {
        "xyz": xyz[keep],
        "rgb": rgb[keep],
        "label": label[keep],
        "orig_index": np.flatnonzero(keep).astype(np.int64),
        "lcc": scores[keep].astype(np.float32),
        "raw_count": np.asarray(len(xyz), dtype=np.int64),
        "raw_label": label.astype(np.int64),
    }
