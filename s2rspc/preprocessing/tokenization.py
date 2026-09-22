"""Superline support regions and the paper-aligned point-token cache."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..config import TokenizationConfig
from .lcc import normalize_rgb
from .superline import pca_stats


TOKEN_CACHE_SCHEMA_VERSION = 2

DOWNSAMPLED_FEATURE_NAMES = (
    "bin_num_points",
    "bin_length",
    "bin_pca_width",
    "bin_width_q",
    "bin_linearness",
    "bin_density_len",
    "frag_num_points",
    "log_frag_num_points",
    "frag_length",
    "frag_width",
    "frag_linearness",
    "frag_density_len",
    "frag_length_width_ratio",
    "token_order_norm",
    "base_spacing",
    "bin_size",
    "bin_width_to_frag_width",
    "bin_density_to_frag_density",
)

STRUCTURAL_DESCRIPTOR_NAMES = (
    "local_linearity",
    "superline_linearity",
    "superline_length_to_width",
    "normalized_token_order",
    "local_width_to_superline_width",
    "local_density_to_superline_density",
)

STRUCTURAL_DESCRIPTOR_SOURCE_NAMES = (
    "bin_linearness",
    "frag_linearness",
    "frag_length_width_ratio",
    "token_order_norm",
    "bin_width_to_frag_width",
    "bin_density_to_frag_density",
)

TOKEN_ATTRIBUTE_NAMES = (
    "local_color_contrast",
    "local_length_over_spacing",
    "local_width_over_spacing",
    "superline_width_over_spacing",
    "superline_length_over_spacing",
    "local_point_density",
)


def _principal_axis(points: np.ndarray) -> np.ndarray:
    axis = np.asarray(pca_stats(points)["tangent"], dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return axis / norm


def _local_width(
    points: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    quantile: float,
) -> float:
    if len(points) <= 1:
        return 0.0
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
    relative = points - center[None, :]
    axial = relative @ tangent
    perpendicular = relative - np.outer(axial, tangent)
    distance = np.linalg.norm(perpendicular, axis=1)
    return float(np.quantile(distance, quantile)) if len(distance) else 0.0


def _observed_representative(
    points: np.ndarray,
    original_index: np.ndarray,
    tangent: np.ndarray,
    center_weight: float,
    perpendicular_weight: float,
) -> Tuple[np.ndarray, int, float]:
    center = points.mean(axis=0).astype(np.float32)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
    relative = points - center[None, :]
    axial = relative @ tangent
    perpendicular = relative - np.outer(axial, tangent)
    center_distance = np.linalg.norm(relative, axis=1)
    perpendicular_distance = np.linalg.norm(perpendicular, axis=1)
    score = (
        center_weight * center_distance + perpendicular_weight * perpendicular_distance
    )
    selected = int(np.argmin(score))
    return (
        points[selected].astype(np.float32),
        int(original_index[selected]),
        float(center_distance[selected]),
    )


def _support_regions(
    points: np.ndarray,
    original_index: np.ndarray,
    radius: float,
    stride: float,
    minimum_points: int,
) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    if len(points) <= max(1, int(minimum_points)):
        return [np.arange(len(points), dtype=np.int64)]
    tree = cKDTree(points)
    suppressed = np.zeros(len(points), dtype=bool)
    regions: List[np.ndarray] = []
    for seed in np.argsort(original_index, kind="mergesort"):
        seed = int(seed)
        if suppressed[seed]:
            continue
        neighborhood = np.asarray(
            tree.query_ball_point(points[seed], r=max(radius, 1e-8)),
            dtype=np.int64,
        )
        if len(neighborhood) == 0:
            neighborhood = np.asarray([seed], dtype=np.int64)
        regions.append(np.sort(neighborhood))
        close = np.asarray(
            tree.query_ball_point(points[seed], r=max(stride, 1e-8)),
            dtype=np.int64,
        )
        suppressed[close] = True
    return regions


def _compact_descriptors(features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    indices = [
        DOWNSAMPLED_FEATURE_NAMES.index(name)
        for name in STRUCTURAL_DESCRIPTOR_SOURCE_NAMES
    ]
    compact = np.asarray(features[:, indices], dtype=np.float32).copy()
    for column, name in enumerate(STRUCTURAL_DESCRIPTOR_SOURCE_NAMES):
        if name not in ("bin_linearness", "frag_linearness", "token_order_norm"):
            compact[:, column] = np.log1p(np.maximum(compact[:, column], 0.0))
    return np.nan_to_num(compact).astype(np.float32)


def _token_attributes(features: np.ndarray, lcc: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    spacing = np.maximum(features[:, 14], 1e-6)
    values = np.stack(
        [
            lcc,
            np.log1p(features[:, 1] / spacing),
            np.log1p(features[:, 2] / spacing),
            np.log1p(features[:, 9] / spacing),
            np.log1p(features[:, 8] / spacing),
            np.log1p(features[:, 0] * spacing / np.maximum(features[:, 1], 1e-6)),
        ],
        axis=1,
    )
    return np.nan_to_num(values).astype(np.float32)


def tokenize_superlines(
    candidate: Dict[str, np.ndarray],
    superlines: Dict[str, np.ndarray],
    cfg: TokenizationConfig,
) -> Dict[str, np.ndarray]:
    xyz = np.asarray(candidate["xyz"], dtype=np.float32)
    rgb = normalize_rgb(candidate["rgb"])
    label = np.asarray(candidate["label"], dtype=np.int64)
    lcc = np.asarray(candidate["lcc"], dtype=np.float32)
    original_index = np.asarray(candidate["orig_index"], dtype=np.int64)
    fragment_id = np.asarray(superlines["fragment_id"], dtype=np.int32)
    base_spacing = float(np.asarray(superlines["base_spacing"]).item())
    support_radius = max(cfg.support_radius_factor * base_spacing, base_spacing)
    seed_stride = max(cfg.seed_stride_factor * base_spacing, base_spacing)

    output: Dict[str, List] = {
        "xyz": [],
        "rgb": [],
        "label": [],
        "label_ratio": [],
        "lcc": [],
        "orig_index": [],
        "fragment_id": [],
        "features": [],
        "bin_num_points": [],
        "rep_anchor_dist": [],
        "members": [],
    }

    for current_fragment in sorted(
        int(value) for value in np.unique(fragment_id) if value >= 0
    ):
        fragment_candidate_index = np.where(fragment_id == current_fragment)[0]
        if len(fragment_candidate_index) == 0:
            continue
        fragment_points = xyz[fragment_candidate_index]
        fragment_stats = pca_stats(fragment_points)
        fragment_count = float(len(fragment_candidate_index))
        fragment_length = float(fragment_stats["length"])
        fragment_width = float(fragment_stats["width"])
        fragment_linearity = float(fragment_stats["linearness"])
        fragment_density = fragment_count / max(fragment_length, 1e-6)
        fragment_ratio = fragment_length / (fragment_width + 1e-6)
        regions = _support_regions(
            fragment_points,
            original_index[fragment_candidate_index],
            support_radius,
            seed_stride,
            cfg.minimum_support_points,
        )
        denominator = max(len(regions) - 1, 1)
        for region_position, local_region_index in enumerate(regions):
            candidate_index = fragment_candidate_index[local_region_index]
            region_xyz = xyz[candidate_index]
            if len(region_xyz) < cfg.minimum_support_points:
                continue
            tangent = _principal_axis(region_xyz)
            representative, representative_orig, anchor_distance = (
                _observed_representative(
                    region_xyz,
                    original_index[candidate_index],
                    tangent,
                    cfg.representative_center_weight,
                    cfg.representative_perpendicular_weight,
                )
            )
            label_ratio = float(np.mean(label[candidate_index]))
            region_stats = pca_stats(region_xyz)
            local_length = float(region_stats["length"])
            local_pca_width = float(region_stats["width"])
            local_linearity = float(region_stats["linearness"])
            local_width = _local_width(
                region_xyz,
                representative,
                tangent,
                cfg.local_width_quantile,
            )
            local_density = float(
                len(region_xyz) / max(local_length, base_spacing, 1e-6)
            )
            features = [
                float(len(region_xyz)),
                local_length,
                local_pca_width,
                local_width,
                local_linearity,
                local_density,
                fragment_count,
                math.log1p(fragment_count),
                fragment_length,
                fragment_width,
                fragment_linearity,
                fragment_density,
                fragment_ratio,
                float(region_position / denominator),
                base_spacing,
                support_radius,
                local_width / (fragment_width + 1e-6),
                local_density / (fragment_density + 1e-6),
            ]
            output["xyz"].append(representative)
            output["rgb"].append(rgb[candidate_index].mean(axis=0))
            output["label"].append(int(label_ratio >= cfg.label_positive_ratio))
            output["label_ratio"].append(label_ratio)
            output["lcc"].append(float(lcc[candidate_index].mean()))
            output["orig_index"].append(representative_orig)
            output["fragment_id"].append(current_fragment)
            output["features"].append(features)
            output["bin_num_points"].append(len(region_xyz))
            output["rep_anchor_dist"].append(anchor_distance)
            output["members"].append(candidate_index.astype(np.int64))

    token_count = len(output["label"])
    member_offsets = np.zeros(token_count + 1, dtype=np.int64)
    if token_count:
        member_offsets[1:] = np.cumsum([len(members) for members in output["members"]])
        member_indices = np.concatenate(output["members"]).astype(np.int64)
    else:
        member_indices = np.zeros(0, dtype=np.int64)

    features = np.asarray(output["features"], dtype=np.float32).reshape(
        -1, len(DOWNSAMPLED_FEATURE_NAMES)
    )
    token_lcc = np.asarray(output["lcc"], dtype=np.float32)
    result = {
        "schema_version": np.asarray(TOKEN_CACHE_SCHEMA_VERSION, dtype=np.int16),
        "xyz": np.asarray(output["xyz"], dtype=np.float32).reshape(-1, 3),
        "rgb": np.asarray(output["rgb"], dtype=np.float32).reshape(-1, 3),
        "label": np.asarray(output["label"], dtype=np.int64),
        "label_ratio": np.asarray(output["label_ratio"], dtype=np.float32),
        "lcc": token_lcc,
        "orig_index": np.asarray(output["orig_index"], dtype=np.int64),
        "fragment_id": np.asarray(output["fragment_id"], dtype=np.int32),
        "features": features,
        "structural_descriptors": _compact_descriptors(features),
        "token_attributes": _token_attributes(features, token_lcc),
        "bin_num_points": np.asarray(output["bin_num_points"], dtype=np.int32),
        "rep_anchor_dist": np.asarray(output["rep_anchor_dist"], dtype=np.float32),
        "patch_member_offsets": member_offsets,
        "patch_member_indices": member_indices,
        "candidate_orig_index": original_index.astype(np.int64),
        "candidate_label": label.astype(np.int64),
        "raw_count": np.asarray(
            candidate.get("raw_count", int(original_index.max(initial=-1)) + 1),
            dtype=np.int64,
        ),
    }
    if "raw_label" in candidate:
        result["raw_label"] = np.asarray(candidate["raw_label"], dtype=np.int64)
    return result
