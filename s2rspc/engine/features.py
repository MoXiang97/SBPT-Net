"""Build the six-channel PointMLP input and 12D structural features."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from ..config import ModelConfig
from ..preprocessing.lcc import normalize_rgb


def pointmlp_feature_matrix(tokenized: Dict[str, np.ndarray]) -> np.ndarray:
    xyz = np.asarray(tokenized["xyz"], dtype=np.float32).copy()
    rgb = normalize_rgb(np.asarray(tokenized["rgb"], dtype=np.float32))
    if len(xyz):
        xyz -= xyz.mean(axis=0, keepdims=True)
    return np.concatenate([xyz, np.clip(rgb, 0.0, 1.0)], axis=1).astype(np.float32)


def attribute_mean_std(
    tokenized_clouds: Iterable[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    values = [
        np.asarray(cloud["token_attributes"], dtype=np.float32)
        for cloud in tokenized_clouds
        if len(cloud["token_attributes"])
    ]
    if not values:
        return np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)
    train = np.concatenate(values, axis=0).astype(np.float64)
    mean = train.mean(axis=0)
    std = np.maximum(train.std(axis=0), 0.01)
    return mean.astype(np.float32), std.astype(np.float32)


def prepare_structural_features(
    tokenized: Dict[str, np.ndarray],
    attribute_mean: np.ndarray,
    attribute_std: np.ndarray,
) -> np.ndarray:
    descriptors = np.asarray(tokenized["structural_descriptors"], dtype=np.float32)
    attributes = np.asarray(tokenized["token_attributes"], dtype=np.float32)
    mean = np.asarray(attribute_mean, dtype=np.float32).reshape(1, 6)
    std = np.asarray(attribute_std, dtype=np.float32).reshape(1, 6)
    normalized_attributes = np.clip((attributes - mean) / std, -8.0, 8.0)
    return np.concatenate(
        [descriptors, normalized_attributes.astype(np.float32)], axis=1
    ).astype(np.float32)


def prepare_model_inputs(
    tokenized: Dict[str, np.ndarray],
    attribute_mean: np.ndarray,
    attribute_std: np.ndarray,
    cfg: ModelConfig,
) -> Dict[str, np.ndarray]:
    point_features = pointmlp_feature_matrix(tokenized)
    structural = prepare_structural_features(tokenized, attribute_mean, attribute_std)
    if point_features.shape[1] != cfg.pointmlp_input_dim:
        raise ValueError(
            f"Expected {cfg.pointmlp_input_dim} PointMLP channels, "
            f"got {point_features.shape}"
        )
    if structural.shape[1] != cfg.structural_dim:
        raise ValueError(
            f"Expected {cfg.structural_dim} structural features, "
            f"got {structural.shape}"
        )
    return {
        "point_features": point_features,
        "structural_features": structural,
        "label": np.asarray(tokenized["label"], dtype=np.float32),
    }


def tensors(
    inputs: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        "point_features": torch.from_numpy(inputs["point_features"].T[None]).to(device),
        "structural_features": torch.from_numpy(inputs["structural_features"][None]).to(
            device
        ),
        "label": torch.from_numpy(inputs["label"][None]).to(device),
    }
