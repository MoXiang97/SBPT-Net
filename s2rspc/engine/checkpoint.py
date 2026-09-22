"""Checkpoint contract for the paper-aligned SBPT-Net release."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from ..config import ModelConfig
from ..models.sbptnet import ARCHITECTURE_ID, SBPTNet
from ..preprocessing.tokenization import (
    STRUCTURAL_DESCRIPTOR_NAMES,
    TOKEN_ATTRIBUTE_NAMES,
    TOKEN_CACHE_SCHEMA_VERSION,
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def checkpoint_payload(
    model: SBPTNet,
    attribute_mean: np.ndarray,
    attribute_std: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "architecture_id": ARCHITECTURE_ID,
        "token_cache_schema_version": TOKEN_CACHE_SCHEMA_VERSION,
        "model_config": asdict(model.cfg),
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "descriptor_feature_names": list(STRUCTURAL_DESCRIPTOR_NAMES),
        "token_attribute_names": list(TOKEN_ATTRIBUTE_NAMES),
        "attribute_mean": np.asarray(attribute_mean, dtype=np.float32),
        "attribute_std": np.asarray(attribute_std, dtype=np.float32),
        "fusion": {
            "beta": float(model.cfg.fusion_beta),
            "clip_bound": float(model.cfg.fusion_clip_bound),
        },
        "deviation_clip_bound": float(model.cfg.deviation_clip_bound),
        "metadata": dict(metadata or {}),
    }


def save_checkpoint(
    path: str | Path,
    model: SBPTNet,
    attribute_mean: np.ndarray,
    attribute_std: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, attribute_mean, attribute_std, metadata)
    torch.save(payload, path)
    return payload


def load_checkpoint(
    path: str | Path,
    model_config: ModelConfig,
    device: str | torch.device = "cpu",
) -> Tuple[SBPTNet, np.ndarray, np.ndarray, Dict[str, Any]]:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    observed = checkpoint.get("architecture_id")
    if observed != ARCHITECTURE_ID:
        raise RuntimeError(
            f"Checkpoint architecture {observed!r} does not match "
            f"the paper-aligned architecture {ARCHITECTURE_ID!r}"
        )
    if (
        int(checkpoint.get("token_cache_schema_version", -1))
        != TOKEN_CACHE_SCHEMA_VERSION
    ):
        raise RuntimeError("Checkpoint expects an incompatible token-cache schema")
    if checkpoint.get("descriptor_feature_names") != list(STRUCTURAL_DESCRIPTOR_NAMES):
        raise RuntimeError("Checkpoint descriptor feature order is incompatible")
    if checkpoint.get("token_attribute_names") != list(TOKEN_ATTRIBUTE_NAMES):
        raise RuntimeError("Checkpoint token-attribute order is incompatible")

    stored = checkpoint.get("model_config", {})
    required = {
        "pointmlp_input_dim": model_config.pointmlp_input_dim,
        "structural_dim": model_config.structural_dim,
        "structural_hidden_dim": model_config.structural_hidden_dim,
        "fusion_beta": model_config.fusion_beta,
        "fusion_clip_bound": model_config.fusion_clip_bound,
        "deviation_clip_bound": model_config.deviation_clip_bound,
    }
    for key, expected in required.items():
        if key in stored and stored[key] != expected:
            raise RuntimeError(
                f"Checkpoint model_config[{key!r}]={stored[key]!r} "
                f"does not match {expected!r}"
            )

    model = SBPTNet(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(device)).eval()
    attribute_mean = np.asarray(checkpoint["attribute_mean"], dtype=np.float32).reshape(
        6
    )
    attribute_std = np.asarray(checkpoint["attribute_std"], dtype=np.float32).reshape(6)
    return model, attribute_mean, attribute_std, checkpoint
