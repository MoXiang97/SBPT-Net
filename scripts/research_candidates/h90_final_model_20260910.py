"""Structural head with line descriptors and normalized token attributes.

The model uses six line-structure descriptors and six normalized token
attributes as inputs.
"""
from __future__ import annotations

import math
import random
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

import h90_slim_no_patch_20260910 as Legacy

T = Legacy.T

TRAINABLE_ARMS = (
    "full",
    "without_line_descriptors",
    "without_token_attributes",
    "without_superline_context",
    "without_consistency_loss",
)


class FinalStructuralHead(nn.Module):
    """Line-structure refinement with physically removable components."""

    descriptor_ids = (4, 9, 11, 12, 14, 15)
    attribute_ids = tuple(range(6))

    def __init__(self, config: Mapping[str, object], arm: str = "full"):
        super().__init__()
        if arm not in TRAINABLE_ARMS:
            raise ValueError(arm)
        self.cfg = dict(config)
        self.arm = arm
        self.use_descriptors = arm != "without_line_descriptors"
        self.use_attributes = arm != "without_token_attributes"
        self.use_context = arm != "without_superline_context"
        input_width = 6 * int(self.use_descriptors) + 6 * int(self.use_attributes)
        if input_width == 0:
            raise ValueError("At least one structural feature group is required")

        dropout = float(self.cfg.get("head_dropout", 0.1))
        self.embed = nn.Sequential(
            nn.Linear(input_width, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout)
        )
        if self.use_context:
            self.nbr_embed = nn.Sequential(
                nn.Linear(input_width, 96), nn.LayerNorm(96), nn.GELU()
            )
            self.q = nn.Linear(96, 48)
            self.k = nn.Linear(96, 48)
            self.v = nn.Linear(96, 96)
            self.context = nn.Sequential(
                nn.Linear(192, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout)
            )
        self.out = nn.Linear(96, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _center_features(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        if self.use_descriptors:
            parts.append(batch["d"][:, self.descriptor_ids])
        if self.use_attributes:
            parts.append(batch["extra"][:, self.attribute_ids])
        return torch.cat(parts, dim=-1)

    def _neighbor_features(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        if self.use_descriptors:
            parts.append(batch["nd"][:, :, self.descriptor_ids])
        if self.use_attributes:
            parts.append(batch["nextra"][:, :, self.attribute_ids])
        return torch.cat(parts, dim=-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        base = batch["logit"].clamp(-10, 10)
        hidden = self.embed(self._center_features(batch))
        if self.use_context:
            neighbor = self.nbr_embed(self._neighbor_features(batch))
            score = (self.q(hidden)[:, None] * self.k(neighbor)).sum(-1) / math.sqrt(48)
            score = score.masked_fill(~batch["nm"], -1e4)
            context = (score.softmax(-1)[..., None] * self.v(neighbor)).sum(1)
            hidden = self.context(torch.cat((hidden, context), dim=-1))
        correction = self.out(hidden).squeeze(-1)
        self.aux_logit = correction
        alpha = float(self.cfg.get("geometry_weight", 0.5))
        limit = float(self.cfg.get("base_logit_cap", 10.0))
        return (1.0 - alpha) * base.clamp(-limit, limit) + alpha * correction


def _legacy_columns(model: FinalStructuralHead):
    # Legacy order: six descriptors, one always-zero evidence value, six attributes.
    columns = []
    if model.use_descriptors:
        columns.extend(range(6))
    if model.use_attributes:
        columns.extend(range(7, 13))
    return columns


def copy_from_legacy_state(
    model: FinalStructuralHead, legacy_state: Mapping[str, torch.Tensor]
) -> None:
    """Copy compatible weights while deleting inactive legacy columns/modules."""
    target_state = model.state_dict()
    copied = {}
    columns = _legacy_columns(model)
    for key, target in target_state.items():
        source = legacy_state[key]
        if key in ("embed.0.weight", "nbr_embed.0.weight"):
            source = source[:, columns]
        if source.shape != target.shape:
            raise RuntimeError(f"Cannot convert {key}: {source.shape} -> {target.shape}")
        copied[key] = source.detach().clone()
    model.load_state_dict(copied, strict=True)


def matched_initial_model(
    config: Mapping[str, object], arm: str, seed: int, device: torch.device
) -> FinalStructuralHead:
    """Create each arm from exactly the same compatible legacy initialization."""
    legacy = Legacy.matched_initial_model(config, "slim_no_patch", seed, torch.device("cpu"))
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    model = FinalStructuralHead(config, arm)
    copy_from_legacy_state(model, legacy.state_dict())
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    del legacy
    return model.to(device)


def training_config(config: Mapping[str, object], arm: str):
    if arm not in TRAINABLE_ARMS:
        raise ValueError(arm)
    result = dict(config)
    if arm == "without_consistency_loss":
        result["consistency_weight"] = 0.0
    return result


def parameter_count(arm: str) -> int:
    config = T.candidate_config(42, T.BASE_CHECKPOINTS[42])
    return sum(p.numel() for p in FinalStructuralHead(config, arm).parameters())
