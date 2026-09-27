"""Physical single-component heads for the final-model Seed-42 diagnostic."""
from __future__ import annotations

import math
import random
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

import h90_final_model_20260910 as Final


T = Final.T

ARMS = (
    "line_descriptors_only",
    "token_attributes_only",
    "superline_context_only",
    "superline_context_only_no_consistency",
    "token_encoding_only_no_consistency",
)

CONTEXT_ARMS = (
    "superline_context_only",
    "superline_context_only_no_consistency",
)


class SingleComponentHead(nn.Module):
    """Fuse the frozen initial logit with exactly one named information path."""

    descriptor_ids = Final.FinalStructuralHead.descriptor_ids
    attribute_ids = Final.FinalStructuralHead.attribute_ids

    def __init__(self, config: Mapping[str, object], arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.cfg = dict(config)
        self.arm = arm
        dropout = float(self.cfg.get("head_dropout", 0.1))
        if arm in CONTEXT_ARMS:
            input_width = 1
        elif arm == "token_encoding_only_no_consistency":
            input_width = 12
        else:
            input_width = 6
        self.embed = nn.Sequential(
            nn.Linear(input_width, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout)
        )
        if arm in CONTEXT_ARMS:
            self.nbr_embed = nn.Sequential(
                nn.Linear(1, 96), nn.LayerNorm(96), nn.GELU()
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
        if self.arm == "line_descriptors_only":
            return batch["d"][:, self.descriptor_ids]
        if self.arm == "token_attributes_only":
            return batch["extra"][:, self.attribute_ids]
        if self.arm == "token_encoding_only_no_consistency":
            return torch.cat(
                (
                    batch["d"][:, self.descriptor_ids],
                    batch["extra"][:, self.attribute_ids],
                ),
                dim=-1,
            )
        return batch["logit"].clamp(-10, 10)[:, None] / 5.0

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        base = batch["logit"].clamp(-10, 10)
        hidden = self.embed(self._center_features(batch))
        if self.arm in CONTEXT_ARMS:
            neighbor_input = batch["nl"].clamp(-10, 10)[..., None] / 5.0
            neighbor = self.nbr_embed(neighbor_input)
            score = (self.q(hidden)[:, None] * self.k(neighbor)).sum(-1) / math.sqrt(48)
            score = score.masked_fill(~batch["nm"], -1e4)
            context = (score.softmax(-1)[..., None] * self.v(neighbor)).sum(1)
            hidden = self.context(torch.cat((hidden, context), dim=-1))
        correction = self.out(hidden).squeeze(-1)
        self.aux_logit = correction
        alpha = float(self.cfg.get("geometry_weight", 0.5))
        limit = float(self.cfg.get("base_logit_cap", 10.0))
        return (1.0 - alpha) * base.clamp(-limit, limit) + alpha * correction


def _copy_center_only_from_full(
    target: SingleComponentHead, full: Final.FinalStructuralHead
) -> None:
    state = target.state_dict()
    full_state = full.state_dict()
    columns = range(6) if target.arm == "line_descriptors_only" else range(6, 12)
    copied = {}
    for key, value in state.items():
        source = full_state[key]
        if key == "embed.0.weight":
            source = source[:, list(columns)]
        if source.shape != value.shape:
            raise RuntimeError(f"Cannot initialize {key}: {source.shape} -> {value.shape}")
        copied[key] = source.detach().clone()
    target.load_state_dict(copied, strict=True)


def matched_initial_model(
    config: Mapping[str, object], arm: str, seed: int, device: torch.device
) -> SingleComponentHead:
    """Use compatible full-model columns or a deterministic logit-context seed."""
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == "token_encoding_only_no_consistency":
        full_config = T.candidate_config(seed, T.BASE_CHECKPOINTS[seed])
        return Final.matched_initial_model(
            full_config, "without_superline_context", seed, device
        )
    full_config = T.candidate_config(seed, T.BASE_CHECKPOINTS[seed])
    full = Final.matched_initial_model(full_config, "full", seed, torch.device("cpu"))
    paired_py_state = random.getstate()
    paired_np_state = np.random.get_state()
    paired_cpu_state = torch.get_rng_state()
    paired_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        if arm in CONTEXT_ARMS:
            T.set_seed(int(seed) + 91100)
            model = SingleComponentHead(config, arm)
        else:
            model = SingleComponentHead(config, arm)
            _copy_center_only_from_full(model, full)
    finally:
        random.setstate(paired_py_state)
        np.random.set_state(paired_np_state)
        torch.set_rng_state(paired_cpu_state)
        if paired_cuda_state is not None:
            torch.cuda.set_rng_state_all(paired_cuda_state)
        del full
    return model.to(device)


def training_config(config: Mapping[str, object], arm: str):
    """Request only the batch signals physically consumed by each arm."""
    if arm not in ARMS:
        raise ValueError(arm)
    result = dict(config)
    use_encoding = arm == "token_encoding_only_no_consistency"
    result["context"] = arm in CONTEXT_ARMS
    result["extra_features"] = (
        "all" if arm == "token_attributes_only" or use_encoding else False
    )
    result["descriptor_style_aug"] = arm == "line_descriptors_only" or use_encoding
    result["extra_style_aug"] = arm == "token_attributes_only" or use_encoding
    result["extra_feature_drop"] = (
        0.1 if arm == "token_attributes_only" or use_encoding else 0.0
    )
    if arm.endswith("_no_consistency"):
        result["consistency_weight"] = 0.0
    return result


def parameter_count(arm: str) -> int:
    config = T.candidate_config(42, T.BASE_CHECKPOINTS[42])
    if arm == "token_encoding_only_no_consistency":
        return Final.parameter_count("without_superline_context")
    return sum(parameter.numel() for parameter in SingleComponentHead(config, arm).parameters())
