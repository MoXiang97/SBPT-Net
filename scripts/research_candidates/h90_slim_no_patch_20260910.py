"""Structural refiner without a local-patch encoder.

The forward pass accepts no patch tensor. A full-size donor supplies the
retained weights during initialization.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

import run_h90_component_ablation_20260910 as A

T, H = A.T, A.H
ARMS = ("slim_no_patch", "slim_without_order", "slim_no_descriptor")


class SlimNoPatchHead(nn.Module):
    """H90 line-aware refiner after physical deletion of the patch branch."""

    def __init__(self, config: Mapping[str, object], arm: str = "slim_no_patch"):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.cfg, self.arm = dict(config), arm
        self.dids = [4, 9, 11, 12, 14, 15]
        self.extra_ids = list(range(6))
        center_width = len(self.dids) + 1 + len(self.extra_ids)  # 6 descriptor + zero evidence + 6 extra
        dropout = float(self.cfg.get("head_dropout", 0.1))
        self.embed = nn.Sequential(nn.Linear(center_width, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout))
        self.nbr_embed = nn.Sequential(nn.Linear(center_width, 96), nn.LayerNorm(96), nn.GELU())
        self.q, self.k, self.v = nn.Linear(96, 48), nn.Linear(96, 48), nn.Linear(96, 96)
        self.context = nn.Sequential(nn.Linear(192, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(96, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _descriptor(self, tensor: torch.Tensor) -> torch.Tensor:
        value = tensor[..., self.dids]
        if self.arm == "slim_no_descriptor":
            return value * 0.0
        if self.arm == "slim_without_order":
            value = value.clone()
            value[..., 3] = 0.0  # token_order_norm is the fourth selected descriptor
        return value

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        base = batch["logit"].clamp(-10, 10)
        evidence = torch.zeros_like(base) if self.cfg.get("independent_geometry") else base
        h = self.embed(torch.cat((self._descriptor(batch["d"]), evidence[:, None] / 5.0,
                                  batch["extra"][:, self.extra_ids]), -1))
        neighbor_logit = torch.zeros_like(batch["nl"]) if self.cfg.get("independent_geometry") else batch["nl"].clamp(-10, 10)
        neighbor = self.nbr_embed(torch.cat((self._descriptor(batch["nd"]), neighbor_logit[..., None] / 5.0,
                                             batch["nextra"][:, :, self.extra_ids]), -1))
        score = (self.q(h)[:, None] * self.k(neighbor)).sum(-1) / math.sqrt(48)
        score = score.masked_fill(~batch["nm"], -1e4)
        context = (score.softmax(-1)[..., None] * self.v(neighbor)).sum(1)
        correction = self.out(self.context(torch.cat((h, context), -1))).squeeze(-1)
        self.aux_logit = correction
        alpha = float(self.cfg.get("geometry_weight", 0.5))
        limit = float(self.cfg.get("base_logit_cap", 10.0))
        return (1.0 - alpha) * base.clamp(-limit, limit) + alpha * correction


def copy_from_full_state(model: SlimNoPatchHead, full_state: Mapping[str, torch.Tensor]) -> None:
    """Copy all active weights; patch columns are the final 64 embed columns."""
    slim = model.state_dict()
    copied = {}
    for key, target in slim.items():
        source = full_state[key]
        if key == "embed.0.weight":
            source = source[:, : target.shape[1]]
        if source.shape != target.shape:
            raise RuntimeError(f"Cannot convert {key}: {source.shape} -> {target.shape}")
        copied[key] = source.detach().clone()
    model.load_state_dict(copied, strict=True)


def from_full_checkpoint(payload: Mapping[str, object], arm: str = "slim_no_patch") -> SlimNoPatchHead:
    model = SlimNoPatchHead(payload["config"], arm)
    copy_from_full_state(model, payload["state_dict"])
    return model


def matched_initial_model(config: Mapping[str, object], arm: str, seed: int, device: torch.device) -> SlimNoPatchHead:
    """Return a slim model with the retained initialization/RNG of H90."""
    T.set_seed(int(seed))
    donor = A.AblationHead(config, "no_patch")
    py = __import__("random").getstate()
    np_state = __import__("numpy").random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    model = SlimNoPatchHead(config, arm)
    copy_from_full_state(model, donor.state_dict())
    __import__("random").setstate(py)
    __import__("numpy").random.set_state(np_state)
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    del donor
    return model.to(device)


def effective_parameter_count(arm: str) -> int:
    config = T.candidate_config(42, T.BASE_CHECKPOINTS[42])
    model = SlimNoPatchHead(config, arm)
    total = sum(p.numel() for p in model.parameters())
    if arm == "slim_without_order":
        # One central and one neighbor input column are functionally disabled.
        return total - 96 - 96
    if arm == "slim_no_descriptor":
        return total - 6 * 96 - 6 * 96
    return total
