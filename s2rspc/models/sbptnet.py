"""PointMLP plus the center-only 12D superline-aware token encoder."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig
from .pointmlp import PointMLP


ARCHITECTURE_ID = "SBPT-Net-PointMLP-Structural12D-H96"


def fuse_logits(
    initial_logit: torch.Tensor,
    structural_logit: torch.Tensor,
    beta: float,
    clip_bound: float,
) -> torch.Tensor:
    return (1.0 - float(beta)) * initial_logit.clamp(
        -float(clip_bound), float(clip_bound)
    ) + float(beta) * structural_logit


class StructuralEncoder12D(nn.Module):
    """Encode each center token independently; no neighbor aggregation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.structural_dim != 12 or cfg.structural_hidden_dim != 96:
            raise ValueError("StructuralEncoder12D requires 12D input and H96")
        self.embed = nn.Sequential(
            nn.Linear(cfg.structural_dim, cfg.structural_hidden_dim),
            nn.LayerNorm(cfg.structural_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.structural_dropout),
        )
        self.out = nn.Linear(cfg.structural_hidden_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, structural_features: torch.Tensor) -> torch.Tensor:
        if structural_features.shape[-1] != 12:
            raise ValueError(
                f"Expected 12 structural channels, got {structural_features.shape}"
            )
        return self.out(self.embed(structural_features)).squeeze(-1)


class SBPTNet(nn.Module):
    """Two-stage SBPT-Net inference graph."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        if self.cfg.pointmlp_input_dim != 6:
            raise ValueError("PointMLP input must be centered XYZ plus RGB (6D)")
        self.pointmlp = PointMLP(
            num_classes=self.cfg.pointmlp_classes,
            input_channels=self.cfg.pointmlp_input_dim,
            points=self.cfg.pointmlp_sample_points,
        )
        self.structural_encoder = StructuralEncoder12D(self.cfg)

    def initial_logits(self, point_features: torch.Tensor) -> torch.Tensor:
        outputs = self.pointmlp(point_features)
        class_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        if class_logits.ndim != 3 or class_logits.shape[1] != 2:
            raise ValueError(f"PointMLP returned unexpected shape {class_logits.shape}")
        return class_logits[:, 1, :] - class_logits[:, 0, :]

    def fuse(
        self,
        initial_logit: torch.Tensor,
        structural_logit: torch.Tensor,
    ) -> torch.Tensor:
        return fuse_logits(
            initial_logit,
            structural_logit,
            beta=self.cfg.fusion_beta,
            clip_bound=self.cfg.fusion_clip_bound,
        )

    def forward(
        self,
        point_features: torch.Tensor,
        structural_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial_logit = self.initial_logits(point_features)
        structural_logit = self.structural_encoder(structural_features)
        final_logit = self.fuse(initial_logit, structural_logit)
        return initial_logit, structural_logit, final_logit
