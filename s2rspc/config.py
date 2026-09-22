"""Typed configuration for the paper-aligned public SBPT-Net implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    raw_root: str = "data/raw"
    candidate_root: str = "data/candidates"
    token_root: str = "data/tokens"
    synthetic_subset: str = "synthetic"
    real_subsets: List[str] = field(
        default_factory=lambda: ["shoe_a", "shoe_b", "shoe_c"]
    )
    train_samples: int = 800
    validation_samples: int = 200
    split_policy: str = "style_holdout"
    holdout_styles: List[str] = field(default_factory=lambda: ["TeBu", "XBL"])


@dataclass
class LCCConfig:
    neighbors: int = 256
    threshold: float = 0.38169607520103455
    chunk_size: int = 50000


@dataclass
class SuperlineConfig:
    descriptor_neighbors: int = 20
    graph_neighbors: int = 20
    minimum_linearness: float = 0.18
    minimum_tangent_similarity: float = 0.72
    maximum_descriptor_distance: float = 3.2
    maximum_spatial_factor: float = 8.0
    descriptor_cost_weight: float = 1.0
    tangent_cost_weight: float = 0.25
    distance_cost_weight: float = 0.08
    residual_cost_weight: float = 0.15
    maximum_component_points: int = 4096
    minimum_fragment_points: int = 3
    spacing_sample_size: int = 8000


@dataclass
class TokenizationConfig:
    support_radius_factor: float = 6.0
    seed_stride_factor: float = 2.0
    representative_perpendicular_weight: float = 1.0
    representative_center_weight: float = 0.35
    local_width_quantile: float = 0.70
    label_positive_ratio: float = 0.50
    minimum_support_points: int = 1


@dataclass
class ModelConfig:
    pointmlp_input_dim: int = 6
    pointmlp_classes: int = 2
    pointmlp_sample_points: int = 4096
    structural_dim: int = 12
    structural_hidden_dim: int = 96
    structural_dropout: float = 0.10
    fusion_beta: float = 0.5
    fusion_clip_bound: float = 3.0
    deviation_clip_bound: float = 10.0


@dataclass
class TrainingConfig:
    optimizer: str = "AdamW"
    pointmlp_learning_rate: float = 5e-4
    pointmlp_weight_decay: float = 1e-4
    structural_learning_rate: float = 5e-4
    structural_weight_decay: float = 2e-4
    maximum_epochs: int = 100
    early_stopping_patience: int = 15
    pointmlp_batch_size: int = 8
    structural_batch_clouds: int = 4
    structural_tokens_per_cloud: int = 256
    pointmlp_positive_weight: float = 2.0
    structural_positive_weight: float = 1.0
    dice_weight: float = 0.5
    structural_auxiliary_weight: float = 0.1
    deviation_weight: float = 1e-4
    gradient_clip: float = 5.0
    checkpoint_selection_threshold: float = 0.5
    descriptor_scale_low: float = 0.85
    descriptor_scale_high: float = 1.15
    descriptor_shift_std: float = 0.08
    descriptor_noise_std: float = 0.04
    attribute_scale_low: float = 0.75
    attribute_scale_high: float = 1.25
    attribute_shift_std: float = 0.20
    attribute_noise_std: float = 0.06
    attribute_drop_probability: float = 0.10


@dataclass
class EvaluationConfig:
    threshold_grid: List[float] = field(
        default_factory=lambda: [i / 20 for i in range(1, 20)]
    )
    projection_threshold: float = 0.40
    metric_space: str = "original_point_cloud_space"


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    lcc: LCCConfig = field(default_factory=LCCConfig)
    superline: SuperlineConfig = field(default_factory=SuperlineConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if self.model.pointmlp_input_dim != 6 or self.model.pointmlp_classes != 2:
            raise ValueError(
                "PointMLP must receive centered XYZ and normalized RGB and output two classes"
            )
        if self.model.structural_dim != 12 or self.model.structural_hidden_dim != 96:
            raise ValueError(
                "The structural encoder requires 12D input and 96 hidden features"
            )
        if not 0.0 <= self.model.fusion_beta <= 1.0:
            raise ValueError("fusion_beta must be in [0, 1]")
        if self.model.fusion_clip_bound <= 0 or self.model.deviation_clip_bound <= 0:
            raise ValueError("Clipping bounds must be positive")
        if self.model.fusion_clip_bound == self.model.deviation_clip_bound:
            raise ValueError(
                "Fusion and deviation clipping bounds represent different operations"
            )
        if self.tokenization.representative_center_weight != 0.35:
            raise ValueError(
                "The released representative center weight is lambda_c=0.35"
            )


def _construct(cls, values: Dict[str, Any]):
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = ExperimentConfig(
        seed=int(payload.get("seed", 42)),
        data=_construct(DataConfig, payload.get("data", {})),
        lcc=_construct(LCCConfig, payload.get("lcc", {})),
        superline=_construct(SuperlineConfig, payload.get("superline", {})),
        tokenization=_construct(TokenizationConfig, payload.get("tokenization", {})),
        model=_construct(ModelConfig, payload.get("model", {})),
        training=_construct(TrainingConfig, payload.get("training", {})),
        evaluation=_construct(EvaluationConfig, payload.get("evaluation", {})),
    )
    cfg.validate()
    return cfg
