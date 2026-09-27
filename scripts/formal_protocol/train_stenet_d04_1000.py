#!/usr/bin/env python
"""Final Compact16D-H96 OrderedPT1 STE-Net training entry for D04-1000.

This entry is intentionally locked to the selected final architecture:

* 16 structural descriptors, with ``bin_size`` and the redundant raw
  ``frag_num_points`` excluded;
* three mean-RGB attributes, producing 19D token attributes;
* the local logit and normalized anchor XYZ, producing a 23D context input;
* one same-fragment OrderedPT1 layer with hidden dimension 96.

This file defines the final architecture and also retains a direct training
entry. The locked paper checkpoint was produced by
``train_stenet_sixway_ablation_d04_1000.py`` with a maximum of 100 epochs and
synthetic-validation early stopping patience 15. Checkpoint and threshold
selection use only the synthetic validation set. The 67 real scans are
reserved for frozen evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
BASE_TRAINING_PATH = (
    ROOT / "scripts" / "formal_protocol" / "stenet_training_core_d04_1000.py"
)


def load_base_training_module():
    spec = importlib.util.spec_from_file_location(
        "stenet_training_core_for_compact16d",
        BASE_TRAINING_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base STE-Net training module: {BASE_TRAINING_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


B = load_base_training_module()
R = B.R
M = B.M
T = B.T

FINAL_ARCHITECTURE_ID = "STE-Net-OrderedPT1-Compact16D-H96"
FINAL_DESCRIPTOR_FEATURE_NAMES = (
    "bin_num_points",
    "bin_length",
    "bin_pca_width",
    "bin_width_q",
    "bin_linearness",
    "bin_density_len",
    "log_frag_num_points",
    "frag_length",
    "frag_width",
    "frag_linearness",
    "frag_density_len",
    "frag_length_width_ratio",
    "token_order_norm",
    "base_spacing",
    "bin_width_to_frag_width",
    "bin_density_to_frag_density",
)
FINAL_REMOVED_DESCRIPTOR_FEATURES = ("frag_num_points", "bin_size")
FINAL_CONTEXT_HIDDEN_DIM = 96


def compact_descriptor_feature_names(
    args: argparse.Namespace | None = None,
) -> Tuple[str, ...]:
    del args
    available = tuple(M.DOWNSAMPLED_FEATURE_NAMES)
    missing = [name for name in FINAL_DESCRIPTOR_FEATURE_NAMES if name not in available]
    if missing:
        raise RuntimeError(f"Missing final descriptor features: {missing}")
    return FINAL_DESCRIPTOR_FEATURE_NAMES


def build_compact_config(args: argparse.Namespace):
    cfg = B.build_config(args)
    cfg.model_descriptor_feature_names = compact_descriptor_feature_names(args)
    cfg.use_token_rgb_features = True
    cfg.use_token_neighbor_prob_features = False
    cfg.use_token_polar_features = False
    cfg.use_token_global_context_features = False
    cfg.use_token_run_prob_features = False
    cfg.token_transformer_context_mode = "same_fragment"
    cfg.token_transformer_spatial_k = 0
    cfg.token_transformer_connectivity_gate_mode = "none"
    cfg.token_transformer_connectivity_gate_strength = 0.0
    cfg.token_transformer_edge_message_weight = 0.0
    cfg.token_transformer_hidden_dim = FINAL_CONTEXT_HIDDEN_DIM
    return cfg


class CompactSameFragmentOrderedPointTransformerRefiner(nn.Module):
    """Final OrderedPT1 refiner with no redundant probability input."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = FINAL_CONTEXT_HIDDEN_DIM,
        dropout: float = 0.10,
    ):
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError(f"Compact context input dimension must be positive, got {in_dim}")
        if int(hidden_dim) <= 0:
            raise ValueError(f"Compact hidden dimension must be positive, got {hidden_dim}")
        if int(hidden_dim) != FINAL_CONTEXT_HIDDEN_DIM:
            raise ValueError(
                f"Final context hidden dimension must be {FINAL_CONTEXT_HIDDEN_DIM}, "
                f"got {hidden_dim}"
            )
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.layer = B.SameFragmentOrderedPointTransformerLayer(
            hidden_dim=self.hidden_dim,
            dropout=float(dropout),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        xyz_norm: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"context x must be (N,{self.in_dim}), got {tuple(x.shape)}")
        if xyz_norm.shape != (x.shape[0], 3):
            raise ValueError("xyz_norm must have shape (N,3)")
        h = self.input_proj(x)
        h = self.layer(h, xyz_norm, neighbor_index, neighbor_mask)
        return self.delta_head(h).squeeze(1)


class CompactSTENet(nn.Module):
    """Final Compact16D local estimator plus OrderedPT1 logit refinement."""

    def __init__(self, cfg: object, feature_dim: int):
        super().__init__()
        descriptor_dim = len(M.model_descriptor_feature_names(cfg))
        if descriptor_dim != len(FINAL_DESCRIPTOR_FEATURE_NAMES):
            raise ValueError(
                "Final STE-Net requires the locked 16D structural descriptor"
            )
        expected_feature_dim = descriptor_dim + len(M.TOKEN_RGB_FEATURE_NAMES)
        if int(feature_dim) != expected_feature_dim:
            raise ValueError(
                "Compact STE-Net requires "
                f"{expected_feature_dim}D token attributes, got {feature_dim}"
            )
        encoder = M.PatchPointFeatureConcatEncoder(
            point_dim=int(cfg.patch_point_dim),
            feat_dim=int(feature_dim),
            width=int(cfg.patch_pointnet_width),
            hidden_dim=int(cfg.hidden_dim),
            emb_dim=int(cfg.embedding_dim),
            dropout=float(cfg.dropout),
            backbone_type=str(cfg.raw_backbone_type),
        )
        self.local_estimator = M.PatchPointFeatureConcatClassifier(
            encoder,
            int(cfg.embedding_dim),
            float(cfg.dropout),
        )
        self.context_refiner = CompactSameFragmentOrderedPointTransformerRefiner(
            in_dim=int(feature_dim) + 1 + 3,
            hidden_dim=int(cfg.token_transformer_hidden_dim),
            dropout=float(cfg.token_transformer_dropout),
        )
        self.feature_dim = int(feature_dim)

    def forward(
        self,
        patch_points: torch.Tensor,
        patch_mask: torch.Tensor,
        token_features: torch.Tensor,
        xyz_norm: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_logit = self.local_estimator(patch_points, patch_mask, token_features)
        context_x = torch.cat(
            [
                token_features,
                raw_logit.unsqueeze(1),
                xyz_norm,
            ],
            dim=1,
        )
        delta_logit = self.context_refiner(
            context_x,
            xyz_norm,
            neighbor_index,
            neighbor_mask,
        )
        refined_logit = raw_logit + delta_logit
        return raw_logit, refined_logit, delta_logit


def prepare_compact_model_arrays(
    records: Iterable[Dict[str, object]],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    cfg: object,
) -> None:
    base_spacing_index = M.DOWNSAMPLED_FEATURE_NAMES.index("base_spacing")
    order_window = int(getattr(cfg, "token_transformer_order_window", 8))
    for record in records:
        ds = record["downsampled"]
        n = int(len(ds["label"]))
        feature_raw = M.downsampled_feature_matrix(ds, cfg)
        expected_feature_dim = int(feature_mean.shape[-1])
        if feature_raw.shape != (n, expected_feature_dim):
            raise ValueError(
                "Unexpected compact token-feature shape: "
                f"expected {(n, expected_feature_dim)}, got {feature_raw.shape}"
            )
        feature_x = ((feature_raw - feature_mean) / feature_std).astype(np.float32)

        xyz = np.asarray(ds.get("xyz", np.zeros((n, 3), dtype=np.float32)), dtype=np.float32)
        descriptor_raw = np.asarray(
            ds.get(
                "features",
                np.zeros((n, len(M.DOWNSAMPLED_FEATURE_NAMES)), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if descriptor_raw.ndim == 2 and descriptor_raw.shape[1] > base_spacing_index:
            base_values = descriptor_raw[:, base_spacing_index]
            base_values = base_values[
                np.isfinite(base_values) & (base_values > 1e-8)
            ]
        else:
            base_values = np.zeros(0, dtype=np.float32)
        base_spacing = float(np.median(base_values)) if len(base_values) else 1.0
        if not np.isfinite(base_spacing) or base_spacing <= 0.0:
            base_spacing = 1.0
        center = (
            np.mean(xyz, axis=0, keepdims=True).astype(np.float32)
            if n
            else np.zeros((1, 3), dtype=np.float32)
        )
        xyz_norm = ((xyz - center) / max(base_spacing, 1e-6)).astype(np.float32)
        neighbor_index, neighbor_mask = M.build_same_fragment_window_indices(
            record,
            order_window,
        )
        record["compact_model_arrays"] = {
            "patch_points": np.asarray(ds["patch_points"], dtype=np.float32),
            "patch_mask": np.asarray(ds["patch_mask"], dtype=np.float32),
            "token_features": feature_x,
            "xyz_norm": xyz_norm,
            "neighbor_index": np.asarray(neighbor_index, dtype=np.int64),
            "neighbor_mask": np.asarray(neighbor_mask, dtype=bool),
            "label": np.asarray(ds["label"], dtype=np.float32),
        }


def compact_record_tensors(
    record: Dict[str, object],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    arrays = record["compact_model_arrays"]
    return {
        "patch_points": torch.from_numpy(arrays["patch_points"]).to(device),
        "patch_mask": torch.from_numpy(arrays["patch_mask"]).to(device),
        "token_features": torch.from_numpy(arrays["token_features"]).to(device),
        "xyz_norm": torch.from_numpy(arrays["xyz_norm"]).to(device),
        "neighbor_index": torch.from_numpy(arrays["neighbor_index"]).to(device),
        "neighbor_mask": torch.from_numpy(arrays["neighbor_mask"]).to(device),
        "label": torch.from_numpy(arrays["label"]).to(device),
    }


def compact_forward_record(
    model: CompactSTENet,
    record: Dict[str, object],
    device: torch.device,
    cfg: object,
    augment: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = compact_record_tensors(record, device)
    patch_points = data["patch_points"]
    patch_mask = data["patch_mask"]
    if augment:
        patch_points, patch_mask = M.augment_patch_points_for_domain_randomization(
            patch_points,
            patch_mask,
            cfg,
        )
    raw_logit, refined_logit, delta_logit = model(
        patch_points,
        patch_mask,
        data["token_features"],
        data["xyz_norm"],
        data["neighbor_index"],
        data["neighbor_mask"],
    )
    return raw_logit, refined_logit, delta_logit, data["label"]


# Stable public names used by evaluation, robustness, and figure scripts.
STENet = CompactSTENet
SameFragmentOrderedPointTransformerRefiner = (
    CompactSameFragmentOrderedPointTransformerRefiner
)
build_config = build_compact_config
prepare_model_arrays = prepare_compact_model_arrays
record_tensors = compact_record_tensors
forward_record = compact_forward_record
B.forward_record = compact_forward_record
predict_records = B.predict_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final Compact16D-H96 STE-Net training on D04-1000."
    )
    parser.add_argument("--candidate_dir", type=Path, default=B.DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--cache_root", type=Path, default=B.DEFAULT_CACHE_ROOT)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--files_per_step", type=int, default=2)
    parser.add_argument("--local_lr", type=float, default=5e-4)
    parser.add_argument("--context_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--local_aux_weight", type=float, default=0.5)
    parser.add_argument("--lovasz_weight", type=float, default=0.1)
    parser.add_argument("--delta_reg", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root is None:
        args.output_root = (
            ROOT
            / "outputs"
            / f"Formal_D04_1000_STENet_Compact16D_H96_seed{args.seed}"
        )
    args.output_root = args.output_root.resolve()
    args.candidate_dir = args.candidate_dir.resolve()
    args.cache_root = args.cache_root.resolve()

    if not args.candidate_dir.is_dir():
        raise FileNotFoundError(f"Candidate root does not exist: {args.candidate_dir}")
    if not args.cache_root.is_dir():
        raise FileNotFoundError(f"Token cache does not exist: {args.cache_root}")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_root}")

    cfg = build_compact_config(args)
    cfg.token_transformer_hidden_dim = FINAL_CONTEXT_HIDDEN_DIM
    M.set_global_determinism(int(args.seed))
    manifest_rows, protocol = T.choose_protocol(cfg, 0)
    train_count = sum(row["domain"] == "synthetic" and row["split"] == "train" for row in manifest_rows)
    val_count = sum(row["domain"] == "synthetic" and row["split"] == "val" for row in manifest_rows)
    real_count = sum(row["domain"] == "real" and row["split"] == "test" for row in manifest_rows)
    feature_dim = len(M.active_downsampled_feature_names(cfg))
    descriptor_feature_names = tuple(M.model_descriptor_feature_names(cfg))
    descriptor_dim = len(descriptor_feature_names)
    context_input_dim = feature_dim + 1 + 3
    model = CompactSTENet(cfg, feature_dim)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        "[FINAL COMPACT16D-H96 PROTOCOL] train={} val={} real={} epochs={} "
        "descriptor_dim={} token_dim={} context_dim={} "
        "context_hidden_dim={} params={} device={}".format(
            train_count,
            val_count,
            real_count,
            int(args.epochs),
            descriptor_dim,
            feature_dim,
            context_input_dim,
            int(cfg.token_transformer_hidden_dim),
            parameter_count,
            args.device,
        ),
        flush=True,
    )
    if (train_count, val_count, real_count) != (800, 200, 67):
        raise RuntimeError(
            "The final full-data run requires exactly 800/200/67 files, "
            f"got {train_count}/{val_count}/{real_count}"
        )
    expected_feature_dim = descriptor_dim + len(M.TOKEN_RGB_FEATURE_NAMES)
    if feature_dim != expected_feature_dim:
        raise RuntimeError(
            f"Expected {expected_feature_dim}D token attributes, got {feature_dim}"
        )
    if args.dry_run:
        return 0

    args.output_root.mkdir(parents=True, exist_ok=False)
    protocol_dir = args.output_root / "00_protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(protocol_dir / "split_manifest.csv", index=False)

    records = B.prepare_records(manifest_rows, args.cache_root, cfg)
    train_records = [
        record
        for record in records
        if record["domain"] == "synthetic" and record["split"] == "train"
    ]
    val_records = [
        record
        for record in records
        if record["domain"] == "synthetic" and record["split"] == "val"
    ]
    feature_mean, feature_std = M.feature_mean_std_from_records(
        train_records,
        feature_dim,
        cfg,
    )
    prepare_compact_model_arrays(records, feature_mean, feature_std, cfg)

    # Reuse the locked training/evaluation protocol while routing it through the
    # compact, spatial-plumbing-free forward path defined in this file.
    B.forward_record = compact_forward_record
    model, history = B.train_model(model, train_records, val_records, cfg, args)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(args.output_root / "training_history.csv", index=False)
    thresholds = B.evaluate_and_write(
        model,
        records,
        val_records,
        cfg,
        args.output_root,
    )
    checkpoint = {
        "architecture_id": FINAL_ARCHITECTURE_ID,
        "status": "final_canonical_architecture",
        "local_estimator": B.state_dict_cpu(model.local_estimator),
        "context_refiner": B.state_dict_cpu(model.context_refiner),
        "descriptor_feature_names": list(descriptor_feature_names),
        "removed_descriptor_features": [
            name for name in M.DOWNSAMPLED_FEATURE_NAMES
            if name not in descriptor_feature_names
        ],
        "context_uses_local_probability": False,
        "context_hidden_dim": int(cfg.token_transformer_hidden_dim),
        "token_attribute_dim": int(feature_dim),
        "context_input_dim": int(context_input_dim),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "thresholds": thresholds,
        "seed": int(args.seed),
    }
    torch.save(checkpoint, args.output_root / "stenet_best.pt")

    best_row = history_frame.sort_values(
        ["synthetic_val_f1_at_0_5", "epoch"],
        ascending=[False, True],
    ).iloc[0]
    experiment_manifest = {
        "experiment": "formal_d04_1000_stenet_compact16d_h96",
        "created_at_unix": time.time(),
        "formal_result": True,
        "architecture_id": FINAL_ARCHITECTURE_ID,
        "canonical_architecture": True,
        "full_dataset_run": True,
        "architecture": {
            "descriptor_dim": descriptor_dim,
            "descriptor_feature_names": list(descriptor_feature_names),
            "removed_descriptor_features": [
                name for name in M.DOWNSAMPLED_FEATURE_NAMES
                if name not in descriptor_feature_names
            ],
            "token_attribute_dim": feature_dim,
            "context_input_dim": context_input_dim,
            "context_hidden_dim": int(cfg.token_transformer_hidden_dim),
            "context_uses_local_logit": True,
            "context_uses_local_probability": False,
            "ordered_half_window": int(cfg.token_transformer_order_window),
            "maximum_neighbor_width": 2 * int(cfg.token_transformer_order_window) + 1,
            "spatial_neighbor_arrays_constructed": False,
            "trainable_parameters": parameter_count,
        },
        "training_strategy": (
            f"single optimization process from random initialization with a maximum of "
            f"{int(args.epochs)} epochs"
        ),
        "real_data_role": "evaluated after synthetic-validation checkpoint and threshold selection",
        "protocol": protocol,
        "candidate_root": str(args.candidate_dir),
        "cache_root": str(args.cache_root),
        "output_root": str(args.output_root),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "files_per_step": int(args.files_per_step),
        "local_lr": float(args.local_lr),
        "context_lr": float(args.context_lr),
        "weight_decay": float(args.weight_decay),
        "local_aux_weight": float(args.local_aux_weight),
        "lovasz_weight": float(args.lovasz_weight),
        "delta_reg": float(args.delta_reg),
        "best_epoch": int(best_row["epoch"]),
        "best_synthetic_val_f1_at_0_5": float(
            best_row["synthetic_val_f1_at_0_5"]
        ),
        "checkpoint_selection": "synthetic validation F1 at threshold 0.5",
        "threshold_selection": "synthetic validation IoU only",
        "thresholds": thresholds,
    }
    (args.output_root / "experiment_manifest.json").write_text(
        json.dumps(experiment_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
