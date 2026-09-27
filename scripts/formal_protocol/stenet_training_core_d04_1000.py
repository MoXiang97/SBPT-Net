#!/usr/bin/env python
"""Shared D04-1000 training protocol for the final STE-Net entry.

This module contains data loading, the OrderedPT1 layer, and common
training/evaluation routines. The canonical Compact16D-H96 architecture and
its executable entry point live in ``train_stenet_d04_1000.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts" / "shared_stage2_lib" / "stenet_superline_organization.py"
DEFAULT_CANDIDATE_ROOT = (
    ROOT / "outputs" / "Stage1_LCC_Candidates_D04_1000_from_existing_LCC_fixed381696"
)
DEFAULT_CACHE_ROOT = ROOT / "outputs" / "FormalCache_D04_1000_SourceAnchoredR6_seed42"


def load_runner():
    spec = importlib.util.spec_from_file_location("stenet_source_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import STE-Net runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = load_runner()
T = R.T
M = R.M


def state_dict_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


class SameFragmentOrderedPointTransformerLayer(nn.Module):
    """One same-fragment ordered Point Transformer-style vector-attention block."""

    def __init__(self, hidden_dim: int = 96, dropout: float = 0.10):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.hidden_dim = hidden_dim
        self.phi = nn.Linear(hidden_dim, hidden_dim)
        self.psi = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Linear(hidden_dim, hidden_dim)
        self.theta = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gamma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.capture_attention = False
        self.last_attention: Optional[torch.Tensor] = None
        self.last_attention_mask: Optional[torch.Tensor] = None

    def forward(
        self,
        h: torch.Tensor,
        xyz_norm: torch.Tensor,
        neighbor_index: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        if h.numel() == 0:
            return h
        if h.ndim != 2 or h.shape[1] != self.hidden_dim:
            raise ValueError(f"h must be (N,{self.hidden_dim}), got {tuple(h.shape)}")
        if xyz_norm.shape != (h.shape[0], 3):
            raise ValueError("xyz_norm must have shape (N,3)")
        if neighbor_index.shape != neighbor_mask.shape or neighbor_index.shape[0] != h.shape[0]:
            raise ValueError("ordered neighbor index/mask must have matching (N,W) shapes")

        phi_h = self.phi(h)
        psi_h = self.psi(h)
        alpha_h = self.alpha(h)
        width = int(neighbor_index.shape[1])
        max_pairs_per_chunk = 400_000
        chunk_rows = max(1, max_pairs_per_chunk // max(width, 1))
        message_parts = []
        captured_attention = []
        captured_masks = []

        for start in range(0, int(h.shape[0]), chunk_rows):
            end = min(int(h.shape[0]), start + chunk_rows)
            index_part = neighbor_index[start:end]
            mask_part = neighbor_mask[start:end].to(dtype=torch.bool)
            safe_mask = mask_part.clone()
            valid_rows = safe_mask.any(dim=1)
            if torch.any(~valid_rows):
                safe_mask[~valid_rows, 0] = True

            relative_xyz = xyz_norm[start:end].unsqueeze(1) - xyz_norm[index_part]
            delta = self.theta(relative_xyz)
            attention_logits = self.gamma(
                phi_h[start:end].unsqueeze(1) - psi_h[index_part] + delta
            )
            attention_logits = attention_logits.masked_fill(
                ~safe_mask.unsqueeze(-1),
                torch.finfo(attention_logits.dtype).min,
            )
            attention = torch.softmax(attention_logits, dim=1)
            attention = attention * mask_part.unsqueeze(-1).to(attention.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
            attention = torch.where(
                torch.isfinite(attention),
                attention,
                torch.zeros_like(attention),
            )
            message_parts.append(
                torch.sum(attention * (alpha_h[index_part] + delta), dim=1)
            )
            if self.capture_attention:
                captured_attention.append(attention.detach().cpu())
                captured_masks.append(mask_part.detach().cpu())

        message_all = torch.cat(message_parts, dim=0)
        h = self.norm1(h + self.dropout(self.out(message_all)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        if self.capture_attention:
            self.last_attention = torch.cat(captured_attention, dim=0)
            self.last_attention_mask = torch.cat(captured_masks, dim=0)
        return h


def build_config(args: argparse.Namespace) -> object:
    cfg = M.Config()
    cfg.candidate_dir = str(args.candidate_dir.resolve())
    cfg.max_sim_train = 800
    cfg.max_sim_val = 200
    cfg.synthetic_split_policy = "style_holdout"
    cfg.synthetic_holdout_styles = "TeBu,XBL"
    cfg.split_seed = int(args.seed)
    cfg.device = str(args.device)
    cfg.raw_backbone_type = "dual_pool_res"
    cfg.use_token_rgb_features = True
    cfg.use_token_neighbor_prob_features = False
    cfg.use_token_polar_features = False
    cfg.use_token_global_context_features = False
    cfg.use_token_run_prob_features = False
    cfg.token_transformer_context_mode = "same_fragment"
    cfg.token_transformer_spatial_k = 12
    cfg.token_transformer_spatial_radius_factor = 8.0
    cfg.token_transformer_spatial_residual_weight = 1.0
    cfg.token_transformer_hidden_dim = 96
    cfg.token_transformer_num_layers = 1
    cfg.token_transformer_dropout = 0.10
    cfg.token_transformer_connectivity_gate_mode = "none"
    cfg.token_transformer_connectivity_gate_strength = 0.0
    cfg.token_transformer_edge_message_weight = 0.0
    cfg.raw_patch_aug_mode = "none"
    cfg.raw_patch_dropout = 0.0
    cfg.raw_patch_jitter_std = 0.0
    cfg.raw_patch_scale_jitter = 0.0
    cfg.output_root = str(args.output_root.resolve())
    return cfg


def prepare_records(
    manifest_rows: List[Dict[str, object]],
    cache_root: Path,
    cfg: object,
) -> List[Dict[str, object]]:
    method = dict(R.SUPERLINE3D_STYLE_METHOD)
    method["cache_root"] = cache_root
    records: List[Dict[str, object]] = []
    for row in tqdm(manifest_rows, desc="Load source-anchored token cache"):
        ds = R.build_or_load_downsampled(str(row["path"]), str(row["subset"]), method, cfg)
        records.append({**row, "downsampled": ds})
    M.apply_feature_ablation_to_records(records, cfg)
    return records


def forward_record(
    model: nn.Module,
    record: Dict[str, object],
    device: torch.device,
    cfg: object,
    augment: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del model, record, device, cfg, augment
    raise RuntimeError(
        "The architecture entry must bind its model-specific forward_record"
    )


@torch.no_grad()
def predict_records(
    model: nn.Module,
    records: Iterable[Dict[str, object]],
    device: torch.device,
    cfg: object,
    quiet: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    raw_parts: List[np.ndarray] = []
    refined_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    iterator = records if quiet else tqdm(records, desc="STE-Net inference")
    for record in iterator:
        if len(record["downsampled"]["label"]) == 0:
            raw_prob = np.zeros(0, dtype=np.float32)
            refined_prob = np.zeros(0, dtype=np.float32)
            labels = np.zeros(0, dtype=np.int64)
        else:
            raw_logit, refined_logit, _, y = forward_record(model, record, device, cfg, augment=False)
            raw_prob = torch.sigmoid(raw_logit).cpu().numpy().astype(np.float32)
            refined_prob = torch.sigmoid(refined_logit).cpu().numpy().astype(np.float32)
            labels = y.cpu().numpy().astype(np.int64)
        record["stenet_raw_prob"] = raw_prob
        record["stenet_refined_prob"] = refined_prob
        raw_parts.append(raw_prob)
        refined_parts.append(refined_prob)
        label_parts.append(labels)
    raw_all = np.concatenate(raw_parts) if raw_parts else np.zeros(0, dtype=np.float32)
    refined_all = np.concatenate(refined_parts) if refined_parts else np.zeros(0, dtype=np.float32)
    labels_all = np.concatenate(label_parts) if label_parts else np.zeros(0, dtype=np.int64)
    return raw_all, refined_all, labels_all


def train_model(
    model: nn.Module,
    train_records: List[Dict[str, object]],
    val_records: List[Dict[str, object]],
    cfg: object,
    args: argparse.Namespace,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    device = torch.device(cfg.device)
    model.to(device)
    all_positive = sum(int(np.sum(r["downsampled"]["label"] == 1)) for r in train_records)
    all_negative = sum(int(np.sum(r["downsampled"]["label"] == 0)) for r in train_records)
    pos_weight = torch.tensor(
        [max(float(all_negative), 1.0) / max(float(all_positive), 1.0)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.local_estimator.parameters(), "lr": float(args.local_lr)},
            {"params": model.context_refiner.parameters(), "lr": float(args.context_lr)},
        ],
        weight_decay=float(args.weight_decay),
    )

    best_val_f1 = -1.0
    best_local_state = None
    best_context_state = None
    history: List[Dict[str, float]] = []
    files_per_step = max(1, int(args.files_per_step))

    for epoch in range(int(args.epochs)):
        model.train()
        order = np.arange(len(train_records), dtype=np.int64)
        rng = np.random.default_rng(int(args.seed) + 30011 * (epoch + 1))
        rng.shuffle(order)
        epoch_losses: List[float] = []
        epoch_raw_losses: List[float] = []
        epoch_refined_losses: List[float] = []

        for start in range(0, len(order), files_per_step):
            optimizer.zero_grad(set_to_none=True)
            file_losses: List[torch.Tensor] = []
            file_raw_losses: List[torch.Tensor] = []
            file_refined_losses: List[torch.Tensor] = []
            for record_index in order[start : start + files_per_step]:
                record = train_records[int(record_index)]
                if len(record["downsampled"]["label"]) == 0:
                    continue
                raw_logit, refined_logit, delta_logit, y = forward_record(
                    model,
                    record,
                    device,
                    cfg,
                    augment=True,
                )
                raw_loss = criterion(raw_logit, y)
                refined_loss = criterion(refined_logit, y)
                if float(args.lovasz_weight) > 0.0:
                    refined_loss = refined_loss + float(args.lovasz_weight) * M.lovasz_hinge_loss_from_logits(
                        refined_logit,
                        y,
                    )
                delta_penalty = torch.mean(delta_logit * delta_logit)
                total_loss = (
                    refined_loss
                    + float(args.local_aux_weight) * raw_loss
                    + float(args.delta_reg) * delta_penalty
                )
                file_losses.append(total_loss)
                file_raw_losses.append(raw_loss.detach())
                file_refined_losses.append(refined_loss.detach())
            if not file_losses:
                continue
            batch_loss = torch.stack(file_losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            epoch_losses.append(float(batch_loss.detach().cpu()))
            epoch_raw_losses.append(float(torch.stack(file_raw_losses).mean().cpu()))
            epoch_refined_losses.append(float(torch.stack(file_refined_losses).mean().cpu()))

        _, val_refined, val_y = predict_records(model, val_records, device, cfg, quiet=True)
        val_f1 = M.binary_metrics(val_y, val_refined >= 0.5)["f1"] if len(val_y) else 0.0
        row = {
            "epoch": float(epoch + 1),
            "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "raw_loss": float(np.mean(epoch_raw_losses)) if epoch_raw_losses else 0.0,
            "refined_loss": float(np.mean(epoch_refined_losses)) if epoch_refined_losses else 0.0,
            "synthetic_val_f1_at_0_5": float(val_f1),
        }
        history.append(row)
        print(
            "[STE-Net] epoch={:03d}/{:03d} loss={:.5f} raw={:.5f} refined={:.5f} val_f1@0.5={:.4f}".format(
                epoch + 1,
                int(args.epochs),
                row["loss"],
                row["raw_loss"],
                row["refined_loss"],
                row["synthetic_val_f1_at_0_5"],
            ),
            flush=True,
        )
        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_local_state = state_dict_cpu(model.local_estimator)
            best_context_state = state_dict_cpu(model.context_refiner)

    if best_local_state is not None and best_context_state is not None:
        model.local_estimator.load_state_dict(best_local_state)
        model.context_refiner.load_state_dict(best_context_state)
    return model, history


def evaluate_and_write(
    model: nn.Module,
    records: List[Dict[str, object]],
    val_records: List[Dict[str, object]],
    cfg: object,
    output_root: Path,
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    predict_records(model, records, device, cfg, quiet=False)
    val_raw = np.concatenate([r["stenet_raw_prob"] for r in val_records])
    val_refined = np.concatenate([r["stenet_refined_prob"] for r in val_records])
    val_y = np.concatenate([np.asarray(r["downsampled"]["label"], dtype=np.int64) for r in val_records])
    raw_threshold = float(M.choose_threshold_from_val(val_raw, val_y, cfg))
    refined_threshold = float(M.choose_threshold_from_val(val_refined, val_y, cfg))

    raw_summary = M.aggregate_eval(
        M.eval_records_with_prob_key(records, raw_threshold, cfg, "stenet_raw_prob", desc="Evaluate STE-Net raw")
    )
    raw_summary.insert(0, "method", "stenet")
    raw_summary.insert(1, "evidence_stage", "raw")
    refined_summary = M.aggregate_eval(
        M.eval_records_with_prob_key(
            records,
            refined_threshold,
            cfg,
            "stenet_refined_prob",
            desc="Evaluate STE-Net refined",
        )
    )
    refined_summary.insert(0, "method", "stenet")
    refined_summary.insert(1, "evidence_stage", "same_fragment")
    pd.concat([raw_summary, refined_summary], ignore_index=True).to_csv(
        output_root / "training_summary.csv",
        index=False,
    )
    thresholds = {"raw": raw_threshold, "refined": refined_threshold}
    (output_root / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return thresholds

