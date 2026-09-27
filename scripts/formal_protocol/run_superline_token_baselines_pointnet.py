#!/usr/bin/env python
"""Run baseline backbones on superline-based point tokens.

This formal comparison keeps the source-only protocol and the baseline
backbones, but replaces the candidate-point input with the point tokens used by
SBPT-Net. Token probabilities are projected back to the original point-cloud
space through the same patch-membership rule used for SBPT-Net evaluation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

BASELINE_SCRIPT = (
    ROOT
    / "reference_code"
    / "STS2R_formal_baselines_100ep_w8"
    / "scripts"
    / "03_run_formal_baselines.py"
)


def load_baseline_module():
    spec = importlib.util.spec_from_file_location(
        "sts2r_formal_baselines_for_superline_tokens",
        str(BASELINE_SCRIPT),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import baseline script: {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


B0 = load_baseline_module()

from formal_protocol import evaluate_stenet_full_point_projection as P  # noqa: E402
from formal_protocol import train_stenet_d04_1000 as S  # noqa: E402


MODEL_TITLES = {
    "pointnet": "PointNet",
    "pointnet2": "PointNet++",
    "pointmlp": "PointMLP",
    "dgcnn": "DGCNN",
    "pointtransformer": "Point Transformer",
    "pointnext": "PointNeXt",
    "curvenet": "CurveNet",
}
THRESHOLDS = tuple(float(v) for v in S.M.THRESHOLD_GRID)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_feature_matrix(ds: Mapping[str, np.ndarray]) -> np.ndarray:
    xyz = np.asarray(ds["xyz"], dtype=np.float32)
    rgb = np.asarray(ds["rgb"], dtype=np.float32)
    if len(xyz):
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
    if rgb.size and float(np.nanmax(rgb)) > 1.5:
        rgb = rgb / 255.0
    return np.concatenate([xyz, np.clip(rgb, 0.0, 1.0)], axis=1).astype(np.float32)


class SuperlineTokenDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        split: str,
        sample_points: int = 4096,
    ) -> None:
        self.records = list(records)
        self.split = split
        self.sample_points = int(sample_points)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        ds = record["downsampled"]
        x = token_feature_matrix(ds)
        y = np.asarray(ds["label"], dtype=np.int64)
        if len(y) == 0:
            x = np.zeros((1, 6), dtype=np.float32)
            y = np.zeros((1,), dtype=np.int64)
        if self.split == "train":
            replace = len(y) < self.sample_points
            choice = np.random.choice(len(y), self.sample_points, replace=replace)
            x = x[choice]
            y = y[choice]
        return torch.from_numpy(x.T).float(), torch.from_numpy(y).long()


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        generator=generator,
    )


def build_model(model_name: str, device: torch.device):
    if model_name == "pointnet":
        model = B0.PointNet(num_classes=2, input_channels=6)
    elif model_name == "pointnet2":
        model = B0.PointNet2(num_classes=2, input_channels=6)
    elif model_name == "pointmlp":
        model = B0.PointMLP(num_classes=2, input_channels=6)
    elif model_name == "dgcnn":
        model = B0.DGCNN(num_classes=2, input_channels=6)
    elif model_name == "pointtransformer":
        model = B0.PointTransformerSeg(num_classes=2, input_channels=6)
    elif model_name == "pointnext":
        model = B0.PointNeXt(num_classes=2, input_channels=6)
    elif model_name == "curvenet":
        model = B0.CurveNetSeg(num_classes=2, input_channels=6)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


def forward_logits(model: torch.nn.Module, feat: torch.Tensor) -> torch.Tensor:
    outputs = model(feat)
    return outputs[0] if isinstance(outputs, tuple) else outputs


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    for feat, target in tqdm(loader, desc="train", leave=False):
        feat = feat.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        xyz = feat[:, :3, :]
        rgb = feat[:, 3:6, :]
        optimizer.zero_grad(set_to_none=True)
        logits = forward_logits(model, feat)
        loss = criterion(logits, target, xyz, rgb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
        lab = target.detach().cpu().numpy().reshape(-1)
        tp += int(np.sum((pred == 1) & (lab == 1)))
        fp += int(np.sum((pred == 1) & (lab == 0)))
        fn += int(np.sum((pred == 0) & (lab == 1)))
        tn += int(np.sum((pred == 0) & (lab == 0)))
    metrics = P.binary_metrics_from_counts(tp, fp, fn, tn)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def predict_token_probabilities(
    model: torch.nn.Module,
    record: Mapping[str, object],
    criterion: torch.nn.Module,
    device: torch.device,
    chunk_points: int,
) -> tuple[np.ndarray, float]:
    ds = record["downsampled"]
    x = token_feature_matrix(ds)
    y = np.asarray(ds["label"], dtype=np.int64)
    if len(y) == 0:
        return np.zeros((0,), dtype=np.float32), 0.0

    probs: List[np.ndarray] = []
    weighted_loss = 0.0
    total = 0
    for start in range(0, len(y), chunk_points):
        end = min(start + chunk_points, len(y))
        xb = torch.from_numpy(x[start:end].T[None, :, :]).float().to(device)
        yb = torch.from_numpy(y[start:end][None, :]).long().to(device)
        logits = forward_logits(model, xb)
        loss = criterion(logits, yb, xb[:, :3, :], xb[:, 3:6, :])
        prob = F.softmax(logits, dim=1)[:, 1, :].detach().cpu().numpy().reshape(-1)
        probs.append(prob.astype(np.float32))
        weighted_loss += float(loss.item()) * (end - start)
        total += end - start
    return np.concatenate(probs).astype(np.float32), weighted_loss / max(total, 1)


def prepare_projection_support(
    records: Sequence[Mapping[str, object]],
    cfg: object,
    desc: str,
) -> None:
    pending = [record for record in records if "projection_support" not in record]
    for record in tqdm(pending, desc=desc):
        candidate, memberships, audit = P.reconstruct_patch_memberships(record, cfg)
        raw = P.load_raw_metadata(str(record["subset"]), str(record["sample_id"]))
        record["projection_support"] = {
            "candidate_orig_index": np.asarray(candidate["orig_index"], dtype=np.int64),
            "candidate_count": int(len(candidate["orig_index"])),
            "memberships": memberships,
            "audit": audit,
            "raw_num_points": int(raw["raw_num_points"]),
        }


def evaluate_projected_records(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, object]],
    criterion: torch.nn.Module,
    device: torch.device,
    chunk_points: int,
    threshold: float,
    model_key: str,
    split_name: str,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    total_loss = 0.0
    for record in tqdm(records, desc=f"{model_key}-{split_name}"):
        token_prob, loss = predict_token_probabilities(
            model, record, criterion, device, chunk_points
        )
        support = record["projection_support"]
        candidate_probability, vote_count = P.project_probabilities(
            token_prob,
            support["memberships"],
            support["candidate_count"],
        )
        metrics = P.counts_for_prediction(
            str(record["subset"]),
            str(record["sample_id"]),
            support["candidate_orig_index"],
            candidate_probability,
            threshold,
        )
        rows.append(
            {
                "method": MODEL_TITLES[model_key],
                "method_key": f"{model_key}_superline_tokens",
                "split": split_name,
                "subset": str(record["subset"]),
                "sample_id": str(record["sample_id"]),
                "metric_space": "original_raw_roi_point_space",
                "projection_threshold": float(threshold),
                "token_points": int(len(token_prob)),
                "candidate_points": int(support["candidate_count"]),
                "points": int(support["raw_num_points"]),
                "loss": float(loss),
                **metrics,
            }
        )
        total_loss += loss
        if output_dir is not None:
            pred_dir = output_dir / "projected_predictions" / model_key / split_name / str(record["subset"])
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pred_dir / f"{record['sample_id']}.npz",
                token_probability=token_prob.astype(np.float32),
                candidate_probability=candidate_probability.astype(np.float32),
                candidate_orig_index=support["candidate_orig_index"].astype(np.int64),
                vote_count=vote_count.astype(np.uint16),
                raw_num_points=np.asarray(support["raw_num_points"], dtype=np.int64),
            )
    frame = pd.DataFrame(rows)
    macro = {
        "loss": total_loss / max(len(records), 1),
        "iou": float(frame["iou"].mean()) if len(frame) else 0.0,
        "precision": float(frame["precision"].mean()) if len(frame) else 0.0,
        "recall": float(frame["recall"].mean()) if len(frame) else 0.0,
        "f1": float(frame["f1"].mean()) if len(frame) else 0.0,
    }
    return frame, macro


def select_threshold(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, object]],
    criterion: torch.nn.Module,
    device: torch.device,
    chunk_points: int,
    model_key: str,
) -> tuple[float, pd.DataFrame]:
    candidate_rows: List[Dict[str, object]] = []
    cached: List[tuple[Mapping[str, object], np.ndarray]] = []
    model.eval()
    for record in tqdm(records, desc=f"{model_key}-synthetic-threshold-cache", leave=False):
        token_prob, _ = predict_token_probabilities(
            model, record, criterion, device, chunk_points
        )
        support = record["projection_support"]
        candidate_probability, _ = P.project_probabilities(
            token_prob,
            support["memberships"],
            support["candidate_count"],
        )
        cached.append((record, candidate_probability))
    for threshold in THRESHOLDS:
        per_case: List[float] = []
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for record, candidate_probability in cached:
            support = record["projection_support"]
            metrics = P.counts_for_prediction(
                str(record["subset"]),
                str(record["sample_id"]),
                support["candidate_orig_index"],
                candidate_probability,
                threshold,
            )
            per_case.append(float(metrics["iou"]))
            for key in counts:
                counts[key] += int(metrics[key])
        pooled = P.binary_metrics_from_counts(**counts)
        candidate_rows.append(
            {
                "threshold": float(threshold),
                "miou": float(np.mean(per_case)) if per_case else 0.0,
                "pooled_iou": pooled["iou"],
                "pooled_precision": pooled["precision"],
                "pooled_recall": pooled["recall"],
                "pooled_f1": pooled["f1"],
                **counts,
            }
        )
    grid = pd.DataFrame(candidate_rows)
    best = grid.sort_values(["miou", "pooled_f1", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), grid


def subset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for subset, sub in frame.groupby("subset", sort=False):
        counts = {key: int(sub[key].sum()) for key in ("tp", "fp", "fn", "tn")}
        pooled = P.binary_metrics_from_counts(**counts)
        rows.append(
            {
                "subset": subset,
                "cases": int(len(sub)),
                "mean_iou": float(sub["iou"].mean()),
                "mean_precision": float(sub["precision"].mean()),
                "mean_recall": float(sub["recall"].mean()),
                "mean_f1": float(sub["f1"].mean()),
                "pooled_iou": pooled["iou"],
                "pooled_precision": pooled["precision"],
                "pooled_recall": pooled["recall"],
                "pooled_f1": pooled["f1"],
                **counts,
            }
        )
    if rows:
        mean_row = {
            "subset": "Equal-subset macro mean",
            "cases": int(frame["sample_id"].nunique()),
            "mean_iou": float(np.mean([r["mean_iou"] for r in rows])),
            "mean_precision": float(np.mean([r["mean_precision"] for r in rows])),
            "mean_recall": float(np.mean([r["mean_recall"] for r in rows])),
            "mean_f1": float(np.mean([r["mean_f1"] for r in rows])),
            "pooled_iou": np.nan,
            "pooled_precision": np.nan,
            "pooled_recall": np.nan,
            "pooled_f1": np.nan,
            "tp": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tn": np.nan,
        }
        rows.append(mean_row)
    return pd.DataFrame(rows)


def load_records(args: argparse.Namespace, cfg: object):
    manifest = pd.read_csv(args.split_manifest)
    if args.smoke:
        train_rows = manifest[manifest["split"] == "train"].head(args.smoke_train).to_dict("records")
        val_rows = manifest[manifest["split"] == "val"].head(args.smoke_val).to_dict("records")
        real_rows = (
            manifest[manifest["split"] == "test"]
            .groupby("subset", sort=False)
            .head(args.smoke_real_per_subset)
            .to_dict("records")
        )
    else:
        train_rows = manifest[manifest["split"] == "train"].to_dict("records")
        val_rows = manifest[manifest["split"] == "val"].to_dict("records")
        real_rows = manifest[manifest["split"] == "test"].to_dict("records")
    train_records = S.B.prepare_records(train_rows, args.token_cache_root, cfg)
    val_records = S.B.prepare_records(val_rows, args.token_cache_root, cfg)
    real_records = S.B.prepare_records(real_rows, args.token_cache_root, cfg)
    return train_records, val_records, real_records


def write_epoch_log(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_model(
    model_key: str,
    train_records: Sequence[Mapping[str, object]],
    val_records: Sequence[Mapping[str, object]],
    real_records: Sequence[Mapping[str, object]],
    cfg: object,
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = torch.device(args.device)
    model_dir = args.output_root / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(model_key, device)
    criterion = B0.Stage2Loss(
        pos_weight=2.0,
        use_smooth=False,
        smooth_weight=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_ds = SuperlineTokenDataset(train_records, split="train", sample_points=args.sample_points)
    train_loader = make_loader(
        train_ds,
        args.batch_size,
        True,
        args.num_workers,
        args.seed,
    )

    best_iou = -1.0
    best_epoch = 0
    patience = 0
    epoch_rows: List[Dict[str, object]] = []
    best_path = model_dir / f"best_{model_key}_superline_tokens.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_macro = evaluate_projected_records(
            model,
            val_records,
            criterion,
            device,
            args.chunk_points,
            threshold=0.5,
            model_key=model_key,
            split_name="val_checkpoint",
            output_dir=None,
        )
        scheduler.step()
        is_best = val_macro["iou"] > best_iou
        if is_best:
            best_iou = val_macro["iou"]
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - start),
            "is_best": int(is_best),
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_iou_at_0p5": val_macro["iou"],
            "val_precision_at_0p5": val_macro["precision"],
            "val_recall_at_0p5": val_macro["recall"],
            "val_f1_at_0p5": val_macro["f1"],
            "patience": patience,
        }
        epoch_rows.append(row)
        write_epoch_log(model_dir / "epoch_metrics.csv", epoch_rows)
        print(
            f"[{model_key}] epoch {epoch:03d} "
            f"train_iou={train_metrics['iou']:.4f} "
            f"val_iou@0.5={val_macro['iou']:.4f} "
            f"best={best_iou:.4f}@{best_epoch}"
        )
        if patience >= args.patience:
            print(f"[{model_key}] early stopping at epoch {epoch}")
            break

    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device))

    selected_threshold, threshold_grid = select_threshold(
        model,
        val_records,
        criterion,
        device,
        args.chunk_points,
        model_key,
    )
    threshold_grid.to_csv(model_dir / "synthetic_validation_threshold_grid.csv", index=False)
    val_frame, val_macro = evaluate_projected_records(
        model,
        val_records,
        criterion,
        device,
        args.chunk_points,
        selected_threshold,
        model_key,
        "synthetic_validation",
        output_dir=model_dir,
    )
    prepare_projection_support(
        real_records,
        cfg,
        f"Prepare real projection support for {model_key}",
    )
    real_frame, real_macro = evaluate_projected_records(
        model,
        real_records,
        criterion,
        device,
        args.chunk_points,
        selected_threshold,
        model_key,
        "real_test",
        output_dir=model_dir,
    )
    val_frame.to_csv(model_dir / "synthetic_validation_per_case.csv", index=False)
    real_frame.to_csv(model_dir / "real_per_case.csv", index=False)
    real_subset = subset_summary(real_frame)
    real_subset.to_csv(model_dir / "real_subset_summary.csv", index=False)

    summary = {
        "model_key": model_key,
        "method": f"{MODEL_TITLES[model_key]} on superline point tokens",
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "best_validation_iou_at_threshold_0p5": float(best_iou),
        "selected_projection_threshold": float(selected_threshold),
        "synthetic_validation_miou": float(val_macro["iou"]),
        "real_equal_subset_mean_iou": float(
            real_subset.loc[real_subset["subset"] == "Equal-subset macro mean", "mean_iou"].iloc[0]
        ),
        "real_subset_summary": real_subset.to_dict("records"),
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pointnet", "pointnet2"],
        choices=list(MODEL_TITLES.keys()),
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments" / "SuperlineToken_Baselines_PointNet_20260812")
    parser.add_argument("--split-manifest", type=Path, default=P.SPLIT_MANIFEST)
    parser.add_argument("--candidate-root", type=Path, default=P.CANDIDATE_ROOT)
    parser.add_argument("--token-cache-root", type=Path, default=P.TOKEN_CACHE_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-points", type=int, default=4096)
    parser.add_argument("--chunk-points", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train", type=int, default=16)
    parser.add_argument("--smoke-val", type=int, default=8)
    parser.add_argument("--smoke-real-per-subset", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists: {args.output_root}. Use --overwrite.")
    args.output_root.mkdir(parents=True, exist_ok=True)

    cfg_args = SimpleNamespace(
        candidate_dir=args.candidate_root,
        output_root=args.output_root,
        seed=args.seed,
        device=args.device,
    )
    cfg = S.build_compact_config(cfg_args)
    cfg.device = str(args.device)
    train_records, val_records, real_records = load_records(args, cfg)
    prepare_projection_support(val_records, cfg, "Prepare synthetic projection support")

    protocol = {
        "experiment": "Baseline backbones on superline point tokens",
        "models": list(args.models),
        "seed": int(args.seed),
        "model_seed_policy": "reset the requested seed before each backbone",
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "real_samples": len(real_records),
        "input": "sample-centered XYZ and RGB of superline-based point tokens",
        "projection": P.PROJECTION_VERSION,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "sample_points": int(args.sample_points),
        "chunk_points": int(args.chunk_points),
        "optimizer": "AdamW",
        "initial_learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "checkpoint_selection": "synthetic validation IoU at threshold 0.5",
        "operating_threshold_selection": "synthetic validation macro per-cloud mIoU",
        "metric_space": "original raw shoe-upper ROI point space",
        "real_aggregation": "equal mean over three subset macro means",
        "real_data_role": "final evaluation only",
        "smoke": bool(args.smoke),
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summaries = []
    for model_key in args.models:
        # Each backbone is an independent run for the requested seed. Resetting
        # all RNGs here prevents the model order from changing initialization or
        # sampling, which is required for a fair per-backbone seed comparison.
        set_seed(args.seed)
        summaries.append(run_model(model_key, train_records, val_records, real_records, cfg, args))
    pd.DataFrame(summaries).to_csv(args.output_root / "summary.csv", index=False)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
