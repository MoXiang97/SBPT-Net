#!/usr/bin/env python
"""Run a leakage-safe real-only cross-shoe pilot.

The experiment compares the same point-cloud backbones on (1) fixed-LCC
candidate points and (2) the established superline-based point tokens.  Each
fold trains on two shoe styles and evaluates once on the held-out style.  The
held-out style is never used for checkpoint or operating-threshold selection.

Existing model, loss, token-construction, and projection implementations are
imported from the formal Figure 9 pipeline.  This file only orchestrates the
new split protocol and the matched candidate/token evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from formal_protocol import evaluate_stenet_full_point_projection as P  # noqa: E402
from formal_protocol import run_superline_token_baselines_pointnet as T  # noqa: E402
from formal_protocol import train_stenet_d04_1000 as S  # noqa: E402


STYLE_KEYS = ("shoe_a", "shoe_b", "shoe_c")
VAL_COUNTS = {"shoe_a": 5, "shoe_b": 5, "shoe_c": 3}
FOLDS = {
    "ab_to_c": {"source": ("shoe_a", "shoe_b"), "target": "shoe_c"},
    "ac_to_b": {"source": ("shoe_a", "shoe_c"), "target": "shoe_b"},
    "bc_to_a": {"source": ("shoe_b", "shoe_c"), "target": "shoe_a"},
}
REPRESENTATIONS = ("candidate_points", "superline_tokens")
DEFAULT_MODELS = ("pointnet2", "dgcnn", "pointtransformer")
THRESHOLDS = tuple(float(v) for v in S.M.THRESHOLD_GRID)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_feature_matrix(candidate: Mapping[str, np.ndarray]) -> np.ndarray:
    xyz = np.asarray(candidate["xyz"], dtype=np.float32)
    rgb = np.asarray(candidate["rgb"], dtype=np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.5:
        rgb = rgb / 255.0
    return np.concatenate([xyz, np.clip(rgb, 0.0, 1.0)], axis=1).astype(np.float32)


def representation_arrays(record: Mapping[str, object], representation: str) -> tuple[np.ndarray, np.ndarray]:
    if representation == "candidate_points":
        candidate = record["candidate"]
        return candidate_feature_matrix(candidate), np.asarray(candidate["label"], dtype=np.int64)
    ds = record["downsampled"]
    return T.token_feature_matrix(ds), np.asarray(ds["label"], dtype=np.int64)


class BalancedRealViewDataset(Dataset):
    """Create equal numbers of stochastic views from each source shoe style."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        representation: str,
        sample_points: int,
        views_per_style: int,
    ) -> None:
        self.representation = representation
        self.sample_points = int(sample_points)
        self.views_per_style = int(views_per_style)
        grouped: Dict[str, List[Mapping[str, object]]] = {}
        for record in records:
            grouped.setdefault(str(record["subset"]), []).append(record)
        self.styles = tuple(sorted(grouped))
        if len(self.styles) != 2:
            raise ValueError(f"Each fold requires exactly two source styles, got {self.styles}")
        self.grouped = {key: sorted(value, key=lambda r: str(r["sample_id"])) for key, value in grouped.items()}

    def __len__(self) -> int:
        return len(self.styles) * self.views_per_style

    def __getitem__(self, idx: int):
        style = self.styles[idx % len(self.styles)]
        records = self.grouped[style]
        record = records[int(np.random.randint(0, len(records)))]
        x, y = representation_arrays(record, self.representation)
        if len(y) == 0:
            x = np.zeros((1, 6), dtype=np.float32)
            y = np.zeros((1,), dtype=np.int64)
        replace = len(y) < self.sample_points
        choice = np.random.choice(len(y), self.sample_points, replace=replace)
        return torch.from_numpy(x[choice].T).float(), torch.from_numpy(y[choice]).long()


def make_loader(dataset: Dataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def build_manifest(candidate_root: Path, protocol_dir: Path, split_seed: int) -> tuple[pd.DataFrame, Dict[str, object]]:
    style_files: Dict[str, List[Path]] = {}
    style_val_ids: Dict[str, set[str]] = {}
    for style in STYLE_KEYS:
        files = sorted((candidate_root / style).glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No candidate files found for {style}: {candidate_root / style}")
        style_files[style] = files
        rng = random.Random(f"{split_seed}:{style}")
        shuffled = list(files)
        rng.shuffle(shuffled)
        style_val_ids[style] = {path.stem for path in shuffled[: VAL_COUNTS[style]]}

    rows: List[Dict[str, object]] = []
    fold_audit: Dict[str, object] = {}
    for fold, spec in FOLDS.items():
        source = tuple(spec["source"])
        target = str(spec["target"])
        fold_rows: List[Dict[str, object]] = []
        for style in STYLE_KEYS:
            for path in style_files[style]:
                if style == target:
                    split = "test"
                elif style in source and path.stem in style_val_ids[style]:
                    split = "val"
                elif style in source:
                    split = "train"
                else:
                    continue
                row = {
                    "fold": fold,
                    "domain": "real",
                    "split": split,
                    "subset": style,
                    "style": style,
                    "sample_id": path.stem,
                    "path": str(path.resolve()),
                }
                rows.append(row)
                fold_rows.append(row)
        frame = pd.DataFrame(fold_rows)
        split_sets = {
            split: set(frame.loc[frame["split"] == split, "path"].tolist())
            for split in ("train", "val", "test")
        }
        overlap = {
            "train_val": sorted(split_sets["train"] & split_sets["val"]),
            "train_test": sorted(split_sets["train"] & split_sets["test"]),
            "val_test": sorted(split_sets["val"] & split_sets["test"]),
        }
        if any(overlap.values()):
            raise RuntimeError(f"Split leakage in {fold}: {overlap}")
        if set(frame.loc[frame["split"] == "test", "subset"]) != {target}:
            raise RuntimeError(f"Target-style audit failed for {fold}")
        fold_audit[fold] = {
            "source_styles": list(source),
            "target_style": target,
            "counts": frame["split"].value_counts().sort_index().to_dict(),
            "style_split_counts": {
                f"{style}:{split}": int(count)
                for (style, split), count in frame.groupby(["subset", "split"]).size().items()
            },
            "overlap": overlap,
            "target_absent_from_train_and_val": target not in set(frame.loc[frame["split"].isin(["train", "val"]), "subset"]),
        }

    manifest = pd.DataFrame(rows)
    manifest_path = protocol_dir / "cross_shoe_split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    audit = {
        "split_seed": split_seed,
        "validation_ids": {key: sorted(value) for key, value in style_val_ids.items()},
        "folds": fold_audit,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
    }
    json_dump(protocol_dir / "split_audit.json", audit)
    return manifest, audit


def load_record_store(
    manifest: pd.DataFrame,
    representation: str,
    token_cache_root: Path,
    cfg: object,
) -> Dict[tuple[str, str], Dict[str, object]]:
    records: List[Dict[str, object]] = []
    rows = manifest.drop_duplicates(["subset", "sample_id", "path"]).to_dict("records")
    if representation == "superline_tokens":
        records = S.B.prepare_records(rows, token_cache_root, cfg)
        T.prepare_projection_support(records, cfg, "Prepare token projection support")
    else:
        for row in rows:
            candidate = S.M.load_candidate_npy(str(row["path"]))
            raw = P.load_raw_metadata(str(row["subset"]), str(row["sample_id"]))
            if len(candidate["orig_index"]) and int(np.max(candidate["orig_index"])) >= int(raw["raw_num_points"]):
                raise RuntimeError(f"orig_index out of range: {row['subset']}/{row['sample_id']}")
            records.append(
                {
                    **row,
                    "candidate": candidate,
                    "raw_num_points": int(raw["raw_num_points"]),
                }
            )
    return {(str(record["subset"]), str(record["sample_id"])): record for record in records}


def fold_records_from_store(
    fold_frame: pd.DataFrame,
    record_store: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    records = []
    for row in fold_frame.to_dict("records"):
        key = (str(row["subset"]), str(row["sample_id"]))
        if key not in record_store:
            raise KeyError(f"Missing prepared record: {key}")
        records.append({**record_store[key], **row})
    return (
        [record for record in records if record["split"] == "train"],
        [record for record in records if record["split"] == "val"],
        [record for record in records if record["split"] == "test"],
    )


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    record: Mapping[str, object],
    representation: str,
    device: torch.device,
    chunk_points: int,
) -> np.ndarray:
    x, y = representation_arrays(record, representation)
    if len(y) == 0:
        return np.zeros((0,), dtype=np.float32)
    parts: List[np.ndarray] = []
    model.eval()
    for start in range(0, len(y), chunk_points):
        end = min(start + chunk_points, len(y))
        xb = torch.from_numpy(x[start:end].T[None]).float().to(device)
        logits = T.forward_logits(model, xb)
        parts.append(F.softmax(logits, dim=1)[:, 1, :].cpu().numpy().reshape(-1).astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)


def candidate_space_probabilities(
    record: Mapping[str, object],
    representation: str,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if representation == "candidate_points":
        candidate = record["candidate"]
        return np.asarray(candidate["orig_index"], dtype=np.int64), probabilities
    support = record["projection_support"]
    candidate_probability, _ = P.project_probabilities(
        probabilities,
        support["memberships"],
        support["candidate_count"],
    )
    return np.asarray(support["candidate_orig_index"], dtype=np.int64), candidate_probability


def cache_record_probabilities(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, object]],
    representation: str,
    device: torch.device,
    chunk_points: int,
) -> List[tuple[Mapping[str, object], np.ndarray, np.ndarray]]:
    cached = []
    for record in records:
        prob = predict_probabilities(model, record, representation, device, chunk_points)
        orig_index, candidate_probability = candidate_space_probabilities(record, representation, prob)
        cached.append((record, orig_index, candidate_probability))
    return cached


def evaluate_cached(
    cached: Sequence[tuple[Mapping[str, object], np.ndarray, np.ndarray]],
    threshold: float,
    model_key: str,
    representation: str,
    split: str,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    for record, orig_index, candidate_probability in cached:
        metrics = P.counts_for_prediction(
            str(record["subset"]),
            str(record["sample_id"]),
            orig_index,
            candidate_probability,
            threshold,
        )
        rows.append(
            {
                "model_key": model_key,
                "method": T.MODEL_TITLES[model_key],
                "representation": representation,
                "split": split,
                "subset": str(record["subset"]),
                "sample_id": str(record["sample_id"]),
                "classification_threshold": float(threshold),
                "metric_space": "original_raw_roi_point_space",
                "candidate_points": int(len(orig_index)),
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    macro = {
        key: float(frame[key].mean()) if len(frame) else 0.0
        for key in ("iou", "precision", "recall", "f1")
    }
    return frame, macro


def select_threshold(
    cached: Sequence[tuple[Mapping[str, object], np.ndarray, np.ndarray]],
    model_key: str,
    representation: str,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLDS:
        frame, macro = evaluate_cached(cached, threshold, model_key, representation, "source_validation")
        counts = {key: int(frame[key].sum()) for key in ("tp", "fp", "fn", "tn")}
        pooled = P.binary_metrics_from_counts(**counts)
        rows.append(
            {
                "threshold": threshold,
                "macro_iou": macro["iou"],
                "macro_precision": macro["precision"],
                "macro_recall": macro["recall"],
                "macro_f1": macro["f1"],
                "pooled_f1": pooled["f1"],
                **counts,
            }
        )
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["macro_iou", "pooled_f1", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), grid


def train_one_run(
    fold: str,
    model_key: str,
    representation: str,
    fold_frame: pd.DataFrame,
    record_store: Mapping[tuple[str, str], Mapping[str, object]],
    args: argparse.Namespace,
    run_dir: Path,
) -> Dict[str, object]:
    T.set_seed(args.seed)
    device = torch.device(args.device)
    train_records, val_records, test_records = fold_records_from_store(fold_frame, record_store)
    if args.smoke:
        source_styles = tuple(FOLDS[fold]["source"])
        train_records = [
            record
            for style in source_styles
            for record in [r for r in train_records if r["subset"] == style][:2]
        ]
        val_records = [
            record
            for style in source_styles
            for record in [r for r in val_records if r["subset"] == style][:1]
        ]
        test_records = test_records[: min(2, len(test_records))]
    train_ds = BalancedRealViewDataset(
        train_records,
        representation,
        args.sample_points,
        args.views_per_style,
    )
    loader = make_loader(train_ds, args.batch_size, args.seed)
    model = T.build_model(model_key, device)
    criterion = T.B0.Stage2Loss(pos_weight=2.0, use_smooth=False, smooth_weight=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "best_model.pt"
    best_iou = -1.0
    best_epoch = 0
    patience_count = 0
    epoch_rows: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = T.train_one_epoch(model, loader, criterion, optimizer, device)
        val_cached = cache_record_probabilities(model, val_records, representation, device, args.chunk_points)
        _, val_macro = evaluate_cached(val_cached, 0.5, model_key, representation, "source_validation_checkpoint")
        scheduler.step()
        is_best = val_macro["iou"] > best_iou
        if is_best:
            best_iou = val_macro["iou"]
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience_count += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "seconds": time.time() - start,
                "lr": optimizer.param_groups[0]["lr"],
                "is_best": int(is_best),
                "train_loss": train_metrics["loss"],
                "train_iou_sampled_space": train_metrics["iou"],
                "validation_macro_iou_full_space_at_0p5": val_macro["iou"],
                "validation_macro_precision_full_space_at_0p5": val_macro["precision"],
                "validation_macro_recall_full_space_at_0p5": val_macro["recall"],
                "validation_macro_f1_full_space_at_0p5": val_macro["f1"],
                "patience": patience_count,
            }
        )
        write_csv(run_dir / "epoch_metrics.csv", epoch_rows)
        print(
            f"[{fold}/{model_key}/{representation}] epoch={epoch:03d} "
            f"val_mIoU@0.5={val_macro['iou']:.4f} best={best_iou:.4f}@{best_epoch}",
            flush=True,
        )
        if patience_count >= args.patience:
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    val_cached = cache_record_probabilities(model, val_records, representation, device, args.chunk_points)
    threshold, grid = select_threshold(val_cached, model_key, representation)
    grid.to_csv(run_dir / "source_validation_threshold_grid.csv", index=False)
    val_frame, val_macro = evaluate_cached(
        val_cached, threshold, model_key, representation, "source_validation"
    )
    test_cached = cache_record_probabilities(model, test_records, representation, device, args.chunk_points)
    test_frame, test_macro = evaluate_cached(
        test_cached, threshold, model_key, representation, "held_out_shoe_test"
    )
    val_frame.to_csv(run_dir / "source_validation_per_case.csv", index=False)
    test_frame.to_csv(run_dir / "held_out_shoe_test_per_case.csv", index=False)
    summary = {
        "fold": fold,
        "source_styles": list(FOLDS[fold]["source"]),
        "target_style": FOLDS[fold]["target"],
        "model_key": model_key,
        "model": T.MODEL_TITLES[model_key],
        "representation": representation,
        "seed": args.seed,
        "train_scans": len(train_records),
        "validation_scans": len(val_records),
        "test_scans": len(test_records),
        "views_per_epoch": len(train_ds),
        "best_epoch": best_epoch,
        "best_validation_macro_iou_at_0p5": best_iou,
        "selected_classification_threshold": threshold,
        "validation_macro": val_macro,
        "test_macro": test_macro,
        "metric_space": "original_raw_roi_point_space",
        "target_used_for_selection": False,
        "smoke": args.smoke,
    }
    json_dump(run_dir / "summary.json", summary)
    return summary


def runtime_audit(args: argparse.Namespace, manifest: pd.DataFrame, split_audit: Mapping[str, object]) -> Dict[str, object]:
    dependencies = {
        "runner": Path(__file__).resolve(),
        "figure9_token_runner": (ROOT / "scripts/formal_protocol/run_superline_token_baselines_pointnet.py").resolve(),
        "baseline_implementation": T.BASELINE_SCRIPT.resolve(),
        "token_training_core": (ROOT / "scripts/formal_protocol/stenet_training_core_d04_1000.py").resolve(),
        "token_construction": Path(S.R.__file__).resolve(),
        "projection": Path(P.__file__).resolve(),
    }
    missing_cache = []
    data_rows = []
    unique_rows = manifest.drop_duplicates(["subset", "sample_id", "path"])
    for row in unique_rows.to_dict("records"):
        candidate = S.M.load_candidate_npy(str(row["path"]))
        cache_path = args.token_cache_root / str(row["subset"]) / f"{row['sample_id']}_downsampled.npz"
        if not cache_path.is_file():
            missing_cache.append(str(cache_path))
        raw = P.load_raw_metadata(str(row["subset"]), str(row["sample_id"]))
        data_rows.append(
            {
                "subset": row["subset"],
                "sample_id": row["sample_id"],
                "candidate_points": len(candidate["label"]),
                "candidate_positive_points": int(np.sum(candidate["label"] == 1)),
                "raw_points": int(raw["raw_num_points"]),
                "candidate_sha256": sha256(Path(row["path"])),
                "token_cache": str(cache_path.resolve()),
                "token_cache_sha256": sha256(cache_path) if cache_path.is_file() else "MISSING",
            }
        )
    if missing_cache:
        raise FileNotFoundError(
            "The pilot treats the shared formal token cache as read-only, but entries are missing: "
            + "; ".join(missing_cache[:5])
        )
    data_inventory = args.protocol_dir / "real_data_inventory.csv"
    pd.DataFrame(data_rows).to_csv(data_inventory, index=False)
    audit = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "candidate_root": str(args.candidate_root.resolve()),
        "token_cache_root": str(args.token_cache_root.resolve()),
        "token_cache_policy": "read_only_existing_formal_cache",
        "split_audit": split_audit,
        "data_inventory": str(data_inventory.resolve()),
        "data_inventory_sha256": sha256(data_inventory),
        "dependency_sha256": {name: {"path": str(path), "sha256": sha256(path)} for name, path in dependencies.items()},
    }
    json_dump(args.protocol_dir / "runtime_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=P.CANDIDATE_ROOT)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--token-cache-root", type=Path, default=P.TOKEN_CACHE_ROOT)
    parser.add_argument("--models", nargs="+", choices=list(T.MODEL_TITLES), default=list(DEFAULT_MODELS))
    parser.add_argument("--representations", nargs="+", choices=list(REPRESENTATIONS), default=list(REPRESENTATIONS))
    parser.add_argument("--folds", nargs="+", choices=list(FOLDS), default=list(FOLDS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260908)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--views-per-style", type=int, default=200)
    parser.add_argument("--sample-points", type=int, default=4096)
    parser.add_argument("--chunk-points", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.candidate_root = args.candidate_root.resolve()
    args.offline_root = args.offline_root.resolve()
    args.token_cache_root = args.token_cache_root.resolve()
    P.CANDIDATE_ROOT = args.candidate_root
    P.BASELINE_OFFLINE_ROOT = args.offline_root
    P.TOKEN_CACHE_ROOT = args.token_cache_root
    args.protocol_dir = args.output_root / "00_protocol"
    protected_outputs = [
        args.output_root / "runs",
        args.output_root / "pilot_summary.csv",
        args.output_root / "pilot_summary_partial.csv",
        args.output_root / "completion.json",
    ]
    if not args.prepare_only and any(path.exists() for path in protected_outputs):
        raise FileExistsError(f"Protected formal outputs already exist under: {args.output_root}")
    args.protocol_dir.mkdir(parents=True, exist_ok=True)
    manifest, split_audit = build_manifest(args.candidate_root, args.protocol_dir, args.split_seed)
    audit = runtime_audit(args, manifest, split_audit)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "split_audit": split_audit, "runtime": audit}, indent=2))
        return 0

    cfg_args = SimpleNamespace(
        candidate_dir=args.candidate_root,
        output_root=args.output_root,
        seed=args.seed,
        device=args.device,
    )
    cfg = S.build_compact_config(cfg_args)
    record_stores = {
        representation: load_record_store(manifest, representation, args.token_cache_root, cfg)
        for representation in args.representations
    }
    summaries: List[Dict[str, object]] = []
    for fold in args.folds:
        fold_frame = manifest[manifest["fold"] == fold].copy()
        for model_key in args.models:
            for representation in args.representations:
                run_dir = args.output_root / "runs" / fold / model_key / representation
                summaries.append(
                    train_one_run(
                        fold,
                        model_key,
                        representation,
                        fold_frame,
                        record_stores[representation],
                        args,
                        run_dir,
                    )
                )
                pd.DataFrame(summaries).to_csv(args.output_root / "pilot_summary_partial.csv", index=False)
                json_dump(args.output_root / "pilot_summary_partial.json", summaries)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_root / "pilot_summary.csv", index=False)
    json_dump(args.output_root / "pilot_summary.json", summaries)
    json_dump(args.output_root / "completion.json", {"status": "completed", "runs": len(summaries)})
    print(summary_frame[["fold", "model", "representation", "target_style", "test_macro"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
