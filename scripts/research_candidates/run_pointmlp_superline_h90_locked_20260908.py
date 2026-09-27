#!/usr/bin/env python
"""PointMLP backbone and line-aware structural refiner components."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_SOURCE = ROOT / "scripts" / "model_components"
if str(DIAGNOSTIC_SOURCE) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTIC_SOURCE))

import common as C  # noqa: E402
import search as H  # noqa: E402


DEFAULT_OUTPUT = ROOT / "experiments" / "PointMLP_Superline_H90_Locked_20260908"
SOURCE_CACHE_NAME = "source_only_dataset.pkl"
BASE_CHECKPOINTS = {
    42: ROOT / "experiments" / "SuperlineToken_Baselines_Remaining_CenteredXYZ_20260812" / "pointmlp" / "best_pointmlp_superline_tokens.pt",
    43: ROOT / "experiments" / "SuperlineToken_Backbone_MultiSeed_20260907" / "runs" / "seed43" / "pointmlp" / "pointmlp" / "best_pointmlp_superline_tokens.pt",
    44: ROOT / "experiments" / "SuperlineToken_Backbone_MultiSeed_20260907" / "runs" / "seed44" / "pointmlp" / "pointmlp" / "best_pointmlp_superline_tokens.pt",
}
EXPECTED_BASE_HASHES = {
    42: "64101693ca90c7da7ad4f28cea8dafc5047db599e1c49a5c14faeac532c11906",
    43: "595954acdbd791f7ff78ed60d3e0dcb3cb9df44e6eda9f6034eb03f9d4dd0b4f",
    44: "b36c1624f0120ab351f4dca65edcbe8ec989c21fb42464c0bbb05b13b3b5ef33",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_source_cache(output_root: Path) -> Dict[str, object]:
    """Create a cache whose serialized object has no real-test key or records."""
    destination = output_root / SOURCE_CACHE_NAME
    audit_path = output_root / "source_only_cache_audit.json"
    if destination.exists() or audit_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing source-only cache: {destination}"
        )
    full_data = C.load_data(False)
    counts = {split: len(records) for split, records in full_data.items()}
    if counts != {"train": 800, "val": 200, "test": 67}:
        raise RuntimeError(f"Unexpected full-cache counts: {counts}")
    source_data = {
        "train": full_data["train"],
        "val": full_data["val"],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(source_data, handle, protocol=4)
    temporary.replace(destination)
    del source_data, full_data
    with destination.open("rb") as handle:
        verified = pickle.load(handle)
    verified_counts = {split: len(records) for split, records in verified.items()}
    if set(verified) != {"train", "val"} or verified_counts != {
        "train": 800,
        "val": 200,
    }:
        raise RuntimeError(
            f"Invalid source-only cache contents: keys={list(verified)}, counts={verified_counts}"
        )
    del verified
    audit = {
        "status": "complete",
        "source_cache": str(destination),
        "source_cache_sha256": sha256(destination),
        "serialized_keys": ["train", "val"],
        "counts": verified_counts,
        "real_records_serialized": 0,
        "origin_cache": str(DIAGNOSTIC_SOURCE / "cache" / "dataset.pkl"),
        "origin_cache_sha256": sha256(DIAGNOSTIC_SOURCE / "cache" / "dataset.pkl"),
        "purpose": "physical source/test separation; no model training or selection",
    }
    write_json(audit_path, audit)
    return audit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def candidate_config(seed: int, checkpoint: Path) -> Dict[str, object]:
    return {
        "name": f"locked_line_invariant_consistency_seed{seed}",
        "architecture_id": "pointmlp_superline_line_invariant_consistency_v1",
        "seed": int(seed),
        "base_checkpoint": str(checkpoint),
        "patch": True,
        "context": True,
        "geometry_aux": True,
        "independent_geometry": True,
        "pos_weight": 1.0,
        "reg": 0.0001,
        "lr": 0.0005,
        "weight_decay": 0.0002,
        "augment": True,
        "invariant": True,
        "rotation_invariant": True,
        "geometry_weight": 0.5,
        "base_logit_cap": 3.0,
        "extra_features": "all",
        "aux_weight": 0.1,
        "descriptor_style_aug": True,
        "extra_style_aug": True,
        "extra_feature_drop": 0.1,
        "consistency_weight": 2.0,
    }


def verify_base_checkpoint(seed: int) -> Path:
    checkpoint = BASE_CHECKPOINTS[int(seed)]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed = sha256(checkpoint)
    expected = EXPECTED_BASE_HASHES[int(seed)]
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"Seed {seed} base checkpoint hash mismatch: {observed} != {expected}"
        )
    return checkpoint


def clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def refresh_frozen_logits(
    records: Sequence[Mapping[str, object]],
    checkpoint: Path,
    seed: int,
    device: torch.device,
    stage: str,
) -> None:
    model = C.R.build_model("pointmlp", device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            record_seed = C.seed_for(record)
            torch.manual_seed(record_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(record_seed)
            logits: List[np.ndarray] = []
            features = np.asarray(record["x"], dtype=np.float32)
            for start in range(0, len(features), 4096):
                tensor = torch.from_numpy(features[start : start + 4096].T[None]).to(device)
                output = C.R.forward_logits(model, tensor)
                logits.append((output[:, 1] - output[:, 0]).cpu().numpy().reshape(-1))
            record["logit"] = np.concatenate(logits).astype(np.float32)
            if index % 100 == 0 or index == len(records):
                print(f"seed{seed} {stage}: refreshed frozen logits {index}/{len(records)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_epoch(
    model: H.ResidualHead,
    records: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    tokens_per_cloud: int,
    batch_clouds: int,
) -> float:
    model.train()
    order = np.random.permutation(len(records))
    losses: List[float] = []
    positive_weight = torch.tensor(float(config["pos_weight"]), device="cuda")
    for start in range(0, len(order), int(batch_clouds)):
        selected_records = [records[index] for index in order[start : start + batch_clouds]]
        selected_indices = [
            np.random.choice(
                len(record["y"]),
                min(len(record["y"]), int(tokens_per_cloud)),
                replace=False,
            )
            for record in selected_records
        ]
        augmented = H.batch(selected_records, selected_indices, config, augment=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(augmented)
        target = augmented["y"]
        probability = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=positive_weight,
        )
        loss = loss + 0.5 * (
            1.0
            - (2.0 * (probability * target).sum() + 1.0)
            / (probability.sum() + target.sum() + 1.0)
        )
        loss = loss + float(config["reg"]) * (
            logits - augmented["logit"].clamp(-10, 10)
        ).square().mean()

        auxiliary_logits = model.aux_logit
        auxiliary_probability = torch.sigmoid(auxiliary_logits)
        auxiliary_loss = F.binary_cross_entropy_with_logits(
            auxiliary_logits,
            target,
            pos_weight=positive_weight,
        )
        auxiliary_loss = auxiliary_loss + 0.5 * (
            1.0
            - (2.0 * (auxiliary_probability * target).sum() + 1.0)
            / (auxiliary_probability.sum() + target.sum() + 1.0)
        )
        loss = loss + float(config["aux_weight"]) * auxiliary_loss

        clean = H.batch(selected_records, selected_indices, config, augment=False)
        clean_logits = model(clean)
        consistency = F.mse_loss(
            torch.sigmoid(logits),
            torch.sigmoid(clean_logits),
        )
        loss = loss + float(config["consistency_weight"]) * consistency

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def evaluate_validation_at_fixed_threshold(
    model: H.ResidualHead,
    validation_records: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> tuple[float, List[np.ndarray]]:
    probabilities = H.infer_head(model, validation_records, config)
    score = float(C.summarize(validation_records, probabilities, 0.5)["mean"])
    return score, probabilities


def train_seed(
    seed: int,
    data: Dict[str, list],
    output_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    seed_dir = output_root / f"seed{seed}"
    if seed_dir.exists() and any(seed_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {seed_dir}")
    seed_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = verify_base_checkpoint(seed)
    refresh_frozen_logits(
        list(data["train"]) + list(data["val"]),
        checkpoint,
        seed,
        device,
        "source",
    )
    config = candidate_config(seed, checkpoint)
    set_seed(seed)
    model = H.ResidualHead(config).to(device)
    trainable_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs),
        eta_min=0.00005,
    )

    best_score = -1.0
    best_epoch = 0
    stale = 0
    history: List[Dict[str, object]] = []
    best_path = seed_dir / "best_source_selected.pt"
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.time()
        train_loss = train_epoch(
            model,
            data["train"],
            config,
            optimizer,
            int(args.tokens_per_cloud),
            int(args.batch_clouds),
        )
        validation_score, _ = evaluate_validation_at_fixed_threshold(
            model,
            data["val"],
            config,
        )
        scheduler.step()
        improved = validation_score > best_score
        if improved:
            best_score = validation_score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": clone_state_dict(model),
                    "config": config,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_iou_at_0p5": validation_score,
                    "base_checkpoint": str(checkpoint),
                    "base_sha256": sha256(checkpoint),
                },
                best_path,
            )
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_original_point_mean_iou_at_0p5": validation_score,
            "best_validation_iou_at_0p5": best_score,
            "best_epoch": best_epoch,
            "patience": stale,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - epoch_started),
        }
        history.append(row)
        write_json(seed_dir / "history.json", history)
        print(
            f"seed{seed} epoch {epoch:03d}: loss={train_loss:.5f} "
            f"val_iou@0.5={validation_score:.5f} best={best_score:.5f}@{best_epoch}",
            flush=True,
        )
        if stale >= int(args.patience):
            print(f"seed{seed}: early stopping at epoch {epoch}", flush=True)
            break

    selected = torch.load(best_path, map_location="cpu")
    model.load_state_dict(selected["state_dict"], strict=True)
    model.to(device)
    validation_probabilities = H.infer_head(model, data["val"], config)
    selected_threshold, threshold_grid = C.choose_threshold(
        data["val"],
        validation_probabilities,
    )
    validation_final = C.summarize(
        data["val"],
        validation_probabilities,
        selected_threshold,
    )
    result = {
        "status": "source_selected_complete_real_not_evaluated",
        "architecture_id": config["architecture_id"],
        "seed": seed,
        "base_checkpoint": str(checkpoint),
        "base_sha256": sha256(checkpoint),
        "trainable_parameters": trainable_parameters,
        "best_epoch": best_epoch,
        "best_validation_original_point_mean_iou_at_0p5": best_score,
        "selected_threshold": selected_threshold,
        "synthetic_validation_miou_at_selected_threshold": validation_final["mean"],
        "epochs_executed": len(history),
        "seconds": float(time.time() - started),
        "checkpoint": str(best_path),
        "threshold_grid": [float(value) for value in threshold_grid],
        "config": config,
        "selection_rule": "synthetic validation original-point mean per-scan IoU at fixed threshold 0.5; earlier epoch wins ties",
        "real_data_access": "none during this stage",
        "source_only_cache": str(output_root / SOURCE_CACHE_NAME),
        "source_only_cache_sha256": sha256(output_root / SOURCE_CACHE_NAME),
    }
    selected["selected_threshold"] = selected_threshold
    selected["synthetic_validation_miou_at_selected_threshold"] = validation_final["mean"]
    torch.save(selected, best_path)
    write_json(seed_dir / "source_selection.json", result)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_seed(
    seed: int,
    data: Dict[str, list],
    output_root: Path,
    device: torch.device,
) -> Dict[str, object]:
    seed_dir = output_root / f"seed{seed}"
    selection_path = seed_dir / "source_selection.json"
    checkpoint_path = seed_dir / "best_source_selected.pt"
    if not selection_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Incomplete source selection for seed {seed}")
    source_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint = verify_base_checkpoint(seed)
    refresh_frozen_logits(
        list(data["val"]) + list(data["test"]),
        checkpoint,
        seed,
        device,
        "final-evaluate",
    )

    base_validation_probabilities = [
        torch.sigmoid(torch.from_numpy(record["logit"])).numpy()
        for record in data["val"]
    ]
    base_threshold, _ = C.choose_threshold(data["val"], base_validation_probabilities)
    base_test_probabilities = [
        torch.sigmoid(torch.from_numpy(record["logit"])).numpy()
        for record in data["test"]
    ]
    base_validation = C.summarize(
        data["val"],
        base_validation_probabilities,
        base_threshold,
    )
    base_test = C.summarize(
        data["test"],
        base_test_probabilities,
        base_threshold,
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload["config"]
    model = H.ResidualHead(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    selected_threshold = float(source_selection["selected_threshold"])
    validation_probabilities = H.infer_head(model, data["val"], config)
    test_probabilities = H.infer_head(model, data["test"], config)
    candidate_validation = C.summarize(
        data["val"],
        validation_probabilities,
        selected_threshold,
    )
    candidate_test = C.summarize(
        data["test"],
        test_probabilities,
        selected_threshold,
    )
    result = {
        "seed": seed,
        "architecture_id": config["architecture_id"],
        "base": {
            "selected_threshold": base_threshold,
            "synthetic_validation_miou": base_validation["mean"],
            "real_test_equal_subset_miou": base_test["mean"],
            "real_test_subsets": base_test["subsets"],
        },
        "candidate": {
            "best_epoch": source_selection["best_epoch"],
            "best_validation_iou_at_0p5": source_selection[
                "best_validation_original_point_mean_iou_at_0p5"
            ],
            "selected_threshold": selected_threshold,
            "synthetic_validation_miou": candidate_validation["mean"],
            "real_test_equal_subset_miou": candidate_test["mean"],
            "real_test_subsets": candidate_test["subsets"],
        },
        "paired_real_test_gain": candidate_test["mean"] - base_test["mean"],
        "real_test_access": "one final evaluation after all source-selected checkpoints completed",
    }
    write_json(seed_dir / "final_evaluation.json", result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def aggregate(results: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    rows = list(results)
    summary: Dict[str, object] = {
        "experiment_id": "20260908_pointmlp_superline_h90_locked_three_seed",
        "selection": "all checkpoints and thresholds selected using synthetic validation only",
        "rows": rows,
    }
    for label, extractor in {
        "base_real_test": lambda row: row["base"]["real_test_equal_subset_miou"],
        "candidate_real_test": lambda row: row["candidate"]["real_test_equal_subset_miou"],
        "paired_gain": lambda row: row["paired_real_test_gain"],
    }.items():
        values = np.asarray([extractor(row) for row in rows], dtype=np.float64)
        summary[label] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": [float(value) for value in values],
        }
    return summary


def load_data(stage: str, smoke: bool, output_root: Path) -> Dict[str, list]:
    if stage == "train":
        source_cache = output_root / SOURCE_CACHE_NAME
        if not source_cache.is_file():
            raise FileNotFoundError(
                f"Prepare the physical source-only cache first: {source_cache}"
            )
        with source_cache.open("rb") as handle:
            data = pickle.load(handle)
        if set(data) != {"train", "val"}:
            raise RuntimeError(
                f"Training cache must contain only train and val, observed {list(data)}"
            )
    else:
        data = C.load_data(False)
    C.ensure_extra_features(data)
    if smoke:
        subset = {
            "train": list(data["train"][:8]),
            "val": list(data["val"][:4]),
        }
        if "test" in data:
            subset["test"] = list(data["test"][:3])
        return subset
    counts = {split: len(records) for split, records in data.items()}
    expected = (
        {"train": 800, "val": 200}
        if stage == "train"
        else {"train": 800, "val": 200, "test": 67}
    )
    if counts != expected:
        raise RuntimeError(f"Unexpected dataset counts: {counts}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prepare-source", "train", "final-evaluate"],
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-clouds", type=int, default=4)
    parser.add_argument("--tokens-per-cloud", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unsupported = sorted(set(args.seeds) - set(BASE_CHECKPOINTS))
    if unsupported:
        raise ValueError(f"Unsupported seeds: {unsupported}")
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare-source":
        audit = prepare_source_cache(args.output_root)
        print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
        return 0
    device = torch.device(args.device)
    data = load_data(args.stage, args.smoke, args.output_root)
    if args.stage == "train":
        source_results = []
        for seed in args.seeds:
            source_results.append(train_seed(seed, data, args.output_root, args, device))
        write_json(
            args.output_root / "source_selection_summary.json",
            {
                "status": "source_selection_complete_real_not_evaluated",
                "seeds": list(args.seeds),
                "results": source_results,
            },
        )
        return 0

    missing = [
        seed
        for seed in args.seeds
        if not (args.output_root / f"seed{seed}" / "source_selection.json").is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Refusing final evaluation before all source selections complete: {missing}"
        )
    final_results = [
        evaluate_seed(seed, data, args.output_root, device)
        for seed in args.seeds
    ]
    summary = aggregate(final_results)
    write_json(args.output_root / "final_results.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
