"""Evaluate a frozen checkpoint in the original raw point-cloud space."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch

from ..config import load_config
from ..data.io import load_tokenized_cloud
from ..engine.checkpoint import load_checkpoint
from ..engine.trainer import predict_cloud
from ..evaluation.metrics import binary_metrics
from ..evaluation.projection import project_tokens_to_original_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/sbptnet_sts2r.yaml")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-root", type=Path)
    parser.add_argument("--subsets", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, mean, std, checkpoint = load_checkpoint(args.checkpoint, cfg.model, device)
    threshold = float(
        checkpoint.get("metadata", {}).get(
            "threshold", cfg.evaluation.projection_threshold
        )
    )
    token_root = args.token_root or Path(cfg.data.token_root)
    subsets = args.subsets or cfg.data.real_subsets
    rows = []
    for subset in subsets:
        for path in sorted((token_root / subset).glob("*.npz")):
            cloud = load_tokenized_cloud(path)
            if "raw_label" not in cloud:
                raise ValueError(f"Original labels are absent from {path}")
            probability = predict_cloud(model, cloud, mean, std, device)["probability"]
            projected = project_tokens_to_original_points(cloud, probability, threshold)
            rows.append(
                {
                    "subset": subset,
                    "sample_id": path.stem,
                    **binary_metrics(cloud["raw_label"], projected["full_prediction"]),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_case_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(rows[0]) if rows else ["subset", "sample_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for subset in subsets:
        items = [row for row in rows if row["subset"] == subset]
        summary[subset] = {
            "cases": len(items),
            "miou": float(np.mean([row["iou"] for row in items])) if items else 0.0,
            "mean_precision": (
                float(np.mean([row["precision"] for row in items])) if items else 0.0
            ),
            "mean_recall": (
                float(np.mean([row["recall"] for row in items])) if items else 0.0
            ),
            "mean_f1": float(np.mean([row["f1"] for row in items])) if items else 0.0,
        }
    subset_rows = [summary[name] for name in subsets]
    summary["equal_subset_mean"] = {
        "subsets": len(subset_rows),
        "miou": (
            float(np.mean([item["miou"] for item in subset_rows]))
            if subset_rows
            else 0.0
        ),
        "mean_precision": (
            float(np.mean([item["mean_precision"] for item in subset_rows]))
            if subset_rows
            else 0.0
        ),
        "mean_recall": (
            float(np.mean([item["mean_recall"] for item in subset_rows]))
            if subset_rows
            else 0.0
        ),
        "mean_f1": (
            float(np.mean([item["mean_f1"] for item in subset_rows]))
            if subset_rows
            else 0.0
        ),
        "aggregation": "equal mean over subset macro means",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
