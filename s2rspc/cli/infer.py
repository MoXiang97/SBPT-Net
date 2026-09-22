"""Run preprocessing and frozen SBPT-Net inference on one point cloud."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from ..config import load_config
from ..engine.checkpoint import load_checkpoint
from ..engine.trainer import predict_cloud
from ..evaluation.projection import project_tokens_to_original_points
from ..preprocessing.pipeline import preprocess_point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/sbptnet_sts2r.yaml")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
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
    _, cloud = preprocess_point_cloud(args.input, cfg)
    model, mean, std, checkpoint = load_checkpoint(args.checkpoint, cfg.model, device)
    probability = predict_cloud(model, cloud, mean, std, device)["probability"]
    threshold = float(
        checkpoint.get("metadata", {}).get(
            "threshold", cfg.evaluation.projection_threshold
        )
    )
    projection = project_tokens_to_original_points(cloud, probability, threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **projection, token_probability=probability)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
