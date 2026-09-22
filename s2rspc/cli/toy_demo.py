"""Generate a non-paper toy cloud and exercise preprocessing and inference."""

from __future__ import annotations
import argparse, json, os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

from ..config import load_config
from ..data.io import save_tokenized_cloud
from ..engine.trainer import predict_cloud
from ..models.sbptnet import SBPTNet
from ..preprocessing.pipeline import preprocess_point_cloud


def make_toy_cloud(path: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(
        np.linspace(-1, 1, 70, dtype=np.float32),
        np.linspace(-0.45, 0.45, 30, dtype=np.float32),
    )
    xyz = np.column_stack(
        [
            gx.ravel(),
            gy.ravel(),
            0.025 * np.sin(2.5 * gx).ravel() + rng.normal(0, 0.0015, gx.size),
        ]
    ).astype(np.float32)
    cx = np.linspace(-0.9, 0.9, 180, dtype=np.float32)
    curve = np.column_stack(
        [cx, 0.12 * np.sin(2.8 * cx), 0.025 * np.sin(2.5 * cx) + 0.002]
    ).astype(np.float32)
    xyz = np.vstack([xyz, curve])
    label = np.zeros(len(xyz), dtype=np.float32)
    label[-len(curve) :] = 1
    rgb = np.full((len(xyz), 3), 185, dtype=np.float32)
    rgb[-len(curve) :] = 25
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([xyz, rgb, label]), fmt="%.7f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/sbptnet_sts2r.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("toy_output"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    raw_path = args.output / "toy_cloud.txt"
    make_toy_cloud(raw_path)
    candidate, cloud = preprocess_point_cloud(raw_path, cfg)
    if len(cloud["label"]) == 0:
        raise RuntimeError("Toy preprocessing produced no tokens")
    model = SBPTNet(cfg.model).eval()
    output = predict_cloud(
        model,
        cloud,
        np.zeros(6, np.float32),
        np.ones(6, np.float32),
        torch.device("cpu"),
    )
    save_tokenized_cloud(args.output / "toy_tokens.npz", cloud)
    report = {
        "raw_points": int(len(np.loadtxt(raw_path))),
        "candidate_points": int(len(candidate["xyz"])),
        "tokens": int(len(cloud["label"])),
        "initial_logit_shape": list(output["initial_logit"].shape),
        "structural_logit_shape": list(output["structural_logit"].shape),
        "final_logit_shape": list(output["final_logit"].shape),
    }
    (args.output / "toy_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
