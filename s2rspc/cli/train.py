"""Train SBPT-Net using source synthetic tokens only."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..data.split import source_only_split
from ..engine.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sbptnet_sts2r.yaml"))
    parser.add_argument("--token-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    token_root = args.token_root or Path(cfg.data.token_root)
    synthetic_files = sorted((token_root / cfg.data.synthetic_subset).glob("*.npz"))
    train_files, validation_files = source_only_split(
        synthetic_files,
        cfg.data.train_samples,
        cfg.data.validation_samples,
        cfg.seed,
        cfg.data.holdout_styles if cfg.data.split_policy == "style_holdout" else [],
    )
    train(train_files, validation_files, args.output, cfg, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
