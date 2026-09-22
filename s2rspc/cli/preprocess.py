"""Preprocess raw XYZRGBLabel point clouds into candidates and tokens."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ..config import load_config
from ..data.io import save_candidate_cloud, save_tokenized_cloud
from ..preprocessing.pipeline import preprocess_point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sbptnet_sts2r.yaml"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--token-root", type=Path, default=None)
    parser.add_argument("--subsets", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    candidate_root = args.candidate_root or Path(cfg.data.candidate_root)
    token_root = args.token_root or Path(cfg.data.token_root)
    subsets = args.subsets or [cfg.data.synthetic_subset, *cfg.data.real_subsets]
    files = []
    for subset in subsets:
        files.extend((subset, path) for path in sorted((args.input_root / subset).glob("*.txt")))
    if not files:
        raise FileNotFoundError(f"No TXT point clouds found below {args.input_root}")
    for subset, raw_path in tqdm(files, desc="SBPT-Net preprocessing"):
        candidate_path = candidate_root / subset / f"{raw_path.stem}.npz"
        token_path = token_root / subset / f"{raw_path.stem}.npz"
        if token_path.exists() and not args.overwrite:
            continue
        candidate, tokenized = preprocess_point_cloud(raw_path, cfg)
        save_candidate_cloud(candidate_path, candidate)
        save_tokenized_cloud(token_path, tokenized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
