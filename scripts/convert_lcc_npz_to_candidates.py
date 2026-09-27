"""Convert frozen LCC NPZ records into the N x 9 candidate format."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np


def convert_subset(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for npz_path in sorted(src.glob("*.npz")):
        record = np.load(npz_path, allow_pickle=False)
        columns = [
            record["xyz"],
            record["rgb"],
            record["label"].reshape(-1, 1),
            record["orig_index"].reshape(-1, 1),
            record["lcc_score"].reshape(-1, 1),
        ]
        candidate = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
        np.save(dst / f"{npz_path.stem}.npy", candidate)


def copy_subset(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for npy_path in sorted(src.glob("*.npy")):
        shutil.copy2(npy_path, dst / npy_path.name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lcc-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()

    convert_subset(args.lcc_root / "D04_AppGeoPhys1000", args.candidate_root / "D04_AppGeoPhys1000")
    copy_subset(args.candidate_root / "D04_AppGeoPhys1000", args.candidate_root / "synthetic")
    for subset in ("shoe_a", "shoe_b", "shoe_c"):
        convert_subset(args.lcc_root / subset, args.candidate_root / subset)

    for left, right in zip(
        sorted((args.candidate_root / "D04_AppGeoPhys1000").glob("*.npy")),
        sorted((args.candidate_root / "synthetic").glob("*.npy")),
    ):
        if sha256(left) != sha256(right):
            raise RuntimeError(f"synthetic copy mismatch: {left.name}")


if __name__ == "__main__":
    main()
