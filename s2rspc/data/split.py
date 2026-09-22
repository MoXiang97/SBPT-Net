"""Source-only synthetic training/validation split utilities."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def synthetic_style_id(path: str | Path) -> str:
    stem = Path(path).stem
    match = re.match(r"^design_\d+_([^-_]+)-\d+", stem)
    if match:
        return match.group(1)
    base = stem.split("_fused", 1)[0]
    return base.split("-", 1)[0]


def source_only_split(
    files: Iterable[str | Path],
    train_count: int,
    validation_count: int,
    seed: int,
    holdout_styles: Sequence[str],
) -> Tuple[List[Path], List[Path]]:
    """Create a style-held-out split without consulting any real point cloud."""
    paths = sorted(Path(path) for path in files)
    holdout = {style.strip().lower() for style in holdout_styles if style.strip()}
    if holdout:
        train_pool = [p for p in paths if synthetic_style_id(p).lower() not in holdout]
        validation_pool = [p for p in paths if synthetic_style_id(p).lower() in holdout]
        if len(train_pool) < train_count or len(validation_pool) < validation_count:
            raise RuntimeError(
                "Insufficient files for the requested style-held-out split: "
                f"train_pool={len(train_pool)}, validation_pool={len(validation_pool)}"
            )
    else:
        if len(paths) < train_count + validation_count:
            raise RuntimeError(
                "Insufficient files for the requested disjoint split: "
                f"available={len(paths)}, requested={train_count + validation_count}"
            )
        shuffled = list(paths)
        rng = random.Random(int(seed))
        rng.shuffle(shuffled)
        train = sorted(shuffled[: int(train_count)])
        validation = sorted(
            shuffled[int(train_count) : int(train_count) + int(validation_count)]
        )
        return train, validation
    rng = random.Random(int(seed))
    rng.shuffle(train_pool)
    rng.shuffle(validation_pool)
    train = sorted(train_pool[: int(train_count)])
    validation = sorted(validation_pool[: int(validation_count)])
    if set(train) & set(validation):
        raise RuntimeError("Training and validation splits overlap")
    return train, validation
