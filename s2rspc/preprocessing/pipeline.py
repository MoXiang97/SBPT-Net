"""End-to-end deterministic preprocessing for one point cloud."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..config import ExperimentConfig
from ..data.io import load_raw_point_cloud, pointmlp_record_seed
from .lcc import preserve_lcc_candidates
from .superline import construct_superlines
from .tokenization import tokenize_superlines


def preprocess_point_cloud(
    path: str | Path, cfg: ExperimentConfig
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    path = Path(path)
    raw = load_raw_point_cloud(path)
    candidate = preserve_lcc_candidates(
        raw,
        threshold=cfg.lcc.threshold,
        neighbors=cfg.lcc.neighbors,
        chunk_size=cfg.lcc.chunk_size,
    )
    superlines = construct_superlines(candidate, cfg.superline)
    tokenized = tokenize_superlines(candidate, superlines, cfg.tokenization)
    tokenized["_record_seed"] = np.asarray(
        pointmlp_record_seed(path.parent.name, path.stem), dtype=np.int64
    )
    return candidate, tokenized
