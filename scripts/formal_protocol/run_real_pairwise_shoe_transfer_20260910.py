#!/usr/bin/env python
"""Run leakage-safe directed real-to-real shoe-style transfer experiments.

This entrypoint deliberately reuses the audited real cross-shoe runner for
model construction, losses, superline tokens, checkpoint selection, threshold
calibration, projection, and full original-point evaluation.  It changes only
the domain protocol from two source styles to one source style and defines all
six directed transfers among Shoe A, Shoe B, and Shoe C.

Use ``--views-per-style 400`` for formal runs so that a single-source epoch has
the same 400 stochastic training views as the earlier two-source protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from formal_protocol import run_real_cross_shoe_superline_pilot_20260908 as base  # noqa: E402


PAIRWISE_FOLDS = {
    "a_to_b": {"source": ("shoe_a",), "target": "shoe_b"},
    "b_to_a": {"source": ("shoe_b",), "target": "shoe_a"},
    "a_to_c": {"source": ("shoe_a",), "target": "shoe_c"},
    "b_to_c": {"source": ("shoe_b",), "target": "shoe_c"},
    "c_to_a": {"source": ("shoe_c",), "target": "shoe_a"},
    "c_to_b": {"source": ("shoe_c",), "target": "shoe_b"},
}


class SingleSourceRealViewDataset(Dataset):
    """Create a fixed number of stochastic views from exactly one source style."""

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
        if len(self.styles) != 1:
            raise ValueError(f"Each pairwise fold requires exactly one source style, got {self.styles}")
        self.style = self.styles[0]
        self.records = sorted(grouped[self.style], key=lambda record: str(record["sample_id"]))
        if not self.records:
            raise ValueError(f"No training records found for source style {self.style}")
        if self.views_per_style <= 0:
            raise ValueError("views_per_style must be positive")

    def __len__(self) -> int:
        return self.views_per_style

    def __getitem__(self, idx: int):
        del idx
        record = self.records[int(np.random.randint(0, len(self.records)))]
        x, y = base.representation_arrays(record, self.representation)
        if len(y) == 0:
            x = np.zeros((1, 6), dtype=np.float32)
            y = np.zeros((1,), dtype=np.int64)
        replace = len(y) < self.sample_points
        choice = np.random.choice(len(y), self.sample_points, replace=replace)
        return torch.from_numpy(x[choice].T).float(), torch.from_numpy(y[choice]).long()


def main() -> int:
    base.FOLDS = PAIRWISE_FOLDS
    base.DEFAULT_MODELS = ("pointtransformer",)
    base.BalancedRealViewDataset = SingleSourceRealViewDataset
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
