#!/usr/bin/env python
"""Run the frozen formal backbone script with an explicit training seed.

The original formal entry keeps seed 42 as a class constant. This wrapper does
not edit that source. It imports the exact entry, sets the requested seed, then
calls its unchanged main function with the remaining command-line arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sts2r_formal_seeded as formal_entry

ENTRY = (
    ROOT
    / "reference_code"
    / "STS2R_formal_baselines_100ep_w8"
    / "scripts"
    / "03_run_formal_baselines.py"
)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--training-seed", type=int, required=True)
    known, remaining = parser.parse_known_args()

    formal_entry.Config.SEED = int(known.training_seed)
    formal_entry.set_seed(int(known.training_seed))
    sys.argv = [str(ENTRY), *remaining]
    result = formal_entry.main()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
