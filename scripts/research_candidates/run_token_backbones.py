#!/usr/bin/env python
"""Train and evaluate a backbone with the SBPT-Net token representation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_candidates import run_sbpt_net as D

MODELS = ("pointnet", "pointnet2", "dgcnn", "pointtransformer", "pointnext", "curvenet")


def expected_root(seed: int, model: str) -> Path:
    return ROOT / "outputs" / "reproduction" / "backbones" / f"seed{seed}" / model


def run(seed: int, model: str, output_root: Path, candidate_root: Path, offline_root: Path, split_manifest: Path, token_cache: Path) -> int:
    if seed not in (42, 43, 44) or model not in MODELS:
        raise ValueError((seed, model))
    output_root = Path(output_root).resolve()
    D.CANDIDATE_ROOT = candidate_root.resolve()
    D.OFFLINE_ROOT = offline_root.resolve()
    D.SPLIT_MANIFEST = split_manifest.resolve()
    token_cache = token_cache.resolve()
    if not D.SPLIT_MANIFEST.is_file() or not token_cache.is_dir():
        raise FileNotFoundError("Split manifest or invariant token cache is missing")
    baseline = D.load_runtime_modules()["baseline"]
    argv = [
        str(Path(baseline.__file__)),
        "--models",
        model,
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
        "--split-manifest",
        str(D.SPLIT_MANIFEST),
        "--candidate-root",
        str(D.CANDIDATE_ROOT),
        "--token-cache-root",
        str(token_cache),
        "--device",
        "cuda",
        "--epochs",
        "100",
        "--patience",
        "15",
        "--batch-size",
        "8",
        "--sample-points",
        "4096",
        "--chunk-points",
        "4096",
        "--num-workers",
        "0",
        "--lr",
        "0.0005",
        "--weight-decay",
        "0.0001",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        with D.bind_invariant_tokenization(token_cache):
            code = int(baseline.main())
    finally:
        sys.argv = old_argv
    protocol_path = output_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["formula"] = "r_ij=min_q ||v_ij-(v_ij^T t_q)t_q||_2"
    protocol["edge_construction"] = "undirected union of directed kNN arcs with geometric tie ordering"
    protocol["invariant_superline_code"] = {
        "path": str(Path(D.I.__file__).resolve()),
        "sha256": D.sha256(Path(D.I.__file__)),
    }
    protocol["token_cache"] = str(token_cache)
    D.atomic_json(protocol_path, protocol)
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--token-cache-root", type=Path, required=True)
    args = parser.parse_args()
    return run(
        args.seed,
        args.model,
        args.output_root or expected_root(args.seed, args.model),
        args.candidate_root,
        args.offline_root,
        args.split_manifest,
        args.token_cache_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
