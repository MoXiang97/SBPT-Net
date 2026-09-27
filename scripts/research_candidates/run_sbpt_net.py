#!/usr/bin/env python
"""Train and evaluate SBPT-Net with invariant superline organization."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_candidates import invariant_superline_organization_20260917 as I  # noqa: E402
from scripts.shared_stage2_lib import sce_net_connectivity_graph_tokenization as T  # noqa: E402


SEED = 42
ARMS = ("token_encoding_only_no_consistency",)
EXPERIMENT_ROOT = ROOT / "outputs" / "reproduction" / "seed42"
CANDIDATE_ROOT = ROOT / "data" / "candidates"
OFFLINE_ROOT = ROOT / "data" / "lcc"
SPLIT_MANIFEST = EXPERIMENT_ROOT / "split_manifest.csv"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_RUNTIME_MODULES: dict[str, object] | None = None


def load_runtime_modules() -> dict[str, object]:
    global _RUNTIME_MODULES
    if _RUNTIME_MODULES is not None:
        return _RUNTIME_MODULES
    scripts_dir = ROOT / "scripts"
    research_dir = scripts_dir / "research_candidates"
    for directory in (scripts_dir, research_dir):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    baseline = importlib.import_module("formal_protocol.run_superline_token_baselines_pointnet")
    single_runner = importlib.import_module("run_h90_single_component_diagnostics_20260911")
    _RUNTIME_MODULES = {
        "baseline": baseline,
        "single_runner": single_runner,
        "base_runner": single_runner.Base,
        "locked": single_runner.T,
        "common": single_runner.C,
    }
    return _RUNTIME_MODULES


@contextmanager
def bind_invariant_tokenization(token_cache_root: Path):
    """Temporarily bind every active cache/reconstruction reference to this candidate."""

    baseline = load_runtime_modules()["baseline"]
    cache_root = Path(token_cache_root).resolve()
    bindings = [
        (baseline.S.B, "R", I),
        (baseline.S, "R", I),
        (baseline.P.S, "R", I),
        (baseline.P, "TOKEN_CACHE_ROOT", cache_root),
        (baseline.P, "CANDIDATE_ROOT", CANDIDATE_ROOT.resolve()),
        (baseline.P, "BASELINE_OFFLINE_ROOT", OFFLINE_ROOT.resolve()),
    ]
    old_values = [(owner, name, getattr(owner, name)) for owner, name, _value in bindings]
    old_method_root = I.SUPERLINE3D_STYLE_METHOD["cache_root"]
    try:
        for owner, name, value in bindings:
            setattr(owner, name, value)
        I.SUPERLINE3D_STYLE_METHOD["cache_root"] = cache_root
        yield baseline
    finally:
        I.SUPERLINE3D_STYLE_METHOD["cache_root"] = old_method_root
        for owner, name, old_value in reversed(old_values):
            setattr(owner, name, old_value)


def assert_isolated_root(output_root: Path) -> Path:
    resolved = Path(output_root).resolve()
    expected = EXPERIMENT_ROOT.resolve()
    if resolved != expected:
        raise ValueError(f"Output must use the selected root: {expected}")
    return resolved


def output_paths(output_root: Path) -> dict[str, str]:
    root = assert_isolated_root(output_root)
    return {
        "token_cache": str(root / "token_cache"),
        "pointmlp": str(root / "pointmlp"),
        "h90_cache": str(root / "h90_cache"),
        "strongest": str(root / "strongest"),
        "smoke": str(root / "smoke"),
        "summary": str(root / "summary.json"),
    }


def build_protocol(output_root: Path) -> dict[str, object]:
    root = assert_isolated_root(output_root)
    required = [SPLIT_MANIFEST, CANDIDATE_ROOT, OFFLINE_ROOT, Path(I.__file__)]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    return {
        "experiment_id": f"sbpt_net_invariant_residual_seed{SEED}",
        "purpose": (
            "Test the adopted strongest structural encoder with an orthogonal tangent "
            "residual and an index-invariant undirected kNN edge set."
        ),
        "created_at": now(),
        "seeds": [SEED],
        "arms": list(ARMS),
        "formula": "r_ij=min_q ||v_ij-(v_ij^T t_q)t_q||_2",
        "code": {
            "entrypoint": str(Path(__file__).relative_to(ROOT)),
            "entrypoint_sha256": sha256(Path(__file__)),
            "invariant_superline": str(Path(I.__file__).resolve().relative_to(ROOT)),
            "invariant_superline_sha256": sha256(Path(I.__file__)),
        },
        "data": {
            "candidate_root": str(CANDIDATE_ROOT),
            "offline_root": str(OFFLINE_ROOT),
            "split_manifest": str(SPLIT_MANIFEST),
            "split_manifest_sha256": sha256(SPLIT_MANIFEST),
            "counts": {"train": 800, "val": 200, "test": 67},
            "real_subsets": {"shoe_a": 26, "shoe_b": 24, "shoe_c": 17},
        },
        "training": {
            "pointmlp": {"epochs": 100, "patience": 15, "seed": SEED},
            "structural_encoder": {"epochs": 100, "patience": 15, "seed": SEED},
            "checkpoint_selection": "synthetic validation IoU at threshold 0.5",
            "threshold_selection": "synthetic validation mean per-scan IoU",
            "real_data_role": "evaluation after source checkpoint and threshold selection",
        },
        "checkpoint_selection": {
            "dataset": "synthetic validation",
            "metric": "mean per-scan gauge-line IoU in complete original-point ROI",
            "threshold": 0.5,
            "tie_break": "earlier epoch",
            "early_stopping": "15 consecutive validation non-improvements",
        },
        "threshold_selection": {
            "dataset": "synthetic validation",
            "grid": "0.05 to 0.95 in increments of 0.05",
            "metric": "mean per-scan gauge-line IoU in complete original-point ROI",
            "tie_break": "lowest threshold after numpy argmax",
            "when": "after checkpoint selection",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available_at_protocol_creation": torch.cuda.is_available(),
        },
        "outputs": output_paths(root),
    }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_protocol(output_root: Path) -> dict[str, object]:
    root = assert_isolated_root(output_root)
    path = root / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        Path(__file__): protocol["code"]["entrypoint_sha256"],
        Path(I.__file__): protocol["code"]["invariant_superline_sha256"],
        SPLIT_MANIFEST: protocol["data"]["split_manifest_sha256"],
    }
    for path_to_check, expected in checks.items():
        observed = sha256(path_to_check)
        if observed.lower() != str(expected).lower():
            raise RuntimeError(f"Frozen protocol input changed: {path_to_check}")
    if protocol["seeds"] != [SEED] or tuple(protocol["arms"]) != ARMS:
        raise RuntimeError("Protocol scope changed")
    return protocol


def _pointmlp_paths(root: Path, smoke: bool) -> tuple[Path, Path]:
    if smoke:
        return root / "smoke" / "token_cache", root / "smoke" / "pointmlp"
    return root / "token_cache", root / "pointmlp"


def pointmlp_checkpoint(root: Path, smoke: bool = False) -> Path:
    _cache, output = _pointmlp_paths(root, smoke)
    return output / "pointmlp" / "best_pointmlp_superline_tokens.pt"


def _build_token_cache_item(payload: tuple[dict[str, object], Path, object]) -> dict[str, object]:
    row, cache_root, cfg = payload
    method = dict(I.SUPERLINE3D_STYLE_METHOD)
    method["cache_root"] = Path(cache_root)
    downsampled = I.build_or_load_downsampled(
        str(row["path"]),
        str(row["subset"]),
        method,
        cfg,
    )
    return {
        "subset": str(row["subset"]),
        "sample_id": str(row["sample_id"]),
        "tokens": int(len(downsampled["label"])),
    }


def prebuild_token_cache(
    manifest: pd.DataFrame,
    token_cache: Path,
    cfg: object,
    max_workers: int = 6,
) -> dict[str, object]:
    """Build independent scan caches concurrently before model data loading."""

    rows = manifest.to_dict("records")
    if len({(str(row["subset"]), str(row["sample_id"])) for row in rows}) != len(rows):
        raise RuntimeError("Token-cache work items are not unique")
    started = time.monotonic()
    token_total = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_build_token_cache_item, (row, token_cache, cfg))
            for row in rows
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            token_total += int(result["tokens"])
            if completed % 20 == 0 or completed == len(futures):
                print(f"prebuilt invariant token cache {completed}/{len(futures)}", flush=True)
    return {
        "samples": len(rows),
        "tokens": token_total,
        "workers": int(max_workers),
        "seconds": time.monotonic() - started,
    }


def run_pointmlp_stage(output_root: Path, smoke: bool = False) -> dict[str, object]:
    root = assert_isolated_root(output_root)
    verify_protocol(root)
    token_cache, pointmlp_output = _pointmlp_paths(root, smoke)
    if pointmlp_output.exists():
        raise FileExistsError(pointmlp_output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SBPT-Net training")

    runtime = load_runtime_modules()
    baseline = runtime["baseline"]
    args = SimpleNamespace(
        models=["pointmlp"],
        output_root=pointmlp_output,
        split_manifest=SPLIT_MANIFEST,
        candidate_root=CANDIDATE_ROOT,
        token_cache_root=token_cache,
        device="cuda",
        seed=SEED,
        epochs=1 if smoke else 100,
        patience=1 if smoke else 15,
        batch_size=2 if smoke else 8,
        sample_points=512 if smoke else 4096,
        chunk_points=4096,
        num_workers=0,
        lr=5e-4,
        weight_decay=1e-4,
        smoke=bool(smoke),
        smoke_train=8,
        smoke_val=4,
        smoke_real_per_subset=1,
        overwrite=False,
    )
    pointmlp_output.mkdir(parents=True, exist_ok=False)
    with bind_invariant_tokenization(token_cache):
        baseline.set_seed(SEED)
        cfg = baseline.S.build_compact_config(
            SimpleNamespace(
                candidate_dir=CANDIDATE_ROOT,
                output_root=pointmlp_output,
                seed=SEED,
                device="cuda",
            )
        )
        cfg.device = "cuda"
        cache_prebuild = None
        if not smoke:
            cache_prebuild = prebuild_token_cache(
                pd.read_csv(SPLIT_MANIFEST),
                token_cache,
                cfg,
                max_workers=6,
            )
        train_records, val_records, real_records = baseline.load_records(args, cfg)
        baseline.prepare_projection_support(
            val_records,
            cfg,
            "Prepare invariant synthetic projection support",
        )
        summary = baseline.run_model(
            "pointmlp",
            train_records,
            val_records,
            real_records,
            cfg,
            args,
        )
    checkpoint = pointmlp_checkpoint(root, smoke)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stage = {
        "status": "complete",
        "smoke": bool(smoke),
        "counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(real_records),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "token_cache": str(token_cache),
        "cache_prebuild": cache_prebuild,
        "summary": summary,
        "completed_at": now(),
    }
    atomic_json(pointmlp_output / "stage_audit.json", stage)
    return stage


def _manifest_rows(smoke: bool) -> pd.DataFrame:
    manifest = pd.read_csv(SPLIT_MANIFEST)
    if smoke:
        return pd.concat(
            [
                manifest[manifest["split"] == "train"].head(8),
                manifest[manifest["split"] == "val"].head(4),
                manifest[manifest["split"] == "test"].groupby("subset", sort=False).head(1),
            ],
            ignore_index=True,
        )
    observed = manifest["split"].value_counts().to_dict()
    if observed != {"train": 800, "val": 200, "test": 67}:
        raise RuntimeError(f"Unexpected split counts: {observed}")
    return manifest


def _projection_support(baseline, record: dict[str, object], cfg: object) -> dict[str, object]:
    candidate, memberships, audit = baseline.P.reconstruct_patch_memberships(record, cfg)
    raw = baseline.P.load_raw_metadata(str(record["subset"]), str(record["sample_id"]))
    lengths = np.asarray([len(membership) for membership in memberships], dtype=np.int64)
    if int(lengths.sum()) == 0:
        projection = csr_matrix((len(candidate["xyz"]), len(record["y"])), dtype=np.float32)
    else:
        row_indices = np.concatenate(memberships)
        column_indices = np.repeat(np.arange(len(memberships)), lengths)
        coverage = np.maximum(
            np.bincount(row_indices, minlength=len(candidate["xyz"]))[row_indices],
            1,
        )
        weights = (1.0 / coverage).astype(np.float32)
        projection = csr_matrix(
            (weights, (row_indices, column_indices)),
            shape=(len(candidate["xyz"]), len(record["y"])),
        )
    probe = np.random.default_rng(SEED).uniform(0.0, 1.0, len(record["y"])).astype(np.float32)
    reference, _votes = baseline.P.project_probabilities(
        probe,
        memberships,
        len(candidate["xyz"]),
    )
    np.testing.assert_allclose(projection @ probe, reference, atol=3e-7, rtol=1e-6)
    return {
        "projection": projection,
        "candidate_gt": np.asarray(raw["raw_label"])[candidate["orig_index"]] == 1,
        "raw_positive": int(np.sum(np.asarray(raw["raw_label"]) == 1)),
        "raw_n": int(raw["raw_num_points"]),
        "candidate_orig_index": np.asarray(candidate["orig_index"], dtype=np.int64),
        "projection_audit": audit,
    }


def h90_record_from_downsampled(
    row: dict[str, object],
    downsampled: dict[str, np.ndarray],
    baseline,
    common,
) -> dict[str, object]:
    """Preserve the legacy batch contract while training the center-only arm."""

    return {
        "subset": str(row["subset"]),
        "sample_id": str(row["sample_id"]),
        "split": str(row["split"]),
        "path": str(row["path"]),
        "downsampled": downsampled,
        "x": baseline.token_feature_matrix(downsampled),
        "d": common.descriptor(downsampled),
        "y": np.asarray(downsampled["label"], dtype=np.float32),
        "logit": np.zeros(len(downsampled["label"]), dtype=np.float32),
        "fragment_id": np.asarray(downsampled["fragment_id"], dtype=np.int32),
        "patch": np.asarray(downsampled["patch_points"], dtype=np.float16),
        "patch_mask": np.asarray(downsampled["patch_mask"], dtype=bool),
    }


def build_h90_cache_stage(output_root: Path, smoke: bool = False) -> dict[str, object]:
    root = assert_isolated_root(output_root)
    verify_protocol(root)
    token_cache, _pointmlp_output = _pointmlp_paths(root, smoke)
    checkpoint = pointmlp_checkpoint(root, smoke)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cache_dir = root / ("smoke/h90_cache" if smoke else "h90_cache")
    if cache_dir.exists():
        raise FileExistsError(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=False)

    runtime = load_runtime_modules()
    baseline = runtime["baseline"]
    common = runtime["common"]
    manifest = _manifest_rows(smoke)
    data: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    started = time.monotonic()
    with bind_invariant_tokenization(token_cache):
        cfg = baseline.S.build_compact_config(
            SimpleNamespace(
                candidate_dir=CANDIDATE_ROOT,
                output_root=cache_dir,
                seed=SEED,
                device="cpu",
            )
        )
        for index, row in enumerate(manifest.to_dict("records"), 1):
            cache_path = token_cache / str(row["subset"]) / f"{row['sample_id']}_downsampled.npz"
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            with np.load(cache_path, allow_pickle=False) as archive:
                downsampled = {key: archive[key] for key in archive.files}
            record = h90_record_from_downsampled(row, downsampled, baseline, common)
            if record["split"] != "train":
                record.update(_projection_support(baseline, record, cfg))
            data[str(record["split"])].append(record)
            if index % 50 == 0 or index == len(manifest):
                print(f"prepared H90 cache {index}/{len(manifest)}", flush=True)

    train_descriptors = np.concatenate([record["d"] for record in data["train"]])
    descriptor_mean = train_descriptors.mean(axis=0)
    descriptor_std = np.maximum(train_descriptors.std(axis=0), 0.05)
    for records in data.values():
        for record in records:
            record["d"] = np.clip(
                (record["d"] - descriptor_mean) / descriptor_std,
                -8,
                8,
            ).astype(np.float32)
            record.pop("downsampled", None)

    normalization_path = cache_dir / "normalization.npz"
    np.savez(normalization_path, d_mean=descriptor_mean, d_std=descriptor_std)
    dataset_path = cache_dir / "dataset.pkl"
    temporary = dataset_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(data, handle, protocol=4)
    temporary.replace(dataset_path)
    audit = {
        "status": "complete",
        "smoke": bool(smoke),
        "samples": {split: len(records) for split, records in data.items()},
        "tokens": {
            split: int(sum(len(record["y"]) for record in records))
            for split, records in data.items()
        },
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "normalization": "descriptor statistics fitted on synthetic training tokens only",
        "projection": "exact uniform overlapping patch means verified against registered implementation",
        "pointmlp_checkpoint": str(checkpoint),
        "pointmlp_checkpoint_sha256": sha256(checkpoint),
        "seconds": time.monotonic() - started,
        "completed_at": now(),
    }
    atomic_json(cache_dir / "stage_audit.json", audit)
    return audit


@contextmanager
def bind_pointmlp_checkpoint(checkpoint: Path):
    runtime = load_runtime_modules()
    locked = runtime["locked"]
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    old_path = locked.BASE_CHECKPOINTS[SEED]
    old_hash = locked.EXPECTED_BASE_HASHES[SEED]
    try:
        locked.BASE_CHECKPOINTS[SEED] = checkpoint
        locked.EXPECTED_BASE_HASHES[SEED] = sha256(checkpoint)
        yield locked
    finally:
        locked.BASE_CHECKPOINTS[SEED] = old_path
        locked.EXPECTED_BASE_HASHES[SEED] = old_hash


def _load_h90_data(root: Path, smoke: bool) -> dict[str, list[dict[str, object]]]:
    cache_dir = root / ("smoke/h90_cache" if smoke else "h90_cache")
    dataset = cache_dir / "dataset.pkl"
    audit_path = cache_dir / "stage_audit.json"
    if not dataset.is_file() or not audit_path.is_file():
        raise FileNotFoundError(dataset)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if sha256(dataset).lower() != audit["dataset_sha256"].lower():
        raise RuntimeError("H90 cache hash mismatch")
    with dataset.open("rb") as handle:
        return pickle.load(handle)


def _check_data_counts(data: dict[str, list[dict[str, object]]], smoke: bool) -> None:
    expected = {"train": 8, "val": 4, "test": 3} if smoke else {
        "train": 800,
        "val": 200,
        "test": 67,
    }
    observed = {split: len(records) for split, records in data.items()}
    if observed != expected:
        raise RuntimeError(f"Unexpected H90 data counts: {observed} != {expected}")
    test_counts = {
        subset: sum(record["subset"] == subset for record in data["test"])
        for subset in ("shoe_a", "shoe_b", "shoe_c")
    }
    expected_test = {"shoe_a": 1, "shoe_b": 1, "shoe_c": 1} if smoke else {
        "shoe_a": 26,
        "shoe_b": 24,
        "shoe_c": 17,
    }
    if test_counts != expected_test:
        raise RuntimeError(f"Unexpected real subset counts: {test_counts}")


def _prepare_structural_data(
    root: Path,
    data: dict[str, list[dict[str, object]]],
    checkpoint: Path,
    smoke: bool,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    runtime = load_runtime_modules()
    common = runtime["common"]
    locked = runtime["locked"]
    token_cache, _pointmlp_output = _pointmlp_paths(root, smoke)
    with bind_invariant_tokenization(token_cache):
        extra_stats = common.ensure_extra_features(data)
        locked.set_seed(SEED)
        locked.refresh_frozen_logits(
            data["train"] + data["val"] + data["test"],
            checkpoint,
            SEED,
            torch.device("cuda"),
            "sbpt_net_backbone",
        )
    np.savez(output_dir / "extra_source_normalization.npz", **extra_stats)
    return extra_stats


def run_structural_smoke(
    root: Path,
    data: dict[str, list[dict[str, object]]],
    checkpoint: Path,
) -> dict[str, object]:
    runtime = load_runtime_modules()
    single_runner = runtime["single_runner"]
    locked = runtime["locked"]
    common = runtime["common"]
    A, H, Single = single_runner.A, single_runner.H, single_runner.Single
    smoke_dir = root / "smoke"
    _prepare_structural_data(root, data, checkpoint, True, smoke_dir)
    config = Single.training_config(locked.candidate_config(SEED, checkpoint), ARMS[0])
    locked.set_seed(SEED)
    model = Single.matched_initial_model(config, ARMS[0], SEED, torch.device("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4)
    with single_runner.bind_paired_training_schedule(SEED):
        loss, sampling_hash = A.train_with_sampling_audit(
            model,
            data["train"],
            config,
            optimizer,
        )
    with A.preserve_rng(model):
        val_probabilities = H.infer_head(model, data["val"], config)
        test_probabilities = H.infer_head(model, data["test"], config)
    gradients_finite = all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not np.isfinite(loss) or not gradients_finite:
        raise RuntimeError("Non-finite structural smoke result")
    result = {
        "status": "PASS",
        "not_research_result": True,
        "seed": SEED,
        "arm": ARMS[0],
        "loss": float(loss),
        "gradients_finite": gradients_finite,
        "sampling_sha256": sampling_hash,
        "val4": common.summarize(data["val"], val_probabilities, 0.5),
        "test3": common.summarize(data["test"], test_probabilities, 0.5),
        "completed_at": now(),
    }
    atomic_json(smoke_dir / "structural_smoke.json", result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def run_smoke_stage(output_root: Path) -> dict[str, object]:
    root = assert_isolated_root(output_root)
    verify_protocol(root)
    if (root / "smoke").exists():
        raise FileExistsError(root / "smoke")
    pointmlp = run_pointmlp_stage(root, smoke=True)
    cache = build_h90_cache_stage(root, smoke=True)
    data = _load_h90_data(root, smoke=True)
    _check_data_counts(data, smoke=True)
    checkpoint = pointmlp_checkpoint(root, smoke=True)
    with bind_pointmlp_checkpoint(checkpoint):
        structural = run_structural_smoke(root, data, checkpoint)
    result = {
        "status": "PASS",
        "not_research_result": True,
        "pointmlp": pointmlp,
        "cache": cache,
        "structural": structural,
        "completed_at": now(),
    }
    atomic_json(root / "smoke" / "audit.json", result)
    return result


def train_strongest_source_only(
    data: dict[str, list[dict[str, object]]],
    checkpoint: Path,
    output_dir: Path,
    max_epochs: int = 100,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    runtime = load_runtime_modules()
    single_runner = runtime["single_runner"]
    locked = runtime["locked"]
    common = runtime["common"]
    base_runner = runtime["base_runner"]
    A, H, Single = single_runner.A, single_runner.H, single_runner.Single
    arm = ARMS[0]
    output_dir.mkdir(parents=True, exist_ok=False)
    config = Single.training_config(locked.candidate_config(SEED, checkpoint), arm)
    locked.set_seed(SEED)
    model = Single.matched_initial_model(config, arm, SEED, torch.device("cuda"))
    initial_state = A.state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=5e-5,
    )
    best_score, best_epoch, stale = -1.0, 0, 0
    best_path = output_dir / "best_source_selected.pt"
    history: list[dict[str, object]] = []
    with single_runner.bind_paired_training_schedule(SEED):
        for epoch in range(1, max_epochs + 1):
            started = time.monotonic()
            loss, sampling_hash = A.train_with_sampling_audit(
                model,
                data["train"],
                config,
                optimizer,
            )
            with A.preserve_rng(model):
                validation_probabilities = H.infer_head(model, data["val"], config)
                validation = common.summarize(data["val"], validation_probabilities, 0.5)
            best_score, best_epoch, stale = A.selection_update(
                validation["mean"],
                best_score,
                epoch,
                best_epoch,
                stale,
            )
            if best_epoch == epoch:
                torch.save(
                    {
                        "state_dict": locked.clone_state_dict(model),
                        "config": config,
                        "arm": arm,
                        "seed": SEED,
                        "epoch": epoch,
                        "validation_iou_at_0p5": validation["mean"],
                    },
                    best_path,
                )
            scheduler.step()
            row = {
                "epoch": epoch,
                "loss": float(loss),
                "val_iou_at_0p5": float(validation["mean"]),
                "best_source_epoch": best_epoch,
                "stale": stale,
                "sampling_sha256": sampling_hash,
                "seconds": time.monotonic() - started,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            atomic_json(output_dir / "history.json", history)
            print(
                f"{arm} ep{epoch:03d}: val@.5={validation['mean'] * 100:.3f} "
                f"best_source={best_epoch}",
                flush=True,
            )
            if stale >= 15:
                break

    selected = torch.load(best_path, map_location="cpu")
    model.load_state_dict(selected["state_dict"], strict=True)
    model.to(torch.device("cuda"))
    with A.preserve_rng(model):
        validation_probabilities = H.infer_head(model, data["val"], config)
        reproduced = common.summarize(data["val"], validation_probabilities, 0.5)["mean"]
        if abs(reproduced - selected["validation_iou_at_0p5"]) > 1e-9:
            raise RuntimeError("Selected structural checkpoint did not reproduce")
        threshold, grid = common.choose_threshold(data["val"], validation_probabilities)
        test_probabilities = H.infer_head(model, data["test"], config)
    result = {
        "arm": arm,
        "seed": SEED,
        "selected_epoch": int(selected["epoch"]),
        "selected_threshold": float(threshold),
        "source_iou_at0p5": float(reproduced),
        "val": common.summarize(data["val"], validation_probabilities, threshold),
        "test": common.summarize(data["test"], test_probabilities, threshold),
        "threshold_grid": grid,
        "epochs_executed": len(history),
        "trainable_head_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_state,
        "checkpoint": str(best_path),
        "checkpoint_sha256": sha256(best_path),
        "real_evaluation_policy": "once after source checkpoint and threshold selection",
    }
    base_runner.save_probabilities(
        output_dir / "selected_probabilities.npz",
        validation_probabilities,
        test_probabilities,
    )
    atomic_json(output_dir / "result.json", result)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return result, history


def run_strongest_stage(output_root: Path) -> dict[str, object]:
    root = assert_isolated_root(output_root)
    verify_protocol(root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SBPT-Net training")
    output_dir = root / "strongest"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    data = _load_h90_data(root, smoke=False)
    _check_data_counts(data, smoke=False)
    checkpoint = pointmlp_checkpoint(root, smoke=False)
    runtime = load_runtime_modules()
    base_runner = runtime["base_runner"]
    with bind_pointmlp_checkpoint(checkpoint):
        _prepare_structural_data(root, data, checkpoint, False, output_dir)
        initial = base_runner.initial_classifier_result(data, output_dir)
        result, history = train_strongest_source_only(
            data,
            checkpoint,
            output_dir / ARMS[0],
            max_epochs=100,
        )
    summary = {
        "status": "complete",
        "seed": SEED,
        "arm": ARMS[0],
        "initial_classifier_only": initial,
        "strongest": result,
        "epochs_executed": len(history),
        "completed_at": now(),
    }
    atomic_json(output_dir / "final_results.json", summary)
    return summary


def equal_subset_metrics(result: dict[str, object]) -> dict[str, float]:
    rows = result["test"]["per_case"]
    subsets = ("shoe_a", "shoe_b", "shoe_c")
    metrics: dict[str, float] = {}
    for metric in ("iou", "precision", "recall", "f1"):
        metrics[metric] = float(
            np.mean(
                [
                    np.mean([row[metric] for row in rows if row["subset"] == subset])
                    for subset in subsets
                ]
            )
        )
    metrics.update({subset: float(result["test"]["subsets"][subset]) for subset in subsets})
    metrics["validation_iou"] = float(result["val"]["mean"])
    metrics["selected_epoch"] = int(result["selected_epoch"])
    metrics["selected_threshold"] = float(result["selected_threshold"])
    return metrics


def summarize_stage(output_root: Path) -> dict[str, object]:
    root = assert_isolated_root(output_root)
    verify_protocol(root)
    new_path = root / "strongest" / ARMS[0] / "result.json"
    if not new_path.is_file():
        raise FileNotFoundError(new_path)
    candidate = json.loads(new_path.read_text(encoding="utf-8"))
    summary = {
        "status": "complete",
        "seed": SEED,
        "formula": "orthogonal tangent residual plus undirected kNN union",
        "metrics": equal_subset_metrics(candidate),
        "result": str(new_path),
        "result_sha256": sha256(new_path),
        "completed_at": now(),
    }
    atomic_json(root / "summary.json", summary)
    return summary


def main() -> int:
    global SEED, EXPERIMENT_ROOT, CANDIDATE_ROOT, OFFLINE_ROOT, SPLIT_MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare-protocol", "smoke", "pointmlp", "h90-cache", "strongest", "summarize"),
    )
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 43, 44))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    SEED = args.seed
    CANDIDATE_ROOT = args.candidate_root.resolve()
    OFFLINE_ROOT = args.offline_root.resolve()
    EXPERIMENT_ROOT = (args.output_root or ROOT / "outputs" / "reproduction" / f"seed{SEED}").resolve()
    SPLIT_MANIFEST = EXPERIMENT_ROOT / "split_manifest.csv"
    root = assert_isolated_root(EXPERIMENT_ROOT)
    if args.stage == "prepare-protocol":
        root.mkdir(parents=True, exist_ok=False)
        baseline = load_runtime_modules()["baseline"]
        cfg = baseline.S.build_compact_config(
            SimpleNamespace(candidate_dir=CANDIDATE_ROOT, output_root=root, seed=SEED, device="cpu")
        )
        rows, _info = T.choose_protocol(cfg)
        counts = pd.DataFrame(rows)["split"].value_counts().to_dict()
        if counts != {"train": 800, "val": 200, "test": 67}:
            raise RuntimeError(f"Unexpected split counts: {counts}")
        pd.DataFrame(rows).to_csv(SPLIT_MANIFEST, index=False)
        result = build_protocol(root)
        atomic_json(root / "protocol.json", result)
        atomic_json(root / "status.json", {"status": "prepared", "at": now()})
    elif args.stage == "smoke":
        result = run_smoke_stage(root)
    elif args.stage == "pointmlp":
        result = run_pointmlp_stage(root, smoke=False)
    elif args.stage == "h90-cache":
        result = build_h90_cache_stage(root, smoke=False)
    elif args.stage == "strongest":
        result = run_strongest_stage(root)
    elif args.stage == "summarize":
        result = summarize_stage(root)
        atomic_json(root / "status.json", {"status": "complete", "at": now()})
    else:
        raise AssertionError(args.stage)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
