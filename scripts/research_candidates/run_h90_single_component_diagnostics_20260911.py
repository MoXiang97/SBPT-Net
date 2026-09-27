"""Run three Seed-42 single-component diagnostics after active compute is free."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import psutil
import torch

import h90_single_component_diagnostics_20260911 as Single
import run_h90_final_ablation_seed42_20260910 as Base


A, T, H, C = Base.A, Base.T, Base.H, Base.C
ROOT = Base.ROOT
SEED = 42
ARMS = Single.ARMS
DEFAULT_ARMS = ("superline_context_only",)
PARENT = ROOT / "experiments/H90_Final_Model_Ablation_Seed42_20260910"
OUT_DEFAULT = ROOT / "experiments/H90_Single_Component_Diagnostics_Seed42_20260911"


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_arms(arms):
    selected = tuple(arms)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select one or more distinct singleton arms")
    unknown = set(selected).difference(ARMS)
    if unknown:
        raise ValueError(f"Unknown singleton arms: {sorted(unknown)}")
    return selected


def experiment_definition(arms):
    meanings = {
        "line_descriptors_only": ["line_descriptors"],
        "token_attributes_only": ["token_attributes"],
        "superline_context_only": ["initial_and_neighbor_logits"],
        "superline_context_only_no_consistency": ["initial_and_neighbor_logits"],
        "token_encoding_only_no_consistency": [
            "line_descriptors",
            "token_attributes",
        ],
    }
    selected = normalize_arms(arms)
    return {"arms": list(selected), **{arm: meanings[arm] for arm in selected}}


@contextmanager
def paired_sampling_rng(seed):
    """Keep record/token selection independent of arm-specific augmentation draws."""
    schedule_rng = np.random.RandomState(int(seed))
    original_permutation = np.random.permutation
    original_choice = np.random.choice
    np.random.permutation = schedule_rng.permutation
    np.random.choice = schedule_rng.choice
    try:
        yield
    finally:
        np.random.permutation = original_permutation
        np.random.choice = original_choice


@contextmanager
def bind_paired_training_schedule(seed, repeat_first=False):
    """Give every arm the same per-epoch record and token sampling schedule."""
    original = A.train_with_sampling_audit
    call_index = 0

    def paired(*args, **kwargs):
        nonlocal call_index
        call_index += 1
        schedule_index = 1 if repeat_first else call_index
        with paired_sampling_rng(int(seed) * 100000 + schedule_index):
            return original(*args, **kwargs)

    A.train_with_sampling_audit = paired
    try:
        yield
    finally:
        A.train_with_sampling_audit = original


@contextmanager
def bind_single_component_arms(arms=ARMS):
    """Reuse the audited loop while restoring its module globals afterward."""
    old_module, old_arms = Base.F, Base.ARMS
    Base.F, Base.ARMS = Single, normalize_arms(arms)
    try:
        yield
    finally:
        Base.F, Base.ARMS = old_module, old_arms


def local_code():
    with bind_single_component_arms():
        files = Base.local_code()
    for path in (
        Path(__file__).resolve(),
        Path(Single.__file__).resolve(),
        Path(__file__).with_name("test_h90_single_component_diagnostics_20260911.py").resolve(),
    ):
        files[str(path.relative_to(ROOT))] = T.sha256(path)
    return dict(sorted(files.items()))


def prepare(out: Path, experiment_id: str, arms=DEFAULT_ARMS) -> None:
    if out.exists():
        raise FileExistsError(out)
    arms = normalize_arms(arms)
    without_consistency = all(arm.endswith("_no_consistency") for arm in arms)
    predecessor = {
        "protocol": PARENT / "protocol.json",
        "results": PARENT / "final_results.json",
        "full_checkpoint": PARENT / "clean_selected_seed42.pt",
        "pairing_audit": PARENT / "pairing_audit.json",
    }
    for path in predecessor.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    base_checkpoint = T.verify_base_checkpoint(SEED)
    registry = json.loads((ROOT / "docs/current_experiment_registry.json").read_text(encoding="utf-8"))
    split = ROOT / registry["shared_inputs"]["sbpt_split_manifest"]
    code = local_code()
    protocol = {
        "experiment_id": experiment_id,
        "category": (
            "final_candidate_component_ablation"
            if without_consistency
            else "single_component_ablation"
        ),
        "purpose": (
            "Complete the two missing Seed-42 component rows for the final model after removal of the consistency loss."
            if without_consistency
            else "Identify whether line descriptors, normalized token attributes, or same-superline logit context can independently explain the large gain over the frozen initial classifier."
        ),
        "base_experiment": str(PARENT.relative_to(ROOT)),
        "definition": experiment_definition(arms),
        "code": {"entrypoint": str(Path(__file__).relative_to(ROOT)), "sha256": code},
        "predecessor": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": T.sha256(path)}
            for key, path in predecessor.items()
        },
        "data": {
            "cache": str(A.FULL_CACHE.relative_to(ROOT)),
            "cache_sha256": A.EXPECTED_FULL_CACHE,
            "split_manifest": str(split.relative_to(ROOT)),
            "split_sha256": T.sha256(split),
            "counts": {"train": 800, "val": 200, "test": 67},
            "real_subsets": {"shoe_a": 26, "shoe_b": 24, "shoe_c": 17},
            "base_checkpoint": str(base_checkpoint.relative_to(ROOT)),
            "base_checkpoint_sha256": T.EXPECTED_BASE_HASHES[SEED],
            "normalization": "six token attributes fitted using all 800 synthetic training scans only",
        },
        "seeds": [SEED],
        "arms": list(arms),
        "training": {
            "max_epochs": 100,
            "patience": 15,
            "optimizer": "AdamW",
            "lr": 0.0005,
            "weight_decay": 0.0002,
            "scheduler": "CosineAnnealingLR eta_min=0.00005 T_max=100",
            "batch_clouds": 4,
            "tokens_per_cloud": 256,
            "gradient_clip_norm": 5.0,
            "frozen_backbone": "Seed-42 source-trained PointMLP with superline-token representative input",
            "loss": (
                "fused BCE + 0.5 Dice + 0.0001 deviation + 0.1 structural auxiliary loss; consistency weight fixed to zero"
                if without_consistency
                else "fused BCE + 0.5 Dice + 0.0001 deviation + 0.1 structural auxiliary loss + 2.0 clean/augmented consistency"
            ),
            "fusion": "fixed 1:1 initial/diagnostic logit fusion",
            "pairing": "same frozen backbone, source split, training budget, optimizer, schedule, source selection, threshold grid, and per-epoch sampled token identities",
        },
        "checkpoint_selection": {
            "dataset": "synthetic validation",
            "space": "complete original-point ROI via uniform-overlap projection",
            "aggregation": "mean per-scan gauge-line IoU",
            "metric": "IoU",
            "threshold": 0.5,
            "tie_break": "earlier epoch",
            "early_stopping": "15 consecutive source-validation non-improvements",
        },
        "threshold_selection": {
            "dataset": "synthetic validation",
            "grid": [float(value) for value in C.THRESHOLDS],
            "space": "complete original-point ROI",
            "metric": "mean per-scan gauge-line IoU",
            "tie_break": "lowest threshold",
            "when": "after checkpoint selection",
        },
        "fairness": {
            "control": "frozen initial classifier result from the same Seed-42 PointMLP checkpoint",
            "unchanged": "data, split, frozen backbone, fusion weight, optimizer, training budget, source selection, threshold calibration, and evaluation",
            "differences": "each arm physically exposes one signal path; context-only uses center and same-superline neighbor logits, while token-encoding-only uses the 12 retained descriptors and attributes without neighbor aggregation; consistency weight is zero for the final-table arms",
            "limits": "single-seed diagnostic; context-only has a separately seeded one-dimensional input embedding because no compatible full-model feature column exists",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "psutil": psutil.__version__,
            "cuda": torch.version.cuda,
            "device": "NVIDIA GeForce RTX 4090 runtime verified before execution",
        },
        "queue": {
            "policy": "run immediately after confirming that no other workspace Python experiment is active",
            "timeout_hours": 24,
        },
        "outputs": {
            "directory": str(out),
            "expected_results": "initial classifier plus source-selected requested singleton rows, histories, checkpoints, probabilities, smoke and pairing audits",
        },
        "created_at": now(),
    }
    out.mkdir(parents=True, exist_ok=False)
    for relative in code:
        destination = out / "frozen_code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    T.write_json(out / "protocol.json", protocol)
    T.write_json(out / "status.json", {"status": "prepared", "at": now(), "training_started": False})
    print(json.dumps({"prepared": str(out), "arms": list(arms), "code_files": len(code)}, indent=2), flush=True)


def execute(out: Path, wait: bool) -> None:
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    arms = normalize_arms(protocol["arms"])
    Base.verify(protocol)
    if (out / "runtime_manifest.json").exists():
        raise FileExistsError("Run already started; never overwrite or silently resume")
    if wait:
        A.wait_for_slot(out, protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    runtime = {
        "status": "running",
        "started_at": now(),
        "pid": __import__("os").getpid(),
        "protocol_sha256": T.sha256(out / "protocol.json"),
        "command": sys.argv,
        "gpu": torch.cuda.get_device_name(0),
    }
    T.write_json(out / "runtime_manifest.json", runtime)
    T.write_json(out / "status.json", {"status": "loading_data", "at": now(), "training_started": False})
    data = C.load_data(False)
    A.check_dataset(data, protocol["data"]["counts"])
    counts = {
        subset: sum(record["subset"] == subset for record in data["test"])
        for subset in ("shoe_a", "shoe_b", "shoe_c")
    }
    if counts != protocol["data"]["real_subsets"]:
        raise RuntimeError("Real subset counts changed")
    stats = C.ensure_extra_features(data)
    np.savez(out / "extra_source_normalization.npz", **stats)
    T.write_json(out / "scan_index.json", {
        split_name: [
            {"subset": record["subset"], "sample_id": record["sample_id"]}
            for record in records
        ]
        for split_name, records in data.items()
    })
    T.refresh_frozen_logits(
        data["train"] + data["val"] + data["test"],
        T.BASE_CHECKPOINTS[SEED], SEED, device, "single_component_shared_backbone",
    )
    baseline = Base.initial_classifier_result(data, out)
    with bind_single_component_arms(arms):
        with bind_paired_training_schedule(SEED, repeat_first=True):
            smoke = Base.gpu_smoke(data, out, device)
        results, sampling_by_epoch, initial_states = [], {}, {}
        for arm in arms:
            T.write_json(out / "status.json", {
                "status": "training", "arm": arm, "at": now(), "training_started": True
            })
            with bind_paired_training_schedule(SEED):
                result, history = Base.train_arm(
                    data, arm, out / arm, device, protocol["training"]["max_epochs"]
                )
            results.append(result)
            initial_states[arm] = result["initial_state_sha256"]
            for row in history:
                epoch = str(row["epoch"])
                if epoch in sampling_by_epoch and sampling_by_epoch[epoch] != row["sampling_sha256"]:
                    raise RuntimeError(f"Paired sampling diverged at epoch {epoch}")
                sampling_by_epoch[epoch] = row["sampling_sha256"]
            T.write_json(out / "results_so_far.json", {
                "initial_classifier_only": baseline, "arms": results
            })
    summary = {
        "status": "complete",
        "seed": SEED,
        "initial_classifier_only": baseline,
        "arms": results,
        "gpu_smoke": smoke["status"],
    }
    T.write_json(out / "final_results.json", summary)
    T.write_json(out / "pairing_audit.json", {
        "status": "PASS",
        "initial_state_sha256": initial_states,
        "epoch_sampling_sha256": sampling_by_epoch,
        "note": "Physical singleton architectures have different state shapes; tests verify signal isolation and compatible center-feature initialization.",
    })
    Base.verify(protocol)
    runtime.update(status="complete", ended_at=now())
    T.write_json(out / "runtime_manifest.json", runtime)
    T.write_json(out / "status.json", {
        "status": "complete", "at": now(), "completed_rows": len(results) + 1
    })
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--wait-for-compute", action="store_true")
    parser.add_argument(
        "--experiment-id", default="20260911_h90_single_component_diagnostics_seed42"
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(DEFAULT_ARMS))
    args = parser.parse_args()
    out = args.output_root.resolve()
    if args.prepare:
        prepare(out, args.experiment_id, args.arms)
    else:
        try:
            execute(out, args.wait_for_compute)
        except Exception as error:
            T.write_json(out / "status.json", {
                "status": "failed", "error": repr(error), "at": now()
            })
            raise


if __name__ == "__main__":
    main()
