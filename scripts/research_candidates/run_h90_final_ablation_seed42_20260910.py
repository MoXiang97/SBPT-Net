"""Physical Seed-42 component ablation of the selected final SBPT candidate."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil
import torch

import h90_final_model_20260910 as F


Legacy, A, T, H, C = F.Legacy, F.Legacy.A, F.T, F.Legacy.H, F.Legacy.A.C
ROOT = A.ROOT
SOURCE = ROOT / "experiments/H90_Slim_NoPatch_Selection_20260910"
OUT_DEFAULT = ROOT / "experiments/H90_Final_Model_Ablation_Seed42_20260910"
ARMS = F.TRAINABLE_ARMS
SEED = 42


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def local_code():
    files = {
        Path(__file__).resolve(),
        Path(F.__file__).resolve(),
        Path(Legacy.__file__).resolve(),
        Path(A.__file__).resolve(),
        Path(T.__file__).resolve(),
        Path(H.__file__).resolve(),
        Path(C.__file__).resolve(),
        Path(__file__).with_name("test_h90_final_model_20260910.py").resolve(),
    }
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and ROOT in path.parents:
                files.add(path)
    return {str(path.relative_to(ROOT)): T.sha256(path) for path in sorted(files)}


def prepare(out: Path, experiment_id: str) -> None:
    if out.exists():
        raise FileExistsError(out)
    source_files = {
        "protocol": SOURCE / "protocol.json",
        "results": SOURCE / "final_results.json",
        "selected_seed42": SOURCE / "exact_converted_no_patch.pt",
        "conversion_audit": SOURCE / "exact_conversion.json",
    }
    for path in source_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    base_checkpoint = T.verify_base_checkpoint(SEED)
    registry = json.loads((ROOT / "docs/current_experiment_registry.json").read_text(encoding="utf-8"))
    split = ROOT / registry["shared_inputs"]["sbpt_split_manifest"]
    code = local_code()
    protocol = {
        "experiment_id": experiment_id,
        "category": "component_ablation",
        "purpose": "Physical Seed-42 component ablation of the user-selected final structural model; defer additional seeds until useful rows are identified.",
        "base_experiment": str(SOURCE.relative_to(ROOT)),
        "code": {"entrypoint": str(Path(__file__).relative_to(ROOT)), "sha256": code},
        "predecessor": {key: {"path": str(path.relative_to(ROOT)), "sha256": T.sha256(path)} for key, path in source_files.items()},
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
        "rows": ["initial_classifier_only", *ARMS],
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
            "frozen_backbone": "Seed-42 source-trained PointMLP with the superline-token representative input",
            "loss": "fused BCE + 0.5 Dice + 0.0001 deviation + 0.1 structural auxiliary loss + 2.0 clean/augmented consistency",
            "architecture": "clean 12D structural input; 6 line descriptors + 6 normalized token attributes; same-superline attention; fixed 1:1 initial/structural logit fusion",
            "pairing": "same frozen backbone, compatible initialization, input preparation, sampling, augmentation draws, optimizer, schedule and evaluation",
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
            "control": "clean full final structural head",
            "unchanged": "frozen PointMLP, data and splits, source selection, threshold grid, fusion weight, sampling and training budget",
            "differences": "one physical feature group, the same-superline context module, or the consistency loss is removed in each ablation row",
            "limits": "single-seed component attribution; additional seeds intentionally deferred",
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
        "queue": {"policy": "wait for all other Python processes; never interrupt existing training", "timeout_hours": 24},
        "outputs": {"directory": str(out), "expected_results": "six rows, five training histories, source-selected checkpoints and thresholds, every-epoch real diagnostics, clean-conversion and pairing audits"},
        "created_at": now(),
    }
    out.mkdir(parents=True, exist_ok=False)
    for relative in code:
        destination = out / "frozen_code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    T.write_json(out / "protocol.json", protocol)
    T.write_json(out / "status.json", {"status": "prepared", "at": now(), "training_started": False})
    print(json.dumps({"prepared": str(out), "code_files": len(code)}, indent=2), flush=True)


def verify(protocol) -> None:
    A.verify_hashes(protocol["code"]["sha256"])
    A.verify_hashes({item["path"]: item["sha256"] for item in protocol["predecessor"].values()})
    A.verify_hashes({
        protocol["data"]["cache"]: protocol["data"]["cache_sha256"],
        protocol["data"]["split_manifest"]: protocol["data"]["split_sha256"],
        protocol["data"]["base_checkpoint"]: protocol["data"]["base_checkpoint_sha256"],
    })


def save_probabilities(path: Path, validation, test) -> None:
    np.savez_compressed(path, **{
        f"{split}_{index:04d}": np.asarray(probability, dtype=np.float32)
        for split, probabilities in (("val", validation), ("test", test))
        for index, probability in enumerate(probabilities)
    })


def initial_classifier_result(data, out: Path):
    probabilities = {
        split: [torch.sigmoid(torch.from_numpy(record["logit"])).numpy() for record in data[split]]
        for split in ("val", "test")
    }
    threshold, grid = C.choose_threshold(data["val"], probabilities["val"])
    result = {
        "arm": "initial_classifier_only",
        "seed": SEED,
        "selected_epoch": None,
        "selected_threshold": threshold,
        "source_iou_at0p5": C.summarize(data["val"], probabilities["val"], 0.5)["mean"],
        "val": C.summarize(data["val"], probabilities["val"], threshold),
        "test": C.summarize(data["test"], probabilities["test"], threshold),
        "threshold_grid": grid,
        "trainable_head_parameters": 0,
        "checkpoint": str(T.BASE_CHECKPOINTS[SEED]),
    }
    T.write_json(out / "initial_classifier_only.json", result)
    save_probabilities(out / "initial_classifier_only_probabilities.npz", probabilities["val"], probabilities["test"])
    return result


def exact_clean_conversion(data, out: Path, device: torch.device):
    source = SOURCE / "exact_converted_no_patch.pt"
    payload = torch.load(source, map_location="cpu")
    legacy = Legacy.SlimNoPatchHead(payload["config"]).to(device)
    legacy.load_state_dict(payload["state_dict"], strict=True)
    clean = F.FinalStructuralHead(payload["config"], "full").to(device)
    F.copy_from_legacy_state(clean, payload["state_dict"])
    legacy.eval()
    clean.eval()
    comparisons = {}
    clean_probabilities = {}
    with A.preserve_rng(clean):
        for split in ("val", "test"):
            legacy_probs = H.infer_head(legacy, data[split], payload["config"])
            clean_probs = H.infer_head(clean, data[split], payload["config"])
            maximum = max(float(np.max(np.abs(old - new))) for old, new in zip(legacy_probs, clean_probs))
            mask_equal = all(np.array_equal(old >= 0.4, new >= 0.4) for old, new in zip(legacy_probs, clean_probs))
            if maximum > 2e-6 or not mask_equal:
                raise RuntimeError(f"Clean conversion failed for {split}")
            comparisons[split] = {"max_probability_absolute_error": maximum, "prediction_mask_at0p4_exact": mask_equal}
            clean_probabilities[split] = clean_probs
    checkpoint = out / "clean_selected_seed42.pt"
    torch.save({
        "state_dict": T.clone_state_dict(clean),
        "config": payload["config"],
        "arm": "full",
        "seed": SEED,
        "epoch": payload["epoch"],
        "threshold": 0.4,
        "source": str(source),
    }, checkpoint)
    audit = {
        "status": "PASS",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": T.sha256(source),
        "clean_checkpoint": str(checkpoint),
        "clean_checkpoint_sha256": T.sha256(checkpoint),
        "legacy_parameters": sum(parameter.numel() for parameter in legacy.parameters()),
        "clean_parameters": sum(parameter.numel() for parameter in clean.parameters()),
        "comparisons": comparisons,
    }
    T.write_json(out / "clean_conversion_audit.json", audit)
    save_probabilities(out / "clean_selected_seed42_probabilities.npz", clean_probabilities["val"], clean_probabilities["test"])
    del legacy, clean
    torch.cuda.empty_cache()
    return audit


def gpu_smoke(data, out: Path, device: torch.device):
    rows = []
    sample_hashes = set()
    for arm in ARMS:
        config = F.training_config(T.candidate_config(SEED, T.BASE_CHECKPOINTS[SEED]), arm)
        model = F.matched_initial_model(config, arm, SEED, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0002)
        with A.preserve_rng(model):
            loss, sample_hash = A.train_with_sampling_audit(model, data["train"][:8], config, optimizer)
            validation = H.infer_head(model, data["val"][:4], config)
            real = [next(record for record in data["test"] if record["subset"] == subset) for subset in ("shoe_a", "shoe_b", "shoe_c")]
            test = H.infer_head(model, real, config)
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite smoke loss for {arm}")
        if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
            raise RuntimeError(f"Non-finite smoke gradient for {arm}")
        sample_hashes.add(sample_hash)
        rows.append({
            "arm": arm,
            "loss": loss,
            "sampling_sha256": sample_hash,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "val4": C.summarize(data["val"][:4], validation, 0.5),
            "test3": C.summarize(real, test, 0.5),
        })
        del model, optimizer
        torch.cuda.empty_cache()
    if len(sample_hashes) != 1:
        raise RuntimeError("Smoke sampling differs across arms")
    result = {"status": "PASS", "not_research_result": True, "rows": rows}
    T.write_json(out / "gpu_smoke.json", result)
    return result


def train_arm(data, arm: str, out: Path, device: torch.device, max_epochs: int):
    out.mkdir(parents=True, exist_ok=False)
    config = F.training_config(T.candidate_config(SEED, T.BASE_CHECKPOINTS[SEED]), arm)
    model = F.matched_initial_model(config, arm, SEED, device)
    initial_state = A.state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0.00005)
    best, best_epoch, stale = -1.0, 0, 0
    history = []
    best_path = out / "best_source_selected.pt"
    for epoch in range(1, max_epochs + 1):
        started = time.monotonic()
        loss, sampling = A.train_with_sampling_audit(model, data["train"], config, optimizer)
        with A.preserve_rng(model):
            validation_probs = H.infer_head(model, data["val"], config)
            validation = C.summarize(data["val"], validation_probs, 0.5)
            best, best_epoch, stale = A.selection_update(validation["mean"], best, epoch, best_epoch, stale)
            payload = {
                "state_dict": T.clone_state_dict(model),
                "config": config,
                "arm": arm,
                "seed": SEED,
                "epoch": epoch,
                "validation_iou_at_0p5": validation["mean"],
            }
            torch.save(payload, out / f"epoch_{epoch:03d}.pt")
            if best_epoch == epoch:
                torch.save(payload, best_path)
            test_probs = H.infer_head(model, data["test"], config)
            test = C.summarize(data["test"], test_probs, 0.5)
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": loss,
            "val_iou_at_0p5": validation["mean"],
            "test_iou_at_0p5": test["mean"],
            "test_subsets_at_0p5": test["subsets"],
            "best_source_epoch": best_epoch,
            "stale": stale,
            "sampling_sha256": sampling,
            "seconds": time.monotonic() - started,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        T.write_json(out / "history.json", history)
        T.write_json(out / f"epoch_{epoch:03d}_metrics.json", {"val": validation, "test": test, "threshold": 0.5, "role": "development curve"})
        print(f"{arm} ep{epoch:03d}: val@.5={validation['mean']*100:.3f} test@.5={test['mean']*100:.3f} A/B/C={test['subsets']} best_source={best_epoch}", flush=True)
        if stale >= 15:
            break

    selected = torch.load(best_path, map_location="cpu")
    model.load_state_dict(selected["state_dict"], strict=True)
    model.to(device)
    with A.preserve_rng(model):
        validation_probs = H.infer_head(model, data["val"], config)
        fixed = C.summarize(data["val"], validation_probs, 0.5)["mean"]
        if abs(fixed - selected["validation_iou_at_0p5"]) > 1e-9:
            raise RuntimeError(f"Selected checkpoint reproduction failed for {arm}")
        threshold, grid = C.choose_threshold(data["val"], validation_probs)
        test_probs = H.infer_head(model, data["test"], config)
    result = {
        "arm": arm,
        "seed": SEED,
        "selected_epoch": selected["epoch"],
        "selected_threshold": threshold,
        "source_iou_at0p5": fixed,
        "val": C.summarize(data["val"], validation_probs, threshold),
        "test": C.summarize(data["test"], test_probs, threshold),
        "threshold_grid": grid,
        "epochs_executed": len(history),
        "trainable_head_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_state,
        "checkpoint": str(best_path),
        "checkpoint_sha256": T.sha256(best_path),
    }
    save_probabilities(out / "selected_probabilities.npz", validation_probs, test_probs)
    T.write_json(out / "result.json", result)
    print(f"FINAL {arm}: source_ep={result['selected_epoch']} t={threshold:.2f} TEST={result['test']['mean']*100:.4f} {result['test']['subsets']}", flush=True)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return result, history


def execute(out: Path, wait: bool) -> None:
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    verify(protocol)
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
        "pid": os.getpid(),
        "protocol_sha256": T.sha256(out / "protocol.json"),
        "command": sys.argv,
        "gpu": torch.cuda.get_device_name(0),
    }
    T.write_json(out / "runtime_manifest.json", runtime)
    T.write_json(out / "status.json", {"status": "loading_data", "at": now(), "training_started": False})
    data = C.load_data(False)
    A.check_dataset(data, protocol["data"]["counts"])
    counts = {subset: sum(record["subset"] == subset for record in data["test"]) for subset in ("shoe_a", "shoe_b", "shoe_c")}
    if counts != protocol["data"]["real_subsets"]:
        raise RuntimeError("Real subset counts changed")
    stats = C.ensure_extra_features(data)
    np.savez(out / "extra_source_normalization.npz", **stats)
    T.write_json(out / "scan_index.json", {
        split: [{"subset": record["subset"], "sample_id": record["sample_id"]} for record in records]
        for split, records in data.items()
    })
    T.refresh_frozen_logits(data["train"] + data["val"] + data["test"], T.BASE_CHECKPOINTS[SEED], SEED, device, "final_ablation_shared_backbone")
    base = initial_classifier_result(data, out)
    conversion = exact_clean_conversion(data, out, device)
    smoke = gpu_smoke(data, out, device)
    results = []
    sampling_by_epoch = {}
    initial_states = {}
    for arm in ARMS:
        T.write_json(out / "status.json", {"status": "training", "arm": arm, "at": now(), "training_started": True})
        result, history = train_arm(data, arm, out / arm, device, protocol["training"]["max_epochs"])
        results.append(result)
        initial_states[arm] = result["initial_state_sha256"]
        for row in history:
            epoch = str(row["epoch"])
            if epoch in sampling_by_epoch and sampling_by_epoch[epoch] != row["sampling_sha256"]:
                raise RuntimeError(f"Paired sampling diverged at epoch {epoch}")
            sampling_by_epoch[epoch] = row["sampling_sha256"]
        T.write_json(out / "results_so_far.json", {"initial_classifier_only": base, "arms": results})
    summary = {
        "status": "complete",
        "seed": SEED,
        "initial_classifier_only": base,
        "arms": results,
        "clean_conversion": conversion,
        "gpu_smoke": smoke["status"],
    }
    T.write_json(out / "final_results.json", summary)
    T.write_json(out / "pairing_audit.json", {
        "status": "PASS",
        "initial_state_sha256": initial_states,
        "epoch_sampling_sha256": sampling_by_epoch,
        "note": "Different physical arms have different state shapes; CPU tests verify compatible weight-column identity.",
    })
    verify(protocol)
    runtime.update(status="complete", ended_at=now())
    T.write_json(out / "runtime_manifest.json", runtime)
    T.write_json(out / "status.json", {"status": "complete", "at": now(), "completed_rows": 6})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--wait-for-compute", action="store_true")
    parser.add_argument("--experiment-id", default="20260910_h90_final_model_ablation_seed42")
    args = parser.parse_args()
    out = args.output_root.resolve()
    if args.prepare:
        prepare(out, args.experiment_id)
    else:
        try:
            execute(out, args.wait_for_compute)
        except Exception as error:
            T.write_json(out / "status.json", {"status": "failed", "error": repr(error), "at": now()})
            raise


if __name__ == "__main__":
    main()
