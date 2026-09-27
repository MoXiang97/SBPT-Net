"""Structural input and loss ablation utilities."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

import run_pointmlp_superline_h90_locked_20260908 as T

C, H, ROOT = T.C, T.H, T.ROOT
ARMS = ("full", "no_descriptor", "descriptor_without_order", "no_extra",
        "no_patch", "no_context_information", "no_consistency")
DEFAULT_OUT = ROOT / "experiments/H90_Component_Ablation_Seed42_20260910"
BASE_ROOT = ROOT / "experiments/PointMLP_Superline_H90_Locked_20260908"
FULL_CACHE = T.DIAGNOSTIC_SOURCE / "cache/dataset.pkl"
EXPECTED_FULL_CACHE = "2658922ef79e35734f782a706ffe559e568e489d3ef93a0a234344a647db9534"
HISTORICAL_LOCK = BASE_ROOT / "source_selection_lock_20260909.json"


def historical_inputs():
    # Read frozen expected digests, not newly measured values: avoid manual
    # transcription of 64-character hashes and never silently refresh a lock.
    locked = json.loads(HISTORICAL_LOCK.read_text(encoding="utf-8"))["file_sha256"]
    files = {Path(m.__file__).resolve() for m in (T, C, H, C.R)}
    files.update({BASE_ROOT / "seed42/best_source_selected.pt", BASE_ROOT / "seed42/source_selection.json",
                  BASE_ROOT / "source_only_cache_audit.json"})
    return {str(p.relative_to(ROOT)): locked[str(p.relative_to(ROOT))] for p in files}


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class AblationHead(H.ResidualHead):
    """Matched-size information ablation; no parameter savings are claimed."""
    def __init__(self, config, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(config)
        self.arm = arm
        if arm == "no_patch":
            self.patch_encoder.register_forward_hook(lambda module, args, output: output * 0.0)
        if arm == "no_context_information":
            # Retain the fusion MLP and its central features; remove only the
            # aggregated neighbor message. Same RNG/parameter shapes as control.
            self.context.register_forward_pre_hook(
                lambda module, args: (torch.cat((args[0][:, :96], args[0][:, 96:] * 0.0), -1),))

    def forward(self, batch):
        b = dict(batch)
        if self.arm in ("no_descriptor", "descriptor_without_order"):
            ids = self.dids if self.arm == "no_descriptor" else [12]
            b["d"], b["nd"] = b["d"].clone(), b["nd"].clone()
            b["d"][:, ids] = 0
            b["nd"][:, :, ids] = 0
        if self.arm == "no_extra":
            b["extra"], b["nextra"] = b["extra"] * 0.0, b["nextra"] * 0.0
        return super().forward(b)


@contextmanager
def preserve_rng(model=None):
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mode = model.training if model is not None else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if model is not None:
            model.train(mode)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def selection_update(score, best, epoch, best_epoch, stale):
    # There is deliberately no real-test argument to this function.
    if not np.isfinite(score):
        raise FloatingPointError("Non-finite source validation IoU")
    return (score, epoch, 0) if score > best else (best, best_epoch, stale + 1)


def train_with_sampling_audit(model, records, config, optimizer):
    digest = hashlib.sha256()
    original = H.batch

    def audited_batch(records_, indices, cfg, augment=False):
        if augment:
            for r, idx in zip(records_, indices):
                digest.update((r["subset"] + "/" + r["sample_id"]).encode())
                digest.update(np.asarray(idx, dtype=np.int64).tobytes())
        return original(records_, indices, cfg, augment=augment)

    H.batch = audited_batch
    try:
        loss = T.train_epoch(model, records, config, optimizer, 256, 4)
    finally:
        H.batch = original
    return loss, digest.hexdigest()


def local_code():
    files = {Path(__file__).resolve(), Path(__file__).with_name("test_h90_component_ablation_20260910.py").resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and ROOT in path.parents:
                files.add(path)
    return {str(p.relative_to(ROOT)): T.sha256(p) for p in sorted(files)}


def verify_hashes(mapping):
    for path, expected in mapping.items():
        if T.sha256(ROOT / path).lower() != expected.lower():
            raise RuntimeError("Immutable input changed: " + str(path))


def prepare(out, experiment_id, smoke):
    if out.exists():
        raise FileExistsError(out)
    historic = historical_inputs()
    verify_hashes(historic)
    checkpoint = T.verify_base_checkpoint(42)
    if T.sha256(FULL_CACHE) != EXPECTED_FULL_CACHE:
        raise RuntimeError("Historical full cache changed")
    registry = json.loads((ROOT / "docs/current_experiment_registry.json").read_text(encoding="utf-8"))
    split = ROOT / registry["shared_inputs"]["sbpt_split_manifest"]
    code = local_code()
    environment = {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
                   "psutil": psutil.__version__, "device": "NVIDIA GeForce RTX 4090 (runtime checked)"}
    protocol = {
        "experiment_id": experiment_id, "category": "diagnostic" if smoke else "component_ablation",
        "base_experiment": str(BASE_ROOT / "final_results.json"),
        "historical_input_sha256": historic,
        "historical_lock_sha256": T.sha256(HISTORICAL_LOCK),
        "purpose": "Controlled H90 simplification attribution, not automatic final model adoption",
        "code": {"entrypoint": str(Path(__file__).relative_to(ROOT)), "sha256": code},
        "data": {"cache": str(FULL_CACHE.relative_to(ROOT)), "cache_sha256": EXPECTED_FULL_CACHE,
                 "split_manifest": str(split.relative_to(ROOT)), "split_sha256": T.sha256(split),
                 "counts": {"train": 8 if smoke else 800, "val": 4 if smoke else 200, "test": 3 if smoke else 67},
                 "normalization": "existing descriptor source stats; six extra stats fitted on all 800 synthetic training scans only",
                 "base_checkpoint": str(checkpoint.relative_to(ROOT)), "base_sha256": T.EXPECTED_BASE_HASHES[42]},
        "seeds": [42], "arms": list(ARMS), "smoke": smoke,
        "training": {"max_epochs": 1 if smoke else 100, "patience": 15, "optimizer": "AdamW", "lr": 0.0005,
                     "weight_decay": 0.0002, "scheduler": "CosineAnnealingLR eta_min=0.00005, T_max=max_epochs",
                     "batch_clouds": 4, "tokens_per_cloud": 256, "gradient_clip_norm": 5.0,
                     "base": "frozen pretrained PointMLP; no new PointMLP training",
                     "loss": "fused BCE+0.5Dice+0.0001 deviation +0.1 auxiliary(BCE+0.5Dice)+2.0 clean/aug consistency",
                     "no_consistency": "only consistency coefficient becomes zero; keep both forwards and augmentation",
                     "augmentation": T.candidate_config(42, checkpoint),
                     "pairing": "identical head initialization, all input paths, random draws, per-epoch sampled token hashes",
                     "evaluation_rng": "save/restore Python, NumPy, torch CPU/CUDA RNG and model mode"},
        "checkpoint_selection": {"dataset": "synthetic validation", "space": "full original raw point ROI via uniform-overlap projection",
                                 "aggregation": "mean per-scan gauge-line IoU", "metric": "IoU", "threshold": 0.5,
                                 "tie_break": "earlier epoch", "early_stopping": "15 consecutive source non-improvements"},
        "threshold_selection": {"dataset": "synthetic validation", "grid": [float(x) for x in C.THRESHOLDS],
                                "metric": "mean original-point scan IoU", "tie_break": "lowest threshold", "when": "after source checkpoint fixed"},
        "environment": environment,
        "outputs": {"directory": str(out), "expected_results": "7 arm histories, every-epoch checkpoints+67-scan test metrics, source-selected final results/probabilities, base calibration, sampling audit, run manifest"},
        "fairness": {"control": "new full H90 replay in this same queue", "unchanged": "data, pretrained PointMLP, head shapes/init, augmentation, schedule, sampling, source selection, projection, fusion weights",
                     "differences": "one masked information group or one loss coefficient per arm; mask screens are not physically pruned architectures",
                     "limits": "6 selected descriptor channels plus 6 extra statistics; removing direct token-order channel does not remove cached graph dependence; one seed does not establish equivalence"},
        "queue": {"policy": "wait for other Python processes to exit; require 3 idle polls 30s apart; do not interrupt other jobs", "timeout_hours": 24},
        "created_at": now(),
    }
    out.mkdir(parents=True, exist_ok=False)
    for relative in code:
        dest = out / "frozen_code" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    T.write_json(out / "protocol.json", protocol)
    T.write_json(out / "status.json", {"status": "prepared", "at": now()})
    print(json.dumps({"prepared": str(out), "code_files": len(code)}, ensure_ascii=False), flush=True)


def wait_for_slot(out, protocol):
    exempt = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    deadline, quiet = time.monotonic() + 86400, 0
    while time.monotonic() < deadline:
        other = []
        for p in psutil.process_iter(["pid", "name"]):
            if p.info["pid"] not in exempt and (p.info["name"] or "").lower() in ("python.exe", "pythonw.exe", "python"):
                other.append(p.info["pid"])
        quiet = 0 if other else quiet + 1
        T.write_json(out / "status.json", {"status": "waiting_for_compute", "other_python_pids": other,
                                           "idle_polls": quiet, "at": now(), "training_started": False})
        if quiet >= 3:
            return
        time.sleep(30)
    raise TimeoutError("24h compute queue limit reached; no other process was stopped")


def check_dataset(data, expected):
    if {s: len(v) for s, v in data.items()} != expected:
        raise RuntimeError("Wrong split counts")
    for split, records in data.items():
        keys = [(r["subset"], r["sample_id"]) for r in records]
        if len(set(keys)) != len(keys):
            raise RuntimeError("Duplicate scan: " + split)
        for r in records:
            if split in ("train", "val") and r["subset"] != "synthetic":
                raise RuntimeError("Real record in source split")
            if split != "train":
                if r["projection"].shape != (len(r["candidate_gt"]), len(r["y"])):
                    raise RuntimeError("Projection mismatch")
                if r["raw_n"] < len(r["candidate_gt"]) or r["raw_positive"] < int(r["candidate_gt"].sum()):
                    raise RuntimeError("Incomplete raw-ROI accounting")
    tr, va = [{r["sample_id"] for r in data[s]} for s in ("train", "val")]
    if tr & va:
        raise RuntimeError("Source split overlap")


def save_probs(path, vals, tests):
    np.savez_compressed(path, **{f"{s}_{i:04d}": p for s, pp in (("val", vals), ("test", tests)) for i, p in enumerate(pp)})


def gpu_smoke(data, out, device):
    """Seven tiny engineering checks before any complete optimization arm."""
    rows = []
    for arm in ARMS:
        with preserve_rng():
            T.set_seed(42)
            config = T.candidate_config(42, T.BASE_CHECKPOINTS[42])
            if arm == "no_consistency":
                config["consistency_weight"] = 0.0
            model = AblationHead(config, arm).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0002)
            loss, digest = train_with_sampling_audit(model, data["train"][:8], config, opt)
            if not np.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError("GPU smoke non-finite gradient/loss: " + arm)
            vp = H.infer_head(model, data["val"][:4], config)
            real = [next(r for r in data["test"] if r["subset"] == s) for s in ("shoe_a", "shoe_b", "shoe_c")]
            tp = H.infer_head(model, real, config)
            rows.append({"arm": arm, "loss": loss, "sampling_sha256": digest,
                         "val4": C.summarize(data["val"][:4], vp, 0.5),
                         "test3": C.summarize(real, tp, 0.5)})
            del model, opt
            torch.cuda.empty_cache()
    if len({r["sampling_sha256"] for r in rows}) != 1:
        raise RuntimeError("GPU smoke sampling mismatch")
    T.write_json(out / "gpu_smoke.json", {"status": "PASS", "purpose": "engineering only; 8train/4val/3test, not research results", "rows": rows})


def final_for_arm(model, config, data, selected_path, out, history, arm):
    payload = torch.load(selected_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    with preserve_rng(model):
        vp = H.infer_head(model, data["val"], config)
        fixed = C.summarize(data["val"], vp, 0.5)["mean"]
        if abs(fixed - payload["validation_iou_at_0p5"]) > 1e-9:
            raise RuntimeError("Selected source checkpoint failed reproduction")
        threshold, grid = C.choose_threshold(data["val"], vp)
        selection = {"arm": arm, "epoch": payload["epoch"], "validation_iou_at_0p5": fixed,
                     "threshold": threshold, "grid_scores": grid, "checkpoint_sha256": T.sha256(selected_path)}
        T.write_json(out / "source_selection.json", selection)
        tp = H.infer_head(model, data["test"], config)
        result = {**selection, "epochs_executed": len(history), "val": C.summarize(data["val"], vp, threshold),
                  "test": C.summarize(data["test"], tp, threshold), "kind": "source-selected result on repeatedly observed real benchmark"}
        save_probs(out / "selected_token_probabilities.npz", vp, tp)
        T.write_json(out / "result.json", result)
    return result


def execute(out, wait):
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if (out / "runtime_manifest.json").exists():
        raise FileExistsError("Run already started; do not overwrite or silently resume")
    if wait:
        wait_for_slot(out, protocol)
    verify_hashes(protocol["code"]["sha256"])
    verify_hashes(protocol["historical_input_sha256"])
    verify_hashes({protocol["data"]["cache"]: protocol["data"]["cache_sha256"],
                   protocol["data"]["split_manifest"]: protocol["data"]["split_sha256"],
                   protocol["data"]["base_checkpoint"]: protocol["data"]["base_sha256"]})
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required by immutable H90 batching")
    manifest = {"status": "running", "started_at": now(), "pid": os.getpid(), "protocol_sha256": T.sha256(out / "protocol.json"),
                "gpu": torch.cuda.get_device_name(0), "environment": protocol["environment"], "command": sys.argv}
    T.write_json(out / "runtime_manifest.json", manifest)
    T.write_json(out / "status.json", {"status": "loading_data", "at": now(), "training_started": False})
    data = C.load_data(False)
    check_dataset(data, {"train": 800, "val": 200, "test": 67})
    counts = {s: sum(r["subset"] == s for r in data["test"]) for s in ("shoe_a", "shoe_b", "shoe_c")}
    if counts != {"shoe_a": 26, "shoe_b": 24, "shoe_c": 17}:
        raise RuntimeError("Real subset counts changed")
    extra_inputs = {}
    for records in data.values():
        for r in records:
            if "extra" not in r:
                p = C.P.TOKEN_CACHE_ROOT / r["subset"] / (r["sample_id"] + "_downsampled.npz")
                extra_inputs[str(p.relative_to(ROOT))] = T.sha256(p)
    T.write_json(out / "extra_input_hashes.json", extra_inputs)
    stats = C.ensure_extra_features(data)  # Fit TRAIN only, transform all splits.
    np.savez(out / "extra_source_normalization.npz", **stats)
    if protocol["smoke"]:
        data = {"train": data["train"][:8], "val": data["val"][:4],
                "test": [next(r for r in data["test"] if r["subset"] == s) for s in ("shoe_a", "shoe_b", "shoe_c")]}
    check_dataset(data, protocol["data"]["counts"])
    T.write_json(out / "scan_index.json", {s: [{"subset": r["subset"], "sample_id": r["sample_id"]} for r in rs] for s, rs in data.items()})
    T.refresh_frozen_logits(data["train"] + data["val"] + data["test"], T.BASE_CHECKPOINTS[42], 42, device, "ablation_shared_frozen_backbone")
    bp = {s: [torch.sigmoid(torch.from_numpy(r["logit"])).numpy() for r in data[s]] for s in ("val", "test")}
    bt, grid = C.choose_threshold(data["val"], bp["val"])
    base = {"threshold": bt, "val": C.summarize(data["val"], bp["val"], bt), "test": C.summarize(data["test"], bp["test"], bt)}
    T.write_json(out / "paired_frozen_base.json", base)
    if not protocol["smoke"]:
        expected_base = json.loads((BASE_ROOT / "seed42/final_evaluation.json").read_text(encoding="utf-8"))["base"]
        if abs(base["test"]["mean"] - expected_base["real_test_equal_subset_miou"]) > 1e-9:
            raise RuntimeError("Paired raw-point baseline does not reproduce H90 audit")
        # Reproduce the frozen historical head too: this checks descriptors and
        # extra normalization, not just the XYZRGB PointMLP path.
        payload = torch.load(BASE_ROOT / "seed42/best_source_selected.pt", map_location="cpu")
        old = H.ResidualHead(payload["config"]).to(device)
        old.load_state_dict(payload["state_dict"])
        with preserve_rng(old):
            vp = H.infer_head(old, data["val"], payload["config"])
            observed = C.summarize(data["val"], vp, 0.5)["mean"]
        selected = json.loads((BASE_ROOT / "seed42/source_selection.json").read_text(encoding="utf-8"))
        error = abs(observed - selected["best_validation_original_point_mean_iou_at_0p5"])
        T.write_json(out / "historical_source_reproduction.json", {"val_iou_at_0p5": observed, "absolute_error": error,
                      "historical_checkpoint_sha256": T.sha256(BASE_ROOT / "seed42/best_source_selected.pt")})
        if error > 1e-9:
            raise RuntimeError("Historical H90 source score changed; investigate inputs before ablation")
        del old
        torch.cuda.empty_cache()
    gpu_smoke(data, out, device)
    results, initial_hashes, sampled_hashes = [], {}, {}
    max_epochs = protocol["training"]["max_epochs"]
    for arm in protocol["arms"]:
        if wait and results:
            wait_for_slot(out, protocol)
        arm_out = out / arm
        arm_out.mkdir(exist_ok=False)
        config = T.candidate_config(42, T.BASE_CHECKPOINTS[42])
        if arm == "no_consistency":
            config["consistency_weight"] = 0.0
        T.set_seed(42)
        model = AblationHead(config, arm).to(device)
        initial_hashes[arm] = state_digest(model)
        if len(set(initial_hashes.values())) != 1:
            raise RuntimeError("Different head initialization across arms")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0.00005)
        best, best_epoch, stale, history = -1.0, 0, 0, []
        for epoch in range(1, max_epochs + 1):
            started = time.monotonic()
            T.write_json(out / "status.json", {"status": "training", "arm": arm, "epoch": epoch, "at": now(), "training_started": True})
            loss, sample_hash = train_with_sampling_audit(model, data["train"], config, optimizer)
            if epoch in sampled_hashes and sampled_hashes[epoch] != sample_hash:
                raise RuntimeError("Paired training sample sequence diverged")
            sampled_hashes[epoch] = sample_hash
            with preserve_rng(model):
                vp = H.infer_head(model, data["val"], config)
                vs = C.summarize(data["val"], vp, 0.5)
                # Only this source score updates selection and patience.
                best, best_epoch, stale = selection_update(vs["mean"], best, epoch, best_epoch, stale)
                cp = {"state_dict": T.clone_state_dict(model), "config": config, "arm": arm, "seed": 42,
                      "epoch": epoch, "validation_iou_at_0p5": vs["mean"], "protocol_sha256": manifest["protocol_sha256"]}
                torch.save(cp, arm_out / f"epoch_{epoch:03d}.pt")
                if best_epoch == epoch:
                    torch.save(cp, arm_out / "best_source_selected.pt")
                # No gradients/state updates and no test-selected checkpoint.
                tp = H.infer_head(model, data["test"], config)
                ts = C.summarize(data["test"], tp, 0.5)
            scheduler.step()
            row = {"epoch": epoch, "loss": loss, "val_iou_at_0p5": vs["mean"], "test_iou_at_0p5": ts["mean"],
                   "test_subsets_at_0p5": ts["subsets"], "best_source_epoch": best_epoch, "stale": stale,
                   "seconds": time.monotonic() - started, "sampling_sha256": sample_hash,
                   "next_learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            T.write_json(arm_out / "history.json", history)
            T.write_json(arm_out / f"epoch_{epoch:03d}_metrics.json", {"val": vs, "test": ts, "threshold": 0.5, "role": "diagnostic curves only"})
            print(f"{arm} ep{epoch:03d}: val@.5={vs['mean']*100:.3f} test@.5={ts['mean']*100:.3f} A/B/C={ts['subsets']} best_source_ep={best_epoch} ({row['seconds']:.1f}s)", flush=True)
            if stale >= 15:
                break
        result = final_for_arm(model, config, data, arm_out / "best_source_selected.pt", arm_out, history, arm)
        result["allocated_head_parameters"] = sum(p.numel() for p in model.parameters())
        results.append(result)
        T.write_json(out / "results_so_far.json", {"base": base, "arms": results})
        print(f"FINAL {arm}: selected ep{result['epoch']} threshold={result['threshold']:.2f} TEST={result['test']['mean']*100:.4f} {result['test']['subsets']}", flush=True)
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    verify_hashes(protocol["code"]["sha256"])
    T.write_json(out / "pairing_audit.json", {"status": "PASS", "initial_state_sha256": initial_hashes,
                                              "epoch_sampling_sha256": sampled_hashes, "counts": protocol["data"]["counts"]})
    T.write_json(out / "final_results.json", {"status": "complete", "base": base, "arms": results,
                                              "interpretation": protocol["test_data_role"], "no_formal_adoption": True})
    manifest.update(status="complete", ended_at=now())
    T.write_json(out / "runtime_manifest.json", manifest)
    T.write_json(out / "status.json", {"status": "complete", "at": now(), "completed_arms": len(results)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--experiment-id", default="20260910_h90_component_ablation_seed42")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait-for-compute", action="store_true")
    args = parser.parse_args()
    out = args.output_root.resolve()
    if args.prepare:
        prepare(out, args.experiment_id, args.smoke)
    else:
        try:
            execute(out, args.wait_for_compute)
        except Exception as exc:
            T.write_json(out / "status.json", {"status": "failed", "error": repr(exc), "at": now()})
            raise


if __name__ == "__main__":
    main()
