#!/usr/bin/env python
"""Evaluate the locked STE-Net checkpoint in the original raw ROI point space.

The canonical model predicts one probability per source-anchored local-patch
token. This script reconstructs the deterministic token-to-candidate patch
memberships, averages probabilities where patches overlap, and then scatters
the candidate scores back to the original ROI indices. Points outside the LCC
candidate set are background, matching the full-space convention used by the
formal backbone and PCEDNet-style baselines.

No training is performed. The projected-score threshold is selected on the
locked 200-file synthetic validation split using the same 0.05..0.95 grid as
the canonical token-space evaluation. It is then frozen for the 67 real scans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from formal_protocol import canonical_stenet_d04_1000 as C
from formal_protocol import train_stenet_d04_1000 as S


CANDIDATE_ROOT = (
    ROOT / "outputs" / "Stage1_LCC_Candidates_D04_1000_from_existing_LCC_fixed381696"
)
TOKEN_CACHE_ROOT = ROOT / "outputs" / "FormalCache_D04_1000_SourceAnchoredR6_seed42"
CANONICAL_ROOT = C.CANONICAL_ROOT
CHECKPOINT_PATH = C.CANONICAL_CHECKPOINT
SPLIT_MANIFEST = CANONICAL_ROOT / "00_protocol" / "split_manifest.csv"
BASELINE_OFFLINE_ROOT = ROOT / "data" / "lcc"
BASELINE_REAL_CSV = (
    ROOT
    / "outputs"
    / "Figure_Cross_Domain_Comparison_20260722"
    / "real_per_case_iou_original_point_space_20260728.csv"
)
DEFAULT_OUTPUT_ROOT = CANONICAL_ROOT / "full_point_recomputed"
SYNTHETIC_OFFLINE_SUBSET = "D04_AppGeoPhys1000"
LEVEL_MAP = {"shoe_a": "Shoe A", "shoe_b": "Shoe B", "shoe_c": "Shoe C"}
PROJECTION_VERSION = "token_patch_uniform_mean_to_candidate_then_orig_index_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--token-cache-root", type=Path, default=TOKEN_CACHE_ROOT)
    parser.add_argument("--offline-root", type=Path, default=BASELINE_OFFLINE_ROOT)
    parser.add_argument(
        "--ablation-feature-mode",
        choices=("auto", "full", "patch_only", "features_only"),
        default="auto",
    )
    parser.add_argument(
        "--evidence-head",
        choices=("raw", "refined"),
        default="refined",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--max-real", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(-1)[0])


def binary_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    union = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    iou = tp / union if union else 0.0
    precision = tp / precision_den if precision_den else 0.0
    recall = tp / recall_den if recall_den else 0.0
    f1_den = precision + recall
    return {
        "iou": float(iou),
        "precision": float(tp / precision_den) if precision_den else 0.0,
        "recall": float(tp / recall_den) if recall_den else 0.0,
        "f1": float(2.0 * precision * recall / f1_den) if f1_den else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def offline_archive(subset: str, sample_id: str) -> Path:
    folder = SYNTHETIC_OFFLINE_SUBSET if subset == "synthetic" else subset
    path = BASELINE_OFFLINE_ROOT / folder / f"{sample_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing original-space metadata archive: {path}")
    return path


def load_raw_metadata(subset: str, sample_id: str) -> Dict[str, object]:
    path = offline_archive(subset, sample_id)
    with np.load(path, allow_pickle=False) as data:
        raw_label = np.asarray(data["raw_label"], dtype=np.int64).reshape(-1)
        raw_num_points = int(np.asarray(data["raw_num_points"]).reshape(-1)[0])
        source_path = scalar_string(data["source_path"])
    if len(raw_label) != raw_num_points:
        raise RuntimeError(
            f"raw_label length mismatch for {subset}/{sample_id}: "
            f"{len(raw_label)} != {raw_num_points}"
        )
    return {
        "archive_path": path,
        "raw_label": raw_label,
        "raw_num_points": raw_num_points,
        "source_path": source_path,
    }


def reconstruct_patch_memberships(
    record: Mapping[str, object],
    cfg: object,
) -> tuple[Dict[str, np.ndarray], list[np.ndarray], Dict[str, float]]:
    """Recreate the exact patch membership list and audit it against the token cache."""
    candidate = S.M.load_candidate_npy(str(record["path"]))
    fragment = S.R.superline3d_style_fragments(candidate, cfg)
    ds = record["downsampled"]

    xyz = np.asarray(candidate["xyz"], dtype=np.float32)
    labels = np.asarray(candidate["label"], dtype=np.int64)
    orig_index = np.asarray(candidate["orig_index"], dtype=np.int64)
    frag_ids = np.asarray(fragment["frag_ids"], dtype=np.int32)
    valid_ids = np.asarray(fragment["valid_ids"], dtype=np.int32)
    base_spacing = float(fragment.get("base_spacing", 1.0))
    if not np.isfinite(base_spacing) or base_spacing <= 0:
        base_spacing = 1.0
    patch_radius = max(float(cfg.local_patch_radius_factor) * base_spacing, base_spacing)
    patch_stride = max(
        float(getattr(cfg, "local_patch_stride_factor", 2.0)) * base_spacing,
        base_spacing,
    )

    memberships: list[np.ndarray] = []
    expected_orig: list[int] = []
    expected_frag: list[int] = []
    expected_size: list[int] = []
    expected_ratio: list[float] = []
    expected_label: list[int] = []
    expected_xyz: list[np.ndarray] = []

    for fid in valid_ids:
        fid_int = int(fid)
        idx = np.where(frag_ids == fid_int)[0]
        if len(idx) < int(cfg.min_fragment_points):
            continue
        patches = S.M._local_patch_indices(
            pts=xyz[idx],
            orig_local_order=orig_index[idx],
            radius=patch_radius,
            stride=patch_stride,
            min_points=int(cfg.downsample_min_points_per_bin),
        )
        for patch_local in patches:
            src_idx = idx[np.asarray(patch_local, dtype=np.int64)]
            if len(src_idx) < int(cfg.downsample_min_points_per_bin):
                continue
            pts_g = xyz[src_idx]
            local_tangent = S.M._principal_axis(pts_g)
            center = np.mean(pts_g, axis=0).astype(np.float32)
            group_tangent = S.M._principal_axis(pts_g, fallback=local_tangent)
            rep_xyz, rep_orig, _, _ = S.M._source_anchored_axis_medoid(
                pts_g,
                orig_index[src_idx],
                center,
                group_tangent,
                center_weight=float(getattr(cfg, "local_medoid_center_weight", 0.35)),
                perp_weight=float(getattr(cfg, "local_medoid_perp_weight", 1.0)),
            )
            ratio = float(np.mean(labels[src_idx]))
            memberships.append(np.asarray(src_idx, dtype=np.int64))
            expected_orig.append(int(rep_orig))
            expected_frag.append(fid_int)
            expected_size.append(int(len(src_idx)))
            expected_ratio.append(ratio)
            expected_label.append(
                int(ratio >= float(cfg.downsample_label_positive_ratio))
            )
            expected_xyz.append(np.asarray(rep_xyz, dtype=np.float32))

    token_count = len(np.asarray(ds["label"]))
    if len(memberships) != token_count:
        raise RuntimeError(
            f"Token count reconstruction mismatch for {record['subset']}/{record['sample_id']}: "
            f"{len(memberships)} != {token_count}"
        )
    checks = {
        "orig_index": np.array_equal(
            np.asarray(expected_orig, dtype=np.int64),
            np.asarray(ds["orig_index"], dtype=np.int64),
        ),
        "fragment_id": np.array_equal(
            np.asarray(expected_frag, dtype=np.int32),
            np.asarray(ds["fragment_id"], dtype=np.int32),
        ),
        "bin_num_points": np.array_equal(
            np.asarray(expected_size, dtype=np.int32),
            np.asarray(ds["bin_num_points"], dtype=np.int32),
        ),
        "label": np.array_equal(
            np.asarray(expected_label, dtype=np.int64),
            np.asarray(ds["label"], dtype=np.int64),
        ),
        "label_ratio": np.allclose(
            np.asarray(expected_ratio, dtype=np.float32),
            np.asarray(ds["label_ratio"], dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        ),
        "xyz": np.allclose(
            np.vstack(expected_xyz).astype(np.float32),
            np.asarray(ds["xyz"], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Token reconstruction audit failed for {record['subset']}/"
            f"{record['sample_id']}: {failed}"
        )
    covered = np.zeros(len(xyz), dtype=bool)
    overlap = np.zeros(len(xyz), dtype=np.uint16)
    for member in memberships:
        covered[member] = True
        np.add.at(overlap, member, 1)
    candidate = dict(candidate)
    candidate["fragment_id"] = frag_ids.astype(np.int32)
    candidate["base_spacing"] = np.asarray(base_spacing, dtype=np.float32)
    return candidate, memberships, {
        "token_count": int(token_count),
        "candidate_count": int(len(xyz)),
        "covered_candidate_count": int(np.sum(covered)),
        "covered_candidate_fraction": float(np.mean(covered)) if len(covered) else 0.0,
        "mean_votes_on_covered": (
            float(np.mean(overlap[covered])) if np.any(covered) else 0.0
        ),
        "max_votes": int(np.max(overlap)) if len(overlap) else 0,
        "base_spacing": float(base_spacing),
        "patch_radius": float(patch_radius),
        "patch_stride": float(patch_stride),
    }


def project_probabilities(
    token_probability: np.ndarray,
    memberships: Sequence[np.ndarray],
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(token_probability) != len(memberships):
        raise ValueError(
            f"Token probability/membership mismatch: "
            f"{len(token_probability)} != {len(memberships)}"
        )
    probability_sum = np.zeros(candidate_count, dtype=np.float64)
    vote_count = np.zeros(candidate_count, dtype=np.uint16)
    for probability, member in zip(token_probability, memberships):
        probability_sum[member] += float(probability)
        np.add.at(vote_count, member, 1)
    candidate_probability = np.zeros(candidate_count, dtype=np.float32)
    covered = vote_count > 0
    candidate_probability[covered] = (
        probability_sum[covered] / vote_count[covered].astype(np.float64)
    ).astype(np.float32)
    return candidate_probability, vote_count


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_model(
    device: torch.device,
    output_root: Path,
    checkpoint_path: Path,
    requested_ablation_mode: str,
) -> tuple[
    torch.nn.Module,
    object,
    Mapping[str, object],
    np.ndarray,
    np.ndarray,
    str,
]:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_mode = str(checkpoint.get("ablation_feature_mode", "full"))
    ablation_mode = (
        checkpoint_mode
        if str(requested_ablation_mode) == "auto"
        else str(requested_ablation_mode)
    )
    if checkpoint_mode != ablation_mode:
        raise RuntimeError(
            "Checkpoint/input ablation mismatch: "
            f"checkpoint={checkpoint_mode}, requested={ablation_mode}"
        )
    args = argparse.Namespace(
        candidate_dir=CANDIDATE_ROOT,
        cache_root=TOKEN_CACHE_ROOT,
        output_root=output_root / "runtime",
        seed=int(checkpoint["seed"]),
        device=str(device),
    )
    cfg = S.build_compact_config(args)
    cfg.ablation_feature_mode = (
        "full" if ablation_mode == "features_only" else ablation_mode
    )
    S.M.set_global_determinism(int(args.seed))
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    model = S.CompactSTENet(cfg, int(feature_mean.shape[-1]))
    model.local_estimator.load_state_dict(checkpoint["local_estimator"], strict=True)
    model.context_refiner.load_state_dict(checkpoint["context_refiner"], strict=True)
    model.to(device).eval()
    if checkpoint_path == CHECKPOINT_PATH.resolve() and ablation_mode == "full":
        canonical = C.require_canonical_checkpoint()
        if sha256(checkpoint_path) != str(canonical["checkpoint_sha256"]):
            raise RuntimeError("Canonical checkpoint hash changed during initialization")
    return model, cfg, checkpoint, feature_mean, feature_std, ablation_mode


@torch.no_grad()
def infer_and_project_one(
    row: Mapping[str, object],
    model: torch.nn.Module,
    cfg: object,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    prediction_path: Path,
    checkpoint_path: Path,
    evidence_head: str,
    ablation_mode: str,
) -> Dict[str, object]:
    record = S.B.prepare_records([dict(row)], TOKEN_CACHE_ROOT, cfg)[0]
    if ablation_mode == "features_only":
        patch_points = np.asarray(
            record["downsampled"]["patch_points"],
            dtype=np.float32,
        )
        record["downsampled"]["patch_points"] = np.zeros_like(
            patch_points,
            dtype=np.float32,
        )
    S.prepare_model_arrays([record], feature_mean, feature_std, cfg)
    raw_logit, refined_logit, _, _ = S.compact_forward_record(
        model, record, device, cfg, augment=False
    )
    selected_logit = raw_logit if evidence_head == "raw" else refined_logit
    token_probability = (
        torch.sigmoid(selected_logit).detach().cpu().numpy().astype(np.float32)
    )
    candidate, memberships, audit = reconstruct_patch_memberships(record, cfg)
    candidate_probability, vote_count = project_probabilities(
        token_probability,
        memberships,
        len(candidate["xyz"]),
    )
    candidate_orig_index = np.asarray(candidate["orig_index"], dtype=np.int64)
    raw = load_raw_metadata(str(row["subset"]), str(row["sample_id"]))
    raw_label = np.asarray(raw["raw_label"], dtype=np.int64)
    if len(np.unique(candidate_orig_index)) != len(candidate_orig_index):
        raise RuntimeError(
            f"Duplicate candidate original indices: {row['subset']}/{row['sample_id']}"
        )
    if (
        len(candidate_orig_index)
        and (
            int(candidate_orig_index.min()) < 0
            or int(candidate_orig_index.max()) >= int(raw["raw_num_points"])
        )
    ):
        raise RuntimeError(
            f"Candidate original index out of range: {row['subset']}/{row['sample_id']}"
        )
    candidate_label = np.asarray(candidate["label"], dtype=np.int64)
    if not np.array_equal(candidate_label, raw_label[candidate_orig_index]):
        raise RuntimeError(
            f"Candidate/raw label alignment failed: {row['subset']}/{row['sample_id']}"
        )
    save_npz_atomic(
        prediction_path,
        projection_version=np.asarray(PROJECTION_VERSION),
        checkpoint_sha256=np.asarray(sha256(checkpoint_path)),
        evidence_head=np.asarray(str(evidence_head)),
        ablation_feature_mode=np.asarray(str(ablation_mode)),
        domain=np.asarray(str(row["domain"])),
        subset=np.asarray(str(row["subset"])),
        split=np.asarray(str(row["split"])),
        sample_id=np.asarray(str(row["sample_id"])),
        source_path=np.asarray(str(raw["source_path"])),
        raw_num_points=np.asarray(int(raw["raw_num_points"]), dtype=np.int64),
        raw_positive_points=np.asarray(int(np.sum(raw_label)), dtype=np.int64),
        candidate_orig_index=candidate_orig_index,
        candidate_probability=candidate_probability,
        candidate_vote_count=vote_count,
        token_probability=token_probability,
        token_orig_index=np.asarray(record["downsampled"]["orig_index"], dtype=np.int64),
        token_fragment_id=np.asarray(
            record["downsampled"]["fragment_id"], dtype=np.int32
        ),
    )
    return {
        **audit,
        "domain": str(row["domain"]),
        "subset": str(row["subset"]),
        "split": str(row["split"]),
        "sample_id": str(row["sample_id"]),
        "raw_num_points": int(raw["raw_num_points"]),
        "raw_positive_points": int(np.sum(raw_label)),
        "candidate_positive_points": int(np.sum(candidate_label)),
        "covered_positive_points": int(
            np.sum(candidate_label[np.asarray(vote_count) > 0])
        ),
        "prediction_path": str(prediction_path),
    }


def prediction_path(output_root: Path, row: Mapping[str, object]) -> Path:
    return (
        output_root
        / "projected_probabilities"
        / str(row["subset"])
        / f"{row['sample_id']}.npz"
    )


def ensure_predictions(
    rows: Sequence[Mapping[str, object]],
    output_root: Path,
    model: torch.nn.Module,
    cfg: object,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
    force: bool,
    description: str,
    checkpoint_path: Path,
    evidence_head: str,
    ablation_mode: str,
) -> pd.DataFrame:
    audit_rows: list[Dict[str, object]] = []
    for row in tqdm(rows, desc=description):
        path = prediction_path(output_root, row)
        if path.is_file() and not force:
            with np.load(path, allow_pickle=False) as data:
                if scalar_string(data["projection_version"]) != PROJECTION_VERSION:
                    raise RuntimeError(f"Stale projection cache: {path}")
                if scalar_string(data["checkpoint_sha256"]) != sha256(checkpoint_path):
                    raise RuntimeError(f"Checkpoint mismatch in projection cache: {path}")
                if scalar_string(data["evidence_head"]) != str(evidence_head):
                    raise RuntimeError(f"Evidence-head mismatch in projection cache: {path}")
                if scalar_string(data["ablation_feature_mode"]) != str(ablation_mode):
                    raise RuntimeError(f"Ablation-mode mismatch in projection cache: {path}")
                candidate_count = int(len(data["candidate_probability"]))
                token_count = int(len(data["token_probability"]))
                covered = np.asarray(data["candidate_vote_count"]) > 0
                raw_n = int(np.asarray(data["raw_num_points"]).reshape(-1)[0])
                raw_pos = int(np.asarray(data["raw_positive_points"]).reshape(-1)[0])
                audit_rows.append(
                    {
                        "domain": str(row["domain"]),
                        "subset": str(row["subset"]),
                        "split": str(row["split"]),
                        "sample_id": str(row["sample_id"]),
                        "token_count": token_count,
                        "candidate_count": candidate_count,
                        "covered_candidate_count": int(np.sum(covered)),
                        "covered_candidate_fraction": (
                            float(np.mean(covered)) if candidate_count else 0.0
                        ),
                        "mean_votes_on_covered": (
                            float(np.mean(data["candidate_vote_count"][covered]))
                            if np.any(covered)
                            else 0.0
                        ),
                        "max_votes": (
                            int(np.max(data["candidate_vote_count"]))
                            if candidate_count
                            else 0
                        ),
                        "raw_num_points": raw_n,
                        "raw_positive_points": raw_pos,
                        "prediction_path": str(path),
                        "loaded_from_projection_cache": True,
                    }
                )
            continue
        audit = infer_and_project_one(
            row,
            model,
            cfg,
            feature_mean,
            feature_std,
            device,
            path,
            checkpoint_path,
            evidence_head,
            ablation_mode,
        )
        audit["loaded_from_projection_cache"] = False
        audit_rows.append(audit)
    return pd.DataFrame(audit_rows)


def counts_for_prediction(
    subset: str,
    sample_id: str,
    candidate_orig_index: np.ndarray,
    candidate_probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    raw = load_raw_metadata(subset, sample_id)
    raw_label = np.asarray(raw["raw_label"], dtype=np.int64)
    candidate_orig_index = np.asarray(candidate_orig_index, dtype=np.int64)
    candidate_gt = raw_label[candidate_orig_index] == 1
    candidate_pred = np.asarray(candidate_probability) >= float(threshold)
    tp = int(np.sum(candidate_pred & candidate_gt))
    fp = int(np.sum(candidate_pred & ~candidate_gt))
    fn = int(np.sum(raw_label == 1)) - tp
    tn = len(raw_label) - tp - fp - fn
    return binary_metrics_from_counts(tp, fp, fn, tn)


def select_projection_threshold(
    rows: Sequence[Mapping[str, object]],
    output_root: Path,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    thresholds = tuple(float(value) for value in S.M.THRESHOLD_GRID)
    totals = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in thresholds
    }
    per_case_rows: list[Dict[str, object]] = []
    for row in tqdm(rows, desc="Select full-point threshold on synthetic validation"):
        path = prediction_path(output_root, row)
        with np.load(path, allow_pickle=False) as data:
            candidate_orig_index = np.asarray(data["candidate_orig_index"], dtype=np.int64)
            candidate_probability = np.asarray(
                data["candidate_probability"], dtype=np.float32
            )
        for threshold in thresholds:
            metrics = counts_for_prediction(
                str(row["subset"]),
                str(row["sample_id"]),
                candidate_orig_index,
                candidate_probability,
                threshold,
            )
            for key in ("tp", "fp", "fn", "tn"):
                totals[threshold][key] += int(metrics[key])
            per_case_rows.append(
                {
                    "domain": str(row["domain"]),
                    "subset": str(row["subset"]),
                    "split": str(row["split"]),
                    "sample_id": str(row["sample_id"]),
                    "threshold": float(threshold),
                    "metric_space": "original_raw_roi_point_space",
                    **metrics,
                }
            )
    per_case_frame = pd.DataFrame(per_case_rows)
    grid_rows = []
    for threshold in thresholds:
        counts = totals[threshold]
        pooled = binary_metrics_from_counts(**counts)
        frame = per_case_frame[np.isclose(per_case_frame["threshold"], threshold)]
        grid_rows.append(
            {
                "threshold": float(threshold),
                "metric_space": "original_raw_roi_point_space",
                "aggregation": "macro_mean_over_point_clouds",
                "cases": int(len(frame)),
                "miou": float(frame["iou"].mean()),
                "iou": float(frame["iou"].mean()),
                "precision": float(frame["precision"].mean()),
                "recall": float(frame["recall"].mean()),
                "f1": float(frame["f1"].mean()),
                "pooled_iou": pooled["iou"],
                "pooled_precision": pooled["precision"],
                "pooled_recall": pooled["recall"],
                "pooled_f1": pooled["f1"],
                **counts,
            }
        )
    grid = pd.DataFrame(grid_rows)
    best = grid.sort_values(
        ["miou", "f1", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best["threshold"]), grid, per_case_frame


def evaluate_real(
    rows: Sequence[Mapping[str, object]],
    output_root: Path,
    threshold: float,
) -> pd.DataFrame:
    result: list[Dict[str, object]] = []
    for row in tqdm(rows, desc="Frozen real full-point evaluation"):
        path = prediction_path(output_root, row)
        with np.load(path, allow_pickle=False) as data:
            candidate_orig_index = np.asarray(data["candidate_orig_index"], dtype=np.int64)
            candidate_probability = np.asarray(
                data["candidate_probability"], dtype=np.float32
            )
            raw_num_points = int(np.asarray(data["raw_num_points"]).reshape(-1)[0])
            token_count = int(len(data["token_probability"]))
        metrics = counts_for_prediction(
            str(row["subset"]),
            str(row["sample_id"]),
            candidate_orig_index,
            candidate_probability,
            threshold,
        )
        result.append(
            {
                "method": "STE-Net (Ours)",
                "method_key": "stenet_full_projection",
                "level": LEVEL_MAP[str(row["subset"])],
                "subset": str(row["subset"]),
                "sample_id": str(row["sample_id"]),
                "metric_space": "original_raw_roi_point_space",
                "projection_threshold": float(threshold),
                "points": raw_num_points,
                "candidate_points": int(len(candidate_orig_index)),
                "token_points": token_count,
                **metrics,
            }
        )
    return pd.DataFrame(result)


def summarize_real(real: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, frame in real.groupby("level", sort=False):
        counts = {
            key: int(frame[key].sum()) for key in ("tp", "fp", "fn", "tn")
        }
        pooled = binary_metrics_from_counts(**counts)
        rows.append(
            {
                "level": level,
                "cases": int(len(frame)),
                "pooled_iou": pooled["iou"],
                "pooled_precision": pooled["precision"],
                "pooled_recall": pooled["recall"],
                "pooled_f1": pooled["f1"],
                "macro_case_iou_mean": float(frame["iou"].mean()),
                "macro_case_iou_std": float(frame["iou"].std(ddof=1)),
                "macro_case_precision_mean": float(frame["precision"].mean()),
                "macro_case_recall_mean": float(frame["recall"].mean()),
                "macro_case_f1_mean": float(frame["f1"].mean()),
                **counts,
            }
        )
    summary = pd.DataFrame(rows)
    equal_subset = summary[
        [
            "macro_case_iou_mean",
            "macro_case_precision_mean",
            "macro_case_recall_mean",
            "macro_case_f1_mean",
        ]
    ].mean()
    overall_counts = {
        key: int(real[key].sum()) for key in ("tp", "fp", "fn", "tn")
    }
    overall = binary_metrics_from_counts(**overall_counts)
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "level": "Equal-subset macro mean",
                        "cases": int(len(real)),
                        "pooled_iou": float("nan"),
                        "pooled_precision": float("nan"),
                        "pooled_recall": float("nan"),
                        "pooled_f1": float("nan"),
                        "macro_case_iou_mean": float(
                            equal_subset["macro_case_iou_mean"]
                        ),
                        "macro_case_iou_std": float("nan"),
                        "macro_case_precision_mean": float(
                            equal_subset["macro_case_precision_mean"]
                        ),
                        "macro_case_recall_mean": float(
                            equal_subset["macro_case_recall_mean"]
                        ),
                        "macro_case_f1_mean": float(
                            equal_subset["macro_case_f1_mean"]
                        ),
                        **{key: np.nan for key in overall_counts},
                    },
                    {
                        "level": "All real scans pooled audit",
                        "cases": int(len(real)),
                        "pooled_iou": overall["iou"],
                        "pooled_precision": overall["precision"],
                        "pooled_recall": overall["recall"],
                        "pooled_f1": overall["f1"],
                        "macro_case_iou_mean": float(real["iou"].mean()),
                        "macro_case_iou_std": float(real["iou"].std(ddof=1)),
                        "macro_case_precision_mean": float(real["precision"].mean()),
                        "macro_case_recall_mean": float(real["recall"].mean()),
                        "macro_case_f1_mean": float(real["f1"].mean()),
                        **overall_counts,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    return summary


def merge_with_full_space_baselines(
    real: pd.DataFrame,
    destination: Path,
) -> pd.DataFrame | None:
    if not BASELINE_REAL_CSV.is_file():
        return None
    baseline = pd.read_csv(BASELINE_REAL_CSV)
    baseline = baseline[
        ~baseline["method_key"].astype(str).str.startswith("stenet")
    ].copy()
    baseline["level"] = baseline["subset"].map(LEVEL_MAP)
    if baseline["level"].isna().any():
        raise RuntimeError("Baseline table contains an unknown real-data subset")
    baseline["metric_space"] = "original_raw_roi_point_space"
    common = [
        "method",
        "method_key",
        "level",
        "subset",
        "sample_id",
        "metric_space",
        "points",
        "iou",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    combined = pd.concat([baseline[common], real[common]], ignore_index=True)
    combined.to_csv(destination, index=False)
    return combined


def summarize_all_methods(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for (method, method_key, level), frame in combined.groupby(
        ["method", "method_key", "level"],
        sort=False,
    ):
        counts = {
            key: int(frame[key].sum()) for key in ("tp", "fp", "fn", "tn")
        }
        metrics = binary_metrics_from_counts(**counts)
        rows.append(
            {
                "method": method,
                "method_key": method_key,
                "level": level,
                "cases": int(len(frame)),
                "metric_space": "original_raw_roi_point_space",
                "pooled_iou": metrics["iou"],
                "pooled_precision": metrics["precision"],
                "pooled_recall": metrics["recall"],
                "pooled_f1": metrics["f1"],
                "macro_case_iou_mean": float(frame["iou"].mean()),
                "macro_case_precision_mean": float(frame["precision"].mean()),
                "macro_case_recall_mean": float(frame["recall"].mean()),
                "macro_case_f1_mean": float(frame["f1"].mean()),
                **counts,
            }
        )
    summary = pd.DataFrame(rows)
    overall_rows = []
    for (method, method_key), frame in combined.groupby(
        ["method", "method_key"],
        sort=False,
    ):
        subset_means = frame.groupby("level", sort=False)[
            ["iou", "precision", "recall", "f1"]
        ].mean()
        if len(subset_means) != 3:
            raise RuntimeError(f"Expected three real-data subsets for {method}")
        overall_rows.append(
            {
                "method": method,
                "method_key": method_key,
                "level": "Equal-subset macro mean",
                "cases": int(len(frame)),
                "metric_space": "original_raw_roi_point_space",
                "pooled_iou": float("nan"),
                "pooled_precision": float("nan"),
                "pooled_recall": float("nan"),
                "pooled_f1": float("nan"),
                "macro_case_iou_mean": float(subset_means["iou"].mean()),
                "macro_case_precision_mean": float(
                    subset_means["precision"].mean()
                ),
                "macro_case_recall_mean": float(subset_means["recall"].mean()),
                "macro_case_f1_mean": float(subset_means["f1"].mean()),
                "tp": np.nan,
                "fp": np.nan,
                "fn": np.nan,
                "tn": np.nan,
            }
        )
    return pd.concat([summary, pd.DataFrame(overall_rows)], ignore_index=True)


def summarize_projection_support(
    rows: Sequence[Mapping[str, object]],
    output_root: Path,
) -> pd.DataFrame:
    totals: Dict[str, Dict[str, int]] = {}
    for row in rows:
        subset = str(row["subset"])
        sample_id = str(row["sample_id"])
        with np.load(prediction_path(output_root, row), allow_pickle=False) as data:
            candidate_orig = np.asarray(data["candidate_orig_index"], dtype=np.int64)
            covered = np.asarray(data["candidate_vote_count"]) > 0
        raw = load_raw_metadata(subset, sample_id)
        raw_label = np.asarray(raw["raw_label"], dtype=bool)
        item = totals.setdefault(
            subset,
            {
                "files": 0,
                "raw_points": 0,
                "candidate_points": 0,
                "token_covered_points": 0,
                "raw_positive_points": 0,
                "candidate_positive_points": 0,
                "token_covered_positive_points": 0,
            },
        )
        item["files"] += 1
        item["raw_points"] += int(len(raw_label))
        item["candidate_points"] += int(len(candidate_orig))
        item["token_covered_points"] += int(np.sum(covered))
        item["raw_positive_points"] += int(np.sum(raw_label))
        item["candidate_positive_points"] += int(np.sum(raw_label[candidate_orig]))
        item["token_covered_positive_points"] += int(
            np.sum(raw_label[candidate_orig[covered]])
        )
    summary = pd.DataFrame(
        [{"subset": subset, **values} for subset, values in totals.items()]
    )
    summary["candidate_fraction_of_raw"] = (
        summary["candidate_points"] / summary["raw_points"]
    )
    summary["token_covered_fraction_of_raw"] = (
        summary["token_covered_points"] / summary["raw_points"]
    )
    summary["stage1_positive_recall_upper_bound"] = (
        summary["candidate_positive_points"] / summary["raw_positive_points"]
    )
    summary["token_coverage_positive_recall_upper_bound"] = (
        summary["token_covered_positive_points"] / summary["raw_positive_points"]
    )
    return summary


def load_raw_xyz_rgb(path: Path, expected_points: int) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".txt":
        data = np.loadtxt(path, dtype=np.float32)
    elif path.suffix.lower() == ".npy":
        data = np.load(path)
    else:
        raise ValueError(f"Unsupported raw point-cloud format: {path}")
    if data.ndim != 2 or data.shape[1] < 6 or len(data) != expected_points:
        raise RuntimeError(
            f"Raw visualization point cloud mismatch: {path}, "
            f"shape={data.shape}, expected points={expected_points}"
        )
    xyz = np.asarray(data[:, :3], dtype=np.float32)
    rgb = np.clip(np.asarray(data[:, 3:6], dtype=np.float32) / 255.0, 0.0, 1.0)
    return xyz, rgb


def side_view(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xyz[:, 0], dtype=np.float32)
    z = np.asarray(xyz[:, 2], dtype=np.float32)
    x_mid = 0.5 * (float(x.min()) + float(x.max()))
    z_mid = 0.5 * (float(z.min()) + float(z.max()))
    scale = max(float(x.max() - x.min()), 1e-6)
    return (x - x_mid) / scale, (z - z_mid) / scale


def gt_roi(
    x: np.ndarray,
    z: np.ndarray,
    label: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    positive = np.asarray(label, dtype=bool)
    if not np.any(positive):
        return (float(x.min()), float(x.max())), (float(z.min()), float(z.max()))
    px = x[positive]
    pz = z[positive]
    x0, x1 = np.quantile(px, [0.005, 0.995])
    z0, z1 = np.quantile(pz, [0.005, 0.995])
    x_margin = max(0.04, 0.18 * float(x1 - x0))
    z_margin = max(0.025, 0.25 * float(z1 - z0))
    return (
        (float(x0 - x_margin), float(x1 + x_margin)),
        (float(z0 - z_margin), float(z1 + z_margin)),
    )


def representative_rows(real: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for level, frame in real.groupby("level", sort=False):
        median = float(frame["iou"].median())
        index = (frame["iou"] - median).abs().idxmin()
        chosen.append(real.loc[index])
    return pd.DataFrame(chosen)


def render_visualization(
    real: pd.DataFrame,
    output_root: Path,
    threshold: float,
) -> list[str]:
    selected = representative_rows(real)
    available = []
    for _, row in selected.iterrows():
        raw = load_raw_metadata(str(row["subset"]), str(row["sample_id"]))
        source = Path(str(raw["source_path"]))
        if source.is_file():
            available.append((row, raw, source))
    if not available:
        return []

    columns = (
        "Raw RGB",
        "Ground truth",
        "Token anchors",
        "Full-point reprojection",
        "Error map",
    )
    fig, axes = plt.subplots(
        len(available),
        len(columns),
        figsize=(15.5, 1.35 * len(available) + 0.70),
        dpi=260,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    for row_index, (metric_row, raw, source) in enumerate(available):
        raw_label = np.asarray(raw["raw_label"], dtype=bool)
        xyz, rgb = load_raw_xyz_rgb(source, int(raw["raw_num_points"]))
        x, z = side_view(xyz)
        xlim, zlim = gt_roi(x, z, raw_label)
        order = np.argsort(xyz[:, 1])
        path = (
            output_root
            / "projected_probabilities"
            / str(metric_row["subset"])
            / f"{metric_row['sample_id']}.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            candidate_orig = np.asarray(data["candidate_orig_index"], dtype=np.int64)
            candidate_probability = np.asarray(
                data["candidate_probability"], dtype=np.float32
            )
            token_orig = np.asarray(data["token_orig_index"], dtype=np.int64)
            token_probability = np.asarray(data["token_probability"], dtype=np.float32)
        full_prediction = np.zeros(len(raw_label), dtype=bool)
        full_prediction[candidate_orig] = candidate_probability >= float(threshold)
        token_positive = token_orig[token_probability >= float(threshold)]
        tp = full_prediction & raw_label
        fp = full_prediction & ~raw_label
        fn = ~full_prediction & raw_label

        for column_index, title in enumerate(columns):
            ax = axes[row_index, column_index]
            if title == "Raw RGB":
                ax.scatter(
                    x[order],
                    z[order],
                    c=np.clip(rgb[order], 0.0, 1.0),
                    s=0.35,
                    linewidths=0,
                    marker=".",
                    rasterized=True,
                )
            else:
                ax.scatter(
                    x[order],
                    z[order],
                    c="#D9D9D9",
                    s=0.28,
                    linewidths=0,
                    marker=".",
                    alpha=0.70,
                    rasterized=True,
                )
                if title == "Ground truth":
                    idx = np.flatnonzero(raw_label)
                    ax.scatter(x[idx], z[idx], c="#D7191C", s=0.85, linewidths=0)
                elif title == "Token anchors":
                    ax.scatter(
                        x[token_positive],
                        z[token_positive],
                        c="#7B3294",
                        s=5.2,
                        linewidths=0,
                    )
                elif title == "Full-point reprojection":
                    idx = np.flatnonzero(full_prediction)
                    ax.scatter(x[idx], z[idx], c="#D7191C", s=0.85, linewidths=0)
                else:
                    for mask, color in (
                        (tp, "#1A9850"),
                        (fp, "#D73027"),
                        (fn, "#2C7BB6"),
                    ):
                        idx = np.flatnonzero(mask)
                        ax.scatter(x[idx], z[idx], c=color, s=0.90, linewidths=0)
            ax.set_xlim(*xlim)
            ax.set_ylim(*zlim)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            if row_index == 0:
                ax.set_title(title, fontsize=10.5, pad=7)
            if column_index == 0:
                ax.text(
                    -0.04,
                    0.5,
                    f"{metric_row['level']}\n"
                    f"{metric_row['sample_id']}\n"
                    f"IoU={float(metric_row['iou']):.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9.5,
                )
    fig.suptitle(
        f"Locked STE-Net: token evidence reprojected to the original ROI point space "
        f"(source-selected threshold = {threshold:.2f})\n"
        "Error map: TP green, FP red, FN blue",
        fontsize=12.0,
        y=0.998,
    )
    fig.tight_layout(rect=(0.04, 0.015, 1.0, 0.965), w_pad=0.5, h_pad=0.75)
    png = output_root / "full_point_reprojection_representative_levels.png"
    pdf = output_root / "full_point_reprojection_representative_levels.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def main() -> int:
    global CANDIDATE_ROOT, TOKEN_CACHE_ROOT, BASELINE_OFFLINE_ROOT
    args = parse_args()
    CANDIDATE_ROOT = args.candidate_root.resolve()
    TOKEN_CACHE_ROOT = args.token_cache_root.resolve()
    BASELINE_OFFLINE_ROOT = args.offline_root.resolve()
    output_root = args.output_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    split_manifest = args.split_manifest.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {split_manifest}")
    if not CANDIDATE_ROOT.is_dir():
        raise FileNotFoundError(f"Candidate root does not exist: {CANDIDATE_ROOT}")
    if not TOKEN_CACHE_ROOT.is_dir():
        raise FileNotFoundError(f"Token cache root does not exist: {TOKEN_CACHE_ROOT}")
    if not BASELINE_OFFLINE_ROOT.is_dir():
        raise FileNotFoundError(f"Offline metadata root does not exist: {BASELINE_OFFLINE_ROOT}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = torch.device(args.device)
    (
        model,
        cfg,
        checkpoint,
        feature_mean,
        feature_std,
        ablation_mode,
    ) = load_model(
        device,
        output_root,
        checkpoint_path,
        str(args.ablation_feature_mode),
    )

    manifest = pd.read_csv(split_manifest)
    validation_rows = manifest[
        (manifest["domain"] == "synthetic") & (manifest["split"] == "val")
    ].to_dict("records")
    real_rows = manifest[
        (manifest["domain"] == "real")
        & (manifest["split"] == "test")
        & (manifest["subset"].isin(LEVEL_MAP))
    ].to_dict("records")
    if int(args.max_validation) > 0:
        validation_rows = validation_rows[: int(args.max_validation)]
    if int(args.max_real) > 0:
        real_rows = real_rows[: int(args.max_real)]
    if int(args.max_validation) == 0 and len(validation_rows) != 200:
        raise RuntimeError(f"Expected 200 validation files, found {len(validation_rows)}")
    if int(args.max_real) == 0 and len(real_rows) != 67:
        raise RuntimeError(f"Expected 67 real files, found {len(real_rows)}")

    validation_audit = ensure_predictions(
        validation_rows,
        output_root,
        model,
        cfg,
        feature_mean,
        feature_std,
        device,
        args.force,
        f"Reproject synthetic validation ({ablation_mode}/{args.evidence_head})",
        checkpoint_path,
        str(args.evidence_head),
        ablation_mode,
    )
    threshold, threshold_grid, validation_per_case = select_projection_threshold(
        validation_rows,
        output_root,
    )
    threshold_grid.to_csv(output_root / "synthetic_validation_threshold_grid.csv", index=False)
    validation_per_case.to_csv(
        output_root / "synthetic_validation_per_case_threshold_grid.csv",
        index=False,
    )

    real_audit = ensure_predictions(
        real_rows,
        output_root,
        model,
        cfg,
        feature_mean,
        feature_std,
        device,
        args.force,
        f"Reproject frozen real scans ({ablation_mode}/{args.evidence_head})",
        checkpoint_path,
        str(args.evidence_head),
        ablation_mode,
    )
    audit = pd.concat([validation_audit, real_audit], ignore_index=True)
    audit.to_csv(output_root / "projection_audit.csv", index=False)
    real = evaluate_real(real_rows, output_root, threshold)
    real.to_csv(output_root / "real_per_case_full_point_iou.csv", index=False)
    summary = summarize_real(real)
    summary.to_csv(output_root / "real_subset_summary_full_point.csv", index=False)
    combined = merge_with_full_space_baselines(
        real,
        output_root / "real_per_case_all_methods_original_point_space.csv",
    )
    all_method_summary = None
    if combined is not None:
        all_method_summary = summarize_all_methods(combined)
        all_method_summary.to_csv(
            output_root / "real_subset_summary_all_methods_original_point_space.csv",
            index=False,
        )
    support_summary = summarize_projection_support(
        [*validation_rows, *real_rows],
        output_root,
    )
    support_summary.to_csv(output_root / "projection_support_summary.csv", index=False)
    visualization_paths = []
    if not args.skip_visualization and int(args.max_real) == 0:
        visualization_paths = render_visualization(real, output_root, threshold)

    best_row = threshold_grid.loc[
        np.isclose(threshold_grid["threshold"], threshold)
    ].iloc[0]
    result = {
        "projection_version": PROJECTION_VERSION,
        "canonical_model_id": (
            C.require_canonical_checkpoint()["canonical_model_id"]
            if checkpoint_path == CHECKPOINT_PATH.resolve() and ablation_mode == "full"
            else None
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "split_manifest": str(split_manifest),
        "ablation_feature_mode": ablation_mode,
        "evidence_head": str(args.evidence_head),
        "checkpoint_token_threshold": float(
            checkpoint["thresholds"][str(args.evidence_head)]
        ),
        "projection_rule": (
            "Uniform mean of all overlapping token probabilities on each exact "
            "candidate patch member; uncovered candidates and all non-candidates "
            "are background; candidate scores are scattered by orig_index."
        ),
        "threshold_selection": {
            "data": "locked synthetic validation split only",
            "validation_files": int(len(validation_rows)),
            "grid": [float(value) for value in S.M.THRESHOLD_GRID],
            "selected_threshold": float(threshold),
            "aggregation": "macro_mean_over_point_clouds",
            "validation_miou": float(best_row["miou"]),
            "validation_mean_precision": float(best_row["precision"]),
            "validation_mean_recall": float(best_row["recall"]),
            "validation_mean_f1": float(best_row["f1"]),
            "pooled_full_point_iou_audit": float(best_row["pooled_iou"]),
            "pooled_full_point_f1_audit": float(best_row["pooled_f1"]),
        },
        "real_evaluation": {
            "files": int(len(real_rows)),
            "metric_space": "original_raw_roi_point_space",
            "subset_summary": summary.to_dict("records"),
        },
        "projection_audit": {
            "all_token_reconstruction_checks_passed": True,
            "mean_candidate_coverage": float(audit["covered_candidate_fraction"].mean()),
            "minimum_candidate_coverage": float(audit["covered_candidate_fraction"].min()),
            "mean_votes_on_covered_candidate": float(
                audit["mean_votes_on_covered"].mean()
            ),
            "maximum_votes": int(audit["max_votes"].max()),
        },
        "projection_support_by_subset": support_summary.to_dict("records"),
        "baseline_comparison_merged": combined is not None,
        "all_method_subset_summary_written": all_method_summary is not None,
        "visualizations": visualization_paths,
        "elapsed_seconds": float(time.time() - started),
    }
    (output_root / "FULL_POINT_PROJECTION_RESULT.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
