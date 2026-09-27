#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence-level tokenization ablation for Stage2 V19 Final V1.

This script intentionally does not run V19.6c polar closure or fixed-width
corridor inference. It compares structural tokenization choices using the same
local-patch encoder, selected token attributes, contextual refiner, and
synthetic-validation threshold protocol.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]
MAIN_SCRIPT = SCRIPT_DIR / "sce_net_connectivity_graph_core.py"


def import_main():
    spec = importlib.util.spec_from_file_location("stage2_v19_final_v1", str(MAIN_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import main script: {MAIN_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = import_main()


METHODS = [
    {
        "id": "simple_point_stride",
        "dir": "00_simple_point_stride",
        "desc": "Simple source-anchored point/stride token baseline; no local-patch medoid and no superline/superpoint context.",
        "cache_root": BASE_DIR / "outputs" / "Stage2_SimplePointStrideTokenCache_D04_1000",
    },
    {
        "id": "plain_patch",
        "dir": "01_plain_patch",
        "desc": "Plain spatial local-patch medoid tokenization directly on Stage1 candidates; no superline/superpoint context.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_plain_patch_cache_D04_1000",
    },
    {
        "id": "superpoint",
        "dir": "02_superpoint",
        "desc": "Ordinary voxel-like spatial superpoint/component grouping on Stage1 candidates; no direction consistency or line prior.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_superpoint_cache_D04_1000",
    },
    {
        "id": "isotropic_r2_superline",
        "dir": "03a_isotropic_r2_superline",
        "desc": "Diagnostic isotropic r2 superline baseline, edge(i,j)=distance<=2*base_spacing.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_isotropic_r2_superline_cache_D04_1000",
        "strict_radius_factor": 2.0,
    },
    {
        "id": "isotropic_r4_superline",
        "dir": "03b_isotropic_r4_superline",
        "desc": "Diagnostic isotropic r4 superline baseline, edge(i,j)=distance<=4*base_spacing.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_isotropic_r4_superline_cache_D04_1000",
        "strict_radius_factor": 4.0,
    },
    {
        "id": "isotropic_r6_superline",
        "dir": "03_isotropic_r6_superline",
        "desc": "Current isotropic r6 superline baseline, edge(i,j)=distance<=6*base_spacing.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_isotropic_superline_cache_D04_1000",
        "strict_radius_factor": 6.0,
    },
    {
        "id": "isotropic_r8_superline",
        "dir": "03c_isotropic_r8_superline",
        "desc": "Diagnostic isotropic r8 superline baseline, edge(i,j)=distance<=8*base_spacing.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_isotropic_r8_superline_cache_D04_1000",
        "strict_radius_factor": 8.0,
    },
    {
        "id": "isotropic_r10_superline",
        "dir": "03d_isotropic_r10_superline",
        "desc": "Diagnostic isotropic r10 superline baseline, edge(i,j)=distance<=10*base_spacing.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_isotropic_r10_superline_cache_D04_1000",
        "strict_radius_factor": 10.0,
    },
    {
        "id": "isotropic_r12_superline",
        "dir": "03e_isotropic_r12_superline",
        "desc": "Diagnostic isotropic r12 superline baseline, edge(i,j)=distance<=12*base_spacing.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_isotropic_r12_superline_cache_D04_1000",
        "strict_radius_factor": 12.0,
    },
    {
        "id": "direction_consistent_r2_superline",
        "dir": "05a_direction_consistent_r2_superline",
        "desc": "Diagnostic radius-matched direction-consistent superline; uses 2*base_spacing search radius with transverse and tangent constraints.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_direction_consistent_r2_superline_cache_D04_1000",
        "direction_max_radius_factor": 2.0,
    },
    {
        "id": "direction_consistent_r4_superline",
        "dir": "05b_direction_consistent_r4_superline",
        "desc": "Diagnostic radius-matched direction-consistent superline; uses 4*base_spacing search radius with transverse and tangent constraints.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_direction_consistent_r4_superline_cache_D04_1000",
        "direction_max_radius_factor": 4.0,
    },
    {
        "id": "direction_consistent_superline",
        "dir": "04_direction_consistent_superline",
        "desc": "Direction-consistent superline with local PCA tangent, transverse suppression, tangent similarity, and top-k low-cost edges.",
        "cache_root": BASE_DIR / "outputs" / "bost_superline_cache_D04_1000",
        "direction_max_radius_factor": 8.0,
    },
    {
        "id": "direction_consistent_r6_superline",
        "dir": "05_direction_consistent_r6_superline",
        "desc": "Radius-matched direction-consistent superline; uses the same 6*base_spacing search radius as isotropic_r6_superline, with transverse and tangent constraints.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_direction_consistent_r6_superline_cache_D04_1000",
        "direction_max_radius_factor": 6.0,
    },
    {
        "id": "direction_consistent_r10_superline",
        "dir": "05c_direction_consistent_r10_superline",
        "desc": "Diagnostic radius-matched direction-consistent superline; uses 10*base_spacing search radius with transverse and tangent constraints.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_direction_consistent_r10_superline_cache_D04_1000",
        "direction_max_radius_factor": 10.0,
    },
    {
        "id": "direction_consistent_r12_superline",
        "dir": "05d_direction_consistent_r12_superline",
        "desc": "Diagnostic radius-matched direction-consistent superline; uses 12*base_spacing search radius with transverse and tangent constraints.",
        "cache_root": BASE_DIR / "outputs" / "tokenization_direction_consistent_r12_superline_cache_D04_1000",
        "direction_max_radius_factor": 12.0,
    },
]


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        p = int(self.parent[x])
        while p != int(self.parent[p]):
            self.parent[p] = self.parent[self.parent[p]]
            p = int(self.parent[p])
        return p

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _frag_info_from_ids(candidate: Dict[str, np.ndarray], frag_ids: np.ndarray, base_spacing: float, radius: float) -> Dict[str, object]:
    xyz = candidate["xyz"].astype(np.float32)
    labels = candidate["label"].astype(np.int64)
    valid = sorted([int(x) for x in np.unique(frag_ids) if int(x) >= 0])
    centroids, tangents, lengths, widths, linearnesses, frag_sizes, pos_ratios = [], [], [], [], [], [], []
    for fid in valid:
        idx = np.where(frag_ids == fid)[0]
        pts = xyz[idx]
        st = M.pca_fragment_stats(pts)
        centroids.append(st["centroid"])
        tangents.append(st["tangent"])
        lengths.append(float(st["length"]))
        widths.append(float(st["width"]))
        linearnesses.append(float(st["linearness"]))
        frag_sizes.append(int(len(idx)))
        pos_ratios.append(float(labels[idx].mean()) if len(idx) else 0.0)
    return {
        "frag_ids": frag_ids.astype(np.int32),
        "valid_ids": np.asarray(valid, dtype=np.int32),
        "frag_sizes": np.asarray(frag_sizes, dtype=np.int32),
        "pos_ratios": np.asarray(pos_ratios, dtype=np.float32),
        "centroids": np.vstack(centroids).astype(np.float32) if centroids else np.zeros((0, 3), dtype=np.float32),
        "tangents": np.vstack(tangents).astype(np.float32) if tangents else np.zeros((0, 3), dtype=np.float32),
        "lengths": np.asarray(lengths, dtype=np.float32),
        "widths": np.asarray(widths, dtype=np.float32),
        "linearnesses": np.asarray(linearnesses, dtype=np.float32),
        "base_spacing": float(base_spacing),
        "radius": float(radius),
    }


def ordinary_superpoint_fragments(candidate: Dict[str, np.ndarray], cfg: object) -> Dict[str, object]:
    """Voxel-like spatial component baseline, deliberately not line-aware."""
    xyz = candidate["xyz"].astype(np.float32)
    n = len(xyz)
    if n == 0:
        return _frag_info_from_ids(candidate, np.zeros((0,), dtype=np.int32), 1.0, 1.0)
    base = float(M.estimate_base_spacing(xyz, int(getattr(cfg, "base_spacing_sample_size", 8000))))
    if not np.isfinite(base) or base <= 0:
        base = 1.0
    voxel = max(float(getattr(cfg, "superpoint_voxel_factor", 6.0)) * base, base)
    coords = np.floor((xyz - xyz.min(axis=0, keepdims=True)) / voxel).astype(np.int64)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i, key in enumerate(map(tuple, coords)):
        buckets.setdefault(key, []).append(i)
    frag_ids = np.full(n, -1, dtype=np.int32)
    fid = 0
    min_pts = int(getattr(cfg, "min_fragment_points", 3))
    for key in sorted(buckets):
        idx = buckets[key]
        if len(idx) >= min_pts:
            frag_ids[np.asarray(idx, dtype=np.int64)] = fid
            fid += 1
    return _frag_info_from_ids(candidate, frag_ids, base, voxel)


def estimate_local_tangents(xyz: np.ndarray, k: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    n = len(xyz)
    tangents = np.zeros((n, 3), dtype=np.float32)
    linearness = np.zeros(n, dtype=np.float32)
    if n == 0:
        return tangents, linearness
    tree = cKDTree(xyz)
    kk = min(max(3, int(k)), n)
    d, ind = tree.query(xyz, k=kk)
    if kk == 1:
        ind = ind.reshape(-1, 1)
    for i in range(n):
        pts = xyz[np.asarray(ind[i], dtype=np.int64)]
        st = M.pca_fragment_stats(pts)
        t = np.asarray(st["tangent"], dtype=np.float32)
        norm = float(np.linalg.norm(t))
        tangents[i] = t / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        eig = np.asarray(st["eigvals"], dtype=np.float32)
        if len(eig) >= 2:
            linearness[i] = float((eig[0] - eig[1]) / (eig[0] + 1e-8))
    return tangents, linearness


def direction_consistent_fragments(candidate: Dict[str, np.ndarray], cfg: object) -> Dict[str, object]:
    xyz = candidate["xyz"].astype(np.float32)
    n = len(xyz)
    if n == 0:
        return _frag_info_from_ids(candidate, np.zeros((0,), dtype=np.int32), 1.0, 1.0)
    base = float(M.estimate_base_spacing(xyz, int(getattr(cfg, "base_spacing_sample_size", 8000))))
    if not np.isfinite(base) or base <= 0:
        base = 1.0
    k_tangent = int(getattr(cfg, "direction_tangent_k", 16))
    max_radius = float(getattr(cfg, "direction_max_radius_factor", 8.0)) * base
    max_perp = float(getattr(cfg, "direction_max_perp_factor", 2.5)) * base
    min_sim = float(getattr(cfg, "direction_min_tangent_sim", 0.5))
    top_k = int(getattr(cfg, "direction_top_k_edges", 3))
    candidate_k = int(getattr(cfg, "direction_edge_candidate_k", 32))
    candidate_k = max(candidate_k, top_k + 1)
    tangents, _ = estimate_local_tangents(xyz, k=k_tangent)
    tree = cKDTree(xyz)
    uf = UnionFind(n)
    query_k = min(n, candidate_k + 1)
    for i in range(n):
        _, nbrs = tree.query(xyz[i], k=query_k, distance_upper_bound=max_radius)
        nbrs = np.atleast_1d(nbrs)
        candidates = []
        ti = tangents[i]
        for j in nbrs:
            j = int(j)
            if j == i or j >= n:
                continue
            d = xyz[j] - xyz[i]
            dist = float(np.linalg.norm(d))
            if dist <= 1e-8 or dist > max_radius:
                continue
            parallel = float(abs(np.dot(d, ti)))
            perp_vec = d - float(np.dot(d, ti)) * ti
            perp = float(np.linalg.norm(perp_vec))
            sim = float(abs(np.dot(ti, tangents[j])))
            if perp > max_perp or sim < min_sim:
                continue
            cost = 1.0 * dist + 2.0 * perp + 3.0 * (1.0 - sim)
            candidates.append((cost, j))
        candidates.sort(key=lambda x: x[0])
        for _, j in candidates[:top_k]:
            uf.union(i, j)
    comp: Dict[int, List[int]] = {}
    for i in range(n):
        comp.setdefault(uf.find(i), []).append(i)
    frag_ids = np.full(n, -1, dtype=np.int32)
    fid = 0
    min_pts = int(getattr(cfg, "min_fragment_points", 3))
    for root in sorted(comp):
        idx = comp[root]
        if len(idx) >= min_pts:
            frag_ids[np.asarray(idx, dtype=np.int64)] = fid
            fid += 1
    return _frag_info_from_ids(candidate, frag_ids, base, max_radius)


def token_cache_path(method: Dict[str, object], subset: str, sample_id: str) -> Path:
    return Path(method["cache_root"]) / subset / f"{sample_id}_downsampled.npz"


def build_or_load_method_downsampled(path: str, subset: str, method: Dict[str, object], cfg: object) -> Dict[str, np.ndarray]:
    sid = M.sample_id(path)
    cache = token_cache_path(method, subset, sid)
    ensure_dir(cache.parent)
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=True) as data:
                n = len(data["label"])
                return {
                    "xyz": data["xyz"], "rgb": data["rgb"], "label": data["label"], "label_ratio": data["label_ratio"],
                    "lcc": data["lcc"], "orig_index": data["orig_index"], "fragment_id": data["fragment_id"],
                    "features": data["features"], "bin_num_points": data["bin_num_points"],
                    "rep_anchor_dist": data["rep_anchor_dist"] if "rep_anchor_dist" in data.files else np.zeros(n, dtype=np.float32),
                    "patch_points": data["patch_points"] if "patch_points" in data.files else np.zeros((n, int(cfg.patch_point_count), int(cfg.patch_point_dim)), dtype=np.float32),
                    "patch_mask": data["patch_mask"] if "patch_mask" in data.files else np.zeros((n, int(cfg.patch_point_count)), dtype=np.float32),
                }
        except Exception as e:
            print(f"[WARN] broken tokenization cache removed and regenerated: {cache} ({type(e).__name__}: {e})")
            try:
                cache.unlink()
            except FileNotFoundError:
                pass
    candidate = M.load_candidate_npy(path)
    mid = str(method["id"])
    if mid == "simple_point_stride":
        ds = M.downsample_candidate_simple_point_stride(candidate, cfg)
    elif mid == "plain_patch":
        ds = M.downsample_candidate_spatial_patches_no_superline(candidate, cfg)
    elif mid == "superpoint":
        frag = ordinary_superpoint_fragments(candidate, cfg)
        ds = M.downsample_superlines_to_line_points(candidate, frag, cfg)
    elif mid.startswith("isotropic_r") and mid.endswith("_superline"):
        old_radius = getattr(cfg, "strict_radius_factor", None)
        if "strict_radius_factor" in method:
            setattr(cfg, "strict_radius_factor", float(method["strict_radius_factor"]))
        frag = M.extract_fragments_and_features(candidate, cfg)
        if old_radius is None and hasattr(cfg, "strict_radius_factor"):
            delattr(cfg, "strict_radius_factor")
        elif old_radius is not None:
            setattr(cfg, "strict_radius_factor", old_radius)
        ds = M.downsample_superlines_to_line_points(candidate, frag, cfg)
    elif mid.startswith("direction_consistent") and mid.endswith("_superline"):
        old_direction_radius = getattr(cfg, "direction_max_radius_factor", None)
        if "direction_max_radius_factor" in method:
            setattr(cfg, "direction_max_radius_factor", float(method["direction_max_radius_factor"]))
        frag = direction_consistent_fragments(candidate, cfg)
        if old_direction_radius is None and hasattr(cfg, "direction_max_radius_factor"):
            delattr(cfg, "direction_max_radius_factor")
        elif old_direction_radius is not None:
            setattr(cfg, "direction_max_radius_factor", old_direction_radius)
        ds = M.downsample_superlines_to_line_points(candidate, frag, cfg)
    else:
        raise ValueError(mid)
    np.savez_compressed(cache, **ds)
    return ds


def choose_protocol(cfg: object, debug_limit_files: int = 0) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    cand = M.collect_candidate_files(cfg)
    synth = cand.get(cfg.sim_mode, [])
    total = len(synth)
    requested_train = int(getattr(cfg, "max_sim_train", 0) or 0)
    requested_val = int(getattr(cfg, "max_sim_val", 0) or 0)
    if requested_train > 0 and requested_val > 0 and requested_train + requested_val <= total:
        train_n, val_n = requested_train, requested_val
    elif total == 1000:
        train_n, val_n = 800, 200
    else:
        train_n, val_n = max(0, total - min(100, total)), min(100, total)
    if debug_limit_files > 0:
        train_n = min(train_n, max(1, debug_limit_files))
        val_n = min(val_n, max(1, min(10, debug_limit_files)))
    cfg.max_sim_train = int(train_n)
    cfg.max_sim_val = int(val_n)
    train_files, val_files = M.split_synthetic_train_val(synth, cfg)
    rows = []
    for f in train_files:
        rows.append({"domain": "synthetic", "subset": cfg.sim_mode, "split": "train", "path": f, "sample_id": M.sample_id(f)})
    for f in val_files:
        rows.append({"domain": "synthetic", "subset": cfg.sim_mode, "split": "val", "path": f, "sample_id": M.sample_id(f)})
    for subset in ["shoe_a", "shoe_b", "shoe_c"]:
        files = cand.get(subset, [])
        if debug_limit_files > 0:
            files = files[:debug_limit_files]
        for f in files:
            rows.append({"domain": "real", "subset": subset, "split": "test", "path": f, "sample_id": M.sample_id(f)})
    info = {
        "synthetic_total": total,
        "train_n": len(train_files),
        "val_n": len(val_files),
        "debug_limit_files": int(debug_limit_files),
        "synthetic_split_policy": str(getattr(cfg, "synthetic_split_policy", "style_holdout")),
        "synthetic_holdout_styles": str(getattr(cfg, "synthetic_holdout_styles", "")),
    }
    return rows, info


def geometry_rows(records: List[Dict[str, object]], method_id: str) -> pd.DataFrame:
    rows = []
    for r in records:
        ds = r["downsampled"]
        xyz = np.asarray(ds["xyz"], dtype=np.float32)
        y = np.asarray(ds["label"], dtype=np.int64)
        frag = np.asarray(ds["fragment_id"], dtype=np.int64)
        fids = [int(x) for x in np.unique(frag) if int(x) >= 0]
        pos_ratios, widths, lengths, lines, sizes = [], [], [], [], []
        mixed = 0
        neg_in_pos = []
        for fid in fids:
            idx = np.where(frag == fid)[0]
            if len(idx) == 0:
                continue
            yy = y[idx]
            pr = float(np.mean(yy)) if len(yy) else 0.0
            st = M.pca_fragment_stats(xyz[idx])
            pos_ratios.append(pr)
            widths.append(float(st["width"]))
            lengths.append(float(st["length"]))
            lines.append(float(st["linearness"]))
            sizes.append(int(len(idx)))
            if pr > 0.0 and pr < 1.0:
                mixed += 1
            if pr > 0.0:
                neg_in_pos.append(1.0 - pr)
        rows.append({
            "method": method_id, **{k: r[k] for k in ["domain", "split", "subset", "sample_id", "path"]},
            "fragment_count": len(fids),
            "token_count": int(len(y)),
            "positive_token_ratio": float(np.mean(y)) if len(y) else 0.0,
            "mean_fragment_points": float(np.mean(sizes)) if sizes else 0.0,
            "mean_fragment_length": float(np.mean(lengths)) if lengths else 0.0,
            "mean_fragment_width": float(np.mean(widths)) if widths else 0.0,
            "mean_fragment_linearness": float(np.mean(lines)) if lines else 0.0,
            "mean_fragment_positive_ratio": float(np.mean(pos_ratios)) if pos_ratios else 0.0,
            "mixed_fragment_rate": float(mixed / max(len(fids), 1)),
            "fragment_purity": float(np.mean([max(p, 1.0 - p) for p in pos_ratios])) if pos_ratios else 0.0,
            "gt_coverage_by_fragments": float(np.sum(y[frag >= 0]) / max(np.sum(y), 1)) if len(y) else 0.0,
            "interference_inclusion_proxy": float(np.mean(neg_in_pos)) if neg_in_pos else 0.0,
        })
    return pd.DataFrame(rows)


def export_method_vis(records: List[Dict[str, object]], method_dir: Path, method_id: str, prob_key: str, threshold: float) -> None:
    root = method_dir / "paper_vis" / method_id
    counters: Dict[str, int] = {}
    for r in records:
        if r["domain"] != "real":
            continue
        subset = str(r["subset"])
        if counters.get(subset, 0) >= 3:
            continue
        counters[subset] = counters.get(subset, 0) + 1
        ds = r["downsampled"]
        xyz = np.asarray(ds["xyz"], dtype=np.float32)
        y = np.asarray(ds["label"], dtype=np.int32)
        frag = np.asarray(ds["fragment_id"], dtype=np.int32)
        prob = np.asarray(r.get(prob_key, np.zeros(len(y))), dtype=np.float32)
        pred = (prob >= float(threshold)).astype(np.int32)
        sid = str(r["sample_id"])
        M._write_xyz_rgb_label(str(root / "00_token_or_fragment_id" / f"{subset}_{sid}.txt"), xyz, M._simple_fragid_rgb(frag), frag)
        M._write_xyz_rgb_label(str(root / "01_gt_red_blue" / f"{subset}_{sid}.txt"), xyz, M._simple_rgb_red_blue(y.astype(bool)), y)
        M._write_xyz_rgb_label(str(root / "02_cluster_refiner_pred_red_blue" / f"{subset}_{sid}.txt"), xyz, M._simple_rgb_red_blue(pred.astype(bool)), pred)
        err_rgb, err_label = M._simple_error_rgb_label(y, pred)
        M._write_xyz_rgb_label(str(root / "03_error_tp_fp_fn" / f"{subset}_{sid}.txt"), xyz, err_rgb, err_label)
        # Mixed-fragment debug: label 1 for fragments that contain both positive and negative tokens.
        mixed_frag = np.zeros(len(y), dtype=np.int32)
        for fid in [int(x) for x in np.unique(frag) if int(x) >= 0]:
            idx = np.where(frag == fid)[0]
            if len(idx) and np.any(y[idx] == 1) and np.any(y[idx] == 0):
                mixed_frag[idx] = 1
        M._write_xyz_rgb_label(str(root / "04_mixed_fragment_debug" / f"{subset}_{sid}.txt"), xyz, M._simple_rgb_red_blue(mixed_frag.astype(bool)), mixed_frag)
    export_method_heatmap_overlays(records, method_dir, method_id, threshold)


def _raw_txt_path_for_record(r: Dict[str, object]) -> Path | None:
    sid = str(r["sample_id"])
    subset = str(r["subset"])
    if str(r["domain"]) == "synthetic":
        path = BASE_DIR / "assets" / "STS2R_Synthetic" / f"{sid}.txt"
        return path if path.exists() else None
    mapping = {
        "shoe_a": BASE_DIR / "assets" / "Real_Data" / "Real_ShoeA" / f"{sid}.txt",
        "shoe_b": BASE_DIR / "assets" / "Real_Data" / "Real_ShoeB" / f"{sid}.txt",
        "shoe_c": BASE_DIR / "assets" / "Real_Data" / "Real_ShoeC" / f"{sid}.txt",
    }
    path = mapping.get(subset)
    return path if path is not None and path.exists() else None


def _load_raw_pointcloud_for_record(r: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = _raw_txt_path_for_record(r)
    if path is None:
        return None
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 7:
        raise ValueError(f"Raw point cloud must have at least 7 columns: {path}")
    xyz = arr[:, :3].astype(np.float32)
    rgb = np.clip(arr[:, 3:6], 0, 255).astype(np.uint8)
    label = arr[:, 6].astype(np.int32)
    return xyz, rgb, label


def _prob_to_heat_rgb(prob: np.ndarray) -> np.ndarray:
    """Simple blue-yellow-red heat color for CloudCompare-friendly TXT export."""
    p = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
    rgb = np.zeros((len(p), 3), dtype=np.uint8)
    lo = p < 0.5
    hi = ~lo
    t = np.zeros_like(p)
    t[lo] = p[lo] / 0.5
    rgb[lo, 0] = (40 + 215 * t[lo]).astype(np.uint8)
    rgb[lo, 1] = (110 + 110 * t[lo]).astype(np.uint8)
    rgb[lo, 2] = (210 - 160 * t[lo]).astype(np.uint8)
    t[hi] = (p[hi] - 0.5) / 0.5
    rgb[hi, 0] = 255
    rgb[hi, 1] = (220 - 190 * t[hi]).astype(np.uint8)
    rgb[hi, 2] = (50 - 40 * t[hi]).astype(np.uint8)
    return rgb


def _write_full_cloud_heatmap_txt(
    out_path: Path,
    raw_xyz: np.ndarray,
    raw_rgb: np.ndarray,
    raw_label: np.ndarray,
    overlay_index: np.ndarray,
    overlay_score: np.ndarray,
    threshold: float,
) -> None:
    ensure_dir(out_path.parent)
    n = len(raw_xyz)
    heat = np.full(n, -1.0, dtype=np.float32)
    source = np.zeros(n, dtype=np.int32)
    pred = np.zeros(n, dtype=np.int32)
    valid = (overlay_index >= 0) & (overlay_index < n)
    idx = overlay_index[valid].astype(np.int64)
    score = np.clip(overlay_score[valid].astype(np.float32), 0.0, 1.0)
    # Keep the strongest token/evidence value if multiple tokens map to the same source point.
    np.maximum.at(heat, idx, score)
    source[heat >= 0.0] = 1
    pred[heat >= float(threshold)] = 1

    vis_rgb = np.full((n, 3), 190, dtype=np.uint8)
    mapped = heat >= 0.0
    vis_rgb[mapped] = _prob_to_heat_rgb(heat[mapped])
    # Ground-truth boundary remains visible even when not selected as a token.
    gt_only = (raw_label > 0) & (~mapped)
    vis_rgb[gt_only] = np.array([255, 80, 80], dtype=np.uint8)

    out = np.column_stack([
        raw_xyz.astype(np.float32),
        vis_rgb.astype(np.float32),
        raw_label.astype(np.float32),
        heat.astype(np.float32),
        pred.astype(np.float32),
        source.astype(np.float32),
        raw_rgb.astype(np.float32),
    ])
    np.savetxt(
        out_path,
        out,
        fmt="%.6f %.6f %.6f %.0f %.0f %.0f %.0f %.6f %.0f %.0f %.0f %.0f %.0f",
    )


def _stage1_candidate_overlay_from_record(r: Dict[str, object], raw_len: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(str(r["path"]))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] >= 9:
        orig = arr[:, 7].astype(np.int64)
        score = arr[:, 8].astype(np.float32)
        if len(score):
            mn, mx = float(np.nanmin(score)), float(np.nanmax(score))
            score = (score - mn) / max(mx - mn, 1e-8)
        return orig, np.clip(score, 0.0, 1.0)
    if arr.shape[1] >= 8:
        return arr[:, 7].astype(np.int64), np.ones(len(arr), dtype=np.float32)
    return np.arange(min(len(arr), raw_len), dtype=np.int64), np.ones(min(len(arr), raw_len), dtype=np.float32)


def _export_one_overlay_stage(
    r: Dict[str, object],
    out_path: Path,
    score: np.ndarray,
    threshold: float,
    use_stage1_candidates: bool = False,
) -> None:
    raw = _load_raw_pointcloud_for_record(r)
    if raw is None:
        return
    raw_xyz, raw_rgb, raw_label = raw
    if use_stage1_candidates:
        orig, heat = _stage1_candidate_overlay_from_record(r, len(raw_xyz))
    else:
        ds = r["downsampled"]
        orig = np.asarray(ds["orig_index"], dtype=np.int64)
        heat = np.asarray(score, dtype=np.float32)
    _write_full_cloud_heatmap_txt(out_path, raw_xyz, raw_rgb, raw_label, orig, heat, threshold)


def export_method_heatmap_overlays(records: List[Dict[str, object]], method_dir: Path, method_id: str, threshold: float) -> None:
    """Export CloudCompare-ready full-point-cloud heatmap overlays.

    The learned evidence is token/candidate-level.  For visualization only, it is
    mapped back to the original complete point cloud through orig_index.  Raw
    background points are gray, ground-truth-only boundary points are light red,
    and mapped token/candidate points are heat-colored by probability/evidence.
    """
    root = method_dir / "paper_heatmap_txt" / method_id
    columns = (
        "Columns: X Y Z R G B GT HeatScore Pred SourceFlag OrigR OrigG OrigB\n"
        "R/G/B are visualization colors. HeatScore=-1 means background/no token evidence.\n"
        "SourceFlag=1 means the row is mapped from a candidate/token; 0 means background.\n"
        "These files are for CloudCompare visualization, not for metric computation.\n"
    )
    ensure_dir(root)
    (root / "columns.txt").write_text(columns, encoding="utf-8")

    # Full real final-evidence heatmaps.  This keeps all real samples available
    # while avoiding a combinatorial explosion of progressive visualizations.
    for r in records:
        if r["domain"] != "real":
            continue
        subset = str(r["subset"])
        sid = str(r["sample_id"])
        prob = np.asarray(r.get("cluster_refiner_prob", np.zeros(len(r["downsampled"]["label"]))), dtype=np.float32)
        _export_one_overlay_stage(
            r,
            root / "all_real_cluster_refiner_heatmap" / subset / f"{sid}.txt",
            prob,
            threshold,
        )

    # Key progressive examples: one synthetic-val sample and one sample from
    # each real subset.  They show how evidence gets cleaner across Stage 2.
    counters: Dict[str, int] = {}
    key_records = []
    for r in records:
        if r["domain"] == "synthetic" and r["split"] == "val" and counters.get("synthetic_val", 0) < 1:
            key_records.append(r)
            counters["synthetic_val"] = counters.get("synthetic_val", 0) + 1
        elif r["domain"] == "real" and counters.get(str(r["subset"]), 0) < 1:
            key_records.append(r)
            counters[str(r["subset"])] = counters.get(str(r["subset"]), 0) + 1

    for r in key_records:
        subset = str(r["subset"])
        sid = str(r["sample_id"])
        prefix = f"{subset}_{sid}"
        _export_one_overlay_stage(
            r,
            root / "key_progressive_examples" / "00_stage1_candidates" / f"{prefix}.txt",
            np.zeros(0, dtype=np.float32),
            0.5,
            use_stage1_candidates=True,
        )
        for stage_name, key in [
            ("01_raw_patch_evidence", "prob"),
            ("02_token_transformer_evidence", "token_transformer_prob"),
            ("03_cluster_refiner_evidence", "cluster_refiner_prob"),
        ]:
            prob = np.asarray(r.get(key, np.zeros(len(r["downsampled"]["label"]))), dtype=np.float32)
            _export_one_overlay_stage(
                r,
                root / "key_progressive_examples" / stage_name / f"{prefix}.txt",
                prob,
                threshold,
            )

# Historical staged runner removed; this module now provides final tokenization and protocol helpers only.
