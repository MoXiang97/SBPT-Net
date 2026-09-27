#!/usr/bin/env python
"""Final superline organization support for STE-Net under D04-1000.

This is a principle-based reimplementation for our current PyTorch/NumPy/SciPy
environment. It does not run the official TensorFlow SuperLine3D model. The
module generates line-style fragment IDs for source-anchored local-patch
tokenization and STE-Net evidence learning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


DEFAULT_REPO = Path(__file__).resolve().parents[2]
SHARED_LIB = DEFAULT_REPO / "scripts" / "shared_stage2_lib"
TOKEN_ABLATION = SHARED_LIB / "sce_net_connectivity_graph_tokenization.py"


def load_token_ablation_module():
    spec = importlib.util.spec_from_file_location("stage2_tokenization_ablation", str(TOKEN_ABLATION))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import tokenization ablation module: {TOKEN_ABLATION}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


T = load_token_ablation_module()
M = T.M


SUPERLINE3D_STYLE_METHOD = {
    "id": "superline3d_style_line_partition",
    "dir": "11_superline3d_style_line_partition",
    "desc": (
        "Principle-based SuperLine3D-style line organization: local PCA line "
        "descriptors, tangent-consistent kNN graph, descriptor-distance pruning, "
        "and deterministic connected components. This is not official TensorFlow "
        "SuperLine3D inference."
    ),
    "cache_root": DEFAULT_REPO / "outputs" / "FormalCache_D04_1000_SourceAnchoredR6_seed42",
}


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

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _local_line_descriptors(xyz: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(xyz)
    tangent = np.zeros((n, 3), dtype=np.float32)
    desc = np.zeros((n, 8), dtype=np.float32)
    linearness = np.zeros(n, dtype=np.float32)
    if n == 0:
        return tangent, linearness, desc
    tree = cKDTree(xyz)
    kk = min(max(4, int(k)), n)
    dist, ind = tree.query(xyz, k=kk)
    if kk == 1:
        dist = dist.reshape(-1, 1)
        ind = ind.reshape(-1, 1)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for i in range(n):
        nb = np.asarray(ind[i], dtype=np.int64).reshape(-1)
        pts = xyz[nb]
        cen = pts.mean(axis=0, keepdims=True)
        if len(pts) < 3:
            tangent[i] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            continue
        cov = (pts - cen).T @ (pts - cen) / max(len(pts) - 1, 1)
        vals, vecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(vals)[::-1]
        vals = np.maximum(vals[order].astype(np.float32), 0.0)
        vecs = vecs[:, order].astype(np.float32)
        t = vecs[:, 0]
        if t[0] < 0:
            t = -t
        tangent[i] = t / max(float(np.linalg.norm(t)), 1e-8)
        l1, l2, l3 = float(vals[0]), float(vals[1]), float(vals[2])
        den = max(l1, 1e-8)
        lin = (l1 - l2) / den
        pla = (l2 - l3) / den
        sca = l3 / den
        linearness[i] = lin
        dd = np.asarray(dist[i], dtype=np.float32).reshape(-1)
        desc[i] = np.array([
            lin,
            pla,
            sca,
            1.0 - abs(float(np.dot(vecs[:, 2], up))),
            float(np.mean(dd)),
            float(np.std(dd)),
            float(np.mean(np.abs((pts - xyz[i][None, :]) @ tangent[i]))),
            float(np.mean(np.linalg.norm((pts - xyz[i][None, :]) - (((pts - xyz[i][None, :]) @ tangent[i])[:, None] * tangent[i][None, :]), axis=1))),
        ], dtype=np.float32)
    mean = desc.mean(axis=0, keepdims=True)
    std = desc.std(axis=0, keepdims=True) + 1e-6
    return tangent, linearness, ((desc - mean) / std).astype(np.float32)


def superline3d_style_fragments(candidate: Dict[str, np.ndarray], cfg: object) -> Dict[str, object]:
    xyz = np.asarray(candidate["xyz"], dtype=np.float32)
    n = len(xyz)
    if n == 0:
        return T._frag_info_from_ids(candidate, np.zeros(0, dtype=np.int32), 1.0, 1.0)
    base = float(M.estimate_base_spacing(xyz, int(getattr(cfg, "base_spacing_sample_size", 8000))))
    if not np.isfinite(base) or base <= 0:
        base = 1.0

    k_desc = int(getattr(cfg, "sl3d_style_desc_k", 20))
    k_graph = int(getattr(cfg, "sl3d_style_graph_k", 20))
    min_linearness = float(getattr(cfg, "sl3d_style_min_linearness", 0.18))
    min_tangent_sim = float(getattr(cfg, "sl3d_style_min_tangent_sim", 0.72))
    max_desc_dist = float(getattr(cfg, "sl3d_style_max_desc_dist", 3.2))
    max_spatial_factor = float(getattr(cfg, "sl3d_style_max_spatial_factor", 8.0))
    max_component_points = int(getattr(cfg, "sl3d_style_max_component_points", 4096))

    tangent, linearness, desc = _local_line_descriptors(xyz, k_desc)
    tree = cKDTree(xyz)
    kk = min(max(2, k_graph + 1), n)
    dist, ind = tree.query(xyz, k=kk)
    if kk == 1:
        dist = dist.reshape(-1, 1)
        ind = ind.reshape(-1, 1)

    uf = UnionFind(n)
    max_spatial = max_spatial_factor * base
    edges = []
    for i in range(n):
        for d, j in zip(np.asarray(dist[i]).reshape(-1)[1:], np.asarray(ind[i]).reshape(-1)[1:]):
            j = int(j)
            if j <= i:
                continue
            if float(d) > max_spatial:
                continue
            if min(float(linearness[i]), float(linearness[j])) < min_linearness:
                continue
            sim = abs(float(np.dot(tangent[i], tangent[j])))
            if sim < min_tangent_sim:
                continue
            rel = xyz[j] - xyz[i]
            # SuperLine-style criterion: neighbors should lie along at least one
            # local line direction, but no task-specific transverse/top-k cost is used.
            along_i = abs(float(np.dot(rel, tangent[i])))
            along_j = abs(float(np.dot(rel, tangent[j])))
            perp_i = float(np.linalg.norm(rel - along_i * tangent[i]))
            perp_j = float(np.linalg.norm(rel - along_j * tangent[j]))
            perp = min(perp_i, perp_j)
            desc_dist = float(np.linalg.norm(desc[i] - desc[j]))
            if desc_dist > max_desc_dist:
                continue
            cost = desc_dist + 0.25 * (1.0 - sim) + 0.08 * float(d / max(base, 1e-6)) + 0.15 * float(perp / max(base, 1e-6))
            edges.append((cost, i, j))
    edges.sort(key=lambda x: x[0])
    for _, i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if int(uf.size[ri] + uf.size[rj]) > max_component_points:
            continue
        uf.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    min_pts = int(getattr(cfg, "min_fragment_points", 3))
    frag_ids = np.full(n, -1, dtype=np.int32)
    fid = 0
    for root in sorted(groups):
        idx = groups[root]
        if len(idx) >= min_pts:
            frag_ids[np.asarray(idx, dtype=np.int64)] = fid
            fid += 1
    return T._frag_info_from_ids(candidate, frag_ids, base, max_spatial)


def token_cache_path(method: Dict[str, object], subset: str, sample_id: str) -> Path:
    return Path(method["cache_root"]) / subset / f"{sample_id}_downsampled.npz"


def build_or_load_downsampled(path: str, subset: str, method: Dict[str, object], cfg: object) -> Dict[str, np.ndarray]:
    sid = M.sample_id(path)
    cache = token_cache_path(method, subset, sid)
    T.ensure_dir(cache.parent)
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
        except Exception:
            cache.unlink(missing_ok=True)
    candidate = M.load_candidate_npy(path)
    frag = superline3d_style_fragments(candidate, cfg)
    ds = M.downsample_superlines_to_line_points(candidate, frag, cfg)
    np.savez_compressed(cache, **ds)
    return ds

# Historical staged runner removed; this module now provides final STE-Net support functions only.
