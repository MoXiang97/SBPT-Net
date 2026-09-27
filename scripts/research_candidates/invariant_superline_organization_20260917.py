"""Index-invariant superline construction for SBPT-Net."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from scripts.shared_stage2_lib import stenet_superline_organization as Legacy


T = Legacy.T
M = Legacy.M
UnionFind = Legacy.UnionFind
_local_line_descriptors = Legacy._local_line_descriptors


DEFAULT_EXPERIMENT_ROOT = (
    Legacy.DEFAULT_REPO
    / "experiments"
    / "SBPT_Net"
)
SUPERLINE3D_STYLE_METHOD = {
    "id": "invariant_tangent_residual_superline_20260917",
    "dir": "invariant_tangent_residual_superline_20260917",
    "desc": (
        "Union of directed kNN arcs and a signed "
        "orthogonal tangent projection residual."
    ),
    "cache_root": DEFAULT_EXPERIMENT_ROOT / "token_cache",
}


def orthogonal_tangent_residual(
    displacement: np.ndarray,
    tangent: np.ndarray,
) -> float:
    """Return the norm left after orthogonally projecting onto ``tangent``."""

    displacement = np.asarray(displacement, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 0.0:
        return float(np.linalg.norm(displacement))
    unit_tangent = tangent / tangent_norm
    residual = displacement - float(np.dot(displacement, unit_tangent)) * unit_tangent
    return float(np.linalg.norm(residual))


def _endpoint_coordinate_key(xyz: np.ndarray, i: int, j: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    endpoints = (tuple(float(value) for value in xyz[i]), tuple(float(value) for value in xyz[j]))
    return tuple(sorted(endpoints))  # type: ignore[return-value]


def undirected_candidate_edges(
    xyz: np.ndarray,
    tangent: np.ndarray,
    linearness: np.ndarray,
    desc: np.ndarray,
    base: float,
    cfg: object,
) -> list[tuple[float, int, int]]:
    """Build the union of directed kNN arcs as deterministic undirected edges."""

    xyz = np.asarray(xyz, dtype=np.float32)
    n = len(xyz)
    if n < 2:
        return []

    k_graph = int(getattr(cfg, "sl3d_style_graph_k", 20))
    min_linearness = float(getattr(cfg, "sl3d_style_min_linearness", 0.18))
    min_tangent_sim = float(getattr(cfg, "sl3d_style_min_tangent_sim", 0.72))
    max_desc_dist = float(getattr(cfg, "sl3d_style_max_desc_dist", 3.2))
    max_spatial_factor = float(getattr(cfg, "sl3d_style_max_spatial_factor", 8.0))
    max_spatial = max_spatial_factor * base

    tree = cKDTree(xyz)
    kk = min(max(2, k_graph + 1), n)
    boundary_distance, _boundary_index = tree.query(xyz, k=kk)
    if kk == 1:
        boundary_distance = boundary_distance.reshape(-1, 1)

    unordered_pairs: set[tuple[int, int]] = set()
    for i in range(n):
        boundary = float(np.max(np.asarray(boundary_distance[i]).reshape(-1)))
        tolerance = max(1e-12, abs(boundary) * 1e-7)
        candidates = [int(j) for j in tree.query_ball_point(xyz[i], boundary + tolerance) if int(j) != i]
        candidates.sort(
            key=lambda j: (
                float(np.dot(xyz[j] - xyz[i], xyz[j] - xyz[i])),
                tuple(float(value) for value in xyz[j]),
            )
        )
        for j in candidates[: min(k_graph, n - 1)]:
            unordered_pairs.add((min(i, j), max(i, j)))

    edges: list[tuple[float, int, int]] = []
    for i, j in unordered_pairs:
        displacement = xyz[j] - xyz[i]
        distance = float(np.linalg.norm(displacement))
        if distance > max_spatial:
            continue
        if min(float(linearness[i]), float(linearness[j])) < min_linearness:
            continue
        tangent_similarity = abs(float(np.dot(tangent[i], tangent[j])))
        if tangent_similarity < min_tangent_sim:
            continue
        descriptor_distance = float(np.linalg.norm(desc[i] - desc[j]))
        if descriptor_distance > max_desc_dist:
            continue
        tangent_residual = min(
            orthogonal_tangent_residual(displacement, tangent[i]),
            orthogonal_tangent_residual(displacement, tangent[j]),
        )
        cost = (
            descriptor_distance
            + 0.25 * (1.0 - tangent_similarity)
            + 0.08 * distance / max(base, 1e-6)
            + 0.15 * tangent_residual / max(base, 1e-6)
        )
        edges.append((cost, i, j))

    edges.sort(key=lambda edge: (edge[0], _endpoint_coordinate_key(xyz, edge[1], edge[2])))
    return edges


def superline3d_style_fragments(candidate: dict[str, np.ndarray], cfg: object) -> dict[str, object]:
    """Create bounded line components with index-invariant edge semantics."""

    xyz = np.asarray(candidate["xyz"], dtype=np.float32)
    n = len(xyz)
    if n == 0:
        return T._frag_info_from_ids(candidate, np.zeros(0, dtype=np.int32), 1.0, 1.0)

    base = float(M.estimate_base_spacing(xyz, int(getattr(cfg, "base_spacing_sample_size", 8000))))
    if not np.isfinite(base) or base <= 0.0:
        base = 1.0

    tangent, linearness, desc = _local_line_descriptors(
        xyz,
        int(getattr(cfg, "sl3d_style_desc_k", 20)),
    )
    edges = undirected_candidate_edges(xyz, tangent, linearness, desc, base, cfg)

    union_find = UnionFind(n)
    max_component_points = int(getattr(cfg, "sl3d_style_max_component_points", 4096))
    for _cost, i, j in edges:
        root_i, root_j = union_find.find(i), union_find.find(j)
        if root_i == root_j:
            continue
        if int(union_find.size[root_i] + union_find.size[root_j]) > max_component_points:
            continue
        union_find.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(union_find.find(i), []).append(i)

    min_points = int(getattr(cfg, "min_fragment_points", 3))
    retained_groups = [indices for indices in groups.values() if len(indices) >= min_points]
    retained_groups.sort(
        key=lambda indices: min(tuple(float(value) for value in xyz[index]) for index in indices)
    )
    fragment_ids = np.full(n, -1, dtype=np.int32)
    for fragment_id, indices in enumerate(retained_groups):
        fragment_ids[np.asarray(indices, dtype=np.int64)] = fragment_id

    max_spatial = float(getattr(cfg, "sl3d_style_max_spatial_factor", 8.0)) * base
    return T._frag_info_from_ids(candidate, fragment_ids, base, max_spatial)


def token_cache_path(method: dict[str, object], subset: str, sample_id: str):
    return Legacy.Path(method["cache_root"]) / subset / f"{sample_id}_downsampled.npz"


def build_or_load_downsampled(
    path: str,
    subset: str,
    method: dict[str, object],
    cfg: object,
) -> dict[str, np.ndarray]:
    """Build an invariant token cache with the registered output contract."""

    sample_id = M.sample_id(path)
    cache = token_cache_path(method, subset, sample_id)
    T.ensure_dir(cache.parent)
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=True) as data:
                n = len(data["label"])
                return {
                    "xyz": data["xyz"],
                    "rgb": data["rgb"],
                    "label": data["label"],
                    "label_ratio": data["label_ratio"],
                    "lcc": data["lcc"],
                    "orig_index": data["orig_index"],
                    "fragment_id": data["fragment_id"],
                    "features": data["features"],
                    "bin_num_points": data["bin_num_points"],
                    "rep_anchor_dist": (
                        data["rep_anchor_dist"]
                        if "rep_anchor_dist" in data.files
                        else np.zeros(n, dtype=np.float32)
                    ),
                    "patch_points": (
                        data["patch_points"]
                        if "patch_points" in data.files
                        else np.zeros(
                            (n, int(cfg.patch_point_count), int(cfg.patch_point_dim)),
                            dtype=np.float32,
                        )
                    ),
                    "patch_mask": (
                        data["patch_mask"]
                        if "patch_mask" in data.files
                        else np.zeros((n, int(cfg.patch_point_count)), dtype=np.float32)
                    ),
                }
        except Exception:
            cache.unlink(missing_ok=True)

    raw_candidate = M.load_candidate_npy(path)
    fragments = superline3d_style_fragments(raw_candidate, cfg)
    downsampled = M.downsample_superlines_to_line_points(raw_candidate, fragments, cfg)
    np.savez_compressed(cache, **downsampled)
    return downsampled
