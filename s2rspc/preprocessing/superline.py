"""Invariant line-oriented organization of preserved candidate points."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..config import SuperlineConfig


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)

    def find(self, item: int) -> int:
        parent = int(self.parent[item])
        while parent != int(self.parent[parent]):
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = int(self.parent[parent])
        return parent

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def estimate_base_spacing(xyz: np.ndarray, sample_size: int = 8000) -> float:
    xyz = np.asarray(xyz, dtype=np.float32)
    if len(xyz) <= 1:
        return 1.0
    if len(xyz) > int(sample_size):
        index = np.random.default_rng(2026).choice(
            len(xyz), int(sample_size), replace=False
        )
        points = xyz[index]
    else:
        points = xyz
    tree = cKDTree(points)
    try:
        distance, _ = tree.query(points, k=2, workers=-1)
    except TypeError:
        distance, _ = tree.query(points, k=2)
    nearest = distance[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-8)]
    return float(np.median(nearest)) if len(nearest) else 1.0


def pca_stats(points: np.ndarray) -> Dict[str, np.ndarray | float]:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return {
            "centroid": np.zeros(3, dtype=np.float32),
            "tangent": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "length": 0.0,
            "width": 0.0,
            "linearness": 0.0,
        }
    centroid = points.mean(axis=0)
    if len(points) < 3:
        tangent = (
            points[-1] - points[0] if len(points) > 1 else np.array([1.0, 0.0, 0.0])
        )
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
        projected = (points - centroid) @ tangent
        return {
            "centroid": centroid.astype(np.float32),
            "tangent": tangent.astype(np.float32),
            "length": float(np.ptp(projected)) if len(points) > 1 else 0.0,
            "width": 0.0,
            "linearness": 1.0 if len(points) > 1 else 0.0,
        }
    centered = points - centroid
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    tangent = vectors[:, order[0]].astype(np.float32)
    tangent /= np.linalg.norm(tangent) + 1e-8
    projected = centered @ tangent
    perpendicular = centered - np.outer(projected, tangent)
    return {
        "centroid": centroid.astype(np.float32),
        "tangent": tangent,
        "length": float(np.ptp(projected)),
        "width": float(np.sqrt(np.mean(np.sum(perpendicular * perpendicular, axis=1)))),
        "linearness": float((values[0] - values[1]) / (values[0] + 1e-8)),
    }


def _local_descriptors(
    xyz: np.ndarray, neighbors: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(xyz)
    tangent = np.zeros((count, 3), dtype=np.float32)
    descriptors = np.zeros((count, 8), dtype=np.float32)
    linearness = np.zeros(count, dtype=np.float32)
    if count == 0:
        return tangent, linearness, descriptors
    tree = cKDTree(xyz)
    k = min(max(4, int(neighbors)), count)
    distance, index = tree.query(xyz, k=k)
    if k == 1:
        distance = distance.reshape(-1, 1)
        index = index.reshape(-1, 1)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for point_index in range(count):
        local = xyz[np.asarray(index[point_index], dtype=np.int64).reshape(-1)]
        center = local.mean(axis=0, keepdims=True)
        if len(local) < 3:
            tangent[point_index] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            continue
        covariance = (local - center).T @ (local - center) / max(len(local) - 1, 1)
        values, vectors = np.linalg.eigh(covariance.astype(np.float64))
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order].astype(np.float32), 0.0)
        vectors = vectors[:, order].astype(np.float32)
        direction = vectors[:, 0]
        if direction[0] < 0:
            direction = -direction
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        tangent[point_index] = direction
        first, second, third = (float(values[i]) for i in range(3))
        denominator = max(first, 1e-8)
        linear = (first - second) / denominator
        planar = (second - third) / denominator
        scatter = third / denominator
        linearness[point_index] = linear
        delta = local - xyz[point_index][None, :]
        axial = delta @ direction
        perpendicular = delta - axial[:, None] * direction[None, :]
        distances = np.asarray(distance[point_index], dtype=np.float32).reshape(-1)
        descriptors[point_index] = np.asarray(
            [
                linear,
                planar,
                scatter,
                1.0 - abs(float(np.dot(vectors[:, 2], up))),
                float(np.mean(distances)),
                float(np.std(distances)),
                float(np.mean(np.abs(axial))),
                float(np.mean(np.linalg.norm(perpendicular, axis=1))),
            ],
            dtype=np.float32,
        )
    standardized = (
        (descriptors - descriptors.mean(axis=0, keepdims=True))
        / (descriptors.std(axis=0, keepdims=True) + 1e-6)
    ).astype(np.float32)
    return tangent, linearness, standardized


def orthogonal_tangent_residual(displacement: np.ndarray, tangent: np.ndarray) -> float:
    """Norm left after orthogonally projecting displacement onto tangent."""

    displacement = np.asarray(displacement, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 0.0:
        return float(np.linalg.norm(displacement))
    unit_tangent = tangent / tangent_norm
    residual = displacement - float(np.dot(displacement, unit_tangent)) * unit_tangent
    return float(np.linalg.norm(residual))


def _endpoint_coordinate_key(
    xyz: np.ndarray, left: int, right: int
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    endpoints = (
        tuple(float(value) for value in xyz[left]),
        tuple(float(value) for value in xyz[right]),
    )
    return tuple(sorted(endpoints))  # type: ignore[return-value]


def undirected_candidate_edges(
    xyz: np.ndarray,
    tangent: np.ndarray,
    linearness: np.ndarray,
    descriptor: np.ndarray,
    base_spacing: float,
    cfg: SuperlineConfig,
) -> List[Tuple[float, int, int]]:
    """Union directed kNN arcs into deterministic undirected candidate edges."""

    xyz = np.asarray(xyz, dtype=np.float32)
    count = len(xyz)
    if count < 2:
        return []
    tree = cKDTree(xyz)
    k = min(max(2, int(cfg.graph_neighbors) + 1), count)
    boundary_distance, _ = tree.query(xyz, k=k)
    if k == 1:
        boundary_distance = boundary_distance.reshape(-1, 1)

    unordered_pairs: set[Tuple[int, int]] = set()
    for left in range(count):
        boundary = float(np.max(np.asarray(boundary_distance[left]).reshape(-1)))
        tolerance = max(1e-12, abs(boundary) * 1e-7)
        candidates = [
            int(right)
            for right in tree.query_ball_point(xyz[left], boundary + tolerance)
            if int(right) != left
        ]
        candidates.sort(
            key=lambda right: (
                float(np.dot(xyz[right] - xyz[left], xyz[right] - xyz[left])),
                tuple(float(value) for value in xyz[right]),
            )
        )
        for right in candidates[: min(int(cfg.graph_neighbors), count - 1)]:
            unordered_pairs.add((min(left, right), max(left, right)))

    maximum_spatial = float(cfg.maximum_spatial_factor) * base_spacing
    edges: List[Tuple[float, int, int]] = []
    for left, right in unordered_pairs:
        displacement = xyz[right] - xyz[left]
        distance = float(np.linalg.norm(displacement))
        if distance > maximum_spatial:
            continue
        if (
            min(float(linearness[left]), float(linearness[right]))
            < cfg.minimum_linearness
        ):
            continue
        tangent_similarity = abs(float(np.dot(tangent[left], tangent[right])))
        if tangent_similarity < cfg.minimum_tangent_similarity:
            continue
        descriptor_distance = float(
            np.linalg.norm(descriptor[left] - descriptor[right])
        )
        if descriptor_distance > cfg.maximum_descriptor_distance:
            continue
        residual = min(
            orthogonal_tangent_residual(displacement, tangent[left]),
            orthogonal_tangent_residual(displacement, tangent[right]),
        )
        cost = (
            cfg.descriptor_cost_weight * descriptor_distance
            + cfg.tangent_cost_weight * (1.0 - tangent_similarity)
            + cfg.distance_cost_weight * distance / max(base_spacing, 1e-6)
            + cfg.residual_cost_weight * residual / max(base_spacing, 1e-6)
        )
        edges.append((float(cost), left, right))

    edges.sort(
        key=lambda item: (
            item[0],
            _endpoint_coordinate_key(xyz, item[1], item[2]),
        )
    )
    return edges


def construct_superlines(
    candidate: Dict[str, np.ndarray], cfg: SuperlineConfig
) -> Dict[str, np.ndarray]:
    xyz = np.asarray(candidate["xyz"], dtype=np.float32)
    count = len(xyz)
    base_spacing = estimate_base_spacing(xyz, cfg.spacing_sample_size)
    if count == 0:
        return {
            "fragment_id": np.zeros(0, dtype=np.int32),
            "base_spacing": np.asarray(base_spacing, dtype=np.float32),
        }

    tangent, linearness, descriptor = _local_descriptors(xyz, cfg.descriptor_neighbors)
    edges = undirected_candidate_edges(
        xyz, tangent, linearness, descriptor, base_spacing, cfg
    )
    union_find = UnionFind(count)
    for _, left, right in edges:
        left_root, right_root = union_find.find(left), union_find.find(right)
        if left_root == right_root:
            continue
        if (
            int(union_find.size[left_root] + union_find.size[right_root])
            > cfg.maximum_component_points
        ):
            continue
        union_find.union(left, right)

    groups: Dict[int, List[int]] = {}
    for point_index in range(count):
        groups.setdefault(union_find.find(point_index), []).append(point_index)
    retained = [
        members
        for members in groups.values()
        if len(members) >= cfg.minimum_fragment_points
    ]
    retained.sort(
        key=lambda members: min(
            tuple(float(value) for value in xyz[index]) for index in members
        )
    )
    fragment_id = np.full(count, -1, dtype=np.int32)
    for next_id, members in enumerate(retained):
        fragment_id[np.asarray(members, dtype=np.int64)] = next_id
    return {
        "fragment_id": fragment_id,
        "base_spacing": np.asarray(base_spacing, dtype=np.float32),
    }
