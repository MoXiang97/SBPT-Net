from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.research_candidates import invariant_superline_organization_20260917 as I


class OrthogonalTangentResidualTests(unittest.TestCase):
    def test_residual_is_invariant_to_displacement_and_tangent_signs(self):
        displacement = np.asarray([1.25, -2.0, 0.75], dtype=np.float32)
        tangent = np.asarray([2.0, 1.0, -0.5], dtype=np.float32)
        tangent /= np.linalg.norm(tangent)

        expected = I.orthogonal_tangent_residual(displacement, tangent)

        self.assertAlmostEqual(
            expected,
            I.orthogonal_tangent_residual(-displacement, tangent),
            places=7,
        )
        self.assertAlmostEqual(
            expected,
            I.orthogonal_tangent_residual(displacement, -tangent),
            places=7,
        )
        self.assertAlmostEqual(
            expected,
            I.orthogonal_tangent_residual(-displacement, -tangent),
            places=7,
        )


class UndirectedEdgeTests(unittest.TestCase):
    @staticmethod
    def _coordinate_edges(xyz, edges):
        coordinate_edges = set()
        for _cost, i, j in edges:
            endpoints = tuple(sorted((tuple(xyz[i]), tuple(xyz[j]))))
            coordinate_edges.add(endpoints)
        return coordinate_edges

    def test_edge_union_is_invariant_to_point_row_permutation(self):
        xyz = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.1, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        tangent = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (len(xyz), 1))
        linearness = np.ones(len(xyz), dtype=np.float32)
        desc = np.zeros((len(xyz), 8), dtype=np.float32)
        cfg = SimpleNamespace(
            sl3d_style_graph_k=1,
            sl3d_style_min_linearness=0.0,
            sl3d_style_min_tangent_sim=0.0,
            sl3d_style_max_desc_dist=10.0,
            sl3d_style_max_spatial_factor=100.0,
        )

        edges = I.undirected_candidate_edges(xyz, tangent, linearness, desc, 1.0, cfg)
        permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
        permuted_edges = I.undirected_candidate_edges(
            xyz[permutation],
            tangent[permutation],
            linearness[permutation],
            desc[permutation],
            1.0,
            cfg,
        )

        self.assertEqual(
            self._coordinate_edges(xyz, edges),
            self._coordinate_edges(xyz[permutation], permuted_edges),
        )

    def test_equal_distance_neighbors_use_geometry_tie_break(self):
        xyz = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        tangent = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (len(xyz), 1))
        linearness = np.ones(len(xyz), dtype=np.float32)
        desc = np.zeros((len(xyz), 8), dtype=np.float32)
        cfg = SimpleNamespace(
            sl3d_style_graph_k=1,
            sl3d_style_min_linearness=0.0,
            sl3d_style_min_tangent_sim=0.0,
            sl3d_style_max_desc_dist=10.0,
            sl3d_style_max_spatial_factor=10.0,
        )
        expected = self._coordinate_edges(
            xyz,
            I.undirected_candidate_edges(xyz, tangent, linearness, desc, 1.0, cfg),
        )
        permutation = np.asarray([3, 1, 2, 0], dtype=np.int64)
        observed = self._coordinate_edges(
            xyz[permutation],
            I.undirected_candidate_edges(
                xyz[permutation],
                tangent[permutation],
                linearness[permutation],
                desc[permutation],
                1.0,
                cfg,
            ),
        )
        self.assertEqual(expected, observed)


class FragmentPartitionTests(unittest.TestCase):
    @staticmethod
    def _co_membership(labels):
        labels = np.asarray(labels)
        return (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)

    def test_fragment_partition_is_invariant_to_point_row_permutation(self):
        x = np.asarray([0.0, 0.8, 1.7, 2.7, 3.8, 5.0], dtype=np.float32)
        first = np.column_stack((x, 0.03 * x**2, np.zeros_like(x)))
        second = np.column_stack((x + 0.15, np.full_like(x, 8.0), 0.02 * x))
        xyz = np.vstack((first, second)).astype(np.float32)
        candidate = {
            "xyz": xyz,
            "label": np.zeros(len(xyz), dtype=np.int64),
        }
        cfg = SimpleNamespace(
            base_spacing_sample_size=8000,
            sl3d_style_desc_k=4,
            sl3d_style_graph_k=2,
            sl3d_style_min_linearness=0.0,
            sl3d_style_min_tangent_sim=0.0,
            sl3d_style_max_desc_dist=100.0,
            sl3d_style_max_spatial_factor=3.0,
            sl3d_style_max_component_points=4,
            min_fragment_points=1,
        )

        original = I.superline3d_style_fragments(candidate, cfg)["frag_ids"]
        permutation = np.asarray([8, 2, 10, 0, 6, 4, 11, 5, 1, 9, 3, 7], dtype=np.int64)
        permuted_candidate = {
            "xyz": xyz[permutation],
            "label": candidate["label"][permutation],
        }
        permuted = I.superline3d_style_fragments(permuted_candidate, cfg)["frag_ids"]
        restored = np.empty_like(permuted)
        restored[permutation] = permuted

        np.testing.assert_array_equal(
            self._co_membership(original),
            self._co_membership(restored),
        )


if __name__ == "__main__":
    unittest.main()
