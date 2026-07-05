"""Unit tests for robust skin-weight transfer helpers.

Run:
  cd /data/jzy/code/SPEAR && python -m unittest tests.test_robust_skin_transfer -v
"""
import unittest

import numpy as np

from tools.robust_skin_transfer import (
    REGION_FRONT_LEFT_LEG,
    REGION_FRONT_RIGHT_LEG,
    REGION_HEAD,
    REGION_HIND_LEFT_LEG,
    REGION_HIND_RIGHT_LEG,
    REGION_TAIL,
    REGION_TORSO,
    SkeletonCapsule,
    coarse_region_labels,
    graph_region_labels_from_capsules,
    inpaint_missing_weights,
    keep_top_k_normalized,
    regularize_regions_by_connected_components,
    target_region_labels_from_source_proximity,
    transfer_weights_by_region,
)


class RobustSkinTransferTest(unittest.TestCase):

    def test_keep_top_k_normalized_prunes_and_renormalizes(self):
        weights = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        out = keep_top_k_normalized(weights, k=3)

        self.assertTrue(np.allclose(out[0, :2], 0.0))
        self.assertTrue(np.allclose(out[0, 2:], [0.25, 1.0 / 3.0, 5.0 / 12.0]))
        self.assertAlmostEqual(float(out[0].sum()), 1.0)
        self.assertTrue(np.allclose(out[1], 0.0))

    def test_inpaint_missing_weights_fills_unknown_vertices_without_moving_known(self):
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
        weights = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        known = np.array([True, False, False, True])

        out, filled, n_iters = inpaint_missing_weights(faces, weights, known, max_iterations=8)

        self.assertTrue(filled.all())
        self.assertGreaterEqual(n_iters, 1)
        self.assertTrue(np.allclose(out[0], [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(out[3], [0.0, 1.0, 0.0]))
        self.assertAlmostEqual(float(out[1].sum()), 1.0)
        self.assertAlmostEqual(float(out[2].sum()), 1.0)
        self.assertGreater(out[1, 0], 0.0)
        self.assertGreater(out[1, 1], 0.0)

    def test_coarse_region_labels_split_head_tail_torso_and_limb_side(self):
        vertices = np.array(
            [
                [0.5, 5.0, 0.0],
                [8.0, 7.0, 0.0],
                [7.0, 1.0, 0.5],
                [7.0, 1.0, -0.5],
                [3.0, 1.0, 0.5],
                [5.0, 5.0, 0.0],
            ],
            dtype=np.float64,
        )
        bounds = (
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([10.0, 10.0, 1.0], dtype=np.float64),
        )

        labels = coarse_region_labels(vertices, bounds=bounds)

        self.assertEqual(labels[0], REGION_TAIL)
        self.assertEqual(labels[1], REGION_HEAD)
        self.assertEqual(labels[2], REGION_FRONT_LEFT_LEG)
        self.assertEqual(labels[3], REGION_FRONT_RIGHT_LEG)
        self.assertEqual(labels[4], REGION_HIND_LEFT_LEG)
        self.assertEqual(labels[5], REGION_TORSO)

    def test_transfer_weights_by_region_ignores_nearer_incompatible_face(self):
        source_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        source_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        source_weights = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        source_face_regions = np.array([REGION_TAIL, REGION_HEAD], dtype=np.int64)
        target_vertices = np.array([[0.2, 0.2, 0.0]], dtype=np.float64)
        target_regions = np.array([REGION_HEAD], dtype=np.int64)

        out, matched, _stats = transfer_weights_by_region(
            source_vertices=source_vertices,
            source_faces=source_faces,
            source_weights=source_weights,
            target_vertices=target_vertices,
            source_face_regions=source_face_regions,
            target_regions=target_regions,
            max_distance=None,
        )

        self.assertTrue(matched[0])
        self.assertLess(out[0, 0], 0.01)
        self.assertGreater(out[0, 1], 0.99)

    def test_target_region_labels_use_source_proximity_but_protect_head_tail(self):
        source_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 5.0, 0.0],
                [6.0, 5.0, 0.0],
                [5.0, 6.0, 0.0],
            ],
            dtype=np.float64,
        )
        source_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        source_face_regions = np.array([REGION_TAIL, REGION_TORSO], dtype=np.int64)
        target_vertices = np.array(
            [
                [0.2, 0.2, 0.0],  # geometrically near tail, but protected as head
                [5.2, 5.2, 0.0],  # coarse leg-like label should be pulled back to torso
            ],
            dtype=np.float64,
        )
        coarse = np.array([REGION_HEAD, REGION_FRONT_LEFT_LEG], dtype=np.int64)

        labels = target_region_labels_from_source_proximity(
            source_vertices=source_vertices,
            source_faces=source_faces,
            source_face_regions=source_face_regions,
            target_vertices=target_vertices,
            coarse_target_regions=coarse,
        )

        self.assertEqual(labels[0], REGION_HEAD)
        self.assertEqual(labels[1], REGION_TORSO)

    def test_graph_region_labels_keep_connected_tail_even_when_hind_leg_is_closer(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],   # tail root seed
                [0.8, 0.05, 0.0],  # tail, but closer to hind-leg capsule
                [1.6, 0.05, 0.0],  # tail tip, still closer to hind-leg capsule
                [0.0, 0.7, 0.0],   # torso bridge seed
                [0.8, 0.8, 0.0],   # hind leg root seed
                [1.6, 0.8, 0.0],   # hind leg end
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 3],
                [1, 2, 3],
                [3, 4, 5],
            ],
            dtype=np.int64,
        )
        capsules = [
            SkeletonCapsule(REGION_TAIL, [0.0, 1.0, 0.0], [1.7, 1.0, 0.0], 0.05),
            SkeletonCapsule(REGION_HIND_LEFT_LEG, [0.0, 0.0, 0.0], [1.7, 0.0, 0.0], 0.05),
            SkeletonCapsule(REGION_TORSO, [0.0, 0.7, 0.0], [1.7, 0.7, 0.0], 0.05),
        ]
        seed_labels = np.full(len(vertices), -1, dtype=np.int64)
        seed_labels[0] = REGION_TAIL
        seed_labels[3] = REGION_TORSO
        seed_labels[4] = REGION_HIND_LEFT_LEG
        seed_labels[5] = REGION_HIND_LEFT_LEG

        labels, stats = graph_region_labels_from_capsules(
            vertices=vertices,
            faces=faces,
            capsules=capsules,
            seed_labels=seed_labels,
            unary_weight=0.2,
        )

        self.assertEqual(labels[1], REGION_TAIL)
        self.assertEqual(labels[2], REGION_TAIL)
        self.assertEqual(labels[4], REGION_HIND_LEFT_LEG)
        self.assertEqual(labels[5], REGION_HIND_LEFT_LEG)
        self.assertGreaterEqual(stats["seed_count"], 4)

    def test_graph_region_labels_do_not_hard_seed_distant_coarse_tail(self):
        vertices = np.array(
            [
                [0.0, 1.0, 0.0],  # reliable tail seed, close to tail capsule
                [1.0, 0.2, 0.0],  # coarse tail from bbox, but near torso capsule
                [1.1, 0.0, 0.0],  # torso anchor
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        capsules = [
            SkeletonCapsule(REGION_TAIL, [0.0, 1.0, 0.0], [0.0, 1.2, 0.0], 0.05),
            SkeletonCapsule(REGION_TORSO, [1.0, 0.0, 0.0], [1.2, 0.0, 0.0], 0.05),
        ]
        coarse = np.array([REGION_TAIL, REGION_TAIL, REGION_TORSO], dtype=np.int64)

        labels, stats = graph_region_labels_from_capsules(
            vertices=vertices,
            faces=faces,
            capsules=capsules,
            coarse_labels=coarse,
            unary_weight=1.0,
            seed_distance_ratio=0.05,
        )

        self.assertEqual(labels[0], REGION_TAIL)
        self.assertEqual(labels[1], REGION_TORSO)
        self.assertEqual(labels[2], REGION_TORSO)
        self.assertLess(stats["seed_count"], 3)

    def test_regularize_regions_by_connected_components_only_changes_eligible_shells(self):
        faces = np.array(
            [
                [0, 1, 2],
                [2, 3, 0],
                [4, 5, 6],
            ],
            dtype=np.int64,
        )
        labels = np.array(
            [
                REGION_TORSO,
                REGION_HIND_LEFT_LEG,
                REGION_TORSO,
                REGION_HIND_LEFT_LEG,
                REGION_HEAD,
                REGION_TORSO,
                REGION_HEAD,
            ],
            dtype=np.int64,
        )

        out, stats = regularize_regions_by_connected_components(
            faces=faces,
            labels=labels,
            eligible_regions={REGION_TORSO, REGION_TAIL, REGION_HIND_LEFT_LEG, REGION_HIND_RIGHT_LEG},
            vote_bias={REGION_TORSO: 1.25},
        )

        self.assertTrue(np.all(out[:4] == REGION_TORSO))
        self.assertTrue(np.all(out[4:] == labels[4:]))
        self.assertEqual(stats["changed_vertices"], 2)
        self.assertEqual(stats["regularized_components"], 1)


if __name__ == "__main__":
    unittest.main()
