"""Pure NumPy tests for humanoid part-aware skin-weight transfer."""
import unittest

import numpy as np

from tools.human_part_transfer import (
    HumanRegion,
    collapse_finger_weights_to_palms,
    cross_limb_bridge_face_mask,
    human_ground_artifact_mask,
    source_face_regions,
    source_vertex_regions_from_weights,
    target_regions_from_capsules,
    transfer_human_weights,
)


class HumanPartTransferTest(unittest.TestCase):

    def test_source_regions_follow_vertex_group_mass_for_all_human_parts(self):
        group_names = [
            "Bip01 Spine",
            "Bip01 Head",
            "Bip01 L UpperArm",
            "Bip01 L Forearm",
            "Bip01 L Hand",
            "Bip01 R UpperArm",
            "Bip01 R Forearm",
            "Bip01 R Hand",
            "Bip01 L Thigh",
            "Bip01 L Calf",
            "Bip01 L Foot",
            "Bip01 R Thigh",
            "Bip01 R Calf",
            "Bip01 R Foot",
        ]
        weights = np.eye(len(group_names), dtype=np.float64)

        regions = source_vertex_regions_from_weights(weights, group_names)

        expected = [
            HumanRegion.TORSO,
            HumanRegion.HEAD,
            HumanRegion.LEFT_UPPER_ARM,
            HumanRegion.LEFT_FOREARM,
            HumanRegion.LEFT_PALM,
            HumanRegion.RIGHT_UPPER_ARM,
            HumanRegion.RIGHT_FOREARM,
            HumanRegion.RIGHT_PALM,
            HumanRegion.LEFT_THIGH,
            HumanRegion.LEFT_CALF,
            HumanRegion.LEFT_FOOT,
            HumanRegion.RIGHT_THIGH,
            HumanRegion.RIGHT_CALF,
            HumanRegion.RIGHT_FOOT,
        ]
        self.assertEqual(regions.tolist(), [int(region) for region in expected])

    def test_rocketbox_facial_vertex_groups_classify_as_head(self):
        group_names = [
            "Bip01 REye",
            "Bip01 MJaw",
            "Bip01 RMasseter",
            "Bip01 MUpperLip",
            "Bip01 RCaninus",
            "Bip01 REyeBlinkTop",
            "Bip01 RMouthCorner",
            "Bip01 RCheek",
            "Bip01 MMiddleEyebrow",
            "Bip01 MNose",
            "Bip01 MTongue",
        ]

        regions = source_vertex_regions_from_weights(
            np.eye(len(group_names), dtype=np.float64),
            group_names,
        )

        self.assertEqual(
            regions.tolist(),
            [int(HumanRegion.HEAD)] * len(group_names),
        )

    def test_fingers_classify_as_palms_and_adjacent_joint_face_uses_majority(self):
        names = ["Bip01 L Finger01", "Bip01 L Forearm", "Bip01 Spine"]
        weights = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.0, 1.0, 0.0],
                [0.0, 0.7, 0.3],
            ],
            dtype=np.float64,
        )
        vertex_regions = source_vertex_regions_from_weights(weights, names)
        faces = np.array([[0, 1, 2]], dtype=np.int64)

        face_regions = source_face_regions(faces, vertex_regions)

        self.assertEqual(vertex_regions[0], HumanRegion.LEFT_PALM)
        self.assertEqual(face_regions[0], HumanRegion.LEFT_FOREARM)

    def test_clavicles_classify_as_side_specific_upper_arms(self):
        names = ["Bip01 L Clavicle", "Bip01 R Clavicle"]

        regions = source_vertex_regions_from_weights(np.eye(2), names)

        self.assertEqual(
            regions.tolist(),
            [HumanRegion.LEFT_UPPER_ARM, HumanRegion.RIGHT_UPPER_ARM],
        )

    def test_left_upper_arm_rejects_opposite_clavicle_influence(self):
        source = {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "weights": np.array(
                [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
                dtype=np.float64,
            ),
            "group_names": ["Bip01 L Clavicle", "Bip01 R Clavicle"],
        }
        target = {
            "vertices": np.array([[0.25, 0.25, 0.0]], dtype=np.float64),
            "faces": np.empty((0, 3), dtype=np.int64),
            "regions": np.array([HumanRegion.LEFT_UPPER_ARM], dtype=np.int64),
        }

        weights, stats = transfer_human_weights(source, target)

        self.assertEqual(stats["unmatched"], 0)
        self.assertAlmostEqual(weights[0, 0], 1.0)
        self.assertEqual(weights[0, 1], 0.0)

    def test_capsules_separate_sides_and_each_arm_leg_segment(self):
        ordered_regions = list(HumanRegion)
        capsules = {
            region: (
                np.array([float(index) * 4.0, 0.0, 0.0]),
                np.array([float(index) * 4.0, 1.0, 0.0]),
                0.25,
            )
            for index, region in enumerate(ordered_regions)
        }
        vertices = np.array(
            [[float(index) * 4.0, 0.5, 0.0] for index in range(len(ordered_regions))],
            dtype=np.float64,
        )

        regions = target_regions_from_capsules(vertices, capsules)

        self.assertEqual(regions.tolist(), [int(region) for region in ordered_regions])

    def test_palm_vertices_never_receive_opposite_or_torso_weights(self):
        names = ["Bip01 L Hand", "Bip01 R Hand", "Bip01 Spine"]
        source = {
            "vertices": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "weights": np.array(
                [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.1, 0.3]],
                dtype=np.float64,
            ),
            "group_names": names,
        }
        target = {
            "vertices": np.array(
                [[0.0, 0.2, 0.2], [0.0, 0.3, 0.2], [0.0, 0.2, 0.3]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "capsules": {
                HumanRegion.LEFT_PALM: (
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    0.3,
                )
            },
        }

        weights, stats = transfer_human_weights(source, target)

        self.assertEqual(stats["unmatched"], 0)
        self.assertTrue(np.allclose(weights[:, 0], 1.0))
        self.assertEqual(weights[:, 1].max(), 0.0)
        self.assertEqual(weights[:, 2].max(), 0.0)

    def test_adjacent_joint_blending_is_preserved(self):
        source = {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "weights": np.array([[1.0, 0.0], [0.6, 0.4], [0.6, 0.4]], dtype=np.float64),
            "group_names": ["Bip01 L UpperArm", "Bip01 L Forearm"],
        }
        target = {
            "vertices": np.array([[0.5, 0.25, 0.0]], dtype=np.float64),
            "faces": np.empty((0, 3), dtype=np.int64),
            "regions": np.array([HumanRegion.LEFT_UPPER_ARM], dtype=np.int64),
        }

        weights, stats = transfer_human_weights(source, target)

        self.assertEqual(stats["unmatched"], 0)
        self.assertGreater(weights[0, 0], 0.0)
        self.assertGreater(weights[0, 1], 0.0)
        self.assertAlmostEqual(float(weights[0].sum()), 1.0)

    def test_graph_fill_leaves_no_unmatched_and_top_four_are_normalized(self):
        source = {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "weights": np.array(
                [
                    [0.30, 0.25, 0.20, 0.10, 0.10, 0.05],
                    [0.25, 0.25, 0.20, 0.15, 0.10, 0.05],
                    [0.20, 0.25, 0.25, 0.15, 0.10, 0.05],
                ],
                dtype=np.float64,
            ),
            "group_names": [
                "Bip01 Spine",
                "Bip01 Spine1",
                "Bip01 Spine2",
                "Bip01 Pelvis",
                "Bip01 L Clavicle",
                "Bip01 R Clavicle",
            ],
        }
        target = {
            "vertices": np.array(
                [[0.02, 0.02, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "regions": np.full(3, HumanRegion.TORSO, dtype=np.int64),
        }

        weights, stats = transfer_human_weights(source, target, max_distance=0.15)

        self.assertEqual(stats["initial_unmatched"], 2)
        self.assertEqual(stats["unmatched"], 0)
        self.assertTrue(np.all(np.count_nonzero(weights > 0.0, axis=1) <= 4))
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))

    def test_graph_fill_ignores_faces_bridging_left_and_right_targets(self):
        source = {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "weights": np.ones((3, 1), dtype=np.float64),
            "group_names": ["Bip01 Spine"],
        }
        target = {
            "vertices": np.array(
                [[-2.0, 0.0, 0.0], [0.01, 0.01, 0.0], [0.02, 0.02, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "regions": np.array(
                [
                    HumanRegion.LEFT_UPPER_ARM,
                    HumanRegion.RIGHT_UPPER_ARM,
                    HumanRegion.RIGHT_UPPER_ARM,
                ],
                dtype=np.int64,
            ),
        }

        weights, stats = transfer_human_weights(
            source,
            target,
            max_distance=0.2,
            require_complete=False,
        )

        self.assertEqual(stats["initial_unmatched"], 1)
        self.assertEqual(stats["unmatched"], 1)
        self.assertEqual(stats["unmatched_regions"], {"left_upper_arm": 1})
        self.assertTrue(np.allclose(weights[0], 0.0))
        self.assertTrue(np.allclose(weights[1:].sum(axis=1), 1.0))

    def test_disconnected_component_without_seed_stays_unmatched(self):
        source = {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "weights": np.ones((3, 1), dtype=np.float64),
            "group_names": ["Bip01 Spine"],
        }
        target = {
            "vertices": np.array(
                [
                    [0.01, 0.01, 0.0], [0.02, 0.01, 0.0], [0.01, 0.02, 0.0],
                    [3.0, 0.0, 0.0], [3.1, 0.0, 0.0], [3.0, 0.1, 0.0],
                ],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
            "regions": np.full(6, HumanRegion.TORSO, dtype=np.int64),
        }

        weights, stats = transfer_human_weights(
            source,
            target,
            max_distance=0.2,
            require_complete=False,
        )

        self.assertEqual(stats["initial_unmatched"], 3)
        self.assertEqual(stats["unmatched"], 3)
        self.assertEqual(stats["unmatched_regions"], {"torso": 3})
        self.assertEqual(stats["graph_filled"], 0)
        self.assertTrue(np.allclose(weights[:3].sum(axis=1), 1.0))
        self.assertTrue(np.allclose(weights[3:], 0.0))

    def test_incomplete_transfer_raises_with_count_and_regions_by_default(self):
        source = {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "weights": np.ones((3, 1), dtype=np.float64),
            "group_names": ["Bip01 Spine"],
        }
        target = {
            "vertices": np.array([[3.0, 0.0, 0.0]], dtype=np.float64),
            "faces": np.empty((0, 3), dtype=np.int64),
            "regions": np.array([HumanRegion.TORSO], dtype=np.int64),
        }

        with self.assertRaisesRegex(ValueError, r"1 unmatched.*torso"):
            transfer_human_weights(source, target, max_distance=0.2)

    def test_graph_fill_reapplies_target_region_influence_mask(self):
        source = {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "weights": np.tile([0.3, 0.5, 0.2], (3, 1)),
            "group_names": [
                "Bip01 L UpperArm",
                "Bip01 L Forearm",
                "Bip01 L Hand",
            ],
        }
        target = {
            "vertices": np.array(
                [[0.02, 0.02, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
            "regions": np.array(
                [
                    HumanRegion.LEFT_FOREARM,
                    HumanRegion.LEFT_PALM,
                    HumanRegion.LEFT_PALM,
                ],
                dtype=np.int64,
            ),
        }

        weights, stats = transfer_human_weights(source, target, max_distance=0.15)

        self.assertEqual(stats["initial_unmatched"], 2)
        self.assertEqual(stats["unmatched"], 0)
        self.assertTrue(np.allclose(weights[1:, 0], 0.0))
        self.assertTrue(np.all(weights[1:, 1:3] > 0.0))
        self.assertTrue(np.allclose(weights[1:].sum(axis=1), 1.0))

    def test_finger_mass_collapses_completely_into_matching_palms(self):
        names = [
            "Bip01 L Hand",
            "Bip01 R Hand",
            "Bip01 L Finger0",
            "Bip01 L Finger11",
            "Bip01 R Finger01",
            "Bip01 Spine",
        ]
        weights = np.array(
            [[0.10, 0.05, 0.20, 0.25, 0.30, 0.10]],
            dtype=np.float64,
        )

        collapsed = collapse_finger_weights_to_palms(weights, names)

        self.assertTrue(np.allclose(collapsed[0], [0.55, 0.35, 0.0, 0.0, 0.0, 0.10]))
        self.assertAlmostEqual(float(collapsed[0].sum()), 1.0)

    def test_low_flat_disconnected_component_is_a_ground_card(self):
        vertices = np.array(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
                [-0.2, -0.2, 1.0],
                [0.2, -0.2, 1.0],
                [0.2, 0.2, 3.0],
                [-0.2, 0.2, 3.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
            dtype=np.int64,
        )

        artifact = human_ground_artifact_mask(vertices, faces, min_vertices=4)

        self.assertTrue(artifact[:4].all())
        self.assertFalse(artifact[4:].any())

    def test_foot_dominant_flat_component_is_preserved_as_a_shoe_sole(self):
        vertices = np.array(
            [
                [-2.0, -2.0, 0.0], [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0],
                [-0.2, -0.2, 1.0], [0.2, -0.2, 1.0],
                [0.2, 0.2, 3.0], [-0.2, 0.2, 3.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
            dtype=np.int64,
        )
        regions = np.array(
            [
                HumanRegion.LEFT_FOOT,
                HumanRegion.LEFT_FOOT,
                HumanRegion.LEFT_FOOT,
                HumanRegion.LEFT_CALF,
                HumanRegion.TORSO,
                HumanRegion.TORSO,
                HumanRegion.HEAD,
                HumanRegion.HEAD,
            ],
            dtype=np.int64,
        )

        artifact = human_ground_artifact_mask(
            vertices,
            faces,
            vertex_regions=regions,
            min_vertices=4,
        )

        self.assertFalse(artifact.any())

    def test_only_low_faces_bridging_left_and_right_legs_are_rejected(self):
        vertices = np.array(
            [
                [-1.0, 0.0, 0.2], [1.0, 0.0, 0.2], [0.0, 0.2, 0.2],
                [-1.0, 1.0, 0.3], [-0.8, 1.0, 0.3], [-0.9, 1.2, 0.3],
                [-1.0, 2.0, 1.5], [1.0, 2.0, 1.5], [0.0, 2.2, 1.5],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
        regions = np.array(
            [
                HumanRegion.LEFT_CALF,
                HumanRegion.RIGHT_CALF,
                HumanRegion.LEFT_FOOT,
                HumanRegion.LEFT_CALF,
                HumanRegion.LEFT_FOOT,
                HumanRegion.LEFT_CALF,
                HumanRegion.LEFT_THIGH,
                HumanRegion.RIGHT_THIGH,
                HumanRegion.RIGHT_THIGH,
            ],
            dtype=np.int64,
        )

        bridge = cross_limb_bridge_face_mask(
            vertices,
            faces,
            regions,
            pelvis_height=1.0,
        )

        self.assertEqual(bridge.tolist(), [True, False, False])

    def test_low_crotch_face_with_torso_vertex_is_preserved(self):
        vertices = np.array(
            [[0.0, 0.0, 0.8], [-0.3, 0.0, 0.7], [0.3, 0.0, 0.7]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        regions = np.array(
            [
                HumanRegion.TORSO,
                HumanRegion.LEFT_THIGH,
                HumanRegion.RIGHT_THIGH,
            ],
            dtype=np.int64,
        )

        bridge = cross_limb_bridge_face_mask(
            vertices,
            faces,
            regions,
            pelvis_height=1.0,
        )

        self.assertFalse(bridge[0])

    def test_leg_bridge_crossing_pelvis_height_is_preserved(self):
        vertices = np.array(
            [[-0.3, 0.0, 0.1], [0.3, 0.0, 0.1], [0.0, 0.0, 1.2]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        regions = np.array(
            [
                HumanRegion.LEFT_THIGH,
                HumanRegion.RIGHT_THIGH,
                HumanRegion.LEFT_THIGH,
            ],
            dtype=np.int64,
        )

        bridge = cross_limb_bridge_face_mask(
            vertices,
            faces,
            regions,
            pelvis_height=1.0,
        )

        self.assertFalse(bridge[0])

    def test_cross_limb_bridge_requires_explicit_pelvis_height(self):
        vertices = np.zeros((3, 3), dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        regions = np.array(
            [
                HumanRegion.LEFT_THIGH,
                HumanRegion.RIGHT_THIGH,
                HumanRegion.LEFT_THIGH,
            ],
            dtype=np.int64,
        )

        with self.assertRaises(TypeError):
            cross_limb_bridge_face_mask(vertices, faces, regions)


if __name__ == "__main__":
    unittest.main()
