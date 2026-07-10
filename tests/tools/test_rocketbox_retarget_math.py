import numpy as np

from tools.rocketbox_retarget_math import (
    apply_rest_delta,
    horizontal_alignment,
    loop_residual,
    parent_local,
    rest_delta,
    scaled_root_translation,
)


def rotation_x(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosine, -sine, 0.0],
            [0.0, sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    return np.array(
        [
            [cosine, -sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def assert_matrix_close(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_rest_delta_preserves_motion_across_different_bind_axes():
    source_rest = rotation_z(45)
    source_pose = source_rest @ rotation_x(30)
    target_rest = rotation_z(-20)

    delta = rest_delta(source_rest, source_pose)
    target_pose = apply_rest_delta(target_rest, delta)

    assert_matrix_close(np.linalg.inv(target_rest) @ target_pose, rotation_x(30))


def test_parent_local_extracts_child_transform_from_parent_frame():
    parent_matrix = rotation_z(45)
    child_local = rotation_x(30)

    assert_matrix_close(parent_local(parent_matrix, parent_matrix @ child_local), child_local)
    assert_matrix_close(parent_local(None, child_local), child_local)


def test_horizontal_alignment_uses_motion_frames_for_half_turn_and_identity():
    assert_matrix_close(
        horizontal_alignment(np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])),
        rotation_z(180),
    )
    assert_matrix_close(
        horizontal_alignment(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        np.eye(4),
    )


def test_scaled_root_translation_applies_scale_exactly_once():
    source_translation = np.array([4.0, -2.0, 1.0])

    np.testing.assert_allclose(
        scaled_root_translation(source_translation, 0.25),
        np.array([1.0, -0.5, 0.25]),
    )


def test_loop_residual_subtracts_expected_cycle_displacement_before_measurement():
    start_translation = np.array([0.0, 0.0, 0.0])
    end_translation = np.array([2.1, 0.2, 0.0])
    expected_cycle_displacement = np.array([2.0, 0.0, 0.0])

    assert np.isclose(
        loop_residual(
            start_translation,
            end_translation,
            expected_cycle_displacement,
        ),
        np.hypot(0.1, 0.2),
    )
