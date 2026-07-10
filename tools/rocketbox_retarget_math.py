"""Parent-local matrix helpers for Rocketbox animation retargeting."""

import numpy as np


def rest_delta(rest_local: np.ndarray, pose_local: np.ndarray) -> np.ndarray:
    return np.linalg.inv(rest_local) @ pose_local


def apply_rest_delta(target_rest_local: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return target_rest_local @ delta


def parent_local(parent_matrix: np.ndarray | None, child_matrix: np.ndarray) -> np.ndarray:
    return child_matrix if parent_matrix is None else np.linalg.inv(parent_matrix) @ child_matrix


def horizontal_alignment(
    source_forward: np.ndarray, target_forward: np.ndarray
) -> np.ndarray:
    """Return the Z-axis rotation that maps one horizontal forward frame to another."""
    source_horizontal = np.asarray(source_forward, dtype=float)[:2]
    target_horizontal = np.asarray(target_forward, dtype=float)[:2]
    source_length = np.linalg.norm(source_horizontal)
    target_length = np.linalg.norm(target_horizontal)
    if source_length == 0.0 or target_length == 0.0:
        raise ValueError("forward vectors must have nonzero horizontal components")

    source_unit = source_horizontal / source_length
    target_unit = target_horizontal / target_length
    angle = np.arctan2(
        source_unit[0] * target_unit[1] - source_unit[1] * target_unit[0],
        np.dot(source_unit, target_unit),
    )
    cosine = np.cos(angle)
    sine = np.sin(angle)
    alignment = np.eye(4)
    alignment[:2, :2] = ((cosine, -sine), (sine, cosine))
    return alignment


def scaled_root_translation(root_translation: np.ndarray, scale: float) -> np.ndarray:
    """Apply the target scale to a source root translation exactly once."""
    return np.asarray(root_translation, dtype=float) * scale


def loop_residual(
    start_translation: np.ndarray,
    end_translation: np.ndarray,
    expected_cycle_displacement: np.ndarray,
) -> float:
    """Measure root-position seam error after removing the expected cycle advance."""
    return float(
        np.linalg.norm(
            np.asarray(end_translation, dtype=float)
            - np.asarray(start_translation, dtype=float)
            - np.asarray(expected_cycle_displacement, dtype=float)
        )
    )
