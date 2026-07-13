"""Shared visual-root to acoustic-emitter trajectory conversion."""

from __future__ import annotations

import math

import numpy as np


def acoustic_trajectory(actor_root_trajectory, source_spec: dict) -> np.ndarray:
    """Return a detached trajectory with the configured emitter-height offset."""
    trajectory = np.asarray(actor_root_trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(
            f"actor root trajectory must have shape (N, 3), got {trajectory.shape}"
        )
    offset_m = float(source_spec.get("audio_source_height_offset_m", 0.0))
    if not math.isfinite(offset_m):
        raise ValueError("audio_source_height_offset_m must be finite")
    result = trajectory.copy()
    result[:, 2] += offset_m
    return result
