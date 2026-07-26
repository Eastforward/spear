"""Unit tests for the deterministic gait-direction classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.quadruped_gait_direction import (
    GaitDirectionError,
    classify_gait_direction,
)


def synthetic_walk_frames(
    stance_drift_per_frame=(-0.02, 0.0),
    frame_count=8,
    seed=11,
):
    """In-place walk: the low stance band drifts by a fixed vector per frame
    while the body jitters in place."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(low=(-0.5, -0.2, 0.0), high=(0.5, 0.2, 0.8), size=(600, 3))
    stance = base[:, 2] < 0.06
    frames = []
    for index in range(frame_count):
        frame = base.copy()
        frame[:, :2] += rng.normal(scale=0.001, size=(600, 2))
        frame[stance, 0] += stance_drift_per_frame[0]*index
        frame[stance, 1] += stance_drift_per_frame[1]*index
        frames.append(frame)
    return frames


def test_backward_stance_drift_is_forward_walking():
    result = classify_gait_direction(synthetic_walk_frames((-0.02, 0.0)))
    assert result["classification"] == "forward"
    assert result["walks_head_first"] is True


def test_forward_stance_drift_is_backward_walking():
    result = classify_gait_direction(synthetic_walk_frames((+0.02, 0.0)))
    assert result["classification"] == "backward"
    assert result["walks_head_first"] is False


def test_lateral_stance_drift_is_sideways_walking():
    result = classify_gait_direction(synthetic_walk_frames((0.0, 0.02)))
    assert result["classification"] == "sideways"


def test_static_frames_are_ambiguous():
    result = classify_gait_direction(synthetic_walk_frames((0.0, 0.0)))
    assert result["classification"] == "ambiguous"


def test_mismatched_vertex_counts_are_rejected():
    frames = synthetic_walk_frames()
    frames[2] = frames[2][:-5]
    with pytest.raises(GaitDirectionError, match="vertex indexing"):
        classify_gait_direction(frames)


def test_too_few_frames_are_rejected():
    frames = synthetic_walk_frames()[:2]
    with pytest.raises(GaitDirectionError, match="at least 3"):
        classify_gait_direction(frames)
