"""Unit tests for the deterministic quadruped forward-axis estimator."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.quadruped_forward_estimation import (
    ForwardEstimationError,
    estimate_forward,
    estimate_unsigned_axis_yaw_deg,
    vote_head_end,
)


def synthetic_quadruped(
    yaw_deg=0.0,
    seed=7,
    head=True,
    leg_widths=((0.3, 0.10), (-0.3, 0.16)),
):
    """Standing dog-like point cloud, head toward the yaw direction.

    Z-up world frame.  Torso is an ellipsoid band; the front leg pair is
    narrower than the hind pair; the head is a raised, dense blob past the
    front end of the torso.
    """
    rng = np.random.default_rng(seed)
    torso = rng.normal(size=(2600, 3))*np.array([0.42, 0.13, 0.09])
    torso[:, 2] += 0.52
    legs = []
    for longitudinal, lateral_half_width in leg_widths:
        for side in (-1.0, 1.0):
            column = rng.normal(size=(240, 3))*np.array([0.03, 0.03, 0.0])
            column[:, 0] += longitudinal
            column[:, 1] += side*lateral_half_width
            column[:, 2] = rng.uniform(0.0, 0.42, size=240)
            legs.append(column)
    parts = [torso] + legs
    if head:
        blob = rng.normal(size=(900, 3))*np.array([0.09, 0.07, 0.07])
        blob += np.array([0.62, 0.0, 0.78])
        tail = rng.normal(size=(80, 3))*np.array([0.10, 0.02, 0.02])
        tail += np.array([-0.58, 0.0, 0.62])
        parts += [blob, tail]
    cloud = np.concatenate(parts, axis=0)
    angle = math.radians(yaw_deg)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return cloud @ rotation.T


def angular_difference(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


@pytest.mark.parametrize("yaw", [0.0, 49.325, 180.0, -90.0, 137.5])
def test_estimate_recovers_forward_yaw(yaw):
    cloud = synthetic_quadruped(yaw_deg=yaw)
    estimate = estimate_forward(cloud)
    assert angular_difference(estimate["estimated_front_yaw_deg"], yaw) < 10.0
    assert estimate["head_end_vote"]["confidence"] >= 0.5
    candidates = estimate["candidate_front_yaw_degrees"]
    assert len(candidates) == 2
    assert angular_difference(candidates[0], candidates[1]) == pytest.approx(
        180.0, abs=1e-6
    )


def test_unsigned_axis_is_stable_across_head_flip():
    forward = synthetic_quadruped(yaw_deg=30.0, seed=3)
    backward = synthetic_quadruped(yaw_deg=210.0, seed=3)
    yaw_forward = estimate_unsigned_axis_yaw_deg(forward)
    yaw_backward = estimate_unsigned_axis_yaw_deg(backward)
    assert angular_difference(yaw_forward, yaw_backward) < 3.0 or (
        angular_difference(yaw_forward, yaw_backward + 180.0) < 3.0
    )


def test_symmetric_cloud_reports_low_confidence():
    cloud = synthetic_quadruped(
        yaw_deg=0.0,
        head=False,
        leg_widths=((0.3, 0.13), (-0.3, 0.13)),
    )
    unsigned = estimate_unsigned_axis_yaw_deg(cloud)
    vote = vote_head_end(cloud, unsigned)
    assert vote["confidence"] < 0.5


def test_estimate_rejects_degenerate_input():
    with pytest.raises(ForwardEstimationError):
        estimate_forward(np.zeros((10, 3)))
    with pytest.raises(ForwardEstimationError):
        estimate_forward(np.full((500, 3), np.nan))
