import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_listener_local_xy_uses_forward_and_left_axes():
    from event_constraints import listener_local_xy

    pts = np.array([
        [2.0, 0.0, 0.45],
        [0.0, 1.0, 0.45],
        [-2.0, 0.0, 0.45],
    ])

    local = listener_local_xy(pts, mic_pos_m=(0.0, 0.0, 1.2), mic_yaw_deg=0.0)

    np.testing.assert_allclose(local[:, 0], [2.0, 0.0, -2.0], atol=1e-6)
    np.testing.assert_allclose(local[:, 1], [0.0, 1.0, 0.0], atol=1e-6)


def test_behind_and_not_visible_constraints_pass_for_rear_path():
    from event_constraints import (
        constraint_behind_camera,
        constraint_left_to_right,
        constraint_not_visible,
    )

    traj = np.column_stack([
        np.full(8, -2.0),
        np.linspace(1.0, -1.0, 8),
        np.full(8, 0.45),
    ])

    assert constraint_behind_camera("rear", traj, (0.0, 0.0, 1.2), 0.0).passed
    assert constraint_not_visible("rear", traj, (0.0, 0.0, 1.2), 0.0).passed
    assert constraint_left_to_right("rear", traj, (0.0, 0.0, 1.2), 0.0).passed


def test_visible_min_frames_fails_for_rear_path():
    from event_constraints import constraint_visible_min_frames

    traj = np.array([[-2.0, 0.0, 0.45]] * 5)
    result = constraint_visible_min_frames(
        "rear", traj, (0.0, 0.0, 1.2), 0.0, min_visible_frames=1
    )

    assert not result.passed
    assert result.details["visible_frames"] == 0


def test_stationary_and_actor_distance_constraints():
    from event_constraints import constraint_min_actor_distance, constraint_stationary

    idle = np.array([[2.0, 0.0, 0.45]] * 6)
    moving_far = np.column_stack([
        np.full(6, -2.0),
        np.linspace(1.0, -1.0, 6),
        np.full(6, 0.45),
    ])
    moving_close = np.array([[2.2, 0.0, 0.45]] * 6)

    assert constraint_stationary("front", idle).passed
    assert constraint_min_actor_distance(
        "front", idle, "rear", moving_far, min_distance_m=1.0
    ).passed
    assert not constraint_min_actor_distance(
        "front", idle, "rear", moving_close, min_distance_m=1.0
    ).passed


def test_verify_constraints_reports_failures():
    from event_constraints import ConstraintResult, verify_constraints

    results = verify_constraints([
        ConstraintResult("ok", True, {}),
        ConstraintResult("bad", False, {"why": "example"}),
    ])

    assert not results["passed"]
    assert results["failed"] == ["bad"]
