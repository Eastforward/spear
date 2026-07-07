"""Tests for tools/spike_rlr/path_planner.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from path_planner import plan_path_2d  # noqa: E402


def test_straight_line_when_no_obstacles():
    path = plan_path_2d(
        start_xy=(0.0, 0.0), end_xy=(4.0, 0.0),
        obstacles_xy=[],
        bounds_xy=(-1, -1, 5, 1),
        n_frames=20,
    )
    assert path.shape == (20, 2)
    # Endpoints correct
    assert np.allclose(path[0], [0.0, 0.0], atol=0.01)
    assert np.allclose(path[-1], [4.0, 0.0], atol=0.01)
    # Straight-line: max |y| stays near 0
    assert np.max(np.abs(path[:, 1])) < 0.15


def test_path_avoids_single_obstacle():
    # Straight line X=0 -> X=4 at Y=0; put a 1x1 obstacle centered on Y=0.
    obs = [(1.5, -0.5, 2.5, 0.5)]
    path = plan_path_2d(
        start_xy=(0.0, 0.0), end_xy=(4.0, 0.0),
        obstacles_xy=obs,
        bounds_xy=(-1, -3, 5, 3),
        cell_m=0.10, inflate_m=0.10,
        n_frames=50,
    )
    assert path.shape == (50, 2)
    # No point should be inside the raw obstacle
    for x, y in path:
        assert not (1.5 <= x <= 2.5 and -0.5 <= y <= 0.5), \
            f"point ({x:.2f}, {y:.2f}) inside obstacle"


def test_z_column_appended():
    path = plan_path_2d(
        start_xy=(0.0, 0.0), end_xy=(2.0, 0.0),
        obstacles_xy=[], bounds_xy=(-1, -1, 3, 1),
        n_frames=10, z_m=0.45,
    )
    assert path.shape == (10, 3)
    assert np.allclose(path[:, 2], 0.45)


def test_raises_when_start_inside_obstacle():
    with pytest.raises(RuntimeError, match="start"):
        plan_path_2d(
            start_xy=(0.5, 0.5), end_xy=(4.0, 0.0),
            obstacles_xy=[(0.0, 0.0, 1.0, 1.0)],
            bounds_xy=(-1, -1, 5, 1),
        )


def test_raises_when_end_inside_obstacle():
    with pytest.raises(RuntimeError, match="end"):
        plan_path_2d(
            start_xy=(0.0, 0.0), end_xy=(2.0, 2.0),
            obstacles_xy=[(1.5, 1.5, 2.5, 2.5)],
            bounds_xy=(-1, -1, 5, 5),
        )


def test_path_length_at_least_straight_distance():
    """Detour path must be at least as long as the Euclidean start-to-end distance."""
    obs = [(1.5, -0.5, 2.5, 0.5)]
    path = plan_path_2d(
        start_xy=(0.0, 0.0), end_xy=(4.0, 0.0),
        obstacles_xy=obs,
        bounds_xy=(-1, -3, 5, 3),
        cell_m=0.10, inflate_m=0.10,
        n_frames=50,
    )
    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = seg_lens.sum()
    assert total >= 4.0  # must be at least the euclidean distance


def test_smoothed_path_is_smooth():
    """Chaikin-smoothed path should have small step-to-step direction changes."""
    obs = [(1.5, -0.5, 2.5, 0.5)]
    path = plan_path_2d(
        start_xy=(0.0, 0.0), end_xy=(4.0, 0.0),
        obstacles_xy=obs,
        bounds_xy=(-1, -3, 5, 3),
        cell_m=0.10, inflate_m=0.10,
        n_frames=100,
        chaikin_iters=2,
    )
    # Direction change per step
    diffs = np.diff(path, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    ang_diff = np.diff(np.unwrap(angles))
    # Smoothed path: no huge kinks
    assert np.max(np.abs(ang_diff)) < np.radians(30), \
        f"max direction change {np.degrees(np.max(np.abs(ang_diff))):.1f} deg too big"


def test_n_frames_exact():
    for n in (10, 50, 75, 200):
        path = plan_path_2d(
            start_xy=(0.0, 0.0), end_xy=(3.0, 0.0),
            obstacles_xy=[], bounds_xy=(-1, -1, 4, 1),
            n_frames=n,
        )
        assert len(path) == n
