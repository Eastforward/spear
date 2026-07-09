import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from trajectory_sampler import sample_trajectory, MOTION_STYLES  # noqa: E402


def _ctx(bounds=(-5, -5, 5, 5), obstacles=(), n_frames=30):
    return {"bounds_xy": bounds, "obstacles": list(obstacles),
             "n_frames": n_frames, "fps": 15}


def test_steady_produces_smooth_path():
    src = {"start_pos_m": [-3, 0, 0.45], "end_pos_m": [3, 0, 0.45]}
    ctx = _ctx()
    traj = sample_trajectory(src, ctx, np.random.default_rng(0),
                              motion_style="steady")
    assert traj.shape == (30, 3)
    assert np.allclose(traj[0], src["start_pos_m"], atol=0.1)
    assert np.allclose(traj[-1], src["end_pos_m"], atol=0.1)


def test_stationary_holds_start():
    src = {"start_pos_m": [-3, 0, 0.45], "end_pos_m": [3, 0, 0.45]}
    ctx = _ctx()
    traj = sample_trajectory(src, ctx, np.random.default_rng(0),
                              motion_style="stationary")
    expected = np.tile(np.asarray(src["start_pos_m"], dtype=np.float64),
                       (ctx["n_frames"], 1))
    assert np.allclose(traj, expected)
    speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) * ctx["fps"]
    assert np.allclose(speeds, 0.0)


def test_stop_and_go_has_stopped_and_moving_segments():
    src = {"start_pos_m": [-3, 0, 0.45], "end_pos_m": [3, 0, 0.45]}
    ctx = _ctx(n_frames=60)
    traj = sample_trajectory(src, ctx, np.random.default_rng(0),
                              motion_style="stop_and_go")
    speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) * 15  # m/s
    n_slow = (speeds < 0.05).sum()
    n_fast = (speeds >= 0.05).sum()
    assert n_slow >= 5, f"expected stopped frames, got {n_slow}"
    assert n_fast >= 5, f"expected moving frames, got {n_fast}"


def test_motion_styles_enum_has_expected_names():
    assert set(MOTION_STYLES) == {"steady", "stationary", "stop_and_go"}
