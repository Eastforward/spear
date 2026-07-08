"""Per-source trajectory sampler wrapping path_planner + motion styles.

Motion styles injected:
  - "steady":       full plan_path_2d output resampled to n_frames
  - "stationary":   holds start position for all frames (slight noise ±0.05m)
  - "stop_and_go":  planned path split into 3 segments; middle segment
                     replaced with a hold (stopped), start/end walking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from path_planner import plan_path_2d  # noqa: E402


MOTION_STYLES = ("steady", "stationary", "stop_and_go")


def _obstacles_to_xy(obstacles):
    """Coerce various obstacle formats to (x0, y0, x1, y1) tuples for planner.

    Supported input entries:
      - ((xmin,ymin,zmin), (xmax,ymax,zmax))  3D AABB
      - (xmin,ymin,xmax,ymax)                  already 2D
    """
    out = []
    for entry in obstacles:
        if len(entry) == 2:
            a, b = entry
            out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
        elif len(entry) == 4:
            x0, y0, x1, y1 = entry
            out.append((float(x0), float(y0), float(x1), float(y1)))
        else:
            raise ValueError(f"unsupported obstacle entry: {entry}")
    return out


def sample_trajectory(source_spec, planning_context, rng,
                       motion_style: str = "steady") -> np.ndarray:
    """Sample one trajectory for a source."""
    assert motion_style in MOTION_STYLES, f"unknown motion_style {motion_style!r}"

    start = np.asarray(source_spec["start_pos_m"], dtype=np.float64)
    end = np.asarray(source_spec["end_pos_m"], dtype=np.float64)
    n_frames = int(planning_context["n_frames"])
    z = float(start[2])

    if motion_style == "stationary":
        base = np.tile(start, (n_frames, 1)).astype(np.float64)
        # Small independent jitter to simulate breathing/sway (±5 cm XY)
        jitter = rng.normal(0, 0.02, size=(n_frames, 2))
        base[:, 0] += jitter[:, 0]
        base[:, 1] += jitter[:, 1]
        return base

    # For steady + stop_and_go, first plan the full path
    bounds_xy = tuple(planning_context["bounds_xy"])
    obstacles_xy = _obstacles_to_xy(planning_context.get("obstacles", []))

    if motion_style == "steady":
        return plan_path_2d(
            start_xy=(start[0], start[1]),
            end_xy=(end[0], end[1]),
            obstacles_xy=obstacles_xy,
            bounds_xy=bounds_xy,
            cell_m=0.15, inflate_m=0.15,  # match scene_two_dogs_apartment
            n_frames=n_frames,
            chaikin_iters=2,
            z_m=z,
        )

    # stop_and_go: plan full path, then replace middle frames with a hold
    full_traj = plan_path_2d(
        start_xy=(start[0], start[1]),
        end_xy=(end[0], end[1]),
        obstacles_xy=obstacles_xy,
        bounds_xy=bounds_xy,
        cell_m=0.15, inflate_m=0.15,  # match scene_two_dogs_apartment
        n_frames=n_frames,
        chaikin_iters=2,
        z_m=z,
    )
    n_mid = n_frames // 3
    stop_start = int(rng.integers(n_frames // 4, n_frames // 3 + 1))
    stop_end = min(stop_start + n_mid, n_frames - 1)
    for i in range(stop_start, stop_end):
        full_traj[i] = full_traj[stop_start]
    return full_traj
