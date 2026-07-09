"""Deterministic event constraints for review/demo scene construction.

These checks are intentionally independent from random seeds. Scenario builders
create candidate trajectories, then this module says whether the declared event
contract is actually satisfied in listener-local coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from visibility import batch_frame_visibility


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    passed: bool
    details: dict


def _traj_array(trajectory_m) -> np.ndarray:
    traj = np.asarray(trajectory_m, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 3:
        raise ValueError(f"trajectory_m must have shape (n_frames, 3), got {traj.shape}")
    return traj


def listener_local_xy(points_xyz, mic_pos_m, mic_yaw_deg: float) -> np.ndarray:
    """Return columns (forward_m, left_m) in the listener/camera frame.

    Yaw 0 means camera forward is world +X. Positive local-left is world +Y
    when yaw is zero, matching visibility._mic_local_direction().
    """
    pts = _traj_array(points_xyz) if np.asarray(points_xyz).ndim == 2 else np.asarray([points_xyz], dtype=np.float64)
    mic = np.asarray(mic_pos_m, dtype=np.float64)
    v = pts[:, :2] - mic[:2]
    yr = np.deg2rad(float(mic_yaw_deg))
    c, s = np.cos(yr), np.sin(yr)
    forward = c * v[:, 0] + s * v[:, 1]
    left = -s * v[:, 0] + c * v[:, 1]
    return np.column_stack([forward, left])


def _basic_details(tag: str, local_xy: np.ndarray) -> dict:
    return {
        "tag": tag,
        "min_forward_m": float(np.min(local_xy[:, 0])),
        "max_forward_m": float(np.max(local_xy[:, 0])),
        "start_left_m": float(local_xy[0, 1]),
        "end_left_m": float(local_xy[-1, 1]),
    }


def constraint_behind_camera(
    tag: str,
    trajectory_m,
    mic_pos_m,
    mic_yaw_deg: float,
    margin_m: float = 0.05,
) -> ConstraintResult:
    traj = _traj_array(trajectory_m)
    local = listener_local_xy(traj, mic_pos_m, mic_yaw_deg)
    passed = bool(np.all(local[:, 0] < -abs(margin_m)))
    details = _basic_details(tag, local)
    details["margin_m"] = float(margin_m)
    return ConstraintResult(f"{tag}:behind_camera", passed, details)


def constraint_front_of_camera(
    tag: str,
    trajectory_m,
    mic_pos_m,
    mic_yaw_deg: float,
    margin_m: float = 0.05,
) -> ConstraintResult:
    traj = _traj_array(trajectory_m)
    local = listener_local_xy(traj, mic_pos_m, mic_yaw_deg)
    passed = bool(np.all(local[:, 0] > abs(margin_m)))
    details = _basic_details(tag, local)
    details["margin_m"] = float(margin_m)
    return ConstraintResult(f"{tag}:front_of_camera", passed, details)


def constraint_left_to_right(
    tag: str,
    trajectory_m,
    mic_pos_m,
    mic_yaw_deg: float,
    margin_m: float = 0.01,
) -> ConstraintResult:
    traj = _traj_array(trajectory_m)
    local = listener_local_xy(traj, mic_pos_m, mic_yaw_deg)
    diffs = np.diff(local[:, 1])
    passed = bool(local[0, 1] > abs(margin_m) and local[-1, 1] < -abs(margin_m))
    details = _basic_details(tag, local)
    details.update({
        "margin_m": float(margin_m),
        "net_left_delta_m": float(local[-1, 1] - local[0, 1]),
        "mean_left_step_m": float(np.mean(diffs)) if len(diffs) else 0.0,
    })
    return ConstraintResult(f"{tag}:left_to_right", passed, details)


def constraint_not_visible(
    tag: str,
    trajectory_m,
    mic_pos_m,
    mic_yaw_deg: float,
    fov_h_deg: float = 90.0,
    fov_v_deg: float = 60.0,
    obstacles_xyz: Iterable[tuple] | None = None,
    max_visible_frames: int = 0,
) -> ConstraintResult:
    traj = _traj_array(trajectory_m)
    vis = batch_frame_visibility(
        traj,
        mic_pos_m,
        mic_yaw_deg,
        fov_h_deg=fov_h_deg,
        fov_v_deg=fov_v_deg,
        obstacles_xyz=obstacles_xyz,
    )
    visible_frames = int(np.count_nonzero(vis["visible"]))
    in_fov_frames = int(np.count_nonzero(vis["in_fov"]))
    passed = bool(visible_frames <= int(max_visible_frames))
    return ConstraintResult(
        f"{tag}:not_visible",
        passed,
        {
            "tag": tag,
            "visible_frames": visible_frames,
            "in_fov_frames": in_fov_frames,
            "max_visible_frames": int(max_visible_frames),
        },
    )


def constraint_visible_min_frames(
    tag: str,
    trajectory_m,
    mic_pos_m,
    mic_yaw_deg: float,
    min_visible_frames: int,
    fov_h_deg: float = 90.0,
    fov_v_deg: float = 60.0,
    obstacles_xyz: Iterable[tuple] | None = None,
) -> ConstraintResult:
    traj = _traj_array(trajectory_m)
    vis = batch_frame_visibility(
        traj,
        mic_pos_m,
        mic_yaw_deg,
        fov_h_deg=fov_h_deg,
        fov_v_deg=fov_v_deg,
        obstacles_xyz=obstacles_xyz,
    )
    visible_frames = int(np.count_nonzero(vis["visible"]))
    passed = bool(visible_frames >= int(min_visible_frames))
    return ConstraintResult(
        f"{tag}:visible_min_frames",
        passed,
        {
            "tag": tag,
            "visible_frames": visible_frames,
            "min_visible_frames": int(min_visible_frames),
        },
    )


def constraint_in_fov_min_frames(
    tag: str,
    trajectory_m,
    mic_pos_m,
    mic_yaw_deg: float,
    min_in_fov_frames: int,
    fov_h_deg: float = 90.0,
    fov_v_deg: float = 60.0,
) -> ConstraintResult:
    traj = _traj_array(trajectory_m)
    vis = batch_frame_visibility(
        traj,
        mic_pos_m,
        mic_yaw_deg,
        fov_h_deg=fov_h_deg,
        fov_v_deg=fov_v_deg,
        obstacles_xyz=None,
    )
    in_fov_frames = int(np.count_nonzero(vis["in_fov"]))
    passed = bool(in_fov_frames >= int(min_in_fov_frames))
    return ConstraintResult(
        f"{tag}:in_fov_min_frames",
        passed,
        {
            "tag": tag,
            "in_fov_frames": in_fov_frames,
            "min_in_fov_frames": int(min_in_fov_frames),
        },
    )


def constraint_stationary(
    tag: str,
    trajectory_m,
    max_displacement_m: float = 0.03,
) -> ConstraintResult:
    traj = _traj_array(trajectory_m)
    displacement = np.linalg.norm(traj[:, :2] - traj[0, :2], axis=1)
    max_disp = float(np.max(displacement)) if len(displacement) else 0.0
    return ConstraintResult(
        f"{tag}:stationary",
        bool(max_disp <= float(max_displacement_m)),
        {
            "tag": tag,
            "max_displacement_m": max_disp,
            "limit_m": float(max_displacement_m),
        },
    )


def constraint_min_actor_distance(
    tag_a: str,
    trajectory_a_m,
    tag_b: str,
    trajectory_b_m,
    min_distance_m: float = 0.5,
) -> ConstraintResult:
    a = _traj_array(trajectory_a_m)
    b = _traj_array(trajectory_b_m)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"trajectory length mismatch: {a.shape[0]} vs {b.shape[0]}")
    distances = np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
    min_dist = float(np.min(distances)) if len(distances) else float("inf")
    return ConstraintResult(
        f"{tag_a}:{tag_b}:min_actor_distance",
        bool(min_dist >= float(min_distance_m)),
        {
            "tag_a": tag_a,
            "tag_b": tag_b,
            "min_distance_m": min_dist,
            "limit_m": float(min_distance_m),
            "frame": int(np.argmin(distances)) if len(distances) else None,
        },
    )


def constraint_no_aabb_intersections(
    tag: str,
    trajectory_m,
    obstacles_xy: Iterable[tuple[float, float, float, float]],
    margin_m: float = 0.0,
) -> ConstraintResult:
    traj = _traj_array(trajectory_m)
    hits = []
    for frame, (x, y, _z) in enumerate(traj):
        for idx, (x0, y0, x1, y1) in enumerate(obstacles_xy):
            if (
                x0 - margin_m <= x <= x1 + margin_m
                and y0 - margin_m <= y <= y1 + margin_m
            ):
                hits.append({"frame": int(frame), "obstacle_index": int(idx)})
                break
    return ConstraintResult(
        f"{tag}:no_aabb_intersections",
        len(hits) == 0,
        {"tag": tag, "hit_count": len(hits), "hits": hits[:10]},
    )


def verify_constraints(results: Iterable[ConstraintResult]) -> dict:
    items = list(results)
    failed = [r.name for r in items if not r.passed]
    return {
        "passed": len(failed) == 0,
        "failed": failed,
        "results": [
            {"name": r.name, "passed": bool(r.passed), "details": r.details}
            for r in items
        ],
    }
