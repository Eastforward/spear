"""12 boolean flag functions for scene classification.

Each function's signature takes only what it needs; extra keyword args are
tolerated via **kwargs so orchestrators can pass a superset dict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
from visibility import batch_frame_visibility  # noqa: E402


ALL_FLAGS = [
    # Group A: occlusion
    "occluded_by_furniture", "occluded_by_wall", "never_occluded",
    # Group B: FOV
    "leaves_camera_fov", "stays_in_camera_fov",
    # Group C: spatial
    "crosses_azimuth_zero", "passes_close_to_mic", "far_from_mic_whole_clip",
    # Group D: motion
    "stationary", "steady_walk", "stop_and_go",
    # Group E: multi-source
    "sources_pass_each_other",
]

# ---- Occlusion helpers ----

def _visibility_arrays(traj_xyz, mic_pos, mic_yaw_deg, fov_h_deg, fov_v_deg,
                        furniture_bboxes, wall_bboxes):
    obstacles = list(furniture_bboxes) + list(wall_bboxes)
    return batch_frame_visibility(
        src_xyz_array=np.asarray(traj_xyz), mic_pos=mic_pos, mic_yaw_deg=mic_yaw_deg,
        fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg, obstacles_xyz=obstacles,
    )


def is_occluded_by_furniture(traj_xyz, mic_pos, mic_yaw_deg,
                               fov_h_deg, fov_v_deg,
                               furniture_bboxes, wall_bboxes=(), **kw):
    """True if any frame's ray from mic to source enters a furniture bbox."""
    vis_furn_only = batch_frame_visibility(
        src_xyz_array=np.asarray(traj_xyz), mic_pos=mic_pos,
        mic_yaw_deg=mic_yaw_deg, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg,
        obstacles_xyz=list(furniture_bboxes),
    )
    return bool(vis_furn_only["occluded_by_furniture"].any())


def is_occluded_by_wall(traj_xyz, mic_pos, mic_yaw_deg, fov_h_deg, fov_v_deg,
                          wall_bboxes, furniture_bboxes=(), **kw):
    vis_wall_only = batch_frame_visibility(
        src_xyz_array=np.asarray(traj_xyz), mic_pos=mic_pos,
        mic_yaw_deg=mic_yaw_deg, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg,
        obstacles_xyz=list(wall_bboxes),
    )
    return bool(vis_wall_only["occluded_by_furniture"].any())  # same field name


def is_never_occluded(traj_xyz, mic_pos, mic_yaw_deg, fov_h_deg, fov_v_deg,
                        furniture_bboxes, wall_bboxes, **kw):
    vis = _visibility_arrays(traj_xyz, mic_pos, mic_yaw_deg, fov_h_deg, fov_v_deg,
                              furniture_bboxes, wall_bboxes)
    return bool(not vis["occluded_by_furniture"].any())


# ---- FOV ----

def is_leaves_camera_fov(traj_xyz, mic_pos, mic_yaw_deg, fov_h_deg, fov_v_deg,
                           furniture_bboxes=(), wall_bboxes=(), **kw):
    vis = _visibility_arrays(traj_xyz, mic_pos, mic_yaw_deg, fov_h_deg, fov_v_deg,
                              furniture_bboxes, wall_bboxes)
    return bool(not vis["in_fov"].all())


def is_stays_in_camera_fov(traj_xyz, mic_pos, mic_yaw_deg, fov_h_deg, fov_v_deg,
                             furniture_bboxes=(), wall_bboxes=(), **kw):
    vis = _visibility_arrays(traj_xyz, mic_pos, mic_yaw_deg, fov_h_deg, fov_v_deg,
                              furniture_bboxes, wall_bboxes)
    return bool(vis["in_fov"].all())


# ---- Spatial ----

def is_crosses_azimuth_zero(traj_xyz, mic_pos, mic_yaw_deg, **kw):
    """True if source's mic-local azimuth changes sign at any frame."""
    v = np.asarray(traj_xyz) - np.asarray(mic_pos)
    yr = np.deg2rad(mic_yaw_deg)
    c, s = np.cos(yr), np.sin(yr)
    x_local = c * v[:, 0] + s * v[:, 1]
    y_local = -s * v[:, 0] + c * v[:, 1]
    azi = np.arctan2(y_local, x_local)
    signs = np.sign(azi)
    return bool(1 in signs and -1 in signs)


def is_passes_close_to_mic(traj_xyz, mic_pos, threshold_m=1.0, **kw):
    v = np.asarray(traj_xyz) - np.asarray(mic_pos)
    dist = np.linalg.norm(v, axis=1)
    return bool(dist.min() < threshold_m)


def is_far_from_mic_whole_clip(traj_xyz, mic_pos, threshold_m=4.0, **kw):
    v = np.asarray(traj_xyz) - np.asarray(mic_pos)
    dist = np.linalg.norm(v, axis=1)
    return bool(dist.min() > threshold_m)


# ---- Motion ----

def _speeds_mps(traj_xyz, fps):
    v = np.diff(np.asarray(traj_xyz), axis=0)
    dt = 1.0 / fps
    dist_per_frame = np.linalg.norm(v, axis=1)
    return dist_per_frame / dt


def is_stationary(traj_xyz, fps=15, threshold_mps=0.1, **kw):
    speeds = _speeds_mps(traj_xyz, fps)
    return bool(speeds.mean() < threshold_mps)


def is_steady_walk(traj_xyz, fps=15,
                    min_mean_speed=0.15, max_variance_ratio=0.4, **kw):
    speeds = _speeds_mps(traj_xyz, fps)
    if len(speeds) < 3:
        return False
    mean_s = speeds.mean()
    if mean_s < min_mean_speed:
        return False
    var_ratio = speeds.std() / max(mean_s, 1e-6)
    return bool(var_ratio < max_variance_ratio)


def is_stop_and_go(traj_xyz, fps=15,
                    min_stopped_frames=3, min_moving_frames=3,
                    stop_threshold_mps=0.05, **kw):
    """True if the trajectory has both clearly-stopped and clearly-moving segments."""
    speeds = _speeds_mps(traj_xyz, fps)
    stopped = speeds < stop_threshold_mps
    moving = speeds >= stop_threshold_mps
    return bool(stopped.sum() >= min_stopped_frames and moving.sum() >= min_moving_frames)


# ---- Multi-source ----

def is_sources_pass_each_other(traj_xyz_a, traj_xyz_b, threshold_m=0.5, **kw):
    """True if two sources' minimum inter-source distance is below threshold."""
    a = np.asarray(traj_xyz_a); b = np.asarray(traj_xyz_b)
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n] - b[:n], axis=1)
    return bool(d.min() < threshold_m)
