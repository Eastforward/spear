"""World-space source trajectories for the animated dog.

`gpurir_trajectory` is a byte-identical replica of
gen_rir_multiscene_v77.get_pos_traj so video and audio (given same seed and
params) describe the SAME trajectory shape. See spec Q13=C and Data Flow §
for the sub-sampling contract.

`waypoint_trajectory` (Task 4) is the user-controlled override.
`compute_yaw_from_positions` (Task 4) is the tangent-direction helper.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)

# ---- byte-identical constants from v77 gen_rir_multiscene_v77 -------------
# DO NOT CHANGE THESE VALUES without a matching change in v77 or you break
# the same-seed cross-modal alignment contract (Q13=C, spec F7).
SPEED_BUCKET_STEP = {"A": 5.0, "B": 15.0, "C": 30.0, "D": 50.0}
MIC_HEIGHT_M = 1.2  # matches v77.MIC_HEIGHT
_N_ANCHORS = 10


def gpurir_trajectory(
    *,
    room_size_m,
    n_frames,
    speed_bucket="B",
    source_height_m=0.45,
    traj_aug=True,
    seed,
    traj_pts_full=200,
    large_angle=360.0,
):
    """Return (n_frames, 3) world-frame meters.

    Internally: build a full-resolution length-`traj_pts_full` trajectory using
    the SAME random consumption order as v77.get_pos_traj, then subsample to
    `n_frames` via positions_full[i * traj_pts_full // n_frames].

    Byte-identical alignment with v77.get_pos_traj: v77 uses the GLOBAL
    np.random state. We save/restore the caller's global state so we don't
    clobber it, then seed the global state with `seed` and consume in the
    SAME order as v77's function body.
    """
    room_sz = list(room_size_m)

    _saved_state = np.random.get_state()
    try:
        np.random.seed(int(seed))

        original_az = np.zeros(_N_ANCHORS)
        original_distance = np.zeros(_N_ANCHORS)
        original_az[0] = np.random.uniform(0, large_angle)
        # v77._source_distance also uses np.random.uniform on the global state
        d_max = min(room_sz[0], room_sz[1]) / 2.0 - 0.5
        original_distance[0] = float(np.random.uniform(1.0, max(1.5, d_max)))
        step = SPEED_BUCKET_STEP.get(speed_bucket, 15.0)
        for i in range(1, _N_ANCHORS):
            if traj_aug and np.random.rand() < 0.15:  # random pause (v77 threshold)
                original_az[i] = original_az[i - 1]
                original_distance[i] = original_distance[i - 1]
                continue
            potential_az = original_az[i - 1] + np.random.uniform(-step, step)
            if potential_az < 0:
                original_az[i] = -potential_az
            elif potential_az > large_angle:
                original_az[i] = 2 * large_angle - potential_az
            else:
                original_az[i] = potential_az
            original_az[i] = np.clip(original_az[i], 0, large_angle)
            original_distance[i] = original_distance[i - 1] + np.random.uniform(-0.02, 0.05)
    finally:
        np.random.set_state(_saved_state)

    # ---- interpolate anchors → full-res grid ----
    time_original = np.linspace(0, 1, _N_ANCHORS)
    time_smooth = np.linspace(0, 1, traj_pts_full)
    kind = "cubic" if traj_aug else "linear"
    smooth_azimuth = np.mod(
        interp1d(time_original, original_az, kind=kind)(time_smooth), large_angle
    )
    smooth_distance = interp1d(time_original, original_distance, kind=kind)(time_smooth)

    # ---- polar → cartesian (in room-center coords, add cx/cy) ----
    theta = smooth_azimuth * np.pi / 180.0
    cx, cy = room_sz[0] / 2.0, room_sz[1] / 2.0
    x = smooth_distance * np.cos(theta) + cx
    y = smooth_distance * np.sin(theta) + cy
    z = np.full(traj_pts_full, float(source_height_m))
    positions_full = np.stack([x, y, z], axis=1)

    # ---- Q13=C sub-sample ----
    if n_frames == traj_pts_full:
        return positions_full
    idx = np.array([i * traj_pts_full // n_frames for i in range(n_frames)])
    return positions_full[idx]


def waypoint_trajectory(
    *,
    waypoints_m,
    n_frames,
    room_size_m,
    source_height_m=0.45,
    kind="cubic",
    wall_margin_m=0.1,
):
    """Interpolate a list of (x,y) or (x,y,z) waypoints (meters) to n_frames points.

    Raises ValueError if fewer than 2 waypoints. Any waypoint outside
    [wall_margin_m, room_dim - wall_margin_m] is clipped and a WARNING is logged.
    """
    if len(waypoints_m) < 2:
        raise ValueError(f"waypoint_trajectory needs >= 2 waypoints, got {len(waypoints_m)}")

    rx, ry, _rz = (float(v) for v in room_size_m)
    lo_x, hi_x = wall_margin_m, rx - wall_margin_m
    lo_y, hi_y = wall_margin_m, ry - wall_margin_m

    clipped = []
    any_clip = False
    for wp in waypoints_m:
        wp = list(wp)
        if len(wp) == 2:
            wp = [wp[0], wp[1], source_height_m]
        elif len(wp) != 3:
            raise ValueError(f"waypoint must be (x,y) or (x,y,z), got {wp!r}")
        new_x = float(np.clip(wp[0], lo_x, hi_x))
        new_y = float(np.clip(wp[1], lo_y, hi_y))
        if new_x != wp[0] or new_y != wp[1]:
            any_clip = True
        clipped.append([new_x, new_y, float(wp[2])])

    if any_clip:
        log.warning(
            "waypoint_trajectory: clipped one or more waypoints to room bounds "
            "[%.2f, %.2f] x [%.2f, %.2f]", lo_x, hi_x, lo_y, hi_y,
        )

    wp_arr = np.array(clipped)  # (K, 3)
    K = wp_arr.shape[0]
    t_orig = np.linspace(0.0, 1.0, K)
    t_smooth = np.linspace(0.0, 1.0, n_frames)
    # cubic needs K>=4; fall back to linear if fewer
    use_kind = kind if (kind != "cubic" or K >= 4) else "linear"
    xs = interp1d(t_orig, wp_arr[:, 0], kind=use_kind)(t_smooth)
    ys = interp1d(t_orig, wp_arr[:, 1], kind=use_kind)(t_smooth)
    zs = interp1d(t_orig, wp_arr[:, 2], kind=use_kind)(t_smooth)
    return np.stack([xs, ys, zs], axis=1)


def compute_yaw_from_positions(positions_m, smoothing_window=3):
    """Yaw (degrees) of the forward tangent. +x = 0 deg, +y = 90 deg.

    Frame 0 uses positions[0]→positions[1]. Last frame uses positions[-2]→[-1].
    A boxcar smoothing window of `smoothing_window` frames de-jitters tangent
    directions on noisy paths.
    """
    if positions_m.shape[0] < 2:
        raise ValueError("Need >= 2 positions to compute yaw")

    diffs = np.zeros_like(positions_m[:, :2])
    diffs[:-1] = positions_m[1:, :2] - positions_m[:-1, :2]
    diffs[-1] = diffs[-2]  # replicate last known direction

    if smoothing_window > 1:
        w = int(smoothing_window)
        pad = w // 2
        padded = np.pad(diffs, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(w) / w
        smoothed = np.stack([
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ], axis=1)
        diffs = smoothed[: positions_m.shape[0]]

    yaw_rad = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.rad2deg(yaw_rad)
