"""Frame-level visibility judgment: FOV containment (H+V) + O-vis occlusion.

Used by:
  - Plan 2 flag verifier for `leaves_camera_fov` / `stays_in_camera_fov`
  - Plan 1.5 metadata (source_visible_from_camera_per_frame,
                        source_occluded_by_furniture_per_frame)
  - Topdown 2D render for accurate FOV cone visualization
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def _mic_local_direction(src_xyz, mic_pos, mic_yaw_deg):
    """Return (azimuth_deg, elevation_deg, distance_m) with mic-forward = +X_local
    after rotating by mic_yaw_deg CCW about Z.
    """
    v = np.asarray(src_xyz, dtype=np.float64) - np.asarray(mic_pos, dtype=np.float64)
    yr = np.deg2rad(mic_yaw_deg)
    c, s = np.cos(yr), np.sin(yr)
    # World -> mic-local: rotate by -mic_yaw
    x_local = c * v[..., 0] + s * v[..., 1]
    y_local = -s * v[..., 0] + c * v[..., 1]
    z_local = v[..., 2]
    dist = np.linalg.norm(v, axis=-1)
    azi = np.degrees(np.arctan2(y_local, x_local))
    ele = np.degrees(np.arctan2(z_local, np.hypot(x_local, y_local)))
    return azi, ele, dist


def _ray_intersects_aabb(origin, direction, aabb_min, aabb_max,
                          t_min: float = 0.0, t_max: float = 1.0) -> bool:
    """Slab-based ray-AABB intersection. Ray parameter t in [t_min, t_max]
    (where t=0 is origin, t=1 is origin+direction).
    Returns True if the ray segment enters the box at any t in [t_min, t_max].
    """
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    mn = np.asarray(aabb_min, dtype=np.float64)
    mx = np.asarray(aabb_max, dtype=np.float64)
    tmin = t_min
    tmax = t_max
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if o[i] < mn[i] or o[i] > mx[i]:
                return False
            continue
        t1 = (mn[i] - o[i]) / d[i]
        t2 = (mx[i] - o[i]) / d[i]
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
        if tmax < tmin:
            return False
    return tmax >= tmin


def frame_visibility(src_xyz, mic_pos, mic_yaw_deg: float,
                      fov_h_deg: float = 90.0, fov_v_deg: float = 60.0,
                      obstacles_xyz: Optional[Iterable[Tuple]] = None) -> dict:
    """Return {'in_fov', 'occluded_by_furniture', 'visible'} for one frame.

    Args:
      src_xyz: (x, y, z) SSOT meters
      mic_pos: (x, y, z) SSOT meters
      mic_yaw_deg: mic-forward at yaw=0 is +X world; yaw rotates CCW in XY
      fov_h_deg: total horizontal FOV
      fov_v_deg: total vertical FOV
      obstacles_xyz: iterable of (aabb_min, aabb_max) tuples in SSOT meters
    """
    azi, ele, _ = _mic_local_direction(src_xyz, mic_pos, mic_yaw_deg)
    in_fov = (abs(float(azi)) <= fov_h_deg / 2.0
              and abs(float(ele)) <= fov_v_deg / 2.0)

    occluded = False
    if in_fov and obstacles_xyz is not None:
        origin = np.asarray(mic_pos, dtype=np.float64)
        target = np.asarray(src_xyz, dtype=np.float64)
        direction = target - origin
        for aabb_min, aabb_max in obstacles_xyz:
            # Only count as occluded if the box is between mic and source
            # (t_min > small epsilon to skip mic's own bbox if any)
            if _ray_intersects_aabb(origin, direction, aabb_min, aabb_max,
                                     t_min=0.02, t_max=0.98):
                occluded = True
                break
    return {
        "in_fov": bool(in_fov),
        "occluded_by_furniture": bool(occluded),
        "visible": bool(in_fov and not occluded),
    }


def batch_frame_visibility(src_xyz_array, mic_pos, mic_yaw_deg: float,
                            fov_h_deg: float = 90.0, fov_v_deg: float = 60.0,
                            obstacles_xyz: Optional[Iterable[Tuple]] = None) -> dict:
    """Same as frame_visibility but vectorized over an array of source
    positions (n_frames, 3). Returns dict of np.ndarray of shape (n_frames,).
    """
    src = np.asarray(src_xyz_array, dtype=np.float64)
    n = src.shape[0]
    in_fov = np.zeros(n, dtype=bool)
    occluded = np.zeros(n, dtype=bool)
    visible = np.zeros(n, dtype=bool)
    # Materialize obstacles once so a generator doesn't get exhausted per frame
    obs_list = list(obstacles_xyz) if obstacles_xyz is not None else None
    for i in range(n):
        r = frame_visibility(src[i], mic_pos, mic_yaw_deg,
                              fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg,
                              obstacles_xyz=obs_list)
        in_fov[i] = r["in_fov"]
        occluded[i] = r["occluded_by_furniture"]
        visible[i] = r["visible"]
    return {"in_fov": in_fov, "occluded_by_furniture": occluded, "visible": visible}
