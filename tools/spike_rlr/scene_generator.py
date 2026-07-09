"""Layer-1 scene sampler for Plan 2 M1 dataset.

Draws:
  - mic pose (M-uniform + h-rand, uniform yaw [0, 360°))
  - n_sources (0/1/2 with 20/40/40 distribution)
  - per-source start/end positions (D-uniform + range 0.5-6.0 m from mic)
  - per-source audio assignment (from audio_library)

Returns a SceneSample dataclass consumed by trajectory_sampler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from source_asset_registry import resolve_source_pool


@dataclass
class SceneSample:
    mic_pos_m: tuple
    mic_yaw_deg: float
    source_specs: list  # list of dicts (see scene_generator docstring)
    rng_seed: int = 0


_N_SOURCE_WEIGHTS = np.array([0.20, 0.40, 0.40])  # for n=0,1,2


def _point_in_regions(x, y, valid_regions):
    if valid_regions is None:
        return True
    return any(x0 <= x <= x1 and y0 <= y <= y1
               for x0, y0, x1, y1 in valid_regions)


def _point_inside_obstacles(x, y, obstacles_xyz, inflate_m):
    for (x0, y0, _), (x1, y1, _) in obstacles_xyz:
        if (x0 - inflate_m <= x <= x1 + inflate_m
                and y0 - inflate_m <= y <= y1 + inflate_m):
            return True
    return False


def _segment_intersects_aabb(origin, target, aabb_min, aabb_max,
                             t_min=0.02, t_max=0.98):
    """Return True if the segment origin->target intersects an AABB."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(target, dtype=np.float64) - o
    mn = np.asarray(aabb_min, dtype=np.float64)
    mx = np.asarray(aabb_max, dtype=np.float64)
    lo = float(t_min)
    hi = float(t_max)
    for axis in range(3):
        if abs(d[axis]) < 1e-9:
            if o[axis] < mn[axis] or o[axis] > mx[axis]:
                return False
            continue
        t1 = (mn[axis] - o[axis]) / d[axis]
        t2 = (mx[axis] - o[axis]) / d[axis]
        lo = max(lo, min(t1, t2))
        hi = min(hi, max(t1, t2))
        if hi < lo:
            return False
    return hi >= lo


def _line_of_sight_blocked(mic_pos, src_pos, visibility_obstacles_xyz):
    for aabb_min, aabb_max in visibility_obstacles_xyz:
        if _segment_intersects_aabb(mic_pos, src_pos, aabb_min, aabb_max):
            return True
    return False


def sample_mic_pose(bounds_xy, obstacles_xyz, rng,
                     height_range=(0.5, 1.8),
                     inflate_m=0.3,
                     wall_margin_m=0.3,
                     max_tries=200,
                     valid_regions=None):
    x_min, y_min, x_max, y_max = bounds_xy
    valid_regions = [tuple(map(float, r)) for r in valid_regions] \
        if valid_regions is not None else None
    for _ in range(max_tries):
        x = rng.uniform(x_min + wall_margin_m, x_max - wall_margin_m)
        y = rng.uniform(y_min + wall_margin_m, y_max - wall_margin_m)
        if not _point_in_regions(x, y, valid_regions):
            continue
        if not _point_inside_obstacles(x, y, obstacles_xyz, inflate_m):
            z = rng.uniform(height_range[0], height_range[1])
            yaw = rng.uniform(0.0, 360.0)
            return (float(x), float(y), float(z)), float(yaw)
    raise RuntimeError("failed to sample mic pose in free space")


def sample_n_sources(rng) -> int:
    return int(rng.choice([0, 1, 2], p=_N_SOURCE_WEIGHTS))


def sample_source_position(bounds_xy, obstacles_xyz, mic_pos, rng,
                             distance_range=(0.5, 6.0),
                             z_m=0.45, inflate_m=0.3, max_tries=200,
                             valid_regions=None):
    x_min, y_min, x_max, y_max = bounds_xy
    d_min, d_max = distance_range
    mic_xy = np.array(mic_pos[:2])
    valid_regions = [tuple(map(float, r)) for r in valid_regions] \
        if valid_regions is not None else None
    for _ in range(max_tries):
        x = rng.uniform(x_min + 0.2, x_max - 0.2)
        y = rng.uniform(y_min + 0.2, y_max - 0.2)
        if not _point_in_regions(x, y, valid_regions):
            continue
        d = np.linalg.norm(np.array([x, y]) - mic_xy)
        if not (d_min <= d <= d_max):
            continue
        if not _point_inside_obstacles(x, y, obstacles_xyz, inflate_m):
            return (float(x), float(y), float(z_m))
    raise RuntimeError(
        f"failed to sample source position within {distance_range} m of mic"
    )


def sample_source_position_in_camera_sector(
    bounds_xy, obstacles_xyz, mic_pos, mic_yaw_deg, rng,
    distance_range=(0.5, 6.0), z_m=0.45,
    fov_h_deg=90.0, fov_margin_deg=0.0,
    inflate_m=0.3, max_tries=400, valid_regions=None,
    visibility_obstacles_xyz=None,
):
    """Sample a source point directly inside the camera's horizontal sector.

    This is used for review-visible clips so dual-source-in-view sampling is
    targeted instead of relying on full-apartment random positions.
    """
    x_min, y_min, x_max, y_max = bounds_xy
    d_min, d_max = distance_range
    half_fov = max(1.0, float(fov_h_deg) / 2.0 - max(0.0, fov_margin_deg))
    valid_regions = [tuple(map(float, r)) for r in valid_regions] \
        if valid_regions is not None else None
    visibility_obstacles = (
        [(tuple(a), tuple(b)) for a, b in visibility_obstacles_xyz]
        if visibility_obstacles_xyz is not None else obstacles_xyz
    )
    for _ in range(max_tries):
        rel_yaw = rng.uniform(-half_fov, half_fov)
        yaw = np.deg2rad(float(mic_yaw_deg) + rel_yaw)
        dist = rng.uniform(d_min, d_max)
        x = float(mic_pos[0] + dist * np.cos(yaw))
        y = float(mic_pos[1] + dist * np.sin(yaw))
        if not (x_min + 0.2 <= x <= x_max - 0.2
                and y_min + 0.2 <= y <= y_max - 0.2):
            continue
        if not _point_in_regions(x, y, valid_regions):
            continue
        if _point_inside_obstacles(x, y, obstacles_xyz, inflate_m):
            continue
        src_pos = (x, y, float(z_m))
        if _line_of_sight_blocked(mic_pos, src_pos, visibility_obstacles):
            continue
        return src_pos
    raise RuntimeError("failed to sample source position in camera sector")


def sample_scene(spec_template: dict, audio_lib, rng) -> SceneSample:
    bounds_xy = tuple(spec_template["bounds_xy"])
    obstacles_xyz = [(tuple(a), tuple(b)) for a, b in spec_template["obstacles"]]
    visibility_obstacles_xyz = [
        (tuple(a), tuple(b))
        for a, b in spec_template.get("visibility_obstacles", obstacles_xyz)
    ]
    valid_regions = spec_template.get("valid_regions")
    distance_range = tuple(spec_template.get("distance_range_m", (0.5, 6.0)))
    mic_h_range = tuple(spec_template.get("mic_height_range_m", (0.5, 1.8)))
    source_z = float(spec_template.get("source_height_m", 0.45))
    source_position_mode = spec_template.get("source_position_mode", "random")
    source_pool = resolve_source_pool(spec_template.get("source_pool") or [
        {"tag": "dog_golden", "audio_lookup": "dog_bark"},
        {"tag": "dog_husky", "audio_lookup": "music_piano"},
    ])
    scene_max_tries = int(spec_template.get(
        "scene_max_tries",
        50 if source_position_mode == "camera_sector" else 1,
    ))

    n_override = spec_template.get("n_sources_override")
    n = sample_n_sources(rng) if n_override is None else int(n_override)
    if n < 0:
        raise RuntimeError(f"n_sources_override must be >= 0, got {n}")
    if n > len(source_pool):
        raise RuntimeError(
            f"requested {n} sources but source_pool has only {len(source_pool)}"
        )
    if n > 0:
        chosen_indices = rng.choice(len(source_pool), size=n, replace=False)
    else:
        chosen_indices = []
    last_error = None
    for _ in range(scene_max_tries):
        mic_pos, mic_yaw = sample_mic_pose(bounds_xy, obstacles_xyz, rng,
                                           height_range=mic_h_range,
                                           valid_regions=valid_regions)
        source_specs = []
        try:
            for pool_idx in chosen_indices:
                pool_entry = source_pool[int(pool_idx)]
                tag = pool_entry["tag"]
                audio_cat = pool_entry["audio_lookup"]
                audio_sample = audio_lib.sample(audio_cat, rng)
                if source_position_mode == "camera_sector":
                    start = sample_source_position_in_camera_sector(
                        bounds_xy, obstacles_xyz, mic_pos, mic_yaw, rng,
                        distance_range=distance_range,
                        z_m=source_z,
                        fov_h_deg=float(
                            spec_template.get("camera_fov_h_deg", 90.0)
                        ),
                        fov_margin_deg=float(
                            spec_template.get("source_visible_fov_margin_deg", 0.0)
                        ),
                        valid_regions=valid_regions,
                        visibility_obstacles_xyz=visibility_obstacles_xyz,
                    )
                    end = sample_source_position_in_camera_sector(
                        bounds_xy, obstacles_xyz, mic_pos, mic_yaw, rng,
                        distance_range=distance_range,
                        z_m=source_z,
                        fov_h_deg=float(
                            spec_template.get("camera_fov_h_deg", 90.0)
                        ),
                        fov_margin_deg=float(
                            spec_template.get("source_visible_fov_margin_deg", 0.0)
                        ),
                        valid_regions=valid_regions,
                        visibility_obstacles_xyz=visibility_obstacles_xyz,
                    )
                else:
                    start = sample_source_position(
                        bounds_xy, obstacles_xyz, mic_pos, rng,
                        distance_range=distance_range,
                        z_m=source_z,
                        valid_regions=valid_regions,
                    )
                    end = sample_source_position(
                        bounds_xy, obstacles_xyz, mic_pos, rng,
                        distance_range=distance_range,
                        z_m=source_z,
                        valid_regions=valid_regions,
                    )
                source_spec = {
                    "tag": tag,
                    "audio_lookup": audio_cat,
                    "audio_path": str(audio_sample.path),
                    "is_synthetic": audio_sample.is_synthetic,
                    "category": audio_sample.category,
                    "start_pos_m": list(start),
                    "end_pos_m": list(end),
                }
                if "asset_id" in pool_entry:
                    source_spec["asset_id"] = pool_entry["asset_id"]
                if "asset_class" in pool_entry:
                    source_spec["asset_class"] = pool_entry["asset_class"]
                if "category" in pool_entry:
                    source_spec["asset_category"] = pool_entry["category"]
                if "family" in pool_entry:
                    source_spec["asset_family"] = pool_entry["family"]
                source_specs.append(source_spec)
        except RuntimeError as exc:
            last_error = exc
            continue
        return SceneSample(
            mic_pos_m=mic_pos, mic_yaw_deg=mic_yaw, source_specs=source_specs,
            rng_seed=int(rng.integers(0, 2**31 - 1)),
        )
    raise RuntimeError(f"failed to sample scene: {last_error}")
