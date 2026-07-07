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


@dataclass
class SceneSample:
    mic_pos_m: tuple
    mic_yaw_deg: float
    source_specs: list  # list of dicts (see scene_generator docstring)
    rng_seed: int = 0


_N_SOURCE_WEIGHTS = np.array([0.20, 0.40, 0.40])  # for n=0,1,2


def sample_mic_pose(bounds_xy, obstacles_xyz, rng,
                     height_range=(0.5, 1.8),
                     inflate_m=0.3,
                     wall_margin_m=0.3,
                     max_tries=200):
    x_min, y_min, x_max, y_max = bounds_xy
    for _ in range(max_tries):
        x = rng.uniform(x_min + wall_margin_m, x_max - wall_margin_m)
        y = rng.uniform(y_min + wall_margin_m, y_max - wall_margin_m)
        inside = False
        for (x0, y0, _), (x1, y1, _) in obstacles_xyz:
            if (x0 - inflate_m <= x <= x1 + inflate_m
                and y0 - inflate_m <= y <= y1 + inflate_m):
                inside = True
                break
        if not inside:
            z = rng.uniform(height_range[0], height_range[1])
            yaw = rng.uniform(0.0, 360.0)
            return (float(x), float(y), float(z)), float(yaw)
    raise RuntimeError("failed to sample mic pose in free space")


def sample_n_sources(rng) -> int:
    return int(rng.choice([0, 1, 2], p=_N_SOURCE_WEIGHTS))


def sample_source_position(bounds_xy, obstacles_xyz, mic_pos, rng,
                             distance_range=(0.5, 6.0),
                             z_m=0.45, inflate_m=0.3, max_tries=200):
    x_min, y_min, x_max, y_max = bounds_xy
    d_min, d_max = distance_range
    mic_xy = np.array(mic_pos[:2])
    for _ in range(max_tries):
        x = rng.uniform(x_min + 0.2, x_max - 0.2)
        y = rng.uniform(y_min + 0.2, y_max - 0.2)
        d = np.linalg.norm(np.array([x, y]) - mic_xy)
        if not (d_min <= d <= d_max):
            continue
        inside = False
        for (x0, y0, _), (x1, y1, _) in obstacles_xyz:
            if (x0 - inflate_m <= x <= x1 + inflate_m
                and y0 - inflate_m <= y <= y1 + inflate_m):
                inside = True
                break
        if not inside:
            return (float(x), float(y), float(z_m))
    raise RuntimeError(
        f"failed to sample source position within {distance_range} m of mic"
    )


def sample_scene(spec_template: dict, audio_lib, rng) -> SceneSample:
    bounds_xy = tuple(spec_template["bounds_xy"])
    obstacles_xyz = [(tuple(a), tuple(b)) for a, b in spec_template["obstacles"]]
    distance_range = tuple(spec_template.get("distance_range_m", (0.5, 6.0)))
    mic_h_range = tuple(spec_template.get("mic_height_range_m", (0.5, 1.8)))
    source_z = float(spec_template.get("source_height_m", 0.45))

    mic_pos, mic_yaw = sample_mic_pose(bounds_xy, obstacles_xyz, rng,
                                         height_range=mic_h_range)
    n = sample_n_sources(rng)
    source_specs = []
    for i in range(n):
        # For M1 initial: rig tags are hardcoded (dog_golden, dog_husky).
        # Plan 3 will let sampler pick from approved/ directory.
        tag = "dog_golden" if i == 0 else "dog_husky"
        audio_cat = "dog_bark" if tag == "dog_golden" else "music_piano"
        audio_sample = audio_lib.sample(audio_cat, rng)
        start = sample_source_position(bounds_xy, obstacles_xyz, mic_pos, rng,
                                         distance_range=distance_range,
                                         z_m=source_z)
        end = sample_source_position(bounds_xy, obstacles_xyz, mic_pos, rng,
                                       distance_range=distance_range,
                                       z_m=source_z)
        source_specs.append({
            "tag": tag,
            "audio_lookup": audio_cat,
            "audio_path": str(audio_sample.path),
            "is_synthetic": audio_sample.is_synthetic,
            "category": audio_sample.category,
            "start_pos_m": list(start),
            "end_pos_m": list(end),
        })
    return SceneSample(
        mic_pos_m=mic_pos, mic_yaw_deg=mic_yaw, source_specs=source_specs,
        rng_seed=int(rng.integers(0, 2**31 - 1)),
    )
