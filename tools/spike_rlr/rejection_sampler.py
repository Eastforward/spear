"""Batch generator: sample N scenes, verify flags, count coverage.

For M1: emit exactly n_clips_target clips (no undersampling). Flag
coverage is a soft target — if the natural distribution doesn't cover
some flag ≥3 times, the sampler logs a warning but does not oversample
in Plan 2 (that's Plan 3 with I-in mode).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from scene_generator import sample_scene  # noqa: E402
from trajectory_sampler import sample_trajectory, MOTION_STYLES  # noqa: E402
from flag_verifier import verify_all_flags  # noqa: E402
from flag_definitions import ALL_FLAGS  # noqa: E402


@dataclass
class SamplerConfig:
    n_clips_target: int
    per_flag_min_coverage: int = 3
    max_retries_per_clip: int = 5


def generate_batch(config: SamplerConfig, spec_template, audio_lib, rng,
                     obstacle_context) -> list:
    """Sample n_clips clips + their trajectories + flag verdicts."""
    batch = []
    for i in range(config.n_clips_target):
        succeeded = False
        for attempt in range(config.max_retries_per_clip):
            try:
                scene_sample = sample_scene(spec_template, audio_lib, rng)
            except RuntimeError:
                continue
            trajectories = []
            failed = False
            for src in scene_sample.source_specs:
                motion_style = rng.choice(list(MOTION_STYLES),
                                           p=[0.7, 0.1, 0.2])  # steady dominant
                # Use furniture+walls for path planning so what the sampler
                # validates matches what the UE renderer's planner will do
                # (compose_two_dog_scene_apartment uses both).
                plan_obstacles = (
                    list(obstacle_context.get("furniture_bboxes", []))
                    + list(obstacle_context.get("wall_bboxes", []))
                )
                planning_ctx = {
                    "bounds_xy": spec_template["bounds_xy"],
                    "obstacles": [(bmin, bmax) for bmin, bmax in plan_obstacles],
                    "n_frames": spec_template.get("n_frames", 75),
                    "fps": spec_template.get("fps", 15),
                }
                try:
                    # Even for stationary motion, first try steady planning to
                    # validate reachability. UE side (compose_two_dog_scene_apartment)
                    # ALWAYS re-plans regardless of motion_style, so if steady
                    # fails here the UE render will also fail. This catches
                    # unreachable endpoints up-front instead of losing 30 s of
                    # UE render time.
                    _ = sample_trajectory(
                        source_spec=src, planning_context=planning_ctx,
                        rng=np.random.default_rng(0),  # cheap deterministic try
                        motion_style="steady",
                    )
                    # Now sample the actual motion style
                    traj = sample_trajectory(
                        source_spec=src, planning_context=planning_ctx,
                        rng=rng, motion_style=str(motion_style),
                    )
                    trajectories.append(traj)
                except RuntimeError:
                    failed = True
                    break
            if failed:
                continue
            # Compute flags
            stub_spec_for_verifier = {
                "mic": {"pos_m": list(scene_sample.mic_pos_m),
                         "yaw_deg": scene_sample.mic_yaw_deg},
                "camera_configs": [{"fov_deg": spec_template.get("camera_fov_h_deg", 90),
                                      "fov_v_deg": spec_template.get("camera_fov_v_deg", 60)}],
                "render_config": {"fps": spec_template.get("fps", 15)},
            }
            flags = verify_all_flags(
                spec_dict=stub_spec_for_verifier,
                trajectories=trajectories,
                furniture_bboxes=obstacle_context.get("furniture_bboxes", []),
                wall_bboxes=obstacle_context.get("wall_bboxes", []),
            )
            batch.append({
                "scene_sample": scene_sample,
                "trajectories": trajectories,
                "flags": flags,
            })
            succeeded = True
            break
        if not succeeded:
            raise RuntimeError(
                f"clip {i}: exhausted {config.max_retries_per_clip} retries"
            )

    # Coverage report
    coverage = {f: 0 for f in ALL_FLAGS}
    for clip in batch:
        for name, v in clip["flags"].items():
            if v:
                coverage[name] += 1
    print("[rejection_sampler] flag coverage:")
    for name in ALL_FLAGS:
        marker = "OK" if coverage[name] >= config.per_flag_min_coverage else "LOW"
        print(f"  [{marker}] {name}: {coverage[name]}")

    return batch
