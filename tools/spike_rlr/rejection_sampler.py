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
from visibility import batch_frame_visibility  # noqa: E402


@dataclass
class SamplerConfig:
    n_clips_target: int
    per_flag_min_coverage: int = 3
    max_retries_per_clip: int = 20   # was 5; bumped because Limit-3 fix
    # (steady-plan validation) rejects more scenes up-front now
    min_visible_frames_per_source: int = 0
    min_joint_visible_frames: int = 0
    camera_yaw_search_step_deg: float = 2.0
    visible_fov_margin_deg: float = 0.0


def visible_frame_masks(scene_sample, trajectories, spec_template,
                        obstacle_context,
                        fov_margin_deg: float = 0.0) -> list[np.ndarray]:
    """Return per-source camera-visible source-center frame masks."""
    obstacles = (
        list(obstacle_context.get("furniture_bboxes", []))
        + list(obstacle_context.get("wall_bboxes", []))
    )
    fov_margin = max(0.0, float(fov_margin_deg))
    fov_h = max(1.0, float(spec_template.get("camera_fov_h_deg", 90.0))
                - 2.0 * fov_margin)
    fov_v = max(1.0, float(spec_template.get("camera_fov_v_deg", 60.0))
                - 2.0 * fov_margin)
    masks = []
    for traj in trajectories:
        vis = batch_frame_visibility(
            np.asarray(traj),
            scene_sample.mic_pos_m,
            scene_sample.mic_yaw_deg,
            fov_h_deg=fov_h,
            fov_v_deg=fov_v,
            obstacles_xyz=obstacles,
        )
        masks.append(np.asarray(vis["visible"], dtype=bool))
    return masks


def visible_frame_counts(scene_sample, trajectories, spec_template,
                         obstacle_context,
                         fov_margin_deg: float = 0.0) -> list[int]:
    """Count camera-visible source-center frames for each trajectory."""
    counts = []
    for mask in visible_frame_masks(
        scene_sample, trajectories, spec_template, obstacle_context,
        fov_margin_deg=fov_margin_deg,
    ):
        counts.append(int(mask.sum()))
    return counts


def joint_visible_frame_count(scene_sample, trajectories, spec_template,
                              obstacle_context,
                              fov_margin_deg: float = 0.0) -> int:
    """Count frames where all source centers are visible simultaneously."""
    masks = visible_frame_masks(
        scene_sample, trajectories, spec_template, obstacle_context,
        fov_margin_deg=fov_margin_deg,
    )
    if not masks:
        return 0
    return int(np.logical_and.reduce(masks).sum())


def optimize_camera_yaw_for_visible_sources(
    scene_sample,
    trajectories,
    spec_template,
    obstacle_context,
    yaw_step_deg: float = 2.0,
    fov_margin_deg: float = 0.0,
    prefer_joint_visible: bool = False,
) -> list[int]:
    """Choose the camera yaw that maximizes per-source visible frames.

    This is deterministic target sampling over the camera-yaw space. It is
    used by review-visible generation so "make every source auditable" is not
    left to whatever random yaw happened to be sampled with the mic.
    """
    if not trajectories:
        return []
    if yaw_step_deg <= 0.0:
        raise ValueError("yaw_step_deg must be positive")

    old_yaw = float(scene_sample.mic_yaw_deg)
    best_yaw = old_yaw
    best_counts = visible_frame_counts(
        scene_sample, trajectories, spec_template, obstacle_context,
        fov_margin_deg=fov_margin_deg,
    )

    def score(counts: list[int], yaw: float) -> tuple:
        yaw_delta = abs(((yaw - old_yaw + 180.0) % 360.0) - 180.0)
        joint_count = joint_visible_frame_count(
            scene_sample, trajectories, spec_template, obstacle_context,
            fov_margin_deg=fov_margin_deg,
        )
        if prefer_joint_visible:
            return (joint_count, min(counts), sum(counts), -yaw_delta)
        return (min(counts), sum(counts), joint_count, -yaw_delta)

    best_score = score(best_counts, best_yaw)
    candidate_yaws = np.arange(0.0, 360.0, yaw_step_deg, dtype=np.float64)
    for yaw in candidate_yaws:
        scene_sample.mic_yaw_deg = float(yaw)
        counts = visible_frame_counts(
            scene_sample, trajectories, spec_template, obstacle_context,
            fov_margin_deg=fov_margin_deg,
        )
        candidate_score = score(counts, float(yaw))
        if candidate_score > best_score:
            best_score = candidate_score
            best_yaw = float(yaw)
            best_counts = counts

    scene_sample.mic_yaw_deg = best_yaw % 360.0
    return best_counts


def meets_min_visible_frames(scene_sample, trajectories, spec_template,
                             obstacle_context,
                             min_visible_frames: int = 0,
                             fov_margin_deg: float = 0.0,
                             min_joint_visible_frames: int = 0) -> bool:
    """Review-only gate: require every source to be visibly auditable."""
    if min_visible_frames <= 0 and min_joint_visible_frames <= 0:
        return True
    if not trajectories:
        return False
    counts = visible_frame_counts(
        scene_sample, trajectories, spec_template, obstacle_context,
        fov_margin_deg=fov_margin_deg,
    )
    if min_visible_frames > 0 and any(
        count < min_visible_frames for count in counts
    ):
        return False
    if min_joint_visible_frames > 0 and joint_visible_frame_count(
        scene_sample, trajectories, spec_template, obstacle_context,
        fov_margin_deg=fov_margin_deg,
    ) < min_joint_visible_frames:
        return False
    return True


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
            motion_styles = []
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
                    "valid_regions": spec_template.get("valid_regions"),
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
                    src["motion_style"] = str(motion_style)
                    trajectories.append(traj)
                    motion_styles.append(str(motion_style))
                except RuntimeError:
                    failed = True
                    break
            if failed:
                continue
            if (config.min_visible_frames_per_source > 0
                    or config.min_joint_visible_frames > 0):
                optimize_camera_yaw_for_visible_sources(
                    scene_sample,
                    trajectories,
                    spec_template,
                    obstacle_context,
                    yaw_step_deg=config.camera_yaw_search_step_deg,
                    fov_margin_deg=config.visible_fov_margin_deg,
                    prefer_joint_visible=config.min_joint_visible_frames > 0,
                )
            if not meets_min_visible_frames(
                scene_sample,
                trajectories,
                spec_template,
                obstacle_context,
                config.min_visible_frames_per_source,
                fov_margin_deg=config.visible_fov_margin_deg,
                min_joint_visible_frames=config.min_joint_visible_frames,
            ):
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
                "motion_styles": motion_styles,
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
