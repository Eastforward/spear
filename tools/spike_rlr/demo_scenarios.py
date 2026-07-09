"""Deterministic review/demo scenario builders.

These builders are not dataset samplers. They create a named event directly
from listener pose + path-planner geometry, then verify the result with
event_constraints before returning a spec.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

_SPEAR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SPEAR_ROOT / "tools"))
sys.path.insert(0, str(_SPEAR_ROOT / "tools" / "spike_rlr"))

from animal_audio import resolve_animal_audio_path  # noqa: E402
from apartment_builtin_obstacles import (  # noqa: E402
    apartment_builtin_visual_obstacle_bboxes_xyz,
)
from event_constraints import (  # noqa: E402
    ConstraintResult,
    constraint_behind_camera,
    constraint_front_of_camera,
    constraint_in_fov_min_frames,
    constraint_left_to_right,
    constraint_min_actor_distance,
    constraint_no_aabb_intersections,
    constraint_not_visible,
    constraint_stationary,
    verify_constraints,
)
from flag_verifier import verify_flag_details  # noqa: E402
from path_planner import plan_path_2d  # noqa: E402
from scene_two_dogs_apartment import (  # noqa: E402
    _kept_furniture_bboxes,
    _planning_bounds,
    _shell_wall_bboxes,
    _static_obstacle_bboxes,
    _valid_region_bboxes,
)
from source_asset_registry import resolve_source_pool_entry  # noqa: E402
from speech_audio import pick_speech_sample  # noqa: E402
from visibility import batch_frame_visibility  # noqa: E402


DEFAULT_BASE_SPEC = _SPEAR_ROOT / "data" / "apartment_v1_spec.json"


class UnsatisfiableScenarioError(RuntimeError):
    """Raised when no deterministic candidate satisfies the requested event."""


def _load_apartment_planning_context(spec: dict) -> tuple[list, tuple, list]:
    cats = json.loads(
        (_SPEAR_ROOT / "tools" / "spike_rlr" / "apartment_furniture_categories.json").read_text()
    )
    obstacles = _static_obstacle_bboxes(spec, cats)
    bounds = _planning_bounds(spec)
    valid_regions = _valid_region_bboxes(spec)
    return obstacles, bounds, valid_regions


def _load_apartment_wall_planning_context(spec: dict) -> tuple[list, tuple, list]:
    """Planning context where only walls/room bounds are hard blockers.

    This is for review/demo events where the acoustic source is treated as a
    point. The animal mesh may visually clip furniture, but the source center
    must not pass through walls or leave the apartment valid regions.
    """
    obstacles = _shell_wall_bboxes(spec)
    bounds = _planning_bounds(spec)
    valid_regions = _valid_region_bboxes(spec)
    return obstacles, bounds, valid_regions


def _flag_obstacle_context(spec: dict) -> tuple[list, list]:
    cats = json.loads(
        (_SPEAR_ROOT / "tools" / "spike_rlr" / "apartment_furniture_categories.json").read_text()
    )
    furniture_xy = _kept_furniture_bboxes(spec, cats)
    furniture_xyz = [
        ((x0, y0, 0.0), (x1, y1, 1.5))
        for x0, y0, x1, y1 in furniture_xy
    ]
    furniture_xyz.extend(apartment_builtin_visual_obstacle_bboxes_xyz(spec))
    wall_xyz = [
        ((x0, y0, 0.0), (x1, y1, 2.8))
        for x0, y0, x1, y1 in _shell_wall_bboxes(spec)
    ]
    return furniture_xyz, wall_xyz


def compute_demo_flags(spec: dict) -> dict:
    """Compute the standard 12 Plan-2 flags for a deterministic demo spec."""
    return compute_demo_flag_details(spec)["aggregate"]


def compute_demo_flag_details(spec: dict) -> dict:
    """Compute aggregate, per-source, and pairwise flags for a demo spec."""
    furniture_bboxes, wall_bboxes = _flag_obstacle_context(spec)
    source_specs = [src for src in spec.get("sources", []) if "trajectory_m" in src]
    trajectories = [
        np.asarray(src["trajectory_m"], dtype=np.float64)
        for src in source_specs
    ]
    return verify_flag_details(
        spec_dict=spec,
        trajectories=trajectories,
        furniture_bboxes=furniture_bboxes,
        wall_bboxes=wall_bboxes,
        source_tags=[src.get("tag", f"source_{i:04d}") for i, src in enumerate(source_specs)],
    )


def _write_spec_and_flags(spec: dict, out_spec_path: str | Path | None) -> None:
    if out_spec_path is None:
        return
    out_path = Path(out_spec_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2) + "\n")
    details = compute_demo_flag_details(spec)
    (out_path.parent / "flags.json").write_text(
        json.dumps(details["aggregate"], indent=2) + "\n"
    )
    (out_path.parent / "flag_details.json").write_text(
        json.dumps(details, indent=2) + "\n"
    )


def _constant_trajectory(pos_m, n_frames: int) -> list[list[float]]:
    p = np.asarray(pos_m, dtype=np.float64)
    return np.tile(p, (n_frames, 1)).tolist()


def _planned_trajectory(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    z_m: float,
    n_frames: int,
    obstacles_xy: list,
    bounds_xy: tuple,
    valid_regions: list,
    clearance_m: float,
) -> np.ndarray:
    return plan_path_2d(
        start_xy=start_xy,
        end_xy=end_xy,
        obstacles_xy=obstacles_xy,
        bounds_xy=bounds_xy,
        cell_m=0.15,
        inflate_m=clearance_m,
        n_frames=n_frames,
        chaikin_iters=2,
        z_m=z_m,
        valid_xy_rects=valid_regions,
    )


def _dog_source(
    tag: str,
    audio_lookup: str,
    trajectory_m,
    wanted_anim: str,
    motion_style: str,
    notes: str,
    mute_audio: bool = False,
    audio_clip_start_s: float | None = None,
    audio_clip_duration_s: float | None = None,
    audio_repeat_interval_s: float | None = None,
) -> dict:
    traj = np.asarray(trajectory_m, dtype=np.float64)
    src = {
        "tag": tag,
        "audio_lookup": audio_lookup,
        "kind": "event_demo",
        "start_pos_m": traj[0].round(6).tolist(),
        "end_pos_m": traj[-1].round(6).tolist(),
        "trajectory_m": traj.round(6).tolist(),
        "motion": "linear_uniform",
        "motion_style": motion_style,
        "wanted_anim": wanted_anim,
        "notes": notes,
    }
    if mute_audio:
        src["mute_audio"] = True
    else:
        src["audio_path"] = str(resolve_animal_audio_path(tag, audio_lookup=audio_lookup))
    if audio_clip_start_s is not None:
        src["audio_clip_start_s"] = float(audio_clip_start_s)
    if audio_clip_duration_s is not None:
        src["audio_clip_duration_s"] = float(audio_clip_duration_s)
    if audio_repeat_interval_s is not None:
        src["audio_repeat_interval_s"] = float(audio_repeat_interval_s)
    return src


def _human_speech_source(
    entry: dict,
    trajectory_m,
    *,
    speech_root: str | Path | None,
    wanted_anim: str,
    motion_style: str,
    notes: str,
    source_role: str = "visible_human_speaker",
    facing_yaw_deg: float | None = None,
) -> dict:
    traj = np.asarray(trajectory_m, dtype=np.float64)
    sample = pick_speech_sample(
        root=speech_root,
        rng=np.random.default_rng(0),
        duration_range_s=(1.0, 5.0),
    )
    src = {
        **entry,
        "kind": "event_demo",
        "source_role": source_role,
        "start_pos_m": traj[0].round(6).tolist(),
        "end_pos_m": traj[-1].round(6).tolist(),
        "trajectory_m": traj.round(6).tolist(),
        "motion": "linear_uniform",
        "motion_style": motion_style,
        "wanted_anim": wanted_anim,
        "audio_lookup": "speech",
        "audio_path": str(sample.path),
        "speech_corpus": sample.corpus,
        "speaker_id": sample.speaker_id,
        "transcript": sample.transcript,
        "notes": notes,
    }
    if speech_root is not None:
        src["speech_root"] = str(speech_root)
    if facing_yaw_deg is not None:
        src["facing_yaw_deg"] = float(facing_yaw_deg) % 360.0
    src.setdefault("walking_forward_yaw_offset_deg", 90.0)
    src.setdefault("actor_scale", 1.0)
    src.setdefault("actor_z_lift_cm", 14.0)
    return src


def _listener_local_point(
    mic_pos_m,
    mic_yaw_deg: float,
    *,
    forward_m: float,
    left_m: float,
    z_m: float,
) -> np.ndarray:
    mic = np.asarray(mic_pos_m, dtype=np.float64)
    yaw = np.deg2rad(float(mic_yaw_deg))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            mic[0] + c * forward_m - s * left_m,
            mic[1] + s * forward_m + c * left_m,
            z_m,
        ],
        dtype=np.float64,
    )


def _constraint_starts_outside_fov_ends_inside(
    tag: str,
    trajectory_m,
    mic_pos_m,
    mic_yaw_deg: float,
    fov_h_deg: float = 90.0,
    fov_v_deg: float = 60.0,
) -> ConstraintResult:
    traj = np.asarray(trajectory_m, dtype=np.float64)
    vis = batch_frame_visibility(
        traj,
        mic_pos_m,
        mic_yaw_deg,
        fov_h_deg=fov_h_deg,
        fov_v_deg=fov_v_deg,
        obstacles_xyz=None,
    )
    passed = bool(not vis["in_fov"][0] and vis["in_fov"][-1])
    return ConstraintResult(
        f"{tag}:starts_outside_fov_ends_inside_fov",
        passed,
        {
            "tag": tag,
            "start_in_fov": bool(vis["in_fov"][0]),
            "end_in_fov": bool(vis["in_fov"][-1]),
            "in_fov_frames": int(np.count_nonzero(vis["in_fov"])),
        },
    )


def compose_front_idle_rear_left_to_right_demo(
    base_spec_path: str | Path = DEFAULT_BASE_SPEC,
    out_spec_path: str | Path | None = None,
) -> dict:
    """One visible front idle dog + one invisible rear L-to-R walking dog.

    The rear dog is intentionally behind the listener throughout, so it should
    be absent from the main camera and only obvious in topdown/per-source audio.
    """
    with open(base_spec_path) as f:
        spec = copy.deepcopy(json.load(f))
    if spec.get("spec_version") != "apartment_v1":
        raise ValueError("front_idle_rear_left_to_right_demo currently targets apartment_v1")

    n_frames = int(spec["render_config"]["n_frames"])
    z_m = float(spec.get("source_height_m", 0.45))
    mic = spec["mic"]["pos_m"]
    yaw = 140.0
    spec["mic"]["yaw_deg"] = yaw
    spec["mic"]["forward"] = [
        float(np.cos(np.deg2rad(yaw))),
        float(np.sin(np.deg2rad(yaw))),
        0.0,
    ]
    if spec.get("camera_configs"):
        spec["camera_configs"][0]["yaw_deg"] = yaw
    obstacles, bounds, valid_regions = _load_apartment_planning_context(spec)

    front_pos = np.asarray([-1.5, 0.15, z_m], dtype=np.float64)
    front_traj = np.asarray(_constant_trajectory(front_pos, n_frames), dtype=np.float64)

    rear_candidates = [
        # Stronger binaural sweep. Zero extra clearance means the point path
        # avoids wall/furniture AABBs, but does not reserve a large body radius.
        {"start_xy": (3.575, -5.0), "end_xy": (6.2, 0.9), "clearance_m": 0.0},
        {"start_xy": (4.1, -5.0), "end_xy": (6.2, 0.9), "clearance_m": 0.0},
        {"start_xy": (5.675, -2.9), "end_xy": (5.675, 0.3), "clearance_m": 0.15},
        {"start_xy": (5.15, -2.9), "end_xy": (5.675, 0.3), "clearance_m": 0.15},
    ]

    last_report = None
    selected = None
    for cand in rear_candidates:
        try:
            rear_traj = _planned_trajectory(
                cand["start_xy"],
                cand["end_xy"],
                z_m,
                n_frames,
                obstacles,
                bounds,
                valid_regions,
                clearance_m=float(cand["clearance_m"]),
            )
        except Exception as exc:
            last_report = {
                "passed": False,
                "failed": ["path_planner"],
                "results": [{
                    "name": "path_planner",
                    "passed": False,
                    "details": {"candidate": cand, "error": str(exc)},
                }],
            }
            continue

        report = verify_constraints([
            constraint_front_of_camera("dog_golden", front_traj, mic, yaw),
            constraint_in_fov_min_frames(
                "dog_golden", front_traj, mic, yaw, min_in_fov_frames=max(1, n_frames // 2)
            ),
            constraint_stationary("dog_golden", front_traj),
            constraint_no_aabb_intersections("dog_golden", front_traj, obstacles),
            constraint_behind_camera("dog_beagle_v2", rear_traj, mic, yaw),
            constraint_not_visible("dog_beagle_v2", rear_traj, mic, yaw),
            constraint_left_to_right("dog_beagle_v2", rear_traj, mic, yaw, margin_m=1.8),
            constraint_no_aabb_intersections("dog_beagle_v2", rear_traj, obstacles),
            constraint_min_actor_distance(
                "dog_golden", front_traj, "dog_beagle_v2", rear_traj, min_distance_m=1.0
            ),
        ])
        last_report = report
        if report["passed"]:
            selected = {**cand, "trajectory_m": rear_traj}
            break

    if selected is None:
        raise UnsatisfiableScenarioError(json.dumps(last_report, indent=2))

    rear_traj = selected["trajectory_m"]
    spec["description"] = (
        "Deterministic apartment review demo: visible front dog_golden idles; "
        "rear dog_beagle_v2 walks a strong listener-left to listener-right sweep "
        "behind the camera."
    )
    spec["event_controls"] = [
        {
            "name": "front_idle_source",
            "tag": "dog_golden",
            "type": "stationary_source",
            "contract": ["front_of_camera", "in_fov_min_frames", "stationary"],
        },
        {
            "name": "rear_left_to_right_source",
            "tag": "dog_beagle_v2",
            "type": "planned_listener_local_trajectory",
            "contract": ["behind_camera", "not_visible", "left_to_right"],
            "selected_candidate": {
                "start_xy": list(selected["start_xy"]),
                "end_xy": list(selected["end_xy"]),
                "clearance_m": float(selected["clearance_m"]),
            },
        },
    ]
    spec["sources"] = [
        _dog_source(
            "dog_golden",
            "silent",
            front_traj,
            wanted_anim="Idle",
            motion_style="stationary",
            notes="Front visible idle dog, controlled by front_idle_source event.",
            mute_audio=True,
        ),
        _dog_source(
            "dog_beagle_v2",
            "dog_sharp_bark",
            rear_traj,
            wanted_anim="Walking",
            motion_style="walking",
            notes="Rear invisible listener-left-to-right dog, controlled by rear_left_to_right_source event.",
            audio_clip_start_s=2.0,
            audio_clip_duration_s=0.35,
            audio_repeat_interval_s=0.7,
        ),
    ]
    spec["event_constraint_report"] = last_report

    _write_spec_and_flags(spec, out_spec_path)
    return spec


def compose_front_idle_left_rear_to_right_front_demo(
    base_spec_path: str | Path = DEFAULT_BASE_SPEC,
    out_spec_path: str | Path | None = None,
) -> dict:
    """Visible front idle dog + beagle walking from left-rear to right-front.

    Unlike the rear-only demo, the moving beagle starts behind/outside the
    camera FOV and ends in the camera's right-front sector. This uses point
    source planning: walls/room bounds are hard blockers, but furniture/body
    radius is not inflated so narrow passages remain usable for review events.
    """
    with open(base_spec_path) as f:
        spec = copy.deepcopy(json.load(f))
    if spec.get("spec_version") != "apartment_v1":
        raise ValueError("front_idle_left_rear_to_right_front_demo currently targets apartment_v1")

    n_frames = int(spec["render_config"]["n_frames"])
    z_m = float(spec.get("source_height_m", 0.45))
    mic = spec["mic"]["pos_m"]
    yaw = 140.0
    spec["mic"]["yaw_deg"] = yaw
    spec["mic"]["forward"] = [
        float(np.cos(np.deg2rad(yaw))),
        float(np.sin(np.deg2rad(yaw))),
        0.0,
    ]
    if spec.get("camera_configs"):
        spec["camera_configs"][0]["yaw_deg"] = yaw
    spec["source_collision_policy"] = "walls_only_center"

    wall_obstacles, bounds, valid_regions = _load_apartment_wall_planning_context(spec)

    front_pos = np.asarray([-1.5, 0.15, z_m], dtype=np.float64)
    front_traj = np.asarray(_constant_trajectory(front_pos, n_frames), dtype=np.float64)

    moving_candidates = [
        {
            # Listener-local approx: rear-left (-1.3m forward, +2.5m left)
            # to right-front (+3.0m forward, -2.0m left). The previous
            # far-corner path was a 2.6 m/s slide for a Walking animation.
            "start_xy": (-0.111111, -2.600735),
            "end_xy": (-0.493274, 3.633433),
            "clearance_m": 0.0,
            "min_in_fov_frames": 18,
        },
        {
            # Slightly farther rear-left fallback; still below brisk walking
            # speed for a 5 s review clip.
            "start_xy": (0.271911, -2.922129),
            "end_xy": (-0.493274, 3.633433),
            "clearance_m": 0.0,
            "min_in_fov_frames": 18,
        },
    ]

    last_report = None
    selected = None
    for cand in moving_candidates:
        try:
            rear_traj = _planned_trajectory(
                cand["start_xy"],
                cand["end_xy"],
                z_m,
                n_frames,
                wall_obstacles,
                bounds,
                valid_regions,
                clearance_m=float(cand["clearance_m"]),
            )
        except Exception as exc:
            last_report = {
                "passed": False,
                "failed": ["path_planner"],
                "results": [{
                    "name": "path_planner",
                    "passed": False,
                    "details": {"candidate": cand, "error": str(exc)},
                }],
            }
            continue

        report = verify_constraints([
            constraint_front_of_camera("dog_golden", front_traj, mic, yaw),
            constraint_in_fov_min_frames(
                "dog_golden", front_traj, mic, yaw, min_in_fov_frames=max(1, n_frames // 2)
            ),
            constraint_stationary("dog_golden", front_traj),
            constraint_no_aabb_intersections("dog_golden", front_traj, wall_obstacles),
            constraint_left_to_right("dog_beagle_v2", rear_traj, mic, yaw, margin_m=0.8),
            constraint_in_fov_min_frames(
                "dog_beagle_v2",
                rear_traj,
                mic,
                yaw,
                min_in_fov_frames=int(cand["min_in_fov_frames"]),
            ),
            _constraint_starts_outside_fov_ends_inside(
                "dog_beagle_v2",
                rear_traj,
                mic,
                yaw,
                fov_h_deg=float(spec["camera_configs"][0]["fov_deg"]),
                fov_v_deg=60.0,
            ),
            constraint_no_aabb_intersections("dog_beagle_v2", rear_traj, wall_obstacles),
            constraint_min_actor_distance(
                "dog_golden", front_traj, "dog_beagle_v2", rear_traj, min_distance_m=1.0
            ),
        ])
        last_report = report
        if report["passed"]:
            selected = {**cand, "trajectory_m": rear_traj}
            break

    if selected is None:
        raise UnsatisfiableScenarioError(json.dumps(last_report, indent=2))

    rear_traj = selected["trajectory_m"]
    spec["description"] = (
        "Deterministic apartment review demo: visible front dog_golden idles; "
        "dog_beagle_v2 starts left-rear/outside FOV and walks into the "
        "right-front camera sector."
    )
    spec["event_controls"] = [
        {
            "name": "front_idle_source",
            "tag": "dog_golden",
            "type": "stationary_source",
            "contract": ["front_of_camera", "in_fov_min_frames", "stationary"],
        },
        {
            "name": "left_rear_to_right_front_source",
            "tag": "dog_beagle_v2",
            "type": "planned_listener_local_trajectory",
            "contract": [
                "starts_outside_fov",
                "ends_inside_fov",
                "left_to_right",
                "walls_only_center",
            ],
            "selected_candidate": {
                "start_xy": list(selected["start_xy"]),
                "end_xy": list(selected["end_xy"]),
                "clearance_m": float(selected["clearance_m"]),
            },
        },
    ]
    spec["sources"] = [
        _dog_source(
            "dog_golden",
            "silent",
            front_traj,
            wanted_anim="Idle",
            motion_style="stationary",
            notes="Front visible idle dog, controlled by front_idle_source event.",
            mute_audio=True,
        ),
        _dog_source(
            "dog_beagle_v2",
            "dog_sharp_bark",
            rear_traj,
            wanted_anim="Walking",
            motion_style="walking",
            notes=(
                "Left-rear to right-front dog, controlled by "
                "left_rear_to_right_front_source event. Uses walls-only "
                "center-point collision."
            ),
            audio_clip_start_s=2.0,
            audio_clip_duration_s=0.35,
            audio_repeat_interval_s=0.7,
        ),
    ]
    spec["event_constraint_report"] = last_report

    _write_spec_and_flags(spec, out_spec_path)
    return spec


def compose_visible_human_speech_demo(
    base_spec_path: str | Path = DEFAULT_BASE_SPEC,
    out_spec_path: str | Path | None = None,
    *,
    human_asset_id: str = "human_male_blue_hoodie_0001",
    speech_root: str | Path | None = None,
) -> dict:
    """One visible stationary human speaker using real speech audio."""
    with open(base_spec_path) as f:
        spec = copy.deepcopy(json.load(f))
    if spec.get("spec_version") != "apartment_v1":
        raise ValueError("visible_human_speech_demo currently targets apartment_v1")

    n_frames = int(spec["render_config"]["n_frames"])
    speech_z_m = 1.55
    mic = spec["mic"]["pos_m"]
    yaw = 140.0
    spec["mic"]["yaw_deg"] = yaw
    spec["mic"]["forward"] = [
        float(np.cos(np.deg2rad(yaw))),
        float(np.sin(np.deg2rad(yaw))),
        0.0,
    ]
    spec["mic"]["notes"] = (
        "yaw=140 world, matching the approved apartment front-source review "
        "view; this avoids the yaw=180 kitchen-island occlusion."
    )
    if spec.get("camera_configs"):
        spec["camera_configs"][0]["yaw_deg"] = yaw
    spec["source_height_m"] = speech_z_m
    spec["source_collision_policy"] = "walls_only_center"

    human_entry = resolve_source_pool_entry({"asset_id": human_asset_id})
    human_pos = np.asarray([-1.5, 1.5, speech_z_m], dtype=np.float64)
    human_traj = np.asarray(_constant_trajectory(human_pos, n_frames), dtype=np.float64)
    facing_yaw = float(
        np.degrees(np.arctan2(mic[1] - human_pos[1], mic[0] - human_pos[0]))
        % 360.0
    )

    report = verify_constraints([
        constraint_front_of_camera(
            human_entry["tag"],
            human_traj,
            mic,
            yaw,
        ),
        constraint_in_fov_min_frames(
            human_entry["tag"],
            human_traj,
            mic,
            yaw,
            min_in_fov_frames=n_frames,
        ),
        constraint_stationary(human_entry["tag"], human_traj),
    ])
    if not report["passed"]:
        raise UnsatisfiableScenarioError(json.dumps(report, indent=2))

    spec["description"] = (
        "Deterministic apartment review demo: one approved Flux/Hunyuan human "
        "speaker stands in front of the camera and emits real LibriTTS speech."
    )
    spec["event_controls"] = [
        {
            "name": "visible_human_speaker",
            "asset_id": human_asset_id,
            "tag": human_entry["tag"],
            "type": "stationary_human_speech_source",
            "contract": [
                "front_of_camera",
                "in_fov_all_frames",
                "stationary",
                "faces_listener",
            ],
            "selected_position_m": human_pos.round(6).tolist(),
            "facing_yaw_deg": facing_yaw,
        }
    ]
    spec["sources"] = [
        _human_speech_source(
            human_entry,
            human_traj,
            speech_root=speech_root,
            wanted_anim="Standing_Idle",
            motion_style="stationary",
            notes=(
                "Visible human speaker controlled by visible_human_speaker "
                "event. Uses approved Flux/Hunyuan appearance and Mixamo "
                "Standing_Idle loop."
            ),
            facing_yaw_deg=facing_yaw,
        )
    ]
    spec["event_constraint_report"] = report

    _write_spec_and_flags(spec, out_spec_path)
    return spec


def compose_visible_moving_human_speech_demo(
    base_spec_path: str | Path = DEFAULT_BASE_SPEC,
    out_spec_path: str | Path | None = None,
    *,
    human_asset_id: str = "human_male_blue_hoodie_0001",
    speech_root: str | Path | None = None,
) -> dict:
    """One visible human speaker walking left-to-right in front of camera."""
    with open(base_spec_path) as f:
        spec = copy.deepcopy(json.load(f))
    if spec.get("spec_version") != "apartment_v1":
        raise ValueError("visible_moving_human_speech_demo currently targets apartment_v1")

    n_frames = int(spec["render_config"]["n_frames"])
    speech_z_m = 1.55
    mic = spec["mic"]["pos_m"]
    yaw = 140.0
    spec["mic"]["yaw_deg"] = yaw
    spec["mic"]["forward"] = [
        float(np.cos(np.deg2rad(yaw))),
        float(np.sin(np.deg2rad(yaw))),
        0.0,
    ]
    spec["mic"]["notes"] = (
        "yaw=140 world, matching the approved apartment front-source review "
        "view; this keeps the walking human visible."
    )
    if spec.get("camera_configs"):
        spec["camera_configs"][0]["yaw_deg"] = yaw
    spec["source_height_m"] = speech_z_m
    spec["source_collision_policy"] = "full_static_aabb_center"

    human_entry = resolve_source_pool_entry({"asset_id": human_asset_id})
    start = _listener_local_point(
        mic,
        yaw,
        forward_m=1.8,
        left_m=1.2,
        z_m=speech_z_m,
    )
    end = _listener_local_point(
        mic,
        yaw,
        forward_m=1.8,
        left_m=-1.2,
        z_m=speech_z_m,
    )
    human_traj = np.linspace(start, end, n_frames, dtype=np.float64)
    obstacles, _bounds, _valid_regions = _load_apartment_planning_context(spec)
    report = verify_constraints([
        constraint_front_of_camera(
            human_entry["tag"],
            human_traj,
            mic,
            yaw,
        ),
        constraint_in_fov_min_frames(
            human_entry["tag"],
            human_traj,
            mic,
            yaw,
            min_in_fov_frames=n_frames,
        ),
        constraint_left_to_right(
            human_entry["tag"],
            human_traj,
            mic,
            yaw,
            margin_m=0.8,
        ),
        constraint_no_aabb_intersections(
            human_entry["tag"],
            human_traj,
            obstacles,
        ),
    ])
    if not report["passed"]:
        raise UnsatisfiableScenarioError(json.dumps(report, indent=2))

    spec["description"] = (
        "Deterministic apartment review demo: one approved Flux/Hunyuan male "
        "human speaker walks from camera-left to camera-right while emitting "
        "real LibriTTS speech."
    )
    spec["event_controls"] = [
        {
            "name": "visible_moving_human_speaker",
            "asset_id": human_asset_id,
            "tag": human_entry["tag"],
            "type": "linear_visible_human_speech_trajectory",
            "contract": [
                "front_of_camera",
                "in_fov_all_frames",
                "left_to_right",
                "full_static_aabb_center",
            ],
            "listener_local_start_m": [1.8, 1.2],
            "listener_local_end_m": [1.8, -1.2],
        }
    ]
    spec["sources"] = [
        _human_speech_source(
            human_entry,
            human_traj,
            speech_root=speech_root,
            wanted_anim="Walking",
            motion_style="walking",
            source_role="visible_moving_human_speaker",
            notes=(
                "Visible moving human speaker controlled by "
                "visible_moving_human_speaker event. Uses approved "
                "Flux/Hunyuan appearance and Mixamo Walking loop."
            ),
        )
    ]
    spec["event_constraint_report"] = report

    _write_spec_and_flags(spec, out_spec_path)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-spec", default=str(DEFAULT_BASE_SPEC))
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--scenario",
        default="front_idle_rear_left_to_right",
        choices=[
            "front_idle_rear_left_to_right",
            "front_idle_left_rear_to_right_front",
            "visible_human_speech",
            "visible_moving_human_speech",
        ],
    )
    ap.add_argument("--speech-root")
    args = ap.parse_args()
    builders = {
        "front_idle_rear_left_to_right": compose_front_idle_rear_left_to_right_demo,
        "front_idle_left_rear_to_right_front": compose_front_idle_left_rear_to_right_front_demo,
        "visible_human_speech": compose_visible_human_speech_demo,
        "visible_moving_human_speech": compose_visible_moving_human_speech_demo,
    }
    if args.scenario in ("visible_human_speech", "visible_moving_human_speech"):
        spec = builders[args.scenario](
            args.base_spec,
            args.out,
            speech_root=args.speech_root,
        )
    else:
        spec = builders[args.scenario](args.base_spec, args.out)
    report = spec["event_constraint_report"]
    print(f"[demo] wrote {args.out}")
    print(f"[demo] constraints passed={report['passed']} failed={report['failed']}")


if __name__ == "__main__":
    main()
