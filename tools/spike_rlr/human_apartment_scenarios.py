"""Deterministic, technical-only multi-human apartment review scenarios."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

from event_constraints import (
    constraint_front_of_camera,
    constraint_in_fov_min_frames,
    constraint_min_actor_distance,
    constraint_no_aabb_intersections,
    constraint_stationary,
    verify_constraints,
)
from scene_two_dogs_apartment import _static_obstacle_bboxes
from speech_audio import pick_speech_sample, speech_sample_source_fields


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_SPEC = REPO_ROOT / "data" / "apartment_v1_spec.json"
DEFAULT_OUT_ROOT = (
    REPO_ROOT
    / "tmp"
    / "hy3d_rocketbox_template_fit_v1"
    / "human_apartment_examples_v1"
)
DEFAULT_SPEECH_ROOT = Path("/data/datasets/LibriTTS")

MALE_TAG = "hy3d_rocketbox_male_adult_01_spike"
FEMALE_TAG = "hy3d_rocketbox_female_adult_01_spike"
MINIMUM_SOURCE_SEPARATION_M = 0.75
ROCKETBOX_FORWARD_YAW_OFFSET_DEG = 90.0
ROCKETBOX_WALK_PLAY_RATE = 0.45


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _trajectory(start, end, n_frames: int) -> np.ndarray:
    return np.linspace(
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
        int(n_frames),
    )


def _curved_trajectory(start, end, n_frames: int, curve_offset_m: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    chord_xy = end[:2] - start[:2]
    chord_length = float(np.linalg.norm(chord_xy))
    if chord_length < 1e-6:
        raise ValueError("curved trajectory requires distinct endpoints")
    perpendicular = np.asarray([-chord_xy[1], chord_xy[0]], dtype=np.float64)
    perpendicular /= chord_length
    control = (start + end) * 0.5
    control[:2] += perpendicular * float(curve_offset_m)
    t = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)[:, None]
    return ((1.0 - t) ** 2 * start) + (2.0 * (1.0 - t) * t * control) + (
        t ** 2 * end
    )


def _facing_yaw_deg(position, mic_position) -> float:
    delta = np.asarray(mic_position, dtype=np.float64) - np.asarray(
        position, dtype=np.float64
    )
    return float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)


def _human_source(
    *,
    tag: str,
    identity_gender: str,
    trajectory: np.ndarray,
    action: str,
    audio_height_m: float,
    speech_sample=None,
    mic_position=None,
) -> dict:
    moving = action == "Walking"
    source = {
        "tag": tag,
        "usage_scope": "technical_spike_only",
        "formal_registry_promotion": False,
        "identity_gender": identity_gender,
        "kind": "moving" if moving else "stationary",
        "motion": "quadratic_bezier_raw" if moving else "stationary",
        "motion_style": "walking" if moving else "stationary",
        "wanted_anim": action,
        "trajectory_m": trajectory.round(9).tolist(),
        "start_pos_m": trajectory[0].round(9).tolist(),
        "end_pos_m": trajectory[-1].round(9).tolist(),
        "audio_source_height_offset_m": float(audio_height_m),
        "walking_forward_yaw_offset_deg": ROCKETBOX_FORWARD_YAW_OFFSET_DEG,
        "animation_play_rate": ROCKETBOX_WALK_PLAY_RATE if moving else 1.0,
        "actor_scale": 1.0,
        "actor_z_lift_cm": 0.0,
    }
    if moving:
        source["notes"] = "Stable Rocketbox template uses reviewed Walking action."
    else:
        source["facing_yaw_deg"] = _facing_yaw_deg(trajectory[0], mic_position)
        source["notes"] = "Stable Rocketbox template uses reviewed Standing_Idle action."

    if speech_sample is None:
        source.update({"audio_lookup": "silent", "mute_audio": True})
    else:
        source.update(speech_sample_source_fields(speech_sample))
        source["mute_audio"] = False
    return source


def _scenario_base(base_spec: dict) -> dict:
    spec = copy.deepcopy(base_spec)
    if spec.get("spec_version") != "apartment_v1":
        raise ValueError("human apartment examples require spec_version apartment_v1")
    spec.update({
        "usage_scope": "technical_spike_only",
        "formal_registry_promotion": False,
        "source_height_m": 0.0,
        "source_collision_policy": "furniture_and_walls",
        "minimum_source_separation_m": MINIMUM_SOURCE_SEPARATION_M,
        "furniture_mode": "subset",
        "furniture_include_categories": ["core", "decoration"],
        "furniture_include_actors_extra": [],
        "furniture_exclude_actors": [],
    })
    spec["mic"].update({
        "pos_m": [0.5, 0.15, 1.2],
        "yaw_deg": 145.0,
        "forward": [
            float(np.cos(np.deg2rad(145.0))),
            float(np.sin(np.deg2rad(145.0))),
            0.0,
        ],
        "type_rlr": "binaural_native",
    })
    spec["camera_configs"] = [{
        "name": "view0",
        "pos_m": [0.5, 0.15, 1.2],
        "yaw_deg": 145.0,
        "fov_deg": 90.0,
    }]
    spec["render_config"].update({
        "width": 960,
        "height": 720,
        "fps": 15,
        "n_frames": 75,
        "duration_s": 5.0,
        "streaming_warmup_frames": 120,
        "camera_warmup_frames": 40,
    })
    spec["audio_config"].update({
        "sample_rate_hz": 16000,
        "duration_s": 5.0,
        "n_samples": 80000,
        "output_channels": 2,
    })
    return spec


def _validate_scenario(spec: dict) -> dict:
    categories = json.loads(
        (REPO_ROOT / "tools/spike_rlr/apartment_furniture_categories.json").read_text(
            encoding="utf-8"
        )
    )
    obstacles = _static_obstacle_bboxes(spec, categories)
    mic_position = spec["mic"]["pos_m"]
    mic_yaw = float(spec["mic"]["yaw_deg"])
    fov = float(spec["camera_configs"][0]["fov_deg"])
    constraints = []
    sources = spec["sources"]
    for source in sources:
        trajectory = np.asarray(source["trajectory_m"], dtype=np.float64)
        visual_trajectory = trajectory.copy()
        visual_trajectory[:, 2] += 1.0
        constraints.extend([
            constraint_front_of_camera(
                source["tag"], visual_trajectory, mic_position, mic_yaw
            ),
            constraint_in_fov_min_frames(
                source["tag"],
                visual_trajectory,
                mic_position,
                mic_yaw,
                min_in_fov_frames=len(trajectory),
                fov_h_deg=fov,
                fov_v_deg=60.0,
            ),
            constraint_no_aabb_intersections(
                source["tag"], trajectory, obstacles
            ),
        ])
        if source["wanted_anim"] == "Standing_Idle":
            constraints.append(constraint_stationary(source["tag"], trajectory))
    for source_a, source_b in combinations(sources, 2):
        constraints.append(constraint_min_actor_distance(
            source_a["tag"],
            source_a["trajectory_m"],
            source_b["tag"],
            source_b["trajectory_m"],
            min_distance_m=float(spec["minimum_source_separation_m"]),
        ))
    report = verify_constraints(constraints)
    if not report["passed"]:
        raise ValueError(f"human apartment scenario constraints failed: {report['failed']}")
    return report


def build_human_scenario_specs(base_spec: dict, male_speech, female_speech) -> dict[str, dict]:
    n_frames = 75
    mic_position = [0.5, 0.15, 1.2]
    baseline_walk = _curved_trajectory(
        [-1.68, 1.55, 0.0],
        [-2.60, 2.45, 0.0],
        n_frames,
        curve_offset_m=-0.32,
    )
    idle = _trajectory([-3.35, 1.55, 0.0], [-3.35, 1.55, 0.0], n_frames)

    male_pass = _curved_trajectory(
        [-1.55, 1.30, 0.0],
        [-2.55, 2.40, 0.0],
        n_frames,
        curve_offset_m=0.32,
    )
    direction = male_pass[-1] - male_pass[0]
    perpendicular = np.asarray([-direction[1], direction[0], 0.0])
    perpendicular /= np.linalg.norm(perpendicular[:2])
    pass_offset = 0.90 * perpendicular
    female_pass = male_pass[::-1].copy() + pass_offset

    definitions = {
        "male_walk_female_idle": [
            (MALE_TAG, "M", baseline_walk, "Walking", 1.55, male_speech),
            (FEMALE_TAG, "F", idle, "Standing_Idle", 1.50, None),
        ],
        "female_walk_male_idle": [
            (MALE_TAG, "M", idle, "Standing_Idle", 1.55, None),
            (FEMALE_TAG, "F", baseline_walk, "Walking", 1.50, female_speech),
        ],
        "dual_walk_pass": [
            (MALE_TAG, "M", male_pass, "Walking", 1.55, male_speech),
            (FEMALE_TAG, "F", female_pass, "Walking", 1.50, female_speech),
        ],
    }

    scenarios = {}
    for scenario_id, source_defs in definitions.items():
        spec = _scenario_base(base_spec)
        spec["scenario_id"] = scenario_id
        spec["description"] = (
            f"Deterministic stable-template human apartment review: {scenario_id}."
        )
        spec["sources"] = [
            _human_source(
                tag=tag,
                identity_gender=gender,
                trajectory=trajectory,
                action=action,
                audio_height_m=audio_height,
                speech_sample=speech,
                mic_position=mic_position,
            )
            for tag, gender, trajectory, action, audio_height, speech in source_defs
        ]
        spec["event_constraint_report"] = _validate_scenario(spec)
        scenarios[scenario_id] = spec
    return scenarios


def build_human_turnaround_scenario_specs(base_spec: dict, male_speech) -> dict[str, dict]:
    """Build one screen-horizontal out-and-back clip with one hard reversal."""
    spec = _scenario_base(base_spec)
    n_frames = int(spec["render_config"]["n_frames"])
    camera_yaw_rad = np.deg2rad(float(spec["camera_configs"][0]["yaw_deg"]))
    camera_right = np.asarray([
        np.sin(camera_yaw_rad),
        -np.cos(camera_yaw_rad),
        0.0,
    ])
    center = np.asarray([-2.15, 1.95, 0.0], dtype=np.float64)
    start = center - 0.75 * camera_right
    turn = center + 0.75 * camera_right
    outbound_frames = 38
    trajectory = np.vstack([
        _trajectory(start, turn, outbound_frames),
        _trajectory(turn, start, n_frames - outbound_frames),
    ])
    outbound_delta = turn[:2] - start[:2]
    outbound_yaw = float(
        np.degrees(np.arctan2(outbound_delta[1], outbound_delta[0])) % 360.0
    )
    semantic_yaw = np.concatenate([
        np.full(outbound_frames, outbound_yaw, dtype=np.float64),
        np.full(n_frames - outbound_frames, (outbound_yaw + 180.0) % 360.0),
    ])

    source = _human_source(
        tag=MALE_TAG,
        identity_gender="M",
        trajectory=trajectory,
        action="Walking",
        audio_height_m=1.55,
        speech_sample=male_speech,
        mic_position=spec["mic"]["pos_m"],
    )
    source.update({
        "motion": "piecewise_linear_raw",
        "facing_yaw_deg_per_frame": semantic_yaw.round(9).tolist(),
        "turnaround_frame_index": outbound_frames,
        "notes": (
            "Stable Rocketbox Walking moves left-to-right in camera space, "
            "then reverses semantic facing once and returns right-to-left."
        ),
    })
    spec.update({
        "scenario_id": "male_walk_turnaround",
        "description": (
            "Single stable-template male walks left-to-right, turns once, "
            "then walks right-to-left."
        ),
        "rig_direction_check_windows": [
            {"label": "left_to_right", "frame_a": 12, "frame_b": 28},
            {"label": "right_to_left", "frame_a": 48, "frame_b": 64},
        ],
        "sources": [source],
    })
    spec["event_constraint_report"] = _validate_scenario(spec)
    return {"male_walk_turnaround": spec}


def write_human_scenario_bundle(
    *,
    out_root: Path | str = DEFAULT_OUT_ROOT,
    base_spec_path: Path | str = DEFAULT_BASE_SPEC,
    speech_root: Path | str = DEFAULT_SPEECH_ROOT,
    scenario_set: str = "standard",
) -> Path:
    out_root = Path(out_root).resolve()
    base_spec_path = Path(base_spec_path).resolve()
    speech_root = Path(speech_root).resolve()
    license_path = speech_root / "LICENSE.txt"
    if not license_path.is_file():
        raise FileNotFoundError(f"LibriTTS license is missing: {license_path}")

    male_speech = pick_speech_sample(
        root=speech_root,
        rng=np.random.default_rng(20260711),
        duration_range_s=(5.0, 8.0),
        speaker_gender="M",
    )
    base_spec = json.loads(base_spec_path.read_text(encoding="utf-8"))
    if scenario_set == "standard":
        female_speech = pick_speech_sample(
            root=speech_root,
            rng=np.random.default_rng(20260712),
            duration_range_s=(5.0, 8.0),
            speaker_gender="F",
        )
        scenarios = build_human_scenario_specs(base_spec, male_speech, female_speech)
    elif scenario_set == "turnaround":
        scenarios = build_human_turnaround_scenario_specs(base_spec, male_speech)
    else:
        raise ValueError(f"unsupported human scenario_set: {scenario_set!r}")

    descriptors = {}
    for scenario_id, spec in scenarios.items():
        spec_path = _atomic_write_json(out_root / "specs" / f"{scenario_id}.json", spec)
        descriptors[scenario_id] = {
            "clip_id": scenario_id,
            "spec_path": str(spec_path.resolve()),
            "spec_sha256": _sha256_file(spec_path),
            "output_dir": str((out_root / "clips" / scenario_id).resolve()),
        }
    manifest = {
        "schema_version": "human_apartment_scenario_bundle_v1",
        "scenario_set": scenario_set,
        "usage_scope": "technical_spike_only",
        "formal_registry_promotion": False,
        "base_spec": {
            "path": str(base_spec_path),
            "sha256": _sha256_file(base_spec_path),
        },
        "speech_license": {
            "corpus": "LibriTTS",
            "spdx": "CC-BY-4.0",
            "path": str(license_path),
            "sha256": _sha256_file(license_path),
        },
        "scenarios": descriptors,
    }
    return _atomic_write_json(out_root / "scenario_bundle.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--base-spec", default=str(DEFAULT_BASE_SPEC))
    parser.add_argument("--speech-root", default=str(DEFAULT_SPEECH_ROOT))
    parser.add_argument(
        "--scenario-set",
        choices=("standard", "turnaround"),
        default="standard",
    )
    args = parser.parse_args()
    print(write_human_scenario_bundle(
        out_root=args.out_root,
        base_spec_path=args.base_spec,
        speech_root=args.speech_root,
        scenario_set=args.scenario_set,
    ))


if __name__ == "__main__":
    main()
