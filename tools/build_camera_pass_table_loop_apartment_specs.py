"""Build six Rocketbox reviews: camera right-rear -> left-front -> table loop."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SPEAR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEAR_ROOT / "tools"))
sys.path.insert(0, str(SPEAR_ROOT / "tools/spike_rlr"))

from build_rocketbox_table_loop_apartment_specs import (  # noqa: E402
    DEFAULT_BASE_MANIFEST,
    DEFAULT_VARIANT_SPEC,
    DEFAULT_VARIANT_UE_MANIFEST,
    RECOLORED_REPRESENTATIVE_ID,
    RECOLORED_ROLE,
    REPRESENTATIVE_BASES,
    TARGET_TABLE_ACTOR,
    TARGET_TABLE_BBOX_SSOT_M,
    build_rounded_table_loop,
    build_table_loop_spec,
)
from scene_two_dogs_apartment import _static_obstacle_bboxes  # noqa: E402
from speech_audio import pick_speech_sample, speech_sample_source_fields  # noqa: E402


DEFAULT_OUT_ROOT = (
    SPEAR_ROOT / "tmp/rocketbox_camera_pass_table_loop_apartment_review_v2"
)
LIVING_ROOM_TABLE_ACTOR = (
    "Meshes/07_table/LivingRoom_Table:SM_table_sofa_44"
)
CAMERA_RIGHT_M = 0.8
CAMERA_REAR_M = 3.2
CAMERA_LEFT_M = 0.8
CAMERA_FRONT_M = 2.0
ENTRY_GUIDE_POINTS = (
    (1.65, -1.05),
    (-0.10, -0.90),
    (-1.40, -0.90),
)
CORNER_CUT_M = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _line(start, end, count=128):
    return np.linspace(start, end, int(count), dtype=np.float64)


def _rounded_polyline(points, *, corner_cut_m=CORNER_CUT_M):
    points = np.asarray(points, dtype=np.float64)
    pieces = []
    cursor = points[0]
    for index in range(1, len(points) - 1):
        previous = points[index - 1]
        vertex = points[index]
        following = points[index + 1]
        incoming = vertex - previous
        outgoing = following - vertex
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        cut = min(
            float(corner_cut_m),
            incoming_length * 0.25,
            outgoing_length * 0.25,
        )
        before = vertex - incoming / incoming_length * cut
        after = vertex + outgoing / outgoing_length * cut
        pieces.append(_line(cursor, before))
        t = np.linspace(0.0, 1.0, 96, dtype=np.float64)[:, None]
        curve = (1.0 - t) ** 2 * before + 2.0 * (1.0 - t) * t * vertex + t**2 * after
        pieces.append(curve[1:])
        cursor = after
    pieces.append(_line(cursor, points[-1]))
    return np.concatenate(
        [piece if index == 0 else piece[1:] for index, piece in enumerate(pieces)]
    )


def _resample_constant_speed(dense_xy, n_frames):
    distances = np.linalg.norm(np.diff(dense_xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    targets = np.linspace(0.0, float(cumulative[-1]), int(n_frames))
    xy = np.column_stack(
        (
            np.interp(targets, cumulative, dense_xy[:, 0]),
            np.interp(targets, cumulative, dense_xy[:, 1]),
        )
    )
    return np.column_stack((xy, np.zeros(int(n_frames), dtype=np.float64)))


def _nearest_frame(trajectory, point_xy):
    distances = np.linalg.norm(
        trajectory[:, :2] - np.asarray(point_xy, dtype=np.float64),
        axis=1,
    )
    return int(np.argmin(distances))


def build_camera_pass_table_loop_trajectory(
    *,
    n_frames: int,
    camera_pos_m,
    camera_yaw_deg: float,
):
    if int(n_frames) < 180:
        raise ValueError("camera-pass table loop needs at least 180 frames")
    camera_xy = np.asarray(camera_pos_m[:2], dtype=np.float64)
    yaw = math.radians(float(camera_yaw_deg))
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    right = np.array([math.sin(yaw), -math.cos(yaw)], dtype=np.float64)
    start = camera_xy + CAMERA_RIGHT_M * right - CAMERA_REAR_M * forward
    left_front = camera_xy - CAMERA_LEFT_M * right + CAMERA_FRONT_M * forward
    loop_dense = build_rounded_table_loop(n_frames=4096)[:, :2]
    loop_start = loop_dense[0]
    entry_points = np.asarray(
        [start, *ENTRY_GUIDE_POINTS, left_front, loop_start],
        dtype=np.float64,
    )
    entry_dense = _rounded_polyline(entry_points)
    dense = np.concatenate((entry_dense, loop_dense[1:]), axis=0)
    trajectory = _resample_constant_speed(dense, n_frames)
    path_length_m = float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())
    entry_path_length_m = float(
        np.linalg.norm(np.diff(entry_dense, axis=0), axis=1).sum()
    )
    dense_path_length_m = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
    loop_entry_frame = int(
        round(entry_path_length_m / dense_path_length_m * (int(n_frames) - 1))
    )
    left_front_frame = _nearest_frame(
        trajectory[: loop_entry_frame + 1],
        left_front,
    )
    start_delta = trajectory[0, :2] - camera_xy
    left_front_delta = left_front - camera_xy
    contract = {
        "schema": "camera_right_rear_left_front_table_loop_v2",
        "camera_pos_m": [float(v) for v in camera_pos_m],
        "camera_yaw_deg": float(camera_yaw_deg),
        "camera_forward_xy": forward.tolist(),
        "camera_right_xy": right.tolist(),
        "requested_start_components_m": {
            "right": float(np.dot(start_delta, right)),
            "rear": float(-np.dot(start_delta, forward)),
        },
        "requested_left_front_components_m": {
            "left": float(-np.dot(left_front_delta, right)),
            "front": float(np.dot(left_front_delta, forward)),
        },
        "start_xy_m": start.tolist(),
        "left_front_xy_m": left_front.tolist(),
        "table_loop_entry_xy_m": loop_start.tolist(),
        "left_front_nearest_frame": left_front_frame,
        "table_loop_entry_nearest_frame": loop_entry_frame,
        "entry_path_length_m": entry_path_length_m,
        "path_length_m": path_length_m,
        "target_actor": TARGET_TABLE_ACTOR,
        "target_bbox_ssot_m": list(TARGET_TABLE_BBOX_SSOT_M),
        "table_loop_direction": "counter_clockwise",
        "table_loop_turns": 1.0,
        "old_approved_assets_modified": False,
    }
    if (
        contract["requested_start_components_m"]["right"] < 0.7
        or contract["requested_start_components_m"]["rear"] < 3.0
        or contract["requested_left_front_components_m"]["left"] < 0.7
        or contract["requested_left_front_components_m"]["front"] < 1.8
    ):
        raise ValueError("camera-relative trajectory contract is not satisfied")
    return trajectory, contract


def _minimum_aabb_clearance(trajectory, obstacles):
    minimum = math.inf
    hit = None
    for frame_index, point in enumerate(trajectory[:, :2]):
        x, y = point
        for obstacle_index, (x0, y0, x1, y1) in enumerate(obstacles):
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            distance = math.hypot(dx, dy)
            if distance < minimum:
                minimum = distance
                hit = (frame_index, obstacle_index, (x0, y0, x1, y1))
    return float(minimum), hit


def build_spec(
    template,
    *,
    role_label,
    speech_sample,
    fps=15,
    duration_s=18.0,
):
    spec = build_table_loop_spec(
        template,
        role_label=role_label,
        fps=fps,
        duration_s=duration_s,
    )
    n_frames = int(round(fps * duration_s))
    camera = spec["camera_configs"][0]
    trajectory, contract = build_camera_pass_table_loop_trajectory(
        n_frames=n_frames,
        camera_pos_m=camera["pos_m"],
        camera_yaw_deg=camera["yaw_deg"],
    )
    source = spec["sources"][0]
    inferred_gender = "female" if "female" in role_label else "male"
    source.setdefault("gender", inferred_gender)
    source.update(
        {
            "start_pos_m": np.round(trajectory[0], 6).tolist(),
            "end_pos_m": np.round(trajectory[-1], 6).tolist(),
            "trajectory_m": np.round(trajectory, 6).tolist(),
            "motion": "explicit_camera_pass_table_loop",
            "motion_style": "camera_pass_then_round_table_loop",
            "wanted_anim": "Walking",
            "mute_audio": False,
            "audio_event_start_s": 5.0,
        }
    )
    source.update(speech_sample_source_fields(speech_sample))
    spec["audio_config"] = {
        "sample_rate_hz": 16000,
        "duration_s": float(duration_s),
        "n_samples": int(round(16000 * duration_s)),
        "output_channels": 2,
        "backend": "RLR_native_binaural",
        "speech_timing_contract": {
            "event_start_s": 5.0,
            "left_front_pass_s": contract["left_front_nearest_frame"] / fps,
            "policy": "speech begins as the actor approaches the visible left-front pass",
        },
    }
    exclusions = list(spec.get("furniture_exclude_actors", []))
    if LIVING_ROOM_TABLE_ACTOR not in exclusions:
        exclusions.append(LIVING_ROOM_TABLE_ACTOR)
    spec["furniture_exclude_actors"] = exclusions
    categories = json.loads(
        (SPEAR_ROOT / "tools/spike_rlr/apartment_furniture_categories.json").read_text(
            encoding="utf-8"
        )
    )
    obstacles = _static_obstacle_bboxes(spec, categories)
    minimum_clearance_m, closest = _minimum_aabb_clearance(trajectory, obstacles)
    if minimum_clearance_m <= 0.0:
        raise ValueError(f"camera-pass trajectory intersects obstacle: {closest}")
    contract.update(
        {
            "status": "passed",
            "minimum_center_aabb_clearance_m": minimum_clearance_m,
            "closest_obstacle_record": closest,
            "excluded_non_target_living_room_table": LIVING_ROOM_TABLE_ACTOR,
            "average_speed_mps": contract["path_length_m"] / duration_s,
        }
    )
    spec["description"] = (
        f"Native Rocketbox {role_label} starts camera-right/rear, passes "
        "camera-left/front, then walks one counter-clockwise loop around the round table."
    )
    spec["trajectory_profile"] = "camera_right_rear_left_front_table_loop_v2"
    spec["camera_pass_table_loop_contract"] = contract
    spec.pop("table_loop_contract", None)
    milestone_frames = [
        2,
        contract["left_front_nearest_frame"],
        contract["table_loop_entry_nearest_frame"],
        int(round(n_frames * 0.72)),
    ]
    spec["rig_direction_check_windows"] = [
        {
            "label": label,
            "frame_a": max(0, frame - 2),
            "frame_b": min(n_frames - 1, frame + 2),
        }
        for label, frame in zip(
            ("camera_right_rear", "camera_left_front", "table_loop_entry", "table_loop_late"),
            milestone_frames,
        )
    ]
    return spec


def _load_json(path: Path, label: str):
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--variant-spec", type=Path, default=DEFAULT_VARIANT_SPEC)
    parser.add_argument("--variant-ue-manifest", type=Path, default=DEFAULT_VARIANT_UE_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=18.0)
    parser.add_argument("--speech-seed", type=int, default=8120)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base_manifest_path = args.base_manifest.resolve()
    base_manifest = _load_json(base_manifest_path, "base review manifest")
    base_records = {
        record["base_avatar_id"]: record for record in base_manifest["records"]
    }
    out_root = args.out_root.resolve()
    records = []

    def publish(
        *,
        representative_id,
        role_label,
        template_path,
        ue_manifest_path,
        authored_height_cm,
        speech_index,
    ):
        template = _load_json(Path(template_path), f"{role_label} template")
        gender = str(
            template["sources"][0].get("gender")
            or ("female" if "female" in role_label else "male")
        )
        sample = pick_speech_sample(
            rng=np.random.default_rng(args.speech_seed + speech_index),
            duration_range_s=(3.0, 8.0),
            speaker_gender=gender,
        )
        spec = build_spec(
            template,
            role_label=role_label,
            speech_sample=sample,
            fps=args.fps,
            duration_s=args.duration_s,
        )
        source = spec["sources"][0]
        tag = source["tag"]
        spec_path = out_root / "specs" / tag / "camera_pass_table_loop_walking.json"
        _atomic_json(spec_path, spec)
        output_dir = out_root / "clips" / tag / "camera_pass_table_loop_walking"
        records.append(
            {
                "base_avatar_id": representative_id,
                "role_label": role_label,
                "tag": tag,
                "authored_height_cm": authored_height_cm,
                "actor_scale": 1.0,
                "template_spec": str(Path(template_path).resolve()),
                "template_spec_sha256": _sha256(Path(template_path).resolve()),
                "ue_import_manifest": str(Path(ue_manifest_path).resolve()),
                "ue_import_manifest_sha256": _sha256(Path(ue_manifest_path).resolve()),
                "speech": source["speech_provenance"],
                "actions": {
                    "Walking": {
                        "spec": str(spec_path),
                        "spec_sha256": _sha256(spec_path),
                        "clip_id": f"{tag}_camera_pass_table_loop_walking_v2",
                        "output_dir": str(output_dir),
                    }
                },
            }
        )

    for index, (role_label, avatar_id) in enumerate(REPRESENTATIVE_BASES):
        record = base_records[avatar_id]
        publish(
            representative_id=avatar_id,
            role_label=role_label,
            template_path=record["actions"]["Walking"]["spec"],
            ue_manifest_path=record["ue_import_manifest"],
            authored_height_cm=record.get("authored_height_cm"),
            speech_index=index,
        )

    variant_manifest = _load_json(args.variant_ue_manifest, "variant UE manifest")
    publish(
        representative_id=RECOLORED_REPRESENTATIVE_ID,
        role_label=RECOLORED_ROLE,
        template_path=args.variant_spec,
        ue_manifest_path=args.variant_ue_manifest,
        authored_height_cm=variant_manifest.get("runtime_contract", {}).get("bounds", {}).get("height_cm"),
        speech_index=len(REPRESENTATIVE_BASES),
    )

    manifest_path = out_root / "representative_spec_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema": "rocketbox_camera_pass_table_loop_specs_v2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usage_scope": "research_candidate",
            "base_manifest": str(base_manifest_path),
            "base_manifest_sha256": _sha256(base_manifest_path),
            "trajectory_profile": "camera_right_rear_left_front_table_loop_v2",
            "avatar_count": len(records),
            "clip_count": len(records),
            "fps": int(args.fps),
            "duration_s": float(args.duration_s),
            "speech_license": {
                "corpus": "LibriTTS",
                "license_path": "/data/datasets/LibriTTS/LICENSE.txt",
                "license_sha256": _sha256(Path("/data/datasets/LibriTTS/LICENSE.txt")),
            },
            "records": records,
        },
    )
    print(
        f"CAMERA_PASS_TABLE_LOOP_SPECS_OK avatars={len(records)} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
