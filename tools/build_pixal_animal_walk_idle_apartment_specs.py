"""Build Walk plus stationary Idle Apartment reviews for passed Pixal animals."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPORTS = (
    SPEAR_ROOT
    / "tmp/pixal_animal_backend_substitution_v1/generated_batch_v1/ue_import_jobs.json"
)
DEFAULT_TEMPLATE = (
    SPEAR_ROOT
    / "tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/specs/"
    "dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking.json"
)
DEFAULT_OUT_ROOT = (
    SPEAR_ROOT / "tmp/pixal_animal_walk_idle_apartment_review_v2"
)


PROFILE = {
    "cat_siamese_v1": {
        "species": "cat",
        "breed": "siamese",
        "actor_scale": 0.09,
        "audio_lookup": "cat_meow",
        "audio_height_m": 0.25,
    },
    "cat_tabby": {
        "species": "cat",
        "breed": "tabby",
        "actor_scale": 0.09,
        "audio_lookup": "cat_meow",
        "audio_height_m": 0.25,
    },
    "dog_beagle_v2": {
        "species": "dog",
        "breed": "beagle",
        "actor_scale": 0.08,
        "audio_lookup": "dog_bark",
        "audio_height_m": 0.35,
    },
    "dog_golden": {
        "species": "dog",
        "breed": "golden_retriever",
        "actor_scale": 0.13,
        "audio_lookup": "dog_bark",
        "audio_height_m": 0.45,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, payload) -> None:
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


def build_pair(template: dict, *, ue_tag: str, legacy_tag: str) -> dict[str, dict]:
    profile = PROFILE[legacy_tag]
    walking = copy.deepcopy(template)
    walking["description"] = (
        f"Pixal3D {legacy_tag} starts camera-right/rear, passes camera-left/front, "
        "then walks one counter-clockwise loop around the round table."
    )
    walking["usage_scope"] = "research_candidate"
    walking["formal_registration_authorized"] = False
    source = walking["sources"][0]
    source.update(
        {
            "tag": ue_tag,
            "asset_class": "animal",
            "species": profile["species"],
            "breed": profile["breed"],
            "kind": "moving",
            "wanted_anim": "Walking",
            "actor_scale": profile["actor_scale"],
            "actor_z_lift_cm": 0.0,
            "walking_forward_yaw_offset_deg": 0.0,
            "animation_play_rate": 1.0,
            "ground_snap_to_floor": True,
            "ground_snap_max_abs_correction_cm": 20.0,
            "audio_lookup": profile["audio_lookup"],
            "audio_source_height_offset_m": profile["audio_height_m"],
            "adaptive_repeat_short_calls": True,
            "strict_audio": True,
            "mute_audio": False,
        }
    )
    walking["camera_pass_table_loop_contract"]["animal_scale_rationale"] = {
        "actor_scale": profile["actor_scale"],
        "policy": "species-appropriate head-height canary; packaged bounds are authoritative",
    }

    idle = copy.deepcopy(walking)
    trajectory = walking["sources"][0]["trajectory_m"]
    contract = walking["camera_pass_table_loop_contract"]
    idle_frame = int(contract["left_front_nearest_frame"])
    idle_position = copy.deepcopy(trajectory[idle_frame])
    idle_source = idle["sources"][0]
    idle_source.update(
        {
            "kind": "stationary",
            "wanted_anim": "Idle",
            "motion": "stationary_left_front",
            "motion_style": "stationary_visible_left_front",
            "start_pos_m": idle_position,
            "end_pos_m": idle_position,
            "trajectory_m": [copy.deepcopy(idle_position) for _ in trajectory],
        }
    )
    idle["description"] = (
        f"Pixal3D {legacy_tag} holds Idle at the visible camera-left/front waypoint; "
        "the actor is not translated while playing an in-place Idle."
    )
    idle["trajectory_profile"] = "stationary_camera_left_front_v1"
    idle["rig_direction_check_windows"] = []
    idle["stationary_idle_contract"] = {
        "status": "passed",
        "frame_count": len(trajectory),
        "position_m": idle_position,
        "maximum_position_delta_m": 0.0,
        "source_waypoint_frame": idle_frame,
    }
    return {"walking": walking, "idle": idle}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", type=Path, default=DEFAULT_IMPORTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    imports_path = args.imports.resolve()
    template_path = args.template.resolve()
    imports = json.loads(imports_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    out_root = args.out_root.resolve()
    records = []
    legacy_tags = {job["legacy_tag"] for job in imports["jobs"]}
    if legacy_tags != set(PROFILE):
        raise ValueError(f"profile/import mismatch: {sorted(legacy_tags)}")
    for job in imports["jobs"]:
        pair = build_pair(
            template, ue_tag=job["tag"], legacy_tag=job["legacy_tag"]
        )
        action_records = {}
        for motion, spec in pair.items():
            clip_name = f"camera_pass_table_loop_{motion}"
            spec_path = out_root / "specs" / job["tag"] / f"{clip_name}.json"
            output_dir = out_root / "clips" / job["tag"] / clip_name
            _atomic_json(spec_path, spec)
            action = spec["sources"][0]["wanted_anim"]
            action_records[action] = {
                "motion": motion,
                "spec": str(spec_path),
                "spec_evidence": _record(spec_path),
                "output_dir": str(output_dir),
                "clip_id": f"{job['tag']}_{clip_name}_v1",
            }
        records.append(
            {
                "base_avatar_id": job["legacy_tag"],
                "tag": job["tag"],
                "legacy_tag": job["legacy_tag"],
                "source_glb": {
                    "path": job["rigged_glb"],
                    "sha256": job["rigged_glb_sha256"],
                },
                "actions": action_records,
            }
        )
    manifest = {
        "schema": "pixal_animal_walk_idle_apartment_specs_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "trajectory_policy": "Walking uses camera right/rear -> left/front -> one table loop; Idle remains stationary at left/front",
        "audio_policy": "species-matched short calls are energy-segmented and repeated with silent gaps",
        "avatar_count": len(records),
        "clip_count": sum(len(record["actions"]) for record in records),
        "imports": _record(imports_path),
        "template": _record(template_path),
        "records": records,
    }
    _atomic_json(out_root / "spec_manifest.json", manifest)
    print(
        f"PIXAL_ANIMAL_WALK_IDLE_SPECS_OK clips={manifest['clip_count']} root={out_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
