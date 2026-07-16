"""Build the Pixal3D pug Apartment review on the shared camera/table path."""
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
DEFAULT_HUMAN_SPEC = (
    SPEAR_ROOT
    / "tmp/rocketbox_camera_pass_table_loop_apartment_review_v2/specs/"
    "rocketbox_adults_male_adult_01_original_ue_v1/"
    "camera_pass_table_loop_walking.json"
)
DEFAULT_OUT_ROOT = (
    SPEAR_ROOT / "tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1"
)
DEFAULT_PIXAL_MANIFEST = (
    SPEAR_ROOT
    / "tmp/pixal_animal_backend_substitution_v1/approved/"
    "dog_pug_pixal_canary_v1/pixal_generation_manifest.json"
)
DEFAULT_RUNTIME_MESH = (
    SPEAR_ROOT
    / "tmp/pixal_animal_backend_substitution_v1/approved/"
    "dog_pug_pixal_canary_v1/mesh_runtime_100000_double_sided.glb"
)
DEFAULT_RIGGED_GLB = (
    SPEAR_ROOT
    / "tmp/pixal_animal_backend_substitution_v1/rigged/"
    "dog_pug_pixal_canary_v1_rigged_v4_flipx_100k_double_sided.glb"
)
TAG = "dog_pug_pixal_canary_v2_100k"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing Pixal animal evidence: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


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


def build_spec(template: dict) -> dict:
    source_template = template["sources"][0]
    allowed_top_level = {
        "apartment_furniture_map",
        "apartment_shell_map",
        "camera_configs",
        "camera_pass_table_loop_contract",
        "furniture_exclude_actors",
        "furniture_include_actors_extra",
        "furniture_include_categories",
        "furniture_mode",
        "mic",
        "minimum_source_separation_m",
        "render_config",
        "rig_direction_check_windows",
        "room_backend",
        "source_collision_policy",
        "source_height_m",
        "spec_version",
        "trajectory_profile",
    }
    spec = {
        key: copy.deepcopy(value)
        for key, value in template.items()
        if key in allowed_top_level
    }
    duration_s = float(spec["render_config"]["duration_s"])
    sample_rate_hz = 16000
    spec.update(
        {
            "description": (
                "Pixal3D pug starts camera-right/rear, passes camera-left/front, "
                "then walks one counter-clockwise loop around the round table."
            ),
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
            "audio_config": {
                "sample_rate_hz": sample_rate_hz,
                "duration_s": duration_s,
                "n_samples": int(round(sample_rate_hz * duration_s)),
                "output_channels": 2,
                "backend": "RLR_native_binaural",
                "animal_event_policy": (
                    "energy-segment short calls and repeat as separated events; "
                    "never seamless-loop"
                ),
            },
            "sources": [
                {
                    "tag": TAG,
                    "asset_class": "animal",
                    "species": "dog",
                    "breed": "pug",
                    "kind": "moving",
                    "start_pos_m": copy.deepcopy(source_template["start_pos_m"]),
                    "end_pos_m": copy.deepcopy(source_template["end_pos_m"]),
                    "trajectory_m": copy.deepcopy(source_template["trajectory_m"]),
                    "motion": "explicit_camera_pass_table_loop",
                    "motion_style": "camera_pass_then_round_table_loop",
                    "wanted_anim": "Walking",
                    "walking_forward_yaw_offset_deg": 0.0,
                    "actor_scale": 0.15,
                    "actor_z_lift_cm": 0.0,
                    "animation_play_rate": 1.0,
                    "ground_snap_to_floor": True,
                    "ground_snap_max_abs_correction_cm": 20.0,
                    "audio_lookup": "dog_bark",
                    "audio_source_height_offset_m": 0.45,
                    "adaptive_repeat_short_calls": True,
                    "strict_audio": True,
                    "mute_audio": False,
                }
            ],
        }
    )
    spec["camera_pass_table_loop_contract"]["animal_center_clearance_note"] = (
        "The inherited path has >=0.20 m center-to-AABB clearance; the target "
        "pug footprint is checked again by packaged runtime bounds evidence."
    )
    return spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_HUMAN_SPEC)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--pixal-manifest", type=Path, default=DEFAULT_PIXAL_MANIFEST)
    parser.add_argument("--runtime-mesh", type=Path, default=DEFAULT_RUNTIME_MESH)
    parser.add_argument("--rigged-glb", type=Path, default=DEFAULT_RIGGED_GLB)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    template_path = args.template.resolve()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    spec = build_spec(template)
    out_root = args.out_root.resolve()
    spec_path = out_root / "specs" / TAG / "camera_pass_table_loop_walking.json"
    output_dir = out_root / "clips" / TAG / "camera_pass_table_loop_walking"
    _atomic_json(spec_path, spec)
    manifest = {
        "schema": "pixal_animal_camera_pass_table_loop_specs_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "backend": "Pixal3D",
        "tag": TAG,
        "template_spec": _descriptor(template_path),
        "pixal_generation_manifest": _descriptor(args.pixal_manifest),
        "runtime_mesh": _descriptor(args.runtime_mesh),
        "rigged_glb": _descriptor(args.rigged_glb),
        "spec": _descriptor(spec_path),
        "output_dir": str(output_dir),
        "clip_id": f"{TAG}_camera_pass_table_loop_walking_v1",
        "review_gate_policy": (
            "research candidate may render with SPEAR_SKIP_REVIEW_GATE=1; "
            "formal registry promotion remains forbidden"
        ),
    }
    manifest_path = out_root / "spec_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(f"PIXAL_ANIMAL_CAMERA_PASS_SPEC_OK manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
