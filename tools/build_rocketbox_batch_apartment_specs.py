"""Build paired Apartment Walking/Standing-Idle specs for native Rocketbox humans."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = SPEAR_ROOT / "tmp/rocketbox_route1_inventory_v1/inventory.json"
DEFAULT_UE_MANIFEST_ROOT = SPEAR_ROOT / "tmp/rocketbox_batch_native_ue_import_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _common_spec(*, fps: int, duration_s: float, description: str) -> dict:
    n_frames = int(round(float(fps) * float(duration_s)))
    if fps <= 0 or duration_s <= 0 or n_frames < 2:
        raise ValueError("Apartment render duration/fps must produce at least two frames")
    return {
        "spec_version": "apartment_v1",
        "description": description,
        "usage_scope": "research_candidate",
        "room_backend": "apartment_shell",
        "apartment_shell_map": "data/apartment_shell_map.json",
        "apartment_furniture_map": "data/apartment_furniture_map.json",
        "furniture_mode": "subset",
        "furniture_include_categories": ["core", "decoration"],
        "furniture_include_actors_extra": [],
        "furniture_exclude_actors": [],
        "source_collision_policy": "furniture_and_walls",
        "minimum_source_separation_m": 0.0,
        "mic": {
            "pos_m": [0.5, 0.15, 1.2],
            "yaw_deg": 145.0,
            "forward": [-1.0, 0.0, 0.0],
            "type_rlr": "binaural_native",
        },
        "camera_configs": [
            {
                "name": "view0",
                "pos_m": [0.5, 0.15, 1.2],
                "yaw_deg": 145.0,
                "fov_deg": 75.0,
            }
        ],
        "render_config": {
            "width": 960,
            "height": 720,
            "fps": int(fps),
            "n_frames": n_frames,
            "duration_s": float(duration_s),
            "streaming_warmup_frames": 120,
            "camera_warmup_frames": 40,
        },
        "source_height_m": 0.0,
    }


def _validate_inputs(avatar: dict, ue_manifest: dict) -> None:
    avatar_id = avatar.get("base_avatar_id")
    tag = f"{avatar_id}_original_ue_v1"
    bounds = ue_manifest.get("runtime_contract", {}).get("bounds", {})
    if (
        avatar.get("inventory_status") != "passed"
        or avatar.get("height_contract", {}).get("status") != "passed"
        or avatar.get("height_contract", {}).get("actor_scale") != 1.0
        or ue_manifest.get("schema") != "rocketbox_batch_native_ue_import_v1"
        or ue_manifest.get("base_avatar_id") != avatar_id
        or ue_manifest.get("asset_id") != avatar.get("legacy_asset_id")
        or ue_manifest.get("tag") != tag
        or ue_manifest.get("usage_scope") != "research_candidate"
        or ue_manifest.get("formal_registration_authorized") is not False
        or ue_manifest.get("reload_verification", {}).get("status") != "passed"
        or ue_manifest.get("runtime_contract", {}).get("actor_scale") != 1.0
        or ue_manifest.get("runtime_contract", {}).get("bone_count") != 80
        or bounds.get("authored_height_preserved") is not True
        or bounds.get("height_passed") is not True
        or bounds.get("ground_passed") is not True
        or set(ue_manifest.get("content", {}).get("animations", {}))
        != {"Walking", "Standing_Idle"}
    ):
        raise ValueError(f"Rocketbox avatar is not Apartment spec-ready: {avatar_id}")


def build_avatar_specs(
    avatar: dict,
    ue_manifest: dict,
    *,
    fps: int = 15,
    duration_s: float = 5.0,
) -> dict[str, dict]:
    """Return paired action specs without changing authored human scale."""
    _validate_inputs(avatar, ue_manifest)
    avatar_id = avatar["base_avatar_id"]
    tag = f"{avatar_id}_original_ue_v1"
    height = avatar["height_contract"]
    ue_height_cm = float(
        ue_manifest["runtime_contract"]["bounds"]["height_cm"]
    )
    height_contract = {
        "actor_scale": 1.0,
        "authored_height_preserved": True,
        "ceiling_cm": float(height["apartment_ceiling_cm"]),
        "minimum_headroom_cm": float(height["minimum_ceiling_headroom_cm"]),
        "authored_height_cm": float(height["authored_height_cm"]),
        "expected_ue_height_cm": ue_height_cm,
        "expected_ceiling_headroom_cm": (
            float(height["apartment_ceiling_cm"]) - ue_height_cm
        ),
    }
    source_common = {
        "tag": tag,
        "base_avatar_id": avatar_id,
        "asset_id": avatar["legacy_asset_id"],
        "asset_class": "human",
        "category": avatar["category"],
        "demographic": avatar["demographic"],
        "gender": avatar["gender"],
        "authored_height_cm": float(height["authored_height_cm"]),
        "expected_ue_height_cm": ue_height_cm,
        "expected_height_range_cm": [float(value) for value in height["allowed_height_cm"]],
        "audio_lookup": "silent",
        "mute_audio": True,
        "audio_source_height_offset_m": round(
            float(height["mouth_audio_height_cm"]) / 100.0, 6
        ),
        "walking_forward_yaw_offset_deg": 90.0,
        "animation_play_rate": 1.0,
        "actor_scale": 1.0,
        "actor_z_lift_cm": 0.0,
        "ground_snap_to_floor": True,
        "ground_snap_max_abs_correction_cm": 15.0,
    }

    walking = _common_spec(
        fps=fps,
        duration_s=duration_s,
        description=f"Native {avatar_id} Walking at authored height in apartment_0000.",
    )
    walking["apartment_height_contract"] = height_contract
    walking["sources"] = [
        {
            **source_common,
            "kind": "moving",
            "start_pos_m": [-1.68, 1.55, 0.0],
            "end_pos_m": [-2.6, 2.45, 0.0],
            "motion": "linear_uniform",
            "wanted_anim": "Walking",
        }
    ]

    idle = _common_spec(
        fps=fps,
        duration_s=duration_s,
        description=(
            f"Native {avatar_id} Standing Idle at authored height in apartment_0000."
        ),
    )
    idle["apartment_height_contract"] = height_contract
    idle["sources"] = [
        {
            **source_common,
            "kind": "stationary",
            "start_pos_m": [-1.68, 1.55, 0.0],
            "end_pos_m": [-1.68, 1.55, 0.0],
            "motion": "linear_uniform_raw",
            "wanted_anim": "Standing_Idle",
            "facing_yaw_deg": 136.0,
        }
    ]
    return {"Walking": walking, "Standing_Idle": idle}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--ue-manifest-root", type=Path, default=DEFAULT_UE_MANIFEST_ROOT)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--avatar-id", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory_path = args.inventory.resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    avatars = inventory.get("avatars", [])
    if (
        inventory.get("schema_version") != "rocketbox_human_inventory_v1"
        or inventory.get("automatic_checks", {}).get("overall") != "passed"
        or inventory.get("population", {}).get("total") != len(avatars)
    ):
        raise RuntimeError("Rocketbox inventory is not spec-ready")
    requested = set(args.avatar_id)
    if args.all == bool(requested):
        raise ValueError("select exactly one of --all or one-or-more --avatar-id")
    selected = [
        avatar
        for avatar in sorted(avatars, key=lambda item: item["base_avatar_id"])
        if args.all or avatar["base_avatar_id"] in requested
    ]
    selected_ids = {avatar["base_avatar_id"] for avatar in selected}
    if requested - selected_ids:
        raise ValueError(f"unknown Rocketbox avatar ids: {sorted(requested - selected_ids)}")

    out_root = args.out_root.resolve()
    records = []
    for avatar in selected:
        avatar_id = avatar["base_avatar_id"]
        tag = f"{avatar_id}_original_ue_v1"
        ue_manifest_path = (
            args.ue_manifest_root.resolve() / tag / "ue_import_manifest.json"
        )
        ue_manifest = json.loads(ue_manifest_path.read_text(encoding="utf-8"))
        specs = build_avatar_specs(
            avatar,
            ue_manifest,
            fps=args.fps,
            duration_s=args.duration_s,
        )
        action_records = {}
        for action, spec in specs.items():
            action_slug = "walking" if action == "Walking" else "idle"
            spec_path = out_root / "specs" / tag / f"{action_slug}.json"
            _atomic_json(spec_path, spec)
            action_records[action] = {
                "spec": str(spec_path),
                "spec_sha256": _sha256(spec_path),
                "clip_id": f"{tag}_{action_slug}",
                "output_dir": str(out_root / "clips" / tag / action_slug),
            }
        records.append(
            {
                "base_avatar_id": avatar_id,
                "tag": tag,
                "ue_import_manifest": str(ue_manifest_path),
                "ue_import_manifest_sha256": _sha256(ue_manifest_path),
                "authored_height_cm": avatar["height_contract"]["authored_height_cm"],
                "actor_scale": 1.0,
                "actions": action_records,
            }
        )

    manifest_path = out_root / "batch_spec_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema": "rocketbox_batch_apartment_specs_v1",
            "generated_at": _utc_now(),
            "usage_scope": "research_candidate",
            "inventory": str(inventory_path),
            "inventory_sha256": _sha256(inventory_path),
            "avatar_count": len(records),
            "clip_count": len(records) * 2,
            "fps": args.fps,
            "duration_s": args.duration_s,
            "records": records,
        },
    )
    print(
        f"ROCKETBOX_APARTMENT_SPECS_OK avatars={len(records)} "
        f"clips={len(records) * 2} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
