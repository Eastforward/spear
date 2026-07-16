"""Build six representative Rocketbox Walking specs that loop around the round table."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SPEAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MANIFEST = (
    SPEAR_ROOT
    / "tmp/rocketbox_batch_apartment_review_v1/batch_spec_manifest.json"
)
DEFAULT_VARIANT_SPEC = (
    SPEAR_ROOT / "tmp/rocketbox_native_ue_v3/specs/shirt_blue_walk.json"
)
DEFAULT_VARIANT_UE_MANIFEST = (
    SPEAR_ROOT
    / "tmp/rocketbox_native_ue_import_v3"
    / "rocketbox_male_adult_01_shirt_blue_ue_v3"
    / "ue_import_manifest.json"
)
DEFAULT_OUT_ROOT = SPEAR_ROOT / "tmp/rocketbox_table_loop_apartment_review_v1"

REPRESENTATIVE_BASES = (
    ("adult_male", "rocketbox_adults_male_adult_01"),
    ("adult_female", "rocketbox_adults_female_adult_01"),
    ("male_child", "rocketbox_children_male_child_01"),
    ("female_child", "rocketbox_children_female_child_01"),
    ("female_nurse", "rocketbox_professions_medical_female_01"),
)
RECOLORED_ROLE = "recolored_adult_male_blue_shirt"
RECOLORED_REPRESENTATIVE_ID = "rocketbox_variant_male_adult_01_shirt_blue"

TARGET_TABLE_ACTOR = (
    "Meshes/07_table/Round_Table:SM_table_circular_polySurface65_47"
)
TARGET_TABLE_BBOX_SSOT_M = (-2.831796, 3.41286, -0.633337, 5.611319)
LOOP_BOUNDS_SSOT_M = (-3.28, 2.55, 0.02, 6.15)
LOOP_CORNER_RADIUS_M = 0.35


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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


def _line(start, end, count=512) -> np.ndarray:
    return np.linspace(start, end, int(count), dtype=np.float64)


def _arc(center, radius, start_deg, end_deg, count=512) -> np.ndarray:
    angles = np.deg2rad(
        np.linspace(start_deg, end_deg, int(count), dtype=np.float64)
    )
    return np.column_stack(
        (
            float(center[0]) + float(radius) * np.cos(angles),
            float(center[1]) + float(radius) * np.sin(angles),
        )
    )


def build_rounded_table_loop(*, n_frames: int) -> np.ndarray:
    """Return one closed, constant-speed, counter-clockwise rounded loop."""
    if int(n_frames) < 60:
        raise ValueError("table loop needs at least 60 frames")
    left, bottom, right, top = LOOP_BOUNDS_SSOT_M
    radius = LOOP_CORNER_RADIUS_M
    if right - left <= 2.0 * radius or top - bottom <= 2.0 * radius:
        raise ValueError("rounded table-loop bounds cannot contain their corners")
    middle_x = (left + right) / 2.0
    pieces = [
        _line((middle_x, bottom), (right - radius, bottom)),
        _arc((right - radius, bottom + radius), radius, -90.0, 0.0),
        _line((right, bottom + radius), (right, top - radius)),
        _arc((right - radius, top - radius), radius, 0.0, 90.0),
        _line((right - radius, top), (left + radius, top)),
        _arc((left + radius, top - radius), radius, 90.0, 180.0),
        _line((left, top - radius), (left, bottom + radius)),
        _arc((left + radius, bottom + radius), radius, 180.0, 270.0),
        _line((left + radius, bottom), (middle_x, bottom)),
    ]
    dense = np.concatenate(
        [piece if index == 0 else piece[1:] for index, piece in enumerate(pieces)]
    )
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, float(cumulative[-1]), int(n_frames))
    xy = np.column_stack(
        (
            np.interp(targets, cumulative, dense[:, 0]),
            np.interp(targets, cumulative, dense[:, 1]),
        )
    )
    trajectory = np.column_stack((xy, np.zeros(int(n_frames), dtype=np.float64)))
    trajectory[-1] = trajectory[0]
    return trajectory


def _direction_windows(n_frames: int) -> list[dict]:
    fractions = (
        ("bottom", 0.02, 0.07),
        ("right", 0.21, 0.27),
        ("top", 0.46, 0.52),
        ("left", 0.71, 0.77),
    )
    windows = []
    for label, start, end in fractions:
        frame_a = int(round((int(n_frames) - 1) * start))
        frame_b = int(round((int(n_frames) - 1) * end))
        windows.append(
            {"label": label, "frame_a": frame_a, "frame_b": frame_b}
        )
    return windows


def build_table_loop_spec(
    template: dict,
    *,
    role_label: str,
    fps: int = 15,
    duration_s: float = 12.0,
) -> dict:
    """Convert one approved linear Walking spec to a closed round-table loop."""
    spec = copy.deepcopy(template)
    n_frames = int(round(float(fps) * float(duration_s)))
    if (
        spec.get("spec_version") != "apartment_v1"
        or spec.get("usage_scope") != "research_candidate"
        or len(spec.get("sources", [])) != 1
        or len(spec.get("camera_configs", [])) != 1
        or fps <= 0
        or duration_s <= 0.0
        or n_frames < 60
    ):
        raise ValueError("representative Rocketbox template is not table-loop ready")
    source = spec["sources"][0]
    if (
        source.get("wanted_anim") != "Walking"
        or source.get("actor_scale") != 1.0
        or float(source.get("walking_forward_yaw_offset_deg", math.nan)) != 90.0
    ):
        raise ValueError("representative source changed its approved Walking contract")

    trajectory = build_rounded_table_loop(n_frames=n_frames)
    rounded_trajectory = np.round(trajectory, 6)
    step_lengths = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
    path_length = float(step_lengths.sum())
    start = [float(value) for value in rounded_trajectory[0]]

    spec["description"] = (
        f"Native Rocketbox {role_label} Walking one loop around the round table "
        "at authored height in apartment_0000."
    )
    spec["render_config"].update(
        {
            "fps": int(fps),
            "n_frames": n_frames,
            "duration_s": float(duration_s),
        }
    )
    spec["camera_configs"][0]["fov_deg"] = 105.0
    spec["rig_direction_check_windows"] = _direction_windows(n_frames)
    spec["trajectory_profile"] = "rounded_round_table_loop_v1"
    spec["table_loop_contract"] = {
        "schema": "rocketbox_round_table_loop_v1",
        "status": "passed",
        "target_actor": TARGET_TABLE_ACTOR,
        "target_bbox_ssot_m": list(TARGET_TABLE_BBOX_SSOT_M),
        "loop_bounds_ssot_m": list(LOOP_BOUNDS_SSOT_M),
        "corner_radius_m": LOOP_CORNER_RADIUS_M,
        "closed_loop": True,
        "direction": "counter_clockwise",
        "path_length_m": path_length,
        "average_speed_mps": path_length / float(duration_s),
        "camera_fov_deg": 105.0,
        "old_approved_assets_modified": False,
    }
    source.update(
        {
            "kind": "moving",
            "start_pos_m": start,
            "end_pos_m": start,
            "motion": "explicit_rounded_table_loop",
            "motion_style": "round_table_loop",
            "trajectory_m": rounded_trajectory.tolist(),
            "wanted_anim": "Walking",
        }
    )
    source.pop("facing_yaw_deg", None)
    source.pop("facing_yaw_deg_per_frame", None)
    return spec


def _load_json(path: Path, label: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--variant-spec", type=Path, default=DEFAULT_VARIANT_SPEC)
    parser.add_argument(
        "--variant-ue-manifest", type=Path, default=DEFAULT_VARIANT_UE_MANIFEST
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_manifest_path = args.base_manifest.resolve()
    base_manifest = _load_json(base_manifest_path, "base review manifest")
    if base_manifest.get("schema") != "rocketbox_batch_apartment_specs_v1":
        raise ValueError("base Rocketbox review manifest schema changed")
    base_records = {
        record["base_avatar_id"]: record
        for record in base_manifest.get("records", [])
    }
    out_root = args.out_root.resolve()
    records = []

    def publish_record(
        *,
        representative_id: str,
        role_label: str,
        template_path: Path,
        ue_manifest_path: Path,
        authored_height_cm,
    ) -> None:
        template = _load_json(template_path, f"{role_label} Walking template")
        spec = build_table_loop_spec(
            template,
            role_label=role_label,
            fps=args.fps,
            duration_s=args.duration_s,
        )
        source = spec["sources"][0]
        tag = str(source["tag"])
        spec_path = out_root / "specs" / tag / "table_loop_walking.json"
        _atomic_json(spec_path, spec)
        output_dir = out_root / "clips" / tag / "table_loop_walking"
        records.append(
            {
                "base_avatar_id": representative_id,
                "role_label": role_label,
                "tag": tag,
                "asset_id": source["asset_id"],
                "authored_height_cm": authored_height_cm,
                "actor_scale": 1.0,
                "template_spec": str(template_path.resolve()),
                "template_spec_sha256": _sha256(template_path.resolve()),
                "ue_import_manifest": str(ue_manifest_path.resolve()),
                "ue_import_manifest_sha256": _sha256(ue_manifest_path.resolve()),
                "actions": {
                    "Walking": {
                        "spec": str(spec_path),
                        "spec_sha256": _sha256(spec_path),
                        "clip_id": f"{tag}_table_loop_walking_v1",
                        "output_dir": str(output_dir),
                    }
                },
            }
        )

    for role_label, avatar_id in REPRESENTATIVE_BASES:
        if avatar_id not in base_records:
            raise ValueError(f"missing representative base avatar: {avatar_id}")
        record = base_records[avatar_id]
        publish_record(
            representative_id=avatar_id,
            role_label=role_label,
            template_path=Path(record["actions"]["Walking"]["spec"]),
            ue_manifest_path=Path(record["ue_import_manifest"]),
            authored_height_cm=record.get("authored_height_cm"),
        )

    variant_spec_path = args.variant_spec.resolve()
    variant_ue_manifest_path = args.variant_ue_manifest.resolve()
    variant_manifest = _load_json(
        variant_ue_manifest_path, "recolored male UE import manifest"
    )
    if (
        variant_manifest.get("schema") != "rocketbox_native_ue_import_v3"
        or variant_manifest.get("tag")
        != "rocketbox_male_adult_01_shirt_blue_ue_v3"
        or variant_manifest.get("usage_scope") != "research_candidate"
        or variant_manifest.get("reload_verification", {}).get("status") != "passed"
    ):
        raise ValueError("recolored adult male is not UE reload-verified")
    publish_record(
        representative_id=RECOLORED_REPRESENTATIVE_ID,
        role_label=RECOLORED_ROLE,
        template_path=variant_spec_path,
        ue_manifest_path=variant_ue_manifest_path,
        authored_height_cm=variant_manifest.get("runtime_contract", {})
        .get("bounds", {})
        .get("height_cm"),
    )

    manifest_path = out_root / "representative_spec_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema": "rocketbox_representative_table_loop_specs_v1",
            "generated_at": _utc_now(),
            "usage_scope": "research_candidate",
            "base_manifest": str(base_manifest_path),
            "base_manifest_sha256": _sha256(base_manifest_path),
            "trajectory_profile": "rounded_round_table_loop_v1",
            "target_actor": TARGET_TABLE_ACTOR,
            "avatar_count": len(records),
            "clip_count": len(records),
            "fps": int(args.fps),
            "duration_s": float(args.duration_s),
            "records": records,
        },
    )
    print(
        f"ROCKETBOX_TABLE_LOOP_SPECS_OK avatars={len(records)} "
        f"clips={len(records)} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
