#!/usr/bin/env python3
"""Build a one-asset pre-animation cardinal-direction review manifest."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Sequence


SCHEMA = "controlled_animal_pose_direction_manual_review_manifest_v2"
AVENGINE_ROOT = Path(__file__).resolve().parents[3]


class CanaryError(RuntimeError):
    """Raised when a canary artifact or identity is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_without(value: dict[str, Any], key: str) -> str:
    payload = {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def safe_identity(value: str, label: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value) is None:
        raise CanaryError(f"unsafe {label}: {value!r}")
    return value


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        relative = path.relative_to(AVENGINE_ROOT.resolve())
    except ValueError as error:
        raise CanaryError(f"artifact escaped AVEngine: {path}") from error
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise CanaryError(f"missing or unsafe artifact: {path}")
    return {
        "absolute_path": str(path),
        "server_path": "/" + relative.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def parse_attributes(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise CanaryError("--attribute must use KEY=VALUE")
        key, item = value.split("=", 1)
        safe_identity(key, "attribute key")
        if not item or key in result:
            raise CanaryError(f"empty or duplicate attribute: {value}")
        result[key] = item
    return result


def build(args: argparse.Namespace) -> Path:
    asset_id = safe_identity(args.asset_id, "asset id")
    species = safe_identity(args.species, "species")
    if species not in {"cat", "dog", "horse"}:
        raise CanaryError(f"unsupported review species: {species}")
    breed = safe_identity(args.breed, "breed")
    entry = {
        "asset_id": asset_id,
        "species": species,
        "breed": breed,
        "profile_schema_id": safe_identity(args.profile_schema_id, "profile id"),
        "sampled_attributes": parse_attributes(args.attribute),
        "artifacts": {
            "reference_image": record(args.reference_image),
            "i23d_raw_glb": record(args.i23d_raw_glb),
            "generation_manifest": record(args.generation_manifest),
            # A rejected decimation may still be used for a yaw-only preview;
            # this field never authorizes binding, skinning, or registration.
            "prebind_lod_glb": record(args.direction_preview_glb),
            "static_contact_sheet": record(args.static_contact_sheet),
            "static_top_view": record(args.static_top_view),
        },
        "generation_provenance": {
            "reference_generator": args.reference_generator,
            "reference_sha256": sha256_file(args.reference_image.resolve()),
            "i23d_backend": args.i23d_backend,
            "i23d_sha256": sha256_file(args.i23d_raw_glb.resolve()),
            "direction_preview_geometry_status": args.preview_geometry_status,
            "direction_preview_only": True,
            "old_asset_mesh_used": False,
            "target_animation_generated_for_this_decision": False,
        },
        "orientation_contract": {
            "preview_coordinate_frame": "gltf_y_up",
            "initial_preview_pretransform": "identity",
            "automatic_orientation_inference": "disabled",
            "hidden_reflection_or_mirror": "forbidden",
            "manual_rotation_only": True,
            "allowed_yaw_degrees": [-90, 0, 90, 180],
            "target_torso_spine_longitudinal_axis": "positive_x",
            "target_up_axis": "positive_y",
            "review_alignment_target": "torso_and_spine_not_visible_nose",
            "fine_yaw_compensation": "forbidden_reject_source_pose_instead",
            "source_pose_rejection_conditions": [
                "head_not_aligned_with_torso",
                "torso_or_spine_twisted_or_diagonal",
                "front_and_hind_leg_planes_inconsistent",
                "feet_not_on_one_ground_plane",
                "tail_intersects_hind_legs",
            ],
            "binding_pretransform": "not_authorized_by_this_visual_gate",
        },
        "current_evidence_status": {
            "walking_direction": "new_canary_pending_manual_review",
            "source_pose": "manual_review_required",
            "binding_orientation": "manual_cardinal_review_required",
            "runtime_topology": "separate_strict_gate_in_progress",
            "formal_dataset_asset": False,
        },
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "pre_animation_manual_cardinal_review_required",
        "formal_dataset_registration_authorized": False,
        "asset_count": 1,
        "species_counts": {species: 1},
        "entries": [entry],
        "safety": {
            "source_assets_modified": False,
            "historical_decisions_modified": False,
            "historical_media_modified": False,
            "browser_decisions_are_transform_overlays_only": True,
            "direction_approval_does_not_authorize_binding": True,
        },
    }
    manifest["manifest_sha256"] = hash_without(manifest, "manifest_sha256")
    output_root = args.output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise CanaryError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        destination = staging / "review_manifest.json"
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        observed = json.loads(destination.read_text(encoding="utf-8"))
        if observed["manifest_sha256"] != hash_without(observed, "manifest_sha256"):
            raise CanaryError("manifest hash readback failed")
        os.rename(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root / "review_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--breed", required=True)
    parser.add_argument("--profile-schema-id", required=True)
    parser.add_argument("--attribute", action="append", default=[])
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--i23d-raw-glb", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--direction-preview-glb", type=Path, required=True)
    parser.add_argument("--static-contact-sheet", type=Path, required=True)
    parser.add_argument("--static-top-view", type=Path, required=True)
    parser.add_argument("--reference-generator", default="FLUX.2 Klein")
    parser.add_argument("--i23d-backend", required=True)
    parser.add_argument("--preview-geometry-status", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = build(args)
    except (CanaryError, OSError, ValueError) as error:
        print(f"GENERATED_ANIMAL_DIRECTION_CANARY_FAILED {error}")
        return 2
    print(f"GENERATED_ANIMAL_DIRECTION_CANARY_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
