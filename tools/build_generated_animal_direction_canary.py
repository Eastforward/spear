#!/usr/bin/env python3
"""Build a one-asset pre-animation manual-direction review manifest."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Sequence


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_quadruped_i23d_geometry


CARDINAL_SCHEMA = "controlled_animal_pose_direction_manual_review_manifest_v2"
TWO_STAGE_SCHEMA = "controlled_animal_pose_direction_manual_review_manifest_v3"
# Backward-compatible public name used by the original cardinal-only tests and
# callers.  New generated meshes opt into v3 explicitly.
SCHEMA = CARDINAL_SCHEMA
AVENGINE_ROOT = Path(__file__).resolve().parents[3]


class CanaryError(RuntimeError):
    """Raised when a canary artifact or identity is invalid."""


def _unsigned_axis_residual_deg(value: float) -> float:
    """Normalize a head/tail-agnostic longitudinal-axis angle to [-90, 90)."""

    normalized = (float(value) + 90.0) % 180.0 - 90.0
    if math.isclose(normalized, 0.0, abs_tol=1.0e-9):
        return 0.0
    return normalized


def declared_view_canonicalization_audit(
    mesh_path: Path,
    *,
    canonicalization_yaw_deg: float,
    maximum_residual_deg: float,
) -> dict[str, Any]:
    """Verify a declared camera-view transform without estimating a transform.

    The transform is fixed by the source-view contract.  Geometry is measured
    only to accept or reject that one-shot result; its measured angle never
    becomes the applied yaw.
    """

    if not 0.0 <= float(canonicalization_yaw_deg) <= 45.0:
        raise CanaryError("declared-view canonicalization yaw must be in [0, 45]")
    if not 0.0 < float(maximum_residual_deg) <= 5.0:
        raise CanaryError("post-canonical torso residual must be in (0, 5]")
    geometry = audit_quadruped_i23d_geometry.audit(
        mesh_path.resolve(), "declared_source_view_one_shot"
    )
    observed = geometry["torso_midline"][
        "sensitivity_global_axis_yaw_degrees"
    ]
    residuals = [
        _unsigned_axis_residual_deg(
            float(value) - float(canonicalization_yaw_deg)
        )
        for value in observed
    ]
    maximum_observed = max(abs(value) for value in residuals)
    passed = maximum_observed <= float(maximum_residual_deg)
    result = {
        "schema": "controlled_animal_declared_view_canonicalization_audit_v1",
        "policy": "fixed_declared_view_transform_verify_only_no_yaw_inference",
        "declared_canonicalization_yaw_deg": float(canonicalization_yaw_deg),
        "observed_unsigned_torso_axis_yaw_degrees": observed,
        "postcanonical_residual_yaw_degrees": residuals,
        "maximum_absolute_residual_yaw_deg": maximum_observed,
        "maximum_allowed_residual_yaw_deg": float(maximum_residual_deg),
        "status": (
            "passed_declared_view_canonicalization"
            if passed
            else "rejected_declared_view_residual"
        ),
        "applied_yaw_was_inferred_from_geometry": False,
        "single_attempt_failure_requires_new_profile_not_seed_retry": True,
        "geometry_measurement": geometry,
    }
    if not passed:
        raise CanaryError(
            "one-shot mesh does not satisfy the declared source-view yaw: "
            f"maximum residual {maximum_observed:.3f} deg exceeds "
            f"{maximum_residual_deg:.3f} deg"
        )
    return result


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
    declared_view = args.declared_view_canonicalization_yaw_deg is not None
    if declared_view and args.manual_residual_yaw:
        raise CanaryError(
            "declared-view canonicalization and manual residual yaw are mutually exclusive"
        )
    two_stage = bool(args.manual_residual_yaw or declared_view)
    canonicalization_audit = None
    if declared_view:
        canonicalization_audit = declared_view_canonicalization_audit(
            args.i23d_raw_glb,
            canonicalization_yaw_deg=(
                args.declared_view_canonicalization_yaw_deg
            ),
            maximum_residual_deg=args.maximum_postcanonical_residual_deg,
        )
        declared_yaw = float(args.declared_view_canonicalization_yaw_deg)
        orientation_contract = {
            "preview_coordinate_frame": "gltf_y_up",
            "initial_preview_pretransform": (
                "fixed_declared_source_view_canonicalization"
            ),
            "automatic_orientation_inference": "disabled",
            "hidden_reflection_or_mirror": "forbidden",
            "axis_alignment_policy": (
                "deterministic_declared_camera_view_canonicalization_v1"
            ),
            "declared_axis_alignment_yaw_deg": declared_yaw,
            "maximum_postcanonical_residual_yaw_deg": float(
                args.maximum_postcanonical_residual_deg
            ),
            "manual_axis_alignment_allowed": False,
            "manual_cardinal_head_tail_yaw_degrees": [0, 180],
            "target_torso_spine_longitudinal_axis": "positive_x",
            "target_up_axis": "positive_y",
            "review_alignment_target": "torso_and_spine_not_visible_nose",
            "source_pose_rejection_conditions": [
                "head_not_aligned_with_torso",
                "torso_or_spine_internally_twisted",
                "front_and_hind_leg_planes_inconsistent",
                "feet_not_on_one_ground_plane",
                "tail_intersects_hind_legs",
            ],
            "binding_pretransform": "requires_this_reviewed_total_yaw",
        }
        state_classification = (
            "pre_animation_declared_axis_canonicalization_then_manual_head_tail"
        )
        binding_orientation = (
            "declared_axis_canonicalization_passed_manual_head_tail_required"
        )
    elif two_stage:
        orientation_contract = {
            "preview_coordinate_frame": "gltf_y_up",
            "initial_preview_pretransform": "identity",
            "automatic_orientation_inference": "disabled",
            "hidden_reflection_or_mirror": "forbidden",
            "manual_rotation_only": True,
            "manual_axis_alignment_yaw": {
                "allowed": True,
                "step_degrees": [1, 5, 15],
                "range_degrees": [-45, 45],
                "visual_authority": "torso_and_spine_longitudinal_axis",
                "must_not_follow": "turned_head_or_tail",
            },
            "manual_cardinal_head_tail_yaw_degrees": [-90, 0, 90, 180],
            "target_torso_spine_longitudinal_axis": "positive_x",
            "target_up_axis": "positive_y",
            "review_alignment_target": "torso_and_spine_not_visible_nose",
            "fine_yaw_compensation": (
                "manual_torso_axis_alignment_only_not_automatic_or_head_compensation"
            ),
            "source_pose_rejection_conditions": [
                "head_not_aligned_with_torso",
                "torso_or_spine_internally_twisted",
                "front_and_hind_leg_planes_inconsistent",
                "feet_not_on_one_ground_plane",
                "tail_intersects_hind_legs",
            ],
            "binding_pretransform": "not_authorized_by_this_visual_gate",
        }
        state_classification = "pre_animation_manual_two_stage_yaw_review_required"
        binding_orientation = "manual_two_stage_yaw_required"
    else:
        orientation_contract = {
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
        }
        state_classification = "pre_animation_manual_cardinal_review_required"
        binding_orientation = "manual_cardinal_review_required"

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
        "orientation_contract": orientation_contract,
        "current_evidence_status": {
            "walking_direction": "new_canary_pending_manual_review",
            "source_pose": "manual_review_required",
            "binding_orientation": binding_orientation,
            "runtime_topology": "separate_strict_gate_in_progress",
            "formal_dataset_asset": False,
        },
    }
    if canonicalization_audit is not None:
        entry["declared_view_canonicalization_audit"] = canonicalization_audit
    manifest: dict[str, Any] = {
        "schema": TWO_STAGE_SCHEMA if two_stage else CARDINAL_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": state_classification,
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
    parser.add_argument(
        "--manual-residual-yaw",
        action="store_true",
        help=(
            "Use the v3 two-stage manual gate: first align the rigid torso axis "
            "with small reviewer-controlled yaw steps, then choose a cardinal "
            "head/tail orientation. No automatic orientation is inferred."
        ),
    )
    parser.add_argument(
        "--declared-view-canonicalization-yaw-deg",
        type=float,
        help=(
            "Apply one profile-declared source-view yaw before review. The yaw "
            "is never inferred from the mesh; geometry only verifies the fixed "
            "transform, and the reviewer then chooses 0/180 head-tail direction."
        ),
    )
    parser.add_argument(
        "--maximum-postcanonical-residual-deg",
        type=float,
        default=3.0,
        help="Reject the single attempt when the fixed-view residual exceeds this.",
    )
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
