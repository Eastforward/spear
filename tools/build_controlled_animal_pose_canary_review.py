#!/usr/bin/env python3
"""Build a manual cardinal-direction overlay for new animal canaries.

The builder joins static-qualified source assets with their non-destructive
100k LOD, isolated Walk/Idle media, and agent animation decision.  It does not
promote that agent decision: every entry remains pending the browser's manual
source-pose and cardinal-orientation gate.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_controlled_animal_direction_review as shared
from tools import controlled_source_asset_schema as contracts
from tools import register_controlled_animal_source_assets as source_registry
from tools import review_controlled_animal_animation_candidates as animation_decisions
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_animation_reviews as animation_reviews
from tools import run_controlled_animal_lod_binding as lod_binding


SCHEMA = shared.SCHEMA
SPEAR_ROOT = shared.SPEAR_ROOT
AVENGINE_ROOT = shared.AVENGINE_ROOT
APARTMENT_SPEC_SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
APARTMENT_STATUS_SCHEMA = "controlled_animal_apartment_render_status_v1"
APARTMENT_MEDIA = {
    "review": "side_by_side_review_annotated.mp4",
    "main": "apartment_v1_view0.mp4",
    "topdown": "topdown_review.mp4",
}


class CanaryReviewError(RuntimeError):
    """Raised when a canary evidence bundle cannot be authenticated."""


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return shared._hash_without(value, key)


def _load_registry(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    payload = shared._read_json(path)
    if (
        payload.get("schema") != source_registry.REGISTRY_SCHEMA
        or payload.get("registry_sha256")
        != _hash_without(payload, "registry_sha256")
        or payload.get("source_asset_count")
        != len(payload.get("source_assets", []))
    ):
        raise CanaryReviewError(f"invalid source registry: {path}")
    sources: dict[str, Any] = {}
    for index in payload["source_assets"]:
        source_path = shared._relative_artifact(
            path.parent, index.get("source_asset", {}), "source_asset_v2"
        )
        source = shared._read_json(source_path)
        asset_id = source.get("asset_id")
        if (
            source.get("schema") != contracts.SOURCE_ASSET_SCHEMA
            or asset_id != index.get("asset_id")
            or asset_id in sources
            or source.get("asset_class") != "animal"
        ):
            raise CanaryReviewError("source registry identity changed")
        sources[str(asset_id)] = {"payload": source, "path": source_path}
    return path, payload, sources


def _load_lod(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    payload = shared._read_json(path)
    if (
        payload.get("schema") != lod_binding.BATCH_SCHEMA
        or payload.get("batch_sha256") != _hash_without(payload, "batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise CanaryReviewError(f"invalid LOD binding batch: {path}")
    attempts: dict[str, Any] = {}
    for attempt in payload.get("attempts", []):
        asset_id = attempt.get("asset_id")
        if (
            not asset_id
            or asset_id in attempts
            or attempt.get("status") != "passed_lod_binding_glb_readback"
        ):
            raise CanaryReviewError("LOD binding coverage/status changed")
        lod_path = shared._relative_artifact(
            path.parent, attempt.get("artifacts", {}).get("lod_glb", {}), "LOD GLB"
        )
        rigged_path = shared._relative_artifact(
            path.parent,
            attempt.get("artifacts", {}).get("rigged_glb", {}),
            "rigged GLB",
        )
        attempts[str(asset_id)] = {
            "payload": attempt,
            "lod_glb": lod_path,
            "rigged_glb": rigged_path,
        }
    return path, payload, attempts


def _load_animation_review(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    payload = shared._read_json(path)
    if (
        payload.get("schema") != animation_reviews.BATCH_SCHEMA
        or payload.get("batch_sha256") != _hash_without(payload, "batch_sha256")
        or payload.get("status") != "rendered_pending_visual_qa"
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise CanaryReviewError(f"invalid animation review batch: {path}")
    reviews: dict[str, Any] = {}
    for index in payload.get("reviews", []):
        asset_id = index.get("asset_id")
        if (
            not asset_id
            or asset_id in reviews
            or index.get("status") != "rendered_pending_visual_qa"
        ):
            raise CanaryReviewError("animation review coverage/status changed")
        review_path = shared._relative_artifact(
            path.parent, index.get("review", {}), "animation review manifest"
        )
        review = shared._read_json(review_path)
        if (
            review.get("schema") != animation_reviews.REVIEW_SCHEMA
            or review.get("asset_id") != asset_id
            or review.get("review_sha256")
            != _hash_without(review, "review_sha256")
        ):
            raise CanaryReviewError("animation review identity/hash changed")
        videos = {
            name: shared._relative_artifact(
                path.parent,
                index.get("videos", {}).get(name, {}),
                f"{name} video",
            )
            for name in ("walking_side", "walking_front", "idle_side")
        }
        reviews[str(asset_id)] = {
            "payload": review,
            "path": review_path,
            "videos": videos,
        }
    return path, payload, reviews


def _load_animation_decisions(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    payload = shared._read_json(path)
    if (
        payload.get("schema") != animation_decisions.DECISION_BATCH_SCHEMA
        or payload.get("decision_batch_sha256")
        != _hash_without(payload, "decision_batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise CanaryReviewError(f"invalid animation decision batch: {path}")
    decisions: dict[str, Any] = {}
    for index in payload.get("decisions", []):
        record_path = shared._relative_artifact(
            path.parent, index.get("record", {}), "animation decision"
        )
        decision = shared._read_json(record_path)
        asset_id = decision.get("asset_id")
        if (
            decision.get("schema") != animation_decisions.DECISION_SCHEMA
            or decision.get("decision_sha256")
            != _hash_without(decision, "decision_sha256")
            or decision.get("decision")
            not in {"approved_for_ue_apartment", "rejected"}
            or asset_id != index.get("asset_id")
            or asset_id in decisions
        ):
            raise CanaryReviewError("animation decision identity/hash changed")
        decisions[str(asset_id)] = {"payload": decision, "path": record_path}
    return path, payload, decisions


def _checked_local_file(path: Path, label: str) -> Path:
    raw_path = path if path.is_absolute() else Path.cwd() / path
    if raw_path.is_symlink():
        raise CanaryReviewError(f"missing or unsafe {label}: {raw_path}")
    path = raw_path.resolve()
    try:
        path.relative_to(AVENGINE_ROOT.resolve())
    except ValueError as error:
        raise CanaryReviewError(f"{label} escaped AVEngine root: {path}") from error
    if not path.is_file():
        raise CanaryReviewError(f"missing or unsafe {label}: {path}")
    return path


def _load_apartment_specs(
    path: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    path = _checked_local_file(path, "Apartment spec manifest")
    payload = shared._read_json(path)
    records = payload.get("records", [])
    if (
        payload.get("schema") != APARTMENT_SPEC_SCHEMA
        or payload.get("manifest_sha256") != _hash_without(payload, "manifest_sha256")
        or payload.get("formal_registration_authorized") is not False
        or payload.get("avatar_count") != len(records)
        or payload.get("clip_count") != 2 * len(records)
    ):
        raise CanaryReviewError(f"invalid Apartment spec manifest: {path}")

    status_path = _checked_local_file(path.parent / "batch_status.json", "Apartment batch status")
    status = shared._read_json(status_path)
    try:
        status_manifest = Path(str(status.get("manifest", ""))).resolve()
    except (OSError, RuntimeError) as error:
        raise CanaryReviewError("invalid Apartment batch manifest path") from error
    if (
        status.get("schema") != APARTMENT_STATUS_SCHEMA
        or status_manifest != path
        or status.get("current_failures") != []
        or status.get("incomplete_jobs") != []
        or status.get("failed_job_count") != 0
        or status.get("incomplete_job_count") != 0
        or status.get("passed_job_count") != payload.get("clip_count")
    ):
        raise CanaryReviewError(f"Apartment batch did not pass completely: {status_path}")

    expected_jobs: set[tuple[str, str]] = set()
    apartment: dict[str, Any] = {}
    for record in records:
        asset_id = record.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id or asset_id in apartment:
            raise CanaryReviewError("Apartment asset identity/coverage changed")
        actions = record.get("actions", {})
        if set(actions) != {"Walking", "Idle"}:
            raise CanaryReviewError(f"Apartment Walk/Idle pair is incomplete: {asset_id}")
        media: dict[str, Path] = {}
        for action in ("Walking", "Idle"):
            action_record = actions[action]
            if action_record.get("motion") != action.lower():
                raise CanaryReviewError(f"Apartment action identity changed: {asset_id}/{action}")
            raw_output_dir = Path(str(action_record.get("output_dir", "")))
            if not raw_output_dir.is_absolute() or raw_output_dir.is_symlink():
                raise CanaryReviewError(
                    f"invalid Apartment output root: {asset_id}/{action}"
                )
            output_dir = raw_output_dir.resolve()
            try:
                output_dir.relative_to(path.parent.resolve())
            except ValueError as error:
                raise CanaryReviewError(
                    f"Apartment output escaped its spec root: {asset_id}/{action}"
                ) from error
            expected_jobs.add((asset_id, action))
            for view, filename in APARTMENT_MEDIA.items():
                media[f"apartment_{action.lower()}_{view}"] = _checked_local_file(
                    output_dir / "videos" / filename,
                    f"Apartment {action} {view} video",
                )
        apartment[asset_id] = {
            "payload": record,
            "media": media,
            "spec_manifest": path,
            "batch_status": status_path,
        }

    passed_jobs = {
        (item.get("base_avatar_id"), item.get("action"))
        for item in status.get("current_results", [])
        if item.get("status") == "passed" and item.get("stage") == "finalize"
    }
    if passed_jobs != expected_jobs:
        raise CanaryReviewError("Apartment passed job coverage differs from spec manifest")
    return path, payload, status_path, status, apartment


def _walking_status(decision: Mapping[str, Any]) -> str:
    if decision.get("decision") == "approved_for_ue_apartment":
        return "new_canary_animation_agent_approved_pending_human_review"
    return "new_canary_animation_rejected"


def _entry(
    source_record: Mapping[str, Any],
    lod_record: Mapping[str, Any],
    animation_record: Mapping[str, Any],
    decision_record: Mapping[str, Any],
    *,
    lod_batch_path: Path,
    animation_batch_path: Path,
    apartment_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = source_record["payload"]
    asset_id = str(source["asset_id"])
    static_review_path = shared._source_artifact(source, "static_review_manifest")
    static_pose = shared._load_static_pose_evidence(source, static_review_path)
    paths = {
        "source_asset": source_record["path"],
        "pixal_raw_glb": shared._source_artifact(source, "pixal_raw_glb"),
        "pixal_input_rgba": shared._source_artifact(source, "pixal_input_rgba"),
        "prebind_lod_glb": lod_record["lod_glb"],
        "static_contact_sheet": shared._source_artifact(
            source, "static_contact_sheet"
        ),
        "static_top_view": static_pose["static_top_view"],
        "static_review_manifest": static_review_path,
        "current_bound_glb": lod_record["rigged_glb"],
        "walking_side": animation_record["videos"]["walking_side"],
        "walking_front": animation_record["videos"]["walking_front"],
        "idle_side": animation_record["videos"]["idle_side"],
        "animation_review_manifest": animation_record["path"],
        "animation_review_batch": animation_batch_path,
        "animation_decision": decision_record["path"],
        "lod_binding_batch": lod_batch_path,
    }
    if apartment_record is not None:
        paths.update(apartment_record["media"])
        paths["apartment_spec_manifest"] = apartment_record["spec_manifest"]
        paths["apartment_batch_status"] = apartment_record["batch_status"]
    taxonomy = source["taxonomy"]
    return {
        "asset_id": asset_id,
        "species": taxonomy["species"],
        "breed": taxonomy["breed"],
        "profile_schema_id": source["profile_schema_id"],
        "sampled_attributes": copy.deepcopy(source["sampled_attributes"]),
        "artifacts": {
            name: shared._record(path, root=AVENGINE_ROOT)
            for name, path in paths.items()
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
                "paws_not_on_one_ground_plane",
            ],
            "binding_pretransform": "not_authorized_by_this_visual_gate",
        },
        "current_evidence_status": {
            "historical_automatic_direction_check": "not_used",
            "walking_direction": _walking_status(decision_record["payload"]),
            "source_pose": "manual_revalidation_required",
            "binding_orientation": "manual_cardinal_revalidation_required",
            "idle_media": "new_canary_pending_human_review",
            "apartment_walk_idle": (
                "passed_pending_human_direction_review"
                if apartment_record is not None
                else "not_available"
            ),
            "formal_dataset_asset": False,
        },
    }


def build_manifest(
    registries: Sequence[Path],
    lod_batches: Sequence[Path],
    animation_batches: Sequence[Path],
    decision_batches: Sequence[Path],
    output_root: Path,
    apartment_spec_manifests: Sequence[Path] = (),
) -> Path:
    lengths = {
        len(registries), len(lod_batches), len(animation_batches), len(decision_batches)
    }
    if lengths != {len(registries)} or not registries:
        raise CanaryReviewError("parallel evidence bundle counts must match and be nonzero")
    output_root = output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise CanaryReviewError(f"refusing to replace output: {output_root}")

    apartment_records: dict[str, Any] = {}
    apartment_manifest_records = []
    for apartment_path in apartment_spec_manifests:
        (
            apartment_path,
            apartment_payload,
            status_path,
            _status,
            loaded_records,
        ) = _load_apartment_specs(apartment_path)
        if set(apartment_records).intersection(loaded_records):
            raise CanaryReviewError("duplicate asset across Apartment manifests")
        apartment_records.update(loaded_records)
        apartment_manifest_records.append(
            {
                **shared._record(apartment_path, root=AVENGINE_ROOT),
                "manifest_sha256": apartment_payload["manifest_sha256"],
                "batch_status": shared._record(status_path, root=AVENGINE_ROOT),
            }
        )

    entries = []
    registry_records = []
    seen = set()
    for registry_path, lod_path, animation_path, decisions_path in zip(
        registries, lod_batches, animation_batches, decision_batches, strict=True
    ):
        registry_path, registry, sources = _load_registry(registry_path)
        lod_path, _lod_batch, lod_attempts = _load_lod(lod_path)
        animation_path, _animation_batch, animation = _load_animation_review(
            animation_path
        )
        decisions_path, _decision_batch, decisions = _load_animation_decisions(
            decisions_path
        )
        if not (set(sources) == set(lod_attempts) == set(animation) == set(decisions)):
            raise CanaryReviewError("registry/LOD/animation/decision asset coverage differs")
        if seen.intersection(sources):
            raise CanaryReviewError("duplicate asset across canary bundles")
        seen.update(sources)
        registry_records.append(
            {
                **shared._record(registry_path, root=AVENGINE_ROOT),
                "registry_sha256": registry["registry_sha256"],
            }
        )
        for asset_id in sources:
            apartment_record = apartment_records.get(asset_id)
            if (
                apartment_record is not None
                and decisions[asset_id]["payload"].get("decision")
                != "approved_for_ue_apartment"
            ):
                raise CanaryReviewError(
                    f"Apartment media attached to rejected animation: {asset_id}"
                )
            entries.append(
                _entry(
                    sources[asset_id],
                    lod_attempts[asset_id],
                    animation[asset_id],
                    decisions[asset_id],
                    lod_batch_path=lod_path,
                    animation_batch_path=animation_path,
                    apartment_record=apartment_record,
                )
            )

    unknown_apartment_assets = set(apartment_records).difference(seen)
    if unknown_apartment_assets:
        raise CanaryReviewError(
            "Apartment assets are absent from canary evidence: "
            + ", ".join(sorted(unknown_apartment_assets))
        )

    entries.sort(key=lambda item: (item["species"], item["breed"], item["asset_id"]))
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "new_canary_manual_pose_and_cardinal_review_required",
        "formal_dataset_registration_authorized": False,
        "source_registries": registry_records,
        "apartment_spec_manifests": apartment_manifest_records,
        "asset_count": len(entries),
        "species_counts": {
            species: sum(item["species"] == species for item in entries)
            for species in ("cat", "dog")
        },
        "entries": entries,
        "user_visual_override": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "scope": "new exact-side cat/dog source-pose canaries",
            "decision": "pending_manual_source_pose_and_cardinal_direction_review",
            "reason": "Agent animation decisions remain advisory until the human cardinal gate is saved.",
        },
        "safety": {
            "source_assets_modified": False,
            "source_registries_modified": False,
            "historical_media_modified": False,
            "historical_decisions_modified": False,
            "browser_decisions_are_transform_overlays_only": True,
        },
    }
    manifest["manifest_sha256"] = _hash_without(manifest, "manifest_sha256")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        destination = staging / "review_manifest.json"
        contracts.write_json_no_replace(destination, manifest)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise CanaryReviewError("review output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "review_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", action="append", required=True, type=Path)
    parser.add_argument("--lod-binding-batch", action="append", required=True, type=Path)
    parser.add_argument("--animation-review-batch", action="append", required=True, type=Path)
    parser.add_argument("--animation-decisions", action="append", required=True, type=Path)
    parser.add_argument("--apartment-spec-manifest", action="append", default=[], type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = build_manifest(
            args.registry,
            args.lod_binding_batch,
            args.animation_review_batch,
            args.animation_decisions,
            args.output_root,
            args.apartment_spec_manifest,
        )
        manifest = shared._read_json(manifest_path)
    except (CanaryReviewError, shared.DirectionReviewError, OSError, ValueError) as error:
        print(f"CONTROLLED_ANIMAL_CANARY_REVIEW_FAILED {error}")
        return 2
    print(
        "CONTROLLED_ANIMAL_CANARY_REVIEW_OK "
        f"assets={manifest['asset_count']} output={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
