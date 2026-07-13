#!/usr/bin/env python3
"""Build a non-overwriting manual source-pose/cardinal-direction manifest.

The published manifest is an overlay on top of the immutable source-asset and
Apartment evidence.  It deliberately does not edit an existing registry,
binding, GLB, animation decision, or video.  The companion browser UI records
only a manual source-pose verdict and a rotation-only cardinal transform in a
separate mutable state directory.  No automatic orientation or hidden mirror
is part of this review contract.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts


SCHEMA = "controlled_animal_pose_direction_manual_review_manifest_v2"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
DEFAULT_REGISTRY = (
    SPEAR_ROOT
    / "tmp/controlled_source_asset_execution_v1/"
    "animal_source_assets_apartment_31_final_v1_20260713/registry_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    SPEAR_ROOT
    / "tmp/controlled_source_asset_execution_v1/"
    "controlled_animal_pose_direction_manual_review_v2_20260713"
)


class DirectionReviewError(RuntimeError):
    """Raised when an immutable input cannot be authenticated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DirectionReviewError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise DirectionReviewError(f"JSON object required: {path}")
    return value


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    result = {
        "absolute_path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if root is not None:
        result["server_path"] = "/" + path.relative_to(root.resolve()).as_posix()
    return result


def _verify_path(path: Path, artifact: Mapping[str, Any], label: str) -> Path:
    path = path.resolve()
    expected_size = artifact.get("size_bytes")
    if (
        path.is_symlink()
        or not path.is_file()
        or (expected_size is not None and path.stat().st_size != expected_size)
        or _sha256_file(path) != artifact.get("sha256")
    ):
        raise DirectionReviewError(f"{label} changed: {path}")
    return path


def _source_artifact(source: Mapping[str, Any], name: str) -> Path:
    artifact = source.get("artifacts", {}).get(name)
    if not isinstance(artifact, dict):
        raise DirectionReviewError(
            f"missing {name}: {source.get('asset_id', '<unknown>')}"
        )
    roots = {
        "spear_repo": SPEAR_ROOT,
        "avengine_repo": AVENGINE_ROOT,
    }
    root = roots.get(str(artifact.get("root_id")))
    if root is None:
        raise DirectionReviewError(f"unsupported artifact root for {name}")
    path = (root / str(artifact.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise DirectionReviewError(f"{name} escaped its declared root") from error
    return _verify_path(path, artifact, name)


def _relative_artifact(root: Path, artifact: Mapping[str, Any], label: str) -> Path:
    path = (root / str(artifact.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise DirectionReviewError(f"{label} escaped its evidence root") from error
    return _verify_path(path, artifact, label)


def _load_animation_evidence(
    source: Mapping[str, Any], decision_path: Path
) -> dict[str, Path]:
    decision = _read_json(decision_path)
    if (
        decision.get("schema") != "avengine_controlled_animal_animation_decision_v1"
        or decision.get("asset_id") != source.get("asset_id")
        or decision.get("decision") != "approved_for_ue_apartment"
        or decision.get("decision_sha256")
        != _hash_without(decision, "decision_sha256")
    ):
        raise DirectionReviewError("historical animation decision is invalid")
    review_record = decision.get("review", {})
    review_path = _verify_path(
        Path(str(review_record.get("path", ""))),
        review_record,
        "animation review manifest",
    )
    review = _read_json(review_path)
    if (
        review.get("schema") != "avengine_controlled_animal_animation_review_v1"
        or review.get("asset_id") != source.get("asset_id")
        or review.get("review_sha256") != _hash_without(review, "review_sha256")
    ):
        raise DirectionReviewError("historical animation review is invalid")
    review_batch_root = review_path.parents[2]
    media = {
        name: _relative_artifact(
            review_batch_root,
            review.get("videos", {}).get(name, {}).get("video", {}),
            name,
        )
        for name in ("walking_side", "walking_front", "idle_side")
    }

    review_batch_path = review_batch_root / "animation_review_batch_manifest.json"
    review_batch = _read_json(review_batch_path)
    if (
        review_batch.get("schema")
        != "avengine_controlled_animal_animation_review_batch_v1"
        or review_batch.get("batch_sha256")
        != _hash_without(review_batch, "batch_sha256")
    ):
        raise DirectionReviewError("animation review batch is invalid")
    lod_record = review_batch.get("lod_binding_batch", {})
    lod_batch_path = _verify_path(
        Path(str(lod_record.get("path", ""))), lod_record, "LOD binding batch"
    )
    lod_batch = _read_json(lod_batch_path)
    if (
        lod_batch.get("schema") != "avengine_controlled_animal_lod_binding_batch_v1"
        or lod_batch.get("batch_sha256") != _hash_without(lod_batch, "batch_sha256")
    ):
        raise DirectionReviewError("LOD binding batch is invalid")
    attempts = {
        item.get("asset_id"): item for item in lod_batch.get("attempts", [])
    }
    attempt = attempts.get(source.get("asset_id"))
    if not attempt or attempt.get("status") != "passed_lod_binding_glb_readback":
        raise DirectionReviewError("asset is missing from its LOD binding batch")
    prebind_lod = _relative_artifact(
        lod_batch_path.parent,
        attempt.get("artifacts", {}).get("lod_glb", {}),
        "pre-bind runtime LOD",
    )
    return {
        **media,
        "prebind_lod": prebind_lod,
        "animation_review_manifest": review_path,
        "animation_review_batch": review_batch_path,
        "lod_binding_batch": lod_batch_path,
    }


def _load_static_pose_evidence(
    source: Mapping[str, Any], review_path: Path
) -> dict[str, Path]:
    review = _read_json(review_path)
    if (
        review.get("schema") != "avengine_controlled_animal_static_review_v1"
        or review.get("instance_id") != source.get("asset_id")
        or review.get("review_sha256") != _hash_without(review, "review_sha256")
    ):
        raise DirectionReviewError("historical static review is invalid")
    static_batch_root = review_path.parents[1]
    top_view = _relative_artifact(
        static_batch_root,
        review.get("views", {}).get("top", {}),
        "static top view",
    )
    return {"static_review_manifest": review_path, "static_top_view": top_view}


def _entry(source: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    asset_id = str(source.get("asset_id", ""))
    taxonomy = source.get("taxonomy", {})
    species = taxonomy.get("species")
    if species not in {"cat", "dog"}:
        raise DirectionReviewError(f"unsupported species for {asset_id}: {species}")

    decision_path = _source_artifact(source, "animation_decision")
    animation = _load_animation_evidence(source, decision_path)
    static_review_path = _source_artifact(source, "static_review_manifest")
    static_pose = _load_static_pose_evidence(source, static_review_path)
    paths = {
        "source_asset": source_path,
        "pixal_raw_glb": _source_artifact(source, "pixal_raw_glb"),
        "pixal_input_rgba": _source_artifact(source, "pixal_input_rgba"),
        "prebind_lod_glb": animation["prebind_lod"],
        "static_contact_sheet": _source_artifact(source, "static_contact_sheet"),
        "static_top_view": static_pose["static_top_view"],
        "static_review_manifest": static_pose["static_review_manifest"],
        "current_bound_glb": _source_artifact(source, "rigged_walk_idle_glb"),
        "walking_side": animation["walking_side"],
        "walking_front": animation["walking_front"],
        "idle_side": animation["idle_side"],
        "apartment_walking_review": _source_artifact(
            source, "apartment_walking_annotated_review_video"
        ),
        "apartment_walking_main": _source_artifact(
            source, "apartment_walking_apartment_video"
        ),
        "apartment_walking_topdown": _source_artifact(
            source, "apartment_walking_topdown_review_video"
        ),
        "animation_decision": decision_path,
        "animation_review_manifest": animation["animation_review_manifest"],
        "animation_review_batch": animation["animation_review_batch"],
        "lod_binding_batch": animation["lod_binding_batch"],
    }
    return {
        "asset_id": asset_id,
        "species": species,
        "breed": taxonomy.get("breed"),
        "profile_schema_id": source.get("profile_schema_id"),
        "sampled_attributes": copy.deepcopy(source.get("sampled_attributes", {})),
        "artifacts": {
            name: _record(path, root=AVENGINE_ROOT) for name, path in paths.items()
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
            "historical_automatic_direction_check": "invalidated",
            "walking_direction": "rejected_by_user_visual_review",
            "source_pose": "manual_revalidation_required",
            "binding_orientation": "manual_cardinal_revalidation_required",
            "idle_media": "retained_as_diagnostic_only_pending_revalidation",
            "formal_dataset_asset": False,
        },
    }


def build_manifest(registry_path: Path, output_root: Path) -> Path:
    registry_path = Path(registry_path).resolve()
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise DirectionReviewError(f"refusing to replace output: {output_root}")
    registry = _read_json(registry_path)
    if (
        registry.get("schema")
        != "avengine_controlled_animal_apartment_source_asset_registry_v1"
        or registry.get("registry_sha256")
        != _hash_without(registry, "registry_sha256")
        or registry.get("source_asset_count") != len(registry.get("source_assets", []))
    ):
        raise DirectionReviewError("final controlled-animal registry is invalid")

    entries = []
    seen = set()
    for index in registry["source_assets"]:
        source_record = index.get("source_asset", {})
        source_path = _relative_artifact(
            registry_path.parent, source_record, "source_asset_v2"
        )
        source = _read_json(source_path)
        asset_id = source.get("asset_id")
        if (
            asset_id in seen
            or asset_id != index.get("asset_id")
            or source.get("schema") != "source_asset_v2"
        ):
            raise DirectionReviewError("source-asset registry identity changed")
        seen.add(asset_id)
        entries.append(_entry(source, source_path))

    entries.sort(key=lambda item: (item["species"], str(item["breed"]), item["asset_id"]))
    if not entries:
        raise DirectionReviewError("direction review selection is empty")
    species_counts = {
        species: sum(item["species"] == species for item in entries)
        for species in ("cat", "dog")
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "source_pose_and_cardinal_direction_revalidation_required",
        "formal_dataset_registration_authorized": False,
        "source_registry": _record(registry_path, root=AVENGINE_ROOT),
        "source_registry_sha256": registry["registry_sha256"],
        "asset_count": len(entries),
        "species_counts": species_counts,
        "entries": entries,
        "user_visual_override": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "scope": "all current controlled cat/dog source poses and Walking outputs",
            "decision": "rejected_pending_manual_source_pose_and_cardinal_direction_review",
            "reason": (
                "User reported cats with turned heads and inconsistent front/hind leg "
                "planes, plus dogs running backward/sideways and beagles with mismatched "
                "leg planes. Fine yaw is a reviewer trial, not a valid repair for a "
                "malformed source pose."
            ),
            "automatic_orientation_inference_allowed": False,
            "manual_cardinal_rotation_only": True,
        },
        "safety": {
            "source_assets_modified": False,
            "source_registries_modified": False,
            "historical_decisions_modified": False,
            "historical_media_modified": False,
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
        os.chmod(destination, 0o444)
        if output_root.exists() or output_root.is_symlink():
            raise DirectionReviewError("direction review output appeared concurrently")
        os.rename(staging, output_root)
    except Exception:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()
        raise
    return output_root / "review_manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        path = build_manifest(args.registry, args.output_root)
        manifest = _read_json(path)
    except DirectionReviewError as error:
        print(f"CONTROLLED_ANIMAL_DIRECTION_REVIEW_BUILD_FAILED {error}")
        return 2
    print(
        "CONTROLLED_ANIMAL_DIRECTION_REVIEW_BUILD_OK "
        f"assets={manifest['asset_count']} cats={manifest['species_counts']['cat']} "
        f"dogs={manifest['species_counts']['dog']} output={path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
