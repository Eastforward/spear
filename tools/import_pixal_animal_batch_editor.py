"""UE Editor batch wrapper for authenticated, create-only Pixal animal gates.

Environment:
  PIXAL_ANIMAL_IMPORT_PREPARATION
      Absolute ``ue_import_preparation_manifest.json`` path.
  PIXAL_ANIMAL_IMPORT_PREPARATION_SHA256
      External SHA-256 pin for the preparation-manifest file bytes.
  PIXAL_ANIMAL_IMPORT_MANIFEST
      Absolute ``ue_import_jobs.json`` path.  It must be the file authenticated
      by the preparation manifest.
  PIXAL_ANIMAL_IMPORT_MANIFEST_SHA256
      External SHA-256 pin for the batch-manifest file bytes.
  PIXAL_ANIMAL_IMPORT_RESULT
      Absolute, previously absent result-receipt path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import runpy
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import unreal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import compose_target_native_generated_quadruped_owner_review as presentation
from tools import controlled_source_asset_schema as contracts

SCRIPT_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = SCRIPT_DIR.parent
SPEAR_TMP_BRIDGE = (SPEAR_ROOT / "tmp").absolute()
IMPORT_ONE = SCRIPT_DIR / "import_gate_animal_editor.py"
BATCH_SCHEMA = "pixal_animal_ue_import_batch_v2"
LEGACY_BATCH_SCHEMA = "pixal_animal_ue_import_batch_v1"
PREPARATION_SCHEMA = "avengine_user_approved_generated_animal_ue_import_preparation_v3"
LEGACY_PREPARATION_SCHEMAS = frozenset(
    {
        "avengine_user_approved_generated_animal_ue_import_preparation_v1",
        "avengine_user_approved_generated_animal_ue_import_preparation_v2",
    }
)
DECISION_FREEZE_RECEIPT_SCHEMA = (
    "avengine_target_native_generated_animal_animation_decision_freeze_receipt_v2"
)
LEGACY_DECISION_FREEZE_RECEIPT_SCHEMAS = frozenset(
    {"avengine_target_native_generated_animal_animation_decision_freeze_receipt_v1"}
)
USER_INSTRUCTION_AUTHORITY = {
    "mode": "caller_assertion_v1",
    "cryptographic_user_identity_verified": False,
    "policy": (
        "the caller is responsible for invoking this tool only after "
        "an explicit user instruction"
    ),
}
SOURCE_REGISTRY_VALIDATION_MODES = {
    "frozen_historical_preflight_v1",
    "current_exact_rebuild",
    "direct_source_authority_v1",
}
RESULT_SCHEMA = "pixal_animal_ue_import_result_v2"
JOB_TYPE = "user_approved_generated_animal"
EXPECTED_ACTIONS = ["Idle", "Walking"]
IMPORT_JOB_IDENTITY_SCHEMA = "pixal_animal_ue_import_job_identity_v1"
IMPORT_JOB_RECEIPT_SCHEMA = "pixal_animal_ue_import_job_receipt_v1"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
BATCH_FIELDS = {
    "schema",
    "status",
    "state_classification",
    "formal_dataset_registration_authorized",
    "job_type",
    "job_count",
    "jobs",
    "non_destructive_policy",
    "batch_sha256",
}
JOB_FIELDS = {
    "job_type",
    "asset_id",
    "legacy_tag",
    "tag",
    "profile_schema_id",
    "sampled_attributes",
    "expected_actions",
    "rigged_glb",
    "rigged_glb_sha256",
    "source_registry_sha256",
    "source_asset_sha256",
    "request_sha256",
    "animation_decision_file_sha256",
    "animation_decision_sha256",
}
TRANSCODED_JOB_FIELDS = JOB_FIELDS | {
    "upstream_rigged_glb",
    "upstream_rigged_glb_sha256",
    "texture_transcode_manifest",
    "texture_transcode_manifest_sha256",
    "texture_transcode_manifest_size_bytes",
}
TEXTURE_TRANSCODE_SCHEMA = "glb_embedded_webp_to_png_transcode_v1"
TEXTURE_TRANSCODE_PURPOSE = "UE_5.5_interchange_compatibility"
TEXTURE_TRANSCODE_FIELDS = {
    "schema",
    "purpose",
    "geometry_skin_animation_byte_graph_changed",
    "input",
    "output",
    "images",
}
TEXTURE_TRANSCODE_IMAGE_FIELDS = {
    "image_index",
    "pixel_size",
    "rgba_sha256",
    "source_webp_sha256",
    "source_size_bytes",
    "png_sha256",
    "png_size_bytes",
}
PREPARATION_FIELDS = {
    "schema",
    "status",
    "state_classification",
    "formal_dataset_registration_authorized",
    "created_at",
    "canonical_identity",
    "source_asset",
    "source_asset_registry",
    "source_asset_registry_sha256",
    "expected_source_asset_registry_file_sha256",
    "source_asset_registry_validation_mode",
    "source_artifact_roots",
    "authenticated_source_artifact_count",
    "animation_review",
    "animation_review_schema",
    "animation_decision",
    "expected_animation_decision_file_sha256",
    "animation_decision_sha256",
    "animation_decision_freeze_receipt",
    "expected_animation_decision_freeze_receipt_file_sha256",
    "animation_decision_freeze_receipt_sha256",
    "presentation_evidence",
    "user_instruction_authority",
    "reviewed_animated_glb",
    "authenticated_review_artifact_count",
    "ue_import_jobs",
    "automatic_checks",
    "manifest_sha256",
}
PRESENTATION_EVIDENCE_FIELDS = {
    "presentation_receipt",
    "expected_presentation_receipt_file_sha256",
    "presentation_receipt_sha256",
    "output_video",
}
PRESENTATION_AUTOMATIC_CHECKS = {
    "presentation_receipt_raw_file_sha256_reauthenticated": True,
    "presentation_receipt_internal_sha256_reauthenticated": True,
    "presentation_exact_v4_review_sha256_reauthenticated": True,
    "presentation_output_video_bytes_and_directory_reauthenticated": True,
}
MOTION_STYLE_APPROVAL_SCHEMA = (
    "avengine_generated_animal_motion_style_approval_v1"
)
CURRENT_ASSET_SHORT_READBACK_SCHEMA = (
    "avengine_generated_animal_current_asset_short_readback_v1"
)
MOTION_STYLE_AND_CURRENT_READBACK_MODE = (
    "motion_style_approval_plus_current_asset_short_readback_v1"
)
MOTION_STYLE_AND_CURRENT_READBACK_EVIDENCE_FIELDS = {
    "mode",
    "motion_style_approval",
    "current_asset_short_readback",
}
MOTION_STYLE_AND_CURRENT_READBACK_AUTOMATIC_CHECKS = {
    "motion_style_approval_file_and_video_sha256_reauthenticated": True,
    "motion_style_idle_walking_approval_reauthenticated": True,
    "current_asset_short_readback_file_and_video_sha256_reauthenticated": True,
    "current_reviewed_glb_geometry_and_idle_walking_binding_reauthenticated": True,
}
CURRENT_ASSET_SHORT_READBACK_CHECKS = {
    "current_review_action_media_bound": True,
    "reviewed_animated_glb_bound": True,
    "idle_action_present": True,
    "walking_action_present": True,
}
PREPARATION_COMMON_AUTOMATIC_CHECK_FIELDS = {
    "source_registry_and_preflight_reauthenticated",
    "source_registry_matched_external_expected_sha256",
    "source_asset_v2_validated_against_request_and_profile",
    "source_registry_workspace_bound_to_review_target",
    "all_source_asset_artifacts_and_licenses_reauthenticated",
    "raw_pixal_geometry_to_tokenrig_target_closure_reauthenticated",
    "all_generated_animation_automatic_gates_passed",
    "all_generated_animation_stage_lineage_reauthenticated",
    "support_plane_v2_output_readback_reauthenticated",
    "all_six_animation_review_media_reauthenticated",
    "all_six_animation_review_media_ffprobed",
    "all_six_animation_render_encode_receipts_reauthenticated",
    "human_animation_approval_matched_external_expected_sha256",
    "animation_decision_freeze_receipt_reauthenticated",
    "user_instruction_authority_preserved_without_cryptographic_upgrade",
    "reviewed_glb_has_embedded_skin_weights_and_exact_idle_walking_actions",
    "job_identity_and_attributes_copied_exactly_from_source_asset_v2",
    "ue_import_jobs_v2_identity_and_self_hash_validated",
    "fresh_non_destructive_ue_tag_derived_from_canonical_asset_id",
    "no_ue_execution_performed",
    "overall",
}
PREPARATION_AUTOMATIC_CHECK_FIELDS = (
    PREPARATION_COMMON_AUTOMATIC_CHECK_FIELDS | set(PRESENTATION_AUTOMATIC_CHECKS)
)
COMPACT_PREPARATION_AUTOMATIC_CHECK_FIELDS = (
    PREPARATION_COMMON_AUTOMATIC_CHECK_FIELDS
    | set(MOTION_STYLE_AND_CURRENT_READBACK_AUTOMATIC_CHECKS)
)
DECISION_FREEZE_RECEIPT_FIELDS = {
    "schema",
    "status",
    "state_classification",
    "formal_dataset_registration_authorized",
    "source_asset_registry",
    "expected_source_asset_registry_file_sha256",
    "source_asset_registry_validation_mode",
    "source_asset",
    "animation_review",
    "expected_animation_review_file_sha256",
    "user_instruction_binding",
    "user_instruction_authority",
    "authenticated_review_artifact_count",
    "animation_decision",
    "decision_sha256",
    "presentation_evidence",
    "receipt_sha256",
}
CANONICAL_IDENTITY_FIELDS = {
    "asset_id",
    "legacy_tag",
    "tag",
    "profile_schema_id",
    "profile_sha256",
    "request_sha256",
    "taxonomy",
    "fixed_attributes",
    "sampled_attributes",
    "target_physical_profile",
}
DESCRIPTOR_FIELDS = {"path", "sha256", "size_bytes"}
JOB_IDENTITY_FIELDS = {
    "schema",
    "job_type",
    "asset_id",
    "legacy_tag",
    "tag",
    "expected_actions",
    "rigged_glb",
    "rigged_glb_sha256",
    "input_manifest_sha256",
    "batch_sha256",
    "job_identity_sha256",
}
JOB_RECEIPT_FIELDS = {
    "schema",
    "tag",
    "job_identity_sha256",
    "skeletal_mesh",
    "walking_animation",
    "blueprint",
    "actions",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"value is not canonical JSON: {error}") from error


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256({name: item for name, item in value.items() if name != key})


def _load_json_file(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if (
            expected_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise RuntimeError(f"external {label} SHA-256 pin mismatched")
        value = contracts.strict_json_loads(raw)
    except (OSError, contracts.StrictJSONError) as error:
        raise RuntimeError(f"{label} is not readable strict JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _load_json_text(value: str, label: str) -> dict[str, Any]:
    try:
        payload = contracts.strict_json_loads(value)
    except (TypeError, contracts.StrictJSONError) as error:
        raise RuntimeError(f"{label} is not strict JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA-256")
    return value


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _uses_only_exact_tmp_bridge(path: Path, label: str) -> Path:
    literal = _lexical_absolute(path)
    symlinks = [
        component
        for component in reversed((literal, *literal.parents))
        if component.is_symlink()
    ]
    unexpected = [component for component in symlinks if component != SPEAR_TMP_BRIDGE]
    if unexpected:
        raise RuntimeError(
            f"{label} cannot use symlink path component: {unexpected[0]}"
        )
    if SPEAR_TMP_BRIDGE in symlinks:
        try:
            literal.resolve().relative_to(SPEAR_TMP_BRIDGE.resolve())
        except ValueError as error:
            raise RuntimeError(f"{label} escaped the exact SPEAR/tmp bridge") from error
    return literal


def _direct_absolute_file(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"{label} path is not absolute: {path}")
    path = _uses_only_exact_tmp_bridge(path, label)
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"{label} is missing or indirect: {path}")
    return resolved


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def _validate_result_target(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"Pixal UE import result path is not absolute: {path}")
    path = _uses_only_exact_tmp_bridge(path, "Pixal UE import result")
    if path.exists() or path.is_symlink():
        raise RuntimeError(
            f"refusing to replace existing Pixal UE import result: {path}"
        )
    return path


def _read_glb_document(
    path: Path,
    *,
    include_binary: bool = False,
) -> dict | tuple[dict, bytes]:
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise RuntimeError(f"source is not a GLB 2.0 file: {path}")
    version, declared_size = struct.unpack_from("<II", raw, 4)
    if version != 2 or declared_size != len(raw):
        raise RuntimeError(f"invalid GLB header or JSON chunk: {path}")
    offset = 12
    chunks = []
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise RuntimeError(f"truncated GLB chunk header: {path}")
        chunk_size, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk_end = offset + chunk_size
        if chunk_size <= 0 or chunk_end > len(raw):
            raise RuntimeError(f"truncated GLB chunk: {path}")
        chunks.append((chunk_type, raw[offset:chunk_end]))
        offset = chunk_end
    if (
        offset != len(raw)
        or not chunks
        or chunks[0][0] != 0x4E4F534A
        or any(chunk_type not in {0x4E4F534A, 0x004E4942} for chunk_type, _ in chunks)
        or sum(chunk_type == 0x4E4F534A for chunk_type, _ in chunks) != 1
    ):
        raise RuntimeError(f"invalid GLB chunk graph: {path}")
    try:
        document = contracts.strict_json_loads(
            chunks[0][1].rstrip(b" \t\r\n\x00")
        )
    except contracts.StrictJSONError as error:
        raise RuntimeError(f"invalid GLB JSON document: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"GLB JSON document is not an object: {path}")
    if include_binary:
        if (
            len(chunks) != 2
            or chunks[1][0] != 0x004E4942
            or not isinstance(document.get("buffers"), list)
            or len(document["buffers"]) != 1
            or not isinstance(document["buffers"][0], dict)
            or "uri" in document["buffers"][0]
        ):
            raise RuntimeError(f"GLB embedded BIN contract is invalid: {path}")
        byte_length = document["buffers"][0].get("byteLength")
        binary_chunk = chunks[1][1]
        if (
            isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or byte_length <= 0
            or byte_length > len(binary_chunk)
            or len(binary_chunk) - byte_length > 3
            or any(binary_chunk[byte_length:])
        ):
            raise RuntimeError(f"GLB embedded BIN length is invalid: {path}")
        return document, binary_chunk[:byte_length]
    return document


def _validate_ue_compatible_glb(job: dict, source: Path) -> None:
    document = _read_glb_document(source)
    animations = document.get("animations")
    animation_names = (
        [animation.get("name") for animation in animations]
        if isinstance(animations, list)
        and all(isinstance(animation, dict) for animation in animations)
        else []
    )
    if (
        set(animation_names) != set(EXPECTED_ACTIONS)
        or len(animation_names) != len(EXPECTED_ACTIONS)
        or len(animation_names) != len(set(animation_names))
    ):
        raise RuntimeError(
            "UE import source must declare exactly the unique Idle and "
            f"Walking actions: {source}"
        )
    required_value = document.get("extensionsRequired", [])
    used_value = document.get("extensionsUsed", [])
    images = document.get("images", [])
    textures = document.get("textures", [])
    if (
        not isinstance(required_value, list)
        or any(not isinstance(value, str) for value in required_value)
        or len(required_value) != len(set(required_value))
        or not isinstance(used_value, list)
        or any(not isinstance(value, str) for value in used_value)
        or len(used_value) != len(set(used_value))
        or not isinstance(images, list)
        or any(not isinstance(image, dict) for image in images)
        or not isinstance(textures, list)
        or any(not isinstance(texture, dict) for texture in textures)
        or any(
            "extensions" in texture
            and not isinstance(texture["extensions"], dict)
            for texture in textures
        )
    ):
        raise RuntimeError(f"GLB extension/image contract is invalid: {source}")
    required = set(required_value)
    used = set(used_value)
    webp_images = [
        index
        for index, image in enumerate(images)
        if image.get("mimeType") == "image/webp"
    ]
    texture_webp = any(
        isinstance(texture.get("extensions"), dict)
        and "EXT_texture_webp" in texture["extensions"]
        for texture in textures
    )
    if (
        "EXT_texture_webp" in required
        or "EXT_texture_webp" in used
        or texture_webp
        or webp_images
    ):
        raise RuntimeError(
            "UE 5.5 import source still requires embedded WebP; run "
            f"transcode_glb_webp_to_png.py first: {source}"
        )
    manifest_value = job.get("texture_transcode_manifest")
    if manifest_value is None:
        return
    _load_texture_transcode_manifest(job, source)


def _validate_file_descriptor(
    value: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != DESCRIPTOR_FIELDS:
        raise RuntimeError(f"{label} descriptor fields are invalid")
    raw_path = value.get("path")
    sha256 = _require_sha256(value.get("sha256"), f"{label} descriptor hash")
    size_bytes = value.get("size_bytes")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise RuntimeError(f"{label} descriptor values are invalid")
    path = _direct_absolute_file(raw_path, label)
    if (
        (expected_path is not None and path != expected_path)
        or path.stat().st_size != size_bytes
        or _sha256(path) != sha256
    ):
        raise RuntimeError(f"{label} descriptor does not authenticate its file")
    return {"path": str(path), "sha256": sha256, "size_bytes": size_bytes}


def _has_idle_walking_actions(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(action, str) for action in value)
        and set(value) == set(EXPECTED_ACTIONS)
    )


def _uses_compact_approval_evidence(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("mode") == MOTION_STYLE_AND_CURRENT_READBACK_MODE
    )


def _validate_extended_file_descriptor(
    value: Any,
    label: str,
    *,
    internal_hash_field: str,
) -> tuple[dict[str, Any], str]:
    expected_fields = DESCRIPTOR_FIELDS | {internal_hash_field}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise RuntimeError(f"{label} descriptor fields are invalid")
    descriptor = _validate_file_descriptor(
        {field: value[field] for field in DESCRIPTOR_FIELDS},
        label,
    )
    internal_hash = _require_sha256(
        value.get(internal_hash_field),
        f"{label} {internal_hash_field}",
    )
    return descriptor, internal_hash


def _validate_compact_approval_evidence(
    value: Any,
    *,
    expected_asset_id: str,
    review: dict[str, Any],
    review_descriptor: dict[str, Any],
    reviewed_glb_descriptor: dict[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != MOTION_STYLE_AND_CURRENT_READBACK_EVIDENCE_FIELDS
        or value.get("mode") != MOTION_STYLE_AND_CURRENT_READBACK_MODE
    ):
        raise RuntimeError("compact motion approval evidence fields are invalid")

    style_descriptor, expected_approval_sha256 = (
        _validate_extended_file_descriptor(
            value.get("motion_style_approval"),
            "motion-style approval",
            internal_hash_field="approval_sha256",
        )
    )
    style = _load_json_file(
        Path(style_descriptor["path"]),
        "motion-style approval",
        expected_sha256=style_descriptor["sha256"],
    )
    if (
        set(style)
        != {
            "schema",
            "status",
            "actions",
            "evidence_video",
            "approval_sha256",
        }
        or style.get("schema") != MOTION_STYLE_APPROVAL_SCHEMA
        or style.get("status") != "approved_for_idle_walking_motion_style"
        or not _has_idle_walking_actions(style.get("actions"))
        or style.get("approval_sha256")
        != _hash_without(style, "approval_sha256")
        or style.get("approval_sha256") != expected_approval_sha256
    ):
        raise RuntimeError("motion-style approval contract is invalid")
    style_video = _validate_file_descriptor(
        style.get("evidence_video"),
        "motion-style evidence video",
    )
    if style["evidence_video"] != style_video:
        raise RuntimeError("motion-style evidence video descriptor is non-canonical")

    readback_descriptor, expected_readback_sha256 = (
        _validate_extended_file_descriptor(
            value.get("current_asset_short_readback"),
            "current-asset short-readback receipt",
            internal_hash_field="receipt_sha256",
        )
    )
    readback = _load_json_file(
        Path(readback_descriptor["path"]),
        "current-asset short-readback receipt",
        expected_sha256=readback_descriptor["sha256"],
    )
    media = (
        review.get("outputs", {}).get("media")
        if isinstance(review.get("outputs"), dict)
        else None
    )
    if (
        not isinstance(media, dict)
        or not isinstance(media.get("idle_side"), dict)
        or not isinstance(media.get("walking_side"), dict)
    ):
        raise RuntimeError(
            "current animation review lacks Idle/Walking side readbacks"
        )
    expected_action_readbacks = {
        "Idle": {
            field: media["idle_side"].get(field)
            for field in ("path", "sha256", "size_bytes")
        },
        "Walking": {
            field: media["walking_side"].get(field)
            for field in ("path", "sha256", "size_bytes")
        },
    }
    expected_review_descriptor = {
        field: review_descriptor[field] for field in DESCRIPTOR_FIELDS
    }
    expected_glb_descriptor = {
        field: reviewed_glb_descriptor[field] for field in DESCRIPTOR_FIELDS
    }
    if (
        set(readback)
        != {
            "schema",
            "status",
            "asset_id",
            "animation_review",
            "reviewed_animated_glb",
            "actions",
            "action_readbacks",
            "checks",
            "receipt_sha256",
        }
        or readback.get("schema") != CURRENT_ASSET_SHORT_READBACK_SCHEMA
        or readback.get("status")
        != "passed_current_asset_geometry_and_actions"
        or readback.get("asset_id") != expected_asset_id
        or readback.get("animation_review") != expected_review_descriptor
        or readback.get("reviewed_animated_glb") != expected_glb_descriptor
        or not _has_idle_walking_actions(readback.get("actions"))
        or readback.get("action_readbacks") != expected_action_readbacks
        or readback.get("checks") != CURRENT_ASSET_SHORT_READBACK_CHECKS
        or readback.get("receipt_sha256")
        != _hash_without(readback, "receipt_sha256")
        or readback.get("receipt_sha256") != expected_readback_sha256
    ):
        raise RuntimeError(
            "current-asset short-readback identity binding is invalid"
        )
    action_videos = {
        action: _validate_file_descriptor(
            descriptor,
            f"current-asset {action} short-readback video",
        )
        for action, descriptor in expected_action_readbacks.items()
    }
    if any(
        expected_action_readbacks[action] != descriptor
        for action, descriptor in action_videos.items()
    ):
        raise RuntimeError(
            "current-asset short-readback video descriptor is non-canonical"
        )
    if any(
        style_video["sha256"] == descriptor["sha256"]
        for descriptor in action_videos.values()
    ):
        raise RuntimeError(
            "motion-style video cannot replace current-asset readback"
        )
    return {
        "mode": MOTION_STYLE_AND_CURRENT_READBACK_MODE,
        "motion_style_approval": {
            **style_descriptor,
            "approval_sha256": expected_approval_sha256,
        },
        "current_asset_short_readback": {
            **readback_descriptor,
            "receipt_sha256": expected_readback_sha256,
        },
    }


def _load_texture_transcode_manifest(
    job: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    manifest_path = _direct_absolute_file(
        job.get("texture_transcode_manifest"),
        "texture transcode manifest",
    )
    manifest_descriptor = _validate_file_descriptor(
        {
            "path": str(manifest_path),
            "sha256": job.get("texture_transcode_manifest_sha256"),
            "size_bytes": job.get("texture_transcode_manifest_size_bytes"),
        },
        "texture transcode manifest",
        expected_path=manifest_path,
    )
    manifest = _load_json_file(
        manifest_path,
        "texture transcode manifest",
        expected_sha256=manifest_descriptor["sha256"],
    )
    if (
        manifest_path.stat().st_size != manifest_descriptor["size_bytes"]
        or _sha256(manifest_path) != manifest_descriptor["sha256"]
    ):
        raise RuntimeError("texture transcode manifest changed during validation")
    images = manifest.get("images")
    if (
        set(manifest) != TEXTURE_TRANSCODE_FIELDS
        or manifest.get("schema") != TEXTURE_TRANSCODE_SCHEMA
        or manifest.get("purpose") != TEXTURE_TRANSCODE_PURPOSE
        or manifest.get("geometry_skin_animation_byte_graph_changed") is not False
        or not isinstance(images, list)
        or not images
    ):
        raise RuntimeError("texture transcode manifest contract is invalid")
    _validate_file_descriptor(
        manifest.get("output"),
        "texture transcode output",
        expected_path=source,
    )
    seen_indices = set()
    for image in images:
        if not isinstance(image, dict) or set(image) != TEXTURE_TRANSCODE_IMAGE_FIELDS:
            raise RuntimeError("texture transcode image evidence fields are invalid")
        index = image.get("image_index")
        pixel_size = image.get("pixel_size")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in seen_indices
            or not isinstance(pixel_size, list)
            or len(pixel_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in pixel_size
            )
            or isinstance(image.get("source_size_bytes"), bool)
            or not isinstance(image.get("source_size_bytes"), int)
            or image["source_size_bytes"] <= 0
            or isinstance(image.get("png_size_bytes"), bool)
            or not isinstance(image.get("png_size_bytes"), int)
            or image["png_size_bytes"] <= 0
        ):
            raise RuntimeError("texture transcode image evidence values are invalid")
        for field in ("rgba_sha256", "source_webp_sha256", "png_sha256"):
            _require_sha256(image.get(field), f"texture transcode image {field}")
        seen_indices.add(index)
    return manifest


def _buffer_view_bytes(
    *,
    document: Mapping[str, Any],
    binary: bytes,
    view_index: Any,
    label: str,
) -> bytes:
    buffer_views = document.get("bufferViews")
    if (
        not isinstance(buffer_views, list)
        or isinstance(view_index, bool)
        or not isinstance(view_index, int)
        or not 0 <= view_index < len(buffer_views)
        or not isinstance(buffer_views[view_index], dict)
    ):
        raise RuntimeError(f"{label} bufferView is invalid")
    view = buffer_views[view_index]
    buffer_index = view.get("buffer", 0)
    byte_offset = view.get("byteOffset", 0)
    byte_length = view.get("byteLength")
    if (
        buffer_index != 0
        or isinstance(byte_offset, bool)
        or not isinstance(byte_offset, int)
        or byte_offset < 0
        or isinstance(byte_length, bool)
        or not isinstance(byte_length, int)
        or byte_length <= 0
        or byte_offset + byte_length > len(binary)
    ):
        raise RuntimeError(f"{label} bufferView range is invalid")
    return binary[byte_offset : byte_offset + byte_length]


def _without_webp_extension(
    document: Mapping[str, Any],
    key: str,
) -> list[str] | None:
    values = document.get(key)
    if values is None:
        return None
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
        or len(values) != len(set(values))
    ):
        raise RuntimeError(f"reviewed GLB {key} is invalid")
    filtered = [value for value in values if value != "EXT_texture_webp"]
    return filtered or None


def _validate_texture_transcode_graph(
    *,
    reviewed_document: dict[str, Any],
    reviewed_binary: bytes,
    compatible_document: dict[str, Any],
    compatible_binary: bytes,
    image_evidence: list[dict[str, Any]],
) -> None:
    reviewed_images = reviewed_document.get("images")
    compatible_images = compatible_document.get("images")
    reviewed_views = reviewed_document.get("bufferViews")
    compatible_views = compatible_document.get("bufferViews")
    reviewed_textures = reviewed_document.get("textures")
    compatible_textures = compatible_document.get("textures")
    if (
        not isinstance(reviewed_images, list)
        or not reviewed_images
        or not all(isinstance(image, dict) for image in reviewed_images)
        or not isinstance(compatible_images, list)
        or len(compatible_images) != len(reviewed_images)
        or not all(isinstance(image, dict) for image in compatible_images)
        or not isinstance(reviewed_views, list)
        or not reviewed_views
        or not all(isinstance(view, dict) for view in reviewed_views)
        or not isinstance(compatible_views, list)
        or not all(isinstance(view, dict) for view in compatible_views)
        or not isinstance(reviewed_textures, list)
        or not all(isinstance(texture, dict) for texture in reviewed_textures)
        or not isinstance(compatible_textures, list)
        or not all(isinstance(texture, dict) for texture in compatible_textures)
    ):
        raise RuntimeError("texture transcode GLB graph is incomplete")
    webp_indices = {
        index
        for index, image in enumerate(reviewed_images)
        if image.get("mimeType") == "image/webp"
    }
    evidence_by_index = {
        image["image_index"]: image
        for image in image_evidence
    }
    if (
        not webp_indices
        or set(evidence_by_index) != webp_indices
        or [image["image_index"] for image in image_evidence]
        != sorted(webp_indices)
        or len(compatible_views) != len(reviewed_views) + len(webp_indices)
        or compatible_views[: len(reviewed_views)] != reviewed_views
        or len(compatible_textures) != len(reviewed_textures)
        or compatible_binary[: len(reviewed_binary)] != reviewed_binary
    ):
        raise RuntimeError(
            "texture transcode changed the original GLB byte graph"
        )

    expected_document = copy.deepcopy(reviewed_document)
    expected_document["bufferViews"] = copy.deepcopy(compatible_views)
    expected_document["buffers"][0]["byteLength"] = len(compatible_binary)
    expected_images = expected_document["images"]
    cursor = len(reviewed_binary)
    for ordinal, image_index in enumerate(sorted(webp_indices)):
        evidence = evidence_by_index[image_index]
        expected_offset = (cursor + 3) & ~3
        output_view_index = len(reviewed_views) + ordinal
        output_view = compatible_views[output_view_index]
        if (
            set(output_view) != {"buffer", "byteOffset", "byteLength"}
            or output_view.get("buffer") != 0
            or output_view.get("byteOffset") != expected_offset
            or output_view.get("byteLength") != evidence["png_size_bytes"]
            or any(compatible_binary[cursor:expected_offset])
        ):
            raise RuntimeError(
                "texture transcode appended PNG bufferView is invalid"
            )
        source_image = reviewed_images[image_index]
        source_bytes = _buffer_view_bytes(
            document=reviewed_document,
            binary=reviewed_binary,
            view_index=source_image.get("bufferView"),
            label=f"reviewed WebP image {image_index}",
        )
        png_bytes = _buffer_view_bytes(
            document=compatible_document,
            binary=compatible_binary,
            view_index=output_view_index,
            label=f"compatible PNG image {image_index}",
        )
        if (
            len(source_bytes) != evidence["source_size_bytes"]
            or hashlib.sha256(source_bytes).hexdigest()
            != evidence["source_webp_sha256"]
            or len(png_bytes) != evidence["png_size_bytes"]
            or hashlib.sha256(png_bytes).hexdigest() != evidence["png_sha256"]
        ):
            raise RuntimeError(
                "texture transcode image bytes do not match the manifest"
            )
        expected_image = copy.deepcopy(source_image)
        expected_image["bufferView"] = output_view_index
        expected_image["mimeType"] = "image/png"
        expected_images[image_index] = expected_image
        cursor = expected_offset + len(png_bytes)
    if cursor != len(compatible_binary):
        raise RuntimeError(
            "texture transcode output has unauthenticated appended bytes"
        )
    for index, image in enumerate(reviewed_images):
        if index not in webp_indices and compatible_images[index] != image:
            raise RuntimeError("texture transcode changed a non-WebP image")

    expected_textures = expected_document["textures"]
    for texture in expected_textures:
        extensions = texture.get("extensions")
        webp = (
            extensions.get("EXT_texture_webp")
            if isinstance(extensions, dict)
            else None
        )
        if webp is None:
            continue
        if (
            not isinstance(webp, dict)
            or set(webp) != {"source"}
            or isinstance(webp.get("source"), bool)
            or webp.get("source") not in webp_indices
        ):
            raise RuntimeError("reviewed GLB EXT_texture_webp routing is invalid")
        texture["source"] = webp["source"]
        del extensions["EXT_texture_webp"]
        if not extensions:
            texture.pop("extensions", None)
    for key in ("extensionsUsed", "extensionsRequired"):
        filtered = _without_webp_extension(reviewed_document, key)
        if filtered is None:
            expected_document.pop(key, None)
        else:
            expected_document[key] = filtered
    if compatible_document != expected_document:
        raise RuntimeError(
            "texture transcode changed GLB structure or PBR routing"
        )


def _validate_texture_transcode_lineage(
    job: dict[str, Any],
    *,
    source: Path,
    reviewed_source: Path,
) -> None:
    manifest = _load_texture_transcode_manifest(job, source)
    upstream = _direct_absolute_file(
        job.get("upstream_rigged_glb"),
        "upstream owner-reviewed GLB",
    )
    upstream_sha256 = _require_sha256(
        job.get("upstream_rigged_glb_sha256"),
        "upstream owner-reviewed GLB hash",
    )
    _validate_file_descriptor(
        manifest.get("input"),
        "texture transcode input",
        expected_path=reviewed_source,
    )
    if (
        upstream != reviewed_source
        or upstream_sha256 != _sha256(reviewed_source)
        or source == reviewed_source
        or job["rigged_glb_sha256"] == upstream_sha256
    ):
        raise RuntimeError(
            "texture transcode job does not bind the reviewed upstream runtime"
        )
    reviewed_document, reviewed_binary = _read_glb_document(
        reviewed_source,
        include_binary=True,
    )
    source_document, source_binary = _read_glb_document(
        source,
        include_binary=True,
    )
    _validate_texture_transcode_graph(
        reviewed_document=reviewed_document,
        reviewed_binary=reviewed_binary,
        compatible_document=source_document,
        compatible_binary=source_binary,
        image_evidence=manifest["images"],
    )


def _resolve_relative_descriptor(
    root: Path,
    value: Any,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != DESCRIPTOR_FIELDS:
        raise RuntimeError(f"{label} descriptor fields are invalid")
    raw_path = value.get("path")
    sha256 = _require_sha256(value.get("sha256"), f"{label} descriptor hash")
    size_bytes = value.get("size_bytes")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or "\\" in raw_path
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise RuntimeError(f"{label} descriptor values are invalid")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise RuntimeError(f"{label} descriptor path escapes its preparation root")
    unresolved = root.joinpath(*relative.parts)
    path = _direct_absolute_file(str(unresolved), label)
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"{label} descriptor path escapes its preparation root"
        ) from error
    if path.stat().st_size != size_bytes or _sha256(path) != sha256:
        raise RuntimeError(f"{label} descriptor does not authenticate its file")
    return path, {"path": raw_path, "sha256": sha256, "size_bytes": size_bytes}


def _validate_decision_freeze_receipt(
    preparation: dict[str, Any],
) -> dict[str, Any]:
    receipt = _validate_file_descriptor(
        preparation.get("animation_decision_freeze_receipt"),
        "animation decision freeze receipt",
    )
    expected_receipt_file_sha256 = _require_sha256(
        preparation.get("expected_animation_decision_freeze_receipt_file_sha256"),
        "external animation decision freeze receipt hash",
    )
    expected_receipt_sha256 = _require_sha256(
        preparation.get("animation_decision_freeze_receipt_sha256"),
        "animation decision freeze receipt internal hash",
    )
    if receipt["sha256"] != expected_receipt_file_sha256:
        raise RuntimeError("animation decision freeze receipt external anchor changed")
    receipt_path = Path(receipt["path"])
    payload = _load_json_file(
        receipt_path,
        "animation decision freeze receipt",
        expected_sha256=expected_receipt_file_sha256,
    )
    if payload.get("schema") in LEGACY_DECISION_FREEZE_RECEIPT_SCHEMAS:
        raise RuntimeError(
            "legacy animation decision freeze receipt v1 is audit-only; "
            "regenerate a v2 receipt bound to an authenticated presentation"
        )
    instruction = payload.get("user_instruction_binding")
    authority = payload.get("user_instruction_authority")
    evidence_value = payload.get("presentation_evidence")
    compact_evidence = _uses_compact_approval_evidence(evidence_value)
    expected_instruction_fields = (
        {
            "decision",
            "motion_style_approval_file_sha256",
            "current_asset_short_readback_file_sha256",
            "current_asset_readback_is_machine_gate",
        }
        if compact_evidence
        else {
            "decision",
            "review_sha256",
            "all_six_checks_explicit",
            "presentation_receipt_file_sha256",
        }
    )
    if (
        set(payload) != DECISION_FREEZE_RECEIPT_FIELDS
        or payload.get("schema") != DECISION_FREEZE_RECEIPT_SCHEMA
        or payload.get("status") != "frozen"
        or payload.get("state_classification") != "research_candidate"
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("receipt_sha256") != _hash_without(payload, "receipt_sha256")
        or payload.get("receipt_sha256") != expected_receipt_sha256
        or authority != USER_INSTRUCTION_AUTHORITY
        or preparation.get("user_instruction_authority") != USER_INSTRUCTION_AUTHORITY
        or payload.get("source_asset_registry_validation_mode")
        not in SOURCE_REGISTRY_VALIDATION_MODES
        or not isinstance(instruction, dict)
        or set(instruction) != expected_instruction_fields
        or instruction.get("decision") != "approved_for_ue_apartment"
        or (
            not compact_evidence
            and instruction.get("all_six_checks_explicit") is not True
        )
        or (
            compact_evidence
            and instruction.get("current_asset_readback_is_machine_gate")
            is not True
        )
        or isinstance(payload.get("authenticated_review_artifact_count"), bool)
        or not isinstance(payload.get("authenticated_review_artifact_count"), int)
        or payload["authenticated_review_artifact_count"] <= 0
    ):
        raise RuntimeError("animation decision freeze receipt authority is invalid")

    source_asset = _validate_file_descriptor(
        preparation.get("source_asset"),
        "canonical source asset",
    )
    source_registry = _validate_file_descriptor(
        preparation.get("source_asset_registry"),
        "source asset registry",
    )
    review = _validate_file_descriptor(
        preparation.get("animation_review"),
        "animation review",
    )
    reviewed_glb = _validate_file_descriptor(
        preparation.get("reviewed_animated_glb"),
        "reviewed animated GLB",
    )
    decision = _validate_file_descriptor(
        preparation.get("animation_decision"),
        "animation decision",
    )
    review_payload = None
    for descriptor, label in (
        (source_asset, "canonical source asset"),
        (source_registry, "source asset registry"),
        (review, "animation review"),
        (decision, "animation decision"),
    ):
        loaded = _load_json_file(Path(descriptor["path"]), label)
        if descriptor is review:
            review_payload = loaded
    if review_payload is None:
        raise RuntimeError("animation review could not be authenticated")
    if (
        payload.get("source_asset") != source_asset
        or payload.get("source_asset_registry") != source_registry
        or payload.get("animation_review") != review
        or payload.get("expected_source_asset_registry_file_sha256")
        != preparation.get("expected_source_asset_registry_file_sha256")
        or payload.get("source_asset_registry_validation_mode")
        != preparation.get("source_asset_registry_validation_mode")
        or payload.get("expected_animation_review_file_sha256") != review["sha256"]
        or (
            not compact_evidence
            and instruction.get("review_sha256") != review["sha256"]
        )
        or isinstance(payload.get("authenticated_review_artifact_count"), bool)
        or payload.get("authenticated_review_artifact_count")
        != preparation.get("authenticated_review_artifact_count")
        or payload.get("decision_sha256")
        != preparation.get("animation_decision_sha256")
    ):
        raise RuntimeError("animation decision freeze receipt artifact lineage changed")
    canonical_identity = preparation.get("canonical_identity")
    if compact_evidence and not isinstance(canonical_identity, dict):
        raise RuntimeError("preparation canonical identity is invalid")
    if compact_evidence:
        canonical_evidence = _validate_compact_approval_evidence(
            evidence_value,
            expected_asset_id=str(canonical_identity.get("asset_id", "")),
            review=review_payload,
            review_descriptor=review,
            reviewed_glb_descriptor=reviewed_glb,
        )
        if (
            canonical_evidence != evidence_value
            or evidence_value != preparation.get("presentation_evidence")
            or instruction.get("motion_style_approval_file_sha256")
            != canonical_evidence["motion_style_approval"]["sha256"]
            or instruction.get("current_asset_short_readback_file_sha256")
            != canonical_evidence["current_asset_short_readback"]["sha256"]
        ):
            raise RuntimeError(
                "animation decision freeze receipt compact approval binding changed"
            )
    else:
        presentation_evidence = evidence_value
        if (
            not isinstance(presentation_evidence, dict)
            or set(presentation_evidence) != PRESENTATION_EVIDENCE_FIELDS
            or presentation_evidence != preparation.get("presentation_evidence")
            or instruction.get("presentation_receipt_file_sha256")
            != presentation_evidence.get(
                "expected_presentation_receipt_file_sha256"
            )
        ):
            raise RuntimeError(
                "animation decision freeze receipt presentation binding changed"
            )
        presentation_receipt = _validate_file_descriptor(
            presentation_evidence.get("presentation_receipt"),
            "owner-review presentation receipt",
        )
        expected_presentation_receipt_sha256 = _require_sha256(
            presentation_evidence.get("expected_presentation_receipt_file_sha256"),
            "external owner-review presentation receipt hash",
        )
        if presentation_receipt["sha256"] != expected_presentation_receipt_sha256:
            raise RuntimeError(
                "owner-review presentation receipt external anchor changed"
            )
        try:
            presentation_payload, raw_presentation_record = (
                presentation.load_presentation_receipt(
                    Path(presentation_receipt["path"]),
                    expected_presentation_receipt_sha256,
                    expected_source_review_sha256=review["sha256"],
                )
            )
        except (
            presentation.PresentationContractError,
            contracts.StrictJSONError,
            OSError,
            ValueError,
        ) as error:
            raise RuntimeError(
                f"animation decision presentation evidence is invalid: {error}"
            ) from error
        canonical_presentation = {
            "presentation_receipt": raw_presentation_record,
            "expected_presentation_receipt_file_sha256": (
                expected_presentation_receipt_sha256
            ),
            "presentation_receipt_sha256": presentation_payload[
                "receipt_sha256"
            ],
            "output_video": presentation_payload["output"],
        }
        if canonical_presentation != presentation_evidence:
            raise RuntimeError(
                "animation decision presentation evidence is non-canonical"
            )
    decision_path, decision_relative = _resolve_relative_descriptor(
        receipt_path.parent,
        payload.get("animation_decision"),
        "freeze receipt animation decision",
    )
    if (
        decision_path != Path(decision["path"])
        or decision_relative["sha256"] != decision["sha256"]
    ):
        raise RuntimeError("animation decision freeze receipt decision binding changed")
    decision_payload = _load_json_file(decision_path, "animation decision")
    if (
        decision_payload.get("decision") != "approved_for_ue_apartment"
        or decision_payload.get("review_sha256") != review["sha256"]
        or decision_payload.get("decision_sha256")
        != _hash_without(decision_payload, "decision_sha256")
        or decision_payload.get("decision_sha256")
        != preparation.get("animation_decision_sha256")
    ):
        raise RuntimeError(
            "animation decision freeze receipt approved decision changed"
        )
    return receipt


def _validate_preparation_anchor(
    preparation_path: Path,
    expected_preparation_sha256: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preparation = _load_json_file(
        preparation_path,
        "UE import preparation",
        expected_sha256=expected_preparation_sha256,
    )
    if preparation.get("schema") in LEGACY_PREPARATION_SCHEMAS:
        raise RuntimeError(
            "legacy UE import preparation v1/v2 is audit-only; regenerate v3 "
            "with an authenticated presentation-bound decision-freeze receipt"
        )
    if (
        set(preparation) != PREPARATION_FIELDS
        or preparation.get("schema") != PREPARATION_SCHEMA
        or preparation.get("status") != "ready_for_new_ue_import"
        or preparation.get("state_classification") != "research_candidate"
        or preparation.get("formal_dataset_registration_authorized") is not False
        or preparation.get("manifest_sha256")
        != _hash_without(preparation, "manifest_sha256")
    ):
        raise RuntimeError("UE import preparation contract is invalid")
    automatic_checks = preparation.get("automatic_checks")
    expected_automatic_check_fields = (
        COMPACT_PREPARATION_AUTOMATIC_CHECK_FIELDS
        if _uses_compact_approval_evidence(
            preparation.get("presentation_evidence")
        )
        else PREPARATION_AUTOMATIC_CHECK_FIELDS
    )
    if (
        not isinstance(automatic_checks, dict)
        or set(automatic_checks) != expected_automatic_check_fields
        or any(
            automatic_checks.get(field) is not True
            for field in expected_automatic_check_fields - {"overall"}
        )
        or automatic_checks.get("overall") != "passed"
    ):
        raise RuntimeError("UE import preparation checks do not authorize execution")
    _validate_decision_freeze_receipt(preparation)

    anchored_path, batch_descriptor = _resolve_relative_descriptor(
        preparation_path.parent,
        preparation.get("ue_import_jobs"),
        "UE import batch",
    )
    if (
        anchored_path != manifest_path
        or batch_descriptor["sha256"] != expected_manifest_sha256
    ):
        raise RuntimeError(
            "external UE import manifest path/SHA does not match preparation anchor"
        )
    preparation_descriptor = {
        "path": str(preparation_path),
        "sha256": expected_preparation_sha256,
        "size_bytes": preparation_path.stat().st_size,
        "manifest_sha256": preparation["manifest_sha256"],
    }
    return (
        preparation,
        preparation_descriptor,
        {
            "path": str(manifest_path),
            "sha256": batch_descriptor["sha256"],
            "size_bytes": batch_descriptor["size_bytes"],
        },
    )


def _validate_batch_payload(
    manifest: dict[str, Any],
    preparation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if manifest.get("schema") == LEGACY_BATCH_SCHEMA:
        raise RuntimeError(
            "legacy Pixal UE import batch v1 cannot authorize writes; "
            "regenerate an authenticated v2 batch"
        )
    if (
        set(manifest) != BATCH_FIELDS
        or manifest.get("schema") != BATCH_SCHEMA
        or manifest.get("status") != "ready_for_new_ue_import"
        or manifest.get("state_classification") != "research_candidate"
        or manifest.get("formal_dataset_registration_authorized") is not False
        or manifest.get("job_type") != JOB_TYPE
        or manifest.get("job_count") != 1
        or manifest.get("batch_sha256") != _hash_without(manifest, "batch_sha256")
    ):
        raise RuntimeError("Pixal animal UE batch top-level contract is invalid")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != manifest["job_count"]:
        raise RuntimeError("Pixal animal UE batch job count is invalid")

    canonical_identity = preparation.get("canonical_identity")
    if (
        not isinstance(canonical_identity, dict)
        or set(canonical_identity) != CANONICAL_IDENTITY_FIELDS
    ):
        raise RuntimeError("preparation canonical identity is invalid")
    job = jobs[0]
    if (
        not isinstance(job, dict)
        or set(job) not in (JOB_FIELDS, TRANSCODED_JOB_FIELDS)
    ):
        raise RuntimeError("Pixal animal UE job fields are invalid")
    asset_id = job.get("asset_id")
    expected_tag = f"pixal_{asset_id}"
    if (
        job.get("job_type") != JOB_TYPE
        or not isinstance(asset_id, str)
        or len(asset_id) > 96
        or CANONICAL_ID_PATTERN.fullmatch(asset_id) is None
        or job.get("legacy_tag") != asset_id
        or job.get("tag") != expected_tag
        or len(expected_tag) > 104
        or job.get("expected_actions") != EXPECTED_ACTIONS
        or not isinstance(job.get("rigged_glb"), str)
        or not Path(job["rigged_glb"]).is_absolute()
        or not isinstance(job.get("profile_schema_id"), str)
        or not job["profile_schema_id"]
        or not isinstance(job.get("sampled_attributes"), dict)
    ):
        raise RuntimeError("Pixal animal UE job identity/actions are invalid")
    for field in (
        "rigged_glb_sha256",
        "source_registry_sha256",
        "source_asset_sha256",
        "request_sha256",
        "animation_decision_file_sha256",
        "animation_decision_sha256",
    ):
        _require_sha256(job.get(field), f"Pixal animal UE job {field}")

    expected_policy = (
        f"new unique gate_{expected_tag} content directories; never rewrite or "
        "reuse any historical generated-animal UE job/content directory"
    )
    if manifest.get("non_destructive_policy") != expected_policy:
        raise RuntimeError("Pixal animal UE non-destructive policy is invalid")

    for field in (
        "asset_id",
        "legacy_tag",
        "tag",
        "profile_schema_id",
        "sampled_attributes",
        "request_sha256",
    ):
        if job[field] != canonical_identity.get(field):
            raise RuntimeError(f"Pixal animal UE job identity diverged at {field}")
    source_asset = _validate_file_descriptor(
        preparation.get("source_asset"), "canonical source asset"
    )
    source_registry = _validate_file_descriptor(
        preparation.get("source_asset_registry"), "source asset registry"
    )
    animation_decision = _validate_file_descriptor(
        preparation.get("animation_decision"), "animation decision"
    )
    reviewed_glb = _validate_file_descriptor(
        preparation.get("reviewed_animated_glb"), "reviewed animated GLB"
    )
    source = _direct_absolute_file(job["rigged_glb"], "UE import source")
    if _sha256(source) != job["rigged_glb_sha256"]:
        raise RuntimeError("Pixal animal UE job source hash is invalid")
    if set(job) == TRANSCODED_JOB_FIELDS:
        _validate_texture_transcode_lineage(
            job,
            source=source,
            reviewed_source=Path(reviewed_glb["path"]),
        )
        reviewed_runtime_lineage_valid = True
    else:
        reviewed_runtime_lineage_valid = (
            source == Path(reviewed_glb["path"])
            and job["rigged_glb_sha256"] == reviewed_glb["sha256"]
        )
    if (
        job["source_asset_sha256"] != source_asset["sha256"]
        or job["source_registry_sha256"] != source_registry["sha256"]
        or preparation.get("expected_source_asset_registry_file_sha256")
        != source_registry["sha256"]
        or job["animation_decision_file_sha256"] != animation_decision["sha256"]
        or preparation.get("expected_animation_decision_file_sha256")
        != animation_decision["sha256"]
        or job["animation_decision_sha256"]
        != preparation.get("animation_decision_sha256")
        or not reviewed_runtime_lineage_valid
    ):
        raise RuntimeError("Pixal animal UE job artifact lineage is invalid")

    job_identity_sha256 = _json_sha256(job)
    batch_identity = {
        "schema": manifest["schema"],
        "job_type": manifest["job_type"],
        "job_count": manifest["job_count"],
        "batch_sha256": manifest["batch_sha256"],
        "asset_ids": [asset_id],
        "tags": [expected_tag],
        "job_identity_sha256s": [job_identity_sha256],
    }
    return jobs, batch_identity


def _authenticate_import_contract() -> dict[str, Any]:
    preparation_path = _direct_absolute_file(
        _required_environment("PIXAL_ANIMAL_IMPORT_PREPARATION"),
        "UE import preparation manifest",
    )
    expected_preparation_sha256 = _require_sha256(
        _required_environment("PIXAL_ANIMAL_IMPORT_PREPARATION_SHA256"),
        "external UE import preparation hash",
    )
    manifest_path = _direct_absolute_file(
        _required_environment("PIXAL_ANIMAL_IMPORT_MANIFEST"),
        "UE import batch manifest",
    )
    expected_manifest_sha256 = _require_sha256(
        _required_environment("PIXAL_ANIMAL_IMPORT_MANIFEST_SHA256"),
        "external UE import batch hash",
    )
    result_path = _validate_result_target(
        _required_environment("PIXAL_ANIMAL_IMPORT_RESULT")
    )
    preparation, preparation_descriptor, manifest_descriptor = (
        _validate_preparation_anchor(
            preparation_path,
            expected_preparation_sha256,
            manifest_path,
            expected_manifest_sha256,
        )
    )
    manifest = _load_json_file(
        manifest_path,
        "UE import batch",
        expected_sha256=expected_manifest_sha256,
    )
    jobs, batch_identity = _validate_batch_payload(manifest, preparation)

    validated_jobs = []
    for job in jobs:
        source = _direct_absolute_file(
            job["rigged_glb"], f"rigged GLB for {job['tag']}"
        )
        if _sha256(source) != job["rigged_glb_sha256"]:
            raise RuntimeError(f"source hash mismatch for {job['tag']}: {source}")
        _validate_ue_compatible_glb(job, source)
        validated_jobs.append(
            {
                "job": job,
                "source": source,
                "job_identity_sha256": _json_sha256(job),
            }
        )
    manifest_descriptor["batch_sha256"] = manifest["batch_sha256"]
    return {
        "preparation": preparation,
        "preparation_descriptor": preparation_descriptor,
        "manifest": manifest,
        "manifest_descriptor": manifest_descriptor,
        "manifest_path": manifest_path,
        "result_path": result_path,
        "batch_identity": batch_identity,
        "validated_jobs": validated_jobs,
    }


def _require_import_contract_unchanged(contract: dict[str, Any]) -> None:
    observed = _authenticate_import_contract()
    if observed != contract:
        raise RuntimeError(
            "authenticated UE import authority graph changed during execution"
        )


def _content_targets(tag: str) -> tuple[str, str, str]:
    mesh_dir = f"/Game/MyAssets/Audioset/Meshes/gate_{tag}"
    bp_dir = f"/Game/MyAssets/Audioset/Blueprints/gate_{tag}"
    bp_path = f"{bp_dir}/BP_gate_{tag}"
    return mesh_dir, bp_dir, bp_path


def _assert_content_targets_absent(tags: list[str]) -> None:
    existing = []
    for tag in tags:
        mesh_dir, bp_dir, bp_path = _content_targets(tag)
        existing.extend(
            directory
            for directory in (mesh_dir, bp_dir)
            if unreal.EditorAssetLibrary.does_directory_exist(directory_path=directory)
        )
        if unreal.EditorAssetLibrary.does_asset_exist(asset_path=bp_path):
            existing.append(bp_path)
    if existing:
        raise RuntimeError(
            "refusing to replace existing Pixal UE content target(s): "
            + ", ".join(sorted(set(existing)))
        )


def _write_result_no_replace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise RuntimeError(
            f"refusing to replace existing Pixal UE import result: {path}"
        ) from error


def _object_path(asset_path: str) -> tuple[str, str, str]:
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=asset_path)
    class_path = data.get_editor_property(name="asset_class_path")
    class_name = str(class_path.get_editor_property(name="asset_name"))
    package_dir = str(data.get_editor_property(name="package_path"))
    asset_name = str(data.get_editor_property(name="asset_name"))
    return class_name, asset_name, f"{package_dir}/{asset_name}.{asset_name}"


def _run_one(
    validated_job: dict[str, Any],
    manifest: dict[str, Any],
    input_manifest_sha256: str,
) -> dict:
    job = validated_job["job"]
    source = validated_job["source"]
    job_identity_sha256 = validated_job["job_identity_sha256"]
    identity = {
        "schema": IMPORT_JOB_IDENTITY_SCHEMA,
        "job_type": job["job_type"],
        "asset_id": job["asset_id"],
        "legacy_tag": job["legacy_tag"],
        "tag": job["tag"],
        "expected_actions": job["expected_actions"],
        "rigged_glb": str(source),
        "rigged_glb_sha256": job["rigged_glb_sha256"],
        "input_manifest_sha256": input_manifest_sha256,
        "batch_sha256": manifest["batch_sha256"],
        "job_identity_sha256": job_identity_sha256,
    }
    if set(identity) != JOB_IDENTITY_FIELDS:
        raise RuntimeError("internal per-job UE import identity is incomplete")
    environment = {
        "GATE_TAG": job["tag"],
        "GATE_RIGGED_GLB": str(source),
        "GATE_IMPORT_JOB_JSON": _canonical_json(identity),
    }
    saved = {
        name: os.environ.get(name)
        for name in (*environment, "GATE_IMPORT_RECEIPT_JSON")
    }
    try:
        os.environ.update(environment)
        os.environ.pop("GATE_IMPORT_RECEIPT_JSON", None)
        runpy.run_path(str(IMPORT_ONE), run_name="__main__")
        receipt_value = os.environ.get("GATE_IMPORT_RECEIPT_JSON")
        if receipt_value is None:
            raise RuntimeError(
                f"per-asset UE importer emitted no receipt for {job['tag']}"
            )
        receipt = _load_json_text(receipt_value, "per-asset UE import receipt")
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if (
        set(receipt) != JOB_RECEIPT_FIELDS
        or receipt.get("schema") != IMPORT_JOB_RECEIPT_SCHEMA
        or receipt.get("tag") != job["tag"]
        or receipt.get("job_identity_sha256") != job_identity_sha256
        or receipt.get("actions") != EXPECTED_ACTIONS
    ):
        raise RuntimeError(f"per-asset UE import receipt mismatch: {job['tag']}")
    return receipt


def _readback_import(
    validated_job: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    job = validated_job["job"]
    source = validated_job["source"]
    mesh_dir, _bp_dir, bp_path = _content_targets(job["tag"])
    imported = unreal.EditorAssetLibrary.list_assets(
        directory_path=mesh_dir, recursive=True
    )
    skeletal_meshes = []
    animation_paths: dict[str, str] = {}
    for asset in imported:
        class_name, asset_name, object_path = _object_path(asset)
        if class_name == "SkeletalMesh":
            skeletal_meshes.append(object_path)
        elif class_name == "AnimSequence":
            if asset_name in animation_paths:
                raise RuntimeError(
                    f"duplicate AnimSequence action for {job['tag']}: {asset_name}"
                )
            animation_paths[asset_name] = object_path
    if (
        len(skeletal_meshes) != 1
        or set(animation_paths) != set(EXPECTED_ACTIONS)
        or len(animation_paths) != len(EXPECTED_ACTIONS)
        or not unreal.EditorAssetLibrary.does_asset_exist(asset_path=bp_path)
        or receipt["skeletal_mesh"] != skeletal_meshes[0]
        or receipt["walking_animation"] != animation_paths["Walking"]
        or receipt["blueprint"] != bp_path
    ):
        raise RuntimeError(
            f"UE import exact readback failed for {job['tag']}: "
            f"skeletal_meshes={skeletal_meshes} "
            f"actions={sorted(animation_paths)} bp={bp_path}"
        )
    return {
        "job_type": job["job_type"],
        "asset_id": job["asset_id"],
        "tag": job["tag"],
        "legacy_tag": job["legacy_tag"],
        "job_identity_sha256": validated_job["job_identity_sha256"],
        "source": str(source),
        "source_sha256": job["rigged_glb_sha256"],
        "mesh_content_dir": mesh_dir,
        "skeletal_mesh": skeletal_meshes[0],
        "walking_animation": animation_paths["Walking"],
        "blueprint": bp_path,
        "asset_count": len(imported),
        "assets": sorted(imported),
        "actions": EXPECTED_ACTIONS,
        "status": "passed",
    }


def _snapshot_content_targets(tags: list[str]) -> list[dict[str, Any]]:
    snapshots = []
    for tag in tags:
        mesh_dir, bp_dir, bp_path = _content_targets(tag)
        directories = []
        assets = []
        errors = []
        for directory in (mesh_dir, bp_dir):
            try:
                if unreal.EditorAssetLibrary.does_directory_exist(
                    directory_path=directory
                ):
                    directories.append(directory)
                    assets.extend(
                        unreal.EditorAssetLibrary.list_assets(
                            directory_path=directory,
                            recursive=True,
                        )
                    )
            except Exception as error:
                errors.append(f"{directory}: {type(error).__name__}: {error}")
        try:
            blueprint_exists = unreal.EditorAssetLibrary.does_asset_exist(
                asset_path=bp_path
            )
        except Exception as error:
            blueprint_exists = None
            errors.append(f"{bp_path}: {type(error).__name__}: {error}")
        snapshots.append(
            {
                "tag": tag,
                "directories": sorted(set(directories)),
                "assets": sorted(set(assets)),
                "blueprint": bp_path,
                "blueprint_exists": blueprint_exists,
                "readback_errors": errors,
                "manual_cleanup_required": bool(
                    directories or assets or blueprint_exists or errors
                ),
            }
        )
    return snapshots


def _base_result(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "preparation_manifest": contract["preparation_descriptor"],
        "input_manifest": contract["manifest_descriptor"],
        "batch_identity": contract["batch_identity"],
        "non_destructive_policy": contract["manifest"]["non_destructive_policy"],
    }


def main() -> None:
    contract = _authenticate_import_contract()
    tags = [item["job"]["tag"] for item in contract["validated_jobs"]]

    # This is the last preflight before the first UE mutation.  Every file,
    # identity, hash, policy, action contract, and destination is authenticated
    # before the per-asset importer can create either content directory.
    _assert_content_targets_absent(tags)
    _require_import_contract_unchanged(contract)

    results = []
    write_started = False
    active_job: dict[str, Any] | None = None
    try:
        for validated_job in contract["validated_jobs"]:
            active_job = validated_job["job"]
            write_started = True
            receipt = _run_one(
                validated_job,
                contract["manifest"],
                contract["manifest_descriptor"]["sha256"],
            )
            results.append(_readback_import(validated_job, receipt))
        success_result = _base_result(contract)
        _require_import_contract_unchanged(contract)
        success_result.update(
            {
                "status": "passed",
                "passed_count": len(results),
                "results": results,
            }
        )
        _write_result_no_replace(contract["result_path"], success_result)
    except Exception as error:
        if write_started:
            failure_result = _base_result(contract)
            failure_result.update(
                {
                    "status": "failed_write_residue_requires_manual_audit",
                    "passed_count": len(results),
                    "results": results,
                    "failure": {
                        "job_type": (
                            active_job.get("job_type")
                            if active_job is not None
                            else None
                        ),
                        "asset_id": (
                            active_job.get("asset_id")
                            if active_job is not None
                            else None
                        ),
                        "tag": (
                            active_job.get("tag") if active_job is not None else None
                        ),
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "residue": _snapshot_content_targets(tags),
                        "existing_content_was_deleted": False,
                    },
                }
            )
            try:
                _write_result_no_replace(contract["result_path"], failure_result)
            except Exception as receipt_error:
                raise RuntimeError(
                    "UE import failed and its exclusive failure receipt could "
                    f"not be written: import_error={error}; "
                    f"receipt_error={receipt_error}"
                ) from error
        raise

    print(
        "PIXAL_ANIMAL_UE_BATCH_IMPORT_OK "
        f"jobs={len(results)} result={contract['result_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
