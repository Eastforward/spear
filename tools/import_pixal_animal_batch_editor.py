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

from tools import controlled_source_asset_schema as contracts


SCRIPT_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = SCRIPT_DIR.parent
SPEAR_TMP_BRIDGE = (SPEAR_ROOT / "tmp").absolute()
IMPORT_ONE = SCRIPT_DIR / "import_gate_animal_editor.py"
BATCH_SCHEMA = "pixal_animal_ue_import_batch_v2"
LEGACY_BATCH_SCHEMA = "pixal_animal_ue_import_batch_v1"
PREPARATION_SCHEMA = "avengine_user_approved_generated_animal_ue_import_preparation_v2"
LEGACY_PREPARATION_SCHEMA = (
    "avengine_user_approved_generated_animal_ue_import_preparation_v1"
)
DECISION_FREEZE_RECEIPT_SCHEMA = (
    "avengine_target_native_generated_animal_animation_decision_freeze_receipt_v1"
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
    "user_instruction_authority",
    "reviewed_animated_glb",
    "authenticated_review_artifact_count",
    "ue_import_jobs",
    "automatic_checks",
    "manifest_sha256",
}
PREPARATION_AUTOMATIC_CHECK_FIELDS = {
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


def _read_glb_document(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise RuntimeError(f"source is not a GLB 2.0 file: {path}")
    version, declared_size = struct.unpack_from("<II", raw, 4)
    json_size, json_type = struct.unpack_from("<I4s", raw, 12)
    if (
        version != 2
        or declared_size != len(raw)
        or json_type != b"JSON"
        or 20 + json_size > len(raw)
    ):
        raise RuntimeError(f"invalid GLB header or JSON chunk: {path}")
    try:
        document = contracts.strict_json_loads(
            raw[20 : 20 + json_size].rstrip(b" \x00")
        )
    except contracts.StrictJSONError as error:
        raise RuntimeError(f"invalid GLB JSON document: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"GLB JSON document is not an object: {path}")
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
    images = document.get("images", [])
    if (
        not isinstance(required_value, list)
        or any(not isinstance(value, str) for value in required_value)
        or not isinstance(images, list)
        or any(not isinstance(image, dict) for image in images)
    ):
        raise RuntimeError(f"GLB extension/image contract is invalid: {source}")
    required = set(required_value)
    webp_images = [
        index
        for index, image in enumerate(images)
        if image.get("mimeType") == "image/webp"
    ]
    if "EXT_texture_webp" in required or webp_images:
        raise RuntimeError(
            "UE 5.5 import source still requires embedded WebP; run "
            f"transcode_glb_webp_to_png.py first: {source}"
        )
    manifest_value = job.get("texture_transcode_manifest")
    if manifest_value is None:
        return
    manifest_path = _direct_absolute_file(
        manifest_value,
        "texture transcode manifest",
    )
    manifest = _load_json_file(manifest_path, "texture transcode manifest")
    if (
        manifest.get("schema") != "glb_embedded_webp_to_png_transcode_v1"
        or manifest.get("geometry_skin_animation_byte_graph_changed") is not False
        or Path(manifest.get("output", {}).get("path", "")).resolve() != source
        or manifest.get("output", {}).get("sha256") != _sha256(source)
    ):
        raise RuntimeError(f"texture transcode evidence does not authenticate {source}")


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
    instruction = payload.get("user_instruction_binding")
    authority = payload.get("user_instruction_authority")
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
        or set(instruction) != {"decision", "review_sha256", "all_six_checks_explicit"}
        or instruction.get("decision") != "approved_for_ue_apartment"
        or instruction.get("all_six_checks_explicit") is not True
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
    decision = _validate_file_descriptor(
        preparation.get("animation_decision"),
        "animation decision",
    )
    for descriptor, label in (
        (source_asset, "canonical source asset"),
        (source_registry, "source asset registry"),
        (review, "animation review"),
        (decision, "animation decision"),
    ):
        _load_json_file(Path(descriptor["path"]), label)
    if (
        payload.get("source_asset") != source_asset
        or payload.get("source_asset_registry") != source_registry
        or payload.get("animation_review") != review
        or payload.get("expected_source_asset_registry_file_sha256")
        != preparation.get("expected_source_asset_registry_file_sha256")
        or payload.get("source_asset_registry_validation_mode")
        != preparation.get("source_asset_registry_validation_mode")
        or payload.get("expected_animation_review_file_sha256") != review["sha256"]
        or instruction.get("review_sha256") != review["sha256"]
        or isinstance(payload.get("authenticated_review_artifact_count"), bool)
        or payload.get("authenticated_review_artifact_count")
        != preparation.get("authenticated_review_artifact_count")
        or payload.get("decision_sha256")
        != preparation.get("animation_decision_sha256")
    ):
        raise RuntimeError("animation decision freeze receipt artifact lineage changed")
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
    if preparation.get("schema") == LEGACY_PREPARATION_SCHEMA:
        raise RuntimeError(
            "legacy UE import preparation v1 is audit-only; regenerate v2 "
            "with an authenticated decision-freeze receipt"
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
    if (
        not isinstance(automatic_checks, dict)
        or set(automatic_checks) != PREPARATION_AUTOMATIC_CHECK_FIELDS
        or any(
            automatic_checks.get(field) is not True
            for field in PREPARATION_AUTOMATIC_CHECK_FIELDS - {"overall"}
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
    if not isinstance(job, dict) or set(job) != JOB_FIELDS:
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
        or Path(job["rigged_glb"]).resolve() != Path(reviewed_glb["path"])
        or job["rigged_glb_sha256"] != reviewed_glb["sha256"]
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
