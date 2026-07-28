#!/usr/bin/env python3
"""Derive a new UE import job from one canonical, user-approved animal.

This bridge is deliberately non-executing: it reauthenticates the immutable
``source_asset_v2``, the generated-animation review and its human approval, then
publishes a fresh canonical UE job.  It never edits an older job or invokes UE.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import compose_target_native_generated_quadruped_owner_review as presentation
from tools import controlled_source_asset_schema as contracts
from tools import register_controlled_animal_source_assets as source_registry
from tools import run_target_native_generated_quadruped_review as generated_review
from tools.generated_animal_forward_contract import (
    ForwardContractError,
    assert_declared_motion_basis,
    assert_declared_motion_donor_artifact,
    expected_motion_donor_contract,
    load_forward_declaration,
)
from tools.generated_animal_tokenrig_closure import (
    validate_tokenrig_closure_manifest,
)

SCHEMA = "avengine_user_approved_generated_animal_ue_import_preparation_v3"
LEGACY_PREPARATION_SCHEMAS = frozenset(
    {
        "avengine_user_approved_generated_animal_ue_import_preparation_v1",
        "avengine_user_approved_generated_animal_ue_import_preparation_v2",
    }
)
IMPORT_SCHEMA = "pixal_animal_ue_import_batch_v2"
IMPORT_JOB_TYPE = "user_approved_generated_animal"
TEXTURE_TRANSCODE_SCHEMA = "glb_embedded_webp_to_png_transcode_v1"
TEXTURE_TRANSCODE_PURPOSE = "UE_5.5_interchange_compatibility"
TEXTURE_TRANSCODE_FIELDS = frozenset(
    {
        "schema",
        "purpose",
        "geometry_skin_animation_byte_graph_changed",
        "input",
        "output",
        "images",
    }
)
TEXTURE_TRANSCODE_IMAGE_FIELDS = frozenset(
    {
        "image_index",
        "pixel_size",
        "rgba_sha256",
        "source_webp_sha256",
        "source_size_bytes",
        "png_sha256",
        "png_size_bytes",
    }
)
DECISION_SCHEMA = "avengine_controlled_animal_animation_decision_v1"
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
DECISION_FREEZE_RECEIPT_FIELDS = frozenset(
    {
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
)
PRESENTATION_EVIDENCE_FIELDS = frozenset(
    {
        "presentation_receipt",
        "expected_presentation_receipt_file_sha256",
        "presentation_receipt_sha256",
        "output_video",
    }
)
PRESENTATION_AUTOMATIC_CHECKS = {
    "presentation_receipt_raw_file_sha256_reauthenticated": True,
    "presentation_receipt_internal_sha256_reauthenticated": True,
    "presentation_exact_v4_review_sha256_reauthenticated": True,
    "presentation_output_video_bytes_and_directory_reauthenticated": True,
}
DECISION_CHECK_FIELDS = frozenset(
    {
        "walking_direction",
        "walking_limb_deformation",
        "walking_ground_contact",
        "idle_ground_contact",
        "body_stability",
        "detached_geometry_absent",
    }
)
LEGACY_GENERATED_REVIEW_SCHEMA = (
    "avengine_target_native_generated_quadruped_review_run_v3"
)
BRANCHED_GENERATED_REVIEW_SCHEMA = (
    "avengine_target_native_generated_quadruped_review_run_v4"
)
GENERATED_REVIEW_SCHEMAS = frozenset(
    {
        LEGACY_GENERATED_REVIEW_SCHEMA,
        BRANCHED_GENERATED_REVIEW_SCHEMA,
    }
)
REQUIRED_MEDIA = frozenset(
    {
        "walking_side",
        "walking_front",
        "walking_rear",
        "idle_side",
        "idle_front",
        "idle_rear",
    }
)
REQUIRED_REVIEW_OUTPUTS = frozenset(
    {
        "heading_manifest",
        "rig_audit",
        "support_plane_manifest",
        "support_plane_output_readback",
        "retargeted_animated_glb",
        "animated_glb",
        "retarget_manifest",
        "gait_direction_audit_initial",
        "gait_direction_audit",
        "deformation_audit_initial",
        "deformation_audit",
    }
)
REVIEW_FILE_INPUTS = frozenset(
    {
        "target_rig_glb",
        "heading_review_evidence",
        "source_motion_glb",
    }
)
REVIEW_INPUTS = REVIEW_FILE_INPUTS | frozenset({"target_rig_lineage"})
REVIEW_OUTPUT_SCALARS = frozenset(
    {
        "gait_direction_status",
        "deformation_overall",
    }
)
REVIEW_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "created_at",
        "status",
        "formal_dataset_registration_authorized",
        "forward_contract",
        "pipeline_order",
        "automatic_admission_gates",
        "inputs",
        "outputs",
        "timings_seconds",
    }
)
LEGACY_REVIEW_GATE_FIELDS = frozenset(
    {
        "heading",
        "rig",
        "support_plane",
        "retarget_export_front_axis",
        "gait_initial",
        "deformation_initial",
        "weight_repair_policy",
        "weight_repair_triggered",
        "weight_repair",
        "gait_final",
        "deformation_final",
        "all_automatic_gates_passed",
    }
)
REVIEW_GATE_FIELDS = LEGACY_REVIEW_GATE_FIELDS | frozenset(
    {
        "weight_repair_strategy",
        "weight_repair_branch",
        "weight_repair_attempts",
        "weight_repair_final_artifact",
    }
)
LEGACY_FORWARD_CONTRACT_FIELDS = frozenset(
    {
        "mode",
        "declaration",
        "target_species",
        "motion_donor_tag",
        "motion_donor",
        "motion_donor_contract",
        "derived_motion_basis",
        "target_asset_authority",
    }
)
FORWARD_CONTRACT_FIELDS = LEGACY_FORWARD_CONTRACT_FIELDS | frozenset(
    {"target_rig_lineage"}
)
REGISTRY_FIELDS = frozenset(
    {
        "schema",
        "state_classification",
        "formal_dataset_registration_authorized",
        "preflight",
        "pixal_batch",
        "static_decision_batch",
        "source_asset_count",
        "source_assets",
        "automatic_checks",
        "registry_sha256",
    }
)
DERIVED_REGISTRY_FIELDS = REGISTRY_FIELDS | frozenset(
    {"derived_static_decisions"}
)
REGISTRY_SOURCE_INDEX_FIELDS = frozenset(
    {
        "asset_id",
        "profile_schema_id",
        "request_sha256",
        "sampled_attributes",
        "attribute_evidence",
        "source_asset",
        "state_classification",
        "next_gate",
    }
)
REGISTRY_AUTOMATIC_CHECKS = {
    "all_requests_reauthenticated": True,
    "all_pixal_input_attempt_request_identities_reauthenticated": True,
    "all_pixal_outputs_reauthenticated": True,
    "all_static_decisions_reauthenticated": True,
    "all_source_asset_v2_validated_against_request_and_profile": True,
    "all_physical_measurements_pending": True,
    "all_animation_ue_audio_qa_pending": True,
    "all_rights_blockers_preserved": True,
    "overall": "passed",
}
DERIVED_REGISTRY_AUTOMATIC_CHECKS = (
    source_registry.DERIVED_REGISTRY_AUTOMATIC_CHECKS
)
PIXAL_BATCH_AUTOMATIC_CHECKS = {
    "all_inputs_reauthenticated": True,
    "all_model_revisions_pinned": True,
    "all_jobs_have_unique_request": True,
    "all_jobs_claimed_once_by_dynamic_queue": True,
    "one_pixal_invocation_per_frozen_request": True,
    "seed_retry_forbidden": True,
    "candidate_ranking_or_best_of_n_forbidden": True,
    "all_outputs_glb2_readable": True,
    "all_outputs_have_pbr_material_and_texture": True,
    "no_generated_asset_has_been_registered": True,
    "overall": "passed",
}
STATIC_DECISION_BATCH_AUTOMATIC_CHECKS = {
    "all_review_hashes_reauthenticated": True,
    "all_multiview_artifacts_reauthenticated": True,
    "all_instances_have_one_decision": True,
    "all_metric_size_evidence_deferred": True,
    "no_formal_registration_authorized": True,
    "overall": "passed",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_DEFORMATION_SAMPLES = 24
REQUIRED_REVIEW_FRAMES = 8
SPEAR_ROOT = Path(__file__).resolve().parents[1]
SPEAR_TMP_BRIDGE = (SPEAR_ROOT / "tmp").absolute()
DEFAULT_ARTIFACT_ROOTS = {
    "spear_repo": SPEAR_ROOT,
    "models_root": Path("/data/models"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise contracts.ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_automatic_checks(
    value: Any, expected: Mapping[str, Any], label: str
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(expected)
        or any(
            value.get(name) != expected_value
            for name, expected_value in expected.items()
        )
    ):
        raise contracts.ContractError(f"{label} automatic checks are invalid")


def _strict_json_loads(value: str, label: str) -> Any:
    try:
        return contracts.strict_json_loads(value)
    except contracts.StrictJSONError as error:
        raise contracts.ContractError(f"{label} is not strict JSON: {error}") from error


def _load_strict_json(path: Path, label: str) -> Any:
    path = _direct_file(path, label)
    try:
        return contracts.load_json(path)
    except contracts.ContractError as error:
        raise contracts.ContractError(f"{label} is not strict JSON: {error}") from error
    except OSError as error:
        raise contracts.ContractError(f"{label} is not readable: {path}") from error


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _symlink_components(path: Path) -> list[Path]:
    absolute = _lexical_absolute(path)
    return [
        component
        for component in reversed((absolute, *absolute.parents))
        if component.is_symlink()
    ]


def _uses_only_exact_tmp_bridge(path: Path, label: str) -> tuple[Path, bool]:
    literal = _lexical_absolute(path)
    symlinks = _symlink_components(literal)
    unexpected = [component for component in symlinks if component != SPEAR_TMP_BRIDGE]
    if unexpected:
        raise contracts.ContractError(
            f"{label} cannot use symlink path component: {unexpected[0]}"
        )
    bridge_used = SPEAR_TMP_BRIDGE in symlinks
    if bridge_used:
        try:
            literal.resolve().relative_to(SPEAR_TMP_BRIDGE.resolve())
        except ValueError as error:
            raise contracts.ContractError(
                f"{label} escaped the exact SPEAR/tmp bridge"
            ) from error
    return literal, bridge_used


def _direct_file(path: Path, label: str) -> Path:
    literal, bridge_used = _uses_only_exact_tmp_bridge(path, label)
    resolved = literal.resolve()
    if bridge_used:
        try:
            resolved.relative_to(SPEAR_TMP_BRIDGE.resolve())
        except ValueError as error:
            raise contracts.ContractError(
                f"{label} escaped the exact SPEAR/tmp bridge"
            ) from error
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise contracts.ContractError(f"{label} is missing or unsafe: {resolved}")
    return resolved


def _new_output_path(path: Path, label: str) -> Path:
    literal, _bridge_used = _uses_only_exact_tmp_bridge(path, label)
    if literal.exists() or literal.is_symlink():
        raise contracts.ContractError(f"refusing to replace {label}: {literal}")
    return literal


def _contained_file(path: Path, root: Path, label: str) -> Path:
    path = _direct_file(path, label)
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise contracts.ContractError(f"{label} escaped its artifact root") from error
    return path


def _absolute_record(path: Path) -> dict[str, Any]:
    path = _direct_file(path, "artifact")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    path = _direct_file(path, "published artifact")
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_descriptor(
    value: Any,
    label: str,
    *,
    expected_path: Path | None = None,
    allow_metadata: bool = False,
) -> Path:
    if not isinstance(value, Mapping):
        raise contracts.ContractError(f"{label} descriptor is missing")
    required = {"path", "sha256", "size_bytes"}
    if not required.issubset(value) or (not allow_metadata and set(value) != required):
        raise contracts.ContractError(f"{label} descriptor fields are invalid")
    raw_path = value.get("path")
    sha256 = value.get("sha256")
    size_bytes = value.get("size_bytes")
    if (
        not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise contracts.ContractError(f"{label} descriptor values are invalid")
    literal = Path(raw_path)
    path = _direct_file(literal, label)
    if expected_path is not None and path != expected_path.resolve():
        raise contracts.ContractError(f"{label} path identity changed")
    if path.stat().st_size != size_bytes or _sha256_file(path) != sha256:
        raise contracts.ContractError(f"{label} changed")
    if path.suffix.lower() == ".json":
        _load_strict_json(path, label)
    return path


def _verify_review_descriptor(
    value: Any,
    label: str,
    *,
    review_root: Path,
    allow_metadata: bool = False,
) -> Path:
    path = _verify_descriptor(value, label, allow_metadata=allow_metadata)
    return _contained_file(path, review_root, label)


def _verify_path_hash_descriptor(
    value: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256"}
        or not isinstance(value.get("path"), str)
        or not Path(value["path"]).is_absolute()
    ):
        raise contracts.ContractError(f"{label} descriptor is invalid")
    _require_sha256(value.get("sha256"), f"{label} sha256")
    path = _direct_file(Path(value["path"]), label)
    if expected_path is not None and path != expected_path.resolve():
        raise contracts.ContractError(f"{label} path identity changed")
    if _sha256_file(path) != value["sha256"]:
        raise contracts.ContractError(f"{label} changed")
    if path.suffix.lower() == ".json":
        _load_strict_json(path, label)
    return path


def _verify_relative_descriptor(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise contracts.ContractError(f"{label} descriptor fields are invalid")
    relative_value = value.get("path")
    if not isinstance(relative_value, str):
        raise contracts.ContractError(f"{label} descriptor path is invalid")
    relative = Path(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise contracts.ContractError(f"{label} descriptor path is invalid")
    literal = root.resolve() / relative
    path = _contained_file(literal, root, label)
    _require_sha256(value.get("sha256"), f"{label} sha256")
    size = value.get("size_bytes")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
        or _sha256_file(path) != value["sha256"]
    ):
        raise contracts.ContractError(f"{label} changed")
    if path.suffix.lower() == ".json":
        _load_strict_json(path, label)
    return path


def parse_artifact_roots(values: Sequence[str]) -> dict[str, Path]:
    roots = {
        name: _lexical_absolute(path) for name, path in DEFAULT_ARTIFACT_ROOTS.items()
    }
    for value in values:
        if "=" not in value:
            raise contracts.ContractError("artifact roots must use ROOT_ID=PATH")
        root_id, raw_path = value.split("=", 1)
        if (
            not root_id
            or not raw_path
            or "/" in root_id
            or "\\" in root_id
            or not Path(raw_path).is_absolute()
        ):
            raise contracts.ContractError("artifact root is invalid")
        root, _bridge_used = _uses_only_exact_tmp_bridge(
            Path(raw_path),
            f"artifact root {root_id}",
        )
        roots[root_id] = root
    return roots


def _resolve_root_artifact(value: Any, roots: Mapping[str, Path], label: str) -> Path:
    if not isinstance(value, Mapping):
        raise contracts.ContractError(f"{label} is not an artifact descriptor")
    required = {"root_id", "path", "sha256", "size_bytes"}
    if set(value) != required:
        raise contracts.ContractError(f"{label} artifact fields are invalid")
    root_id = value.get("root_id")
    relative = Path(str(value.get("path", "")))
    if (
        root_id not in roots
        or relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
    ):
        raise contracts.ContractError(f"{label} artifact root/path is invalid")
    root, _bridge_used = _uses_only_exact_tmp_bridge(
        Path(roots[root_id]),
        f"{label} artifact root",
    )
    literal = root / relative
    resolved = _direct_file(literal, label)
    allowed = False
    try:
        resolved.relative_to(root.resolve())
        allowed = True
    except ValueError:
        if root_id == "spear_repo" and relative.parts[0] == "tmp":
            try:
                resolved.relative_to((root / "tmp").resolve())
                allowed = True
            except ValueError:
                pass
    if not allowed:
        raise contracts.ContractError(f"{label} escaped its artifact root")
    if resolved.stat().st_size != value.get("size_bytes") or _sha256_file(
        resolved
    ) != value.get("sha256"):
        raise contracts.ContractError(f"{label} artifact changed")
    if resolved.suffix.lower() == ".json":
        _load_strict_json(resolved, label)
    return resolved


def load_source_registry_anchor(
    registry_path: Path,
    source_asset_path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    registry_path = _direct_file(registry_path, "source asset registry")
    _require_sha256(expected_file_sha256, "expected source asset registry file sha256")
    if _sha256_file(registry_path) != expected_file_sha256:
        raise contracts.ContractError(
            "source asset registry does not match externally expected SHA-256"
        )
    registry = _load_strict_json(registry_path, "source asset registry")
    derived_registry = (
        isinstance(registry, dict)
        and registry.get("schema") == source_registry.DERIVED_REGISTRY_SCHEMA
    )
    expected_registry_fields = (
        DERIVED_REGISTRY_FIELDS if derived_registry else REGISTRY_FIELDS
    )
    expected_registry_checks = (
        DERIVED_REGISTRY_AUTOMATIC_CHECKS
        if derived_registry
        else REGISTRY_AUTOMATIC_CHECKS
    )
    if (
        not isinstance(registry, dict)
        or set(registry) != expected_registry_fields
        or registry.get("schema")
        not in {
            source_registry.REGISTRY_SCHEMA,
            source_registry.DERIVED_REGISTRY_SCHEMA,
        }
        or registry.get("state_classification") != "research_candidate"
        or registry.get("formal_dataset_registration_authorized") is not False
        or registry.get("registry_sha256")
        != source_registry._hash_without(registry, "registry_sha256")
    ):
        raise contracts.ContractError("source asset registry contract/hash is invalid")
    _require_exact_automatic_checks(
        registry.get("automatic_checks"),
        expected_registry_checks,
        "source asset registry",
    )

    preflight_descriptor = registry.get("preflight")
    expected_preflight_fields = {
        "path",
        "sha256",
        "preflight_sha256",
        "validation_mode",
    }
    if (
        not isinstance(preflight_descriptor, Mapping)
        or set(preflight_descriptor) != expected_preflight_fields
        or not isinstance(preflight_descriptor.get("path"), str)
        or not Path(preflight_descriptor["path"]).is_absolute()
        or preflight_descriptor.get("validation_mode")
        not in {"frozen_historical_preflight_v1", "current_exact_rebuild"}
    ):
        raise contracts.ContractError(
            "source asset registry preflight descriptor is invalid"
        )
    _require_sha256(
        preflight_descriptor.get("sha256"), "source registry preflight sha256"
    )
    _require_sha256(
        preflight_descriptor.get("preflight_sha256"),
        "source registry internal preflight sha256",
    )
    preflight_path = _direct_file(
        Path(preflight_descriptor["path"]), "source registry preflight"
    )
    if _sha256_file(preflight_path) != preflight_descriptor["sha256"]:
        raise contracts.ContractError("source registry preflight changed")
    _load_strict_json(preflight_path, "source registry preflight")
    frozen = preflight_descriptor["validation_mode"] == "frozen_historical_preflight_v1"
    preflight, requests, profiles = source_registry.load_source_contract(
        preflight_path, frozen_historical_preflight=frozen
    )
    if preflight.get("preflight_sha256") != preflight_descriptor["preflight_sha256"]:
        raise contracts.ContractError("source registry preflight identity changed")

    pixal_descriptor = registry.get("pixal_batch")
    expected_pixal_fields = {"path", "sha256", "batch_sha256"}
    if (
        not isinstance(pixal_descriptor, Mapping)
        or set(pixal_descriptor) != expected_pixal_fields
        or not isinstance(pixal_descriptor.get("path"), str)
        or not Path(pixal_descriptor["path"]).is_absolute()
    ):
        raise contracts.ContractError(
            "source asset registry Pixal batch descriptor is invalid"
        )
    _require_sha256(
        pixal_descriptor.get("sha256"), "source registry Pixal batch sha256"
    )
    _require_sha256(
        pixal_descriptor.get("batch_sha256"),
        "source registry internal Pixal batch sha256",
    )
    pixal_path = _direct_file(
        Path(pixal_descriptor["path"]), "source registry Pixal batch"
    )
    if _sha256_file(pixal_path) != pixal_descriptor["sha256"]:
        raise contracts.ContractError("source registry Pixal batch changed")
    pixal_payload = _load_strict_json(pixal_path, "source registry Pixal batch")
    if (
        not isinstance(pixal_payload, dict)
        or pixal_payload.get("schema") != source_registry.pixal_runner.BATCH_SCHEMA
        or pixal_payload.get("status") != "passed_generation_and_glb_readback"
        or pixal_payload.get("state_classification") != "research_candidate"
        or pixal_payload.get("formal_dataset_registration_authorized") is not False
        or pixal_payload.get("batch_sha256")
        != source_registry._hash_without(pixal_payload, "batch_sha256")
        or pixal_payload.get("batch_sha256") != pixal_descriptor["batch_sha256"]
    ):
        raise contracts.ContractError(
            "source registry Pixal batch contract/hash is invalid"
        )
    _require_exact_automatic_checks(
        pixal_payload.get("automatic_checks"),
        PIXAL_BATCH_AUTOMATIC_CHECKS,
        "source registry Pixal batch",
    )
    pixal_inputs_descriptor = pixal_payload.get("pixal_inputs")
    if (
        not isinstance(pixal_inputs_descriptor, Mapping)
        or set(pixal_inputs_descriptor) != {"path", "sha256", "manifest_sha256"}
        or not isinstance(pixal_inputs_descriptor.get("path"), str)
        or not Path(pixal_inputs_descriptor["path"]).is_absolute()
    ):
        raise contracts.ContractError(
            "source registry Pixal inputs descriptor is invalid"
        )
    _require_sha256(
        pixal_inputs_descriptor.get("sha256"),
        "source registry Pixal inputs file sha256",
    )
    _require_sha256(
        pixal_inputs_descriptor.get("manifest_sha256"),
        "source registry internal Pixal inputs sha256",
    )
    pixal_inputs_path = _direct_file(
        Path(pixal_inputs_descriptor["path"]),
        "source registry Pixal inputs manifest",
    )
    if _sha256_file(pixal_inputs_path) != pixal_inputs_descriptor["sha256"]:
        raise contracts.ContractError("source registry Pixal inputs manifest changed")
    _load_strict_json(
        pixal_inputs_path,
        "source registry Pixal inputs manifest",
    )
    (
        authenticated_pixal_inputs_path,
        pixal_inputs_manifest,
    ) = source_registry.pixal_runner.load_pixal_inputs(pixal_inputs_path)
    if (
        authenticated_pixal_inputs_path != pixal_inputs_path
        or pixal_inputs_manifest.get("manifest_sha256")
        != pixal_inputs_descriptor["manifest_sha256"]
    ):
        raise contracts.ContractError("source registry Pixal inputs identity changed")
    input_jobs, attempts = source_registry.validate_pixal_request_identity(
        pixal_payload, pixal_inputs_manifest, requests
    )

    decision_descriptor = registry.get("static_decision_batch")
    expected_decision_fields = {"path", "sha256", "decision_batch_sha256"}
    if (
        not isinstance(decision_descriptor, Mapping)
        or set(decision_descriptor) != expected_decision_fields
        or not isinstance(decision_descriptor.get("path"), str)
        or not Path(decision_descriptor["path"]).is_absolute()
    ):
        raise contracts.ContractError(
            "source asset registry static decision batch descriptor is invalid"
        )
    _require_sha256(
        decision_descriptor.get("sha256"),
        "source registry static decision batch sha256",
    )
    _require_sha256(
        decision_descriptor.get("decision_batch_sha256"),
        "source registry internal static decision batch sha256",
    )
    decision_batch_path = _direct_file(
        Path(decision_descriptor["path"]),
        "source registry static decision batch",
    )
    if _sha256_file(decision_batch_path) != decision_descriptor["sha256"]:
        raise contracts.ContractError("source registry static decision batch changed")
    decision_batch_payload = _load_strict_json(
        decision_batch_path,
        "source registry static decision batch",
    )
    if (
        not isinstance(decision_batch_payload, dict)
        or decision_batch_payload.get("schema")
        != source_registry.static_decisions.DECISION_BATCH_SCHEMA
        or decision_batch_payload.get("status") != "completed"
        or decision_batch_payload.get("decision_batch_sha256")
        != source_registry._hash_without(
            decision_batch_payload, "decision_batch_sha256"
        )
        or decision_batch_payload.get("decision_batch_sha256")
        != decision_descriptor["decision_batch_sha256"]
    ):
        raise contracts.ContractError(
            "source registry static decision batch contract/hash is invalid"
        )
    _require_exact_automatic_checks(
        decision_batch_payload.get("automatic_checks"),
        STATIC_DECISION_BATCH_AUTOMATIC_CHECKS,
        "source registry static decision batch",
    )
    (
        authenticated_decision_batch_path,
        authenticated_decision_batch,
        decisions,
    ) = source_registry.load_decision_batch(decision_batch_path)
    if (
        authenticated_decision_batch_path != decision_batch_path
        or authenticated_decision_batch.get("decision_batch_sha256")
        != decision_descriptor["decision_batch_sha256"]
    ):
        raise contracts.ContractError(
            "source registry static decision batch identity changed"
        )
    authority_decisions = decisions
    if derived_registry:
        if set(decisions) != set(attempts) or any(
            decision.get("payload", {}).get("decision") != "rejected"
            for decision in decisions.values()
        ):
            raise contracts.ContractError(
                "derived source registry must preserve complete raw rejections"
            )
        derived_indexes = registry.get("derived_static_decisions")
        if not isinstance(derived_indexes, list) or not derived_indexes:
            raise contracts.ContractError(
                "derived source registry decisions are missing"
            )
        authority_decisions = {}
        for index in derived_indexes:
            if not isinstance(index, Mapping) or set(index) != {
                "path",
                "sha256",
                "decision_sha256",
            }:
                raise contracts.ContractError(
                    "derived source registry decision index is invalid"
                )
            path_value = index.get("path")
            if (
                not isinstance(path_value, str)
                or not Path(path_value).is_absolute()
            ):
                raise contracts.ContractError(
                    "derived source registry decision path is invalid"
                )
            _require_sha256(
                index.get("sha256"),
                "derived static decision file sha256",
            )
            _require_sha256(
                index.get("decision_sha256"),
                "derived static decision internal sha256",
            )
            decision_path = _direct_file(
                Path(path_value),
                "derived static decision",
            )
            if _sha256_file(decision_path) != index["sha256"]:
                raise contracts.ContractError("derived static decision changed")
            payload = source_registry.derived_static_decisions.validate_decision(
                _load_strict_json(decision_path, "derived static decision")
            )
            instance_id = payload.get("instance_id")
            if (
                payload.get("decision")
                != source_registry.derived_static_decisions.APPROVED
                or payload.get("decision_sha256") != index["decision_sha256"]
                or instance_id in authority_decisions
                or instance_id not in attempts
            ):
                raise contracts.ContractError(
                    "derived static decision authority is invalid"
                )
            authority_decisions[instance_id] = {
                "path": decision_path,
                "payload": payload,
            }
        approved_ids = set(authority_decisions)
    else:
        approved_ids = source_registry.approved_attempt_ids(decisions, attempts)

    entries = registry.get("source_assets")
    if (
        not isinstance(entries, list)
        or not entries
        or registry.get("source_asset_count") != len(entries)
    ):
        raise contracts.ContractError("source asset registry index coverage is invalid")
    source_asset_path = _direct_file(source_asset_path, "source_asset_v2")
    selected: Mapping[str, Any] | None = None
    selected_request: Mapping[str, Any] | None = None
    selected_profile: Mapping[str, Any] | None = None
    seen_asset_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != REGISTRY_SOURCE_INDEX_FIELDS:
            raise contracts.ContractError(
                "source asset registry index entry is invalid"
            )
        asset_id = entry.get("asset_id")
        if not isinstance(asset_id, str) or asset_id in seen_asset_ids:
            raise contracts.ContractError(
                "source asset registry contains a duplicate identity"
            )
        seen_asset_ids.add(asset_id)
        indexed_path = _verify_relative_descriptor(
            entry["source_asset"],
            registry_path.parent,
            f"source asset registry entry {asset_id}",
        )
        request = requests.get(asset_id)
        if not isinstance(request, Mapping):
            raise contracts.ContractError(
                "source asset registry request identity is missing"
            )
        profile = profiles.get(request.get("profile_schema_id"))
        if not isinstance(profile, Mapping):
            raise contracts.ContractError(
                "source asset registry profile identity is missing"
            )
        payload = contracts.validate_source_asset_v2(
            _load_strict_json(
                indexed_path,
                f"source asset registry entry {asset_id}",
            ),
            request=request,
            profile=profile,
        )
        indexed_identity = {
            "asset_id": payload["asset_id"],
            "profile_schema_id": payload["profile_schema_id"],
            "request_sha256": payload["request_sha256"],
            "sampled_attributes": payload["sampled_attributes"],
            "state_classification": payload["state_classification"],
        }
        decision = authority_decisions.get(asset_id)
        if (
            any(
                contracts.canonical_json(entry.get(name))
                != contracts.canonical_json(value)
                for name, value in indexed_identity.items()
            )
            or entry.get("next_gate") != "lod_then_species_rig_binding"
            or not isinstance(decision, Mapping)
            or contracts.canonical_json(entry.get("attribute_evidence"))
            != contracts.canonical_json(
                decision.get("payload", {}).get("attribute_evidence")
            )
        ):
            raise contracts.ContractError(
                "source asset registry index identity/evidence changed"
            )
        if indexed_path == source_asset_path:
            if selected is not None:
                raise contracts.ContractError(
                    "source asset registry selected the source more than once"
                )
            selected = entry
            selected_request = request
            selected_profile = profile
    if seen_asset_ids != approved_ids or set(input_jobs) != set(attempts):
        raise contracts.ContractError(
            "source asset registry authority coverage is invalid"
        )
    if selected is None:
        raise contracts.ContractError(
            "source_asset_v2 is not anchored by the selected registry"
        )

    if selected_request is None or selected_profile is None:
        raise contracts.ContractError(
            "source asset registry selected identity is incomplete"
        )
    return (
        registry_path,
        registry,
        dict(selected_request),
        dict(selected_profile),
        preflight_descriptor["validation_mode"],
    )


def load_source_asset(
    path: Path,
    artifact_roots: Mapping[str, Path],
    *,
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    require_derived_authority: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    path = _direct_file(path, "source_asset_v2")
    payload = contracts.validate_source_asset_v2(
        _load_strict_json(path, "source_asset_v2"),
        request=request,
        profile=profile,
    )
    if (
        payload["asset_class"] != "animal"
        or payload["state_classification"] != "research_candidate"
        or not payload["asset_id"].endswith(payload["request_sha256"][:12])
        or set(payload["rig"]["actions"]) != {"Idle", "Walking"}
    ):
        raise contracts.ContractError(
            "source asset is not a canonical research animal identity"
        )
    authenticated: dict[str, Path] = {}
    for role, artifact in sorted(payload["artifacts"].items()):
        authenticated[f"artifact:{role}"] = _resolve_root_artifact(
            artifact, artifact_roots, f"source asset {role}"
        )
    for index, license_record in enumerate(payload["rights"]["licenses"]):
        authenticated[f"license:{index}"] = _resolve_root_artifact(
            license_record, artifact_roots, f"source asset license {index}"
        )
    derived_required = {
        "artifact:pixal_raw_glb",
        "artifact:raw_static_decision",
        "artifact:derived_repaired_glb",
        "artifact:derived_geometry_closure",
        "artifact:derived_repair_manifest",
        "artifact:derived_geometry_audit",
        "artifact:derived_static_review_manifest",
        "artifact:derived_static_decision",
    }
    present_derived_roles = derived_required.intersection(authenticated)
    derived_specific_roles = {
        role for role in derived_required if role.startswith("artifact:derived_")
    }
    if require_derived_authority and present_derived_roles != derived_required:
        raise contracts.ContractError(
            "derived source registry selected an asset without complete repair authority"
        )
    if derived_specific_roles.intersection(authenticated):
        if not derived_required.issubset(authenticated):
            raise contracts.ContractError(
                "derived source asset authority artifacts are incomplete"
            )
        decision = (
            source_registry.derived_static_decisions.validate_decision(
                _load_strict_json(
                    authenticated["artifact:derived_static_decision"],
                    "derived static decision",
                )
            )
        )
        review_path = authenticated["artifact:derived_static_review_manifest"]
        review = source_registry.derived_review_contract.validate_review(
            _load_strict_json(review_path, "derived static review")
        )
        bindings = {
            "pixal_raw_glb": (
                review["source_authorities"]["raw_pixal_glb"],
                authenticated["artifact:pixal_raw_glb"],
            ),
            "raw_static_decision": (
                review["source_authorities"]["raw_static_decision"]["file"],
                authenticated["artifact:raw_static_decision"],
            ),
            "derived_repaired_glb": (
                review["derived_geometry"]["repaired_glb"],
                authenticated["artifact:derived_repaired_glb"],
            ),
            "derived_geometry_closure": (
                review["derived_geometry"]["geometry_closure"],
                authenticated["artifact:derived_geometry_closure"],
            ),
            "derived_repair_manifest": (
                review["derived_geometry"]["repair_manifest"],
                authenticated["artifact:derived_repair_manifest"],
            ),
            "derived_geometry_audit": (
                review["derived_geometry"]["independent_geometry_audit"],
                authenticated["artifact:derived_geometry_audit"],
            ),
        }
        for label, (descriptor, expected_path) in bindings.items():
            observed_path = _direct_file(
                Path(descriptor["path"]),
                f"derived source {label}",
            )
            if (
                observed_path != expected_path
                or observed_path.stat().st_size != descriptor["size_bytes"]
                or _sha256_file(observed_path) != descriptor["sha256"]
            ):
                raise contracts.ContractError(
                    f"derived source {label} authority changed"
                )
        if (
            decision["decision"]
            != source_registry.derived_static_decisions.APPROVED
            or decision["instance_id"] != payload["asset_id"]
            or decision["review_binding"]["review_file"]["sha256"]
            != _sha256_file(review_path)
            or decision["review_binding"]["internal_review_sha256"]
            != review["review_sha256"]
            or review["instance_identity"]["instance_id"] != payload["asset_id"]
            or review["source_authorities"]["raw_static_decision"]["decision"]
            != "rejected"
        ):
            raise contracts.ContractError(
                "derived source decision/review identity changed"
            )
    return path, payload, authenticated


def _read_glb_document(
    path: Path,
    *,
    include_binary: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise contracts.ContractError("reviewed GLB header is truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise contracts.ContractError("reviewed runtime is not a complete GLB2")
    offset = 12
    raw_document: bytes | None = None
    binary_chunks: list[bytes] = []
    chunk_types: list[int] = []
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise contracts.ContractError("reviewed GLB chunk header is truncated")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_length <= 0 or chunk_end > len(payload):
            raise contracts.ContractError("reviewed GLB chunk is truncated")
        chunk = payload[offset:chunk_end]
        chunk_types.append(chunk_type)
        if chunk_type == 0x4E4F534A:
            if raw_document is not None:
                raise contracts.ContractError("reviewed GLB has multiple JSON chunks")
            raw_document = chunk
        elif chunk_type == 0x004E4942:
            binary_chunks.append(chunk)
        offset = chunk_end
    if (
        offset != len(payload)
        or raw_document is None
        or chunk_types != [0x4E4F534A, 0x004E4942]
    ):
        raise contracts.ContractError(
            "reviewed GLB lacks a complete embedded skinned-mesh payload"
        )
    try:
        document = _strict_json_loads(
            raw_document.rstrip(b" \t\r\n\x00").decode("utf-8"),
            "reviewed GLB JSON",
        )
    except UnicodeDecodeError as error:
        raise contracts.ContractError("reviewed GLB JSON is invalid") from error
    if not isinstance(document, dict):
        raise contracts.ContractError("reviewed GLB JSON must be an object")
    buffers = document.get("buffers")
    buffer_views = document.get("bufferViews")
    accessors = document.get("accessors")
    nodes = document.get("nodes")
    meshes = document.get("meshes")
    skins = document.get("skins")
    animations = document.get("animations")
    if (
        not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], Mapping)
        or "uri" in buffers[0]
        or isinstance(buffers[0].get("byteLength"), bool)
        or not isinstance(buffers[0].get("byteLength"), int)
        or buffers[0]["byteLength"] <= 0
        or len(binary_chunks) != 1
        or buffers[0]["byteLength"] > len(binary_chunks[0])
        or len(binary_chunks[0]) - buffers[0]["byteLength"] > 3
        or any(binary_chunks[0][buffers[0]["byteLength"] :])
        or not isinstance(buffer_views, list)
        or not buffer_views
        or not isinstance(accessors, list)
        or not accessors
        or not isinstance(nodes, list)
        or not nodes
        or not isinstance(meshes, list)
        or not meshes
        or not isinstance(skins, list)
        or len(skins) != 1
        or not isinstance(animations, list)
    ):
        raise contracts.ContractError(
            "reviewed GLB lacks a complete embedded skinned-mesh payload"
        )

    def accessor(index: Any, label: str) -> Mapping[str, Any]:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(accessors)
            or not isinstance(accessors[index], Mapping)
        ):
            raise contracts.ContractError(f"reviewed GLB {label} accessor is invalid")
        value = accessors[index]
        count = value.get("count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(value.get("bufferView"), bool)
            or not isinstance(value.get("bufferView"), int)
            or not 0 <= value["bufferView"] < len(buffer_views)
        ):
            raise contracts.ContractError(
                f"reviewed GLB {label} accessor is incomplete"
            )
        return value

    skin = skins[0]
    joints = skin.get("joints") if isinstance(skin, Mapping) else None
    if (
        not isinstance(joints, list)
        or len(joints) < 5
        or len(joints) != len(set(joints))
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(nodes)
            for index in joints
        )
    ):
        raise contracts.ContractError("reviewed GLB skin joint binding is incomplete")
    inverse_bind = accessor(skin.get("inverseBindMatrices"), "inverse-bind matrix")
    if inverse_bind.get("type") != "MAT4" or inverse_bind.get("count") != len(joints):
        raise contracts.ContractError(
            "reviewed GLB inverse-bind matrices do not cover the skin"
        )

    skinned_nodes = [
        node
        for node in nodes
        if isinstance(node, Mapping)
        and isinstance(node.get("mesh"), int)
        and node.get("skin") == 0
    ]
    if not skinned_nodes:
        raise contracts.ContractError(
            "reviewed GLB has no node binding a mesh to its skin"
        )
    for node in skinned_nodes:
        mesh_index = node["mesh"]
        if (
            isinstance(mesh_index, bool)
            or not 0 <= mesh_index < len(meshes)
            or not isinstance(meshes[mesh_index], Mapping)
        ):
            raise contracts.ContractError("reviewed GLB skinned mesh index is invalid")
        primitives = meshes[mesh_index].get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise contracts.ContractError("reviewed GLB skinned mesh has no primitives")
        for primitive in primitives:
            attributes = (
                primitive.get("attributes") if isinstance(primitive, Mapping) else None
            )
            required_attributes = {
                "POSITION",
                "JOINTS_0",
                "WEIGHTS_0",
            }
            if not isinstance(attributes, Mapping) or not required_attributes.issubset(
                attributes
            ):
                raise contracts.ContractError(
                    "reviewed GLB primitive lacks position/joint/weight accessors"
                )
            position = accessor(attributes["POSITION"], "position")
            joints_accessor = accessor(attributes["JOINTS_0"], "joint")
            weights_accessor = accessor(attributes["WEIGHTS_0"], "weight")
            if (
                joints_accessor.get("type") != "VEC4"
                or weights_accessor.get("type") != "VEC4"
                or position.get("count") != joints_accessor.get("count")
                or position.get("count") != weights_accessor.get("count")
            ):
                raise contracts.ContractError(
                    "reviewed GLB skin accessors do not cover every vertex"
                )

    action_names: list[str] = []
    for animation in animations:
        if not isinstance(animation, Mapping):
            raise contracts.ContractError("reviewed GLB animation entry is invalid")
        name = animation.get("name")
        samplers = animation.get("samplers")
        channels = animation.get("channels")
        if (
            not isinstance(name, str)
            or not isinstance(samplers, list)
            or not samplers
            or not isinstance(channels, list)
            or not channels
        ):
            raise contracts.ContractError(
                "reviewed GLB animation channels are incomplete"
            )
        action_names.append(name)
        for sampler in samplers:
            if not isinstance(sampler, Mapping):
                raise contracts.ContractError(
                    "reviewed GLB animation sampler is invalid"
                )
            accessor(sampler.get("input"), "animation input")
            accessor(sampler.get("output"), "animation output")
        for channel in channels:
            target = channel.get("target") if isinstance(channel, Mapping) else None
            sampler_index = (
                channel.get("sampler") if isinstance(channel, Mapping) else None
            )
            if (
                isinstance(sampler_index, bool)
                or not isinstance(sampler_index, int)
                or not 0 <= sampler_index < len(samplers)
                or not isinstance(target, Mapping)
                or target.get("path") not in {"translation", "rotation", "scale"}
                or isinstance(target.get("node"), bool)
                or not isinstance(target.get("node"), int)
                or not 0 <= target["node"] < len(nodes)
            ):
                raise contracts.ContractError(
                    "reviewed GLB animation channel target is invalid"
                )
    if include_binary:
        return document, binary_chunks[0][: buffers[0]["byteLength"]]
    return document


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
        or not isinstance(buffer_views[view_index], Mapping)
    ):
        raise contracts.ContractError(f"{label} bufferView is invalid")
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
        raise contracts.ContractError(f"{label} bufferView range is invalid")
    return binary[byte_offset : byte_offset + byte_length]


def _decoded_rgba(
    payload: bytes,
    *,
    expected_format: str,
    label: str,
) -> tuple[list[int], bytes]:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.format != expected_format:
                raise contracts.ContractError(
                    f"{label} payload is not {expected_format}"
                )
            opened.load()
            rgba = opened.convert("RGBA")
            return list(rgba.size), rgba.tobytes()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise contracts.ContractError(f"{label} payload cannot be decoded") from error


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
        raise contracts.ContractError(f"reviewed GLB {key} is invalid")
    filtered = [value for value in values if value != "EXT_texture_webp"]
    return filtered or None


def _authenticate_texture_transcode_graph(
    *,
    reviewed_document: dict[str, Any],
    reviewed_binary: bytes,
    compatible_document: dict[str, Any],
    compatible_binary: bytes,
    image_evidence: Sequence[Mapping[str, Any]],
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
        or not all(isinstance(image, Mapping) for image in reviewed_images)
        or not isinstance(compatible_images, list)
        or len(compatible_images) != len(reviewed_images)
        or not all(isinstance(image, Mapping) for image in compatible_images)
        or not isinstance(reviewed_views, list)
        or not reviewed_views
        or not all(isinstance(view, Mapping) for view in reviewed_views)
        or not isinstance(compatible_views, list)
        or not all(isinstance(view, Mapping) for view in compatible_views)
        or not isinstance(reviewed_textures, list)
        or not all(isinstance(texture, Mapping) for texture in reviewed_textures)
        or not isinstance(compatible_textures, list)
        or not all(
            isinstance(texture, Mapping) for texture in compatible_textures
        )
    ):
        raise contracts.ContractError("texture transcode GLB graph is incomplete")

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
        raise contracts.ContractError(
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
            raise contracts.ContractError(
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
        source_size, source_rgba = _decoded_rgba(
            source_bytes,
            expected_format="WEBP",
            label=f"reviewed WebP image {image_index}",
        )
        output_size, output_rgba = _decoded_rgba(
            png_bytes,
            expected_format="PNG",
            label=f"compatible PNG image {image_index}",
        )
        rgba_sha256 = hashlib.sha256(source_rgba).hexdigest()
        if (
            len(source_bytes) != evidence["source_size_bytes"]
            or hashlib.sha256(source_bytes).hexdigest()
            != evidence["source_webp_sha256"]
            or len(png_bytes) != evidence["png_size_bytes"]
            or hashlib.sha256(png_bytes).hexdigest() != evidence["png_sha256"]
            or source_size != evidence["pixel_size"]
            or output_size != evidence["pixel_size"]
            or output_rgba != source_rgba
            or rgba_sha256 != evidence["rgba_sha256"]
        ):
            raise contracts.ContractError(
                "texture transcode image bytes/pixels do not match the manifest"
            )
        expected_image = copy.deepcopy(source_image)
        expected_image["bufferView"] = output_view_index
        expected_image["mimeType"] = "image/png"
        expected_images[image_index] = expected_image
        cursor = expected_offset + len(png_bytes)

    if cursor != len(compatible_binary):
        raise contracts.ContractError(
            "texture transcode output has unauthenticated appended bytes"
        )
    for index, image in enumerate(reviewed_images):
        if index not in webp_indices and compatible_images[index] != image:
            raise contracts.ContractError(
                "texture transcode changed a non-WebP image"
            )

    expected_textures = expected_document["textures"]
    for index, texture in enumerate(expected_textures):
        extensions = texture.get("extensions")
        webp = (
            extensions.get("EXT_texture_webp")
            if isinstance(extensions, Mapping)
            else None
        )
        if webp is None:
            continue
        if (
            not isinstance(webp, Mapping)
            or set(webp) != {"source"}
            or isinstance(webp.get("source"), bool)
            or webp.get("source") not in webp_indices
        ):
            raise contracts.ContractError(
                "reviewed GLB EXT_texture_webp routing is invalid"
            )
        texture["source"] = webp["source"]
        del extensions["EXT_texture_webp"]
        if not extensions:
            texture.pop("extensions", None)
        expected_textures[index] = texture

    for key in ("extensionsUsed", "extensionsRequired"):
        filtered = _without_webp_extension(reviewed_document, key)
        if filtered is None:
            expected_document.pop(key, None)
        else:
            expected_document[key] = filtered
    if compatible_document != expected_document:
        raise contracts.ContractError(
            "texture transcode changed GLB structure or PBR routing"
        )
    if (
        "EXT_texture_webp" in compatible_document.get("extensionsUsed", [])
        or "EXT_texture_webp"
        in compatible_document.get("extensionsRequired", [])
        or any(
            isinstance(texture, Mapping)
            and isinstance(texture.get("extensions"), Mapping)
            and "EXT_texture_webp" in texture["extensions"]
            for texture in compatible_textures
        )
    ):
        raise contracts.ContractError(
            "UE-compatible GLB retains EXT_texture_webp"
        )


def _load_finite_json(path: Path, label: str) -> dict[str, Any]:
    payload = _load_strict_json(path, label)
    if not isinstance(payload, dict):
        raise contracts.ContractError(f"{label} must be a JSON object")
    return payload


def _authenticate_texture_transcode(
    *,
    reviewed_glb: Path,
    ue_compatible_glb_path: Path | None,
    texture_transcode_manifest_path: Path | None,
) -> dict[str, Any] | None:
    if (ue_compatible_glb_path is None) != (
        texture_transcode_manifest_path is None
    ):
        raise contracts.ContractError(
            "UE-compatible GLB and texture transcode manifest must be supplied together"
        )
    if ue_compatible_glb_path is None:
        return None

    reviewed_glb = _direct_file(reviewed_glb, "owner-reviewed animated GLB")
    ue_compatible_glb = _direct_file(
        ue_compatible_glb_path,
        "UE-compatible animated GLB",
    )
    texture_transcode_manifest = _direct_file(
        texture_transcode_manifest_path,
        "texture transcode manifest",
    )
    payload = _load_finite_json(
        texture_transcode_manifest,
        "texture transcode manifest",
    )
    images = payload.get("images")
    if (
        set(payload) != TEXTURE_TRANSCODE_FIELDS
        or payload.get("schema") != TEXTURE_TRANSCODE_SCHEMA
        or payload.get("purpose") != TEXTURE_TRANSCODE_PURPOSE
        or payload.get("geometry_skin_animation_byte_graph_changed") is not False
        or not isinstance(images, list)
        or not images
    ):
        raise contracts.ContractError("texture transcode manifest contract is invalid")
    input_glb = _verify_descriptor(
        payload.get("input"),
        "texture transcode input",
        expected_path=reviewed_glb,
    )
    output_glb = _verify_descriptor(
        payload.get("output"),
        "texture transcode output",
        expected_path=ue_compatible_glb,
    )
    if (
        input_glb == output_glb
        or payload["input"]["sha256"] == payload["output"]["sha256"]
    ):
        raise contracts.ContractError(
            "texture transcode output must be a distinct immutable derivative"
        )

    seen_indices: set[int] = set()
    for image in images:
        if (
            not isinstance(image, Mapping)
            or set(image) != TEXTURE_TRANSCODE_IMAGE_FIELDS
        ):
            raise contracts.ContractError(
                "texture transcode image evidence fields are invalid"
            )
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
            raise contracts.ContractError(
                "texture transcode image evidence values are invalid"
            )
        for field in ("rgba_sha256", "source_webp_sha256", "png_sha256"):
            _require_sha256(image.get(field), f"texture transcode image {field}")
        seen_indices.add(index)

    reviewed_document, reviewed_binary = _read_glb_document(
        input_glb,
        include_binary=True,
    )
    compatible_document, compatible_binary = _read_glb_document(
        output_glb,
        include_binary=True,
    )
    _authenticate_texture_transcode_graph(
        reviewed_document=reviewed_document,
        reviewed_binary=reviewed_binary,
        compatible_document=compatible_document,
        compatible_binary=compatible_binary,
        image_evidence=images,
    )
    return {
        "ue_compatible_glb": ue_compatible_glb,
        "texture_transcode_manifest": texture_transcode_manifest,
        "manifest": payload,
    }


def _texture_transcode_job_fields(
    texture_transcode: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = _direct_file(
        Path(texture_transcode["texture_transcode_manifest"]),
        "texture transcode manifest",
    )
    return {
        "texture_transcode_manifest": str(manifest_path),
        "texture_transcode_manifest_sha256": _sha256_file(manifest_path),
        "texture_transcode_manifest_size_bytes": manifest_path.stat().st_size,
    }


def _runner_call(label: str, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (
        ForwardContractError,
        RuntimeError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        raise contracts.ContractError(f"{label} rejected: {error}") from error


def _review_pipeline_order(*, repair_branch: str) -> list[str]:
    order = [
        "heading",
        "rig_audit",
        "support_plane",
        "support_plane_readback",
        "retarget",
        "gait_direction",
        "deformation",
    ]
    repair_stages = generated_review.WEIGHT_REPAIR_BRANCH_PIPELINE_STAGES.get(
        repair_branch
    )
    if repair_stages is None:
        raise contracts.ContractError(
            f"generated animation repair branch is unsupported: {repair_branch!r}"
        )
    order.extend(repair_stages)
    for action in ("walking", "idle"):
        for view in ("side", "front", "rear"):
            order.extend(
                [
                    f"render_{action}_{view}",
                    f"encode_{action}_{view}",
                ]
            )
    order.append("media_readback")
    return order


def _preauthenticate_nested_json_descriptors(
    value: Any,
    *,
    root: Path,
    label: str,
    seen: set[Path] | None = None,
) -> None:
    """Strictly parse JSON evidence before a delegated lineage validator."""

    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        if isinstance(raw_path, str) and raw_path:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                if ".." in candidate.parts:
                    raise contracts.ContractError(
                        f"{label} nested JSON descriptor escaped its root"
                    )
                candidate = root / candidate
            suffix = candidate.suffix.lower()
            if suffix in {".json", ".jsonl"}:
                path = _direct_file(candidate, f"{label} nested JSON evidence")
                if path not in seen:
                    seen.add(path)
                    if suffix == ".json":
                        _load_strict_json(path, f"{label} nested JSON evidence")
                    else:
                        try:
                            lines = path.read_bytes().splitlines()
                        except OSError as error:
                            raise contracts.ContractError(
                                f"{label} nested JSONL evidence is unreadable"
                            ) from error
                        if not lines:
                            raise contracts.ContractError(
                                f"{label} nested JSONL evidence is empty"
                            )
                        for index, line in enumerate(lines, start=1):
                            try:
                                contracts.strict_json_loads(line)
                            except contracts.StrictJSONError as error:
                                raise contracts.ContractError(
                                    f"{label} nested JSONL line {index} "
                                    f"is not strict JSON: {error}"
                                ) from error
        for child in value.values():
            _preauthenticate_nested_json_descriptors(
                child,
                root=root,
                label=label,
                seen=seen,
            )
    elif isinstance(value, list):
        for child in value:
            _preauthenticate_nested_json_descriptors(
                child,
                root=root,
                label=label,
                seen=seen,
            )


def _source_workspace(
    source_asset: Mapping[str, Any],
    source_artifacts: Mapping[str, Path],
) -> tuple[str, Path]:
    pixal_path = source_artifacts.get("artifact:pixal_raw_glb")
    if pixal_path is None:
        raise contracts.ContractError(
            "source asset lacks its authenticated Pixal geometry authority"
        )
    workspaces_root = (SPEAR_ROOT / "tmp/new_animal_assets").resolve()
    try:
        relative = pixal_path.resolve().relative_to(workspaces_root)
    except ValueError as error:
        raise contracts.ContractError(
            "source Pixal geometry escaped the generated-animal workspace"
        ) from error
    if not relative.parts:
        raise contracts.ContractError(
            "source Pixal geometry has no generated-animal workspace"
        )
    workspace = relative.parts[0]
    if (
        not workspace
        or "/" in workspace
        or "\\" in workspace
        or source_asset.get("asset_class") != "animal"
    ):
        raise contracts.ContractError(
            "source generated-animal workspace identity is invalid"
        )
    return workspace, workspaces_root / workspace


def _validate_target_rig_lineage(
    value: Any,
    *,
    target_rig_glb: Path,
    source_asset: Mapping[str, Any],
    source_artifacts: Mapping[str, Path],
    authenticated: dict[str, Path],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise contracts.ContractError(
            "generated animation target-rig lineage descriptor is missing"
        )
    closure = value.get("closure_manifest")
    if (
        not isinstance(closure, Mapping)
        or set(closure) != {"path", "sha256", "size_bytes", "manifest_sha256"}
        or not isinstance(closure.get("path"), str)
        or not Path(closure["path"]).is_absolute()
    ):
        raise contracts.ContractError(
            "generated animation TokenRig closure descriptor is invalid"
        )
    expected_file_sha256 = _require_sha256(
        closure.get("sha256"), "TokenRig closure manifest file sha256"
    )
    _require_sha256(
        closure.get("manifest_sha256"),
        "TokenRig closure internal manifest sha256",
    )
    closure_path = _direct_file(Path(closure["path"]), "TokenRig closure manifest")
    closure_size = closure.get("size_bytes")
    if (
        isinstance(closure_size, bool)
        or not isinstance(closure_size, int)
        or closure_size <= 0
        or closure_path.stat().st_size != closure_size
        or _sha256_file(closure_path) != expected_file_sha256
    ):
        raise contracts.ContractError("TokenRig closure manifest changed")
    closure_payload = _load_strict_json(
        closure_path,
        "TokenRig closure manifest",
    )
    _preauthenticate_nested_json_descriptors(
        closure_payload,
        root=closure_path.parent,
        label="TokenRig closure",
    )
    observed = _runner_call(
        "TokenRig source closure",
        validate_tokenrig_closure_manifest,
        closure_path,
        expected_manifest_sha256=expected_file_sha256,
        expected_target_rig_glb=target_rig_glb,
    )
    if contracts.canonical_json(value) != contracts.canonical_json(observed):
        raise contracts.ContractError(
            "review target-rig lineage contradicts its authenticated closure"
        )
    raw_pixal = source_artifacts.get("artifact:pixal_raw_glb")
    if raw_pixal is None:
        raise contracts.ContractError(
            "source asset lacks its authenticated raw Pixal GLB"
        )
    _runner_call(
        "TokenRig closure raw Pixal source",
        generated_review.require_file_binding,
        observed["lineage"]["raw_pixal_glb"],
        raw_pixal,
        "TokenRig closure raw Pixal source",
    )
    tokenrig_input = source_artifacts.get(
        "artifact:derived_repaired_glb",
        raw_pixal,
    )
    _runner_call(
        "TokenRig closure target geometry",
        generated_review.require_file_binding,
        observed["lineage"]["tokenrig_input"],
        tokenrig_input,
        "TokenRig closure target geometry",
    )
    workspace, _workspace_root = _source_workspace(source_asset, source_artifacts)
    if observed.get("asset_id") != workspace:
        raise contracts.ContractError(
            "TokenRig closure and source registry workspace identities differ"
        )
    readback = observed.get("geometry_readback")
    if not isinstance(readback, Mapping) or set(readback) != {
        "path",
        "sha256",
        "size_bytes",
        "manifest_sha256",
    }:
        raise contracts.ContractError(
            "TokenRig closure geometry-readback descriptor is invalid"
        )
    readback_path = _verify_descriptor(
        {name: readback[name] for name in ("path", "sha256", "size_bytes")},
        "TokenRig closure geometry readback",
    )
    _require_sha256(
        readback.get("manifest_sha256"),
        "TokenRig closure geometry-readback internal sha256",
    )
    authenticated["target_rig_lineage:closure_manifest"] = closure_path
    authenticated["target_rig_lineage:geometry_readback"] = readback_path
    return dict(observed)


def _validate_forward_contract(
    value: Any,
    *,
    review_schema: str,
    target_rig_lineage: Any,
    source_asset: Mapping[str, Any],
    source_artifacts: Mapping[str, Path],
    inputs: Mapping[str, Path],
    authenticated: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_fields = (
        LEGACY_FORWARD_CONTRACT_FIELDS
        if review_schema == LEGACY_GENERATED_REVIEW_SCHEMA
        else FORWARD_CONTRACT_FIELDS
    )
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise contracts.ContractError(
            "generated animation forward contract fields are invalid"
        )
    if value.get("mode") != "forward_declaration_v1":
        raise contracts.ContractError(
            "generated animation review did not use a forward declaration"
        )
    declaration_path = _verify_descriptor(
        value.get("declaration"),
        "review forward declaration",
        expected_path=inputs["heading_review_evidence"],
    )
    authenticated["forward:declaration"] = declaration_path
    declaration = _runner_call(
        "forward declaration",
        load_forward_declaration,
        declaration_path,
    )
    _runner_call(
        "forward declaration target rig",
        generated_review.require_file_binding,
        declaration.get("input_glb"),
        inputs["target_rig_glb"],
        "forward declaration target rig",
    )
    species = source_asset.get("taxonomy", {}).get("species")
    donor_tag = value.get("motion_donor_tag")
    donor_contract = _runner_call(
        "motion donor contract",
        expected_motion_donor_contract,
        donor_tag,
    )
    donor_readback = _runner_call(
        "motion donor artifact",
        assert_declared_motion_donor_artifact,
        donor_tag,
        inputs["source_motion_glb"],
    )
    basis = value.get("derived_motion_basis")
    if (
        species != value.get("target_species")
        or donor_contract.get("target_species") != species
        or value.get("motion_donor_contract") != donor_contract
        or value.get("motion_donor") != donor_readback
        or declaration.get("motion_donor_tag") != donor_tag
        or declaration.get("expected_motion_basis") != basis
    ):
        raise contracts.ContractError(
            "forward declaration species/donor authority changed"
        )
    if not isinstance(basis, Mapping) or set(basis) != {
        "motion_basis_yaw_deg",
        "side_chain_mode",
    }:
        raise contracts.ContractError("forward motion basis is invalid")
    if review_schema == BRANCHED_GENERATED_REVIEW_SCHEMA:
        observed_target_lineage = _validate_target_rig_lineage(
            target_rig_lineage,
            target_rig_glb=inputs["target_rig_glb"],
            source_asset=source_asset,
            source_artifacts=source_artifacts,
            authenticated=authenticated,
        )
        if contracts.canonical_json(value.get("target_rig_lineage")) != (
            contracts.canonical_json(observed_target_lineage)
        ):
            raise contracts.ContractError(
                "forward contract and review input TokenRig lineages differ"
            )
    _runner_call(
        "forward motion basis",
        assert_declared_motion_basis,
        donor_tag,
        basis["motion_basis_yaw_deg"],
        basis["side_chain_mode"],
    )
    target_authority = value.get("target_asset_authority")
    if (
        not isinstance(target_authority, Mapping)
        or set(target_authority)
        != {
            "input_glb",
            "mesh_geometry_source",
            "motion_donor_geometry_used",
        }
        or target_authority.get("input_glb")
        != {
            "path": str(inputs["target_rig_glb"]),
            "sha256": _sha256_file(inputs["target_rig_glb"]),
            "size_bytes": inputs["target_rig_glb"].stat().st_size,
        }
        or target_authority.get("mesh_geometry_source") != "target_rig_glb"
        or target_authority.get("motion_donor_geometry_used") is not False
    ):
        raise contracts.ContractError("forward target mesh authority changed")
    workspace, workspace_root = _source_workspace(source_asset, source_artifacts)
    if declaration.get("asset_workspace") != workspace:
        raise contracts.ContractError(
            "forward declaration and source registry workspace differ"
        )
    _contained_file(
        inputs["target_rig_glb"],
        workspace_root,
        "review target rig",
    )
    return dict(declaration), dict(donor_contract)


def _repair_branch_contract(
    gates: Mapping[str, Any],
    *,
    review_schema: str,
) -> tuple[bool, str]:
    gate_fields = set(gates)
    if review_schema == LEGACY_GENERATED_REVIEW_SCHEMA:
        if gate_fields != LEGACY_REVIEW_GATE_FIELDS:
            raise contracts.ContractError(
                "legacy v3 animation review repair gate fields are invalid"
            )
        branch = "primary" if gates.get("weight_repair_triggered") else "not_needed"
        return True, branch
    if (
        review_schema != BRANCHED_GENERATED_REVIEW_SCHEMA
        or gate_fields != REVIEW_GATE_FIELDS
    ):
        raise contracts.ContractError(
            "branched v4 animation review repair gate fields are invalid"
        )

    branch = gates.get("weight_repair_branch")
    attempts = gates.get("weight_repair_attempts")
    if (
        branch not in generated_review.WEIGHT_REPAIR_BRANCH_STAGES
        or not isinstance(attempts, list)
        or any(
            not isinstance(attempt, Mapping)
            or set(attempt) != set(generated_review.WEIGHT_REPAIR_ATTEMPT_GATE_FIELDS)
            for attempt in attempts
        )
    ):
        raise contracts.ContractError(
            "generated animation weight-repair branch/attempts are invalid"
        )
    for attempt in attempts:
        extension = attempt.get("maximum_extension_ratio_of_rest_diagonal")
        remaining = attempt.get("remaining_seed_edge_count")
        if (
            isinstance(extension, bool)
            or not isinstance(extension, (int, float))
            or not math.isfinite(extension)
            or extension < 0.0
            or isinstance(remaining, bool)
            or not isinstance(remaining, int)
            or remaining < 0
        ):
            raise contracts.ContractError(
                "generated animation weight-repair attempt metrics are invalid"
            )
    _runner_call(
        "weight-repair branch consistency",
        generated_review.require_weight_repair_branch_consistency,
        branch,
        attempts,
    )
    expected_strategy = generated_review.WEIGHT_REPAIR_BRANCH_STRATEGY[branch]
    expected_final = generated_review.weight_repair_final_artifact(branch)
    expected_triggered = branch != "not_needed"
    expected_status = (
        generated_review.WEIGHT_REPAIR_READY_STATUS
        if expected_triggered
        else "not_needed"
    )
    if (
        gates.get("weight_repair_strategy") != expected_strategy
        or gates.get("weight_repair_final_artifact") != expected_final
        or gates.get("weight_repair_triggered") is not expected_triggered
        or gates.get("weight_repair") != expected_status
    ):
        raise contracts.ContractError(
            "generated animation weight-repair branch authority contradicts itself"
        )
    return False, branch


def _repair_output_names(*, legacy: bool, branch: str) -> set[str]:
    if branch == "not_needed":
        return set()
    if legacy:
        return {"weight_repair_manifest"}
    names = {"weight_repair_manifest"}
    for stage in generated_review.WEIGHT_REPAIR_BRANCH_STAGES[branch]:
        contract = generated_review.WEIGHT_REPAIR_STAGE_CONTRACTS[stage]
        names.add(contract["output_glb_output_descriptor"])
        names.add(contract["manifest_output_descriptor"])
    return names


def _weight_repair_parameters(stage: str) -> Mapping[str, Any]:
    parameters = {
        "primary": generated_review.WEIGHT_REPAIR_PARAMETERS,
        "fallback_a": generated_review.WEIGHT_REPAIR_FALLBACK_PARAMETERS,
        "fallback_b": generated_review.WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS,
    }.get(stage)
    if parameters is None:
        raise contracts.ContractError(
            f"generated animation weight-repair stage is unsupported: {stage!r}"
        )
    return parameters


def _validate_weight_repair_lineage(
    *,
    gates: Mapping[str, Any],
    outputs: Mapping[str, Any],
    authenticated: Mapping[str, Path],
    retargeted_glb: Path,
    final_glb: Path,
    legacy: bool,
    branch: str,
) -> None:
    if branch == "not_needed":
        if (
            final_glb != retargeted_glb
            or gates.get("weight_repair") != "not_needed"
            or outputs["gait_direction_audit"]
            != outputs["gait_direction_audit_initial"]
            or outputs["deformation_audit"] != outputs["deformation_audit_initial"]
        ):
            raise contracts.ContractError(
                "unrepaired review changed its final animation/audits"
            )
        return

    if legacy:
        repair = _runner_call(
            "legacy primary weight repair manifest",
            generated_review.require_weight_repair,
            authenticated["output:weight_repair_manifest"],
            retargeted_glb,
            final_glb,
            expected_parameters=generated_review.WEIGHT_REPAIR_PARAMETERS,
        )
        if (
            repair.get("status") != generated_review.WEIGHT_REPAIR_READY_STATUS
            or gates.get("weight_repair") != repair["status"]
        ):
            raise contracts.ContractError(
                "legacy primary weight-repair gate contradicts its manifest"
            )
        return

    attempts: list[dict[str, Any]] = []
    stages = generated_review.WEIGHT_REPAIR_BRANCH_STAGES[branch]
    final_stage = generated_review.WEIGHT_REPAIR_BRANCH_FINAL_STAGE[branch]
    for stage in stages:
        contract = generated_review.WEIGHT_REPAIR_STAGE_CONTRACTS[stage]
        manifest_name = contract["manifest_output_descriptor"]
        output_name = contract["output_glb_output_descriptor"]
        if stage == "fallback_b":
            input_glb = authenticated["output:weight_repair_fallback_a_glb"]
        else:
            input_glb = retargeted_glb
        manifest_path = authenticated[f"output:{manifest_name}"]
        output_glb = authenticated[f"output:{output_name}"]
        repair = _runner_call(
            f"{stage} weight repair manifest",
            generated_review.require_weight_repair,
            manifest_path,
            input_glb,
            output_glb,
            expected_parameters=_weight_repair_parameters(stage),
            allow_incomplete=stage != final_stage,
        )
        attempts.append(
            generated_review.weight_repair_attempt_record(
                stage,
                repair,
                manifest_path,
                output_glb,
            )
        )

    _runner_call(
        "authenticated weight-repair branch consistency",
        generated_review.require_weight_repair_branch_consistency,
        branch,
        attempts,
    )
    expected_gate_attempts = generated_review.weight_repair_gate_attempts(attempts)
    final_artifact = generated_review.weight_repair_final_artifact(branch)
    final_glb_name = final_artifact["glb_output_descriptor"]
    final_manifest_name = final_artifact["manifest_output_descriptor"]
    final_manifest = authenticated[f"output:{final_manifest_name}"]
    if (
        gates.get("weight_repair_attempts") != expected_gate_attempts
        or gates.get("weight_repair_final_artifact") != final_artifact
        or outputs["animated_glb"] != outputs[final_glb_name]
        or final_glb != authenticated[f"output:{final_glb_name}"]
        or outputs["weight_repair_manifest"] != outputs[final_manifest_name]
        or authenticated["output:weight_repair_manifest"] != final_manifest
        or gates.get("weight_repair") != attempts[-1]["status"]
    ):
        raise contracts.ContractError(
            "generated animation final weight-repair lineage is inconsistent"
        )


def _validate_review_lineage(
    *,
    review_path: Path,
    review: Mapping[str, Any],
    source_asset: Mapping[str, Any],
    source_artifacts: Mapping[str, Path],
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Any],
    authenticated: dict[str, Path],
) -> Path:
    review_root = review_path.parent
    forward, donor_contract = _validate_forward_contract(
        review.get("forward_contract"),
        review_schema=review["schema"],
        target_rig_lineage=review["inputs"].get("target_rig_lineage"),
        source_asset=source_asset,
        source_artifacts=source_artifacts,
        inputs=inputs,
        authenticated=authenticated,
    )
    basis = review["forward_contract"]["derived_motion_basis"]
    target_axis = forward["target_front_axis"]

    heading_path = authenticated["output:heading_manifest"]
    heading_payload = _load_finite_json(heading_path, "heading manifest")
    heading_glb = _verify_review_descriptor(
        heading_payload.get("output"),
        "heading output GLB",
        review_root=review_root,
    )
    authenticated["lineage:heading_glb"] = heading_glb
    _runner_call(
        "heading manifest",
        generated_review.require_heading_manifest,
        heading_path,
        inputs["target_rig_glb"],
        inputs["heading_review_evidence"],
        heading_glb,
        reviewed_source_front_yaw_deg=forward["reviewed_source_front_yaw_deg"],
        target_front_axis=target_axis,
    )

    rig_path = authenticated["output:rig_audit"]
    _runner_call(
        "rig audit",
        generated_review.require_rig_audit,
        rig_path,
        heading_glb,
        front_axis=target_axis,
    )

    support_path = authenticated["output:support_plane_manifest"]
    support_payload = _load_finite_json(support_path, "support-plane manifest")
    leveled_glb = _verify_review_descriptor(
        support_payload.get("output"),
        "support-plane output GLB",
        review_root=review_root,
    )
    authenticated["lineage:leveled_glb"] = leveled_glb
    _runner_call(
        "support-plane manifest/readback",
        generated_review.require_support_plane,
        support_path,
        heading_glb,
        leveled_glb,
        readback_path=authenticated["output:support_plane_output_readback"],
        plane_source="mesh-foot-bottoms",
        review_evidence=rig_path,
        front_axis=target_axis,
    )

    retargeted_glb = authenticated["output:retargeted_animated_glb"]
    retarget_path = authenticated["output:retarget_manifest"]
    retarget_payload = _load_finite_json(retarget_path, "retarget manifest")
    rotation_transfer = retarget_payload.get("rotation_transfer")
    motion_amplitude = (
        rotation_transfer.get("motion_amplitude")
        if isinstance(rotation_transfer, Mapping)
        else None
    )
    if (
        isinstance(motion_amplitude, bool)
        or not isinstance(motion_amplitude, (int, float))
        or not 0.0 < motion_amplitude <= 1.0
    ):
        raise contracts.ContractError("retarget motion amplitude is invalid")
    _runner_call(
        "retarget manifest",
        generated_review.require_retarget_manifest,
        retarget_path,
        leveled_glb,
        inputs["source_motion_glb"],
        retargeted_glb,
        target_front_axis=target_axis,
        motion_amplitude=motion_amplitude,
        motion_basis_yaw_deg=basis["motion_basis_yaw_deg"],
        side_chain_mode=basis["side_chain_mode"],
        motion_donor_contract=donor_contract,
    )

    gates = review["automatic_admission_gates"]
    initial_gait_path = authenticated["output:gait_direction_audit_initial"]
    initial_deformation_path = authenticated["output:deformation_audit_initial"]
    initial_gait = _runner_call(
        "initial gait audit",
        generated_review.require_gait_audit,
        initial_gait_path,
        retargeted_glb,
        "initial gait audit",
    )
    initial_deformation = _runner_call(
        "initial deformation audit",
        generated_review.require_deformation_audit,
        initial_deformation_path,
        retargeted_glb,
        "initial deformation audit",
        expected_samples=REQUIRED_DEFORMATION_SAMPLES,
        require_pass=False,
    )
    final_glb = authenticated["output:animated_glb"]
    legacy_repair_contract, repair_branch = _repair_branch_contract(
        gates,
        review_schema=review["schema"],
    )
    _validate_weight_repair_lineage(
        gates=gates,
        outputs=outputs,
        authenticated=authenticated,
        retargeted_glb=retargeted_glb,
        final_glb=final_glb,
        legacy=legacy_repair_contract,
        branch=repair_branch,
    )

    final_gait = _runner_call(
        "final gait audit",
        generated_review.require_gait_audit,
        authenticated["output:gait_direction_audit"],
        final_glb,
        "final gait audit",
    )
    final_deformation = _runner_call(
        "final deformation audit",
        generated_review.require_deformation_audit,
        authenticated["output:deformation_audit"],
        final_glb,
        "final deformation audit",
        expected_samples=REQUIRED_DEFORMATION_SAMPLES,
        require_pass=True,
    )
    if (
        gates["gait_initial"] != initial_gait["status"]
        or gates["deformation_initial"] != initial_deformation["overall"]
        or gates["gait_final"] != final_gait["status"]
        or gates["deformation_final"] != final_deformation["overall"]
        or outputs.get("gait_direction_status") != final_gait["status"]
        or outputs.get("deformation_overall") != final_deformation["overall"]
    ):
        raise contracts.ContractError(
            "review automatic gates contradict their authenticated audits"
        )
    return final_glb


def load_animation_review(
    path: Path,
    *,
    source_asset: Mapping[str, Any],
    source_artifacts: Mapping[str, Path],
) -> tuple[Path, dict[str, Any], Path, dict[str, Path]]:
    path = _direct_file(path, "generated animation review")
    payload = _load_finite_json(path, "generated animation review")
    gates = payload.get("automatic_admission_gates", {})
    if (
        set(payload) != REVIEW_TOP_LEVEL_FIELDS
        or payload.get("schema") not in GENERATED_REVIEW_SCHEMAS
        or payload.get("status") != "research_candidate_pending_human_review"
        or payload.get("formal_dataset_registration_authorized") is not False
        or not isinstance(gates, Mapping)
        or gates.get("all_automatic_gates_passed") is not True
        or gates.get("rig") != "passed"
        or gates.get("heading") != "positive-x"
        or gates.get("support_plane") != "mesh-foot-bottoms"
        or gates.get("retarget_export_front_axis") != "positive-x"
        or gates.get("gait_initial") != "pass"
        or gates.get("gait_final") != "pass"
        or gates.get("deformation_final") != "passed"
        or gates.get("weight_repair_policy") not in {"auto", "always", "never"}
        or not isinstance(gates.get("weight_repair_triggered"), bool)
    ):
        raise contracts.ContractError(
            "generated animation review automatic gates are incomplete"
        )
    legacy_repair_contract, repair_branch = _repair_branch_contract(
        gates,
        review_schema=payload["schema"],
    )
    try:
        expected_repair = generated_review.weight_repair_required(
            gates["weight_repair_policy"], gates.get("deformation_initial")
        )
    except (RuntimeError, ValueError) as error:
        raise contracts.ContractError(
            f"generated animation repair gate is invalid: {error}"
        ) from error
    if gates["weight_repair_triggered"] is not expected_repair:
        raise contracts.ContractError(
            "generated animation repair trigger contradicts its policy"
        )

    pipeline_order = payload.get("pipeline_order")
    expected_pipeline = _review_pipeline_order(repair_branch=repair_branch)
    timings = payload.get("timings_seconds")
    if (
        pipeline_order != expected_pipeline
        or not isinstance(timings, Mapping)
        or set(timings) != set(expected_pipeline)
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            for value in timings.values()
        )
        or not isinstance(payload.get("created_at"), str)
        or not payload["created_at"].strip()
    ):
        raise contracts.ContractError(
            "generated animation review pipeline/timings are invalid"
        )
    _runner_call(
        "weight-repair pipeline order",
        generated_review.require_weight_repair_pipeline_order,
        pipeline_order,
        repair_branch,
    )

    authenticated: dict[str, Path] = {}
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    expected_inputs = (
        REVIEW_FILE_INPUTS
        if payload["schema"] == LEGACY_GENERATED_REVIEW_SCHEMA
        else REVIEW_INPUTS
    )
    expected_outputs = (
        set(REQUIRED_REVIEW_OUTPUTS) | set(REVIEW_OUTPUT_SCALARS) | {"media"}
    )
    if payload["schema"] == BRANCHED_GENERATED_REVIEW_SCHEMA:
        expected_outputs.add("media_lineage")
    expected_outputs.update(
        _repair_output_names(
            legacy=legacy_repair_contract,
            branch=repair_branch,
        )
    )
    if (
        not isinstance(inputs, Mapping)
        or set(inputs) != expected_inputs
        or not isinstance(outputs, Mapping)
        or set(outputs) != expected_outputs
    ):
        raise contracts.ContractError("generated animation review I/O is invalid")
    for name, descriptor in inputs.items():
        if name == "target_rig_lineage":
            continue
        authenticated[f"input:{name}"] = _verify_descriptor(
            descriptor, f"review input {name}"
        )
    review_root = path.parent
    descriptor_names = set(REQUIRED_REVIEW_OUTPUTS) | _repair_output_names(
        legacy=legacy_repair_contract,
        branch=repair_branch,
    )
    for name in sorted(descriptor_names):
        authenticated[f"output:{name}"] = _verify_review_descriptor(
            outputs[name],
            f"review output {name}",
            review_root=review_root,
        )

    media = outputs.get("media")
    if not isinstance(media, Mapping) or set(media) != REQUIRED_MEDIA:
        raise contracts.ContractError(
            "generated animation review media coverage changed"
        )
    media_fields = {
        "path",
        "sha256",
        "size_bytes",
        "codec",
        "width",
        "height",
        "frame_count",
        "frame_rate",
        "duration_seconds",
    }
    for name, descriptor in sorted(media.items()):
        if not isinstance(descriptor, Mapping) or set(descriptor) != media_fields:
            raise contracts.ContractError(
                f"review media descriptor fields are invalid: {name}"
            )
        media_path = _verify_review_descriptor(
            descriptor,
            f"review media {name}",
            review_root=review_root,
            allow_metadata=True,
        )
        if (
            descriptor.get("codec") != "h264"
            or descriptor.get("frame_count") != REQUIRED_REVIEW_FRAMES
            or descriptor.get("width") != 512
            or descriptor.get("height") != 384
            or not isinstance(descriptor.get("frame_rate"), str)
            or not descriptor["frame_rate"]
            or isinstance(descriptor.get("duration_seconds"), bool)
            or not isinstance(descriptor.get("duration_seconds"), (int, float))
            or not math.isfinite(float(descriptor["duration_seconds"]))
            or descriptor["duration_seconds"] <= 0
        ):
            raise contracts.ContractError(f"review media metadata is invalid: {name}")
        probed = _runner_call(
            f"review media {name} ffprobe",
            generated_review.verify_video,
            media_path,
            REQUIRED_REVIEW_FRAMES,
        )
        if dict(descriptor) != probed:
            raise contracts.ContractError(
                f"review media metadata contradicts ffprobe: {name}"
            )
        authenticated[f"media:{name}"] = media_path

    media_lineage = outputs.get("media_lineage")
    if payload["schema"] == BRANCHED_GENERATED_REVIEW_SCHEMA:
        if (
            not isinstance(media_lineage, Mapping)
            or set(media_lineage) != REQUIRED_MEDIA
        ):
            raise contracts.ContractError(
                "generated animation review media-lineage coverage changed"
            )
        for name, lineage in sorted(media_lineage.items()):
            if not isinstance(lineage, Mapping) or set(lineage) != {
                "render_manifest",
                "encode_manifest",
            }:
                raise contracts.ContractError(
                    f"review media-lineage fields are invalid: {name}"
                )
            for kind in ("render_manifest", "encode_manifest"):
                authenticated[f"media_lineage:{name}:{kind}"] = (
                    _verify_review_descriptor(
                        lineage[kind],
                        f"review media-lineage {name} {kind}",
                        review_root=review_root,
                    )
                )
    elif media_lineage is not None:
        raise contracts.ContractError(
            "legacy v3 animation review unexpectedly contains media lineage"
        )

    input_paths = {name: authenticated[f"input:{name}"] for name in REVIEW_FILE_INPUTS}
    animated_path = _validate_review_lineage(
        review_path=path,
        review=payload,
        source_asset=source_asset,
        source_artifacts=source_artifacts,
        inputs=input_paths,
        outputs=outputs,
        authenticated=authenticated,
    )
    if payload["schema"] == BRANCHED_GENERATED_REVIEW_SCHEMA:
        labels = tuple(spec[0] for spec in generated_review.REVIEW_MEDIA_SPECS)
        if set(labels) != REQUIRED_MEDIA:
            raise contracts.ContractError(
                "runner and bridge review-media label contracts differ"
            )
        runner_media, runner_lineage = _runner_call(
            "review render/encode media lineage",
            generated_review.require_review_media_set,
            {"review_root": review_root / "05_review"},
            labels,
            animated_path,
            REQUIRED_REVIEW_FRAMES,
        )
        if contracts.canonical_json(media) != contracts.canonical_json(
            runner_media
        ) or contracts.canonical_json(media_lineage) != contracts.canonical_json(
            runner_lineage
        ):
            raise contracts.ContractError(
                "review media/lineage contradicts authenticated render and encode receipts"
            )
    document = _read_glb_document(animated_path)
    action_names = [
        animation.get("name")
        for animation in document.get("animations", [])
        if isinstance(animation, Mapping)
    ]
    if (
        len(action_names) != 2
        or set(action_names) != {"Idle", "Walking"}
        or len(document.get("skins", [])) != 1
        or not document.get("meshes")
    ):
        raise contracts.ContractError(
            "reviewed GLB must contain one skin and exactly Idle/Walking"
        )
    return path, payload, animated_path, authenticated


def load_animation_decision(
    path: Path,
    *,
    expected_file_sha256: str,
    source_asset: Mapping[str, Any],
    review_path: Path,
    review_payload: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    expected_file_sha256 = _require_sha256(
        expected_file_sha256, "expected animation decision file sha256"
    )
    path = _direct_file(path, "animation decision")
    if _sha256_file(path) != expected_file_sha256:
        raise contracts.ContractError(
            "animation decision does not match the external expected SHA-256"
        )
    payload = _load_finite_json(path, "animation decision")
    expected_fields = {
        "schema",
        "asset_id",
        "review_sha256",
        "decision",
        "checks",
        "caveats",
        "notes",
        "review",
        "state_classification",
        "formal_dataset_registration_authorized",
        "next_gate",
        "decision_sha256",
    }
    checks = payload.get("checks", {}) if isinstance(payload, Mapping) else {}
    caveats = payload.get("caveats") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or payload.get("schema") != DECISION_SCHEMA
        or payload.get("asset_id") != source_asset["asset_id"]
        or payload.get("decision") != "approved_for_ue_apartment"
        or payload.get("state_classification") != "research_candidate"
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("next_gate")
        != "ue_import_metric_trajectory_audio_and_apartment_media"
        or payload.get("decision_sha256") != _hash_without(payload, "decision_sha256")
        or not isinstance(checks, Mapping)
        or set(checks) != DECISION_CHECK_FIELDS
        or not all(value is True for value in checks.values())
        or not isinstance(caveats, list)
        or any(not isinstance(value, str) or not value for value in caveats)
        or len(caveats) != len(set(caveats))
        or not isinstance(payload.get("notes"), str)
        or not payload["notes"].strip()
        or review_payload.get("schema") not in GENERATED_REVIEW_SCHEMAS
        or review_payload.get("status") != "research_candidate_pending_human_review"
        or review_payload.get("formal_dataset_registration_authorized") is not False
        or review_payload.get("automatic_admission_gates", {}).get(
            "all_automatic_gates_passed"
        )
        is not True
    ):
        raise contracts.ContractError(
            "animation decision is missing, changed, or not approved"
        )
    review_descriptor = payload["review"]
    _verify_descriptor(
        review_descriptor, "approved animation review", expected_path=review_path
    )
    if payload["review_sha256"] != review_descriptor["sha256"]:
        raise contracts.ContractError("animation decision review identity changed")
    return path, payload


def _file_guard(path: Path, label: str) -> dict[str, Any]:
    path = _direct_file(path, label)
    current = os.stat(path, follow_symlinks=False)
    return {
        "path": str(path),
        "device": current.st_dev,
        "inode": current.st_ino,
        "mode": stat.S_IMODE(current.st_mode),
        "link_count": current.st_nlink,
        "size_bytes": current.st_size,
        "mtime_ns": current.st_mtime_ns,
        "ctime_ns": current.st_ctime_ns,
    }


def _require_unchanged_guard(
    expected: Mapping[str, Any], path: Path, label: str
) -> None:
    if _file_guard(path, label) != expected:
        raise contracts.ContractError(f"{label} identity changed during authentication")


def load_presentation_evidence(
    value: Any,
    *,
    review_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if (
        not isinstance(value, Mapping)
        or set(value) != PRESENTATION_EVIDENCE_FIELDS
        or not isinstance(value.get("presentation_receipt"), Mapping)
        or set(value["presentation_receipt"]) != {"path", "sha256", "size_bytes"}
    ):
        raise contracts.ContractError(
            "decision freeze receipt presentation evidence fields are invalid"
        )
    expected_receipt_sha256 = _require_sha256(
        value.get("expected_presentation_receipt_file_sha256"),
        "expected presentation receipt file sha256",
    )
    descriptor = value["presentation_receipt"]
    if (
        not isinstance(descriptor.get("path"), str)
        or not Path(descriptor["path"]).is_absolute()
        or descriptor.get("sha256") != expected_receipt_sha256
    ):
        raise contracts.ContractError(
            "decision freeze receipt presentation descriptor is invalid"
        )
    receipt_path = _direct_file(
        Path(descriptor["path"]), "owner-review presentation receipt"
    )
    receipt_guard = _file_guard(receipt_path, "owner-review presentation receipt")
    review_path = _direct_file(review_path, "generated animation review")
    review_sha256 = _sha256_file(review_path)
    try:
        receipt_payload, raw_record = presentation.load_presentation_receipt(
            receipt_path,
            expected_receipt_sha256,
            expected_source_review_sha256=review_sha256,
        )
    except (
        presentation.PresentationContractError,
        contracts.StrictJSONError,
        OSError,
        ValueError,
    ) as error:
        raise contracts.ContractError(
            f"owner-review presentation receipt is invalid: {error}"
        ) from error
    _require_unchanged_guard(
        receipt_guard,
        receipt_path,
        "owner-review presentation receipt",
    )
    if (
        raw_record != descriptor
        or receipt_payload.get("source_review") != _absolute_record(review_path)
        or receipt_payload.get("expected_source_review_sha256") != review_sha256
        or value.get("presentation_receipt_sha256")
        != receipt_payload.get("receipt_sha256")
        or value.get("output_video") != receipt_payload.get("output")
    ):
        raise contracts.ContractError(
            "decision freeze receipt presentation authority binding changed"
        )

    output = receipt_payload["output"]
    output_path_value = output.get("path")
    if (
        not isinstance(output_path_value, str)
        or not Path(output_path_value).is_absolute()
    ):
        raise contracts.ContractError(
            "owner-review presentation output path is invalid"
        )
    output_path = _direct_file(
        Path(output_path_value), "owner-review presentation output video"
    )
    output_guard = _file_guard(output_path, "owner-review presentation output video")
    if (
        receipt_path.name != presentation.RECEIPT_NAME
        or output_path.name != presentation.OUTPUT_VIDEO_NAME
        or receipt_path.parent != output_path.parent
        or {path.name for path in receipt_path.parent.iterdir()}
        != {presentation.RECEIPT_NAME, presentation.OUTPUT_VIDEO_NAME}
        or _absolute_record(output_path)
        != {key: output[key] for key in ("path", "sha256", "size_bytes")}
    ):
        raise contracts.ContractError(
            "owner-review presentation output bytes/directory binding changed"
        )
    directory_stat = os.stat(receipt_path.parent, follow_symlinks=False)
    receipt_stat = os.stat(receipt_path, follow_symlinks=False)
    output_stat = os.stat(output_path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or stat.S_IMODE(directory_stat.st_mode) != 0o555
        or not stat.S_ISREG(receipt_stat.st_mode)
        or stat.S_IMODE(receipt_stat.st_mode) != 0o444
        or receipt_stat.st_nlink != 1
        or not stat.S_ISREG(output_stat.st_mode)
        or stat.S_IMODE(output_stat.st_mode) != 0o444
        or output_stat.st_nlink != 1
    ):
        raise contracts.ContractError(
            "owner-review presentation output is not a sealed publication"
        )
    _require_unchanged_guard(
        output_guard,
        output_path,
        "owner-review presentation output video",
    )
    _require_unchanged_guard(
        receipt_guard,
        receipt_path,
        "owner-review presentation receipt",
    )
    canonical_evidence = {
        "presentation_receipt": raw_record,
        "expected_presentation_receipt_file_sha256": expected_receipt_sha256,
        "presentation_receipt_sha256": receipt_payload["receipt_sha256"],
        "output_video": copy.deepcopy(receipt_payload["output"]),
    }
    if value != canonical_evidence:
        raise contracts.ContractError(
            "decision freeze receipt presentation evidence is non-canonical"
        )
    return canonical_evidence, receipt_payload, output_path


def load_animation_decision_freeze_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    source_registry_path: Path,
    expected_source_registry_sha256: str,
    source_registry_validation_mode: str,
    source_asset_path: Path,
    review_path: Path,
    decision_path: Path,
    decision_payload: Mapping[str, Any],
    authenticated_review_artifact_count: int,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    expected_file_sha256 = _require_sha256(
        expected_file_sha256,
        "expected animation decision freeze receipt file sha256",
    )
    path = _direct_file(path, "animation decision freeze receipt")
    if _sha256_file(path) != expected_file_sha256:
        raise contracts.ContractError(
            "animation decision freeze receipt does not match the external "
            "expected SHA-256"
        )
    payload = _load_finite_json(path, "animation decision freeze receipt")
    if (
        isinstance(payload, Mapping)
        and payload.get("schema") in LEGACY_DECISION_FREEZE_RECEIPT_SCHEMAS
    ):
        raise contracts.ContractError(
            "legacy v1 animation decision freeze receipts are audit-only and "
            "cannot authorize UE import preparation v3"
        )
    instruction = payload.get("user_instruction_binding")
    authority = payload.get("user_instruction_authority")
    review_sha256 = _sha256_file(review_path)
    if (
        set(payload) != DECISION_FREEZE_RECEIPT_FIELDS
        or payload.get("schema") != DECISION_FREEZE_RECEIPT_SCHEMA
        or payload.get("status") != "frozen"
        or payload.get("state_classification") != "research_candidate"
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("receipt_sha256") != _hash_without(payload, "receipt_sha256")
        or payload.get("expected_source_asset_registry_file_sha256")
        != expected_source_registry_sha256
        or payload.get("source_asset_registry_validation_mode")
        != source_registry_validation_mode
        or payload.get("expected_animation_review_file_sha256") != review_sha256
        or not isinstance(instruction, Mapping)
        or set(instruction)
        != {
            "decision",
            "review_sha256",
            "all_six_checks_explicit",
            "presentation_receipt_file_sha256",
        }
        or instruction.get("decision") != "approved_for_ue_apartment"
        or instruction.get("review_sha256") != review_sha256
        or instruction.get("all_six_checks_explicit") is not True
        or authority != USER_INSTRUCTION_AUTHORITY
        or isinstance(payload.get("authenticated_review_artifact_count"), bool)
        or not isinstance(
            payload.get("authenticated_review_artifact_count"),
            int,
        )
        or payload["authenticated_review_artifact_count"]
        != authenticated_review_artifact_count
        or payload.get("decision_sha256") != decision_payload.get("decision_sha256")
    ):
        raise contracts.ContractError(
            "animation decision freeze receipt authority contract is invalid"
        )
    _verify_descriptor(
        payload.get("source_asset_registry"),
        "freeze receipt source registry",
        expected_path=source_registry_path,
    )
    _verify_descriptor(
        payload.get("source_asset"),
        "freeze receipt source asset",
        expected_path=source_asset_path,
    )
    _verify_descriptor(
        payload.get("animation_review"),
        "freeze receipt animation review",
        expected_path=review_path,
    )
    if (
        payload.get("source_asset_registry") != _absolute_record(source_registry_path)
        or payload.get("source_asset") != _absolute_record(source_asset_path)
        or payload.get("animation_review") != _absolute_record(review_path)
    ):
        raise contracts.ContractError(
            "animation decision freeze receipt descriptors are non-canonical"
        )
    bound_decision = _verify_relative_descriptor(
        payload.get("animation_decision"),
        path.parent,
        "freeze receipt animation decision",
    )
    if bound_decision != decision_path or payload.get(
        "animation_decision"
    ) != _relative_record(decision_path, path.parent):
        raise contracts.ContractError(
            "animation decision freeze receipt decision identity changed"
        )
    presentation_evidence, _presentation_payload, _output_video = (
        load_presentation_evidence(
            payload.get("presentation_evidence"),
            review_path=review_path,
        )
    )
    if (
        instruction.get("presentation_receipt_file_sha256")
        != presentation_evidence["expected_presentation_receipt_file_sha256"]
    ):
        raise contracts.ContractError(
            "animation decision freeze receipt presentation instruction binding changed"
        )
    return path, payload, presentation_evidence


def _directory_guard(path: Path, label: str) -> dict[str, Any]:
    literal, _bridge_used = _uses_only_exact_tmp_bridge(path, label)
    resolved = literal.resolve()
    current = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode):
        raise contracts.ContractError(f"{label} is not a directory")
    return {
        "path": str(resolved),
        "device": current.st_dev,
        "inode": current.st_ino,
        "mode": stat.S_IMODE(current.st_mode),
        "mtime_ns": current.st_mtime_ns,
        "ctime_ns": current.st_ctime_ns,
    }


def _require_owned_directory_identity(
    path: Path,
    expected: tuple[int, int],
    label: str,
) -> None:
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise contracts.ContractError(f"{label} disappeared") from error
    if (
        path.is_symlink()
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != expected
    ):
        raise contracts.ContractError(f"{label} identity changed")


def _open_directory_no_follow(path: Path, label: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise contracts.ContractError(f"cannot open stable {label}: {path}") from error
    current = os.fstat(descriptor)
    if not stat.S_ISDIR(current.st_mode):
        os.close(descriptor)
        raise contracts.ContractError(f"{label} is not a directory")
    return descriptor


def _require_path_matches_directory_fd(path: Path, descriptor: int, label: str) -> None:
    try:
        resolved = presentation.resolve_directory(path, label)
    except presentation.PresentationContractError as error:
        raise contracts.ContractError(
            f"{label} no longer names the held directory or satisfies the "
            f"path policy: {error}"
        ) from error
    path_stat = os.stat(resolved, follow_symlinks=False)
    descriptor_stat = os.fstat(descriptor)
    if (
        resolved != path
        or not stat.S_ISDIR(path_stat.st_mode)
        or (path_stat.st_dev, path_stat.st_ino)
        != (descriptor_stat.st_dev, descriptor_stat.st_ino)
    ):
        raise contracts.ContractError(f"{label} no longer names the held directory")


def _create_staging_at(
    parent_fd: int, output_name: str
) -> tuple[str, int, tuple[int, int]]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(128):
        name = f".{output_name}.{secrets.token_hex(12)}.staging"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except Exception:
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
        current = os.fstat(descriptor)
        return name, descriptor, (current.st_dev, current.st_ino)
    raise contracts.ContractError(
        "could not allocate a unique preparation staging name"
    )


def _write_json_at(directory_fd: int, name: str, value: Any) -> dict[str, Any]:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError as error:
        raise contracts.ContractError(
            f"refusing to replace preparation staging artifact: {name}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return {
        "path": name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def _seal_staging_at(directory_fd: int, expected_names: frozenset[str]) -> None:
    if set(os.listdir(directory_fd)) != expected_names:
        raise contracts.ContractError(
            "UE import preparation staging artifact set changed"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for name in sorted(expected_names):
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        try:
            current = os.fstat(descriptor)
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise contracts.ContractError(
                    f"UE import preparation staging artifact is unsafe: {name}"
                )
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
    os.fchmod(directory_fd, 0o555)
    os.fsync(directory_fd)


def _require_staging_records_at(
    directory_fd: int,
    expected_records: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(os.listdir(directory_fd)) != set(expected_records):
        raise contracts.ContractError(
            "UE import preparation staging artifact set changed before publication"
        )
    directory_stat = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or stat.S_IMODE(directory_stat.st_mode) != 0o555
    ):
        raise contracts.ContractError(
            "UE import preparation staging directory seal changed"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for name, expected in sorted(expected_records.items()):
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        try:
            before = os.fstat(descriptor)
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        guard_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            any(
                getattr(before, field) != getattr(after, field)
                for field in guard_fields
            )
            or not stat.S_ISREG(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o444
            or after.st_nlink != 1
        ):
            raise contracts.ContractError(
                f"UE import preparation staging artifact changed: {name}"
            )
        encoded = b"".join(chunks)
        observed = {
            "path": name,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
        }
        if observed != expected:
            raise contracts.ContractError(
                f"UE import preparation staging artifact bytes changed: {name}"
            )


def _atomic_publish_no_replace_at(
    parent_fd: int,
    staging_name: str,
    output_name: str,
    *,
    staging_fd: int | None = None,
    expected_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    try:
        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2")
    except AttributeError as error:
        raise contracts.ContractError(
            "atomic no-replace UE preparation publication is unavailable"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if staging_fd is not None or expected_records is not None:
        if staging_fd is None or expected_records is None:
            raise contracts.ContractError(
                "staging descriptor and records must be supplied together"
            )
        _require_staging_records_at(staging_fd, expected_records)
    result = renameat2(
        parent_fd,
        os.fsencode(staging_name),
        parent_fd,
        os.fsencode(output_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise contracts.ContractError(
            f"refusing to replace UE import preparation output: {output_name}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise contracts.ContractError(
            "atomic no-replace UE preparation publication is unsupported"
        )
    raise OSError(error_number, os.strerror(error_number), output_name)


def _remove_owned_staging_at(
    parent_fd: int,
    name: str,
    identity: tuple[int, int],
    allowed_names: frozenset[str],
) -> bool:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(name, flags, dir_fd=parent_fd)
    except (FileNotFoundError, NotADirectoryError, OSError):
        return False
    try:
        current = os.fstat(directory_fd)
        if (current.st_dev, current.st_ino) != identity:
            return False
        children = set(os.listdir(directory_fd))
        if not children.issubset(allowed_names):
            return False
        os.fchmod(directory_fd, 0o700)
        for child in sorted(children):
            child_stat = os.stat(
                child,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISDIR(child_stat.st_mode):
                return False
            os.unlink(child, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        return False
    os.rmdir(name, dir_fd=parent_fd)
    return True


def _authenticate_import_authority(
    *,
    source_registry_manifest_path: Path,
    expected_source_registry_sha256: str,
    source_asset_path: Path,
    animation_review_path: Path,
    animation_decision_path: Path,
    expected_animation_decision_sha256: str,
    animation_decision_freeze_receipt_path: Path,
    expected_animation_decision_freeze_receipt_sha256: str,
    roots: Mapping[str, Path],
) -> dict[str, Any]:
    (
        source_registry_path,
        source_registry_payload,
        source_request,
        source_profile,
        source_registry_validation_mode,
    ) = load_source_registry_anchor(
        source_registry_manifest_path,
        source_asset_path,
        expected_file_sha256=expected_source_registry_sha256,
    )
    source_path, source_asset, source_artifacts = load_source_asset(
        source_asset_path,
        roots,
        request=source_request,
        profile=source_profile,
        require_derived_authority=(
            source_registry_payload.get("schema")
            == source_registry.DERIVED_REGISTRY_SCHEMA
        ),
    )
    (
        review_path,
        review,
        animated_glb,
        review_artifacts,
    ) = load_animation_review(
        animation_review_path,
        source_asset=source_asset,
        source_artifacts=source_artifacts,
    )
    if review.get("schema") != BRANCHED_GENERATED_REVIEW_SCHEMA:
        raise contracts.ContractError(
            "UE import preparation v3 requires a v4 animation review with "
            "branch, render/encode lineage, and owner-review presentation "
            "evidence; legacy v3 is audit-only"
        )
    decision_path, decision = load_animation_decision(
        animation_decision_path,
        expected_file_sha256=expected_animation_decision_sha256,
        source_asset=source_asset,
        review_path=review_path,
        review_payload=review,
    )
    (
        freeze_receipt_path,
        freeze_receipt,
        presentation_evidence,
    ) = load_animation_decision_freeze_receipt(
        animation_decision_freeze_receipt_path,
        expected_file_sha256=expected_animation_decision_freeze_receipt_sha256,
        source_registry_path=source_registry_path,
        expected_source_registry_sha256=expected_source_registry_sha256,
        source_registry_validation_mode=source_registry_validation_mode,
        source_asset_path=source_path,
        review_path=review_path,
        decision_path=decision_path,
        decision_payload=decision,
        authenticated_review_artifact_count=len(review_artifacts),
    )
    guarded_paths = {
        source_registry_path,
        source_path,
        review_path,
        decision_path,
        freeze_receipt_path,
        animated_glb,
        *source_artifacts.values(),
        *review_artifacts.values(),
        Path(presentation_evidence["presentation_receipt"]["path"]),
        Path(presentation_evidence["output_video"]["path"]),
    }
    file_guards = {
        str(path): _file_guard(path, f"authority artifact {path.name}")
        for path in sorted(guarded_paths, key=str)
    }
    presentation_root = Path(
        presentation_evidence["presentation_receipt"]["path"]
    ).parent
    return {
        "source_registry_path": source_registry_path,
        "source_registry_payload": source_registry_payload,
        "source_request": source_request,
        "source_profile": source_profile,
        "source_registry_validation_mode": source_registry_validation_mode,
        "source_path": source_path,
        "source_asset": source_asset,
        "source_artifacts": source_artifacts,
        "review_path": review_path,
        "review": review,
        "animated_glb": animated_glb,
        "review_artifacts": review_artifacts,
        "decision_path": decision_path,
        "decision": decision,
        "freeze_receipt_path": freeze_receipt_path,
        "freeze_receipt": freeze_receipt,
        "presentation_evidence": presentation_evidence,
        "file_guards": file_guards,
        "presentation_directory_guard": _directory_guard(
            presentation_root,
            "owner-review presentation directory",
        ),
    }


def prepare_import(
    *,
    source_registry_manifest_path: Path,
    expected_source_registry_sha256: str,
    source_asset_path: Path,
    animation_review_path: Path,
    animation_decision_path: Path,
    expected_animation_decision_sha256: str,
    animation_decision_freeze_receipt_path: Path,
    expected_animation_decision_freeze_receipt_sha256: str,
    output_root: Path,
    artifact_roots: Mapping[str, Path] | None = None,
    ue_compatible_glb_path: Path | None = None,
    texture_transcode_manifest_path: Path | None = None,
) -> Path:
    roots = {
        name: _uses_only_exact_tmp_bridge(
            Path(path),
            f"artifact root {name}",
        )[0]
        for name, path in (artifact_roots or DEFAULT_ARTIFACT_ROOTS).items()
    }
    authority_arguments = {
        "source_registry_manifest_path": source_registry_manifest_path,
        "expected_source_registry_sha256": expected_source_registry_sha256,
        "source_asset_path": source_asset_path,
        "animation_review_path": animation_review_path,
        "animation_decision_path": animation_decision_path,
        "expected_animation_decision_sha256": expected_animation_decision_sha256,
        "animation_decision_freeze_receipt_path": (
            animation_decision_freeze_receipt_path
        ),
        "expected_animation_decision_freeze_receipt_sha256": (
            expected_animation_decision_freeze_receipt_sha256
        ),
        "roots": roots,
    }
    authority = _authenticate_import_authority(**authority_arguments)
    source_registry_path = authority["source_registry_path"]
    source_registry_payload = authority["source_registry_payload"]
    source_registry_validation_mode = authority["source_registry_validation_mode"]
    source_path = authority["source_path"]
    source_asset = authority["source_asset"]
    source_artifacts = authority["source_artifacts"]
    review_path = authority["review_path"]
    review = authority["review"]
    animated_glb = authority["animated_glb"]
    texture_transcode = _authenticate_texture_transcode(
        reviewed_glb=animated_glb,
        ue_compatible_glb_path=ue_compatible_glb_path,
        texture_transcode_manifest_path=texture_transcode_manifest_path,
    )
    import_glb = (
        texture_transcode["ue_compatible_glb"]
        if texture_transcode is not None
        else animated_glb
    )
    review_artifacts = authority["review_artifacts"]
    decision_path = authority["decision_path"]
    decision = authority["decision"]
    freeze_receipt_path = authority["freeze_receipt_path"]
    freeze_receipt = authority["freeze_receipt"]
    presentation_evidence = authority["presentation_evidence"]

    output_root = _new_output_path(Path(output_root), "output")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_parent_literal, _bridge_used = _uses_only_exact_tmp_bridge(
        output_root.parent,
        "output parent",
    )
    output_parent = output_parent_literal.resolve()
    if not output_parent.is_dir() or output_parent.is_symlink():
        raise contracts.ContractError("output parent is missing or unsafe")
    output_root = output_parent / output_root.name
    _new_output_path(output_root, "output")
    parent_fd = _open_directory_no_follow(
        output_parent,
        "UE import preparation output parent",
    )
    staging_fd = -1
    staging_name = ""
    staging_identity = (0, 0)
    expected_staging_names = frozenset(
        {
            "ue_import_jobs.json",
            "ue_import_preparation_manifest.json",
        }
    )
    published = False
    publication_path_verified = False
    try:
        _require_path_matches_directory_fd(
            output_parent,
            parent_fd,
            "UE import preparation output parent",
        )
        staging_name, staging_fd, staging_identity = _create_staging_at(
            parent_fd,
            output_root.name,
        )
        output_parent_guard = _directory_guard(output_parent, "output parent")
        asset_id = source_asset["asset_id"]
        tag = f"pixal_{asset_id}"
        import_job = {
            "job_type": IMPORT_JOB_TYPE,
            "asset_id": asset_id,
            "legacy_tag": asset_id,
            "tag": tag,
            "profile_schema_id": source_asset["profile_schema_id"],
            "sampled_attributes": copy.deepcopy(
                source_asset["sampled_attributes"]
            ),
            "expected_actions": ["Idle", "Walking"],
            "rigged_glb": str(import_glb),
            "rigged_glb_sha256": _sha256_file(import_glb),
            "source_registry_sha256": _sha256_file(source_registry_path),
            "source_asset_sha256": _sha256_file(source_path),
            "request_sha256": source_asset["request_sha256"],
            "animation_decision_file_sha256": _sha256_file(decision_path),
            "animation_decision_sha256": decision["decision_sha256"],
        }
        if texture_transcode is not None:
            import_job.update(
                {
                    "upstream_rigged_glb": str(animated_glb),
                    "upstream_rigged_glb_sha256": _sha256_file(animated_glb),
                    **_texture_transcode_job_fields(texture_transcode),
                }
            )
        import_payload = {
            "schema": IMPORT_SCHEMA,
            "status": "ready_for_new_ue_import",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "job_type": IMPORT_JOB_TYPE,
            "job_count": 1,
            "jobs": [import_job],
            "non_destructive_policy": (
                f"new unique gate_{tag} content directories; never rewrite or "
                "reuse any historical generated-animal UE job/content directory"
            ),
        }
        import_payload["batch_sha256"] = _hash_without(import_payload, "batch_sha256")
        import_record = _write_json_at(
            staging_fd,
            "ue_import_jobs.json",
            import_payload,
        )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "ready_for_new_ue_import",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "created_at": _utc_now(),
            "canonical_identity": {
                "asset_id": asset_id,
                "legacy_tag": asset_id,
                "tag": tag,
                "profile_schema_id": source_asset["profile_schema_id"],
                "profile_sha256": source_asset["profile_sha256"],
                "request_sha256": source_asset["request_sha256"],
                "taxonomy": copy.deepcopy(source_asset["taxonomy"]),
                "fixed_attributes": copy.deepcopy(source_asset["fixed_attributes"]),
                "sampled_attributes": copy.deepcopy(source_asset["sampled_attributes"]),
                "target_physical_profile": copy.deepcopy(
                    source_asset["target_physical_profile"]
                ),
            },
            "source_asset": _absolute_record(source_path),
            "source_asset_registry": _absolute_record(source_registry_path),
            "source_asset_registry_sha256": source_registry_payload["registry_sha256"],
            "expected_source_asset_registry_file_sha256": (
                expected_source_registry_sha256
            ),
            "source_asset_registry_validation_mode": (source_registry_validation_mode),
            "source_artifact_roots": {
                name: str(path) for name, path in sorted(roots.items())
            },
            "authenticated_source_artifact_count": len(source_artifacts),
            "animation_review": _absolute_record(review_path),
            "animation_review_schema": review["schema"],
            "animation_decision": _absolute_record(decision_path),
            "expected_animation_decision_file_sha256": (
                expected_animation_decision_sha256
            ),
            "animation_decision_sha256": decision["decision_sha256"],
            "animation_decision_freeze_receipt": _absolute_record(freeze_receipt_path),
            "expected_animation_decision_freeze_receipt_file_sha256": (
                expected_animation_decision_freeze_receipt_sha256
            ),
            "animation_decision_freeze_receipt_sha256": freeze_receipt[
                "receipt_sha256"
            ],
            "presentation_evidence": copy.deepcopy(presentation_evidence),
            "user_instruction_authority": copy.deepcopy(
                freeze_receipt["user_instruction_authority"]
            ),
            "reviewed_animated_glb": _absolute_record(animated_glb),
            "authenticated_review_artifact_count": len(review_artifacts),
            "ue_import_jobs": import_record,
            "automatic_checks": {
                "source_registry_and_preflight_reauthenticated": True,
                "source_registry_matched_external_expected_sha256": True,
                "source_asset_v2_validated_against_request_and_profile": True,
                "source_registry_workspace_bound_to_review_target": True,
                "all_source_asset_artifacts_and_licenses_reauthenticated": True,
                "raw_pixal_geometry_to_tokenrig_target_closure_reauthenticated": True,
                "all_generated_animation_automatic_gates_passed": True,
                "all_generated_animation_stage_lineage_reauthenticated": True,
                "support_plane_v2_output_readback_reauthenticated": True,
                "all_six_animation_review_media_reauthenticated": True,
                "all_six_animation_review_media_ffprobed": True,
                "all_six_animation_render_encode_receipts_reauthenticated": True,
                "human_animation_approval_matched_external_expected_sha256": True,
                "animation_decision_freeze_receipt_reauthenticated": True,
                **PRESENTATION_AUTOMATIC_CHECKS,
                "user_instruction_authority_preserved_without_cryptographic_upgrade": True,
                "reviewed_glb_has_embedded_skin_weights_and_exact_idle_walking_actions": True,
                "job_identity_and_attributes_copied_exactly_from_source_asset_v2": True,
                "ue_import_jobs_v2_identity_and_self_hash_validated": True,
                "fresh_non_destructive_ue_tag_derived_from_canonical_asset_id": True,
                "no_ue_execution_performed": True,
                "overall": "passed",
            },
        }
        manifest["manifest_sha256"] = _hash_without(manifest, "manifest_sha256")
        manifest_record = _write_json_at(
            staging_fd,
            "ue_import_preparation_manifest.json",
            manifest,
        )
        _seal_staging_at(staging_fd, expected_staging_names)
        staging_stat = os.fstat(staging_fd)
        if (staging_stat.st_dev, staging_stat.st_ino) != staging_identity:
            raise contracts.ContractError(
                "UE import preparation staging directory identity changed"
            )
        final_authority = _authenticate_import_authority(**authority_arguments)
        if final_authority != authority:
            raise contracts.ContractError(
                "UE import preparation authority graph changed during generation"
            )
        final_texture_transcode = _authenticate_texture_transcode(
            reviewed_glb=final_authority["animated_glb"],
            ue_compatible_glb_path=ue_compatible_glb_path,
            texture_transcode_manifest_path=texture_transcode_manifest_path,
        )
        if final_texture_transcode != texture_transcode:
            raise contracts.ContractError(
                "texture transcode authority graph changed during generation"
            )
        if (
            manifest["presentation_evidence"]
            != final_authority["presentation_evidence"]
            or manifest["presentation_evidence"]
            != freeze_receipt["presentation_evidence"]
        ):
            raise contracts.ContractError(
                "UE import preparation presentation evidence changed during generation"
            )
        _require_owned_directory_identity(
            output_parent / staging_name,
            staging_identity,
            "UE import preparation staging directory",
        )
        if _directory_guard(output_parent, "output parent") != output_parent_guard:
            raise contracts.ContractError(
                "UE import preparation output parent changed during generation"
            )
        _require_path_matches_directory_fd(
            output_parent,
            parent_fd,
            "UE import preparation output parent",
        )
        _atomic_publish_no_replace_at(
            parent_fd,
            staging_name,
            output_root.name,
            staging_fd=staging_fd,
            expected_records={
                "ue_import_jobs.json": import_record,
                "ue_import_preparation_manifest.json": manifest_record,
            },
        )
        published = True
        _require_path_matches_directory_fd(
            output_parent,
            parent_fd,
            "published UE import preparation output parent",
        )
        publication_path_verified = True
        os.fsync(parent_fd)
        return output_root / "ue_import_preparation_manifest.json"
    except Exception:
        cleanup_name = output_root.name if published else staging_name
        if staging_name and (not published or not publication_path_verified):
            _remove_owned_staging_at(
                parent_fd,
                cleanup_name,
                staging_identity,
                expected_staging_names,
            )
        raise
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(parent_fd)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry-manifest", required=True, type=Path)
    parser.add_argument(
        "--expected-source-registry-sha256",
        required=True,
        help=(
            "Externally recorded SHA-256 of the complete source-registry "
            "manifest; the registry's self-hash is not an authority anchor."
        ),
    )
    parser.add_argument("--source-asset", required=True, type=Path)
    parser.add_argument("--animation-review", required=True, type=Path)
    parser.add_argument("--animation-decision", required=True, type=Path)
    parser.add_argument(
        "--expected-animation-decision-sha256",
        required=True,
        help=(
            "Externally recorded SHA-256 of the complete animation-decision "
            "file; the decision's self-hash is not an approval authority."
        ),
    )
    parser.add_argument(
        "--animation-decision-freeze-receipt",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-animation-decision-freeze-receipt-sha256",
        required=True,
        help=(
            "Externally recorded SHA-256 of the complete decision-freeze receipt file."
        ),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--ue-compatible-glb",
        type=Path,
        help=(
            "Existing PNG-textured GLB derivative of the exact owner-reviewed "
            "animated GLB. Must be paired with --texture-transcode-manifest."
        ),
    )
    parser.add_argument(
        "--texture-transcode-manifest",
        type=Path,
        help=(
            "Existing transcode_glb_webp_to_png.py manifest. Must be paired "
            "with --ue-compatible-glb."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        metavar="ROOT_ID=PATH",
        help="Add or override a source_asset_v2 artifact root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = prepare_import(
            source_registry_manifest_path=args.source_registry_manifest,
            expected_source_registry_sha256=args.expected_source_registry_sha256,
            source_asset_path=args.source_asset,
            animation_review_path=args.animation_review,
            animation_decision_path=args.animation_decision,
            expected_animation_decision_sha256=(
                args.expected_animation_decision_sha256
            ),
            animation_decision_freeze_receipt_path=(
                args.animation_decision_freeze_receipt
            ),
            expected_animation_decision_freeze_receipt_sha256=(
                args.expected_animation_decision_freeze_receipt_sha256
            ),
            output_root=args.output_root,
            artifact_roots=parse_artifact_roots(args.artifact_root),
            ue_compatible_glb_path=args.ue_compatible_glb,
            texture_transcode_manifest_path=args.texture_transcode_manifest,
        )
        payload = _load_strict_json(manifest, "UE import preparation output")
    except (contracts.ContractError, OSError) as error:
        print(
            f"USER_APPROVED_GENERATED_ANIMAL_UE_PREP_FAILED {error}",
            file=sys.stderr,
        )
        return 2
    print(
        "USER_APPROVED_GENERATED_ANIMAL_UE_PREP_OK "
        f"asset={payload['canonical_identity']['asset_id']} output={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
