#!/usr/bin/env python3
"""Build authenticated Apartment Walk/Idle specs for one user-approved generated animal."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import (
    prepare_user_approved_generated_animal_ue_imports as preparation_bridge,
)
from tools import register_controlled_animal_source_assets as source_registry
from tools import run_target_native_generated_quadruped_review as generated_review
from tools.spike_rlr.animal_audio import bind_pinned_animal_audio_contract


CONFIG_SCHEMA = "user_approved_generated_animal_apartment_config_v1"
OUTPUT_SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
PHYSICAL_MEASUREMENT = "shoulder_height_cm"
RIG_SEMANTIC_EVIDENCE_SCHEMA = "controlled_animal_rig_semantic_evidence_v1"
LEGACY_UNGATED_GENERATED_REVIEW_SCHEMA = (
    "avengine_target_native_generated_quadruped_review_run_v1"
)
LEGACY_GENERATED_REVIEW_SCHEMAS = frozenset(
    {
        LEGACY_UNGATED_GENERATED_REVIEW_SCHEMA,
        "avengine_target_native_generated_quadruped_review_run_v2",
        "avengine_target_native_generated_quadruped_review_run_v3",
    }
)
FORMAL_GENERATED_REVIEW_SCHEMA = (
    "avengine_target_native_generated_quadruped_review_run_v4"
)
GENERATED_REVIEW_SCHEMAS = LEGACY_GENERATED_REVIEW_SCHEMAS | {
    FORMAL_GENERATED_REVIEW_SCHEMA
}
RETARGET_SCHEMA = "avengine_generated_quadruped_retarget_v5"
WEIGHT_REPAIR_SCHEMA = "avengine_motion_aware_quadruped_weight_repair_v2"
DECISION_SCHEMA = "avengine_controlled_animal_animation_decision_v1"
FORMAL_BATCH_SCHEMA = "pixal_animal_ue_import_batch_v2"
LEGACY_BATCH_SCHEMA = "pixal_animal_ue_import_batch_v1"
FORMAL_RESULT_SCHEMA = "pixal_animal_ue_import_result_v2"
LEGACY_RESULT_SCHEMA = "pixal_animal_ue_import_result_v1"
PREPARATION_SCHEMA = preparation_bridge.SCHEMA
LEGACY_PREPARATION_SCHEMA = (
    "avengine_user_approved_generated_animal_ue_import_preparation_v1"
)
DECISION_FREEZE_RECEIPT_SCHEMA = preparation_bridge.DECISION_FREEZE_RECEIPT_SCHEMA
USER_INSTRUCTION_AUTHORITY = copy.deepcopy(
    preparation_bridge.USER_INSTRUCTION_AUTHORITY
)
DECISION_FREEZE_RECEIPT_FIELDS = preparation_bridge.DECISION_FREEZE_RECEIPT_FIELDS
SOURCE_REGISTRY_VALIDATION_MODES = frozenset(
    {"frozen_historical_preflight_v1", "current_exact_rebuild"}
)
JOB_TYPE = "user_approved_generated_animal"
EXPECTED_ACTIONS = ["Idle", "Walking"]
CANONICAL_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
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
DECISION_FIELDS = frozenset(
    {
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
)
BATCH_FIELDS = frozenset(
    {
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
)
JOB_FIELDS = frozenset(
    {
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
)
RESULT_FIELDS = frozenset(
    {
        "schema",
        "generated_at",
        "state_classification",
        "formal_dataset_registration_authorized",
        "preparation_manifest",
        "input_manifest",
        "batch_identity",
        "non_destructive_policy",
        "status",
        "passed_count",
        "results",
    }
)
RESULT_ITEM_FIELDS = frozenset(
    {
        "job_type",
        "asset_id",
        "tag",
        "legacy_tag",
        "job_identity_sha256",
        "source",
        "source_sha256",
        "mesh_content_dir",
        "skeletal_mesh",
        "walking_animation",
        "blueprint",
        "asset_count",
        "assets",
        "actions",
        "status",
    }
)
BATCH_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "job_type",
        "job_count",
        "batch_sha256",
        "asset_ids",
        "tags",
        "job_identity_sha256s",
    }
)
PREPARATION_FIELDS = frozenset(
    {
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
)
PREPARATION_AUTOMATIC_CHECK_FIELDS = frozenset(
    {
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
)
CANONICAL_IDENTITY_FIELDS = frozenset(
    {
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
)
SPEAR_ROOT = Path(__file__).resolve().parents[1]
SPEAR_TMP_BRIDGE = (SPEAR_ROOT / "tmp").absolute()
V4_FINAL_WEIGHT_REPAIR_ARTIFACTS = {
    branch: generated_review.weight_repair_final_artifact(branch)
    for branch in generated_review.WEIGHT_REPAIR_BRANCH_STAGES
}
TARGET_PHYSICAL_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "control_attribute",
        "selected_value",
        "measurement",
        "mode",
        "reference_value_cm",
        "reference_provenance",
        "scale_ratio",
        "tolerance_cm",
        "target_value_cm",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _has_symlink_component(path: Path) -> bool:
    try:
        _uses_only_exact_tmp_bridge(path, "path")
    except contracts.ContractError:
        return True
    return False


def _direct_file(path: Path, label: str) -> Path:
    raw, bridge_used = _uses_only_exact_tmp_bridge(Path(path), label)
    resolved = raw.resolve()
    if bridge_used:
        try:
            resolved.relative_to(SPEAR_TMP_BRIDGE.resolve())
        except ValueError as error:
            raise contracts.ContractError(
                f"{label} escaped the exact SPEAR/tmp bridge"
            ) from error
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise contracts.ContractError(f"missing or unsafe {label}: {raw}")
    return resolved


def _new_output_path(path: Path, label: str) -> Path:
    literal, _bridge_used = _uses_only_exact_tmp_bridge(Path(path), label)
    if literal.exists() or literal.is_symlink():
        raise contracts.ContractError(f"refusing to replace {label}: {literal}")
    return literal


def _canonical_hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    return hashlib.sha256(contracts.canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise contracts.ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _authenticate_external_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> Path:
    expected_sha256 = _require_sha256(
        expected_sha256,
        f"external {label} SHA-256",
    )
    path = _direct_file(path, label)
    if _sha256(path) != expected_sha256:
        raise contracts.ContractError(f"external {label} SHA-256 pin mismatched")
    return path


def _artifact(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    path = _direct_file(path, "artifact")
    published = Path(published_path) if published_path is not None else path
    if _has_symlink_component(published):
        raise contracts.ContractError(
            f"artifact publication path cannot contain a symlink: {published}"
        )
    return {
        "path": str(published.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _descriptor_matches(record: Mapping[str, Any]) -> bool:
    try:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            return False
        raw = Path(str(record["path"]))
        size = record["size_bytes"]
        sha256 = record["sha256"]
        if (
            _has_symlink_component(raw)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            return False
        path = raw.resolve()
        return bool(
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == size
            and _sha256(path) == sha256
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _validate_absolute_descriptor(
    value: Any,
    label: str,
    *,
    extra_fields: frozenset[str] = frozenset(),
    expected_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    fields = {"path", "sha256", "size_bytes"} | set(extra_fields)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise contracts.ContractError(f"{label} descriptor fields changed")
    raw_path = value.get("path")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not Path(raw_path).is_absolute()
    ):
        raise contracts.ContractError(f"{label} descriptor path must be absolute")
    base = {
        "path": raw_path,
        "sha256": value.get("sha256"),
        "size_bytes": value.get("size_bytes"),
    }
    if not _descriptor_matches(base):
        raise contracts.ContractError(f"{label} descriptor does not bind its file")
    path = Path(raw_path).resolve()
    if expected_path is not None and path != Path(expected_path).resolve():
        raise contracts.ContractError(f"{label} descriptor binds the wrong file")
    return path, copy.deepcopy(dict(value))


def _validate_relative_descriptor(
    value: Any,
    label: str,
    *,
    root: Path,
    expected_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise contracts.ContractError(f"{label} descriptor fields changed")
    raw_path = value.get("path")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or "\\" in raw_path
        or Path(raw_path).is_absolute()
    ):
        raise contracts.ContractError(f"{label} must use a safe relative path")
    parts = Path(raw_path).parts
    if "." in parts or ".." in parts:
        raise contracts.ContractError(f"{label} relative path escapes its root")
    unresolved = root.joinpath(*parts)
    if _has_symlink_component(unresolved):
        raise contracts.ContractError(f"{label} cannot use a symlink path")
    path = unresolved.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise contracts.ContractError(
            f"{label} relative path escapes its root"
        ) from error
    if (
        path != Path(expected_path).resolve()
        or not path.is_file()
        or path.is_symlink()
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] <= 0
        or path.stat().st_size != value["size_bytes"]
        or _sha256(path) != _require_sha256(value.get("sha256"), f"{label} hash")
    ):
        raise contracts.ContractError(f"{label} descriptor does not bind its file")
    return copy.deepcopy(dict(value))


def _load(path: Path) -> dict[str, Any]:
    path = _direct_file(path, "JSON input")
    try:
        value = contracts.load_json(path)
    except (OSError, contracts.ContractError) as error:
        raise contracts.ContractError(
            f"cannot load strict JSON {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise contracts.ContractError(f"JSON object required: {path}")
    return value


def _positive_finite(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_integer(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise contracts.ContractError(f"{label} must be an integer")
    if positive and value <= 0:
        raise contracts.ContractError(f"{label} must be positive")
    if nonnegative and value < 0:
        raise contracts.ContractError(f"{label} must be nonnegative")
    return value


def _require_number(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if not _finite_number(value):
        raise contracts.ContractError(f"{label} must be a finite number")
    number = float(value)
    if positive and number <= 0.0:
        raise contracts.ContractError(f"{label} must be positive")
    if nonnegative and number < 0.0:
        raise contracts.ContractError(f"{label} must be nonnegative")
    return number


def _require_vector(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise contracts.ContractError(f"{label} must contain exactly three numbers")
    return tuple(
        _require_number(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )


def _validate_template_numeric_contract(template: Mapping[str, Any]) -> None:
    render = template.get("render_config")
    audio = template.get("audio_config")
    sources = template.get("sources")
    camera_contract = template.get("camera_pass_table_loop_contract")
    windows = template.get("rig_direction_check_windows")
    if not isinstance(render, Mapping):
        raise contracts.ContractError("template render_config must be an object")
    if not isinstance(audio, Mapping):
        raise contracts.ContractError("template audio_config must be an object")
    if (
        not isinstance(sources, list)
        or len(sources) != 1
        or not isinstance(sources[0], Mapping)
    ):
        raise contracts.ContractError("template must contain exactly one source")
    if not isinstance(camera_contract, Mapping):
        raise contracts.ContractError(
            "template camera_pass_table_loop_contract must be an object"
        )
    if not isinstance(windows, list) or not windows:
        raise contracts.ContractError(
            "template rig_direction_check_windows must be non-empty"
        )

    _require_number(
        render.get("duration_s"), "template render_config.duration_s", positive=True
    )
    _require_integer(render.get("fps"), "template render_config.fps", positive=True)
    frame_count = _require_integer(
        render.get("n_frames"), "template render_config.n_frames", positive=True
    )
    for name in ("width", "height", "resolution_x", "resolution_y"):
        if name in render:
            _require_integer(
                render[name], f"template render_config.{name}", positive=True
            )
    for name in ("streaming_warmup_frames", "camera_warmup_frames"):
        if name in render:
            _require_integer(
                render[name], f"template render_config.{name}", nonnegative=True
            )

    _require_number(
        audio.get("duration_s"), "template audio_config.duration_s", positive=True
    )
    _require_integer(
        audio.get("sample_rate_hz"),
        "template audio_config.sample_rate_hz",
        positive=True,
    )

    source = sources[0]
    trajectory = source.get("trajectory_m")
    if not isinstance(trajectory, list) or not trajectory:
        raise contracts.ContractError("template trajectory_m must be non-empty")
    trajectory_vectors = [
        _require_vector(point, f"template trajectory_m[{index}]")
        for index, point in enumerate(trajectory)
    ]
    if len(trajectory_vectors) != frame_count:
        raise contracts.ContractError(
            "template trajectory length must match render_config.n_frames"
        )
    start = _require_vector(source.get("start_pos_m"), "template start_pos_m")
    end = _require_vector(source.get("end_pos_m"), "template end_pos_m")
    if start != trajectory_vectors[0] or end != trajectory_vectors[-1]:
        raise contracts.ContractError(
            "template start/end positions must bind the trajectory endpoints"
        )

    left_front_frame = _require_integer(
        camera_contract.get("left_front_nearest_frame"),
        "template left-front frame",
        nonnegative=True,
    )
    if left_front_frame >= frame_count:
        raise contracts.ContractError(
            "template left-front frame must be an in-range integer"
        )

    for index, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise contracts.ContractError(
                f"template rig direction window {index} must be an object"
            )
        if "frame_a" in window or "frame_b" in window:
            pair = ("frame_a", "frame_b")
        elif "start_frame" in window or "end_frame" in window:
            pair = ("start_frame", "end_frame")
        else:
            raise contracts.ContractError(
                f"template rig direction window {index} has no frame bounds"
            )
        bounds = [
            _require_integer(
                window.get(name),
                f"template rig direction window {index}.{name}",
                nonnegative=True,
            )
            for name in pair
        ]
        if bounds[0] > bounds[1] or bounds[1] >= frame_count:
            raise contracts.ContractError(
                f"template rig direction window {index} is out of range"
            )
        if "label" in window and (
            not isinstance(window["label"], str) or not window["label"]
        ):
            raise contracts.ContractError(
                f"template rig direction window {index}.label is invalid"
            )


def _validate_target_physical_profile(
    value: Any, sampled_attributes: Mapping[str, str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != TARGET_PHYSICAL_PROFILE_FIELDS:
        raise contracts.ContractError("target_physical_profile fields changed")
    profile = dict(value)
    control = profile.get("control_attribute")
    selected = profile.get("selected_value")
    provenance = profile.get("reference_provenance")
    if (
        not isinstance(profile.get("profile_id"), str)
        or not profile["profile_id"]
        or not isinstance(control, str)
        or not control
        or control not in sampled_attributes
        or not isinstance(selected, str)
        or selected != sampled_attributes[control]
        or profile.get("measurement") != PHYSICAL_MEASUREMENT
        or profile.get("mode")
        not in {"relative_to_profile_reference", "absolute_measurement"}
        or not all(
            _positive_finite(profile.get(name))
            for name in (
                "reference_value_cm",
                "scale_ratio",
                "tolerance_cm",
                "target_value_cm",
            )
        )
        or not isinstance(provenance, Mapping)
        or set(provenance) != {"status", "source_id", "artifact", "notes"}
        or provenance.get("status") not in {"provisional", "verified"}
        or not isinstance(provenance.get("source_id"), str)
        or not provenance["source_id"]
        or not isinstance(provenance.get("notes"), str)
        or not provenance["notes"]
    ):
        raise contracts.ContractError("target_physical_profile is invalid")
    expected_target = float(profile["reference_value_cm"]) * float(
        profile["scale_ratio"]
    )
    if not math.isclose(
        float(profile["target_value_cm"]),
        expected_target,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise contracts.ContractError(
            "target_physical_profile target does not match reference and scale"
        )
    artifact = provenance.get("artifact")
    if artifact is not None and not _descriptor_matches(artifact):
        raise contracts.ContractError(
            "target_physical_profile reference artifact changed"
        )
    if provenance["status"] == "verified" and artifact is None:
        raise contracts.ContractError(
            "verified target_physical_profile requires a reference artifact"
        )
    return copy.deepcopy(profile)


def _descriptor_binds_file(record: Any, path: Path, expected_sha256: str) -> bool:
    if not isinstance(record, Mapping) or not _descriptor_matches(record):
        return False
    try:
        raw_expected = Path(path)
        if _has_symlink_component(raw_expected):
            return False
        expected = raw_expected.resolve()
        return bool(
            Path(str(record["path"])).resolve() == expected
            and record["sha256"] == expected_sha256
            and record["size_bytes"] == expected.stat().st_size
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _require_front_semantic_chains(payload: Mapping[str, Any]) -> None:
    schema = payload.get("schema")
    if schema == WEIGHT_REPAIR_SCHEMA:
        chains = payload.get("semantic_rig", {}).get("chains")
    elif schema == RETARGET_SCHEMA:
        inference = payload.get("semantic_inference")
        if (
            not isinstance(inference, Mapping)
            or inference.get("bone_name_independent_target") is not True
            or inference.get("complete_target_bone_coverage") is not True
        ):
            raise contracts.ContractError(
                "retarget semantic inference is not bone-name independent"
            )
        chains = inference.get("chains")
    else:
        raise contracts.ContractError(
            f"unsupported generated rig semantic evidence schema: {schema}"
        )
    if not isinstance(chains, Mapping):
        raise contracts.ContractError("generated rig semantic chains are missing")
    selected: list[str] = []
    for label in ("front_side_negative", "front_side_positive"):
        chain = chains.get(label)
        if (
            not isinstance(chain, list)
            or len(chain) < 2
            or any(not isinstance(name, str) or not name for name in chain)
            or len(set(chain)) != len(chain)
        ):
            raise contracts.ContractError(
                f"generated rig semantic chain is invalid: {label}"
            )
        selected.append(chain[0])
    if len(set(selected)) != 2:
        raise contracts.ContractError(
            "generated rig front upper semantic bones are not distinct"
        )


def _authenticated_rig_semantic_evidence(
    decision: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Bind generated bone semantics to the exact final Walk/Idle GLB.

    Older manually repaired assets predate the generated-quadruped runner and
    retain descriptive or Quaternius vertex-group names.  Those reviews do not
    manufacture semantic evidence here.  A recognized runner review, however,
    must provide a complete hash-bound semantic artifact or fail closed.
    """

    review_path = _direct_file(
        Path(str(decision["review"]["path"])), "generated animal review"
    )
    review = _load(review_path)
    if review.get("schema") not in GENERATED_REVIEW_SCHEMAS:
        return None
    outputs = review.get("outputs")
    gates = review.get("automatic_admission_gates")
    runtime = _direct_file(Path(str(job["rigged_glb"])), "generated animal runtime GLB")
    runtime_sha256 = str(job["rigged_glb_sha256"])
    if (
        not isinstance(outputs, Mapping)
        or review.get("status") != "research_candidate_pending_human_review"
        or review.get("formal_dataset_registration_authorized") is not False
        or not _descriptor_binds_file(
            outputs.get("animated_glb"), runtime, runtime_sha256
        )
    ):
        raise contracts.ContractError(
            "generated animal review does not bind the approved final GLB"
        )
    if review.get("schema") == LEGACY_UNGATED_GENERATED_REVIEW_SCHEMA:
        return None
    if (
        not isinstance(gates, Mapping)
        or gates.get("all_automatic_gates_passed") is not True
    ):
        raise contracts.ContractError(
            "generated animal automatic admission gates did not pass"
        )

    if review.get("schema") == FORMAL_GENERATED_REVIEW_SCHEMA:
        branch = gates.get("weight_repair_branch")
        expected_final = V4_FINAL_WEIGHT_REPAIR_ARTIFACTS.get(str(branch))
        final = gates.get("weight_repair_final_artifact")
        if (
            expected_final is None
            or final != expected_final
            or not _descriptor_binds_file(
                outputs.get(expected_final["glb_output_descriptor"]),
                runtime,
                runtime_sha256,
            )
            or outputs.get("animated_glb")
            != outputs.get(expected_final["glb_output_descriptor"])
        ):
            raise contracts.ContractError(
                "generated animal v4 final weight-repair artifact changed"
            )
        manifest_role = expected_final["manifest_output_descriptor"]
        if manifest_role is None:
            role = "retarget_manifest"
            expected_schema = RETARGET_SCHEMA
            binding_field = "export"
            semantic_kind = "bone_name_independent_retarget"
        else:
            role = manifest_role
            if outputs.get("weight_repair_manifest") != outputs.get(role):
                raise contracts.ContractError(
                    "generated animal v4 final weight-repair manifest alias changed"
                )
            expected_schema = WEIGHT_REPAIR_SCHEMA
            binding_field = "output"
            semantic_kind = "motion_aware_weight_repair"
    elif outputs.get("weight_repair_manifest") is not None:
        role = "weight_repair_manifest"
        expected_schema = WEIGHT_REPAIR_SCHEMA
        binding_field = "output"
        semantic_kind = "motion_aware_weight_repair"
    else:
        role = "retarget_manifest"
        expected_schema = RETARGET_SCHEMA
        binding_field = "export"
        semantic_kind = "bone_name_independent_retarget"
    descriptor = outputs.get(role)
    if not isinstance(descriptor, Mapping) or not _descriptor_matches(descriptor):
        raise contracts.ContractError(
            f"generated animal review {role} is missing or changed"
        )
    evidence = _load(Path(str(descriptor["path"])))
    if evidence.get("schema") != expected_schema or not _descriptor_binds_file(
        evidence.get(binding_field), runtime, runtime_sha256
    ):
        raise contracts.ContractError(
            f"generated animal {role} does not bind the approved final GLB"
        )
    if expected_schema == WEIGHT_REPAIR_SCHEMA:
        authority = evidence.get("authority_contract")
        if not isinstance(authority, Mapping) or any(
            authority.get(field) is not True
            for field in (
                "native_mesh_geometry_preserved",
                "native_mesh_topology_preserved",
                "pbr_material_preserved",
                "fitted_skeleton_rest_matrices_preserved",
                "approved_animation_curves_preserved",
                "only_vertex_weights_modified_in_memory",
            )
        ):
            raise contracts.ContractError(
                "generated animal weight-repair authority did not pass"
            )
    _require_front_semantic_chains(evidence)
    return {
        "schema": RIG_SEMANTIC_EVIDENCE_SCHEMA,
        "kind": semantic_kind,
        "artifact": copy.deepcopy(dict(descriptor)),
        "semantic_schema": expected_schema,
        "source_glb_sha256": runtime_sha256,
    }


def _expected_non_destructive_policy(tag: str) -> str:
    return (
        f"new unique gate_{tag} content directories; never rewrite or "
        "reuse any historical generated-animal UE job/content directory"
    )


def _validate_formal_decision(
    decision: Mapping[str, Any],
    *,
    decision_path: Path,
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    runtime: Path,
) -> dict[str, Any]:
    checks = decision.get("checks")
    caveats = decision.get("caveats")
    if (
        set(decision) != DECISION_FIELDS
        or decision.get("schema") != DECISION_SCHEMA
        or decision.get("asset_id") != config["asset_id"]
        or decision.get("decision") != "approved_for_ue_apartment"
        or decision.get("state_classification") != "research_candidate"
        or decision.get("formal_dataset_registration_authorized") is not False
        or decision.get("next_gate")
        != "ue_import_metric_trajectory_audio_and_apartment_media"
        or decision.get("decision_sha256")
        != _canonical_hash_without(decision, "decision_sha256")
        or not isinstance(checks, Mapping)
        or set(checks) != DECISION_CHECK_FIELDS
        or any(value is not True for value in checks.values())
        or not isinstance(caveats, list)
        or any(not isinstance(value, str) or not value for value in caveats)
        or len(caveats) != len(set(caveats))
        or not isinstance(decision.get("notes"), str)
        or not decision["notes"].strip()
    ):
        raise contracts.ContractError(
            "formal user animation decision is missing, changed, or not approved"
        )
    review_path, review_descriptor = _validate_absolute_descriptor(
        decision.get("review"),
        "approved v4 animation review",
    )
    if decision.get("review_sha256") != review_descriptor["sha256"]:
        raise contracts.ContractError("animation decision review identity changed")
    review = _load(review_path)
    outputs = review.get("outputs")
    gates = review.get("automatic_admission_gates")
    if (
        review.get("schema") != FORMAL_GENERATED_REVIEW_SCHEMA
        or review.get("status") != "research_candidate_pending_human_review"
        or review.get("formal_dataset_registration_authorized") is not False
        or not isinstance(outputs, Mapping)
        or not isinstance(gates, Mapping)
        or gates.get("all_automatic_gates_passed") is not True
        or not _descriptor_binds_file(
            outputs.get("animated_glb"),
            runtime,
            str(job["rigged_glb_sha256"]),
        )
    ):
        raise contracts.ContractError(
            "formal Apartment publication requires the exact approved v4 review"
        )
    if (
        _sha256(decision_path)
        != _require_sha256(
            job.get("animation_decision_file_sha256"),
            "UE job animation decision file hash",
        )
        or job.get("animation_decision_sha256") != decision["decision_sha256"]
    ):
        raise contracts.ContractError("UE job does not bind the approved decision")
    return review


def _validate_source_registry_identity(
    preparation: Mapping[str, Any],
    *,
    source_asset_path: Path,
    source_registry_path: Path,
    source_asset_descriptor: Mapping[str, Any],
    source_registry_descriptor: Mapping[str, Any],
    config: Mapping[str, Any],
    job: Mapping[str, Any],
) -> None:
    source_asset = _load(source_asset_path)
    registry = _load(source_registry_path)
    if (
        source_asset.get("schema") != contracts.SOURCE_ASSET_SCHEMA
        or source_asset.get("asset_id") != job.get("asset_id")
        or source_asset.get("profile_schema_id") != job.get("profile_schema_id")
        or source_asset.get("request_sha256") != job.get("request_sha256")
        or source_asset.get("sampled_attributes") != job.get("sampled_attributes")
        or source_asset.get("taxonomy")
        != {"species": config.get("species"), "breed": config.get("breed")}
        or source_asset.get("fixed_attributes")
        != preparation.get("canonical_identity", {}).get("fixed_attributes")
        or source_asset.get("target_physical_profile")
        != config.get("target_physical_profile")
    ):
        raise contracts.ContractError(
            "UE import preparation canonical source asset identity changed"
        )
    if (
        set(registry) != preparation_bridge.REGISTRY_FIELDS
        or registry.get("schema") != source_registry.REGISTRY_SCHEMA
        or registry.get("state_classification") != "research_candidate"
        or registry.get("formal_dataset_registration_authorized") is not False
        or registry.get("registry_sha256")
        != _canonical_hash_without(registry, "registry_sha256")
        or registry.get("registry_sha256")
        != preparation.get("source_asset_registry_sha256")
        or registry.get("automatic_checks")
        != preparation_bridge.REGISTRY_AUTOMATIC_CHECKS
    ):
        raise contracts.ContractError(
            "UE import preparation source registry identity changed"
        )
    validation_mode = preparation.get("source_asset_registry_validation_mode")
    preflight = registry.get("preflight")
    if (
        validation_mode not in SOURCE_REGISTRY_VALIDATION_MODES
        or not isinstance(preflight, Mapping)
        or preflight.get("validation_mode") != validation_mode
    ):
        raise contracts.ContractError(
            "UE import preparation source registry validation mode changed"
        )
    entries = registry.get("source_assets")
    matching = (
        [
            entry
            for entry in entries
            if isinstance(entry, Mapping)
            and entry.get("asset_id") == job.get("asset_id")
        ]
        if isinstance(entries, list)
        else []
    )
    if (
        isinstance(registry.get("source_asset_count"), bool)
        or not isinstance(registry.get("source_asset_count"), int)
        or registry.get("source_asset_count") != len(entries or [])
        or len(matching) != 1
        or set(matching[0]) != preparation_bridge.REGISTRY_SOURCE_INDEX_FIELDS
        or matching[0].get("profile_schema_id") != job.get("profile_schema_id")
        or matching[0].get("request_sha256") != job.get("request_sha256")
        or matching[0].get("sampled_attributes") != job.get("sampled_attributes")
        or matching[0].get("state_classification") != "research_candidate"
    ):
        raise contracts.ContractError(
            "UE import preparation source registry entry changed"
        )
    registry_source_descriptor = _validate_relative_descriptor(
        matching[0].get("source_asset"),
        "source registry canonical source asset",
        root=source_registry_path.parent,
        expected_path=source_asset_path,
    )
    if (
        registry_source_descriptor["sha256"] != source_asset_descriptor["sha256"]
        or source_asset_descriptor["sha256"] != job.get("source_asset_sha256")
        or source_registry_descriptor["sha256"] != job.get("source_registry_sha256")
    ):
        raise contracts.ContractError(
            "UE import preparation source/registry file anchors changed"
        )


def _validate_decision_freeze_receipt(
    preparation: Mapping[str, Any],
    *,
    receipt_path: Path,
    expected_receipt_file_sha256: str,
    source_asset_descriptor: Mapping[str, Any],
    source_registry_descriptor: Mapping[str, Any],
    review_descriptor: Mapping[str, Any],
    decision_descriptor: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, receipt_descriptor = _validate_absolute_descriptor(
        preparation.get("animation_decision_freeze_receipt"),
        "UE import preparation animation decision freeze receipt",
        expected_path=receipt_path,
    )
    if (
        receipt_descriptor["sha256"] != expected_receipt_file_sha256
        or preparation.get("expected_animation_decision_freeze_receipt_file_sha256")
        != expected_receipt_file_sha256
    ):
        raise contracts.ContractError(
            "UE import preparation freeze receipt external anchor changed"
        )
    receipt = _load(receipt_path)
    instruction = receipt.get("user_instruction_binding")
    authority = receipt.get("user_instruction_authority")
    if (
        set(receipt) != DECISION_FREEZE_RECEIPT_FIELDS
        or receipt.get("schema") != DECISION_FREEZE_RECEIPT_SCHEMA
        or receipt.get("status") != "frozen"
        or receipt.get("state_classification") != "research_candidate"
        or receipt.get("formal_dataset_registration_authorized") is not False
        or receipt.get("receipt_sha256")
        != _canonical_hash_without(receipt, "receipt_sha256")
        or receipt.get("receipt_sha256")
        != preparation.get("animation_decision_freeze_receipt_sha256")
        or authority != USER_INSTRUCTION_AUTHORITY
        or preparation.get("user_instruction_authority") != USER_INSTRUCTION_AUTHORITY
        or not isinstance(instruction, Mapping)
        or set(instruction) != {"decision", "review_sha256", "all_six_checks_explicit"}
        or instruction.get("decision") != "approved_for_ue_apartment"
        or instruction.get("review_sha256") != review_descriptor["sha256"]
        or instruction.get("all_six_checks_explicit") is not True
    ):
        raise contracts.ContractError(
            "animation decision freeze receipt authority contract is invalid"
        )
    if (
        receipt.get("source_asset") != source_asset_descriptor
        or receipt.get("source_asset_registry") != source_registry_descriptor
        or receipt.get("animation_review") != review_descriptor
        or receipt.get("expected_source_asset_registry_file_sha256")
        != preparation.get("expected_source_asset_registry_file_sha256")
        or receipt.get("source_asset_registry_validation_mode")
        != preparation.get("source_asset_registry_validation_mode")
        or receipt.get("expected_animation_review_file_sha256")
        != review_descriptor["sha256"]
        or isinstance(receipt.get("authenticated_review_artifact_count"), bool)
        or not isinstance(receipt.get("authenticated_review_artifact_count"), int)
        or receipt.get("authenticated_review_artifact_count")
        != preparation.get("authenticated_review_artifact_count")
        or receipt.get("decision_sha256")
        != preparation.get("animation_decision_sha256")
        or receipt.get("decision_sha256") != decision.get("decision_sha256")
    ):
        raise contracts.ContractError(
            "animation decision freeze receipt artifact lineage changed"
        )
    receipt_decision = _validate_relative_descriptor(
        receipt.get("animation_decision"),
        "freeze receipt animation decision",
        root=receipt_path.parent,
        expected_path=Path(str(decision_descriptor["path"])),
    )
    if receipt_decision["sha256"] != decision_descriptor["sha256"]:
        raise contracts.ContractError(
            "animation decision freeze receipt decision identity changed"
        )
    return receipt_descriptor, receipt


def _validate_preparation_anchor(
    value: Any,
    *,
    expected_preparation_path: Path,
    expected_preparation_file_sha256: str,
    receipt_path: Path,
    expected_receipt_file_sha256: str,
    jobs: Mapping[str, Any],
    jobs_path: Path,
    decision: Mapping[str, Any],
    decision_path: Path,
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    runtime: Path,
) -> dict[str, Any]:
    preparation_path, descriptor = _validate_absolute_descriptor(
        value,
        "UE import preparation",
        extra_fields=frozenset({"manifest_sha256"}),
        expected_path=expected_preparation_path,
    )
    if descriptor["sha256"] != expected_preparation_file_sha256:
        raise contracts.ContractError(
            "UE import result preparation external anchor changed"
        )
    descriptor_manifest_sha256 = _require_sha256(
        descriptor.get("manifest_sha256"),
        "UE import preparation descriptor manifest hash",
    )
    preparation = _load(preparation_path)
    if preparation.get("schema") == LEGACY_PREPARATION_SCHEMA:
        raise contracts.ContractError(
            "legacy UE import preparation v1 is audit-only; regenerate v2 "
            "with an authenticated decision-freeze receipt"
        )
    automatic_checks = preparation.get("automatic_checks")
    canonical_identity = preparation.get("canonical_identity")
    if (
        set(preparation) != PREPARATION_FIELDS
        or preparation.get("schema") != PREPARATION_SCHEMA
        or preparation.get("status") != "ready_for_new_ue_import"
        or preparation.get("state_classification") != "research_candidate"
        or preparation.get("formal_dataset_registration_authorized") is not False
        or preparation.get("manifest_sha256")
        != _canonical_hash_without(preparation, "manifest_sha256")
        or descriptor_manifest_sha256 != preparation.get("manifest_sha256")
        or not isinstance(preparation.get("created_at"), str)
        or not preparation["created_at"]
        or not isinstance(automatic_checks, Mapping)
        or set(automatic_checks) != PREPARATION_AUTOMATIC_CHECK_FIELDS
        or any(
            automatic_checks.get(field) is not True
            for field in PREPARATION_AUTOMATIC_CHECK_FIELDS - {"overall"}
        )
        or automatic_checks.get("overall") != "passed"
        or not isinstance(canonical_identity, Mapping)
        or set(canonical_identity) != CANONICAL_IDENTITY_FIELDS
    ):
        raise contracts.ContractError("UE import preparation contract is invalid")

    for field in (
        "asset_id",
        "legacy_tag",
        "tag",
        "profile_schema_id",
        "sampled_attributes",
        "request_sha256",
    ):
        if canonical_identity.get(field) != job.get(field):
            raise contracts.ContractError(
                f"UE import preparation canonical identity changed at {field}"
            )
    if canonical_identity.get("target_physical_profile") != config.get(
        "target_physical_profile"
    ):
        raise contracts.ContractError(
            "UE import preparation physical profile differs from Apartment config"
        )
    for field in ("profile_sha256",):
        _require_sha256(
            canonical_identity.get(field),
            f"UE import preparation canonical {field}",
        )
    taxonomy = canonical_identity.get("taxonomy")
    if (
        not isinstance(taxonomy, Mapping)
        or set(taxonomy) != {"species", "breed"}
        or taxonomy.get("species") != config.get("species")
        or taxonomy.get("breed") != config.get("breed")
        or not isinstance(canonical_identity.get("fixed_attributes"), Mapping)
    ):
        raise contracts.ContractError(
            "UE import preparation canonical taxonomy/attributes are invalid"
        )

    anchored_jobs = _validate_relative_descriptor(
        preparation.get("ue_import_jobs"),
        "UE import preparation batch",
        root=preparation_path.parent,
        expected_path=jobs_path,
    )
    if anchored_jobs["sha256"] != _sha256(jobs_path) or jobs.get(
        "batch_sha256"
    ) != _canonical_hash_without(jobs, "batch_sha256"):
        raise contracts.ContractError("UE import preparation batch anchor changed")

    _, decision_descriptor = _validate_absolute_descriptor(
        preparation.get("animation_decision"),
        "UE import preparation animation decision",
        expected_path=decision_path,
    )
    review_path = Path(str(decision["review"]["path"])).resolve()
    _, review_descriptor = _validate_absolute_descriptor(
        preparation.get("animation_review"),
        "UE import preparation animation review",
        expected_path=review_path,
    )
    _, runtime_descriptor = _validate_absolute_descriptor(
        preparation.get("reviewed_animated_glb"),
        "UE import preparation reviewed GLB",
        expected_path=runtime,
    )
    source_asset_path, source_asset_descriptor = _validate_absolute_descriptor(
        preparation.get("source_asset"),
        "UE import preparation source asset",
    )
    source_registry_path, source_registry_descriptor = _validate_absolute_descriptor(
        preparation.get("source_asset_registry"),
        "UE import preparation source registry",
    )
    if (
        decision_descriptor["sha256"] != _sha256(decision_path)
        or preparation.get("expected_animation_decision_file_sha256")
        != decision_descriptor["sha256"]
        or preparation.get("animation_decision_sha256") != decision["decision_sha256"]
        or review_descriptor["sha256"] != decision["review_sha256"]
        or preparation.get("animation_review_schema") != FORMAL_GENERATED_REVIEW_SCHEMA
        or runtime_descriptor["sha256"] != job["rigged_glb_sha256"]
        or source_asset_descriptor["sha256"] != job["source_asset_sha256"]
        or source_registry_descriptor["sha256"] != job["source_registry_sha256"]
        or preparation.get("expected_source_asset_registry_file_sha256")
        != source_registry_descriptor["sha256"]
    ):
        raise contracts.ContractError("UE import preparation artifact lineage changed")
    _require_sha256(
        preparation.get("source_asset_registry_sha256"),
        "source registry internal hash",
    )
    if (
        not isinstance(preparation.get("source_asset_registry_validation_mode"), str)
        or not preparation["source_asset_registry_validation_mode"]
        or not isinstance(preparation.get("source_artifact_roots"), Mapping)
        or not preparation["source_artifact_roots"]
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(path, str)
            or not Path(path).is_absolute()
            for name, path in preparation["source_artifact_roots"].items()
        )
        or isinstance(preparation.get("authenticated_source_artifact_count"), bool)
        or not isinstance(preparation.get("authenticated_source_artifact_count"), int)
        or preparation["authenticated_source_artifact_count"] <= 0
        or isinstance(preparation.get("authenticated_review_artifact_count"), bool)
        or not isinstance(preparation.get("authenticated_review_artifact_count"), int)
        or preparation["authenticated_review_artifact_count"] <= 0
        or source_asset_path != Path(str(preparation["source_asset"]["path"])).resolve()
    ):
        raise contracts.ContractError(
            "UE import preparation audit counts/roots changed"
        )
    _validate_source_registry_identity(
        preparation,
        source_asset_path=source_asset_path,
        source_registry_path=source_registry_path,
        source_asset_descriptor=source_asset_descriptor,
        source_registry_descriptor=source_registry_descriptor,
        config=config,
        job=job,
    )
    _validate_decision_freeze_receipt(
        preparation,
        receipt_path=receipt_path,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        source_asset_descriptor=source_asset_descriptor,
        source_registry_descriptor=source_registry_descriptor,
        review_descriptor=review_descriptor,
        decision_descriptor=decision_descriptor,
        decision=decision,
    )
    return preparation


def _authenticate_formal_v2(
    *,
    config: Mapping[str, Any],
    jobs: Mapping[str, Any],
    result: Mapping[str, Any],
    decision: Mapping[str, Any],
    jobs_path: Path,
    result_path: Path,
    decision_path: Path,
    preparation_path: Path,
    expected_preparation_file_sha256: str,
    receipt_path: Path,
    expected_receipt_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del result_path
    job_list = jobs.get("jobs")
    if (
        set(jobs) != BATCH_FIELDS
        or jobs.get("schema") != FORMAL_BATCH_SCHEMA
        or jobs.get("status") != "ready_for_new_ue_import"
        or jobs.get("state_classification") != "research_candidate"
        or jobs.get("formal_dataset_registration_authorized") is not False
        or jobs.get("job_type") != JOB_TYPE
        or isinstance(jobs.get("job_count"), bool)
        or not isinstance(jobs.get("job_count"), int)
        or jobs.get("job_count") != 1
        or not isinstance(job_list, list)
        or len(job_list) != 1
        or jobs.get("batch_sha256") != _canonical_hash_without(jobs, "batch_sha256")
    ):
        raise contracts.ContractError("formal UE import batch v2 contract is invalid")
    job = job_list[0]
    expected_tag = f"pixal_{config['asset_id']}"
    if (
        not isinstance(job, Mapping)
        or set(job) != JOB_FIELDS
        or job.get("job_type") != JOB_TYPE
        or CANONICAL_ID_PATTERN.fullmatch(config["asset_id"]) is None
        or len(config["asset_id"]) > 96
        or job.get("asset_id") != config["asset_id"]
        or job.get("legacy_tag") != config["asset_id"]
        or job.get("tag") != expected_tag
        or len(expected_tag) > 104
        or config["tag"] != expected_tag
        or job.get("profile_schema_id") != config["profile_schema_id"]
        or job.get("sampled_attributes") != config["sampled_attributes"]
        or job.get("expected_actions") != EXPECTED_ACTIONS
        or not isinstance(job.get("rigged_glb"), str)
        or jobs.get("non_destructive_policy")
        != _expected_non_destructive_policy(expected_tag)
    ):
        raise contracts.ContractError("formal UE import job identity/actions changed")
    for field in (
        "rigged_glb_sha256",
        "source_registry_sha256",
        "source_asset_sha256",
        "request_sha256",
        "animation_decision_file_sha256",
        "animation_decision_sha256",
    ):
        _require_sha256(job.get(field), f"formal UE job {field}")
    runtime_raw = Path(str(job.get("rigged_glb", "")))
    if not runtime_raw.is_absolute() or _has_symlink_component(runtime_raw):
        raise contracts.ContractError("formal UE job runtime GLB path is unsafe")
    runtime = _direct_file(runtime_raw, "formal UE job runtime GLB")
    if _sha256(runtime) != job["rigged_glb_sha256"]:
        raise contracts.ContractError("formal UE job runtime GLB hash changed")

    _validate_formal_decision(
        decision,
        decision_path=decision_path,
        config=config,
        job=job,
        runtime=runtime,
    )
    preparation = _validate_preparation_anchor(
        result.get("preparation_manifest"),
        expected_preparation_path=preparation_path,
        expected_preparation_file_sha256=expected_preparation_file_sha256,
        receipt_path=receipt_path,
        expected_receipt_file_sha256=expected_receipt_file_sha256,
        jobs=jobs,
        jobs_path=jobs_path,
        decision=decision,
        decision_path=decision_path,
        config=config,
        job=job,
        runtime=runtime,
    )

    result_list = result.get("results")
    imported = result_list[0] if isinstance(result_list, list) and result_list else None
    if (
        set(result) != RESULT_FIELDS
        or result.get("schema") != FORMAL_RESULT_SCHEMA
        or not isinstance(result.get("generated_at"), str)
        or not result["generated_at"]
        or result.get("state_classification") != "research_candidate"
        or result.get("formal_dataset_registration_authorized") is not False
        or result.get("status") != "passed"
        or isinstance(result.get("passed_count"), bool)
        or not isinstance(result.get("passed_count"), int)
        or result.get("passed_count") != 1
        or not isinstance(result_list, list)
        or len(result_list) != 1
        or result.get("non_destructive_policy") != jobs["non_destructive_policy"]
        or not isinstance(imported, Mapping)
        or set(imported) != RESULT_ITEM_FIELDS
    ):
        raise contracts.ContractError(
            "formal UE import result v2 coverage is incomplete"
        )

    _, input_descriptor = _validate_absolute_descriptor(
        result.get("input_manifest"),
        "UE import result input batch",
        extra_fields=frozenset({"batch_sha256"}),
        expected_path=jobs_path,
    )
    if input_descriptor.get("batch_sha256") != jobs["batch_sha256"]:
        raise contracts.ContractError("UE import result batch self-hash changed")
    expected_job_identity_sha256 = _canonical_sha256(job)
    expected_batch_identity = {
        "schema": FORMAL_BATCH_SCHEMA,
        "job_type": JOB_TYPE,
        "job_count": 1,
        "batch_sha256": jobs["batch_sha256"],
        "asset_ids": [config["asset_id"]],
        "tags": [expected_tag],
        "job_identity_sha256s": [expected_job_identity_sha256],
    }
    if (
        not isinstance(result.get("batch_identity"), Mapping)
        or set(result["batch_identity"]) != BATCH_IDENTITY_FIELDS
        or result["batch_identity"] != expected_batch_identity
    ):
        raise contracts.ContractError("UE import result batch identity changed")

    mesh_dir = f"/Game/MyAssets/Audioset/Meshes/gate_{expected_tag}"
    blueprint = (
        f"/Game/MyAssets/Audioset/Blueprints/gate_{expected_tag}/BP_gate_{expected_tag}"
    )
    idle_animation = f"{mesh_dir}/Idle.Idle"
    expected_walking_animation = f"{mesh_dir}/Walking.Walking"
    assets = imported.get("assets")
    asset_count = imported.get("asset_count")
    source_raw = Path(str(imported.get("source", "")))
    skeletal_mesh = imported.get("skeletal_mesh")
    walking_animation = imported.get("walking_animation")
    if (
        imported.get("job_type") != JOB_TYPE
        or imported.get("asset_id") != config["asset_id"]
        or imported.get("legacy_tag") != config["asset_id"]
        or imported.get("tag") != expected_tag
        or imported.get("job_identity_sha256") != expected_job_identity_sha256
        or not source_raw.is_absolute()
        or _has_symlink_component(source_raw)
        or source_raw.resolve() != runtime
        or imported.get("source_sha256") != job["rigged_glb_sha256"]
        or imported.get("mesh_content_dir") != mesh_dir
        or not isinstance(skeletal_mesh, str)
        or not skeletal_mesh.startswith(f"{mesh_dir}/")
        or not isinstance(walking_animation, str)
        or walking_animation != expected_walking_animation
        or imported.get("blueprint") != blueprint
        or isinstance(asset_count, bool)
        or not isinstance(asset_count, int)
        or asset_count < 3
        or not isinstance(assets, list)
        or len(assets) != asset_count
        or assets != sorted(assets)
        or len(set(assets)) != len(assets)
        or any(
            not isinstance(value, str) or not value.startswith(f"{mesh_dir}/")
            for value in assets
        )
        or skeletal_mesh not in assets
        or skeletal_mesh in {idle_animation, walking_animation}
        or idle_animation not in assets
        or walking_animation not in assets
        or imported.get("actions") != EXPECTED_ACTIONS
        or imported.get("status") != "passed"
    ):
        raise contracts.ContractError("formal UE object readback identity changed")
    if preparation.get("animation_decision_sha256") != decision["decision_sha256"]:
        raise contracts.ContractError("formal UE preparation decision identity changed")
    return dict(job), dict(decision)


def _authenticate_legacy_audit(
    *,
    config: Mapping[str, Any],
    jobs: Mapping[str, Any],
    result: Mapping[str, Any],
    decision: Mapping[str, Any],
    jobs_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    job_list = jobs.get("jobs", [])
    result_list = result.get("results", [])
    if (
        jobs.get("schema") != LEGACY_BATCH_SCHEMA
        or not isinstance(job_list, list)
        or len(job_list) != 1
        or result.get("schema") != LEGACY_RESULT_SCHEMA
        or _has_symlink_component(Path(str(result.get("input_manifest", ""))))
        or Path(str(result.get("input_manifest", ""))).resolve() != jobs_path.resolve()
        or isinstance(result.get("passed_count"), bool)
        or not isinstance(result.get("passed_count"), int)
        or result.get("passed_count") != 1
        or not isinstance(result_list, list)
        or len(result_list) != 1
    ):
        raise contracts.ContractError("legacy UE import audit coverage is incomplete")
    job = job_list[0]
    imported = result_list[0]
    runtime_raw = Path(str(job.get("rigged_glb", "")))
    runtime = runtime_raw.resolve()
    if (
        job.get("legacy_tag") != config["asset_id"]
        or job.get("asset_id") != config["asset_id"]
        or job.get("tag") != config["tag"]
        or job.get("profile_schema_id") != config["profile_schema_id"]
        or job.get("sampled_attributes") != config["sampled_attributes"]
        or set(job.get("expected_actions", [])) != set(EXPECTED_ACTIONS)
        or _has_symlink_component(runtime_raw)
        or runtime.is_symlink()
        or not runtime.is_file()
        or _sha256(runtime) != job.get("rigged_glb_sha256")
        or imported.get("legacy_tag") != config["asset_id"]
        or imported.get("tag") != config["tag"]
        or imported.get("source_sha256") != job.get("rigged_glb_sha256")
        or set(imported.get("actions", [])) != set(EXPECTED_ACTIONS)
        or imported.get("status") != "passed"
    ):
        raise contracts.ContractError("legacy UE identity or action readback changed")
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("asset_id") != config["asset_id"]
        or decision.get("decision") != "approved_for_ue_apartment"
        or decision.get("state_classification") != "research_candidate"
        or decision.get("formal_dataset_registration_authorized") is not False
        or decision.get("decision_sha256")
        != _canonical_hash_without(decision, "decision_sha256")
        or not _descriptor_matches(decision.get("review", {}))
        or decision.get("review_sha256") != decision.get("review", {}).get("sha256")
    ):
        raise contracts.ContractError("legacy user animation decision changed")
    review = _load(Path(str(decision["review"]["path"])))
    outputs = review.get("outputs")
    if (
        review.get("schema") not in LEGACY_GENERATED_REVIEW_SCHEMAS
        or review.get("status") != "research_candidate_pending_human_review"
        or review.get("formal_dataset_registration_authorized") is not False
        or not isinstance(outputs, Mapping)
        or not _descriptor_binds_file(
            outputs.get("animated_glb"),
            runtime,
            str(job["rigged_glb_sha256"]),
        )
    ):
        raise contracts.ContractError(
            "legacy audit requires a hash-bound generated review schema v1-v3"
        )
    return dict(job), dict(decision)


def _authenticate(
    *,
    config_path: Path,
    jobs_path: Path,
    result_path: Path,
    decision_path: Path,
    preparation_path: Path | None = None,
    expected_preparation_file_sha256: str | None = None,
    receipt_path: Path | None = None,
    expected_receipt_file_sha256: str | None = None,
    allow_legacy_audit: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load(config_path)
    jobs = _load(jobs_path)
    result = _load(result_path)
    decision = _load(decision_path)
    required_config = {
        "schema",
        "asset_id",
        "tag",
        "profile_schema_id",
        "species",
        "breed",
        "sampled_attributes",
        "target_physical_profile",
        "actor_scale",
        "walking_forward_yaw_offset_deg",
        "ground_snap_max_abs_correction_cm",
        "audio_lookup",
        "audio_source_height_offset_m",
        "scale_rationale",
        "state_classification",
        "formal_dataset_registration_authorized",
    }
    if set(config) != required_config or config.get("schema") != CONFIG_SCHEMA:
        raise contracts.ContractError(
            "generated animal Apartment config fields changed"
        )
    if (
        config.get("state_classification") != "research_candidate"
        or config.get("formal_dataset_registration_authorized") is not False
        or not str(config.get("tag", "")).startswith("pixal_")
        or not all(
            isinstance(config.get(name), str) and config[name]
            for name in (
                "asset_id",
                "profile_schema_id",
                "species",
                "breed",
                "audio_lookup",
            )
        )
        or not isinstance(config.get("sampled_attributes"), dict)
        or not config["sampled_attributes"]
        or not all(
            isinstance(name, str) and name and isinstance(value, str) and value
            for name, value in config["sampled_attributes"].items()
        )
    ):
        raise contracts.ContractError("generated animal Apartment config is invalid")
    config["target_physical_profile"] = _validate_target_physical_profile(
        config["target_physical_profile"], config["sampled_attributes"]
    )
    actor_scale_raw = config["actor_scale"]
    yaw_raw = config["walking_forward_yaw_offset_deg"]
    ground_limit_raw = config["ground_snap_max_abs_correction_cm"]
    audio_height_raw = config["audio_source_height_offset_m"]
    if (
        not _finite_number(actor_scale_raw)
        or not _finite_number(yaw_raw)
        or not _finite_number(ground_limit_raw)
        or not _finite_number(audio_height_raw)
    ):
        raise contracts.ContractError(
            "scale, cardinal direction, ground snap, and audio height must be finite numbers"
        )
    actor_scale = float(actor_scale_raw)
    yaw = float(yaw_raw)
    ground_limit = float(ground_limit_raw)
    audio_height = float(audio_height_raw)
    if (
        not 0.01 <= actor_scale <= 2.0
        or abs(yaw / 90.0 - round(yaw / 90.0)) > 1e-9
        or not 0.0 < ground_limit <= 200.0
        or not 0.0 <= audio_height <= 5.0
    ):
        raise contracts.ContractError(
            "scale, cardinal direction, ground snap, or audio height is unsafe"
        )

    schema_pair = (jobs.get("schema"), result.get("schema"))
    if schema_pair == (FORMAL_BATCH_SCHEMA, FORMAL_RESULT_SCHEMA):
        if (
            preparation_path is None
            or expected_preparation_file_sha256 is None
            or receipt_path is None
            or expected_receipt_file_sha256 is None
        ):
            raise contracts.ContractError(
                "formal v4/v2 authentication requires external preparation "
                "and freeze-receipt anchors"
            )
        job, decision = _authenticate_formal_v2(
            config=config,
            jobs=jobs,
            result=result,
            decision=decision,
            jobs_path=jobs_path,
            result_path=result_path,
            decision_path=decision_path,
            preparation_path=preparation_path,
            expected_preparation_file_sha256=(expected_preparation_file_sha256),
            receipt_path=receipt_path,
            expected_receipt_file_sha256=expected_receipt_file_sha256,
        )
    elif schema_pair == (LEGACY_BATCH_SCHEMA, LEGACY_RESULT_SCHEMA):
        job, decision = _authenticate_legacy_audit(
            config=config,
            jobs=jobs,
            result=result,
            decision=decision,
            jobs_path=jobs_path,
        )
        if not allow_legacy_audit:
            raise contracts.ContractError(
                "legacy UE v1 and generated review v1-v3 are audit-only; "
                "regenerate an authenticated v4 review and v2 UE import result"
            )
    else:
        raise contracts.ContractError(
            "UE import batch/result schemas are mixed, unsupported, or downgraded"
        )
    return config, job, decision


def _build_pair(
    template: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    walking = copy.deepcopy(template)
    walking.update(
        {
            "description": (
                f"User-approved generated {config['breed']} starts camera-right/rear, "
                "passes camera-left/front, then walks one counter-clockwise table loop."
            ),
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
        }
    )
    source = walking["sources"][0]
    source.update(
        {
            "tag": config["tag"],
            "asset_id": config["asset_id"],
            "asset_class": "animal",
            "species": config["species"],
            "breed": config["breed"],
            "profile_schema_id": config["profile_schema_id"],
            "sampled_attributes": copy.deepcopy(config["sampled_attributes"]),
            "target_physical_profile": copy.deepcopy(config["target_physical_profile"]),
            "kind": "moving",
            "motion": "explicit_camera_pass_table_loop",
            "motion_style": "camera_pass_then_round_table_loop",
            "wanted_anim": "Walking",
            "walking_forward_yaw_offset_deg": config["walking_forward_yaw_offset_deg"],
            "actor_scale": config["actor_scale"],
            "actor_z_lift_cm": 0.0,
            "animation_play_rate": 1.0,
            "ground_snap_to_floor": True,
            "ground_snap_max_abs_correction_cm": config[
                "ground_snap_max_abs_correction_cm"
            ],
            "audio_lookup": config["audio_lookup"],
            "audio_source_height_offset_m": config["audio_source_height_offset_m"],
            "adaptive_repeat_short_calls": True,
            "strict_audio": True,
            "mute_audio": False,
            "controlled_animal_gate": copy.deepcopy(gate),
        }
    )
    bind_pinned_animal_audio_contract(source)
    walking["camera_pass_table_loop_contract"]["animal_scale_rationale"] = {
        "actor_scale": config["actor_scale"],
        "policy": config["scale_rationale"],
        "profile_schema_id": config["profile_schema_id"],
        "sampled_size": config["target_physical_profile"]["selected_value"],
        "target_measurement": config["target_physical_profile"]["measurement"],
        "target_value_cm": config["target_physical_profile"]["target_value_cm"],
    }

    idle = copy.deepcopy(walking)
    trajectory = walking["sources"][0]["trajectory_m"]
    idle_frame = walking["camera_pass_table_loop_contract"]["left_front_nearest_frame"]
    if (
        isinstance(idle_frame, bool)
        or not isinstance(idle_frame, int)
        or not isinstance(trajectory, list)
        or not 0 <= idle_frame < len(trajectory)
    ):
        raise contracts.ContractError(
            "template left-front frame must be an in-range integer"
        )
    idle_position = copy.deepcopy(trajectory[idle_frame])
    idle_source = idle["sources"][0]
    idle_source.update(
        {
            "kind": "stationary",
            "wanted_anim": "Idle",
            "motion": "stationary_left_front",
            "motion_style": "stationary_visible_left_front",
            "start_pos_m": idle_position,
            "end_pos_m": idle_position,
            "trajectory_m": [copy.deepcopy(idle_position) for _ in trajectory],
        }
    )
    idle["description"] = (
        f"User-approved generated {config['breed']} holds Idle at the visible "
        "camera-left/front waypoint without translation."
    )
    idle["trajectory_profile"] = "stationary_camera_left_front_v1"
    idle.pop("rig_direction_check_windows", None)
    idle["stationary_idle_contract"] = {
        "status": "passed",
        "frame_count": len(trajectory),
        "position_m": idle_position,
        "maximum_position_delta_m": 0.0,
        "source_waypoint_frame": idle_frame,
    }
    return {"Walking": walking, "Idle": idle}


def audit_inputs(
    *,
    config_path: Path,
    ue_jobs: Path,
    ue_result: Path,
    animation_decision: Path,
    ue_preparation: Path | None = None,
    expected_ue_preparation_sha256: str | None = None,
    expected_ue_jobs_sha256: str | None = None,
    expected_ue_result_sha256: str | None = None,
    expected_animation_decision_sha256: str | None = None,
    animation_decision_freeze_receipt: Path | None = None,
    expected_animation_decision_freeze_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate either current v4/v2 inputs or read-only legacy v1 inputs.

    This function never publishes specs.  Legacy v1-v3 evidence is retained
    solely so historical runs remain inspectable; ``build_specs`` deliberately
    calls the same authenticator without the legacy allowance.
    """

    formal_anchor_values = (
        ue_preparation,
        expected_ue_preparation_sha256,
        expected_ue_jobs_sha256,
        expected_ue_result_sha256,
        expected_animation_decision_sha256,
        animation_decision_freeze_receipt,
        expected_animation_decision_freeze_receipt_sha256,
    )
    if any(value is not None for value in formal_anchor_values):
        if any(value is None for value in formal_anchor_values):
            raise contracts.ContractError(
                "formal audit external anchors must be provided together"
            )
        ue_preparation = _authenticate_external_file(
            Path(ue_preparation),
            str(expected_ue_preparation_sha256),
            "UE preparation",
        )
        ue_jobs = _authenticate_external_file(
            ue_jobs,
            str(expected_ue_jobs_sha256),
            "UE jobs",
        )
        ue_result = _authenticate_external_file(
            ue_result,
            str(expected_ue_result_sha256),
            "UE result",
        )
        animation_decision = _authenticate_external_file(
            animation_decision,
            str(expected_animation_decision_sha256),
            "animation decision",
        )
        animation_decision_freeze_receipt = _authenticate_external_file(
            Path(animation_decision_freeze_receipt),
            str(expected_animation_decision_freeze_receipt_sha256),
            "animation decision freeze receipt",
        )
    for path in (config_path, ue_jobs, ue_result, animation_decision):
        _artifact(path)
    config, job, decision = _authenticate(
        config_path=config_path.resolve(),
        jobs_path=ue_jobs.resolve(),
        result_path=ue_result.resolve(),
        decision_path=animation_decision.resolve(),
        preparation_path=ue_preparation,
        expected_preparation_file_sha256=expected_ue_preparation_sha256,
        receipt_path=animation_decision_freeze_receipt,
        expected_receipt_file_sha256=(
            expected_animation_decision_freeze_receipt_sha256
        ),
        allow_legacy_audit=True,
    )
    rig_semantic_evidence = _authenticated_rig_semantic_evidence(decision, job)
    jobs = _load(ue_jobs)
    result = _load(ue_result)
    review = _load(Path(str(decision["review"]["path"])))
    formal_publishable = (
        jobs.get("schema") == FORMAL_BATCH_SCHEMA
        and result.get("schema") == FORMAL_RESULT_SCHEMA
        and review.get("schema") == FORMAL_GENERATED_REVIEW_SCHEMA
        and rig_semantic_evidence is not None
    )
    return {
        "status": "authenticated",
        "mode": "formal_v4_v2" if formal_publishable else "legacy_audit_only",
        "asset_id": config["asset_id"],
        "tag": job["tag"],
        "review_schema": review["schema"],
        "ue_batch_schema": jobs["schema"],
        "ue_result_schema": result["schema"],
        "rig_semantic_evidence_status": (
            "authenticated"
            if rig_semantic_evidence is not None
            else "legacy_v1_unavailable"
        ),
        "formal_apartment_publication_authorized": formal_publishable,
    }


def build_specs(
    *,
    config_path: Path,
    ue_jobs: Path,
    ue_result: Path,
    animation_decision: Path,
    ue_preparation: Path,
    expected_ue_preparation_sha256: str,
    expected_ue_jobs_sha256: str,
    expected_ue_result_sha256: str,
    expected_animation_decision_sha256: str,
    animation_decision_freeze_receipt: Path,
    expected_animation_decision_freeze_receipt_sha256: str,
    template: Path,
    output_root: Path,
) -> Path:
    ue_preparation = _authenticate_external_file(
        ue_preparation,
        expected_ue_preparation_sha256,
        "UE preparation",
    )
    ue_jobs = _authenticate_external_file(
        ue_jobs,
        expected_ue_jobs_sha256,
        "UE jobs",
    )
    ue_result = _authenticate_external_file(
        ue_result,
        expected_ue_result_sha256,
        "UE result",
    )
    animation_decision = _authenticate_external_file(
        animation_decision,
        expected_animation_decision_sha256,
        "animation decision",
    )
    animation_decision_freeze_receipt = _authenticate_external_file(
        animation_decision_freeze_receipt,
        expected_animation_decision_freeze_receipt_sha256,
        "animation decision freeze receipt",
    )
    for path in (config_path, template):
        _artifact(path)
    config, job, decision = _authenticate(
        config_path=config_path.resolve(),
        jobs_path=ue_jobs,
        result_path=ue_result,
        decision_path=animation_decision,
        preparation_path=ue_preparation,
        expected_preparation_file_sha256=expected_ue_preparation_sha256,
        receipt_path=animation_decision_freeze_receipt,
        expected_receipt_file_sha256=(
            expected_animation_decision_freeze_receipt_sha256
        ),
    )
    rig_semantic_evidence = _authenticated_rig_semantic_evidence(decision, job)
    preparation = _load(ue_preparation)
    freeze_receipt = _load(animation_decision_freeze_receipt)
    template_payload = _load(template)
    _validate_template_numeric_contract(template_payload)
    output_root = _new_output_path(output_root, "output")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    _new_output_path(output_root, "output")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        decision_artifact = _artifact(animation_decision)
        decision_artifact["decision_sha256"] = decision["decision_sha256"]
        preparation_artifact = _artifact(ue_preparation)
        preparation_artifact["manifest_sha256"] = preparation["manifest_sha256"]
        receipt_artifact = _artifact(animation_decision_freeze_receipt)
        receipt_artifact["receipt_sha256"] = freeze_receipt["receipt_sha256"]
        gate = {
            "schema": "controlled_animal_apartment_gate_v1",
            "status": "approved_for_research_candidate_apartment",
            "asset_id": config["asset_id"],
            "tag": config["tag"],
            "animation_decision": decision_artifact,
            "animation_decision_freeze_receipt": receipt_artifact,
            "ue_import_preparation": preparation_artifact,
            "ue_import_result": _artifact(ue_result),
            "ue_source_sha256": job["rigged_glb_sha256"],
            "user_instruction_authority": copy.deepcopy(USER_INSTRUCTION_AUTHORITY),
            "formal_dataset_registration_authorized": False,
        }
        actions: dict[str, Any] = {}
        for action, spec in _build_pair(
            template_payload, config=config, gate=gate
        ).items():
            motion = action.lower()
            clip_name = f"camera_pass_table_loop_{motion}"
            relative_spec = Path("specs") / config["tag"] / f"{clip_name}.json"
            staged_spec = staging / relative_spec
            contracts.write_json_no_replace(staged_spec, spec)
            published_spec = output_root / relative_spec
            actions[action] = {
                "motion": motion,
                "spec": str(published_spec),
                "spec_evidence": _artifact(staged_spec, published_path=published_spec),
                "output_dir": str(output_root / "clips" / config["tag"] / clip_name),
                "clip_id": f"{config['tag']}_{clip_name}_v1",
            }
        record: dict[str, Any] = {
            "base_avatar_id": config["asset_id"],
            "asset_id": config["asset_id"],
            "tag": config["tag"],
            "profile_schema_id": config["profile_schema_id"],
            "species": config["species"],
            "breed": config["breed"],
            "sampled_attributes": copy.deepcopy(config["sampled_attributes"]),
            "target_physical_profile": copy.deepcopy(config["target_physical_profile"]),
            "source_glb": {
                "path": job["rigged_glb"],
                "sha256": job["rigged_glb_sha256"],
            },
            "actions": actions,
        }
        if rig_semantic_evidence is not None:
            record["rig_semantic_evidence"] = rig_semantic_evidence
        manifest: dict[str, Any] = {
            "schema": OUTPUT_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
            "trajectory_policy": "Walking uses camera right/rear -> left/front -> one table loop; Idle is stationary at left/front",
            "audio_policy": "species-matched short calls are segmented and repeated with silent gaps",
            "avatar_count": 1,
            "clip_count": 2,
            "inputs": {
                "config": _artifact(config_path),
                "ue_import_jobs": _artifact(ue_jobs),
                "ue_import_result": _artifact(ue_result),
                "ue_import_preparation": preparation_artifact,
                "animation_decision": decision_artifact,
                "animation_decision_freeze_receipt": receipt_artifact,
                "template": _artifact(template),
            },
            "records": [record],
        }
        manifest["manifest_sha256"] = contracts.manifest_sha256(manifest)
        contracts.write_json_no_replace(staging / "spec_manifest.json", manifest)
        os.rename(staging, output_root)
        return output_root / "spec_manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ue-jobs", required=True, type=Path)
    parser.add_argument("--ue-result", required=True, type=Path)
    parser.add_argument("--animation-decision", required=True, type=Path)
    parser.add_argument("--ue-preparation", type=Path)
    parser.add_argument("--expected-ue-preparation-sha256")
    parser.add_argument("--expected-ue-jobs-sha256")
    parser.add_argument("--expected-ue-result-sha256")
    parser.add_argument("--expected-animation-decision-sha256")
    parser.add_argument("--animation-decision-freeze-receipt", type=Path)
    parser.add_argument("--expected-animation-decision-freeze-receipt-sha256")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="authenticate inputs without publishing Apartment specs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.audit_only:
            audit = audit_inputs(
                config_path=args.config,
                ue_jobs=args.ue_jobs,
                ue_result=args.ue_result,
                animation_decision=args.animation_decision,
                ue_preparation=args.ue_preparation,
                expected_ue_preparation_sha256=(args.expected_ue_preparation_sha256),
                expected_ue_jobs_sha256=args.expected_ue_jobs_sha256,
                expected_ue_result_sha256=args.expected_ue_result_sha256,
                expected_animation_decision_sha256=(
                    args.expected_animation_decision_sha256
                ),
                animation_decision_freeze_receipt=(
                    args.animation_decision_freeze_receipt
                ),
                expected_animation_decision_freeze_receipt_sha256=(
                    args.expected_animation_decision_freeze_receipt_sha256
                ),
            )
            print(
                "USER_APPROVED_GENERATED_ANIMAL_APARTMENT_AUDIT_OK "
                f"mode={audit['mode']} asset={audit['asset_id']}"
            )
            return 0
        required_formal = {
            "--template": args.template,
            "--output-root": args.output_root,
            "--ue-preparation": args.ue_preparation,
            "--expected-ue-preparation-sha256": (args.expected_ue_preparation_sha256),
            "--expected-ue-jobs-sha256": args.expected_ue_jobs_sha256,
            "--expected-ue-result-sha256": args.expected_ue_result_sha256,
            "--expected-animation-decision-sha256": (
                args.expected_animation_decision_sha256
            ),
            "--animation-decision-freeze-receipt": (
                args.animation_decision_freeze_receipt
            ),
            "--expected-animation-decision-freeze-receipt-sha256": (
                args.expected_animation_decision_freeze_receipt_sha256
            ),
        }
        missing = [name for name, value in required_formal.items() if value is None]
        if missing:
            raise contracts.ContractError(
                "formal publication requires: " + ", ".join(missing)
            )
        manifest = build_specs(
            config_path=args.config,
            ue_jobs=args.ue_jobs,
            ue_result=args.ue_result,
            animation_decision=args.animation_decision,
            ue_preparation=args.ue_preparation,
            expected_ue_preparation_sha256=(args.expected_ue_preparation_sha256),
            expected_ue_jobs_sha256=args.expected_ue_jobs_sha256,
            expected_ue_result_sha256=args.expected_ue_result_sha256,
            expected_animation_decision_sha256=(
                args.expected_animation_decision_sha256
            ),
            animation_decision_freeze_receipt=(args.animation_decision_freeze_receipt),
            expected_animation_decision_freeze_receipt_sha256=(
                args.expected_animation_decision_freeze_receipt_sha256
            ),
            template=args.template,
            output_root=args.output_root,
        )
    except (contracts.ContractError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"USER_APPROVED_GENERATED_ANIMAL_APARTMENT_SPECS_FAILED {error}")
        return 2
    print(f"USER_APPROVED_GENERATED_ANIMAL_APARTMENT_SPECS_OK manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
