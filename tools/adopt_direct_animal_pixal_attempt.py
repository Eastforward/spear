#!/usr/bin/env python3
"""Adopt one authenticated, already-completed direct animal Pixal3D attempt.

This command never starts ImageGen, ISNet, Pixal3D, or any other inference.
It accepts only the two explicitly documented research routes below, verifies
their complete recorded lineage, and publishes a create-only byte copy that
the controlled-animal static-review runner can consume.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_mesh_efficiency
from tools import controlled_animal_isnet_worker as isnet
from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import human_attribute_pixal_contract as staged_pixal
from tools import prepare_controlled_animal_pixal_inputs as pixal_inputs
from tools import rocketbox_native_material_canary as immutable
from tools import route2_human_contract_common as contract_common
from tools import run_controlled_animal_pixal_jobs as pixal_runner


SPEC_SCHEMA = "avengine_direct_animal_pixal_adoption_spec_v1"
BATCH_SCHEMA = "avengine_direct_animal_pixal_adopted_batch_v1"
SOURCE_AUTHORITY_SCHEMA = "avengine_direct_animal_source_authority_v1"
DIRECT_PROFILE_IDENTITY_SCHEMA = (
    "avengine_direct_animal_semantic_profile_identity_v1"
)

IMAGEGEN_ROUTE = "imagegen_pixal3d_animal_research_v1"
IMAGEGEN_SOURCE_KIND = "hash_bound_imagegen_isnet_v1"
IMAGEGEN_REFERENCE_LABEL = "hash-bound ImageGen input"

SHADOW_CLEANUP_ROUTE = "flux2_pixal3d_animal_input_repair_research_v1"
SHADOW_CLEANUP_SOURCE_KIND = "bounded_shadow_cleaned_same_source_same_seed_v1"
SHADOW_CLEANUP_REFERENCE_LABEL = "bounded shadow-cleaned input"

ROUTE_CONTRACTS = {
    IMAGEGEN_ROUTE: {
        "source_kind": IMAGEGEN_SOURCE_KIND,
        "reference_label": IMAGEGEN_REFERENCE_LABEL,
        "evidence_roles": frozenset(
            {
                "generation_receipt",
                "objective_2d_review",
                "isnet_jobs",
                "isnet_status",
                "pixal_jobs",
                "pixal_worker_status",
                "pixal_attempt_manifest",
                "pixal_raw_glb",
            }
        ),
    },
    SHADOW_CLEANUP_ROUTE: {
        "source_kind": SHADOW_CLEANUP_SOURCE_KIND,
        "reference_label": SHADOW_CLEANUP_REFERENCE_LABEL,
        "evidence_roles": frozenset(
            {
                "controlled_request",
                "shadow_cleanup_manifest",
                "pixal_jobs",
                "pixal_worker_status",
                "pixal_attempt_manifest",
                "pixal_raw_glb",
            }
        ),
    },
}

_SPEC_FIELDS = frozenset(
    {
        "schema",
        "route",
        "source_kind",
        "instance_id",
        "state_classification",
        "formal_dataset_registration_authorized",
        "evidence",
        "spec_sha256",
    }
)
_BATCH_FIELDS = frozenset(
    {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "route",
        "source_kind",
        "reference_label",
        "source_spec",
        "models",
        "parameters",
        "job_count",
        "passed_count",
        "failed_count",
        "attempts",
        "copied_artifacts",
        "execution_evidence",
        "automatic_checks",
        "batch_sha256",
    }
)
_ATTEMPT_FIELDS = frozenset(
    {
        "instance_id",
        "execution_job_id",
        "request_sha256",
        "profile_schema_id",
        "sampled_attributes",
        "target_physical_profile",
        "gpu",
        "seed",
        "attempt_ordinal",
        "one_shot_execution",
        "pixal_input",
        "output",
        "attempt_manifest",
        "mesh_readback",
        "timings",
        "status",
        "next_gate",
    }
)
_FILE_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_SOURCE_AUTHORITY_FIELDS = frozenset(
    {
        "schema",
        "state_classification",
        "formal_dataset_registration_authorized",
        "adopted_batch",
        "source_spec",
        "instance_id",
        "profile_schema_id",
        "profile_sha256",
        "request_sha256",
        "taxonomy",
        "fixed_attributes",
        "lineage_group_id",
        "acoustic_profile",
        "authority_sha256",
    }
)
_SOURCE_AUTHORITY_SEMANTICS_FIELDS = frozenset(
    {
        "taxonomy",
        "fixed_attributes",
        "lineage_group_id",
        "acoustic_profile",
    }
)
_MODEL_REVISIONS = {
    "pixal3d": pixal_inputs.PIXAL_MODEL_REVISION,
    "dino": pixal_inputs.DINO_REVISION,
}
_PARAMETERS = {"resolution": 1024, "manual_fov": 0.2, "low_vram": False}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1

# The cleanup receipt freezes this >2^53 seed as an exact decimal string.
# Execution-side JSON must retain the exact Python integer.  Rounded Number
# serializations are not equivalent and are rejected without float coercion.
BRITISH_SEED_DECIMAL_ATTESTATION = "3750817048556488970"
BRITISH_EXECUTION_SEED = 3750817048556488970


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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise contracts.ContractError(f"{label} must be an object")
    return value


def _require_exact_fields(
    value: Any, fields: frozenset[str], label: str
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, label)
    if set(mapping) != fields:
        raise contracts.ContractError(f"{label} fields are invalid")
    return mapping


def _is_seed(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < (1 << 63)
    )


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_canonical_identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or contract_common.CANONICAL_ID_RE.fullmatch(value) is None
    ):
        raise contracts.ContractError(
            f"{label} must be one canonical lower-case identifier segment"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise contracts.ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise contracts.ContractError(f"{label} relative path is invalid")
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or value != posix.as_posix()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise contracts.ContractError(
            f"{label} must be canonical relative POSIX syntax without dot components"
        )
    return Path(*posix.parts)


def _declared_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise contracts.ContractError(f"{label} path is invalid")
    raw = Path(value)
    unresolved = raw if raw.is_absolute() else SPEAR_ROOT / raw
    if unresolved.is_symlink():
        raise contracts.ContractError(f"{label} must not be a leaf symlink")
    return unresolved.resolve()


SPEAR_ROOT = Path(__file__).resolve().parents[1]


def _same_path(left: Any, right: Any, label: str) -> None:
    if _declared_path(left, f"{label} left") != _declared_path(
        right, f"{label} right"
    ):
        raise contracts.ContractError(f"{label} paths differ")


def _authenticate_file_record(
    record: Any,
    label: str,
    *,
    exact: bool = True,
) -> Path:
    mapping = _require_mapping(record, label)
    if (exact and set(mapping) != _FILE_FIELDS) or not _FILE_FIELDS.issubset(mapping):
        raise contracts.ContractError(f"{label} file record is invalid")
    path = _declared_path(mapping.get("path"), label)
    size = mapping.get("size_bytes")
    digest = mapping.get("sha256")
    if (
        not path.is_file()
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or path.stat().st_size != size
        or not isinstance(digest, str)
        or len(digest) != 64
        or _sha256_file(path) != digest
    ):
        raise contracts.ContractError(f"{label} changed")
    return path


def _authenticate_path_hash(path_value: Any, digest: Any, label: str) -> Path:
    path = _declared_path(path_value, label)
    if (
        not path.is_file()
        or not isinstance(digest, str)
        or len(digest) != 64
        or _sha256_file(path) != digest
    ):
        raise contracts.ContractError(f"{label} changed")
    return path


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        relative = path.relative_to(root.resolve())
    except ValueError as error:
        raise contracts.ContractError("copied artifact escaped adoption root") from error
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_record_path(record: Any, root: Path, label: str) -> Path:
    mapping = _require_exact_fields(record, _FILE_FIELDS, label)
    relative = _canonical_relative_path(mapping.get("path"), label)
    unresolved = root / relative
    if unresolved.is_symlink():
        raise contracts.ContractError(f"{label} must not be a leaf symlink")
    path = unresolved.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise contracts.ContractError(f"{label} escaped the adopted batch") from error
    if (
        not path.is_file()
        or path.stat().st_size != mapping.get("size_bytes")
        or _sha256_file(path) != mapping.get("sha256")
    ):
        raise contracts.ContractError(f"{label} changed")
    return path


def _contained_destination(root: Path, relative: Path, label: str) -> Path:
    canonical = _canonical_relative_path(relative.as_posix(), label)
    unresolved = Path(root).absolute() / canonical
    if unresolved.is_symlink():
        raise contracts.ContractError(f"{label} must not be a leaf symlink")
    resolved_root = Path(root).resolve()
    try:
        unresolved.resolve(strict=False).relative_to(resolved_root)
    except ValueError as error:
        raise contracts.ContractError(f"{label} escaped its staging root") from error
    return unresolved


def _validate_staged_glb(
    path: Path,
    *,
    staging: Path,
    input_rgba: Path,
    label: str,
) -> None:
    try:
        staged_pixal.validate_staged_pixal_glb(
            path,
            staging=staging,
            input_rgba=input_rgba,
        )
    except staged_pixal.PixalContractError as error:
        raise contracts.ContractError(f"{label} failed deep PBR validation: {error}") from error


def _load_json_record(record: Any, label: str) -> tuple[Path, Any]:
    path = _authenticate_file_record(record, label)
    return path, contracts.load_json(path)


def _reject_formal_authorization(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "formal_dataset_registration_authorized" and item is not False:
                raise contracts.ContractError(f"{label} contains formal authorization")
            _reject_formal_authorization(item, label)
    elif isinstance(value, list):
        for item in value:
            _reject_formal_authorization(item, label)


def _validate_target_physical_profile(
    target: Any, sampled_attributes: Any
) -> dict[str, Any]:
    target = _require_mapping(target, "target physical profile")
    sampled = _require_mapping(sampled_attributes, "sampled attributes")
    required = {
        "control_attribute",
        "measurement",
        "mode",
        "profile_id",
        "reference_value_cm",
        "selected_value",
        "target_value_cm",
        "tolerance_cm",
    }
    if not required.issubset(target):
        raise contracts.ContractError("target physical profile is incomplete")
    control = target.get("control_attribute")
    if (
        not isinstance(control, str)
        or not control
        or control not in sampled
        or target.get("selected_value") != sampled[control]
        or target.get("mode")
        not in {"relative_to_profile_reference", "absolute_measurement"}
    ):
        raise contracts.ContractError("target physical control binding is invalid")
    for name in ("reference_value_cm", "target_value_cm", "tolerance_cm"):
        if not _is_number(target.get(name)) or float(target[name]) <= 0:
            raise contracts.ContractError(f"target physical {name} is invalid")
    for name in ("measurement", "profile_id"):
        if not isinstance(target.get(name), str) or not target[name]:
            raise contracts.ContractError(f"target physical {name} is invalid")
    return copy.deepcopy(dict(target))


def _validate_dynamic_request_common(
    controlled: Any,
    *,
    schema: str,
    route: str,
    instance_id: str,
    require_request_sha256: bool = True,
) -> Mapping[str, Any]:
    controlled = _require_mapping(controlled, "controlled request")
    if (
        controlled.get("schema") != schema
        or controlled.get("route") != route
        or controlled.get("asset_class") != "animal"
        or controlled.get("instance_id") != instance_id
        or controlled.get("formal_dataset_registration_authorized") is not False
        or not _is_seed(controlled.get("generation_seed"))
    ):
        raise contracts.ContractError("controlled request identity is invalid")
    required_text = ["execution_job_id", "profile_schema_id"]
    if require_request_sha256:
        required_text.append("request_sha256")
    for name in required_text:
        value = controlled.get(name)
        if (
            not isinstance(value, str)
            or not value
            or (name == "request_sha256" and len(value) != 64)
        ):
            raise contracts.ContractError(f"controlled request {name} is invalid")
    rig = _require_mapping(controlled.get("rig_profile"), "rig profile")
    if (
        rig.get("actions") != ["Walking", "Idle"]
        or rig.get("front_axis") != "positive_x"
        or not isinstance(rig.get("profile_id"), str)
        or not isinstance(rig.get("skeleton_family"), str)
    ):
        raise contracts.ContractError("dynamic rig profile is invalid")
    _validate_target_physical_profile(
        controlled.get("target_physical_profile"),
        controlled.get("sampled_attributes"),
    )
    return controlled


def _validate_one_shot(value: Any) -> None:
    try:
        one_shot.validate_stage_record(value, "pixal3d")
    except one_shot.PolicyError as error:
        raise contracts.ContractError(str(error)) from error


def _validate_rgba_and_alpha(
    candidate_path: Path, alpha_path: Path, rgba_path: Path
) -> tuple[int, int, tuple[int, int, int, int]]:
    with (
        Image.open(candidate_path) as candidate_opened,
        Image.open(alpha_path) as alpha_opened,
        Image.open(rgba_path) as rgba_opened,
    ):
        candidate_opened.load()
        alpha_opened.load()
        rgba_opened.load()
        candidate = candidate_opened.convert("RGB")
        if alpha_opened.mode != "L" or rgba_opened.mode != "RGBA":
            raise contracts.ContractError("ISNet alpha/RGBA image modes changed")
        if candidate.size != alpha_opened.size or candidate.size != rgba_opened.size:
            raise contracts.ContractError("ISNet image dimensions differ")
        if rgba_opened.getchannel("A").tobytes() != alpha_opened.tobytes():
            raise contracts.ContractError("ISNet RGBA alpha differs from alpha artifact")
        if rgba_opened.convert("RGB").tobytes() != candidate.tobytes():
            raise contracts.ContractError("ISNet RGBA RGB differs from source candidate")
        extrema = alpha_opened.getextrema()
        thresholded = alpha_opened.point(
            lambda pixel: 255 if pixel >= 128 else 0, mode="L"
        )
        bbox = thresholded.getbbox()
        if bbox is None:
            raise contracts.ContractError("ISNet alpha has no foreground")
        return candidate.width, candidate.height, bbox


def _validate_imagegen_lineage(
    spec: Mapping[str, Any],
    paths: Mapping[str, Path],
    payloads: Mapping[str, Any],
) -> dict[str, Any]:
    instance_id = spec["instance_id"]
    receipt = _require_mapping(payloads["generation_receipt"], "generation receipt")
    if (
        receipt.get("schema")
        != "avengine_builtin_imagegen_reference_generation_receipt_v3"
        or receipt.get("state_classification") != "research_candidate"
        or receipt.get("formal_dataset_registration_authorized") is not False
    ):
        raise contracts.ContractError("ImageGen generation receipt is invalid")
    tool = _require_mapping(receipt.get("tool"), "ImageGen tool receipt")
    if (
        tool.get("mode") != "codex_builtin_image_gen"
        or tool.get("provider_model_revision") is not None
        or tool.get("seed") is not None
        or tool.get("replayability") != "output_hash_bound_only"
        or tool.get("invocation_ordinal") != 0
        or tool.get("invocations_for_method") != 1
        or tool.get("images_for_method") != 1
    ):
        raise contracts.ContractError("unknown or replayable ImageGen provider claim")
    method = _require_mapping(receipt.get("method_revision"), "ImageGen method revision")
    if (
        method.get("this_is_not_a_pixel3d_seed_retry") is not True
        or method.get("pixel3d_seed_retry_allowed") is not False
        or method.get("candidate_ranking_allowed") is not False
    ):
        raise contracts.ContractError("ImageGen method permits retry or ranking")
    candidate_record = _require_mapping(receipt.get("output"), "ImageGen output")
    candidate_path = _authenticate_file_record(
        candidate_record, "ImageGen output", exact=False
    )
    if candidate_record.get("mode") != "RGB":
        raise contracts.ContractError("ImageGen output mode changed")
    reference_input = receipt.get("reference_input")
    reference_input_path = None
    if reference_input is not None:
        reference_input_path = _authenticate_file_record(
            reference_input, "ImageGen reference input", exact=False
        )
        if (
            method.get("supersedes_source_candidate_sha256")
            != reference_input.get("sha256")
        ):
            raise contracts.ContractError("ImageGen edit lineage hash differs")

    review = _require_mapping(payloads["objective_2d_review"], "objective 2D review")
    if (
        review.get("schema") != "avengine_research_animal_image_2d_review_v1"
        or review.get("decision") != "approved_for_dynamic_feasibility_only"
        or review.get("downstream_gate")
        != "one_hash_bound_new_pixel3d_checkpoint_authorized"
    ):
        raise contracts.ContractError("objective 2D review did not authorize checkpoint")
    review_candidate = _require_mapping(review.get("candidate"), "review candidate")
    review_receipt = _require_mapping(
        review.get("generation_receipt"), "review generation receipt"
    )
    _authenticate_file_record(review_candidate, "review candidate")
    _authenticate_file_record(review_receipt, "review generation receipt")
    if (
        review_candidate.get("sha256") != candidate_record.get("sha256")
        or _declared_path(review_candidate.get("path"), "review candidate")
        != candidate_path
        or review_receipt.get("sha256")
        != spec["evidence"]["generation_receipt"]["sha256"]
        or _declared_path(review_receipt.get("path"), "review receipt")
        != paths["generation_receipt"]
    ):
        raise contracts.ContractError("objective review lineage differs from receipt")
    constraints = _require_mapping(review.get("constraints"), "review constraints")
    if (
        constraints.get("pixel3d_invocations_allowed") != 1
        or constraints.get("pixel3d_seed_retry_allowed") is not False
        or constraints.get("pixel3d_candidate_ranking_allowed") is not False
        or constraints.get("formal_dataset_registration_authorized") is not False
    ):
        raise contracts.ContractError("objective review constraints were weakened")

    jobs = _require_mapping(payloads["isnet_jobs"], "ISNet jobs")
    if jobs.get("schema") != isnet.JOBS_SCHEMA:
        raise contracts.ContractError("ISNet jobs schema changed")
    isnet_jobs = jobs.get("jobs")
    if not isinstance(isnet_jobs, list) or len(isnet_jobs) != 1:
        raise contracts.ContractError("ISNet job coverage is invalid")
    isnet_job = _require_exact_fields(
        isnet_jobs[0],
        frozenset(
            {
                "instance_id",
                "candidate_path",
                "candidate_sha256",
                "alpha_path",
                "rgba_path",
            }
        ),
        "ISNet job",
    )
    if (
        isnet_job.get("instance_id") != instance_id
        or isnet_job.get("candidate_sha256") != candidate_record.get("sha256")
    ):
        raise contracts.ContractError("ISNet job identity differs")
    _same_path(isnet_job.get("candidate_path"), str(candidate_path), "ISNet candidate")

    status = _require_mapping(payloads["isnet_status"], "ISNet status")
    status_jobs = status.get("jobs")
    if (
        status.get("schema") != isnet.STATUS_SCHEMA
        or status.get("status") != "passed"
        or status.get("passed_count") != 1
        or status.get("failed_count") != 0
        or not isinstance(status_jobs, list)
        or len(status_jobs) != 1
        or status.get("model")
        != {
            "name": "isnet-general-use",
            "path": str(isnet.MODEL_PATH),
            "sha256": isnet.MODEL_SHA256,
        }
    ):
        raise contracts.ContractError("ISNet status/model contract changed")
    status_job = _require_mapping(status_jobs[0], "ISNet status job")
    if (
        status_job.get("instance_id") != instance_id
        or status_job.get("status") != "passed"
        or status_job.get("alpha_extrema") != [0, 255]
    ):
        raise contracts.ContractError("ISNet status job did not pass")
    alpha_path = _authenticate_path_hash(
        status_job.get("alpha_path"), status_job.get("alpha_sha256"), "ISNet alpha"
    )
    rgba_path = _authenticate_path_hash(
        status_job.get("rgba_path"), status_job.get("rgba_sha256"), "ISNet RGBA"
    )
    _same_path(isnet_job.get("alpha_path"), str(alpha_path), "ISNet alpha")
    _same_path(isnet_job.get("rgba_path"), str(rgba_path), "ISNet RGBA")
    segmentation = _require_mapping(review.get("segmentation"), "review segmentation")
    if (
        segmentation.get("alpha_sha256") != status_job.get("alpha_sha256")
        or segmentation.get("rgba_sha256") != status_job.get("rgba_sha256")
        or segmentation.get("threshold") != 128
    ):
        raise contracts.ContractError("objective review/ISNet hashes differ")
    _same_path(segmentation.get("alpha_path"), str(alpha_path), "review alpha")
    _same_path(segmentation.get("rgba_path"), str(rgba_path), "review RGBA")
    width, height, bbox = _validate_rgba_and_alpha(
        candidate_path, alpha_path, rgba_path
    )
    if (
        status_job.get("foreground_bbox_xyxy") != list(bbox)
        or segmentation.get("foreground_bbox_xyxy")
        != [bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1]
        or candidate_record.get("canvas") != [width, height]
    ):
        raise contracts.ContractError("ISNet/review foreground geometry differs")

    return {
        "candidate_path": candidate_path,
        "alpha_path": alpha_path,
        "pixal_input_path": rgba_path,
        "reference_input_path": reference_input_path,
    }


def _validate_british_seed(value: Any, label: str) -> int:
    if not _is_seed(value) or value != BRITISH_EXECUTION_SEED:
        raise contracts.ContractError(f"{label} differs from frozen British seed")
    return int(value)


def _validate_shadow_cleanup_images(
    candidate_path: Path,
    original_alpha_path: Path,
    original_rgba_path: Path,
    cleaned_alpha_path: Path,
    cleaned_rgba_path: Path,
    removed_mask_path: Path,
) -> None:
    with (
        Image.open(candidate_path) as candidate_opened,
        Image.open(original_alpha_path) as original_alpha_opened,
        Image.open(original_rgba_path) as original_rgba_opened,
        Image.open(cleaned_alpha_path) as cleaned_alpha_opened,
        Image.open(cleaned_rgba_path) as cleaned_rgba_opened,
        Image.open(removed_mask_path) as removed_opened,
    ):
        for image in (
            candidate_opened,
            original_alpha_opened,
            original_rgba_opened,
            cleaned_alpha_opened,
            cleaned_rgba_opened,
            removed_opened,
        ):
            image.load()
        if (
            original_alpha_opened.mode != "L"
            or original_rgba_opened.mode != "RGBA"
            or cleaned_alpha_opened.mode != "L"
            or cleaned_rgba_opened.mode != "RGBA"
            or removed_opened.mode != "L"
            or len(
                {
                    candidate_opened.size,
                    original_alpha_opened.size,
                    original_rgba_opened.size,
                    cleaned_alpha_opened.size,
                    cleaned_rgba_opened.size,
                    removed_opened.size,
                }
            )
            != 1
        ):
            raise contracts.ContractError("shadow-cleanup image contract changed")
        candidate_rgb = candidate_opened.convert("RGB").tobytes()
        original_rgba = original_rgba_opened
        if (
            original_rgba.getchannel("A").tobytes()
            != original_alpha_opened.tobytes()
            or original_rgba.convert("RGB").tobytes() != candidate_rgb
            or cleaned_rgba_opened.getchannel("A").tobytes()
            != cleaned_alpha_opened.tobytes()
        ):
            raise contracts.ContractError("shadow-cleanup source/RGBA channels differ")
        before_alpha = original_alpha_opened.tobytes()
        after_alpha = cleaned_alpha_opened.tobytes()
        removed = removed_opened.tobytes()
        before_rgb = original_rgba.convert("RGB").tobytes()
        after_rgb = cleaned_rgba_opened.convert("RGB").tobytes()
        for index, (before, after, mask) in enumerate(
            zip(before_alpha, after_alpha, removed, strict=True)
        ):
            if (mask and (before == 0 or after != 0)) or (
                not mask and after != before
            ):
                raise contracts.ContractError("shadow cleanup alpha/mask differs")
            if after and before_rgb[index * 3 : index * 3 + 3] != after_rgb[
                index * 3 : index * 3 + 3
            ]:
                raise contracts.ContractError("shadow cleanup altered visible subject RGB")


def _validate_shadow_cleanup_lineage(
    spec: Mapping[str, Any],
    paths: Mapping[str, Path],
    payloads: Mapping[str, Any],
) -> dict[str, Any]:
    instance_id = spec["instance_id"]
    request = _validate_dynamic_request_common(
        payloads["controlled_request"],
        schema="avengine_research_animal_pixal_input_repair_request_v1",
        route=SHADOW_CLEANUP_ROUTE,
        instance_id=instance_id,
        require_request_sha256=False,
    )
    expected_request_fields = frozenset(
        {
            "schema",
            "asset_class",
            "execution_job_id",
            "formal_dataset_registration_authorized",
            "generation_seed",
            "instance_id",
            "method_revision",
            "profile_schema_id",
            "reference",
            "rejected_predecessor",
            "rig_profile",
            "route",
            "sampled_attributes",
            "target_physical_profile",
        }
    )
    _require_exact_fields(request, expected_request_fields, "input-repair request")
    current_seed = _validate_british_seed(
        request.get("generation_seed"), "input-repair request seed"
    )
    method = _require_mapping(request.get("method_revision"), "input-repair method")
    if method != {
        "candidate_ranking_allowed": False,
        "input_repair_kind": "bounded_ground_shadow_alpha_cleanup_v1",
        "pixal_seed_unchanged": True,
        "same_source_candidate": True,
        "seed_retry_allowed": False,
        "this_is_not_a_seed_retry": True,
    }:
        raise contracts.ContractError("input-repair method is not same-source/same-seed")

    reference = _require_exact_fields(
        request.get("reference"),
        frozenset(
            {
                "cleaned_pixal_input",
                "input_repair_manifest",
                "original_pixal_input",
                "source",
            }
        ),
        "input-repair reference",
    )
    candidate_path = _authenticate_file_record(
        reference["source"], "input-repair source"
    )
    original_rgba_path = _authenticate_file_record(
        reference["original_pixal_input"], "original Pixal input"
    )
    cleaned_rgba_path = _authenticate_file_record(
        reference["cleaned_pixal_input"], "cleaned Pixal input"
    )
    cleanup_record_path = _authenticate_file_record(
        reference["input_repair_manifest"], "shadow cleanup manifest"
    )
    if (
        cleanup_record_path != paths["shadow_cleanup_manifest"]
        or reference["input_repair_manifest"]["sha256"]
        != spec["evidence"]["shadow_cleanup_manifest"]["sha256"]
    ):
        raise contracts.ContractError("request/cleanup manifest evidence differs")

    cleanup = _require_mapping(
        payloads["shadow_cleanup_manifest"], "shadow cleanup manifest"
    )
    if (
        cleanup.get("schema")
        != "avengine_generated_animal_opaque_ground_shadow_cleanup_v2"
        or cleanup.get("state_classification")
        != "technical_spike_input_repair_only"
        or cleanup.get("formal_dataset_registration_authorized") is not False
        or cleanup.get("execution")
        != {
            "class": "deterministic_cpu_only",
            "connectivity": 8,
            "coordinate_convention": "xyxy_half_open",
            "gpu_or_pixal_started": False,
        }
    ):
        raise contracts.ContractError("shadow cleanup execution contract changed")
    decision = _require_mapping(cleanup.get("decision"), "shadow cleanup decision")
    required_decisions = {
        "abdomen_and_leg_evidence_preserved": True,
        "dynamic_asset_admission_authorized": False,
        "opaque_ground_shadow_cleanup_passed": True,
        "paw_bottom_evidence_preserved": True,
        "pixal_seed_unchanged_for_next_run": True,
        "source_candidate_unchanged": True,
    }
    if any(decision.get(key) is not value for key, value in required_decisions.items()):
        raise contracts.ContractError("shadow cleanup decision was weakened")
    cleanup_source = _require_mapping(cleanup.get("source"), "shadow cleanup source")
    attestation = cleanup_source.get(
        "pixal_seed_decimal_frozen_for_controlled_rerun"
    )
    if attestation != BRITISH_SEED_DECIMAL_ATTESTATION:
        raise contracts.ContractError("British cleanup decimal seed attestation changed")
    cleanup_candidate = _authenticate_file_record(
        cleanup_source.get("candidate"), "cleanup candidate", exact=False
    )
    original_alpha_path = _authenticate_file_record(
        cleanup_source.get("input_alpha"), "cleanup input alpha", exact=False
    )
    cleanup_original_rgba = _authenticate_file_record(
        cleanup_source.get("input_rgba"), "cleanup input RGBA", exact=False
    )
    if cleanup_candidate != candidate_path or cleanup_original_rgba != original_rgba_path:
        raise contracts.ContractError("cleanup source lineage differs from request")
    outputs = _require_mapping(cleanup.get("outputs"), "shadow cleanup outputs")
    if set(outputs) != {"cleaned_alpha", "cleaned_rgba", "removed_shadow_mask"}:
        raise contracts.ContractError("shadow cleanup output roles changed")
    cleaned_alpha_path = _authenticate_file_record(
        outputs["cleaned_alpha"], "cleaned alpha", exact=False
    )
    cleanup_cleaned_rgba = _authenticate_file_record(
        outputs["cleaned_rgba"], "cleaned RGBA", exact=False
    )
    removed_mask_path = _authenticate_file_record(
        outputs["removed_shadow_mask"], "removed shadow mask", exact=False
    )
    if cleanup_cleaned_rgba != cleaned_rgba_path:
        raise contracts.ContractError("cleanup/request cleaned RGBA differs")
    _validate_shadow_cleanup_images(
        candidate_path,
        original_alpha_path,
        original_rgba_path,
        cleaned_alpha_path,
        cleaned_rgba_path,
        removed_mask_path,
    )
    measurements = _require_mapping(
        cleanup.get("measurements"), "shadow cleanup measurements"
    )
    paw_evidence = measurements.get("paw_bottom_evidence")
    body_evidence = measurements.get("protected_body_evidence")
    if (
        measurements.get("selected_shadow_pixel_count") != 1223
        or measurements.get("removed_alpha_pixel_count") != 1223
        or not isinstance(paw_evidence, list)
        or len(paw_evidence) != 4
        or not isinstance(body_evidence, list)
        or len(body_evidence) != 2
        or any(item.get("byte_exact_preserved") is not True for item in paw_evidence)
        or any(item.get("byte_exact_preserved") is not True for item in body_evidence)
    ):
        raise contracts.ContractError("shadow cleanup protected evidence changed")

    rejected = _require_exact_fields(
        request.get("rejected_predecessor"),
        frozenset({"bounded_repair_rejection", "pixal_manifest", "pixal_raw"}),
        "rejected predecessor",
    )
    rejection_path = _authenticate_file_record(
        rejected["bounded_repair_rejection"], "bounded repair rejection"
    )
    predecessor_manifest_path = _authenticate_file_record(
        rejected["pixal_manifest"], "predecessor Pixal manifest"
    )
    predecessor_raw_path = _authenticate_file_record(
        rejected["pixal_raw"], "predecessor Pixal raw"
    )
    predecessor = _require_mapping(
        contracts.load_json(predecessor_manifest_path), "predecessor manifest"
    )
    predecessor_seed = _validate_british_seed(
        predecessor.get("parameters", {}).get("seed"), "predecessor parameter seed"
    )
    _validate_british_seed(
        predecessor.get("controlled_request", {}).get("generation_seed"),
        "predecessor request seed",
    )
    if (
        predecessor.get("backend") != "pixal3d"
        or predecessor.get("output", {}).get("sha256")
        != rejected["pixal_raw"]["sha256"]
        or predecessor.get("output", {}).get("bytes")
        != rejected["pixal_raw"]["size_bytes"]
        or predecessor_seed
        != predecessor.get("controlled_request", {}).get("generation_seed")
    ):
        raise contracts.ContractError("predecessor Pixal lineage is invalid")
    _same_path(
        predecessor.get("output", {}).get("path"),
        str(predecessor_raw_path),
        "predecessor raw",
    )
    _validate_one_shot(predecessor.get("one_shot_execution"))
    rejection = _require_mapping(
        contracts.load_json(rejection_path), "bounded repair rejection"
    )
    rejection_lineage = _require_mapping(
        rejection.get("lineage"), "bounded repair rejection lineage"
    )
    if (
        rejection.get("schema") != "avengine_pixal_same_mesh_mirrored_limb_repair_v1"
        or rejection.get("formal_dataset_registration_authorized") is not False
        or rejection.get("decision", {}).get("status")
        != "rejected_bounded_local_repair_preflight"
        or rejection.get("decision", {}).get("cat_semantic_retarget_authorized")
        is not False
        or rejection_lineage.get("pixal_manifest", {}).get("sha256")
        != rejected["pixal_manifest"]["sha256"]
        or rejection_lineage.get("pixal_source", {}).get("sha256")
        != rejected["pixal_raw"]["sha256"]
        or rejection_lineage.get("approved_reference", {}).get("sha256")
        != reference["source"]["sha256"]
    ):
        raise contracts.ContractError("bounded repair rejection lineage differs")
    for name in (
        "approved_reference",
        "owner_review",
        "pixal_manifest",
        "pixal_source",
        "static_decision",
    ):
        _authenticate_file_record(
            rejection_lineage.get(name), f"rejection lineage {name}"
        )

    return {
        "candidate_path": candidate_path,
        "original_alpha_path": original_alpha_path,
        "original_rgba_path": original_rgba_path,
        "cleaned_alpha_path": cleaned_alpha_path,
        "pixal_input_path": cleaned_rgba_path,
        "removed_mask_path": removed_mask_path,
        "predecessor_manifest_path": predecessor_manifest_path,
        "predecessor_raw_path": predecessor_raw_path,
        "rejection_path": rejection_path,
        "current_seed": current_seed,
    }


def _validate_pixal_attempt(
    spec: Mapping[str, Any],
    paths: Mapping[str, Path],
    payloads: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    route = spec["route"]
    instance_id = spec["instance_id"]
    raw_jobs = payloads["pixal_jobs"]
    if not isinstance(raw_jobs, list) or len(raw_jobs) != 1:
        raise contracts.ContractError("direct Pixal jobs must contain exactly one job")
    job = _require_mapping(raw_jobs[0], "direct Pixal job")
    expected_job_fields = {
        "legacy_tag",
        "candidate_tag",
        "seed",
        "attempt_ordinal",
        "one_shot_execution",
        "reference",
        "output",
        "manifest",
        "public_output",
        "public_manifest",
        "controlled_request",
    }
    if route == IMAGEGEN_ROUTE:
        expected_job_fields.add("rig_mode")
    if set(job) != expected_job_fields:
        raise contracts.ContractError("direct Pixal job fields changed")
    controlled_schema = (
        "avengine_research_animal_pixal_request_v1"
        if route == IMAGEGEN_ROUTE
        else "avengine_research_animal_pixal_input_repair_request_v1"
    )
    controlled = _validate_dynamic_request_common(
        job.get("controlled_request"),
        schema=controlled_schema,
        route=route,
        instance_id=instance_id,
    )
    if route == IMAGEGEN_ROUTE:
        expected_request_fields = frozenset(
            {
                "schema",
                "asset_class",
                "execution_job_id",
                "generation_seed",
                "instance_id",
                "profile_schema_id",
                "method_freeze_sha256",
                "request_sha256",
                "rig_profile",
                "route",
                "sampled_attributes",
                "target_physical_profile",
                "formal_dataset_registration_authorized",
            }
        )
        _require_exact_fields(
            controlled, expected_request_fields, "ImageGen Pixal request"
        )
        if (
            controlled.get("method_freeze_sha256")
            != spec["evidence"]["generation_receipt"]["sha256"]
            or controlled.get("request_sha256")
            != _hash_without(controlled, "request_sha256")
            or job.get("rig_mode") != "animated_transfer"
        ):
            raise contracts.ContractError("ImageGen Pixal request hash/rig mode differs")
    else:
        source_request = payloads["controlled_request"]
        embedded_base = {
            name: copy.deepcopy(value)
            for name, value in controlled.items()
            if name not in {"request_record", "request_sha256"}
        }
        request_record = _require_exact_fields(
            controlled.get("request_record"), _FILE_FIELDS, "request record"
        )
        if (
            embedded_base != source_request
            or request_record.get("sha256")
            != spec["evidence"]["controlled_request"]["sha256"]
            or request_record.get("size_bytes")
            != spec["evidence"]["controlled_request"]["size_bytes"]
            or controlled.get("request_sha256") != request_record.get("sha256")
        ):
            raise contracts.ContractError("input-repair request record differs")
        _same_path(
            request_record.get("path"),
            spec["evidence"]["controlled_request"]["path"],
            "input-repair request record",
        )

    seed = job.get("seed")
    if (
        not _is_seed(seed)
        or seed != controlled.get("generation_seed")
        or job.get("attempt_ordinal") != 0
        or job.get("legacy_tag") != instance_id
    ):
        raise contracts.ContractError("direct Pixal job seed/attempt identity changed")
    if route == SHADOW_CLEANUP_ROUTE:
        _validate_british_seed(seed, "direct Pixal job seed")
    _validate_one_shot(job.get("one_shot_execution"))
    reference = _require_mapping(job.get("reference"), "direct Pixal reference")
    pixal_input = _authenticate_file_record(
        reference.get("pixal_input"), "direct Pixal input"
    )
    if pixal_input != lineage["pixal_input_path"]:
        raise contracts.ContractError("direct Pixal input differs from source lineage")
    source = _authenticate_file_record(reference.get("source"), "direct Pixal source")
    if source != lineage["candidate_path"]:
        raise contracts.ContractError("direct Pixal source differs from source lineage")
    if route == IMAGEGEN_ROUTE:
        if (
            set(reference) != {"source", "pixal_input", "normalization"}
            or reference.get("normalization") != "pinned_isnet_general_use_alpha_v1"
        ):
            raise contracts.ContractError("ImageGen Pixal normalization changed")
    else:
        repair = _require_mapping(reference.get("input_repair"), "input repair binding")
        if (
            set(reference)
            != {"source", "pixal_input", "normalization", "input_repair"}
            or reference.get("normalization")
            != "pinned_isnet_general_use_alpha_plus_bounded_shadow_cleanup_v1"
            or repair
            != {
                "manifest_path": str(paths["shadow_cleanup_manifest"]),
                "manifest_sha256": spec["evidence"]["shadow_cleanup_manifest"][
                    "sha256"
                ],
                "manifest_size_bytes": spec["evidence"]["shadow_cleanup_manifest"][
                    "size_bytes"
                ],
                "method": "bounded_ground_shadow_alpha_cleanup_v1",
            }
        ):
            # Path aliases are allowed, so compare the path separately.
            if (
                set(repair)
                != {
                    "manifest_path",
                    "manifest_sha256",
                    "manifest_size_bytes",
                    "method",
                }
                or repair.get("manifest_sha256")
                != spec["evidence"]["shadow_cleanup_manifest"]["sha256"]
                or repair.get("manifest_size_bytes")
                != spec["evidence"]["shadow_cleanup_manifest"]["size_bytes"]
                or repair.get("method")
                != "bounded_ground_shadow_alpha_cleanup_v1"
            ):
                raise contracts.ContractError("input-repair Pixal binding changed")
            _same_path(
                repair.get("manifest_path"),
                str(paths["shadow_cleanup_manifest"]),
                "input-repair manifest",
            )

    for name, expected in (
        ("output", paths["pixal_raw_glb"]),
        ("public_output", paths["pixal_raw_glb"]),
        ("manifest", paths["pixal_attempt_manifest"]),
        ("public_manifest", paths["pixal_attempt_manifest"]),
    ):
        _same_path(job.get(name), str(expected), f"Pixal job {name}")

    worker = _require_exact_fields(
        payloads["pixal_worker_status"],
        frozenset(
            {
                "failed_count",
                "finished_at",
                "gpu",
                "jobs",
                "low_vram",
                "model_load_seconds",
                "passed_count",
                "scheduling_mode",
                "schema",
                "started_at",
            }
        ),
        "Pixal worker status",
    )
    worker_jobs = worker.get("jobs")
    if (
        worker.get("schema") != pixal_runner.WORKER_SCHEMA
        or worker.get("scheduling_mode") != "fixed_partition_v1"
        or worker.get("failed_count") != 0
        or worker.get("passed_count") != 1
        or worker.get("low_vram") is not False
        or not isinstance(worker.get("gpu"), int)
        or isinstance(worker.get("gpu"), bool)
        or worker["gpu"] < 0
        or not isinstance(worker_jobs, list)
        or len(worker_jobs) != 1
    ):
        raise contracts.ContractError("Pixal worker was not one fixed partition")
    worker_job = _require_exact_fields(
        worker_jobs[0],
        frozenset(
            {
                "attempt_ordinal",
                "candidate_tag",
                "finished_at",
                "legacy_tag",
                "manifest",
                "output",
                "output_sha256",
                "seed",
                "started_at",
                "status",
                "wall_seconds",
            }
        ),
        "Pixal worker job",
    )
    if (
        worker_job.get("legacy_tag") != instance_id
        or worker_job.get("candidate_tag") != job.get("candidate_tag")
        or worker_job.get("seed") != seed
        or worker_job.get("attempt_ordinal") != 0
        or worker_job.get("status") != "passed"
        or worker_job.get("output_sha256")
        != spec["evidence"]["pixal_raw_glb"]["sha256"]
    ):
        raise contracts.ContractError("Pixal worker/job evidence differs")
    _same_path(worker_job.get("output"), str(paths["pixal_raw_glb"]), "worker output")
    _same_path(
        worker_job.get("manifest"),
        str(paths["pixal_attempt_manifest"]),
        "worker manifest",
    )

    manifest = _require_exact_fields(
        payloads["pixal_attempt_manifest"],
        frozenset(
            {
                "backend",
                "controlled_request",
                "dino",
                "input",
                "model",
                "one_shot_execution",
                "output",
                "parameters",
                "timings",
            }
        ),
        "Pixal attempt manifest",
    )
    if (
        manifest.get("backend") != "pixal3d"
        or manifest.get("controlled_request") != controlled
        or manifest.get("model", {}).get("revision")
        != pixal_inputs.PIXAL_MODEL_REVISION
        or manifest.get("dino", {}).get("revision") != pixal_inputs.DINO_REVISION
        or manifest.get("parameters") != {**_PARAMETERS, "seed": seed}
        or manifest.get("one_shot_execution") != job.get("one_shot_execution")
        or manifest.get("input", {}).get("mode") != "RGBA"
        or manifest.get("input", {}).get("alpha_min") != 0
        or manifest.get("input", {}).get("alpha_max") != 255
        or manifest.get("input", {}).get("sha256")
        != reference["pixal_input"]["sha256"]
        or manifest.get("output", {}).get("sha256")
        != spec["evidence"]["pixal_raw_glb"]["sha256"]
        or manifest.get("output", {}).get("bytes")
        != spec["evidence"]["pixal_raw_glb"]["size_bytes"]
    ):
        raise contracts.ContractError("Pixal attempt manifest contract changed")
    _same_path(manifest["input"].get("path"), str(pixal_input), "manifest input")
    _same_path(
        manifest["output"].get("path"), str(paths["pixal_raw_glb"]), "manifest output"
    )
    timings = _require_mapping(manifest.get("timings"), "Pixal timings")
    if (
        timings.get("model_reused") is not True
        or not _is_number(timings.get("persistent_worker_model_load_seconds"))
        or not _is_number(timings.get("inference_and_export_seconds"))
        or float(timings["inference_and_export_seconds"]) <= 0
    ):
        raise contracts.ContractError("Pixal timing evidence changed")

    _validate_staged_glb(
        paths["pixal_raw_glb"],
        staging=paths["pixal_raw_glb"].parent,
        input_rgba=lineage["pixal_input_path"],
        label="source Pixal GLB",
    )
    mesh = audit_mesh_efficiency.mesh_stats(paths["pixal_raw_glb"])
    if (
        not isinstance(mesh, dict)
        or mesh.get("exists") is not True
        or mesh.get("bytes") != spec["evidence"]["pixal_raw_glb"]["size_bytes"]
        or any(
            not isinstance(mesh.get(name), int) or mesh[name] <= 0
            for name in ("vertices", "triangles", "meshes", "primitives", "materials")
        )
        or mesh.get("textures", 0) < 2
        or mesh.get("images", 0) < 2
        or mesh.get("skins") != 0
        or mesh.get("animations") != 0
    ):
        raise contracts.ContractError("Pixal GLB/PBR readback is incomplete")
    mesh_readback = {
        name: value for name, value in mesh.items() if name not in {"path", "exists"}
    }
    return {
        "job": copy.deepcopy(dict(job)),
        "controlled": copy.deepcopy(dict(controlled)),
        "seed": seed,
        "gpu": worker["gpu"],
        "pixal_input_path": pixal_input,
        "mesh_readback": mesh_readback,
        "timings": copy.deepcopy(dict(timings)),
    }


def load_adoption_spec(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    unresolved = Path(path).absolute()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(f"adoption spec is missing: {unresolved}")
    path = unresolved.resolve()
    spec = _require_exact_fields(
        contracts.load_json(path), _SPEC_FIELDS, "direct adoption spec"
    )
    _require_canonical_identifier(spec.get("instance_id"), "adoption instance_id")
    route = spec.get("route")
    contract = ROUTE_CONTRACTS.get(route)
    if (
        spec.get("schema") != SPEC_SCHEMA
        or contract is None
        or spec.get("source_kind") != contract["source_kind"]
        or spec.get("state_classification") != "research_candidate"
        or spec.get("formal_dataset_registration_authorized") is not False
        or spec.get("spec_sha256") != _hash_without(spec, "spec_sha256")
    ):
        raise contracts.ContractError("direct adoption spec contract/hash is invalid")
    evidence = _require_mapping(spec.get("evidence"), "adoption evidence")
    if set(evidence) != contract["evidence_roles"]:
        raise contracts.ContractError("adoption evidence roles are invalid")
    paths: dict[str, Path] = {}
    payloads: dict[str, Any] = {}
    for role in sorted(evidence):
        record = _require_exact_fields(
            evidence[role], _FILE_FIELDS, f"adoption evidence {role}"
        )
        if not isinstance(record.get("path"), str) or not Path(
            record["path"]
        ).is_absolute():
            raise contracts.ContractError(
                f"adoption evidence {role} must use an absolute source path"
            )
        evidence_path = _authenticate_file_record(
            record, f"adoption evidence {role}"
        )
        paths[role] = evidence_path
        if role != "pixal_raw_glb":
            payloads[role] = contracts.load_json(evidence_path)
    _reject_formal_authorization(spec, "adoption spec")
    _reject_formal_authorization(payloads, "adoption evidence")
    if route == IMAGEGEN_ROUTE:
        lineage = _validate_imagegen_lineage(spec, paths, payloads)
    else:
        lineage = _validate_shadow_cleanup_lineage(spec, paths, payloads)
    attempt = _validate_pixal_attempt(spec, paths, payloads, lineage)
    bundle_sources: dict[str, Path] = dict(paths)
    if route == IMAGEGEN_ROUTE:
        bundle_sources.update(
            {
                "source_candidate": lineage["candidate_path"],
                "isnet_alpha": lineage["alpha_path"],
                "pixal_input_rgba": lineage["pixal_input_path"],
            }
        )
        if lineage["reference_input_path"] is not None:
            bundle_sources["imagegen_reference_input"] = lineage[
                "reference_input_path"
            ]
    else:
        bundle_sources.update(
            {
                "source_candidate": lineage["candidate_path"],
                "original_input_alpha": lineage["original_alpha_path"],
                "original_pixal_input": lineage["original_rgba_path"],
                "shadow_cleaned_alpha": lineage["cleaned_alpha_path"],
                "pixal_input_rgba": lineage["pixal_input_path"],
                "removed_shadow_mask": lineage["removed_mask_path"],
                "predecessor_pixal_manifest": lineage["predecessor_manifest_path"],
                "predecessor_pixal_raw": lineage["predecessor_raw_path"],
                "bounded_repair_rejection": lineage["rejection_path"],
            }
        )
    context = {
        **attempt,
        "bundle_sources": bundle_sources,
    }
    return path, copy.deepcopy(dict(spec)), context


def _copy_no_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one path without replacing a concurrent peer."""

    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise contracts.ContractError(
            "atomic direct-adoption publication requires Linux renameat2"
        )
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            number, "refusing to replace direct-adoption output", destination
        )
    raise OSError(number, os.strerror(number), destination)


def _bundle_destination(role: str, source: Path, instance_id: str) -> Path:
    if role == "pixal_raw_glb":
        return Path(instance_id) / "pixal_raw_1024.glb"
    if role == "pixal_attempt_manifest":
        return Path(instance_id) / "pixal_raw_1024.manifest.json"
    if role == "pixal_input_rgba":
        return Path(instance_id) / "pixal_input_rgba.png"
    suffix = "".join(source.suffixes) or ".bin"
    return Path("evidence") / f"{role}{suffix}"


def adopt_attempt(spec_path: Path, output_root: Path) -> Path:
    spec_path, spec, context = load_adoption_spec(spec_path)
    instance_id = _require_canonical_identifier(
        spec.get("instance_id"), "adoption instance_id"
    )
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        copied_spec = _contained_destination(
            staging,
            Path("evidence") / "adoption_spec.json",
            "copied adoption spec destination",
        )
        _copy_no_replace(spec_path, copied_spec)
        copied_artifacts: dict[str, dict[str, Any]] = {}
        destinations: set[Path] = set()
        for role, source in sorted(context["bundle_sources"].items()):
            relative = _bundle_destination(role, source, instance_id)
            if relative in destinations:
                raise contracts.ContractError("adoption bundle destination collision")
            destinations.add(relative)
            destination = _contained_destination(
                staging, relative, f"copied artifact {role} destination"
            )
            _copy_no_replace(source, destination)
            copied_artifacts[role] = _relative_record(destination, staging)
        controlled = context["controlled"]
        attempt = {
            "instance_id": instance_id,
            "execution_job_id": controlled["execution_job_id"],
            "request_sha256": controlled["request_sha256"],
            "profile_schema_id": controlled["profile_schema_id"],
            "sampled_attributes": controlled["sampled_attributes"],
            "target_physical_profile": controlled["target_physical_profile"],
            "gpu": context["gpu"],
            "seed": context["seed"],
            "attempt_ordinal": 0,
            "one_shot_execution": context["job"]["one_shot_execution"],
            "pixal_input": copied_artifacts["pixal_input_rgba"],
            "output": copied_artifacts["pixal_raw_glb"],
            "attempt_manifest": copied_artifacts["pixal_attempt_manifest"],
            "mesh_readback": context["mesh_readback"],
            "timings": context["timings"],
            "status": "passed_generation_and_glb_readback",
            "next_gate": "static_visual_qa",
        }
        contract = ROUTE_CONTRACTS[spec["route"]]
        batch: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "status": "passed_generation_and_glb_readback",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "route": spec["route"],
            "source_kind": spec["source_kind"],
            "reference_label": contract["reference_label"],
            "source_spec": {
                **_relative_record(copied_spec, staging),
                "spec_sha256": spec["spec_sha256"],
            },
            "models": copy.deepcopy(_MODEL_REVISIONS),
            "parameters": copy.deepcopy(_PARAMETERS),
            "job_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "attempts": [attempt],
            "copied_artifacts": copied_artifacts,
            "execution_evidence": {
                "mode": "adopted_fixed_partition_research_v1",
                "scheduling_mode": "fixed_partition_v1",
                "recorded_attempt_count": 1,
                "hidden_attempt_exclusion_proven": False,
                "inference_performed_by_adopter": False,
                "source_worker_gpu": context["gpu"],
            },
            "automatic_checks": {
                "source_spec_hash_reauthenticated": True,
                "route_specific_lineage_reauthenticated": True,
                "one_shot_and_no_retry_reauthenticated": True,
                "fixed_partition_worker_reauthenticated": True,
                "glb_and_pbr_readback_reauthenticated": True,
                "all_published_artifacts_are_byte_copies": True,
                "formal_registration_remains_unauthorized": True,
                "overall": "passed",
            },
        }
        batch["batch_sha256"] = _hash_without(batch, "batch_sha256")
        batch_path = _contained_destination(
            staging, Path("pixal_batch_manifest.json"), "adopted batch destination"
        )
        contracts.write_json_no_replace(batch_path, batch)
        load_adopted_batch(batch_path)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("adoption output appeared concurrently")
        _rename_no_replace(staging, output_root)
        return output_root / "pixal_batch_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def load_adopted_batch(path: Path) -> tuple[Path, dict[str, Any]]:
    unresolved = Path(path).absolute()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(f"adopted Pixal batch is missing: {unresolved}")
    path = unresolved.resolve()
    root = path.parent
    batch = _require_exact_fields(
        contracts.load_json(path), _BATCH_FIELDS, "adopted Pixal batch"
    )
    route = batch.get("route")
    contract = ROUTE_CONTRACTS.get(route)
    attempts = batch.get("attempts")
    if (
        batch.get("schema") != BATCH_SCHEMA
        or contract is None
        or batch.get("source_kind") != contract["source_kind"]
        or batch.get("reference_label") != contract["reference_label"]
        or batch.get("status") != "passed_generation_and_glb_readback"
        or batch.get("state_classification") != "research_candidate"
        or batch.get("formal_dataset_registration_authorized") is not False
        or batch.get("models") != _MODEL_REVISIONS
        or batch.get("parameters") != _PARAMETERS
        or batch.get("job_count") != 1
        or batch.get("passed_count") != 1
        or batch.get("failed_count") != 0
        or not isinstance(attempts, list)
        or len(attempts) != 1
        or batch.get("batch_sha256") != _hash_without(batch, "batch_sha256")
    ):
        raise contracts.ContractError("adopted Pixal batch contract/hash is invalid")
    attempt_value = _require_mapping(attempts[0], "adopted attempt")
    if batch.get("execution_evidence") != {
        "mode": "adopted_fixed_partition_research_v1",
        "scheduling_mode": "fixed_partition_v1",
        "recorded_attempt_count": 1,
        "hidden_attempt_exclusion_proven": False,
        "inference_performed_by_adopter": False,
        "source_worker_gpu": attempt_value.get("gpu"),
    }:
        raise contracts.ContractError("adopted execution evidence changed")
    if batch.get("automatic_checks") != {
        "source_spec_hash_reauthenticated": True,
        "route_specific_lineage_reauthenticated": True,
        "one_shot_and_no_retry_reauthenticated": True,
        "fixed_partition_worker_reauthenticated": True,
        "glb_and_pbr_readback_reauthenticated": True,
        "all_published_artifacts_are_byte_copies": True,
        "formal_registration_remains_unauthorized": True,
        "overall": "passed",
    }:
        raise contracts.ContractError("adopted automatic checks changed")
    source_spec_record = _require_mapping(batch.get("source_spec"), "source spec")
    if set(source_spec_record) != _FILE_FIELDS | {"spec_sha256"}:
        raise contracts.ContractError("source spec record fields changed")
    copied_spec = _relative_record_path(
        {name: source_spec_record[name] for name in _FILE_FIELDS},
        root,
        "copied source spec",
    )
    _, spec, context = load_adoption_spec(copied_spec)
    if (
        spec.get("spec_sha256") != source_spec_record.get("spec_sha256")
        or spec.get("route") != route
        or spec.get("source_kind") != batch.get("source_kind")
    ):
        raise contracts.ContractError("adopted batch/source spec differs")

    copied = _require_mapping(batch.get("copied_artifacts"), "copied artifacts")
    expected_roles = set(context["bundle_sources"])
    if set(copied) != expected_roles:
        raise contracts.ContractError("copied artifact roles changed")
    copied_paths: dict[str, Path] = {}
    for role, record in copied.items():
        copied_path = _relative_record_path(record, root, f"copied artifact {role}")
        source = context["bundle_sources"][role]
        if (
            record.get("sha256") != _sha256_file(source)
            or record.get("size_bytes") != source.stat().st_size
        ):
            raise contracts.ContractError(f"copied artifact {role} is not source bytes")
        copied_paths[role] = copied_path

    attempt = _require_exact_fields(attempt_value, _ATTEMPT_FIELDS, "adopted attempt")
    _require_canonical_identifier(
        attempt.get("instance_id"), "adopted attempt instance_id"
    )
    controlled = context["controlled"]
    expected_attempt_scalars = {
        "instance_id": spec["instance_id"],
        "execution_job_id": controlled["execution_job_id"],
        "request_sha256": controlled["request_sha256"],
        "profile_schema_id": controlled["profile_schema_id"],
        "sampled_attributes": controlled["sampled_attributes"],
        "target_physical_profile": controlled["target_physical_profile"],
        "gpu": context["gpu"],
        "seed": context["seed"],
        "attempt_ordinal": 0,
        "one_shot_execution": context["job"]["one_shot_execution"],
        "timings": context["timings"],
        "status": "passed_generation_and_glb_readback",
        "next_gate": "static_visual_qa",
    }
    if any(attempt.get(name) != value for name, value in expected_attempt_scalars.items()):
        raise contracts.ContractError("adopted attempt differs from source attempt")
    for field, role in (
        ("pixal_input", "pixal_input_rgba"),
        ("output", "pixal_raw_glb"),
        ("attempt_manifest", "pixal_attempt_manifest"),
    ):
        if attempt.get(field) != copied.get(role):
            raise contracts.ContractError(f"adopted attempt {field} binding changed")
    _validate_staged_glb(
        copied_paths["pixal_raw_glb"],
        staging=root,
        input_rgba=copied_paths["pixal_input_rgba"],
        label="adopted Pixal GLB",
    )
    live = audit_mesh_efficiency.mesh_stats(copied_paths["pixal_raw_glb"])
    mesh_readback = {
        name: value for name, value in live.items() if name not in {"path", "exists"}
    }
    if (
        live.get("exists") is not True
        or attempt.get("mesh_readback") != mesh_readback
        or mesh_readback != context["mesh_readback"]
    ):
        raise contracts.ContractError("adopted GLB/PBR readback changed")
    _reject_formal_authorization(batch, "adopted batch")
    return path, copy.deepcopy(dict(batch))


def _absolute_file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _direct_profile_sha256(
    *,
    profile_schema_id: str,
    taxonomy: Mapping[str, Any],
    fixed_attributes: Mapping[str, Any],
    lineage_group_id: str,
    rig_profile: Mapping[str, Any],
    acoustic_profile: Mapping[str, Any],
) -> str:
    """Hash the finite semantic profile completed by the direct authority."""

    return _json_sha256(
        {
            "schema": DIRECT_PROFILE_IDENTITY_SCHEMA,
            "profile_schema_id": profile_schema_id,
            "asset_class": "animal",
            "lineage_group_id": lineage_group_id,
            "taxonomy": copy.deepcopy(dict(taxonomy)),
            "fixed_attributes": copy.deepcopy(dict(fixed_attributes)),
            "rig_profile": copy.deepcopy(dict(rig_profile)),
            "acoustic_profile": copy.deepcopy(dict(acoustic_profile)),
        }
    )


def _validate_direct_semantics(
    *,
    profile_schema_id: Any,
    taxonomy: Any,
    fixed_attributes: Any,
    lineage_group_id: Any,
    acoustic_profile: Any,
    sampled_attributes: Any,
    rig_profile: Any,
) -> str:
    profile_schema_id = _require_canonical_identifier(
        profile_schema_id, "direct authority profile_schema_id"
    )
    lineage_group_id = _require_canonical_identifier(
        lineage_group_id, "direct authority lineage_group_id"
    )
    taxonomy_value = contracts._validate_attribute_values(
        taxonomy, "direct authority taxonomy"
    )
    if set(taxonomy_value) != {"species", "breed"}:
        raise contracts.ContractError(
            "direct authority taxonomy must contain exactly species and breed"
        )
    for name, value in taxonomy_value.items():
        _require_canonical_identifier(
            value, f"direct authority taxonomy.{name}"
        )
    fixed_value = contracts._validate_attribute_values(
        fixed_attributes, "direct authority fixed_attributes"
    )
    sampled_value = contracts._validate_attribute_values(
        sampled_attributes, "authenticated direct sampled_attributes"
    )
    overlap = (
        (set(taxonomy_value) & set(fixed_value))
        | (set(taxonomy_value) & set(sampled_value))
        | (set(fixed_value) & set(sampled_value))
    )
    if overlap:
        raise contracts.ContractError(
            f"direct authority attribute names overlap: {sorted(overlap)}"
        )
    validated_rig = contracts._validate_rig_profile(
        rig_profile, asset_class="animal"
    )
    validated_acoustic = contracts._validate_acoustic_profile(
        acoustic_profile,
        available_attributes=(
            set(taxonomy_value) | set(fixed_value) | set(sampled_value)
        ),
    )
    return _direct_profile_sha256(
        profile_schema_id=profile_schema_id,
        taxonomy=taxonomy_value,
        fixed_attributes=fixed_value,
        lineage_group_id=lineage_group_id,
        rig_profile=validated_rig,
        acoustic_profile=validated_acoustic,
    )


def build_direct_source_authority(
    adopted_batch_path: Path,
    *,
    taxonomy: Mapping[str, Any],
    fixed_attributes: Mapping[str, Any],
    lineage_group_id: str,
    acoustic_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the small explicit semantic authority for one adopted batch.

    Sampled attributes, the physical target, rig profile, and model revisions
    intentionally remain in the authenticated adopted batch/source attempt.
    """

    adopted_batch_path, batch = load_adopted_batch(adopted_batch_path)
    attempt = batch["attempts"][0]
    source_spec_record = batch["source_spec"]
    source_spec_path = _relative_record_path(
        {name: source_spec_record[name] for name in _FILE_FIELDS},
        adopted_batch_path.parent,
        "copied source spec",
    )
    _, _spec, adoption_context = load_adoption_spec(source_spec_path)
    profile_sha256 = _validate_direct_semantics(
        profile_schema_id=attempt["profile_schema_id"],
        taxonomy=taxonomy,
        fixed_attributes=fixed_attributes,
        lineage_group_id=lineage_group_id,
        acoustic_profile=acoustic_profile,
        sampled_attributes=attempt["sampled_attributes"],
        rig_profile=adoption_context["controlled"]["rig_profile"],
    )
    authority: dict[str, Any] = {
        "schema": SOURCE_AUTHORITY_SCHEMA,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "adopted_batch": {
            "file": _absolute_file_record(adopted_batch_path),
            "batch_sha256": batch["batch_sha256"],
        },
        "source_spec": {
            "file": _absolute_file_record(source_spec_path),
            "spec_sha256": source_spec_record["spec_sha256"],
        },
        "instance_id": attempt["instance_id"],
        "profile_schema_id": attempt["profile_schema_id"],
        "profile_sha256": profile_sha256,
        "request_sha256": attempt["request_sha256"],
        "taxonomy": copy.deepcopy(dict(taxonomy)),
        "fixed_attributes": copy.deepcopy(dict(fixed_attributes)),
        "lineage_group_id": lineage_group_id,
        "acoustic_profile": copy.deepcopy(dict(acoustic_profile)),
    }
    authority["authority_sha256"] = _hash_without(
        authority, "authority_sha256"
    )
    return authority


def load_direct_source_authority(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Authenticate a direct semantic authority and its complete adopted source."""

    unresolved = Path(path).absolute()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(
            f"direct source authority is missing: {unresolved}"
        )
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "direct source authority file SHA-256")
        path = _authenticate_path_hash(
            str(unresolved),
            expected_sha256,
            "direct source authority",
        )
    else:
        path = unresolved.resolve()
    authority = _require_exact_fields(
        contracts.load_json(path),
        _SOURCE_AUTHORITY_FIELDS,
        "direct source authority",
    )
    if (
        authority.get("schema") != SOURCE_AUTHORITY_SCHEMA
        or authority.get("state_classification") != "research_candidate"
        or authority.get("formal_dataset_registration_authorized") is not False
        or authority.get("authority_sha256")
        != _hash_without(authority, "authority_sha256")
    ):
        raise contracts.ContractError(
            "direct source authority contract/hash is invalid"
        )
    instance_id = _require_canonical_identifier(
        authority.get("instance_id"), "direct authority instance_id"
    )
    _require_sha256(
        authority.get("profile_sha256"), "direct authority profile_sha256"
    )
    _require_sha256(
        authority.get("request_sha256"), "direct authority request_sha256"
    )

    batch_authority = _require_exact_fields(
        authority.get("adopted_batch"),
        frozenset({"file", "batch_sha256"}),
        "direct authority adopted_batch",
    )
    batch_file = _require_exact_fields(
        batch_authority.get("file"),
        _FILE_FIELDS,
        "direct authority adopted batch file",
    )
    if not Path(batch_file["path"]).is_absolute():
        raise contracts.ContractError(
            "direct authority adopted batch path must be absolute"
        )
    adopted_batch_path = _authenticate_file_record(
        batch_file, "direct authority adopted batch file"
    )
    adopted_batch_path, batch = load_adopted_batch(adopted_batch_path)
    if batch_authority.get("batch_sha256") != batch.get("batch_sha256"):
        raise contracts.ContractError(
            "direct authority adopted batch identity changed"
        )

    source_authority = _require_exact_fields(
        authority.get("source_spec"),
        frozenset({"file", "spec_sha256"}),
        "direct authority source_spec",
    )
    source_file = _require_exact_fields(
        source_authority.get("file"),
        _FILE_FIELDS,
        "direct authority source spec file",
    )
    if not Path(source_file["path"]).is_absolute():
        raise contracts.ContractError(
            "direct authority source spec path must be absolute"
        )
    authority_source_spec_path = _authenticate_file_record(
        source_file, "direct authority source spec file"
    )
    batch_source_record = batch["source_spec"]
    batch_source_spec_path = _relative_record_path(
        {name: batch_source_record[name] for name in _FILE_FIELDS},
        adopted_batch_path.parent,
        "adopted batch source spec",
    )
    if (
        authority_source_spec_path != batch_source_spec_path
        or source_file["sha256"] != batch_source_record["sha256"]
        or source_file["size_bytes"] != batch_source_record["size_bytes"]
        or source_authority.get("spec_sha256")
        != batch_source_record.get("spec_sha256")
    ):
        raise contracts.ContractError(
            "direct authority source spec identity changed"
        )
    source_spec_path, source_spec, adoption_context = load_adoption_spec(
        batch_source_spec_path
    )
    attempt = batch["attempts"][0]
    controlled = adoption_context["controlled"]
    if (
        instance_id != attempt.get("instance_id")
        or instance_id != source_spec.get("instance_id")
        or authority.get("profile_schema_id")
        != attempt.get("profile_schema_id")
        or authority.get("profile_schema_id")
        != controlled.get("profile_schema_id")
        or authority.get("request_sha256") != attempt.get("request_sha256")
        or authority.get("request_sha256") != controlled.get("request_sha256")
    ):
        raise contracts.ContractError(
            "direct authority adopted attempt identity changed"
        )
    expected_profile_sha256 = _validate_direct_semantics(
        profile_schema_id=authority["profile_schema_id"],
        taxonomy=authority["taxonomy"],
        fixed_attributes=authority["fixed_attributes"],
        lineage_group_id=authority["lineage_group_id"],
        acoustic_profile=authority["acoustic_profile"],
        sampled_attributes=attempt["sampled_attributes"],
        rig_profile=controlled["rig_profile"],
    )
    if authority["profile_sha256"] != expected_profile_sha256:
        raise contracts.ContractError(
            "direct authority semantic profile hash changed"
        )
    _reject_formal_authorization(authority, "direct source authority")
    return (
        path,
        copy.deepcopy(dict(authority)),
        {
            "adopted_batch_path": adopted_batch_path,
            "adopted_batch": copy.deepcopy(dict(batch)),
            "attempt": copy.deepcopy(dict(attempt)),
            "source_spec_path": source_spec_path,
            "source_spec": copy.deepcopy(dict(source_spec)),
            "adoption_context": copy.deepcopy(dict(adoption_context)),
        },
    )


def _load_source_authority_semantics(path: Path) -> dict[str, Any]:
    unresolved = Path(path).absolute()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(
            f"direct source authority semantics are missing: {unresolved}"
        )
    semantics = _require_mapping(
        contracts.load_json(unresolved.resolve()),
        "direct source authority semantics",
    )
    if not _SOURCE_AUTHORITY_SEMANTICS_FIELDS.issubset(semantics):
        raise contracts.ContractError(
            "direct source authority semantics are incomplete"
        )
    return {
        name: copy.deepcopy(semantics[name])
        for name in _SOURCE_AUTHORITY_SEMANTICS_FIELDS
    }


def _write_direct_source_authority_no_replace(
    path: Path,
    authority: Mapping[str, Any],
) -> Path:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        payload = (contracts.canonical_json(authority) + "\n").encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".staging",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        load_direct_source_authority(
            temporary,
            expected_sha256=_sha256_file(temporary),
        )
        _rename_no_replace(temporary, destination)
        temporary = None
        directory_fd = os.open(
            destination.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination.resolve()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--source-authority-output",
        type=Path,
        help=(
            "optionally publish a canonical direct source authority after "
            "successful adoption"
        ),
    )
    parser.add_argument(
        "--source-authority-semantics",
        type=Path,
        help=(
            "JSON object containing taxonomy, fixed_attributes, "
            "lineage_group_id, and acoustic_profile"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if (args.source_authority_output is None) != (
            args.source_authority_semantics is None
        ):
            raise ValueError(
                "--source-authority-output and "
                "--source-authority-semantics must be provided together"
            )
        semantics = (
            _load_source_authority_semantics(args.source_authority_semantics)
            if args.source_authority_semantics is not None
            else None
        )
        manifest = adopt_attempt(args.spec, args.output_root)
        authority_path: Path | None = None
        if args.source_authority_output is not None:
            assert semantics is not None
            authority = build_direct_source_authority(
                manifest,
                **semantics,
            )
            authority_path = _write_direct_source_authority_no_replace(
                args.source_authority_output,
                authority,
            )
    except (contracts.ContractError, OSError, ValueError) as error:
        print(f"DIRECT_ANIMAL_PIXAL_ADOPTION_FAILED {error}", file=sys.stderr)
        return 2
    print(f"DIRECT_ANIMAL_PIXAL_ADOPTION_OK output={manifest}")
    if authority_path is not None:
        print(f"DIRECT_ANIMAL_SOURCE_AUTHORITY_OK output={authority_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
