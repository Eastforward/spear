#!/usr/bin/env python3
"""Run fail-closed downstream admission for approved generated static objects.

The runner is deliberately category-blind.  Every instance supplies only a
reviewed heading authority, a reviewed emitter-anchor authority, and one
generic watertight parameter set.  The runner binds those frozen semantic
authorities to the newly produced hashes without changing their reviewed
meaning.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_mesh_efficiency
from tools import controlled_source_asset_schema as contracts
from tools import generated_asset_emitter_contract as emitter_contract
from tools import review_controlled_static_object_candidates as static_decisions
from tools import rocketbox_native_material_canary as immutable

PLAN_SCHEMA = "avengine_controlled_static_object_admission_plan_v1"
ANCHOR_AUTHORITY_SCHEMA = "avengine_static_emitter_anchor_authority_v1"
BATCH_SCHEMA = "avengine_controlled_static_object_admission_batch_v1"
JOB_RECEIPT_SCHEMA = "avengine_controlled_static_object_admission_receipt_v1"
STAGE_RECEIPT_SCHEMA = "avengine_controlled_static_object_stage_receipt_v1"
COMMAND_INPUT_MANIFEST_SCHEMA = "avengine_controlled_static_object_command_input_manifest_v1"
FAILURE_SCHEMA = "avengine_controlled_static_object_admission_failure_v1"
STATIC_ASSET_CLASS = "static_object"
STATIC_ROUTE = "flux2_pixal3d_static_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/data/jzy/.local/bin/blender")
WATERTIGHT_TOOL = (
    SPEAR_ROOT / "tools/blender_create_watertight_textured_proxy_mesh.py"
)
FINALIZER_TOOL = SPEAR_ROOT / "tools/blender_finalize_generated_static_object.py"
EMITTER_TOOL = SPEAR_ROOT / "tools/blender_measure_generated_static_emitter.py"
# The finalizer and emitter import this module through ``tools``.  Preserve
# that package-relative layout in the staged runtime snapshot as well.
COMMAND_PYTHON_DEPENDENCIES = {
    "generated_asset_emitter_contract.py": SPEAR_ROOT
    / "tools/generated_asset_emitter_contract.py",
}
STAGE_PYTHON_DEPENDENCIES = {
    "watertight": (),
    "finalization": ("generated_asset_emitter_contract.py",),
    "emitter_measurement": ("generated_asset_emitter_contract.py",),
}
STAGE_TIMEOUTS = {
    "watertight": 10800,
    "finalization": 1800,
    "emitter_measurement": 1800,
}
WATERTIGHT_PARAMETER_FIELDS = {
    "voxel_resolution",
    "target_faces",
    "smooth_iterations",
    "shrinkwrap_strength",
    "post_shrinkwrap_smooth_iterations",
    "torso_fold_repair_iterations",
    "attribute_transfer_backend",
    "bake_resolution",
    "base_color_encoding_policy",
    "base_color_gain",
    "double_sided",
}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class AdmissionError(ValueError):
    """Raised when static-object admission cannot publish a complete batch."""


def _command_tool_sources() -> dict[str, Path]:
    """Return live constants so tests and explicit pins remain injectable."""

    return {
        "watertight": WATERTIGHT_TOOL,
        "finalization": FINALIZER_TOOL,
        "emitter_measurement": EMITTER_TOOL,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _require_sha256(value: Any, label: str) -> str:
    try:
        return emitter_contract.require_sha256(value, label)
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error


def _require_identifier(value: Any, label: str) -> str:
    try:
        return emitter_contract.require_identifier(value, label)
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contracts.write_json_no_replace(path, dict(payload))


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without ever replacing a peer's root."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AdmissionError("atomic no-replace publication requires Linux renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
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
        raise FileExistsError(number, "refusing to replace existing admission root", destination)
    raise OSError(number, os.strerror(number), destination)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a generated manifest during pre-publication rebasing."""

    path = path.resolve()
    temporary = path.parent / f".{path.name}.rebase_{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise AdmissionError(f"manifest rebase scratch path already exists: {temporary}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_record(path: Path, *, recorded_path: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    public = recorded_path.resolve() if recorded_path is not None else path
    return {
        "path": str(public),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_direct_file(path_value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise AdmissionError(f"{label} path is invalid")
    unresolved = Path(path_value)
    if not unresolved.is_absolute():
        unresolved = base / unresolved
    if unresolved.is_symlink():
        raise AdmissionError(f"{label} must not be a symlink")
    path = unresolved.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise AdmissionError(f"{label} is missing or empty: {path}")
    return path


def _resolve_file_record(
    record: Any,
    *,
    base: Path,
    label: str,
    require_within_base: bool,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise AdmissionError(f"{label} file record is invalid")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise AdmissionError(f"{label} file-record path is invalid")
    unresolved = Path(path_value)
    if not unresolved.is_absolute():
        unresolved = base / unresolved
    if unresolved.is_symlink():
        raise AdmissionError(f"{label} must not be a symlink")
    path = unresolved.resolve()
    if require_within_base:
        try:
            path.relative_to(base.resolve())
        except ValueError as error:
            raise AdmissionError(f"{label} escaped its immutable root") from error
    size = record.get("size_bytes")
    if (
        not path.is_file()
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
        or _sha256_file(path) != _require_sha256(record.get("sha256"), f"{label} hash")
    ):
        raise AdmissionError(f"{label} hash/size changed")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return emitter_contract.load_json_object(path, label)
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error


def _mesh_stats(path: Path, label: str) -> dict[str, Any]:
    try:
        stats = audit_mesh_efficiency.mesh_stats(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise AdmissionError(f"{label} GLB readback failed") from error
    if (
        not isinstance(stats, dict)
        or stats.get("exists") is not True
        or stats.get("vertices", 0) <= 0
        or stats.get("triangles", 0) <= 0
        or stats.get("materials", 0) <= 0
        or stats.get("textures", 0) <= 0
        or stats.get("skins") != 0
        or stats.get("animations") != 0
    ):
        raise AdmissionError(f"{label} must be textured, unskinned, and unanimated")
    return {
        key: value for key, value in stats.items() if key not in {"path", "exists"}
    }


def _load_review_lineage(value: Any) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Re-open decisions' static review and Pixal lineage, not just its digest."""

    if not isinstance(value, Mapping) or set(value) != {
        "path", "sha256", "review_batch_sha256"
    }:
        raise AdmissionError("decision/static-review batch binding is invalid")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise AdmissionError("decision/static-review batch path is invalid")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        raise AdmissionError("decision/static-review batch is missing or unsafe")
    path = path.resolve()
    if _sha256_file(path) != _require_sha256(value.get("sha256"), "static review batch hash"):
        raise AdmissionError("decision/static-review batch file changed")
    try:
        review_path, review_batch, reviews = static_decisions.load_review_batch(path)
    except (contracts.ContractError, OSError, json.JSONDecodeError) as error:
        raise AdmissionError(f"decision static-review/Pixal lineage is invalid: {error}") from error
    if (
        review_path != path
        or review_batch.get("review_batch_sha256")
        != _require_sha256(value.get("review_batch_sha256"), "static review batch content hash")
    ):
        raise AdmissionError("decision/static-review batch content binding changed")
    return path, review_batch, reviews


def load_decision_batch(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    unresolved = Path(path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise AdmissionError(f"static-object decision batch is missing: {path}")
    path = unresolved.resolve()
    batch = _load_json(path, "static-object decision batch")
    review_path, review_batch, reviews = _load_review_lineage(
        batch.get("static_object_review_batch") if isinstance(batch, Mapping) else None
    )
    decisions_index = batch.get("decisions")
    decision_count = batch.get("decision_count")
    approved_count_value = batch.get("approved_count")
    rejected_count_value = batch.get("rejected_count")
    counts_are_valid = all(
        not isinstance(value, bool) and isinstance(value, int) and value >= 0
        for value in (
            decision_count,
            approved_count_value,
            rejected_count_value,
        )
    )
    if (
        batch.get("schema") != static_decisions.DECISION_BATCH_SCHEMA
        or batch.get("status") != "completed"
        or batch.get("state_classification") != "research_candidate"
        or batch.get("asset_class") != STATIC_ASSET_CLASS
        or batch.get("route") != STATIC_ROUTE
        or batch.get("formal_dataset_registration_authorized") is not False
        or batch.get("decision_batch_sha256")
        != _hash_without(batch, "decision_batch_sha256")
        or batch.get("automatic_checks", {}).get("overall") != "passed"
        or not isinstance(decisions_index, list)
        or not counts_are_valid
        or decision_count != len(decisions_index)
        or approved_count_value + rejected_count_value != len(decisions_index)
    ):
        raise AdmissionError("static-object decision batch contract/hash is invalid")

    root = path.parent
    decisions: dict[str, dict[str, Any]] = {}
    approved_count = 0
    rejected_count = 0
    for index in batch["decisions"]:
        if not isinstance(index, Mapping):
            raise AdmissionError("static-object decision index is invalid")
        instance_id = index.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or instance_id in decisions
            or index.get("decision") not in {
                static_decisions.APPROVED,
                static_decisions.REJECTED,
            }
        ):
            raise AdmissionError("static-object decision identity/status is invalid")
        _require_identifier(instance_id, "decision instance_id")
        decision_path = _resolve_file_record(
            index.get("record"),
            base=root,
            label=f"{instance_id} decision record",
            require_within_base=True,
        )
        decision = _load_json(decision_path, f"{instance_id} decision")
        review = reviews.get(instance_id)
        if (
            review is None
            or
            decision.get("schema") != static_decisions.DECISION_SCHEMA
            or decision.get("instance_id") != instance_id
            or decision.get("decision") != index["decision"]
            or decision.get("decision_sha256") != index.get("decision_sha256")
            or decision.get("decision_sha256")
            != _hash_without(decision, "decision_sha256")
            or decision.get("asset_class") != STATIC_ASSET_CLASS
            or decision.get("route") != STATIC_ROUTE
            or decision.get("formal_dataset_registration_authorized") is not False
            or decision.get("request_sha256") != index.get("request_sha256")
            or decision.get("profile_sha256") != index.get("profile_sha256")
            or decision.get("review")
            != _file_record(review["path"])
            or decision.get("pixal_output")
            != _file_record(review["pixal_output_path"])
        ):
            raise AdmissionError(
                f"{instance_id} static-object decision contract/hash changed"
            )
        _require_sha256(decision["request_sha256"], "decision request_sha256")
        _require_sha256(decision["profile_sha256"], "decision profile_sha256")
        pixal_path = _resolve_file_record(
            decision.get("pixal_output"),
            base=root,
            label=f"{instance_id} approved Pixal output",
            require_within_base=False,
        )
        if decision["pixal_output"]["sha256"] != index.get("pixal_output_sha256"):
            raise AdmissionError(f"{instance_id} decision/Pixal index hash changed")
        physical = decision.get("target_physical_profile")
        if (
            not isinstance(physical, Mapping)
            or physical.get("control_attribute") is not None
            or physical.get("measurement") != "height_cm"
        ):
            raise AdmissionError(
                f"{instance_id} static physical profile is not finalizer-compatible"
            )
        for field in ("target_value_cm", "tolerance_cm"):
            value = physical.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise AdmissionError(
                    f"{instance_id} physical {field} must be finite and positive"
                )
        if decision["decision"] == static_decisions.APPROVED:
            if decision.get("next_gate") != "watertight_then_static_finalization":
                raise AdmissionError(f"{instance_id} approved next gate changed")
            approved_count += 1
        else:
            if decision.get("next_gate") != "stop":
                raise AdmissionError(f"{instance_id} rejected next gate changed")
            rejected_count += 1
        decisions[instance_id] = {
            "index": copy.deepcopy(dict(index)),
            "payload": decision,
            "path": decision_path,
            "record": _file_record(decision_path),
            "pixal_path": pixal_path,
            "pixal_record": _file_record(pixal_path),
            "pixal_mesh_readback": _mesh_stats(
                pixal_path, f"{instance_id} raw Pixal"
            ),
        }
    if (
        approved_count != batch["approved_count"]
        or rejected_count != batch["rejected_count"]
    ):
        raise AdmissionError("static-object decision counts changed")
    if approved_count == 0:
        raise AdmissionError("static-object admission has no approved decisions")
    return path, batch, decisions


def validate_watertight_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != WATERTIGHT_PARAMETER_FIELDS:
        raise AdmissionError("watertight parameter fields are invalid")
    result = copy.deepcopy(dict(value))

    integer_ranges = {
        "voxel_resolution": (96, 512),
        "target_faces": (10000, 1000000),
        "smooth_iterations": (0, 8),
        "post_shrinkwrap_smooth_iterations": (0, 8),
        "torso_fold_repair_iterations": (0, 20),
    }
    for field, (minimum, maximum) in integer_ranges.items():
        item = result[field]
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not minimum <= item <= maximum
        ):
            raise AdmissionError(
                f"watertight {field} must be in [{minimum}, {maximum}]"
            )
    strength = result["shrinkwrap_strength"]
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or not 0.0 <= float(strength) <= 1.0
    ):
        raise AdmissionError("watertight shrinkwrap_strength must be in [0, 1]")
    result["shrinkwrap_strength"] = float(strength)
    if result["attribute_transfer_backend"] not in {"bake", "bvh", "data-transfer"}:
        raise AdmissionError("unsupported watertight attribute transfer backend")
    if result["bake_resolution"] not in {512, 1024, 2048, 4096}:
        raise AdmissionError("unsupported watertight bake resolution")
    if result["base_color_encoding_policy"] not in {
        "preserve-bake",
        "srgb-to-linear",
    }:
        raise AdmissionError("unsupported base-color encoding policy")
    gain = result["base_color_gain"]
    if (
        not isinstance(gain, list)
        or len(gain) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 < float(item) <= 2.0
            for item in gain
        )
    ):
        raise AdmissionError("base_color_gain must contain three values in (0, 2]")
    result["base_color_gain"] = [float(item) for item in gain]
    if not isinstance(result["double_sided"], bool):
        raise AdmissionError("watertight double_sided must be boolean")
    return result


def _validate_heading_authority(
    path: Path,
    decision: Mapping[str, Any],
    pixal_path: Path,
) -> dict[str, Any]:
    payload = _load_json(path, "static heading authority")
    required = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "input_glb_sha256",
        "review_artifact",
        "reviewed_source_front_yaw_deg",
        "target_front_axis",
        "decision",
        "formal_dataset_registration_authorized",
    }
    if (
        set(payload) != required
        or payload.get("schema") != "avengine_static_heading_review_v1"
        or payload.get("target_front_axis") != "positive-x"
        or payload.get("decision") != "approved_for_positive_x_normalization"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise AdmissionError("static heading authority contract is invalid")
    for field in ("instance_id", "request_sha256", "profile_sha256"):
        if payload[field] != decision[field]:
            raise AdmissionError(f"static heading authority {field} changed")
    if _require_sha256(
        payload["input_glb_sha256"], "heading authority input hash"
    ) != _sha256_file(pixal_path):
        raise AdmissionError("heading authority must bind the approved raw Pixal GLB")
    try:
        emitter_contract.validate_file_record(
            payload["review_artifact"], label="static heading review artifact"
        )
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error
    yaw = payload["reviewed_source_front_yaw_deg"]
    if (
        isinstance(yaw, bool)
        or not isinstance(yaw, (int, float))
        or not math.isfinite(float(yaw))
        or not -180.0 <= float(yaw) <= 180.0
    ):
        raise AdmissionError("reviewed source front yaw must be in [-180, 180]")
    return payload


def _validate_anchor_authority(
    path: Path,
    decision: Mapping[str, Any],
    pixal_path: Path,
) -> dict[str, Any]:
    payload = _load_json(path, "static emitter anchor authority")
    required = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "input_glb_sha256",
        "anchor_id",
        "anchor_type",
        "semantic_role",
        "selection",
        "review_evidence",
        "formal_dataset_registration_authorized",
    }
    if (
        set(payload) != required
        or payload.get("schema") != ANCHOR_AUTHORITY_SCHEMA
        or payload.get("anchor_type") != "object_speaker"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise AdmissionError("static emitter anchor authority contract is invalid")
    for field in ("instance_id", "request_sha256", "profile_sha256"):
        if payload[field] != decision[field]:
            raise AdmissionError(f"static anchor authority {field} changed")
    if _require_sha256(
        payload["input_glb_sha256"], "anchor authority input hash"
    ) != _sha256_file(pixal_path):
        raise AdmissionError("static anchor authority must bind the approved raw Pixal GLB")
    _require_identifier(payload["anchor_id"], "anchor authority anchor_id")
    _require_identifier(payload["semantic_role"], "anchor authority semantic_role")
    try:
        emitter_contract.validate_file_record(
            payload["review_evidence"], label="anchor authority review evidence"
        )
        emitter_contract._validate_selection(payload["selection"])
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error
    return payload


def load_plan(
    path: Path,
    *,
    decision_batch: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    unresolved = Path(path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise AdmissionError(f"static-object admission plan is missing: {path}")
    path = unresolved.resolve()
    plan = _load_json(path, "static-object admission plan")
    if (
        set(plan)
        != {
            "schema",
            "decision_batch_sha256",
            "instances",
            "formal_dataset_registration_authorized",
            "plan_sha256",
        }
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("decision_batch_sha256")
        != decision_batch["decision_batch_sha256"]
        or plan.get("formal_dataset_registration_authorized") is not False
        or plan.get("plan_sha256") != _hash_without(plan, "plan_sha256")
        or not isinstance(plan.get("instances"), list)
        or not plan["instances"]
    ):
        raise AdmissionError("static-object admission plan contract/hash is invalid")

    approved_ids = {
        instance_id
        for instance_id, decision in decisions.items()
        if decision["payload"]["decision"] == static_decisions.APPROVED
    }
    jobs: dict[str, dict[str, Any]] = {}
    for item in plan["instances"]:
        if not isinstance(item, Mapping) or set(item) != {
            "instance_id",
            "heading_evidence_path",
            "anchor_spec_path",
            "watertight_parameters",
        }:
            raise AdmissionError("admission plan instance fields are invalid")
        instance_id = item.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or instance_id in jobs
            or instance_id not in approved_ids
        ):
            raise AdmissionError("admission plan instance coverage/identity is invalid")
        heading_path = _resolve_direct_file(
            item["heading_evidence_path"],
            base=path.parent,
            label=f"{instance_id} heading authority",
        )
        anchor_path = _resolve_direct_file(
            item["anchor_spec_path"],
            base=path.parent,
            label=f"{instance_id} anchor authority",
        )
        decision = decisions[instance_id]
        heading = _validate_heading_authority(
            heading_path,
            decision["payload"],
            decision["pixal_path"],
        )
        anchor = _validate_anchor_authority(
            anchor_path, decision["payload"], decision["pixal_path"]
        )
        jobs[instance_id] = {
            "instance_id": instance_id,
            "decision": decision,
            "heading_authority": heading,
            "heading_authority_path": heading_path,
            "heading_authority_record": _file_record(heading_path),
            "anchor_authority": anchor,
            "anchor_authority_path": anchor_path,
            "anchor_authority_record": _file_record(anchor_path),
            "watertight_parameters": validate_watertight_parameters(
                item["watertight_parameters"]
            ),
        }
    if set(jobs) != approved_ids:
        raise AdmissionError("admission plan must cover every approved decision once")
    return path, plan, jobs


def validate_admission_inputs(
    decision_batch_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    decision_path, decision_batch, decisions = load_decision_batch(
        decision_batch_path
    )
    resolved_plan, plan, jobs = load_plan(
        plan_path,
        decision_batch=decision_batch,
        decisions=decisions,
    )
    return {
        "decision_batch_path": decision_path,
        "decision_batch": decision_batch,
        "decision_batch_record": _file_record(decision_path),
        "plan_path": resolved_plan,
        "plan": plan,
        "plan_record": _file_record(resolved_plan),
        "jobs": jobs,
    }


def _assert_file_unchanged(
    path: Path,
    expected_record: Mapping[str, Any],
    label: str,
) -> None:
    if _file_record(path) != expected_record:
        raise AdmissionError(f"{label} changed during admission")


def _assert_job_sources_unchanged(job: Mapping[str, Any]) -> None:
    decision = job["decision"]
    _assert_file_unchanged(
        decision["path"], decision["record"], f"{job['instance_id']} decision"
    )
    _assert_file_unchanged(
        decision["pixal_path"],
        decision["pixal_record"],
        f"{job['instance_id']} raw Pixal GLB",
    )
    _assert_file_unchanged(
        job["heading_authority_path"],
        job["heading_authority_record"],
        f"{job['instance_id']} heading authority",
    )
    _assert_file_unchanged(
        job["anchor_authority_path"],
        job["anchor_authority_record"],
        f"{job['instance_id']} anchor authority",
    )


def _assert_admission_sources_unchanged(inputs: Mapping[str, Any]) -> None:
    _assert_file_unchanged(
        inputs["decision_batch_path"],
        inputs["decision_batch_record"],
        "decision batch",
    )
    _assert_file_unchanged(inputs["plan_path"], inputs["plan_record"], "admission plan")
    for instance_id in sorted(inputs["jobs"]):
        _assert_job_sources_unchanged(inputs["jobs"][instance_id])


def _copy_frozen_file(source: Path, expected: Mapping[str, Any], destination: Path, label: str) -> dict[str, Any]:
    """Byte-copy one authenticated input, rejecting a change before or during copy."""

    _assert_file_unchanged(source, expected, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        destination.chmod(0o444)
    except FileExistsError as error:
        raise AdmissionError(f"frozen input destination already exists: {destination}") from error
    copied = _file_record(destination)
    if copied["sha256"] != expected["sha256"] or copied["size_bytes"] != expected["size_bytes"]:
        raise AdmissionError(f"{label} changed while being frozen")
    return copied


def _command_source_snapshot(path: Path, label: str) -> dict[str, Any]:
    """Capture both the configured command path and its resolved bytes."""

    configured = Path(path).absolute()
    if not configured.is_file():
        raise AdmissionError(f"pinned {label} is missing: {configured}")
    resolved = configured.resolve()
    return {
        "configured_path": str(configured),
        "resolved_path": str(resolved),
        "record": _file_record(resolved),
    }


def _assert_command_source_unchanged(
    snapshot: Mapping[str, Any], label: str
) -> None:
    configured = Path(snapshot["configured_path"])
    resolved = Path(snapshot["resolved_path"])
    if (
        not configured.is_file()
        or configured.resolve() != resolved
        or _file_record(resolved) != snapshot["record"]
    ):
        raise AdmissionError(f"pinned {label} changed during admission")


def _snapshot_command_inputs(staging: Path) -> dict[str, Any]:
    """Freeze invoked Python tools and pin the non-copyable Blender binary."""

    blender = _command_source_snapshot(BLENDER, "Blender binary")
    tools_snapshot = {
        stage: _command_source_snapshot(path, f"{stage} tool")
        for stage, path in _command_tool_sources().items()
    }
    dependencies = {
        name: _command_source_snapshot(path, f"{name} dependency")
        for name, path in COMMAND_PYTHON_DEPENDENCIES.items()
    }
    runtime_tools = staging / ".runtime_commands" / "tools"
    frozen_tools: dict[str, Path] = {}
    for stage, source in tools_snapshot.items():
        destination = runtime_tools / Path(source["resolved_path"]).name
        record = _copy_frozen_file(
            Path(source["resolved_path"]),
            source["record"],
            destination,
            f"{stage} tool",
        )
        frozen_tools[stage] = destination
        source["frozen_path"] = str(destination)
        source["frozen_record"] = record
    for name, source in dependencies.items():
        destination = runtime_tools / name
        record = _copy_frozen_file(
            Path(source["resolved_path"]),
            source["record"],
            destination,
            f"{name} dependency",
        )
        source["frozen_path"] = str(destination)
        source["frozen_record"] = record

    # Frozen source is not merely copied: prevent a stage from unlinking and
    # replacing a read-only file through its writable parent directory.
    runtime_tools.chmod(0o555)
    runtime_tools.parent.chmod(0o555)
    return {
        "blender": blender,
        "tools": tools_snapshot,
        "dependencies": dependencies,
        "frozen_tools": frozen_tools,
    }


def _assert_command_inputs_unchanged(command_inputs: Mapping[str, Any]) -> None:
    _assert_command_source_unchanged(command_inputs["blender"], "Blender binary")
    for stage, snapshot in command_inputs["tools"].items():
        _assert_command_source_unchanged(snapshot, f"{stage} tool")
        _assert_file_unchanged(
            Path(snapshot["frozen_path"]),
            snapshot["frozen_record"],
            f"frozen {stage} tool",
        )
    for name, snapshot in command_inputs["dependencies"].items():
        _assert_command_source_unchanged(snapshot, f"{name} dependency")
        _assert_file_unchanged(
            Path(snapshot["frozen_path"]),
            snapshot["frozen_record"],
            f"frozen {name} dependency",
        )


def _snapshot_inputs(inputs: Mapping[str, Any], staging: Path) -> None:
    """Freeze every runner-consumed external input before invoking Blender."""

    _assert_admission_sources_unchanged(inputs)
    for instance_id, job in inputs["jobs"].items():
        root = staging / "input_snapshots" / "instances" / instance_id
        decision = job["decision"]
        heading_evidence = emitter_contract.validate_file_record(
            job["heading_authority"]["review_artifact"], label="heading review artifact"
        )
        anchor_evidence = emitter_contract.validate_file_record(
            job["anchor_authority"]["review_evidence"], label="anchor review evidence"
        )
        frozen_pixal = root / "raw_pixal.glb"
        frozen_heading_review = root / "heading_review.bin"
        frozen_anchor_review = root / "anchor_review.bin"
        frozen_decision = staging / ".runtime_inputs" / instance_id / "decision.json"
        frozen_heading = staging / ".runtime_inputs" / instance_id / "heading.json"
        frozen_anchor = staging / ".runtime_inputs" / instance_id / "anchor.json"
        pixal_record = _copy_frozen_file(decision["pixal_path"], decision["pixal_record"], frozen_pixal, f"{instance_id} raw Pixal GLB")
        heading_review_record = _copy_frozen_file(
            heading_evidence, job["heading_authority"]["review_artifact"], frozen_heading_review, f"{instance_id} heading review evidence"
        )
        anchor_review_record = _copy_frozen_file(
            anchor_evidence, job["anchor_authority"]["review_evidence"], frozen_anchor_review, f"{instance_id} anchor review evidence"
        )
        runtime_decision = copy.deepcopy(decision["payload"])
        runtime_decision["pixal_output"] = _file_record(frozen_pixal)
        runtime_decision["decision_sha256"] = _hash_without(runtime_decision, "decision_sha256")
        _write_json_exclusive(frozen_decision, runtime_decision)
        runtime_heading = copy.deepcopy(job["heading_authority"])
        runtime_heading["review_artifact"] = _file_record(frozen_heading_review)
        _write_json_exclusive(frozen_heading, runtime_heading)
        runtime_anchor = copy.deepcopy(job["anchor_authority"])
        runtime_anchor["review_evidence"] = _file_record(frozen_anchor_review)
        _write_json_exclusive(frozen_anchor, runtime_anchor)
        job["runtime"] = {
            "pixal_path": frozen_pixal,
            "pixal_record": pixal_record,
            "decision_path": frozen_decision,
            "decision_record": _file_record(frozen_decision),
            "decision_payload": runtime_decision,
            "heading_path": frozen_heading,
            "heading_record": _file_record(frozen_heading),
            "heading_review_path": frozen_heading_review,
            "heading_review_record": heading_review_record,
            "anchor_path": frozen_anchor,
            "anchor_record": _file_record(frozen_anchor),
            "anchor_review_path": frozen_anchor_review,
            "anchor_review_record": anchor_review_record,
        }


def _assert_job_sources_unchanged(job: Mapping[str, Any]) -> None:
    """After snapshotting, no stage may consume a live external authority."""

    runtime = job.get("runtime")
    if runtime is None:
        # This supports validate-only and command construction without staging.
        decision = job["decision"]
        _assert_file_unchanged(decision["path"], decision["record"], f"{job['instance_id']} decision")
        _assert_file_unchanged(decision["pixal_path"], decision["pixal_record"], f"{job['instance_id']} raw Pixal GLB")
        return
    for name, path_key, record_key in (
        ("frozen raw Pixal GLB", "pixal_path", "pixal_record"),
        ("frozen decision", "decision_path", "decision_record"),
        ("frozen heading authority", "heading_path", "heading_record"),
        ("frozen anchor authority", "anchor_path", "anchor_record"),
        ("frozen heading review", "heading_review_path", "heading_review_record"),
        ("frozen anchor review", "anchor_review_path", "anchor_review_record"),
    ):
        expected = runtime[record_key]
        _assert_file_unchanged(runtime[path_key], expected, f"{job['instance_id']} {name}")


def _job_paths(root: Path, instance_id: str) -> dict[str, Path]:
    job = root / "instances" / instance_id
    return {
        "job_root": job,
        "watertight_dir": job / "01_watertight",
        "watertight_glb": job / "01_watertight" / "watertight.glb",
        "watertight_manifest": (
            job / "01_watertight" / "watertight_manifest.json"
        ),
        "watertight_log": job / "01_watertight" / "blender.log",
        "watertight_command_inputs": (
            job / "01_watertight" / "command_input_manifest.json"
        ),
        "finalization_dir": job / "02_finalization",
        "bound_heading": job / "02_finalization" / "bound_heading_evidence.json",
        "final_glb": job / "02_finalization" / "finalized.glb",
        "finalization_manifest": (
            job / "02_finalization" / "finalization_manifest.json"
        ),
        "finalization_log": job / "02_finalization" / "blender.log",
        "finalization_command_inputs": (
            job / "02_finalization" / "command_input_manifest.json"
        ),
        "emitter_dir": job / "03_emitter",
        "bound_anchor": job / "03_emitter" / "bound_anchor_spec.json",
        "emitter_measurement": job / "03_emitter" / "emitter_measurement.json",
        "marker_glb": job / "03_emitter" / "emitter_marker.glb",
        "emitter_log": job / "03_emitter" / "blender.log",
        "emitter_command_inputs": (
            job / "03_emitter" / "command_input_manifest.json"
        ),
        "job_receipt": job / "admission_receipt.json",
    }


def build_stage_commands(
    job: Mapping[str, Any],
    work_root: Path,
    *,
    command_inputs: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[str]], dict[str, Path]]:
    paths = _job_paths(Path(work_root).resolve(), job["instance_id"])
    params = job["watertight_parameters"]
    runtime = job.get("runtime")
    pixal_path = runtime["pixal_path"] if runtime else job["decision"]["pixal_path"]
    decision_path = runtime["decision_path"] if runtime else job["decision"]["path"]
    blender_path = (
        Path(command_inputs["blender"]["resolved_path"])
        if command_inputs is not None
        else BLENDER
    )
    tools = (
        command_inputs["frozen_tools"]
        if command_inputs is not None
        else _command_tool_sources()
    )
    watertight = [
        str(blender_path),
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        str(tools["watertight"]),
        "--",
        "--source",
        str(pixal_path),
        "--output",
        str(paths["watertight_glb"]),
        "--manifest",
        str(paths["watertight_manifest"]),
        "--voxel-resolution",
        str(params["voxel_resolution"]),
        "--target-faces",
        str(params["target_faces"]),
        "--smooth-iterations",
        str(params["smooth_iterations"]),
        "--shrinkwrap-strength",
        str(params["shrinkwrap_strength"]),
        "--post-shrinkwrap-smooth-iterations",
        str(params["post_shrinkwrap_smooth_iterations"]),
        "--torso-fold-repair-iterations",
        str(params["torso_fold_repair_iterations"]),
        "--attribute-transfer-backend",
        str(params["attribute_transfer_backend"]),
        "--bake-resolution",
        str(params["bake_resolution"]),
        "--base-color-encoding-policy",
        str(params["base_color_encoding_policy"]),
        "--base-color-gain",
        *(str(item) for item in params["base_color_gain"]),
    ]
    if params["double_sided"]:
        watertight.append("--double-sided")
    finalization = [
        str(blender_path),
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        str(tools["finalization"]),
        "--",
        "--input-glb",
        str(paths["watertight_glb"]),
        "--watertight-manifest",
        str(paths["watertight_manifest"]),
        "--static-decision",
        str(decision_path),
        "--heading-evidence",
        str(paths["bound_heading"]),
        "--output",
        str(paths["final_glb"]),
        "--manifest",
        str(paths["finalization_manifest"]),
    ]
    emitter = [
        str(blender_path),
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        str(tools["emitter_measurement"]),
        "--",
        "--input-glb",
        str(paths["final_glb"]),
        "--finalization-manifest",
        str(paths["finalization_manifest"]),
        "--anchor-spec",
        str(paths["bound_anchor"]),
        "--output",
        str(paths["emitter_measurement"]),
        "--marker-glb",
        str(paths["marker_glb"]),
    ]
    return {
        "watertight": watertight,
        "finalization": finalization,
        "emitter_measurement": emitter,
    }, paths


def _execute_command(
    stage: str,
    command: Sequence[str],
    log_path: Path,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started = time.perf_counter()
    error_text = None
    returncode = None
    with log_path.open("xb") as log:
        try:
            completed = subprocess.run(
                list(command),
                cwd=SPEAR_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=STAGE_TIMEOUTS[stage],
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            error_text = str(error)
        log.flush()
        os.fsync(log.fileno())
    execution = {
        "stage": stage,
        "command": list(command),
        "command_sha256": _json_sha256(list(command)),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "returncode": returncode,
        "timeout_seconds": STAGE_TIMEOUTS[stage],
        "error": error_text,
        "log_path": log_path,
    }
    return execution


def _run_stage_command(
    stage: str,
    command: Sequence[str],
    log_path: Path,
    execution_log: list[dict[str, Any]],
) -> dict[str, Any]:
    execution = _execute_command(stage, command, log_path)
    execution_log.append(execution)
    if execution["returncode"] != 0:
        raise AdmissionError(
            f"{stage} command failed returncode={execution['returncode']} "
            f"error={execution['error']}"
        )
    return execution


def _validate_watertight(
    manifest_path: Path,
    output_glb: Path,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = job.get("runtime")
    pixal_path = runtime["pixal_path"] if runtime else job["decision"]["pixal_path"]
    payload = _load_json(manifest_path, "watertight manifest")
    if (
        payload.get("schema")
        != "avengine_watertight_textured_runtime_proxy_v1"
        or payload.get("status")
        != "research_candidate_pending_static_and_animation_qa"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise AdmissionError("watertight manifest contract is not passed")
    try:
        emitter_contract.validate_file_record(
            payload.get("input"),
            label="watertight raw Pixal input",
            expected_path=pixal_path,
        )
        emitter_contract.validate_file_record(
            payload.get("attribute_input"),
            label="watertight attribute input",
            expected_path=pixal_path,
        )
        emitter_contract.validate_file_record(
            payload.get("output"),
            label="watertight output",
            expected_path=output_glb,
        )
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error
    final = payload.get("topology", {}).get("final")
    if (
        not isinstance(final, Mapping)
        or final.get("boundary_edges") != 0
        or final.get("wire_edges") != 0
        or final.get("nonmanifold_edges_over_two_faces") != 0
    ):
        raise AdmissionError("watertight boundary/nonmanifold gate is not passed")
    authority = payload.get("authority_contract")
    if (
        not isinstance(authority, Mapping)
        or authority.get("approved_skeleton_or_animation_touched") is not False
    ):
        raise AdmissionError("watertight static authority contract changed")
    actual = payload.get("parameters")
    if not isinstance(actual, Mapping):
        raise AdmissionError("watertight parameter readback is missing")
    for field, expected in job["watertight_parameters"].items():
        if actual.get(field) != expected:
            raise AdmissionError(f"watertight parameter readback changed: {field}")
    voxel_size = actual.get("voxel_size")
    if (
        isinstance(voxel_size, bool)
        or not isinstance(voxel_size, (int, float))
        or not math.isfinite(float(voxel_size))
        or float(voxel_size) <= 0.0
    ):
        raise AdmissionError("watertight voxel-size readback is invalid")
    _mesh_stats(output_glb, f"{job['instance_id']} watertight output")
    return payload


def _bind_heading_authority(
    job: Mapping[str, Any],
    watertight_glb: Path,
    output: Path,
) -> dict[str, Any]:
    runtime = job.get("runtime")
    source_path = runtime["heading_path"] if runtime else job["heading_authority_path"]
    source_record = runtime["heading_record"] if runtime else job["heading_authority_record"]
    pixal_path = runtime["pixal_path"] if runtime else job["decision"]["pixal_path"]
    decision = runtime["decision_payload"] if runtime else job["decision"]["payload"]
    if _file_record(source_path) != source_record:
        raise AdmissionError("heading authority changed during admission")
    source = _validate_heading_authority(
        source_path, decision, pixal_path,
    )
    bound = copy.deepcopy(source)
    bound["input_glb_sha256"] = _sha256_file(watertight_glb)
    if {
        key: value for key, value in bound.items() if key != "input_glb_sha256"
    } != {
        key: value for key, value in source.items() if key != "input_glb_sha256"
    }:
        raise AdmissionError("heading binding changed reviewed semantic fields")
    _write_json_exclusive(output, bound)
    return bound


def _validate_finalization(
    manifest_path: Path,
    output_glb: Path,
    watertight_glb: Path,
    job: Mapping[str, Any],
    bound_heading_path: Path,
) -> dict[str, Any]:
    try:
        payload = emitter_contract.validate_static_finalization(
            manifest_path, output_glb
        )
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error
    decision = job["decision"]["payload"]
    if any(
        payload.get(field) != decision[field]
        for field in ("instance_id", "request_sha256", "profile_sha256")
    ):
        raise AdmissionError("finalization identity lineage changed")
    try:
        emitter_contract.validate_file_record(
            payload.get("input"),
            label="finalization watertight input",
            expected_path=watertight_glb,
        )
        emitter_contract.validate_file_record(
            payload.get("heading", {}).get("evidence"),
            label="bound heading evidence",
            expected_path=bound_heading_path,
        )
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error
    if (
        payload.get("physical_scale", {}).get("passed") is not True
        or payload.get("grounding", {}).get("passed") is not True
        or payload.get("scene_readback", {}).get("no_rig_or_animation") is not True
    ):
        raise AdmissionError("finalization scale/ground/static readback is not passed")
    _mesh_stats(output_glb, f"{job['instance_id']} finalized output")
    return payload


def _bind_anchor_authority(
    job: Mapping[str, Any],
    final_glb: Path,
    finalization_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    runtime = job.get("runtime")
    source_path = runtime["anchor_path"] if runtime else job["anchor_authority_path"]
    source_record = runtime["anchor_record"] if runtime else job["anchor_authority_record"]
    pixal_path = runtime["pixal_path"] if runtime else job["decision"]["pixal_path"]
    decision = runtime["decision_payload"] if runtime else job["decision"]["payload"]
    if _file_record(source_path) != source_record:
        raise AdmissionError("anchor authority changed during admission")
    source = _validate_anchor_authority(
        source_path, decision, pixal_path,
    )
    bound = {
        "schema": emitter_contract.STATIC_ANCHOR_SPEC_SCHEMA,
        **{
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key not in {"schema", "input_glb_sha256"}
        },
        "finalized_glb_sha256": _sha256_file(final_glb),
        "finalization_manifest_sha256": _sha256_file(finalization_manifest),
    }
    preserved = {
        key: value
        for key, value in bound.items()
        if key
        not in {
            "schema",
            "finalized_glb_sha256",
            "finalization_manifest_sha256",
        }
    }
    if preserved != {
        key: value
        for key, value in source.items()
        if key not in {"schema", "input_glb_sha256"}
    }:
        raise AdmissionError("anchor binding changed reviewed semantic fields")
    _write_json_exclusive(output, bound)
    try:
        emitter_contract.validate_static_anchor_spec(
            output,
            finalization_path=finalization_manifest,
            input_glb=final_glb,
            finalization=_load_json(
                finalization_manifest, "static finalization manifest"
            ),
        )
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error
    return bound


def _validate_emitter_measurement(
    measurement_path: Path,
    marker_glb: Path,
    final_glb: Path,
    finalization_manifest: Path,
    bound_anchor_path: Path,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    finalization = _load_json(
        finalization_manifest, "static finalization manifest"
    )
    try:
        emitter_contract.validate_static_finalization(
            finalization_manifest, final_glb
        )
        emitter_contract.validate_static_anchor_spec(
            bound_anchor_path,
            finalization_path=finalization_manifest,
            input_glb=final_glb,
            finalization=finalization,
        )
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error
    payload = _load_json(measurement_path, "static emitter measurement")
    decision = job["decision"]["payload"]
    if (
        payload.get("schema") != emitter_contract.MEASUREMENT_SCHEMA
        or payload.get("status") != "measured_pending_marker_visual_review"
        or payload.get("asset_class") != STATIC_ASSET_CLASS
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("coordinate_system") != emitter_contract.COORDINATE_SYSTEM
        or any(
            payload.get(field) != decision[field]
            for field in ("instance_id", "request_sha256", "profile_sha256")
        )
    ):
        raise AdmissionError("static emitter measurement contract/identity changed")
    try:
        emitter_contract.validate_file_record(
            payload.get("input"),
            label="emitter finalized input",
            expected_path=final_glb,
        )
        emitter_contract.validate_file_record(
            payload.get("finalization_manifest"),
            label="emitter finalization manifest",
            expected_path=finalization_manifest,
        )
        emitter_contract.validate_file_record(
            payload.get("anchor_spec"),
            label="emitter bound anchor spec",
            expected_path=bound_anchor_path,
        )
        emitter_contract.validate_file_record(
            payload.get("marker_review", {}).get("marker_glb"),
            label="emitter marker GLB",
            expected_path=marker_glb,
        )
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error
    if payload.get("marker_review", {}).get("visual_review") != "pending":
        raise AdmissionError("emitter marker review must remain pending")
    anchor = payload.get("emitter_anchor")
    if (
        not isinstance(anchor, Mapping)
        or anchor.get("animation_required") is not False
        or anchor.get("asset_specific_not_class_template") is not True
    ):
        raise AdmissionError("static emitter anchor result contract changed")
    bound_anchor = _load_json(bound_anchor_path, "bound static emitter anchor")
    for field, expected in {
        "anchor_id": bound_anchor["anchor_id"],
        "anchor_type": bound_anchor["anchor_type"],
        "semantic_role": bound_anchor["semantic_role"],
        "method": bound_anchor["selection"]["method"],
        "aggregation": bound_anchor["selection"]["aggregation"],
    }.items():
        if anchor.get(field) != expected:
            raise AdmissionError(f"static emitter anchor {field} differs from authority")
    try:
        offset = emitter_contract.require_finite_vector(anchor.get("offset_m"), 3, "emitter offset")
        bounds = payload.get("asset_bounds")
        if not isinstance(bounds, Mapping):
            raise AdmissionError("static emitter bounds are missing")
        minimum = emitter_contract.require_finite_vector(bounds.get("minimum_m"), 3, "emitter bounds minimum")
        maximum = emitter_contract.require_finite_vector(bounds.get("maximum_m"), 3, "emitter bounds maximum")
        extent = emitter_contract.require_finite_vector(bounds.get("extent_m"), 3, "emitter bounds extent")
    except emitter_contract.EmitterContractError as error:
        raise AdmissionError(str(error)) from error
    for axis in range(3):
        if maximum[axis] <= minimum[axis] or not math.isclose(
            extent[axis], maximum[axis] - minimum[axis], abs_tol=1.0e-6
        ):
            raise AdmissionError("static emitter bounds are inconsistent")
        if not minimum[axis] - 1.0e-6 <= offset[axis] <= maximum[axis] + 1.0e-6:
            raise AdmissionError("static emitter offset lies outside finalized bounds")
    radius = payload.get("marker_review", {}).get("marker_radius_m")
    if (
        isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(float(radius))
        or float(radius) <= 0.0
    ):
        raise AdmissionError("static emitter marker radius is invalid")
    return payload


def _run_job(
    job: Mapping[str, Any],
    staging: Path,
    execution_log: list[dict[str, Any]],
    command_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_job_sources_unchanged(job)
    _assert_command_inputs_unchanged(command_inputs)
    commands, paths = build_stage_commands(
        job, staging, command_inputs=command_inputs
    )
    for directory in (
        paths["watertight_dir"],
        paths["finalization_dir"],
        paths["emitter_dir"],
    ):
        directory.mkdir(parents=True, exist_ok=False)

    _assert_command_inputs_unchanged(command_inputs)
    _run_stage_command(
        "watertight",
        commands["watertight"],
        paths["watertight_log"],
        execution_log,
    )
    _assert_command_inputs_unchanged(command_inputs)
    watertight = _validate_watertight(
        paths["watertight_manifest"], paths["watertight_glb"], job
    )
    heading = _bind_heading_authority(
        job, paths["watertight_glb"], paths["bound_heading"]
    )

    _assert_job_sources_unchanged(job)
    _assert_command_inputs_unchanged(command_inputs)
    _run_stage_command(
        "finalization",
        commands["finalization"],
        paths["finalization_log"],
        execution_log,
    )
    _assert_command_inputs_unchanged(command_inputs)
    finalization = _validate_finalization(
        paths["finalization_manifest"],
        paths["final_glb"],
        paths["watertight_glb"],
        job,
        paths["bound_heading"],
    )
    anchor = _bind_anchor_authority(
        job,
        paths["final_glb"],
        paths["finalization_manifest"],
        paths["bound_anchor"],
    )

    _assert_job_sources_unchanged(job)
    _assert_command_inputs_unchanged(command_inputs)
    _run_stage_command(
        "emitter_measurement",
        commands["emitter_measurement"],
        paths["emitter_log"],
        execution_log,
    )
    _assert_command_inputs_unchanged(command_inputs)
    emitter = _validate_emitter_measurement(
        paths["emitter_measurement"],
        paths["marker_glb"],
        paths["final_glb"],
        paths["finalization_manifest"],
        paths["bound_anchor"],
        job,
    )
    return {
        "job": job,
        "paths": paths,
        "commands": commands,
        "command_inputs": command_inputs,
        "executions": execution_log[-3:],
        "watertight": watertight,
        "bound_heading": heading,
        "finalization": finalization,
        "bound_anchor": anchor,
        "emitter": emitter,
    }


def _rebase_record_path(
    record: Mapping[str, Any],
    *,
    old_path: Path,
    new_path: Path,
    label: str,
) -> dict[str, Any]:
    if Path(record.get("path", "")).resolve() != old_path.resolve():
        raise AdmissionError(f"{label} path changed before publication")
    result = copy.deepcopy(dict(record))
    result["path"] = str(new_path.resolve())
    return result


def _rebase_job_manifests(
    result: dict[str, Any],
    staging: Path,
    output_root: Path,
) -> dict[str, Any]:
    paths = result["paths"]

    def public(path: Path) -> Path:
        return output_root / path.resolve().relative_to(staging.resolve())

    hashes_before = {
        "watertight_manifest": _sha256_file(paths["watertight_manifest"]),
        "finalization_manifest": _sha256_file(paths["finalization_manifest"]),
        "bound_anchor": _sha256_file(paths["bound_anchor"]),
        "emitter_measurement": _sha256_file(paths["emitter_measurement"]),
    }

    watertight = copy.deepcopy(result["watertight"])
    runtime = result["job"]["runtime"]
    watertight["input"] = _rebase_record_path(
        watertight["input"], old_path=runtime["pixal_path"],
        new_path=public(runtime["pixal_path"]), label="watertight frozen raw input",
    )
    watertight["attribute_input"] = _rebase_record_path(
        watertight["attribute_input"], old_path=runtime["pixal_path"],
        new_path=public(runtime["pixal_path"]), label="watertight frozen attribute input",
    )
    watertight["output"] = _rebase_record_path(
        watertight["output"],
        old_path=paths["watertight_glb"],
        new_path=public(paths["watertight_glb"]),
        label="watertight output",
    )
    _replace_json(paths["watertight_manifest"], watertight)

    bound_heading = _load_json(paths["bound_heading"], "bound static heading")
    bound_heading["review_artifact"] = _rebase_record_path(
        bound_heading["review_artifact"], old_path=runtime["heading_review_path"],
        new_path=public(runtime["heading_review_path"]), label="frozen heading review",
    )
    _replace_json(paths["bound_heading"], bound_heading)

    finalization = copy.deepcopy(result["finalization"])
    finalization["input"] = _rebase_record_path(
        finalization["input"],
        old_path=paths["watertight_glb"],
        new_path=public(paths["watertight_glb"]),
        label="finalization input",
    )
    finalization["output"] = _rebase_record_path(
        finalization["output"],
        old_path=paths["final_glb"],
        new_path=public(paths["final_glb"]),
        label="finalization output",
    )
    finalization["heading"]["evidence"] = _file_record(
        paths["bound_heading"], recorded_path=public(paths["bound_heading"])
    )
    _replace_json(paths["finalization_manifest"], finalization)

    anchor = _load_json(paths["bound_anchor"], "bound static anchor")
    anchor["review_evidence"] = _rebase_record_path(
        anchor["review_evidence"], old_path=runtime["anchor_review_path"],
        new_path=public(runtime["anchor_review_path"]), label="frozen anchor review",
    )
    anchor["finalization_manifest_sha256"] = _sha256_file(
        paths["finalization_manifest"]
    )
    _replace_json(paths["bound_anchor"], anchor)

    emitter = copy.deepcopy(result["emitter"])
    emitter["input"] = _rebase_record_path(
        emitter["input"],
        old_path=paths["final_glb"],
        new_path=public(paths["final_glb"]),
        label="emitter input",
    )
    emitter["finalization_manifest"] = _file_record(
        paths["finalization_manifest"],
        recorded_path=public(paths["finalization_manifest"]),
    )
    emitter["anchor_spec"] = _file_record(
        paths["bound_anchor"],
        recorded_path=public(paths["bound_anchor"]),
    )
    emitter["marker_review"]["marker_glb"] = _rebase_record_path(
        emitter["marker_review"]["marker_glb"],
        old_path=paths["marker_glb"],
        new_path=public(paths["marker_glb"]),
        label="emitter marker",
    )
    _replace_json(paths["emitter_measurement"], emitter)

    # Runtime-only authority copies are no longer referenced after their bound
    # descendants have been authenticated.  Do not publish stale staging paths.
    for key in ("decision_path", "heading_path", "anchor_path"):
        runtime[key].unlink(missing_ok=False)

    hashes_after = {
        "watertight_manifest": _sha256_file(paths["watertight_manifest"]),
        "finalization_manifest": _sha256_file(paths["finalization_manifest"]),
        "bound_anchor": _sha256_file(paths["bound_anchor"]),
        "emitter_measurement": _sha256_file(paths["emitter_measurement"]),
    }
    result.update(
        {
            "watertight": watertight,
            "finalization": finalization,
            "bound_anchor": anchor,
            "emitter": emitter,
            "path_rebinding": {
                "policy": "staging_to_atomic_public_root_paths_only_v1",
                "staging_root": str(staging),
                "public_root": str(output_root),
                "hashes_before": hashes_before,
                "hashes_after": hashes_after,
                "semantic_authority_fields_changed": False,
            },
        }
    )
    return result


def _public_command_path(value: str, staging_root: Path, public_root: Path) -> str:
    try:
        return str(
            public_root
            / Path(value).resolve().relative_to(staging_root.resolve())
        )
    except (TypeError, ValueError):
        return value


def _command_input_manifest_payload(
    *,
    instance_id: str,
    stage: str,
    command: Sequence[str],
    command_inputs: Mapping[str, Any],
    staging: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build the immutable byte identity of one actually invoked command."""

    if stage not in STAGE_PYTHON_DEPENDENCIES:
        raise AdmissionError(f"unknown command-input stage: {stage}")
    public_command = [
        _public_command_path(value, staging, output_root) for value in command
    ]
    tool = command_inputs["tools"][stage]
    _assert_file_unchanged(
        Path(tool["frozen_path"]), tool["frozen_record"], f"frozen {stage} tool"
    )

    def public_frozen_record(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        frozen = Path(snapshot["frozen_path"])
        _assert_file_unchanged(
            frozen, snapshot["frozen_record"], f"frozen command input {frozen.name}"
        )
        return {
            **copy.deepcopy(dict(snapshot["frozen_record"])),
            "path": _public_command_path(str(frozen), staging, output_root),
        }

    tool_record = public_frozen_record(tool)
    dependencies = {
        name: public_frozen_record(command_inputs["dependencies"][name])
        for name in STAGE_PYTHON_DEPENDENCIES[stage]
    }
    blender = command_inputs["blender"]
    payload: dict[str, Any] = {
        "schema": COMMAND_INPUT_MANIFEST_SCHEMA,
        "instance_id": instance_id,
        "stage": stage,
        "command": public_command,
        "command_sha256": _json_sha256(public_command),
        "blender": {
            "configured_path": blender["configured_path"],
            "resolved_path": blender["resolved_path"],
            "record": copy.deepcopy(blender["record"]),
        },
        "python_tool": tool_record,
        "python_dependencies": dependencies,
        "formal_dataset_registration_authorized": False,
    }
    python_index = (
        public_command.index("--python") if "--python" in public_command else -1
    )
    if (
        len(public_command) < 1
        or public_command[0] != payload["blender"]["record"]["path"]
        or python_index < 0
        or python_index + 1 >= len(public_command)
        or public_command[python_index + 1] != tool_record["path"]
    ):
        raise AdmissionError(f"{stage} command does not use its sealed inputs")
    payload["manifest_sha256"] = _hash_without(payload, "manifest_sha256")
    return payload


def _write_command_input_manifest(
    *,
    instance_id: str,
    stage: str,
    execution: Mapping[str, Any],
    command_inputs: Mapping[str, Any],
    staging: Path,
    output_root: Path,
    path: Path,
    public_path: Path,
) -> dict[str, Any]:
    payload = _command_input_manifest_payload(
        instance_id=instance_id,
        stage=stage,
        command=execution["command"],
        command_inputs=command_inputs,
        staging=staging,
        output_root=output_root,
    )
    _write_json_exclusive(path, payload)
    return _file_record(path, recorded_path=public_path)


def _stage_receipt(
    *,
    instance_id: str,
    stage: str,
    execution: Mapping[str, Any],
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    validation: Mapping[str, Any],
    command_input_manifest: Mapping[str, Any],
    path: Path,
    public_path: Path,
) -> dict[str, Any]:
    log_path = execution["log_path"]
    public_log = public_path.parent / log_path.name
    staging_root = path.parents[3]
    public_root = public_path.parents[3]

    payload: dict[str, Any] = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "instance_id": instance_id,
        "stage": stage,
        "status": "passed",
        "command": [
            _public_command_path(value, staging_root, public_root)
            for value in execution["command"]
        ],
        "command_sha256": _json_sha256(
            [
                _public_command_path(value, staging_root, public_root)
                for value in execution["command"]
            ]
        ),
        "command_input_manifest": copy.deepcopy(dict(command_input_manifest)),
        "execution": {
            key: copy.deepcopy(value)
            for key, value in execution.items()
            if key not in {"command", "command_sha256", "log_path", "stage"}
        },
        "log": _file_record(log_path, recorded_path=public_log),
        "inputs": copy.deepcopy(dict(inputs)),
        "outputs": copy.deepcopy(dict(outputs)),
        "validation": copy.deepcopy(dict(validation)),
        "formal_dataset_registration_authorized": False,
    }
    payload["receipt_sha256"] = _hash_without(payload, "receipt_sha256")
    _write_json_exclusive(path, payload)
    return payload


def _publish_job_receipts(
    result: dict[str, Any],
    staging: Path,
    output_root: Path,
) -> dict[str, Any]:
    job = result["job"]
    paths = result["paths"]

    def public(path: Path) -> Path:
        return output_root / path.resolve().relative_to(staging.resolve())

    executions = {item["stage"]: item for item in result["executions"]}
    _assert_command_inputs_unchanged(result["command_inputs"])
    command_input_manifests = {
        "watertight": _write_command_input_manifest(
            instance_id=job["instance_id"],
            stage="watertight",
            execution=executions["watertight"],
            command_inputs=result["command_inputs"],
            staging=staging,
            output_root=output_root,
            path=paths["watertight_command_inputs"],
            public_path=public(paths["watertight_command_inputs"]),
        ),
        "finalization": _write_command_input_manifest(
            instance_id=job["instance_id"],
            stage="finalization",
            execution=executions["finalization"],
            command_inputs=result["command_inputs"],
            staging=staging,
            output_root=output_root,
            path=paths["finalization_command_inputs"],
            public_path=public(paths["finalization_command_inputs"]),
        ),
        "emitter_measurement": _write_command_input_manifest(
            instance_id=job["instance_id"],
            stage="emitter_measurement",
            execution=executions["emitter_measurement"],
            command_inputs=result["command_inputs"],
            staging=staging,
            output_root=output_root,
            path=paths["emitter_command_inputs"],
            public_path=public(paths["emitter_command_inputs"]),
        ),
    }
    stage_receipts = {}
    watertight_receipt_path = paths["watertight_dir"] / "stage_receipt.json"
    stage_receipts["watertight"] = _stage_receipt(
        instance_id=job["instance_id"],
        stage="watertight",
        execution=executions["watertight"],
        inputs={
            "decision": copy.deepcopy(job["decision"]["record"]),
            "pixal_output": copy.deepcopy(job["decision"]["pixal_record"]),
            "watertight_parameters": copy.deepcopy(
                job["watertight_parameters"]
            ),
        },
        outputs={
            "watertight_glb": _file_record(
                paths["watertight_glb"],
                recorded_path=public(paths["watertight_glb"]),
            ),
            "watertight_manifest": _file_record(
                paths["watertight_manifest"],
                recorded_path=public(paths["watertight_manifest"]),
            ),
        },
        validation={
            "boundary_edges": 0,
            "nonmanifold_edges_over_two_faces": 0,
            "no_rig_or_animation": True,
        },
        command_input_manifest=command_input_manifests["watertight"],
        path=watertight_receipt_path,
        public_path=public(watertight_receipt_path),
    )
    finalization_receipt_path = paths["finalization_dir"] / "stage_receipt.json"
    stage_receipts["finalization"] = _stage_receipt(
        instance_id=job["instance_id"],
        stage="finalization",
        execution=executions["finalization"],
        inputs={
            "watertight_glb": _file_record(
                paths["watertight_glb"],
                recorded_path=public(paths["watertight_glb"]),
            ),
            "watertight_manifest": _file_record(
                paths["watertight_manifest"],
                recorded_path=public(paths["watertight_manifest"]),
            ),
            "heading_authority": copy.deepcopy(job["heading_authority_record"]),
            "bound_heading_evidence": _file_record(
                paths["bound_heading"],
                recorded_path=public(paths["bound_heading"]),
            ),
        },
        outputs={
            "finalized_glb": _file_record(
                paths["final_glb"], recorded_path=public(paths["final_glb"])
            ),
            "finalization_manifest": _file_record(
                paths["finalization_manifest"],
                recorded_path=public(paths["finalization_manifest"]),
            ),
        },
        validation={
            "heading_passed": True,
            "physical_scale_passed": True,
            "grounding_passed": True,
            "no_rig_or_animation": True,
        },
        command_input_manifest=command_input_manifests["finalization"],
        path=finalization_receipt_path,
        public_path=public(finalization_receipt_path),
    )
    emitter_receipt_path = paths["emitter_dir"] / "stage_receipt.json"
    stage_receipts["emitter_measurement"] = _stage_receipt(
        instance_id=job["instance_id"],
        stage="emitter_measurement",
        execution=executions["emitter_measurement"],
        inputs={
            "finalized_glb": _file_record(
                paths["final_glb"], recorded_path=public(paths["final_glb"])
            ),
            "finalization_manifest": _file_record(
                paths["finalization_manifest"],
                recorded_path=public(paths["finalization_manifest"]),
            ),
            "anchor_authority": copy.deepcopy(job["anchor_authority_record"]),
            "bound_anchor_spec": _file_record(
                paths["bound_anchor"],
                recorded_path=public(paths["bound_anchor"]),
            ),
        },
        outputs={
            "emitter_measurement": _file_record(
                paths["emitter_measurement"],
                recorded_path=public(paths["emitter_measurement"]),
            ),
            "marker_glb": _file_record(
                paths["marker_glb"], recorded_path=public(paths["marker_glb"])
            ),
        },
        validation={"marker_visual_review": "pending"},
        command_input_manifest=command_input_manifests["emitter_measurement"],
        path=emitter_receipt_path,
        public_path=public(emitter_receipt_path),
    )

    receipt_records = {
        "watertight": _relative_record(watertight_receipt_path, staging),
        "finalization": _relative_record(finalization_receipt_path, staging),
        "emitter_measurement": _relative_record(emitter_receipt_path, staging),
    }
    receipt: dict[str, Any] = {
        "schema": JOB_RECEIPT_SCHEMA,
        "status": "passed_pending_emitter_marker_review",
        "asset_class": STATIC_ASSET_CLASS,
        "route": STATIC_ROUTE,
        "instance_id": job["instance_id"],
        "request_sha256": job["decision"]["payload"]["request_sha256"],
        "profile_sha256": job["decision"]["payload"]["profile_sha256"],
        "decision": copy.deepcopy(job["decision"]["record"]),
        "pixal_output": copy.deepcopy(job["decision"]["pixal_record"]),
        "heading_authority": copy.deepcopy(job["heading_authority_record"]),
        "anchor_authority": copy.deepcopy(job["anchor_authority_record"]),
        "bound_heading_evidence": _file_record(
            paths["bound_heading"],
            recorded_path=public(paths["bound_heading"]),
        ),
        "bound_anchor_spec": _file_record(
            paths["bound_anchor"],
            recorded_path=public(paths["bound_anchor"]),
        ),
        "watertight_glb": _file_record(
            paths["watertight_glb"],
            recorded_path=public(paths["watertight_glb"]),
        ),
        "finalized_glb": _file_record(
            paths["final_glb"], recorded_path=public(paths["final_glb"])
        ),
        "emitter_measurement": _file_record(
            paths["emitter_measurement"],
            recorded_path=public(paths["emitter_measurement"]),
        ),
        "emitter_marker_glb": _file_record(
            paths["marker_glb"], recorded_path=public(paths["marker_glb"])
        ),
        "stage_receipts": receipt_records,
        "path_rebinding": copy.deepcopy(result["path_rebinding"]),
        "marker_review": "pending",
        "next_gate": "emitter_marker_visual_review",
        "formal_dataset_registration_authorized": False,
    }
    receipt["receipt_sha256"] = _hash_without(receipt, "receipt_sha256")
    _write_json_exclusive(paths["job_receipt"], receipt)
    return {
        "instance_id": job["instance_id"],
        "request_sha256": receipt["request_sha256"],
        "profile_sha256": receipt["profile_sha256"],
        "status": receipt["status"],
        "receipt_sha256": receipt["receipt_sha256"],
        "job_receipt": _relative_record(paths["job_receipt"], staging),
        "marker_review": "pending",
    }


def _validate_published_job(
    result: Mapping[str, Any],
    output_root: Path,
) -> None:
    instance_id = result["job"]["instance_id"]
    paths = _job_paths(output_root, instance_id)
    _validate_watertight(
        paths["watertight_manifest"],
        paths["watertight_glb"],
        result["job"],
    )
    _validate_finalization(
        paths["finalization_manifest"],
        paths["final_glb"],
        paths["watertight_glb"],
        result["job"],
        paths["bound_heading"],
    )
    _validate_emitter_measurement(
        paths["emitter_measurement"],
        paths["marker_glb"],
        paths["final_glb"],
        paths["finalization_manifest"],
        paths["bound_anchor"],
        result["job"],
    )
    receipt = _load_json(paths["job_receipt"], f"{instance_id} admission receipt")
    if (
        receipt.get("schema") != JOB_RECEIPT_SCHEMA
        or receipt.get("receipt_sha256")
        != _hash_without(receipt, "receipt_sha256")
        or receipt.get("marker_review") != "pending"
        or receipt.get("formal_dataset_registration_authorized") is not False
    ):
        raise AdmissionError(f"{instance_id} published admission receipt changed")
    for record in receipt.get("stage_receipts", {}).values():
        stage_path = _resolve_file_record(
            record,
            base=output_root,
            label=f"{instance_id} stage receipt",
            require_within_base=True,
        )
        stage = _load_json(stage_path, f"{instance_id} stage receipt")
        if (
            stage.get("schema") != STAGE_RECEIPT_SCHEMA
            or stage.get("receipt_sha256")
            != _hash_without(stage, "receipt_sha256")
            or stage.get("formal_dataset_registration_authorized") is not False
        ):
            raise AdmissionError(f"{instance_id} stage receipt changed")
        _validate_published_command_input_manifest(
            stage.get("command_input_manifest"),
            instance_id=instance_id,
            stage=stage.get("stage"),
            command=stage.get("command"),
            output_root=output_root,
        )


def _validate_published_command_input_manifest(
    record: Any,
    *,
    instance_id: str,
    stage: Any,
    command: Any,
    output_root: Path,
) -> None:
    if not isinstance(stage, str) or stage not in STAGE_PYTHON_DEPENDENCIES:
        raise AdmissionError(f"{instance_id} stage command-input stage is invalid")
    manifest_path = _resolve_file_record(
        record,
        base=output_root,
        label=f"{instance_id} {stage} command-input manifest",
        require_within_base=True,
    )
    manifest = _load_json(manifest_path, f"{instance_id} {stage} command-input manifest")
    required = {
        "schema",
        "instance_id",
        "stage",
        "command",
        "command_sha256",
        "blender",
        "python_tool",
        "python_dependencies",
        "formal_dataset_registration_authorized",
        "manifest_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("schema") != COMMAND_INPUT_MANIFEST_SCHEMA
        or manifest.get("instance_id") != instance_id
        or manifest.get("stage") != stage
        or manifest.get("command") != command
        or manifest.get("command_sha256") != _json_sha256(command)
        or manifest.get("manifest_sha256") != _hash_without(manifest, "manifest_sha256")
        or manifest.get("formal_dataset_registration_authorized") is not False
    ):
        raise AdmissionError(f"{instance_id} {stage} command-input manifest changed")
    blender = manifest.get("blender")
    if (
        not isinstance(blender, Mapping)
        or set(blender) != {"configured_path", "resolved_path", "record"}
        or not isinstance(blender.get("configured_path"), str)
        or not isinstance(blender.get("resolved_path"), str)
        or not Path(blender["configured_path"]).is_absolute()
        or not Path(blender["resolved_path"]).is_absolute()
    ):
        raise AdmissionError(f"{instance_id} {stage} Blender identity is invalid")
    blender_path = _resolve_file_record(
        blender["record"],
        base=output_root,
        label=f"{instance_id} {stage} Blender binary",
        require_within_base=False,
    )
    if blender_path != Path(blender["resolved_path"]).resolve():
        raise AdmissionError(f"{instance_id} {stage} Blender record path changed")
    tool_path = output_root / ".runtime_commands" / "tools" / _command_tool_sources()[stage].name
    resolved_tool = _resolve_file_record(
        manifest.get("python_tool"),
        base=output_root,
        label=f"{instance_id} {stage} frozen Python tool",
        require_within_base=True,
    )
    if resolved_tool != tool_path.resolve():
        raise AdmissionError(f"{instance_id} {stage} frozen Python tool path changed")
    dependencies = manifest.get("python_dependencies")
    expected_dependencies = set(STAGE_PYTHON_DEPENDENCIES[stage])
    if not isinstance(dependencies, Mapping) or set(dependencies) != expected_dependencies:
        raise AdmissionError(f"{instance_id} {stage} Python dependency coverage changed")
    for name in expected_dependencies:
        dependency_path = output_root / ".runtime_commands" / "tools" / name
        resolved = _resolve_file_record(
            dependencies[name],
            base=output_root,
            label=f"{instance_id} {stage} frozen dependency {name}",
            require_within_base=True,
        )
        if resolved != dependency_path.resolve():
            raise AdmissionError(
                f"{instance_id} {stage} frozen dependency path changed: {name}"
            )
    python_index = command.index("--python") if isinstance(command, list) and "--python" in command else -1
    if (
        not isinstance(command, list)
        or len(command) < 2
        or command[0] != blender["record"]["path"]
        or python_index < 0
        or python_index + 1 >= len(command)
        or command[python_index + 1] != manifest["python_tool"]["path"]
    ):
        raise AdmissionError(f"{instance_id} {stage} command/input binding changed")


def _validate_published_batch(output_root: Path) -> None:
    """Rehash published stage command inputs from their receipt seals."""

    batch_path = output_root / "static_object_admission_batch_manifest.json"
    batch = _load_json(batch_path, "published static-object admission batch")
    if (
        batch.get("schema") != BATCH_SCHEMA
        or batch.get("batch_sha256") != _hash_without(batch, "batch_sha256")
        or not isinstance(batch.get("jobs"), list)
    ):
        raise AdmissionError("published static-object admission batch changed")
    for job in batch["jobs"]:
        if not isinstance(job, Mapping) or not isinstance(job.get("instance_id"), str):
            raise AdmissionError("published static-object admission job is invalid")
        instance_id = job["instance_id"]
        receipt_path = _resolve_file_record(
            job.get("job_receipt"),
            base=output_root,
            label=f"{instance_id} published job receipt",
            require_within_base=True,
        )
        receipt = _load_json(receipt_path, f"{instance_id} published job receipt")
        if (
            receipt.get("schema") != JOB_RECEIPT_SCHEMA
            or receipt.get("instance_id") != instance_id
            or receipt.get("receipt_sha256") != _hash_without(receipt, "receipt_sha256")
            or not isinstance(receipt.get("stage_receipts"), Mapping)
            or set(receipt["stage_receipts"]) != set(STAGE_PYTHON_DEPENDENCIES)
        ):
            raise AdmissionError(f"{instance_id} published job receipt changed")
        for stage_name, stage_record in receipt["stage_receipts"].items():
            stage_path = _resolve_file_record(
                stage_record,
                base=output_root,
                label=f"{instance_id} {stage_name} published stage receipt",
                require_within_base=True,
            )
            stage = _load_json(stage_path, f"{instance_id} {stage_name} published stage receipt")
            if (
                stage.get("schema") != STAGE_RECEIPT_SCHEMA
                or stage.get("stage") != stage_name
                or stage.get("receipt_sha256") != _hash_without(stage, "receipt_sha256")
            ):
                raise AdmissionError(f"{instance_id} {stage_name} stage receipt changed")
            _validate_published_command_input_manifest(
                stage.get("command_input_manifest"),
                instance_id=instance_id,
                stage=stage_name,
                command=stage.get("command"),
                output_root=output_root,
            )


def _validate_staged_rebase(result: Mapping[str, Any], staging: Path, output_root: Path) -> None:
    """Validate rebased public paths against their still-private staged bytes."""

    def public(path: Path) -> Path:
        return output_root / path.resolve().relative_to(staging.resolve())

    def record_at(record: Mapping[str, Any], physical: Path, label: str) -> None:
        if Path(record.get("path", "")).resolve() != public(physical).resolve():
            raise AdmissionError(f"{label} public path rebasing changed")
        if (
            record.get("sha256") != _sha256_file(physical)
            or record.get("size_bytes") != physical.stat().st_size
        ):
            raise AdmissionError(f"{label} public record hash/size changed")

    paths = result["paths"]
    runtime = result["job"]["runtime"]
    watertight = _load_json(paths["watertight_manifest"], "rebased watertight manifest")
    record_at(watertight["input"], runtime["pixal_path"], "watertight frozen input")
    record_at(watertight["attribute_input"], runtime["pixal_path"], "watertight frozen attribute input")
    record_at(watertight["output"], paths["watertight_glb"], "watertight output")
    heading = _load_json(paths["bound_heading"], "rebased heading authority")
    record_at(heading["review_artifact"], runtime["heading_review_path"], "heading review")
    finalization = _load_json(paths["finalization_manifest"], "rebased finalization manifest")
    record_at(finalization["input"], paths["watertight_glb"], "finalization input")
    record_at(finalization["output"], paths["final_glb"], "finalization output")
    record_at(finalization["heading"]["evidence"], paths["bound_heading"], "finalization heading")
    anchor = _load_json(paths["bound_anchor"], "rebased anchor authority")
    record_at(anchor["review_evidence"], runtime["anchor_review_path"], "anchor review")
    if anchor["finalization_manifest_sha256"] != _sha256_file(paths["finalization_manifest"]):
        raise AdmissionError("rebased anchor finalization hash changed")
    emitter = _load_json(paths["emitter_measurement"], "rebased emitter measurement")
    record_at(emitter["input"], paths["final_glb"], "emitter input")
    record_at(emitter["finalization_manifest"], paths["finalization_manifest"], "emitter finalization")
    record_at(emitter["anchor_spec"], paths["bound_anchor"], "emitter anchor")
    record_at(emitter["marker_review"]["marker_glb"], paths["marker_glb"], "emitter marker")
    receipt = _load_json(paths["job_receipt"], "staged admission receipt")
    if (
        receipt.get("schema") != JOB_RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != _hash_without(receipt, "receipt_sha256")
        or receipt.get("decision") != result["job"]["decision"]["record"]
        or receipt.get("pixal_output") != result["job"]["decision"]["pixal_record"]
        or receipt.get("heading_authority") != result["job"]["heading_authority_record"]
        or receipt.get("anchor_authority") != result["job"]["anchor_authority_record"]
        or receipt.get("formal_dataset_registration_authorized") is not False
    ):
        raise AdmissionError("staged admission receipt lineage changed")
    expected_stages = {"watertight", "finalization", "emitter_measurement"}
    if set(receipt.get("stage_receipts", {})) != expected_stages:
        raise AdmissionError("staged admission receipt stage coverage changed")
    executions = {item["stage"]: item for item in result["executions"]}
    command_manifest_paths = {
        "watertight": paths["watertight_command_inputs"],
        "finalization": paths["finalization_command_inputs"],
        "emitter_measurement": paths["emitter_command_inputs"],
    }
    for stage_record in receipt["stage_receipts"].values():
        stage_path = _resolve_file_record(
            stage_record, base=staging, label="staged stage receipt", require_within_base=True
        )
        stage = _load_json(stage_path, "staged stage receipt")
        if (
            stage.get("schema") != STAGE_RECEIPT_SCHEMA
            or stage.get("receipt_sha256") != _hash_without(stage, "receipt_sha256")
            or stage.get("formal_dataset_registration_authorized") is not False
        ):
            raise AdmissionError("staged stage receipt changed")
        stage_name = stage.get("stage")
        if stage_name not in expected_stages:
            raise AdmissionError("staged stage receipt command-input stage changed")
        manifest_path = command_manifest_paths[stage_name]
        record_at(
            stage.get("command_input_manifest"),
            manifest_path,
            f"{stage_name} command-input manifest",
        )
        manifest = _load_json(manifest_path, f"{stage_name} command-input manifest")
        expected_manifest = _command_input_manifest_payload(
            instance_id=result["job"]["instance_id"],
            stage=stage_name,
            command=executions[stage_name]["command"],
            command_inputs=result["command_inputs"],
            staging=staging,
            output_root=output_root,
        )
        if manifest != expected_manifest or stage.get("command") != manifest["command"]:
            raise AdmissionError("staged command-input manifest changed")


def _preserve_failure(
    work_root: Path,
    output_root: Path,
    inputs: Mapping[str, Any],
    execution_log: Sequence[Mapping[str, Any]],
    failed_instance: str | None,
    failed_stage: str | None,
    error: BaseException,
    *,
    work_root_is_sealed: bool,
) -> Path:
    failure_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    failure_root = output_root.parent / (
        f"{output_root.name}.failed_{failure_stamp}_{os.getpid()}"
    )
    root = work_root
    payload = {
        "schema": FAILURE_SCHEMA,
        "status": "failed_preserved_as_evidence",
        "failed_at": _utc_now(),
        "failed_instance_id": failed_instance,
        "failed_stage": failed_stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "decision_batch": copy.deepcopy(inputs["decision_batch_record"]),
        "plan": copy.deepcopy(inputs["plan_record"]),
        "executed_commands": [
            {
                key: copy.deepcopy(value)
                for key, value in execution.items()
                if key != "log_path"
            }
            for execution in execution_log
        ],
        "relocation": {
            "working_root": str(root),
            "failure_root": str(failure_root),
        },
        "formal_dataset_registration_authorized": False,
    }
    if work_root_is_sealed:
        # A RENAME_NOREPLACE collision happens after the private root has been
        # sealed.  Never reopen that root just to add a manifest: copy its
        # evidence into a fresh writable sibling, add the manifest there, then
        # seal and atomically publish that separate failure root.
        root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.failed.",
                suffix=".staging",
                dir=output_root.parent,
            )
        )
        shutil.copytree(work_root, root, dirs_exist_ok=True, symlinks=True)
        root.chmod(0o755)
    else:
        root.mkdir(parents=True, exist_ok=True)
    payload["relocation"]["failure_working_root"] = str(root)
    failure_path = root / "admission_failure_manifest.json"
    if not failure_path.exists() and not failure_path.is_symlink():
        _write_json_exclusive(failure_path, payload)
    immutable._seal_readonly_tree(root)
    _rename_noreplace(root, failure_root)
    return failure_root


def run_admission(
    decision_batch_path: Path,
    plan_path: Path,
    output_root: Path,
    *,
    validate_only: bool = False,
) -> Path | None:
    inputs = validate_admission_inputs(decision_batch_path, plan_path)
    if validate_only:
        return None
    if not BLENDER.is_file() or not all(
        path.is_file() for path in (WATERTIGHT_TOOL, FINALIZER_TOOL, EMITTER_TOOL)
    ):
        raise AdmissionError("pinned Blender/static admission tools are missing")
    requested_output_root = Path(output_root)
    # Resolve only after inspecting the requested leaf.  In particular,
    # ``Path.resolve`` hides a dangling final symlink and would otherwise turn
    # it into an unintended publication target.
    if requested_output_root.is_symlink():
        raise AdmissionError(
            f"output root must not be a symlink: {requested_output_root}"
        )
    output_parent = requested_output_root.parent.resolve()
    output_root = output_parent / requested_output_root.name
    if output_root.exists() or output_root.is_symlink():
        raise AdmissionError(f"refusing to replace output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    results: list[dict[str, Any]] = []
    execution_log: list[dict[str, Any]] = []
    failed_instance = None
    failed_stage = None
    published = False
    staging_sealed = False
    try:
        _snapshot_inputs(inputs, staging)
        command_inputs = _snapshot_command_inputs(staging)
        for instance_id in sorted(inputs["jobs"]):
            failed_instance = instance_id
            before = len(execution_log)
            try:
                result = _run_job(
                    inputs["jobs"][instance_id],
                    staging,
                    execution_log,
                    command_inputs,
                )
            except Exception:
                latest = execution_log[-1] if len(execution_log) > before else None
                failed_stage = latest["stage"] if latest else "pre_stage_validation"
                raise
            results.append(result)

        rebased = [
            _rebase_job_manifests(result, staging, output_root)
            for result in results
        ]
        job_records = [
            _publish_job_receipts(result, staging, output_root)
            for result in rebased
        ]
        batch: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "status": "passed_all_instances_pending_emitter_marker_review",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "asset_class": STATIC_ASSET_CLASS,
            "route": STATIC_ROUTE,
            "decision_batch": {
                **copy.deepcopy(inputs["decision_batch_record"]),
                "decision_batch_sha256": inputs["decision_batch"][
                    "decision_batch_sha256"
                ],
            },
            "plan": {
                **copy.deepcopy(inputs["plan_record"]),
                "plan_sha256": inputs["plan"]["plan_sha256"],
            },
            "job_count": len(job_records),
            "passed_count": len(job_records),
            "failed_count": 0,
            "jobs": sorted(job_records, key=lambda item: item["instance_id"]),
            "marker_review": {
                "status": "pending",
                "next_gate": "emitter_marker_visual_review",
            },
            "automatic_checks": {
                "all_decisions_and_plan_authorities_reauthenticated": True,
                "all_watertight_boundary_and_nonmanifold_gates_passed": True,
                "all_heading_scale_and_grounding_gates_passed": True,
                "all_static_emitter_measurements_hash_bound": True,
                "all_marker_reviews_pending": True,
                "fail_first_no_partial_batch_published": True,
                "no_formal_registration_authorized": True,
                "overall": "passed",
            },
        }
        batch["batch_sha256"] = _hash_without(batch, "batch_sha256")
        manifest = staging / "static_object_admission_batch_manifest.json"
        _write_json_exclusive(manifest, batch)
        # All semantic/output checks run while the tree is private.  The final
        # rename is therefore only publication, never a validation boundary.
        for result in rebased:
            _validate_staged_rebase(result, staging, output_root)
        staged_batch = _load_json(manifest, "staged static-object admission batch")
        if (
            staged_batch.get("batch_sha256") != _hash_without(staged_batch, "batch_sha256")
            or staged_batch.get("formal_dataset_registration_authorized") is not False
            or staged_batch.get("marker_review", {}).get("status") != "pending"
        ):
            raise AdmissionError("staged static-object admission batch changed")
        staging_sealed = True
        immutable._seal_readonly_tree(staging)
        _rename_noreplace(staging, output_root)
        published = True
        _validate_published_batch(output_root)
        return output_root / manifest.name
    except Exception as error:
        root = output_root if published and output_root.exists() else staging
        try:
            failure_root = _preserve_failure(
                root,
                output_root,
                inputs,
                execution_log,
                failed_instance,
                failed_stage,
                error,
                work_root_is_sealed=staging_sealed,
            )
        except (AdmissionError, contracts.ContractError, OSError) as preserve_error:
            raise AdmissionError(
                f"{error}; additionally failed to preserve evidence: {preserve_error}"
            ) from error
        raise AdmissionError(
            f"{error}; failure_evidence={failure_root}"
        ) from error


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-batch", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = run_admission(
            args.decision_batch,
            args.plan,
            args.output_root,
            validate_only=args.validate_only,
        )
        if args.validate_only:
            inputs = validate_admission_inputs(args.decision_batch, args.plan)
            print(
                "CONTROLLED_STATIC_OBJECT_ADMISSION_VALID "
                f"jobs={len(inputs['jobs'])}"
            )
            return 0
        batch = contracts.load_json(result)
    except (
        AdmissionError,
        contracts.ContractError,
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"CONTROLLED_STATIC_OBJECT_ADMISSION_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_STATIC_OBJECT_ADMISSION_OK "
        f"jobs={batch['job_count']} output={result}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
