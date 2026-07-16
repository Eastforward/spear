#!/usr/bin/env python3
"""Immutable identity, lineage, output-root, and branch contract for Route-2 humans."""

from __future__ import annotations

import copy
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any, Mapping

from tools import route2_human_contract_common as common


SCHEMA = "route2_human_instance_contract_v1"
FILENAME = "route2_human_instance_contract_v1.json"
CASE_KINDS = {"base_avatar", "attribute_instance"}
COORDINATE_FRAME = {"front": "negative-y", "up": "positive-z", "floor_z_m": 0.0}
BASE_AVATAR_IDS = frozenset(
    {"rocketbox_male_adult_01", "rocketbox_female_adult_01"}
)
ATTRIBUTE_CASE_BASE = {
    "tall_man": "rocketbox_male_adult_01",
    "short_woman": "rocketbox_female_adult_01",
    "glasses": "rocketbox_male_adult_01",
    "hat": "rocketbox_female_adult_01",
    "short_sleeve_color": "rocketbox_male_adult_01",
    "trousers": "rocketbox_female_adult_01",
    "shoes": "rocketbox_male_adult_01",
}
BASE_LINEAGE_ROLES = {
    "source_image": "approved_soft_t_source",
    "flux_candidate": "flux2_candidate_image",
    "flux_manifest": "flux2_candidate_manifest",
    "source_review": "source_reference_review",
    "pixal_input_rgba": "pixal_input_rgba",
    "pixal_manifest": "pixal_manifest",
    "pixal_pbr_glb": "pixal_pbr_glb",
}
ATTRIBUTE_LINEAGE_ROLES = {
    "base_qualified_candidate": "base_qualified_candidate",
    "attribute_candidate_manifest": "attribute_candidate_manifest",
    "attribute_agent_decision": "attribute_agent_decision",
    "candidate_rgba": "agent_accepted_rgba",
    "pixal_job": "pixal_attribute_job",
    "pixal_attempt": "pixal_attempt_ledger",
    "pixal_manifest": "pixal_manifest",
    "pixal_pbr_glb": "pixal_pbr_glb",
}
FLUX2_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
PIXAL3D_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
DINO_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
PIXAL3D_LICENSE_SHA256 = "31d37e9c4fee1e0cd2196bccd592e8a2c30bfa17ea177d70ad25f977ba6bd9c0"
DINO_LICENSE_SHA256 = "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"
MODEL_ROOT = Path("/data/models")
PIXAL3D_SNAPSHOT_RELATIVE = (
    Path("hub/models--TencentARC--Pixal3D/snapshots") / PIXAL3D_REVISION
)
DINO_SNAPSHOT_RELATIVE = (
    Path("hub/models--camenduru--dinov3-vitl16-pretrain-lvd1689m/snapshots")
    / DINO_REVISION
)
MODEL_SNAPSHOT_CONTRACTS = {
    PIXAL3D_REVISION: {
        "relative_path": PIXAL3D_SNAPSHOT_RELATIVE,
        "license": "LICENSE",
        "license_sha256": PIXAL3D_LICENSE_SHA256,
        "required_files": {
            "pipeline.json",
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        },
    },
    DINO_REVISION: {
        "relative_path": DINO_SNAPSHOT_RELATIVE,
        "license": "LICENSE.md",
        "license_sha256": DINO_LICENSE_SHA256,
        "required_files": {
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
        },
    },
}
PIXAL_PYTHON_EXECUTABLE = Path(
    "/data/jzy/miniconda3/envs/avengine-3dgen/bin/python3.10"
)
_MODEL_SNAPSHOT_CACHE: dict[
    tuple[str, str], tuple[tuple[Any, ...], dict[str, Any]]
] = {}
PIXAL_ATTRIBUTE_ATTEMPT_SCHEMA = "pixal3d_human_attribute_attempt_v1"
PIXAL_ATTRIBUTE_START_SCHEMA = "pixal3d_human_attribute_attempt_start_v1"
PIXAL_ATTRIBUTE_EXECUTION_LOG_SCHEMA = "pixal3d_human_attribute_execution_log_v1"
PIXAL_ATTRIBUTE_FAILURE_BUNDLE_SCHEMA = "pixal3d_human_attribute_failure_bundle_v1"
PIXAL_EXECUTOR_FIELDS = frozenset(
    {
        "kind",
        "argv",
        "execution_authorized",
        "atomic_no_replace",
        "path",
        "sha256",
        "size_bytes",
    }
)
PIXAL_ATTRIBUTE_JOB_FIELDS = frozenset(
    {
        "schema",
        "case_id",
        "asset_id",
        "base_asset_id",
        "state_classification",
        "input_rgba",
        "candidate_manifest",
        "agent_2d_decision",
        "model_revision",
        "dino_revision",
        "parameters",
        "wrapper",
        "output_glb",
        "output_manifest",
        "output_policy",
        "executor",
    }
)
PIXAL_ATTRIBUTE_ATTEMPT_FIELDS = frozenset(
    {
        "schema",
        "attempt_id",
        "status",
        "case_id",
        "asset_id",
        "base_avatar_id",
        "job",
        "executor",
        "start_ledger",
        "execution_log",
        "execution_guard",
        "argv",
        "environment",
        "wrapper",
        "started_at_utc",
        "finished_at_utc",
        "returncode",
        "preflight_reauthenticated",
        "postflight_reauthenticated",
        "staging",
        "publication",
        "model_inventory",
        "licenses",
        "output_glb",
        "output_manifest",
        "failure_evidence",
    }
)
PIXAL_ATTRIBUTE_START_FIELDS = frozenset(
    {
        "schema",
        "attempt_id",
        "status",
        "case_id",
        "asset_id",
        "base_avatar_id",
        "job",
        "executor",
        "execution_guard_before",
        "argv",
        "started_at_utc",
        "staging",
        "publication_policy",
    }
)
PIXAL_ATTRIBUTE_EXECUTION_LOG_FIELDS = frozenset(
    {
        "schema",
        "attempt_id",
        "returncode",
        "logical_argv",
        "staged_command",
        "stdout",
        "stderr",
        "success_sentinel",
    }
)
PIXAL_ATTRIBUTE_FAILURE_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "attempt_id",
        "status",
        "case_id",
        "asset_id",
        "base_avatar_id",
        "job",
        "start_ledger",
        "failure_stage",
        "error",
        "returncode",
        "artifacts",
    }
)
PIXAL_ENVIRONMENT_FIELDS = frozenset(
    {
        "python_executable",
        "python_executable_record",
        "python_version",
        "torch_version",
        "cuda_version",
        "cuda_visible_devices",
        "cuda_available",
        "cuda_device_count",
        "cuda_device_name",
        "cuda_device_uuid",
        "attention_backend",
        "hf_hub_cache",
        "hf_hub_offline",
        "transformers_offline",
        "torch_home",
        "opencv_io_enable_openexr",
        "pytorch_cuda_alloc_conf",
    }
)
DEFAULT_BRANCH_DAG: dict[str, Any] = {
    "entry_branch": "direct",
    "nodes": [
        {"branch_id": "direct", "relative_root": "."},
        {"branch_id": "fitted_skeleton", "relative_root": "fitted_skeleton_v1"},
        {
            "branch_id": "sanitized_weights",
            "relative_root": "fitted_skeleton_v1/sanitized_weights_v1",
        },
    ],
    "edges": [
        ["direct", "fitted_skeleton"],
        ["fitted_skeleton", "sanitized_weights"],
    ],
    "finalizable_branches": ["direct", "fitted_skeleton", "sanitized_weights"],
}


class InstanceContractError(RuntimeError):
    """The instance contract is stale, non-canonical, incomplete, or contradictory."""


def _is_descendant(child: str, parent: str) -> bool:
    if parent == ".":
        return child != "."
    parent_path = PurePosixPath(parent)
    child_path = PurePosixPath(child)
    return child_path != parent_path and parent_path in child_path.parents


def validate_branch_dag(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "entry_branch",
        "nodes",
        "edges",
        "finalizable_branches",
    }:
        raise InstanceContractError("allowed branch DAG fields are incomplete or unexpected")
    entry = common.require_id(value.get("entry_branch"), "entry_branch", InstanceContractError)
    nodes_value = value.get("nodes")
    if not isinstance(nodes_value, list) or not nodes_value:
        raise InstanceContractError("allowed branch DAG nodes must be a non-empty list")
    nodes: list[dict[str, str]] = []
    positions: dict[str, int] = {}
    roots: dict[str, str] = {}
    for index, node in enumerate(nodes_value):
        if not isinstance(node, Mapping) or set(node) != {"branch_id", "relative_root"}:
            raise InstanceContractError("allowed branch DAG node fields are invalid")
        branch_id = common.require_id(
            node.get("branch_id"), "branch_id", InstanceContractError
        )
        relative_root = common.require_relative_root(
            node.get("relative_root"), InstanceContractError
        )
        if branch_id in positions:
            raise InstanceContractError(f"duplicate branch_id: {branch_id}")
        if relative_root in roots.values():
            raise InstanceContractError(f"duplicate branch relative_root: {relative_root}")
        positions[branch_id] = index
        roots[branch_id] = relative_root
        nodes.append({"branch_id": branch_id, "relative_root": relative_root})
    if entry not in positions:
        raise InstanceContractError("entry_branch names an unknown branch")
    if roots[entry] != ".":
        raise InstanceContractError("entry branch relative_root must be '.'")
    if any(branch != entry and root == "." for branch, root in roots.items()):
        raise InstanceContractError("only the entry branch may use relative_root '.'")

    edges_value = value.get("edges")
    if not isinstance(edges_value, list):
        raise InstanceContractError("allowed branch DAG edges must be a list")
    edges: list[list[str]] = []
    seen_edges: set[tuple[str, str]] = set()
    outgoing: dict[str, list[str]] = {branch: [] for branch in positions}
    indegree = {branch: 0 for branch in positions}
    for edge in edges_value:
        if not isinstance(edge, list) or len(edge) != 2:
            raise InstanceContractError("each allowed branch edge must contain two branch IDs")
        parent, child = edge
        if parent not in positions or child not in positions:
            raise InstanceContractError("allowed branch edge names an unknown branch")
        if parent == child or (parent, child) in seen_edges:
            raise InstanceContractError("allowed branch DAG has a self-edge or duplicate edge")
        seen_edges.add((parent, child))
        outgoing[parent].append(child)
        indegree[child] += 1
        edges.append([parent, child])
    ready = [branch for branch in positions if indegree[branch] == 0]
    visited: list[str] = []
    remaining = dict(indegree)
    while ready:
        ready.sort(key=positions.__getitem__)
        branch = ready.pop(0)
        visited.append(branch)
        for child in outgoing[branch]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if len(visited) != len(nodes):
        raise InstanceContractError("allowed branch DAG contains a cycle")
    if indegree[entry] != 0:
        raise InstanceContractError("allowed branch DAG entry branch has an incoming edge")
    if any(branch != entry and degree == 0 for branch, degree in indegree.items()):
        raise InstanceContractError("allowed branch DAG contains an unreachable root")
    for parent, child in seen_edges:
        if not _is_descendant(roots[child], roots[parent]):
            raise InstanceContractError(
                "child branch relative_root must be nested under its parent branch"
            )
    if visited != [node["branch_id"] for node in nodes]:
        raise InstanceContractError("allowed branch DAG nodes are not in canonical topological order")

    finalizable_value = value.get("finalizable_branches")
    if (
        not isinstance(finalizable_value, list)
        or len(finalizable_value) != len(set(finalizable_value))
        or any(branch not in positions for branch in finalizable_value)
    ):
        raise InstanceContractError("finalizable_branches contains an unknown or duplicate branch")
    finalizable = [str(branch) for branch in finalizable_value]
    if finalizable != [node["branch_id"] for node in nodes if node["branch_id"] in finalizable]:
        raise InstanceContractError("finalizable_branches is not in canonical branch order")
    return {
        "entry_branch": entry,
        "nodes": nodes,
        "edges": edges,
        "finalizable_branches": finalizable,
    }


def _source_record(lineage_id: str, descriptor: Any) -> dict[str, Any]:
    common.require_id(lineage_id, "source lineage id", InstanceContractError)
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"role", "path", "root"}:
        raise InstanceContractError(
            f"source lineage {lineage_id!r} must contain exactly role, path, and root"
        )
    role = common.require_id(
        descriptor.get("role"), "source lineage role", InstanceContractError
    )
    root = common.require_real_directory(
        Path(descriptor["root"]), "source lineage root", InstanceContractError
    )
    record = common.file_record(
        Path(descriptor["path"]),
        root=root,
        description=f"source lineage {lineage_id}",
        error_type=InstanceContractError,
    )
    return {
        "role": role,
        "path": record["path"],
        "root": str(root),
        "relative_path": record["relative_path"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
        "mode": record["mode"],
    }


def _validate_source_record(lineage_id: str, descriptor: Any) -> dict[str, Any]:
    common.require_id(lineage_id, "source lineage id", InstanceContractError)
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "role",
        "path",
        "root",
        "relative_path",
        "sha256",
        "size_bytes",
        "mode",
    }:
        raise InstanceContractError(f"source lineage {lineage_id!r} descriptor is invalid")
    role = common.require_id(
        descriptor.get("role"), "source lineage role", InstanceContractError
    )
    root = common.require_real_directory(
        Path(str(descriptor.get("root"))), "source lineage root", InstanceContractError
    )
    relative_value = descriptor.get("relative_path")
    if not isinstance(relative_value, str) or relative_value in {"", "."}:
        raise InstanceContractError(f"source lineage {lineage_id!r} relative_path is invalid")
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise InstanceContractError(
            f"source lineage {lineage_id!r} relative_path is non-canonical"
        )
    expected_path = common.absolute(root.joinpath(*relative.parts))
    supplied_path = common.absolute(Path(str(descriptor.get("path"))))
    if supplied_path != expected_path:
        raise InstanceContractError(f"source lineage {lineage_id!r} path is not canonical")
    current = common.file_record(
        supplied_path,
        root=root,
        description=f"source lineage {lineage_id}",
        error_type=InstanceContractError,
    )
    if descriptor.get("sha256") != current["sha256"]:
        raise InstanceContractError(f"source lineage {lineage_id!r} SHA-256 changed")
    if descriptor.get("size_bytes") != current["size_bytes"]:
        raise InstanceContractError(f"source lineage {lineage_id!r} size changed")
    if descriptor.get("mode") != current["mode"]:
        raise InstanceContractError(f"source lineage {lineage_id!r} mode changed")
    return {
        "role": role,
        "path": current["path"],
        "root": str(root),
        "relative_path": relative_value,
        "sha256": current["sha256"],
        "size_bytes": current["size_bytes"],
        "mode": current["mode"],
    }


def _validate_case(
    *, asset_id: str, base_avatar_id: str, case_id: Any, case_kind: Any
) -> dict[str, str]:
    case_id = common.require_id(case_id, "case_id", InstanceContractError)
    case_kind = common.require_id(case_kind, "case_kind", InstanceContractError)
    if case_kind not in CASE_KINDS:
        raise InstanceContractError(f"case_kind is not allowed: {case_kind!r}")
    if case_kind == "base_avatar" and (
        asset_id != base_avatar_id or case_id != asset_id
    ):
        raise InstanceContractError(
            "base_avatar case requires asset_id, base_avatar_id, and case_id to match"
        )
    if case_kind == "base_avatar" and base_avatar_id not in BASE_AVATAR_IDS:
        raise InstanceContractError("base_avatar_id is not a pinned Route-2 base avatar")
    if case_kind == "attribute_instance":
        expected_base = ATTRIBUTE_CASE_BASE.get(case_id)
        if asset_id != f"route2_{case_id}_v1" or expected_base is None:
            raise InstanceContractError("attribute instance case_id or asset_id is not pinned")
        if base_avatar_id != expected_base:
            raise InstanceContractError(
                f"base_avatar_id does not match the pinned {case_id} profile"
            )
    return {"case_id": case_id, "kind": case_kind}


def _validate_lineage_profile(
    lineage: Mapping[str, Mapping[str, Any]], *, case_kind: str
) -> None:
    expected = BASE_LINEAGE_ROLES if case_kind == "base_avatar" else ATTRIBUTE_LINEAGE_ROLES
    if set(lineage) != set(expected):
        raise InstanceContractError(
            "source_lineage roles do not match the pinned case profile: "
            f"expected={sorted(expected)} actual={sorted(lineage)}"
        )
    for lineage_id, expected_role in expected.items():
        descriptor = lineage[lineage_id]
        if not isinstance(descriptor, Mapping) or descriptor.get("role") != expected_role:
            raise InstanceContractError(
                f"source lineage role is not pinned: {lineage_id} -> {expected_role}"
            )


def _lineage_json(record: Mapping[str, Any], description: str) -> dict[str, Any]:
    payload, current = common.load_json_mapping_record(
        Path(str(record["path"])),
        root=Path(str(record["root"])),
        description=description,
        error_type=InstanceContractError,
    )
    if any(
        current[key] != record[key]
        for key in ("path", "relative_path", "sha256", "size_bytes", "mode")
    ):
        raise InstanceContractError(
            f"{description} changed between lineage authentication and parsing"
        )
    common.reject_user_approval(payload, InstanceContractError, description)
    return payload


def _require_png(record: Mapping[str, Any], description: str) -> None:
    data, current = common.read_file_snapshot(
        Path(str(record["path"])),
        root=Path(str(record["root"])),
        description=description,
        error_type=InstanceContractError,
    )
    if any(
        current[key] != record[key]
        for key in ("path", "relative_path", "sha256", "size_bytes", "mode")
    ):
        raise InstanceContractError(f"{description} changed during PNG authentication")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise InstanceContractError(f"{description} is not a PNG")


def _require_exact_mapping(
    value: Any, fields: frozenset[str] | set[str], description: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise InstanceContractError(
            f"{description} fields are incomplete or unexpected"
        )
    return value


def _require_embedded_record(
    value: Any,
    expected: Mapping[str, Any],
    description: str,
    *,
    extra_fields: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    fields = frozenset({"path", "sha256", "size_bytes"}) | extra_fields
    record = _require_exact_mapping(value, fields, description)
    for key in ("path", "sha256", "size_bytes"):
        if record.get(key) != expected.get(key):
            raise InstanceContractError(f"{description} does not match authenticated lineage")
    return record


def _require_current_embedded_record(value: Any, description: str) -> dict[str, Any]:
    record = _require_exact_mapping(
        value, frozenset({"path", "sha256", "size_bytes"}), description
    )
    path_value = record.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise InstanceContractError(f"{description} path must be absolute")
    path = common.absolute(Path(path_value))
    current = common.file_record(
        path,
        root=path.parent,
        description=description,
        error_type=InstanceContractError,
    )
    for key in ("path", "sha256", "size_bytes"):
        if record.get(key) != current.get(key):
            raise InstanceContractError(f"{description} changed")
    return {key: current[key] for key in ("path", "sha256", "size_bytes")}


def _parse_utc_timestamp(value: Any, description: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise InstanceContractError(f"{description} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise InstanceContractError(f"{description} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise InstanceContractError(f"{description} must be UTC")
    return parsed


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_metadata_key(
    snapshot: Path, *, revision: str, description: str
) -> tuple[Any, ...]:
    repository_root = snapshot.parents[1]
    entries: list[tuple[Any, ...]] = []
    for path in sorted(
        snapshot.rglob("*"),
        key=lambda item: item.relative_to(snapshot).as_posix(),
    ):
        relative = path.relative_to(snapshot).as_posix()
        logical = os.lstat(path)
        if stat.S_ISLNK(logical.st_mode):
            raw_target: str | None = os.readlink(path)
            resolved = common.absolute(path.parent / raw_target)
            try:
                resolved.relative_to(repository_root)
            except ValueError as error:
                raise InstanceContractError(
                    f"{description} symlink target escapes its model repository"
                ) from error
            target = os.stat(resolved, follow_symlinks=False)
            kind = "symlink"
            resolved_value: str | None = str(resolved)
            target_identity: tuple[int, ...] | None = _stat_identity(target)
        else:
            raw_target = None
            resolved_value = str(path)
            target_identity = _stat_identity(logical)
            if stat.S_ISDIR(logical.st_mode):
                kind = "directory"
            elif stat.S_ISREG(logical.st_mode):
                kind = "regular_file"
            else:
                kind = "other"
        entries.append(
            (
                relative,
                kind,
                _stat_identity(logical),
                raw_target,
                resolved_value,
                target_identity,
            )
        )
    required = set(MODEL_SNAPSHOT_CONTRACTS[revision]["required_files"]) | {
        str(MODEL_SNAPSHOT_CONTRACTS[revision]["license"])
    }
    file_entries = {
        entry[0]
        for entry in entries
        if entry[1] in {"regular_file", "symlink"}
    }
    if not required.issubset(file_entries):
        raise InstanceContractError(
            f"{description} metadata is missing pinned model/license files"
        )
    return (
        str(snapshot),
        revision,
        _stat_identity(os.stat(snapshot, follow_symlinks=False)),
        tuple(entries),
    )


def _snapshot_inventory_uncached(
    snapshot: Path, *, revision: str, description: str
) -> dict[str, Any]:
    contract = MODEL_SNAPSHOT_CONTRACTS[revision]
    repository_root = snapshot.parents[1]
    repository_root = common.require_real_directory(
        repository_root, f"{description} repository", InstanceContractError
    )
    records: list[dict[str, Any]] = []
    for path in sorted(snapshot.rglob("*"), key=lambda item: item.relative_to(snapshot).as_posix()):
        relative = path.relative_to(snapshot).as_posix()
        current = os.lstat(path)
        if stat.S_ISDIR(current.st_mode):
            if path.is_symlink():
                raise InstanceContractError(f"{description} contains a symlinked directory")
            continue
        if stat.S_ISLNK(current.st_mode):
            raw_target = os.readlink(path)
            target = common.absolute(path.parent / raw_target)
            try:
                target.relative_to(repository_root)
            except ValueError as error:
                raise InstanceContractError(
                    f"{description} symlink target escapes its model repository"
                ) from error
            record = common.hash_file_snapshot(
                target,
                root=repository_root,
                description=f"{description} inventory file {relative}",
                error_type=InstanceContractError,
            )
            content_sha256 = record["sha256"]
            if common.SHA256_RE.fullmatch(target.name) is not None:
                if content_sha256 != target.name:
                    raise InstanceContractError(
                        f"{description} content-addressed blob SHA-256 changed"
                    )
                hash_source = "authenticated_bytes_and_huggingface_lfs_name"
            else:
                hash_source = "authenticated_file_bytes"
            records.append(
                {
                    "relative_path": relative,
                    "storage": "repository_symlink",
                    "link_target": raw_target,
                    "target_relative_path": record["relative_path"],
                    "sha256": content_sha256,
                    "hash_source": hash_source,
                    "size_bytes": record["size_bytes"],
                    "mode": record["mode"],
                }
            )
        elif stat.S_ISREG(current.st_mode):
            record = common.hash_file_snapshot(
                path,
                root=snapshot,
                description=f"{description} inventory file {relative}",
                error_type=InstanceContractError,
            )
            records.append(
                {
                    "relative_path": relative,
                    "storage": "direct_regular_file",
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                    "mode": record["mode"],
                }
            )
        else:
            raise InstanceContractError(f"{description} contains a non-file entry: {relative}")
    by_path = {record["relative_path"]: record for record in records}
    required = set(contract["required_files"]) | {str(contract["license"])}
    if not records or not required.issubset(by_path):
        raise InstanceContractError(
            f"{description} inventory is empty or missing pinned model/license files"
        )
    license_record = by_path[str(contract["license"])]
    if license_record["sha256"] != contract["license_sha256"]:
        raise InstanceContractError(f"{description} license file hash is not pinned")
    digest = hashlib.sha256(
        common.canonical_json(records).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(snapshot),
        "revision": revision,
        "file_count": len(records),
        "inventory_sha256": digest,
        "license": license_record,
    }


def _snapshot_inventory(
    snapshot: Path, *, revision: str, description: str
) -> dict[str, Any]:
    before = _snapshot_metadata_key(
        snapshot, revision=revision, description=description
    )
    cache_key = (str(snapshot), revision)
    cached = _MODEL_SNAPSHOT_CACHE.get(cache_key)
    if cached is not None and cached[0] == before:
        return {**copy.deepcopy(cached[1]), "cache_hit": True}
    evidence = _snapshot_inventory_uncached(
        snapshot, revision=revision, description=description
    )
    after = _snapshot_metadata_key(
        snapshot, revision=revision, description=description
    )
    if after != before:
        raise InstanceContractError(f"{description} metadata changed during hashing")
    stored = copy.deepcopy(evidence)
    _MODEL_SNAPSHOT_CACHE[cache_key] = (after, stored)
    return {**copy.deepcopy(stored), "cache_hit": False}


def _validate_model_snapshot(value: Any, revision: str, description: str) -> dict[str, Any]:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise InstanceContractError(f"{description} path must be absolute")
    snapshot = common.absolute(Path(value))
    model_root = common.require_real_directory(
        MODEL_ROOT, "canonical model root", InstanceContractError
    )
    contract = MODEL_SNAPSHOT_CONTRACTS.get(revision)
    if contract is None:
        raise InstanceContractError(f"{description} revision is not supported")
    expected_snapshot = common.absolute(model_root / contract["relative_path"])
    if snapshot != expected_snapshot:
        raise InstanceContractError(f"{description} is not the canonical fixed snapshot")
    snapshot = common.require_real_directory(
        snapshot, description, InstanceContractError
    )
    if snapshot.name != revision:
        raise InstanceContractError(f"{description} revision directory is not pinned")
    return _snapshot_inventory(snapshot, revision=revision, description=description)


def model_snapshot_evidence(revision: str) -> dict[str, Any]:
    contract = MODEL_SNAPSHOT_CONTRACTS.get(revision)
    if contract is None:
        raise InstanceContractError("model snapshot revision is not supported")
    snapshot = common.absolute(
        common.require_real_directory(
            MODEL_ROOT, "canonical model root", InstanceContractError
        )
        / contract["relative_path"]
    )
    return _validate_model_snapshot(
        str(snapshot), revision, f"model snapshot {revision}"
    )


def _filesystem_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _execution_guard_file(path: Path, description: str) -> dict[str, Any]:
    path = common.absolute(path)
    parent = common.require_real_directory(
        path.parent,
        f"{description} parent",
        InstanceContractError,
    )
    record = common.file_record(
        path,
        root=parent,
        description=description,
        error_type=InstanceContractError,
    )
    current = os.stat(path, follow_symlinks=False)
    parent_current = os.stat(parent, follow_symlinks=False)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_size != record["size_bytes"]
        or f"{stat.S_IMODE(current.st_mode):04o}" != record["mode"]
    ):
        raise InstanceContractError(f"{description} identity changed during guard")
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
        "mode": record["mode"],
        "identity": _filesystem_identity(current),
        "parent": {
            "path": str(parent),
            "identity": _filesystem_identity(parent_current),
        },
    }


def _model_execution_metadata_guard(
    revision: str,
    description: str,
) -> dict[str, Any]:
    contract = MODEL_SNAPSHOT_CONTRACTS[revision]
    model_root = common.require_real_directory(
        MODEL_ROOT,
        "canonical model root",
        InstanceContractError,
    )
    snapshot = common.require_real_directory(
        model_root / contract["relative_path"],
        description,
        InstanceContractError,
    )
    metadata_key = _snapshot_metadata_key(
        snapshot,
        revision=revision,
        description=description,
    )
    guarded_directories: set[Path] = set()

    def add_ancestry(path: Path) -> None:
        current = common.absolute(path)
        while True:
            try:
                current.relative_to(model_root)
            except ValueError:
                break
            if current.is_dir() and not current.is_symlink():
                guarded_directories.add(current)
            if current == model_root:
                break
            current = current.parent

    add_ancestry(snapshot)
    for entry in metadata_key[3]:
        relative, _, _, _, resolved_value, _ = entry
        add_ancestry((snapshot / relative).parent)
        if isinstance(resolved_value, str):
            add_ancestry(Path(resolved_value).parent)
    directory_records = []
    for directory in sorted(guarded_directories, key=str):
        validated = common.require_real_directory(
            directory,
            f"{description} guarded directory",
            InstanceContractError,
        )
        directory_records.append(
            {
                "path": str(validated),
                "identity": _filesystem_identity(
                    os.stat(validated, follow_symlinks=False)
                ),
            }
        )
    metadata_payload = {
        "snapshot_metadata_key": metadata_key,
        "guarded_directories": directory_records,
    }
    return {
        "path": str(snapshot),
        "revision": revision,
        "entry_count": len(metadata_key[3]),
        "metadata_sha256": hashlib.sha256(
            common.canonical_json(metadata_payload).encode("utf-8")
        ).hexdigest(),
    }


def _pixal_execution_guard_once() -> dict[str, Any]:
    from tools import human_attribute_pixal_contract as pixal_contract

    core = {
        "schema": "pixal3d_execution_guard_v1",
        "scope": [
            "python_executable",
            "pixal_wrapper",
            "atomic_executor",
            "pixal_model_snapshot_metadata",
            "dino_model_snapshot_metadata",
        ],
        "files": {
            "python_executable": _execution_guard_file(
                PIXAL_PYTHON_EXECUTABLE,
                "Pixal execution-guard Python executable",
            ),
            "wrapper": _execution_guard_file(
                Path(pixal_contract.PIXAL_WRAPPER_PATH),
                "Pixal execution-guard wrapper",
            ),
            "executor": _execution_guard_file(
                Path(pixal_contract.EXECUTOR_PATH),
                "Pixal execution-guard executor",
            ),
        },
        "models": {
            "pixal": _model_execution_metadata_guard(
                PIXAL3D_REVISION,
                "Pixal3D execution-guard snapshot",
            ),
            "dino": _model_execution_metadata_guard(
                DINO_REVISION,
                "DINO execution-guard snapshot",
            ),
        },
    }
    return {
        **core,
        "guard_sha256": hashlib.sha256(
            common.canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }


def pixal_execution_guard_evidence() -> dict[str, Any]:
    """Return a twice-stable guard whose ctime scope detects swap-and-restore."""

    return common.stable_mapping_snapshot(
        _pixal_execution_guard_once,
        InstanceContractError,
        "Pixal execution guard",
    )


def _validate_filesystem_identity(value: Any, description: str) -> dict[str, int]:
    fields = {"device", "inode", "mode", "size_bytes", "mtime_ns", "ctime_ns"}
    identity = _require_exact_mapping(value, fields, description)
    if any(
        not isinstance(identity.get(field), int)
        or isinstance(identity.get(field), bool)
        or identity[field] < 0
        for field in fields
    ):
        raise InstanceContractError(f"{description} values are invalid")
    return dict(identity)


def _validate_pixal_execution_guard(value: Any, description: str) -> dict[str, Any]:
    guard = _require_exact_mapping(
        value,
        frozenset({"schema", "scope", "files", "models", "guard_sha256"}),
        description,
    )
    expected_scope = [
        "python_executable",
        "pixal_wrapper",
        "atomic_executor",
        "pixal_model_snapshot_metadata",
        "dino_model_snapshot_metadata",
    ]
    if (
        guard.get("schema") != "pixal3d_execution_guard_v1"
        or guard.get("scope") != expected_scope
        or not isinstance(guard.get("guard_sha256"), str)
        or common.SHA256_RE.fullmatch(guard["guard_sha256"]) is None
    ):
        raise InstanceContractError(f"{description} schema or scope is invalid")
    from tools import human_attribute_pixal_contract as pixal_contract

    expected_file_paths = {
        "python_executable": common.absolute(PIXAL_PYTHON_EXECUTABLE),
        "wrapper": common.absolute(Path(pixal_contract.PIXAL_WRAPPER_PATH)),
        "executor": common.absolute(Path(pixal_contract.EXECUTOR_PATH)),
    }
    files = _require_exact_mapping(
        guard.get("files"), set(expected_file_paths), f"{description} files"
    )
    validated_files: dict[str, Any] = {}
    for role, expected_path in expected_file_paths.items():
        record = _require_exact_mapping(
            files.get(role),
            frozenset(
                {"path", "sha256", "size_bytes", "mode", "identity", "parent"}
            ),
            f"{description} {role}",
        )
        parent = _require_exact_mapping(
            record.get("parent"),
            frozenset({"path", "identity"}),
            f"{description} {role} parent",
        )
        if (
            record.get("path") != str(expected_path)
            or not isinstance(record.get("sha256"), str)
            or common.SHA256_RE.fullmatch(record["sha256"]) is None
            or not isinstance(record.get("size_bytes"), int)
            or isinstance(record.get("size_bytes"), bool)
            or record["size_bytes"] <= 0
            or not isinstance(record.get("mode"), str)
            or not isinstance(parent.get("path"), str)
            or parent["path"] != str(expected_path.parent)
        ):
            raise InstanceContractError(f"{description} {role} record is invalid")
        validated_files[role] = {
            **dict(record),
            "identity": _validate_filesystem_identity(
                record.get("identity"), f"{description} {role} identity"
            ),
            "parent": {
                "path": parent["path"],
                "identity": _validate_filesystem_identity(
                    parent.get("identity"),
                    f"{description} {role} parent identity",
                ),
            },
        }
    models = _require_exact_mapping(
        guard.get("models"), frozenset({"pixal", "dino"}), f"{description} models"
    )
    expected_models = {
        "pixal": PIXAL3D_REVISION,
        "dino": DINO_REVISION,
    }
    validated_models: dict[str, Any] = {}
    for role, revision in expected_models.items():
        record = _require_exact_mapping(
            models.get(role),
            frozenset({"path", "revision", "entry_count", "metadata_sha256"}),
            f"{description} {role} model",
        )
        expected_path = common.absolute(
            MODEL_ROOT / MODEL_SNAPSHOT_CONTRACTS[revision]["relative_path"]
        )
        if (
            record.get("path") != str(expected_path)
            or record.get("revision") != revision
            or not isinstance(record.get("entry_count"), int)
            or isinstance(record.get("entry_count"), bool)
            or record["entry_count"] <= 0
            or not isinstance(record.get("metadata_sha256"), str)
            or common.SHA256_RE.fullmatch(record["metadata_sha256"]) is None
        ):
            raise InstanceContractError(f"{description} {role} model guard is invalid")
        validated_models[role] = dict(record)
    normalized = {
        "schema": "pixal3d_execution_guard_v1",
        "scope": expected_scope,
        "files": validated_files,
        "models": validated_models,
    }
    expected_digest = hashlib.sha256(
        common.canonical_json(normalized).encode("utf-8")
    ).hexdigest()
    if guard.get("guard_sha256") != expected_digest:
        raise InstanceContractError(f"{description} digest is invalid")
    return {**normalized, "guard_sha256": expected_digest}


def _probe_pixal_python_runtime(
    executable: Path, environment: Mapping[str, Any]
) -> dict[str, Any]:
    probe = (
        "import json,platform,torch,cv2,o_voxel;"
        "available=torch.cuda.is_available();count=torch.cuda.device_count();"
        "props=torch.cuda.get_device_properties(0) if available and count==1 else None;"
        "print(json.dumps({'python_version':platform.python_version(),"
        "'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda,"
        "'cuda_available':available,'cuda_device_count':count,"
        "'cuda_device_name':torch.cuda.get_device_name(0) if props else '',"
        "'cuda_device_uuid':str(props.uuid) if props else ''}))"
    )
    process_environment = os.environ.copy()
    for key, environment_key in (
        ("CUDA_VISIBLE_DEVICES", "cuda_visible_devices"),
        ("HF_HUB_CACHE", "hf_hub_cache"),
        ("HF_HUB_OFFLINE", "hf_hub_offline"),
        ("TRANSFORMERS_OFFLINE", "transformers_offline"),
        ("TORCH_HOME", "torch_home"),
        ("OPENCV_IO_ENABLE_OPENEXR", "opencv_io_enable_openexr"),
        ("PYTORCH_CUDA_ALLOC_CONF", "pytorch_cuda_alloc_conf"),
    ):
        process_environment[key] = str(environment[environment_key])
    try:
        completed = subprocess.run(
            [str(executable), "-I", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=process_environment,
        )
        value = json.loads(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise InstanceContractError(
            f"Pixal Python runtime probe failed: {error}"
        ) from error
    if not isinstance(value, dict) or set(value) != {
        "python_version",
        "torch_version",
        "cuda_version",
        "cuda_available",
        "cuda_device_count",
        "cuda_device_name",
        "cuda_device_uuid",
    }:
        raise InstanceContractError("Pixal Python runtime probe is incomplete")
    for key in (
        "python_version",
        "torch_version",
        "cuda_version",
        "cuda_device_name",
        "cuda_device_uuid",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise InstanceContractError("Pixal Python runtime versions are incomplete")
    if value.get("cuda_available") is not True or value.get("cuda_device_count") != 1:
        raise InstanceContractError("Pixal Python runtime versions are incomplete")
    return value


def _validate_pixal_attribute_job(
    pixal_job: Mapping[str, Any],
    lineage: Mapping[str, Mapping[str, Any]],
    *,
    asset_id: str,
    base_avatar_id: str,
    case_id: str,
) -> tuple[list[str], dict[str, Any]]:
    from tools import human_attribute_pixal_contract as pixal_contract

    _require_exact_mapping(
        pixal_job, PIXAL_ATTRIBUTE_JOB_FIELDS, "Pixal attribute job"
    )
    if (
        pixal_job.get("schema") != "pixal3d_human_attribute_job_v1"
        or pixal_job.get("case_id") != case_id
        or pixal_job.get("asset_id") != asset_id
        or pixal_job.get("base_asset_id") != base_avatar_id
        or pixal_job.get("state_classification") != "research_candidate"
        or pixal_job.get("model_revision") != PIXAL3D_REVISION
        or pixal_job.get("dino_revision") != DINO_REVISION
        or pixal_job.get("parameters")
        != {"seed": 42, "manual_fov": 0.2, "resolution": 1024, "low_vram": True}
        or pixal_job.get("output_glb") != lineage["pixal_pbr_glb"]["path"]
        or pixal_job.get("output_manifest") != lineage["pixal_manifest"]["path"]
        or pixal_job.get("output_policy") != "atomic_no_replace"
    ):
        raise InstanceContractError("Pixal attribute job identity or parameters are invalid")
    output_glb = common.absolute(Path(lineage["pixal_pbr_glb"]["path"]))
    output_manifest = common.absolute(Path(lineage["pixal_manifest"]["path"]))
    asset_root = output_glb.parent
    if (
        asset_root.name != asset_id
        or output_glb.name != "canary_1024_seed42.glb"
        or output_manifest != output_glb.with_suffix(".manifest.json")
        or common.absolute(Path(lineage["pixal_job"]["path"]))
        != asset_root.parent / f"{asset_id}.pixal_job.json"
    ):
        raise InstanceContractError("Pixal attribute job/output paths are not canonical")
    _require_embedded_record(
        pixal_job.get("candidate_manifest"),
        lineage["attribute_candidate_manifest"],
        "Pixal job candidate manifest",
    )
    decision_record = _require_embedded_record(
        pixal_job.get("agent_2d_decision"),
        lineage["attribute_agent_decision"],
        "Pixal job agent 2D decision",
        extra_fields=frozenset({"status"}),
    )
    if decision_record.get("status") != "agent_qa_passed_pending_user_acceptance":
        raise InstanceContractError("Pixal job agent 2D decision status is not accepted")
    rgba_record = _require_embedded_record(
        pixal_job.get("input_rgba"),
        lineage["candidate_rgba"],
        "Pixal job input RGBA",
        extra_fields=frozenset({"mode", "size", "alpha_min", "alpha_max"}),
    )
    size = rgba_record.get("size")
    alpha_min = rgba_record.get("alpha_min")
    alpha_max = rgba_record.get("alpha_max")
    if (
        rgba_record.get("mode") != "RGBA"
        or not isinstance(size, list)
        or len(size) != 2
        or any(
            not isinstance(number, int) or isinstance(number, bool) or number <= 0
            for number in size
        )
        or not isinstance(alpha_min, int)
        or isinstance(alpha_min, bool)
        or not isinstance(alpha_max, int)
        or isinstance(alpha_max, bool)
        or not 0 <= alpha_min < 255
        or not 0 < alpha_max <= 255
        or alpha_min > alpha_max
    ):
        raise InstanceContractError("Pixal job input RGBA metadata is invalid")
    wrapper = _require_current_embedded_record(
        pixal_job.get("wrapper"), "Pixal job wrapper"
    )
    if (
        wrapper["path"] != str(common.absolute(pixal_contract.PIXAL_WRAPPER_PATH))
        or wrapper["sha256"] != pixal_contract.PIXAL_WRAPPER_SHA256
        or pixal_contract.PIXAL3D_REVISION != PIXAL3D_REVISION
        or pixal_contract.DINO_REVISION != DINO_REVISION
    ):
        raise InstanceContractError("Pixal job wrapper is not the exact pinned executor")
    expected_argv = [
        wrapper["path"],
        "--backend",
        "pixal3d",
        "--image",
        lineage["candidate_rgba"]["path"],
        "--output",
        lineage["pixal_pbr_glb"]["path"],
        "--gpu",
        "3",
        "--seed",
        "42",
        "--resolution",
        "1024",
        "--manual-fov",
        "0.2",
        "--low-vram",
    ]
    executor = _require_exact_mapping(
        pixal_job.get("executor"),
        PIXAL_EXECUTOR_FIELDS,
        "Pixal job executor",
    )
    executor_code = _require_current_embedded_record(
        {key: executor.get(key) for key in ("path", "sha256", "size_bytes")},
        "Pixal atomic executor code",
    )
    expected_executor_path = common.absolute(Path(pixal_contract.EXECUTOR_PATH))
    if (
        executor.get("kind") != "atomic_pixal3d_executor_v1"
        or executor.get("argv") != expected_argv
        or executor.get("execution_authorized") is not True
        or executor.get("atomic_no_replace") is not True
        or executor_code["path"] != str(expected_executor_path)
    ):
        raise InstanceContractError("Pixal job must use the authorized atomic executor")
    return expected_argv, executor_code


def _validate_pixal_attempt_start_ledger(
    descriptor: Any,
    *,
    expected_path: Path,
    attempt_id: str,
    case_id: str,
    asset_id: str,
    base_avatar_id: str,
    job_record: Mapping[str, Any],
    executor_record: Mapping[str, Any],
    argv: list[str],
    started_at_utc: str,
    staging_path: Path,
    expected_execution_guard_before: Mapping[str, Any] | None,
) -> dict[str, Any]:
    supplied = _require_exact_mapping(
        descriptor,
        frozenset({"path", "sha256", "size_bytes"}),
        "Pixal attempt start ledger descriptor",
    )
    expected_path = common.absolute(expected_path)
    payload, record = common.load_json_mapping_record(
        expected_path,
        root=expected_path.parent,
        description="Pixal attempt start ledger",
        error_type=InstanceContractError,
        require_mode=0o444,
    )
    core_record = {
        key: record[key] for key in ("path", "sha256", "size_bytes")
    }
    if dict(supplied) != core_record:
        raise InstanceContractError("Pixal attempt start ledger descriptor changed")
    _require_exact_mapping(
        payload, PIXAL_ATTRIBUTE_START_FIELDS, "Pixal attempt start ledger"
    )
    common.reject_user_approval(
        payload, InstanceContractError, "Pixal attempt start ledger"
    )
    start_staging = _require_exact_mapping(
        payload.get("staging"),
        frozenset({"path", "created"}),
        "Pixal attempt start staging",
    )
    staging_prefix = f".{asset_id}.{attempt_id}."
    if (
        staging_path.parent != expected_path.parents[2]
        or not staging_path.name.startswith(staging_prefix)
        or not staging_path.name.endswith(".staging")
        or len(staging_path.name)
        <= len(staging_prefix) + len(".staging")
    ):
        raise InstanceContractError(
            "Pixal attempt start staging path is not bound to its asset and attempt"
        )
    guard_before = _validate_pixal_execution_guard(
        payload.get("execution_guard_before"),
        "Pixal attempt start execution guard",
    )
    if (
        expected_execution_guard_before is not None
        and guard_before != dict(expected_execution_guard_before)
    ):
        raise InstanceContractError(
            "Pixal attempt start execution guard does not match the completed attempt"
        )
    _parse_utc_timestamp(payload.get("started_at_utc"), "Pixal attempt ledger start")
    if (
        payload.get("schema") != PIXAL_ATTRIBUTE_START_SCHEMA
        or payload.get("attempt_id") != attempt_id
        or payload.get("status") != "started"
        or payload.get("case_id") != case_id
        or payload.get("asset_id") != asset_id
        or payload.get("base_avatar_id") != base_avatar_id
        or payload.get("job") != dict(job_record)
        or payload.get("executor") != dict(executor_record)
        or payload.get("execution_guard_before") != guard_before
        or payload.get("argv") != argv
        or payload.get("started_at_utc") != started_at_utc
        or start_staging.get("path") != str(staging_path)
        or start_staging.get("created") is not True
        or payload.get("publication_policy") != "atomic_no_replace"
    ):
        raise InstanceContractError("Pixal attempt start ledger is inconsistent")
    return {"payload": payload, "record": core_record}


def _validate_pixal_execution_log(
    descriptor: Any,
    *,
    expected_path: Path,
    attempt_id: str,
    returncode: int,
    logical_argv: list[str],
    python_executable: str,
    staging_path: Path,
    public_glb: Path,
    public_manifest: Path,
) -> dict[str, Any]:
    supplied = _require_exact_mapping(
        descriptor,
        frozenset({"path", "sha256", "size_bytes"}),
        "Pixal execution log descriptor",
    )
    expected_path = common.absolute(expected_path)
    payload, record = common.load_json_mapping_record(
        expected_path,
        root=expected_path.parent,
        description="Pixal execution log",
        error_type=InstanceContractError,
        require_mode=0o444,
    )
    core_record = {
        key: record[key] for key in ("path", "sha256", "size_bytes")
    }
    if dict(supplied) != core_record:
        raise InstanceContractError("Pixal execution log descriptor changed")
    _require_exact_mapping(
        payload, PIXAL_ATTRIBUTE_EXECUTION_LOG_FIELDS, "Pixal execution log"
    )
    common.reject_user_approval(payload, InstanceContractError, "Pixal execution log")
    staged_argv = list(logical_argv)
    try:
        output_index = staged_argv.index("--output") + 1
    except ValueError as error:  # pragma: no cover - the job validator pins this argument.
        raise InstanceContractError("Pixal execution argv has no output argument") from error
    staged_argv[output_index] = str(staging_path / public_glb.name)
    expected_sentinel = str(staging_path / public_manifest.name)
    stdout = payload.get("stdout")
    stderr = payload.get("stderr")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise InstanceContractError("Pixal execution log streams are invalid")
    nonempty_stdout = [line for line in stdout.splitlines() if line.strip()]
    if (
        payload.get("schema") != PIXAL_ATTRIBUTE_EXECUTION_LOG_SCHEMA
        or payload.get("attempt_id") != attempt_id
        or payload.get("returncode") != returncode
        or payload.get("logical_argv") != logical_argv
        or payload.get("staged_command") != [python_executable, *staged_argv]
        or payload.get("success_sentinel") != expected_sentinel
        or not nonempty_stdout
        or nonempty_stdout[-1] != expected_sentinel
        or stdout.splitlines().count(expected_sentinel) != 1
    ):
        raise InstanceContractError(
            "Pixal execution log or unique success sentinel is inconsistent"
        )
    return {"payload": payload, "record": core_record}


def _validate_pixal_attribute_attempt(
    attempt: Mapping[str, Any],
    pixal_job: Mapping[str, Any],
    expected_argv: list[str],
    expected_executor: Mapping[str, Any],
    lineage: Mapping[str, Mapping[str, Any]],
    *,
    asset_id: str,
    base_avatar_id: str,
    case_id: str,
    model_evidence: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_exact_mapping(
        attempt, PIXAL_ATTRIBUTE_ATTEMPT_FIELDS, "Pixal attempt provenance"
    )
    if (
        attempt.get("schema") != PIXAL_ATTRIBUTE_ATTEMPT_SCHEMA
        or attempt.get("status") != "succeeded"
        or attempt.get("case_id") != case_id
        or attempt.get("asset_id") != asset_id
        or attempt.get("base_avatar_id") != base_avatar_id
    ):
        raise InstanceContractError("Pixal attempt provenance identity is invalid")
    attempt_id = common.require_id(
        attempt.get("attempt_id"), "Pixal attempt_id", InstanceContractError
    )
    job_record = _require_embedded_record(
        attempt.get("job"), lineage["pixal_job"], "Pixal attempt job"
    )
    executor_record = _require_exact_mapping(
        attempt.get("executor"),
        frozenset({"path", "sha256", "size_bytes"}),
        "Pixal attempt executor",
    )
    if dict(executor_record) != dict(expected_executor):
        raise InstanceContractError("Pixal attempt executor does not match the job")
    _require_current_embedded_record(executor_record, "Pixal attempt executor")
    guard_pair = _require_exact_mapping(
        attempt.get("execution_guard"),
        frozenset({"before", "after", "unchanged"}),
        "Pixal attempt execution guard",
    )
    guard_before = _validate_pixal_execution_guard(
        guard_pair.get("before"),
        "Pixal attempt execution guard before",
    )
    guard_after = _validate_pixal_execution_guard(
        guard_pair.get("after"),
        "Pixal attempt execution guard after",
    )
    current_guard = pixal_execution_guard_evidence()
    if (
        guard_pair.get("unchanged") is not True
        or guard_before != guard_after
        or guard_after != current_guard
    ):
        raise InstanceContractError(
            "Pixal execution-critical files or model metadata changed during execution"
        )
    if attempt.get("argv") != expected_argv:
        raise InstanceContractError("Pixal attempt argv does not match the immutable job")
    wrapper = _require_embedded_record(
        attempt.get("wrapper"), pixal_job["wrapper"], "Pixal attempt wrapper"
    )
    _require_current_embedded_record(wrapper, "Pixal attempt wrapper")
    environment = _require_exact_mapping(
        attempt.get("environment"),
        PIXAL_ENVIRONMENT_FIELDS,
        "Pixal attempt environment provenance",
    )
    string_fields = PIXAL_ENVIRONMENT_FIELDS - {
        "python_executable",
        "python_executable_record",
        "cuda_available",
        "cuda_device_count",
    }
    if any(
        not isinstance(environment.get(field), str) or not environment.get(field)
        for field in string_fields
    ):
        raise InstanceContractError("Pixal attempt environment provenance is incomplete")
    if (
        environment.get("cuda_available") is not True
        or not isinstance(environment.get("cuda_device_count"), int)
        or isinstance(environment.get("cuda_device_count"), bool)
        or environment.get("cuda_device_count") != 1
    ):
        raise InstanceContractError("Pixal attempt CUDA device visibility is invalid")
    python_executable = environment.get("python_executable")
    if not isinstance(python_executable, str) or not Path(python_executable).is_absolute():
        raise InstanceContractError("Pixal attempt Python executable must be absolute")
    pinned_environment = {
        "cuda_visible_devices": "3",
        "attention_backend": "sdpa",
        "hf_hub_cache": "/data/models/hub",
        "hf_hub_offline": "1",
        "transformers_offline": "1",
        "torch_home": "/data/models/torch",
        "opencv_io_enable_openexr": "1",
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
    }
    if any(environment.get(key) != value for key, value in pinned_environment.items()):
        raise InstanceContractError("Pixal attempt environment is not the pinned offline runtime")
    executable = common.absolute(Path(python_executable))
    expected_executable = common.absolute(PIXAL_PYTHON_EXECUTABLE)
    if executable != expected_executable:
        raise InstanceContractError("Pixal attempt Python executable is not pinned")
    executable_record_before = common.file_record(
        executable,
        root=executable.parent,
        description="Pixal Python executable",
        error_type=InstanceContractError,
    )
    executable_mode = os.stat(executable, follow_symlinks=False).st_mode
    if executable_mode & 0o111 == 0:
        raise InstanceContractError("Pixal Python executable is not executable")
    supplied_executable_record = _require_exact_mapping(
        environment.get("python_executable_record"),
        frozenset({"path", "sha256", "size_bytes", "mode"}),
        "Pixal Python executable record",
    )
    expected_executable_record = {
        key: executable_record_before[key]
        for key in ("path", "sha256", "size_bytes", "mode")
    }
    if dict(supplied_executable_record) != expected_executable_record:
        raise InstanceContractError("Pixal Python executable descriptor changed")
    runtime = _probe_pixal_python_runtime(executable, environment)
    executable_record_after = common.file_record(
        executable,
        root=executable.parent,
        description="Pixal Python executable",
        error_type=InstanceContractError,
    )
    if executable_record_after != executable_record_before:
        raise InstanceContractError("Pixal Python executable changed during runtime probe")
    owner_guard_after_probe = pixal_execution_guard_evidence()
    if owner_guard_after_probe != current_guard:
        raise InstanceContractError(
            "Pixal execution guard changed during the owner runtime probe"
        )
    if any(environment.get(key) != value for key, value in runtime.items()):
        raise InstanceContractError(
            "Pixal attempt Python/Torch/CUDA versions do not match the executable"
        )
    started = _parse_utc_timestamp(
        attempt.get("started_at_utc"), "Pixal attempt start"
    )
    finished = _parse_utc_timestamp(
        attempt.get("finished_at_utc"), "Pixal attempt finish"
    )
    if finished < started:
        raise InstanceContractError("Pixal attempt finish precedes its start")
    if (
        not isinstance(attempt.get("returncode"), int)
        or isinstance(attempt.get("returncode"), bool)
        or attempt.get("returncode") != 0
        or attempt.get("preflight_reauthenticated") is not True
        or attempt.get("postflight_reauthenticated") is not True
        or attempt.get("failure_evidence") != []
    ):
        raise InstanceContractError("Pixal attempt did not prove a clean authenticated success")
    staging = _require_exact_mapping(
        attempt.get("staging"),
        frozenset({"path", "created", "preserved_after_success"}),
        "Pixal attempt staging evidence",
    )
    staging_value = staging.get("path")
    asset_root = common.absolute(Path(lineage["pixal_pbr_glb"]["path"])).parent
    staging_path = (
        common.absolute(Path(staging_value))
        if isinstance(staging_value, str) and Path(staging_value).is_absolute()
        else None
    )
    if (
        staging_path is None
        or staging_path.parent != asset_root.parent
        or not staging_path.name.endswith(".staging")
        or staging.get("created") is not True
        or staging.get("preserved_after_success") is not False
        or os.path.lexists(staging_path)
    ):
        raise InstanceContractError("Pixal attempt staging evidence is invalid")
    expected_start_path = (
        asset_root.parent
        / ".attempts"
        / asset_id
        / f"{attempt_id}.started.json"
    )
    _validate_pixal_attempt_start_ledger(
        attempt.get("start_ledger"),
        expected_path=expected_start_path,
        attempt_id=attempt_id,
        case_id=case_id,
        asset_id=asset_id,
        base_avatar_id=base_avatar_id,
        job_record=job_record,
        executor_record=executor_record,
        argv=expected_argv,
        started_at_utc=str(attempt.get("started_at_utc")),
        staging_path=staging_path,
        expected_execution_guard_before=guard_before,
    )
    _validate_pixal_execution_log(
        attempt.get("execution_log"),
        expected_path=asset_root / "execution.log",
        attempt_id=attempt_id,
        returncode=0,
        logical_argv=expected_argv,
        python_executable=str(python_executable),
        staging_path=staging_path,
        public_glb=common.absolute(Path(lineage["pixal_pbr_glb"]["path"])),
        public_manifest=common.absolute(Path(lineage["pixal_manifest"]["path"])),
    )
    if attempt.get("publication") != {
        "policy": "atomic_no_replace",
        "glb_published": True,
        "manifest_published": True,
    }:
        raise InstanceContractError("Pixal attempt publication was not atomic no-replace")
    inventory = _require_exact_mapping(
        attempt.get("model_inventory"),
        frozenset(
            {
                "pixal_revision",
                "dino_revision",
                "pixal_snapshot_inventory_sha256",
                "dino_snapshot_inventory_sha256",
            }
        ),
        "Pixal attempt model inventory provenance",
    )
    if (
        inventory.get("pixal_revision") != PIXAL3D_REVISION
        or inventory.get("dino_revision") != DINO_REVISION
        or inventory.get("pixal_snapshot_inventory_sha256")
        != model_evidence["pixal"]["inventory_sha256"]
        or inventory.get("dino_snapshot_inventory_sha256")
        != model_evidence["dino"]["inventory_sha256"]
    ):
        raise InstanceContractError("Pixal attempt model inventory provenance is invalid")
    expected_licenses = {
        "pixal_license_sha256": model_evidence["pixal"]["license"]["sha256"],
        "dino_license_sha256": model_evidence["dino"]["license"]["sha256"],
    }
    if attempt.get("licenses") != expected_licenses:
        raise InstanceContractError("Pixal attempt license snapshots are not pinned")
    _require_embedded_record(
        attempt.get("output_glb"), lineage["pixal_pbr_glb"], "Pixal attempt output GLB"
    )
    _require_embedded_record(
        attempt.get("output_manifest"),
        lineage["pixal_manifest"],
        "Pixal attempt output manifest",
    )
    if common.absolute(Path(lineage["pixal_attempt"]["path"])) != asset_root / "pixal_attempt.json":
        raise InstanceContractError("Pixal attempt path is not canonical")


def _validate_pixal_attribute_failure_bundle_once(bundle_dir: Path) -> dict[str, Any]:
    bundle = common.require_real_directory(
        bundle_dir,
        "Pixal failure bundle",
        InstanceContractError,
        mode=0o555,
    )
    manifest_path = bundle / "failure_manifest.json"
    payload, _ = common.load_json_mapping_record(
        manifest_path,
        root=bundle,
        description="Pixal failure bundle manifest",
        error_type=InstanceContractError,
        require_mode=0o444,
    )
    _require_exact_mapping(
        payload,
        PIXAL_ATTRIBUTE_FAILURE_BUNDLE_FIELDS,
        "Pixal failure bundle manifest",
    )
    common.reject_user_approval(
        payload, InstanceContractError, "Pixal failure bundle manifest"
    )
    attempt_id = common.require_id(
        payload.get("attempt_id"), "Pixal failure attempt_id", InstanceContractError
    )
    case_id = common.require_id(
        payload.get("case_id"), "Pixal failure case_id", InstanceContractError
    )
    asset_id = common.require_id(
        payload.get("asset_id"), "Pixal failure asset_id", InstanceContractError
    )
    base_avatar_id = common.require_id(
        payload.get("base_avatar_id"),
        "Pixal failure base_avatar_id",
        InstanceContractError,
    )
    if (
        payload.get("schema") != PIXAL_ATTRIBUTE_FAILURE_BUNDLE_SCHEMA
        or payload.get("status") != "failed"
        or ATTRIBUTE_CASE_BASE.get(case_id) != base_avatar_id
        or asset_id != f"route2_{case_id}_v1"
        or bundle.name != attempt_id
        or bundle.parent.name != asset_id
        or bundle.parent.parent.name != ".failed_attempts"
    ):
        raise InstanceContractError("Pixal failure bundle identity/path is invalid")
    output_root = bundle.parent.parent.parent
    job_path = output_root / f"{asset_id}.pixal_job.json"
    job_payload, job_file = common.load_json_mapping_record(
        job_path,
        root=output_root,
        description="failed Pixal job",
        error_type=InstanceContractError,
        require_mode=0o444,
    )
    job_record = {
        key: job_file[key] for key in ("path", "sha256", "size_bytes")
    }
    supplied_job = _require_exact_mapping(
        payload.get("job"),
        frozenset({"path", "sha256", "size_bytes"}),
        "failed Pixal job descriptor",
    )
    if (
        dict(supplied_job) != job_record
    ):
        raise InstanceContractError("Pixal failure bundle job binding is invalid")
    argv, current_executor = _validate_pixal_failure_job(
        job_payload,
        job_path=job_path,
        output_root=output_root,
        case_id=case_id,
        asset_id=asset_id,
        base_avatar_id=base_avatar_id,
    )
    start_path = (
        output_root / ".attempts" / asset_id / f"{attempt_id}.started.json"
    )
    start_probe, _ = common.load_json_mapping_record(
        start_path,
        root=start_path.parent,
        description="failed Pixal start ledger",
        error_type=InstanceContractError,
        require_mode=0o444,
    )
    start_staging = _require_exact_mapping(
        start_probe.get("staging"),
        frozenset({"path", "created"}),
        "failed Pixal start staging",
    )
    staging_value = start_staging.get("path")
    if not isinstance(staging_value, str) or not Path(staging_value).is_absolute():
        raise InstanceContractError("failed Pixal staging path is invalid")
    staging_path = common.absolute(Path(staging_value))
    if staging_path.parent != output_root or not staging_path.name.endswith(".staging"):
        raise InstanceContractError("failed Pixal staging path is not canonical")
    _validate_pixal_attempt_start_ledger(
        payload.get("start_ledger"),
        expected_path=start_path,
        attempt_id=attempt_id,
        case_id=case_id,
        asset_id=asset_id,
        base_avatar_id=base_avatar_id,
        job_record=job_record,
        executor_record=current_executor,
        argv=argv,
        started_at_utc=str(start_probe.get("started_at_utc")),
        staging_path=staging_path,
        expected_execution_guard_before=None,
    )
    common.require_id(
        payload.get("failure_stage"),
        "Pixal failure stage",
        InstanceContractError,
    )
    error_value = _require_exact_mapping(
        payload.get("error"),
        frozenset({"type", "message"}),
        "Pixal failure error",
    )
    if any(
        not isinstance(error_value.get(field), str) or not error_value[field]
        for field in ("type", "message")
    ):
        raise InstanceContractError("Pixal failure error is incomplete")
    returncode = payload.get("returncode")
    if returncode is not None and (
        not isinstance(returncode, int) or isinstance(returncode, bool)
    ):
        raise InstanceContractError("Pixal failure returncode is invalid")

    records: list[dict[str, Any]] = []
    for path in sorted(bundle.rglob("*"), key=lambda item: item.relative_to(bundle).as_posix()):
        relative = path.relative_to(bundle).as_posix()
        if path.is_symlink():
            raise InstanceContractError("Pixal failure bundle contains a symlink")
        if path.is_dir():
            common.require_real_directory(
                path,
                f"Pixal failure bundle directory {relative}",
                InstanceContractError,
                mode=0o555,
            )
            continue
        if path == manifest_path:
            continue
        record = common.file_record(
            path,
            root=bundle,
            description=f"Pixal failure artifact {relative}",
            error_type=InstanceContractError,
            require_mode=0o444,
        )
        records.append(
            {
                key: record[key]
                for key in ("relative_path", "sha256", "size_bytes", "mode")
            }
        )
    if payload.get("artifacts") != records:
        raise InstanceContractError("Pixal failure bundle inventory changed")
    return {**dict(payload), "validated_inventory": records}


def validate_pixal_attribute_failure_bundle(bundle_dir: Path) -> dict[str, Any]:
    return common.stable_mapping_snapshot(
        lambda: _validate_pixal_attribute_failure_bundle_once(bundle_dir),
        InstanceContractError,
        "Pixal failure bundle",
    )


def _offline_file_descriptor(
    value: Any,
    *,
    description: str,
    extra_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    fields = frozenset({"path", "sha256", "size_bytes"}) | extra_fields
    record = _require_exact_mapping(value, fields, description)
    path = record.get("path")
    digest = record.get("sha256")
    size = record.get("size_bytes")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or not isinstance(digest, str)
        or common.SHA256_RE.fullmatch(digest) is None
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise InstanceContractError(f"{description} descriptor is invalid")
    return dict(record)


def _validate_pixal_failure_job(
    job: Mapping[str, Any],
    *,
    job_path: Path,
    output_root: Path,
    case_id: str,
    asset_id: str,
    base_avatar_id: str,
) -> tuple[list[str], dict[str, Any]]:
    """Authenticate the immutable failed job without requiring live model scans."""

    from tools import human_attribute_pixal_contract as pixal_contract
    from tools.spike_rlr import human_attribute_review

    _require_exact_mapping(job, PIXAL_ATTRIBUTE_JOB_FIELDS, "failed Pixal job")
    common.reject_user_approval(job, InstanceContractError, "failed Pixal job")
    if (
        job.get("schema") != "pixal3d_human_attribute_job_v1"
        or job.get("case_id") != case_id
        or job.get("asset_id") != asset_id
        or job.get("base_asset_id") != base_avatar_id
        or job.get("state_classification") != "research_candidate"
        or job.get("model_revision") != PIXAL3D_REVISION
        or job.get("dino_revision") != DINO_REVISION
        or job.get("parameters")
        != {"seed": 42, "manual_fov": 0.2, "resolution": 1024, "low_vram": True}
        or job.get("output_policy") != "atomic_no_replace"
    ):
        raise InstanceContractError("failed Pixal job identity or parameters are invalid")

    output_glb = output_root / asset_id / "canary_1024_seed42.glb"
    output_manifest = output_glb.with_suffix(".manifest.json")
    if (
        common.absolute(job_path) != output_root / f"{asset_id}.pixal_job.json"
        or job.get("output_glb") != str(output_glb)
        or job.get("output_manifest") != str(output_manifest)
    ):
        raise InstanceContractError("failed Pixal job/output paths are not canonical")

    candidate = _offline_file_descriptor(
        job.get("candidate_manifest"),
        description="failed Pixal candidate manifest",
    )
    decision = _offline_file_descriptor(
        job.get("agent_2d_decision"),
        description="failed Pixal agent 2D decision",
        extra_fields=frozenset({"status"}),
    )
    rgba = _offline_file_descriptor(
        job.get("input_rgba"),
        description="failed Pixal input RGBA",
        extra_fields=frozenset({"mode", "size", "alpha_min", "alpha_max"}),
    )
    candidate_path = common.absolute(Path(candidate["path"]))
    decision_path = common.absolute(Path(decision["path"]))
    rgba_path = common.absolute(Path(rgba["path"]))
    if (
        candidate_path.name != "candidate_manifest.json"
        or decision_path
        != common.absolute(human_attribute_review.decision_path(candidate_path.parent))
        or rgba_path != candidate_path.parent / "candidate_rgba.png"
        or decision.get("status")
        != "agent_qa_passed_pending_user_acceptance"
    ):
        raise InstanceContractError("failed Pixal candidate/decision/input paths are invalid")
    size = rgba.get("size")
    alpha_min = rgba.get("alpha_min")
    alpha_max = rgba.get("alpha_max")
    if (
        rgba.get("mode") != "RGBA"
        or not isinstance(size, list)
        or len(size) != 2
        or any(
            not isinstance(number, int) or isinstance(number, bool) or number <= 0
            for number in size
        )
        or not isinstance(alpha_min, int)
        or isinstance(alpha_min, bool)
        or not isinstance(alpha_max, int)
        or isinstance(alpha_max, bool)
        or not 0 <= alpha_min < 255
        or not 0 < alpha_max <= 255
        or alpha_min > alpha_max
    ):
        raise InstanceContractError("failed Pixal input RGBA metadata is invalid")

    wrapper = _require_current_embedded_record(
        job.get("wrapper"), "failed Pixal wrapper"
    )
    if (
        wrapper["path"]
        != str(common.absolute(Path(pixal_contract.PIXAL_WRAPPER_PATH)))
        or wrapper["sha256"] != pixal_contract.PIXAL_WRAPPER_SHA256
    ):
        raise InstanceContractError("failed Pixal wrapper is not pinned")
    expected_argv = [
        wrapper["path"],
        "--backend",
        "pixal3d",
        "--image",
        str(rgba_path),
        "--output",
        str(output_glb),
        "--gpu",
        "3",
        "--seed",
        "42",
        "--resolution",
        "1024",
        "--manual-fov",
        "0.2",
        "--low-vram",
    ]
    executor = _require_exact_mapping(
        job.get("executor"), PIXAL_EXECUTOR_FIELDS, "failed Pixal executor"
    )
    executor_record = _require_current_embedded_record(
        {key: executor.get(key) for key in ("path", "sha256", "size_bytes")},
        "failed Pixal executor code",
    )
    if (
        executor_record["path"]
        != str(common.absolute(Path(pixal_contract.EXECUTOR_PATH)))
        or executor.get("kind") != "atomic_pixal3d_executor_v1"
        or executor.get("argv") != expected_argv
        or executor.get("execution_authorized") is not True
        or executor.get("atomic_no_replace") is not True
    ):
        raise InstanceContractError("failed Pixal executor argv or binding is invalid")
    return expected_argv, executor_record


def _require_glb_index(
    value: Any,
    inventory: list[Any],
    description: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value >= len(inventory)
    ):
        raise InstanceContractError(f"{description} index is invalid")
    return value


def _pixal_glb_buffer_view(
    document: Mapping[str, Any],
    binary: bytes,
    index: Any,
    description: str,
) -> tuple[Mapping[str, Any], int, int]:
    views = document.get("bufferViews")
    if not isinstance(views, list):
        raise InstanceContractError(f"{description} bufferView inventory is invalid")
    view_index = _require_glb_index(index, views, f"{description} bufferView")
    view = views[view_index]
    if not isinstance(view, Mapping):
        raise InstanceContractError(f"{description} bufferView is invalid")
    buffer_index = view.get("buffer")
    offset = view.get("byteOffset", 0)
    length = view.get("byteLength")
    buffers = document.get("buffers")
    declared_length = (
        buffers[0].get("byteLength")
        if isinstance(buffers, list)
        and len(buffers) == 1
        and isinstance(buffers[0], Mapping)
        else None
    )
    if (
        buffer_index != 0
        or isinstance(buffer_index, bool)
        or not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
        or not isinstance(declared_length, int)
        or isinstance(declared_length, bool)
        or offset + length > declared_length
        or offset + length > len(binary)
    ):
        raise InstanceContractError(f"{description} bufferView range is invalid")
    stride = view.get("byteStride")
    if stride is not None and (
        not isinstance(stride, int)
        or isinstance(stride, bool)
        or not 4 <= stride <= 252
        or stride % 4 != 0
    ):
        raise InstanceContractError(f"{description} bufferView stride is invalid")
    return view, offset, offset + length


def _validate_pixal_mesh_accessor(
    document: Mapping[str, Any],
    binary: bytes,
    accessor_index: int,
    *,
    expected_type: str,
    expected_component_type: int,
    description: str,
) -> int:
    accessors = document["accessors"]
    accessor = accessors[accessor_index]
    if not isinstance(accessor, Mapping) or "sparse" in accessor:
        raise InstanceContractError(f"{description} accessor is invalid")
    count = accessor.get("count")
    byte_offset = accessor.get("byteOffset", 0)
    if (
        accessor.get("type") != expected_type
        or accessor.get("componentType") != expected_component_type
        or accessor.get("normalized", False) is not False
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or not isinstance(byte_offset, int)
        or isinstance(byte_offset, bool)
        or byte_offset < 0
    ):
        raise InstanceContractError(f"{description} accessor metadata is invalid")
    view, start, end = _pixal_glb_buffer_view(
        document,
        binary,
        accessor.get("bufferView"),
        description,
    )
    component_size = 4
    component_count = {"VEC2": 2, "VEC3": 3}[expected_type]
    element_size = component_size * component_count
    stride = view.get("byteStride", element_size)
    if stride < element_size:
        raise InstanceContractError(f"{description} accessor stride is too small")
    required_end = start + byte_offset + (count - 1) * stride + element_size
    if required_end > end:
        raise InstanceContractError(f"{description} accessor exceeds its bufferView")
    return count


def _validate_pixal_packed_image(
    payload: bytes,
    mime_type: str,
    description: str,
) -> None:
    from PIL import Image, UnidentifiedImageError

    expected_format = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }.get(mime_type)
    if expected_format is None or not payload:
        raise InstanceContractError(f"{description} MIME type or payload is invalid")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if (
                image.format != expected_format
                or image.width <= 0
                or image.height <= 0
            ):
                raise InstanceContractError(
                    f"{description} decoded image format or dimensions are invalid"
                )
    except (UnidentifiedImageError, OSError) as error:
        raise InstanceContractError(f"{description} packed image cannot be decoded") from error


def _pixal_texture_image_index(
    texture: Any,
    images: list[Any],
    description: str,
    *,
    extensions_used: Any,
) -> int:
    if not isinstance(texture, Mapping):
        raise InstanceContractError(f"{description} descriptor is invalid")
    source = texture.get("source")
    extension_source: Any = None
    extensions = texture.get("extensions")
    if isinstance(extensions, Mapping):
        webp = extensions.get("EXT_texture_webp")
        if isinstance(webp, Mapping):
            if (
                not isinstance(extensions_used, list)
                or "EXT_texture_webp" not in extensions_used
            ):
                raise InstanceContractError(
                    f"{description} uses undeclared EXT_texture_webp"
                )
            extension_source = webp.get("source")
    if source is None:
        source = extension_source
    elif extension_source is not None and extension_source != source:
        raise InstanceContractError(
            f"{description} core and EXT_texture_webp image references disagree"
        )
    return _require_glb_index(source, images, f"{description} image")


def _validate_pixal_pbr_document(
    document: Mapping[str, Any],
    binary: bytes | None,
    description: str,
) -> None:
    """Require real textured PBR mesh structure from the authenticated GLB bytes."""

    buffers = document.get("buffers")
    meshes = document.get("meshes")
    materials = document.get("materials")
    textures = document.get("textures")
    images = document.get("images")
    accessors = document.get("accessors")
    buffer_views = document.get("bufferViews")
    if any(
        not isinstance(value, list) or not value
        for value in (meshes, materials, textures, images, accessors, buffer_views)
    ):
        raise InstanceContractError(
            f"{description} must contain non-empty mesh, material, texture, image, "
            "accessor, and bufferView inventories"
        )
    if (
        not isinstance(binary, bytes)
        or not binary
        or not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], Mapping)
        or "uri" in buffers[0]
        or not isinstance(buffers[0].get("byteLength"), int)
        or isinstance(buffers[0].get("byteLength"), bool)
        or buffers[0]["byteLength"] <= 0
        or buffers[0]["byteLength"] > len(binary)
        or len(binary) - buffers[0]["byteLength"] > 3
    ):
        raise InstanceContractError(
            f"{description} must contain one non-empty packed GLB BIN buffer"
        )

    referenced_materials: set[int] = set()
    for mesh_index, mesh in enumerate(meshes):
        if not isinstance(mesh, Mapping):
            raise InstanceContractError(f"{description} mesh {mesh_index} is invalid")
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise InstanceContractError(
                f"{description} mesh {mesh_index} has no primitives"
            )
        for primitive_index, primitive in enumerate(primitives):
            if not isinstance(primitive, Mapping):
                raise InstanceContractError(
                    f"{description} mesh {mesh_index} primitive {primitive_index} is invalid"
                )
            attributes = primitive.get("attributes")
            if not isinstance(attributes, Mapping):
                raise InstanceContractError(
                    f"{description} mesh {mesh_index} primitive {primitive_index} "
                    "has no attributes"
                )
            position_index = _require_glb_index(
                attributes.get("POSITION"),
                accessors,
                f"{description} POSITION accessor",
            )
            texcoord_index = _require_glb_index(
                attributes.get("TEXCOORD_0"),
                accessors,
                f"{description} TEXCOORD_0 accessor",
            )
            position_count = _validate_pixal_mesh_accessor(
                document,
                binary,
                position_index,
                expected_type="VEC3",
                expected_component_type=5126,
                description=f"{description} POSITION",
            )
            texcoord_count = _validate_pixal_mesh_accessor(
                document,
                binary,
                texcoord_index,
                expected_type="VEC2",
                expected_component_type=5126,
                description=f"{description} TEXCOORD_0",
            )
            if texcoord_count != position_count:
                raise InstanceContractError(
                    f"{description} POSITION/TEXCOORD_0 counts are invalid"
                )
            referenced_materials.add(
                _require_glb_index(
                    primitive.get("material"),
                    materials,
                    f"{description} primitive material",
                )
            )

    referenced_textures: set[int] = set()
    for material_index in sorted(referenced_materials):
        material = materials[material_index]
        if not isinstance(material, Mapping):
            raise InstanceContractError(
                f"{description} material {material_index} is invalid"
            )
        pbr = material.get("pbrMetallicRoughness")
        if not isinstance(pbr, Mapping):
            raise InstanceContractError(
                f"{description} material {material_index} has no PBR definition"
            )
        for field in ("baseColorTexture", "metallicRoughnessTexture"):
            reference = pbr.get(field)
            if not isinstance(reference, Mapping):
                raise InstanceContractError(
                    f"{description} material {material_index} has no {field}"
                )
            referenced_textures.add(
                _require_glb_index(
                    reference.get("index"),
                    textures,
                    f"{description} {field}",
                )
            )

    referenced_images = {
        _pixal_texture_image_index(
            textures[texture_index],
            images,
            f"{description} texture {texture_index}",
            extensions_used=document.get("extensionsUsed"),
        )
        for texture_index in referenced_textures
    }
    for image_index in sorted(referenced_images):
        image = images[image_index]
        if not isinstance(image, Mapping):
            raise InstanceContractError(
                f"{description} image {image_index} is invalid"
            )
        uri = image.get("uri")
        buffer_view = image.get("bufferView")
        if isinstance(uri, str):
            try:
                header, encoded = uri.split(",", 1)
                mime_type = header.removeprefix("data:").removesuffix(";base64")
                if header != f"data:{mime_type};base64" or not encoded:
                    raise ValueError("non-canonical data URI")
                payload = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as error:
                raise InstanceContractError(
                    f"{description} image {image_index} has invalid base64 data URI"
                ) from error
            _validate_pixal_packed_image(
                payload,
                mime_type,
                f"{description} image {image_index}",
            )
        else:
            _, start, end = _pixal_glb_buffer_view(
                document,
                binary,
                buffer_view,
                f"{description} image {image_index}",
            )
            _validate_pixal_packed_image(
                binary[start:end],
                image.get("mimeType"),
                f"{description} image {image_index}",
            )


def _validate_base_lineage_semantics(
    lineage: Mapping[str, Mapping[str, Any]], *, asset_id: str
) -> None:
    source = lineage["source_image"]
    candidate = lineage["flux_candidate"]
    pixal_input = lineage["pixal_input_rgba"]
    for record, description in (
        (source, "approved source image"),
        (candidate, "FLUX candidate image"),
        (pixal_input, "Pixal input RGBA"),
    ):
        _require_png(record, description)
    flux = _lineage_json(lineage["flux_manifest"], "FLUX candidate manifest")
    if (
        flux.get("schema_version") != "human_reference_candidate_v1"
        or flux.get("asset_id") != asset_id
        or flux.get("model_revision") != FLUX2_REVISION
        or flux.get("input_sha256") != source["sha256"]
        or flux.get("output_sha256") != candidate["sha256"]
    ):
        raise InstanceContractError("FLUX source/candidate lineage is invalid")
    review = _lineage_json(lineage["source_review"], "source reference review")
    if (
        review.get("schema_version") != "human_reference_review_v1"
        or review.get("asset_id") != asset_id
        or review.get("decision") != "approved"
        or review.get("source_sha256") != source["sha256"]
        or review.get("candidate_sha256") != candidate["sha256"]
        or review.get("candidate_manifest_sha256") != lineage["flux_manifest"]["sha256"]
    ):
        raise InstanceContractError("approved source reference lineage is invalid")
    pixal = _lineage_json(lineage["pixal_manifest"], "Pixal manifest")
    pixal_model = _require_exact_mapping(
        pixal.get("model"),
        frozenset({"snapshot", "revision"}),
        "base Pixal manifest model",
    )
    pixal_dino = _require_exact_mapping(
        pixal.get("dino"),
        frozenset({"snapshot", "revision"}),
        "base Pixal manifest DINO",
    )
    pixal_manifest_input = pixal.get("input")
    pixal_manifest_output = pixal.get("output")
    if (
        pixal.get("backend") != "pixal3d"
        or pixal_model.get("revision") != PIXAL3D_REVISION
        or pixal_dino.get("revision") != DINO_REVISION
        or not isinstance(pixal_manifest_input, Mapping)
        or pixal_manifest_input.get("path") != pixal_input["path"]
        or pixal_manifest_input.get("sha256") != pixal_input["sha256"]
        or not isinstance(pixal_manifest_output, Mapping)
        or pixal_manifest_output.get("path") != lineage["pixal_pbr_glb"]["path"]
        or pixal_manifest_output.get("sha256")
        != lineage["pixal_pbr_glb"]["sha256"]
        or pixal_manifest_output.get("bytes")
        != lineage["pixal_pbr_glb"]["size_bytes"]
        or pixal.get("parameters")
        != {"low_vram": True, "manual_fov": 0.2, "resolution": 1024, "seed": 42}
    ):
        raise InstanceContractError("Pixal manifest/model/output lineage is invalid")
    _validate_model_snapshot(
        pixal_model.get("snapshot"), PIXAL3D_REVISION, "Pixal3D model snapshot"
    )
    _validate_model_snapshot(
        pixal_dino.get("snapshot"), DINO_REVISION, "DINO model snapshot"
    )
    glb_document, glb_binary, current_glb = common.load_glb_document_binary_record(
        Path(str(lineage["pixal_pbr_glb"]["path"])),
        root=Path(str(lineage["pixal_pbr_glb"]["root"])),
        description="Pixal PBR GLB",
        error_type=InstanceContractError,
    )
    if any(
        current_glb[key] != lineage["pixal_pbr_glb"][key]
        for key in ("path", "relative_path", "sha256", "size_bytes", "mode")
    ):
        raise InstanceContractError("Pixal PBR GLB changed during parsing")
    _validate_pixal_pbr_document(glb_document, glb_binary, "Pixal PBR GLB")


def _validate_attribute_lineage_semantics(
    lineage: Mapping[str, Mapping[str, Any]],
    *,
    asset_id: str,
    base_avatar_id: str,
    case_id: str,
) -> None:
    from tools import route2_human_qualified_candidate as qualified
    from tools.spike_rlr import human_attribute_review

    try:
        base = qualified.validate_qualified_candidate(
            Path(str(lineage["base_qualified_candidate"]["path"]))
        )
    except qualified.QualificationError as error:
        raise InstanceContractError(f"base qualified candidate is not current: {error}") from error
    if base.get("asset_id") != base_avatar_id or base.get("base_avatar_id") != base_avatar_id:
        raise InstanceContractError("base qualified candidate identity does not match base_avatar_id")
    candidate_path = Path(str(lineage["attribute_candidate_manifest"]["path"]))
    candidate_payload = _lineage_json(
        lineage["attribute_candidate_manifest"], "attribute candidate manifest"
    )
    base_qualification = _require_exact_mapping(
        candidate_payload.get("base_route2_qualification"),
        frozenset(
            {
                "asset_id",
                "status",
                "qualified_candidate",
                "final_branch",
                "review_dir",
            }
        ),
        "attribute candidate base qualification",
    )
    _require_embedded_record(
        base_qualification.get("qualified_candidate"),
        lineage["base_qualified_candidate"],
        "attribute candidate qualified base",
    )
    final_branch = _require_exact_mapping(
        base_qualification.get("final_branch"),
        frozenset({"branch_id", "path", "relative_root"}),
        "attribute candidate qualified base branch",
    )
    if (
        base_qualification.get("asset_id") != base_avatar_id
        or base_qualification.get("status")
        != "agent_qa_passed_pending_user_acceptance"
        or dict(final_branch) != base.get("final_branch")
        or base_qualification.get("review_dir") != base.get("dynamic", {}).get("review_dir")
    ):
        raise InstanceContractError(
            "attribute candidate base qualification does not bind the exact qualified branch"
        )
    try:
        decision = human_attribute_review.assert_agent_2d_qa_passed(candidate_path.parent)
    except human_attribute_review.AttributeReviewError as error:
        raise InstanceContractError(f"attribute owner review is not accepted: {error}") from error
    if (
        candidate_path.name != "candidate_manifest.json"
        or decision.get("case_id") != case_id
        or decision.get("base_asset_id") != base_avatar_id
        or decision.get("downstream_asset_id") != asset_id
        or lineage["attribute_agent_decision"]["path"]
        != str(human_attribute_review.decision_path(candidate_path.parent))
        or lineage["attribute_agent_decision"]["sha256"]
        != common.sha256_file(human_attribute_review.decision_path(candidate_path.parent))
    ):
        raise InstanceContractError("attribute candidate/agent decision lineage is invalid")
    rgba_record = candidate_payload.get("artifacts", {}).get("candidate_rgba.png")
    if (
        not isinstance(rgba_record, Mapping)
        or rgba_record.get("sha256") != lineage["candidate_rgba"]["sha256"]
        or rgba_record.get("size_bytes") != lineage["candidate_rgba"]["size_bytes"]
    ):
        raise InstanceContractError("attribute candidate RGBA lineage is invalid")
    pixal_job = _lineage_json(lineage["pixal_job"], "Pixal attribute job")
    expected_argv, expected_executor = _validate_pixal_attribute_job(
        pixal_job,
        lineage,
        asset_id=asset_id,
        base_avatar_id=base_avatar_id,
        case_id=case_id,
    )
    attempt = _lineage_json(lineage["pixal_attempt"], "Pixal attempt ledger")
    pixal = _lineage_json(lineage["pixal_manifest"], "attribute Pixal manifest")
    pixal_input = _require_exact_mapping(
        pixal.get("input"),
        frozenset({"path", "sha256", "mode", "size", "alpha_min", "alpha_max"}),
        "attribute Pixal manifest input",
    )
    pixal_output = _require_exact_mapping(
        pixal.get("output"),
        frozenset({"path", "sha256", "bytes"}),
        "attribute Pixal manifest output",
    )
    pixal_model = _require_exact_mapping(
        pixal.get("model"),
        frozenset({"snapshot", "revision"}),
        "attribute Pixal manifest model",
    )
    pixal_dino = _require_exact_mapping(
        pixal.get("dino"),
        frozenset({"snapshot", "revision"}),
        "attribute Pixal manifest DINO",
    )
    if (
        set(pixal)
        != {
            "backend",
            "asset_id",
            "case_id",
            "base_avatar_id",
            "input",
            "output",
            "model",
            "dino",
            "parameters",
        }
        or pixal.get("backend") != "pixal3d"
        or pixal.get("asset_id") != asset_id
        or pixal.get("case_id") != case_id
        or pixal.get("base_avatar_id") != base_avatar_id
        or pixal_input.get("path") != lineage["candidate_rgba"]["path"]
        or pixal_input.get("sha256") != lineage["candidate_rgba"]["sha256"]
        or pixal_input.get("mode") != "RGBA"
        or pixal_input.get("size") != pixal_job["input_rgba"]["size"]
        or pixal_input.get("alpha_min") != pixal_job["input_rgba"]["alpha_min"]
        or pixal_input.get("alpha_max") != pixal_job["input_rgba"]["alpha_max"]
        or pixal_model.get("revision") != PIXAL3D_REVISION
        or pixal_dino.get("revision") != DINO_REVISION
        or pixal_output.get("path") != lineage["pixal_pbr_glb"]["path"]
        or pixal_output.get("sha256") != lineage["pixal_pbr_glb"]["sha256"]
        or pixal_output.get("bytes") != lineage["pixal_pbr_glb"]["size_bytes"]
        or pixal.get("parameters")
        != {"low_vram": True, "manual_fov": 0.2, "resolution": 1024, "seed": 42}
    ):
        raise InstanceContractError("attribute Pixal manifest lineage is invalid")
    pixal_snapshot_evidence = _validate_model_snapshot(
        pixal_model.get("snapshot"), PIXAL3D_REVISION, "Pixal3D model snapshot"
    )
    dino_snapshot_evidence = _validate_model_snapshot(
        pixal_dino.get("snapshot"), DINO_REVISION, "DINO model snapshot"
    )
    _validate_pixal_attribute_attempt(
        attempt,
        pixal_job,
        expected_argv,
        expected_executor,
        lineage,
        asset_id=asset_id,
        base_avatar_id=base_avatar_id,
        case_id=case_id,
        model_evidence={
            "pixal": pixal_snapshot_evidence,
            "dino": dino_snapshot_evidence,
        },
    )
    for lineage_id in ("pixal_job", "pixal_attempt", "pixal_manifest", "pixal_pbr_glb"):
        if lineage[lineage_id]["mode"] != "0444":
            raise InstanceContractError(
                f"attribute Pixal evidence must be immutable: {lineage_id}"
            )
    glb_document, glb_binary, current_glb = common.load_glb_document_binary_record(
        Path(str(lineage["pixal_pbr_glb"]["path"])),
        root=Path(str(lineage["pixal_pbr_glb"]["root"])),
        description="attribute Pixal PBR GLB",
        error_type=InstanceContractError,
    )
    if any(
        current_glb[key] != lineage["pixal_pbr_glb"][key]
        for key in ("path", "relative_path", "sha256", "size_bytes", "mode")
    ):
        raise InstanceContractError("attribute Pixal PBR GLB changed during parsing")
    _validate_pixal_pbr_document(
        glb_document,
        glb_binary,
        "attribute Pixal PBR GLB",
    )


def _validate_lineage_semantics(
    lineage: Mapping[str, Mapping[str, Any]],
    *,
    asset_id: str,
    base_avatar_id: str,
    case: Mapping[str, str],
) -> None:
    if case["kind"] == "base_avatar":
        _validate_base_lineage_semantics(lineage, asset_id=asset_id)
    else:
        _validate_attribute_lineage_semantics(
            lineage,
            asset_id=asset_id,
            base_avatar_id=base_avatar_id,
            case_id=case["case_id"],
        )


def build_instance_contract(
    *,
    asset_id: str,
    base_avatar_id: str,
    case_id: str,
    case_kind: str,
    output_root: Path,
    source_lineage: Mapping[str, Mapping[str, Any]],
    allowed_branch_dag: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    asset_id = common.require_id(asset_id, "asset_id", InstanceContractError)
    base_avatar_id = common.require_id(
        base_avatar_id, "base_avatar_id", InstanceContractError
    )
    case = _validate_case(
        asset_id=asset_id,
        base_avatar_id=base_avatar_id,
        case_id=case_id,
        case_kind=case_kind,
    )
    root = common.require_real_directory(
        output_root, "canonical output root", InstanceContractError
    )
    if root.name != asset_id:
        raise InstanceContractError("canonical output root name must equal asset_id")
    dag = validate_branch_dag(
        copy.deepcopy(DEFAULT_BRANCH_DAG if allowed_branch_dag is None else allowed_branch_dag)
    )
    if dag != DEFAULT_BRANCH_DAG:
        raise InstanceContractError("allowed branch DAG must equal the pinned branch DAG")
    if not isinstance(source_lineage, Mapping) or not source_lineage:
        raise InstanceContractError("source_lineage must be a non-empty mapping")
    _validate_lineage_profile(source_lineage, case_kind=case["kind"])
    lineage = {
        lineage_id: _source_record(lineage_id, source_lineage[lineage_id])
        for lineage_id in sorted(source_lineage)
    }
    _validate_lineage_semantics(
        lineage, asset_id=asset_id, base_avatar_id=base_avatar_id, case=case
    )
    result = {
        "schema": SCHEMA,
        "asset_id": asset_id,
        "base_avatar_id": base_avatar_id,
        "case": case,
        "coordinate_frame": dict(COORDINATE_FRAME),
        "canonical_output_root": str(root),
        "source_lineage": lineage,
        "allowed_branch_dag": dag,
        "state_classification": "research_candidate",
        "publication": {"no_replace": True, "artifact_mode": "0444"},
        "user_acceptance": "pending_user_review",
    }
    common.reject_user_approval(result, InstanceContractError, "instance contract")
    return result


def _validate_payload(
    payload: Any, *, expected_output_root: Path | None = None
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "asset_id",
        "base_avatar_id",
        "case",
        "coordinate_frame",
        "canonical_output_root",
        "source_lineage",
        "allowed_branch_dag",
        "state_classification",
        "publication",
        "user_acceptance",
    }:
        raise InstanceContractError("instance contract fields are incomplete or unexpected")
    common.reject_user_approval(payload, InstanceContractError, "instance contract")
    if payload.get("schema") != SCHEMA:
        raise InstanceContractError(f"instance contract schema must be {SCHEMA}")
    asset_id = common.require_id(payload.get("asset_id"), "asset_id", InstanceContractError)
    base_avatar_id = common.require_id(
        payload.get("base_avatar_id"), "base_avatar_id", InstanceContractError
    )
    case_value = payload.get("case")
    if not isinstance(case_value, Mapping) or set(case_value) != {"case_id", "kind"}:
        raise InstanceContractError("instance case descriptor is invalid")
    case = _validate_case(
        asset_id=asset_id,
        base_avatar_id=base_avatar_id,
        case_id=case_value.get("case_id"),
        case_kind=case_value.get("kind"),
    )
    if payload.get("coordinate_frame") != COORDINATE_FRAME:
        raise InstanceContractError("instance coordinate frame must be FRONT -Y, UP +Z, floor Z=0")
    root_value = payload.get("canonical_output_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise InstanceContractError("canonical output root must be absolute")
    root = common.require_real_directory(
        Path(root_value), "canonical output root", InstanceContractError
    )
    if root.name != asset_id:
        raise InstanceContractError("canonical output root name must equal asset_id")
    if expected_output_root is not None and root != common.absolute(expected_output_root):
        raise InstanceContractError("canonical output root does not match the contract path")
    lineage_value = payload.get("source_lineage")
    if not isinstance(lineage_value, Mapping) or not lineage_value:
        raise InstanceContractError("source_lineage must be a non-empty mapping")
    _validate_lineage_profile(lineage_value, case_kind=case["kind"])
    if list(lineage_value) != sorted(lineage_value):
        raise InstanceContractError("source_lineage keys are not in canonical order")
    lineage = {
        lineage_id: _validate_source_record(lineage_id, descriptor)
        for lineage_id, descriptor in lineage_value.items()
    }
    _validate_lineage_semantics(
        lineage, asset_id=asset_id, base_avatar_id=base_avatar_id, case=case
    )
    dag = validate_branch_dag(payload.get("allowed_branch_dag"))
    if dag != payload.get("allowed_branch_dag"):
        raise InstanceContractError("allowed branch DAG is not canonical")
    if dag != DEFAULT_BRANCH_DAG:
        raise InstanceContractError("allowed branch DAG must equal the pinned branch DAG")
    if payload.get("state_classification") != "research_candidate":
        raise InstanceContractError("Route-2 instance must remain research_candidate")
    if payload.get("publication") != {"no_replace": True, "artifact_mode": "0444"}:
        raise InstanceContractError("instance contract publication policy changed")
    if payload.get("user_acceptance") != "pending_user_review":
        raise InstanceContractError("instance contract may only await user review")
    return {
        "schema": SCHEMA,
        "asset_id": asset_id,
        "base_avatar_id": base_avatar_id,
        "case": case,
        "coordinate_frame": dict(COORDINATE_FRAME),
        "canonical_output_root": str(root),
        "source_lineage": lineage,
        "allowed_branch_dag": dag,
        "state_classification": "research_candidate",
        "publication": {"no_replace": True, "artifact_mode": "0444"},
        "user_acceptance": "pending_user_review",
    }


def publish_instance_contract(payload: Mapping[str, Any]) -> Path:
    validated = common.stable_mapping_snapshot(
        lambda: _validate_payload(payload),
        InstanceContractError,
        "instance contract source lineage",
    )
    destination = Path(validated["canonical_output_root"]) / FILENAME

    def validate_prelink() -> None:
        current = common.stable_mapping_snapshot(
            lambda: _validate_payload(payload),
            InstanceContractError,
            "instance contract pre-publication lineage",
        )
        if current != validated:
            raise InstanceContractError(
                "instance contract source lineage changed during pre-publication validation"
            )

    return common.write_json_immutable_noreplace(
        destination,
        validated,
        InstanceContractError,
        "Route-2 human instance contract",
        prelink_validator=validate_prelink,
    )


def _validate_instance_contract_once(path: Path) -> dict[str, Any]:
    supplied = common.absolute(path)
    if supplied.name != FILENAME:
        raise InstanceContractError(f"instance contract must be named {FILENAME}")
    root = common.require_real_directory(
        supplied.parent, "canonical output root", InstanceContractError
    )
    payload, _ = common.load_json_mapping_record(
        supplied,
        root=root,
        description="instance contract",
        error_type=InstanceContractError,
        require_mode=0o444,
    )
    return _validate_payload(payload, expected_output_root=root)


def validate_instance_contract(path: Path) -> dict[str, Any]:
    return common.stable_mapping_snapshot(
        lambda: _validate_instance_contract_once(path),
        InstanceContractError,
        "instance contract",
    )


def branch_descriptor(contract: Mapping[str, Any], branch_id: str) -> dict[str, str]:
    branch_id = common.require_id(branch_id, "branch_id", InstanceContractError)
    dag = validate_branch_dag(contract.get("allowed_branch_dag"))
    for node in dag["nodes"]:
        if node["branch_id"] == branch_id:
            return dict(node)
    raise InstanceContractError(f"branch_id is not allowed by the instance contract: {branch_id}")


def resolve_branch_root(contract: Mapping[str, Any], branch_id: str) -> Path:
    descriptor = branch_descriptor(contract, branch_id)
    root = common.require_real_directory(
        Path(str(contract.get("canonical_output_root"))),
        "canonical output root",
        InstanceContractError,
    )
    relative = descriptor["relative_root"]
    candidate = root if relative == "." else common.absolute(root / relative)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise InstanceContractError("branch root escapes canonical output root") from error
    current = candidate
    existing_ancestor: Path | None = None
    while current != root.parent:
        if os.path.lexists(current):
            existing_ancestor = current
            break
        current = current.parent
    if existing_ancestor is None:
        raise InstanceContractError("canonical output root disappeared")
    if existing_ancestor == candidate:
        common.require_real_directory(candidate, "branch root", InstanceContractError)
    elif existing_ancestor.is_symlink() or not existing_ancestor.is_dir():
        raise InstanceContractError("branch root has a symlinked or non-directory ancestor")
    return candidate


def contract_file_record(contract_path: Path) -> dict[str, Any]:
    contract = validate_instance_contract(contract_path)
    root = Path(contract["canonical_output_root"])
    return common.file_record(
        contract_path,
        root=root,
        description="instance contract",
        error_type=InstanceContractError,
        require_mode=0o444,
    )


__all__ = [
    "ATTRIBUTE_CASE_BASE",
    "ATTRIBUTE_LINEAGE_ROLES",
    "BASE_AVATAR_IDS",
    "BASE_LINEAGE_ROLES",
    "CASE_KINDS",
    "COORDINATE_FRAME",
    "DEFAULT_BRANCH_DAG",
    "FILENAME",
    "InstanceContractError",
    "DINO_REVISION",
    "PIXAL_ATTRIBUTE_ATTEMPT_FIELDS",
    "PIXAL_ATTRIBUTE_ATTEMPT_SCHEMA",
    "PIXAL_ATTRIBUTE_EXECUTION_LOG_FIELDS",
    "PIXAL_ATTRIBUTE_EXECUTION_LOG_SCHEMA",
    "PIXAL_ATTRIBUTE_FAILURE_BUNDLE_FIELDS",
    "PIXAL_ATTRIBUTE_FAILURE_BUNDLE_SCHEMA",
    "PIXAL_ATTRIBUTE_JOB_FIELDS",
    "PIXAL_ATTRIBUTE_START_FIELDS",
    "PIXAL_ATTRIBUTE_START_SCHEMA",
    "PIXAL_ENVIRONMENT_FIELDS",
    "PIXAL_EXECUTOR_FIELDS",
    "PIXAL_PYTHON_EXECUTABLE",
    "PIXAL3D_REVISION",
    "SCHEMA",
    "branch_descriptor",
    "build_instance_contract",
    "contract_file_record",
    "publish_instance_contract",
    "model_snapshot_evidence",
    "pixal_execution_guard_evidence",
    "resolve_branch_root",
    "validate_branch_dag",
    "validate_instance_contract",
    "validate_pixal_attribute_failure_bundle",
]
