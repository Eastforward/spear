#!/usr/bin/env python3
"""Produce and validate a fail-closed direct-native quadruped review.

This route is deliberately distinct from the generated-asset route.  It
accepts only an immutable bounded-identity realization whose native Idle and
Walking clips survived the authored edit.  It does not rank candidates,
invoke Pixel3D, invoke TokenRig, retarget motion, or repair weights.

The output is a research review candidate.  Human approval, UE asset-bound
readback, formal registration, and Native merge remain separate gates.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import stat
import struct
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence, Union


SPEAR_ROOT = Path(__file__).resolve().parents[1]
TOOLS = SPEAR_ROOT / "tools"
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import bounded_quadruped_identity_contract as identity  # noqa: E402
from tools.run_target_native_generated_quadruped_review import (  # noqa: E402
    DEFORMATION_THRESHOLDS,
    REVIEW_MEDIA_FPS,
    REVIEW_MEDIA_HEIGHT,
    REVIEW_MEDIA_SPECS,
    REVIEW_MEDIA_WIDTH,
    require_deformation_audit,
    require_review_media_set,
)


SCHEMA = "avengine_direct_native_authored_quadruped_animation_review_v2"
REALIZATION_SCHEMA = "avengine_bounded_quadruped_identity_realization_v1"
STATUS = "research_candidate_pending_human_review"
REVIEW_FRAMES = 24
DEFORMATION_SAMPLES = 24
SHA256_PATTERN = identity.SHA256_PATTERN
MEDIA_LABELS = tuple(item[0] for item in REVIEW_MEDIA_SPECS)
REVIEW_STAGES = (
    "deformation_audit",
    *(
        stage
        for label in MEDIA_LABELS
        for stage in (f"render_{label}", f"encode_{label}")
    ),
)
RECEIPT_NAME = "review_receipt.json"
RECEIPT_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "created",
        "status",
        "state",
        "formal",
        "route",
        "lineage",
        "toolchain",
        "automatic_admission_gates",
        "outputs",
        "artifact_closure",
        "authority_boundary",
        "next_gate",
        "review_sha256",
    }
)
REALIZATION_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "created_at",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "lineage",
        "toolchain",
        "toolchain_publication_snapshots",
        "publication_security_boundary",
        "operations",
        "invariants",
        "output_readback",
        "blender_reimport",
        "outputs",
        "authority_boundary",
        "next_gate",
    }
)
REALIZATION_INVARIANT_FIELDS = frozenset(
    {
        "before",
        "after_in_memory",
        "topology_unchanged_in_memory",
        "weights_unchanged_in_memory",
        "skeleton_unchanged_in_memory",
        "actions_unchanged_in_memory",
        "canonical_skinned_surface_sha256_before_export",
        "canonical_skinned_surface_sha256_after_reimport",
        "canonical_roundtrip_precision_decimals",
        "canonical_geometry_and_weights_survived_reimport",
        "canonical_triangle_surface_and_weight_clusters_survived_reimport",
        "skeleton_hierarchy_sha256_before_export",
        "skeleton_hierarchy_sha256_after_reimport",
        "raw_source_skin_signature",
        "raw_output_skin_signature",
        "maximum_bind_matrix_semantic_delta",
        "maximum_bind_matrix_semantic_delta_gate",
        "bind_world_surface_sha256_before_export",
        "bind_world_surface_sha256_after_reimport",
        "skeleton_survived_reimport",
        "animation_semantics_before_export",
        "animation_semantics_after_reimport",
        "animation_semantic_comparison",
        "raw_source_animation_time_signature",
        "raw_output_animation_time_signature",
        "raw_animation_timelines_and_interpolation_unchanged",
        "idle_and_walking_actions_survived_reimport",
        "canonical_surface_precision_diagnostic",
    }
)
REALIZATION_ROOT_FILES = frozenset(
    {"bounded_identity.glb", "identity_coat.png", "realization_manifest.json"}
)
REALIZATION_EVIDENCE_FILES = frozenset(
    {
        "source_snapshot.glb",
        "identity_plan.json",
        "preflight_receipt.json",
        "machine_execution_authorization.json",
        "bounded_quadruped_identity_contract.py",
        "blender_realize_bounded_quadruped_identity.py",
    }
)
SOURCE_SNAPSHOT_NAMES = {
    "realization_manifest.json": "source_realization_manifest.json",
    "bounded_identity.glb": "source_bounded_identity.glb",
    "identity_coat.png": "source_identity_coat.png",
    "evidence/source_snapshot.glb": "source_source_snapshot.glb",
    "evidence/identity_plan.json": "source_identity_plan.json",
    "evidence/preflight_receipt.json": "source_preflight_receipt.json",
    "evidence/machine_execution_authorization.json": (
        "source_machine_execution_authorization.json"
    ),
    "evidence/bounded_quadruped_identity_contract.py": "source_contract.py",
    "evidence/blender_realize_bounded_quadruped_identity.py": "source_executor.py",
}
REVIEW_TOOL_FILES = {
    "review_runner": Path(__file__).resolve(),
    "producer_contract_runtime": TOOLS / "bounded_quadruped_identity_contract.py",
    "deformation_auditor": TOOLS / "blender_audit_skinned_deformation.py",
    "animation_renderer": TOOLS / "blender_render_glb_animation.py",
    "renderer_semantics": TOOLS / "generated_quadruped_semantics.py",
    "renderer_morphotype_guide": TOOLS / "quadruped_morphotype_guide.py",
    "media_encoder": TOOLS / "encode_quadruped_review_media.py",
    "strict_review_contract": TOOLS / "run_target_native_generated_quadruped_review.py",
    "controlled_source_asset_schema": TOOLS / "controlled_source_asset_schema.py",
    "generated_animal_forward_contract": (
        TOOLS / "generated_animal_forward_contract.py"
    ),
    "generated_animal_support_plane_contract": (
        TOOLS / "generated_animal_support_plane_contract.py"
    ),
    "generated_animal_tokenrig_closure": (
        TOOLS / "generated_animal_tokenrig_closure.py"
    ),
}
ROUTE = {
    "pixal3d_used": False,
    "tokenrig_used": False,
    "native_animated_source_used": True,
    "candidate_ranking_used": False,
    "retry_selection_used": False,
}
AUTHORIZED_RESEARCH_OUTPUT_SCOPES = frozenset(
    {"new_research_candidate_sealed_at_publication_only"}
)
EXPECTED_REALIZATION_PUBLICATION_SECURITY_BOUNDARY = {
    "protocol": "posix_held_dirfd_renameat2_noreplace_postrename_reverify_v2",
    "rename_semantics": {
        "rename_is_publication_boundary_not_readiness_claim": True,
        "post_rename_exact_held_fd_reverification_required": True,
        "failed_post_rename_verification_retains_final": True,
    },
    "parent_directory_policy": {
        "owner_uid_must_equal_effective_uid": True,
        "other_writable_forbidden": True,
        "group_writable_requires_uid_matched_effective_private_group": True,
        "uid_matched_group_privacy_is_environment_assumption": True,
        "path_inode_uid_gid_mode_reauthenticated": True,
    },
    "posix_same_uid_limit": {
        "absolute_immutability_claimed": False,
        "malicious_same_uid_writer_can_mutate_after_verification": True,
        "mode_0444_0555_is_only_accidental_write_hardening": True,
    },
    "consumer_requirement": {
        "external_expected_manifest_raw_sha256_required": True,
        "rehash_entire_declared_file_closure_before_use": True,
    },
}


class DirectNativeReviewError(RuntimeError):
    """The direct-native candidate or its review evidence failed closed."""


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DirectNativeReviewError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise DirectNativeReviewError(f"non-finite JSON value is forbidden: {value}")


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except DirectNativeReviewError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise DirectNativeReviewError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise DirectNativeReviewError(f"{label} must be a JSON object")
    require_finite(value, label)
    return value


def require_finite(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DirectNativeReviewError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            require_finite(item, label)
        return
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item, label)
        return
    raise DirectNativeReviewError(f"{label} contains a non-JSON value")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def require_exact_keys(
    value: Any, expected: Union[set[str], frozenset[str]], label: str
):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise DirectNativeReviewError(
            f"{label} fields changed: expected={sorted(expected)} observed={observed}"
        )
    return value


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise DirectNativeReviewError(f"{label} is not a lowercase SHA-256")
    return value


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _guard(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def open_absolute_directory(path: Path) -> tuple[int, tuple[int, int]]:
    """Open every absolute component with O_NOFOLLOW.

    The returned descriptor, rather than the final pathname, is the authority
    used for all subsequent reads or publication.
    """

    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute():
        raise DirectNativeReviewError("directory path must be absolute")
    descriptor = os.open("/", _directory_flags())
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise DirectNativeReviewError("unsafe directory component")
            next_descriptor = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        current = os.fstat(descriptor)
        if not stat.S_ISDIR(current.st_mode):
            raise DirectNativeReviewError(f"not a directory: {absolute}")
        return descriptor, _identity(current)
    except Exception:
        os.close(descriptor)
        raise


def stable_read_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    required_mode: Optional[int] = None,
) -> tuple[bytes, tuple[int, int], os.stat_result]:
    if not name or "/" in name or name in {".", ".."}:
        raise DirectNativeReviewError(f"unsafe {label} filename")
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    descriptor = os.open(name, _file_flags(), dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _guard(opened) != _guard(before)
            or (
                required_mode is not None
                and stat.S_IMODE(opened.st_mode) != required_mode
            )
        ):
            raise DirectNativeReviewError(f"{label} is not an immutable regular file")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    payload = b"".join(chunks)
    if (
        _guard(opened) != _guard(after)
        or _guard(opened) != _guard(path_after)
        or len(payload) != opened.st_size
    ):
        raise DirectNativeReviewError(f"{label} changed while read")
    return payload, _identity(opened), opened


def stable_read_absolute(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_fd, _ = open_absolute_directory(absolute.parent)
    try:
        payload, _, current = stable_read_at(parent_fd, absolute.name, label=label)
    finally:
        os.close(parent_fd)
    return payload, {
        "path": str(absolute),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "mode": oct(stat.S_IMODE(current.st_mode)),
        "nlink": current.st_nlink,
    }


def relative_record(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}


@dataclass
class AuthenticatedRealization:
    root: Path
    root_fd: int
    evidence_fd: int
    manifest: dict[str, Any]
    raw_manifest: bytes
    artifacts: dict[str, bytes]
    plan: identity.IdentityPlan

    def close(self) -> None:
        first_error = None
        for attribute in ("evidence_fd", "root_fd"):
            descriptor = getattr(self, attribute)
            if descriptor < 0:
                continue
            setattr(self, attribute, -1)
            try:
                os.close(descriptor)
            except OSError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _require_relative_binding(
    record: Any, relative_path: str, payload: bytes, label: str
) -> None:
    require_exact_keys(record, {"path", "sha256", "size_bytes"}, label)
    if record != relative_record(relative_path, payload):
        raise DirectNativeReviewError(f"{label} does not bind exact bytes")


def _require_external_and_publication_snapshot_binding(
    record: Any,
    *,
    snapshot_relative: str,
    snapshot_payload: bytes,
    label: str,
) -> None:
    require_exact_keys(
        record,
        {"path", "sha256", "size_bytes", "publication_snapshot"},
        label,
    )
    _require_relative_binding(
        record["publication_snapshot"],
        snapshot_relative,
        snapshot_payload,
        f"{label} publication snapshot",
    )
    if (
        record["sha256"] != sha256_bytes(snapshot_payload)
        or record["size_bytes"] != len(snapshot_payload)
        or not isinstance(record["path"], str)
        or not Path(record["path"]).is_absolute()
    ):
        raise DirectNativeReviewError(
            f"{label} publication snapshot does not close its source descriptor"
        )


def _validate_realization_publication_security_boundary(value: Any) -> None:
    if value != EXPECTED_REALIZATION_PUBLICATION_SECURITY_BOUNDARY:
        raise DirectNativeReviewError(
            "realization publication security boundary changed or weakened"
        )


def _validate_raw_animation_timing(invariants: Mapping[str, Any]) -> dict[str, Any]:
    source = invariants.get("raw_source_animation_time_signature")
    output = invariants.get("raw_output_animation_time_signature")
    expected_fields = {
        "actions",
        "fractional_24fps_time_count",
        "preserves_non_integer_24fps_times",
        "sha256",
    }
    require_exact_keys(source, expected_fields, "source raw animation timing")
    require_exact_keys(output, expected_fields, "output raw animation timing")
    if source != output:
        raise DirectNativeReviewError(
            "source/output raw animation timing signatures are not exactly equal"
        )
    actions = source["actions"]
    require_exact_keys(actions, {"Idle", "Walking"}, "raw animation actions")
    all_times: set[float] = set()
    for action_name, timelines in actions.items():
        if not isinstance(timelines, list) or not timelines:
            raise DirectNativeReviewError(f"{action_name} has no raw timelines")
        previous_key = None
        for timeline in timelines:
            require_exact_keys(
                timeline,
                {"interpolation", "sample_count", "times_seconds"},
                f"{action_name} raw timeline",
            )
            interpolation = timeline["interpolation"]
            times = timeline["times_seconds"]
            if (
                interpolation not in {"LINEAR", "STEP", "CUBICSPLINE"}
                or not isinstance(times, list)
                or not times
                or timeline["sample_count"] != len(times)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in times
                )
                or any(left >= right for left, right in zip(times, times[1:]))
            ):
                raise DirectNativeReviewError(
                    f"{action_name} raw timeline is malformed"
                )
            key = (interpolation, tuple(times))
            if previous_key is not None and key <= previous_key:
                raise DirectNativeReviewError(
                    f"{action_name} raw timelines are not canonical"
                )
            previous_key = key
            all_times.update(float(value) for value in times)
    fractional = {
        value for value in all_times if abs(value * 24.0 - round(value * 24.0)) > 1.0e-5
    }
    if (
        not fractional
        or source["fractional_24fps_time_count"] != len(fractional)
        or source["preserves_non_integer_24fps_times"] is not True
        or source["sha256"] != canonical_sha256(actions)
    ):
        raise DirectNativeReviewError(
            "raw timing signature summary contradicts exact timelines"
        )
    return dict(source)


def _nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise DirectNativeReviewError(f"{label} must be a finite nonnegative number")
    return float(value)


def _validate_dense_animation_gate(
    invariants: Mapping[str, Any], plan: identity.IdentityPlan
) -> dict[str, Any]:
    comparison = invariants.get("animation_semantic_comparison")
    require_exact_keys(
        comparison,
        {
            "maximum_animation_duration_delta_frames",
            "maximum_animation_duration_delta_frames_gate",
            "maximum_root_trajectory_delta",
            "maximum_root_trajectory_delta_gate",
            "maximum_skinned_world_vertex_delta",
            "maximum_skinned_world_vertex_delta_gate",
            "per_action",
            "joint_world_matrix_hashes_recorded_but_not_used_as_blender_rig_gate",
            "overall",
        },
        "dense animation comparison",
    )
    phases = [index / 80 for index in range(81)]
    per_action = comparison["per_action"]
    require_exact_keys(per_action, {"Idle", "Walking"}, "dense animation actions")
    observed_skin = []
    observed_root = []
    for action_name in ("Idle", "Walking"):
        samples = per_action[action_name]
        if not isinstance(samples, list) or len(samples) != 81:
            raise DirectNativeReviewError(
                f"{action_name} must contain exactly 81 dense phases"
            )
        for expected_phase, sample in zip(phases, samples):
            require_exact_keys(
                sample,
                {
                    "phase",
                    "source_evaluated_vertex_count",
                    "output_evaluated_vertex_count",
                    "maximum_skinned_world_vertex_delta",
                    "within_gate",
                    "maximum_root_trajectory_delta",
                    "root_trajectory_within_gate",
                },
                f"{action_name} dense phase",
            )
            if (
                not math.isclose(
                    float(sample["phase"]),
                    expected_phase,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not isinstance(sample["source_evaluated_vertex_count"], int)
                or isinstance(sample["source_evaluated_vertex_count"], bool)
                or sample["source_evaluated_vertex_count"] <= 0
                or not isinstance(sample["output_evaluated_vertex_count"], int)
                or isinstance(sample["output_evaluated_vertex_count"], bool)
                or sample["output_evaluated_vertex_count"] <= 0
            ):
                raise DirectNativeReviewError(
                    f"{action_name} dense phase identity changed"
                )
            skin = _nonnegative_number(
                sample["maximum_skinned_world_vertex_delta"],
                f"{action_name} skinned delta",
            )
            root = _nonnegative_number(
                sample["maximum_root_trajectory_delta"],
                f"{action_name} root delta",
            )
            if sample["within_gate"] is not (
                skin <= plan.maximum_sampled_skinned_world_delta
            ) or sample["root_trajectory_within_gate"] is not (
                root <= plan.maximum_sampled_skinned_world_delta
            ):
                raise DirectNativeReviewError(
                    f"{action_name} dense phase boolean contradicts measurement"
                )
            observed_skin.append(skin)
            observed_root.append(root)
    maximum_skin = max(observed_skin)
    maximum_root = max(observed_root)
    duration = _nonnegative_number(
        comparison["maximum_animation_duration_delta_frames"],
        "maximum animation duration delta",
    )
    expected = {
        "maximum_skinned_world_vertex_delta": maximum_skin,
        "maximum_skinned_world_vertex_delta_gate": (
            plan.maximum_sampled_skinned_world_delta
        ),
        "maximum_root_trajectory_delta": maximum_root,
        "maximum_root_trajectory_delta_gate": (
            plan.maximum_sampled_skinned_world_delta
        ),
        "maximum_animation_duration_delta_frames": duration,
        "maximum_animation_duration_delta_frames_gate": (
            plan.maximum_animation_duration_delta_frames
        ),
    }
    for field, value in expected.items():
        if not math.isclose(
            float(comparison[field]), value, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise DirectNativeReviewError(
                f"dense animation aggregate contradicts phases: {field}"
            )
    if (
        maximum_skin > plan.maximum_sampled_skinned_world_delta
        or maximum_root > plan.maximum_sampled_skinned_world_delta
        or duration > plan.maximum_animation_duration_delta_frames
        or comparison["overall"] != "passed"
        or comparison[
            "joint_world_matrix_hashes_recorded_but_not_used_as_blender_rig_gate"
        ]
        is not True
    ):
        raise DirectNativeReviewError("dense animation comparison did not pass")
    for field in (
        "animation_semantics_before_export",
        "animation_semantics_after_reimport",
    ):
        evidence = invariants.get(field)
        require_exact_keys(
            evidence,
            {"actions", "precision_decimals", "sample_phases", "sha256"},
            field,
        )
        if (
            evidence["precision_decimals"]
            != plan.roundtrip_canonical_precision_decimals
            or evidence["sample_phases"] != phases
            or set(evidence["actions"]) != {"Idle", "Walking"}
        ):
            raise DirectNativeReviewError(f"{field} dense sampling changed")
        for action_name in ("Idle", "Walking"):
            action = evidence["actions"][action_name]
            require_exact_keys(
                action,
                {"action_name", "duration_frames", "root_bone", "samples"},
                f"{field} {action_name}",
            )
            samples = action["samples"]
            if (
                not isinstance(action["action_name"], str)
                or action_name.lower() not in action["action_name"].lower()
                or action["root_bone"] != plan.motion_root_bone
                or not isinstance(samples, list)
                or len(samples) != 81
                or _nonnegative_number(action["duration_frames"], f"{field} duration")
                <= 0
            ):
                raise DirectNativeReviewError(f"{field} action evidence changed")
            for expected_phase, sample in zip(phases, samples):
                require_exact_keys(
                    sample,
                    {
                        "phase",
                        "joint_world_matrices_sha256",
                        "skinned_world_surface_sha256",
                    },
                    f"{field} {action_name} sample",
                )
                if not math.isclose(
                    float(sample["phase"]),
                    expected_phase,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise DirectNativeReviewError(
                        f"{field} {action_name} phase grid changed"
                    )
                require_sha256(
                    sample["joint_world_matrices_sha256"],
                    f"{field} joint matrices",
                )
                require_sha256(
                    sample["skinned_world_surface_sha256"],
                    f"{field} skinned surface",
                )
    return {
        "phases_per_action": 81,
        "phase_interval": "index/80",
        "maximum_skinned_world_vertex_delta": maximum_skin,
        "maximum_skinned_world_vertex_delta_gate": (
            plan.maximum_sampled_skinned_world_delta
        ),
        "maximum_root_trajectory_delta": maximum_root,
        "maximum_root_trajectory_delta_gate": (
            plan.maximum_sampled_skinned_world_delta
        ),
        "maximum_animation_duration_delta_frames": duration,
        "maximum_animation_duration_delta_frames_gate": (
            plan.maximum_animation_duration_delta_frames
        ),
        "status": "passed",
    }


def _glb_json_and_binary(payload: bytes) -> tuple[dict[str, Any], bytes]:
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise DirectNativeReviewError("output GLB header is invalid")
    version, total_length = struct.unpack_from("<II", payload, 4)
    if version != 2 or total_length != len(payload):
        raise DirectNativeReviewError("output GLB length/version is invalid")
    offset = 12
    document = None
    binary = b""
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise DirectNativeReviewError("output GLB chunk header is truncated")
        length, kind = struct.unpack_from("<I4s", payload, offset)
        offset += 8
        chunk = payload[offset : offset + length]
        offset += length
        if len(chunk) != length:
            raise DirectNativeReviewError("output GLB chunk is truncated")
        if kind == b"JSON":
            if document is not None:
                raise DirectNativeReviewError("output GLB has duplicate JSON chunks")
            document = strict_json_bytes(chunk.rstrip(b" \x00"), "output GLB JSON")
        elif kind == b"BIN\x00":
            if binary:
                raise DirectNativeReviewError("output GLB has duplicate BIN chunks")
            binary = chunk
        else:
            raise DirectNativeReviewError("output GLB has an unsupported chunk")
    if document is None or not binary:
        raise DirectNativeReviewError("output GLB JSON/BIN closure is incomplete")
    return document, binary


def _validate_glb_and_coat(
    glb_payload: bytes,
    coat_payload: bytes,
    manifest: Mapping[str, Any],
    plan: identity.IdentityPlan,
) -> dict[str, Any]:
    document, binary = _glb_json_and_binary(glb_payload)
    meshes = document.get("meshes")
    skins = document.get("skins")
    animations = document.get("animations")
    materials = document.get("materials")
    images = document.get("images")
    textures = document.get("textures")
    if not all(
        isinstance(item, list)
        for item in (meshes, skins, animations, materials, images, textures)
    ):
        raise DirectNativeReviewError("output GLB descriptor arrays are malformed")
    action_names = [item.get("name") for item in animations]
    for expected in ("Idle", "Walking"):
        if (
            sum(
                isinstance(name, str) and expected.lower() in name.lower()
                for name in action_names
            )
            != 1
        ):
            raise DirectNativeReviewError(
                f"output GLB action identity changed: {expected}"
            )
    if not (
        len(meshes) == 1
        and len(skins) == 1
        and len(animations) == 2
        and len(materials) == 1
        and len(images) == 1
        and len(textures) == 1
    ):
        raise DirectNativeReviewError("output GLB inventory changed")
    primitives = [
        primitive for mesh in meshes for primitive in mesh.get("primitives", [])
    ]
    required_attributes = {
        "POSITION",
        "NORMAL",
        "TEXCOORD_0",
        "JOINTS_0",
        "WEIGHTS_0",
    }
    if not primitives or any(
        not isinstance(primitive, dict)
        or not required_attributes.issubset(primitive.get("attributes", {}))
        or not isinstance(primitive.get("material"), int)
        for primitive in primitives
    ):
        raise DirectNativeReviewError("output GLB primitive lost UV/skin/material")
    image = images[0]
    view_index = image.get("bufferView")
    views = document.get("bufferViews", [])
    if (
        image.get("mimeType") != "image/png"
        or "uri" in image
        or not isinstance(view_index, int)
        or not 0 <= view_index < len(views)
    ):
        raise DirectNativeReviewError("output GLB embedded PNG descriptor changed")
    view = views[view_index]
    start = int(view.get("byteOffset", 0))
    end = start + int(view.get("byteLength", 0))
    if view.get("buffer", 0) != 0 or start < 0 or end > len(binary) or end <= start:
        raise DirectNativeReviewError("output GLB embedded PNG range is invalid")
    expected_codes = {
        identity.srgb_rgba8(plan.base_color),
        identity.srgb_rgba8(plan.white_color),
    }
    external_codes = identity.png_rgba8_code_values(coat_payload)
    embedded_codes = identity.png_rgba8_code_values(binary[start:end])
    if external_codes != expected_codes or embedded_codes != expected_codes:
        raise DirectNativeReviewError("coat sRGB readback was not applied exactly once")
    readback = manifest["output_readback"]
    require_exact_keys(
        readback,
        {
            "animation_channel_counts",
            "animation_names",
            "base_color_texture_color_space",
            "embedded_png",
            "embedded_png_srgb_readback",
            "image_count",
            "material_count",
            "mesh_count",
            "primitive_count",
            "removal_object_absent",
            "required_attributes_present",
            "skin_count",
            "texture_count",
        },
        "GLB output descriptor",
    )
    if (
        readback["mesh_count"] != 1
        or readback["skin_count"] != 1
        or readback["material_count"] != 1
        or readback["image_count"] != 1
        or readback["texture_count"] != 1
        or readback["primitive_count"] != len(primitives)
        or readback["embedded_png"] is not True
        or readback["required_attributes_present"] is not True
        or readback["removal_object_absent"] is not True
        or readback["base_color_texture_color_space"] != "sRGB"
        or readback["animation_names"] != action_names
    ):
        raise DirectNativeReviewError("GLB output descriptor contradicts raw GLB")
    color_readback = readback["embedded_png_srgb_readback"]
    require_exact_keys(
        color_readback,
        {
            "base_rgba8",
            "white_rgba8",
            "unique_rgba8_values",
            "encoding",
            "sRGB_transfer_applied_exactly_once",
        },
        "embedded sRGB readback",
    )
    if (
        color_readback["encoding"] != "PNG_RGBA8_sRGB_code_values"
        or color_readback["sRGB_transfer_applied_exactly_once"] is not True
        or {tuple(item) for item in color_readback["unique_rgba8_values"]}
        != expected_codes
        or tuple(color_readback["base_rgba8"]) != identity.srgb_rgba8(plan.base_color)
        or tuple(color_readback["white_rgba8"]) != identity.srgb_rgba8(plan.white_color)
    ):
        raise DirectNativeReviewError("embedded sRGB descriptor contradicts pixels")
    return {
        "mesh_count": 1,
        "skin_count": 1,
        "animation_names": action_names,
        "material_count": 1,
        "texture_count": 1,
        "image_count": 1,
        "primitive_count": len(primitives),
        "required_attributes": sorted(required_attributes),
        "sRGB_transfer_applied_exactly_once": True,
        "base_rgba8": list(identity.srgb_rgba8(plan.base_color)),
        "white_rgba8": list(identity.srgb_rgba8(plan.white_color)),
        "status": "passed",
    }


def _require_srgb_descriptor(
    value: Any, plan: identity.IdentityPlan, label: str
) -> None:
    require_exact_keys(
        value,
        {
            "base_rgba8",
            "white_rgba8",
            "unique_rgba8_values",
            "encoding",
            "sRGB_transfer_applied_exactly_once",
        },
        label,
    )
    expected_base = identity.srgb_rgba8(plan.base_color)
    expected_white = identity.srgb_rgba8(plan.white_color)
    if (
        value["encoding"] != "PNG_RGBA8_sRGB_code_values"
        or value["sRGB_transfer_applied_exactly_once"] is not True
        or tuple(value["base_rgba8"]) != expected_base
        or tuple(value["white_rgba8"]) != expected_white
        or {tuple(item) for item in value["unique_rgba8_values"]}
        != {expected_base, expected_white}
    ):
        raise DirectNativeReviewError(f"{label} contradicts exact sRGB codes")


def _validate_operations(
    manifest: Mapping[str, Any], plan: identity.IdentityPlan
) -> None:
    operations = require_exact_keys(
        manifest["operations"],
        {
            "ear_morph",
            "removed_object",
            "removed_pose_bone_custom_shape_assignments",
            "deterministic_uv",
            "deterministic_coat",
        },
        "realization operations",
    )
    ear = require_exact_keys(
        operations["ear_morph"],
        {
            "maximum_displacement",
            "maximum_displacement_bbox_diagonal_ratio",
            "moved_vertex_count",
            "output_masks_mirrored",
            "selected_position_sha256_after",
            "selected_position_sha256_before",
            "unselected_position_sha256_after",
            "unselected_position_sha256_before",
            "unselected_positions_unchanged",
        },
        "ear morph operation",
    )
    if (
        ear["moved_vertex_count"]
        != len(plan.positive.indices) + len(plan.negative.indices)
        or ear["output_masks_mirrored"] is not True
        or ear["unselected_positions_unchanged"] is not True
        or ear["unselected_position_sha256_before"]
        != ear["unselected_position_sha256_after"]
        or _nonnegative_number(
            ear["maximum_displacement_bbox_diagonal_ratio"],
            "ear displacement ratio",
        )
        > plan.maximum_displacement_diagonal_ratio
    ):
        raise DirectNativeReviewError("ear morph operation contradicted its plan")
    for field in (
        "selected_position_sha256_after",
        "selected_position_sha256_before",
        "unselected_position_sha256_after",
        "unselected_position_sha256_before",
    ):
        require_sha256(ear[field], f"ear morph {field}")
    if (
        operations["removed_object"] != plan.removal_object
        or not isinstance(
            operations["removed_pose_bone_custom_shape_assignments"], list
        )
        or len(operations["removed_pose_bone_custom_shape_assignments"])
        != plan.removal_custom_shape_assignments
        or len(set(operations["removed_pose_bone_custom_shape_assignments"]))
        != plan.removal_custom_shape_assignments
    ):
        raise DirectNativeReviewError("removed custom-shape object closure changed")
    uv = require_exact_keys(
        operations["deterministic_uv"],
        {"method", "angle_limit_radians", "island_margin"},
        "deterministic UV operation",
    )
    if (
        uv["method"] != "blender_smart_project_v1"
        or uv["angle_limit_radians"] != plan.uv_angle_limit
        or uv["island_margin"] != plan.uv_island_margin
    ):
        raise DirectNativeReviewError("deterministic UV operation changed")
    coat = require_exact_keys(
        operations["deterministic_coat"],
        {
            "external_png_srgb_readback",
            "face_role_counts",
            "resolution",
            "white_texel_count_after_bleed",
            "white_texel_fraction_after_bleed",
        },
        "deterministic coat operation",
    )
    _require_srgb_descriptor(
        coat["external_png_srgb_readback"], plan, "external coat sRGB readback"
    )
    require_exact_keys(
        coat["face_role_counts"], {"base", "white"}, "coat face-role counts"
    )
    if (
        coat["resolution"] != [plan.uv_resolution, plan.uv_resolution]
        or any(
            not isinstance(coat["face_role_counts"][role], int)
            or isinstance(coat["face_role_counts"][role], bool)
            or coat["face_role_counts"][role] <= 0
            for role in ("base", "white")
        )
        or not isinstance(coat["white_texel_count_after_bleed"], int)
        or isinstance(coat["white_texel_count_after_bleed"], bool)
        or coat["white_texel_count_after_bleed"] <= 0
        or not 0.0
        < _nonnegative_number(
            coat["white_texel_fraction_after_bleed"], "white texel fraction"
        )
        < 1.0
    ):
        raise DirectNativeReviewError("deterministic coat operation is invalid")
    reimport = require_exact_keys(
        manifest["blender_reimport"],
        {"disable_bone_shape", "synthetic_bone_shape_object_absent"},
        "Blender reimport",
    )
    if (
        reimport["disable_bone_shape"] is not True
        or reimport["synthetic_bone_shape_object_absent"] is not True
    ):
        raise DirectNativeReviewError("Blender reimport retained synthetic bone shape")


def _validate_structural_invariants(
    invariants: Mapping[str, Any], plan: identity.IdentityPlan
) -> dict[str, Any]:
    require_exact_keys(
        invariants, REALIZATION_INVARIANT_FIELDS, "realization invariants"
    )
    required_true = (
        "topology_unchanged_in_memory",
        "weights_unchanged_in_memory",
        "skeleton_unchanged_in_memory",
        "actions_unchanged_in_memory",
        "canonical_geometry_and_weights_survived_reimport",
        "canonical_triangle_surface_and_weight_clusters_survived_reimport",
        "skeleton_survived_reimport",
        "raw_animation_timelines_and_interpolation_unchanged",
        "idle_and_walking_actions_survived_reimport",
    )
    if any(invariants.get(field) is not True for field in required_true):
        raise DirectNativeReviewError("structural/skin/skeleton invariant did not pass")
    before = invariants.get("before")
    after = invariants.get("after_in_memory")
    require_exact_keys(
        before,
        {
            "inventory",
            "topology_sha256",
            "weights_sha256",
            "skeleton_sha256",
            "actions_sha256",
            "canonical_skinned_surface_sha256",
        },
        "pre-edit invariant descriptor",
    )
    require_exact_keys(
        after,
        {
            "inventory",
            "topology_sha256",
            "weights_sha256",
            "skeleton_sha256",
            "actions_sha256",
            "canonical_skinned_surface_sha256",
            "uv_layer_names",
            "uv_loop_count",
        },
        "post-edit invariant descriptor",
    )
    for field in (
        "topology_sha256",
        "weights_sha256",
        "skeleton_sha256",
        "actions_sha256",
    ):
        require_sha256(before[field], f"before {field}")
        if before[field] != after[field]:
            raise DirectNativeReviewError(f"in-memory invariant changed: {field}")
    diagnostic = invariants["canonical_surface_precision_diagnostic"]
    require_exact_keys(
        diagnostic,
        {"3", "4", "5", "6", "7"},
        "canonical surface precision diagnostic",
    )
    equality_pairs = (
        (
            "canonical_skinned_surface_sha256_before_export",
            "canonical_skinned_surface_sha256_after_reimport",
        ),
        (
            "skeleton_hierarchy_sha256_before_export",
            "skeleton_hierarchy_sha256_after_reimport",
        ),
        (
            "bind_world_surface_sha256_before_export",
            "bind_world_surface_sha256_after_reimport",
        ),
    )
    for left, right in equality_pairs:
        require_sha256(invariants.get(left), left)
        if invariants[left] != invariants.get(right):
            raise DirectNativeReviewError(f"roundtrip invariant changed: {left}")
    source_skin = invariants.get("raw_source_skin_signature")
    output_skin = invariants.get("raw_output_skin_signature")
    skin_fields = {
        "joint_count",
        "joint_hierarchy_and_order_sha256",
        "raw_inverse_bind_matrices_sha256",
        "precision_decimals",
        "bind_semantic_residual_sha256",
        "semantic_sha256",
        "maximum_bind_residual_magnitude",
    }
    require_exact_keys(source_skin, skin_fields, "source skin signature")
    require_exact_keys(output_skin, skin_fields, "output skin signature")
    if (
        source_skin["joint_count"] <= 0
        or source_skin["joint_count"] != output_skin["joint_count"]
        or source_skin["joint_hierarchy_and_order_sha256"]
        != output_skin["joint_hierarchy_and_order_sha256"]
        or source_skin["precision_decimals"]
        != plan.roundtrip_canonical_precision_decimals
        or output_skin["precision_decimals"]
        != plan.roundtrip_canonical_precision_decimals
    ):
        raise DirectNativeReviewError("raw skin hierarchy/order changed")
    maximum_bind_delta = _nonnegative_number(
        invariants.get("maximum_bind_matrix_semantic_delta"),
        "maximum bind semantic delta",
    )
    if (
        invariants.get("maximum_bind_matrix_semantic_delta_gate")
        != plan.maximum_bind_matrix_semantic_delta
        or maximum_bind_delta > plan.maximum_bind_matrix_semantic_delta
    ):
        raise DirectNativeReviewError("bind semantic delta exceeded its gate")
    return {
        "topology_unchanged_in_memory": True,
        "weights_unchanged_in_memory": True,
        "skeleton_unchanged_in_memory": True,
        "actions_unchanged_in_memory": True,
        "roundtrip_geometry_and_weight_clusters_passed": True,
        "joint_count": source_skin["joint_count"],
        "maximum_bind_matrix_semantic_delta": maximum_bind_delta,
        "maximum_bind_matrix_semantic_delta_gate": (
            plan.maximum_bind_matrix_semantic_delta
        ),
        "status": "passed",
    }


def authenticate_realization(
    manifest_path: Path, expected_raw_sha256: str
) -> AuthenticatedRealization:
    require_sha256(expected_raw_sha256, "expected realization manifest SHA-256")
    absolute_manifest = Path(os.path.abspath(os.fspath(manifest_path)))
    if absolute_manifest.name != "realization_manifest.json":
        raise DirectNativeReviewError(
            "realization manifest must be named realization_manifest.json"
        )
    root_fd, _ = open_absolute_directory(absolute_manifest.parent)
    evidence_fd = os.open("evidence", _directory_flags(), dir_fd=root_fd)
    try:
        root_entries = set(os.listdir(root_fd))
        evidence_entries = set(os.listdir(evidence_fd))
        if root_entries != set(REALIZATION_ROOT_FILES) | {"evidence"}:
            raise DirectNativeReviewError("realization root closure changed")
        if evidence_entries != set(REALIZATION_EVIDENCE_FILES):
            raise DirectNativeReviewError("realization evidence closure changed")
        artifacts: dict[str, bytes] = {}
        for name in sorted(REALIZATION_ROOT_FILES):
            payload, _, _ = stable_read_at(
                root_fd,
                name,
                label=f"realization {name}",
                required_mode=0o444,
            )
            artifacts[name] = payload
        for name in sorted(REALIZATION_EVIDENCE_FILES):
            payload, _, _ = stable_read_at(
                evidence_fd,
                name,
                label=f"realization evidence/{name}",
                required_mode=0o444,
            )
            artifacts[f"evidence/{name}"] = payload
        root_stat = os.fstat(root_fd)
        evidence_stat = os.fstat(evidence_fd)
        if (
            stat.S_IMODE(root_stat.st_mode) != 0o555
            or stat.S_IMODE(evidence_stat.st_mode) != 0o555
        ):
            raise DirectNativeReviewError("realization directories are not sealed 0555")
        raw_manifest = artifacts["realization_manifest.json"]
        if sha256_bytes(raw_manifest) != expected_raw_sha256:
            raise DirectNativeReviewError(
                "external realization manifest SHA-256 authority mismatch"
            )
        manifest = strict_json_bytes(raw_manifest, "realization manifest")
        require_exact_keys(
            manifest, REALIZATION_TOP_LEVEL_FIELDS, "realization manifest"
        )
        if (
            manifest["schema"] != REALIZATION_SCHEMA
            or manifest["status"] != "research_candidate_pending_final_animation_review"
            or manifest["state_classification"] != "research_candidate"
            or manifest["formal_dataset_registration_authorized"] is not False
            or manifest["next_gate"] != "final_six_view_animation_and_identity_review"
        ):
            raise DirectNativeReviewError("realization status/authority is invalid")
        _validate_realization_publication_security_boundary(
            manifest["publication_security_boundary"]
        )
        outputs = require_exact_keys(
            manifest["outputs"], {"glb", "coat_texture"}, "realization outputs"
        )
        _require_relative_binding(
            outputs["glb"],
            "bounded_identity.glb",
            artifacts["bounded_identity.glb"],
            "realization GLB",
        )
        _require_relative_binding(
            outputs["coat_texture"],
            "identity_coat.png",
            artifacts["identity_coat.png"],
            "realization coat",
        )
        lineage = require_exact_keys(
            manifest["lineage"],
            {"source", "plan", "preflight", "machine_execution_authorization"},
            "realization lineage",
        )
        lineage_paths = {
            "source": "evidence/source_snapshot.glb",
            "plan": "evidence/identity_plan.json",
            "preflight": "evidence/preflight_receipt.json",
            "machine_execution_authorization": (
                "evidence/machine_execution_authorization.json"
            ),
        }
        for label, relative in lineage_paths.items():
            _require_external_and_publication_snapshot_binding(
                lineage[label],
                snapshot_relative=relative,
                snapshot_payload=artifacts[relative],
                label=f"realization {label}",
            )
        toolchain = require_exact_keys(
            manifest["toolchain"], {"blender", "contract", "executor"}, "toolchain"
        )
        snapshots = require_exact_keys(
            manifest["toolchain_publication_snapshots"],
            {"contract", "executor"},
            "tool publication snapshots",
        )
        for label, relative in (
            ("contract", "evidence/bounded_quadruped_identity_contract.py"),
            ("executor", "evidence/blender_realize_bounded_quadruped_identity.py"),
        ):
            require_exact_keys(
                toolchain[label], {"path", "sha256", "size_bytes"}, f"{label} tool"
            )
            _require_relative_binding(
                snapshots[label],
                relative,
                artifacts[relative],
                f"{label} publication tool snapshot",
            )
            if (
                toolchain[label]["sha256"] != sha256_bytes(artifacts[relative])
                or toolchain[label]["size_bytes"] != len(artifacts[relative])
                or not isinstance(toolchain[label]["path"], str)
                or not Path(toolchain[label]["path"]).is_absolute()
            ):
                raise DirectNativeReviewError(
                    f"{label} publication tool snapshot does not close its descriptor"
                )
        blender = require_exact_keys(
            toolchain["blender"],
            {"binary", "build_hash", "version"},
            "realization Blender identity",
        )
        if not all(
            isinstance(blender[field], str) and blender[field] for field in blender
        ):
            raise DirectNativeReviewError("realization Blender identity is incomplete")
        plan_value = strict_json_bytes(
            artifacts["evidence/identity_plan.json"], "identity plan snapshot"
        )
        try:
            plan = identity.load_plan(plan_value)
        except identity.IdentityContractError as error:
            raise DirectNativeReviewError(f"identity plan rejected: {error}") from error
        preflight = strict_json_bytes(
            artifacts["evidence/preflight_receipt.json"],
            "preflight receipt snapshot",
        )
        if (
            preflight.get("schema") != identity.PREFLIGHT_SCHEMA
            or preflight.get("status") != "pending_machine_execution_authorization"
            or preflight.get("state_classification") != "research_candidate"
            or preflight.get("formal_dataset_registration_authorized") is not False
            or preflight.get("execution_authorized") is not False
            or preflight.get("automatic_checks", {}).get("overall")
            != "passed_preflight_only"
            or preflight.get("plan", {}).get("sha256")
            != sha256_bytes(artifacts["evidence/identity_plan.json"])
            or preflight.get("source", {}).get("sha256")
            != sha256_bytes(artifacts["evidence/source_snapshot.glb"])
        ):
            raise DirectNativeReviewError("preflight receipt closure did not pass")
        authorization = strict_json_bytes(
            artifacts["evidence/machine_execution_authorization.json"],
            "machine execution authorization snapshot",
        )
        require_exact_keys(
            authorization,
            {
                "schema",
                "decision",
                "preflight_receipt_sha256",
                "plan_sha256",
                "source_sha256",
                "checks",
                "notes",
                "authorization_basis",
                "authorized_actor_type",
                "authorized_output_scope",
                "state_classification",
                "formal_dataset_registration_authorized",
                "user_approval_inferred",
            },
            "machine execution authorization",
        )
        if (
            authorization["schema"] != identity.AUTHORIZATION_SCHEMA
            or authorization["decision"]
            != "authorized_for_new_bounded_research_candidate_output"
            or authorization["preflight_receipt_sha256"]
            != sha256_bytes(artifacts["evidence/preflight_receipt.json"])
            or authorization["plan_sha256"]
            != sha256_bytes(artifacts["evidence/identity_plan.json"])
            or authorization["source_sha256"]
            != sha256_bytes(artifacts["evidence/source_snapshot.glb"])
            or set(authorization["checks"]) != identity.AUTHORIZATION_CHECKS
            or any(value is not True for value in authorization["checks"].values())
            or authorization["authorization_basis"]
            != "scoped_task_continue_asset_identity_repair"
            or authorization["authorized_actor_type"] != "codex_task_agent"
            or authorization["authorized_output_scope"]
            not in AUTHORIZED_RESEARCH_OUTPUT_SCOPES
            or authorization["state_classification"] != "research_candidate"
            or authorization["formal_dataset_registration_authorized"] is not False
            or authorization["user_approval_inferred"] is not False
        ):
            raise DirectNativeReviewError(
                "machine execution authorization binding is invalid"
            )
        invariants = manifest["invariants"]
        if not isinstance(invariants, Mapping):
            raise DirectNativeReviewError("realization invariants are missing")
        _validate_raw_animation_timing(invariants)
        _validate_dense_animation_gate(invariants, plan)
        _validate_structural_invariants(invariants, plan)
        _validate_glb_and_coat(
            artifacts["bounded_identity.glb"],
            artifacts["identity_coat.png"],
            manifest,
            plan,
        )
        _validate_operations(manifest, plan)
        authority = require_exact_keys(
            manifest["authority_boundary"],
            {
                "owner_animation_decision",
                "breed_identity_decision",
                "ue_asset_bound_readback",
                "dynamic_audio",
                "native_merge_authorized",
            },
            "realization authority boundary",
        )
        if (
            authority["owner_animation_decision"] != "pending"
            or authority["breed_identity_decision"] != "pending"
            or authority["ue_asset_bound_readback"] != "pending"
            or authority["dynamic_audio"] != "pending"
            or authority["native_merge_authorized"] is not False
        ):
            raise DirectNativeReviewError("realization claims excess authority")
        return AuthenticatedRealization(
            root=absolute_manifest.parent,
            root_fd=root_fd,
            evidence_fd=evidence_fd,
            manifest=manifest,
            raw_manifest=raw_manifest,
            artifacts=artifacts,
            plan=plan,
        )
    except Exception:
        os.close(evidence_fd)
        os.close(root_fd)
        raise


@dataclass
class DirectoryHandle:
    relative: str
    descriptor: int
    identity: tuple[int, int]


class SecureReviewPublication:
    """A held-dirfd publication with quarantine-only failure handling."""

    def __init__(self, output_root: Path):
        self.output_root = Path(os.path.abspath(os.fspath(output_root)))
        if not self.output_root.name or self.output_root.name in {".", ".."}:
            raise DirectNativeReviewError("unsafe output-root basename")
        self.parent_path = self.output_root.parent
        self.parent_fd, self.parent_identity = open_absolute_directory(self.parent_path)
        self.staging_name = (
            f".{self.output_root.name}.{secrets.token_hex(12)}.quarantine"
        )
        self.published = False
        self.root_fd = -1
        self.root_identity = (-1, -1)
        self.directories: dict[str, DirectoryHandle] = {}
        try:
            try:
                os.stat(
                    self.output_root.name,
                    dir_fd=self.parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise DirectNativeReviewError(
                    f"refusing to replace output root: {self.output_root}"
                )
            os.mkdir(self.staging_name, 0o700, dir_fd=self.parent_fd)
            self.root_fd = os.open(
                self.staging_name, _directory_flags(), dir_fd=self.parent_fd
            )
            self.root_identity = _identity(os.fstat(self.root_fd))
            entry = os.stat(
                self.staging_name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            if _identity(entry) != self.root_identity:
                raise DirectNativeReviewError("private staging identity changed")
            self.directories[""] = DirectoryHandle("", self.root_fd, self.root_identity)
            self._mkdir("evidence")
            self._mkdir("media")
            for label in MEDIA_LABELS:
                self._mkdir(f"media/{label}_frames")
            self._require_proc_path()
        except Exception:
            self.close()
            raise

    @property
    def proc_root(self) -> Path:
        return Path(f"/proc/self/fd/{self.root_fd}")

    def proc_path(self, relative: str) -> Path:
        return self.proc_root / relative

    def _require_proc_path(self) -> None:
        current = os.stat(self.proc_root)
        if (
            not stat.S_ISDIR(current.st_mode)
            or _identity(current) != self.root_identity
            or _identity(os.fstat(self.root_fd)) != self.root_identity
        ):
            raise DirectNativeReviewError("/proc/self/fd staging identity changed")

    def _mkdir(self, relative: str) -> None:
        parent_relative, name = (
            relative.rsplit("/", 1) if "/" in relative else ("", relative)
        )
        parent = self.directories[parent_relative]
        os.mkdir(name, 0o700, dir_fd=parent.descriptor)
        descriptor = os.open(name, _directory_flags(), dir_fd=parent.descriptor)
        current = os.fstat(descriptor)
        self.directories[relative] = DirectoryHandle(
            relative, descriptor, _identity(current)
        )

    def write_exclusive(self, relative: str, payload: bytes) -> None:
        parent_relative, name = (
            relative.rsplit("/", 1) if "/" in relative else ("", relative)
        )
        parent = self.directories[parent_relative]
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent.descriptor,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise DirectNativeReviewError("publication write made no progress")
                offset += written
            os.fsync(descriptor)
            current = os.fstat(descriptor)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or current.st_size != len(payload)
            ):
                raise DirectNativeReviewError("publication write is unsafe")
        finally:
            os.close(descriptor)

    def rewrite_owned(self, relative: str, payload: bytes) -> None:
        parent_relative, name = (
            relative.rsplit("/", 1) if "/" in relative else ("", relative)
        )
        parent = self.directories[parent_relative]
        before = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent.descriptor,
        )
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _identity(opened) != _identity(before)
            ):
                raise DirectNativeReviewError("owned artifact identity changed")
            os.ftruncate(descriptor, 0)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise DirectNativeReviewError("owned rewrite made no progress")
                offset += written
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            if _identity(after) != _identity(opened) or after.st_size != len(payload):
                raise DirectNativeReviewError("owned artifact raced during rewrite")
        finally:
            os.close(descriptor)

    def read_file(self, relative: str, required_mode: Optional[int] = None) -> bytes:
        parent_relative, name = (
            relative.rsplit("/", 1) if "/" in relative else ("", relative)
        )
        parent = self.directories[parent_relative]
        payload, _, _ = stable_read_at(
            parent.descriptor,
            name,
            label=f"review artifact {relative}",
            required_mode=required_mode,
        )
        return payload

    def _scan(self, *, required_mode: Optional[int]) -> dict[str, dict[str, Any]]:
        expected_directories = set(self.directories)
        observed_directories = {""}
        records: dict[str, dict[str, Any]] = {}
        pending = [""]
        while pending:
            relative = pending.pop()
            handle = self.directories.get(relative)
            if handle is None:
                raise DirectNativeReviewError(
                    f"publication contains an unowned directory: {relative}"
                )
            current = os.fstat(handle.descriptor)
            if (
                not stat.S_ISDIR(current.st_mode)
                or _identity(current) != handle.identity
                or (
                    required_mode is not None and stat.S_IMODE(current.st_mode) != 0o555
                )
            ):
                raise DirectNativeReviewError(
                    f"publication directory identity/mode changed: {relative}"
                )
            for name in os.listdir(handle.descriptor):
                if not name or "/" in name or name in {".", ".."}:
                    raise DirectNativeReviewError("unsafe publication entry")
                child = f"{relative}/{name}" if relative else name
                entry = os.stat(name, dir_fd=handle.descriptor, follow_symlinks=False)
                if stat.S_ISDIR(entry.st_mode):
                    expected = self.directories.get(child)
                    if expected is None or _identity(entry) != expected.identity:
                        raise DirectNativeReviewError(
                            f"unknown/replaced directory quarantined: {child}"
                        )
                    observed_directories.add(child)
                    pending.append(child)
                elif stat.S_ISREG(entry.st_mode):
                    payload, _, _ = stable_read_at(
                        handle.descriptor,
                        name,
                        label=f"publication {child}",
                        required_mode=required_mode,
                    )
                    records[child] = relative_record(child, payload)
                else:
                    raise DirectNativeReviewError(
                        f"unsafe publication entry quarantined: {child}"
                    )
        if observed_directories != expected_directories:
            raise DirectNativeReviewError("publication directory inventory changed")
        return records

    def seal(self, expected_files: set[str]) -> dict[str, dict[str, Any]]:
        self._require_proc_path()
        records = self._scan(required_mode=None)
        if set(records) != expected_files:
            raise DirectNativeReviewError(
                "publication exact file inventory changed before seal"
            )
        for relative in sorted(records):
            parent_relative, name = (
                relative.rsplit("/", 1) if "/" in relative else ("", relative)
            )
            descriptor = os.open(
                name, _file_flags(), dir_fd=self.directories[parent_relative].descriptor
            )
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for relative in sorted(
            self.directories, key=lambda value: value.count("/"), reverse=True
        ):
            os.fchmod(self.directories[relative].descriptor, 0o555)
            os.fsync(self.directories[relative].descriptor)
        sealed = self._scan(required_mode=0o444)
        if sealed != records:
            raise DirectNativeReviewError("publication bytes changed while sealing")
        return sealed

    def verify(self, expected: Mapping[str, Mapping[str, Any]]) -> None:
        records = self._scan(required_mode=0o444)
        if records != dict(expected):
            raise DirectNativeReviewError("sealed publication bytes/inventory changed")

    def _require_parent_path_still_held(self) -> None:
        check_fd, check_identity = open_absolute_directory(self.parent_path)
        try:
            if (
                check_identity != self.parent_identity
                or _identity(os.fstat(self.parent_fd)) != self.parent_identity
            ):
                raise DirectNativeReviewError(
                    "publication parent path no longer names held directory"
                )
        finally:
            os.close(check_fd)

    def _require_published_entry(self) -> None:
        self._require_parent_path_still_held()
        final_entry = os.stat(
            self.output_root.name,
            dir_fd=self.parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(final_entry.st_mode)
            or _identity(final_entry) != self.root_identity
        ):
            raise DirectNativeReviewError(
                "published name does not bind the held staging inode"
            )

    def publish(self, expected: Mapping[str, Mapping[str, Any]]) -> None:
        self._require_parent_path_still_held()
        self.verify(expected)
        try:
            renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2")
        except AttributeError as error:
            raise DirectNativeReviewError(
                "atomic no-replace publication is unavailable"
            ) from error
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            self.parent_fd,
            os.fsencode(self.staging_name),
            self.parent_fd,
            os.fsencode(self.output_root.name),
            1,
        )
        if result != 0:
            number = ctypes.get_errno()
            if number == errno.EEXIST:
                raise DirectNativeReviewError(
                    f"refusing concurrent output root: {self.output_root}"
                )
            if number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
                raise DirectNativeReviewError(
                    "atomic no-replace publication is unsupported"
                )
            raise OSError(number, os.strerror(number), self.output_root.name)
        self.published = True
        # Rename is the publication boundary, not a readiness claim.  Bind the
        # final name both before and after the expensive held-fd closure scan so
        # a same-UID replacement in that window cannot be reported as success.
        self._require_published_entry()
        self.verify(expected)
        self._require_published_entry()
        os.fsync(self.parent_fd)
        self._require_published_entry()

    def quarantine(self) -> None:
        """Intentionally leave an unpublished private root in place.

        There is no per-file unlink or rmdir cleanup.  An attacker cannot turn
        a guard-to-unlink race into deletion of an external same-UID victim.
        """

        return

    def close(self) -> None:
        first_error = None
        for relative, handle in sorted(
            self.directories.items(),
            key=lambda item: item[0].count("/"),
            reverse=True,
        ):
            if handle.descriptor >= 0:
                descriptor = handle.descriptor
                handle.descriptor = -1
                try:
                    os.close(descriptor)
                except OSError as error:
                    if first_error is None:
                        first_error = error
        if self.root_fd >= 0 and "" not in self.directories:
            descriptor = self.root_fd
            self.root_fd = -1
            try:
                os.close(descriptor)
            except OSError as error:
                if first_error is None:
                    first_error = error
        self.root_fd = -1
        if self.parent_fd >= 0:
            descriptor = self.parent_fd
            self.parent_fd = -1
            try:
                os.close(descriptor)
            except OSError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


@dataclass
class ExecutableIdentity:
    name: str
    path: Path
    payload_guard: tuple[int, ...]
    descriptor: int
    record: dict[str, Any]
    version_argv: list[str]
    version_stdout: str

    def verify(self) -> None:
        current_fd = os.fstat(self.descriptor)
        current_path = os.stat(self.path, follow_symlinks=False)
        if (
            _guard(current_fd) != self.payload_guard
            or _guard(current_path) != self.payload_guard
        ):
            raise DirectNativeReviewError(
                f"{self.name} executable identity/bytes changed"
            )

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            os.close(descriptor)


def authenticate_executable(
    value: str, *, name: str, version_arguments: list[str]
) -> ExecutableIdentity:
    candidate = Path(value)
    if candidate.parent == Path("."):
        resolved = shutil.which(value)
        if resolved is None:
            raise DirectNativeReviewError(f"{name} executable was not found")
        candidate = Path(resolved)
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    parent_fd, _ = open_absolute_directory(candidate.parent)
    try:
        payload, identity_pair, current = stable_read_at(
            parent_fd, candidate.name, label=f"{name} executable"
        )
        descriptor = os.open(candidate.name, _file_flags(), dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    if not os.access(candidate, os.X_OK):
        os.close(descriptor)
        raise DirectNativeReviewError(f"{name} executable is not executable")
    result = subprocess.run(
        [str(candidate), *version_arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    after = os.fstat(descriptor)
    if _identity(after) != identity_pair or _guard(after) != _guard(current):
        os.close(descriptor)
        raise DirectNativeReviewError(f"{name} changed during version readback")
    return ExecutableIdentity(
        name=name,
        path=candidate,
        payload_guard=_guard(current),
        descriptor=descriptor,
        record={
            "path": str(candidate),
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
            "device": current.st_dev,
            "inode": current.st_ino,
        },
        version_argv=[str(candidate), *version_arguments],
        version_stdout=(result.stdout + result.stderr).strip(),
    )


def _read_exact_descriptor(descriptor: int, expected_size: int) -> bytes:
    chunks = []
    offset = 0
    while offset < expected_size:
        block = os.pread(descriptor, min(1024 * 1024, expected_size - offset), offset)
        if not block:
            break
        chunks.append(block)
        offset += len(block)
    payload = b"".join(chunks)
    if len(payload) != expected_size:
        raise DirectNativeReviewError("held review tool read was truncated")
    return payload


@dataclass
class AuthenticatedToolFile:
    name: str
    path: Path
    payload: bytes
    payload_guard: tuple[int, ...]
    descriptor: int
    record: dict[str, Any]

    def verify(self) -> None:
        if self.descriptor < 0:
            raise DirectNativeReviewError(f"{self.name} review tool is closed")
        current_fd = os.fstat(self.descriptor)
        current_path = os.stat(self.path, follow_symlinks=False)
        if (
            _guard(current_fd) != self.payload_guard
            or _guard(current_path) != self.payload_guard
            or sha256_bytes(_read_exact_descriptor(self.descriptor, current_fd.st_size))
            != self.record["sha256"]
        ):
            raise DirectNativeReviewError(
                f"{self.name} review tool identity/bytes changed"
            )

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            os.close(descriptor)


def authenticate_tool_file(path: Path, *, name: str) -> AuthenticatedToolFile:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_fd, _ = open_absolute_directory(absolute.parent)
    descriptor = -1
    try:
        payload, identity_pair, current = stable_read_at(
            parent_fd,
            absolute.name,
            label=f"{name} review tool",
        )
        descriptor = os.open(absolute.name, _file_flags(), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if _identity(opened) != identity_pair or _guard(opened) != _guard(current):
            raise DirectNativeReviewError(f"{name} review tool changed while held")
        if sha256_bytes(
            _read_exact_descriptor(descriptor, opened.st_size)
        ) != sha256_bytes(payload):
            raise DirectNativeReviewError(
                f"{name} held review tool bytes differ from snapshot"
            )
        result = AuthenticatedToolFile(
            name=name,
            path=absolute,
            payload=payload,
            payload_guard=_guard(opened),
            descriptor=descriptor,
            record={
                "path": str(absolute),
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            },
        )
        descriptor = -1
        result.verify()
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _close_producer_resources(
    publication: Optional[SecureReviewPublication],
    executables: Mapping[str, ExecutableIdentity],
    review_tools: Mapping[str, AuthenticatedToolFile],
    realization: AuthenticatedRealization,
) -> None:
    resources = [
        *([publication] if publication is not None else []),
        *executables.values(),
        *review_tools.values(),
        realization,
    ]
    first_error = None
    for resource in resources:
        try:
            resource.close()
        except OSError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def blender_command(blender: Path, script: Path, arguments: list[str]) -> list[str]:
    return [
        str(blender),
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        str(script),
        "--",
        *arguments,
    ]


def media_paths(root: Path, label: str) -> dict[str, Path]:
    return {
        "frame_dir": root / "media" / f"{label}_frames",
        "render_manifest": root / "media" / f"{label}_render_manifest.json",
        "video": root / "media" / f"{label}.mp4",
        "encode_manifest": root / "media" / f"{label}_encode_manifest.json",
    }


def build_commands(
    *,
    root: Path,
    blender: Path,
    python: Path,
    input_glb: Path,
    tool_paths: Optional[Mapping[str, Path]] = None,
) -> list[tuple[str, list[str]]]:
    scripts = (
        {
            "deformation_auditor": TOOLS / "blender_audit_skinned_deformation.py",
            "animation_renderer": TOOLS / "blender_render_glb_animation.py",
            "media_encoder": TOOLS / "encode_quadruped_review_media.py",
        }
        if tool_paths is None
        else dict(tool_paths)
    )
    if set(scripts) != {
        "deformation_auditor",
        "animation_renderer",
        "media_encoder",
    } or any(not Path(path).is_absolute() for path in scripts.values()):
        raise DirectNativeReviewError("review execution tool paths are incomplete")
    commands = [
        (
            "deformation_audit",
            blender_command(
                blender,
                scripts["deformation_auditor"],
                [
                    "--input",
                    str(input_glb),
                    "--output",
                    str(root / "evidence/deformation_audit.json"),
                    "--action",
                    "Walking",
                    "--action",
                    "Idle",
                    "--samples",
                    str(DEFORMATION_SAMPLES),
                ],
            ),
        )
    ]
    for label, action, view, yaw in REVIEW_MEDIA_SPECS:
        paths = media_paths(root, label)
        commands.append(
            (
                f"render_{label}",
                blender_command(
                    blender,
                    scripts["animation_renderer"],
                    [
                        "--input",
                        str(input_glb),
                        "--action",
                        action,
                        "--output-dir",
                        str(paths["frame_dir"]),
                        "--manifest",
                        str(paths["render_manifest"]),
                        "--n-frames",
                        str(REVIEW_FRAMES),
                        "--width",
                        str(REVIEW_MEDIA_WIDTH),
                        "--height",
                        str(REVIEW_MEDIA_HEIGHT),
                        "--fps",
                        str(REVIEW_MEDIA_FPS),
                        "--samples",
                        "16",
                        "--view",
                        view,
                        "--asset-yaw-deg",
                        str(yaw),
                        "--trajectory-distance-ratio",
                        "0",
                        "--ground-plane",
                        "--engine",
                        "BLENDER_EEVEE_NEXT",
                    ],
                ),
            )
        )
        commands.append(
            (
                f"encode_{label}",
                [
                    str(python),
                    str(scripts["media_encoder"]),
                    "--input-glb",
                    str(input_glb),
                    "--render-manifest",
                    str(paths["render_manifest"]),
                    "--frame-dir",
                    str(paths["frame_dir"]),
                    "--label",
                    label,
                    "--action",
                    action,
                    "--view",
                    view,
                    "--asset-yaw-deg",
                    str(yaw),
                    "--n-frames",
                    str(REVIEW_FRAMES),
                    "--width",
                    str(REVIEW_MEDIA_WIDTH),
                    "--height",
                    str(REVIEW_MEDIA_HEIGHT),
                    "--fps",
                    str(REVIEW_MEDIA_FPS),
                    "--output",
                    str(paths["video"]),
                    "--manifest",
                    str(paths["encode_manifest"]),
                ],
            )
        )
    return commands


def _replace_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        if value == old:
            return new
        if value.startswith(old + "/"):
            return new + value[len(old) :]
        return value
    if isinstance(value, list):
        return [_replace_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_prefix(item, old, new) for key, item in value.items()}
    return value


def _published_file_record(root: Path, relative: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(root / relative),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def normalize_generated_lineage(
    publication: SecureReviewPublication,
) -> None:
    old_prefix = str(publication.proc_root)
    new_prefix = str(publication.output_root)
    deformation_relative = "evidence/deformation_audit.json"
    deformation = strict_json_bytes(
        publication.read_file(deformation_relative), "deformation audit"
    )
    publication.rewrite_owned(
        deformation_relative,
        pretty_json_bytes(_replace_prefix(deformation, old_prefix, new_prefix)),
    )
    for label in MEDIA_LABELS:
        render_relative = f"media/{label}_render_manifest.json"
        render = strict_json_bytes(
            publication.read_file(render_relative), f"{label} render manifest"
        )
        render = _replace_prefix(render, old_prefix, new_prefix)
        render_bytes = pretty_json_bytes(render)
        publication.rewrite_owned(render_relative, render_bytes)
        encode_relative = f"media/{label}_encode_manifest.json"
        encode = strict_json_bytes(
            publication.read_file(encode_relative), f"{label} encode manifest"
        )
        encode = _replace_prefix(encode, old_prefix, new_prefix)
        encode["render_manifest"] = _published_file_record(
            publication.output_root, render_relative, render_bytes
        )
        artifacts = [dict(item["artifact"]) for item in render["frames"]]
        encode["frame_set"] = {
            "count": len(artifacts),
            "sha256": canonical_sha256(artifacts),
            "artifacts": artifacts,
        }
        encode_bytes = pretty_json_bytes(encode)
        publication.rewrite_owned(encode_relative, encode_bytes)


def _expected_review_files() -> set[str]:
    result = {
        RECEIPT_NAME,
        "evidence/deformation_audit.json",
    }
    result.update(f"evidence/{name}" for name in SOURCE_SNAPSHOT_NAMES.values())
    result.update(f"evidence/tool_{name}.py" for name in REVIEW_TOOL_FILES)
    for label in MEDIA_LABELS:
        result.update(
            {
                f"media/{label}.mp4",
                f"media/{label}_render_manifest.json",
                f"media/{label}_encode_manifest.json",
            }
        )
        result.update(
            f"media/{label}_frames/frame_{index:04d}.png"
            for index in range(REVIEW_FRAMES)
        )
    return result


def _command_records(
    commands: list[tuple[str, list[str]]],
    *,
    old_prefix: str,
    new_prefix: str,
) -> list[dict[str, Any]]:
    return [
        {
            "stage": stage,
            "observed_argv": argv,
            "published_equivalent_argv": [
                _replace_prefix(argument, old_prefix, new_prefix) for argument in argv
            ],
        }
        for stage, argv in commands
    ]


def _media_subprocess_contract_for_root(
    review_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    ffmpeg_records = []
    ffprobe_records = []
    for label in MEDIA_LABELS:
        paths = media_paths(review_root, label)
        ffmpeg_records.append(
            {
                "label": label,
                "resolved_executable_identity": "toolchain.executables.ffmpeg",
                "observed_argv": [
                    "ffmpeg",
                    "-nostdin",
                    "-n",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(REVIEW_MEDIA_FPS),
                    "-start_number",
                    "0",
                    "-i",
                    str(paths["frame_dir"] / "frame_%04d.png"),
                    "-frames:v",
                    str(REVIEW_FRAMES),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(paths["video"]),
                ],
            }
        )
        ffprobe_records.append(
            {
                "label": label,
                "resolved_executable_identity": "toolchain.executables.ffprobe",
                "observed_argv": [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    (
                        "stream=codec_name,width,height,nb_frames,"
                        "r_frame_rate,avg_frame_rate:format=duration"
                    ),
                    "-of",
                    "json",
                    str(paths["video"]),
                ],
            }
        )
    return {"ffmpeg": ffmpeg_records, "ffprobe": ffprobe_records}


def _media_subprocess_contract(
    publication: SecureReviewPublication,
) -> dict[str, list[dict[str, Any]]]:
    return _media_subprocess_contract_for_root(publication.proc_root)


def _normalize_historical_proc_argv(
    argv: Any,
    *,
    published_root: Path,
    expected_proc_prefix: Optional[str],
    label: str,
) -> tuple[list[str], Optional[str]]:
    if not isinstance(argv, list) or not argv:
        raise DirectNativeReviewError(f"{label} argv is incomplete")
    normalized = []
    proc_prefix = expected_proc_prefix
    marker = "/proc/self/fd/"
    for argument in argv:
        if not isinstance(argument, str):
            raise DirectNativeReviewError(f"{label} argv contains a non-string")
        if argument.startswith(marker):
            remainder = argument[len(marker) :]
            descriptor, separator, suffix = remainder.partition("/")
            if not descriptor.isdigit():
                raise DirectNativeReviewError(
                    f"{label} argv contains an invalid held-fd path"
                )
            current_prefix = f"{marker}{descriptor}"
            if proc_prefix is None:
                proc_prefix = current_prefix
            elif current_prefix != proc_prefix:
                raise DirectNativeReviewError(
                    f"{label} argv changed held publication roots"
                )
            argument = str(published_root)
            if separator:
                argument = f"{argument}/{suffix}"
        normalized.append(argument)
    return normalized, proc_prefix


def _records_to_closure(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    artifacts = [
        {
            "path": relative,
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for relative, record in sorted(records.items())
        if relative != RECEIPT_NAME
    ]
    return {
        "artifacts": artifacts,
        "inventory_sha256": canonical_sha256(artifacts),
        "receipt_excluded_from_artifact_inventory_to_avoid_self_reference": True,
        "exact_directory_inventory_required": True,
    }


def _output_descriptor(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    media = {}
    for label in MEDIA_LABELS:
        media[label] = {
            "video": dict(records[f"media/{label}.mp4"]),
            "render_manifest": dict(records[f"media/{label}_render_manifest.json"]),
            "encode_manifest": dict(records[f"media/{label}_encode_manifest.json"]),
            "frames": [
                dict(records[f"media/{label}_frames/frame_{index:04d}.png"])
                for index in range(REVIEW_FRAMES)
            ],
        }
    return {
        "deformation_audit": dict(records["evidence/deformation_audit.json"]),
        "media": media,
    }


def build_receipt(
    *,
    realization: AuthenticatedRealization,
    realization_manifest_path: Path,
    expected_realization_sha256: str,
    records_without_receipt: Mapping[str, Mapping[str, Any]],
    tool_snapshots: Mapping[str, Mapping[str, Any]],
    executables: Mapping[str, ExecutableIdentity],
    commands: list[tuple[str, list[str]]],
    publication: SecureReviewPublication,
    deformation: Mapping[str, Any],
) -> dict[str, Any]:
    invariants = realization.manifest["invariants"]
    dense = _validate_dense_animation_gate(invariants, realization.plan)
    timing = _validate_raw_animation_timing(invariants)
    structure = _validate_structural_invariants(invariants, realization.plan)
    glb_readback = _validate_glb_and_coat(
        realization.artifacts["bounded_identity.glb"],
        realization.artifacts["identity_coat.png"],
        realization.manifest,
        realization.plan,
    )
    receipt = {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "state": {
            "classification": "research_candidate",
            "owner_animation_decision": "pending",
            "owner_identity_decision": "pending",
            "ue_asset_bound_readback": "pending",
            "native_merge_authorized": False,
        },
        "formal": {
            "formal_dataset_registration_authorized": False,
            "ue_import_authorized": False,
            "native_merge_authorized": False,
        },
        "route": dict(ROUTE),
        "lineage": {
            "realization_manifest": {
                "external_path": str(
                    Path(os.path.abspath(os.fspath(realization_manifest_path)))
                ),
                "external_raw_sha256_authority": expected_realization_sha256,
                "immutable_snapshot": dict(
                    records_without_receipt["evidence/source_realization_manifest.json"]
                ),
            },
            "reviewed_glb": dict(
                records_without_receipt["evidence/source_bounded_identity.glb"]
            ),
            "reviewed_coat": dict(
                records_without_receipt["evidence/source_identity_coat.png"]
            ),
            "bounded_realization_exact_root_authenticated": True,
            "bounded_realization_exact_evidence_authenticated": True,
        },
        "toolchain": {
            "executables": {
                name: {
                    "identity": dict(executable.record),
                    "version_argv": list(executable.version_argv),
                    "version_stdout": executable.version_stdout,
                }
                for name, executable in executables.items()
            },
            "tools": {
                name: {
                    "external": {
                        "path": str(REVIEW_TOOL_FILES[name]),
                        "sha256": record["sha256"],
                        "size_bytes": record["size_bytes"],
                    },
                    "immutable_snapshot": dict(record),
                }
                for name, record in tool_snapshots.items()
            },
            "commands": _command_records(
                commands,
                old_prefix=str(publication.proc_root),
                new_prefix=str(publication.output_root),
            ),
            "media_subprocess_contract": _media_subprocess_contract(publication),
        },
        "automatic_admission_gates": {
            "realization_external_raw_sha256_authenticated": True,
            "realization_exact_immutable_closure_authenticated": True,
            "raw_idle_walking_timing_signature_exactly_equal": True,
            "raw_timing_signature": timing,
            "dense_animation_semantic_gate": dense,
            "structural_skin_skeleton_bind_gate": structure,
            "srgb_and_glb_readback_gate": glb_readback,
            "deformation_audit": {
                "samples_per_action": DEFORMATION_SAMPLES,
                "thresholds": dict(DEFORMATION_THRESHOLDS),
                "overall": deformation["overall"],
                "status": "passed",
            },
            "six_view_media": {
                "labels": list(MEDIA_LABELS),
                "frames_per_view": REVIEW_FRAMES,
                "resolution": [REVIEW_MEDIA_WIDTH, REVIEW_MEDIA_HEIGHT],
                "fps": REVIEW_MEDIA_FPS,
                "render_encode_video_lineage_authenticated": True,
                "status": "passed",
            },
            "all_automatic_gates_passed": True,
        },
        "outputs": _output_descriptor(records_without_receipt),
        "artifact_closure": _records_to_closure(records_without_receipt),
        "authority_boundary": {
            "automatic_technical_review_completed": True,
            "owner_animation_decision": "pending",
            "owner_identity_decision": "pending",
            "ue_asset_bound_readback": "pending",
            "formal_dataset_registration_authorized": False,
            "native_merge_authorized": False,
            "same_uid_continuous_write_adversary_out_of_scope": True,
            "pathnames_are_not_authority": True,
            "caller_supplied_raw_review_sha256_required": True,
        },
        "next_gate": "exact_artifact_bound_human_animation_and_identity_review",
    }
    receipt["review_sha256"] = canonical_sha256(receipt)
    require_exact_keys(receipt, RECEIPT_TOP_LEVEL_FIELDS, "review receipt")
    return receipt


def _read_tree_record(root: Path, relative: str) -> dict[str, Any]:
    payload, _ = stable_read_absolute(root / relative, f"review closure {relative}")
    return relative_record(relative, payload)


def scan_sealed_tree(root_fd: int) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Read one exact sealed tree using only held O_NOFOLLOW dirfds."""

    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) != 0o555:
        raise DirectNativeReviewError("sealed tree root is not a 0555 directory")
    files: dict[str, dict[str, Any]] = {}
    directories = {""}
    pending: list[tuple[str, int]] = [("", os.dup(root_fd))]
    try:
        while pending:
            relative, descriptor = pending.pop()
            try:
                current = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(current.st_mode)
                    or stat.S_IMODE(current.st_mode) != 0o555
                ):
                    raise DirectNativeReviewError(
                        f"sealed tree directory is unsafe: {relative}"
                    )
                for name in os.listdir(descriptor):
                    if not name or "/" in name or name in {".", ".."}:
                        raise DirectNativeReviewError("sealed tree entry is unsafe")
                    child = f"{relative}/{name}" if relative else name
                    entry = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if stat.S_ISDIR(entry.st_mode):
                        child_fd = os.open(name, _directory_flags(), dir_fd=descriptor)
                        opened = os.fstat(child_fd)
                        if _identity(opened) != _identity(entry):
                            os.close(child_fd)
                            raise DirectNativeReviewError(
                                f"sealed tree directory raced: {child}"
                            )
                        directories.add(child)
                        pending.append((child, child_fd))
                    elif stat.S_ISREG(entry.st_mode):
                        payload, _, _ = stable_read_at(
                            descriptor,
                            name,
                            label=f"sealed tree {child}",
                            required_mode=0o444,
                        )
                        files[child] = relative_record(child, payload)
                    else:
                        raise DirectNativeReviewError(
                            f"sealed tree contains a link/special file: {child}"
                        )
            finally:
                os.close(descriptor)
    except Exception:
        for _, descriptor in pending:
            os.close(descriptor)
        raise
    return files, directories


def stable_read_relative_at(root_fd: int, relative: str, label: str) -> bytes:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts or "." in parts:
        raise DirectNativeReviewError(f"unsafe relative path for {label}")
    directory_fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, _directory_flags(), dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        payload, _, _ = stable_read_at(
            directory_fd,
            parts[-1],
            label=label,
            required_mode=0o444,
        )
        return payload
    finally:
        os.close(directory_fd)


def authenticate_embedded_realization(
    root_fd: int,
    *,
    expected_raw_manifest_sha256: str,
) -> tuple[dict[str, Any], identity.IdentityPlan, dict[str, Any]]:
    """Re-run the realization gates from the review's immutable flat closure."""

    flat_paths = {
        source_relative: f"evidence/{snapshot_name}"
        for source_relative, snapshot_name in SOURCE_SNAPSHOT_NAMES.items()
    }
    artifacts = {
        source_relative: stable_read_relative_at(
            root_fd, review_relative, f"embedded realization {source_relative}"
        )
        for source_relative, review_relative in flat_paths.items()
    }
    raw_manifest = artifacts["realization_manifest.json"]
    if sha256_bytes(raw_manifest) != expected_raw_manifest_sha256:
        raise DirectNativeReviewError(
            "embedded realization raw manifest authority mismatch"
        )
    manifest = strict_json_bytes(raw_manifest, "embedded realization manifest")
    require_exact_keys(manifest, REALIZATION_TOP_LEVEL_FIELDS, "embedded realization")
    if (
        manifest["schema"] != REALIZATION_SCHEMA
        or manifest["status"] != "research_candidate_pending_final_animation_review"
        or manifest["state_classification"] != "research_candidate"
        or manifest["formal_dataset_registration_authorized"] is not False
        or manifest["next_gate"] != "final_six_view_animation_and_identity_review"
    ):
        raise DirectNativeReviewError("embedded realization status is invalid")
    _validate_realization_publication_security_boundary(
        manifest["publication_security_boundary"]
    )
    outputs = require_exact_keys(
        manifest["outputs"], {"glb", "coat_texture"}, "embedded outputs"
    )
    _require_relative_binding(
        outputs["glb"],
        "bounded_identity.glb",
        artifacts["bounded_identity.glb"],
        "embedded realization GLB",
    )
    _require_relative_binding(
        outputs["coat_texture"],
        "identity_coat.png",
        artifacts["identity_coat.png"],
        "embedded realization coat",
    )
    lineage = require_exact_keys(
        manifest["lineage"],
        {"source", "plan", "preflight", "machine_execution_authorization"},
        "embedded realization lineage",
    )
    for label, relative in (
        ("source", "evidence/source_snapshot.glb"),
        ("plan", "evidence/identity_plan.json"),
        ("preflight", "evidence/preflight_receipt.json"),
        (
            "machine_execution_authorization",
            "evidence/machine_execution_authorization.json",
        ),
    ):
        record = lineage[label]
        require_exact_keys(
            record,
            {"path", "sha256", "size_bytes", "publication_snapshot"},
            f"embedded {label}",
        )
        _require_relative_binding(
            record["publication_snapshot"],
            relative,
            artifacts[relative],
            f"embedded {label} publication snapshot",
        )
        if record["sha256"] != sha256_bytes(artifacts[relative]) or record[
            "size_bytes"
        ] != len(artifacts[relative]):
            raise DirectNativeReviewError(f"embedded {label} descriptor changed")
    snapshots = require_exact_keys(
        manifest["toolchain_publication_snapshots"],
        {"contract", "executor"},
        "embedded tool publication snapshots",
    )
    toolchain = require_exact_keys(
        manifest["toolchain"], {"blender", "contract", "executor"}, "embedded toolchain"
    )
    for label, relative in (
        ("contract", "evidence/bounded_quadruped_identity_contract.py"),
        ("executor", "evidence/blender_realize_bounded_quadruped_identity.py"),
    ):
        _require_relative_binding(
            snapshots[label],
            relative,
            artifacts[relative],
            f"embedded {label} snapshot",
        )
        require_exact_keys(
            toolchain[label], {"path", "sha256", "size_bytes"}, f"embedded {label}"
        )
        if (
            toolchain[label]["sha256"] != sha256_bytes(artifacts[relative])
            or toolchain[label]["size_bytes"] != len(artifacts[relative])
            or not isinstance(toolchain[label]["path"], str)
            or not Path(toolchain[label]["path"]).is_absolute()
        ):
            raise DirectNativeReviewError(f"embedded {label} descriptor changed")
    blender = require_exact_keys(
        toolchain["blender"],
        {"binary", "build_hash", "version"},
        "embedded realization Blender identity",
    )
    if not all(isinstance(blender[field], str) and blender[field] for field in blender):
        raise DirectNativeReviewError(
            "embedded realization Blender identity is incomplete"
        )
    plan_value = strict_json_bytes(
        artifacts["evidence/identity_plan.json"], "embedded identity plan"
    )
    try:
        plan = identity.load_plan(plan_value)
    except identity.IdentityContractError as error:
        raise DirectNativeReviewError(
            f"embedded identity plan rejected: {error}"
        ) from error
    preflight = strict_json_bytes(
        artifacts["evidence/preflight_receipt.json"], "embedded preflight"
    )
    if (
        preflight.get("schema") != identity.PREFLIGHT_SCHEMA
        or preflight.get("status") != "pending_machine_execution_authorization"
        or preflight.get("state_classification") != "research_candidate"
        or preflight.get("execution_authorized") is not False
        or preflight.get("formal_dataset_registration_authorized") is not False
        or preflight.get("automatic_checks", {}).get("overall")
        != "passed_preflight_only"
        or preflight.get("plan", {}).get("sha256")
        != sha256_bytes(artifacts["evidence/identity_plan.json"])
        or preflight.get("source", {}).get("sha256")
        != sha256_bytes(artifacts["evidence/source_snapshot.glb"])
    ):
        raise DirectNativeReviewError("embedded preflight did not pass")
    authorization = strict_json_bytes(
        artifacts["evidence/machine_execution_authorization.json"],
        "embedded authorization",
    )
    require_exact_keys(
        authorization,
        {
            "schema",
            "decision",
            "preflight_receipt_sha256",
            "plan_sha256",
            "source_sha256",
            "checks",
            "notes",
            "authorization_basis",
            "authorized_actor_type",
            "authorized_output_scope",
            "state_classification",
            "formal_dataset_registration_authorized",
            "user_approval_inferred",
        },
        "embedded authorization",
    )
    if (
        authorization["schema"] != identity.AUTHORIZATION_SCHEMA
        or authorization["decision"]
        != "authorized_for_new_bounded_research_candidate_output"
        or authorization["preflight_receipt_sha256"]
        != sha256_bytes(artifacts["evidence/preflight_receipt.json"])
        or authorization["plan_sha256"]
        != sha256_bytes(artifacts["evidence/identity_plan.json"])
        or authorization["source_sha256"]
        != sha256_bytes(artifacts["evidence/source_snapshot.glb"])
        or set(authorization["checks"]) != identity.AUTHORIZATION_CHECKS
        or any(value is not True for value in authorization["checks"].values())
        or authorization["authorization_basis"]
        != "scoped_task_continue_asset_identity_repair"
        or authorization["authorized_actor_type"] != "codex_task_agent"
        or authorization["authorized_output_scope"]
        not in AUTHORIZED_RESEARCH_OUTPUT_SCOPES
        or authorization["state_classification"] != "research_candidate"
        or authorization["formal_dataset_registration_authorized"] is not False
        or authorization["user_approval_inferred"] is not False
    ):
        raise DirectNativeReviewError("embedded authorization binding is invalid")
    invariants = manifest["invariants"]
    if not isinstance(invariants, Mapping):
        raise DirectNativeReviewError("embedded realization invariants are missing")
    timing = _validate_raw_animation_timing(invariants)
    dense = _validate_dense_animation_gate(invariants, plan)
    structure = _validate_structural_invariants(invariants, plan)
    glb_readback = _validate_glb_and_coat(
        artifacts["bounded_identity.glb"],
        artifacts["identity_coat.png"],
        manifest,
        plan,
    )
    _validate_operations(manifest, plan)
    authority = require_exact_keys(
        manifest["authority_boundary"],
        {
            "owner_animation_decision",
            "breed_identity_decision",
            "ue_asset_bound_readback",
            "dynamic_audio",
            "native_merge_authorized",
        },
        "embedded realization authority boundary",
    )
    if authority != {
        "owner_animation_decision": "pending",
        "breed_identity_decision": "pending",
        "ue_asset_bound_readback": "pending",
        "dynamic_audio": "pending",
        "native_merge_authorized": False,
    }:
        raise DirectNativeReviewError("embedded realization claims excess authority")
    return (
        manifest,
        plan,
        {
            "raw_timing_signature": timing,
            "dense_animation_semantic_gate": dense,
            "structural_skin_skeleton_bind_gate": structure,
            "srgb_and_glb_readback_gate": glb_readback,
        },
    )


def _validate_receipt_shape(receipt: Mapping[str, Any]) -> None:
    require_exact_keys(receipt, RECEIPT_TOP_LEVEL_FIELDS, "review receipt")
    try:
        created = datetime.fromisoformat(receipt["created"])
    except (TypeError, ValueError) as error:
        raise DirectNativeReviewError("review creation time is invalid") from error
    if created.tzinfo is None:
        raise DirectNativeReviewError("review creation time must be timezone-aware")
    require_exact_keys(
        receipt["state"],
        {
            "classification",
            "owner_animation_decision",
            "owner_identity_decision",
            "ue_asset_bound_readback",
            "native_merge_authorized",
        },
        "review state",
    )
    require_exact_keys(
        receipt["authority_boundary"],
        {
            "automatic_technical_review_completed",
            "owner_animation_decision",
            "owner_identity_decision",
            "ue_asset_bound_readback",
            "formal_dataset_registration_authorized",
            "native_merge_authorized",
            "same_uid_continuous_write_adversary_out_of_scope",
            "pathnames_are_not_authority",
            "caller_supplied_raw_review_sha256_required",
        },
        "review authority boundary",
    )
    lineage = require_exact_keys(
        receipt["lineage"],
        {
            "realization_manifest",
            "reviewed_glb",
            "reviewed_coat",
            "bounded_realization_exact_root_authenticated",
            "bounded_realization_exact_evidence_authenticated",
        },
        "review lineage",
    )
    require_exact_keys(
        lineage["realization_manifest"],
        {
            "external_path",
            "external_raw_sha256_authority",
            "immutable_snapshot",
        },
        "review realization lineage",
    )
    require_sha256(
        lineage["realization_manifest"]["external_raw_sha256_authority"],
        "review realization raw SHA-256",
    )
    for label in ("immutable_snapshot",):
        require_exact_keys(
            lineage["realization_manifest"][label],
            {"path", "sha256", "size_bytes"},
            f"review realization {label}",
        )
    for label in ("reviewed_glb", "reviewed_coat"):
        require_exact_keys(
            lineage[label], {"path", "sha256", "size_bytes"}, f"review {label}"
        )
    toolchain = require_exact_keys(
        receipt["toolchain"],
        {"executables", "tools", "commands", "media_subprocess_contract"},
        "review toolchain",
    )
    require_exact_keys(
        toolchain["executables"],
        {"blender", "ffmpeg", "ffprobe", "python"},
        "executables",
    )
    version_arguments = {
        "blender": ["--version"],
        "ffmpeg": ["-version"],
        "ffprobe": ["-version"],
        "python": ["--version"],
    }
    for label, executable in toolchain["executables"].items():
        require_exact_keys(
            executable,
            {"identity", "version_argv", "version_stdout"},
            f"{label} executable",
        )
        require_exact_keys(
            executable["identity"],
            {"path", "sha256", "size_bytes", "device", "inode"},
            f"{label} executable identity",
        )
        require_sha256(executable["identity"]["sha256"], f"{label} executable SHA-256")
        executable_path = executable["identity"]["path"]
        if (
            not isinstance(executable_path, str)
            or not Path(executable_path).is_absolute()
            or executable["version_argv"]
            != [executable_path, *version_arguments[label]]
            or not isinstance(executable["version_stdout"], str)
            or not executable["version_stdout"]
            or any(
                not isinstance(executable["identity"][field], int)
                or isinstance(executable["identity"][field], bool)
                or executable["identity"][field] < 0
                for field in ("size_bytes", "device", "inode")
            )
        ):
            raise DirectNativeReviewError(f"{label} version evidence is incomplete")
    require_exact_keys(
        toolchain["tools"], set(REVIEW_TOOL_FILES), "review execution tools"
    )
    for label, tool in toolchain["tools"].items():
        require_exact_keys(
            tool, {"external", "immutable_snapshot"}, f"{label} review tool"
        )
        require_exact_keys(
            tool["external"],
            {"path", "sha256", "size_bytes"},
            f"{label} external tool",
        )
        require_exact_keys(
            tool["immutable_snapshot"],
            {"path", "sha256", "size_bytes"},
            f"{label} tool snapshot",
        )
        if (
            not isinstance(tool["external"]["path"], str)
            or not Path(tool["external"]["path"]).is_absolute()
            or tool["immutable_snapshot"]["path"] != f"evidence/tool_{label}.py"
            or tool["external"]["sha256"] != tool["immutable_snapshot"]["sha256"]
            or tool["external"]["size_bytes"]
            != tool["immutable_snapshot"]["size_bytes"]
        ):
            raise DirectNativeReviewError(f"{label} tool snapshot binding changed")
    commands = toolchain["commands"]
    if not isinstance(commands, list) or [
        command.get("stage") for command in commands
    ] != list(REVIEW_STAGES):
        raise DirectNativeReviewError("review command lineage must contain 13 stages")
    for command in commands:
        require_exact_keys(
            command,
            {"stage", "observed_argv", "published_equivalent_argv"},
            "review command",
        )
        if (
            not isinstance(command["stage"], str)
            or not isinstance(command["observed_argv"], list)
            or not command["observed_argv"]
            or not isinstance(command["published_equivalent_argv"], list)
            or len(command["observed_argv"])
            != len(command["published_equivalent_argv"])
        ):
            raise DirectNativeReviewError("review command argv is incomplete")
        stage = command["stage"]
        observed = command["observed_argv"]
        published = command["published_equivalent_argv"]
        if stage == "deformation_audit":
            executable_name = "blender"
            tool_name = "deformation_auditor"
            script_index = 5
        elif stage.startswith("render_"):
            executable_name = "blender"
            tool_name = "animation_renderer"
            script_index = 5
        else:
            executable_name = "python"
            tool_name = "media_encoder"
            script_index = 1
        if (
            observed[0] != toolchain["executables"][executable_name]["identity"]["path"]
            or published[0]
            != toolchain["executables"][executable_name]["identity"]["path"]
            or len(observed) <= script_index
            or len(published) <= script_index
            or observed[script_index]
            != toolchain["tools"][tool_name]["external"]["path"]
            or published[script_index]
            != toolchain["tools"][tool_name]["external"]["path"]
        ):
            raise DirectNativeReviewError(
                f"review command executable/tool binding changed: {stage}"
            )
    media_contract = require_exact_keys(
        toolchain["media_subprocess_contract"],
        {"ffmpeg", "ffprobe"},
        "media subprocess contract",
    )
    for executable_name in ("ffmpeg", "ffprobe"):
        records = media_contract[executable_name]
        if (
            not isinstance(records, list)
            or len(records) != len(MEDIA_LABELS)
            or [record.get("label") for record in records] != list(MEDIA_LABELS)
        ):
            raise DirectNativeReviewError(
                f"{executable_name} subprocess coverage changed"
            )
        for record in records:
            require_exact_keys(
                record,
                {"label", "resolved_executable_identity", "observed_argv"},
                f"{executable_name} subprocess",
            )
            if (
                record["label"] not in MEDIA_LABELS
                or record["resolved_executable_identity"]
                != f"toolchain.executables.{executable_name}"
                or not isinstance(record["observed_argv"], list)
                or not record["observed_argv"]
                or record["observed_argv"][0] != executable_name
            ):
                raise DirectNativeReviewError(
                    f"{executable_name} subprocess lineage changed"
                )
    gates = require_exact_keys(
        receipt["automatic_admission_gates"],
        {
            "realization_external_raw_sha256_authenticated",
            "realization_exact_immutable_closure_authenticated",
            "raw_idle_walking_timing_signature_exactly_equal",
            "raw_timing_signature",
            "dense_animation_semantic_gate",
            "structural_skin_skeleton_bind_gate",
            "srgb_and_glb_readback_gate",
            "deformation_audit",
            "six_view_media",
            "all_automatic_gates_passed",
        },
        "automatic admission gates",
    )
    require_exact_keys(
        gates["dense_animation_semantic_gate"],
        {
            "phases_per_action",
            "phase_interval",
            "maximum_skinned_world_vertex_delta",
            "maximum_skinned_world_vertex_delta_gate",
            "maximum_root_trajectory_delta",
            "maximum_root_trajectory_delta_gate",
            "maximum_animation_duration_delta_frames",
            "maximum_animation_duration_delta_frames_gate",
            "status",
        },
        "dense animation receipt gate",
    )
    require_exact_keys(
        gates["structural_skin_skeleton_bind_gate"],
        {
            "topology_unchanged_in_memory",
            "weights_unchanged_in_memory",
            "skeleton_unchanged_in_memory",
            "actions_unchanged_in_memory",
            "roundtrip_geometry_and_weight_clusters_passed",
            "joint_count",
            "maximum_bind_matrix_semantic_delta",
            "maximum_bind_matrix_semantic_delta_gate",
            "status",
        },
        "structural receipt gate",
    )
    require_exact_keys(
        gates["srgb_and_glb_readback_gate"],
        {
            "mesh_count",
            "skin_count",
            "animation_names",
            "material_count",
            "texture_count",
            "image_count",
            "primitive_count",
            "required_attributes",
            "sRGB_transfer_applied_exactly_once",
            "base_rgba8",
            "white_rgba8",
            "status",
        },
        "sRGB/GLB receipt gate",
    )
    require_exact_keys(
        gates["deformation_audit"],
        {"samples_per_action", "thresholds", "overall", "status"},
        "deformation receipt gate",
    )
    require_exact_keys(
        gates["six_view_media"],
        {
            "labels",
            "frames_per_view",
            "resolution",
            "fps",
            "render_encode_video_lineage_authenticated",
            "status",
        },
        "six-view receipt gate",
    )
    outputs = require_exact_keys(
        receipt["outputs"], {"deformation_audit", "media"}, "review outputs"
    )
    require_exact_keys(
        outputs["deformation_audit"],
        {"path", "sha256", "size_bytes"},
        "deformation output",
    )
    require_exact_keys(outputs["media"], set(MEDIA_LABELS), "review media outputs")
    for label, media in outputs["media"].items():
        require_exact_keys(
            media,
            {"video", "render_manifest", "encode_manifest", "frames"},
            f"{label} media output",
        )
        for descriptor in ("video", "render_manifest", "encode_manifest"):
            require_exact_keys(
                media[descriptor],
                {"path", "sha256", "size_bytes"},
                f"{label} {descriptor}",
            )
        if not isinstance(media["frames"], list) or len(media["frames"]) != 24:
            raise DirectNativeReviewError(f"{label} frame output count changed")
        for frame in media["frames"]:
            require_exact_keys(
                frame, {"path", "sha256", "size_bytes"}, f"{label} frame"
            )
    if (
        receipt["schema"] != SCHEMA
        or receipt["status"] != STATUS
        or receipt["route"] != ROUTE
        or receipt["formal"]
        != {
            "formal_dataset_registration_authorized": False,
            "ue_import_authorized": False,
            "native_merge_authorized": False,
        }
        or receipt["state"].get("classification") != "research_candidate"
        or receipt["state"].get("owner_animation_decision") != "pending"
        or receipt["state"].get("owner_identity_decision") != "pending"
        or receipt["state"].get("ue_asset_bound_readback") != "pending"
        or receipt["state"].get("native_merge_authorized") is not False
        or receipt["authority_boundary"]
        != {
            "automatic_technical_review_completed": True,
            "owner_animation_decision": "pending",
            "owner_identity_decision": "pending",
            "ue_asset_bound_readback": "pending",
            "formal_dataset_registration_authorized": False,
            "native_merge_authorized": False,
            "same_uid_continuous_write_adversary_out_of_scope": True,
            "pathnames_are_not_authority": True,
            "caller_supplied_raw_review_sha256_required": True,
        }
        or lineage["bounded_realization_exact_root_authenticated"] is not True
        or lineage["bounded_realization_exact_evidence_authenticated"] is not True
        or not isinstance(lineage["realization_manifest"]["external_path"], str)
        or not Path(lineage["realization_manifest"]["external_path"]).is_absolute()
        or receipt["next_gate"]
        != "exact_artifact_bound_human_animation_and_identity_review"
    ):
        raise DirectNativeReviewError("review receipt status/route/authority changed")
    expected_review_sha = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "review_sha256"}
    )
    if receipt["review_sha256"] != expected_review_sha:
        raise DirectNativeReviewError("review canonical SHA-256 is invalid")
    required_true = (
        "realization_external_raw_sha256_authenticated",
        "realization_exact_immutable_closure_authenticated",
        "raw_idle_walking_timing_signature_exactly_equal",
        "all_automatic_gates_passed",
    )
    if any(gates.get(field) is not True for field in required_true):
        raise DirectNativeReviewError("review automatic gate did not pass")
    _validate_raw_animation_timing(
        {
            "raw_source_animation_time_signature": gates["raw_timing_signature"],
            "raw_output_animation_time_signature": gates["raw_timing_signature"],
        }
    )
    structure = gates["structural_skin_skeleton_bind_gate"]
    srgb = gates["srgb_and_glb_readback_gate"]
    if (
        gates.get("dense_animation_semantic_gate", {}).get("phases_per_action") != 81
        or gates.get("dense_animation_semantic_gate", {}).get("phase_interval")
        != "index/80"
        or gates.get("deformation_audit", {}).get("samples_per_action")
        != DEFORMATION_SAMPLES
        or gates.get("deformation_audit", {}).get("thresholds")
        != DEFORMATION_THRESHOLDS
        or gates.get("deformation_audit", {}).get("overall") != "passed"
        or gates.get("deformation_audit", {}).get("status") != "passed"
        or gates.get("six_view_media", {}).get("labels") != list(MEDIA_LABELS)
        or gates.get("six_view_media", {}).get("frames_per_view") != REVIEW_FRAMES
        or gates.get("six_view_media", {}).get("resolution")
        != [REVIEW_MEDIA_WIDTH, REVIEW_MEDIA_HEIGHT]
        or gates.get("six_view_media", {}).get("fps") != REVIEW_MEDIA_FPS
        or gates.get("six_view_media", {}).get(
            "render_encode_video_lineage_authenticated"
        )
        is not True
        or gates.get("six_view_media", {}).get("status") != "passed"
        or gates.get("dense_animation_semantic_gate", {}).get("status") != "passed"
        or any(
            structure.get(field) is not True
            for field in (
                "topology_unchanged_in_memory",
                "weights_unchanged_in_memory",
                "skeleton_unchanged_in_memory",
                "actions_unchanged_in_memory",
                "roundtrip_geometry_and_weight_clusters_passed",
            )
        )
        or structure.get("status") != "passed"
        or srgb.get("sRGB_transfer_applied_exactly_once") is not True
        or srgb.get("status") != "passed"
    ):
        raise DirectNativeReviewError("review fixed thresholds/media contract changed")


def _validate_fixed_protocol_tree_bindings(
    receipt: Mapping[str, Any],
    tree_records: Mapping[str, Mapping[str, Any]],
    tree_directories: set[str],
) -> None:
    closure = receipt["artifact_closure"]
    require_exact_keys(
        closure,
        {
            "artifacts",
            "inventory_sha256",
            "receipt_excluded_from_artifact_inventory_to_avoid_self_reference",
            "exact_directory_inventory_required",
        },
        "review artifact closure",
    )
    artifacts = closure["artifacts"]
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or closure["inventory_sha256"] != canonical_sha256(artifacts)
        or closure["receipt_excluded_from_artifact_inventory_to_avoid_self_reference"]
        is not True
        or closure["exact_directory_inventory_required"] is not True
    ):
        raise DirectNativeReviewError("review artifact closure hash is invalid")
    protocol_files = _expected_review_files()
    if set(tree_records) != protocol_files:
        raise DirectNativeReviewError(
            "published review file inventory changed from the fixed protocol"
        )
    expected_artifacts = [
        dict(tree_records[relative])
        for relative in sorted(protocol_files - {RECEIPT_NAME})
    ]
    if artifacts != expected_artifacts:
        raise DirectNativeReviewError(
            "review artifact closure differs from the fixed protocol inventory"
        )
    expected_directories = {
        "",
        "evidence",
        "media",
        *(f"media/{label}_frames" for label in MEDIA_LABELS),
    }
    if tree_directories != expected_directories:
        raise DirectNativeReviewError(
            "published review exact directory inventory changed"
        )
    if receipt["outputs"] != _output_descriptor(tree_records):
        raise DirectNativeReviewError(
            "receipt outputs differ from the fixed protocol paths"
        )
    lineage = receipt["lineage"]
    if (
        lineage["realization_manifest"]["immutable_snapshot"]
        != tree_records["evidence/source_realization_manifest.json"]
    ):
        raise DirectNativeReviewError("realization snapshot closure changed")
    expected_lineage_records = {
        "reviewed_glb": tree_records["evidence/source_bounded_identity.glb"],
        "reviewed_coat": tree_records["evidence/source_identity_coat.png"],
    }
    for label, expected_record in expected_lineage_records.items():
        if lineage[label] != expected_record:
            raise DirectNativeReviewError(
                f"{label} differs from its fixed immutable review artifact"
            )
    for label, tool in receipt["toolchain"]["tools"].items():
        expected_record = tree_records[f"evidence/tool_{label}.py"]
        if tool["immutable_snapshot"] != expected_record:
            raise DirectNativeReviewError(
                f"{label} review tool snapshot closure changed"
            )


def _validate_execution_lineage_against_root(
    receipt: Mapping[str, Any],
    root: Path,
) -> None:
    expected_commands = build_commands(
        root=root,
        blender=Path(
            receipt["toolchain"]["executables"]["blender"]["identity"]["path"]
        ),
        python=Path(receipt["toolchain"]["executables"]["python"]["identity"]["path"]),
        input_glb=root / "evidence/source_bounded_identity.glb",
        tool_paths={
            label: Path(receipt["toolchain"]["tools"][label]["external"]["path"])
            for label in (
                "deformation_auditor",
                "animation_renderer",
                "media_encoder",
            )
        },
    )
    proc_prefix = None
    for command_record, (stage, expected_argv) in zip(
        receipt["toolchain"]["commands"], expected_commands
    ):
        if (
            command_record["stage"] != stage
            or command_record["published_equivalent_argv"] != expected_argv
        ):
            raise DirectNativeReviewError(
                f"published review command plan changed: {stage}"
            )
        normalized, proc_prefix = _normalize_historical_proc_argv(
            command_record["observed_argv"],
            published_root=root,
            expected_proc_prefix=proc_prefix,
            label=f"review command {stage}",
        )
        if normalized != expected_argv:
            raise DirectNativeReviewError(
                f"observed review command plan changed: {stage}"
            )
    expected_media_contract = _media_subprocess_contract_for_root(root)
    for executable_name in ("ffmpeg", "ffprobe"):
        observed_records = receipt["toolchain"]["media_subprocess_contract"][
            executable_name
        ]
        for observed_record, expected_record in zip(
            observed_records, expected_media_contract[executable_name]
        ):
            if (
                observed_record["label"] != expected_record["label"]
                or observed_record["resolved_executable_identity"]
                != expected_record["resolved_executable_identity"]
            ):
                raise DirectNativeReviewError(
                    f"{executable_name} subprocess identity/label changed"
                )
            normalized, proc_prefix = _normalize_historical_proc_argv(
                observed_record["observed_argv"],
                published_root=root,
                expected_proc_prefix=proc_prefix,
                label=(f"{executable_name} subprocess {observed_record['label']}"),
            )
            if normalized != expected_record["observed_argv"]:
                raise DirectNativeReviewError(
                    f"{executable_name} subprocess argv changed"
                )
    if proc_prefix is None:
        raise DirectNativeReviewError(
            "review command lineage lacks a held publication root"
        )


def _validate_embedded_gate_parity(
    receipt_gates: Mapping[str, Any],
    embedded_gates: Mapping[str, Any],
) -> None:
    for label, expected_gate in embedded_gates.items():
        if receipt_gates.get(label) != expected_gate:
            raise DirectNativeReviewError(
                f"receipt {label} contradicts embedded realization evidence"
            )


def _require_validation_ffprobe(
    receipt: Mapping[str, Any],
    ffprobe: ExecutableIdentity,
) -> None:
    recorded_ffprobe = receipt["toolchain"]["executables"]["ffprobe"]
    current_ffprobe = {
        "identity": dict(ffprobe.record),
        "version_argv": list(ffprobe.version_argv),
        "version_stdout": ffprobe.version_stdout,
    }
    if recorded_ffprobe != current_ffprobe:
        raise DirectNativeReviewError(
            "validation ffprobe does not match the receipt executable identity"
        )
    ffprobe.verify()
    resolved_ffprobe = shutil.which("ffprobe", path=str(ffprobe.path.parent))
    if resolved_ffprobe is None or not Path(resolved_ffprobe).samefile(ffprobe.path):
        raise DirectNativeReviewError(
            "validation ffprobe is not the executable resolved for media readback"
        )


def validate_published_review(
    root: Path,
    *,
    expected_raw_review_sha256: str,
    ffprobe: ExecutableIdentity,
) -> dict[str, Any]:
    """Reauthenticate a published review using caller-provided raw authority."""

    require_sha256(expected_raw_review_sha256, "expected raw review SHA-256")
    root = Path(os.path.abspath(os.fspath(root)))
    root_fd, _ = open_absolute_directory(root)
    try:
        tree_records, tree_directories = scan_sealed_tree(root_fd)
        receipt_payload, _, _ = stable_read_at(
            root_fd,
            RECEIPT_NAME,
            label="published review receipt",
            required_mode=0o444,
        )
        if sha256_bytes(receipt_payload) != expected_raw_review_sha256:
            raise DirectNativeReviewError(
                "caller raw review SHA-256 authority mismatch"
            )
        receipt = strict_json_bytes(receipt_payload, "published review receipt")
        _validate_receipt_shape(receipt)
        _validate_fixed_protocol_tree_bindings(
            receipt,
            tree_records,
            tree_directories,
        )
        lineage = receipt["lineage"]["realization_manifest"]
        _validate_execution_lineage_against_root(receipt, root)
        _embedded_manifest, _embedded_plan, embedded_gates = (
            authenticate_embedded_realization(
                root_fd,
                expected_raw_manifest_sha256=lineage["external_raw_sha256_authority"],
            )
        )
        gates = receipt["automatic_admission_gates"]
        _validate_embedded_gate_parity(gates, embedded_gates)
        _require_validation_ffprobe(receipt, ffprobe)
        deformation_path = root / "evidence/deformation_audit.json"
        reviewed_glb = root / "evidence/source_bounded_identity.glb"
        deformation = require_deformation_audit(
            deformation_path,
            reviewed_glb,
            "direct-native deformation audit",
            expected_samples=DEFORMATION_SAMPLES,
            require_pass=True,
        )
        old_path = os.environ.get("PATH")
        try:
            ffprobe.verify()
            os.environ["PATH"] = str(ffprobe.path.parent)
            paths = {"review_root": root / "media"}
            media, media_lineage = require_review_media_set(
                paths,
                MEDIA_LABELS,
                reviewed_glb,
                REVIEW_FRAMES,
            )
        finally:
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path
        ffprobe.verify()
        if set(media) != set(MEDIA_LABELS) or set(media_lineage) != set(MEDIA_LABELS):
            raise DirectNativeReviewError("six-view media set is incomplete")
        expected_deformation_gate = {
            "samples_per_action": DEFORMATION_SAMPLES,
            "thresholds": dict(DEFORMATION_THRESHOLDS),
            "overall": deformation["overall"],
            "status": "passed",
        }
        if gates["deformation_audit"] != expected_deformation_gate:
            raise DirectNativeReviewError(
                "receipt deformation gate contradicts validated audit evidence"
            )
        return receipt
    finally:
        os.close(root_fd)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realization-manifest", type=Path)
    parser.add_argument("--expected-realization-manifest-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Authenticate inputs and print the exact expensive command plan only.",
    )
    parser.add_argument(
        "--validate-review-root",
        type=Path,
        help="Reauthenticate an already-published v2 review without rendering.",
    )
    parser.add_argument("--expected-review-sha256")
    return parser.parse_args(argv)


def _require_producer_arguments(args) -> None:
    if (
        args.realization_manifest is None
        or args.expected_realization_manifest_sha256 is None
        or args.output_root is None
    ):
        raise DirectNativeReviewError(
            "producer requires --realization-manifest, "
            "--expected-realization-manifest-sha256, and --output-root"
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.validate_review_root is not None:
        if args.expected_review_sha256 is None:
            raise DirectNativeReviewError(
                "--validate-review-root requires --expected-review-sha256"
            )
        ffprobe = authenticate_executable(
            args.ffprobe, name="ffprobe", version_arguments=["-version"]
        )
        try:
            receipt = validate_published_review(
                args.validate_review_root,
                expected_raw_review_sha256=args.expected_review_sha256,
                ffprobe=ffprobe,
            )
        finally:
            ffprobe.close()
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "validated",
                    "review_sha256": receipt["review_sha256"],
                },
                indent=2,
            )
        )
        return 0

    _require_producer_arguments(args)
    realization = authenticate_realization(
        args.realization_manifest,
        args.expected_realization_manifest_sha256,
    )
    executables: dict[str, ExecutableIdentity] = {}
    review_tools: dict[str, AuthenticatedToolFile] = {}
    publication: Optional[SecureReviewPublication] = None
    resources_closed = False
    try:
        executables["blender"] = authenticate_executable(
            args.blender, name="Blender", version_arguments=["--version"]
        )
        executables["ffmpeg"] = authenticate_executable(
            args.ffmpeg, name="ffmpeg", version_arguments=["-version"]
        )
        executables["ffprobe"] = authenticate_executable(
            args.ffprobe, name="ffprobe", version_arguments=["-version"]
        )
        executables["python"] = authenticate_executable(
            os.path.realpath(sys.executable),
            name="Python",
            version_arguments=["--version"],
        )
        for name, path in REVIEW_TOOL_FILES.items():
            review_tools[name] = authenticate_tool_file(path, name=name)
        output_root = Path(os.path.abspath(os.fspath(args.output_root)))
        preview_commands = build_commands(
            root=output_root,
            blender=executables["blender"].path,
            python=executables["python"].path,
            input_glb=output_root / "evidence/source_bounded_identity.glb",
        )
        if args.validate_only:
            preview = json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "validated_plan_only_no_expensive_commands_run",
                    "route": ROUTE,
                    "review_frames": REVIEW_FRAMES,
                    "deformation_samples": DEFORMATION_SAMPLES,
                    "commands": preview_commands,
                },
                indent=2,
            )
            _close_producer_resources(
                publication,
                executables,
                review_tools,
                realization,
            )
            resources_closed = True
            print(preview)
            return 0
        publication = SecureReviewPublication(output_root)
        for source_relative, snapshot_name in SOURCE_SNAPSHOT_NAMES.items():
            publication.write_exclusive(
                f"evidence/{snapshot_name}",
                realization.artifacts[source_relative],
            )
        tool_snapshots: dict[str, dict[str, Any]] = {}
        for name, tool in review_tools.items():
            relative = f"evidence/tool_{name}.py"
            publication.write_exclusive(relative, tool.payload)
            tool_snapshots[name] = relative_record(relative, tool.payload)
        input_glb = publication.proc_path("evidence/source_bounded_identity.glb")
        commands = build_commands(
            root=publication.proc_root,
            blender=executables["blender"].path,
            python=executables["python"].path,
            input_glb=input_glb,
        )
        executable_path = os.pathsep.join(
            dict.fromkeys(
                [
                    str(executables["ffmpeg"].path.parent),
                    str(executables["ffprobe"].path.parent),
                ]
            )
        )
        for basename in ("ffmpeg", "ffprobe"):
            resolved = shutil.which(basename, path=executable_path)
            expected = executables[basename].path
            if resolved is None or not Path(resolved).samefile(expected):
                raise DirectNativeReviewError(
                    f"explicit {basename} is not the executable resolved for media stages"
                )
        command_environment = dict(os.environ)
        command_environment["PATH"] = executable_path
        for stage, command in commands:
            for executable in executables.values():
                executable.verify()
            for tool in review_tools.values():
                tool.verify()
            subprocess.run(
                command,
                cwd=SPEAR_ROOT,
                env=command_environment,
                check=True,
            )
            for executable in executables.values():
                executable.verify()
            for tool in review_tools.values():
                tool.verify()
            if stage == "deformation_audit":
                require_deformation_audit(
                    publication.proc_path("evidence/deformation_audit.json"),
                    input_glb,
                    "direct-native deformation audit",
                    expected_samples=DEFORMATION_SAMPLES,
                    require_pass=True,
                )
        proc_paths = {"review_root": publication.proc_path("media")}
        old_path = os.environ.get("PATH")
        try:
            os.environ["PATH"] = executable_path
            require_review_media_set(
                proc_paths,
                MEDIA_LABELS,
                input_glb,
                REVIEW_FRAMES,
            )
        finally:
            if old_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = old_path
        for tool in review_tools.values():
            tool.verify()
        normalize_generated_lineage(publication)
        provisional_files = _expected_review_files() - {RECEIPT_NAME}
        records_without_receipt = publication._scan(required_mode=None)
        if set(records_without_receipt) != provisional_files:
            raise DirectNativeReviewError(
                "review output exact inventory changed before receipt"
            )
        normalized_deformation = strict_json_bytes(
            publication.read_file("evidence/deformation_audit.json"),
            "normalized deformation audit",
        )
        receipt = build_receipt(
            realization=realization,
            realization_manifest_path=args.realization_manifest,
            expected_realization_sha256=args.expected_realization_manifest_sha256,
            records_without_receipt=records_without_receipt,
            tool_snapshots=tool_snapshots,
            executables=executables,
            commands=commands,
            publication=publication,
            deformation=normalized_deformation,
        )
        publication.write_exclusive(RECEIPT_NAME, pretty_json_bytes(receipt))
        sealed = publication.seal(_expected_review_files())
        publication.publish(sealed)
        raw_review_sha256 = sealed[RECEIPT_NAME]["sha256"]
        validate_published_review(
            publication.output_root,
            expected_raw_review_sha256=raw_review_sha256,
            ffprobe=executables["ffprobe"],
        )
        for tool in review_tools.values():
            tool.verify()
        publication._require_published_entry()
        publication.verify(sealed)
        success = (
            "DIRECT_NATIVE_AUTHORED_QUADRUPED_REVIEW_OK "
            f"output={publication.output_root} "
            f"review_raw_sha256={raw_review_sha256} "
            f"review_canonical_sha256={receipt['review_sha256']}"
        )
        _close_producer_resources(
            publication,
            executables,
            review_tools,
            realization,
        )
        resources_closed = True
        print(success)
        return 0
    except Exception:
        if publication is not None:
            publication.quarantine()
        raise
    finally:
        if not resources_closed:
            _close_producer_resources(
                publication,
                executables,
                review_tools,
                realization,
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        DirectNativeReviewError,
        identity.IdentityContractError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(
            f"DIRECT_NATIVE_AUTHORED_QUADRUPED_REVIEW_FAILED {error}", file=sys.stderr
        )
        raise SystemExit(2)
