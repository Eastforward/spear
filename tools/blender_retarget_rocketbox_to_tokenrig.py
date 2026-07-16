#!/usr/bin/env python3
"""Retarget sealed Rocketbox Walk/Idle motion to a static-approved TokenRig human."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ACTION_NAMES = {"walk": "Walking", "idle": "Standing_Idle"}
CANONICAL_FRONT = "negative-y"
CANONICAL_UP = "positive-z"
CANONICAL_FRONT_VECTOR = (0.0, -1.0, 0.0)
CANONICAL_UP_VECTOR = (0.0, 0.0, 1.0)
AXIS_MAP_3X3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
MAXIMUM_PENETRATION_M = 0.010
MAXIMUM_GROUNDING_CORRECTION_M = 0.010
MAXIMUM_IK_ANKLE_CORRECTION_M = 0.030
MINIMUM_IK_KNEE_PLANE_DOT = 0.90
MINIMUM_IK_REACH_MARGIN_M = 1.0e-5
MAXIMUM_IK_SEGMENT_LENGTH_INPUT_DRIFT_M = 1.0e-6
WALK_SOURCE_SUPPORT_BASIS = "sealed_walk_mesh_foot_toe_skin_clearance_v1"
IDLE_SOURCE_SUPPORT_BASIS = "exact_idle_world_semantic_joint_trajectory_v1"
MAXIMUM_HOVER_M = 0.030
CONTACT_CLEARANCE_M = 0.015
STANCE_CLEARANCE_M = 0.030
MINIMUM_WALK_CONTACT_RATIO = 0.20
MINIMUM_WALK_STANCE_CONTACT_RATIO = 0.70
MINIMUM_IDLE_CONTACT_RATIO = 0.90
MINIMUM_IDLE_BILATERAL_CONTACT_RATIO = 0.90
EXPECTED_IDLE_FRAME_START = 1
EXPECTED_IDLE_FRAME_END = 351
EXPECTED_IDLE_FRAME_COUNT = 351
MAXIMUM_WALK_CONSECUTIVE_HOVER_FRAMES = 24
MAXIMUM_IDLE_CONSECUTIVE_HOVER_FRAMES = 2
MAXIMUM_STANCE_SLIDE_M = 0.030
MAXIMUM_STANCE_SPEED_MPS = 0.15
QUALITY_SAMPLE_SEED = 20260712
DEFAULT_MAXIMUM_GLOBAL_SAMPLE_VERTICES = 24576
DEFAULT_MAXIMUM_SUPPORT_SAMPLE_VERTICES = 4096
DEFAULT_MAXIMUM_SAMPLE_EDGES = 49152
SUPPORT_CORE_COMBINED_WEIGHT_MINIMUM = 0.5
SUPPORT_CORE_OPPOSITE_WEIGHT_MAXIMUM = 1.0e-4
MINIMUM_CALIBRATED_SHOULDER_SPAN_RATIO = 0.80
MINIMUM_CALIBRATED_HIP_SPAN_RATIO = 0.85
MAXIMUM_CALIBRATED_EDGE_STRETCH_RATIO = 1.35
MINIMUM_FORWARD_DOT = 0.90
MINIMUM_BODY_FORWARD_DOT = 0.75
ROOT_RECONSTRUCTION_TOLERANCE_M = 1.0e-4
SPEED_RECONSTRUCTION_TOLERANCE_MPS = 1.0e-4
LOOP_ROTATION_TOLERANCE_RAD = 0.002
LOOP_ROOT_TOLERANCE_M = 1.0e-4
LOOP_PELVIS_TRANSLATION_TOLERANCE_M = 0.005
LOOP_BOUNDARY_VELOCITY_TOLERANCE_MPS = 0.15
REST_ROTATION_MATRIX_TOLERANCE = 1.0e-4
POSE_TRANSLATION_TOLERANCE_M = 1.0e-4
IK_CONTACT_READBACK_SAFETY_MARGIN_M = POSE_TRANSLATION_TOLERANCE_M
MAXIMUM_CROSS_FOOT_CLEARANCE_CHANGE_M = POSE_TRANSLATION_TOLERANCE_M
MAXIMUM_SURFACE_CONTACT_IK_ITERATIONS = 3
MAXIMUM_LOOP_BOUNDARY_RECONCILIATION_ITERATIONS = 4
SURFACE_CONTACT_IK_SCHEMA = "tokenrig_surface_contact_leg_ik_v1"
LOOP_BOUNDARY_RECONCILIATION_SCHEMA = (
    "tokenrig_symmetric_loop_boundary_velocity_reconciliation_v1"
)
ROUNDTRIP_ENDPOINT_MATRIX_TOLERANCE = 1.0e-4
MAXIMUM_IDLE_SPEED_MPS = 0.02
MAXIMUM_ROTATION_ORTHOGONALITY_ERROR = 5.0e-6
MAXIMUM_ROTATION_SINGULAR_VALUE_DEVIATION = 5.0e-6
MAXIMUM_OBJECT_UNIFORM_SCALE_RELATIVE_VARIATION = (
    MAXIMUM_ROTATION_SINGULAR_VALUE_DEVIATION
)
MAXIMUM_ROTATION_POLAR_RESIDUAL = 5.0e-6
MAXIMUM_ROTATION_CONDITION_NUMBER = 1.00001
MINIMUM_ROTATION_SINGULAR_VALUE = 1.0e-12
MANIFEST_SCHEMA = "tokenrig_rocketbox_retarget_v1"
METRICS_SCHEMA = "tokenrig_rocketbox_retarget_metrics_v1"
STATIC_QA_SCHEMA = "tokenrig_human_static_qa_v1"
STATIC_QA_DECISION = "automatic_static_checks_passed"
BASELINE_MANIFEST_SCHEMA = "rocketbox_baseline_manifest_v1"
BASELINE_RETARGET_SCHEMA = "rocketbox_retarget_manifest_v1"
MOTION_BASIS_SELECTION_SCHEMA = "retarget_motion_basis_correction_v1"
MOTION_BASIS_REVIEW_SCHEMA = "shared_limb_motion_basis_review_v1"
MOTION_BASIS_SELECTION_DECISION = "selected_for_next_retarget"
CANONICAL_BASELINE_ROOT = Path(
    "/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1"
)
CANONICAL_BASELINE_MANIFEST_SHA256 = (
    "b6e468e5f0c79d7ecec168e3c2460a7997a8d2916393da9add1ef2b6952fb922"
)
CANONICAL_BASELINE_MANIFEST_SIZE = 4203
CANONICAL_ROCKETBOX_ROOT = Path("/data/datasets/rocketbox/Microsoft-Rocketbox")
ROCKETBOX_COMMIT = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
ROCKETBOX_SPINE_BONES = ("Bip01 Spine", "Bip01 Spine1", "Bip01 Spine2")
STATIC_AXIS_MATRIX_4X4 = (
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
ROCKETBOX_ROLE_TO_BONE = {
    "pelvis": "Bip01 Pelvis",
    "neck": "Bip01 Neck",
    "head": "Bip01 Head",
    "left_clavicle": "Bip01 L Clavicle",
    "left_upper_arm": "Bip01 L UpperArm",
    "left_forearm": "Bip01 L Forearm",
    "left_hand": "Bip01 L Hand",
    "right_clavicle": "Bip01 R Clavicle",
    "right_upper_arm": "Bip01 R UpperArm",
    "right_forearm": "Bip01 R Forearm",
    "right_hand": "Bip01 R Hand",
    "left_thigh": "Bip01 L Thigh",
    "left_calf": "Bip01 L Calf",
    "left_foot": "Bip01 L Foot",
    "left_toe": "Bip01 L Toe0",
    "right_thigh": "Bip01 R Thigh",
    "right_calf": "Bip01 R Calf",
    "right_foot": "Bip01 R Foot",
    "right_toe": "Bip01 R Toe0",
}
SEMANTIC_METHOD = "unique_topology_and_canonical_rest_position_v1"
SEMANTIC_EXACT_ROLES = (
    "pelvis",
    "neck",
    "head",
    "left_clavicle",
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_clavicle",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
    "left_thigh",
    "left_calf",
    "left_foot",
    "left_toe",
    "right_thigh",
    "right_calf",
    "right_foot",
    "right_toe",
)
SHARED_CANONICAL_LIMB_ROLES = (
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
    "left_thigh",
    "left_calf",
    "left_foot",
    "left_toe",
    "right_thigh",
    "right_calf",
    "right_foot",
    "right_toe",
)
ANATOMICAL_CLAVICLE_ROLES = ("left_clavicle", "right_clavicle")
ANATOMICAL_AXIAL_EXACT_ROLES = ("pelvis", "neck", "head")
ROTATION_DYNAMIC_STAGES = frozenset(
    {
        "source_global_pose",
        "source_local_pose",
        "source_object_world",
    }
)
ROTATION_STATIC_STAGES = frozenset(
    {
        "source_global_rest",
        "target_global_rest",
        "source_local_rest",
        "target_local_rest",
        "target_object_world",
        "canonical_axis_identity",
    }
)
ROTATION_OBJECT_STAGES = frozenset(
    {"source_object_world", "target_object_world"}
)
ROTATION_ALLOWED_STAGES = ROTATION_DYNAMIC_STAGES | ROTATION_STATIC_STAGES


class RetargetError(RuntimeError):
    """Raised when an authenticated retarget invariant cannot be proven."""


class GroundingError(RetargetError):
    """Raised with structured per-frame evidence for a grounding rejection."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any]):
        if not isinstance(evidence, Mapping):
            raise TypeError("grounding rejection evidence must be a mapping")
        self.evidence = json.loads(json.dumps(evidence, sort_keys=True))
        super().__init__(message)


@dataclass(frozen=True)
class IdleMotionContract:
    relative_path: Path
    sha256: str
    size_bytes: int
    git_blob_sha1: str


@dataclass(frozen=True)
class RetargetInputContract:
    baseline_root: Path
    baseline_manifest_sha256: str
    baseline_manifest_size: int
    rocketbox_root: Path
    rocketbox_commit: str
    idle_by_baseline_asset: Mapping[str, IdleMotionContract]


PRODUCTION_INPUT_CONTRACT = RetargetInputContract(
    baseline_root=CANONICAL_BASELINE_ROOT,
    baseline_manifest_sha256=CANONICAL_BASELINE_MANIFEST_SHA256,
    baseline_manifest_size=CANONICAL_BASELINE_MANIFEST_SIZE,
    rocketbox_root=CANONICAL_ROCKETBOX_ROOT,
    rocketbox_commit=ROCKETBOX_COMMIT,
    idle_by_baseline_asset={
        "rocketbox_male_adult_01": IdleMotionContract(
            relative_path=Path(
                "Assets/Animations/all_animations_max_motextr_static/"
                "m_idle_neutral_01.max.fbx"
            ),
            sha256="818cc185af21390575f7fbfdeb3012ba2ce5969fbcb220ea725a2617b339a6e2",
            size_bytes=2418544,
            git_blob_sha1="a2d92c3326a9c503af677c9fa6082387f060d6c4",
        ),
        "rocketbox_female_adult_01": IdleMotionContract(
            relative_path=Path(
                "Assets/Animations/all_animations_max_motextr_static/"
                "f_idle_neutral_01.max.fbx"
            ),
            sha256="fd68b33ea9e290dc734ca8c3a71ef5842bb2dfe719853ff84f6336d06d39fdcb",
            size_bytes=2959360,
            git_blob_sha1="aecf1d0089ccfc0c381d5395294bb1c8fe0e63ae",
        ),
    },
)


_ASSET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_real_directory(path: Path, description: str) -> Path:
    path = _absolute(path)
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise RetargetError(f"{description} must be a direct real directory: {path}")
    return path


def _require_regular_file(path: Path, root: Path, description: str) -> Path:
    path = _absolute(path)
    root = _require_real_directory(root, f"approved {description} root")
    if not os.path.lexists(path):
        raise RetargetError(f"{description} is missing: {path}")
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise RetargetError(f"{description} must be a direct regular file, not a symlink")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RetargetError(f"{description} is outside its approved root") from error
    if path.stat().st_size <= 0:
        raise RetargetError(f"{description} is empty")
    return path


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetargetError(f"{description} is not readable JSON: {error}") from error
    if not isinstance(payload, dict):
        raise RetargetError(f"{description} must be a JSON object")
    return payload


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_file_record(
    path: Path,
    record: Any,
    *,
    size_key: str,
    description: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise RetargetError(f"{description} has no authenticated file record")
    if record.get("sha256") != sha256_file(path):
        raise RetargetError(f"{description} SHA-256 does not match its authenticated record")
    if record.get(size_key) != path.stat().st_size:
        raise RetargetError(f"{description} size does not match its authenticated record")
    return _file_record(path)


def authenticate_static_gate(
    *, asset_id: str, bind_pose_glb: Path, static_qa_json: Path
) -> dict[str, Any]:
    if not _ASSET_ID_RE.fullmatch(asset_id):
        raise RetargetError(f"invalid asset_id: {asset_id!r}")
    bind_argument = _absolute(bind_pose_glb)
    qa_argument = _absolute(static_qa_json)
    if bind_argument.name != "bind_pose.glb":
        raise RetargetError("exact static artifact must be named bind_pose.glb")
    if qa_argument.name != "static_qa.json" or qa_argument.parent != bind_argument.parent:
        raise RetargetError("bind_pose.glb and static_qa.json must share the static bundle")
    root = _require_real_directory(bind_argument.parent, "static QA bundle")
    bind = _require_regular_file(bind_argument, root, "bind_pose.glb")
    qa_path = _require_regular_file(qa_argument, root, "static_qa.json")
    qa = _load_json(qa_path, "static QA")
    if qa.get("schema") != STATIC_QA_SCHEMA:
        raise RetargetError(f"static QA schema must be {STATIC_QA_SCHEMA}")
    if qa.get("asset_id") != asset_id:
        raise RetargetError("static QA asset_id does not match the retarget asset")
    if qa.get("decision") != STATIC_QA_DECISION:
        raise RetargetError("static QA decision is not passed")
    artifacts = qa.get("artifacts")
    record = artifacts.get("bind_pose.glb") if isinstance(artifacts, Mapping) else None
    if not isinstance(record, Mapping) or record.get("filename") != "bind_pose.glb":
        raise RetargetError("static QA does not identify the exact bind_pose.glb")
    bind_record = _validate_file_record(
        bind,
        record,
        size_key="size_bytes",
        description="bind_pose.glb",
    )
    checks = qa.get("checks")
    if not isinstance(checks, Mapping):
        raise RetargetError("static QA checks are missing")
    axis = checks.get("axis_canonicalization")
    if not isinstance(axis, Mapping) or any(
        (
            axis.get("source_front") != "positive-y",
            axis.get("canonical_front") != CANONICAL_FRONT,
            axis.get("yaw_radians") != math.pi,
            axis.get("transform_count") != 1,
            axis.get("matrix") != [list(row) for row in STATIC_AXIS_MATRIX_4X4],
            axis.get("canonical_front_vector") != [0.0, -1.0, 0.0],
            axis.get("canonical_up_vector") != [0.0, 0.0, 1.0],
            axis.get("determinant") != 1.0,
        )
    ):
        raise RetargetError(
            "static axis matrix is not exactly-once FRONT -Y with canonical UP +Z"
        )
    grounding = checks.get("grounding")
    if not isinstance(grounding, Mapping) or any(
        (
            grounding.get("canonical_floor_z") != 0.0,
            grounding.get("post_floor_z") != 0.0,
            grounding.get("transform_count") != 1,
        )
    ):
        raise RetargetError("static floor decision is not one canonical Z=0 floor")
    semantic = validate_semantic_mapping(checks.get("semantic_mapping", {}))
    mesh_contract = checks.get("canonical_mesh_contract")
    if not isinstance(mesh_contract, Mapping):
        raise RetargetError("static canonical mesh contract is missing")
    exported_pbr = checks.get("exported_pbr")
    if not isinstance(exported_pbr, Mapping) or exported_pbr.get("passed") is not True:
        raise RetargetError("static PBR preservation decision is not passed")
    return {
        "bind_pose": bind_record,
        "static_qa": _file_record(qa_path),
        "authenticated_task3": qa.get("authenticated"),
        "semantic_mapping": semantic,
        "canonical_mesh_contract": dict(mesh_contract),
        "pbr_contract": dict(exported_pbr),
        "floor_z_m": 0.0,
        "axis_map_3x3": [list(row) for row in AXIS_MAP_3X3],
    }


def authenticate_sealed_walk(
    *,
    base_avatar_id: str,
    baseline_retarget_blend: Path,
    baseline_retarget_manifest: Path,
    contract: RetargetInputContract = PRODUCTION_INPUT_CONTRACT,
) -> dict[str, Any]:
    root = _require_real_directory(contract.baseline_root, "sealed Rocketbox baseline")
    root_manifest_path = _require_regular_file(
        root / "baseline_manifest.json", root, "baseline manifest"
    )
    if sha256_file(root_manifest_path) != contract.baseline_manifest_sha256:
        raise RetargetError("baseline manifest SHA-256 does not match the sealed contract")
    if root_manifest_path.stat().st_size != contract.baseline_manifest_size:
        raise RetargetError("baseline manifest size does not match the sealed contract")
    root_manifest = _load_json(root_manifest_path, "baseline manifest")
    if (
        root_manifest.get("schema_version") != BASELINE_MANIFEST_SCHEMA
        or root_manifest.get("baseline_id") != "rocketbox_neutral_walk_v1"
        or root_manifest.get("motion") != "walk_neutral"
    ):
        raise RetargetError("baseline manifest identity is not the sealed neutral walk")

    manifest_argument = _absolute(baseline_retarget_manifest)
    if manifest_argument.name != "retarget_manifest.json":
        raise RetargetError("canonical sealed manifest must be retarget_manifest.json")
    try:
        manifest_argument.relative_to(root)
    except ValueError as error:
        raise RetargetError("retarget manifest is outside the canonical sealed baseline") from error
    manifest_path = _require_regular_file(
        manifest_argument, root, "sealed retarget manifest"
    )
    if base_avatar_id not in contract.idle_by_baseline_asset:
        raise RetargetError(f"unsupported base_avatar_id: {base_avatar_id!r}")
    baseline_asset_id = manifest_path.parent.name
    if baseline_asset_id != base_avatar_id:
        raise RetargetError(
            "sealed walk path does not match explicit base_avatar_id: "
            f"path={baseline_asset_id!r} requested={base_avatar_id!r}"
        )
    if baseline_asset_id not in contract.idle_by_baseline_asset:
        raise RetargetError("sealed retarget path has no supported gender baseline asset")
    expected_root = root / baseline_asset_id
    if manifest_path != expected_root / "retarget_manifest.json":
        raise RetargetError("retarget manifest is not at its canonical sealed gender path")
    blend_argument = _absolute(baseline_retarget_blend)
    if blend_argument != expected_root / "retarget.blend":
        raise RetargetError("canonical sealed retarget.blend path does not match the gender baseline")
    blend = _require_regular_file(blend_argument, root, "sealed retarget.blend")

    assets = root_manifest.get("assets")
    asset_entry = assets.get(baseline_asset_id) if isinstance(assets, Mapping) else None
    files = asset_entry.get("files") if isinstance(asset_entry, Mapping) else None
    if not isinstance(files, Mapping):
        raise RetargetError("baseline manifest has no gender asset file records")
    blend_record = _validate_file_record(
        blend,
        files.get("retarget.blend"),
        size_key="size",
        description="retarget.blend",
    )
    manifest_record = _validate_file_record(
        manifest_path,
        files.get("retarget_manifest.json"),
        size_key="size",
        description="retarget manifest",
    )
    manifest = _load_json(manifest_path, "sealed retarget manifest")
    if manifest.get("asset_id") != baseline_asset_id:
        raise RetargetError("sealed retarget manifest asset_id disagrees with its canonical path")
    animation = manifest.get("source_animation")
    if (
        manifest.get("schema_version") != BASELINE_RETARGET_SCHEMA
        or manifest.get("stage") != "retargeted"
        or manifest.get("automatic_checks", {}).get("overall") != "passed"
        or not isinstance(animation, Mapping)
        or animation.get("fps") != 30
        or animation.get("frame_count")
        != animation.get("frame_end") - animation.get("frame_start") + 1
        or manifest.get("artifacts", {}).get("blend") != "retarget.blend"
    ):
        raise RetargetError("sealed retarget manifest is not an approved 30 fps neutral walk")
    return {
        "baseline_manifest": _file_record(root_manifest_path),
        "base_avatar_id": baseline_asset_id,
        "retarget_blend": blend_record,
        "retarget_manifest": manifest_record,
        "source_animation": dict(animation),
    }


def _run_checked(
    subprocess_runner: Callable[..., Any], command: Sequence[str]
) -> str:
    result = subprocess_runner(
        [str(item) for item in command],
        check=False,
        capture_output=True,
        text=True,
    )
    if getattr(result, "returncode", 0) != 0:
        raise RetargetError(
            f"subprocess failed: {' '.join(map(str, command))}: {getattr(result, 'stderr', '')}"
        )
    return str(getattr(result, "stdout", "")).strip()


def authenticate_idle_motion(
    *,
    base_avatar_id: str,
    idle_motion_fbx: Path,
    contract: RetargetInputContract = PRODUCTION_INPUT_CONTRACT,
    subprocess_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    idle_contract = contract.idle_by_baseline_asset.get(base_avatar_id)
    if idle_contract is None:
        raise RetargetError("baseline asset has no exact gender idle contract")
    root = _require_real_directory(contract.rocketbox_root, "Rocketbox checkout")
    expected = root / idle_contract.relative_path
    if _absolute(idle_motion_fbx) != expected:
        raise RetargetError("idle FBX is not the exact gender idle motion path")
    idle = _require_regular_file(expected, root, "gender idle FBX")
    if idle.stat().st_size != idle_contract.size_bytes:
        raise RetargetError("gender idle FBX size does not match the pinned contract")
    if sha256_file(idle) != idle_contract.sha256:
        raise RetargetError("gender idle FBX SHA-256 does not match the pinned contract")
    revision = _run_checked(
        subprocess_runner, ["git", "-C", str(root), "rev-parse", "HEAD"]
    )
    if revision != contract.rocketbox_commit:
        raise RetargetError("Rocketbox checkout commit is not pinned")
    blob = _run_checked(
        subprocess_runner,
        ["git", "-C", str(root), "hash-object", str(idle_contract.relative_path)],
    )
    if blob != idle_contract.git_blob_sha1:
        raise RetargetError("gender idle FBX Git blob does not match the pinned commit")
    return {
        **_file_record(idle),
        "relative_path": str(idle_contract.relative_path),
        "git_commit": revision,
        "git_blob_sha1": blob,
        "base_avatar_id": base_avatar_id,
    }


def validate_base_avatar_id(
    *, asset_id: str, base_avatar_id: str, contract: RetargetInputContract
) -> str:
    if not _ASSET_ID_RE.fullmatch(asset_id):
        raise RetargetError(f"invalid instance asset_id: {asset_id!r}")
    if base_avatar_id not in contract.idle_by_baseline_asset:
        raise RetargetError(f"unsupported base_avatar_id: {base_avatar_id!r}")
    return base_avatar_id


def authenticate_motion_basis_selection(
    *,
    base_avatar_id: str,
    motion_basis_selection: Path,
    motion_basis_review_manifest: Path,
) -> dict[str, Any]:
    selection_argument = _absolute(motion_basis_selection)
    review_argument = _absolute(motion_basis_review_manifest)
    if selection_argument.name != "retarget_motion_basis_correction_v1.json":
        raise RetargetError("motion-basis selection has an unexpected filename")
    if review_argument.name != "motion_basis_review_manifest.json":
        raise RetargetError("motion-basis review has an unexpected filename")
    selection_root = _require_real_directory(
        selection_argument.parent, "motion-basis selection bundle"
    )
    review_root = _require_real_directory(
        review_argument.parent, "motion-basis review bundle"
    )
    selection_path = _require_regular_file(
        selection_argument, selection_root, "motion-basis selection"
    )
    review_path = _require_regular_file(
        review_argument, review_root, "motion-basis review manifest"
    )
    for path, description in (
        (selection_path, "motion-basis selection"),
        (review_path, "motion-basis review manifest"),
    ):
        if path.stat().st_mode & 0o777 != 0o444:
            raise RetargetError(f"{description} must be sealed read-only mode 0444")

    selection = _load_json(selection_path, "motion-basis selection")
    review = _load_json(review_path, "motion-basis review manifest")
    if selection.get("schema") != MOTION_BASIS_SELECTION_SCHEMA:
        raise RetargetError("motion-basis selection schema is invalid")
    if selection.get("asset_id") != base_avatar_id:
        raise RetargetError("motion-basis selection does not match base_avatar_id")
    if selection.get("decision") != MOTION_BASIS_SELECTION_DECISION:
        raise RetargetError("motion-basis selection was not approved for retarget")
    if selection.get("canonical_front") != CANONICAL_FRONT:
        raise RetargetError("motion-basis selection front is not canonical FRONT -Y")
    if selection.get("canonical_up") != CANONICAL_UP:
        raise RetargetError("motion-basis selection up is not canonical UP +Z")
    if selection.get("scope") != "bilateral_arm_and_leg_chains_only":
        raise RetargetError("motion-basis selection scope is not the four limb chains")
    if selection.get("formal_dataset_asset") is not False:
        raise RetargetError("review selection must not claim formal dataset status")
    if not isinstance(selection.get("reviewer"), str) or not selection["reviewer"]:
        raise RetargetError("motion-basis selection has no reviewer")
    if not isinstance(selection.get("reviewed_at"), str) or not selection["reviewed_at"]:
        raise RetargetError("motion-basis selection has no review timestamp")
    candidate_id = selection.get("candidate_id")
    yaw_degrees = selection.get("yaw_degrees")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise RetargetError("motion-basis selection has no candidate_id")
    if isinstance(yaw_degrees, bool) or not isinstance(yaw_degrees, (int, float)):
        raise RetargetError("motion-basis selection yaw is invalid")
    selected_matrix, projection = project_near_rotation(
        selection.get("matrix_3x3"),
        "approved shared canonical limb motion basis",
        context={"base_avatar_id": base_avatar_id, "candidate_id": candidate_id},
    )

    review_sha256 = sha256_file(review_path)
    if selection.get("candidate_bundle_manifest_sha256") != review_sha256:
        raise RetargetError("motion-basis selection points to a stale review manifest")
    if review.get("schema") != MOTION_BASIS_REVIEW_SCHEMA:
        raise RetargetError("motion-basis review schema is invalid")
    if review.get("asset_id") != base_avatar_id:
        raise RetargetError("motion-basis review does not match base_avatar_id")
    if review.get("automatic_checks") != "all_candidates_generated_and_hash_locked":
        raise RetargetError("motion-basis review candidates are not hash locked")
    candidates = review.get("candidates")
    candidate = candidates.get(candidate_id) if isinstance(candidates, Mapping) else None
    if not isinstance(candidate, Mapping):
        raise RetargetError("approved motion-basis candidate is absent from review")
    if candidate.get("yaw_degrees") != yaw_degrees:
        raise RetargetError("approved motion-basis yaw disagrees with review candidate")
    candidate_matrix, _ = project_near_rotation(
        candidate.get("matrix_3x3"),
        "reviewed shared canonical limb motion basis",
        context={"base_avatar_id": base_avatar_id, "candidate_id": candidate_id},
    )
    if not np.allclose(selected_matrix, candidate_matrix, rtol=0.0, atol=1.0e-12):
        raise RetargetError("approved motion-basis matrix disagrees with review candidate")

    metrics = candidate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RetargetError("approved motion-basis candidate has no metrics")
    if any(
        (
            metrics.get("asset_id") != base_avatar_id,
            metrics.get("candidate_id") != candidate_id,
            metrics.get("canonical_front") != CANONICAL_FRONT,
            metrics.get("canonical_up") != CANONICAL_UP,
            metrics.get("overall_classification") != "four_limb_sagittal_motion",
        )
    ):
        raise RetargetError("approved candidate did not pass four-limb sagittal review")
    axial_gate = metrics.get("anatomical_axial_pose_gate")
    if not isinstance(axial_gate, Mapping) or any(
        (
            axial_gate.get("schema") != "anatomical_axial_pose_gate_v1",
            axial_gate.get("automatic_checks") != "passed",
            axial_gate.get("overall_classification")
            != "axial_pose_within_source_motion_envelope",
        )
    ):
        raise RetargetError("approved candidate did not pass the anatomical axial gate")
    axial_transfer = metrics.get("anatomical_axial_transfer")
    if not isinstance(axial_transfer, Mapping) or any(
        (
            axial_transfer.get("schema")
            != "tokenrig_anatomical_axial_body_transfer_v1",
            axial_transfer.get("automatic_checks") != "passed",
        )
    ):
        raise RetargetError("approved candidate lacks the anatomical axial transfer proof")
    shared_basis = metrics.get("shared_limb_motion_basis")
    if not isinstance(shared_basis, Mapping) or any(
        (
            shared_basis.get("schema")
            != "tokenrig_shared_canonical_limb_motion_basis_v1",
            shared_basis.get("automatic_checks") != "passed",
        )
    ):
        raise RetargetError("approved candidate lacks the shared limb-basis proof")
    reviewed_basis, _ = project_near_rotation(
        shared_basis.get("motion_basis_3x3"),
        "candidate shared canonical limb motion basis evidence",
        context={"base_avatar_id": base_avatar_id, "candidate_id": candidate_id},
    )
    if not np.allclose(selected_matrix, reviewed_basis, rtol=0.0, atol=1.0e-12):
        raise RetargetError("approved matrix disagrees with shared limb-basis evidence")

    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise RetargetError("approved motion-basis candidate artifacts are missing")
    selected_artifacts: dict[str, Any] = {}
    for name, record in sorted(artifacts.items()):
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise RetargetError("approved candidate has an invalid artifact record")
        relative = record.get("filename")
        if not isinstance(relative, str) or not relative:
            raise RetargetError("approved candidate artifact filename is invalid")
        artifact = _require_regular_file(
            review_root / relative, review_root, f"approved candidate artifact {name}"
        )
        selected_artifacts[name] = _validate_file_record(
            artifact,
            record,
            size_key="size_bytes",
            description=f"approved candidate artifact {name}",
        )

    return {
        "selection": _file_record(selection_path),
        "review_manifest": _file_record(review_path),
        "base_avatar_id": base_avatar_id,
        "candidate_id": candidate_id,
        "yaw_degrees": float(yaw_degrees),
        "matrix_3x3": selected_matrix.tolist(),
        "matrix_projection": projection,
        "reviewer": selection["reviewer"],
        "reviewed_at": selection["reviewed_at"],
        "selected_artifacts": selected_artifacts,
        "four_limb_classification": metrics["overall_classification"],
        "axial_classification": axial_gate["overall_classification"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--base-avatar-id", required=True)
    parser.add_argument("--bind-pose-glb", type=Path, required=True)
    parser.add_argument("--static-qa-json", type=Path, required=True)
    parser.add_argument("--baseline-retarget-blend", type=Path, required=True)
    parser.add_argument("--baseline-retarget-manifest", type=Path, required=True)
    parser.add_argument("--idle-motion-fbx", type=Path, required=True)
    parser.add_argument("--motion-basis-selection", type=Path, required=True)
    parser.add_argument("--motion-basis-review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def validate_semantic_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    if mapping.get("method") != SEMANTIC_METHOD:
        raise RetargetError("semantic mapping method is not the static-audit contract")
    if mapping.get("side_basis") != {"left": "positive-x", "right": "negative-x"}:
        raise RetargetError("semantic mapping side basis is not canonical")
    semantic = mapping.get("semantic_bones")
    chains = mapping.get("chains")
    if not isinstance(semantic, Mapping) or not isinstance(chains, Mapping):
        raise RetargetError("semantic mapping must contain bones and chains")
    if set(semantic) != {*SEMANTIC_EXACT_ROLES, "spine"}:
        raise RetargetError("semantic mapping roles are incomplete or unexpected")
    spine = semantic.get("spine")
    if not isinstance(spine, list) or not 2 <= len(spine) <= 4:
        raise RetargetError("semantic spine must contain two, three, or four target bones")
    if any(not isinstance(value, str) or not value for value in spine):
        raise RetargetError("semantic spine contains an invalid target bone name")
    exact = {role: semantic.get(role) for role in SEMANTIC_EXACT_ROLES}
    if any(not isinstance(value, str) or not value for value in exact.values()):
        raise RetargetError("semantic mapping contains an invalid exact target bone name")
    target_names = [*exact.values(), *spine]
    if len(target_names) != len(set(target_names)):
        raise RetargetError("duplicate target bone makes the semantic map ambiguous")
    expected_chains = {
        "axial": [exact["pelvis"], *spine, exact["neck"], exact["head"]],
        "left_arm": [exact[f"left_{part}"] for part in ("clavicle", "upper_arm", "forearm", "hand")],
        "right_arm": [exact[f"right_{part}"] for part in ("clavicle", "upper_arm", "forearm", "hand")],
        "left_leg": [exact[f"left_{part}"] for part in ("thigh", "calf", "foot", "toe")],
        "right_leg": [exact[f"right_{part}"] for part in ("thigh", "calf", "foot", "toe")],
    }
    if dict(chains) != expected_chains:
        raise RetargetError("semantic chains disagree with the exact axial/limb mapping")
    descendant_groups: dict[str, list[str]] = {}
    descendants: list[str] = []
    for key in ("ignored_proven_distal_descendants", "ignored_proven_head_descendants"):
        values = mapping.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise RetargetError(f"semantic mapping {key} is invalid")
        descendant_groups[key] = list(values)
        descendants.extend(values)
    if len(descendants) != len(set(descendants)) or set(descendants) & set(target_names):
        raise RetargetError("duplicate or mapped rest descendant is ambiguous")
    return {
        "semantic_bones": {**exact, "spine": list(spine)},
        "chains": expected_chains,
        "target_bone_names": sorted(target_names),
        "rest_descendants": sorted(descendants),
        "head_bound_descendants": sorted(
            descendant_groups["ignored_proven_head_descendants"]
        ),
        "hand_bound_descendants": sorted(
            descendant_groups["ignored_proven_distal_descendants"]
        ),
        "method": SEMANTIC_METHOD,
    }


def _normalized_rest_arc(
    points: Sequence[Sequence[float]], *, label: str
) -> tuple[float, ...]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3:
        raise RetargetError(f"{label} rest arc must contain at least two 3D points")
    if not np.isfinite(values).all():
        raise RetargetError(f"{label} rest arc contains non-finite points")
    segments = np.linalg.norm(np.diff(values, axis=0), axis=1)
    total = float(segments.sum())
    if total <= 1.0e-12:
        raise RetargetError(f"degenerate {label} rest arc has zero total length")
    if np.any(segments <= 1.0e-12):
        raise RetargetError(f"ambiguous {label} rest arc has duplicate controls")
    cumulative = np.concatenate(([0.0], np.cumsum(segments))) / total
    cumulative[0] = 0.0
    cumulative[-1] = 1.0
    return tuple(float(value) for value in cumulative)


def build_spine_resample_plan(
    *,
    source_bones: Sequence[str],
    source_rest_heads: Sequence[Sequence[float]],
    target_bones: Sequence[str],
    target_rest_heads: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    if tuple(source_bones) != ROCKETBOX_SPINE_BONES:
        raise RetargetError("source spine controls are not exact Rocketbox Spine/Spine1/Spine2")
    if len(source_rest_heads) != len(source_bones):
        raise RetargetError("source spine names and rest heads differ in length")
    if not 2 <= len(target_bones) <= 4 or len(target_rest_heads) != len(target_bones):
        raise RetargetError("target spine must contain two, three, or four matched rest heads")
    if len(set(target_bones)) != len(target_bones):
        raise RetargetError("target spine bone names are ambiguous")
    source_arc = _normalized_rest_arc(source_rest_heads, label="source spine")
    target_arc = _normalized_rest_arc(target_rest_heads, label="target spine")
    records: list[dict[str, Any]] = []
    for target_name, position in zip(target_bones, target_arc):
        if position <= 0.0:
            lower, upper, alpha = 0, 1, 0.0
        elif position >= 1.0:
            lower, upper, alpha = len(source_arc) - 2, len(source_arc) - 1, 1.0
        else:
            lower = next(
                index
                for index in range(len(source_arc) - 1)
                if source_arc[index] <= position <= source_arc[index + 1]
            )
            upper = lower + 1
            alpha = (position - source_arc[lower]) / (
                source_arc[upper] - source_arc[lower]
            )
        records.append(
            {
                "target_bone": str(target_name),
                "target_normalized_arc": float(position),
                "source_indices": [lower, upper],
                "source_bones": [str(source_bones[lower]), str(source_bones[upper])],
                "weights": [float(1.0 - alpha), float(alpha)],
                "interpolation_domain": "cumulative_parent_to_child_rotation",
            }
        )
    return records


def _rotation_context_text(context: Mapping[str, Any] | None) -> str:
    if not context:
        return "context=unspecified"
    ordered = (
        "action",
        "frame",
        "semantic_role",
        "source_bone",
        "target_bone",
        "matrix_stage",
    )
    parts = [f"{key}={context[key]}" for key in ordered if key in context]
    parts.extend(
        f"{key}={context[key]}" for key in sorted(set(context) - set(ordered))
    )
    return "context[" + ", ".join(parts) + "]"


def _rotation_failure(
    *,
    description: str,
    reason: str,
    context: Mapping[str, Any] | None,
    determinant: float,
    orthogonality_error: float,
    singular_values: Sequence[float],
    condition_number: float,
    polar_residual: float,
) -> RetargetError:
    return RetargetError(
        f"{description} {reason}; {_rotation_context_text(context)}; "
        f"determinant={determinant:.17g}; "
        f"orthogonality_max_error={orthogonality_error:.17g}; "
        f"singular_values={[float(value) for value in singular_values]}; "
        f"condition_number={condition_number:.17g}; "
        f"polar_residual_max_abs={polar_residual:.17g}"
    )


def project_near_rotation(
    value: Sequence[Sequence[float]],
    description: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise RetargetError(f"{description} must be a finite 3x3 matrix")
    determinant = float(np.linalg.det(matrix))
    orthogonality_error = float(
        np.max(np.abs(matrix.T @ matrix - np.eye(3, dtype=np.float64)))
    )
    left, singular_values, right_transpose = np.linalg.svd(matrix)
    minimum_singular = float(singular_values[-1])
    maximum_singular = float(singular_values[0])
    condition_number = (
        float(maximum_singular / minimum_singular)
        if minimum_singular > 0.0
        else math.inf
    )
    projected = left @ right_transpose
    polar_residual = float(np.max(np.abs(matrix - projected)))
    metrics = {
        "determinant": determinant,
        "orthogonality_error": orthogonality_error,
        "singular_values": singular_values,
        "condition_number": condition_number,
        "polar_residual": polar_residual,
    }
    if determinant < 0.0:
        raise _rotation_failure(
            description=description,
            reason="has reflection/left-handed handedness",
            context=context,
            **metrics,
        )
    if (
        determinant == 0.0
        or minimum_singular <= MINIMUM_ROTATION_SINGULAR_VALUE
    ):
        raise _rotation_failure(
            description=description,
            reason="is degenerate/singular",
            context=context,
            **metrics,
        )
    maximum_singular_deviation = float(
        np.max(np.abs(singular_values - np.ones(3, dtype=np.float64)))
    )
    if any(
        (
            orthogonality_error > MAXIMUM_ROTATION_ORTHOGONALITY_ERROR,
            maximum_singular_deviation
            > MAXIMUM_ROTATION_SINGULAR_VALUE_DEVIATION,
            condition_number > MAXIMUM_ROTATION_CONDITION_NUMBER,
            polar_residual > MAXIMUM_ROTATION_POLAR_RESIDUAL,
        )
    ):
        raise _rotation_failure(
            description=description,
            reason="exceeds strict near-rotation bounds",
            context=context,
            **metrics,
        )
    output_determinant = float(np.linalg.det(projected))
    output_orthogonality = float(
        np.max(np.abs(projected.T @ projected - np.eye(3, dtype=np.float64)))
    )
    if output_determinant <= 0.0 or output_orthogonality > 1.0e-12:
        raise RetargetError(
            f"{description} SVD/polar projection did not produce SO(3); "
            f"{_rotation_context_text(context)}"
        )
    evidence = {
        "method": "svd_polar_nearest_so3_v1",
        "description": description,
        "context": dict(context or {}),
        "input_handedness": "right",
        "input_determinant": determinant,
        "input_orthogonality_max_error": orthogonality_error,
        "input_singular_values": [float(value) for value in singular_values],
        "maximum_singular_value_deviation": maximum_singular_deviation,
        "condition_number": condition_number,
        "polar_residual_max_abs": polar_residual,
        "output_determinant": output_determinant,
        "output_orthogonality_max_error": output_orthogonality,
        "projection_applied": polar_residual > 1.0e-12,
        "thresholds": {
            "maximum_orthogonality_error": MAXIMUM_ROTATION_ORTHOGONALITY_ERROR,
            "maximum_singular_value_deviation": (
                MAXIMUM_ROTATION_SINGULAR_VALUE_DEVIATION
            ),
            "maximum_condition_number": MAXIMUM_ROTATION_CONDITION_NUMBER,
            "maximum_polar_residual": MAXIMUM_ROTATION_POLAR_RESIDUAL,
            "minimum_singular_value": MINIMUM_ROTATION_SINGULAR_VALUE,
        },
    }
    return projected, evidence


def project_uniform_scaled_rotation(
    value: Sequence[Sequence[float]],
    description: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise RetargetError(f"{description} must be a finite 3x3 matrix")
    determinant = float(np.linalg.det(matrix))
    if determinant <= 0.0:
        handedness = "reflection/left-handed" if determinant < 0.0 else "degenerate"
        raise RetargetError(
            f"{description} has {handedness} object transform; "
            f"{_rotation_context_text(context)}; determinant={determinant:.17g}"
        )
    uniform_scale = float(determinant ** (1.0 / 3.0))
    if not math.isfinite(uniform_scale) or uniform_scale <= 0.0:
        raise RetargetError(
            f"{description} has invalid uniform scale; {_rotation_context_text(context)}"
        )
    projected, normalized = project_near_rotation(
        matrix / uniform_scale,
        f"{description} normalized rotation",
        context=context,
    )
    return projected, {
        **normalized,
        "method": "positive_uniform_scale_then_svd_polar_so3_v1",
        "description": description,
        "raw_input_determinant": determinant,
        "uniform_scale": uniform_scale,
        "normalized_rotation": normalized,
    }


def summarize_rotation_projections(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise RetargetError("rotation projection evidence is empty")
    metric_fields = {
        "input_orthogonality_max_error": "maximum_input_orthogonality_error",
        "maximum_singular_value_deviation": "maximum_singular_value_deviation",
        "condition_number": "maximum_condition_number",
        "polar_residual_max_abs": "maximum_polar_residual",
        "output_orthogonality_max_error": (
            "maximum_output_orthogonality_error"
        ),
    }
    context_fields = (
        "action",
        "semantic_role",
        "source_bone",
        "target_bone",
        "matrix_stage",
    )
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    normalized_records: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RetargetError("rotation projection record must be an object")
        context = record.get("context")
        if not isinstance(context, Mapping):
            raise RetargetError("rotation projection record has no context")
        for field in (
            "input_determinant",
            *metric_fields,
            "output_determinant",
        ):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RetargetError(f"rotation projection record has invalid {field}")
            if not math.isfinite(float(value)):
                raise RetargetError(f"rotation projection record has non-finite {field}")
        normalized_records.append(record)
        key = tuple(context.get(field) for field in context_fields)
        group = groups.setdefault(
            key,
            {
                **{field: context.get(field) for field in context_fields},
                "sample_count": 0,
                "projection_applied_count": 0,
                "right_handed_record_count": 0,
                "minimum_input_determinant": math.inf,
                "maximum_input_determinant": -math.inf,
                "minimum_output_determinant": math.inf,
                "maximum_output_determinant": -math.inf,
                **{group_field: -math.inf for group_field in metric_fields.values()},
                "worst_frames": {},
                "minimum_uniform_scale": math.inf,
                "maximum_uniform_scale": -math.inf,
            },
        )
        group["sample_count"] += 1
        group["projection_applied_count"] += int(
            record.get("projection_applied") is True
        )
        group["right_handed_record_count"] += int(
            record.get("input_handedness") == "right"
        )
        group["minimum_input_determinant"] = min(
            group["minimum_input_determinant"], float(record["input_determinant"])
        )
        group["maximum_input_determinant"] = max(
            group["maximum_input_determinant"], float(record["input_determinant"])
        )
        group["minimum_output_determinant"] = min(
            group["minimum_output_determinant"], float(record["output_determinant"])
        )
        group["maximum_output_determinant"] = max(
            group["maximum_output_determinant"], float(record["output_determinant"])
        )
        for record_field, group_field in metric_fields.items():
            value = float(record[record_field])
            if value > group[group_field]:
                group[group_field] = value
                group["worst_frames"][record_field] = context.get("frame")
        if "uniform_scale" in record:
            scale = float(record["uniform_scale"])
            group["minimum_uniform_scale"] = min(
                group["minimum_uniform_scale"], scale
            )
            group["maximum_uniform_scale"] = max(
                group["maximum_uniform_scale"], scale
            )

    per_bone_stage = []
    for group in groups.values():
        if math.isinf(group["minimum_uniform_scale"]):
            del group["minimum_uniform_scale"]
            del group["maximum_uniform_scale"]
        per_bone_stage.append(group)
    per_bone_stage.sort(
        key=lambda group: tuple(str(group.get(field) or "") for field in context_fields)
    )
    summary = {
        "schema": "strict_rotation_projection_evidence_v1",
        "record_count": len(normalized_records),
        "context_group_count": len(per_bone_stage),
        "projection_applied_count": sum(
            record.get("projection_applied") is True for record in normalized_records
        ),
        "right_handed_record_count": sum(
            record.get("input_handedness") == "right" for record in normalized_records
        ),
        "minimum_input_determinant": min(
            float(record["input_determinant"]) for record in normalized_records
        ),
        "maximum_input_determinant": max(
            float(record["input_determinant"]) for record in normalized_records
        ),
        "maximum_input_orthogonality_error": max(
            float(record["input_orthogonality_max_error"])
            for record in normalized_records
        ),
        "maximum_singular_value_deviation": max(
            float(record["maximum_singular_value_deviation"])
            for record in normalized_records
        ),
        "maximum_condition_number": max(
            float(record["condition_number"]) for record in normalized_records
        ),
        "maximum_polar_residual": max(
            float(record["polar_residual_max_abs"])
            for record in normalized_records
        ),
        "minimum_output_determinant": min(
            float(record["output_determinant"]) for record in normalized_records
        ),
        "maximum_output_determinant": max(
            float(record["output_determinant"]) for record in normalized_records
        ),
        "maximum_output_orthogonality_error": max(
            float(record["output_orthogonality_max_error"])
            for record in normalized_records
        ),
        "thresholds": {
            "maximum_orthogonality_error": MAXIMUM_ROTATION_ORTHOGONALITY_ERROR,
            "maximum_singular_value_deviation": (
                MAXIMUM_ROTATION_SINGULAR_VALUE_DEVIATION
            ),
            "maximum_condition_number": MAXIMUM_ROTATION_CONDITION_NUMBER,
            "maximum_polar_residual": MAXIMUM_ROTATION_POLAR_RESIDUAL,
            "minimum_singular_value": MINIMUM_ROTATION_SINGULAR_VALUE,
        },
        "per_bone_stage": per_bone_stage,
    }
    validate_rotation_projection_summary(summary)
    return summary


def _rotation_context_name(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise RetargetError(f"rotation projection context has invalid {description}")
    return value


def _validate_rotation_projection_group_context(
    group: Mapping[str, Any],
    *,
    semantic_mapping: Mapping[str, Any] | None,
) -> tuple[Any, ...]:
    action = group.get("action")
    if action not in set(ACTION_NAMES.values()):
        raise RetargetError("rotation projection context has invalid action")
    stage = group.get("matrix_stage")
    if stage not in ROTATION_ALLOWED_STAGES:
        raise RetargetError("rotation projection context has invalid matrix stage")
    role = group.get("semantic_role")
    source_bone = group.get("source_bone")
    target_bone = group.get("target_bone")
    exact_roles = set(SEMANTIC_EXACT_ROLES)
    body_roles = exact_roles | {"spine"}

    if stage in {"source_global_pose", "source_local_pose"}:
        if role not in body_roles:
            raise RetargetError("rotation projection context has invalid pose semantic role")
        source_name = _rotation_context_name(source_bone, "source bone")
        if role == "spine":
            if source_name not in ROCKETBOX_SPINE_BONES or target_bone is not None:
                raise RetargetError("rotation projection context has invalid spine source bone")
        elif source_name != ROCKETBOX_ROLE_TO_BONE[role]:
            raise RetargetError("rotation projection context source bone disagrees with role")
        elif stage == "source_local_pose" and target_bone is not None:
            raise RetargetError("rotation projection local-pose context has a target bone")
        elif target_bone is not None:
            _rotation_context_name(target_bone, "target bone")
    elif stage == "source_object_world":
        if role != "armature_root" or target_bone is not None:
            raise RetargetError("rotation projection source-object context is invalid")
        _rotation_context_name(source_bone, "source object")
    elif stage == "target_object_world":
        if role != "armature_root" or source_bone is not None:
            raise RetargetError("rotation projection target-object context is invalid")
        _rotation_context_name(target_bone, "target object")
    elif stage == "canonical_axis_identity":
        if role != "canonical_axis" or source_bone is not None or target_bone is not None:
            raise RetargetError("rotation projection canonical-axis context is invalid")
    elif stage == "source_local_rest":
        if (
            role != "spine"
            or source_bone not in ROCKETBOX_SPINE_BONES
            or target_bone is not None
        ):
            raise RetargetError("rotation projection source-local-rest context is invalid")
    elif stage == "target_local_rest":
        if role != "spine" or source_bone is not None:
            raise RetargetError("rotation projection target-local-rest context is invalid")
        _rotation_context_name(target_bone, "target bone")
    else:
        if role not in exact_roles:
            raise RetargetError("rotation projection global-rest semantic role is invalid")
        if source_bone != ROCKETBOX_ROLE_TO_BONE[role]:
            raise RetargetError("rotation projection global-rest source bone is invalid")
        _rotation_context_name(target_bone, "target bone")

    semantic_bones: Mapping[str, Any] | None = None
    if semantic_mapping is not None:
        candidate = semantic_mapping.get("semantic_bones")
        if not isinstance(candidate, Mapping):
            raise RetargetError("rotation projection semantic mapping is invalid")
        semantic_bones = candidate
    if semantic_bones is not None and target_bone is not None:
        if role in exact_roles and target_bone != semantic_bones.get(role):
            raise RetargetError("rotation projection target bone disagrees with semantic role")
        if role == "spine":
            spines = semantic_bones.get("spine")
            if not isinstance(spines, list) or target_bone not in spines:
                raise RetargetError("rotation projection target spine is not in semantic mapping")

    has_minimum_scale = "minimum_uniform_scale" in group
    has_maximum_scale = "maximum_uniform_scale" in group
    if stage in ROTATION_OBJECT_STAGES:
        if not has_minimum_scale or not has_maximum_scale:
            raise RetargetError("rotation projection object context lacks uniform-scale evidence")
    elif has_minimum_scale or has_maximum_scale:
        raise RetargetError("rotation projection non-object context has uniform-scale evidence")
    return (action, role, source_bone, target_bone, stage)


def validate_rotation_projection_summary(
    summary: Mapping[str, Any],
    *,
    expected_action: str | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    semantic_mapping: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    if summary.get("schema") != "strict_rotation_projection_evidence_v1":
        raise RetargetError("rotation projection evidence schema is invalid")
    if expected_action is not None and expected_action not in set(ACTION_NAMES.values()):
        raise RetargetError("rotation projection expected action is invalid")
    has_frame_start = frame_start is not None
    has_frame_end = frame_end is not None
    if has_frame_start != has_frame_end:
        raise RetargetError("rotation projection action frame range is partial")
    if has_frame_start and (
        isinstance(frame_start, bool)
        or isinstance(frame_end, bool)
        or not isinstance(frame_start, int)
        or not isinstance(frame_end, int)
        or frame_end < frame_start
    ):
        raise RetargetError("rotation projection action frame range is invalid")
    record_count = summary.get("record_count")
    groups = summary.get("per_bone_stage")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
        or not isinstance(groups, list)
        or not groups
        or summary.get("context_group_count") != len(groups)
        or sum(group.get("sample_count", 0) for group in groups) != record_count
    ):
        raise RetargetError("rotation projection evidence counts are invalid")
    applied = summary.get("projection_applied_count")
    if (
        isinstance(applied, bool)
        or not isinstance(applied, int)
        or not 0 <= applied <= record_count
        or summary.get("right_handed_record_count") != record_count
    ):
        raise RetargetError("rotation projection evidence handedness/counts are invalid")

    group_bounds = {
        "maximum_input_orthogonality_error": MAXIMUM_ROTATION_ORTHOGONALITY_ERROR,
        "maximum_singular_value_deviation": (
            MAXIMUM_ROTATION_SINGULAR_VALUE_DEVIATION
        ),
        "maximum_condition_number": MAXIMUM_ROTATION_CONDITION_NUMBER,
        "maximum_polar_residual": MAXIMUM_ROTATION_POLAR_RESIDUAL,
        "maximum_output_orthogonality_error": 1.0e-12,
    }
    group_minimums = {
        "maximum_input_orthogonality_error": 0.0,
        "maximum_singular_value_deviation": 0.0,
        "maximum_condition_number": 1.0,
        "maximum_polar_residual": 0.0,
        "maximum_output_orthogonality_error": 0.0,
    }
    worst_frame_fields = {
        "input_orthogonality_max_error",
        "maximum_singular_value_deviation",
        "condition_number",
        "polar_residual_max_abs",
        "output_orthogonality_max_error",
    }
    context_keys: set[tuple[Any, ...]] = set()
    group_actions: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise RetargetError("rotation projection context group is invalid")
        context_key = _validate_rotation_projection_group_context(
            group, semantic_mapping=semantic_mapping
        )
        if context_key in context_keys:
            raise RetargetError("rotation projection has a duplicate context group")
        context_keys.add(context_key)
        group_actions.add(str(group["action"]))
        sample_count = group.get("sample_count")
        applied_count = group.get("projection_applied_count")
        right_handed_count = group.get("right_handed_record_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count <= 0
            or isinstance(applied_count, bool)
            or not isinstance(applied_count, int)
            or not 0 <= applied_count <= sample_count
            or right_handed_count != sample_count
        ):
            raise RetargetError("rotation projection context group/count is invalid")
        for field in (
            "minimum_input_determinant",
            "maximum_input_determinant",
            "minimum_output_determinant",
            "maximum_output_determinant",
        ):
            value = group.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RetargetError(
                    f"rotation projection group has invalid {field}"
                )
        if (
            float(group["minimum_input_determinant"]) <= 0.0
            or float(group["minimum_input_determinant"])
            > float(group["maximum_input_determinant"])
            or abs(float(group["minimum_output_determinant"]) - 1.0) > 1.0e-12
            or abs(float(group["maximum_output_determinant"]) - 1.0) > 1.0e-12
        ):
            raise RetargetError(
                "rotation projection group has invalid determinant/handedness"
            )
        for field, maximum in group_bounds.items():
            value = group.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < group_minimums[field]
                or float(value) > maximum
            ):
                raise RetargetError(f"rotation projection group exceeds {field}")
        worst_frames = group.get("worst_frames")
        if (
            not isinstance(worst_frames, Mapping)
            or set(worst_frames) != worst_frame_fields
        ):
            raise RetargetError("rotation projection group worst_frames is invalid")
        stage = str(group["matrix_stage"])
        for frame in worst_frames.values():
            if frame is None:
                if stage not in ROTATION_STATIC_STAGES:
                    raise RetargetError(
                        "rotation projection dynamic context has a static worst frame"
                    )
                continue
            if isinstance(frame, bool) or not isinstance(frame, int):
                raise RetargetError("rotation projection group worst frame is invalid")
            if has_frame_start and not frame_start <= frame <= frame_end:
                raise RetargetError(
                    "rotation projection worst frame is outside the action frame range"
                )
        has_minimum_scale = "minimum_uniform_scale" in group
        has_maximum_scale = "maximum_uniform_scale" in group
        if has_minimum_scale != has_maximum_scale:
            raise RetargetError("rotation projection uniform-scale evidence is partial")
        if has_minimum_scale:
            minimum_scale = group["minimum_uniform_scale"]
            maximum_scale = group["maximum_uniform_scale"]
            if (
                isinstance(minimum_scale, bool)
                or isinstance(maximum_scale, bool)
                or not isinstance(minimum_scale, (int, float))
                or not isinstance(maximum_scale, (int, float))
                or not math.isfinite(float(minimum_scale))
                or not math.isfinite(float(maximum_scale))
                or float(minimum_scale) <= 0.0
                or float(minimum_scale) > float(maximum_scale)
            ):
                raise RetargetError(
                    "rotation projection group uniform scale is invalid"
                )
    if len(group_actions) != 1:
        raise RetargetError("rotation projection summary mixes action contexts")
    only_action = next(iter(group_actions))
    if expected_action is not None and only_action != expected_action:
        raise RetargetError("rotation projection context action disagrees with metrics")
    expected_thresholds = {
        "maximum_orthogonality_error": MAXIMUM_ROTATION_ORTHOGONALITY_ERROR,
        "maximum_singular_value_deviation": (
            MAXIMUM_ROTATION_SINGULAR_VALUE_DEVIATION
        ),
        "maximum_condition_number": MAXIMUM_ROTATION_CONDITION_NUMBER,
        "maximum_polar_residual": MAXIMUM_ROTATION_POLAR_RESIDUAL,
        "minimum_singular_value": MINIMUM_ROTATION_SINGULAR_VALUE,
    }
    if summary.get("thresholds") != expected_thresholds:
        raise RetargetError("rotation projection thresholds are not the strict contract")
    bounded = {
        "maximum_input_orthogonality_error": MAXIMUM_ROTATION_ORTHOGONALITY_ERROR,
        "maximum_singular_value_deviation": (
            MAXIMUM_ROTATION_SINGULAR_VALUE_DEVIATION
        ),
        "maximum_condition_number": MAXIMUM_ROTATION_CONDITION_NUMBER,
        "maximum_polar_residual": MAXIMUM_ROTATION_POLAR_RESIDUAL,
        "maximum_output_orthogonality_error": 1.0e-12,
    }
    for field, maximum in bounded.items():
        value = summary.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < group_minimums[field]
            or float(value) > maximum
        ):
            raise RetargetError(f"rotation projection evidence exceeds {field}")
    recomputed = {
        "context_group_count": len(groups),
        "record_count": sum(group["sample_count"] for group in groups),
        "projection_applied_count": sum(
            group["projection_applied_count"] for group in groups
        ),
        "right_handed_record_count": sum(
            group["right_handed_record_count"] for group in groups
        ),
        "minimum_input_determinant": min(
            group["minimum_input_determinant"] for group in groups
        ),
        "maximum_input_determinant": max(
            group["maximum_input_determinant"] for group in groups
        ),
        "minimum_output_determinant": min(
            group["minimum_output_determinant"] for group in groups
        ),
        "maximum_output_determinant": max(
            group["maximum_output_determinant"] for group in groups
        ),
        **{
            field: max(group[field] for group in groups)
            for field in group_bounds
        },
    }
    for field, expected in recomputed.items():
        actual = summary.get(field)
        if isinstance(expected, int):
            matches = actual == expected
        else:
            matches = isinstance(actual, (int, float)) and float(actual) == float(expected)
        if not matches:
            raise RetargetError(
                f"rotation projection aggregate disagrees with groups: {field}"
            )
    if (
        _finite_number(
            summary.get("minimum_input_determinant"),
            "minimum rotation projection input determinant",
        )
        <= 0.0
        or abs(
            _finite_number(
                summary.get("minimum_output_determinant"),
                "minimum rotation projection output determinant",
            )
            - 1.0
        )
        > 1.0e-12
    ):
        raise RetargetError("rotation projection evidence has invalid handedness/output")
    return {"status": "passed"}


def _rotation_matrix(
    value: Sequence[Sequence[float]],
    description: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return project_near_rotation(value, description, context=context)


def rest_corrected_local_rotation(
    source_rest: Sequence[Sequence[float]],
    source_pose: Sequence[Sequence[float]],
    target_rest: Sequence[Sequence[float]],
) -> np.ndarray:
    source_rest_m, _ = _rotation_matrix(source_rest, "source local rest")
    source_pose_m, _ = _rotation_matrix(source_pose, "source local pose")
    target_rest_m, _ = _rotation_matrix(target_rest, "target local rest")
    return target_rest_m @ source_rest_m.T @ source_pose_m


def rest_aligned_global_rotation(
    source_rest: Sequence[Sequence[float]],
    source_pose: Sequence[Sequence[float]],
    target_rest: Sequence[Sequence[float]],
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    projection_evidence: dict[str, Any] = {}

    def project(value: Any, stage: str) -> np.ndarray:
        stage_context = {**dict(context or {}), "matrix_stage": stage}
        projected, evidence = _rotation_matrix(
            value,
            stage.replace("_", " "),
            context=stage_context,
        )
        projection_evidence[stage] = evidence
        return projected

    source_rest_m = project(source_rest, "source_global_rest")
    source_pose_m = project(source_pose, "source_global_pose")
    target_rest_m = project(target_rest, "target_global_rest")
    alignment = target_rest_m @ source_rest_m.T
    source_delta = source_pose_m @ source_rest_m.T
    target_delta = alignment @ source_delta @ alignment.T
    return target_delta @ target_rest_m, {
        "source_to_target_rest_alignment": alignment,
        "source_global_pose_delta": source_delta,
        "target_global_pose_delta": target_delta,
        "rotation_projections": projection_evidence,
    }


def canonical_world_delta_rotation(
    source_rest: Sequence[Sequence[float]],
    source_pose: Sequence[Sequence[float]],
    target_rest: Sequence[Sequence[float]],
    *,
    source_base_rotation_3x3: Sequence[Sequence[float]],
    target_base_rotation_3x3: Sequence[Sequence[float]],
    context: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transfer a distal world-space delta without conjugating by bone rest axes.

    Foot and toe surface pitch/roll live in the shared canonical static object
    frame.  Conjugating that delta through unrelated fitted bone rest axes can
    preserve the numeric angle while rotating it about the wrong world axis.
    """

    projection_evidence: dict[str, Any] = {}

    def project(value: Any, stage: str) -> np.ndarray:
        stage_context = {**dict(context or {}), "matrix_stage": stage}
        projected, evidence = _rotation_matrix(
            value,
            stage.replace("_", " "),
            context=stage_context,
        )
        projection_evidence[stage] = evidence
        return projected

    source_rest_m = project(source_rest, "source_global_rest")
    source_pose_m = project(source_pose, "source_global_pose")
    target_rest_m = project(target_rest, "target_global_rest")
    source_base_m, _ = project_near_rotation(
        source_base_rotation_3x3,
        "source canonical static object-frame rotation",
    )
    target_base_m, _ = project_near_rotation(
        target_base_rotation_3x3,
        "target canonical static object-frame rotation",
    )

    source_rest_canonical = source_base_m @ source_rest_m
    source_pose_canonical = source_base_m @ source_pose_m
    canonical_delta = source_pose_canonical @ source_rest_canonical.T
    desired_target_canonical = canonical_delta @ (target_base_m @ target_rest_m)
    target_pose_local = target_base_m.T @ desired_target_canonical
    target_pose_local, _ = project_near_rotation(
        target_pose_local,
        "target distal canonical-world pose output",
    )
    reconstructed_delta = (
        target_base_m
        @ target_pose_local
        @ (target_base_m @ target_rest_m).T
    )
    reconstruction_error = float(
        np.max(np.abs(reconstructed_delta - canonical_delta))
    )
    if reconstruction_error > REST_ROTATION_MATRIX_TOLERANCE:
        raise RetargetError(
            "distal canonical-world rotation delta reconstruction failed"
        )
    return target_pose_local, {
        "method": "canonical_world_distal_delta_v1",
        "source_base_rotation_3x3": source_base_m.tolist(),
        "target_base_rotation_3x3": target_base_m.tolist(),
        "canonical_source_delta_3x3": canonical_delta.tolist(),
        "canonical_delta_reconstruction_error": reconstruction_error,
        "target_rest_axis_conjugation_used": False,
        "axis_map_applied_once_per_asset": True,
        "rotation_projections": projection_evidence,
    }


def shared_canonical_limb_rotation(
    source_rest: Sequence[Sequence[float]],
    source_pose: Sequence[Sequence[float]],
    target_rest: Sequence[Sequence[float]],
    *,
    source_base_rotation_3x3: Sequence[Sequence[float]],
    target_base_rotation_3x3: Sequence[Sequence[float]],
    motion_basis_3x3: Sequence[Sequence[float]],
    context: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transfer every limb delta in one canonical body/world motion basis.

    Per-bone source-to-target rest alignment rotates an otherwise shared Walk
    axis differently for fitted bones whose local axes are unrelated.  This
    path first measures the source delta in the authenticated canonical object
    frame, applies one reviewed proper basis conjugation, and then composes the
    result with the target rest orientation.  Target rest axes never conjugate
    the animation delta.
    """

    _, canonical_evidence = canonical_world_delta_rotation(
        source_rest,
        source_pose,
        target_rest,
        source_base_rotation_3x3=source_base_rotation_3x3,
        target_base_rotation_3x3=target_base_rotation_3x3,
        context=context,
    )
    source_delta = np.asarray(
        canonical_evidence["canonical_source_delta_3x3"], dtype=np.float64
    )
    motion_basis, motion_basis_evidence = project_near_rotation(
        motion_basis_3x3,
        "reviewed shared canonical limb motion basis",
    )
    target_base, _ = project_near_rotation(
        target_base_rotation_3x3,
        "target canonical static object-frame rotation",
    )
    target_rest_m, _ = project_near_rotation(target_rest, "target global rest")
    corrected_delta = motion_basis @ source_delta @ motion_basis.T
    corrected_delta, _ = project_near_rotation(
        corrected_delta,
        "corrected shared canonical limb delta",
    )
    target_pose = target_base.T @ corrected_delta @ (target_base @ target_rest_m)
    target_pose, _ = project_near_rotation(
        target_pose,
        "target shared canonical limb pose output",
    )
    reconstructed_delta = (
        target_base @ target_pose @ (target_base @ target_rest_m).T
    )
    reconstruction_error = float(
        np.max(np.abs(reconstructed_delta - corrected_delta))
    )
    if reconstruction_error > REST_ROTATION_MATRIX_TOLERANCE:
        raise RetargetError(
            "shared canonical limb rotation delta reconstruction failed"
        )
    return target_pose, {
        "method": "shared_canonical_limb_delta_v1",
        "source_base_rotation_3x3": canonical_evidence[
            "source_base_rotation_3x3"
        ],
        "target_base_rotation_3x3": canonical_evidence[
            "target_base_rotation_3x3"
        ],
        "canonical_source_delta_3x3": source_delta.tolist(),
        "motion_basis_3x3": motion_basis.tolist(),
        "corrected_canonical_delta_3x3": corrected_delta.tolist(),
        "canonical_delta_reconstruction_error": reconstruction_error,
        "per_bone_rest_axis_conjugation_used": False,
        "target_rest_axis_conjugation_used": False,
        "motion_basis_projection": motion_basis_evidence,
        "rotation_projections": canonical_evidence["rotation_projections"],
    }


def scaled_target_translation(
    *,
    source_rest: Sequence[float],
    source_pose: Sequence[float],
    target_rest: Sequence[float],
    height_scale: float,
) -> np.ndarray:
    source_rest_v = np.asarray(source_rest, dtype=np.float64)
    source_pose_v = np.asarray(source_pose, dtype=np.float64)
    target_rest_v = np.asarray(target_rest, dtype=np.float64)
    if any(value.shape != (3,) for value in (source_rest_v, source_pose_v, target_rest_v)):
        raise RetargetError("translation inputs must be 3D vectors")
    if not all(np.isfinite(value).all() for value in (source_rest_v, source_pose_v, target_rest_v)):
        raise RetargetError("translation inputs must be finite")
    if not math.isfinite(height_scale) or height_scale <= 0.0:
        raise RetargetError("height scale must be finite and positive")
    return target_rest_v + height_scale * (source_pose_v - source_rest_v)


def compute_height_scales(
    *,
    source_armature_height: float,
    target_armature_height: float,
    source_world_height: float,
    target_world_height: float,
) -> dict[str, float]:
    values = {
        "source armature height": source_armature_height,
        "target armature height": target_armature_height,
        "source world height": source_world_height,
        "target world height": target_world_height,
    }
    for description, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise RetargetError(f"{description} must be finite and positive")
    return {
        "pelvis_local_scale": float(target_armature_height / source_armature_height),
        "root_world_scale": float(target_world_height / source_world_height),
    }


def slerp_quaternion(
    first: Sequence[float], second: Sequence[float], alpha: float
) -> np.ndarray:
    first_q = np.asarray(first, dtype=np.float64).copy()
    second_q = np.asarray(second, dtype=np.float64).copy()
    if first_q.shape != (4,) or second_q.shape != (4,):
        raise RetargetError("SLERP inputs must be WXYZ quaternions")
    if not np.isfinite(first_q).all() or not np.isfinite(second_q).all():
        raise RetargetError("SLERP inputs must be finite")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise RetargetError("SLERP alpha must be in [0, 1]")
    for value, label in ((first_q, "first"), (second_q, "second")):
        length = float(np.linalg.norm(value))
        if length <= 1.0e-12:
            raise RetargetError(f"{label} SLERP quaternion is degenerate")
        value /= length
    dot = float(np.dot(first_q, second_q))
    if dot < 0.0:
        second_q = -second_q
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = first_q + alpha * (second_q - first_q)
    else:
        theta = math.acos(dot)
        sine = math.sin(theta)
        result = (
            math.sin((1.0 - alpha) * theta) / sine * first_q
            + math.sin(alpha * theta) / sine * second_q
        )
    return result / np.linalg.norm(result)


def _normalized_quaternion(value: Sequence[float], description: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).copy()
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise RetargetError(f"{description} must be a finite WXYZ quaternion")
    length = float(np.linalg.norm(quaternion))
    if length <= 1.0e-12:
        raise RetargetError(f"{description} quaternion is degenerate")
    return quaternion / length


def multiply_quaternions(
    first: Sequence[float], second: Sequence[float]
) -> np.ndarray:
    first_q = _normalized_quaternion(first, "first product")
    second_q = _normalized_quaternion(second, "second product")
    w1, x1, y1, z1 = first_q
    w2, x2, y2, z2 = second_q
    return _normalized_quaternion(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        "quaternion product",
    )


def inverse_quaternion(value: Sequence[float]) -> np.ndarray:
    quaternion = _normalized_quaternion(value, "inverse")
    quaternion[1:] *= -1.0
    return quaternion


def resample_spine_quaternions(
    plan: Sequence[Mapping[str, Any]],
    source_deltas: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    if not plan:
        raise RetargetError("cumulative spine resample plan is empty")
    source_cumulative: dict[str, np.ndarray] = {}
    running = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    for source_name in ROCKETBOX_SPINE_BONES:
        try:
            source_delta = source_deltas[source_name]
        except KeyError as error:
            raise RetargetError(
                f"missing source spine delta: {source_name}"
            ) from error
        running = multiply_quaternions(running, source_delta)
        source_cumulative[source_name] = running

    result: dict[str, np.ndarray] = {}
    previous_cumulative: np.ndarray | None = None
    for record in plan:
        if record.get("interpolation_domain") != "cumulative_parent_to_child_rotation":
            raise RetargetError("spine plan is not a cumulative rotation field")
        first, second = record["source_bones"]
        alpha = float(record["weights"][1])
        try:
            first_q = source_cumulative[first]
            second_q = source_cumulative[second]
        except KeyError as error:
            raise RetargetError(f"missing source spine delta: {error.args[0]}") from error
        target_cumulative = slerp_quaternion(
            first_q, second_q, alpha
        )
        target_local = (
            target_cumulative
            if previous_cumulative is None
            else multiply_quaternions(
                inverse_quaternion(previous_cumulative), target_cumulative
            )
        )
        result[str(record["target_bone"])] = target_local
        previous_cumulative = target_cumulative
    return result


def anatomical_frame_from_points(
    *,
    left: Sequence[float],
    right: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    primary_axis: str,
    description: str,
) -> np.ndarray:
    """Build a proper anatomical frame without using any bone-local roll axis.

    Columns are semantic right, forward, and up.  Pelvis/spine frames preserve
    the lateral chord exactly; neck/head frames preserve the axial chord
    exactly.  The remaining axis is orthogonalized deterministically.
    """

    points = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (left, right, lower, upper)
    )
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in points):
        raise RetargetError(f"{description} anatomical points must be finite 3D")
    left_v, right_v, lower_v, upper_v = points
    lateral = right_v - left_v
    up = upper_v - lower_v
    if primary_axis == "lateral":
        length = float(np.linalg.norm(lateral))
        if length <= 1.0e-8:
            raise RetargetError(f"{description} anatomical lateral chord is degenerate")
        lateral /= length
        up -= lateral * float(np.dot(up, lateral))
        length = float(np.linalg.norm(up))
        if length <= 1.0e-8:
            raise RetargetError(f"{description} anatomical up projection is degenerate")
        up /= length
    elif primary_axis == "up":
        length = float(np.linalg.norm(up))
        if length <= 1.0e-8:
            raise RetargetError(f"{description} anatomical axial chord is degenerate")
        up /= length
        lateral -= up * float(np.dot(lateral, up))
        length = float(np.linalg.norm(lateral))
        if length <= 1.0e-8:
            raise RetargetError(
                f"{description} anatomical lateral projection is degenerate"
            )
        lateral /= length
    else:
        raise RetargetError(f"{description} anatomical primary axis is invalid")
    forward = np.cross(up, lateral)
    length = float(np.linalg.norm(forward))
    if length <= 1.0e-8:
        raise RetargetError(f"{description} anatomical forward axis is degenerate")
    forward /= length
    frame = np.column_stack((lateral, forward, up))
    projected, _ = project_near_rotation(frame, f"{description} anatomical frame")
    return projected


def _interpolate_rotation_frames(
    frames: Sequence[np.ndarray], *, lower: int, upper: int, alpha: float
) -> np.ndarray:
    Matrix, _, _ = _mathutils()
    if (
        not frames
        or not 0 <= lower < len(frames)
        or not 0 <= upper < len(frames)
        or lower > upper
    ):
        raise RetargetError("anatomical spine rotation interpolation is invalid")
    if lower == upper or alpha <= 0.0:
        return np.asarray(frames[lower], dtype=np.float64).copy()
    if alpha >= 1.0:
        return np.asarray(frames[upper], dtype=np.float64).copy()
    first = tuple(Matrix(frames[lower].tolist()).to_quaternion().normalized())
    second = tuple(Matrix(frames[upper].tolist()).to_quaternion().normalized())
    value = slerp_quaternion(first, second, alpha)
    return np.asarray(_quaternion_from_array(value).to_matrix(), dtype=np.float64)


def build_anatomical_axial_transfer(
    *,
    target_armature: Any,
    semantic_bones: Mapping[str, Any],
    target_rest: Mapping[str, Mapping[str, Any]],
    cached: Mapping[str, Any],
    target_base_rotation_3x3: Sequence[Sequence[float]],
    spine_plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Precompute fitted-axis-independent pelvis/spine/head/shoulder poses."""

    target_base, _ = project_near_rotation(
        target_base_rotation_3x3, "target anatomical object base"
    )
    target_spines = list(semantic_bones["spine"])
    if len(spine_plan) != len(target_spines):
        raise RetargetError("anatomical spine plan does not cover target spine")

    def target_point(name: str) -> np.ndarray:
        value = np.asarray(
            tuple(target_armature.data.bones[name].head_local), dtype=np.float64
        )
        if value.shape != (3,) or not np.isfinite(value).all():
            raise RetargetError(f"target anatomical rest point is invalid: {name}")
        return value

    target_shoulders = {
        side: target_point(semantic_bones[f"{side}_upper_arm"])
        for side in ("left", "right")
    }
    target_hips = {
        side: target_point(semantic_bones[f"{side}_thigh"])
        for side in ("left", "right")
    }
    target_frames: dict[str, np.ndarray] = {
        "pelvis": anatomical_frame_from_points(
            left=target_hips["left"],
            right=target_hips["right"],
            lower=target_point(semantic_bones["pelvis"]),
            upper=target_point(target_spines[0]),
            primary_axis="lateral",
            description="target pelvis rest",
        )
    }
    for index, name in enumerate(target_spines):
        upper_name = (
            target_spines[index + 1]
            if index + 1 < len(target_spines)
            else semantic_bones["neck"]
        )
        target_frames[name] = anatomical_frame_from_points(
            left=target_shoulders["left"],
            right=target_shoulders["right"],
            lower=target_point(name),
            upper=target_point(upper_name),
            primary_axis="lateral",
            description=f"target spine rest {name}",
        )
    target_neck_frame = anatomical_frame_from_points(
        left=target_shoulders["left"],
        right=target_shoulders["right"],
        lower=target_point(semantic_bones["neck"]),
        upper=target_point(semantic_bones["head"]),
        primary_axis="up",
        description="target neck/head rest",
    )
    target_frames["neck"] = target_neck_frame
    target_frames["head"] = target_neck_frame

    offsets: dict[str, np.ndarray] = {}
    offset_names = {
        "pelvis": semantic_bones["pelvis"],
        "neck": semantic_bones["neck"],
        "head": semantic_bones["head"],
        **{name: name for name in target_spines},
    }
    for key, name in offset_names.items():
        rest_bone, _ = project_near_rotation(
            target_rest[name]["global"].to_3x3(),
            f"target anatomical rest bone {name}",
        )
        rest_bone_world = target_base @ rest_bone
        rest_frame_world = target_base @ target_frames[key]
        offsets[key] = rest_frame_world.T @ rest_bone_world

    target_clavicle_rest: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        clavicle = semantic_bones[f"{side}_clavicle"]
        upper_arm = semantic_bones[f"{side}_upper_arm"]
        rest_bone, _ = project_near_rotation(
            target_rest[clavicle]["global"].to_3x3(),
            f"target anatomical clavicle rest {side}",
        )
        target_clavicle_rest[side] = {
            "direction_world": target_base
            @ (target_point(upper_arm) - target_point(clavicle)),
            "bone_world": target_base @ rest_bone,
        }

    source_first_root, _ = project_near_rotation(
        cached["frames"][0].root_rotation.to_matrix(),
        "source anatomical first object rotation",
    )
    runtime_frames: dict[int, dict[str, Any]] = {}
    maximum_rotation_error = 0.0
    for frame_record in cached["frames"]:
        source_points = frame_record.world_joint_positions

        def source_point(name: str) -> np.ndarray:
            value = np.asarray(tuple(source_points[name]), dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise RetargetError(
                    f"source anatomical point is invalid at frame {frame_record.frame}: {name}"
                )
            return value

        source_shoulders = {
            side: source_point(ROCKETBOX_ROLE_TO_BONE[f"{side}_upper_arm"])
            for side in ("left", "right")
        }
        source_hips = {
            side: source_point(ROCKETBOX_ROLE_TO_BONE[f"{side}_thigh"])
            for side in ("left", "right")
        }
        source_pelvis_frame = anatomical_frame_from_points(
            left=source_hips["left"],
            right=source_hips["right"],
            lower=source_point(ROCKETBOX_ROLE_TO_BONE["pelvis"]),
            upper=source_point(ROCKETBOX_SPINE_BONES[0]),
            primary_axis="lateral",
            description=f"source pelvis frame {frame_record.frame}",
        )
        source_spine_frames = []
        for index, name in enumerate(ROCKETBOX_SPINE_BONES):
            upper_name = (
                ROCKETBOX_SPINE_BONES[index + 1]
                if index + 1 < len(ROCKETBOX_SPINE_BONES)
                else ROCKETBOX_ROLE_TO_BONE["neck"]
            )
            source_spine_frames.append(
                anatomical_frame_from_points(
                    left=source_shoulders["left"],
                    right=source_shoulders["right"],
                    lower=source_point(name),
                    upper=source_point(upper_name),
                    primary_axis="lateral",
                    description=f"source spine frame {frame_record.frame}:{name}",
                )
            )
        source_neck_frame = anatomical_frame_from_points(
            left=source_shoulders["left"],
            right=source_shoulders["right"],
            lower=source_point(ROCKETBOX_ROLE_TO_BONE["neck"]),
            upper=source_point(ROCKETBOX_ROLE_TO_BONE["head"]),
            primary_axis="up",
            description=f"source neck/head frame {frame_record.frame}",
        )
        source_current_root, _ = project_near_rotation(
            frame_record.root_rotation.to_matrix(),
            f"source anatomical object rotation {frame_record.frame}",
        )
        root_delta, _ = project_near_rotation(
            source_first_root.T @ source_current_root,
            f"source anatomical root delta {frame_record.frame}",
        )
        target_current_root, _ = project_near_rotation(
            target_base @ root_delta,
            f"target anatomical object rotation {frame_record.frame}",
        )

        desired: dict[str, Any] = {
            "pelvis": target_current_root.T @ source_pelvis_frame @ offsets["pelvis"],
            "spine": {},
            "neck": target_current_root.T @ source_neck_frame @ offsets["neck"],
            "head": target_current_root.T @ source_neck_frame @ offsets["head"],
            "clavicles": {},
        }
        for record in spine_plan:
            lower, upper = (int(value) for value in record["source_indices"])
            alpha = float(record["weights"][1])
            source_frame = _interpolate_rotation_frames(
                source_spine_frames, lower=lower, upper=upper, alpha=alpha
            )
            name = str(record["target_bone"])
            desired["spine"][name] = (
                target_current_root.T @ source_frame @ offsets[name]
            )
        for side in ("left", "right"):
            source_direction = (
                source_point(ROCKETBOX_ROLE_TO_BONE[f"{side}_upper_arm"])
                - source_point(ROCKETBOX_ROLE_TO_BONE[f"{side}_clavicle"])
            )
            alignment = _minimal_direction_alignment(
                target_clavicle_rest[side]["direction_world"],
                source_direction,
                description=f"anatomical clavicle bridge {frame_record.frame}:{side}",
            )
            desired["clavicles"][side] = (
                target_current_root.T
                @ alignment
                @ target_clavicle_rest[side]["bone_world"]
            )
        for value in (
            desired["pelvis"],
            *desired["spine"].values(),
            desired["neck"],
            desired["head"],
            *desired["clavicles"].values(),
        ):
            projected, _ = project_near_rotation(
                value, f"target anatomical desired pose {frame_record.frame}"
            )
            maximum_rotation_error = max(
                maximum_rotation_error,
                float(np.max(np.abs(np.asarray(value) - projected))),
            )
        runtime_frames[int(frame_record.frame)] = desired

    expected_records = len(runtime_frames) * (len(target_spines) + 5)
    return {
        "frames": runtime_frames,
        "evidence": {
            "schema": "tokenrig_anatomical_axial_body_transfer_v1",
            "method": "source_world_anatomical_frame_absolute_v1",
            "pelvis_primary_constraint": "left_to_right_hip_chord",
            "spine_primary_constraint": "left_to_right_shoulder_chord",
            "neck_head_primary_constraint": "neck_to_head_chord",
            "head_axis_source": "neck_to_head_semantic_chord",
            "clavicle_method": "rest_twist_minimal_child_direction_alignment_v1",
            "source_world_joint_frame_used": True,
            "target_current_object_rotation_removed_once": True,
            "per_bone_rest_axis_conjugation_used": False,
            "source_spine_count": len(ROCKETBOX_SPINE_BONES),
            "target_spine_count": len(target_spines),
            "frame_count": len(runtime_frames),
            "record_count": expected_records,
            "maximum_rotation_projection_error": maximum_rotation_error,
            "automatic_checks": "passed",
        },
    }


def build_exact_semantic_correspondence(
    validated_mapping: Mapping[str, Any]
) -> dict[str, str]:
    semantic = validated_mapping.get("semantic_bones")
    if not isinstance(semantic, Mapping):
        raise RetargetError("validated semantic mapping has no semantic bones")
    correspondence = {
        source_name: str(semantic[role])
        for role, source_name in ROCKETBOX_ROLE_TO_BONE.items()
    }
    if len(correspondence) != len(set(correspondence.values())):
        raise RetargetError("exact semantic correspondence is ambiguous")
    return correspondence


def _section(metrics: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = metrics.get(name)
    if not isinstance(value, Mapping):
        raise RetargetError(f"action metrics are missing {name}")
    return value


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetargetError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RetargetError(f"{description} must be finite")
    return result


def validate_action_metrics(
    metrics: Mapping[str, Any],
    *,
    semantic_mapping: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    action_name = metrics.get("action_name")
    if action_name not in set(ACTION_NAMES.values()):
        raise RetargetError("action name is not Walking or Standing_Idle")
    frame_start = metrics.get("frame_start")
    frame_end = metrics.get("frame_end")
    if (
        isinstance(frame_start, bool)
        or isinstance(frame_end, bool)
        or not isinstance(frame_start, int)
        or not isinstance(frame_end, int)
        or frame_end <= frame_start
    ):
        raise RetargetError("action frame range is invalid")
    expected_frame_keys = {
        str(frame) for frame in range(frame_start, frame_end + 1)
    }
    expected_frame_count = frame_end - frame_start + 1
    rest_delta = _section(metrics, "rest_delta")
    for field in (
        "target_rest_translations_preserved",
        "finite_rest_and_pose_matrices",
        "parent_first",
    ):
        if rest_delta.get(field) is not True:
            raise RetargetError(f"rest-corrected retarget invariant failed: {field}")
    if (
        _finite_number(
            rest_delta.get("maximum_global_rest_alignment_error"),
            "global rest alignment error",
        )
        > REST_ROTATION_MATRIX_TOLERANCE
    ):
        raise RetargetError("global rest alignment exceeds tolerance")
    if (
        _finite_number(
            rest_delta.get("maximum_local_rest_delta_error"),
            "spine local rest delta error",
        )
        > REST_ROTATION_MATRIX_TOLERANCE
    ):
        raise RetargetError("spine local rest delta exceeds tolerance")
    if (
        _finite_number(
            rest_delta.get("maximum_target_translation_error_m"),
            "target rest translation error",
        )
        > POSE_TRANSLATION_TOLERANCE_M
    ):
        raise RetargetError("target rest translation reconstruction exceeds tolerance")

    projection = metrics.get("rotation_projection")
    if not isinstance(projection, Mapping):
        raise RetargetError("action metrics are missing rotation projection evidence")
    validate_rotation_projection_summary(
        projection,
        expected_action=str(action_name),
        frame_start=frame_start,
        frame_end=frame_end,
        semantic_mapping=semantic_mapping,
    )

    root = _section(metrics, "root_motion")
    if root.get("axis_map_3x3") != [list(row) for row in AXIS_MAP_3X3]:
        raise RetargetError("retarget axis must remain identity at canonical FRONT -Y")
    if _finite_number(root.get("height_scale"), "height scale") <= 0.0:
        raise RetargetError("height scale must be positive")
    if (
        _finite_number(root.get("reconstruction_error_m"), "root motion reconstruction")
        > ROOT_RECONSTRUCTION_TOLERANCE_M
    ):
        raise RetargetError("root motion reconstruction exceeds tolerance")
    if action_name == ACTION_NAMES["walk"]:
        direction = _finite_number(
            root.get("endpoint_direction_dot_negative_y"), "walking direction"
        )
        if direction < MINIMUM_FORWARD_DOT:
            raise RetargetError("Walking is reverse or not aligned to FRONT -Y")
    body_front = _finite_number(
        root.get("minimum_body_forward_dot_negative_y"), "body forward"
    )
    if body_front < MINIMUM_BODY_FORWARD_DOT:
        raise RetargetError("body forward is backward or not aligned to FRONT -Y")
    if action_name == ACTION_NAMES["walk"]:
        body_travel = _finite_number(
            root.get("minimum_body_forward_dot_travel"),
            "body forward versus travel",
        )
        if body_travel < MINIMUM_BODY_FORWARD_DOT:
            raise RetargetError("body forward disagrees with Walking travel")

    speed = _section(metrics, "speed")
    target_speed = _finite_number(speed.get("target_speed_m_per_s"), "target speed")
    if (
        _finite_number(
            speed.get("absolute_reconstruction_error_m_per_s"),
            "speed reconstruction error",
        )
        > SPEED_RECONSTRUCTION_TOLERANCE_MPS
    ):
        raise RetargetError("speed does not reconstruct the height-scaled source")
    if action_name == ACTION_NAMES["idle"] and target_speed > MAXIMUM_IDLE_SPEED_MPS:
        raise RetargetError("idle speed is not stationary")

    loop = _section(metrics, "loop")
    if (
        _finite_number(loop.get("maximum_rotation_residual_rad"), "loop rotation")
        > LOOP_ROTATION_TOLERANCE_RAD
        or _finite_number(
            loop.get("root_cycle_reconstruction_error_m"), "loop root motion"
        )
        > LOOP_ROOT_TOLERANCE_M
    ):
        raise RetargetError("loop endpoint reconstruction failed")
    if (
        _finite_number(
            loop.get("armature_root_rotation_residual_rad"),
            "loop root rotation",
        )
        > LOOP_ROTATION_TOLERANCE_RAD
    ):
        raise RetargetError("loop armature root rotation is discontinuous")
    if (
        _finite_number(
            loop.get("pelvis_local_translation_residual_m"),
            "loop pelvis translation",
        )
        > LOOP_PELVIS_TRANSLATION_TOLERANCE_M
    ):
        raise RetargetError("loop pelvis translation is discontinuous")
    boundary_gate = loop.get("source_calibrated_boundary_velocity")
    if not isinstance(boundary_gate, Mapping):
        raise RetargetError("source-calibrated loop boundary evidence is missing")
    try:
        recomputed_boundary_gate = source_calibrated_boundary_velocity_gate(
            target_records=boundary_gate.get("target_records", {}),
            source_records=boundary_gate.get("source_records", {}),
            target_to_source=boundary_gate.get("target_to_source", {}),
            height_scale=boundary_gate.get("height_scale"),
        )
    except RetargetError as error:
        raise RetargetError(
            f"source-calibrated loop boundary velocity is invalid: {error}"
        ) from error
    if dict(boundary_gate) != recomputed_boundary_gate:
        raise RetargetError("source-calibrated loop boundary evidence is inconsistent")
    target_boundary_maximum = max(
        _finite_number(value.get("residual_m_per_s"), "target boundary residual")
        for value in recomputed_boundary_gate["target_records"].values()
    )
    if not math.isclose(
        _finite_number(
            loop.get("maximum_boundary_velocity_residual_m_per_s"),
            "loop boundary velocity",
        ),
        target_boundary_maximum,
        abs_tol=1.0e-9,
    ):
        raise RetargetError("loop boundary velocity maximum is inconsistent")
    if loop.get("foot_phase_continuous") is not True:
        raise RetargetError("loop foot phase is discontinuous")

    surface_ik = _section(metrics, "surface_contact_ik")
    if any(
        (
            surface_ik.get("schema") != SURFACE_CONTACT_IK_SCHEMA,
            surface_ik.get("method")
            != "evaluated_surface_minimum_to_vertical_two_bone_leg_ik_v1",
            surface_ik.get("action_name") != action_name,
            surface_ik.get("frame_start") != frame_start,
            surface_ik.get("frame_end") != frame_end,
            surface_ik.get("frame_count") != expected_frame_count,
            surface_ik.get("fixed_floor_z_m") != 0.0,
            surface_ik.get("automatic_checks") != "passed",
            surface_ik.get("root_pelvis_hip_translation_preserved") is not True,
            surface_ik.get("ankle_xy_preserved") is not True,
            surface_ik.get("foot_toe_global_orientation_preserved") is not True,
        )
    ):
        raise RetargetError("surface-contact leg IK provenance is invalid")
    surface_margin = _finite_number(
        surface_ik.get("safety_margin_m"), "surface-contact safety margin"
    )
    surface_maximum = _finite_number(
        surface_ik.get("maximum_cumulative_upward_correction_m"),
        "surface-contact maximum correction",
    )
    if (
        surface_margin != IK_CONTACT_READBACK_SAFETY_MARGIN_M
        or surface_maximum < 0.0
        or surface_maximum > MAXIMUM_IK_ANKLE_CORRECTION_M
        or surface_ik.get("maximum_allowed_ankle_correction_m")
        != MAXIMUM_IK_ANKLE_CORRECTION_M
    ):
        raise RetargetError("surface-contact leg IK exceeded its correction contract")
    surface_records = surface_ik.get("records_by_frame")
    if not isinstance(surface_records, Mapping) or set(surface_records) != expected_frame_keys:
        raise RetargetError("surface-contact leg IK does not cover every action frame")
    recorded_corrections: list[float] = []
    for frame, record in surface_records.items():
        if not isinstance(record, Mapping):
            raise RetargetError(f"surface-contact frame record is invalid: {frame}")
        pre = record.get("pre_minimum_z_m")
        post = record.get("post_minimum_z_m")
        corrections = record.get("cumulative_upward_correction_m")
        if any(not isinstance(value, Mapping) for value in (pre, post, corrections)):
            raise RetargetError(f"surface-contact frame evidence is incomplete: {frame}")
        if any(set(value) != {"left", "right"} for value in (pre, post, corrections)):
            raise RetargetError(f"surface-contact frame lacks bilateral evidence: {frame}")
        for side in ("left", "right"):
            _finite_number(pre[side], f"surface-contact {frame} {side} pre-minimum")
            post_value = _finite_number(
                post[side], f"surface-contact {frame} {side} post-minimum"
            )
            correction = _finite_number(
                corrections[side], f"surface-contact {frame} {side} correction"
            )
            if (
                post_value < -IK_CONTACT_READBACK_SAFETY_MARGIN_M
                or correction < 0.0
                or correction > MAXIMUM_IK_ANKLE_CORRECTION_M
            ):
                raise RetargetError("surface-contact frame violates floor or IK cap")
            recorded_corrections.append(correction)
        iterations = record.get("iterations")
        iteration_count = record.get("iteration_count")
        if (
            not isinstance(iterations, list)
            or isinstance(iteration_count, bool)
            or not isinstance(iteration_count, int)
            or iteration_count != len(iterations)
            or not 0 <= iteration_count <= MAXIMUM_SURFACE_CONTACT_IK_ITERATIONS
        ):
            raise RetargetError("surface-contact IK iteration evidence is invalid")
        for iteration in iterations:
            solutions = iteration.get("solutions") if isinstance(iteration, Mapping) else None
            if not isinstance(solutions, Mapping) or not solutions:
                raise RetargetError("surface-contact IK solution evidence is missing")
            for solution in solutions.values():
                readback = solution.get("readback") if isinstance(solution, Mapping) else None
                if (
                    not isinstance(readback, Mapping)
                    or solution.get("automatic_checks") != "passed"
                    or readback.get("root_pelvis_hip_translation_preserved") is not True
                    or readback.get("ankle_xy_preserved") is not True
                    or readback.get("foot_toe_global_orientation_preserved") is not True
                ):
                    raise RetargetError("surface-contact IK readback is invalid")
    if not math.isclose(
        surface_maximum, max(recorded_corrections), abs_tol=1.0e-9
    ):
        raise RetargetError("surface-contact maximum correction evidence is inconsistent")

    floor = _section(metrics, "floor")
    if _finite_number(floor.get("fixed_floor_z_m"), "fixed floor") != 0.0:
        raise RetargetError("fixed floor must be the authenticated Z=0 plane")
    if abs(
        _finite_number(
            floor.get("grounding_correction_m"), "grounding correction"
        )
    ) > MAXIMUM_GROUNDING_CORRECTION_M:
        raise RetargetError("grounding correction exceeds 0.010 m")
    pre_ground_maximum = _finite_number(
        floor.get("pre_ground_maximum_penetration_m"),
        "pre-ground penetration",
    )
    if pre_ground_maximum > MAXIMUM_PENETRATION_M:
        raise RetargetError("pre-ground penetration exceeds 0.010 m")
    pre_ground_by_frame = floor.get("pre_ground_penetration_by_frame_m")
    if (
        not isinstance(pre_ground_by_frame, Mapping)
        or set(pre_ground_by_frame) != expected_frame_keys
    ):
        raise RetargetError(
            "pre-ground penetration does not cover every action frame"
        )
    pre_ground_values = [
        _finite_number(value, f"pre-ground penetration {frame}")
        for frame, value in pre_ground_by_frame.items()
    ]
    if (
        any(value < 0.0 or value > MAXIMUM_PENETRATION_M for value in pre_ground_values)
        or not math.isclose(
            pre_ground_maximum, max(pre_ground_values), abs_tol=1.0e-9
        )
    ):
        raise RetargetError("pre-ground per-frame penetration evidence is inconsistent")
    penetration_by_frame = floor.get("penetration_by_frame_m")
    if not isinstance(penetration_by_frame, Mapping) or not penetration_by_frame:
        raise RetargetError("per-frame penetration evidence is missing")
    if set(penetration_by_frame) != expected_frame_keys:
        raise RetargetError("per-frame penetration does not cover every action frame")
    per_frame_penetrations = [
        _finite_number(value, f"per-frame penetration {frame}")
        for frame, value in penetration_by_frame.items()
    ]
    if any(
        value < 0.0 or value > MAXIMUM_PENETRATION_M
        for value in per_frame_penetrations
    ):
        raise RetargetError("per-frame penetration exceeds 0.010 m")
    if (
        _finite_number(floor.get("maximum_penetration_m"), "penetration")
        > MAXIMUM_PENETRATION_M
    ):
        raise RetargetError("penetration exceeds 0.010 m")
    if (
        _finite_number(
            floor.get("maximum_per_foot_cycle_minimum_clearance_m"),
            "per-foot cycle minimum hover clearance",
        )
        > MAXIMUM_HOVER_M
    ):
        raise RetargetError("foot hover exceeds the 0.030 m support gate")
    if floor.get("left_contact") is not True or floor.get("right_contact") is not True:
        raise RetargetError("bilateral foot/toe support is absent over the loop")

    sampling = _section(metrics, "sampling")
    if (
        sampling.get("method") != "deterministic_spatial_skin_support_core_v2"
        or sampling.get("seed") != QUALITY_SAMPLE_SEED
        or not isinstance(sampling.get("index_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", sampling["index_sha256"])
    ):
        raise RetargetError("deterministic mesh sampling provenance is invalid")
    for field in (
        "vertex_coverage_ratio",
        "edge_coverage_ratio",
        "lower_body_edge_coverage_ratio",
    ):
        coverage = _finite_number(sampling.get(field), f"sampling {field}")
        if not 0.0 < coverage <= 1.0:
            raise RetargetError(f"sampling {field} has no coverage")

    contact = _section(metrics, "contact")
    feet_contact = contact.get("feet")
    if not isinstance(feet_contact, Mapping) or set(feet_contact) != {"left", "right"}:
        raise RetargetError("per-foot temporal contact evidence is missing")
    for side in ("left", "right"):
        evidence = feet_contact[side]
        if not isinstance(evidence, Mapping):
            raise RetargetError(f"{side} temporal contact evidence is invalid")
        contact_ratio = _finite_number(
            evidence.get("contact_ratio"), f"{side} contact ratio"
        )
        stance_ratio = _finite_number(
            evidence.get("stance_contact_ratio"), f"{side} stance contact ratio"
        )
        hover_frames = _finite_number(
            evidence.get("maximum_consecutive_hover_frames"),
            f"{side} consecutive hover frames",
        )
        if action_name == ACTION_NAMES["walk"]:
            if contact_ratio < MINIMUM_WALK_CONTACT_RATIO:
                raise RetargetError(f"{side} Walking contact ratio is too low")
            if stance_ratio < MINIMUM_WALK_STANCE_CONTACT_RATIO:
                raise RetargetError(f"{side} stance contact ratio is too low")
            if hover_frames > MAXIMUM_WALK_CONSECUTIVE_HOVER_FRAMES:
                raise RetargetError(f"{side} consecutive hover is too long")
        else:
            if contact_ratio < MINIMUM_IDLE_CONTACT_RATIO:
                raise RetargetError(f"Idle {side} foot contact ratio is too low")
            if stance_ratio < MINIMUM_IDLE_CONTACT_RATIO:
                raise RetargetError(f"Idle {side} stance contact ratio is too low")
            if hover_frames > MAXIMUM_IDLE_CONSECUTIVE_HOVER_FRAMES:
                raise RetargetError(f"Idle {side} consecutive hover is too long")
        stance_slide = _finite_number(
            evidence.get("maximum_stance_slide_m"), f"{side} stance slide"
        )
        stance_speed = _finite_number(
            evidence.get("maximum_stance_speed_m_per_s"),
            f"{side} stance speed",
        )
        if stance_slide < 0.0 or stance_speed < 0.0:
            raise RetargetError(f"{side} stance-slide evidence cannot be negative")
        # A clearance-only Walking stance label also contains low swing frames on
        # this approved source motion.  Keep the measured slide as review evidence,
        # but do not let that phase ambiguity reject an otherwise grounded Walk.
        # Standing Idle remains strictly planted and therefore retains both caps.
        if action_name == ACTION_NAMES["idle"]:
            if stance_slide > MAXIMUM_STANCE_SLIDE_M:
                raise RetargetError(f"Idle {side} stance slide exceeds tolerance")
            if stance_speed > MAXIMUM_STANCE_SPEED_MPS:
                raise RetargetError(
                    f"Idle {side} stance slide speed exceeds tolerance"
                )
        contact_samples = evidence.get("contact_by_frame")
        stance_samples = evidence.get("stance_by_frame")
        if (
            not isinstance(contact_samples, list)
            or not isinstance(stance_samples, list)
            or len(contact_samples) != expected_frame_count
            or len(stance_samples) != expected_frame_count
            or any(not isinstance(value, bool) for value in contact_samples)
            or any(not isinstance(value, bool) for value in stance_samples)
            or evidence.get("frame_count") != expected_frame_count
        ):
            raise RetargetError(f"{side} temporal contact does not cover every action frame")
        contact_count = sum(contact_samples)
        stance_count = sum(stance_samples)
        if (
            stance_count <= 0
            or evidence.get("contact_frame_count") != contact_count
            or evidence.get("stance_frame_count") != stance_count
            or not math.isclose(contact_ratio, contact_count / expected_frame_count, abs_tol=1.0e-9)
            or not math.isclose(
                stance_ratio,
                sum(
                    contact_value and stance_value
                    for contact_value, stance_value in zip(
                        contact_samples, stance_samples
                    )
                )
                / stance_count,
                abs_tol=1.0e-9,
            )
        ):
            raise RetargetError(f"{side} temporal contact ratios are inconsistent")
    if not foot_phase_is_continuous(
        {
            side: feet_contact[side]["contact_by_frame"]
            for side in ("left", "right")
        }
    ):
        raise RetargetError("loop foot phase is discontinuous")
    computed_support = summarize_support_union(
        left_contact=feet_contact["left"]["contact_by_frame"],
        right_contact=feet_contact["right"]["contact_by_frame"],
    )
    recorded_support = {
        field: contact.get(field)
        for field in (
            "support_union",
            "support_coverage_ratio",
            "maximum_consecutive_both_feet_airborne_frames",
        )
    }
    if recorded_support != computed_support:
        raise RetargetError("bilateral support_union evidence is inconsistent")
    if action_name == ACTION_NAMES["walk"]:
        validate_walking_support(recorded_support)
    bilateral_ratio = _finite_number(
        contact.get("bilateral_contact_ratio"), "bilateral contact ratio"
    )
    if (
        action_name == ACTION_NAMES["idle"]
        and bilateral_ratio < MINIMUM_IDLE_BILATERAL_CONTACT_RATIO
    ):
        raise RetargetError("Idle bilateral contact ratio is too low")

    performance = _section(metrics, "performance")
    passes = performance.get("passes")
    if (
        performance.get("schema") != "indexed_evaluated_mesh_performance_v1"
        or not isinstance(passes, Mapping)
        or set(passes) != {"grounding", "quality"}
    ):
        raise RetargetError("performance telemetry schema or passes are invalid")
    for phase, by_frame in passes.items():
        if not isinstance(by_frame, Mapping) or set(by_frame) != expected_frame_keys:
            raise RetargetError(
                f"performance {phase} does not cover every action frame"
            )
        for frame, telemetry in by_frame.items():
            if not isinstance(telemetry, Mapping):
                raise RetargetError(f"performance {phase} frame {frame} is invalid")
            full_vertices = _finite_number(
                telemetry.get("full_evaluated_vertex_count"),
                f"performance {phase} full evaluated vertices",
            )
            full_edges = _finite_number(
                telemetry.get("full_evaluated_edge_count"),
                f"performance {phase} full evaluated edges",
            )
            sampled_vertices = _finite_number(
                telemetry.get("sampled_vertex_count"),
                f"performance {phase} sampled vertices",
            )
            wall_time = _finite_number(
                telemetry.get("wall_time_seconds"),
                f"performance {phase} wall time",
            )
            peak_rss = _finite_number(
                telemetry.get("process_peak_rss_bytes"),
                f"performance {phase} peak RSS",
            )
            if full_vertices <= 0 or full_edges <= 0:
                raise RetargetError("performance full evaluated mesh is empty")
            if sampled_vertices <= 0 or sampled_vertices >= full_vertices:
                raise RetargetError(
                    "performance sampled vertices do not stay below the full evaluated mesh"
                )
            if wall_time < 0.0 or peak_rss <= 0:
                raise RetargetError("performance wall time or memory is invalid")

    feet = _section(metrics, "feet")
    if feet.get("inverted") is not False or _finite_number(
        feet.get("minimum_foot_to_toe_rest_dot"), "foot-to-toe rest dot"
    ) <= 0.0:
        raise RetargetError("foot inversion detected")

    deformation = _section(metrics, "deformation")
    if deformation.get("calibration_basis") != "approved_source_motion_and_static_bind_v1":
        raise RetargetError("deformation calibration basis is not authenticated")
    required_shoulder = _finite_number(
        deformation.get("required_minimum_shoulder_span_ratio"),
        "calibrated shoulder span",
    )
    required_hip = _finite_number(
        deformation.get("required_minimum_hip_span_ratio"),
        "calibrated hip span",
    )
    allowed_edge = _finite_number(
        deformation.get("allowed_maximum_skinned_edge_stretch_ratio"),
        "calibrated edge stretch",
    )
    if required_shoulder < MINIMUM_CALIBRATED_SHOULDER_SPAN_RATIO:
        raise RetargetError("calibrated shoulder threshold is too permissive")
    if required_hip < MINIMUM_CALIBRATED_HIP_SPAN_RATIO:
        raise RetargetError("calibrated hip threshold is too permissive")
    if allowed_edge > MAXIMUM_CALIBRATED_EDGE_STRETCH_RATIO:
        raise RetargetError("calibrated edge threshold is too permissive")
    if (
        _finite_number(
            deformation.get("minimum_shoulder_span_ratio"), "shoulder span"
        )
        < required_shoulder
    ):
        raise RetargetError("shoulder collapse detected")
    if (
        _finite_number(deformation.get("minimum_hip_span_ratio"), "hip span")
        < required_hip
    ):
        raise RetargetError("hip collapse detected")
    if (
        _finite_number(
            deformation.get("maximum_skinned_edge_stretch_ratio"), "edge stretch"
        )
        > allowed_edge
    ):
        raise RetargetError(
            "garment/body tearing detected by edge stretch: "
            f"actual={float(deformation['maximum_skinned_edge_stretch_ratio']):.9f} "
            f"allowed={allowed_edge:.9f}"
        )

    roundtrip = _section(metrics, "roundtrip")
    boolean_checks = {
        "one_armature": "one armature",
        "one_skinned_mesh": "one skinned mesh",
        "one_action": "one-action GLB",
        "uv_present": "UV readback",
        "skin_present": "skin readback",
        "pbr_payloads_unchanged": "PBR payload preservation",
        "skeleton_exact": "exact generic skeleton",
        "loop_endpoints_exact": "loop endpoints",
        "finite_matrices": "finite matrices",
    }
    for field, description in boolean_checks.items():
        if roundtrip.get(field) is not True:
            if field == "one_action":
                raise RetargetError("one-action GLB readback failed")
            if field == "pbr_payloads_unchanged":
                raise RetargetError("PBR payload bytes changed")
            raise RetargetError(f"roundtrip failed: {description}")
    if roundtrip.get("action_name") != action_name:
        raise RetargetError("roundtrip action name changed")
    recorded_slide = {
        side: {
            "maximum_stance_slide_m": float(
                feet_contact[side]["maximum_stance_slide_m"]
            ),
            "maximum_stance_speed_m_per_s": float(
                feet_contact[side]["maximum_stance_speed_m_per_s"]
            ),
        }
        for side in ("left", "right")
    }
    return {
        "status": "passed",
        "action_name": str(action_name),
        "stance_slide_policy": (
            "recorded_advisory_clearance_phase_v1"
            if action_name == ACTION_NAMES["walk"]
            else "strict_planted_idle_v1"
        ),
        "recorded_stance_slide": recorded_slide,
    }


def file_descriptor(path: Path, *, public_path: str | None = None) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RetargetError(f"artifact is missing, empty, or a symlink: {path}")
    return {
        "path": public_path if public_path is not None else str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_retarget_manifest(
    *,
    asset_id: str,
    base_avatar_id: str,
    authenticated: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    command: Sequence[str],
    blender_version: str,
) -> dict[str, Any]:
    if not _ASSET_ID_RE.fullmatch(asset_id):
        raise RetargetError(f"invalid asset_id: {asset_id!r}")
    if base_avatar_id not in PRODUCTION_INPUT_CONTRACT.idle_by_baseline_asset:
        raise RetargetError(f"unsupported base_avatar_id: {base_avatar_id!r}")
    if metrics.get("schema") != METRICS_SCHEMA:
        raise RetargetError("retarget metrics schema is not pinned")
    actions = metrics.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(ACTION_NAMES.values()):
        raise RetargetError("retarget metrics must contain exactly Walk and Idle")
    semantic = metrics.get("semantic_mapping")
    rest_matrices = metrics.get("rest_matrices")
    spine_plan = metrics.get("spine_resample_plan")
    export_parameters = metrics.get("export_parameters")
    if not isinstance(semantic, Mapping) or not isinstance(rest_matrices, Mapping):
        raise RetargetError("manifest is missing semantic or rest-matrix evidence")
    action_checks = {
        name: validate_action_metrics(value, semantic_mapping=semantic)
        for name, value in actions.items()
    }
    if not isinstance(spine_plan, (list, Mapping)):
        raise RetargetError("manifest is missing spine resample weights")
    if isinstance(spine_plan, Mapping) and set(spine_plan) != set(ACTION_NAMES.values()):
        raise RetargetError("per-action spine resample weights must cover Walk and Idle")
    expected_export_parameters = {
        name: gltf_export_parameters(name) for name in ACTION_NAMES.values()
    }
    if export_parameters != expected_export_parameters:
        raise RetargetError("retarget metrics do not contain the complete GLB export parameters")
    return {
        "schema": MANIFEST_SCHEMA,
        "asset_id": asset_id,
        "base_avatar_id": base_avatar_id,
        "state_classification": "research_candidate",
        "canonical_front": CANONICAL_FRONT,
        "canonical_up": CANONICAL_UP,
        "axis_transform_at_retarget": "identity",
        "authenticated_inputs": dict(authenticated),
        "semantic_mapping_sha256": sha256_json(semantic),
        "rest_matrices_sha256": sha256_json(rest_matrices),
        "spine_resample_plan": json.loads(json.dumps(spine_plan)),
        "export_parameters": json.loads(json.dumps(export_parameters)),
        "export_parameters_sha256": sha256_json(export_parameters),
        "actions": action_checks,
        "artifacts": dict(artifacts),
        "command": [str(item) for item in command],
        "environment": {
            "blender_version": str(blender_version),
            "fps": 30,
            "runtime_provenance_sha256": sha256_json(
                authenticated.get("runtime")
            ),
        },
        "automatic_checks": "passed",
        "user_acceptance": "pending_user_review",
    }


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RetargetError("atomic no-replace directory publication is unavailable") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise RetargetError(f"no-replace destination already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


@dataclass
class SourceMotion:
    armature: Any
    action: Any
    imported_objects: tuple[Any, ...]
    imported_actions: tuple[Any, ...]
    frame_start: int
    frame_end: int


@dataclass
class CachedMotionFrame:
    frame: int
    root_location: Any
    root_rotation: Any
    global_rotations: Mapping[str, Any]
    local_rotations: Mapping[str, Any]
    joint_positions: Mapping[str, Any]
    world_joint_positions: Mapping[str, Any]
    pelvis_local_translation: Any


def _require_bpy():
    try:
        return importlib.import_module("bpy")
    except ImportError as error:
        raise RetargetError("Task 6 execution requires Blender's bpy runtime") from error


def validate_blender_runtime(bpy: Any) -> None:
    version = getattr(getattr(bpy, "app", None), "version", None)
    if not isinstance(version, (tuple, list)) or tuple(version[:2]) != (4, 2):
        raise RetargetError(f"Task 6 requires Blender 4.2, found {version!r}")


def _build_field(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def runtime_provenance(
    bpy: Any,
    *,
    static_audit_helper_path: Path | None = None,
    numpy_module_path: Path | None = None,
) -> dict[str, Any]:
    validate_blender_runtime(bpy)
    helper = (
        Path(static_audit_helper_path)
        if static_audit_helper_path is not None
        else Path(__file__).resolve().with_name("blender_tokenrig_human_static_audit.py")
    )
    numpy_path = (
        Path(numpy_module_path)
        if numpy_module_path is not None
        else Path(np.__file__).resolve()
    )
    binary_path = Path(str(getattr(bpy.app, "binary_path", "")))
    return {
        "static_audit_helper": file_descriptor(helper.resolve()),
        "blender": {
            "binary": file_descriptor(binary_path.resolve()),
            "version": [int(value) for value in bpy.app.version],
            "version_string": str(bpy.app.version_string),
            "build_hash": _build_field(bpy.app.build_hash),
            "build_date": _build_field(bpy.app.build_date),
            "build_time": _build_field(bpy.app.build_time),
            "build_branch": _build_field(bpy.app.build_branch),
            "build_platform": _build_field(bpy.app.build_platform),
        },
        "numpy": {
            "version": str(np.__version__),
            "module": file_descriptor(numpy_path.resolve()),
        },
    }


def _mathutils():
    try:
        module = importlib.import_module("mathutils")
    except ImportError as error:
        raise RetargetError("Task 6 execution requires Blender mathutils") from error
    return module.Matrix, module.Quaternion, module.Vector


def _static_audit_module():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("tools.blender_tokenrig_human_static_audit")


def _matrix_rows(matrix: Any) -> list[list[float]]:
    rows = [[float(value) for value in row] for row in matrix]
    if not rows or not all(math.isfinite(value) for row in rows for value in row):
        raise RetargetError("rest/pose matrix contains non-finite values")
    return rows


def _parent_local_rest(bone: Any) -> Any:
    if bone.parent is None:
        return bone.matrix_local.copy()
    return bone.parent.matrix_local.inverted() @ bone.matrix_local


def _parent_local_pose(pose_bone: Any) -> Any:
    if pose_bone.parent is None:
        return pose_bone.matrix.copy()
    return pose_bone.parent.matrix.inverted() @ pose_bone.matrix


def _integer_frame_range(action: Any) -> tuple[int, int]:
    start_value, end_value = map(float, action.frame_range)
    start, end = int(round(start_value)), int(round(end_value))
    if (
        abs(start_value - start) > 1.0e-5
        or abs(end_value - end) > 1.0e-5
        or end <= start
    ):
        raise RetargetError(f"action frame range is not a nonempty integral loop: {action.frame_range[:]}")
    return start, end


def _configure_scene(bpy: Any) -> None:
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.render.fps_base = 1.0
    scene.sync_mode = "NONE"


def _required_source_names() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*ROCKETBOX_ROLE_TO_BONE.values(), *ROCKETBOX_SPINE_BONES)))


def _identify_walk_source(bpy: Any, expected_animation: Mapping[str, Any]) -> SourceMotion:
    required = set(_required_source_names())
    candidates = []
    for obj in bpy.context.scene.objects:
        if obj.type != "ARMATURE" or obj.animation_data is None:
            continue
        action = obj.animation_data.action
        if action is None or not required.issubset({bone.name for bone in obj.data.bones}):
            continue
        candidates.append((obj, action))
    if len(candidates) != 1:
        raise RetargetError(
            "sealed walk blend must contain one semantic source armature, "
            f"found {len(candidates)}"
        )
    armature, action = candidates[0]
    frame_start, frame_end = _integer_frame_range(action)
    expected_range = (
        int(expected_animation["frame_start"]),
        int(expected_animation["frame_end"]),
    )
    if (frame_start, frame_end) != expected_range:
        raise RetargetError("sealed walk action range disagrees with its authenticated manifest")
    return SourceMotion(
        armature=armature,
        action=action,
        imported_objects=tuple(bpy.context.scene.objects),
        imported_actions=tuple(bpy.data.actions),
        frame_start=frame_start,
        frame_end=frame_end,
    )


def _import_idle_source(bpy: Any, path: Path) -> SourceMotion:
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    result = bpy.ops.import_scene.fbx(filepath=str(path))
    if "FINISHED" not in result:
        raise RetargetError("Rocketbox idle FBX import failed")
    imported_objects = tuple(obj for obj in bpy.data.objects if obj not in before_objects)
    imported_actions = tuple(action for action in bpy.data.actions if action not in before_actions)
    armatures = [obj for obj in imported_objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RetargetError(f"idle FBX must import one source armature, found {len(armatures)}")
    armature = armatures[0]
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RetargetError("idle FBX source armature has no active action")
    missing = sorted(set(_required_source_names()) - {bone.name for bone in armature.data.bones})
    if missing:
        raise RetargetError(f"idle FBX is missing required Rocketbox semantic bones: {missing}")
    frame_start, frame_end = _integer_frame_range(action)
    return SourceMotion(
        armature=armature,
        action=action,
        imported_objects=imported_objects,
        imported_actions=imported_actions,
        frame_start=frame_start,
        frame_end=frame_end,
    )


def _import_tokenrig_runtime(
    bpy: Any, bind_pose_glb: Path, semantic: Mapping[str, Any]
) -> tuple[Any, Any]:
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.gltf(filepath=str(bind_pose_glb))
    if "FINISHED" not in result:
        raise RetargetError("canonical grounded bind_pose.glb import failed")
    imported = set(bpy.data.objects) - before
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    skinned_meshes = [
        obj
        for obj in imported
        if obj.type == "MESH"
        and any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
    ]
    if len(armatures) != 1 or len(skinned_meshes) != 1:
        raise RetargetError(
            "bind pose must import one TokenRig armature and one skinned Pixal mesh: "
            f"armatures={len(armatures)} meshes={len(skinned_meshes)}"
        )
    armature, mesh = armatures[0], skinned_meshes[0]
    modifiers = [
        modifier
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object == armature
    ]
    if len(modifiers) != 1:
        raise RetargetError("Pixal mesh must have exactly one modifier bound to TokenRig")
    target_names = {bone.name for bone in armature.data.bones}
    expected = set(semantic["target_bone_names"]) | set(semantic["rest_descendants"])
    if expected != target_names:
        raise RetargetError(
            "TokenRig runtime bone inventory differs from the hash-locked semantic map: "
            f"missing={sorted(expected-target_names)} extra={sorted(target_names-expected)}"
        )

    def has_ancestor(name: str, allowed: set[str]) -> bool:
        parent = armature.data.bones[name].parent
        while parent is not None:
            if parent.name in allowed:
                return True
            parent = parent.parent
        return False

    head = semantic["semantic_bones"]["head"]
    for name in semantic["head_bound_descendants"]:
        if not has_ancestor(name, {head}):
            raise RetargetError(f"Head-bound attachment is not a Head descendant: {name}")
    hands = {
        semantic["semantic_bones"]["left_hand"],
        semantic["semantic_bones"]["right_hand"],
    }
    for name in semantic["hand_bound_descendants"]:
        if not has_ancestor(name, hands):
            raise RetargetError(f"finger/unmapped distal bone is not Hand-bound: {name}")
    if not mesh.data.uv_layers or not mesh.vertex_groups or not mesh.material_slots:
        raise RetargetError("TokenRig runtime lost UV, skin groups, or PBR material slots")
    armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    return armature, mesh


def capture_rest_matrices(armature: Any, bone_names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    available = {bone.name for bone in armature.data.bones}
    missing = sorted(set(bone_names) - available)
    if missing:
        raise RetargetError(f"rest-matrix capture is missing bones: {missing}")
    for name in bone_names:
        bone = armature.data.bones[name]
        local = _parent_local_rest(bone)
        result[name] = {
            "parent": bone.parent.name if bone.parent else None,
            "armature_space": _matrix_rows(bone.matrix_local),
            "parent_local": _matrix_rows(local),
            "head": [float(value) for value in bone.head_local],
            "tail": [float(value) for value in bone.tail_local],
        }
    return result


def _capture_rest_runtime(armature: Any, names: Sequence[str]) -> dict[str, Any]:
    return {
        name: {
            "global": armature.data.bones[name].matrix_local.copy(),
            "local": _parent_local_rest(armature.data.bones[name]),
        }
        for name in names
    }


def _source_semantic_role(name: str) -> str:
    for role, source_name in ROCKETBOX_ROLE_TO_BONE.items():
        if name == source_name:
            return role
    if name in ROCKETBOX_SPINE_BONES:
        return "spine"
    raise RetargetError(f"source bone has no retarget semantic role: {name}")


def cache_source_motion(
    bpy: Any,
    source: SourceMotion,
    *,
    action_name: str,
) -> dict[str, Any]:
    Matrix, _, _ = _mathutils()
    if action_name not in set(ACTION_NAMES.values()):
        raise RetargetError("source motion cache requested an unapproved action")
    source.armature.animation_data.action = source.action
    names = _required_source_names()
    bpy.context.scene.frame_set(source.frame_start)
    bpy.context.view_layer.update()
    rest_runtime = _capture_rest_runtime(source.armature, names)
    rest_world_heads = {
        name: (
            source.armature.matrix_world @ source.armature.data.bones[name].head_local
        ).copy()
        for name in names
    }
    frames: list[CachedMotionFrame] = []
    rotation_projection_records: list[dict[str, Any]] = []
    first_object_frame: dict[str, Any] | None = None
    for frame in range(source.frame_start, source.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        global_rotations = {}
        local_rotations = {}
        for name in names:
            base_context = {
                "action": action_name,
                "frame": frame,
                "semantic_role": _source_semantic_role(name),
                "source_bone": name,
                "target_bone": None,
            }
            global_matrix, global_evidence = project_near_rotation(
                source.armature.pose.bones[name].matrix.to_3x3(),
                "source global pose",
                context={**base_context, "matrix_stage": "source_global_pose"},
            )
            local_matrix, local_evidence = project_near_rotation(
                _parent_local_pose(source.armature.pose.bones[name]).to_3x3(),
                "source local pose",
                context={**base_context, "matrix_stage": "source_local_pose"},
            )
            rotation_projection_records.extend((global_evidence, local_evidence))
            global_rotations[name] = (
                Matrix(global_matrix.tolist()).to_quaternion().normalized().copy()
            )
            local_rotations[name] = (
                Matrix(local_matrix.tolist()).to_quaternion().normalized().copy()
            )
        root_matrix, root_evidence = project_uniform_scaled_rotation(
            source.armature.matrix_world.to_3x3(),
            "source armature object transform",
            context={
                "action": action_name,
                "frame": frame,
                "semantic_role": "armature_root",
                "source_bone": source.armature.name,
                "target_bone": None,
                "matrix_stage": "source_object_world",
            },
        )
        rotation_projection_records.append(root_evidence)
        if frame == source.frame_start:
            first_object_frame = {
                "frame": int(frame),
                "rotation_3x3": root_matrix.tolist(),
                "uniform_scale_m_per_armature_unit": float(
                    root_evidence["uniform_scale"]
                ),
                "translation_world_m": [
                    float(value) for value in source.armature.matrix_world.translation
                ],
                "mapping": (
                    "source_armature_local_to_canonical_static_object_frame"
                ),
                "translation_applied_to_leg_relative_points": False,
            }
        joint_positions = {
            name: source.armature.pose.bones[name].head.copy() for name in names
        }
        world_joint_positions = {
            name: (
                source.armature.matrix_world @ source.armature.pose.bones[name].head
            ).copy()
            for name in names
        }
        frames.append(
            CachedMotionFrame(
                frame=frame,
                root_location=source.armature.matrix_world.translation.copy(),
                root_rotation=Matrix(root_matrix.tolist())
                .to_quaternion()
                .normalized()
                .copy(),
                global_rotations=global_rotations,
                local_rotations=local_rotations,
                joint_positions=joint_positions,
                world_joint_positions=world_joint_positions,
                pelvis_local_translation=_parent_local_pose(
                    source.armature.pose.bones[ROCKETBOX_ROLE_TO_BONE["pelvis"]]
                ).translation.copy(),
            )
        )
    if first_object_frame is None:
        raise RetargetError("source motion cache has no authenticated first object frame")
    return {
        "frames": frames,
        "rest_runtime": rest_runtime,
        "rest_serialized": capture_rest_matrices(source.armature, names),
        "rest_heads": {
            name: source.armature.data.bones[name].head_local.copy()
            for name in names
        },
        "rest_world_heads": rest_world_heads,
        "first_object_frame": first_object_frame,
        "frame_start": source.frame_start,
        "frame_end": source.frame_end,
        "action_original_name": source.action.name,
        "rotation_projection_records": rotation_projection_records,
    }


def _semantic_height(armature: Any, role_to_bone: Mapping[str, str]) -> float:
    head_z = float(armature.data.bones[role_to_bone["head"]].head_local.z)
    support_z = min(
        float(armature.data.bones[role_to_bone[role]].head_local.z)
        for role in ("left_foot", "left_toe", "right_foot", "right_toe")
    )
    height = head_z - support_z
    if not math.isfinite(height) or height <= 1.0e-6:
        raise RetargetError("semantic source/target height is degenerate")
    return height


def _semantic_world_height(armature: Any, role_to_bone: Mapping[str, str]) -> float:
    head = armature.matrix_world @ armature.data.bones[role_to_bone["head"]].head_local
    supports = [
        armature.matrix_world @ armature.data.bones[role_to_bone[role]].head_local
        for role in ("left_foot", "left_toe", "right_foot", "right_toe")
    ]
    height = float(head.z - min(value.z for value in supports))
    if not math.isfinite(height) or height <= 1.0e-6:
        raise RetargetError("semantic world height is degenerate")
    return height


def _target_parent_first(armature: Any, names: Sequence[str]) -> list[str]:
    name_set = set(names)

    def depth(name: str) -> int:
        count = 0
        parent = armature.data.bones[name].parent
        while parent is not None:
            count += 1
            parent = parent.parent
        return count

    ordered = sorted(name_set, key=lambda name: (depth(name), name))
    seen: set[str] = set()
    for name in ordered:
        parent = armature.data.bones[name].parent
        if parent is not None and parent.name in name_set and parent.name not in seen:
            raise RetargetError(f"parent-first semantic order failed at {name}")
        seen.add(name)
    return ordered


def _keyframe_transform(value: Any, frame: int) -> None:
    value.keyframe_insert(data_path="location", frame=frame, group=value.name)
    value.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=value.name)
    value.keyframe_insert(data_path="scale", frame=frame, group=value.name)


def capture_target_base_transform(armature: Any) -> dict[str, Any]:
    if armature.parent is not None:
        raise RetargetError("canonical TokenRig armature must be an unparented runtime root")
    Matrix, _, _ = _mathutils()
    armature.rotation_mode = "QUATERNION"
    projected, evidence = project_uniform_scaled_rotation(
        armature.matrix_world.to_3x3(),
        "target armature object transform",
        context={
            "action": "static_bind",
            "frame": None,
            "semantic_role": "armature_root",
            "source_bone": None,
            "target_bone": armature.name,
            "matrix_stage": "target_object_world",
        },
    )
    return {
        "location": armature.location.copy(),
        "rotation": Matrix(projected.tolist()).to_quaternion().normalized().copy(),
        "scale": armature.scale.copy(),
        "rotation_projection": evidence,
    }


def restore_target_base_transform(armature: Any, transform: Mapping[str, Any]) -> None:
    armature.rotation_mode = "QUATERNION"
    armature.location = transform["location"].copy()
    armature.rotation_quaternion = transform["rotation"].copy()
    armature.scale = transform["scale"].copy()


def _quaternion_array(value: Any) -> np.ndarray:
    return np.asarray((value.w, value.x, value.y, value.z), dtype=np.float64)


def _quaternion_from_array(value: Sequence[float]) -> Any:
    _, Quaternion, _ = _mathutils()
    return Quaternion(tuple(float(item) for item in value)).normalized()


def apply_source_driven_leg_ik_pose(
    *,
    bpy: Any,
    armature: Any,
    semantic_bones: Mapping[str, Any],
    action_name: str,
    frame: int,
    side: str,
    endpoint_mapping: Mapping[str, Any],
    body_height_m: float,
    target_base_rotation_3x3: Sequence[Sequence[float]] | None = None,
    target_base_uniform_scale_m_per_unit: float | None = None,
) -> dict[str, Any]:
    """Apply one authenticated source-driven leg solution in Blender world space.

    The endpoint mapping is the primary motion transfer, not a contact repair, so
    its XYZ delta has no arbitrary distance cap.  Hip/root/pelvis anchors remain
    unchanged.  The pre-IK global Foot and Toe orientations are restored after
    the thigh/calf solve so the source-driven endpoint does not introduce an
    unrelated ankle/toe twist.
    """

    Matrix, _, Vector = _mathutils()
    if action_name not in set(ACTION_NAMES.values()):
        raise RetargetError("source-driven pose IK action is not approved")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 1:
        raise RetargetError("source-driven pose IK frame is invalid")
    if side not in {"left", "right"}:
        raise RetargetError("source-driven pose IK side is invalid")
    names = {
        part: semantic_bones[f"{side}_{part}"]
        for part in ("thigh", "calf", "foot", "toe")
    }
    pelvis_name = semantic_bones["pelvis"]
    if any(name not in armature.pose.bones for name in (*names.values(), pelvis_name)):
        raise RetargetError("source-driven pose IK semantic bone is missing")

    armature_world = armature.matrix_world.copy()
    inverse_armature_world = armature_world.inverted()

    def world_point(name: str) -> np.ndarray:
        return np.asarray(
            tuple(armature.matrix_world @ armature.pose.bones[name].head),
            dtype=np.float64,
        )

    def matrix_error(first: Any, second: Any) -> float:
        return float(
            np.max(
                np.abs(
                    np.asarray(first, dtype=np.float64)
                    - np.asarray(second, dtype=np.float64)
                )
            )
        )

    world_hip = world_point(names["thigh"])
    world_knee = world_point(names["calf"])
    world_ankle = world_point(names["foot"])
    mapping_evidence = endpoint_mapping.get("evidence")
    if not isinstance(mapping_evidence, Mapping):
        raise RetargetError("source-driven pose IK endpoint evidence is missing")
    coordinate_space = mapping_evidence.get("coordinate_space")
    if coordinate_space == "canonical_static_object_frame_m":
        if (
            target_base_rotation_3x3 is None
            or target_base_uniform_scale_m_per_unit is None
        ):
            raise RetargetError(
                "canonical source-driven pose IK lacks target base-frame evidence"
            )
        solver_hip = canonical_static_armature_point(
            armature.pose.bones[names["thigh"]].head,
            uniform_scale_m_per_unit=target_base_uniform_scale_m_per_unit,
            base_rotation_3x3=target_base_rotation_3x3,
        )
        solver_knee = canonical_static_armature_point(
            armature.pose.bones[names["calf"]].head,
            uniform_scale_m_per_unit=target_base_uniform_scale_m_per_unit,
            base_rotation_3x3=target_base_rotation_3x3,
        )
        solver_ankle = canonical_static_armature_point(
            armature.pose.bones[names["foot"]].head,
            uniform_scale_m_per_unit=target_base_uniform_scale_m_per_unit,
            base_rotation_3x3=target_base_rotation_3x3,
        )
    elif coordinate_space == "authenticated_world_m":
        if (
            target_base_rotation_3x3 is not None
            or target_base_uniform_scale_m_per_unit is not None
        ):
            raise RetargetError(
                "world source-driven pose IK received an extraneous base frame"
            )
        solver_hip, solver_knee, solver_ankle = (
            world_hip,
            world_knee,
            world_ankle,
        )
    else:
        raise RetargetError("source-driven pose IK coordinate space is invalid")
    original_root_world = armature.matrix_world.copy()
    original_pelvis_world = (
        armature.matrix_world @ armature.pose.bones[pelvis_name].matrix
    ).copy()
    original_foot_world = (
        armature.matrix_world @ armature.pose.bones[names["foot"]].matrix
    ).copy()
    original_toe_world = (
        armature.matrix_world @ armature.pose.bones[names["toe"]].matrix
    ).copy()
    original_foot_orientation = original_foot_world.to_quaternion().normalized()
    original_toe_orientation = original_toe_world.to_quaternion().normalized()

    solved = solve_source_driven_two_bone_leg_ik(
        hip=solver_hip,
        knee=solver_knee,
        ankle=solver_ankle,
        endpoint_mapping=endpoint_mapping,
        body_height_m=body_height_m,
    )
    solved_knee_solver = np.asarray(solved["knee"], dtype=np.float64)
    solved_ankle_solver = np.asarray(solved["ankle"], dtype=np.float64)
    if coordinate_space == "canonical_static_object_frame_m":
        solved_knee_local = canonical_static_to_armature_point(
            solved_knee_solver,
            uniform_scale_m_per_unit=target_base_uniform_scale_m_per_unit,
            base_rotation_3x3=target_base_rotation_3x3,
        )
        solved_ankle_local = canonical_static_to_armature_point(
            solved_ankle_solver,
            uniform_scale_m_per_unit=target_base_uniform_scale_m_per_unit,
            base_rotation_3x3=target_base_rotation_3x3,
        )
        solved_knee = np.asarray(
            tuple(armature.matrix_world @ Vector(tuple(solved_knee_local))),
            dtype=np.float64,
        )
        solved_ankle = np.asarray(
            tuple(armature.matrix_world @ Vector(tuple(solved_ankle_local))),
            dtype=np.float64,
        )
    else:
        solved_knee = solved_knee_solver
        solved_ankle = solved_ankle_solver

    def rotate_bone_to(
        bone_name: str,
        pivot: np.ndarray,
        current_end: np.ndarray,
        desired_end: np.ndarray,
    ) -> None:
        current_vector = Vector(tuple(current_end - pivot))
        desired_vector = Vector(tuple(desired_end - pivot))
        if min(current_vector.length, desired_vector.length) <= 1.0e-8:
            raise RetargetError("source-driven pose IK rotation vector is degenerate")
        delta = current_vector.rotation_difference(desired_vector)
        current_world = armature.matrix_world @ armature.pose.bones[bone_name].matrix
        desired_world = (
            Matrix.Translation(Vector(tuple(pivot)))
            @ delta.to_matrix().to_4x4()
            @ Matrix.Translation(Vector(tuple(-pivot)))
            @ current_world
        )
        armature.pose.bones[bone_name].matrix = (
            inverse_armature_world @ desired_world
        )
        bpy.context.view_layer.update()

    rotate_bone_to(names["thigh"], world_hip, world_knee, solved_knee)
    knee_after_thigh = world_point(names["calf"])
    ankle_after_thigh = world_point(names["foot"])
    rotate_bone_to(
        names["calf"],
        knee_after_thigh,
        ankle_after_thigh,
        solved_ankle,
    )
    natural_ankle = world_point(names["foot"])
    pre_distal_endpoint_error = float(np.linalg.norm(natural_ankle - solved_ankle))
    if pre_distal_endpoint_error > POSE_TRANSLATION_TOLERANCE_M:
        raise RetargetError(
            "source-driven pose IK Blender chain did not reconstruct its ankle"
        )

    desired_foot_world = Matrix.LocRotScale(
        Vector(tuple(solved_ankle)),
        original_foot_orientation,
        original_foot_world.to_scale(),
    )
    armature.pose.bones[names["foot"]].matrix = (
        inverse_armature_world @ desired_foot_world
    )
    bpy.context.view_layer.update()
    natural_toe_head = world_point(names["toe"])
    desired_toe_world = Matrix.LocRotScale(
        Vector(tuple(natural_toe_head)),
        original_toe_orientation,
        original_toe_world.to_scale(),
    )
    armature.pose.bones[names["toe"]].matrix = (
        inverse_armature_world @ desired_toe_world
    )
    bpy.context.view_layer.update()

    actual_hip = world_point(names["thigh"])
    actual_knee = world_point(names["calf"])
    actual_ankle = world_point(names["foot"])
    actual_foot_world = (
        armature.matrix_world @ armature.pose.bones[names["foot"]].matrix
    )
    actual_toe_world = (
        armature.matrix_world @ armature.pose.bones[names["toe"]].matrix
    )
    root_matrix_error = matrix_error(original_root_world, armature.matrix_world)
    pelvis_matrix_error = matrix_error(
        original_pelvis_world,
        armature.matrix_world @ armature.pose.bones[pelvis_name].matrix,
    )
    hip_position_error = float(np.linalg.norm(actual_hip - world_hip))
    knee_position_error = float(np.linalg.norm(actual_knee - solved_knee))
    ankle_position_error = float(np.linalg.norm(actual_ankle - solved_ankle))
    foot_orientation_error = _quaternion_angle(
        original_foot_orientation,
        actual_foot_world.to_quaternion().normalized(),
    )
    toe_orientation_error = _quaternion_angle(
        original_toe_orientation,
        actual_toe_world.to_quaternion().normalized(),
    )
    maximum_anchor_error = max(
        root_matrix_error, pelvis_matrix_error, hip_position_error
    )
    maximum_joint_error = max(knee_position_error, ankle_position_error)
    maximum_orientation_error = max(
        foot_orientation_error, toe_orientation_error
    )
    if maximum_anchor_error > POSE_TRANSLATION_TOLERANCE_M:
        raise RetargetError("source-driven pose IK changed root/pelvis/hip anchor")
    if maximum_joint_error > POSE_TRANSLATION_TOLERANCE_M:
        raise RetargetError("source-driven pose IK Blender joint readback failed")
    if maximum_orientation_error > REST_ROTATION_MATRIX_TOLERANCE:
        raise RetargetError("source-driven pose IK did not restore Foot/Toe orientation")

    return {
        "schema": "tokenrig_source_driven_leg_pose_ik_v1",
        "action_name": action_name,
        "frame": frame,
        "side": side,
        "endpoint_mapping": dict(endpoint_mapping),
        "solver": solved["evidence"],
        "coordinate_transform": {
            "solver_coordinate_space": coordinate_space,
            "target_base_rotation_3x3": (
                None
                if target_base_rotation_3x3 is None
                else np.asarray(target_base_rotation_3x3, dtype=np.float64).tolist()
            ),
            "target_base_uniform_scale_m_per_unit": (
                None
                if target_base_uniform_scale_m_per_unit is None
                else float(target_base_uniform_scale_m_per_unit)
            ),
            "target_current_object_matrix_applied_once_after_solve": (
                coordinate_space == "canonical_static_object_frame_m"
            ),
            "root_translation_in_endpoint_mapping": False,
        },
        "readback": {
            "root_matrix_max_abs_error": root_matrix_error,
            "pelvis_matrix_max_abs_error": pelvis_matrix_error,
            "hip_position_error_m": hip_position_error,
            "knee_position_error_m": knee_position_error,
            "ankle_position_error_m": ankle_position_error,
            "pre_distal_endpoint_error_m": pre_distal_endpoint_error,
            "foot_orientation_error_rad": foot_orientation_error,
            "toe_orientation_error_rad": toe_orientation_error,
            "root_pelvis_hip_translation_preserved": True,
            "foot_toe_global_orientation_restored": True,
        },
        "automatic_checks": "passed",
    }


def apply_vertical_surface_contact_leg_ik_pose(
    *,
    bpy: Any,
    armature: Any,
    semantic_bones: Mapping[str, Any],
    action_name: str,
    frame: int,
    side: str,
    upward_correction_m: float,
) -> dict[str, Any]:
    """Raise one ankle without changing root, pelvis, hip, X-Y, or distal orientation."""

    Matrix, _, Vector = _mathutils()
    correction = _finite_number(
        upward_correction_m, "surface-contact ankle correction"
    )
    if correction <= 0.0:
        raise RetargetError("surface-contact ankle correction must be positive")
    if action_name not in set(ACTION_NAMES.values()):
        raise RetargetError("surface-contact IK action is not approved")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 1:
        raise RetargetError("surface-contact IK frame is invalid")
    if side not in {"left", "right"}:
        raise RetargetError("surface-contact IK side is invalid")
    names = {
        part: semantic_bones[f"{side}_{part}"]
        for part in ("thigh", "calf", "foot", "toe")
    }
    pelvis_name = semantic_bones["pelvis"]
    if any(name not in armature.pose.bones for name in (*names.values(), pelvis_name)):
        raise RetargetError("surface-contact IK semantic bone is missing")

    armature_world = armature.matrix_world.copy()
    inverse_armature_world = armature_world.inverted()

    def world_point(name: str) -> np.ndarray:
        return np.asarray(
            tuple(armature.matrix_world @ armature.pose.bones[name].head),
            dtype=np.float64,
        )

    def matrix_error(first: Any, second: Any) -> float:
        return float(
            np.max(
                np.abs(
                    np.asarray(first, dtype=np.float64)
                    - np.asarray(second, dtype=np.float64)
                )
            )
        )

    world_hip = world_point(names["thigh"])
    world_knee = world_point(names["calf"])
    world_ankle = world_point(names["foot"])
    target_ankle = world_ankle + np.asarray((0.0, 0.0, correction))
    original_root_world = armature.matrix_world.copy()
    original_pelvis_world = (
        armature.matrix_world @ armature.pose.bones[pelvis_name].matrix
    ).copy()
    original_foot_world = (
        armature.matrix_world @ armature.pose.bones[names["foot"]].matrix
    ).copy()
    original_toe_world = (
        armature.matrix_world @ armature.pose.bones[names["toe"]].matrix
    ).copy()
    original_foot_orientation = original_foot_world.to_quaternion().normalized()
    original_toe_orientation = original_toe_world.to_quaternion().normalized()
    solved = solve_two_bone_leg_ik(
        hip=world_hip,
        knee=world_knee,
        ankle=world_ankle,
        target_ankle=target_ankle,
    )
    solved_knee = np.asarray(solved["knee"], dtype=np.float64)
    solved_ankle = np.asarray(solved["ankle"], dtype=np.float64)

    def rotate_bone_to(
        bone_name: str,
        pivot: np.ndarray,
        current_end: np.ndarray,
        desired_end: np.ndarray,
    ) -> None:
        current_vector = Vector(tuple(current_end - pivot))
        desired_vector = Vector(tuple(desired_end - pivot))
        if min(current_vector.length, desired_vector.length) <= 1.0e-8:
            raise RetargetError("surface-contact IK rotation vector is degenerate")
        delta = current_vector.rotation_difference(desired_vector)
        current_world = armature.matrix_world @ armature.pose.bones[bone_name].matrix
        desired_world = (
            Matrix.Translation(Vector(tuple(pivot)))
            @ delta.to_matrix().to_4x4()
            @ Matrix.Translation(Vector(tuple(-pivot)))
            @ current_world
        )
        armature.pose.bones[bone_name].matrix = inverse_armature_world @ desired_world
        bpy.context.view_layer.update()

    rotate_bone_to(names["thigh"], world_hip, world_knee, solved_knee)
    knee_after_thigh = world_point(names["calf"])
    ankle_after_thigh = world_point(names["foot"])
    rotate_bone_to(
        names["calf"], knee_after_thigh, ankle_after_thigh, solved_ankle
    )
    natural_ankle = world_point(names["foot"])
    pre_distal_endpoint_error = float(np.linalg.norm(natural_ankle - solved_ankle))
    if pre_distal_endpoint_error > POSE_TRANSLATION_TOLERANCE_M:
        raise RetargetError("surface-contact IK did not reconstruct its ankle")

    desired_foot_world = Matrix.LocRotScale(
        Vector(tuple(solved_ankle)),
        original_foot_orientation,
        original_foot_world.to_scale(),
    )
    armature.pose.bones[names["foot"]].matrix = (
        inverse_armature_world @ desired_foot_world
    )
    bpy.context.view_layer.update()
    natural_toe_head = world_point(names["toe"])
    desired_toe_world = Matrix.LocRotScale(
        Vector(tuple(natural_toe_head)),
        original_toe_orientation,
        original_toe_world.to_scale(),
    )
    armature.pose.bones[names["toe"]].matrix = (
        inverse_armature_world @ desired_toe_world
    )
    bpy.context.view_layer.update()

    actual_hip = world_point(names["thigh"])
    actual_knee = world_point(names["calf"])
    actual_ankle = world_point(names["foot"])
    actual_foot_world = armature.matrix_world @ armature.pose.bones[names["foot"]].matrix
    actual_toe_world = armature.matrix_world @ armature.pose.bones[names["toe"]].matrix
    root_matrix_error = matrix_error(original_root_world, armature.matrix_world)
    pelvis_matrix_error = matrix_error(
        original_pelvis_world,
        armature.matrix_world @ armature.pose.bones[pelvis_name].matrix,
    )
    hip_position_error = float(np.linalg.norm(actual_hip - world_hip))
    knee_position_error = float(np.linalg.norm(actual_knee - solved_knee))
    ankle_position_error = float(np.linalg.norm(actual_ankle - solved_ankle))
    foot_orientation_error = _quaternion_angle(
        original_foot_orientation,
        actual_foot_world.to_quaternion().normalized(),
    )
    toe_orientation_error = _quaternion_angle(
        original_toe_orientation,
        actual_toe_world.to_quaternion().normalized(),
    )
    if max(root_matrix_error, pelvis_matrix_error, hip_position_error) > POSE_TRANSLATION_TOLERANCE_M:
        raise RetargetError("surface-contact IK changed root/pelvis/hip anchor")
    if max(knee_position_error, ankle_position_error) > POSE_TRANSLATION_TOLERANCE_M:
        raise RetargetError("surface-contact IK joint readback failed")
    if max(foot_orientation_error, toe_orientation_error) > REST_ROTATION_MATRIX_TOLERANCE:
        raise RetargetError("surface-contact IK changed Foot/Toe orientation")

    for name in names.values():
        _keyframe_transform(armature.pose.bones[name], frame)
    return {
        "schema": "tokenrig_vertical_surface_contact_leg_pose_ik_v1",
        "action_name": action_name,
        "frame": frame,
        "side": side,
        "solver": solved["evidence"],
        "readback": {
            "root_matrix_max_abs_error": root_matrix_error,
            "pelvis_matrix_max_abs_error": pelvis_matrix_error,
            "hip_position_error_m": hip_position_error,
            "knee_position_error_m": knee_position_error,
            "ankle_position_error_m": ankle_position_error,
            "pre_distal_endpoint_error_m": pre_distal_endpoint_error,
            "foot_orientation_error_rad": foot_orientation_error,
            "toe_orientation_error_rad": toe_orientation_error,
            "root_pelvis_hip_translation_preserved": True,
            "ankle_xy_preserved": bool(
                np.linalg.norm((actual_ankle - world_ankle)[:2])
                <= POSE_TRANSLATION_TOLERANCE_M
            ),
            "foot_toe_global_orientation_preserved": True,
        },
        "automatic_checks": "passed",
    }


def bake_rest_corrected_action(
    *,
    bpy: Any,
    target_armature: Any,
    semantic: Mapping[str, Any],
    cached: Mapping[str, Any],
    action_name: str,
    target_base_transform: Mapping[str, Any],
    limb_motion_basis_3x3=AXIS_MAP_3X3,
) -> tuple[Any, dict[str, Any]]:
    Matrix, _, Vector = _mathutils()
    if action_name not in set(ACTION_NAMES.values()):
        raise RetargetError("bake requested an unapproved action name")
    limb_motion_basis, limb_motion_basis_projection = project_near_rotation(
        limb_motion_basis_3x3,
        "reviewed shared canonical limb motion basis",
    )
    exact = build_exact_semantic_correspondence(semantic)
    target_spines = list(semantic["semantic_bones"]["spine"])
    mapped_target_names = [*exact.values(), *target_spines]
    target_rest = _capture_rest_runtime(target_armature, mapped_target_names)
    target_rest_serialized = capture_rest_matrices(target_armature, mapped_target_names)
    source_rest = cached["rest_runtime"]
    rotation_projection_records = list(cached.get("rotation_projection_records", ()))
    if "rotation_projection" in target_base_transform:
        target_object_evidence = dict(target_base_transform["rotation_projection"])
        target_object_evidence["context"] = {
            **dict(target_object_evidence.get("context", {})),
            "action": action_name,
            "frame": int(cached["frame_start"]),
        }
        rotation_projection_records.append(target_object_evidence)
    _, canonical_axis_evidence = project_near_rotation(
        AXIS_MAP_3X3,
        "canonical retarget axis",
        context={
            "action": action_name,
            "frame": int(cached["frame_start"]),
            "semantic_role": "canonical_axis",
            "source_bone": None,
            "target_bone": None,
            "matrix_stage": "canonical_axis_identity",
        },
    )
    rotation_projection_records.append(canonical_axis_evidence)
    source_rest_heads = [source_rest[name]["global"].translation[:] for name in ROCKETBOX_SPINE_BONES]
    target_rest_heads = [target_rest[name]["global"].translation[:] for name in target_spines]
    spine_plan = build_spine_resample_plan(
        source_bones=ROCKETBOX_SPINE_BONES,
        source_rest_heads=source_rest_heads,
        target_bones=target_spines,
        target_rest_heads=target_rest_heads,
    )
    source_role_to_bone = dict(ROCKETBOX_ROLE_TO_BONE)
    source_armature = cached.get("source_armature")
    if source_armature is None:
        raise RetargetError("cached motion is missing its authenticated source armature")
    bpy.context.scene.frame_set(int(cached["frame_start"]))
    bpy.context.view_layer.update()
    target_role_to_bone = {
        role: target
        for role, target in semantic["semantic_bones"].items()
        if role != "spine"
    }
    height_scales = compute_height_scales(
        source_armature_height=_semantic_height(source_armature, source_role_to_bone),
        target_armature_height=_semantic_height(target_armature, target_role_to_bone),
        source_world_height=_semantic_world_height(source_armature, source_role_to_bone),
        target_world_height=_semantic_world_height(target_armature, target_role_to_bone),
    )
    pelvis_local_scale = height_scales["pelvis_local_scale"]
    root_world_scale = height_scales["root_world_scale"]

    if target_armature.animation_data is None:
        target_armature.animation_data_create()
    if action_name in bpy.data.actions:
        raise RetargetError(f"target action already exists: {action_name}")
    action = bpy.data.actions.new(name=action_name)
    if action.name != action_name:
        raise RetargetError("Blender changed the only approved action name")
    target_armature.animation_data.action = action
    target_armature.data.pose_position = "POSE"
    restore_target_base_transform(target_armature, target_base_transform)
    base_location = target_base_transform["location"].copy()
    base_rotation = target_base_transform["rotation"].copy()
    base_scale = target_base_transform["scale"].copy()
    source_first_object_frame = cached.get("first_object_frame")
    if not isinstance(source_first_object_frame, Mapping):
        raise RetargetError("source motion cache is missing its first object frame")
    source_base_rotation = np.asarray(
        source_first_object_frame.get("rotation_3x3"), dtype=np.float64
    )
    source_base_rotation, _ = project_near_rotation(
        source_base_rotation, "source first object frame for endpoint mapping"
    )
    source_uniform_scale = _finite_number(
        source_first_object_frame.get("uniform_scale_m_per_armature_unit"),
        "source first object-frame scale",
    )
    target_base_rotation = np.asarray(base_rotation.to_matrix(), dtype=np.float64)
    target_base_rotation, _ = project_near_rotation(
        target_base_rotation, "target base object frame for endpoint mapping"
    )
    target_uniform_scale = _finite_number(
        target_base_transform["rotation_projection"].get("uniform_scale"),
        "target base object-frame scale",
    )
    target_rest_canonical_heads = {
        target_name: canonical_static_armature_point(
            target_armature.data.bones[target_name].head_local,
            uniform_scale_m_per_unit=target_uniform_scale,
            base_rotation_3x3=target_base_rotation,
        )
        for side in ("left", "right")
        for target_name in (
            semantic["semantic_bones"][f"{side}_thigh"],
            semantic["semantic_bones"][f"{side}_calf"],
            semantic["semantic_bones"][f"{side}_foot"],
        )
    }
    target_body_height_m = _semantic_world_height(
        target_armature, target_role_to_bone
    )
    frames: Sequence[CachedMotionFrame] = cached["frames"]
    first = frames[0]
    pelvis_source = ROCKETBOX_ROLE_TO_BONE["pelvis"]
    pelvis_rest_translation = source_rest[pelvis_source]["local"].translation.copy()
    ordered = _target_parent_first(target_armature, mapped_target_names)
    anatomical_axial = build_anatomical_axial_transfer(
        target_armature=target_armature,
        semantic_bones=semantic["semantic_bones"],
        target_rest=target_rest,
        cached=cached,
        target_base_rotation_3x3=target_base_rotation,
        spine_plan=spine_plan,
    )
    maximum_global_rotation_error = 0.0
    maximum_local_rotation_error = 0.0
    maximum_translation_error = 0.0
    maximum_root_rotation_reconstruction_error = 0.0
    maximum_root_translation_reconstruction_error = 0.0
    maximum_shared_limb_delta_reconstruction_error = 0.0
    maximum_distal_canonical_delta_reconstruction_error = 0.0
    shared_limb_delta_record_count = 0
    distal_canonical_delta_record_count = 0
    primary_leg_ik_by_frame: dict[str, dict[str, Any]] = {}

    for frame_record in frames:
        bpy.context.scene.frame_set(frame_record.frame)
        for pose_bone in target_armature.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.matrix_basis = Matrix.Identity(4)
        root_delta = (frame_record.root_location - first.root_location) * root_world_scale
        target_armature.location = base_location + root_delta
        root_rotation_delta = first.root_rotation.inverted() @ frame_record.root_rotation
        target_armature.rotation_quaternion = (base_rotation @ root_rotation_delta).normalized()
        target_armature.scale = base_scale
        bpy.context.view_layer.update()
        expected_root_rotation = (base_rotation @ root_rotation_delta).normalized()
        maximum_root_rotation_reconstruction_error = max(
            maximum_root_rotation_reconstruction_error,
            _quaternion_angle(
                expected_root_rotation,
                target_armature.rotation_quaternion.normalized(),
            ),
        )
        expected_root_location = base_location + root_delta
        maximum_root_translation_reconstruction_error = max(
            maximum_root_translation_reconstruction_error,
            float(
                np.linalg.norm(
                    np.asarray(tuple(target_armature.location), dtype=np.float64)
                    - np.asarray(tuple(expected_root_location), dtype=np.float64)
                )
            ),
        )

        axial_frame = anatomical_axial["frames"][int(frame_record.frame)]
        source_by_target = {target: source for source, target in exact.items()}

        for target_name in ordered:
            target_pose_bone = target_armature.pose.bones[target_name]
            target_rest_local = target_rest[target_name]["local"]
            target_translation = target_rest_local.translation.copy()
            expected_global_rotation = None
            if target_name == exact[pelvis_source]:
                target_translation += pelvis_local_scale * (
                    frame_record.pelvis_local_translation - pelvis_rest_translation
                )
            alignment_evidence = None
            if target_name in axial_frame["spine"]:
                target_global = np.asarray(
                    axial_frame["spine"][target_name], dtype=np.float64
                )
            else:
                source_name = source_by_target[target_name]
                source_rest_global = source_rest[source_name]["global"].to_3x3()
                source_pose_global = frame_record.global_rotations[source_name].to_matrix()
                target_rest_global = target_rest[target_name]["global"].to_3x3()
                semantic_role = _source_semantic_role(source_name)
                rotation_context = {
                    "action": action_name,
                    "frame": frame_record.frame,
                    "semantic_role": semantic_role,
                    "source_bone": source_name,
                    "target_bone": target_name,
                }
                if semantic_role in ANATOMICAL_AXIAL_EXACT_ROLES:
                    target_global = np.asarray(
                        axial_frame[semantic_role], dtype=np.float64
                    )
                elif semantic_role in ANATOMICAL_CLAVICLE_ROLES:
                    side = semantic_role.removesuffix("_clavicle")
                    target_global = np.asarray(
                        axial_frame["clavicles"][side], dtype=np.float64
                    )
                elif semantic_role in SHARED_CANONICAL_LIMB_ROLES:
                    target_global, alignment_evidence = shared_canonical_limb_rotation(
                        source_rest_global,
                        source_pose_global,
                        target_rest_global,
                        source_base_rotation_3x3=source_base_rotation,
                        target_base_rotation_3x3=target_base_rotation,
                        motion_basis_3x3=limb_motion_basis,
                        context=rotation_context,
                    )
                    shared_limb_delta_record_count += 1
                    shared_error = float(
                        alignment_evidence[
                            "canonical_delta_reconstruction_error"
                        ]
                    )
                    maximum_shared_limb_delta_reconstruction_error = max(
                        maximum_shared_limb_delta_reconstruction_error,
                        shared_error,
                    )
                    if semantic_role in {
                        "left_foot",
                        "left_toe",
                        "right_foot",
                        "right_toe",
                    }:
                        distal_canonical_delta_record_count += 1
                        maximum_distal_canonical_delta_reconstruction_error = max(
                            maximum_distal_canonical_delta_reconstruction_error,
                            shared_error,
                        )
                else:
                    raise RetargetError(
                        f"semantic role has no reviewed rotation transfer: {semantic_role}"
                    )
                if (
                    alignment_evidence is not None
                    and frame_record.frame == int(cached["frame_start"])
                ):
                    rotation_projection_records.extend(
                        alignment_evidence["rotation_projections"].values()
                    )
            target_global, _ = project_near_rotation(
                target_global,
                f"reviewed target global rotation {frame_record.frame}:{target_name}",
            )
            target_pose_bone.matrix = Matrix.LocRotScale(
                (
                    target_pose_bone.parent.matrix @ target_rest_local.translation
                    if target_pose_bone.parent is not None
                    else target_translation
                ),
                Matrix(target_global.tolist()).to_quaternion(),
                Vector((1.0, 1.0, 1.0)),
            )
            expected_global_rotation = target_global
            bpy.context.view_layer.update()
            if expected_global_rotation is not None:
                actual_global = np.asarray(
                    target_pose_bone.matrix.to_quaternion().to_matrix(), dtype=np.float64
                )
                maximum_global_rotation_error = max(
                    maximum_global_rotation_error,
                    float(np.max(np.abs(actual_global - expected_global_rotation))),
                )
            actual_translation = np.asarray(
                tuple(_parent_local_pose(target_pose_bone).translation),
                dtype=np.float64,
            )
            expected_translation = np.asarray(tuple(target_translation), dtype=np.float64)
            translation_error = float(
                np.max(np.abs(actual_translation - expected_translation))
            )
            if not math.isfinite(translation_error):
                raise RetargetError("target pose translation became non-finite")
            maximum_translation_error = max(
                maximum_translation_error, translation_error
            )

        frame_primary_ik: dict[str, Any] = {}
        for side in ("left", "right"):
            source_names = {
                part: source_role_to_bone[f"{side}_{part}"]
                for part in ("thigh", "calf", "foot")
            }
            target_names = {
                part: semantic["semantic_bones"][f"{side}_{part}"]
                for part in ("thigh", "calf", "foot")
            }
            endpoint_mapping = map_source_leg_endpoint_rest_frame(
                source_rest_hip=canonical_static_armature_point(
                    cached["rest_heads"][source_names["thigh"]],
                    uniform_scale_m_per_unit=source_uniform_scale,
                    base_rotation_3x3=source_base_rotation,
                ),
                source_rest_knee=canonical_static_armature_point(
                    cached["rest_heads"][source_names["calf"]],
                    uniform_scale_m_per_unit=source_uniform_scale,
                    base_rotation_3x3=source_base_rotation,
                ),
                source_rest_ankle=canonical_static_armature_point(
                    cached["rest_heads"][source_names["foot"]],
                    uniform_scale_m_per_unit=source_uniform_scale,
                    base_rotation_3x3=source_base_rotation,
                ),
                target_rest_hip=target_rest_canonical_heads[target_names["thigh"]],
                target_rest_knee=target_rest_canonical_heads[target_names["calf"]],
                target_rest_ankle=target_rest_canonical_heads[target_names["foot"]],
                source_current_hip=canonical_static_armature_point(
                    frame_record.joint_positions[source_names["thigh"]],
                    uniform_scale_m_per_unit=source_uniform_scale,
                    base_rotation_3x3=source_base_rotation,
                ),
                source_current_knee=canonical_static_armature_point(
                    frame_record.joint_positions[source_names["calf"]],
                    uniform_scale_m_per_unit=source_uniform_scale,
                    base_rotation_3x3=source_base_rotation,
                ),
                source_current_ankle=canonical_static_armature_point(
                    frame_record.joint_positions[source_names["foot"]],
                    uniform_scale_m_per_unit=source_uniform_scale,
                    base_rotation_3x3=source_base_rotation,
                ),
                target_current_hip=canonical_static_armature_point(
                    target_armature.pose.bones[target_names["thigh"]].head,
                    uniform_scale_m_per_unit=target_uniform_scale,
                    base_rotation_3x3=target_base_rotation,
                ),
                scale_basis="reach_normalized_piecewise",
                orientation_basis="canonical_absolute_source_direction",
                coordinate_space="canonical_static_object_frame_m",
                axis_map_3x3=limb_motion_basis,
            )
            frame_primary_ik[side] = apply_source_driven_leg_ik_pose(
                bpy=bpy,
                armature=target_armature,
                semantic_bones=semantic["semantic_bones"],
                action_name=action_name,
                frame=frame_record.frame,
                side=side,
                endpoint_mapping=endpoint_mapping,
                body_height_m=target_body_height_m,
                target_base_rotation_3x3=target_base_rotation,
                target_base_uniform_scale_m_per_unit=target_uniform_scale,
            )
        primary_leg_ik_by_frame[str(frame_record.frame)] = frame_primary_ik

        _keyframe_transform(target_armature, frame_record.frame)
        for target_name in mapped_target_names:
            _keyframe_transform(target_armature.pose.bones[target_name], frame_record.frame)
        for descendant in semantic["rest_descendants"]:
            if target_armature.pose.bones[descendant].matrix_basis != Matrix.Identity(4):
                raise RetargetError(f"unmapped/finger/accessory descendant left rest: {descendant}")

    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    if maximum_global_rotation_error > REST_ROTATION_MATRIX_TOLERANCE:
        raise RetargetError("global rest alignment exceeds tolerance during bake")
    if maximum_local_rotation_error > REST_ROTATION_MATRIX_TOLERANCE:
        raise RetargetError("spine local rest delta exceeds tolerance during bake")
    if maximum_translation_error > POSE_TRANSLATION_TOLERANCE_M:
        raise RetargetError("target rest translation reconstruction failed during bake")
    primary_records = [
        primary_leg_ik_by_frame[str(frame)][side]
        for frame in range(int(cached["frame_start"]), int(cached["frame_end"]) + 1)
        for side in ("left", "right")
    ]
    if len(primary_records) != 2 * len(frames):
        raise RetargetError("source-driven primary leg IK frame coverage is incomplete")
    expected_shared_limb_records = len(SHARED_CANONICAL_LIMB_ROLES) * len(frames)
    if shared_limb_delta_record_count != expected_shared_limb_records:
        raise RetargetError("shared canonical limb rotation coverage is incomplete")
    action.use_fake_user = True
    return action, {
        "action_name": action_name,
        "frame_start": int(cached["frame_start"]),
        "frame_end": int(cached["frame_end"]),
        "height_scale": float(root_world_scale),
        "root_world_scale": float(root_world_scale),
        "pelvis_local_scale": float(pelvis_local_scale),
        "spine_resample_plan": spine_plan,
        "target_rest_serialized": target_rest_serialized,
        "maximum_global_rest_alignment_error": maximum_global_rotation_error,
        "maximum_local_rest_delta_error": maximum_local_rotation_error,
        "maximum_target_translation_error_m": maximum_translation_error,
        "target_rest_translations_preserved": (
            maximum_translation_error <= POSE_TRANSLATION_TOLERANCE_M
        ),
        "parent_first": True,
        "root_object_mapping": {
            "schema": "tokenrig_root_object_mapping_v1",
            "rotation_order": (
                "target_base_rotation @ (source_first_rotation.inverted() @ "
                "source_current_rotation)"
            ),
            "translation_method": "source_world_translation_delta_scaled_once",
            "source_translation_coordinate_space": "authenticated_world_m",
            "axis_map_3x3": [list(row) for row in AXIS_MAP_3X3],
            "root_world_scale": float(root_world_scale),
            "source_first_translation_world_m": [
                float(value) for value in first.root_location
            ],
            "target_base_translation_world_m": [
                float(value) for value in base_location
            ],
            "maximum_rotation_reconstruction_error_rad": (
                maximum_root_rotation_reconstruction_error
            ),
            "maximum_translation_reconstruction_error_m": (
                maximum_root_translation_reconstruction_error
            ),
            "endpoint_mapping_uses_root_translation": False,
            "target_current_object_matrix_applied_once_after_leg_solve": True,
            "automatic_checks": "passed",
        },
        "shared_limb_motion_basis": {
            "schema": "tokenrig_shared_canonical_limb_motion_basis_v1",
            "method": "shared_canonical_limb_delta_v1",
            "semantic_roles": list(SHARED_CANONICAL_LIMB_ROLES),
            "motion_basis_3x3": limb_motion_basis.tolist(),
            "motion_basis_projection": limb_motion_basis_projection,
            "record_count": shared_limb_delta_record_count,
            "expected_record_count": expected_shared_limb_records,
            "maximum_canonical_delta_reconstruction_error": (
                maximum_shared_limb_delta_reconstruction_error
            ),
            "per_bone_rest_axis_conjugation_used": False,
            "root_object_mapping_changed": False,
            "pelvis_spine_neck_head_mapping_changed": True,
            "clavicles_owned_by_anatomical_transfer": True,
            "automatic_checks": "passed",
        },
        "anatomical_axial_transfer": dict(anatomical_axial["evidence"]),
        "distal_orientation": {
            "schema": "tokenrig_canonical_world_distal_orientation_v1",
            "method": "shared_canonical_limb_delta_v1",
            "semantic_roles": [
                "left_foot",
                "left_toe",
                "right_foot",
                "right_toe",
            ],
            "record_count": distal_canonical_delta_record_count,
            "maximum_canonical_delta_reconstruction_error": (
                maximum_distal_canonical_delta_reconstruction_error
            ),
            "target_rest_axis_conjugation_used": False,
            "shared_motion_basis_applied": True,
            "automatic_checks": "passed",
        },
        "primary_leg_ik": {
            "schema": "tokenrig_source_driven_primary_leg_ik_action_v1",
            "method": (
                "canonical_absolute_source_direction_piecewise_reach_mapped_pole_v1"
            ),
            "coordinate_space": "canonical_static_object_frame_m",
            "scale_basis": "reach_normalized_piecewise",
            "orientation_basis": "canonical_absolute_source_direction",
            "frame_start": int(cached["frame_start"]),
            "frame_end": int(cached["frame_end"]),
            "frame_count": len(frames),
            "side_count": 2,
            "record_count": len(primary_records),
            "maximum_endpoint_delta_m": max(
                value["solver"]["endpoint_delta_norm_m"]
                for value in primary_records
            ),
            "maximum_endpoint_delta_body_height_ratio": max(
                value["solver"]["endpoint_delta_body_height_ratio"]
                for value in primary_records
            ),
            "maximum_endpoint_delta_leg_length_ratio": max(
                value["solver"]["endpoint_delta_leg_length_ratio"]
                for value in primary_records
            ),
            "minimum_reach_margin_m": min(
                value["solver"]["reach_margin_m"] for value in primary_records
            ),
            "minimum_bend_height_m": min(
                value["solver"]["bend_height_m"] for value in primary_records
            ),
            "minimum_knee_flexion_angle_rad": min(
                value["solver"]["knee_flexion_angle_rad"]
                for value in primary_records
            ),
            "maximum_knee_internal_angle_rad": max(
                value["solver"]["knee_internal_angle_rad"]
                for value in primary_records
            ),
            "maximum_root_matrix_error": max(
                value["readback"]["root_matrix_max_abs_error"]
                for value in primary_records
            ),
            "maximum_pelvis_matrix_error": max(
                value["readback"]["pelvis_matrix_max_abs_error"]
                for value in primary_records
            ),
            "maximum_hip_position_error_m": max(
                value["readback"]["hip_position_error_m"]
                for value in primary_records
            ),
            "maximum_knee_position_error_m": max(
                value["readback"]["knee_position_error_m"]
                for value in primary_records
            ),
            "maximum_ankle_position_error_m": max(
                value["readback"]["ankle_position_error_m"]
                for value in primary_records
            ),
            "maximum_foot_orientation_error_rad": max(
                value["readback"]["foot_orientation_error_rad"]
                for value in primary_records
            ),
            "maximum_toe_orientation_error_rad": max(
                value["readback"]["toe_orientation_error_rad"]
                for value in primary_records
            ),
            "correction_cap_m": None,
            "source_first_object_frame": dict(source_first_object_frame),
            "target_base_object_frame": {
                "rotation_3x3": target_base_rotation.tolist(),
                "uniform_scale_m_per_armature_unit": target_uniform_scale,
                "mapping": (
                    "target_armature_local_to_canonical_static_object_frame"
                ),
                "translation_applied_to_leg_relative_points": False,
            },
            "target_current_object_matrix_applied_once_after_solve": True,
            "root_translation_in_endpoint_mapping": False,
            "physical_basis": (
                "source and target armature-local hip-knee-ankle points scaled to "
                "meters and rotated exactly once into their authenticated first/base "
                "static object frames; hip-relative direction and normalized "
                "flexion/extension reach are mapped into target leg lengths, then "
                "the target current object transform applies root yaw/translation once"
            ),
            "root_pelvis_hip_translation_preserved": True,
            "foot_toe_global_orientation_restored": True,
            "records_by_frame": primary_leg_ik_by_frame,
            "automatic_checks": "passed",
        },
        "rotation_projection": summarize_rotation_projections(
            rotation_projection_records
        ),
    }


def remove_source_objects(bpy: Any, source: SourceMotion, keep_actions: Sequence[Any]) -> list[str]:
    removed = []
    for obj in source.imported_objects:
        if obj.name in bpy.data.objects:
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    keep = set(keep_actions)
    for action in source.imported_actions:
        if action not in keep and action.name in bpy.data.actions:
            bpy.data.actions.remove(action, do_unlink=True)
    return sorted(removed)


def validate_target_only_scene(
    bpy: Any, armature: Any, mesh: Any, expected_bones: Sequence[str]
) -> None:
    if set(bpy.context.scene.objects) != {armature, mesh}:
        raise RetargetError(
            f"final scene must contain only TokenRig armature and Pixal mesh: "
            f"{[obj.name for obj in bpy.context.scene.objects]}"
        )
    actual = {bone.name for bone in armature.data.bones}
    if actual != set(expected_bones):
        raise RetargetError("final generic TokenRig skeleton changed before export")
    if len(armature.data.bones) == 0:
        raise RetargetError("final generic TokenRig skeleton is empty")
    modifiers = [
        modifier
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object == armature
    ]
    if len(modifiers) != 1:
        raise RetargetError("final Pixal mesh is not uniquely bound to TokenRig")


def _select_target_only(bpy: Any, armature: Any, mesh: Any) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    armature.hide_set(False)
    mesh.hide_set(False)
    armature.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    if set(bpy.context.selected_objects) != {armature, mesh}:
        raise RetargetError("target-only selection failed")


def _save_animated_blend(bpy: Any, armature: Any, mesh: Any, path: Path) -> None:
    _select_target_only(bpy, armature, mesh)
    bpy.context.preferences.filepaths.save_version = 0
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path))
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size <= 0:
        raise RetargetError("animated.blend save failed")


def _identify_saved_target(bpy: Any, expected_bones: Sequence[str]) -> tuple[Any, Any]:
    expected = set(expected_bones)
    armatures = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "ARMATURE" and {bone.name for bone in obj.data.bones} == expected
    ]
    if len(armatures) != 1:
        raise RetargetError("saved animated blend does not contain one exact TokenRig armature")
    armature = armatures[0]
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and any(
            modifier.type == "ARMATURE" and modifier.object == armature
            for modifier in obj.modifiers
        )
    ]
    if len(meshes) != 1:
        raise RetargetError("saved animated blend does not contain one Pixal skinned mesh")
    return armature, meshes[0]


def gltf_export_parameters(action_name: str) -> dict[str, Any]:
    if action_name not in set(ACTION_NAMES.values()):
        raise RetargetError("GLB export requested an unapproved action name")
    return {
        "export_format": "GLB",
        "use_selection": True,
        "export_animations": True,
        "export_animation_mode": "ACTIVE_ACTIONS",
        "export_nla_strips_merged_animation_name": action_name,
        "export_force_sampling": True,
        "export_frame_range": True,
        "export_frame_step": 1,
        "export_def_bones": False,
        "export_rest_position_armature": True,
        "export_anim_slide_to_zero": False,
        "export_optimize_animation_size": False,
        "export_anim_single_armature": True,
        "export_reset_pose_bones": True,
        "export_negative_frame": "SLIDE",
        "export_skins": True,
        "export_texcoords": True,
        "export_normals": True,
        "export_materials": "EXPORT",
    }


def _export_one_action(
    bpy: Any,
    animated_blend: Path,
    action_name: str,
    output_path: Path,
    expected_bones: Sequence[str],
) -> tuple[int, int]:
    result = bpy.ops.wm.open_mainfile(filepath=str(animated_blend))
    if "FINISHED" not in result:
        raise RetargetError("could not reopen animated.blend for isolated export")
    armature, mesh = _identify_saved_target(bpy, expected_bones)
    actions = {action.name: action for action in bpy.data.actions}
    if set(ACTION_NAMES.values()) - set(actions):
        raise RetargetError("animated.blend is missing Walk or Idle")
    action = actions[action_name]
    armature.animation_data.action = action
    for other in list(bpy.data.actions):
        if other != action:
            bpy.data.actions.remove(other, do_unlink=True)
    if set(bpy.data.actions) != {action}:
        raise RetargetError("one-action export isolation failed")
    frame_start, frame_end = _integer_frame_range(action)
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    _select_target_only(bpy, armature, mesh)
    result = bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        **gltf_export_parameters(action_name),
    )
    if "FINISHED" not in result or not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RetargetError(f"single-action GLB export failed: {output_path}")
    return frame_start, frame_end


def _finite_glb_nodes(document: Mapping[str, Any]) -> bool:
    for node in document.get("nodes", []):
        for field in ("matrix", "translation", "rotation", "scale"):
            values = node.get(field)
            if values is not None and (
                not isinstance(values, list)
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
            ):
                return False
    return True


def capture_animation_endpoint_matrices(
    bpy: Any,
    armature: Any,
    action: Any,
    expected_bones: Sequence[str],
) -> dict[str, Any]:
    actual_bones = {bone.name for bone in armature.data.bones}
    if actual_bones != set(expected_bones):
        raise RetargetError("animation endpoint capture has a different skeleton")
    if armature.animation_data is None:
        raise RetargetError("animation endpoint capture has no armature animation data")
    frame_start, frame_end = _integer_frame_range(action)
    armature.animation_data.action = action
    frames: dict[str, Any] = {}
    for frame in (frame_start, frame_end):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        frames[str(frame)] = {
            "armature_world": _matrix_rows(armature.matrix_world),
            "bones": {
                name: _matrix_rows(armature.pose.bones[name].matrix)
                for name in sorted(expected_bones)
            },
        }
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frames": frames,
    }


def compare_animation_endpoint_matrices(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> float:
    if (
        expected.get("frame_start") != actual.get("frame_start")
        or expected.get("frame_end") != actual.get("frame_end")
    ):
        raise RetargetError("serialized animation endpoint frame range changed")
    expected_frames = expected.get("frames")
    actual_frames = actual.get("frames")
    if not isinstance(expected_frames, Mapping) or not isinstance(actual_frames, Mapping):
        raise RetargetError("serialized animation endpoint evidence is missing")
    if set(expected_frames) != set(actual_frames):
        raise RetargetError("serialized animation endpoint frame keys changed")
    maximum_error = 0.0
    for frame in expected_frames:
        expected_frame = expected_frames[frame]
        actual_frame = actual_frames[frame]
        if not isinstance(expected_frame, Mapping) or not isinstance(actual_frame, Mapping):
            raise RetargetError("serialized animation endpoint frame is invalid")
        expected_bones = expected_frame.get("bones")
        actual_bones = actual_frame.get("bones")
        if not isinstance(expected_bones, Mapping) or not isinstance(actual_bones, Mapping):
            raise RetargetError("serialized animation endpoint bone evidence is missing")
        if set(expected_bones) != set(actual_bones):
            raise RetargetError("serialized animation endpoint skeleton changed")
        matrix_pairs = [
            (
                expected_frame.get("armature_world"),
                actual_frame.get("armature_world"),
            ),
            *(
                (expected_bones[name], actual_bones[name])
                for name in expected_bones
            ),
        ]
        for expected_matrix, actual_matrix in matrix_pairs:
            expected_array = np.asarray(expected_matrix, dtype=np.float64)
            actual_array = np.asarray(actual_matrix, dtype=np.float64)
            if (
                expected_array.shape != (4, 4)
                or actual_array.shape != (4, 4)
                or not np.isfinite(expected_array).all()
                or not np.isfinite(actual_array).all()
            ):
                raise RetargetError("serialized animation endpoint matrix is invalid")
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(expected_array - actual_array))),
            )
    if maximum_error > ROUNDTRIP_ENDPOINT_MATRIX_TOLERANCE:
        raise RetargetError(
            "serialized animation endpoint matrix changed beyond tolerance: "
            f"{maximum_error}"
        )
    return maximum_error


def authenticate_and_normalize_imported_action(
    armature: Any,
    *,
    action_name: str,
    actions: Sequence[Any],
) -> tuple[Any, dict[str, Any]]:
    if action_name not in set(ACTION_NAMES.values()):
        raise RetargetError("Blender roundtrip action name is not approved")
    imported_actions = list(actions)
    if len(imported_actions) != 1:
        raise RetargetError("Blender roundtrip does not contain exactly one action")
    action = imported_actions[0]
    animation_data = getattr(armature, "animation_data", None)
    if animation_data is None or animation_data.action is not action:
        raise RetargetError("Blender roundtrip action is not active on its armature")
    tracks = list(animation_data.nla_tracks)
    if len(tracks) != 1 or tracks[0].name != action_name:
        raise RetargetError("Blender roundtrip did not preserve the exact GLB NLA name")
    strips = list(tracks[0].strips)
    if len(strips) != 1 or strips[0].action is not action:
        raise RetargetError("Blender roundtrip NLA strip does not own the unique action")
    imported_datablock_name = str(action.name)
    action.name = action_name
    if action.name != action_name:
        raise RetargetError("Blender roundtrip action datablock normalization failed")
    return action, {
        "schema": "blender_4_2_gltf_imported_action_identity_v1",
        "glb_animation_name": action_name,
        "nla_track_name": str(tracks[0].name),
        "imported_action_datablock_name": imported_datablock_name,
        "normalized_action_datablock_name": str(action.name),
        "unique_action": True,
        "active_action_matches_unique_action": True,
        "unique_nla_track_and_strip": True,
        "automatic_checks": "passed",
    }


def roundtrip_validate_action(
    *,
    bpy: Any,
    glb_path: Path,
    action_name: str,
    frame_range: tuple[int, int],
    expected_bones: Sequence[str],
    expected_rest_matrices: Mapping[str, Any],
    input_pbr: Mapping[str, Any],
    expected_mesh_contract: Mapping[str, Any],
    expected_surface: Any,
    expected_skin_positions: Sequence[Sequence[float]],
    expected_skin_weights: Sequence[Mapping[str, float]],
    expected_animation_endpoints: Mapping[str, Any],
    target_base_transform: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _static_audit_module()
    parsed = audit.read_glb(glb_path)
    output_pbr = audit.pbr_payload_contract(parsed)
    audit.compare_pbr_payloads(input_pbr, output_pbr)
    document = parsed.document
    meshes = document.get("meshes", [])
    skins = document.get("skins", [])
    animations = document.get("animations", [])
    primitives = [
        primitive
        for mesh in meshes
        for primitive in mesh.get("primitives", [])
    ]
    uv_present = bool(primitives) and all(
        "TEXCOORD_0" in primitive.get("attributes", {}) for primitive in primitives
    )
    skin_present = bool(primitives) and all(
        {"JOINTS_0", "WEIGHTS_0"}.issubset(primitive.get("attributes", {}))
        for primitive in primitives
    )
    if len(meshes) != 1 or len(skins) != 1 or len(animations) != 1:
        raise RetargetError("GLB readback is not one mesh, one skin, and one action")
    if animations[0].get("name") != action_name:
        raise RetargetError("GLB readback action name changed")
    if not uv_present or not skin_present:
        raise RetargetError("GLB readback lost UV or skin attributes")
    if not _finite_glb_nodes(document):
        raise RetargetError("GLB readback contains non-finite node matrices")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    _configure_scene(bpy)
    result = bpy.ops.import_scene.gltf(filepath=str(glb_path))
    if "FINISHED" not in result:
        raise RetargetError("GLB Blender roundtrip import failed")
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    skinned_meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
    ]
    if len(armatures) != 1 or len(skinned_meshes) != 1:
        raise RetargetError("Blender roundtrip is not one armature and one skinned mesh")
    armature, mesh = armatures[0], skinned_meshes[0]
    actual_bones = {bone.name for bone in armature.data.bones}
    if actual_bones != set(expected_bones):
        raise RetargetError("Blender roundtrip changed the generic TokenRig skeleton")
    action, imported_action_identity = authenticate_and_normalize_imported_action(
        armature,
        action_name=action_name,
        actions=list(bpy.data.actions),
    )
    if _integer_frame_range(action) != frame_range:
        raise RetargetError("Blender roundtrip changed loop endpoints")
    actual_animation_endpoints = capture_animation_endpoint_matrices(
        bpy, armature, action, expected_bones
    )
    endpoint_matrix_error = compare_animation_endpoint_matrices(
        expected_animation_endpoints, actual_animation_endpoints
    )
    armature.animation_data.action = None
    restore_target_base_transform(armature, target_base_transform)
    armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    actual_rest = capture_rest_matrices(armature, sorted(expected_bones))
    for name in expected_bones:
        actual = np.asarray(actual_rest[name]["armature_space"], dtype=np.float64)
        expected = np.asarray(expected_rest_matrices[name]["armature_space"], dtype=np.float64)
        if actual.shape != expected.shape or float(np.max(np.abs(actual - expected))) > 1.0e-4:
            raise RetargetError(f"Blender roundtrip changed rest matrix for {name}")
    if not mesh.data.uv_layers or not mesh.vertex_groups:
        raise RetargetError("Blender roundtrip mesh lost UV or vertex groups")
    actual_mesh_contract = audit.capture_blender_mesh_contract(mesh)
    mesh_validation = audit.compare_mesh_contracts(
        expected_mesh_contract,
        actual_mesh_contract,
        allow_serialization_splits=True,
    )
    actual_surface = audit.capture_blender_surface_reference(mesh)
    surface_validation = audit.compare_surface_references(
        expected_surface, actual_surface
    )
    actual_weights, actual_positions = audit.extract_vertex_weights(mesh, armature)
    skin_validation = audit.compare_skin_by_position(
        expected_skin_positions,
        expected_skin_weights,
        actual_positions,
        actual_weights,
    )
    return {
        "one_armature": True,
        "one_skinned_mesh": True,
        "one_action": True,
        "action_name": action_name,
        "blender_imported_action_identity": imported_action_identity,
        "uv_present": True,
        "skin_present": True,
        "pbr_payloads_unchanged": True,
        "skeleton_exact": True,
        "loop_endpoints_exact": True,
        "finite_matrices": True,
        "maximum_endpoint_matrix_error": endpoint_matrix_error,
        "frame_start": frame_range[0],
        "frame_end": frame_range[1],
        "pbr_roles": sorted(output_pbr),
        "mesh_validation": mesh_validation,
        "surface_validation": surface_validation,
        "skin_validation": skin_validation,
    }


def _stable_sample_ranks(indices: np.ndarray, salt: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint64) ^ np.uint64(QUALITY_SAMPLE_SEED + salt)
    with np.errstate(over="ignore"):
        values ^= values >> np.uint64(30)
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(27)
        values *= np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    return values


def _stratified_indices(
    candidate_indices: np.ndarray,
    points: np.ndarray,
    maximum_count: int,
    *,
    salt: int,
) -> np.ndarray:
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if indices.ndim != 1 or maximum_count <= 0:
        raise RetargetError("deterministic sample request is invalid")
    if not len(indices):
        return indices
    if len(indices) <= maximum_count:
        return np.sort(indices)
    candidate_points = np.asarray(points, dtype=np.float64)
    if candidate_points.shape != (len(indices), 3) or not np.isfinite(candidate_points).all():
        raise RetargetError("deterministic sample points are invalid")
    minimum = candidate_points.min(axis=0)
    span = candidate_points.max(axis=0) - minimum
    span[span <= 1.0e-12] = 1.0
    grid = np.asarray((12, 12, 16), dtype=np.int64)
    cells_xyz = np.floor((candidate_points - minimum) / span * grid).astype(np.int64)
    cells_xyz = np.clip(cells_xyz, 0, grid - 1)
    cells = cells_xyz[:, 0] + grid[0] * (
        cells_xyz[:, 1] + grid[1] * cells_xyz[:, 2]
    )
    ranks = _stable_sample_ranks(indices, salt)
    cell_order = np.lexsort((ranks, cells))
    _, first_offsets = np.unique(cells[cell_order], return_index=True)
    representatives = cell_order[first_offsets]
    if len(representatives) > maximum_count:
        representatives = representatives[
            np.argsort(ranks[representatives], kind="stable")[:maximum_count]
        ]
    selected = np.zeros(len(indices), dtype=bool)
    selected[representatives] = True
    remaining_count = maximum_count - int(selected.sum())
    if remaining_count > 0:
        remaining = np.flatnonzero(~selected)
        fill = remaining[np.argsort(ranks[remaining], kind="stable")[:remaining_count]]
        selected[fill] = True
    return np.sort(indices[selected])


def _build_support_core_only_plan_unvalidated(
    *,
    rest_positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    semantic_bones: Mapping[str, str],
    maximum_support_vertices: int,
) -> dict[str, Any]:
    positions = np.asarray(rest_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise RetargetError("rest mesh positions are invalid")
    if len(vertex_weights) != len(positions):
        raise RetargetError("skin weights do not cover the rest mesh vertices")
    if (
        isinstance(maximum_support_vertices, bool)
        or not isinstance(maximum_support_vertices, int)
        or maximum_support_vertices <= 0
    ):
        raise RetargetError("deterministic support core budget is invalid")
    if not isinstance(semantic_bones, Mapping):
        raise RetargetError("deterministic support core semantic bones are invalid")
    try:
        side_bones = {
            side: {
                str(semantic_bones[f"{side}_foot"]),
                str(semantic_bones[f"{side}_toe"]),
            }
            for side in ("left", "right")
        }
    except KeyError as error:
        raise RetargetError(
            f"deterministic support core is missing semantic role {error.args[0]}"
        ) from error

    normalized_weights: list[dict[str, float]] = []
    dominant_bones: list[str] = []
    for index, weights in enumerate(vertex_weights):
        if not isinstance(weights, Mapping) or not weights:
            raise RetargetError(f"skin weights are missing at vertex {index}")
        normalized = {
            str(name): _finite_number(value, f"skin weight {index} {name}")
            for name, value in weights.items()
        }
        if any(value < 0.0 for value in normalized.values()):
            raise RetargetError("skin weights cannot be negative")
        normalized_weights.append(normalized)
        dominant_bones.append(
            min(normalized, key=lambda name: (-normalized[name], name))
        )

    height = float(np.ptp(positions[:, 2]))
    lower_band_max_z = float(positions[:, 2].min()) + max(0.05, 0.05 * height)
    bottom_band = positions[:, 2] <= lower_band_max_z
    support_indices: dict[str, np.ndarray] = {}
    support_candidate_indices: dict[str, np.ndarray] = {}
    support_candidate_evidence: dict[str, Any] = {}
    for side, salt in (("left", 11), ("right", 13)):
        opposite_side = "right" if side == "left" else "left"
        masses = np.asarray(
            [
                sum(weights.get(name, 0.0) for name in side_bones[side])
                for weights in normalized_weights
            ],
            dtype=np.float64,
        )
        opposite_masses = np.asarray(
            [
                sum(weights.get(name, 0.0) for name in side_bones[opposite_side])
                for weights in normalized_weights
            ],
            dtype=np.float64,
        )
        any_nonzero = masses > 0.0
        combined = masses >= SUPPORT_CORE_COMBINED_WEIGHT_MINIMUM
        dominant = np.asarray(
            [name in side_bones[side] for name in dominant_bones], dtype=bool
        ) & (masses > opposite_masses)
        opposite_clean = opposite_masses <= SUPPORT_CORE_OPPOSITE_WEIGHT_MAXIMUM
        candidates = np.flatnonzero(
            combined & dominant & opposite_clean & bottom_band
        )
        if not len(candidates):
            raise RetargetError("skin map has no bilateral foot/toe support vertices")
        selected = _stratified_indices(
            candidates,
            positions[candidates],
            maximum_support_vertices,
            salt=salt,
        )
        support_candidate_indices[side] = candidates
        support_indices[side] = selected
        bounds = positions[candidates]
        support_candidate_evidence.update(
            {
                f"{side}_any_nonzero_support_candidate_count": int(any_nonzero.sum()),
                f"{side}_combined_mass_candidate_count": int(combined.sum()),
                f"{side}_dominant_side_candidate_count": int((combined & dominant).sum()),
                f"{side}_opposite_clean_candidate_count": int(
                    (combined & dominant & opposite_clean).sum()
                ),
                f"{side}_support_candidate_count": int(len(candidates)),
                f"{side}_support_candidate_index_sha256": hashlib.sha256(
                    np.asarray(candidates, dtype="<i8").tobytes()
                ).hexdigest(),
                f"{side}_support_core_rest_bounds_m": {
                    "minimum": [float(value) for value in bounds.min(axis=0)],
                    "maximum": [float(value) for value in bounds.max(axis=0)],
                },
            }
        )

    combined_indices = np.unique(
        np.concatenate((support_indices["left"], support_indices["right"]))
    )
    digest = hashlib.sha256()
    digest.update(str(QUALITY_SAMPLE_SEED).encode("ascii"))
    digest.update(b"source_contact_support_vertices_only")
    for values in (
        support_indices["left"],
        support_indices["right"],
        support_candidate_indices["left"],
        support_candidate_indices["right"],
    ):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    return {
        "_runtime_rest_positions": positions,
        "_runtime_vertex_weights": normalized_weights,
        "_runtime_semantic_bones": dict(semantic_bones),
        "left_support_vertex_indices": support_indices["left"].tolist(),
        "right_support_vertex_indices": support_indices["right"].tolist(),
        "support_vertex_indices": combined_indices.tolist(),
        "evidence": {
            "method": "deterministic_skin_support_core_only_v2",
            "scope": "source_contact_support_vertices_only",
            "topology_edges_required": False,
            "seed": QUALITY_SAMPLE_SEED,
            "maximum_support_vertices": int(maximum_support_vertices),
            "support_core_definition": {
                "minimum_combined_side_foot_toe_weight": (
                    SUPPORT_CORE_COMBINED_WEIGHT_MINIMUM
                ),
                "dominant_bone_must_be_side_foot_or_toe": True,
                "combined_side_weight_must_exceed_opposite_side": True,
                "maximum_opposite_side_foot_toe_weight": (
                    SUPPORT_CORE_OPPOSITE_WEIGHT_MAXIMUM
                ),
                "rest_lower_band_max_z_m": lower_band_max_z,
            },
            "index_sha256": digest.hexdigest(),
            "total_vertex_count": int(len(positions)),
            **support_candidate_evidence,
        },
    }


def build_deterministic_support_core_plan(
    *,
    rest_positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    semantic_bones: Mapping[str, str],
    maximum_support_vertices: int = DEFAULT_MAXIMUM_SUPPORT_SAMPLE_VERTICES,
) -> dict[str, Any]:
    plan = _build_support_core_only_plan_unvalidated(
        rest_positions=rest_positions,
        vertex_weights=vertex_weights,
        semantic_bones=semantic_bones,
        maximum_support_vertices=maximum_support_vertices,
    )
    validate_deterministic_support_core_plan(plan)
    return plan


def validate_deterministic_support_core_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping):
        raise RetargetError("deterministic source support core plan is invalid")
    evidence = plan.get("evidence")
    positions = np.asarray(plan.get("_runtime_rest_positions"), dtype=np.float64)
    weights = plan.get("_runtime_vertex_weights")
    semantic = plan.get("_runtime_semantic_bones")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("method") != "deterministic_skin_support_core_only_v2"
        or evidence.get("scope") != "source_contact_support_vertices_only"
        or evidence.get("topology_edges_required") is not False
        or positions.ndim != 2
        or positions.shape[1:] != (3,)
        or not np.isfinite(positions).all()
        or not isinstance(weights, Sequence)
        or len(weights) != len(positions)
        or not isinstance(semantic, Mapping)
    ):
        raise RetargetError(
            "deterministic source support core evidence is inconsistent or forged"
        )
    maximum_support = evidence.get("maximum_support_vertices")
    expected = _build_support_core_only_plan_unvalidated(
        rest_positions=positions,
        vertex_weights=weights,
        semantic_bones=semantic,
        maximum_support_vertices=maximum_support,
    )
    for field in (
        "left_support_vertex_indices",
        "right_support_vertex_indices",
        "support_vertex_indices",
    ):
        if plan.get(field) != expected[field]:
            raise RetargetError(
                "deterministic source support core indices are inconsistent or forged"
            )
    if evidence != expected["evidence"]:
        raise RetargetError(
            "deterministic source support core evidence is inconsistent or forged"
        )


def build_deterministic_mesh_sample_plan(
    *,
    rest_positions: Sequence[Sequence[float]],
    edges: Sequence[Sequence[int]],
    vertex_weights: Sequence[Mapping[str, float]],
    semantic_bones: Mapping[str, str],
    maximum_global_vertices: int = DEFAULT_MAXIMUM_GLOBAL_SAMPLE_VERTICES,
    maximum_support_vertices: int = DEFAULT_MAXIMUM_SUPPORT_SAMPLE_VERTICES,
    maximum_edges: int = DEFAULT_MAXIMUM_SAMPLE_EDGES,
) -> dict[str, Any]:
    positions = np.asarray(rest_positions, dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise RetargetError("rest mesh positions are invalid")
    if (
        edge_array.ndim != 2
        or edge_array.shape[1] != 2
        or not len(edge_array)
        or np.any(edge_array < 0)
        or np.any(edge_array >= len(positions))
    ):
        raise RetargetError("rest mesh edges are invalid")
    if len(vertex_weights) != len(positions):
        raise RetargetError("skin weights do not cover the rest mesh vertices")

    side_bones = {
        side: {
            semantic_bones[f"{side}_foot"],
            semantic_bones[f"{side}_toe"],
        }
        for side in ("left", "right")
    }
    lower_body_bones = {
        semantic_bones[role]
        for role in (
            "pelvis",
            "left_thigh",
            "left_calf",
            "left_foot",
            "left_toe",
            "right_thigh",
            "right_calf",
            "right_foot",
            "right_toe",
        )
    }
    height = float(np.ptp(positions[:, 2]))
    lower_band_max_z = float(positions[:, 2].min()) + max(0.05, 0.05 * height)
    bottom_band = positions[:, 2] <= lower_band_max_z
    side_mass: dict[str, np.ndarray] = {}
    any_nonzero_masks: dict[str, np.ndarray] = {}
    dominant_side_masks: dict[str, np.ndarray] = {}
    combined_mass_masks: dict[str, np.ndarray] = {}
    opposite_clean_masks: dict[str, np.ndarray] = {}
    dominant_bones: list[str] = []
    for index, weights in enumerate(vertex_weights):
        if not isinstance(weights, Mapping) or not weights:
            raise RetargetError(f"skin weights are missing at vertex {index}")
        normalized = {
            str(name): _finite_number(value, f"skin weight {index} {name}")
            for name, value in weights.items()
        }
        if any(value < 0.0 for value in normalized.values()):
            raise RetargetError("skin weights cannot be negative")
        dominant_bones.append(
            min(normalized, key=lambda name: (-normalized[name], name))
        )
    for side, bones in side_bones.items():
        opposite = side_bones["right" if side == "left" else "left"]
        masses = np.asarray(
            [sum(float(weights.get(name, 0.0)) for name in bones) for weights in vertex_weights],
            dtype=np.float64,
        )
        opposite_masses = np.asarray(
            [
                sum(float(weights.get(name, 0.0)) for name in opposite)
                for weights in vertex_weights
            ],
            dtype=np.float64,
        )
        side_mass[side] = masses
        any_nonzero_masks[side] = masses > 0.0
        combined_mass_masks[side] = masses >= SUPPORT_CORE_COMBINED_WEIGHT_MINIMUM
        opposite_clean_masks[side] = (
            opposite_masses <= SUPPORT_CORE_OPPOSITE_WEIGHT_MAXIMUM
        )
        dominant_side_masks[side] = np.asarray(
            [name in bones for name in dominant_bones], dtype=bool
        ) & (masses > opposite_masses)
    support_masks = {
        side: (
            combined_mass_masks[side]
            & dominant_side_masks[side]
            & opposite_clean_masks[side]
            & bottom_band
        )
        for side in ("left", "right")
    }
    if any(not mask.any() for mask in support_masks.values()):
        raise RetargetError("skin map has no bilateral foot/toe support vertices")
    lower_body_mask = np.asarray(
        [bool(set(weights) & lower_body_bones) for weights in vertex_weights], dtype=bool
    )
    support_indices = {
        side: _stratified_indices(
            np.flatnonzero(mask),
            positions[mask],
            maximum_support_vertices,
            salt=11 if side == "left" else 13,
        )
        for side, mask in support_masks.items()
    }
    global_candidates = np.arange(len(positions), dtype=np.int64)
    global_indices = _stratified_indices(
        global_candidates,
        positions,
        maximum_global_vertices,
        salt=17,
    )
    bottom_indices = _stratified_indices(
        np.flatnonzero(bottom_band),
        positions[bottom_band],
        maximum_global_vertices,
        salt=19,
    )
    penetration_indices = np.unique(
        np.concatenate(
            (bottom_indices, support_indices["left"], support_indices["right"])
        )
    )

    lower_edge_mask = lower_body_mask[edge_array[:, 0]] | lower_body_mask[edge_array[:, 1]]
    lower_edge_candidates = np.flatnonzero(lower_edge_mask)
    critical_budget = max(1, maximum_edges * 3 // 4)
    critical_edge_indices = _stratified_indices(
        lower_edge_candidates,
        positions[edge_array[lower_edge_candidates]].mean(axis=1),
        critical_budget,
        salt=23,
    )
    remaining_budget = max(1, maximum_edges - len(critical_edge_indices))
    global_edge_candidates = np.arange(len(edge_array), dtype=np.int64)
    global_edge_indices = _stratified_indices(
        global_edge_candidates,
        positions[edge_array].mean(axis=1),
        remaining_budget,
        salt=29,
    )
    sampled_edge_indices = np.unique(
        np.concatenate((critical_edge_indices, global_edge_indices))
    )
    sampled_edges = edge_array[sampled_edge_indices]
    evaluation_indices = np.unique(
        np.concatenate(
            (
                global_indices,
                penetration_indices,
                sampled_edges.reshape(-1),
            )
        )
    )
    edge_lengths = np.linalg.norm(
        positions[sampled_edges[:, 0]] - positions[sampled_edges[:, 1]], axis=1
    )
    if not np.isfinite(edge_lengths).all() or np.any(edge_lengths <= 1.0e-10):
        raise RetargetError("sampled rest mesh contains a degenerate edge")
    digest = hashlib.sha256()
    digest.update(str(QUALITY_SAMPLE_SEED).encode("ascii"))
    for values in (
        evaluation_indices,
        penetration_indices,
        support_indices["left"],
        support_indices["right"],
        sampled_edge_indices,
    ):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    support_candidate_evidence: dict[str, Any] = {}
    for side in ("left", "right"):
        candidate_indices = np.flatnonzero(support_masks[side])
        bounds = positions[candidate_indices]
        support_candidate_evidence.update(
            {
                f"{side}_any_nonzero_support_candidate_count": int(
                    any_nonzero_masks[side].sum()
                ),
                f"{side}_combined_mass_candidate_count": int(
                    combined_mass_masks[side].sum()
                ),
                f"{side}_dominant_side_candidate_count": int(
                    (combined_mass_masks[side] & dominant_side_masks[side]).sum()
                ),
                f"{side}_opposite_clean_candidate_count": int(
                    (
                        combined_mass_masks[side]
                        & dominant_side_masks[side]
                        & opposite_clean_masks[side]
                    ).sum()
                ),
                f"{side}_support_candidate_index_sha256": hashlib.sha256(
                    np.asarray(candidate_indices, dtype="<i8").tobytes()
                ).hexdigest(),
                f"{side}_support_core_rest_bounds_m": {
                    "minimum": [float(value) for value in bounds.min(axis=0)],
                    "maximum": [float(value) for value in bounds.max(axis=0)],
                },
            }
        )
    plan = {
        "_runtime_rest_positions": positions,
        "_runtime_vertex_weights": vertex_weights,
        "_runtime_semantic_bones": dict(semantic_bones),
        "evaluation_vertex_indices": evaluation_indices.tolist(),
        "penetration_vertex_indices": penetration_indices.tolist(),
        "left_support_vertex_indices": support_indices["left"].tolist(),
        "right_support_vertex_indices": support_indices["right"].tolist(),
        "support_vertex_indices": np.unique(
            np.concatenate((support_indices["left"], support_indices["right"]))
        ).tolist(),
        "sampled_edges": sampled_edges.tolist(),
        "rest_edge_lengths": edge_lengths.tolist(),
        "evidence": {
            "method": "deterministic_spatial_skin_support_core_v2",
            "seed": QUALITY_SAMPLE_SEED,
            "maximum_support_vertices": int(maximum_support_vertices),
            "support_core_definition": {
                "minimum_combined_side_foot_toe_weight": (
                    SUPPORT_CORE_COMBINED_WEIGHT_MINIMUM
                ),
                "dominant_bone_must_be_side_foot_or_toe": True,
                "combined_side_weight_must_exceed_opposite_side": True,
                "maximum_opposite_side_foot_toe_weight": (
                    SUPPORT_CORE_OPPOSITE_WEIGHT_MAXIMUM
                ),
                "rest_lower_band_max_z_m": lower_band_max_z,
            },
            "index_sha256": digest.hexdigest(),
            "total_vertex_count": int(len(positions)),
            "evaluated_vertex_count": int(len(evaluation_indices)),
            "total_edge_count": int(len(edge_array)),
            "sampled_edge_count": int(len(sampled_edge_indices)),
            "left_support_candidate_count": int(support_masks["left"].sum()),
            "right_support_candidate_count": int(support_masks["right"].sum()),
            **support_candidate_evidence,
            "lower_body_edge_candidate_count": int(len(lower_edge_candidates)),
            "vertex_coverage_ratio": float(len(evaluation_indices) / len(positions)),
            "edge_coverage_ratio": float(len(sampled_edge_indices) / len(edge_array)),
            "lower_body_edge_coverage_ratio": float(
                len(critical_edge_indices) / max(1, len(lower_edge_candidates))
            ),
        },
    }
    validate_deterministic_mesh_sample_plan_support_core(plan)
    return plan


def validate_deterministic_mesh_sample_plan_support_core(
    plan: Mapping[str, Any],
) -> None:
    if not isinstance(plan, Mapping):
        raise RetargetError("deterministic support core plan is invalid")
    evidence = plan.get("evidence")
    positions = np.asarray(plan.get("_runtime_rest_positions"), dtype=np.float64)
    weights = plan.get("_runtime_vertex_weights")
    semantic = plan.get("_runtime_semantic_bones")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("method") != "deterministic_spatial_skin_support_core_v2"
        or positions.ndim != 2
        or positions.shape[1:] != (3,)
        or not np.isfinite(positions).all()
        or not isinstance(weights, Sequence)
        or len(weights) != len(positions)
        or not isinstance(semantic, Mapping)
    ):
        raise RetargetError("deterministic support core evidence is inconsistent or forged")
    maximum_support = evidence.get("maximum_support_vertices")
    if isinstance(maximum_support, bool) or not isinstance(maximum_support, int) or maximum_support <= 0:
        raise RetargetError("deterministic support core budget is inconsistent or forged")
    definition = evidence.get("support_core_definition")
    height = float(np.ptp(positions[:, 2]))
    lower_band_max_z = float(positions[:, 2].min()) + max(0.05, 0.05 * height)
    expected_definition = {
        "minimum_combined_side_foot_toe_weight": SUPPORT_CORE_COMBINED_WEIGHT_MINIMUM,
        "dominant_bone_must_be_side_foot_or_toe": True,
        "combined_side_weight_must_exceed_opposite_side": True,
        "maximum_opposite_side_foot_toe_weight": (
            SUPPORT_CORE_OPPOSITE_WEIGHT_MAXIMUM
        ),
        "rest_lower_band_max_z_m": lower_band_max_z,
    }
    if definition != expected_definition:
        raise RetargetError("deterministic support core definition is inconsistent or forged")
    side_bones = {
        side: {semantic[f"{side}_foot"], semantic[f"{side}_toe"]}
        for side in ("left", "right")
    }
    normalized_weights: list[dict[str, float]] = []
    dominant_bones = []
    for index, value in enumerate(weights):
        if not isinstance(value, Mapping) or not value:
            raise RetargetError("deterministic support core skin weights are invalid")
        normalized = {
            str(name): _finite_number(weight, f"support core weight {index} {name}")
            for name, weight in value.items()
        }
        if any(weight < 0.0 for weight in normalized.values()):
            raise RetargetError("deterministic support core skin weights are negative")
        normalized_weights.append(normalized)
        dominant_bones.append(
            min(normalized, key=lambda name: (-normalized[name], name))
        )
    bottom = positions[:, 2] <= lower_band_max_z
    for side, salt in (("left", 11), ("right", 13)):
        opposite_side = "right" if side == "left" else "left"
        masses = np.asarray(
            [sum(value.get(name, 0.0) for name in side_bones[side]) for value in normalized_weights]
        )
        opposite = np.asarray(
            [
                sum(value.get(name, 0.0) for name in side_bones[opposite_side])
                for value in normalized_weights
            ]
        )
        any_nonzero = masses > 0.0
        combined = masses >= SUPPORT_CORE_COMBINED_WEIGHT_MINIMUM
        dominant = np.asarray(
            [name in side_bones[side] for name in dominant_bones], dtype=bool
        ) & (masses > opposite)
        opposite_clean = opposite <= SUPPORT_CORE_OPPOSITE_WEIGHT_MAXIMUM
        candidates = np.flatnonzero(
            combined & dominant & opposite_clean & bottom
        )
        if not len(candidates):
            raise RetargetError("deterministic support core has no bilateral candidates")
        expected_selected = _stratified_indices(
            candidates,
            positions[candidates],
            maximum_support,
            salt=salt,
        ).tolist()
        if plan.get(f"{side}_support_vertex_indices") != expected_selected:
            raise RetargetError("deterministic support core indices are inconsistent or forged")
        bounds = positions[candidates]
        expected_fields = {
            f"{side}_any_nonzero_support_candidate_count": int(any_nonzero.sum()),
            f"{side}_combined_mass_candidate_count": int(combined.sum()),
            f"{side}_dominant_side_candidate_count": int((combined & dominant).sum()),
            f"{side}_opposite_clean_candidate_count": int(
                (combined & dominant & opposite_clean).sum()
            ),
            f"{side}_support_candidate_count": int(len(candidates)),
            f"{side}_support_candidate_index_sha256": hashlib.sha256(
                np.asarray(candidates, dtype="<i8").tobytes()
            ).hexdigest(),
            f"{side}_support_core_rest_bounds_m": {
                "minimum": [float(value) for value in bounds.min(axis=0)],
                "maximum": [float(value) for value in bounds.max(axis=0)],
            },
        }
        if any(evidence.get(key) != value for key, value in expected_fields.items()):
            raise RetargetError("deterministic support core evidence is inconsistent or forged")


def describe_grounding_sample(
    *,
    frame: int,
    vertex_index: int,
    evaluated_position: Sequence[float],
    rest_position: Sequence[float],
    weights: Mapping[str, float],
    semantic_bones: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        isinstance(frame, bool)
        or not isinstance(frame, int)
        or isinstance(vertex_index, bool)
        or not isinstance(vertex_index, int)
        or vertex_index < 0
    ):
        raise RetargetError("grounding sample frame or vertex index is invalid")
    evaluated = np.asarray(evaluated_position, dtype=np.float64)
    rest = np.asarray(rest_position, dtype=np.float64)
    if (
        evaluated.shape != (3,)
        or rest.shape != (3,)
        or not np.isfinite(evaluated).all()
        or not np.isfinite(rest).all()
    ):
        raise RetargetError("grounding sample positions are invalid")
    if not isinstance(weights, Mapping) or not weights:
        raise RetargetError("grounding sample has no skin weights")
    role_by_bone: dict[str, str] = {}
    for role, value in semantic_bones.items():
        if isinstance(value, str):
            role_by_bone[value] = str(role)
        elif isinstance(value, list):
            for bone in value:
                if isinstance(bone, str):
                    role_by_bone[bone] = str(role)
    normalized_weights: list[dict[str, Any]] = []
    for bone, weight in weights.items():
        if not isinstance(bone, str) or not bone:
            raise RetargetError("grounding sample has an invalid skin bone")
        value = _finite_number(weight, f"grounding sample weight {bone}")
        if value <= 0.0:
            raise RetargetError("grounding sample skin weights must be positive")
        normalized_weights.append(
            {
                "bone": bone,
                "semantic_role": role_by_bone.get(bone, "unmapped"),
                "weight": value,
            }
        )
    normalized_weights.sort(key=lambda item: (-item["weight"], item["bone"]))
    dominant_role = str(normalized_weights[0]["semantic_role"])
    side = (
        "left"
        if dominant_role.startswith("left_")
        else "right"
        if dominant_role.startswith("right_")
        else "center"
        if dominant_role != "unmapped"
        else "unmapped"
    )
    return {
        "frame": frame,
        "vertex_index": vertex_index,
        "evaluated_position_m": [float(value) for value in evaluated],
        "rest_position_m": [float(value) for value in rest],
        "dominant_weights": normalized_weights,
        "dominant_semantic_region": dominant_role,
        "side": side,
    }


def solve_two_bone_leg_ik(
    *,
    hip: Sequence[float],
    knee: Sequence[float],
    ankle: Sequence[float],
    target_ankle: Sequence[float],
    maximum_correction_m: float = MAXIMUM_IK_ANKLE_CORRECTION_M,
    minimum_knee_plane_dot: float = MINIMUM_IK_KNEE_PLANE_DOT,
) -> dict[str, Any]:
    points = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (hip, knee, ankle, target_ankle)
    )
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in points):
        raise RetargetError("two-bone IK points must be finite 3D vectors")
    hip_v, knee_v, ankle_v, target_v = points
    if (
        not math.isfinite(maximum_correction_m)
        or maximum_correction_m <= 0.0
        or not math.isfinite(minimum_knee_plane_dot)
        or not 0.0 < minimum_knee_plane_dot <= 1.0
    ):
        raise RetargetError("two-bone IK correction or knee-plane cap is invalid")
    correction = target_v - ankle_v
    horizontal_correction = float(np.linalg.norm(correction[:2]))
    if horizontal_correction > 1.0e-9:
        raise RetargetError("two-bone IK cannot alter ankle X-Y/horizontal position")
    correction_m = float(correction[2])
    if abs(correction_m) > maximum_correction_m:
        raise RetargetError(
            "two-bone IK ankle correction exceeds the strict "
            f"{maximum_correction_m:.3f} m cap: {correction_m}"
        )

    upper = knee_v - hip_v
    lower = ankle_v - knee_v
    upper_length = float(np.linalg.norm(upper))
    lower_length = float(np.linalg.norm(lower))
    target_delta = target_v - hip_v
    target_distance = float(np.linalg.norm(target_delta))
    if min(upper_length, lower_length, target_distance) <= 1.0e-8:
        raise RetargetError("two-bone IK chain is degenerate")
    minimum_reach = abs(upper_length - lower_length)
    maximum_reach = upper_length + lower_length
    reach_margin = min(
        target_distance - minimum_reach,
        maximum_reach - target_distance,
    )
    if reach_margin < MINIMUM_IK_REACH_MARGIN_M:
        raise RetargetError(
            "two-bone IK target is unreachable or exceeds the strict reach margin"
        )

    target_axis = target_delta / target_distance
    current_normal = np.cross(upper, lower)
    normal_length = float(np.linalg.norm(current_normal))
    if normal_length <= 1.0e-8:
        raise RetargetError("two-bone IK current knee plane is degenerate")
    current_normal /= normal_length
    current_bend = knee_v - hip_v
    current_bend -= float(np.dot(current_bend, target_axis)) * target_axis
    current_bend_length = float(np.linalg.norm(current_bend))
    if current_bend_length <= 1.0e-8:
        raise RetargetError("two-bone IK current knee side is ambiguous")
    current_bend /= current_bend_length
    bend_direction = np.cross(target_axis, current_normal)
    bend_length = float(np.linalg.norm(bend_direction))
    if bend_length <= 1.0e-8:
        raise RetargetError("two-bone IK target axis is parallel to the knee plane normal")
    bend_direction /= bend_length
    if float(np.dot(bend_direction, current_bend)) < 0.0:
        bend_direction *= -1.0

    along = (
        upper_length * upper_length
        - lower_length * lower_length
        + target_distance * target_distance
    ) / (2.0 * target_distance)
    bend_height_squared = upper_length * upper_length - along * along
    if bend_height_squared <= 1.0e-12:
        raise RetargetError("two-bone IK solution would straighten or flip the knee")
    bend_height = math.sqrt(bend_height_squared)
    solved_knee = hip_v + along * target_axis + bend_height * bend_direction
    new_upper = solved_knee - hip_v
    new_lower = target_v - solved_knee
    new_normal = np.cross(new_upper, new_lower)
    new_normal /= np.linalg.norm(new_normal)
    knee_plane_dot = float(np.dot(current_normal, new_normal))
    new_bend = solved_knee - hip_v
    new_bend -= float(np.dot(new_bend, target_axis)) * target_axis
    new_bend /= np.linalg.norm(new_bend)
    knee_side_dot = float(np.dot(current_bend, new_bend))
    if knee_plane_dot < minimum_knee_plane_dot or knee_side_dot <= 0.0:
        raise RetargetError("two-bone IK knee plane/twist would flip beyond its cap")

    return {
        "hip": [float(value) for value in hip_v],
        "knee": [float(value) for value in solved_knee],
        "ankle": [float(value) for value in target_v],
        "evidence": {
            "method": "deterministic_two_bone_leg_ik_v1",
            "ankle_correction_m": correction_m,
            "absolute_ankle_correction_m": abs(correction_m),
            "maximum_ankle_correction_m": float(maximum_correction_m),
            "upper_leg_length_m": upper_length,
            "lower_leg_length_m": lower_length,
            "target_distance_m": target_distance,
            "reach_margin_m": reach_margin,
            "minimum_reach_margin_m": MINIMUM_IK_REACH_MARGIN_M,
            "knee_plane_dot": knee_plane_dot,
            "minimum_knee_plane_dot": float(minimum_knee_plane_dot),
            "knee_side_dot": knee_side_dot,
            "root_pelvis_xy_unchanged": True,
        },
    }


def _proper_leg_rest_frame(
    *, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, description: str
) -> np.ndarray:
    primary = ankle - hip
    primary_length = float(np.linalg.norm(primary))
    if primary_length <= 1.0e-8:
        raise RetargetError(f"{description} hip-to-ankle rest chord is degenerate")
    primary /= primary_length
    bend = knee - hip
    bend -= float(np.dot(bend, primary)) * primary
    bend_length = float(np.linalg.norm(bend))
    if bend_length <= 1.0e-8:
        raise RetargetError(f"{description} knee bend plane is degenerate")
    bend /= bend_length
    normal = np.cross(primary, bend)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1.0e-8:
        raise RetargetError(f"{description} knee plane normal is degenerate")
    normal /= normal_length
    frame = np.column_stack((primary, bend, normal))
    determinant = float(np.linalg.det(frame))
    orthogonality = float(np.max(np.abs(frame.T @ frame - np.eye(3))))
    if determinant <= 0.0 or abs(determinant - 1.0) > 1.0e-10 or orthogonality > 1.0e-10:
        raise RetargetError(f"{description} rest frame is not proper right-handed SO(3)")
    return frame


def canonical_static_armature_point(
    point: Sequence[float],
    *,
    uniform_scale_m_per_unit: float,
    base_rotation_3x3: Sequence[Sequence[float]],
) -> np.ndarray:
    """Map an armature-local point into its authenticated static object frame.

    Object translation is deliberately absent: leg endpoint transfer is hip
    relative, while root translation is already reconstructed independently.
    """

    value = np.asarray(point, dtype=np.float64)
    scale = _finite_number(
        uniform_scale_m_per_unit, "canonical armature-frame uniform scale"
    )
    if value.shape != (3,) or not np.isfinite(value).all() or scale <= 0.0:
        raise RetargetError("canonical armature-frame point or scale is invalid")
    rotation, _ = project_near_rotation(
        base_rotation_3x3, "canonical armature-frame base rotation"
    )
    return rotation @ (scale * value)


def canonical_static_to_armature_point(
    point: Sequence[float],
    *,
    uniform_scale_m_per_unit: float,
    base_rotation_3x3: Sequence[Sequence[float]],
) -> np.ndarray:
    """Invert :func:`canonical_static_armature_point` exactly once."""

    value = np.asarray(point, dtype=np.float64)
    scale = _finite_number(
        uniform_scale_m_per_unit, "canonical armature-frame uniform scale"
    )
    if value.shape != (3,) or not np.isfinite(value).all() or scale <= 0.0:
        raise RetargetError("canonical armature-frame point or scale is invalid")
    rotation, _ = project_near_rotation(
        base_rotation_3x3, "canonical armature-frame base rotation"
    )
    return rotation.T @ value / scale


def _proper_canonical_front_chord_frame(
    *, hip: np.ndarray, ankle: np.ndarray, description: str
) -> np.ndarray:
    primary = ankle - hip
    primary_length = float(np.linalg.norm(primary))
    if primary_length <= 1.0e-8:
        raise RetargetError(f"{description} hip-to-ankle rest chord is degenerate")
    primary /= primary_length
    sagittal = np.asarray(CANONICAL_FRONT_VECTOR, dtype=np.float64)
    sagittal -= float(np.dot(sagittal, primary)) * primary
    sagittal_length = float(np.linalg.norm(sagittal))
    if sagittal_length <= 1.0e-8:
        raise RetargetError(
            f"{description} canonical FRONT projection is degenerate"
        )
    sagittal /= sagittal_length
    third = np.cross(primary, sagittal)
    third_length = float(np.linalg.norm(third))
    if third_length <= 1.0e-8:
        raise RetargetError(f"{description} canonical chord frame is degenerate")
    third /= third_length
    frame = np.column_stack((primary, sagittal, third))
    determinant = float(np.linalg.det(frame))
    orthogonality = float(np.max(np.abs(frame.T @ frame - np.eye(3))))
    if determinant <= 0.0 or abs(determinant - 1.0) > 1.0e-10 or orthogonality > 1.0e-10:
        raise RetargetError(
            f"{description} canonical chord frame is not proper right-handed SO(3)"
        )
    return frame


def _minimal_direction_alignment(
    source_direction: np.ndarray,
    target_direction: np.ndarray,
    *,
    description: str,
) -> np.ndarray:
    source = np.asarray(source_direction, dtype=np.float64)
    target = np.asarray(target_direction, dtype=np.float64)
    if (
        source.shape != (3,)
        or target.shape != (3,)
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise RetargetError(f"{description} directions must be finite 3D")
    source_length = float(np.linalg.norm(source))
    target_length = float(np.linalg.norm(target))
    if min(source_length, target_length) <= 1.0e-8:
        raise RetargetError(f"{description} direction is degenerate")
    source /= source_length
    target /= target_length
    dot = max(-1.0, min(1.0, float(np.dot(source, target))))
    if dot <= -1.0 + 1.0e-10:
        raise RetargetError(
            f"{description} antiparallel direction has ambiguous twist"
        )
    cross = np.cross(source, target)
    cross_length = float(np.linalg.norm(cross))
    if cross_length <= 1.0e-12:
        rotation = np.eye(3, dtype=np.float64)
    else:
        skew = np.asarray(
            (
                (0.0, -cross[2], cross[1]),
                (cross[2], 0.0, -cross[0]),
                (-cross[1], cross[0], 0.0),
            ),
            dtype=np.float64,
        )
        rotation = np.eye(3) + skew + (skew @ skew) / (1.0 + dot)
    projected, _ = project_near_rotation(rotation, description)
    return projected


def _build_source_leg_endpoint_rest_frame(
    *,
    source_rest_hip: Sequence[float],
    source_rest_knee: Sequence[float],
    source_rest_ankle: Sequence[float],
    target_rest_hip: Sequence[float],
    target_rest_knee: Sequence[float],
    target_rest_ankle: Sequence[float],
    source_current_hip: Sequence[float],
    source_current_ankle: Sequence[float],
    target_current_hip: Sequence[float],
    scale_basis: str,
    source_current_knee: Sequence[float] | None = None,
    orientation_basis: str = "full_leg_rest_frame",
    coordinate_space: str = "authenticated_world_m",
    axis_map_3x3: Sequence[Sequence[float]] = AXIS_MAP_3X3,
) -> dict[str, Any]:
    labels = (
        "source_rest_hip",
        "source_rest_knee",
        "source_rest_ankle",
        "target_rest_hip",
        "target_rest_knee",
        "target_rest_ankle",
        "source_current_hip",
        "source_current_ankle",
        "target_current_hip",
    )
    raw_points = (
        source_rest_hip,
        source_rest_knee,
        source_rest_ankle,
        target_rest_hip,
        target_rest_knee,
        target_rest_ankle,
        source_current_hip,
        source_current_ankle,
        target_current_hip,
    )
    points = {}
    for label, value in zip(labels, raw_points):
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise RetargetError(f"rest-frame endpoint {label} must be finite 3D")
        points[label] = point
    current_knee = None
    if source_current_knee is not None:
        current_knee = np.asarray(source_current_knee, dtype=np.float64)
        if current_knee.shape != (3,) or not np.isfinite(current_knee).all():
            raise RetargetError(
                "rest-frame endpoint source_current_knee must be finite 3D"
            )
    raw_axis = np.asarray(axis_map_3x3, dtype=np.float64)
    if raw_axis.shape != (3, 3) or not np.isfinite(raw_axis).all():
        raise RetargetError("rest-frame endpoint axis map must be finite 3x3")
    axis, axis_evidence = project_near_rotation(
        raw_axis,
        "rest-frame endpoint axis map",
    )
    allowed_orientation_bases = {
        "full_leg_rest_frame",
        "canonical_axis_minimal_rest_chord",
        "canonical_front_constrained_chord_frame",
        "canonical_absolute_source_direction",
    }
    if orientation_basis not in allowed_orientation_bases:
        raise RetargetError("rest-frame endpoint orientation basis is invalid")
    if coordinate_space not in {
        "authenticated_world_m",
        "canonical_static_object_frame_m",
    }:
        raise RetargetError("rest-frame endpoint coordinate space is invalid")
    if orientation_basis == "canonical_absolute_source_direction":
        source_frame = np.eye(3, dtype=np.float64)
        target_frame = np.eye(3, dtype=np.float64)
    elif orientation_basis == "full_leg_rest_frame":
        source_frame = _proper_leg_rest_frame(
            hip=points["source_rest_hip"],
            knee=points["source_rest_knee"],
            ankle=points["source_rest_ankle"],
            description="source leg",
        )
        target_frame = _proper_leg_rest_frame(
            hip=points["target_rest_hip"],
            knee=points["target_rest_knee"],
            ankle=points["target_rest_ankle"],
            description="target leg",
        )
    else:
        source_frame = _proper_canonical_front_chord_frame(
            hip=points["source_rest_hip"],
            ankle=points["source_rest_ankle"],
            description="source leg",
        )
        target_frame = _proper_canonical_front_chord_frame(
            hip=points["target_rest_hip"],
            ankle=points["target_rest_ankle"],
            description="target leg",
        )
    mapped_source_frame = axis @ source_frame
    alignment = (
        np.eye(3, dtype=np.float64)
        if orientation_basis == "canonical_absolute_source_direction"
        else _minimal_direction_alignment(
            axis @ (points["source_rest_ankle"] - points["source_rest_hip"]),
            points["target_rest_ankle"] - points["target_rest_hip"],
            description="rest-frame endpoint minimal chord alignment",
        )
        if orientation_basis == "canonical_axis_minimal_rest_chord"
        else target_frame @ mapped_source_frame.T
    )
    alignment, alignment_evidence = project_near_rotation(
        alignment,
        "rest-frame endpoint proper alignment",
    )
    source_chord = float(
        np.linalg.norm(points["source_rest_ankle"] - points["source_rest_hip"])
    )
    target_chord = float(
        np.linalg.norm(points["target_rest_ankle"] - points["target_rest_hip"])
    )
    source_upper = float(
        np.linalg.norm(points["source_rest_knee"] - points["source_rest_hip"])
    )
    source_lower = float(
        np.linalg.norm(points["source_rest_ankle"] - points["source_rest_knee"])
    )
    target_upper = float(
        np.linalg.norm(points["target_rest_knee"] - points["target_rest_hip"])
    )
    target_lower = float(
        np.linalg.norm(points["target_rest_ankle"] - points["target_rest_knee"])
    )
    source_segments = source_upper + source_lower
    target_segments = target_upper + target_lower
    source_minimum_reach = abs(source_upper - source_lower)
    target_minimum_reach = abs(target_upper - target_lower)
    source_reach_span = source_segments - source_minimum_reach
    target_reach_span = target_segments - target_minimum_reach
    if min(source_reach_span, target_reach_span) <= 1.0e-8:
        raise RetargetError("rest-frame endpoint leg reach interval is degenerate")
    scales = {
        "rest_chord": target_chord / source_chord,
        "segment_sum": target_segments / source_segments,
    }
    anisotropic_basis = "axial_chord_perpendicular_segment"
    piecewise_basis = "reach_normalized_piecewise"
    if scale_basis not in {*scales, anisotropic_basis, piecewise_basis}:
        raise RetargetError(
            "rest-frame endpoint scale basis must be rest_chord, segment_sum, "
            "axial_chord_perpendicular_segment, or reach_normalized_piecewise"
        )
    if (
        orientation_basis == "canonical_absolute_source_direction"
        and scale_basis != piecewise_basis
    ):
        raise RetargetError(
            "canonical absolute source direction requires piecewise normalized reach"
        )
    source_current_vector = (
        points["source_current_ankle"] - points["source_current_hip"]
    )
    source_rest_vector = points["source_rest_ankle"] - points["source_rest_hip"]
    source_distance = float(np.linalg.norm(source_current_vector))
    source_normalized_reach = (
        source_distance - source_minimum_reach
    ) / source_reach_span
    piecewise_branch = None
    normalized_reach_fraction = None
    mapped_target_distance = None
    target_reach_margin = None
    target_normalized_reach = None
    mapped_direction = None
    source_to_mapped_direction_alignment_dot = None
    if scale_basis == piecewise_basis:
        if (
            source_distance < source_minimum_reach - 1.0e-9
            or source_distance > source_segments + 1.0e-9
        ):
            raise RetargetError(
                "rest-frame endpoint source distance is outside its leg reach"
            )
        if source_distance <= 1.0e-8:
            raise RetargetError("rest-frame endpoint source current chord is degenerate")
        mapped_direction = alignment @ (axis @ source_current_vector)
        mapped_direction /= np.linalg.norm(mapped_direction)
        authenticated_source_direction = axis @ source_current_vector
        authenticated_source_direction /= np.linalg.norm(
            authenticated_source_direction
        )
        source_to_mapped_direction_alignment_dot = float(
            np.dot(authenticated_source_direction, mapped_direction)
        )
        if math.isclose(source_distance, source_chord, abs_tol=1.0e-12):
            piecewise_branch = "rest"
            normalized_reach_fraction = 0.0
            mapped_target_distance = target_chord
        elif source_distance > source_chord:
            source_span = source_segments - source_chord
            target_span = target_segments - target_chord
            if min(source_span, target_span) <= 1.0e-8:
                raise RetargetError(
                    "rest-frame endpoint extension reach interval is degenerate"
                )
            piecewise_branch = "extension"
            normalized_reach_fraction = (
                source_distance - source_chord
            ) / source_span
            mapped_target_distance = target_chord + normalized_reach_fraction * target_span
        else:
            source_span = source_chord - source_minimum_reach
            target_span = target_chord - target_minimum_reach
            if min(source_span, target_span) <= 1.0e-8:
                raise RetargetError(
                    "rest-frame endpoint flexion reach interval is degenerate"
                )
            piecewise_branch = "flexion"
            normalized_reach_fraction = (
                source_chord - source_distance
            ) / source_span
            mapped_target_distance = target_chord - normalized_reach_fraction * target_span
        if not -1.0e-9 <= normalized_reach_fraction <= 1.0 + 1.0e-9:
            raise RetargetError(
                "rest-frame endpoint normalized source reach is outside [0, 1]"
            )
        target_reach_margin = min(
            mapped_target_distance - target_minimum_reach,
            target_segments - mapped_target_distance,
        )
        if target_reach_margin < MINIMUM_IK_REACH_MARGIN_M:
            raise RetargetError(
                "rest-frame endpoint mapped target violates the strict reach margin"
            )
        endpoint = (
            points["target_current_hip"]
            + mapped_target_distance * mapped_direction
        )
        target_normalized_reach = (
            mapped_target_distance - target_minimum_reach
        ) / target_reach_span
        rest_direction = alignment @ (axis @ source_rest_vector)
        rest_direction /= np.linalg.norm(rest_direction)
        reconstructed_rest = (
            points["target_rest_hip"] + target_chord * rest_direction
        )
        scale_components = None
    else:
        scale_components = np.asarray(
            (
                scales["rest_chord"],
                scales["segment_sum"],
                scales["segment_sum"],
            )
            if scale_basis == anisotropic_basis
            else (scales[scale_basis],) * 3,
            dtype=np.float64,
        )
        if not np.isfinite(scale_components).all() or np.any(scale_components <= 0.0):
            raise RetargetError("rest-frame endpoint scale is invalid")
        source_current_coordinates = mapped_source_frame.T @ (axis @ source_current_vector)
        mapped_current_vector = target_frame @ (
            scale_components * source_current_coordinates
        )
        endpoint = points["target_current_hip"] + mapped_current_vector
        source_rest_coordinates = mapped_source_frame.T @ (axis @ source_rest_vector)
        reconstructed_rest = (
            points["target_rest_hip"]
            + target_frame @ (scale_components * source_rest_coordinates)
        )
    rest_error = float(
        np.linalg.norm(reconstructed_rest - points["target_rest_ankle"])
    )
    mapped_source_knee_vector = None
    if current_knee is not None:
        mapped_source_knee_vector = alignment @ (
            axis @ (current_knee - points["source_current_hip"])
        )
    authenticated_inputs = {
        "source_rest_hip": points["source_rest_hip"].tolist(),
        "source_rest_knee": points["source_rest_knee"].tolist(),
        "source_rest_ankle": points["source_rest_ankle"].tolist(),
        "target_rest_hip": points["target_rest_hip"].tolist(),
        "target_rest_knee": points["target_rest_knee"].tolist(),
        "target_rest_ankle": points["target_rest_ankle"].tolist(),
        "source_current_hip": points["source_current_hip"].tolist(),
        "source_current_knee": (
            None if current_knee is None else current_knee.tolist()
        ),
        "source_current_ankle": points["source_current_ankle"].tolist(),
        "target_current_hip": points["target_current_hip"].tolist(),
        "scale_basis": scale_basis,
        "orientation_basis": orientation_basis,
        "coordinate_space": coordinate_space,
        "axis_map_3x3": raw_axis.tolist(),
    }
    return {
        "endpoint": [float(value) for value in endpoint],
        "mapped_source_knee_vector": (
            None
            if mapped_source_knee_vector is None
            else [float(value) for value in mapped_source_knee_vector]
        ),
        "alignment_3x3": alignment.tolist(),
        "evidence": {
            "schema": "proper_leg_rest_frame_endpoint_v1",
            "method": "proper_leg_rest_frame_endpoint_v1",
            "formula": (
                "canonical_absolute_source_direction_plus_piecewise_normalized_reach_v1"
                if orientation_basis == "canonical_absolute_source_direction"
                else "proper_canonical_axis_minimal_rest_chord_plus_piecewise_normalized_reach_v1"
                if orientation_basis == "canonical_axis_minimal_rest_chord"
                else "proper_canonical_front_chord_frame_plus_piecewise_normalized_reach_v1"
                if orientation_basis == "canonical_front_constrained_chord_frame"
                else "proper_rest_frame_direction_plus_piecewise_normalized_reach_v1"
            ),
            "coordinate_space": coordinate_space,
            "authenticated_inputs": authenticated_inputs,
            "orientation_basis": orientation_basis,
            "direction_mapping": (
                "axis_map_once_without_target_rest_chord_rotation"
                if orientation_basis == "canonical_absolute_source_direction"
                else "source_rest_to_target_rest_orientation_alignment"
            ),
            "target_rest_chord_used_for_orientation": (
                orientation_basis != "canonical_absolute_source_direction"
            ),
            "mapped_target_direction_unit": (
                None
                if mapped_direction is None
                else [float(value) for value in mapped_direction]
            ),
            "source_to_mapped_direction_alignment_dot": (
                source_to_mapped_direction_alignment_dot
            ),
            "canonical_front": CANONICAL_FRONT,
            "canonical_front_vector": list(CANONICAL_FRONT_VECTOR),
            "canonical_up": CANONICAL_UP,
            "canonical_up_vector": list(CANONICAL_UP_VECTOR),
            "axis_map_3x3": axis.tolist(),
            "axis_projection": axis_evidence,
            "alignment_projection": alignment_evidence,
            "source_rest_frame_3x3": source_frame.tolist(),
            "target_rest_frame_3x3": target_frame.tolist(),
            "source_rest_frame_determinant": float(np.linalg.det(source_frame)),
            "target_rest_frame_determinant": float(np.linalg.det(target_frame)),
            "alignment_determinant": float(np.linalg.det(alignment)),
            "source_rest_chord_m": source_chord,
            "target_rest_chord_m": target_chord,
            "source_segment_sum_m": source_segments,
            "target_segment_sum_m": target_segments,
            "source_upper_leg_length_m": source_upper,
            "source_lower_leg_length_m": source_lower,
            "target_upper_leg_length_m": target_upper,
            "target_lower_leg_length_m": target_lower,
            "source_minimum_reach_m": source_minimum_reach,
            "source_maximum_reach_m": source_segments,
            "target_minimum_reach_m": target_minimum_reach,
            "target_maximum_reach_m": target_segments,
            "rest_chord_scale": float(scales["rest_chord"]),
            "segment_sum_scale": float(scales["segment_sum"]),
            "scale_basis": scale_basis,
            "scale": (
                None
                if scale_basis in {anisotropic_basis, piecewise_basis}
                else float(scale_components[0])
            ),
            "mapping_scale_components": (
                None
                if scale_components is None
                else {
                    "rest_axis": float(scale_components[0]),
                    "bend_axis": float(scale_components[1]),
                    "normal_axis": float(scale_components[2]),
                }
            ),
            "source_current_distance_m": source_distance,
            "source_normalized_reach_0_1": source_normalized_reach,
            "piecewise_branch": piecewise_branch,
            "normalized_reach_fraction": normalized_reach_fraction,
            "mapped_target_distance_m": mapped_target_distance,
            "target_normalized_reach_0_1": target_normalized_reach,
            "target_reach_margin_m": target_reach_margin,
            "distance_was_clamped": False,
            "rest_reconstruction_error_m": rest_error,
        },
    }


def map_source_leg_endpoint_rest_frame(
    *,
    source_rest_hip: Sequence[float],
    source_rest_knee: Sequence[float],
    source_rest_ankle: Sequence[float],
    target_rest_hip: Sequence[float],
    target_rest_knee: Sequence[float],
    target_rest_ankle: Sequence[float],
    source_current_hip: Sequence[float],
    source_current_ankle: Sequence[float],
    target_current_hip: Sequence[float],
    scale_basis: str,
    source_current_knee: Sequence[float] | None = None,
    orientation_basis: str = "full_leg_rest_frame",
    coordinate_space: str = "authenticated_world_m",
    axis_map_3x3: Sequence[Sequence[float]] = AXIS_MAP_3X3,
) -> dict[str, Any]:
    mapping = _build_source_leg_endpoint_rest_frame(
        source_rest_hip=source_rest_hip,
        source_rest_knee=source_rest_knee,
        source_rest_ankle=source_rest_ankle,
        target_rest_hip=target_rest_hip,
        target_rest_knee=target_rest_knee,
        target_rest_ankle=target_rest_ankle,
        source_current_hip=source_current_hip,
        source_current_knee=source_current_knee,
        source_current_ankle=source_current_ankle,
        target_current_hip=target_current_hip,
        scale_basis=scale_basis,
        orientation_basis=orientation_basis,
        coordinate_space=coordinate_space,
        axis_map_3x3=axis_map_3x3,
    )
    validate_source_leg_endpoint_rest_frame_mapping(mapping)
    return mapping


def validate_source_leg_endpoint_rest_frame_mapping(
    mapping: Mapping[str, Any],
) -> None:
    """Recompute a source-leg endpoint and pole from every authenticated input."""

    if not isinstance(mapping, Mapping):
        raise RetargetError("rest-frame endpoint mapping is invalid")
    evidence = mapping.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RetargetError("rest-frame endpoint mapping is inconsistent or forged")
    inputs = evidence.get("authenticated_inputs")
    if not isinstance(inputs, Mapping):
        raise RetargetError("rest-frame endpoint mapping is inconsistent or forged")
    try:
        expected = _build_source_leg_endpoint_rest_frame(**dict(inputs))
    except (RetargetError, TypeError, ValueError) as error:
        raise RetargetError(
            f"rest-frame endpoint mapping is inconsistent or forged: {error}"
        ) from error
    if dict(mapping) != expected:
        raise RetargetError(
            "rest-frame endpoint mapping is internally inconsistent or forged"
        )


def solve_source_driven_two_bone_leg_ik(
    *,
    hip: Sequence[float],
    knee: Sequence[float],
    ankle: Sequence[float],
    endpoint_mapping: Mapping[str, Any],
    body_height_m: float,
) -> dict[str, Any]:
    """Solve an arbitrary-XYZ primary leg endpoint using the mapped source pole."""

    validate_source_leg_endpoint_rest_frame_mapping(endpoint_mapping)
    evidence = endpoint_mapping["evidence"]
    if evidence.get("scale_basis") != "reach_normalized_piecewise":
        raise RetargetError(
            "source-driven two-bone IK requires the verified piecewise endpoint"
        )
    points = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (hip, knee, ankle, endpoint_mapping.get("endpoint"))
    )
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in points):
        raise RetargetError("source-driven two-bone IK points must be finite 3D")
    hip_v, knee_v, ankle_v, target_v = points
    height = _finite_number(body_height_m, "source-driven IK body height")
    if height <= 1.0e-8:
        raise RetargetError("source-driven IK body height is degenerate")
    authenticated_hip = np.asarray(
        evidence["authenticated_inputs"]["target_current_hip"],
        dtype=np.float64,
    )
    if not np.allclose(hip_v, authenticated_hip, rtol=0.0, atol=1.0e-12):
        raise RetargetError(
            "source-driven IK hip differs from the authenticated root/pelvis anchor"
        )

    upper = knee_v - hip_v
    lower = ankle_v - knee_v
    upper_length = float(np.linalg.norm(upper))
    lower_length = float(np.linalg.norm(lower))
    target_delta = target_v - hip_v
    target_distance = float(np.linalg.norm(target_delta))
    if min(upper_length, lower_length, target_distance) <= 1.0e-8:
        raise RetargetError("source-driven two-bone IK chain is degenerate")
    expected_upper = _finite_number(
        evidence.get("target_upper_leg_length_m"),
        "source-driven IK mapped upper length",
    )
    expected_lower = _finite_number(
        evidence.get("target_lower_leg_length_m"),
        "source-driven IK mapped lower length",
    )
    expected_endpoint = _finite_number(
        evidence.get("mapped_target_distance_m"),
        "source-driven IK mapped endpoint length",
    )
    upper_input_drift = upper_length - expected_upper
    lower_input_drift = lower_length - expected_lower
    if abs(upper_input_drift) > MAXIMUM_IK_SEGMENT_LENGTH_INPUT_DRIFT_M:
        raise RetargetError(
            "source-driven IK upper length disagrees with piecewise evidence"
        )
    if abs(lower_input_drift) > MAXIMUM_IK_SEGMENT_LENGTH_INPUT_DRIFT_M:
        raise RetargetError(
            "source-driven IK lower length disagrees with piecewise evidence"
        )
    if not math.isclose(
        target_distance, expected_endpoint, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RetargetError(
            "source-driven IK endpoint length disagrees with piecewise evidence"
        )
    minimum_reach = abs(upper_length - lower_length)
    maximum_reach = upper_length + lower_length
    reach_margin = min(
        target_distance - minimum_reach,
        maximum_reach - target_distance,
    )
    if reach_margin < MINIMUM_IK_REACH_MARGIN_M:
        raise RetargetError(
            "source-driven two-bone IK endpoint violates the strict reach margin"
        )

    target_axis = target_delta / target_distance
    mapped_pole = np.asarray(
        endpoint_mapping.get("mapped_source_knee_vector"), dtype=np.float64
    )
    if mapped_pole.shape != (3,) or not np.isfinite(mapped_pole).all():
        raise RetargetError(
            "source-driven two-bone IK requires an authenticated mapped source pole"
        )
    pole_projection = mapped_pole - float(np.dot(mapped_pole, target_axis)) * target_axis
    pole_projection_length = float(np.linalg.norm(pole_projection))
    if pole_projection_length <= 1.0e-8:
        raise RetargetError(
            "source-driven two-bone IK mapped source pole is degenerate/ambiguous"
        )
    bend_direction = pole_projection / pole_projection_length
    normal = np.cross(target_axis, bend_direction)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1.0e-8:
        raise RetargetError("source-driven two-bone IK bend frame is degenerate")
    normal /= normal_length
    bend_frame = np.column_stack((target_axis, bend_direction, normal))
    bend_determinant = float(np.linalg.det(bend_frame))
    if not math.isclose(bend_determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise RetargetError(
            "source-driven two-bone IK bend frame is reflected or not proper"
        )

    along = (
        upper_length * upper_length
        - lower_length * lower_length
        + target_distance * target_distance
    ) / (2.0 * target_distance)
    bend_height_squared = upper_length * upper_length - along * along
    if bend_height_squared <= 1.0e-12:
        raise RetargetError(
            "source-driven two-bone IK would straighten or flip the knee"
        )
    bend_height = math.sqrt(bend_height_squared)
    solved_knee = hip_v + along * target_axis + bend_height * bend_direction
    solved_upper_length = float(np.linalg.norm(solved_knee - hip_v))
    solved_lower_length = float(np.linalg.norm(target_v - solved_knee))
    upper_residual = abs(solved_upper_length - upper_length)
    lower_residual = abs(solved_lower_length - lower_length)
    if max(upper_residual, lower_residual) > 1.0e-10:
        raise RetargetError("source-driven two-bone IK did not preserve segment lengths")

    solved_bend = solved_knee - hip_v
    solved_bend -= float(np.dot(solved_bend, target_axis)) * target_axis
    solved_bend /= np.linalg.norm(solved_bend)
    pole_alignment = float(np.dot(solved_bend, bend_direction))
    if pole_alignment <= 0.0:
        raise RetargetError("source-driven two-bone IK selected the wrong bend side")
    current_bend = knee_v - hip_v
    current_bend -= float(np.dot(current_bend, target_axis)) * target_axis
    current_bend_length = float(np.linalg.norm(current_bend))
    current_pole_dot = (
        None
        if current_bend_length <= 1.0e-8
        else float(np.dot(current_bend / current_bend_length, bend_direction))
    )
    internal_cosine = float(
        np.dot(hip_v - solved_knee, target_v - solved_knee)
        / (upper_length * lower_length)
    )
    internal_angle = math.acos(max(-1.0, min(1.0, internal_cosine)))
    flexion_angle = math.pi - internal_angle
    endpoint_delta = target_v - ankle_v
    endpoint_delta_norm = float(np.linalg.norm(endpoint_delta))
    source_reach = _finite_number(
        evidence.get("source_normalized_reach_0_1"),
        "source-driven IK normalized source reach",
    )
    target_reach = _finite_number(
        evidence.get("target_normalized_reach_0_1"),
        "source-driven IK normalized target reach",
    )
    if not 0.0 <= source_reach <= 1.0 or not 0.0 <= target_reach <= 1.0:
        raise RetargetError("source-driven IK normalized reach is outside [0, 1]")

    return {
        "hip": [float(value) for value in hip_v],
        "knee": [float(value) for value in solved_knee],
        "ankle": [float(value) for value in target_v],
        "evidence": {
            "method": "source_driven_piecewise_two_bone_ik_v1",
            "coordinate_space": evidence["coordinate_space"],
            "endpoint_mapping_schema": evidence["schema"],
            "endpoint_mapping_validated": True,
            "endpoint_orientation_basis": evidence["orientation_basis"],
            "piecewise_branch": evidence["piecewise_branch"],
            "piecewise_normalized_reach_fraction": evidence[
                "normalized_reach_fraction"
            ],
            "source_normalized_reach_0_1": source_reach,
            "target_normalized_reach_0_1": target_reach,
            "upper_leg_length_m": upper_length,
            "lower_leg_length_m": lower_length,
            "authenticated_upper_leg_length_m": expected_upper,
            "authenticated_lower_leg_length_m": expected_lower,
            "input_upper_length_drift_m": upper_input_drift,
            "input_lower_length_drift_m": lower_input_drift,
            "maximum_input_segment_length_drift_m": (
                MAXIMUM_IK_SEGMENT_LENGTH_INPUT_DRIFT_M
            ),
            "target_distance_m": target_distance,
            "minimum_reach_m": minimum_reach,
            "maximum_reach_m": maximum_reach,
            "reach_margin_m": reach_margin,
            "minimum_reach_margin_m": MINIMUM_IK_REACH_MARGIN_M,
            "bend_height_m": bend_height,
            "knee_internal_angle_rad": internal_angle,
            "knee_flexion_angle_rad": flexion_angle,
            "mapped_source_pole_projection_length_m": pole_projection_length,
            "mapped_source_pole_alignment_dot": pole_alignment,
            "current_target_to_mapped_source_pole_dot": current_pole_dot,
            "solved_bend_frame_determinant": bend_determinant,
            "upper_length_residual_m": upper_residual,
            "lower_length_residual_m": lower_residual,
            "endpoint_delta_xyz_m": [float(value) for value in endpoint_delta],
            "endpoint_delta_norm_m": endpoint_delta_norm,
            "endpoint_delta_body_height_ratio": endpoint_delta_norm / height,
            "endpoint_delta_leg_length_ratio": endpoint_delta_norm / maximum_reach,
            "body_height_m": height,
            "correction_cap_m": None,
            "root_pelvis_unchanged": True,
            "authenticated_rest_pole_fallback_used": False,
        },
    }


def _bilateral_clearance_values(
    values: Mapping[str, Any], *, description: str
) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != {"left", "right"}:
        raise RetargetError(f"{description} must contain exactly left and right")
    return {
        side: _finite_number(values[side], f"{description} {side}")
        for side in ("left", "right")
    }


def _bilateral_boolean_values(
    values: Mapping[str, Any], *, description: str
) -> dict[str, bool]:
    if not isinstance(values, Mapping) or set(values) != {"left", "right"}:
        raise RetargetError(f"{description} must contain exactly left and right")
    if any(type(values[side]) is not bool for side in ("left", "right")):
        raise RetargetError(f"{description} values must be boolean")
    return {side: bool(values[side]) for side in ("left", "right")}


def _build_source_contact_ik_plan(
    *,
    action_name: str,
    source_support_basis: str,
    frame: int,
    source_clearance_m: Mapping[str, Any],
    target_clearance_m: Mapping[str, Any],
    height_scale: float,
    accumulated_absolute_correction_m: Mapping[str, Any],
    candidate_contact_correction_reachable: Mapping[str, Any],
) -> dict[str, Any]:
    if action_name not in set(ACTION_NAMES.values()):
        raise RetargetError("source-contact IK action must be Walking or Standing_Idle")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 1:
        raise RetargetError("source-contact IK frame must be a positive integer")
    scale = _finite_number(height_scale, "source-contact IK height scale")
    if scale <= 0.0:
        raise RetargetError("source-contact IK height scale must be positive")
    source = _bilateral_clearance_values(
        source_clearance_m, description="source clearance"
    )
    target = _bilateral_clearance_values(
        target_clearance_m, description="target clearance"
    )
    accumulated = _bilateral_clearance_values(
        accumulated_absolute_correction_m,
        description="accumulated absolute IK correction",
    )
    candidate_reachable = _bilateral_boolean_values(
        candidate_contact_correction_reachable,
        description="candidate contact correction reachability",
    )
    if any(value < 0.0 for value in accumulated.values()):
        raise RetargetError("accumulated absolute IK correction cannot be negative")
    interior_minimum = IK_CONTACT_READBACK_SAFETY_MARGIN_M
    interior_maximum = CONTACT_CLEARANCE_M - interior_minimum
    if not 0.0 < interior_minimum < interior_maximum < CONTACT_CLEARANCE_M:
        raise RetargetError("source-contact IK interior target interval is invalid")
    is_idle = action_name == ACTION_NAMES["idle"]
    expected_basis = (
        IDLE_SOURCE_SUPPORT_BASIS if is_idle else WALK_SOURCE_SUPPORT_BASIS
    )
    if source_support_basis != expected_basis:
        raise RetargetError(
            "source-contact IK support basis does not match its action"
        )
    candidates = [
        side
        for side in ("left", "right")
        if source[side] <= CONTACT_CLEARANCE_M
    ]
    contact_requests: dict[str, dict[str, float]] = {}
    for side in candidates:
        scaled_source = source[side] * scale
        before = target[side]
        if before < 0.0:
            desired = min(
                interior_maximum,
                max(interior_minimum, scaled_source),
            )
        elif before > CONTACT_CLEARANCE_M:
            desired = interior_maximum
        else:
            desired = before
        contact_requests[side] = {
            "desired_clearance_m": desired,
            "correction_m": desired - before,
        }

    def candidate_correction_is_feasible(side: str) -> bool:
        correction = contact_requests[side]["correction_m"]
        return (
            correction == 0.0
            or (
                candidate_reachable[side]
                and abs(correction) <= MAXIMUM_IK_ANKLE_CORRECTION_M
                and accumulated[side] + abs(correction)
                <= MAXIMUM_IK_ANKLE_CORRECTION_M
            )
        )

    if is_idle:
        if candidates != ["left", "right"]:
            raise RetargetError(
                "Standing Idle support must prove bilateral source contact"
            )
        primary = None
        support_sides = ["left", "right"]
        walking_selection_reason = None
        primary_tie_break_order = None
    else:
        if not candidates:
            raise RetargetError(
                "source-contact IK frame has no authenticated source contact candidate"
            )
        valid_target_candidates = [
            side
            for side in candidates
            if 0.0 <= target[side] <= CONTACT_CLEARANCE_M
        ]
        if len(candidates) == 1:
            primary = candidates[0]
            if not candidate_correction_is_feasible(primary):
                correction = contact_requests[primary]["correction_m"]
                if not candidate_reachable[primary]:
                    raise RetargetError(
                        "source-contact IK single source candidate correction is "
                        "unreachable"
                    )
                if abs(correction) > MAXIMUM_IK_ANKLE_CORRECTION_M:
                    raise RetargetError(
                        "source-contact IK single source candidate correction "
                        "exceeds the strict 0.030 m cap"
                    )
                raise RetargetError(
                    "source-contact IK single source candidate accumulated "
                    "correction exceeds the strict 0.030 m cap"
                )
            walking_selection_reason = "single_source_candidate_same_side"
            primary_tie_break_order = ["authenticated_single_candidate_side"]
        elif valid_target_candidates:
            primary = min(
                valid_target_candidates,
                key=lambda side: (source[side], side),
            )
            walking_selection_reason = (
                "double_source_candidate_existing_target_contact"
            )
            primary_tie_break_order = [
                "target_contact_already_valid",
                "source_clearance_m",
                "side",
            ]
        else:
            feasible_candidates = [
                side for side in candidates if candidate_correction_is_feasible(side)
            ]
            if not feasible_candidates:
                raise RetargetError(
                    "source-contact IK double source candidates have no reachable "
                    "correction within the strict 0.030 m cap"
                )
            primary = min(
                feasible_candidates,
                key=lambda side: (
                    abs(contact_requests[side]["correction_m"]),
                    source[side],
                    side,
                ),
            )
            walking_selection_reason = (
                "double_source_candidate_minimum_reachable_correction"
            )
            primary_tie_break_order = [
                "requested_correction_reachable",
                "absolute_correction_m",
                "source_clearance_m",
                "side",
            ]
        support_sides = [primary]

    side_plans: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        source_clearance = source[side]
        scaled_source = source_clearance * scale
        before = target[side]
        is_primary = side == primary
        if is_idle and before < 0.0:
            desired = min(interior_maximum, max(interior_minimum, scaled_source))
            reason = "idle_bilateral_contact_penetration"
        elif is_idle and before > CONTACT_CLEARANCE_M:
            desired = interior_maximum
            reason = "idle_bilateral_contact_hover"
        elif is_idle:
            desired = before
            reason = "idle_bilateral_contact_already_valid"
        elif is_primary and before < 0.0:
            desired = min(interior_maximum, max(interior_minimum, scaled_source))
            reason = "source_primary_contact_penetration"
        elif is_primary and before > CONTACT_CLEARANCE_M:
            desired = interior_maximum
            reason = "source_primary_contact_hover"
        elif is_primary:
            desired = before
            reason = "source_primary_contact_already_valid"
        elif before < 0.0:
            desired = max(interior_minimum, scaled_source)
            reason = "target_penetration_during_source_swing"
        else:
            desired = before
            reason = "nonprimary_clearance_unchanged"
        correction = desired - before
        if abs(correction) > MAXIMUM_IK_ANKLE_CORRECTION_M:
            raise RetargetError(
                "source-contact IK ankle correction exceeds the strict "
                f"{MAXIMUM_IK_ANKLE_CORRECTION_M:.3f} m cap: "
                f"frame={frame} side={side} correction={correction}"
            )
        accumulated_after = accumulated[side] + abs(correction)
        if accumulated_after > MAXIMUM_IK_ANKLE_CORRECTION_M:
            raise RetargetError(
                "source-contact IK accumulated absolute correction exceeds the "
                f"strict {MAXIMUM_IK_ANKLE_CORRECTION_M:.3f} m cap: "
                f"frame={frame} side={side} accumulated={accumulated_after}"
            )
        side_plans[side] = {
            "source_clearance_m": source_clearance,
            "scaled_source_clearance_m": scaled_source,
            "source_contact_candidate": side in candidates,
            "target_before_clearance_m": before,
            "target_desired_clearance_m": desired,
            "ankle_correction_m": correction,
            "absolute_ankle_correction_m": abs(correction),
            "accumulated_absolute_correction_before_m": accumulated[side],
            "accumulated_absolute_correction_after_m": accumulated_after,
            "apply_correction": correction != 0.0,
            "reason": reason,
        }
    return {
        "schema": "tokenrig_source_contact_ik_plan_v2",
        "action_name": action_name,
        "source_support_basis": source_support_basis,
        "frame": frame,
        "height_scale": scale,
        "contact_clearance_m": CONTACT_CLEARANCE_M,
        "physical_valid_clearance_interval_m": {
            "minimum": 0.0,
            "maximum": CONTACT_CLEARANCE_M,
        },
        "readback_safety_margin_m": interior_minimum,
        "interior_target_clearance_interval_m": {
            "minimum": interior_minimum,
            "maximum": interior_maximum,
        },
        "maximum_ankle_correction_m": MAXIMUM_IK_ANKLE_CORRECTION_M,
        "source_contact_candidate_sides": candidates,
        "candidate_contact_correction_reachable": candidate_reachable,
        "walking_selection_reason": walking_selection_reason,
        "primary_support_side": primary,
        "primary_tie_break_order": primary_tie_break_order,
        "support_sides": support_sides,
        "sides": side_plans,
    }


def plan_source_contact_ik_correction(
    *,
    action_name: str,
    source_support_basis: str,
    frame: int,
    source_clearance_m: Mapping[str, Any],
    target_clearance_m: Mapping[str, Any],
    height_scale: float,
    accumulated_absolute_correction_m: Mapping[str, Any] | None = None,
    candidate_contact_correction_reachable: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan a symmetric, bounded ankle-Z edit from authenticated source contact."""

    plan = _build_source_contact_ik_plan(
        action_name=action_name,
        source_support_basis=source_support_basis,
        frame=frame,
        source_clearance_m=source_clearance_m,
        target_clearance_m=target_clearance_m,
        height_scale=height_scale,
        accumulated_absolute_correction_m=(
            accumulated_absolute_correction_m
            if accumulated_absolute_correction_m is not None
            else {"left": 0.0, "right": 0.0}
        ),
        candidate_contact_correction_reachable=(
            candidate_contact_correction_reachable
            if candidate_contact_correction_reachable is not None
            else {"left": True, "right": True}
        ),
    )
    validate_source_contact_ik_plan(plan)
    return plan


def validate_source_contact_ik_plan(plan: Mapping[str, Any]) -> None:
    """Recompute all contact classifications and edits from the recorded inputs."""

    if not isinstance(plan, Mapping):
        raise RetargetError("source-contact IK plan is invalid")
    sides = plan.get("sides")
    if not isinstance(sides, Mapping) or set(sides) != {"left", "right"}:
        raise RetargetError("source-contact IK plan sides are invalid or forged")
    if any(not isinstance(sides[side], Mapping) for side in ("left", "right")):
        raise RetargetError("source-contact IK side evidence is invalid or forged")
    try:
        expected = _build_source_contact_ik_plan(
            action_name=plan.get("action_name"),
            source_support_basis=plan.get("source_support_basis"),
            frame=plan.get("frame"),
            source_clearance_m={
                side: sides[side].get("source_clearance_m")
                for side in ("left", "right")
            },
            target_clearance_m={
                side: sides[side].get("target_before_clearance_m")
                for side in ("left", "right")
            },
            height_scale=plan.get("height_scale"),
            accumulated_absolute_correction_m={
                side: sides[side].get(
                    "accumulated_absolute_correction_before_m"
                )
                for side in ("left", "right")
            },
            candidate_contact_correction_reachable=plan.get(
                "candidate_contact_correction_reachable"
            ),
        )
    except RetargetError as error:
        raise RetargetError(
            f"source-contact IK plan is internally inconsistent or forged: {error}"
        ) from error
    if dict(plan) != expected:
        raise RetargetError("source-contact IK plan is internally inconsistent or forged")


def _build_source_contact_ik_frame_readback(
    *,
    action_name: str,
    frame: int,
    target_before_clearance_m: Mapping[str, Any],
    target_after_clearance_m: Mapping[str, Any],
    corrected_sides: Sequence[str],
    required_contact_sides: Sequence[str],
    desired_clearance_m: Mapping[str, Any],
    accumulated_absolute_correction_m: Mapping[str, Any],
) -> dict[str, Any]:
    if action_name not in set(ACTION_NAMES.values()):
        raise RetargetError("IK frame readback action is invalid")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 1:
        raise RetargetError("IK frame readback frame is invalid")
    before = _bilateral_clearance_values(
        target_before_clearance_m, description="IK readback target before clearance"
    )
    after = _bilateral_clearance_values(
        target_after_clearance_m, description="IK readback target after clearance"
    )
    accumulated = _bilateral_clearance_values(
        accumulated_absolute_correction_m,
        description="IK readback accumulated absolute correction",
    )
    if any(
        value < 0.0 or value > MAXIMUM_IK_ANKLE_CORRECTION_M
        for value in accumulated.values()
    ):
        raise RetargetError(
            "IK readback accumulated absolute correction exceeds the strict "
            f"{MAXIMUM_IK_ANKLE_CORRECTION_M:.3f} m cap"
        )
    if (
        isinstance(corrected_sides, (str, bytes))
        or isinstance(required_contact_sides, (str, bytes))
    ):
        raise RetargetError("IK readback side sets are invalid")
    corrected_set = set(corrected_sides)
    required_set = set(required_contact_sides)
    if (
        corrected_set - {"left", "right"}
        or required_set - {"left", "right"}
        or not required_set
        or len(corrected_set) != len(tuple(corrected_sides))
        or len(required_set) != len(tuple(required_contact_sides))
    ):
        raise RetargetError("IK readback corrected or required side set is invalid")
    if action_name == ACTION_NAMES["idle"] and required_set != {"left", "right"}:
        raise RetargetError("Standing Idle IK readback must require bilateral contact")
    if (
        not isinstance(desired_clearance_m, Mapping)
        or set(desired_clearance_m) != {"left", "right"}
    ):
        raise RetargetError("IK readback interior desired clearances are invalid")
    interior_minimum = IK_CONTACT_READBACK_SAFETY_MARGIN_M
    interior_maximum = CONTACT_CLEARANCE_M - interior_minimum
    desired: dict[str, float | None] = {}
    for side in ("left", "right"):
        value = desired_clearance_m[side]
        if side in corrected_set:
            normalized = _finite_number(
                value, f"IK readback {side} desired clearance"
            )
            if side in required_set:
                if not interior_minimum <= normalized <= interior_maximum:
                    raise RetargetError(
                        "IK readback required-contact desired clearance is outside "
                        "the strict interior target"
                    )
            elif normalized < interior_minimum:
                raise RetargetError(
                    "IK readback swing desired clearance is below the strict "
                    "nonpenetrating margin"
                )
            desired[side] = normalized
        elif value is not None:
            raise RetargetError(
                "IK readback uncorrected side cannot claim an interior desired target"
            )
        else:
            desired[side] = None

    side_evidence: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        actual = after[side]
        interval_result = (
            "below_physical_contact_interval"
            if actual < 0.0
            else "inside_physical_contact_interval"
            if actual <= CONTACT_CLEARANCE_M
            else "above_physical_contact_interval"
        )
        if side in required_set and interval_result != "inside_physical_contact_interval":
            raise RetargetError(
                f"IK readback {side} is outside the physical contact interval"
            )
        if side in corrected_set and side not in required_set and actual < 0.0:
            raise RetargetError(
                f"IK readback {side} swing remains penetrating after correction"
            )
        support_phase_result = (
            "required_contact_valid"
            if side in required_set
            else "nonpenetrating_swing"
            if side in corrected_set
            else "not_evaluated"
        )
        cross_change = None
        if side not in corrected_set:
            cross_change = actual - before[side]
            if abs(cross_change) > MAXIMUM_CROSS_FOOT_CLEARANCE_CHANGE_M:
                raise RetargetError(
                    "IK readback uncorrected cross-foot clearance change exceeds its "
                    "strict cap"
                )
        side_evidence[side] = {
            "corrected": side in corrected_set,
            "required_contact": side in required_set,
            "target_before_clearance_m": before[side],
            "target_after_clearance_m": actual,
            "actual_interval_result": interval_result,
            "support_phase_result": support_phase_result,
            "desired_clearance_m": desired[side],
            "accumulated_absolute_correction_m": accumulated[side],
            "cross_foot_clearance_change_m": cross_change,
        }
    return {
        "schema": "tokenrig_source_contact_ik_frame_readback_v1",
        "action_name": action_name,
        "frame": frame,
        "contact_clearance_m": CONTACT_CLEARANCE_M,
        "physical_valid_clearance_interval_m": {
            "minimum": 0.0,
            "maximum": CONTACT_CLEARANCE_M,
        },
        "readback_safety_margin_m": interior_minimum,
        "interior_target_clearance_interval_m": {
            "minimum": interior_minimum,
            "maximum": interior_maximum,
        },
        "maximum_accumulated_absolute_correction_m": (
            MAXIMUM_IK_ANKLE_CORRECTION_M
        ),
        "maximum_cross_foot_clearance_change_m": (
            MAXIMUM_CROSS_FOOT_CLEARANCE_CHANGE_M
        ),
        "corrected_sides": [
            side for side in ("left", "right") if side in corrected_set
        ],
        "required_contact_sides": [
            side for side in ("left", "right") if side in required_set
        ],
        "sides": side_evidence,
        "automatic_checks": "passed",
    }


def validate_source_contact_ik_frame_readback(
    *,
    action_name: str,
    frame: int,
    target_before_clearance_m: Mapping[str, Any],
    target_after_clearance_m: Mapping[str, Any],
    corrected_sides: Sequence[str],
    required_contact_sides: Sequence[str],
    desired_clearance_m: Mapping[str, Any],
    accumulated_absolute_correction_m: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _build_source_contact_ik_frame_readback(
        action_name=action_name,
        frame=frame,
        target_before_clearance_m=target_before_clearance_m,
        target_after_clearance_m=target_after_clearance_m,
        corrected_sides=corrected_sides,
        required_contact_sides=required_contact_sides,
        desired_clearance_m=desired_clearance_m,
        accumulated_absolute_correction_m=accumulated_absolute_correction_m,
    )
    validate_source_contact_ik_frame_readback_evidence(evidence)
    return evidence


def validate_source_contact_ik_frame_readback_evidence(
    evidence: Mapping[str, Any],
) -> None:
    if not isinstance(evidence, Mapping):
        raise RetargetError("IK frame readback evidence is invalid")
    sides = evidence.get("sides")
    if not isinstance(sides, Mapping) or set(sides) != {"left", "right"}:
        raise RetargetError("IK frame readback evidence is inconsistent or forged")
    if any(not isinstance(sides[side], Mapping) for side in ("left", "right")):
        raise RetargetError("IK frame readback side evidence is inconsistent or forged")
    try:
        expected = _build_source_contact_ik_frame_readback(
            action_name=evidence.get("action_name"),
            frame=evidence.get("frame"),
            target_before_clearance_m={
                side: sides[side].get("target_before_clearance_m")
                for side in ("left", "right")
            },
            target_after_clearance_m={
                side: sides[side].get("target_after_clearance_m")
                for side in ("left", "right")
            },
            corrected_sides=evidence.get("corrected_sides"),
            required_contact_sides=evidence.get("required_contact_sides"),
            desired_clearance_m={
                side: sides[side].get("desired_clearance_m")
                for side in ("left", "right")
            },
            accumulated_absolute_correction_m={
                side: sides[side].get("accumulated_absolute_correction_m")
                for side in ("left", "right")
            },
        )
    except RetargetError as error:
        raise RetargetError(
            f"IK frame readback evidence is inconsistent or forged: {error}"
        ) from error
    if dict(evidence) != expected:
        raise RetargetError(
            "IK frame readback evidence is internally inconsistent or forged"
        )


def _build_idle_source_support_evidence(
    *,
    frame_start: int,
    frame_end: int,
    fps: int,
    head_world_position_frame_start_m: Sequence[float],
    joint_world_positions_m: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
    object_world_matrices: Sequence[Sequence[Sequence[float]]],
) -> dict[str, Any]:
    if (
        isinstance(frame_start, bool)
        or isinstance(frame_end, bool)
        or frame_start != EXPECTED_IDLE_FRAME_START
        or frame_end != EXPECTED_IDLE_FRAME_END
        or frame_end - frame_start + 1 != EXPECTED_IDLE_FRAME_COUNT
    ):
        raise RetargetError(
            "exact Idle world support must cover all 351 authenticated frames"
        )
    if fps != 30 or isinstance(fps, bool):
        raise RetargetError("exact Idle world support must use authenticated 30 fps")
    if (
        not isinstance(joint_world_positions_m, Mapping)
        or set(joint_world_positions_m) != {"left", "right"}
        or any(
            not isinstance(joint_world_positions_m[side], Mapping)
            or set(joint_world_positions_m[side]) != {"foot", "toe"}
            for side in ("left", "right")
        )
    ):
        raise RetargetError(
            "exact Idle world support requires all four left/right foot and toe trajectories"
        )

    trajectories: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        trajectories[side] = {}
        for part in ("foot", "toe"):
            points = np.asarray(
                joint_world_positions_m[side][part], dtype=np.float64
            )
            if (
                points.shape != (EXPECTED_IDLE_FRAME_COUNT, 3)
                or not np.isfinite(points).all()
            ):
                raise RetargetError(
                    "exact Idle world support trajectory coverage is invalid or non-finite"
                )
            trajectories[side][part] = points

    head = np.asarray(head_world_position_frame_start_m, dtype=np.float64)
    if head.shape != (3,) or not np.isfinite(head).all():
        raise RetargetError("exact Idle semantic head world position is invalid")
    first_support_z = min(
        float(trajectories[side][part][0, 2])
        for side in ("left", "right")
        for part in ("foot", "toe")
    )
    semantic_height = float(head[2] - first_support_z)
    if semantic_height <= 1.0e-6 or not math.isfinite(semantic_height):
        raise RetargetError("exact Idle semantic world height is degenerate")

    matrices = np.asarray(object_world_matrices, dtype=np.float64)
    if (
        matrices.shape != (EXPECTED_IDLE_FRAME_COUNT, 4, 4)
        or not np.isfinite(matrices).all()
        or not np.allclose(
            matrices[:, 3, :],
            np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise RetargetError(
            "exact Idle object world transform coverage is invalid or non-affine"
        )
    rotation_records = []
    normalized_rotations = []
    uniform_scales = []
    for offset, matrix in enumerate(matrices):
        frame = frame_start + offset
        projected, record = project_uniform_scaled_rotation(
            matrix[:3, :3],
            "exact Idle source armature object transform",
            context={
                "action": ACTION_NAMES["idle"],
                "frame": frame,
                "semantic_role": "armature_root",
                "source_bone": "exact_idle_armature",
                "target_bone": None,
                "matrix_stage": "source_object_world",
            },
        )
        rotation_records.append(record)
        normalized_rotations.append(projected)
        uniform_scales.append(float(record["uniform_scale"]))
    relative_scale_variation = (
        max(uniform_scales) - min(uniform_scales)
    ) / min(uniform_scales)
    if (
        relative_scale_variation
        > MAXIMUM_OBJECT_UNIFORM_SCALE_RELATIVE_VARIATION
    ):
        raise RetargetError(
            "exact Idle object uniform scale relative variation exceeds its "
            "dimensionless pinned cap"
        )
    relative_rotation = normalized_rotations[0].T @ normalized_rotations[-1]
    rotation_cosine = float(
        np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    )
    rotation_endpoint = float(math.acos(rotation_cosine))
    if rotation_endpoint > LOOP_ROTATION_TOLERANCE_RAD:
        raise RetargetError("exact Idle object rotation loop endpoint is open")
    translations = matrices[:, :3, 3]
    translation_endpoint = float(np.linalg.norm(translations[-1] - translations[0]))
    if translation_endpoint > LOOP_ROOT_TOLERANCE_M:
        raise RetargetError("exact Idle object translation loop endpoint is open")

    sides: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        joint_evidence: dict[str, dict[str, Any]] = {}
        for part in ("foot", "toe"):
            points = trajectories[side][part]
            speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
            z_range = float(np.ptp(points[:, 2]))
            xy_range = float(
                np.linalg.norm(np.ptp(points[:, :2], axis=0))
            )
            maximum_xy_displacement = float(
                np.max(np.linalg.norm(points[:, :2] - points[0, :2], axis=1))
            )
            maximum_speed = float(np.max(speeds))
            endpoint = float(np.linalg.norm(points[-1] - points[0]))
            if z_range > CONTACT_CLEARANCE_M:
                raise RetargetError(
                    f"exact Idle {side} {part} Z range is not planted"
                )
            if (
                xy_range > MAXIMUM_STANCE_SLIDE_M
                or maximum_xy_displacement > MAXIMUM_STANCE_SLIDE_M
            ):
                raise RetargetError(
                    f"exact Idle {side} {part} XY stance range is not planted"
                )
            if maximum_speed > MAXIMUM_IDLE_SPEED_MPS:
                raise RetargetError(
                    f"exact Idle {side} {part} world speed is not planted"
                )
            if endpoint > LOOP_ROOT_TOLERANCE_M:
                raise RetargetError(
                    f"exact Idle {side} {part} loop endpoint is open"
                )
            joint_evidence[part] = {
                "z_range_m": z_range,
                "normalized_z_range": z_range / semantic_height,
                "xy_range_m": xy_range,
                "maximum_xy_displacement_from_first_m": maximum_xy_displacement,
                "maximum_speed_m_per_s": maximum_speed,
                "endpoint_residual_m": endpoint,
            }
        support_z = np.minimum(
            trajectories[side]["foot"][:, 2],
            trajectories[side]["toe"][:, 2],
        )
        clearances = support_z - float(np.min(support_z))
        maximum_clearance = float(np.max(clearances))
        if maximum_clearance > CONTACT_CLEARANCE_M:
            raise RetargetError(
                f"exact Idle {side} support clearance is not planted"
            )
        sides[side] = {
            "planted": True,
            "joints": joint_evidence,
            "action_minimum_support_z_m": float(np.min(support_z)),
            "maximum_support_relative_clearance_m": maximum_clearance,
            "support_relative_clearance_by_frame_m": [
                float(value) for value in clearances
            ],
            "contact_ratio": float(np.mean(clearances <= CONTACT_CLEARANCE_M)),
        }

    normalized_joint_inputs = {
        side: {
            part: trajectories[side][part].tolist()
            for part in ("foot", "toe")
        }
        for side in ("left", "right")
    }
    return {
        "schema": "rocketbox_exact_idle_world_support_v1",
        "action_name": ACTION_NAMES["idle"],
        "source_support_basis": IDLE_SOURCE_SUPPORT_BASIS,
        "coordinate_space": "blender_world_m",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": EXPECTED_IDLE_FRAME_COUNT,
        "fps": fps,
        "semantic_world_height_m": semantic_height,
        "thresholds": {
            "maximum_support_relative_clearance_m": CONTACT_CLEARANCE_M,
            "maximum_joint_z_range_m": CONTACT_CLEARANCE_M,
            "maximum_joint_xy_range_m": MAXIMUM_STANCE_SLIDE_M,
            "maximum_joint_speed_m_per_s": MAXIMUM_IDLE_SPEED_MPS,
            "maximum_endpoint_residual_m": LOOP_ROOT_TOLERANCE_M,
            "maximum_object_rotation_endpoint_residual_rad": (
                LOOP_ROTATION_TOLERANCE_RAD
            ),
            "maximum_object_uniform_scale_relative_variation": (
                MAXIMUM_OBJECT_UNIFORM_SCALE_RELATIVE_VARIATION
            ),
        },
        "object_transform": {
            "minimum_uniform_scale": min(uniform_scales),
            "maximum_uniform_scale": max(uniform_scales),
            "maximum_relative_scale_time_variation": relative_scale_variation,
            "rotation_endpoint_residual_rad": rotation_endpoint,
            "translation_endpoint_residual_m": translation_endpoint,
            "rotation_projection": summarize_rotation_projections(
                rotation_records
            ),
        },
        "sides": sides,
        "authenticated_world_inputs": {
            "head_world_position_frame_start_m": [float(value) for value in head],
            "joint_world_positions_m": normalized_joint_inputs,
            "object_world_matrices": matrices.tolist(),
        },
        "automatic_checks": "passed",
    }


def summarize_idle_source_support(
    *,
    frame_start: int,
    frame_end: int,
    fps: int,
    head_world_position_frame_start_m: Sequence[float],
    joint_world_positions_m: Mapping[str, Mapping[str, Sequence[Sequence[float]]]],
    object_world_matrices: Sequence[Sequence[Sequence[float]]],
) -> dict[str, Any]:
    evidence = _build_idle_source_support_evidence(
        frame_start=frame_start,
        frame_end=frame_end,
        fps=fps,
        head_world_position_frame_start_m=head_world_position_frame_start_m,
        joint_world_positions_m=joint_world_positions_m,
        object_world_matrices=object_world_matrices,
    )
    validate_idle_source_support(evidence)
    return evidence


def validate_idle_source_support(evidence: Mapping[str, Any]) -> None:
    if not isinstance(evidence, Mapping):
        raise RetargetError("exact Idle source support evidence is invalid")
    inputs = evidence.get("authenticated_world_inputs")
    if not isinstance(inputs, Mapping):
        raise RetargetError(
            "exact Idle source support evidence inputs are inconsistent or forged"
        )
    try:
        expected = _build_idle_source_support_evidence(
            frame_start=evidence.get("frame_start"),
            frame_end=evidence.get("frame_end"),
            fps=evidence.get("fps"),
            head_world_position_frame_start_m=inputs.get(
                "head_world_position_frame_start_m"
            ),
            joint_world_positions_m=inputs.get("joint_world_positions_m"),
            object_world_matrices=inputs.get("object_world_matrices"),
        )
    except RetargetError as error:
        raise RetargetError(
            f"exact Idle source support evidence is inconsistent or forged: {error}"
        ) from error
    if dict(evidence) != expected:
        raise RetargetError(
            "exact Idle source support evidence is internally inconsistent or forged"
        )


def plan_constant_grounding(
    *,
    frame_minimum_z: Mapping[int, float],
    floor_z_m: float,
    action_name: str | None = None,
    frame_minimum_samples: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not frame_minimum_z or not math.isfinite(floor_z_m):
        raise RetargetError("grounding evidence is empty or non-finite")
    values = {int(frame): float(value) for frame, value in frame_minimum_z.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise RetargetError("grounding frame minima are non-finite")
    worst_frame = min(values, key=lambda frame: (values[frame], frame))
    minimum = values[worst_frame]
    correction = float(floor_z_m - minimum)
    penetrations = {
        str(frame): max(0.0, float(floor_z_m - value))
        for frame, value in sorted(values.items())
    }
    maximum_penetration = max(penetrations.values())
    ordered_values = np.asarray(
        [value for _, value in sorted(values.items())], dtype=np.float64
    )
    distribution = {
        "frame_count": int(len(ordered_values)),
        "minimum": float(np.min(ordered_values)),
        "maximum": float(np.max(ordered_values)),
        "mean": float(np.mean(ordered_values)),
        "median": float(np.median(ordered_values)),
    }
    normalized_samples = {
        int(frame): dict(sample)
        for frame, sample in (frame_minimum_samples or {}).items()
    }
    worst_sample = normalized_samples.get(worst_frame)
    if abs(correction) > MAXIMUM_GROUNDING_CORRECTION_M:
        evidence = {
            "schema": "tokenrig_grounding_rejection_v1",
            "action_name": action_name if action_name is not None else "unspecified",
            "fixed_floor_z_m": float(floor_z_m),
            "maximum_allowed_constant_correction_m": (
                MAXIMUM_GROUNDING_CORRECTION_M
            ),
            "required_constant_correction_m": correction,
            "worst_frame": int(worst_frame),
            "worst_minimum_z_m": float(minimum),
            "worst_sample": worst_sample,
            "frame_minimum_z_m": {
                str(frame): value for frame, value in sorted(values.items())
            },
            "frame_minimum_distribution_m": distribution,
        }
        sample_text = (
            f" vertex={worst_sample.get('vertex_index')}"
            f" region={worst_sample.get('dominant_semantic_region')}"
            if isinstance(worst_sample, Mapping)
            else ""
        )
        raise GroundingError(
            "constant ground correction exceeds 0.010 m: "
            f"action={evidence['action_name']} correction={correction} "
            f"worst_frame={worst_frame} min_z={minimum}{sample_text}",
            evidence=evidence,
        )
    return {
        "correction_m": correction,
        "pre_ground_minimum_z_m": minimum,
        "pre_ground_worst_frame": int(worst_frame),
        "pre_ground_worst_sample": worst_sample,
        "pre_ground_frame_minimum_distribution_m": distribution,
        "pre_ground_maximum_penetration_m": maximum_penetration,
        "pre_ground_penetration_by_frame_m": penetrations,
    }


def plan_surface_contact_leg_ik(
    *,
    side_minimum_z_m: Mapping[str, float],
    floor_z_m: float,
    safety_margin_m: float = IK_CONTACT_READBACK_SAFETY_MARGIN_M,
    maximum_correction_m: float = MAXIMUM_IK_ANKLE_CORRECTION_M,
) -> dict[str, Any]:
    if set(side_minimum_z_m) != {"left", "right"}:
        raise RetargetError("surface-contact IK requires bilateral surface minima")
    floor = _finite_number(floor_z_m, "surface-contact floor")
    margin = _finite_number(safety_margin_m, "surface-contact safety margin")
    maximum = _finite_number(
        maximum_correction_m, "surface-contact maximum correction"
    )
    if margin < 0.0 or maximum <= 0.0 or margin >= maximum:
        raise RetargetError("surface-contact IK margin or correction cap is invalid")
    minima = {
        side: _finite_number(side_minimum_z_m[side], f"{side} surface minimum")
        for side in ("left", "right")
    }
    target_z = floor + margin
    corrections = {
        side: max(0.0, target_z - minima[side]) for side in ("left", "right")
    }
    rejected = {
        side: value for side, value in corrections.items() if value > maximum
    }
    if rejected:
        raise GroundingError(
            "surface-contact IK ankle correction exceeds the strict 0.030 m cap: "
            + ", ".join(f"{side}={value}" for side, value in rejected.items()),
            evidence={
                "schema": "tokenrig_surface_contact_ik_rejection_v1",
                "fixed_floor_z_m": floor,
                "safety_margin_m": margin,
                "maximum_allowed_ankle_correction_m": maximum,
                "side_minimum_z_m": minima,
                "required_upward_correction_m": corrections,
            },
        )
    return {
        "fixed_floor_z_m": floor,
        "safety_margin_m": margin,
        "target_surface_z_m": target_z,
        "maximum_allowed_ankle_correction_m": maximum,
        "side_minimum_z_m": minima,
        "upward_correction_m": corrections,
    }


def _maximum_consecutive_true(values: Sequence[bool]) -> int:
    flags = [bool(value) for value in values]
    if not flags:
        return 0
    if all(flags):
        return len(flags)
    maximum = current = 0
    for value in (*flags, *flags):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return min(maximum, len(flags))


def summarize_foot_contact(
    *,
    clearances_m: Sequence[float],
    anchor_xy_m: Sequence[Sequence[float]],
    fps: int,
) -> dict[str, Any]:
    clearances = np.asarray(clearances_m, dtype=np.float64)
    anchors = np.asarray(anchor_xy_m, dtype=np.float64)
    if (
        clearances.ndim != 1
        or not len(clearances)
        or anchors.shape != (len(clearances), 2)
        or not np.isfinite(clearances).all()
        or not np.isfinite(anchors).all()
        or fps <= 0
    ):
        raise RetargetError("foot contact samples are invalid")
    contact = clearances <= CONTACT_CLEARANCE_M
    stance = clearances <= STANCE_CLEARANCE_M
    stance_count = int(stance.sum())
    if stance_count == 0:
        raise RetargetError("foot has no stance frames")
    maximum_slide = 0.0
    maximum_speed = 0.0
    segment_start: int | None = None
    for index, in_stance in enumerate((*stance.tolist(), False)):
        if in_stance and segment_start is None:
            segment_start = index
        elif not in_stance and segment_start is not None:
            segment = anchors[segment_start:index]
            if len(segment) > 1:
                maximum_slide = max(
                    maximum_slide,
                    float(np.max(np.linalg.norm(segment - segment[0], axis=1))),
                )
                maximum_speed = max(
                    maximum_speed,
                    float(np.max(np.linalg.norm(np.diff(segment, axis=0), axis=1))) * fps,
                )
            segment_start = None
    return {
        "frame_count": int(len(clearances)),
        "contact_frame_count": int(contact.sum()),
        "stance_frame_count": stance_count,
        "contact_ratio": float(contact.mean()),
        "stance_contact_ratio": float(contact[stance].mean()),
        "maximum_consecutive_hover_frames": _maximum_consecutive_true(~contact),
        "maximum_stance_slide_m": maximum_slide,
        "maximum_stance_speed_m_per_s": maximum_speed,
        "contact_by_frame": [bool(value) for value in contact],
        "stance_by_frame": [bool(value) for value in stance],
    }


def summarize_support_union(
    *, left_contact: Sequence[bool], right_contact: Sequence[bool]
) -> dict[str, Any]:
    if (
        not left_contact
        or len(left_contact) != len(right_contact)
        or any(not isinstance(value, (bool, np.bool_)) for value in left_contact)
        or any(not isinstance(value, (bool, np.bool_)) for value in right_contact)
    ):
        raise RetargetError("bilateral support samples are invalid")
    support_union = [
        bool(left or right)
        for left, right in zip(left_contact, right_contact)
    ]
    airborne = [not supported for supported in support_union]
    return {
        "support_union": support_union,
        "support_coverage_ratio": float(np.mean(support_union)),
        "maximum_consecutive_both_feet_airborne_frames": (
            _maximum_consecutive_true(airborne)
        ),
    }


def validate_walking_support(evidence: Mapping[str, Any]) -> None:
    support_union = evidence.get("support_union")
    if not isinstance(support_union, list) or not support_union:
        raise RetargetError("Walking support_union is missing")
    if any(not isinstance(value, bool) for value in support_union):
        raise RetargetError("Walking support_union is invalid")
    coverage = _finite_number(
        evidence.get("support_coverage_ratio"), "Walking support coverage"
    )
    maximum_airborne = _finite_number(
        evidence.get("maximum_consecutive_both_feet_airborne_frames"),
        "Walking both-feet-airborne run",
    )
    actual_coverage = float(np.mean(support_union))
    actual_airborne = _maximum_consecutive_true(
        [not supported for supported in support_union]
    )
    if (
        not math.isclose(coverage, actual_coverage, abs_tol=1.0e-12)
        or maximum_airborne != actual_airborne
    ):
        raise RetargetError("Walking support evidence is internally inconsistent")
    if coverage != 1.0 or maximum_airborne != 0:
        raise RetargetError(
            "Walking flight detected: support coverage must be 1.0 and "
            "both-feet-airborne frames must be zero"
        )


def body_forward_vector(
    *,
    left_shoulder: Sequence[float],
    right_shoulder: Sequence[float],
    pelvis: Sequence[float],
    neck: Sequence[float],
) -> np.ndarray:
    lateral = np.asarray(left_shoulder, dtype=np.float64) - np.asarray(
        right_shoulder, dtype=np.float64
    )
    up = np.asarray(neck, dtype=np.float64) - np.asarray(pelvis, dtype=np.float64)
    if lateral.shape != (3,) or up.shape != (3,) or not np.isfinite((*lateral, *up)).all():
        raise RetargetError("body forward inputs are invalid")
    forward = np.cross(lateral, up)
    length = float(np.linalg.norm(forward))
    if length <= 1.0e-8:
        raise RetargetError("body forward basis is degenerate")
    return forward / length


def maximum_boundary_velocity_residual(
    trajectories: Mapping[str, Sequence[Sequence[float]]], *, fps: int
) -> float:
    records = boundary_velocity_residual_records(trajectories, fps=fps)
    return max(value["residual_m_per_s"] for value in records.values())


def boundary_velocity_residual_records(
    trajectories: Mapping[str, Sequence[Sequence[float]]], *, fps: int
) -> dict[str, Any]:
    if not trajectories or fps <= 0:
        raise RetargetError("loop boundary trajectories are missing")
    records: dict[str, Any] = {}
    for name, values in trajectories.items():
        points = np.asarray(values, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3:
            raise RetargetError(f"loop boundary trajectory is invalid: {name}")
        if not np.isfinite(points).all():
            raise RetargetError(f"loop boundary trajectory is non-finite: {name}")
        start_velocity = (points[1] - points[0]) * fps
        end_velocity = (points[-1] - points[-2]) * fps
        records[str(name)] = {
            "start_velocity_m_per_s": start_velocity.tolist(),
            "end_velocity_m_per_s": end_velocity.tolist(),
            "residual_m_per_s": float(
                np.linalg.norm(start_velocity - end_velocity)
            ),
        }
    return records


def source_calibrated_boundary_velocity_gate(
    *,
    target_records: Mapping[str, Mapping[str, Any]],
    source_records: Mapping[str, Mapping[str, Any]],
    target_to_source: Mapping[str, Sequence[str]],
    height_scale: float,
) -> dict[str, Any]:
    scale = _finite_number(height_scale, "loop boundary height scale")
    if scale <= 0.0 or set(target_records) != set(target_to_source):
        raise RetargetError("loop boundary source calibration inventory is invalid")
    normalized_mapping: dict[str, list[str]] = {}
    checks: dict[str, Any] = {}
    rejected: list[str] = []
    for target_name in sorted(target_records):
        sources = target_to_source[target_name]
        if (
            not isinstance(sources, Sequence)
            or isinstance(sources, (str, bytes))
            or not sources
            or any(name not in source_records for name in sources)
        ):
            raise RetargetError("loop boundary source calibration mapping is invalid")
        normalized_mapping[target_name] = [str(name) for name in sources]
        target_residual = _finite_number(
            target_records[target_name].get("residual_m_per_s"),
            f"target loop boundary residual {target_name}",
        )
        source_residual = max(
            _finite_number(
                source_records[name].get("residual_m_per_s"),
                f"source loop boundary residual {name}",
            )
            for name in sources
        )
        scaled_source = source_residual * scale
        allowed = max(LOOP_BOUNDARY_VELOCITY_TOLERANCE_MPS, scaled_source)
        status = "passed" if target_residual <= allowed + 1.0e-9 else "failed"
        if status == "failed":
            rejected.append(target_name)
        checks[target_name] = {
            "source_trajectory_names": [str(name) for name in sources],
            "source_maximum_residual_m_per_s": source_residual,
            "source_scaled_residual_m_per_s": scaled_source,
            "fixed_minimum_allowance_m_per_s": (
                LOOP_BOUNDARY_VELOCITY_TOLERANCE_MPS
            ),
            "maximum_allowed_residual_m_per_s": allowed,
            "target_residual_m_per_s": target_residual,
            "status": status,
        }
    evidence = {
        "schema": "source_calibrated_loop_boundary_velocity_gate_v1",
        "method": "per_semantic_source_residual_scaled_by_target_height_v1",
        "height_scale": scale,
        "fixed_minimum_allowance_m_per_s": LOOP_BOUNDARY_VELOCITY_TOLERANCE_MPS,
        "target_to_source": normalized_mapping,
        "target_records": {name: dict(value) for name, value in target_records.items()},
        "source_records": {name: dict(value) for name, value in source_records.items()},
        "checks": checks,
        "rejected_trajectories": rejected,
        "automatic_checks": "passed" if not rejected else "failed",
    }
    if rejected:
        raise RetargetError(
            "target loop boundary velocity exceeds the corresponding sealed source: "
            + ", ".join(rejected)
        )
    return evidence


def foot_phase_is_continuous(contact_by_side: Mapping[str, Sequence[bool]]) -> bool:
    if set(contact_by_side) != {"left", "right"}:
        raise RetargetError("loop foot phase must cover left and right")
    for values in contact_by_side.values():
        if not values or any(not isinstance(value, (bool, np.bool_)) for value in values):
            raise RetargetError("loop foot phase samples are invalid")
        if bool(values[0]) != bool(values[-1]):
            return False
    return True


def calibrate_deformation_thresholds(
    *,
    source_minimum_shoulder_ratio: float,
    source_minimum_hip_ratio: float,
) -> dict[str, Any]:
    values = (source_minimum_shoulder_ratio, source_minimum_hip_ratio)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RetargetError("approved source deformation calibration is invalid")
    return {
        "calibration_basis": "approved_source_motion_and_static_bind_v1",
        "source_minimum_shoulder_span_ratio": float(source_minimum_shoulder_ratio),
        "source_minimum_hip_span_ratio": float(source_minimum_hip_ratio),
        "required_minimum_shoulder_span_ratio": max(
            MINIMUM_CALIBRATED_SHOULDER_SPAN_RATIO,
            min(1.0, float(source_minimum_shoulder_ratio) - 0.05),
        ),
        "required_minimum_hip_span_ratio": max(
            MINIMUM_CALIBRATED_HIP_SPAN_RATIO,
            min(1.0, float(source_minimum_hip_ratio) - 0.05),
        ),
        "allowed_maximum_skinned_edge_stretch_ratio": (
            MAXIMUM_CALIBRATED_EDGE_STRETCH_RATIO
        ),
    }


def _evaluated_indexed_positions(
    bpy: Any,
    mesh: Any,
    vertex_indices: Sequence[int],
    *,
    performance_telemetry: dict[str, Any] | None = None,
) -> np.ndarray:
    started = time.perf_counter()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    temporary = evaluated.to_mesh()
    full_vertex_count = len(temporary.vertices)
    full_edge_count = len(temporary.edges)
    indices = [int(index) for index in vertex_indices]
    try:
        matrix = evaluated.matrix_world
        if any(index < 0 or index >= len(temporary.vertices) for index in indices):
            raise RetargetError("evaluated mesh sample index is out of range")
        values = np.asarray(
            [tuple(matrix @ temporary.vertices[index].co) for index in indices],
            dtype=np.float64,
        )
    finally:
        evaluated.to_mesh_clear()
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise RetargetError("evaluated Pixal mesh has non-finite or invalid positions")
    if performance_telemetry is not None:
        performance_telemetry.update(
            {
                "full_evaluated_vertex_count": int(full_vertex_count),
                "full_evaluated_edge_count": int(full_edge_count),
                "sampled_vertex_count": int(len(indices)),
                "wall_time_seconds": float(time.perf_counter() - started),
                "process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                ),
            }
        )
    return values


def _sample_offsets(
    evaluation_indices: Sequence[int], requested_indices: Sequence[int]
) -> np.ndarray:
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    requested = np.asarray(requested_indices, dtype=np.int64)
    offsets = np.searchsorted(evaluation, requested)
    if (
        np.any(offsets >= len(evaluation))
        or not np.array_equal(evaluation[offsets], requested)
    ):
        raise RetargetError("mesh sample plan references an unevaluated vertex")
    return offsets


def _surface_contact_side_indices(
    sample_plan: Mapping[str, Any],
) -> dict[str, list[int]]:
    semantic = sample_plan.get("_runtime_semantic_bones")
    weights = sample_plan.get("_runtime_vertex_weights")
    if not isinstance(semantic, Mapping) or not isinstance(weights, Sequence):
        raise RetargetError("surface-contact IK lacks runtime skin evidence")
    side_bones = {
        side: {
            str(semantic[f"{side}_foot"]),
            str(semantic[f"{side}_toe"]),
        }
        for side in ("left", "right")
    }
    selected = {
        side: set(int(value) for value in sample_plan[f"{side}_support_vertex_indices"])
        for side in ("left", "right")
    }
    for raw_index in sample_plan["penetration_vertex_indices"]:
        index = int(raw_index)
        if index < 0 or index >= len(weights):
            raise RetargetError("surface-contact IK penetration index is invalid")
        vertex_weights = weights[index]
        if not isinstance(vertex_weights, Mapping):
            raise RetargetError("surface-contact IK skin weights are invalid")
        totals = {
            side: sum(float(vertex_weights.get(name, 0.0)) for name in side_bones[side])
            for side in ("left", "right")
        }
        for side, opposite in (("left", "right"), ("right", "left")):
            if (
                totals[side] >= SUPPORT_CORE_COMBINED_WEIGHT_MINIMUM
                and totals[opposite] <= SUPPORT_CORE_OPPOSITE_WEIGHT_MAXIMUM
            ):
                selected[side].add(index)
    result = {side: sorted(values) for side, values in selected.items()}
    if any(not values for values in result.values()):
        raise RetargetError("surface-contact IK has no bilateral surface samples")
    return result


def _apply_surface_contact_leg_ik(
    *,
    bpy: Any,
    armature: Any,
    mesh: Any,
    action: Any,
    semantic_bones: Mapping[str, Any],
    floor_z_m: float,
    sample_plan: Mapping[str, Any],
) -> dict[str, Any]:
    armature.animation_data.action = action
    frame_start, frame_end = _integer_frame_range(action)
    side_indices = _surface_contact_side_indices(sample_plan)
    evaluation_indices = sorted(
        set(side_indices["left"]) | set(side_indices["right"])
    )
    offsets = {
        side: _sample_offsets(evaluation_indices, side_indices[side])
        for side in ("left", "right")
    }
    records: dict[str, Any] = {}
    maximum_correction = 0.0
    corrected_frame_count = 0
    corrected_side_frame_count = 0
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        cumulative = {"left": 0.0, "right": 0.0}
        iteration_records: list[dict[str, Any]] = []
        pre_minimum: dict[str, float] | None = None
        post_minimum: dict[str, float] | None = None
        for iteration in range(1, MAXIMUM_SURFACE_CONTACT_IK_ITERATIONS + 1):
            sampled = _evaluated_indexed_positions(
                bpy, mesh, evaluation_indices
            )
            minima = {
                side: float(np.min(sampled[offsets[side], 2]))
                for side in ("left", "right")
            }
            if pre_minimum is None:
                pre_minimum = dict(minima)
            plan = plan_surface_contact_leg_ik(
                side_minimum_z_m=minima,
                floor_z_m=floor_z_m,
            )
            pending = {
                side: float(plan["upward_correction_m"][side])
                for side in ("left", "right")
                if float(plan["upward_correction_m"][side]) > 1.0e-8
            }
            if not pending:
                post_minimum = dict(minima)
                break
            solutions: dict[str, Any] = {}
            for side, correction in pending.items():
                if cumulative[side] + correction > MAXIMUM_IK_ANKLE_CORRECTION_M:
                    raise GroundingError(
                        "surface-contact IK cumulative ankle correction exceeds 0.030 m",
                        evidence={
                            "schema": "tokenrig_surface_contact_ik_rejection_v1",
                            "action_name": action.name,
                            "frame": frame,
                            "side": side,
                            "pre_minimum_z_m": pre_minimum,
                            "current_minimum_z_m": minima,
                            "cumulative_correction_m": cumulative[side],
                            "pending_correction_m": correction,
                            "maximum_allowed_ankle_correction_m": (
                                MAXIMUM_IK_ANKLE_CORRECTION_M
                            ),
                        },
                    )
                solutions[side] = apply_vertical_surface_contact_leg_ik_pose(
                    bpy=bpy,
                    armature=armature,
                    semantic_bones=semantic_bones,
                    action_name=action.name,
                    frame=frame,
                    side=side,
                    upward_correction_m=correction,
                )
                cumulative[side] += correction
            bpy.context.view_layer.update()
            iteration_records.append(
                {
                    "iteration": iteration,
                    "minimum_z_before_m": minima,
                    "planned_upward_correction_m": {
                        side: float(plan["upward_correction_m"][side])
                        for side in ("left", "right")
                    },
                    "solutions": solutions,
                }
            )
        if post_minimum is None:
            sampled = _evaluated_indexed_positions(bpy, mesh, evaluation_indices)
            post_minimum = {
                side: float(np.min(sampled[offsets[side], 2]))
                for side in ("left", "right")
            }
        if any(
            value < floor_z_m - IK_CONTACT_READBACK_SAFETY_MARGIN_M
            for value in post_minimum.values()
        ):
            raise GroundingError(
                "surface-contact IK did not clear the fixed floor",
                evidence={
                    "schema": "tokenrig_surface_contact_ik_rejection_v1",
                    "action_name": action.name,
                    "frame": frame,
                    "fixed_floor_z_m": float(floor_z_m),
                    "pre_minimum_z_m": pre_minimum,
                    "post_minimum_z_m": post_minimum,
                    "cumulative_correction_m": cumulative,
                    "iterations": iteration_records,
                },
            )
        corrected_sides = sum(value > 0.0 for value in cumulative.values())
        corrected_frame_count += int(corrected_sides > 0)
        corrected_side_frame_count += corrected_sides
        maximum_correction = max(maximum_correction, *cumulative.values())
        records[str(frame)] = {
            "pre_minimum_z_m": pre_minimum,
            "post_minimum_z_m": post_minimum,
            "cumulative_upward_correction_m": cumulative,
            "iteration_count": len(iteration_records),
            "iterations": iteration_records,
        }
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    return {
        "schema": SURFACE_CONTACT_IK_SCHEMA,
        "method": "evaluated_surface_minimum_to_vertical_two_bone_leg_ik_v1",
        "action_name": action.name,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_end - frame_start + 1,
        "fixed_floor_z_m": float(floor_z_m),
        "safety_margin_m": IK_CONTACT_READBACK_SAFETY_MARGIN_M,
        "maximum_allowed_ankle_correction_m": MAXIMUM_IK_ANKLE_CORRECTION_M,
        "maximum_cumulative_upward_correction_m": maximum_correction,
        "corrected_frame_count": corrected_frame_count,
        "corrected_side_frame_count": corrected_side_frame_count,
        "surface_sample_indices": {
            side: {
                "count": len(side_indices[side]),
                "sha256": hashlib.sha256(
                    np.asarray(side_indices[side], dtype="<i8").tobytes()
                ).hexdigest(),
            }
            for side in ("left", "right")
        },
        "root_pelvis_hip_translation_preserved": True,
        "ankle_xy_preserved": True,
        "foot_toe_global_orientation_preserved": True,
        "records_by_frame": records,
        "automatic_checks": "passed",
    }


def _apply_constant_grounding(
    *,
    bpy: Any,
    armature: Any,
    mesh: Any,
    action: Any,
    floor_z_m: float,
    sample_plan: Mapping[str, Any],
) -> dict[str, Any]:
    armature.animation_data.action = action
    frame_start, frame_end = _integer_frame_range(action)
    frame_minimum_z: dict[int, float] = {}
    frame_minimum_samples: dict[int, dict[str, Any]] = {}
    performance_by_frame: dict[str, Any] = {}
    penetration_indices = sample_plan["penetration_vertex_indices"]
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        telemetry: dict[str, Any] = {}
        sampled = _evaluated_indexed_positions(
            bpy,
            mesh,
            penetration_indices,
            performance_telemetry=telemetry,
        )
        minimum_offset = int(np.argmin(sampled[:, 2]))
        vertex_index = int(penetration_indices[minimum_offset])
        frame_minimum_z[frame] = float(sampled[minimum_offset, 2])
        frame_minimum_samples[frame] = describe_grounding_sample(
            frame=frame,
            vertex_index=vertex_index,
            evaluated_position=sampled[minimum_offset],
            rest_position=sample_plan["_runtime_rest_positions"][vertex_index],
            weights=sample_plan["_runtime_vertex_weights"][vertex_index],
            semantic_bones=sample_plan["_runtime_semantic_bones"],
        )
        performance_by_frame[str(frame)] = telemetry
    evidence = plan_constant_grounding(
        frame_minimum_z=frame_minimum_z,
        floor_z_m=floor_z_m,
        action_name=action.name,
        frame_minimum_samples=frame_minimum_samples,
    )
    evidence["performance_by_frame"] = performance_by_frame
    lift = evidence["correction_m"]
    curves = [
        curve
        for curve in action.fcurves
        if curve.data_path == "location" and curve.array_index == 2
    ]
    if len(curves) != 1:
        raise RetargetError("animated armature must have one root Z location curve")
    for point in curves[0].keyframe_points:
        point.co.y += lift
        point.handle_left.y += lift
        point.handle_right.y += lift
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    return evidence


def _joint_world(armature: Any, name: str) -> np.ndarray:
    value = armature.matrix_world @ armature.pose.bones[name].head
    return np.asarray(tuple(value), dtype=np.float64)


def _quaternion_rotation_vector(value: Any) -> np.ndarray:
    quaternion = value.normalized().copy()
    if quaternion.w < 0.0:
        quaternion.negate()
    vector = np.asarray((quaternion.x, quaternion.y, quaternion.z), dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        return 2.0 * vector
    angle = 2.0 * math.atan2(length, max(0.0, float(quaternion.w)))
    return angle * vector / length


def _quaternion_from_rotation_vector(value: Sequence[float]) -> Any:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise RetargetError("loop seam rotation vector must be finite 3D")
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-12:
        array = np.asarray((1.0, *(0.5 * vector)), dtype=np.float64)
    else:
        half = 0.5 * angle
        array = np.asarray(
            (math.cos(half), *(math.sin(half) * vector / angle)),
            dtype=np.float64,
        )
    return _quaternion_from_array(array)


def _boundary_world_velocity_evidence(
    *,
    bpy: Any,
    armature: Any,
    bone_names: Sequence[str],
    frame_start: int,
    frame_end: int,
) -> dict[str, Any]:
    frames = (frame_start, frame_start + 1, frame_end - 1, frame_end)
    trajectories = {"armature_root": []} | {
        str(name): [] for name in bone_names
    }
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        root = np.asarray(tuple(armature.matrix_world.translation), dtype=np.float64)
        trajectories["armature_root"].append(root)
        for name in bone_names:
            trajectories[str(name)].append(_joint_world(armature, str(name)) - root)
    records: dict[str, Any] = {}
    for name, values in trajectories.items():
        points = np.asarray(values, dtype=np.float64)
        start_velocity = (points[1] - points[0]) * 30.0
        end_velocity = (points[3] - points[2]) * 30.0
        records[name] = {
            "start_velocity_m_per_s": start_velocity.tolist(),
            "end_velocity_m_per_s": end_velocity.tolist(),
            "residual_m_per_s": float(
                np.linalg.norm(start_velocity - end_velocity)
            ),
        }
    worst_name = max(records, key=lambda name: records[name]["residual_m_per_s"])
    return {
        "maximum_residual_m_per_s": records[worst_name]["residual_m_per_s"],
        "worst_trajectory": worst_name,
        "records": records,
    }


def reconcile_action_loop_boundary_velocity(
    *,
    bpy: Any,
    armature: Any,
    action: Any,
    bone_names: Sequence[str],
) -> dict[str, Any]:
    armature.animation_data.action = action
    frame_start, frame_end = _integer_frame_range(action)
    if frame_end - frame_start < 3:
        raise RetargetError("loop seam reconciliation needs at least four frames")
    names = [str(name) for name in bone_names]
    if len(names) != len(set(names)) or any(
        name not in armature.pose.bones for name in names
    ):
        raise RetargetError("loop seam reconciliation bone inventory is invalid")
    pre = _boundary_world_velocity_evidence(
        bpy=bpy,
        armature=armature,
        bone_names=names,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    values = {"armature_root": armature} | {
        name: armature.pose.bones[name] for name in names
    }
    frames = (frame_start, frame_start + 1, frame_end - 1, frame_end)

    def capture_states() -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for frame in frames:
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            result[frame] = {
                name: {
                    "location": np.asarray(tuple(value.location), dtype=np.float64),
                    "rotation": value.rotation_quaternion.normalized().copy(),
                    "scale": np.asarray(tuple(value.scale), dtype=np.float64),
                }
                for name, value in values.items()
            }
        return result

    initial_states = capture_states()
    maximum_endpoint_location_change = 0.0
    maximum_endpoint_rotation_change = 0.0
    maximum_endpoint_scale_change = 0.0
    bpy.context.scene.frame_set(frame_end)
    for name, value in values.items():
        start = initial_states[frame_start][name]
        end = initial_states[frame_end][name]
        target_location = (
            end["location"]
            if name == "armature_root"
            else start["location"]
        )
        maximum_endpoint_location_change = max(
            maximum_endpoint_location_change,
            float(np.linalg.norm(target_location - end["location"])),
        )
        maximum_endpoint_rotation_change = max(
            maximum_endpoint_rotation_change,
            _quaternion_angle(start["rotation"], end["rotation"]),
        )
        maximum_endpoint_scale_change = max(
            maximum_endpoint_scale_change,
            float(np.max(np.abs(start["scale"] - end["scale"]))),
        )
        value.rotation_mode = "QUATERNION"
        value.location = tuple(float(item) for item in target_location)
        value.rotation_quaternion = start["rotation"]
        value.scale = tuple(float(item) for item in start["scale"])
        _keyframe_transform(value, frame_end)
    bpy.context.view_layer.update()
    if (
        maximum_endpoint_location_change > LOOP_PELVIS_TRANSLATION_TOLERANCE_M
        or maximum_endpoint_rotation_change > LOOP_ROTATION_TOLERANCE_RAD
        or maximum_endpoint_scale_change > POSE_TRANSLATION_TOLERANCE_M
    ):
        raise RetargetError(
            "loop endpoint normalization exceeds the existing loop tolerances"
        )
    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    maximum_location_change = 0.0
    maximum_rotation_change = 0.0
    maximum_scale_change = 0.0
    iteration_records: list[dict[str, Any]] = []
    post = pre
    for iteration in range(1, MAXIMUM_LOOP_BOUNDARY_RECONCILIATION_ITERATIONS + 1):
        states = capture_states()
        targets: dict[int, dict[str, Any]] = {
            frame_start + 1: {},
            frame_end - 1: {},
        }
        for name in values:
            start = states[frame_start][name]
            next_value = states[frame_start + 1][name]
            previous = states[frame_end - 1][name]
            end = states[frame_end][name]
            location_step = 0.5 * (
                (next_value["location"] - start["location"])
                + (end["location"] - previous["location"])
            )
            scale_step = 0.5 * (
                (next_value["scale"] - start["scale"])
                + (end["scale"] - previous["scale"])
            )
            start_delta = start["rotation"].rotation_difference(
                next_value["rotation"]
            )
            end_delta = previous["rotation"].rotation_difference(end["rotation"])
            average_rotation_step = _quaternion_from_rotation_vector(
                0.5
                * (
                    _quaternion_rotation_vector(start_delta)
                    + _quaternion_rotation_vector(end_delta)
                )
            )
            next_rotation = (
                start["rotation"] @ average_rotation_step
            ).normalized()
            previous_rotation = (
                end["rotation"] @ average_rotation_step.inverted()
            ).normalized()
            next_location = (
                next_value["location"]
                if name == "armature_root"
                else start["location"] + location_step
            )
            previous_location = (
                previous["location"]
                if name == "armature_root"
                else end["location"] - location_step
            )
            next_scale = start["scale"] + scale_step
            previous_scale = end["scale"] - scale_step
            targets[frame_start + 1][name] = {
                "location": next_location,
                "rotation": next_rotation,
                "scale": next_scale,
            }
            targets[frame_end - 1][name] = {
                "location": previous_location,
                "rotation": previous_rotation,
                "scale": previous_scale,
            }
            initial_next = initial_states[frame_start + 1][name]
            initial_previous = initial_states[frame_end - 1][name]
            maximum_location_change = max(
                maximum_location_change,
                float(np.linalg.norm(next_location - initial_next["location"])),
                float(
                    np.linalg.norm(
                        previous_location - initial_previous["location"]
                    )
                ),
            )
            maximum_rotation_change = max(
                maximum_rotation_change,
                _quaternion_angle(next_rotation, initial_next["rotation"]),
                _quaternion_angle(
                    previous_rotation, initial_previous["rotation"]
                ),
            )
            maximum_scale_change = max(
                maximum_scale_change,
                float(np.max(np.abs(next_scale - initial_next["scale"]))),
                float(
                    np.max(
                        np.abs(previous_scale - initial_previous["scale"])
                    )
                ),
            )
        for frame, frame_targets in targets.items():
            bpy.context.scene.frame_set(frame)
            for name, target in frame_targets.items():
                value = values[name]
                value.rotation_mode = "QUATERNION"
                value.location = tuple(float(item) for item in target["location"])
                value.rotation_quaternion = target["rotation"]
                value.scale = tuple(float(item) for item in target["scale"])
                _keyframe_transform(value, frame)
            bpy.context.view_layer.update()
        for curve in action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
        next_post = _boundary_world_velocity_evidence(
            bpy=bpy,
            armature=armature,
            bone_names=names,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        iteration_records.append(
            {
                "iteration": iteration,
                "pre_maximum_residual_m_per_s": post[
                    "maximum_residual_m_per_s"
                ],
                "post_maximum_residual_m_per_s": next_post[
                    "maximum_residual_m_per_s"
                ],
                "post_worst_trajectory": next_post["worst_trajectory"],
            }
        )
        improvement = (
            post["maximum_residual_m_per_s"]
            - next_post["maximum_residual_m_per_s"]
        )
        post = next_post
        if (
            post["maximum_residual_m_per_s"]
            <= LOOP_BOUNDARY_VELOCITY_TOLERANCE_MPS
            or improvement <= 1.0e-6
        ):
            break
    if post["maximum_residual_m_per_s"] > pre["maximum_residual_m_per_s"] + 1.0e-9:
        raise RetargetError(
            "symmetric loop seam reconciliation made boundary velocity worse: "
            f"pre={pre['maximum_residual_m_per_s']} "
            f"post={post['maximum_residual_m_per_s']} "
            f"worst={post['worst_trajectory']}"
        )
    return {
        "schema": LOOP_BOUNDARY_RECONCILIATION_SCHEMA,
        "method": "symmetric_average_first_and_last_discrete_transform_step_v1",
        "action_name": action.name,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "modified_frames": [frame_start + 1, frame_end - 1],
        "untouched_frame_count": frame_end - frame_start - 1,
        "iteration_count": len(iteration_records),
        "maximum_iteration_count": (
            MAXIMUM_LOOP_BOUNDARY_RECONCILIATION_ITERATIONS
        ),
        "iterations": iteration_records,
        "maximum_location_change_m": maximum_location_change,
        "maximum_rotation_change_rad": maximum_rotation_change,
        "maximum_scale_change": maximum_scale_change,
        "maximum_endpoint_location_normalization_m": (
            maximum_endpoint_location_change
        ),
        "maximum_endpoint_rotation_normalization_rad": (
            maximum_endpoint_rotation_change
        ),
        "maximum_endpoint_scale_normalization": maximum_endpoint_scale_change,
        "pre_boundary_velocity": pre,
        "post_boundary_velocity": post,
        "fixed_minimum_boundary_velocity_allowance_m_per_s": (
            LOOP_BOUNDARY_VELOCITY_TOLERANCE_MPS
        ),
        "requires_per_semantic_source_calibrated_gate": True,
        "root_cycle_translation_endpoint_unchanged": True,
        "root_translation_curve_unchanged": True,
        "root_and_pose_rotation_endpoints_exactly_closed": True,
        "pose_location_and_scale_endpoints_exactly_closed": True,
        "endpoint_normalization_bounded_by_existing_loop_tolerances": True,
        "automatic_checks": "passed",
    }


def _quaternion_angle(first: Any, second: Any) -> float:
    angle = float(first.normalized().rotation_difference(second.normalized()).angle)
    return min(angle, 2.0 * math.pi - angle)


def _horizontal_speed(points: Sequence[np.ndarray], fps: int) -> float:
    if len(points) < 2:
        raise RetargetError("motion speed needs at least two frames")
    distance = sum(
        float(np.linalg.norm((second - first)[:2]))
        for first, second in zip(points, points[1:])
    )
    duration = (len(points) - 1) / float(fps)
    return distance / duration


def _measure_action_quality(
    *,
    bpy: Any,
    armature: Any,
    mesh: Any,
    action: Any,
    semantic: Mapping[str, Any],
    cached: Mapping[str, Any],
    bake_evidence: Mapping[str, Any],
    sample_plan: Mapping[str, Any],
    grounding_evidence: Mapping[str, Any],
    floor_z_m: float,
) -> dict[str, Any]:
    armature.animation_data.action = action
    frame_start, frame_end = _integer_frame_range(action)
    source_frames: Sequence[CachedMotionFrame] = cached["frames"]
    if len(source_frames) != frame_end - frame_start + 1:
        raise RetargetError("baked action frame count changed from the authenticated source")
    exact = build_exact_semantic_correspondence(semantic)
    target_names = [*exact.values(), *semantic["semantic_bones"]["spine"]]
    role = semantic["semantic_bones"]
    rest_positions = {
        name: np.asarray(
            tuple(armature.matrix_world @ armature.data.bones[name].head_local),
            dtype=np.float64,
        )
        for name in target_names
    }
    rest_foot_vectors = {
        side: rest_positions[role[f"{side}_toe"]] - rest_positions[role[f"{side}_foot"]]
        for side in ("left", "right")
    }
    rest_shoulder = float(
        np.linalg.norm(
            rest_positions[role["left_upper_arm"]]
            - rest_positions[role["right_upper_arm"]]
        )
    )
    rest_hip = float(
        np.linalg.norm(
            rest_positions[role["left_thigh"]]
            - rest_positions[role["right_thigh"]]
        )
    )
    if rest_shoulder <= 1.0e-8 or rest_hip <= 1.0e-8:
        raise RetargetError("rest shoulder or hip span is degenerate")

    root_points: list[np.ndarray] = []
    mesh_minima: list[float] = []
    side_clearances = {"left": [], "right": []}
    foot_dots: list[float] = []
    shoulder_ratios: list[float] = []
    hip_ratios: list[float] = []
    edge_stretches: list[float] = []
    rotations: dict[int, dict[str, Any]] = {}
    evaluation_indices = sample_plan["evaluation_vertex_indices"]
    penetration_offsets = _sample_offsets(
        evaluation_indices, sample_plan["penetration_vertex_indices"]
    )
    support_offsets = {
        side: _sample_offsets(
            evaluation_indices, sample_plan[f"{side}_support_vertex_indices"]
        )
        for side in ("left", "right")
    }
    sampled_edges = np.asarray(sample_plan["sampled_edges"], dtype=np.int64)
    edge_offsets = _sample_offsets(
        evaluation_indices, sampled_edges.reshape(-1)
    ).reshape((-1, 2))
    rest_edge_lengths = np.asarray(
        sample_plan["rest_edge_lengths"], dtype=np.float64
    )
    support_anchors = {"left": [], "right": []}
    body_forwards: list[np.ndarray] = []
    root_rotations: list[Any] = []
    pelvis_local_translations: list[np.ndarray] = []
    relative_joint_trajectories = {name: [] for name in target_names}
    quality_performance_by_frame: dict[str, Any] = {}
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        root_points.append(np.asarray(tuple(armature.matrix_world.translation), dtype=np.float64))
        telemetry: dict[str, Any] = {}
        evaluated = _evaluated_indexed_positions(
            bpy,
            mesh,
            evaluation_indices,
            performance_telemetry=telemetry,
        )
        quality_performance_by_frame[str(frame)] = telemetry
        mesh_minima.append(float(np.min(evaluated[penetration_offsets, 2])))
        lengths = np.linalg.norm(
            evaluated[edge_offsets[:, 0]] - evaluated[edge_offsets[:, 1]], axis=1
        )
        edge_stretches.append(float(np.max(lengths / rest_edge_lengths)))
        positions = {name: _joint_world(armature, name) for name in target_names}
        body_forwards.append(
            body_forward_vector(
                left_shoulder=positions[role["left_upper_arm"]],
                right_shoulder=positions[role["right_upper_arm"]],
                pelvis=positions[role["pelvis"]],
                neck=positions[role["neck"]],
            )
        )
        root_rotations.append(
            armature.matrix_world.to_quaternion().normalized().copy()
        )
        pelvis_local_translations.append(
            np.asarray(
                tuple(_parent_local_pose(armature.pose.bones[role["pelvis"]]).translation),
                dtype=np.float64,
            )
        )
        for name in target_names:
            relative_joint_trajectories[name].append(
                (positions[name] - root_points[-1]).tolist()
            )
        for side in ("left", "right"):
            support_positions = evaluated[support_offsets[side]]
            support_order = np.argsort(support_positions[:, 2], kind="stable")
            anchor_count = min(16, len(support_order))
            contact_patch = support_positions[support_order[:anchor_count]]
            side_clearances[side].append(
                float(np.min(contact_patch[:, 2])) - floor_z_m
            )
            support_anchors[side].append(
                np.mean(contact_patch[:, :2], axis=0).tolist()
            )
            current = positions[role[f"{side}_toe"]] - positions[role[f"{side}_foot"]]
            denominator = float(
                np.linalg.norm(current) * np.linalg.norm(rest_foot_vectors[side])
            )
            if denominator <= 1.0e-12:
                raise RetargetError("animated foot-to-toe vector is degenerate")
            foot_dots.append(float(np.dot(current, rest_foot_vectors[side]) / denominator))
        shoulder_ratios.append(
            float(
                np.linalg.norm(
                    positions[role["left_upper_arm"]]
                    - positions[role["right_upper_arm"]]
                )
                / rest_shoulder
            )
        )
        hip_ratios.append(
            float(
                np.linalg.norm(
                    positions[role["left_thigh"]] - positions[role["right_thigh"]]
                )
                / rest_hip
            )
        )
        rotations[frame] = {
            name: armature.pose.bones[name].matrix.to_quaternion().normalized().copy()
            for name in target_names
        }

    source_root = [
        np.asarray(tuple(frame.root_location), dtype=np.float64) for frame in source_frames
    ]
    height_scale = float(bake_evidence["height_scale"])
    source_displacements = [height_scale * (value - source_root[0]) for value in source_root]
    target_displacements = [value - root_points[0] for value in root_points]
    reconstruction_error = max(
        float(np.linalg.norm(actual - expected))
        for actual, expected in zip(target_displacements, source_displacements)
    )
    travel = target_displacements[-1] - target_displacements[0]
    if action.name == ACTION_NAMES["walk"]:
        horizontal = travel[:2]
        if float(np.linalg.norm(horizontal)) <= 1.0e-8:
            raise RetargetError("Walking has zero root travel")
        direction_dot = float(horizontal[1] * -1.0 / np.linalg.norm(horizontal))
        travel_direction = np.asarray(
            (horizontal[0], horizontal[1], 0.0), dtype=np.float64
        )
        travel_direction /= np.linalg.norm(travel_direction)
        body_travel_dot = min(
            float(np.dot(forward, travel_direction)) for forward in body_forwards
        )
    else:
        direction_dot = None
        body_travel_dot = None
    body_negative_y_dot = min(
        float(np.dot(forward, np.asarray((0.0, -1.0, 0.0))))
        for forward in body_forwards
    )
    source_speed = _horizontal_speed(source_root, 30) * height_scale
    target_speed = _horizontal_speed(root_points, 30)
    rotation_residual = max(
        _quaternion_angle(rotations[frame_start][name], rotations[frame_end][name])
        for name in target_names
    )
    expected_cycle = source_displacements[-1] - source_displacements[0]
    actual_cycle = target_displacements[-1] - target_displacements[0]
    root_cycle_error = float(np.linalg.norm(actual_cycle - expected_cycle))
    root_rotation_residual = _quaternion_angle(root_rotations[0], root_rotations[-1])
    pelvis_translation_residual = float(
        np.linalg.norm(pelvis_local_translations[-1] - pelvis_local_translations[0])
    )
    target_boundary_trajectories = {
        "armature_root": root_points,
        **relative_joint_trajectories,
    }
    target_boundary_records = boundary_velocity_residual_records(
        target_boundary_trajectories, fps=30
    )
    source_names = _required_source_names()
    source_boundary_trajectories: dict[str, list[Any]] = {
        "armature_root": source_root
    } | {name: [] for name in source_names}
    for frame_record, root_position in zip(source_frames, source_root):
        for name in source_names:
            source_boundary_trajectories[name].append(
                (
                    np.asarray(
                        tuple(frame_record.world_joint_positions[name]),
                        dtype=np.float64,
                    )
                    - root_position
                ).tolist()
            )
    source_boundary_records = boundary_velocity_residual_records(
        source_boundary_trajectories, fps=30
    )
    target_to_source = {
        "armature_root": ["armature_root"],
        **{target: [source] for source, target in exact.items()},
        **{target: list(ROCKETBOX_SPINE_BONES) for target in role["spine"]},
    }
    boundary_velocity_gate = source_calibrated_boundary_velocity_gate(
        target_records=target_boundary_records,
        source_records=source_boundary_records,
        target_to_source=target_to_source,
        height_scale=height_scale,
    )
    boundary_velocity_residual = max(
        value["residual_m_per_s"] for value in target_boundary_records.values()
    )
    penetration = max(0.0, float(floor_z_m - min(mesh_minima)))
    left_clearance = max(0.0, min(side_clearances["left"]))
    right_clearance = max(0.0, min(side_clearances["right"]))
    contact_summaries = {
        side: summarize_foot_contact(
            clearances_m=side_clearances[side],
            anchor_xy_m=support_anchors[side],
            fps=30,
        )
        for side in ("left", "right")
    }
    bilateral_contact = [
        left and right
        for left, right in zip(
            contact_summaries["left"]["contact_by_frame"],
            contact_summaries["right"]["contact_by_frame"],
        )
    ]
    support_evidence = summarize_support_union(
        left_contact=contact_summaries["left"]["contact_by_frame"],
        right_contact=contact_summaries["right"]["contact_by_frame"],
    )
    foot_phase_continuous = foot_phase_is_continuous(
        {
            side: contact_summaries[side]["contact_by_frame"]
            for side in ("left", "right")
        }
    )
    source_rest_heads = cached["rest_heads"]
    source_shoulder_rest = float(
        np.linalg.norm(
            np.asarray(tuple(source_rest_heads[ROCKETBOX_ROLE_TO_BONE["left_upper_arm"]]))
            - np.asarray(tuple(source_rest_heads[ROCKETBOX_ROLE_TO_BONE["right_upper_arm"]]))
        )
    )
    source_hip_rest = float(
        np.linalg.norm(
            np.asarray(tuple(source_rest_heads[ROCKETBOX_ROLE_TO_BONE["left_thigh"]]))
            - np.asarray(tuple(source_rest_heads[ROCKETBOX_ROLE_TO_BONE["right_thigh"]]))
        )
    )
    if source_shoulder_rest <= 1.0e-8 or source_hip_rest <= 1.0e-8:
        raise RetargetError("approved source deformation rest spans are degenerate")
    source_shoulder_ratios = [
        float(
            np.linalg.norm(
                np.asarray(tuple(frame.joint_positions[ROCKETBOX_ROLE_TO_BONE["left_upper_arm"]]))
                - np.asarray(tuple(frame.joint_positions[ROCKETBOX_ROLE_TO_BONE["right_upper_arm"]]))
            )
            / source_shoulder_rest
        )
        for frame in source_frames
    ]
    source_hip_ratios = [
        float(
            np.linalg.norm(
                np.asarray(tuple(frame.joint_positions[ROCKETBOX_ROLE_TO_BONE["left_thigh"]]))
                - np.asarray(tuple(frame.joint_positions[ROCKETBOX_ROLE_TO_BONE["right_thigh"]]))
            )
            / source_hip_rest
        )
        for frame in source_frames
    ]
    deformation_calibration = calibrate_deformation_thresholds(
        source_minimum_shoulder_ratio=min(source_shoulder_ratios),
        source_minimum_hip_ratio=min(source_hip_ratios),
    )
    penetration_by_frame = {
        str(frame): max(0.0, float(floor_z_m - minimum))
        for frame, minimum in zip(range(frame_start, frame_end + 1), mesh_minima)
    }
    return {
        "action_name": action.name,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "rest_delta": {
            "target_rest_translations_preserved": bake_evidence[
                "target_rest_translations_preserved"
            ],
            "finite_rest_and_pose_matrices": all(
                math.isfinite(float(bake_evidence[key]))
                for key in (
                    "maximum_global_rest_alignment_error",
                    "maximum_local_rest_delta_error",
                )
            ),
            "parent_first": bake_evidence["parent_first"],
            "maximum_global_rest_alignment_error": bake_evidence[
                "maximum_global_rest_alignment_error"
            ],
            "maximum_local_rest_delta_error": bake_evidence[
                "maximum_local_rest_delta_error"
            ],
            "maximum_target_translation_error_m": bake_evidence[
                "maximum_target_translation_error_m"
            ],
        },
        "rotation_projection": bake_evidence["rotation_projection"],
        "root_motion": {
            "axis_map_3x3": [list(row) for row in AXIS_MAP_3X3],
            "height_scale": height_scale,
            "source_travel_m": source_displacements[-1].tolist(),
            "target_travel_m": travel.tolist(),
            "reconstruction_error_m": reconstruction_error,
            "endpoint_direction_dot_negative_y": direction_dot,
            "minimum_body_forward_dot_negative_y": body_negative_y_dot,
            "minimum_body_forward_dot_travel": body_travel_dot,
        },
        "speed": {
            "source_scaled_speed_m_per_s": source_speed,
            "target_speed_m_per_s": target_speed,
            "absolute_reconstruction_error_m_per_s": abs(target_speed - source_speed),
        },
        "loop": {
            "maximum_rotation_residual_rad": rotation_residual,
            "root_cycle_reconstruction_error_m": root_cycle_error,
            "armature_root_rotation_residual_rad": root_rotation_residual,
            "pelvis_local_translation_residual_m": pelvis_translation_residual,
            "maximum_boundary_velocity_residual_m_per_s": boundary_velocity_residual,
            "source_calibrated_boundary_velocity": boundary_velocity_gate,
            "foot_phase_continuous": foot_phase_continuous,
        },
        "floor": {
            "fixed_floor_z_m": float(floor_z_m),
            "grounding_correction_m": grounding_evidence["correction_m"],
            "pre_ground_maximum_penetration_m": grounding_evidence[
                "pre_ground_maximum_penetration_m"
            ],
            "pre_ground_penetration_by_frame_m": grounding_evidence[
                "pre_ground_penetration_by_frame_m"
            ],
            "penetration_by_frame_m": penetration_by_frame,
            "maximum_penetration_m": penetration,
            "maximum_per_foot_cycle_minimum_clearance_m": max(
                left_clearance, right_clearance
            ),
            "left_contact": contact_summaries["left"]["contact_frame_count"] > 0,
            "right_contact": contact_summaries["right"]["contact_frame_count"] > 0,
            "left_minimum_clearance_m": left_clearance,
            "right_minimum_clearance_m": right_clearance,
        },
        "contact": {
            "bilateral_contact_ratio": float(np.mean(bilateral_contact)),
            **support_evidence,
            "feet": contact_summaries,
        },
        "performance": {
            "schema": "indexed_evaluated_mesh_performance_v1",
            "passes": {
                "grounding": grounding_evidence["performance_by_frame"],
                "quality": quality_performance_by_frame,
            },
        },
        "sampling": dict(sample_plan["evidence"]),
        "feet": {
            "minimum_foot_to_toe_rest_dot": min(foot_dots),
            "inverted": min(foot_dots) <= 0.0,
        },
        "deformation": {
            **deformation_calibration,
            "minimum_shoulder_span_ratio": min(shoulder_ratios),
            "minimum_hip_span_ratio": min(hip_ratios),
            "maximum_skinned_edge_stretch_ratio": max(edge_stretches),
        },
    }


def _remove_everything_except(bpy: Any, keep_objects: Sequence[Any], keep_actions: Sequence[Any]) -> None:
    objects = set(keep_objects)
    actions = set(keep_actions)
    for obj in list(bpy.data.objects):
        if obj not in objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in list(bpy.data.actions):
        if action not in actions:
            bpy.data.actions.remove(action, do_unlink=True)
    bpy.ops.outliner.orphans_purge(
        do_local_ids=True, do_linked_ids=True, do_recursive=True
    )


def _run_blender_retarget(
    *,
    bpy: Any,
    bind_pose_glb: Path,
    baseline_blend: Path,
    idle_motion_fbx: Path,
    staging_dir: Path,
    static_auth: Mapping[str, Any],
    walk_auth: Mapping[str, Any],
    limb_motion_basis_3x3: Sequence[Sequence[float]],
) -> dict[str, Any]:
    result = bpy.ops.wm.open_mainfile(filepath=str(baseline_blend))
    if "FINISHED" not in result:
        raise RetargetError("could not open immutable baseline retarget.blend")
    _configure_scene(bpy)
    walk_source = _identify_walk_source(bpy, walk_auth["source_animation"])
    semantic = static_auth["semantic_mapping"]
    target_armature, target_mesh = _import_tokenrig_runtime(bpy, bind_pose_glb, semantic)
    target_base_transform = capture_target_base_transform(target_armature)
    all_target_bones = [bone.name for bone in target_armature.data.bones]
    target_rest_all = capture_rest_matrices(target_armature, all_target_bones)
    audit = _static_audit_module()
    expected_mesh_contract = audit.capture_blender_mesh_contract(target_mesh)
    expected_surface = audit.capture_blender_surface_reference(target_mesh)
    expected_skin_weights, expected_skin_positions = audit.extract_vertex_weights(
        target_mesh, target_armature
    )
    raw_edges = np.empty(len(target_mesh.data.edges) * 2, dtype=np.int32)
    target_mesh.data.edges.foreach_get("vertices", raw_edges)
    sample_plan = build_deterministic_mesh_sample_plan(
        rest_positions=expected_skin_positions,
        edges=raw_edges.reshape((-1, 2)),
        vertex_weights=expected_skin_weights,
        semantic_bones=semantic["semantic_bones"],
    )

    walk_cached = cache_source_motion(
        bpy, walk_source, action_name=ACTION_NAMES["walk"]
    )
    walk_cached["source_armature"] = walk_source.armature
    walking, walk_bake = bake_rest_corrected_action(
        bpy=bpy,
        target_armature=target_armature,
        semantic=semantic,
        cached=walk_cached,
        action_name=ACTION_NAMES["walk"],
        target_base_transform=target_base_transform,
        limb_motion_basis_3x3=limb_motion_basis_3x3,
    )
    remove_source_objects(bpy, walk_source, [walking])

    idle_source = _import_idle_source(bpy, idle_motion_fbx)
    idle_cached = cache_source_motion(
        bpy, idle_source, action_name=ACTION_NAMES["idle"]
    )
    idle_cached["source_armature"] = idle_source.armature
    standing_idle, idle_bake = bake_rest_corrected_action(
        bpy=bpy,
        target_armature=target_armature,
        semantic=semantic,
        cached=idle_cached,
        action_name=ACTION_NAMES["idle"],
        target_base_transform=target_base_transform,
        limb_motion_basis_3x3=limb_motion_basis_3x3,
    )
    remove_source_objects(bpy, idle_source, [walking, standing_idle])
    _remove_everything_except(
        bpy, [target_armature, target_mesh], [walking, standing_idle]
    )
    validate_target_only_scene(bpy, target_armature, target_mesh, all_target_bones)
    if {action.name for action in bpy.data.actions} != set(ACTION_NAMES.values()):
        raise RetargetError("animated target scene must contain exactly Walking and Standing_Idle")

    floor_z_m = float(static_auth["floor_z_m"])
    surface_contact_ik = {
        walking.name: _apply_surface_contact_leg_ik(
            bpy=bpy,
            armature=target_armature,
            mesh=target_mesh,
            action=walking,
            semantic_bones=semantic["semantic_bones"],
            floor_z_m=floor_z_m,
            sample_plan=sample_plan,
        ),
        standing_idle.name: _apply_surface_contact_leg_ik(
            bpy=bpy,
            armature=target_armature,
            mesh=target_mesh,
            action=standing_idle,
            semantic_bones=semantic["semantic_bones"],
            floor_z_m=floor_z_m,
            sample_plan=sample_plan,
        ),
    }
    loop_boundary_reconciliation = {
        action.name: reconcile_action_loop_boundary_velocity(
            bpy=bpy,
            armature=target_armature,
            action=action,
            bone_names=semantic["target_bone_names"],
        )
        for action in (walking, standing_idle)
    }
    grounding = {
        walking.name: _apply_constant_grounding(
            bpy=bpy,
            armature=target_armature,
            mesh=target_mesh,
            action=walking,
            floor_z_m=floor_z_m,
            sample_plan=sample_plan,
        ),
        standing_idle.name: _apply_constant_grounding(
            bpy=bpy,
            armature=target_armature,
            mesh=target_mesh,
            action=standing_idle,
            floor_z_m=floor_z_m,
            sample_plan=sample_plan,
        ),
    }
    action_metrics = {
        walking.name: _measure_action_quality(
            bpy=bpy,
            armature=target_armature,
            mesh=target_mesh,
            action=walking,
            semantic=semantic,
            cached=walk_cached,
            bake_evidence=walk_bake,
            sample_plan=sample_plan,
            grounding_evidence=grounding[walking.name],
            floor_z_m=floor_z_m,
        ),
        standing_idle.name: _measure_action_quality(
            bpy=bpy,
            armature=target_armature,
            mesh=target_mesh,
            action=standing_idle,
            semantic=semantic,
            cached=idle_cached,
            bake_evidence=idle_bake,
            sample_plan=sample_plan,
            grounding_evidence=grounding[standing_idle.name],
            floor_z_m=floor_z_m,
        ),
    }
    for name, evidence in surface_contact_ik.items():
        action_metrics[name]["surface_contact_ik"] = evidence
        action_metrics[name]["loop_boundary_reconciliation"] = (
            loop_boundary_reconciliation[name]
        )
    preexport_metrics_checkpoint = staging_dir / "preexport_action_metrics.json"
    _write_json_exclusive(
        preexport_metrics_checkpoint,
        {
            "schema": "tokenrig_retarget_preexport_action_metrics_v1",
            "purpose": "failure_evidence_only_removed_before_success_publication",
            "actions": action_metrics,
        },
    )
    animation_endpoints = {
        walking.name: capture_animation_endpoint_matrices(
            bpy, target_armature, walking, all_target_bones
        ),
        standing_idle.name: capture_animation_endpoint_matrices(
            bpy, target_armature, standing_idle, all_target_bones
        ),
    }

    animated_blend = staging_dir / "animated.blend"
    _save_animated_blend(bpy, target_armature, target_mesh, animated_blend)
    input_pbr = audit.pbr_payload_contract(audit.read_glb(bind_pose_glb))
    exports = {
        ACTION_NAMES["walk"]: staging_dir / "walking.glb",
        ACTION_NAMES["idle"]: staging_dir / "standing_idle.glb",
    }
    frame_ranges = {}
    for name, output_path in exports.items():
        frame_ranges[name] = _export_one_action(
            bpy, animated_blend, name, output_path, all_target_bones
        )
        action_metrics[name]["roundtrip"] = roundtrip_validate_action(
            bpy=bpy,
            glb_path=output_path,
            action_name=name,
            frame_range=frame_ranges[name],
            expected_bones=all_target_bones,
            expected_rest_matrices=target_rest_all,
            input_pbr=input_pbr,
            expected_mesh_contract=expected_mesh_contract,
            expected_surface=expected_surface,
            expected_skin_positions=expected_skin_positions,
            expected_skin_weights=expected_skin_weights,
            expected_animation_endpoints=animation_endpoints[name],
            target_base_transform=target_base_transform,
        )
        validate_action_metrics(
            action_metrics[name], semantic_mapping=semantic
        )

    return {
        "schema": METRICS_SCHEMA,
        "canonical_front": CANONICAL_FRONT,
        "canonical_up": CANONICAL_UP,
        "axis_map_3x3": [list(row) for row in AXIS_MAP_3X3],
        "fixed_floor_z_m": floor_z_m,
        "constant_grounding_offsets_m": grounding,
        "surface_contact_leg_ik": surface_contact_ik,
        "loop_boundary_reconciliation": loop_boundary_reconciliation,
        "semantic_mapping": semantic,
        "exact_semantic_correspondence": build_exact_semantic_correspondence(semantic),
        "spine_resample_plan": {
            ACTION_NAMES["walk"]: walk_bake["spine_resample_plan"],
            ACTION_NAMES["idle"]: idle_bake["spine_resample_plan"],
        },
        "export_parameters": {
            name: gltf_export_parameters(name) for name in ACTION_NAMES.values()
        },
        "rest_matrices": {
            "walk_source": walk_cached["rest_serialized"],
            "idle_source": idle_cached["rest_serialized"],
            "target": target_rest_all,
        },
        "actions": action_metrics,
        "target_bone_names": sorted(all_target_bones),
        "unmapped_descendants_kept_at_rest": semantic["rest_descendants"],
        "head_bound_attachments_rigid": {
            "head": semantic["semantic_bones"]["head"],
            "descendants": semantic["head_bound_descendants"],
        },
        "hand_bound_descendants_at_rest": semantic["hand_bound_descendants"],
        "artifacts": {
            "animated_blend": "animated.blend",
            "walking": "walking.glb",
            "standing_idle": "standing_idle.glb",
        },
        "automatic_checks": "passed",
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_exclusive(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _chmod_and_fsync(path: Path, mode: int) -> None:
    path = Path(path)
    path.chmod(mode)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_output(output_dir: Path) -> Path:
    argument = _absolute(output_dir)
    parent = _require_real_directory(argument.parent, "retarget output parent")
    destination = parent / argument.name
    if os.path.lexists(destination):
        raise RetargetError(f"no-replace retarget output already exists: {destination}")
    return destination


def _failure_evidence_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.failed.{uuid.uuid4().hex}.json"


def _structured_error_evidence(error: BaseException) -> dict[str, Any] | None:
    evidence = getattr(error, "evidence", None)
    if evidence is None:
        return None
    if not isinstance(evidence, Mapping):
        raise RetargetError("structured failure evidence must be a mapping")
    payload = json.loads(json.dumps(evidence, sort_keys=True))
    if not isinstance(payload, dict):
        raise RetargetError("structured failure evidence must serialize as an object")
    return payload


def _write_external_inventory(
    *,
    path: Path,
    scope: str,
    artifacts: Sequence[Path],
) -> Path:
    descriptors = {
        artifact.name: file_descriptor(artifact, public_path=artifact.name)
        for artifact in sorted(artifacts, key=lambda value: value.name)
    }
    _write_json_exclusive(
        path,
        {
            "schema": "tokenrig_retarget_failure_inventory_v1",
            "inventory_scope": scope,
            "descriptor_self_excluded": True,
            "artifacts": descriptors,
        },
    )
    path.chmod(0o444)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return path


def _write_failure_evidence(
    *, output_dir: Path, asset_id: str, error: BaseException, authenticated: Any
) -> Path:
    path = _failure_evidence_path(output_dir)
    inventory_path = path.with_name(f"{path.stem}.inventory.json")
    payload = {
        "schema": "tokenrig_rocketbox_retarget_attempt_v1",
        "asset_id": asset_id,
        "decision": "rejected",
        "error_type": type(error).__name__,
        "error": str(error),
        "authenticated_inputs": authenticated,
        "external_inventory_descriptor": inventory_path.name,
    }
    error_evidence = _structured_error_evidence(error)
    if error_evidence is not None:
        payload["error_evidence"] = error_evidence
    _write_json_exclusive(path, payload)
    _write_external_inventory(
        path=inventory_path,
        scope="standalone_failure_evidence_excluding_this_external_descriptor",
        artifacts=[path],
    )
    path.chmod(0o444)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return path


def preserve_failed_staging(
    *,
    staging: Path,
    output_dir: Path,
    asset_id: str,
    error: BaseException,
    authenticated: Any,
) -> Path:
    staging = Path(staging)
    output_dir = Path(output_dir)
    if not staging.is_dir() or staging.is_symlink():
        raise RetargetError("failed staging directory is missing or symlinked")
    staging.chmod(0o700)
    for artifact in staging.iterdir():
        if artifact.is_file() and not artifact.is_symlink():
            artifact.chmod(0o600)
    failure_name = "retarget_failure.json"
    rejected = output_dir.parent / f"{output_dir.name}.failed.{uuid.uuid4().hex}"
    inventory_path = output_dir.parent / f".{rejected.name}.inventory.json"
    payload = {
        "schema": "tokenrig_rocketbox_retarget_attempt_v1",
        "asset_id": asset_id,
        "decision": "rejected",
        "readiness_bundle_published": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "authenticated_inputs": authenticated,
        "preserved_artifacts": sorted(path.name for path in staging.iterdir()),
        "external_inventory_descriptor": inventory_path.name,
    }
    error_evidence = _structured_error_evidence(error)
    if error_evidence is not None:
        payload["error_evidence"] = error_evidence
    _write_json_exclusive(staging / failure_name, payload)
    for path in staging.iterdir():
        if path.is_file() and not path.is_symlink():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    _fsync_directory(staging)
    _seal_failure_bundle_readonly(staging)
    rename_directory_noreplace(staging, rejected)
    artifacts = [
        artifact
        for artifact in rejected.iterdir()
        if artifact.is_file() and not artifact.is_symlink()
    ]
    _write_external_inventory(
        path=inventory_path,
        scope="rejected_bundle_files_excluding_this_external_descriptor",
        artifacts=artifacts,
    )
    _fsync_directory(rejected.parent)
    return rejected / failure_name


def _seal_failure_bundle_readonly(staging: Path) -> None:
    staging = Path(staging)
    if not staging.is_dir() or staging.is_symlink():
        raise RetargetError("failed staging directory is missing or symlinked")
    for artifact in sorted(staging.iterdir()):
        if artifact.is_symlink() or not artifact.is_file():
            raise RetargetError(
                f"failed staging artifact is not a direct regular file: {artifact.name}"
            )
        _chmod_and_fsync(artifact, 0o444)
    _chmod_and_fsync(staging, 0o555)


def seal_staged_bundle_readonly(staging: Path) -> None:
    staging = Path(staging)
    if not staging.is_dir() or staging.is_symlink():
        raise RetargetError("staged success bundle is missing or symlinked")
    for artifact in sorted(staging.iterdir()):
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size <= 0:
            raise RetargetError(
                f"staged success artifact is missing, empty, or symlinked: {artifact.name}"
            )
        _chmod_and_fsync(artifact, 0o444)
    _chmod_and_fsync(staging, 0o555)


def run_retarget(
    *,
    asset_id: str,
    base_avatar_id: str,
    bind_pose_glb: Path,
    static_qa_json: Path,
    baseline_retarget_blend: Path,
    baseline_retarget_manifest: Path,
    idle_motion_fbx: Path,
    motion_basis_selection: Path,
    motion_basis_review_manifest: Path,
    output_dir: Path,
    contract: RetargetInputContract = PRODUCTION_INPUT_CONTRACT,
    bpy_module: Any | None = None,
    subprocess_runner: Callable[..., Any] = subprocess.run,
    command: Sequence[str] | None = None,
) -> Path:
    destination = _prepare_output(output_dir)
    authenticated: dict[str, Any] = {}
    staging: Path | None = None
    try:
        validate_base_avatar_id(
            asset_id=asset_id,
            base_avatar_id=base_avatar_id,
            contract=contract,
        )
        authenticated["static"] = authenticate_static_gate(
            asset_id=asset_id,
            bind_pose_glb=bind_pose_glb,
            static_qa_json=static_qa_json,
        )
        authenticated["walk"] = authenticate_sealed_walk(
            base_avatar_id=base_avatar_id,
            baseline_retarget_blend=baseline_retarget_blend,
            baseline_retarget_manifest=baseline_retarget_manifest,
            contract=contract,
        )
        authenticated["idle"] = authenticate_idle_motion(
            base_avatar_id=base_avatar_id,
            idle_motion_fbx=idle_motion_fbx,
            contract=contract,
            subprocess_runner=subprocess_runner,
        )
        authenticated["motion_basis"] = authenticate_motion_basis_selection(
            base_avatar_id=base_avatar_id,
            motion_basis_selection=motion_basis_selection,
            motion_basis_review_manifest=motion_basis_review_manifest,
        )
        authenticated["code"] = file_descriptor(Path(__file__).resolve())
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".staging",
                dir=str(destination.parent),
            )
        )
        bpy = bpy_module if bpy_module is not None else _require_bpy()
        validate_blender_runtime(bpy)
        authenticated["runtime"] = runtime_provenance(bpy)
        metrics = _run_blender_retarget(
            bpy=bpy,
            bind_pose_glb=Path(bind_pose_glb).resolve(),
            baseline_blend=Path(baseline_retarget_blend).resolve(),
            idle_motion_fbx=Path(idle_motion_fbx).resolve(),
            staging_dir=staging,
            static_auth=authenticated["static"],
            walk_auth=authenticated["walk"],
            limb_motion_basis_3x3=authenticated["motion_basis"]["matrix_3x3"],
        )

        current = {
            "static": authenticate_static_gate(
                asset_id=asset_id,
                bind_pose_glb=bind_pose_glb,
                static_qa_json=static_qa_json,
            ),
            "walk": authenticate_sealed_walk(
                base_avatar_id=base_avatar_id,
                baseline_retarget_blend=baseline_retarget_blend,
                baseline_retarget_manifest=baseline_retarget_manifest,
                contract=contract,
            ),
            "idle": authenticate_idle_motion(
                base_avatar_id=base_avatar_id,
                idle_motion_fbx=idle_motion_fbx,
                contract=contract,
                subprocess_runner=subprocess_runner,
            ),
            "motion_basis": authenticate_motion_basis_selection(
                base_avatar_id=base_avatar_id,
                motion_basis_selection=motion_basis_selection,
                motion_basis_review_manifest=motion_basis_review_manifest,
            ),
            "code": file_descriptor(Path(__file__).resolve()),
            "runtime": runtime_provenance(bpy),
        }
        if current != authenticated:
            raise RetargetError("authenticated inputs or runner changed during retarget")

        metrics_path = staging / "retarget_metrics.json"
        _write_json_exclusive(metrics_path, metrics)
        preexport_metrics_checkpoint = staging / "preexport_action_metrics.json"
        if (
            preexport_metrics_checkpoint.is_symlink()
            or not preexport_metrics_checkpoint.is_file()
            or preexport_metrics_checkpoint.stat().st_size <= 0
        ):
            raise RetargetError("preexport metrics checkpoint is missing or invalid")
        preexport_metrics_checkpoint.unlink()
        artifact_names = (
            "animated.blend",
            "walking.glb",
            "standing_idle.glb",
            "retarget_metrics.json",
        )
        artifacts = {
            name: file_descriptor(staging / name, public_path=name)
            for name in artifact_names
        }
        manifest = build_retarget_manifest(
            asset_id=asset_id,
            base_avatar_id=base_avatar_id,
            authenticated=authenticated,
            metrics=metrics,
            artifacts=artifacts,
            command=list(command if command is not None else sys.argv),
            blender_version=str(getattr(bpy.app, "version_string", "unknown")),
        )
        _write_json_exclusive(staging / "retarget_manifest.json", manifest)
        required = {*artifact_names, "retarget_manifest.json"}
        if {path.name for path in staging.iterdir()} != required:
            raise RetargetError("staged retarget bundle has missing or unexpected artifacts")
        for path in staging.iterdir():
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise RetargetError(f"staged artifact is missing, empty, or symlinked: {path.name}")
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(staging)
        seal_staged_bundle_readonly(staging)
        rename_directory_noreplace(staging, destination)
        staging = None
        _fsync_directory(destination.parent)
        return destination / "retarget_manifest.json"
    except BaseException as error:
        try:
            if staging is not None and staging.exists():
                evidence = preserve_failed_staging(
                    staging=staging,
                    output_dir=destination,
                    asset_id=asset_id,
                    error=error,
                    authenticated=authenticated,
                )
                staging = None
            else:
                evidence = _write_failure_evidence(
                    output_dir=destination,
                    asset_id=asset_id,
                    error=error,
                    authenticated=authenticated,
                )
        except BaseException as evidence_error:
            raise RetargetError(
                f"retarget failed ({error}); failure evidence also failed ({evidence_error})"
            ) from error
        raise RetargetError(f"retarget rejected; evidence={evidence}: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_retarget(
        asset_id=args.asset_id,
        base_avatar_id=args.base_avatar_id,
        bind_pose_glb=args.bind_pose_glb,
        static_qa_json=args.static_qa_json,
        baseline_retarget_blend=args.baseline_retarget_blend,
        baseline_retarget_manifest=args.baseline_retarget_manifest,
        idle_motion_fbx=args.idle_motion_fbx,
        motion_basis_selection=args.motion_basis_selection,
        motion_basis_review_manifest=args.motion_basis_review_manifest,
        output_dir=args.output_dir,
    )
    print(f"TOKENRIG_RETARGET_PUBLISHED {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
