#!/usr/bin/env python3
"""Immutable full-snapshot agent review contract for Route-2 2D attributes."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from tools import route2_human_contract_common as route2_common
from tools import route2_human_qualified_candidate as qualified_candidate


PASS_STATUS = "agent_qa_passed_pending_user_acceptance"
REJECTED_STATUS = "rejected"
AGENT_2D_VISUAL_CHECKS = (
    "only_requested_attribute_changed",
    "identity_face_and_hair_preserved",
    "soft_t_pose_and_limb_gaps_preserved",
    "camera_framing_and_background_preserved",
    "target_attribute_is_visually_achieved",
    "mask_boundary_has_no_visible_patch_seam",
    "candidate_rgba_silhouette_is_clean",
    "non_target_clothing_and_foot_contact_preserved",
)
CANDIDATE_ARTIFACTS = frozenset(
    {
        "source.png",
        "raw_candidate.png",
        "candidate.png",
        "source_alpha.png",
        "candidate_alpha.png",
        "candidate_rgba.png",
        "edit_core.png",
        "transition_band.png",
        "protected_guard.png",
        "overlay.png",
        "diff.png",
        "agent_2d_decision.json",
        "generation_attempt.json",
    }
)


class AttributeReviewError(RuntimeError):
    """Raised when a Route-2 2D review snapshot is stale or unauthorized."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decision_path(bundle_dir: Path) -> Path:
    bundle_dir = Path(bundle_dir).absolute()
    return bundle_dir.with_name(f"{bundle_dir.name}.agent_2d_visual_qa.json")


def _regular(path: Path, root: Path, description: str) -> Path:
    path = Path(path).absolute()
    root = Path(root).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise AttributeReviewError(f"{description} must be a direct regular file")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AttributeReviewError(f"{description} is outside its bundle") from error
    return path


def _validate_base_qualification(
    value: Any, *, base_asset_id: str
) -> dict[str, Any]:
    expected_fields = {
        "asset_id",
        "status",
        "qualified_candidate",
        "final_branch",
        "review_dir",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise AttributeReviewError(
            "candidate base qualification has no exact qualified candidate descriptor"
        )
    pointer = value.get("qualified_candidate")
    final_branch = value.get("final_branch")
    if (
        value.get("asset_id") != base_asset_id
        or value.get("status") != PASS_STATUS
        or not isinstance(pointer, Mapping)
        or set(pointer) != {"path", "sha256", "size_bytes"}
        or not isinstance(pointer.get("path"), str)
        or not isinstance(final_branch, Mapping)
        or set(final_branch) != {"branch_id", "path", "relative_root"}
        or not isinstance(value.get("review_dir"), str)
    ):
        raise AttributeReviewError(
            "candidate base qualification qualified candidate descriptor is invalid"
        )
    pointer_path = Path(pointer["path"]).absolute()
    if (
        pointer_path.name != qualified_candidate.FILENAME
        or pointer_path.parent.name != base_asset_id
    ):
        raise AttributeReviewError("candidate qualified candidate path is not canonical")
    try:
        record = route2_common.file_record(
            pointer_path,
            root=pointer_path.parent,
            description="base qualified candidate",
            error_type=AttributeReviewError,
            require_mode=0o444,
        )
        qualified = qualified_candidate.validate_qualified_candidate(pointer_path)
    except (
        AttributeReviewError,
        qualified_candidate.QualificationError,
    ) as error:
        raise AttributeReviewError(
            f"candidate qualified candidate is stale: {error}"
        ) from error
    if (
        pointer.get("sha256") != record["sha256"]
        or pointer.get("size_bytes") != record["size_bytes"]
        or qualified.get("asset_id") != base_asset_id
        or qualified.get("base_avatar_id") != base_asset_id
        or qualified.get("status") != PASS_STATUS
        or qualified.get("final_branch") != dict(final_branch)
        or qualified.get("dynamic", {}).get("review_dir") != value["review_dir"]
    ):
        raise AttributeReviewError(
            "candidate qualified candidate owner snapshot changed"
        )
    return dict(value)


def validated_candidate_snapshot(bundle_dir: Path) -> dict[str, Any]:
    root = Path(bundle_dir).absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise AttributeReviewError("candidate bundle must be a direct real directory")
    manifest_path = _regular(root / "candidate_manifest.json", root, "candidate manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttributeReviewError(f"candidate manifest is unreadable: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "flux2_human_attribute_candidate_v2"
        or manifest.get("state_classification") != "research_candidate"
        or manifest.get("bundle_status") != "generated_pending_agent_2d_visual_qa"
        or manifest.get("agent_qa_status") != "pending_agent_2d_visual_qa"
        or manifest.get("user_acceptance") != "pending_user_review"
        or "user_approved" in json.dumps(manifest)
    ):
        raise AttributeReviewError("candidate manifest review state is invalid")
    case_id = manifest.get("case_id")
    base_asset_id = manifest.get("base_asset_id")
    downstream_asset_id = manifest.get("downstream_asset_id")
    if (
        not isinstance(case_id, str)
        or not isinstance(base_asset_id, str)
        or downstream_asset_id != f"route2_{case_id}_v1"
    ):
        raise AttributeReviewError("candidate lineage identity is invalid")
    base_qualification = _validate_base_qualification(
        manifest.get("base_route2_qualification"), base_asset_id=base_asset_id
    )
    quantitative = manifest.get("quantitative_snapshot")
    if (
        not isinstance(quantitative, Mapping)
        or quantitative.get("automatic_checks") != "passed"
        or quantitative.get("pixel_proof", {}).get("outside_changed_pixels") != 0
        or quantitative.get("pixel_proof", {}).get("outside_max_abs_channel_delta") != 0
        or quantitative.get("pixel_proof", {}).get("transition_is_feathered") is not True
        or quantitative.get("alpha_proof", {}).get("outside_changed_pixels") != 0
        or quantitative.get("case_metrics", {}).get("passed") is not True
        or not all(
            value is True
            for value in quantitative.get("case_metrics", {}).get("checks", {}).values()
        )
    ):
        raise AttributeReviewError("candidate quantitative checks are incomplete or failed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(CANDIDATE_ARTIFACTS):
        raise AttributeReviewError("candidate artifact set is incomplete or unexpected")
    artifact_sha256: dict[str, str] = {}
    artifact_sizes: dict[str, int] = {}
    image_size: tuple[int, int] | None = None
    for filename in sorted(CANDIDATE_ARTIFACTS):
        path = _regular(root / filename, root, filename)
        record = artifacts[filename]
        if (
            not isinstance(record, Mapping)
            or record.get("path") != str(path)
            or record.get("sha256") != sha256_file(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise AttributeReviewError(f"candidate artifact snapshot changed: {filename}")
        if path.stat().st_mode & 0o222:
            raise AttributeReviewError(f"candidate artifact is mutable: {filename}")
        artifact_sha256[filename] = record["sha256"]
        artifact_sizes[filename] = record["size_bytes"]
        if filename.endswith(".png"):
            with Image.open(path) as opened:
                opened.load()
                if image_size is None:
                    image_size = opened.size
                elif opened.size != image_size:
                    raise AttributeReviewError("candidate image canvases differ")
                if filename == "candidate_rgba.png":
                    if opened.mode != "RGBA" or opened.getchannel("A").getextrema() != (0, 255):
                        raise AttributeReviewError("candidate RGBA alpha is invalid")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != set(CANDIDATE_ARTIFACTS) | {"candidate_manifest.json"}:
        raise AttributeReviewError("candidate bundle has missing or unexpected files")
    return {
        "case_id": case_id,
        "base_asset_id": base_asset_id,
        "downstream_asset_id": downstream_asset_id,
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_manifest_size_bytes": manifest_path.stat().st_size,
        "artifact_sha256": artifact_sha256,
        "artifact_sizes": artifact_sizes,
        "quantitative_snapshot": dict(quantitative),
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise AttributeReviewError("atomic no-replace publication requires renameat2")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise AttributeReviewError(f"agent 2D visual QA decision already exists: {destination}")
    raise AttributeReviewError(f"atomic decision publication failed: {os.strerror(number)}")


def record_agent_2d_visual_qa(
    bundle_dir: Path,
    *,
    status: str,
    reviewer: str,
    notes: str,
    checks: Mapping[str, bool],
) -> Path:
    if status not in {PASS_STATUS, REJECTED_STATUS}:
        raise AttributeReviewError("agent 2D visual QA status is invalid")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise AttributeReviewError("agent 2D visual QA reviewer is empty")
    if not isinstance(notes, str) or not notes.strip():
        raise AttributeReviewError("agent 2D visual QA notes are empty")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(AGENT_2D_VISUAL_CHECKS)
        or any(not isinstance(value, bool) for value in checks.values())
        or (status == PASS_STATUS and not all(checks.values()))
    ):
        raise AttributeReviewError("agent 2D visual QA checklist is incomplete or failed")
    snapshot = validated_candidate_snapshot(bundle_dir)
    destination = decision_path(bundle_dir)
    if os.path.lexists(destination):
        raise AttributeReviewError(f"agent 2D visual QA decision already exists: {destination}")
    payload = {
        "schema": "human_attribute_agent_2d_visual_qa_v1",
        "case_id": snapshot["case_id"],
        "base_asset_id": snapshot["base_asset_id"],
        "downstream_asset_id": snapshot["downstream_asset_id"],
        "status": status,
        "reviewer_kind": "agent",
        "reviewer": reviewer.strip(),
        "notes": notes.strip(),
        "checks": dict(checks),
        "snapshot": snapshot,
        "user_acceptance": "pending_user_review",
    }
    if "user_approved" in json.dumps(payload):
        raise AttributeReviewError("agent 2D decision may not claim user approval")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        _rename_noreplace(temporary, destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def assert_agent_2d_qa_passed(bundle_dir: Path) -> dict[str, Any]:
    path = decision_path(bundle_dir)
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise AttributeReviewError("agent 2D visual QA is pending")
    if path.stat().st_mode & 0o222:
        raise AttributeReviewError("agent 2D visual QA decision is mutable")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttributeReviewError(f"agent 2D decision is unreadable: {error}") from error
    try:
        snapshot = validated_candidate_snapshot(bundle_dir)
    except AttributeReviewError as error:
        raise AttributeReviewError(f"candidate review snapshot changed: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "human_attribute_agent_2d_visual_qa_v1"
        or payload.get("case_id") != snapshot["case_id"]
        or payload.get("base_asset_id") != snapshot["base_asset_id"]
        or payload.get("downstream_asset_id") != snapshot["downstream_asset_id"]
        or payload.get("status") != PASS_STATUS
        or payload.get("reviewer_kind") != "agent"
        or payload.get("snapshot") != snapshot
        or payload.get("user_acceptance") != "pending_user_review"
        or "user_approved" in json.dumps(payload)
        or set(payload.get("checks", {})) != set(AGENT_2D_VISUAL_CHECKS)
        or not all(value is True for value in payload["checks"].values())
    ):
        raise AttributeReviewError("agent 2D decision or candidate snapshot changed")
    return payload
