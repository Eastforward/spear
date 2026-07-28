#!/usr/bin/env python3
"""Freeze one explicit user decision for an authenticated target-native v4 review.

This tool never infers a verdict.  The caller must supply the user's explicit
decision, all six explicit check values, and the SHA-256 of the exact review
covered by that instruction.  It reauthenticates the complete source and review
lineage before publishing a new, immutable decision record.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import compose_target_native_generated_quadruped_owner_review as presentation
from tools import prepare_user_approved_generated_animal_ue_imports as bridge


RECEIPT_SCHEMA = bridge.DECISION_FREEZE_RECEIPT_SCHEMA
APPROVED = "approved_for_ue_apartment"
REJECTED = "rejected"
DECISIONS = (APPROVED, REJECTED)
PRESENTATION_EVIDENCE_FIELDS = bridge.PRESENTATION_EVIDENCE_FIELDS
MOTION_STYLE_AND_CURRENT_READBACK_MODE = (
    bridge.MOTION_STYLE_AND_CURRENT_READBACK_MODE
)
MOTION_STYLE_AND_CURRENT_READBACK_EVIDENCE_FIELDS = (
    bridge.MOTION_STYLE_AND_CURRENT_READBACK_EVIDENCE_FIELDS
)
CHECK_ARGUMENTS = {
    "walking_direction": "--walking-direction",
    "walking_limb_deformation": "--walking-limb-deformation",
    "walking_ground_contact": "--walking-ground-contact",
    "idle_ground_contact": "--idle-ground-contact",
    "body_stability": "--body-stability",
    "detached_geometry_absent": "--detached-geometry-absent",
}
PUBLISHED_FILE_NAMES = frozenset(
    {"animation_decision.json", "decision_freeze_receipt.json"}
)


def _explicit_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected the literal true or false")


def _stat_guard(path: Path, current: os.stat_result) -> dict[str, Any]:
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


def _stable_file_snapshot(path: Path, label: str) -> dict[str, Any]:
    """Read one direct file through a stable descriptor and recheck its path."""

    path = bridge._direct_file(path, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        path_before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, flags)
    except OSError as error:
        raise contracts.ContractError(f"cannot open stable {label}: {path}") from error
    try:
        opened = os.fstat(descriptor)
        opened_guard = _stat_guard(path, opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size <= 0
            or _stat_guard(path, path_before) != opened_guard
        ):
            raise contracts.ContractError(f"{label} changed before stable open")
        digest = hashlib.sha256()
        size_bytes = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size_bytes += len(block)
        closed_guard = _stat_guard(path, os.fstat(descriptor))
        if closed_guard != opened_guard or size_bytes != opened.st_size:
            raise contracts.ContractError(f"{label} changed during stable read")
    finally:
        os.close(descriptor)
    try:
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise contracts.ContractError(
            f"{label} disappeared after stable read"
        ) from error
    if _stat_guard(path, path_after) != opened_guard:
        raise contracts.ContractError(
            f"{label} path identity changed during stable read"
        )
    return {
        "record": {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": size_bytes,
        },
        "guard": opened_guard,
    }


def _snapshot_paths(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: _stable_file_snapshot(path, f"authority artifact {name}")
        for name, path in sorted(paths.items())
    }


def _directory_guard_from_fd(
    descriptor: int,
    physical_path: Path,
) -> dict[str, Any]:
    current = os.fstat(descriptor)
    if not stat.S_ISDIR(current.st_mode):
        raise contracts.ContractError("held output parent is not a directory")
    return {
        "path": str(physical_path),
        "device": current.st_dev,
        "inode": current.st_ino,
        "mode": stat.S_IMODE(current.st_mode),
        "mtime_ns": current.st_mtime_ns,
        "ctime_ns": current.st_ctime_ns,
    }


def _open_output_parent(output_root: Path) -> tuple[Path, Path, int, dict[str, Any]]:
    output_root = bridge._new_output_path(Path(output_root), "output")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root = bridge._new_output_path(output_root, "output")
    lexical_parent, _bridge_used = bridge._uses_only_exact_tmp_bridge(
        output_root.parent,
        "animation decision output parent",
    )
    try:
        physical_parent = lexical_parent.resolve(strict=True)
    except OSError as error:
        raise contracts.ContractError(
            "animation decision output parent is missing"
        ) from error
    current = os.stat(physical_parent, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode):
        raise contracts.ContractError(
            "animation decision output parent is not a directory"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(physical_parent, flags)
    except OSError as error:
        raise contracts.ContractError(
            "cannot hold animation decision output parent"
        ) from error
    try:
        guard = _directory_guard_from_fd(parent_fd, physical_parent)
        _require_parent_path_matches_fd(
            lexical_parent,
            physical_parent,
            parent_fd,
            guard,
        )
    except Exception:
        os.close(parent_fd)
        raise
    return physical_parent / output_root.name, lexical_parent, parent_fd, guard


def _require_parent_path_matches_fd(
    lexical_parent: Path,
    physical_parent: Path,
    parent_fd: int,
    expected_guard: Mapping[str, Any],
) -> None:
    current_literal, _bridge_used = bridge._uses_only_exact_tmp_bridge(
        lexical_parent,
        "animation decision output parent",
    )
    try:
        current_physical = current_literal.resolve(strict=True)
        current_stat = os.stat(current_physical, follow_symlinks=False)
    except OSError as error:
        raise contracts.ContractError(
            "animation decision output parent disappeared"
        ) from error
    observed_guard = _directory_guard_from_fd(parent_fd, physical_parent)
    if (
        current_physical != physical_parent
        or not stat.S_ISDIR(current_stat.st_mode)
        or (current_stat.st_dev, current_stat.st_ino)
        != (observed_guard["device"], observed_guard["inode"])
        or observed_guard["device"] != expected_guard["device"]
        or observed_guard["inode"] != expected_guard["inode"]
        or observed_guard["mode"] != expected_guard["mode"]
    ):
        raise contracts.ContractError(
            "animation decision output parent no longer names the held directory"
        )


def _create_staging_at(
    parent_fd: int, output_name: str
) -> tuple[str, int, tuple[int, int]]:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(128):
        name = f".{output_name}.{secrets.token_hex(12)}.staging"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
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
        if not stat.S_ISDIR(current.st_mode):
            os.close(descriptor)
            raise contracts.ContractError(
                "animation decision staging entry is not a directory"
            )
        return name, descriptor, (current.st_dev, current.st_ino)
    raise contracts.ContractError(
        "could not allocate a unique animation decision staging directory"
    )


def _write_json_at(
    directory_fd: int,
    name: str,
    value: Any,
) -> dict[str, Any]:
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError as error:
        raise contracts.ContractError(
            f"refusing to replace animation decision staging artifact: {name}"
        ) from error
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or current.st_size != len(encoded)
        ):
            raise contracts.ContractError(
                f"animation decision staging artifact is unsafe: {name}"
            )
    finally:
        os.close(descriptor)
    return {
        "path": name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }


def _seal_staging_at(directory_fd: int) -> None:
    if set(os.listdir(directory_fd)) != PUBLISHED_FILE_NAMES:
        raise contracts.ContractError("animation decision staging artifact set changed")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for name in sorted(PUBLISHED_FILE_NAMES):
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        try:
            current = os.fstat(descriptor)
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise contracts.ContractError(
                    f"animation decision staging artifact is unsafe: {name}"
                )
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
    os.fchmod(directory_fd, 0o555)
    os.fsync(directory_fd)


def _require_sealed_staging_records(
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
    staging_identity: tuple[int, int],
    expected_records: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(expected_records) != PUBLISHED_FILE_NAMES or any(
        not isinstance(record, Mapping)
        or set(record) != {"path", "sha256", "size_bytes"}
        or record.get("path") != name
        for name, record in expected_records.items()
    ):
        raise contracts.ContractError(
            "animation decision expected staging records are invalid"
        )
    directory_before = os.fstat(staging_fd)
    parent_entry_before = os.stat(
        staging_name,
        dir_fd=parent_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(directory_before.st_mode)
        or stat.S_IMODE(directory_before.st_mode) != 0o555
        or (directory_before.st_dev, directory_before.st_ino) != staging_identity
        or not stat.S_ISDIR(parent_entry_before.st_mode)
        or (parent_entry_before.st_dev, parent_entry_before.st_ino) != staging_identity
        or set(os.listdir(staging_fd)) != PUBLISHED_FILE_NAMES
    ):
        raise contracts.ContractError(
            "animation decision sealed staging directory changed before publication"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for name in sorted(PUBLISHED_FILE_NAMES):
        path_before = os.stat(
            name,
            dir_fd=staging_fd,
            follow_symlinks=False,
        )
        descriptor = os.open(name, flags, dir_fd=staging_fd)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o444
                or opened.st_nlink != 1
                or _stat_guard(Path(name), path_before)
                != _stat_guard(Path(name), opened)
            ):
                raise contracts.ContractError(
                    f"animation decision sealed staging artifact changed: {name}"
                )
            digest = hashlib.sha256()
            size_bytes = 0
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size_bytes += len(block)
            if _stat_guard(Path(name), os.fstat(descriptor)) != _stat_guard(
                Path(name),
                opened,
            ):
                raise contracts.ContractError(
                    f"animation decision sealed staging artifact raced: {name}"
                )
        finally:
            os.close(descriptor)
        path_after = os.stat(
            name,
            dir_fd=staging_fd,
            follow_symlinks=False,
        )
        observed = {
            "path": name,
            "sha256": digest.hexdigest(),
            "size_bytes": size_bytes,
        }
        if (
            _stat_guard(Path(name), path_after) != _stat_guard(Path(name), opened)
            or observed != expected_records[name]
        ):
            raise contracts.ContractError(
                f"animation decision sealed staging raw bytes changed: {name}"
            )
    directory_after = os.fstat(staging_fd)
    parent_entry_after = os.stat(
        staging_name,
        dir_fd=parent_fd,
        follow_symlinks=False,
    )
    if (
        _stat_guard(Path(staging_name), directory_after)
        != _stat_guard(Path(staging_name), directory_before)
        or not stat.S_ISDIR(parent_entry_after.st_mode)
        or (parent_entry_after.st_dev, parent_entry_after.st_ino) != staging_identity
    ):
        raise contracts.ContractError(
            "animation decision sealed staging identity raced before publication"
        )


def _atomic_publish_no_replace(
    parent_fd: int,
    staging_name: str,
    output_name: str,
    *,
    staging_fd: int,
    staging_identity: tuple[int, int],
    expected_records: Mapping[str, Mapping[str, Any]],
    lexical_parent: Path,
    physical_parent: Path,
    expected_parent_guard: Mapping[str, Any],
) -> None:
    _require_parent_path_matches_fd(
        lexical_parent,
        physical_parent,
        parent_fd,
        expected_parent_guard,
    )
    _require_sealed_staging_records(
        parent_fd,
        staging_fd,
        staging_name,
        staging_identity,
        expected_records,
    )
    try:
        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2")
    except AttributeError as error:
        raise contracts.ContractError(
            "atomic no-replace animation decision publication is unavailable"
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
            f"refusing to replace animation decision output: {output_name}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise contracts.ContractError(
            "atomic no-replace animation decision publication is unsupported"
        )
    raise OSError(error_number, os.strerror(error_number), output_name)


def _remove_owned_staging_at(
    parent_fd: int,
    staging_fd: int,
    staging_name: str,
    identity: tuple[int, int],
) -> None:
    """Remove only our two known files through held descriptors.

    Any unknown child, directory child, or identity mismatch is quarantined in
    place and reported.  This function never traverses a path or follows a link.
    """

    held = os.fstat(staging_fd)
    try:
        parent_entry = os.stat(
            staging_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise contracts.ContractError(
            "animation decision staging directory disappeared; quarantined"
        ) from error
    if (
        not stat.S_ISDIR(held.st_mode)
        or not stat.S_ISDIR(parent_entry.st_mode)
        or (held.st_dev, held.st_ino) != identity
        or (parent_entry.st_dev, parent_entry.st_ino) != identity
    ):
        raise contracts.ContractError(
            "animation decision staging identity changed; quarantined"
        )
    children = set(os.listdir(staging_fd))
    unknown = children - PUBLISHED_FILE_NAMES
    if unknown:
        raise contracts.ContractError(
            "animation decision staging contains unknown artifacts; quarantined: "
            + ", ".join(sorted(unknown))
        )
    child_stats: dict[str, os.stat_result] = {}
    for child in sorted(children):
        current = os.stat(child, dir_fd=staging_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
            raise contracts.ContractError(
                f"animation decision staging artifact is unsafe; quarantined: {child}"
            )
        child_stats[child] = current
    os.fchmod(staging_fd, 0o700)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for child in sorted(child_stats):
        descriptor = os.open(child, flags, dir_fd=staging_fd)
        try:
            opened = os.fstat(descriptor)
            expected = child_stats[child]
            if (opened.st_dev, opened.st_ino) != (
                expected.st_dev,
                expected.st_ino,
            ):
                raise contracts.ContractError(
                    f"animation decision staging artifact changed; quarantined: {child}"
                )
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        os.unlink(child, dir_fd=staging_fd)
    os.fsync(staging_fd)
    if os.listdir(staging_fd):
        raise contracts.ContractError(
            "animation decision staging was repopulated; quarantined"
        )
    current = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        raise contracts.ContractError(
            "animation decision staging parent entry changed; quarantined"
        )
    os.rmdir(staging_name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def authenticate_presentation(
    *,
    presentation_receipt_path: Path,
    expected_presentation_receipt_sha256: str,
    animation_review_path: Path,
    expected_animation_review_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_receipt_sha256 = bridge._require_sha256(
        expected_presentation_receipt_sha256,
        "expected presentation receipt file sha256",
    )
    expected_review_sha256 = bridge._require_sha256(
        expected_animation_review_sha256,
        "expected animation review file sha256",
    )
    review_snapshot = _stable_file_snapshot(
        animation_review_path,
        "target-native animation review",
    )
    expected_review_record = review_snapshot["record"]
    review_path = Path(expected_review_record["path"])
    if expected_review_record["sha256"] != expected_review_sha256:
        raise contracts.ContractError(
            "animation review changed before presentation authentication"
        )
    try:
        payload, receipt_record = presentation.load_presentation_receipt(
            Path(presentation_receipt_path),
            expected_receipt_sha256,
            expected_source_review_sha256=expected_review_sha256,
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != presentation.PRESENTATION_SCHEMA
            or payload.get("expected_source_review_sha256") != expected_review_sha256
            or payload.get("source_review") != expected_review_record
            or not isinstance(receipt_record, dict)
            or set(receipt_record) != {"path", "sha256", "size_bytes"}
            or receipt_record.get("sha256") != expected_receipt_sha256
        ):
            raise presentation.PresentationContractError(
                "presentation receipt is not bound to the exact v4 review"
            )
        internal_receipt_sha256 = presentation.require_sha256(
            payload.get("receipt_sha256"),
            "presentation receipt canonical self-hash",
        )
        if internal_receipt_sha256 != presentation.hash_without(
            payload,
            "receipt_sha256",
        ):
            raise presentation.PresentationContractError(
                "presentation receipt canonical self-hash failed"
            )

        receipt_path = presentation.resolve_regular_file(
            Path(receipt_record["path"]),
            "presentation receipt",
        )
        presentation.require_readonly_publication(receipt_path.parent)
        receipt_bytes, receipt_guard = presentation.read_stable_bytes(
            receipt_path,
            "presentation receipt",
        )
        if (
            str(receipt_path) != receipt_record["path"]
            or presentation.sha256_bytes(receipt_bytes) != expected_receipt_sha256
            or len(receipt_bytes) != receipt_record["size_bytes"]
        ):
            raise presentation.PresentationContractError(
                "presentation receipt changed after authentication"
            )

        output = payload.get("output")
        if not isinstance(output, dict):
            raise presentation.PresentationContractError(
                "presentation output object is missing"
            )
        output_path = presentation.resolve_regular_file(
            Path(output.get("path", "")),
            "presentation output video",
        )
        if (
            output_path.parent != receipt_path.parent
            or output_path.name != presentation.OUTPUT_VIDEO_NAME
        ):
            raise presentation.PresentationContractError(
                "presentation output video must be next to its receipt"
            )
        output_bytes, output_guard = presentation.read_stable_bytes(
            output_path,
            "presentation output video",
        )
        if {
            "path": str(output_path),
            "sha256": presentation.sha256_bytes(output_bytes),
            "size_bytes": len(output_bytes),
        } != {key: output.get(key) for key in ("path", "sha256", "size_bytes")}:
            raise presentation.PresentationContractError(
                "presentation output video failed real-byte authentication"
            )
        presentation.require_same_guard(
            receipt_guard,
            receipt_path,
            "presentation receipt",
        )
        presentation.require_same_guard(
            output_guard,
            output_path,
            "presentation output video",
        )
        if (
            _stable_file_snapshot(
                review_path,
                "target-native animation review",
            )
            != review_snapshot
        ):
            raise presentation.PresentationContractError(
                "animation review changed during presentation authentication"
            )
        presentation.require_readonly_publication(receipt_path.parent)
        publication = os.stat(receipt_path.parent, follow_symlinks=False)
        authority_guards = {
            "publication_directory": {
                "path": str(receipt_path.parent),
                "device": publication.st_dev,
                "inode": publication.st_ino,
                "mode": stat.S_IMODE(publication.st_mode),
                "mtime_ns": publication.st_mtime_ns,
                "ctime_ns": publication.st_ctime_ns,
            },
            "animation_review": review_snapshot["guard"],
            "presentation_receipt": receipt_guard,
            "output_video": output_guard,
        }
    except (presentation.PresentationContractError, OSError) as error:
        raise contracts.ContractError(
            f"presentation evidence authentication failed: {error}"
        ) from error

    evidence = {
        "presentation_receipt": copy.deepcopy(receipt_record),
        "expected_presentation_receipt_file_sha256": expected_receipt_sha256,
        "presentation_receipt_sha256": internal_receipt_sha256,
        "output_video": copy.deepcopy(output),
    }
    if set(evidence) != PRESENTATION_EVIDENCE_FIELDS:
        raise contracts.ContractError("presentation evidence fields changed")
    return evidence, authority_guards


def authenticate_motion_style_and_current_readback(
    *,
    motion_style_approval_path: Path,
    expected_motion_style_approval_sha256: str,
    current_asset_short_readback_path: Path,
    expected_current_asset_short_readback_sha256: str,
    expected_asset_id: str,
    animation_review_path: Path,
    animation_review_payload: Mapping[str, Any],
    expected_animation_review_sha256: str,
    reviewed_animated_glb: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate reusable motion style separately from current geometry."""

    expected_review_sha256 = bridge._require_sha256(
        expected_animation_review_sha256,
        "expected animation review file sha256",
    )
    expected_style_sha256 = bridge._require_sha256(
        expected_motion_style_approval_sha256,
        "expected motion-style approval file sha256",
    )
    expected_readback_sha256 = bridge._require_sha256(
        expected_current_asset_short_readback_sha256,
        "expected current-asset short-readback file sha256",
    )
    review_path = bridge._direct_file(
        animation_review_path,
        "target-native animation review",
    )
    animated_glb = bridge._direct_file(
        reviewed_animated_glb,
        "current reviewed animated GLB",
    )
    if bridge._sha256_file(review_path) != expected_review_sha256:
        raise contracts.ContractError(
            "animation review changed before compact approval authentication"
        )
    (
        _style_path,
        _style_payload,
        style_descriptor,
        _style_video,
    ) = bridge.load_motion_style_approval(
        motion_style_approval_path,
        expected_file_sha256=expected_style_sha256,
    )
    (
        _readback_path,
        _readback_payload,
        readback_descriptor,
        _readback_videos,
    ) = bridge.load_current_asset_short_readback(
        current_asset_short_readback_path,
        expected_file_sha256=expected_readback_sha256,
        expected_asset_id=expected_asset_id,
        review_path=review_path,
        review_payload=animation_review_payload,
        reviewed_animated_glb=animated_glb,
    )
    evidence = {
        "mode": MOTION_STYLE_AND_CURRENT_READBACK_MODE,
        "motion_style_approval": style_descriptor,
        "current_asset_short_readback": readback_descriptor,
    }
    canonical_evidence, canonical_paths = (
        bridge.load_motion_style_and_current_readback_evidence(
            evidence,
            expected_asset_id=expected_asset_id,
            review_path=review_path,
            review_payload=animation_review_payload,
            reviewed_animated_glb=animated_glb,
        )
    )
    if canonical_evidence != evidence:
        raise contracts.ContractError(
            "compact animation approval evidence changed during authentication"
        )
    guarded_paths = {
        "animation_review": review_path,
        "reviewed_animated_glb": animated_glb,
        **canonical_paths,
    }
    snapshots = _snapshot_paths(guarded_paths)
    return evidence, {
        name: snapshot["guard"] for name, snapshot in snapshots.items()
    }


def authenticate_review(
    *,
    source_registry_manifest_path: Path,
    expected_source_registry_sha256: str,
    source_asset_path: Path,
    animation_review_path: Path,
    expected_animation_review_sha256: str,
    artifact_roots: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    expected_source_registry_sha256 = bridge._require_sha256(
        expected_source_registry_sha256,
        "expected source registry file sha256",
    )
    expected_animation_review_sha256 = bridge._require_sha256(
        expected_animation_review_sha256,
        "expected animation review file sha256",
    )
    primary_before = {
        "source_asset_registry": _stable_file_snapshot(
            source_registry_manifest_path,
            "source asset registry",
        ),
        "source_asset": _stable_file_snapshot(
            source_asset_path,
            "source asset",
        ),
        "animation_review": _stable_file_snapshot(
            animation_review_path,
            "target-native animation review",
        ),
    }
    review_path = Path(primary_before["animation_review"]["record"]["path"])
    roots = {
        name: bridge._uses_only_exact_tmp_bridge(
            Path(path),
            f"artifact root {name}",
        )[0]
        for name, path in (artifact_roots or bridge.DEFAULT_ARTIFACT_ROOTS).items()
    }
    (
        registry_path,
        registry,
        source_request,
        source_profile,
        registry_mode,
    ) = bridge.load_source_registry_anchor(
        source_registry_manifest_path,
        source_asset_path,
        expected_file_sha256=expected_source_registry_sha256,
    )
    source_path, source_asset, source_artifacts = bridge.load_source_asset(
        source_asset_path,
        roots,
        request=source_request,
        profile=source_profile,
        require_derived_authority=(
            registry.get("schema")
            in {
                bridge.source_registry.LEGACY_DERIVED_REGISTRY_SCHEMA,
                bridge.source_registry.DERIVED_REGISTRY_SCHEMA,
            }
        ),
        expected_raw_static_decision_batch=registry.get(
            "static_decision_batch"
        ),
    )
    (
        authenticated_review_path,
        review,
        animated_glb,
        review_artifacts,
    ) = bridge.load_animation_review(
        review_path,
        source_asset=source_asset,
        source_artifacts=source_artifacts,
    )
    if (
        authenticated_review_path != review_path
        or review.get("schema") != bridge.BRANCHED_GENERATED_REVIEW_SCHEMA
        or review.get("status") != "research_candidate_pending_human_review"
        or review.get("formal_dataset_registration_authorized") is not False
        or review.get("automatic_admission_gates", {}).get("all_automatic_gates_passed")
        is not True
    ):
        raise contracts.ContractError(
            "only a fully authenticated target-native v4 review can be decided"
        )
    graph_paths = {
        "source_asset_registry": registry_path,
        "source_asset": source_path,
        "animation_review": review_path,
        "reviewed_animated_glb": animated_glb,
        **{
            f"source_artifact:{name}": path
            for name, path in sorted(source_artifacts.items())
        },
        **{
            f"review_artifact:{name}": path
            for name, path in sorted(review_artifacts.items())
        },
    }
    authority_graph = _snapshot_paths(graph_paths)
    for name in ("source_asset_registry", "source_asset", "animation_review"):
        if authority_graph[name] != primary_before[name]:
            raise contracts.ContractError(
                f"{name.replace('_', ' ')} changed during complete authentication"
            )
    if (
        authority_graph["source_asset_registry"]["record"]["sha256"]
        != expected_source_registry_sha256
    ):
        raise contracts.ContractError(
            "source asset registry does not match the external expected SHA-256"
        )
    if (
        authority_graph["animation_review"]["record"]["sha256"]
        != expected_animation_review_sha256
    ):
        raise contracts.ContractError(
            "animation review does not match the external expected SHA-256"
        )
    return {
        "registry_path": registry_path,
        "registry": registry,
        "registry_mode": registry_mode,
        "source_request": source_request,
        "source_profile": source_profile,
        "source_path": source_path,
        "source_asset": source_asset,
        "source_artifacts": source_artifacts,
        "review_path": review_path,
        "review": review,
        "animated_glb": animated_glb,
        "review_artifacts": review_artifacts,
        "authority_graph": authority_graph,
    }


def freeze_decision(
    *,
    source_registry_manifest_path: Path,
    expected_source_registry_sha256: str,
    source_asset_path: Path,
    animation_review_path: Path,
    expected_animation_review_sha256: str,
    presentation_receipt_path: Path | None = None,
    expected_presentation_receipt_sha256: str | None = None,
    motion_style_approval_path: Path | None = None,
    expected_motion_style_approval_sha256: str | None = None,
    current_asset_short_readback_path: Path | None = None,
    expected_current_asset_short_readback_sha256: str | None = None,
    decision: str,
    checks: Mapping[str, bool] | None = None,
    caveats: Sequence[str],
    notes: str,
    user_explicit_decision: str | None = None,
    user_explicit_review_sha256: str | None = None,
    user_explicit_presentation_receipt_sha256: str | None = None,
    user_explicit_motion_style_approval_sha256: str | None = None,
    output_root: Path,
    artifact_roots: Mapping[str, Path] | None = None,
) -> Path:
    if decision not in DECISIONS:
        raise contracts.ContractError("animation decision is invalid")
    presentation_arguments = (
        presentation_receipt_path,
        expected_presentation_receipt_sha256,
        user_explicit_presentation_receipt_sha256,
    )
    compact_arguments = (
        motion_style_approval_path,
        expected_motion_style_approval_sha256,
        current_asset_short_readback_path,
        expected_current_asset_short_readback_sha256,
        user_explicit_motion_style_approval_sha256,
    )
    presentation_mode = all(value is not None for value in presentation_arguments)
    compact_mode = all(value is not None for value in compact_arguments)
    if (
        presentation_mode == compact_mode
        or (
            any(value is not None for value in presentation_arguments)
            and not presentation_mode
        )
        or (
            any(value is not None for value in compact_arguments)
            and not compact_mode
        )
    ):
        raise contracts.ContractError(
            "supply exactly one complete animation approval evidence mode"
        )
    if presentation_mode:
        if user_explicit_decision != decision:
            raise contracts.ContractError(
                "the frozen verdict must match the user's explicit decision"
            )
        bridge._require_sha256(
            user_explicit_review_sha256,
            "user-explicit animation review sha256",
        )
        if user_explicit_review_sha256 != expected_animation_review_sha256:
            raise contracts.ContractError(
                "the user's explicit instruction is not bound to the expected "
                "review"
            )
        expected_presentation_receipt_sha256 = bridge._require_sha256(
            expected_presentation_receipt_sha256,
            "expected presentation receipt file sha256",
        )
        bridge._require_sha256(
            user_explicit_presentation_receipt_sha256,
            "user-explicit presentation receipt file sha256",
        )
        if (
            user_explicit_presentation_receipt_sha256
            != expected_presentation_receipt_sha256
        ):
            raise contracts.ContractError(
                "the user's explicit instruction is not bound to the expected "
                "presentation receipt"
            )
    else:
        if decision != APPROVED:
            raise contracts.ContractError(
                "motion-style/current-readback evidence only authorizes approval"
            )
        if (
            user_explicit_decision is not None
            and user_explicit_decision != decision
        ):
            raise contracts.ContractError(
                "an optional compact verdict cannot contradict the authenticated "
                "approval evidence"
            )
        expected_motion_style_approval_sha256 = bridge._require_sha256(
            expected_motion_style_approval_sha256,
            "expected motion-style approval file sha256",
        )
        expected_current_asset_short_readback_sha256 = bridge._require_sha256(
            expected_current_asset_short_readback_sha256,
            "expected current-asset short-readback file sha256",
        )
        bridge._require_sha256(
            user_explicit_motion_style_approval_sha256,
            "user-explicit motion-style approval file sha256",
        )
        if (
            user_explicit_motion_style_approval_sha256
            != expected_motion_style_approval_sha256
        ):
            raise contracts.ContractError(
                "the user's explicit instruction is not bound to the approved "
                "motion-style baseline"
            )
    supplied_checks_are_valid = bool(
        isinstance(checks, Mapping)
        and set(checks) == bridge.DECISION_CHECK_FIELDS
        and all(isinstance(value, bool) for value in checks.values())
    )
    if presentation_mode and not supplied_checks_are_valid:
        raise contracts.ContractError(
            "all six animation decision checks must be explicit booleans"
        )
    if (
        presentation_mode
        and decision == APPROVED
        and not all(checks.values())
    ):
        raise contracts.ContractError("approval requires all six checks to pass")
    if not presentation_mode and checks is not None and not supplied_checks_are_valid:
        raise contracts.ContractError(
            "compact decision checks, when supplied, must be complete booleans"
        )
    if (
        not isinstance(notes, str)
        or not notes.strip()
        or not isinstance(caveats, Sequence)
        or isinstance(caveats, (str, bytes))
        or any(not isinstance(value, str) or not value for value in caveats)
        or len(caveats) != len(set(caveats))
    ):
        raise contracts.ContractError("decision notes/caveats are invalid")

    authority_arguments = {
        "source_registry_manifest_path": source_registry_manifest_path,
        "expected_source_registry_sha256": expected_source_registry_sha256,
        "source_asset_path": source_asset_path,
        "animation_review_path": animation_review_path,
        "expected_animation_review_sha256": expected_animation_review_sha256,
        "artifact_roots": artifact_roots,
    }
    authority = authenticate_review(
        **authority_arguments,
    )
    registry_mode = authority["registry_mode"]
    source_asset = authority["source_asset"]
    review_path = authority["review_path"]
    review_artifacts = authority["review_artifacts"]
    authority_graph = authority["authority_graph"]
    if presentation_mode:
        presentation_evidence, presentation_guards = authenticate_presentation(
            presentation_receipt_path=presentation_receipt_path,
            expected_presentation_receipt_sha256=(
                expected_presentation_receipt_sha256
            ),
            animation_review_path=review_path,
            expected_animation_review_sha256=expected_animation_review_sha256,
        )
    else:
        presentation_evidence, presentation_guards = (
            authenticate_motion_style_and_current_readback(
                motion_style_approval_path=motion_style_approval_path,
                expected_motion_style_approval_sha256=(
                    expected_motion_style_approval_sha256
                ),
                current_asset_short_readback_path=(
                    current_asset_short_readback_path
                ),
                expected_current_asset_short_readback_sha256=(
                    expected_current_asset_short_readback_sha256
                ),
                expected_asset_id=source_asset["asset_id"],
                animation_review_path=review_path,
                animation_review_payload=authority["review"],
                expected_animation_review_sha256=(
                    expected_animation_review_sha256
                ),
                reviewed_animated_glb=authority["animated_glb"],
            )
        )
    if presentation_mode:
        decision_checks = dict(checks)
    else:
        decision_checks = {
            name: True for name in bridge.DECISION_CHECK_FIELDS
        }
        if checks is not None and dict(checks) != decision_checks:
            raise contracts.ContractError(
                "caller-supplied compact checks contradict the authenticated "
                "motion-style and current-review machine gates"
            )
    if presentation_guards["animation_review"] != authority_graph[
        "animation_review"
    ]["guard"]:
        raise contracts.ContractError(
            "animation review authority changed between review and approval "
            "authentication"
        )
    if (
        not presentation_mode
        and presentation_guards["reviewed_animated_glb"]
        != authority_graph["reviewed_animated_glb"]["guard"]
    ):
        raise contracts.ContractError(
            "reviewed animated GLB changed between review and short readback "
            "authentication"
        )
    (
        output_root,
        lexical_output_parent,
        parent_fd,
        output_parent_guard,
    ) = _open_output_parent(Path(output_root))
    staging_fd = -1
    staging_name = ""
    staging_identity = (0, 0)
    published = False
    try:
        _require_parent_path_matches_fd(
            lexical_output_parent,
            output_root.parent,
            parent_fd,
            output_parent_guard,
        )
        staging_name, staging_fd, staging_identity = _create_staging_at(
            parent_fd,
            output_root.name,
        )
        output_parent_guard = _directory_guard_from_fd(
            parent_fd,
            output_root.parent,
        )
        state = "research_candidate" if decision == APPROVED else "rejected"
        next_gate = (
            "ue_import_metric_trajectory_audio_and_apartment_media"
            if decision == APPROVED
            else "stop"
        )
        record: dict[str, Any] = {
            "schema": bridge.DECISION_SCHEMA,
            "asset_id": source_asset["asset_id"],
            "review_sha256": expected_animation_review_sha256,
            "decision": decision,
            "checks": {
                name: decision_checks[name]
                for name in sorted(bridge.DECISION_CHECK_FIELDS)
            },
            "caveats": list(caveats),
            "notes": notes,
            "review": copy.deepcopy(authority_graph["animation_review"]["record"]),
            "state_classification": state,
            "formal_dataset_registration_authorized": False,
            "next_gate": next_gate,
        }
        record["decision_sha256"] = bridge._hash_without(record, "decision_sha256")
        decision_record = _write_json_at(
            staging_fd,
            "animation_decision.json",
            record,
        )
        if presentation_mode:
            user_instruction_binding = {
                "decision": decision,
                "review_sha256": user_explicit_review_sha256,
                "all_six_checks_explicit": True,
                "presentation_receipt_file_sha256": (
                    user_explicit_presentation_receipt_sha256
                ),
            }
        else:
            user_instruction_binding = {
                "decision": decision,
                "motion_style_approval_file_sha256": (
                    user_explicit_motion_style_approval_sha256
                ),
                "current_asset_short_readback_file_sha256": (
                    expected_current_asset_short_readback_sha256
                ),
                "current_asset_readback_is_machine_gate": True,
            }
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "status": "frozen",
            "state_classification": state,
            "formal_dataset_registration_authorized": False,
            "source_asset_registry": copy.deepcopy(
                authority_graph["source_asset_registry"]["record"]
            ),
            "expected_source_asset_registry_file_sha256": (
                expected_source_registry_sha256
            ),
            "source_asset_registry_validation_mode": registry_mode,
            "source_asset": copy.deepcopy(authority_graph["source_asset"]["record"]),
            "animation_review": copy.deepcopy(
                authority_graph["animation_review"]["record"]
            ),
            "expected_animation_review_file_sha256": (expected_animation_review_sha256),
            "presentation_evidence": presentation_evidence,
            "user_instruction_binding": user_instruction_binding,
            "user_instruction_authority": dict(bridge.USER_INSTRUCTION_AUTHORITY),
            "authenticated_review_artifact_count": len(review_artifacts),
            "animation_decision": decision_record,
            "decision_sha256": record["decision_sha256"],
        }
        receipt["receipt_sha256"] = bridge._hash_without(receipt, "receipt_sha256")
        receipt_file_record = _write_json_at(
            staging_fd,
            "decision_freeze_receipt.json",
            receipt,
        )
        _seal_staging_at(staging_fd)
        staging_stat = os.fstat(staging_fd)
        if (
            not stat.S_ISDIR(staging_stat.st_mode)
            or (staging_stat.st_dev, staging_stat.st_ino) != staging_identity
        ):
            raise contracts.ContractError(
                "animation decision staging directory identity changed"
            )

        final_authority = authenticate_review(**authority_arguments)
        if final_authority != authority:
            raise contracts.ContractError(
                "complete source and review authority graph changed before "
                "decision publication"
            )
        if presentation_mode:
            final_presentation, final_presentation_guards = authenticate_presentation(
                presentation_receipt_path=presentation_receipt_path,
                expected_presentation_receipt_sha256=(
                    expected_presentation_receipt_sha256
                ),
                animation_review_path=final_authority["review_path"],
                expected_animation_review_sha256=expected_animation_review_sha256,
            )
        else:
            final_presentation, final_presentation_guards = (
                authenticate_motion_style_and_current_readback(
                    motion_style_approval_path=motion_style_approval_path,
                    expected_motion_style_approval_sha256=(
                        expected_motion_style_approval_sha256
                    ),
                    current_asset_short_readback_path=(
                        current_asset_short_readback_path
                    ),
                    expected_current_asset_short_readback_sha256=(
                        expected_current_asset_short_readback_sha256
                    ),
                    expected_asset_id=final_authority["source_asset"]["asset_id"],
                    animation_review_path=final_authority["review_path"],
                    animation_review_payload=final_authority["review"],
                    expected_animation_review_sha256=(
                        expected_animation_review_sha256
                    ),
                    reviewed_animated_glb=final_authority["animated_glb"],
                )
            )
        if (
            final_presentation != presentation_evidence
            or final_presentation_guards != presentation_guards
            or final_presentation_guards["animation_review"]
            != final_authority["authority_graph"]["animation_review"]["guard"]
            or (
                not presentation_mode
                and final_presentation_guards["reviewed_animated_glb"]
                != final_authority["authority_graph"]["reviewed_animated_glb"][
                    "guard"
                ]
            )
        ):
            raise contracts.ContractError(
                "animation approval evidence changed before decision publication"
            )
        _require_parent_path_matches_fd(
            lexical_output_parent,
            output_root.parent,
            parent_fd,
            output_parent_guard,
        )
        staging_entry = os.stat(
            staging_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(staging_entry.st_mode)
            or (staging_entry.st_dev, staging_entry.st_ino) != staging_identity
        ):
            raise contracts.ContractError(
                "animation decision staging parent entry changed"
            )
        try:
            os.stat(
                output_root.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise contracts.ContractError(
                "animation decision output appeared concurrently"
            )
        _atomic_publish_no_replace(
            parent_fd,
            staging_name,
            output_root.name,
            staging_fd=staging_fd,
            staging_identity=staging_identity,
            expected_records={
                decision_record["path"]: decision_record,
                receipt_file_record["path"]: receipt_file_record,
            },
            lexical_parent=lexical_output_parent,
            physical_parent=output_root.parent,
            expected_parent_guard=output_parent_guard,
        )
        published = True
        os.fsync(parent_fd)
        return output_root / "animation_decision.json"
    except Exception as error:
        if not published and staging_fd >= 0 and staging_name:
            try:
                _remove_owned_staging_at(
                    parent_fd,
                    staging_fd,
                    staging_name,
                    staging_identity,
                )
            except Exception as cleanup_error:
                raise contracts.ContractError(
                    f"{error}; animation decision staging was quarantined: "
                    f"{cleanup_error}"
                ) from error
        raise
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(parent_fd)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry-manifest", required=True, type=Path)
    parser.add_argument("--expected-source-registry-sha256", required=True)
    parser.add_argument("--source-asset", required=True, type=Path)
    parser.add_argument("--animation-review", required=True, type=Path)
    parser.add_argument("--expected-animation-review-sha256", required=True)
    parser.add_argument("--presentation-receipt", type=Path)
    parser.add_argument("--expected-presentation-receipt-sha256")
    parser.add_argument("--motion-style-approval", type=Path)
    parser.add_argument("--expected-motion-style-approval-sha256")
    parser.add_argument("--current-asset-short-readback", type=Path)
    parser.add_argument("--expected-current-asset-short-readback-sha256")
    parser.add_argument("--decision", choices=DECISIONS, required=True)
    parser.add_argument(
        "--user-explicit-decision",
        choices=DECISIONS,
        help=(
            "Required for full-presentation mode and must equal --decision. "
            "Compact mode derives approval from the user-bound motion-style "
            "approval plus the current machine readback."
        ),
    )
    parser.add_argument(
        "--user-explicit-review-sha256",
        help=(
            "For full-presentation mode, SHA-256 of the exact review covered "
            "by the user's instruction. Compact mode uses the current review "
            "only as a machine-bound readback authority."
        ),
    )
    parser.add_argument(
        "--user-explicit-presentation-receipt-sha256",
        help=(
            "Raw file SHA-256 of the exact sealed presentation receipt covered "
            "by the user's instruction; it must equal "
            "--expected-presentation-receipt-sha256."
        ),
    )
    parser.add_argument(
        "--user-explicit-motion-style-approval-sha256",
        help=(
            "Raw file SHA-256 of the reusable Idle/Walking motion-style approval "
            "covered by the user's instruction."
        ),
    )
    for name, flag in CHECK_ARGUMENTS.items():
        parser.add_argument(
            flag,
            dest=name,
            type=_explicit_bool,
            choices=(True, False),
            help=(
                "Required in full-presentation mode. Compact mode derives the "
                "decision checks from authenticated style and current-review "
                "machine gates."
            ),
        )
    parser.add_argument("--notes", required=True)
    parser.add_argument("--caveat", action="append", default=[])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        metavar="ROOT_ID=PATH",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        decision_path = freeze_decision(
            source_registry_manifest_path=args.source_registry_manifest,
            expected_source_registry_sha256=args.expected_source_registry_sha256,
            source_asset_path=args.source_asset,
            animation_review_path=args.animation_review,
            expected_animation_review_sha256=(args.expected_animation_review_sha256),
            presentation_receipt_path=args.presentation_receipt,
            expected_presentation_receipt_sha256=(
                args.expected_presentation_receipt_sha256
            ),
            motion_style_approval_path=args.motion_style_approval,
            expected_motion_style_approval_sha256=(
                args.expected_motion_style_approval_sha256
            ),
            current_asset_short_readback_path=(
                args.current_asset_short_readback
            ),
            expected_current_asset_short_readback_sha256=(
                args.expected_current_asset_short_readback_sha256
            ),
            decision=args.decision,
            checks=(
                None
                if all(getattr(args, name) is None for name in CHECK_ARGUMENTS)
                else {
                    name: getattr(args, name)
                    for name in CHECK_ARGUMENTS
                }
            ),
            caveats=args.caveat,
            notes=args.notes,
            user_explicit_decision=args.user_explicit_decision,
            user_explicit_review_sha256=args.user_explicit_review_sha256,
            user_explicit_presentation_receipt_sha256=(
                args.user_explicit_presentation_receipt_sha256
            ),
            user_explicit_motion_style_approval_sha256=(
                args.user_explicit_motion_style_approval_sha256
            ),
            output_root=args.output_root,
            artifact_roots=bridge.parse_artifact_roots(args.artifact_root),
        )
        payload = bridge._load_finite_json(
            decision_path,
            "frozen animation decision output",
        )
    except (contracts.ContractError, OSError) as error:
        print(
            f"TARGET_NATIVE_ANIMATION_DECISION_FREEZE_FAILED {error}",
            file=sys.stderr,
        )
        return 2
    print(
        "TARGET_NATIVE_ANIMATION_DECISION_FREEZE_OK "
        f"decision={decision_path} "
        f"decision_file_sha256={bridge._sha256_file(decision_path)} "
        f"decision_sha256={payload['decision_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
