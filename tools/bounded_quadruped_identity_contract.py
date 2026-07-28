#!/usr/bin/env python3
"""Pure contracts for one bounded, same-rig quadruped identity derivation.

The module is intentionally Blender-free and standard-library only.  A plan is
never an approval: it describes exact vertex masks, a symmetric bounded
transform, an explicit removable object, deterministic UV parameters, and
profile-driven coat regions.  Execution additionally requires a separate
machine-authorization record bound to the content-addressed preflight receipt
and limited to creation of a new research-candidate output.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import struct
from typing import Any, Mapping, Optional, Sequence
import zlib


PLAN_SCHEMA = "avengine_bounded_quadruped_identity_plan_v1"
PREFLIGHT_SCHEMA = "avengine_bounded_quadruped_identity_preflight_v1"
AUTHORIZATION_SCHEMA = (
    "avengine_bounded_quadruped_identity_machine_execution_authorization_v1"
)
REALIZATION_SCHEMA = "avengine_bounded_quadruped_identity_realization_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AXES = frozenset({"x", "y", "z", "lateral_abs"})
OPERATORS = frozenset({"gte", "lte"})
AUTHORIZATION_CHECKS = frozenset(
    {
        "positive_mask_is_only_one_ear",
        "negative_mask_is_only_one_ear",
        "masks_include_all_duplicate_position_vertices",
        "symmetric_upright_transform_is_acceptable",
        "unskinned_removal_target_is_unwanted",
        "deterministic_coat_regions_are_acceptable",
    }
)
DENSE_ANIMATION_PHASE_INTERVALS = 80


class IdentityContractError(ValueError):
    """Raised when identity evidence is incomplete, ambiguous, or unsafe."""


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise IdentityContractError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise IdentityContractError(f"non-finite JSON number is forbidden: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise IdentityContractError(f"non-finite JSON number is forbidden: {value}")
    return result


def strict_json_loads(payload: bytes) -> Any:
    """Parse unambiguous UTF-8 JSON and reject non-standard numbers."""

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except IdentityContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise IdentityContractError(f"invalid strict JSON: {error}") from error


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stat_guard(value: os.stat_result) -> tuple[int, ...]:
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


def _directory_guard(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_gid,
        value.st_nlink,
    )


def _parent_guard(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_uid,
        value.st_gid,
    )


def _require_safe_parent_permissions(value: os.stat_result) -> None:
    mode = stat.S_IMODE(value.st_mode)
    effective_uid = os.geteuid()
    effective_gid = os.getegid()
    if value.st_uid != effective_uid:
        raise IdentityContractError(
            "publication output parent must be owned by the effective user"
        )
    if mode & stat.S_IWOTH:
        raise IdentityContractError(
            "publication output parent must not be other-writable"
        )
    if mode & stat.S_IWGRP and not (
        value.st_uid == value.st_gid == effective_uid == effective_gid
    ):
        raise IdentityContractError(
            "group-writable publication output parent requires the "
            "effective user's UID-matched effective-group convention"
        )


def _renameat2_noreplace(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
) -> None:
    try:
        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2")
    except AttributeError as error:
        raise IdentityContractError(
            "atomic no-replace rename is unavailable"
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
        source_fd,
        os.fsencode(source_name),
        destination_fd,
        os.fsencode(destination_name),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise IdentityContractError("atomic no-replace rename is unsupported")
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def publication_security_boundary() -> dict[str, Any]:
    """Describe what publication does and explicitly does not guarantee."""

    return {
        "protocol": ("posix_held_dirfd_renameat2_noreplace_postrename_reverify_v2"),
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


@dataclass
class SecurePublication:
    """Held-dirfd, no-replace publication for the bounded identity producer.

    Rename is a publication boundary, not a claim that the result is ready or
    absolutely immutable.  POSIX mode bits cannot stop a malicious process
    running as the same UID from chmod/write after verification.  Consumers
    must receive an expected raw manifest SHA-256 out of band and rehash the
    complete declared file closure before use.
    """

    output_root: Path
    lexical_parent: Path
    physical_parent: Path
    parent_fd: int
    parent_guard: tuple[int, ...]
    staging_name: str
    staging_fd: int
    staging_identity: tuple[int, int]
    evidence_fd: int
    evidence_identity: tuple[int, int]
    root_identities: dict[str, tuple[int, int]]
    evidence_identities: dict[str, tuple[int, int]]
    sealed_staging_nlink: Optional[int] = None
    sealed_evidence_nlink: Optional[int] = None
    published: bool = False
    quarantine_name: Optional[str] = None

    @property
    def staging_path(self) -> Path:
        return Path(f"/proc/self/fd/{self.staging_fd}")

    @property
    def evidence_path(self) -> Path:
        return Path(f"/proc/self/fd/{self.evidence_fd}")

    def _require_proc_fd_paths(self) -> None:
        for path, descriptor, expected in (
            (self.staging_path, self.staging_fd, self.staging_identity),
            (self.evidence_path, self.evidence_fd, self.evidence_identity),
        ):
            try:
                current = os.stat(path)
            except OSError as error:
                raise IdentityContractError(
                    "Linux /proc/self/fd publication paths are unavailable"
                ) from error
            if not stat.S_ISDIR(current.st_mode) or _identity(current) != expected:
                raise IdentityContractError(
                    "Linux /proc/self/fd publication path changed identity"
                )
            if _identity(os.fstat(descriptor)) != expected:
                raise IdentityContractError("held publication directory changed")

    def _write_at(
        self,
        directory_fd: int,
        identities: dict[str, tuple[int, int]],
        name: str,
        payload: bytes,
    ) -> Path:
        self._require_parent_path_matches_fd()
        if "/" in name or not name or name in {".", ".."}:
            raise IdentityContractError("publication filename is unsafe")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise IdentityContractError("publication write made no progress")
                offset += written
            os.fsync(descriptor)
            current = os.fstat(descriptor)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or current.st_size != len(payload)
            ):
                raise IdentityContractError("publication artifact is unsafe")
            identities[name] = _identity(current)
        finally:
            os.close(descriptor)
        self._require_parent_path_matches_fd()
        base = (
            self.staging_path if directory_fd == self.staging_fd else self.evidence_path
        )
        return base / name

    def write_root(self, name: str, payload: bytes) -> Path:
        return self._write_at(
            self.staging_fd,
            self.root_identities,
            name,
            payload,
        )

    def write_evidence(self, name: str, payload: bytes) -> Path:
        return self._write_at(
            self.evidence_fd,
            self.evidence_identities,
            name,
            payload,
        )

    def write_root_json(self, name: str, value: Any) -> Path:
        return self.write_root(
            name,
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n",
        )

    def register_root_file(self, name: str) -> Path:
        self._require_parent_path_matches_fd()
        descriptor = os.open(name, _file_flags(), dir_fd=self.staging_fd)
        try:
            current = os.fstat(descriptor)
            if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                raise IdentityContractError(
                    f"generated publication artifact is unsafe: {name}"
                )
            self.root_identities[name] = _identity(current)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._require_parent_path_matches_fd()
        return self.staging_path / name

    def _read_record(
        self,
        directory_fd: int,
        name: str,
        expected_identity: tuple[int, int],
        *,
        required_mode: Optional[int] = None,
    ) -> dict[str, Any]:
        path_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(name, _file_flags(), dir_fd=directory_fd)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _identity(opened) != expected_identity
                or _stat_guard(path_before) != _stat_guard(opened)
                or (
                    required_mode is not None
                    and stat.S_IMODE(opened.st_mode) != required_mode
                )
            ):
                raise IdentityContractError(
                    f"publication artifact identity changed: {name}"
                )
            digest = hashlib.sha256()
            size_bytes = 0
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size_bytes += len(block)
            after_fd = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            _stat_guard(after_fd) != _stat_guard(opened)
            or _stat_guard(path_after) != _stat_guard(opened)
            or after_fd.st_size != size_bytes
            or path_after.st_size != size_bytes
        ):
            raise IdentityContractError(f"publication artifact raced: {name}")
        return {
            "path": name,
            "sha256": digest.hexdigest(),
            "size_bytes": size_bytes,
        }

    def seal(
        self,
        *,
        expected_root_files: set[str],
        expected_evidence_files: set[str],
    ) -> dict[str, dict[str, Any]]:
        self._require_parent_path_matches_fd()
        self._require_proc_fd_paths()
        if set(os.listdir(self.staging_fd)) != expected_root_files | {"evidence"}:
            raise IdentityContractError("publication staging root file set changed")
        if set(os.listdir(self.evidence_fd)) != expected_evidence_files:
            raise IdentityContractError("publication evidence file set changed")
        if (
            set(self.root_identities) != expected_root_files
            or set(self.evidence_identities) != expected_evidence_files
        ):
            raise IdentityContractError(
                "publication owned file identities are incomplete"
            )
        records: dict[str, dict[str, Any]] = {}
        for name in sorted(expected_root_files):
            records[name] = self._read_record(
                self.staging_fd,
                name,
                self.root_identities[name],
            )
            descriptor = os.open(name, _file_flags(), dir_fd=self.staging_fd)
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in sorted(expected_evidence_files):
            record = self._read_record(
                self.evidence_fd,
                name,
                self.evidence_identities[name],
            )
            record["path"] = f"evidence/{name}"
            records[f"evidence/{name}"] = record
            descriptor = os.open(name, _file_flags(), dir_fd=self.evidence_fd)
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fchmod(self.evidence_fd, 0o555)
        os.fsync(self.evidence_fd)
        os.fchmod(self.staging_fd, 0o555)
        os.fsync(self.staging_fd)
        self.sealed_staging_nlink = os.fstat(self.staging_fd).st_nlink
        self.sealed_evidence_nlink = os.fstat(self.evidence_fd).st_nlink
        self.verify(records)
        return records

    def _require_held_parent_guard(self) -> None:
        held = os.fstat(self.parent_fd)
        if not stat.S_ISDIR(held.st_mode) or _parent_guard(held) != self.parent_guard:
            raise IdentityContractError(
                "held publication output parent inode, owner, group, or mode changed"
            )

    def _require_parent_path_matches_fd(self) -> None:
        try:
            current_parent = self.lexical_parent.resolve(strict=True)
            current = os.stat(current_parent, follow_symlinks=False)
        except OSError as error:
            raise IdentityContractError(
                "publication output parent disappeared"
            ) from error
        self._require_held_parent_guard()
        if (
            current_parent != self.physical_parent
            or not stat.S_ISDIR(current.st_mode)
            or _parent_guard(current) != self.parent_guard
        ):
            raise IdentityContractError(
                "publication output parent no longer names the held directory "
                "with its authenticated inode, owner, group, and mode"
            )

    def verify(
        self,
        expected_records: Mapping[str, Mapping[str, Any]],
        *,
        parent_entry_name: Optional[str] = None,
    ) -> None:
        self._require_parent_path_matches_fd()
        self._require_proc_fd_paths()
        if self.sealed_staging_nlink is None or self.sealed_evidence_nlink is None:
            raise IdentityContractError(
                "publication directory link counts are unsealed"
            )
        entry_name = (
            self.staging_name if parent_entry_name is None else parent_entry_name
        )
        held = os.fstat(self.staging_fd)
        parent_entry = os.stat(
            entry_name,
            dir_fd=self.parent_fd,
            follow_symlinks=False,
        )
        held_evidence = os.fstat(self.evidence_fd)
        evidence_entry = os.stat(
            "evidence",
            dir_fd=self.staging_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(held.st_mode)
            or stat.S_IMODE(held.st_mode) != 0o555
            or _identity(held) != self.staging_identity
            or held.st_nlink != self.sealed_staging_nlink
            or _directory_guard(parent_entry) != _directory_guard(held)
            or not stat.S_ISDIR(held_evidence.st_mode)
            or stat.S_IMODE(held_evidence.st_mode) != 0o555
            or held_evidence.st_nlink != self.sealed_evidence_nlink
            or stat.S_IMODE(evidence_entry.st_mode) != 0o555
            or _identity(evidence_entry) != self.evidence_identity
            or not stat.S_ISDIR(evidence_entry.st_mode)
            or _directory_guard(evidence_entry) != _directory_guard(held_evidence)
        ):
            raise IdentityContractError("sealed publication directory identity changed")
        expected_root = set(self.root_identities) | {"evidence"}
        expected_evidence = set(self.evidence_identities)
        if (
            set(os.listdir(self.staging_fd)) != expected_root
            or set(os.listdir(self.evidence_fd)) != expected_evidence
            or set(expected_records)
            != set(self.root_identities)
            | {f"evidence/{name}" for name in self.evidence_identities}
        ):
            raise IdentityContractError("sealed publication file set changed")
        observed = {}
        for name, identity in sorted(self.root_identities.items()):
            observed[name] = self._read_record(
                self.staging_fd,
                name,
                identity,
                required_mode=0o444,
            )
        for name, identity in sorted(self.evidence_identities.items()):
            record = self._read_record(
                self.evidence_fd,
                name,
                identity,
                required_mode=0o444,
            )
            record["path"] = f"evidence/{name}"
            observed[f"evidence/{name}"] = record
        if observed != expected_records:
            raise IdentityContractError("sealed publication bytes changed")
        held_after = os.fstat(self.staging_fd)
        held_evidence_after = os.fstat(self.evidence_fd)
        parent_entry_after = os.stat(
            entry_name,
            dir_fd=self.parent_fd,
            follow_symlinks=False,
        )
        evidence_entry_after = os.stat(
            "evidence",
            dir_fd=self.staging_fd,
            follow_symlinks=False,
        )
        if (
            set(os.listdir(self.staging_fd)) != expected_root
            or set(os.listdir(self.evidence_fd)) != expected_evidence
            or _directory_guard(held_after) != _directory_guard(held)
            or _directory_guard(parent_entry_after) != _directory_guard(held)
            or _directory_guard(held_evidence_after) != _directory_guard(held_evidence)
            or _directory_guard(evidence_entry_after) != _directory_guard(held_evidence)
        ):
            raise IdentityContractError("sealed publication raced during verification")
        self._require_parent_path_matches_fd()

    def _rename_no_replace(self) -> None:
        try:
            _renameat2_noreplace(
                self.parent_fd,
                self.staging_name,
                self.parent_fd,
                self.output_root.name,
            )
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise IdentityContractError(
                    f"refusing to replace output root: {self.output_root}"
                ) from error
            raise
        # The syscall is the publication boundary.  Mark it immediately so
        # every later failure retains the final entry.
        self.published = True

    def publish(self, expected_records: Mapping[str, Mapping[str, Any]]) -> None:
        self._require_parent_path_matches_fd()
        self.verify(expected_records)
        self._rename_no_replace()
        self._require_parent_path_matches_fd()
        self.verify(
            expected_records,
            parent_entry_name=self.output_root.name,
        )
        os.fsync(self.parent_fd)

    def _quarantine_no_replace(self, quarantine_name: str) -> None:
        _renameat2_noreplace(
            self.parent_fd,
            self.staging_name,
            self.parent_fd,
            quarantine_name,
        )

    def cleanup(self) -> None:
        if self.published or self.quarantine_name is not None:
            return
        self._require_held_parent_guard()
        held_staging = os.fstat(self.staging_fd)
        if (
            not stat.S_ISDIR(held_staging.st_mode)
            or _identity(held_staging) != self.staging_identity
        ):
            raise IdentityContractError(
                "held publication staging identity changed; retained in place"
            )
        try:
            parent_entry = os.stat(
                self.staging_name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as error:
            raise IdentityContractError(
                "publication staging disappeared; held directory retained"
            ) from error
        if (
            not stat.S_ISDIR(parent_entry.st_mode)
            or _identity(parent_entry) != self.staging_identity
        ):
            raise IdentityContractError(
                "publication staging identity changed; replacement retained in place"
            )
        for _attempt in range(128):
            quarantine_name = (
                f".{self.output_root.name}.{secrets.token_hex(12)}.quarantine"
            )
            try:
                self._quarantine_no_replace(quarantine_name)
            except OSError as error:
                if error.errno == errno.EEXIST:
                    continue
                raise IdentityContractError(
                    "publication staging quarantine rename failed; retained"
                ) from error
            # The rename happened.  Record that fact before verification so a
            # verification/fsync failure can never trigger another cleanup.
            self.quarantine_name = quarantine_name
            break
        else:
            raise IdentityContractError(
                "could not allocate no-replace quarantine; staging retained"
            )
        moved_entry = os.stat(
            self.quarantine_name,
            dir_fd=self.parent_fd,
            follow_symlinks=False,
        )
        held_after = os.fstat(self.staging_fd)
        if (
            not stat.S_ISDIR(moved_entry.st_mode)
            or _identity(moved_entry) != self.staging_identity
            or _identity(held_after) != self.staging_identity
        ):
            raise IdentityContractError(
                "quarantine moved a replacement entry; all entries retained"
            )
        try:
            os.stat(
                self.staging_name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise IdentityContractError(
                "publication staging name repopulated after quarantine; retained"
            )
        os.fsync(self.parent_fd)

    def close(self) -> None:
        first_error = None
        for attribute in ("evidence_fd", "staging_fd", "parent_fd"):
            descriptor = getattr(self, attribute)
            if descriptor < 0:
                continue
            # Mark the descriptor consumed before close.  A failing close must
            # not make a finally block retry a descriptor number that the
            # kernel may already have released and reassigned.
            setattr(self, attribute, -1)
            try:
                os.close(descriptor)
            except OSError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def open_secure_publication(output_root: Path) -> SecurePublication:
    output_root = Path(output_root).absolute()
    if not output_root.name or output_root.name in {".", ".."}:
        raise IdentityContractError("output root name is unsafe")
    lexical_parent = output_root.parent
    try:
        physical_parent = lexical_parent.resolve(strict=True)
        parent_path_stat = os.stat(physical_parent, follow_symlinks=False)
    except OSError as error:
        raise IdentityContractError("output parent is missing") from error
    if not stat.S_ISDIR(parent_path_stat.st_mode):
        raise IdentityContractError("output parent is not a directory")
    _require_safe_parent_permissions(parent_path_stat)
    parent_fd = os.open(physical_parent, _directory_flags())
    staging_fd = evidence_fd = None
    staging_name = None
    staging_identity = None
    publication = None
    try:
        parent_guard = _parent_guard(parent_path_stat)
        held_parent = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(held_parent.st_mode)
            or _parent_guard(held_parent) != parent_guard
        ):
            raise IdentityContractError("output parent changed while opened")
        _require_safe_parent_permissions(held_parent)
        try:
            os.stat(
                output_root.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise IdentityContractError(
                f"refusing to replace output root: {output_root}"
            )
        for _attempt in range(128):
            candidate = f".{output_root.name}.{secrets.token_hex(12)}.staging"
            try:
                os.mkdir(candidate, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            staging_name = candidate
            break
        if staging_name is None:
            raise IdentityContractError("could not allocate private staging")
        staging_fd = os.open(staging_name, _directory_flags(), dir_fd=parent_fd)
        staging_identity = _identity(os.fstat(staging_fd))
        staging_entry = os.stat(
            staging_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if _identity(staging_entry) != staging_identity:
            raise IdentityContractError("staging identity changed while opened")
        os.mkdir("evidence", 0o700, dir_fd=staging_fd)
        evidence_fd = os.open("evidence", _directory_flags(), dir_fd=staging_fd)
        evidence_identity = _identity(os.fstat(evidence_fd))
        evidence_entry = os.stat(
            "evidence",
            dir_fd=staging_fd,
            follow_symlinks=False,
        )
        if _identity(evidence_entry) != evidence_identity:
            raise IdentityContractError("evidence identity changed while opened")
        publication = SecurePublication(
            output_root=output_root,
            lexical_parent=lexical_parent,
            physical_parent=physical_parent,
            parent_fd=parent_fd,
            parent_guard=parent_guard,
            staging_name=staging_name,
            staging_fd=staging_fd,
            staging_identity=staging_identity,
            evidence_fd=evidence_fd,
            evidence_identity=evidence_identity,
            root_identities={},
            evidence_identities={},
        )
        publication._require_parent_path_matches_fd()
        publication._require_proc_fd_paths()
        return publication
    except Exception as error:
        cleanup_error = None
        if publication is not None:
            try:
                publication.cleanup()
            except Exception as current_cleanup_error:
                cleanup_error = current_cleanup_error
        elif staging_name is not None and staging_fd is not None:
            try:
                held_staging = os.fstat(staging_fd)
                parent_entry = os.stat(
                    staging_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    staging_identity is None
                    or not stat.S_ISDIR(parent_entry.st_mode)
                    or _identity(held_staging) != staging_identity
                    or _identity(parent_entry) != staging_identity
                ):
                    raise IdentityContractError(
                        "partially initialized staging identity changed; retained"
                    )
                for _attempt in range(128):
                    quarantine_name = (
                        f".{output_root.name}.{secrets.token_hex(12)}.quarantine"
                    )
                    try:
                        _renameat2_noreplace(
                            parent_fd,
                            staging_name,
                            parent_fd,
                            quarantine_name,
                        )
                    except OSError as rename_error:
                        if rename_error.errno == errno.EEXIST:
                            continue
                        raise
                    break
                else:
                    raise IdentityContractError(
                        "could not allocate early publication quarantine"
                    )
                moved_entry = os.stat(
                    quarantine_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(moved_entry.st_mode)
                    or _identity(moved_entry) != staging_identity
                    or _identity(os.fstat(staging_fd)) != staging_identity
                ):
                    raise IdentityContractError(
                        "early quarantine moved a replacement; retained"
                    )
                os.fsync(parent_fd)
            except Exception as current_cleanup_error:
                cleanup_error = current_cleanup_error
        elif staging_name is not None:
            cleanup_error = IdentityContractError(
                "unbound partially initialized staging retained in place"
            )
        if evidence_fd is not None:
            os.close(evidence_fd)
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)
        if cleanup_error is not None:
            raise IdentityContractError(
                f"publication initialization failed; staging retained: {cleanup_error}"
            ) from error
        raise


def require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise IdentityContractError(
            f"{label} fields changed: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise IdentityContractError(f"{label} must be a lowercase SHA-256")
    return value


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise IdentityContractError(f"{label} must be a boolean")
    return value


def require_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise IdentityContractError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def require_number(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise IdentityContractError(
            f"{label} must be finite and in [{minimum}, {maximum}]"
        )
    return float(value)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise IdentityContractError(f"{label} must be a non-empty trimmed string")
    return value


def _validate_indices(value: Any, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in value
        )
    ):
        raise IdentityContractError(f"{label} must contain non-negative integers")
    result = tuple(value)
    if tuple(sorted(set(result))) != result:
        raise IdentityContractError(f"{label} must be sorted and duplicate-free")
    return result


def _validate_color(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise IdentityContractError(f"{label} must be an RGBA array")
    return tuple(
        require_number(item, f"{label}[{index}]", minimum=0.0, maximum=1.0)
        for index, item in enumerate(value)
    )


def srgb_rgba8(value: Sequence[float]) -> tuple[int, int, int, int]:
    if len(value) != 4 or any(
        not math.isfinite(float(channel)) or not 0.0 <= float(channel) <= 1.0
        for channel in value
    ):
        raise IdentityContractError("sRGB colour must contain four channels in [0, 1]")
    return tuple(
        max(0, min(255, int(math.floor(float(channel) * 255.0 + 0.5))))
        for channel in value
    )


def png_rgba8_code_values(payload: bytes) -> set[tuple[int, int, int, int]]:
    """Return decoded RGBA8 codes without applying a colour transfer curve."""

    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise IdentityContractError("coat texture is not a PNG")
    width = height = None
    compressed = bytearray()
    offset = 8
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise IdentityContractError("coat PNG chunk is truncated")
        size = struct.unpack_from(">I", payload, offset)[0]
        chunk_type = payload[offset + 4 : offset + 8]
        start = offset + 8
        end = start + size
        if end + 4 > len(payload):
            raise IdentityContractError("coat PNG chunk payload is truncated")
        chunk = payload[start:end]
        expected_crc = struct.unpack_from(">I", payload, end)[0]
        if zlib.crc32(chunk_type + chunk) & 0xFFFFFFFF != expected_crc:
            raise IdentityContractError("coat PNG chunk CRC is invalid")
        if chunk_type == b"IHDR":
            if len(chunk) != 13:
                raise IdentityContractError("coat PNG IHDR is invalid")
            width, height, depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", chunk)
            )
            if (
                width <= 0
                or height <= 0
                or depth != 8
                or color_type != 6
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                raise IdentityContractError("coat PNG must be non-interlaced RGBA8")
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break
        offset = end + 4
    if width is None or height is None or not compressed:
        raise IdentityContractError("coat PNG is missing IHDR or IDAT")
    try:
        scanlines = zlib.decompress(bytes(compressed))
    except zlib.error as error:
        raise IdentityContractError("coat PNG IDAT is invalid") from error
    row_size = width * 4
    if len(scanlines) != height * (row_size + 1):
        raise IdentityContractError("coat PNG scanline size changed")
    previous = bytearray(row_size)
    codes: set[tuple[int, int, int, int]] = set()
    for row_index in range(height):
        row_start = row_index * (row_size + 1)
        filter_type = scanlines[row_start]
        encoded = scanlines[row_start + 1 : row_start + 1 + row_size]
        decoded = bytearray(row_size)
        for index, raw in enumerate(encoded):
            left = decoded[index - 4] if index >= 4 else 0
            above = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                estimate = left + above - upper_left
                left_delta = abs(estimate - left)
                above_delta = abs(estimate - above)
                upper_left_delta = abs(estimate - upper_left)
                predictor = (
                    left
                    if left_delta <= above_delta and left_delta <= upper_left_delta
                    else above
                    if above_delta <= upper_left_delta
                    else upper_left
                )
            else:
                raise IdentityContractError("coat PNG uses an unknown row filter")
            decoded[index] = (raw + predictor) & 0xFF
        codes.update(
            tuple(decoded[index : index + 4]) for index in range(0, row_size, 4)
        )
        previous = decoded
    return codes


@dataclass(frozen=True)
class EarSide:
    side: str
    indices: tuple[int, ...]
    pivot_y: float
    pivot_z: float
    rotation_degrees: float


@dataclass(frozen=True)
class CoatPredicate:
    axis: str
    operator: str
    value: float


@dataclass(frozen=True)
class CoatRegion:
    region_id: str
    predicates: tuple[CoatPredicate, ...]


@dataclass(frozen=True)
class IdentityPlan:
    raw: Mapping[str, Any]
    source_sha256: str
    source_size_bytes: int
    primary_mesh: str
    armature: str
    motion_root_bone: str
    expected_actions: tuple[str, ...]
    removal_object: str
    removal_vertices: int
    removal_faces: int
    removal_custom_shape_assignments: int
    positive: EarSide
    negative: EarSide
    symmetry_plane_y: float
    symmetry_tolerance: float
    maximum_moved_vertex_fraction: float
    maximum_displacement_diagonal_ratio: float
    roundtrip_canonical_precision_decimals: int
    roundtrip_animation_sample_phases: tuple[float, ...]
    maximum_bind_matrix_semantic_delta: float
    maximum_sampled_skinned_world_delta: float
    maximum_animation_duration_delta_frames: float
    uv_resolution: int
    uv_angle_limit: float
    uv_island_margin: float
    base_color: tuple[float, float, float, float]
    white_color: tuple[float, float, float, float]
    coat_regions: tuple[CoatRegion, ...]


def _validate_side(value: Any, label: str, expected_side: str) -> EarSide:
    if not isinstance(value, Mapping):
        raise IdentityContractError(f"{label} must be an object")
    require_exact_keys(
        value,
        frozenset(
            {
                "side",
                "vertex_indices",
                "pivot_y",
                "pivot_z",
                "rotation_degrees",
            }
        ),
        label,
    )
    if value["side"] != expected_side:
        raise IdentityContractError(f"{label}.side must be {expected_side}")
    rotation = require_number(
        value["rotation_degrees"],
        f"{label}.rotation_degrees",
        minimum=-85.0,
        maximum=85.0,
    )
    if abs(rotation) < 5.0:
        raise IdentityContractError(f"{label} rotation is not a meaningful morph")
    return EarSide(
        side=expected_side,
        indices=_validate_indices(value["vertex_indices"], f"{label}.vertex_indices"),
        pivot_y=require_number(
            value["pivot_y"],
            f"{label}.pivot_y",
            minimum=-1000.0,
            maximum=1000.0,
        ),
        pivot_z=require_number(
            value["pivot_z"],
            f"{label}.pivot_z",
            minimum=-1000.0,
            maximum=1000.0,
        ),
        rotation_degrees=rotation,
    )


def _validate_regions(value: Any) -> tuple[CoatRegion, ...]:
    if not isinstance(value, list) or not value:
        raise IdentityContractError("coat.white_regions must be a non-empty list")
    regions = []
    identifiers = set()
    for index, item in enumerate(value):
        label = f"coat.white_regions[{index}]"
        if not isinstance(item, Mapping):
            raise IdentityContractError(f"{label} must be an object")
        require_exact_keys(item, frozenset({"id", "all"}), label)
        identifier = _nonempty_string(item["id"], f"{label}.id")
        if identifier in identifiers:
            raise IdentityContractError("coat region identifiers must be unique")
        identifiers.add(identifier)
        predicates_value = item["all"]
        if not isinstance(predicates_value, list) or not predicates_value:
            raise IdentityContractError(f"{label}.all must be a non-empty list")
        predicates = []
        axes = set()
        for predicate_index, predicate in enumerate(predicates_value):
            predicate_label = f"{label}.all[{predicate_index}]"
            if not isinstance(predicate, Mapping):
                raise IdentityContractError(f"{predicate_label} must be an object")
            require_exact_keys(
                predicate,
                frozenset({"axis", "op", "value"}),
                predicate_label,
            )
            axis = predicate["axis"]
            operator = predicate["op"]
            if axis not in AXES or operator not in OPERATORS:
                raise IdentityContractError(
                    f"{predicate_label} has an unsupported predicate"
                )
            if (axis, operator) in axes:
                raise IdentityContractError(
                    f"{label} repeats the same axis/operator predicate"
                )
            axes.add((axis, operator))
            predicates.append(
                CoatPredicate(
                    axis=axis,
                    operator=operator,
                    value=require_number(
                        predicate["value"],
                        f"{predicate_label}.value",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                )
            )
        regions.append(CoatRegion(identifier, tuple(predicates)))
    return tuple(regions)


def load_plan(value: Any) -> IdentityPlan:
    """Validate a plan without treating it as permission to execute."""

    if not isinstance(value, Mapping):
        raise IdentityContractError("identity plan must be an object")
    require_exact_keys(
        value,
        frozenset(
            {
                "schema",
                "state_classification",
                "formal_dataset_registration_authorized",
                "execution_authorized",
                "source",
                "scene_contract",
                "ear_morph",
                "uv",
                "coat",
                "gates",
                "authority",
            }
        ),
        "identity plan",
    )
    if (
        value["schema"] != PLAN_SCHEMA
        or value["state_classification"] != "research_candidate"
        or require_bool(
            value["formal_dataset_registration_authorized"],
            "formal_dataset_registration_authorized",
        )
        is not False
        or require_bool(value["execution_authorized"], "execution_authorized")
        is not False
    ):
        raise IdentityContractError(
            "a plan must remain a non-executable research candidate"
        )

    source = value["source"]
    if not isinstance(source, Mapping):
        raise IdentityContractError("source must be an object")
    require_exact_keys(source, frozenset({"sha256", "size_bytes"}), "source")
    source_sha = require_sha256(source["sha256"], "source.sha256")
    source_size = require_int(
        source["size_bytes"],
        "source.size_bytes",
        minimum=1,
        maximum=2**63 - 1,
    )

    scene = value["scene_contract"]
    if not isinstance(scene, Mapping):
        raise IdentityContractError("scene_contract must be an object")
    require_exact_keys(
        scene,
        frozenset(
            {
                "primary_mesh",
                "armature",
                "motion_root_bone",
                "expected_actions",
                "remove_unskinned_object",
            }
        ),
        "scene_contract",
    )
    actions_value = scene["expected_actions"]
    if (
        not isinstance(actions_value, list)
        or len(actions_value) != 2
        or any(not isinstance(item, str) or not item for item in actions_value)
        or len(set(actions_value)) != 2
    ):
        raise IdentityContractError(
            "scene_contract.expected_actions must contain two unique names"
        )
    removal = scene["remove_unskinned_object"]
    if not isinstance(removal, Mapping):
        raise IdentityContractError("remove_unskinned_object must be an object")
    require_exact_keys(
        removal,
        frozenset(
            {
                "name",
                "type",
                "vertex_count",
                "face_count",
                "must_be_unskinned",
                "must_have_no_materials",
                "must_be_only_pose_bone_custom_shape",
                "expected_pose_bone_custom_shape_assignments",
            }
        ),
        "remove_unskinned_object",
    )
    if (
        removal["type"] != "MESH"
        or removal["must_be_unskinned"] is not True
        or removal["must_have_no_materials"] is not True
        or removal["must_be_only_pose_bone_custom_shape"] is not True
    ):
        raise IdentityContractError("removal target guards cannot be weakened")

    morph = value["ear_morph"]
    if not isinstance(morph, Mapping):
        raise IdentityContractError("ear_morph must be an object")
    require_exact_keys(
        morph,
        frozenset(
            {
                "coordinate_space",
                "symmetry_axis",
                "symmetry_plane_y",
                "symmetry_tolerance",
                "positive_y",
                "negative_y",
            }
        ),
        "ear_morph",
    )
    if morph["coordinate_space"] != "mesh_local" or morph["symmetry_axis"] != "Y":
        raise IdentityContractError("ear morph must use mesh-local Y symmetry")
    positive = _validate_side(morph["positive_y"], "positive_y", "positive_y")
    negative = _validate_side(morph["negative_y"], "negative_y", "negative_y")
    if set(positive.indices) & set(negative.indices):
        raise IdentityContractError("ear vertex masks must be disjoint")
    if not math.isclose(
        positive.rotation_degrees,
        -negative.rotation_degrees,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise IdentityContractError("ear rotations must be equal and opposite")
    plane_y = require_number(
        morph["symmetry_plane_y"],
        "ear_morph.symmetry_plane_y",
        minimum=-1000.0,
        maximum=1000.0,
    )
    if not math.isclose(
        0.5 * (positive.pivot_y + negative.pivot_y),
        plane_y,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ) or not math.isclose(
        positive.pivot_z,
        negative.pivot_z,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise IdentityContractError("ear pivots must mirror across the plan plane")

    uv = value["uv"]
    if not isinstance(uv, Mapping):
        raise IdentityContractError("uv must be an object")
    require_exact_keys(
        uv,
        frozenset({"method", "resolution", "angle_limit_radians", "island_margin"}),
        "uv",
    )
    if uv["method"] != "blender_smart_project_v1":
        raise IdentityContractError("unsupported deterministic UV method")
    resolution = require_int(
        uv["resolution"],
        "uv.resolution",
        minimum=256,
        maximum=2048,
    )
    if resolution & (resolution - 1):
        raise IdentityContractError("uv.resolution must be a power of two")

    coat = value["coat"]
    if not isinstance(coat, Mapping):
        raise IdentityContractError("coat must be an object")
    require_exact_keys(
        coat,
        frozenset(
            {
                "method",
                "base_color_srgb",
                "white_color_srgb",
                "white_regions",
            }
        ),
        "coat",
    )
    if coat["method"] != "normalized_face_region_texture_v1":
        raise IdentityContractError("unsupported deterministic coat method")

    gates = value["gates"]
    if not isinstance(gates, Mapping):
        raise IdentityContractError("gates must be an object")
    require_exact_keys(
        gates,
        frozenset(
            {
                "maximum_moved_vertex_fraction",
                "maximum_displacement_bbox_diagonal_ratio",
                "require_all_duplicate_position_vertices",
                "require_exact_topology_in_memory",
                "require_exact_weights_in_memory",
                "require_exact_skeleton_in_memory",
                "require_exact_actions_in_memory",
                "require_embedded_texture_readback",
                "roundtrip_canonical_precision_decimals",
                "roundtrip_animation_sample_phases",
                "maximum_bind_matrix_semantic_delta",
                "maximum_sampled_skinned_world_delta",
                "maximum_animation_duration_delta_frames",
            }
        ),
        "gates",
    )
    if any(
        gates[name] is not True
        for name in (
            "require_all_duplicate_position_vertices",
            "require_exact_topology_in_memory",
            "require_exact_weights_in_memory",
            "require_exact_skeleton_in_memory",
            "require_exact_actions_in_memory",
            "require_embedded_texture_readback",
        )
    ):
        raise IdentityContractError("identity invariant gates cannot be weakened")
    phases_value = gates["roundtrip_animation_sample_phases"]
    if not isinstance(phases_value, list):
        raise IdentityContractError(
            "gates.roundtrip_animation_sample_phases must be an array"
        )
    sample_phases = tuple(
        require_number(
            phase,
            f"gates.roundtrip_animation_sample_phases[{index}]",
            minimum=0.0,
            maximum=1.0,
        )
        for index, phase in enumerate(phases_value)
    )
    required_dense_phases = tuple(
        index / DENSE_ANIMATION_PHASE_INTERVALS
        for index in range(DENSE_ANIMATION_PHASE_INTERVALS + 1)
    )
    if sample_phases != required_dense_phases:
        raise IdentityContractError(
            "roundtrip animation phases must be the complete uniform 81-phase "
            "grid from 0 through 1"
        )

    authority = value["authority"]
    if not isinstance(authority, Mapping):
        raise IdentityContractError("authority must be an object")
    require_exact_keys(
        authority,
        frozenset(
            {
                "source_mesh_is_geometry_authority",
                "only_explicit_ear_vertices_may_move",
                "coat_may_not_change_geometry",
                "machine_execution_authorization_required",
                "user_approval_inferred",
            }
        ),
        "authority",
    )
    if authority != {
        "source_mesh_is_geometry_authority": True,
        "only_explicit_ear_vertices_may_move": True,
        "coat_may_not_change_geometry": True,
        "machine_execution_authorization_required": True,
        "user_approval_inferred": False,
    }:
        raise IdentityContractError("identity authority boundary changed")

    return IdentityPlan(
        raw=value,
        source_sha256=source_sha,
        source_size_bytes=source_size,
        primary_mesh=_nonempty_string(scene["primary_mesh"], "primary_mesh"),
        armature=_nonempty_string(scene["armature"], "armature"),
        motion_root_bone=_nonempty_string(
            scene["motion_root_bone"],
            "motion_root_bone",
        ),
        expected_actions=tuple(actions_value),
        removal_object=_nonempty_string(removal["name"], "removal.name"),
        removal_vertices=require_int(
            removal["vertex_count"],
            "removal.vertex_count",
            minimum=1,
            maximum=10_000_000,
        ),
        removal_faces=require_int(
            removal["face_count"],
            "removal.face_count",
            minimum=1,
            maximum=10_000_000,
        ),
        removal_custom_shape_assignments=require_int(
            removal["expected_pose_bone_custom_shape_assignments"],
            "removal.expected_pose_bone_custom_shape_assignments",
            minimum=1,
            maximum=10_000,
        ),
        positive=positive,
        negative=negative,
        symmetry_plane_y=plane_y,
        symmetry_tolerance=require_number(
            morph["symmetry_tolerance"],
            "ear_morph.symmetry_tolerance",
            minimum=1.0e-9,
            maximum=1.0e-3,
        ),
        maximum_moved_vertex_fraction=require_number(
            gates["maximum_moved_vertex_fraction"],
            "gates.maximum_moved_vertex_fraction",
            minimum=1.0e-6,
            maximum=0.15,
        ),
        maximum_displacement_diagonal_ratio=require_number(
            gates["maximum_displacement_bbox_diagonal_ratio"],
            "gates.maximum_displacement_bbox_diagonal_ratio",
            minimum=1.0e-6,
            maximum=0.12,
        ),
        roundtrip_canonical_precision_decimals=require_int(
            gates["roundtrip_canonical_precision_decimals"],
            "gates.roundtrip_canonical_precision_decimals",
            minimum=3,
            maximum=6,
        ),
        roundtrip_animation_sample_phases=sample_phases,
        maximum_bind_matrix_semantic_delta=require_number(
            gates["maximum_bind_matrix_semantic_delta"],
            "gates.maximum_bind_matrix_semantic_delta",
            minimum=1.0e-7,
            maximum=0.01,
        ),
        maximum_sampled_skinned_world_delta=require_number(
            gates["maximum_sampled_skinned_world_delta"],
            "gates.maximum_sampled_skinned_world_delta",
            minimum=1.0e-7,
            maximum=0.01,
        ),
        maximum_animation_duration_delta_frames=require_number(
            gates["maximum_animation_duration_delta_frames"],
            "gates.maximum_animation_duration_delta_frames",
            minimum=1.0e-7,
            maximum=0.01,
        ),
        uv_resolution=resolution,
        uv_angle_limit=require_number(
            uv["angle_limit_radians"],
            "uv.angle_limit_radians",
            minimum=0.1,
            maximum=math.pi,
        ),
        uv_island_margin=require_number(
            uv["island_margin"],
            "uv.island_margin",
            minimum=0.001,
            maximum=0.1,
        ),
        base_color=_validate_color(coat["base_color_srgb"], "base_color_srgb"),
        white_color=_validate_color(coat["white_color_srgb"], "white_color_srgb"),
        coat_regions=_validate_regions(coat["white_regions"]),
    )


def rotate_ear_point(
    point: Sequence[float],
    side: EarSide,
) -> tuple[float, float, float]:
    """Rotate one explicitly selected point around a mesh-local X pivot line."""

    if len(point) != 3 or any(not math.isfinite(float(item)) for item in point):
        raise IdentityContractError("ear point must contain three finite numbers")
    radians = math.radians(side.rotation_degrees)
    delta_y = float(point[1]) - side.pivot_y
    delta_z = float(point[2]) - side.pivot_z
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return (
        float(point[0]),
        side.pivot_y + cosine * delta_y - sine * delta_z,
        side.pivot_z + sine * delta_y + cosine * delta_z,
    )


def normalized_point(
    point: Sequence[float],
    minimum: Sequence[float],
    maximum: Sequence[float],
    symmetry_plane_y: float,
) -> dict[str, float]:
    if len(point) != 3 or len(minimum) != 3 or len(maximum) != 3:
        raise IdentityContractError("normalization requires 3D points")
    values = {}
    for index, axis in enumerate(("x", "y", "z")):
        extent = float(maximum[index]) - float(minimum[index])
        if not math.isfinite(extent) or extent <= 0.0:
            raise IdentityContractError("normalization bounds are degenerate")
        values[axis] = (float(point[index]) - float(minimum[index])) / extent
    half_lateral_extent = max(
        abs(float(minimum[1]) - symmetry_plane_y),
        abs(float(maximum[1]) - symmetry_plane_y),
    )
    if half_lateral_extent <= 0.0:
        raise IdentityContractError("lateral normalization bounds are degenerate")
    values["lateral_abs"] = (
        abs(float(point[1]) - symmetry_plane_y) / half_lateral_extent
    )
    return values


def coat_role(
    point: Sequence[float],
    minimum: Sequence[float],
    maximum: Sequence[float],
    symmetry_plane_y: float,
    regions: Sequence[CoatRegion],
) -> str:
    values = normalized_point(point, minimum, maximum, symmetry_plane_y)
    for region in regions:
        matches = True
        for predicate in region.predicates:
            observed = values[predicate.axis]
            if predicate.operator == "gte":
                matches = matches and observed >= predicate.value
            else:
                matches = matches and observed <= predicate.value
        if matches:
            return "white"
    return "base"


def validate_execution_authorization(
    value: Any,
    *,
    expected_preflight_sha256: str,
    expected_plan_sha256: str,
    expected_source_sha256: str,
) -> None:
    """Require separately hashed, task-scoped machine authorization."""

    if not isinstance(value, Mapping):
        raise IdentityContractError("execution authorization must be an object")
    require_exact_keys(
        value,
        frozenset(
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
            }
        ),
        "execution authorization",
    )
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["decision"] != "authorized_for_new_bounded_research_candidate_output"
        or value["authorization_basis"] != "scoped_task_continue_asset_identity_repair"
        or value["authorized_actor_type"] != "codex_task_agent"
        or value["authorized_output_scope"]
        != "new_research_candidate_sealed_at_publication_only"
        or value["state_classification"] != "research_candidate"
        or value["formal_dataset_registration_authorized"] is not False
        or value["user_approval_inferred"] is not False
        or value["preflight_receipt_sha256"]
        != require_sha256(expected_preflight_sha256, "expected preflight SHA-256")
        or value["plan_sha256"]
        != require_sha256(expected_plan_sha256, "expected plan SHA-256")
        or value["source_sha256"]
        != require_sha256(expected_source_sha256, "expected source SHA-256")
    ):
        raise IdentityContractError(
            "execution authorization authority binding is invalid"
        )
    checks = value["checks"]
    if (
        not isinstance(checks, Mapping)
        or set(checks) != AUTHORIZATION_CHECKS
        or any(item is not True for item in checks.values())
    ):
        raise IdentityContractError(
            "execution authorization requires every explicit mask/removal/coat check"
        )
    _nonempty_string(value["notes"], "execution authorization notes")
