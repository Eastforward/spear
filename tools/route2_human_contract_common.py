#!/usr/bin/env python3
"""Small fail-closed filesystem primitives for Route-2 human contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, TypeVar


CANONICAL_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9_]{0,126}[a-z0-9])?")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TError = TypeVar("TError", bound=Exception)
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_directory_chain(
    path: Path, description: str, error_type: type[TError]
) -> int:
    candidate = absolute(path)
    if not candidate.is_absolute():  # pragma: no cover - absolute() guarantees this.
        raise error_type(f"{description} must be absolute")
    descriptor = -1
    try:
        descriptor = os.open("/", _DIRECTORY_FLAGS)
        for component in candidate.parts[1:]:
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise error_type(f"{description} contains a non-directory component")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise error_type(f"{description} is not a direct real directory: {candidate}: {error}") from error


def read_file_snapshot(
    path: Path,
    *,
    root: Path,
    description: str,
    error_type: type[TError],
    require_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    root = absolute(root)
    candidate = absolute(path)
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise error_type(f"{description} is outside its authenticated root") from error
    if not relative.parts:
        raise error_type(f"{description} must name a file below its authenticated root")
    directory_fd = _open_directory_chain(root, f"{description} root", error_type)
    file_fd = -1
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child
        file_fd = os.open(relative.parts[-1], _READ_FLAGS, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise error_type(f"{description} must be a non-empty direct regular file")
        mode = stat.S_IMODE(before.st_mode)
        if require_mode is not None and mode != require_mode:
            raise error_type(f"{description} must have mode {require_mode:04o}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise error_type(f"{description} changed while it was being read")
        current_entry = os.stat(
            relative.parts[-1], dir_fd=directory_fd, follow_symlinks=False
        )
        if (current_entry.st_dev, current_entry.st_ino) != (before.st_dev, before.st_ino):
            raise error_type(f"{description} pathname changed while it was being read")
        data = b"".join(chunks)
        if len(data) != before.st_size:
            raise error_type(f"{description} size changed while it was being read")
        verification_fd = _open_directory_chain(
            candidate.parent, f"{description} parent", error_type
        )
        try:
            pinned_parent = os.fstat(directory_fd)
            current_parent = os.fstat(verification_fd)
            if (pinned_parent.st_dev, pinned_parent.st_ino) != (
                current_parent.st_dev,
                current_parent.st_ino,
            ):
                raise error_type(f"{description} authenticated parent changed during read")
        finally:
            os.close(verification_fd)
        final_entry = os.stat(
            relative.parts[-1], dir_fd=directory_fd, follow_symlinks=False
        )
        if any(
            getattr(after, name) != getattr(final_entry, name)
            for name in stable_fields
        ):
            raise error_type(f"{description} pathname changed after its authenticated read")
        return data, {
            "path": str(candidate),
            "relative_path": relative.as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": before.st_size,
            "mode": f"{mode:04o}",
        }
    except OSError as error:
        raise error_type(f"{description} could not be opened without following links: {error}") from error
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(directory_fd)


def sha256_file(path: Path) -> str:
    _, record = read_file_snapshot(
        path,
        root=absolute(path).parent,
        description="file",
        error_type=RuntimeError,
    )
    return record["sha256"]


def hash_file_snapshot(
    path: Path,
    *,
    root: Path,
    description: str,
    error_type: type[TError],
    require_mode: int | None = None,
) -> dict[str, Any]:
    """Stream and hash one authenticated regular-file inode without buffering it."""

    root = absolute(root)
    candidate = absolute(path)
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise error_type(f"{description} is outside its authenticated root") from error
    if not relative.parts:
        raise error_type(f"{description} must name a file below its authenticated root")
    directory_fd = _open_directory_chain(root, f"{description} root", error_type)
    file_fd = -1
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child
        file_fd = os.open(relative.parts[-1], _READ_FLAGS, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise error_type(f"{description} must be a non-empty direct regular file")
        mode = stat.S_IMODE(before.st_mode)
        if require_mode is not None and mode != require_mode:
            raise error_type(f"{description} must have mode {require_mode:04o}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(file_fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(file_fd)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise error_type(f"{description} changed while it was being hashed")
        if size != before.st_size:
            raise error_type(f"{description} size changed while it was being hashed")
        current = os.stat(
            relative.parts[-1], dir_fd=directory_fd, follow_symlinks=False
        )
        if any(getattr(after, name) != getattr(current, name) for name in stable_fields):
            raise error_type(f"{description} pathname changed during hashing")
        verification_fd = _open_directory_chain(
            candidate.parent, f"{description} parent", error_type
        )
        try:
            pinned_parent = os.fstat(directory_fd)
            current_parent = os.fstat(verification_fd)
            if (pinned_parent.st_dev, pinned_parent.st_ino) != (
                current_parent.st_dev,
                current_parent.st_ino,
            ):
                raise error_type(f"{description} authenticated parent changed")
        finally:
            os.close(verification_fd)
        final = os.stat(
            relative.parts[-1], dir_fd=directory_fd, follow_symlinks=False
        )
        if any(getattr(after, name) != getattr(final, name) for name in stable_fields):
            raise error_type(f"{description} pathname changed after hashing")
        return {
            "path": str(candidate),
            "relative_path": relative.as_posix(),
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
            "mode": f"{mode:04o}",
        }
    except OSError as error:
        raise error_type(
            f"{description} could not be opened without following links: {error}"
        ) from error
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(directory_fd)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def reject_user_approval(value: Any, error_type: type[TError], description: str) -> None:
    if "user_approved" in canonical_json(value):
        raise error_type(f"{description} may never claim user_approved")


def require_id(value: Any, field: str, error_type: type[TError]) -> str:
    if not isinstance(value, str) or CANONICAL_ID_RE.fullmatch(value) is None:
        raise error_type(f"{field} must be a canonical lower-case identifier")
    return value


def require_real_directory(
    path: Path,
    description: str,
    error_type: type[TError],
    *,
    mode: int | None = None,
) -> Path:
    candidate = absolute(path)
    descriptor = _open_directory_chain(candidate, description, error_type)
    try:
        current = os.fstat(descriptor)
        if mode is not None and stat.S_IMODE(current.st_mode) != mode:
            raise error_type(f"{description} must have mode {mode:04o}: {candidate}")
        supplied = os.stat(candidate, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (supplied.st_dev, supplied.st_ino):
            raise error_type(f"{description} path changed during validation: {candidate}")
    except OSError as error:
        raise error_type(f"{description} is not a direct real directory: {candidate}: {error}") from error
    finally:
        os.close(descriptor)
    return candidate


def require_contained_regular_file(
    path: Path,
    root: Path,
    description: str,
    error_type: type[TError],
    *,
    mode: int | None = None,
) -> Path:
    read_file_snapshot(
        path,
        root=root,
        description=description,
        error_type=error_type,
        require_mode=mode,
    )
    return absolute(path)


def require_relative_root(value: Any, error_type: type[TError]) -> str:
    if not isinstance(value, str) or not value:
        raise error_type("branch relative_root must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise error_type("branch relative_root must be canonical POSIX relative syntax")
    if value == ".":
        return value
    if any(part in {"", ".", ".."} for part in path.parts):
        raise error_type("branch relative_root may not contain dot or parent components")
    for part in path.parts:
        require_id(part, "branch relative_root component", error_type)
    return value


def load_json_mapping(
    path: Path, description: str, error_type: type[TError]
) -> dict[str, Any]:
    value, _ = load_json_mapping_record(
        path,
        root=absolute(path).parent,
        description=description,
        error_type=error_type,
    )
    return value


def load_json_mapping_record(
    path: Path,
    *,
    root: Path,
    description: str,
    error_type: type[TError],
    require_mode: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        data, record = read_file_snapshot(
            path,
            root=root,
            description=description,
            error_type=error_type,
            require_mode=require_mode,
        )
        value = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise error_type(f"{description} is not readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise error_type(f"{description} must contain a JSON object")
    return value, record


def load_glb_document_binary_record(
    path: Path,
    *,
    root: Path,
    description: str,
    error_type: type[TError],
    require_mode: int | None = None,
) -> tuple[dict[str, Any], bytes | None, dict[str, Any]]:
    data, record = read_file_snapshot(
        path,
        root=root,
        description=description,
        error_type=error_type,
        require_mode=require_mode,
    )
    if len(data) < 20 or data[:4] != b"glTF":
        raise error_type(f"{description} is not a GLB 2.0 file")
    version, declared_length = struct.unpack_from("<II", data, 4)
    if version != 2 or declared_length != len(data):
        raise error_type(f"{description} has an invalid GLB header")
    offset = 12
    json_chunks: list[bytes] = []
    binary_chunks: list[bytes] = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise error_type(f"{description} has a truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(data):
            raise error_type(f"{description} has a truncated GLB chunk payload")
        chunk = data[offset:end]
        offset = end
        if chunk_type == 0x4E4F534A:
            json_chunks.append(chunk)
        elif chunk_type == 0x004E4942:
            binary_chunks.append(chunk)
    if len(json_chunks) != 1 or len(binary_chunks) > 1:
        raise error_type(f"{description} has an invalid GLB chunk inventory")
    try:
        document = json.loads(
            json_chunks[0].rstrip(b" \t\r\n\x00").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise error_type(f"{description} has an invalid GLB JSON chunk: {error}") from error
    if not isinstance(document, dict):
        raise error_type(f"{description} GLB JSON root must be an object")
    binary = binary_chunks[0] if binary_chunks else None
    return document, binary, record


def load_glb_document_record(
    path: Path,
    *,
    root: Path,
    description: str,
    error_type: type[TError],
    require_mode: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, _, record = load_glb_document_binary_record(
        path,
        root=root,
        description=description,
        error_type=error_type,
        require_mode=require_mode,
    )
    return document, record


def file_record(
    path: Path,
    *,
    root: Path,
    description: str,
    error_type: type[TError],
    require_mode: int | None = None,
) -> dict[str, Any]:
    _, record = read_file_snapshot(
        path,
        root=root,
        description=description,
        error_type=error_type,
        require_mode=require_mode,
    )
    return record


def validate_file_record(
    record: Any,
    *,
    expected_path: Path,
    root: Path,
    description: str,
    error_type: type[TError],
    require_mode: int | None = None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise error_type(f"{description} descriptor is missing")
    expected = file_record(
        expected_path,
        root=root,
        description=description,
        error_type=error_type,
        require_mode=require_mode,
    )
    if dict(record) != expected:
        raise error_type(f"{description} descriptor or SHA-256 changed")
    return expected


def stable_mapping_snapshot(
    reader: Callable[[], dict[str, Any]],
    error_type: type[TError],
    description: str,
    *,
    maximum_attempts: int = 3,
) -> dict[str, Any]:
    if maximum_attempts <= 0:
        raise error_type(f"{description} stable snapshot attempt count is invalid")
    for _ in range(maximum_attempts):
        first = reader()
        second = reader()
        if canonical_json(first) == canonical_json(second):
            return second
    raise error_type(f"{description} did not reach a stable twice-identical snapshot")


def fsync_directory(path: Path) -> None:
    descriptor = _open_directory_chain(path, "fsync directory", RuntimeError)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_immutable_noreplace(
    destination: Path,
    value: Mapping[str, Any],
    error_type: type[TError],
    description: str,
    *,
    prelink_validator: Callable[[], None] | None = None,
    postlink_validator: Callable[[], None] | None = None,
) -> Path:
    destination = absolute(destination)
    parent = require_real_directory(destination.parent, f"{description} parent", error_type)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    parent_fd = _open_directory_chain(parent, f"{description} parent", error_type)
    temporary_name = f".{destination.name}.{uuid.uuid4().hex}.staging"
    temporary_fd = -1
    published = False
    try:
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise error_type(f"{description} already exists: {destination}")
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(encoded):
            offset += os.write(temporary_fd, encoded[offset:])
        os.fsync(temporary_fd)
        os.fchmod(temporary_fd, 0o444)
        os.fsync(temporary_fd)
        staged = os.fstat(temporary_fd)
        if prelink_validator is not None:
            prelink_validator()
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise error_type(f"{description} already exists: {destination}") from error
        published = True
        current = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != (staged.st_dev, staged.st_ino)
            or stat.S_IMODE(current.st_mode) != 0o444
            or current.st_size != len(encoded)
        ):
            raise error_type(f"{description} publication inode or mode changed")
        verification_fd = _open_directory_chain(parent, f"{description} parent", error_type)
        try:
            verification = os.fstat(verification_fd)
            pinned = os.fstat(parent_fd)
            if (verification.st_dev, verification.st_ino) != (pinned.st_dev, pinned.st_ino):
                os.unlink(destination.name, dir_fd=parent_fd)
                published = False
                raise error_type(f"{description} parent changed during publication")
        finally:
            os.close(verification_fd)
        validator_after_link = (
            postlink_validator
            if postlink_validator is not None
            else prelink_validator
        )
        if validator_after_link is not None:
            validator_after_link()
        os.fsync(parent_fd)
    except BaseException as error:
        if published:
            try:
                os.unlink(destination.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        if isinstance(error, error_type):
            raise
        if isinstance(error, OSError):
            raise error_type(f"{description} atomic publication failed: {error}") from error
        raise
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    return destination
