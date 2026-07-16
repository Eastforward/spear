"""Seal the approved Rocketbox neutral-walk review as an immutable baseline."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

from rocketbox_motion_review import EXPECTED_ASSET_IDS, assert_pair_approved


BASELINE_ID = "rocketbox_neutral_walk_v1"
BASELINE_SCHEMA = "rocketbox_baseline_manifest_v1"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
BASELINE_FILES = (
    "retarget.blend",
    "retarget.glb",
    "retarget_metrics.json",
    "retarget_manifest.json",
    "motion_review.json",
    "front.mp4",
    "side.mp4",
    "top.mp4",
    "joints.mp4",
    "feet.mp4",
    "source_target.mp4",
    "contact_sheet.png",
)


class BaselineSealError(ValueError):
    """Raised when a baseline cannot be created or is not identical."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, root: Path, description: str) -> Path:
    path = Path(path).absolute()
    root = Path(root).absolute()
    if not os.path.lexists(path):
        raise BaselineSealError(f"{description} is missing: {path}")
    resolved = path.resolve()
    if path != resolved or resolved.parent != root:
        raise BaselineSealError(
            f"{description} must be a regular file directly under its review directory"
        )
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise BaselineSealError(f"{description} must be a regular file")
    return path


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def _collect_manifest(review_root: Path) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for asset_id in EXPECTED_ASSET_IDS:
        review_dir = review_root / asset_id
        resolved_review_dir = review_dir.resolve()
        if review_dir.absolute() != resolved_review_dir:
            raise BaselineSealError(f"{asset_id} review directory is a symlink")
        if not resolved_review_dir.is_dir():
            raise BaselineSealError(f"{asset_id} review directory is missing")
        files = {}
        for filename in BASELINE_FILES:
            path = _regular_file(
                resolved_review_dir / filename,
                resolved_review_dir,
                f"{asset_id}/{filename}",
            )
            files[filename] = _file_record(path)
        assets[asset_id] = {"files": files}
    return {
        "schema_version": BASELINE_SCHEMA,
        "baseline_id": BASELINE_ID,
        "motion": "walk_neutral",
        "artifact_allowlist": list(BASELINE_FILES),
        "assets": assets,
    }


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_fsync(path: Path, content: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_and_verify(source: Path, destination: Path, expected: dict[str, Any]) -> None:
    with source.open("rb") as source_stream, destination.open("xb") as destination_stream:
        while True:
            chunk = source_stream.read(1024 * 1024)
            if not chunk:
                break
            destination_stream.write(chunk)
        destination_stream.flush()
        os.fsync(destination_stream.fileno())
    actual = _file_record(destination)
    if actual != expected:
        raise BaselineSealError(f"copied bytes differ for {source.name}")


def _copy_to_temporary(review_root: Path, temporary_root: Path, manifest: dict[str, Any]) -> None:
    for asset_id in EXPECTED_ASSET_IDS:
        source_dir = review_root / asset_id
        destination_dir = temporary_root / asset_id
        destination_dir.mkdir()
        for filename in BASELINE_FILES:
            source = _regular_file(source_dir / filename, source_dir, f"{asset_id}/{filename}")
            destination = destination_dir / filename
            _copy_and_verify(source, destination, manifest["assets"][asset_id]["files"][filename])
        _fsync_directory(destination_dir)
    _write_fsync(temporary_root / "baseline_manifest.json", _manifest_bytes(manifest))
    _fsync_directory(temporary_root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise BaselineSealError("atomic no-replace publication is unavailable") from error
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
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise BaselineSealError(
            f"baseline destination already exists after concurrent publication: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _assert_existing_identical(output_root: Path, manifest: dict[str, Any]) -> None:
    if not output_root.is_dir() or output_root.is_symlink():
        raise BaselineSealError(f"existing baseline root is not a regular directory: {output_root}")
    manifest_path = output_root / "baseline_manifest.json"
    _regular_file(manifest_path, output_root, "baseline manifest")
    if manifest_path.read_bytes() != _manifest_bytes(manifest):
        raise BaselineSealError("existing baseline manifest differs")

    expected_names = set(EXPECTED_ASSET_IDS) | {"baseline_manifest.json"}
    actual_names = {path.name for path in output_root.iterdir()}
    if actual_names != expected_names:
        raise BaselineSealError("existing baseline artifact set differs")
    for asset_id in EXPECTED_ASSET_IDS:
        asset_dir = output_root / asset_id
        if asset_dir.is_symlink() or not asset_dir.is_dir():
            raise BaselineSealError(f"existing {asset_id} directory differs")
        expected_files = manifest["assets"][asset_id]["files"]
        if {path.name for path in asset_dir.iterdir()} != set(BASELINE_FILES):
            raise BaselineSealError(f"existing {asset_id} artifact set differs")
        for filename in BASELINE_FILES:
            copied = _regular_file(asset_dir / filename, asset_dir, f"existing {asset_id}/{filename}")
            if _file_record(copied) != expected_files[filename]:
                raise BaselineSealError(f"existing {asset_id}/{filename} bytes differ")


def seal_baseline(review_root: Path, output_root: Path) -> dict:
    """Seal both currently approved neutral-walk reviews into ``output_root``."""
    review_root = Path(review_root).absolute()
    output_root = Path(output_root).absolute()
    approvals = assert_pair_approved(review_root)
    if set(approvals) != set(EXPECTED_ASSET_IDS):
        raise BaselineSealError("exact male/female approval pair is required")
    manifest = _collect_manifest(review_root)

    if os.path.lexists(output_root):
        _assert_existing_identical(output_root, manifest)
        return manifest

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    try:
        _copy_to_temporary(review_root, temporary_root, manifest)
        current_approvals = assert_pair_approved(review_root)
        if set(current_approvals) != set(EXPECTED_ASSET_IDS):
            raise BaselineSealError("exact male/female approval pair is required")
        current_manifest = _collect_manifest(review_root)
        if current_manifest != manifest:
            raise BaselineSealError("source artifacts changed during staging")
        _rename_noreplace(temporary_root, output_root)
        temporary_root = None  # type: ignore[assignment]
        _fsync_directory(output_root.parent)
    except BaseException:
        if temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    seal_baseline(args.review_root, args.output_root)
    print("ROCKETBOX_BASELINE_SEALED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
