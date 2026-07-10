"""Verified official-source catalog helpers for Rocketbox avatar review."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen


_AVATAR_SPECS = (
    (
        "rocketbox_male_adult_01",
        "male",
        "Male_Adult_01",
        "m002",
    ),
    (
        "rocketbox_female_adult_01",
        "female",
        "Female_Adult_01",
        "f001",
    ),
)
_REQUIRED_TEXTURE_SUFFIXES = (
    "body_color",
    "body_normal",
    "body_specular",
    "head_color",
    "head_normal",
    "head_specular",
    "opacity_color",
)
_FEMALE_OPTIONAL_TEXTURE_SUFFIXES = ("head_normal_wrinkle",)
RAW_GITHUB_BASE = (
    "https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/"
)
_USER_AGENT = "AVEngine-Rocketbox-Source-Review/1.0"
_SOURCE_MANIFEST_FIELDS = (
    "role",
    "official_rel_path",
    "size",
    "official_git_blob_sha1",
)
_PINNED_REVIEW_AXES = {"forward_axis": "-Y", "up_axis": "+Z"}


def _expected_source_manifest_layout(
    avatar_dir: str, texture_prefix: str, texture_suffixes: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    avatar_root = f"Assets/Avatars/Adults/{avatar_dir}"
    return (
        ("fbx", f"{avatar_root}/Export/{avatar_dir}.fbx"),
        *(
            ("texture", f"{avatar_root}/Textures/{texture_prefix}_{suffix}.tga")
            for suffix in texture_suffixes
        ),
    )


_PINNED_SOURCE_MANIFESTS = {
    "rocketbox_male_adult_01": {
        "sha256": "a0fab2d505a426763ba66802b9d292a87856bf9a0a1d453c1279bea3630d6b62",
        "layout": _expected_source_manifest_layout(
            "Male_Adult_01", "m002", _REQUIRED_TEXTURE_SUFFIXES
        ),
    },
    "rocketbox_female_adult_01": {
        "sha256": "4973307a3cd444d8abeaf507287a33849b5b720a14f72b639e55040475929b24",
        "layout": _expected_source_manifest_layout(
            "Female_Adult_01",
            "f001",
            _REQUIRED_TEXTURE_SUFFIXES + _FEMALE_OPTIONAL_TEXTURE_SUFFIXES,
        ),
    },
}


class OfficialFileError(RuntimeError):
    """An on-disk file differs from its official Git tree record."""


class SourceReviewNotApproved(RuntimeError):
    """Raised when Rocketbox source review lacks required human approval."""


@dataclass(frozen=True)
class OfficialFile:
    rel_path: str
    size: int
    git_sha: str
    local_path: Path


@dataclass(frozen=True)
class RocketboxReviewAsset:
    asset_id: str
    gender: str
    avatar_dir: str
    texture_prefix: str
    fbx: OfficialFile
    textures: tuple[OfficialFile, ...]
    up_axis: str = "+Z"
    forward_axis: str = "-Y"
    missing_required_textures: tuple[str, ...] = ()


def _official_file_from_tree(
    entries: dict[str, dict[str, object]], rel_path: str, sample_root: Path
) -> OfficialFile:
    entry = entries[rel_path]
    return OfficialFile(
        rel_path=rel_path,
        size=int(entry["size"]),
        git_sha=str(entry["sha"]),
        local_path=sample_root / rel_path,
    )


def load_review_assets(
    tree_json: Path, sample_root: Path
) -> dict[str, RocketboxReviewAsset]:
    """Load the two sampled adult avatars from an official Git tree response."""
    tree = json.loads(Path(tree_json).read_text(encoding="utf-8"))
    entries = {
        str(entry["path"]): entry
        for entry in tree["tree"]
        if "path" in entry and "size" in entry and "sha" in entry
    }
    sample_root = Path(sample_root)
    assets: dict[str, RocketboxReviewAsset] = {}

    for asset_id, gender, avatar_dir, texture_prefix in _AVATAR_SPECS:
        avatar_root = f"Assets/Avatars/Adults/{avatar_dir}"
        fbx_rel_path = f"{avatar_root}/Export/{avatar_dir}.fbx"
        if fbx_rel_path not in entries:
            continue

        required_texture_paths = tuple(
            f"{avatar_root}/Textures/{texture_prefix}_{suffix}.tga"
            for suffix in _REQUIRED_TEXTURE_SUFFIXES
        )
        optional_texture_paths = tuple(
            f"{avatar_root}/Textures/{texture_prefix}_{suffix}.tga"
            for suffix in _FEMALE_OPTIONAL_TEXTURE_SUFFIXES
            if gender == "female"
        )
        textures = tuple(
            _official_file_from_tree(entries, rel_path, sample_root)
            for rel_path in required_texture_paths + optional_texture_paths
            if rel_path in entries
        )
        assets[asset_id] = RocketboxReviewAsset(
            asset_id=asset_id,
            gender=gender,
            avatar_dir=avatar_dir,
            texture_prefix=texture_prefix,
            fbx=_official_file_from_tree(entries, fbx_rel_path, sample_root),
            textures=textures,
            missing_required_textures=tuple(
                rel_path for rel_path in required_texture_paths if rel_path not in entries
            ),
        )
    return assets


def git_blob_sha1(path: Path) -> str:
    """Return the SHA-1 Git assigns to the file's blob object."""
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_bytes(payload: bytes) -> str:
    """Return the SHA-1 Git assigns to a blob containing ``payload``."""
    digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def verify_official_file(
    path: Path, expected_size: int, expected_git_sha: str
) -> None:
    """Raise when a local file does not match its official Git blob record."""
    path = Path(path)
    actual_size = path.stat().st_size
    actual_git_sha = git_blob_sha1(path)
    if actual_size != expected_size or actual_git_sha != expected_git_sha:
        raise OfficialFileError(
            f"Official file verification failed for {path}: "
            f"size actual={actual_size} expected={expected_size}; "
            f"Git blob SHA actual={actual_git_sha} expected={expected_git_sha}"
        )


def _official_file_url(rel_path: str) -> str:
    encoded_path = "/".join(quote(segment, safe="") for segment in rel_path.split("/"))
    return f"{RAW_GITHUB_BASE}{encoded_path}"


def _download_official_file(official_file: OfficialFile, opener: Callable) -> None:
    local_path = official_file.local_path
    part_path = local_path.with_name(f"{local_path.name}.part")
    request = Request(
        _official_file_url(official_file.rel_path), headers={"User-Agent": _USER_AGENT}
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with opener(request, timeout=60) as response, part_path.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        verify_official_file(part_path, official_file.size, official_file.git_sha)
        os.replace(part_path, local_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise


def ensure_official_files(
    asset: RocketboxReviewAsset, opener: Callable | None = None
) -> list[Path]:
    """Fetch missing official files and atomically replace invalid local copies."""
    if asset.missing_required_textures:
        missing_paths = ", ".join(asset.missing_required_textures)
        raise OfficialFileError(
            f"Rocketbox asset {asset.asset_id} is missing required official textures: "
            f"{missing_paths}"
        )
    opener = urlopen if opener is None else opener
    downloaded: list[Path] = []
    for official_file in (asset.fbx, *asset.textures):
        local_path = official_file.local_path
        if local_path.exists():
            try:
                verify_official_file(local_path, official_file.size, official_file.git_sha)
                continue
            except OfficialFileError:
                invalid_path = local_path.with_name(f"{local_path.name}.invalid")
                archive_index = 0
                while invalid_path.exists():
                    archive_index += 1
                    invalid_path = local_path.with_name(
                        f"{local_path.name}.invalid.{archive_index}"
                    )
                os.replace(local_path, invalid_path)
        _download_official_file(official_file, opener)
        downloaded.append(local_path)
    return downloaded


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one local source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_asset_files_verified(asset: RocketboxReviewAsset) -> None:
    if asset.missing_required_textures:
        missing_paths = ", ".join(asset.missing_required_textures)
        raise OfficialFileError(
            f"Rocketbox asset {asset.asset_id} is missing required official textures: "
            f"{missing_paths}"
        )
    for official_file in (asset.fbx, *asset.textures):
        verify_official_file(
            official_file.local_path, official_file.size, official_file.git_sha
        )


def _inspection_file_record(official_file: OfficialFile, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "official_rel_path": official_file.rel_path,
        "local_path": str(official_file.local_path.resolve()),
        "size": official_file.size,
        "git_blob_sha1": git_blob_sha1(official_file.local_path),
        "official_git_blob_sha1": official_file.git_sha,
        "sha256": sha256_file(official_file.local_path),
    }


def build_source_inspection(asset: RocketboxReviewAsset) -> dict[str, Any]:
    """Build an auditable inspection record from verified Task 1 local files."""
    _assert_asset_files_verified(asset)
    return {
        "schema_version": "rocketbox_source_inspection_v1",
        "asset_id": asset.asset_id,
        "gender": asset.gender,
        "avatar_dir": asset.avatar_dir,
        "texture_prefix": asset.texture_prefix,
        "up_axis": asset.up_axis,
        "forward_axis": asset.forward_axis,
        "missing_required_textures": list(asset.missing_required_textures),
        "official_files": [
            _inspection_file_record(asset.fbx, "fbx"),
            *[
                _inspection_file_record(texture, "texture")
                for texture in asset.textures
            ],
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)
    return path


def _expected_inspection(asset: RocketboxReviewAsset) -> dict[str, Any]:
    return build_source_inspection(asset)


def write_pending_source_review(
    asset: RocketboxReviewAsset, output_dir: Path, inspection: dict[str, Any]
) -> Path:
    """Write a pending source-review record for a freshly verified inspection."""
    expected_inspection = _expected_inspection(asset)
    if inspection != expected_inspection:
        raise OfficialFileError(
            "Source inspection does not match freshly verified Rocketbox source files"
        )
    review = {
        "schema_version": "rocketbox_human_source_review_v1",
        "asset_id": asset.asset_id,
        "up_axis": asset.up_axis,
        "forward_axis": asset.forward_axis,
        "missing_required_textures": list(asset.missing_required_textures),
        "source_sha256": sha256_file(asset.fbx.local_path),
        "official_files": inspection["official_files"],
        "geometry_status": "pending",
        "appearance_status": "pending",
        "direction_status": "pending",
        "approved_by": None,
        "approved_at": None,
        "notes": None,
    }
    return _write_json(Path(output_dir) / "source_review.json", review)


def _canonical_source_manifest(
    official_files: object,
) -> tuple[tuple[str, str, int, str], ...]:
    if not isinstance(official_files, list):
        raise SourceReviewNotApproved("source manifest official_files must be a list")
    manifest: list[tuple[str, str, int, str]] = []
    for record in official_files:
        if not isinstance(record, dict):
            raise SourceReviewNotApproved("source manifest contains a non-object record")
        try:
            role, rel_path, size, git_sha = (
                record[field] for field in _SOURCE_MANIFEST_FIELDS
            )
        except KeyError as error:
            raise SourceReviewNotApproved(
                f"source manifest record is missing {error.args[0]}"
            ) from error
        if (
            not isinstance(role, str)
            or not isinstance(rel_path, str)
            or type(size) is not int
            or not isinstance(git_sha, str)
        ):
            raise SourceReviewNotApproved("source manifest record has invalid field types")
        manifest.append((role, rel_path, size, git_sha))
    return tuple(manifest)


def _source_manifest_sha256(
    manifest: tuple[tuple[str, str, int, str], ...]
) -> str:
    payload = [dict(zip(_SOURCE_MANIFEST_FIELDS, record)) for record in manifest]
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _verify_pinned_source_manifest(review: dict[str, Any]) -> None:
    asset_id = review.get("asset_id")
    pin = _PINNED_SOURCE_MANIFESTS.get(asset_id)
    if pin is None:
        raise SourceReviewNotApproved(
            f"Unknown Rocketbox production asset_id: {asset_id!r}"
        )
    manifest = _canonical_source_manifest(review.get("official_files"))
    expected_layout = pin["layout"]
    actual_layout = tuple((role, rel_path) for role, rel_path, _, _ in manifest)
    if len(manifest) != len(expected_layout) or actual_layout != expected_layout:
        raise SourceReviewNotApproved(
            "source manifest roles, paths, order, or count do not match the pinned asset"
        )
    if _source_manifest_sha256(manifest) != pin["sha256"]:
        raise SourceReviewNotApproved(
            "source manifest SHA-256 does not match the pinned official asset"
        )


def _verify_review_official_files(review: dict[str, Any]) -> None:
    _verify_pinned_source_manifest(review)
    missing_required_textures = review.get("missing_required_textures")
    if missing_required_textures:
        raise SourceReviewNotApproved(
            "missing_required_textures must be empty before source review approval"
        )
    official_files = review.get("official_files")
    if not isinstance(official_files, list) or not official_files:
        raise SourceReviewNotApproved("official_files must contain verified source files")

    fbx_sha256 = None
    for record in official_files:
        try:
            local_path = Path(record["local_path"])
            expected_size = int(record["size"])
            expected_git_sha = str(record["official_git_blob_sha1"])
            verify_official_file(local_path, expected_size, expected_git_sha)
            actual_sha256 = sha256_file(local_path)
        except (KeyError, OSError, TypeError, ValueError, OfficialFileError) as error:
            raise SourceReviewNotApproved(
                f"official_files verification failed: {error}"
            ) from error
        if actual_sha256 != record.get("sha256"):
            raise SourceReviewNotApproved(
                f"official_files SHA-256 mismatch for {local_path}"
            )
        if record.get("role") == "fbx":
            fbx_sha256 = actual_sha256

    if fbx_sha256 is None or review.get("source_sha256") != fbx_sha256:
        raise SourceReviewNotApproved("source_sha256 does not match the verified FBX")


def _parse_aware_iso8601(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SourceReviewNotApproved("approved_at must be a timezone-aware ISO-8601 time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SourceReviewNotApproved(
            "approved_at must be a timezone-aware ISO-8601 time"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SourceReviewNotApproved("approved_at must be a timezone-aware ISO-8601 time")
    return parsed


def _verify_pinned_review_axes(review: dict[str, Any]) -> None:
    for field, expected in _PINNED_REVIEW_AXES.items():
        if review.get(field) != expected:
            raise SourceReviewNotApproved(f"{field} must equal {expected}")


def assert_source_review_approved(path: Path) -> dict[str, Any]:
    """Return a source review only when verified files have all human approvals."""
    try:
        review = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceReviewNotApproved(f"Could not read source review {path}: {error}") from error
    if review.get("schema_version") != "rocketbox_human_source_review_v1":
        raise SourceReviewNotApproved("schema_version is not rocketbox_human_source_review_v1")
    _verify_pinned_review_axes(review)
    _verify_review_official_files(review)
    for status_name in ("geometry_status", "appearance_status", "direction_status"):
        if review.get(status_name) != "approved":
            raise SourceReviewNotApproved(f"{status_name} must equal approved")
    if not isinstance(review.get("approved_by"), str) or not review["approved_by"].strip():
        raise SourceReviewNotApproved("approved_by must be non-empty")
    _parse_aware_iso8601(review.get("approved_at"))
    return review


def approve_source_review(
    path: Path,
    reviewer: str,
    geometry_status: str,
    appearance_status: str,
    direction_status: str,
    notes: str | None,
) -> Path:
    """Record one explicit human approval; this never infers any status."""
    review_path = Path(path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("schema_version") != "rocketbox_human_source_review_v1":
        raise SourceReviewNotApproved("schema_version is not rocketbox_human_source_review_v1")
    _verify_pinned_review_axes(review)
    _verify_review_official_files(review)
    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    review.update(
        {
            "geometry_status": geometry_status,
            "appearance_status": appearance_status,
            "direction_status": direction_status,
            "approved_by": reviewer,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
    )
    return _write_json(review_path, review)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Rocketbox review commands without doing I/O or network access."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for command_name in ("download", "inspect"):
        command = commands.add_parser(command_name)
        command.add_argument("--tree-json", type=Path, required=True)
        command.add_argument("--sample-root", type=Path, required=True)
        command.add_argument("--asset-id", required=True)
        if command_name == "inspect":
            command.add_argument("--output-dir", type=Path, required=True)

    approve = commands.add_parser("approve")
    approve.add_argument("--review-json", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    for status_name in ("geometry", "appearance", "direction"):
        approve.add_argument(f"--{status_name}", choices=("approved",), required=True)
    approve.add_argument("--notes", required=True)
    return parser.parse_args(argv)


def _load_cli_asset(args: argparse.Namespace) -> RocketboxReviewAsset:
    assets = load_review_assets(args.tree_json, args.sample_root)
    try:
        return assets[args.asset_id]
    except KeyError as error:
        raise ValueError(f"Unknown Rocketbox review asset: {args.asset_id}") from error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "download":
        asset = _load_cli_asset(args)
        ensure_official_files(asset)
        print(
            f"ROCKETBOX_OFFICIAL_FILES_OK asset_id={asset.asset_id} "
            f"files={len((asset.fbx, *asset.textures))}"
        )
        return 0
    if args.command == "inspect":
        asset = _load_cli_asset(args)
        inspection = build_source_inspection(asset)
        inspection_path = _write_json(args.output_dir / "source_inspection.json", inspection)
        review_path = write_pending_source_review(asset, args.output_dir, inspection)
        print(f"ROCKETBOX_SOURCE_INSPECTION_WRITTEN path={inspection_path}")
        print(f"ROCKETBOX_SOURCE_REVIEW_PENDING path={review_path}")
        return 0
    review_path = approve_source_review(
        args.review_json,
        args.reviewer,
        args.geometry,
        args.appearance,
        args.direction,
        args.notes,
    )
    print(f"ROCKETBOX_SOURCE_REVIEW_APPROVED path={review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
