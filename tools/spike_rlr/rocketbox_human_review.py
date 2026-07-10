"""Verified official-source catalog helpers for Rocketbox avatar review."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
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


class OfficialFileError(RuntimeError):
    """An on-disk file differs from its official Git tree record."""


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
                os.replace(local_path, invalid_path)
        _download_official_file(official_file, opener)
        downloaded.append(local_path)
    return downloaded
