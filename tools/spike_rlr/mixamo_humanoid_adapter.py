"""Lightweight Mixamo humanoid import-plan helpers.

This module does not run Unreal import. It turns locally provided FBX files
into a deterministic plan that the later UE adapter can consume.
"""
from __future__ import annotations

from pathlib import Path
import re

from external_data_paths import dataset_spec
from mixamo_probe import discover_mixamo_fbx


_ROLE_ALIASES = {
    "idle": {
        "idle",
        "standingidle",
        "standing_idle",
        "standing idle",
    },
    "walk": {
        "walk",
        "walking",
    },
    "run": {
        "run",
        "running",
    },
}

_MOTION_STYLE_BY_ROLE = {
    "idle": "stationary",
    "walk": "walking",
    "run": "running",
}

_DEFAULT_REQUIRED_ROLES = ("idle", "walk")


def _clip_stem_key(path: Path) -> str:
    stem = Path(path).stem
    compact = re.sub(r"[^a-z0-9]+", "", stem.lower())
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    spaced = re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()
    return compact, normalized, spaced


def classify_mixamo_clip_role(path: Path) -> str | None:
    """Classify a Mixamo FBX filename into the small roles used by smoke tests."""
    if Path(path).suffix.lower() != ".fbx":
        return None
    keys = set(_clip_stem_key(Path(path)))
    for role, aliases in _ROLE_ALIASES.items():
        if keys & aliases:
            return role
    return None


def _source_name(path: Path) -> str:
    return Path(path).stem.replace(" ", "_")


def _clip_entry(root: Path, fbx_path: Path, role: str, ue_folder: str) -> dict:
    source_name = _source_name(fbx_path)
    return {
        "role": role,
        "fbx_path": str(fbx_path),
        "fbx_relative_path": fbx_path.relative_to(root).as_posix(),
        "source_name": source_name,
        "ue_asset_path": f"{ue_folder.rstrip('/')}/{source_name}",
        "wanted_anim": source_name,
        "motion_style": _MOTION_STYLE_BY_ROLE.get(role, "unknown"),
    }


def build_mixamo_humanoid_import_plan(
    root: Path,
    *,
    required_roles: tuple[str, ...] = _DEFAULT_REQUIRED_ROLES,
    ue_folder: str = "/Game/Mixamo/Humanoid",
) -> dict:
    """Build a deterministic humanoid import plan from local Mixamo FBX files."""
    spec = dataset_spec("mixamo")
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{spec.label} dataset root does not exist: {root}. "
            f"{spec.acquisition_hint}"
        )

    chosen: dict[str, dict] = {}
    for fbx_path in discover_mixamo_fbx(root):
        role = classify_mixamo_clip_role(fbx_path)
        if role is None or role not in required_roles or role in chosen:
            continue
        chosen[role] = _clip_entry(root, fbx_path, role, ue_folder)

    missing = [role for role in required_roles if role not in chosen]
    state = "ready" if not missing else "missing_required_clips"
    return {
        "state": state,
        "asset_family": "mixamo_humanoid",
        "skeleton_family": "humanoid",
        "visual_source_class": "human",
        "default_audio_lookup": "speech",
        "ue_folder": ue_folder.rstrip("/"),
        "required_roles": list(required_roles),
        "missing_roles": missing,
        "manual_action": (
            "" if not missing
            else "Add Mixamo FBX files for required roles: "
                 + ", ".join("Walking.fbx" if role == "walk" else f"{role}.fbx"
                             for role in missing)
        ),
        "clips": {role: chosen[role] for role in required_roles if role in chosen},
    }
