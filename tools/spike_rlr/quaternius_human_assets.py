"""Quaternius human source-asset manifest helpers."""
from __future__ import annotations

import datetime
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from source_asset_registry import default_registry_root


DEFAULT_QUATERNIUS_HUMAN_ZIP = Path(
    "/data/datasets/quaternius/raw/"
    "quaternius_ultimate_animated_character_pack_2021_opengameart.zip"
)

_REALISTIC_LABEL_PREFIXES = (
    "Casual",
    "Chef",
    "Cowboy",
    "Doctor",
    "OldClassy",
    "Suit",
    "Worker",
)


@dataclass(frozen=True)
class QuaterniusHumanAsset:
    asset_id: str
    legacy_tag: str
    family: str
    visual_label: str
    archive_path: Path
    fbx_member: str
    obj_member: str | None
    mtl_member: str | None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _is_human_visual_label(label: str) -> bool:
    return any(label.startswith(prefix) for prefix in _REALISTIC_LABEL_PREFIXES)


def _member_for_stem(members: set[str], stem: str, suffix: str) -> str | None:
    needle = f"/OBJ/{stem}{suffix}"
    return next((m for m in members if m.endswith(needle)), None)


def discover_quaternius_human_assets(
    zip_path: Path | str = DEFAULT_QUATERNIUS_HUMAN_ZIP,
) -> list[QuaterniusHumanAsset]:
    archive = Path(zip_path)
    if not archive.exists():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())
    assets: list[QuaterniusHumanAsset] = []
    for member in sorted(members):
        if "/FBX/" not in member or not member.lower().endswith(".fbx"):
            continue
        label = Path(member).stem
        if not _is_human_visual_label(label):
            continue
        family = _slug(label)
        assets.append(QuaterniusHumanAsset(
            asset_id=f"human_{family}_0001",
            legacy_tag=f"human_{family}_v1",
            family=family,
            visual_label=label,
            archive_path=archive,
            fbx_member=member,
            obj_member=_member_for_stem(members, label, ".obj"),
            mtl_member=_member_for_stem(members, label, ".mtl"),
        ))
    return sorted(assets, key=lambda item: item.asset_id)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _archive_member_path(archive: Path, member: str | None) -> str | None:
    if member is None:
        return None
    return f"{archive}!{member}"


def build_quaternius_human_asset_manifest(
    source: QuaterniusHumanAsset,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    description = source.visual_label.replace("_", " ").lower()
    return {
        "schema_version": "source_asset_v1",
        "asset_id": source.asset_id,
        "legacy_tag": source.legacy_tag,
        "asset_class": "human",
        "category": "human",
        "family": source.family,
        "variant": {
            "variant_index": 1,
            "gender_presentation": (
                "female" if source.family.endswith("_female")
                else "male" if source.family.endswith("_male")
                else None
            ),
            "outfit_label": source.family,
            "intended_color_label": None,
        },
        "generation": {
            "source_pipeline": "quaternius_human_pack",
            "model": "ultimate_animated_character_pack_2021",
            "seed": None,
            "positive_prompt": None,
            "negative_prompt": None,
            "text_description": f"low-poly {description} human character",
            "created_at": created_at or _now(),
        },
        "appearance": {
            "dominant_colors": [],
            "color_tags": [],
            "lightness": None,
            "saturation": None,
            "color_measurement_status": "material_colors_pending",
        },
        "visual_assets": {
            "source_archive": str(source.archive_path),
            "source_fbx": _archive_member_path(source.archive_path, source.fbx_member),
            "source_obj": _archive_member_path(source.archive_path, source.obj_member),
            "source_mtl": _archive_member_path(source.archive_path, source.mtl_member),
            "mesh_original": None,
            "mesh_oriented": None,
            "mesh_runtime": None,
            "diffuse": None,
            "review_image": None,
            "review_video": None,
            "runtime_metadata": None,
        },
        "rig": {
            "skeleton_family": "quaternius_human",
            "animations": ["Idle", "Walk"],
            "loop_required": True,
            "animation_source": "embedded_quaternius_fbx",
        },
        "audio": {
            "default_lookup": "speech",
            "allowed_lookups": ["speech", "talking", "conversation", "silent"],
        },
        "review": {
            "overall_status": "needs_review",
            "appearance_status": "needs_review",
            "direction_status": "needs_review",
            "texture_status": "needs_review",
            "rig_status": "needs_review",
            "audio_mapping_status": "needs_review",
            "approved_by": None,
            "approved_at": None,
            "notes": None,
        },
    }


def write_quaternius_human_asset_manifest(
    source: QuaterniusHumanAsset,
    *,
    registry_root: Path | str | None = None,
    created_at: str | None = None,
) -> Path:
    root = Path(registry_root) if registry_root is not None else default_registry_root()
    out = root / "human" / source.family / source.asset_id / "asset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            build_quaternius_human_asset_manifest(source, created_at=created_at),
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return out
