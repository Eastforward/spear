"""Helpers for source asset candidate manifests.

Candidate manifests are written next to generated Hunyuan assets while they are
still pending review. They are not production registry entries until later
gates promote the asset into data/source_assets_v1.
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any


CANDIDATE_MANIFEST_NAME = "source_asset_candidate.json"

REPO_ROOT = Path(__file__).resolve().parents[2]


def _slug(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return out or "unknown"


def _variant_index_from_tag(tag: str) -> int:
    m = re.search(r"_v([0-9]+)$", tag)
    if m:
        return int(m.group(1))
    return 1


def _tag_base(tag: str) -> str:
    return re.sub(r"_v[0-9]+$", "", tag)


def asset_id_for_tag(tag: str, category: str, family: str) -> str:
    del category, family
    return f"{_tag_base(tag)}_{_variant_index_from_tag(tag):04d}"


def _path_for_manifest(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _first_existing(tag_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = tag_dir / name
        if p.exists():
            return p
    return None


def _category_defaults(category: str) -> dict[str, Any]:
    if category == "dog":
        return {
            "skeleton_family": "quaternius_dog",
            "default_lookup": "dog_bark",
            "allowed_lookups": ["dog_bark", "dog_growl", "dog_sharp_bark", "silent"],
        }
    if category == "cat":
        return {
            "skeleton_family": "quaternius_cat",
            "default_lookup": "cat_purring",
            "allowed_lookups": ["cat_meow", "cat_purring", "silent"],
        }
    if category == "human":
        return {
            "skeleton_family": "mixamo_humanoid",
            "default_lookup": "speech",
            "allowed_lookups": ["speech", "talking", "conversation", "silent"],
        }
    return {
        "skeleton_family": "unknown",
        "default_lookup": "silent",
        "allowed_lookups": ["silent"],
    }


def build_hy3d_candidate_manifest(
    tag_dir: Path | str,
    *,
    tag: str,
    species: str,
    breed: str,
    seed: int | None,
    positive_prompt: str,
    flux_model: str | None = None,
    negative_prompt: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    tag_dir = Path(tag_dir)
    category = _slug(species)
    family = _slug(breed)
    defaults = _category_defaults(category)
    now = created_at or datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    mesh_original = _first_existing(tag_dir, ("mesh.obj", "mesh.glb"))

    return {
        "schema_version": "source_asset_v1",
        "asset_id": asset_id_for_tag(tag, category, family),
        "legacy_tag": tag,
        "asset_class": "human" if category == "human" else "animal",
        "category": category,
        "family": family,
        "variant": {
            "variant_index": _variant_index_from_tag(tag),
            "size": None,
            "coat_type": None,
            "intended_color_label": family.replace("_", " "),
        },
        "generation": {
            "source_pipeline": "flux+hunyuan3d",
            "model": "+".join(
                p for p in (flux_model, "hunyuan3d-2.1") if p
            ),
            "seed": seed,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "text_description": f"{breed} {species}",
            "created_at": now,
        },
        "appearance": {
            "dominant_colors": [],
            "color_tags": [],
            "lightness": None,
            "saturation": None,
            "color_measurement_status": "pending",
        },
        "visual_assets": {
            "reference_image": _path_for_manifest(tag_dir / "reference.png"),
            "mesh_original": _path_for_manifest(mesh_original),
            "mesh_oriented": _path_for_manifest(tag_dir / "mesh_oriented.glb"),
            "mesh_runtime": _path_for_manifest(tag_dir / "mesh_runtime.glb"),
            "diffuse": _path_for_manifest(tag_dir / "hy3d_diffuse.jpg"),
            "roughness": _path_for_manifest(tag_dir / "hy3d_roughness.jpg"),
            "metallic": _path_for_manifest(tag_dir / "hy3d_metallic.jpg"),
            "review_image": _path_for_manifest(tag_dir / "direction_preview_review.png"),
            "direction_json": _path_for_manifest(tag_dir / "direction.json"),
            "runtime_metadata": _path_for_manifest(tag_dir / "mesh_runtime.json"),
        },
        "rig": {
            "skeleton_family": defaults["skeleton_family"],
            "animations": ["Idle", "Walking"],
            "loop_required": True,
        },
        "audio": {
            "default_lookup": defaults["default_lookup"],
            "allowed_lookups": defaults["allowed_lookups"],
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


def write_hy3d_candidate_manifest(
    tag_dir: Path | str,
    *,
    tag: str,
    species: str,
    breed: str,
    seed: int | None,
    positive_prompt: str,
    flux_model: str | None = None,
    negative_prompt: str | None = None,
    created_at: str | None = None,
) -> Path:
    tag_dir = Path(tag_dir)
    manifest = build_hy3d_candidate_manifest(
        tag_dir,
        tag=tag,
        species=species,
        breed=breed,
        seed=seed,
        positive_prompt=positive_prompt,
        flux_model=flux_model,
        negative_prompt=negative_prompt,
        created_at=created_at,
    )
    path = tag_dir / CANDIDATE_MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def sync_candidate_manifest_review(
    tag_dir: Path | str,
    direction: dict[str, Any],
) -> Path | None:
    tag_dir = Path(tag_dir)
    path = tag_dir / CANDIDATE_MANIFEST_NAME
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    approved = bool(direction.get("human_approved"))
    manifest.setdefault("review", {})
    manifest["review"]["direction_status"] = "approved" if approved else "rejected"
    manifest["review"]["overall_status"] = (
        "needs_runtime_gate" if approved else "rejected"
    )
    manifest["review"]["approved_by"] = direction.get("human_approved_by")
    manifest["review"]["approved_at"] = direction.get("human_approved_at")
    manifest["review"]["notes"] = direction.get("human_notes")

    visual_assets = manifest.setdefault("visual_assets", {})
    for key, name in (
        ("reference_image", "reference.png"),
        ("mesh_original", "mesh.obj"),
        ("mesh_oriented", "mesh_oriented.glb"),
        ("diffuse", "hy3d_diffuse.jpg"),
        ("roughness", "hy3d_roughness.jpg"),
        ("metallic", "hy3d_metallic.jpg"),
        ("review_image", "direction_preview_review.png"),
        ("direction_json", "direction.json"),
        ("runtime_metadata", "mesh_runtime.json"),
        ("mesh_runtime", "mesh_runtime.glb"),
    ):
        value = _path_for_manifest(tag_dir / name)
        if value is not None:
            visual_assets[key] = value
    if direction.get("mesh_oriented"):
        visual_assets["mesh_oriented"] = direction["mesh_oriented"]

    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
