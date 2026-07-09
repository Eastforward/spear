import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from promote_source_asset import promote_source_asset  # noqa: E402


def _write_approved_candidate(approved_dir: Path, tag: str) -> Path:
    tag_dir = approved_dir / tag
    tag_dir.mkdir(parents=True)
    mesh = tag_dir / "mesh_oriented.glb"
    mesh.write_bytes(b"approved oriented mesh")
    (tag_dir / "mesh_runtime.glb").write_bytes(b"runtime mesh")
    (tag_dir / "mesh_runtime.json").write_text("{}", encoding="utf-8")
    image = Image.new("RGB", (4, 2))
    for x in range(4):
        for y in range(2):
            image.putpixel((x, y), (120, 80, 40) if x < 3 else (230, 210, 180))
    image.save(tag_dir / "hy3d_diffuse.jpg")
    (tag_dir / "hy3d_roughness.jpg").write_bytes(b"roughness")
    (tag_dir / "hy3d_metallic.jpg").write_bytes(b"metallic")
    (tag_dir / "reference.png").write_bytes(b"reference")
    (tag_dir / "direction_preview_review.png").write_bytes(b"review")
    (tag_dir / "direction.json").write_text(json.dumps({
        "algorithm_version": "auto_orient_v1",
        "human_approved": True,
        "human_approved_by": "jzy",
        "human_approved_at": "2026-07-09T00:00:00+00:00",
        "quarantined": False,
        "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    (tag_dir / "source_asset_candidate.json").write_text(json.dumps({
        "schema_version": "source_asset_v1",
        "asset_id": "dog_pug_0001",
        "legacy_tag": tag,
        "asset_class": "animal",
        "category": "dog",
        "family": "pug",
        "variant": {"variant_index": 1, "intended_color_label": "pug"},
        "generation": {"text_description": "pug dog"},
        "appearance": {"dominant_colors": [], "color_measurement_status": "pending"},
        "visual_assets": {},
        "rig": {
            "skeleton_family": "quaternius_dog",
            "animations": ["Idle", "Walking"],
            "loop_required": True,
        },
        "audio": {
            "default_lookup": "dog_bark",
            "allowed_lookups": ["dog_bark", "dog_growl", "dog_sharp_bark", "silent"],
        },
        "review": {"overall_status": "needs_runtime_gate"},
    }), encoding="utf-8")
    return tag_dir


def _write_approved_human_candidate(approved_dir: Path, tag: str) -> Path:
    tag_dir = approved_dir / tag
    tag_dir.mkdir(parents=True)
    mesh = tag_dir / "mesh_oriented.glb"
    mesh.write_bytes(b"approved oriented human mesh")
    (tag_dir / "mesh_runtime.glb").write_bytes(b"walking runtime")
    (tag_dir / "mesh_runtime_walking.glb").write_bytes(b"walking runtime")
    (tag_dir / "mesh_runtime_standing_idle.glb").write_bytes(b"idle runtime")
    (tag_dir / "mesh_runtime.json").write_text(json.dumps({
        "schema_version": "human_mixamo_runtime_v1",
        "runtime_type": "mixamo_humanoid_nearest_skin_transfer",
        "default_animation": "Walking",
        "legacy_runtime": str(tag_dir / "mesh_runtime_walking.glb"),
        "animations": {
            "Walking": {
                "role": "walk",
                "motion_style": "walking",
                "loop": True,
                "glb_path": str(tag_dir / "mesh_runtime_walking.glb"),
                "source_fbx": "raw/Walking.fbx",
            },
            "Standing_Idle": {
                "role": "idle",
                "motion_style": "stationary",
                "loop": True,
                "glb_path": str(tag_dir / "mesh_runtime_standing_idle.glb"),
                "source_fbx": "raw/Standing_Idle.fbx",
            },
        },
    }), encoding="utf-8")
    image = Image.new("RGB", (4, 2), (35, 75, 140))
    image.save(tag_dir / "hy3d_diffuse.jpg")
    (tag_dir / "reference.png").write_bytes(b"reference")
    (tag_dir / "direction_preview_review.png").write_bytes(b"review")
    (tag_dir / "direction.json").write_text(json.dumps({
        "algorithm_version": "manual_human_orientation_v1",
        "human_approved": True,
        "human_approved_by": "jzy",
        "human_approved_at": "2026-07-09T00:00:00+00:00",
        "quarantined": False,
        "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    (tag_dir / "source_asset_candidate.json").write_text(json.dumps({
        "schema_version": "source_asset_v1",
        "asset_id": "human_male_blue_hoodie_0001",
        "legacy_tag": tag,
        "asset_class": "human",
        "category": "human",
        "family": "male_blue_hoodie",
        "variant": {"variant_index": 1, "intended_color_label": "male blue hoodie"},
        "generation": {"text_description": "male blue hoodie human"},
        "appearance": {"dominant_colors": [], "color_measurement_status": "pending"},
        "visual_assets": {},
        "rig": {
            "skeleton_family": "mixamo_humanoid",
            "animations": ["Standing_Idle", "Walking"],
            "loop_required": True,
        },
        "audio": {
            "default_lookup": "speech",
            "allowed_lookups": ["speech", "talking", "conversation", "silent"],
        },
        "review": {"overall_status": "needs_runtime_gate"},
    }), encoding="utf-8")
    return tag_dir


def test_promote_source_asset_writes_registry_manifest(tmp_path):
    approved_dir = tmp_path / "approved"
    registry_root = tmp_path / "registry"
    _write_approved_candidate(approved_dir, "dog_pug_v1")

    asset_path = promote_source_asset(
        "dog_pug_v1",
        approved_dir=approved_dir,
        registry_root=registry_root,
    )

    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    registry = json.loads((registry_root / "registry.json").read_text(encoding="utf-8"))
    assert asset_path == registry_root / "dog" / "pug" / "dog_pug_0001" / "asset.json"
    assert asset["review"]["overall_status"] == "approved"
    assert asset["review"]["rig_status"] == "approved"
    assert asset["visual_assets"]["mesh_runtime"].endswith("mesh_runtime.glb")
    assert asset["appearance"]["dominant_colors"]
    assert asset["appearance"]["dominant_colors"][0]["source"] == "measured_from_texture"
    assert registry["assets"] == [{
        "asset_id": "dog_pug_0001",
        "asset_class": "animal",
        "category": "dog",
        "family": "pug",
        "path": "dog/pug/dog_pug_0001/asset.json",
        "overall_status": "approved",
    }]


def test_promote_source_asset_requires_runtime_gate(tmp_path):
    approved_dir = tmp_path / "approved"
    tag_dir = _write_approved_candidate(approved_dir, "dog_pug_v1")
    (tag_dir / "mesh_runtime.glb").unlink()

    with pytest.raises(RuntimeError, match="mesh_runtime.glb"):
        promote_source_asset(
            "dog_pug_v1",
            approved_dir=approved_dir,
            registry_root=tmp_path / "registry",
        )


def test_promote_human_source_asset_records_mixamo_animation_assets(tmp_path):
    approved_dir = tmp_path / "approved"
    registry_root = tmp_path / "registry"
    _write_approved_human_candidate(approved_dir, "human_male_blue_hoodie_v1")

    asset_path = promote_source_asset(
        "human_male_blue_hoodie_v1",
        approved_dir=approved_dir,
        registry_root=registry_root,
    )

    asset = json.loads(asset_path.read_text(encoding="utf-8"))
    registry = json.loads((registry_root / "registry.json").read_text(encoding="utf-8"))
    assert (
        asset_path
        == registry_root
        / "human"
        / "male_blue_hoodie"
        / "human_male_blue_hoodie_0001"
        / "asset.json"
    )
    assert asset["asset_class"] == "human"
    assert asset["audio"]["default_lookup"] == "speech"
    assert asset["visual_assets"]["mesh_runtime_walking"].endswith(
        "mesh_runtime_walking.glb"
    )
    assert asset["visual_assets"]["mesh_runtime_standing_idle"].endswith(
        "mesh_runtime_standing_idle.glb"
    )
    assert asset["rig"]["runtime_type"] == "mixamo_humanoid_nearest_skin_transfer"
    assert asset["rig"]["default_animation"] == "Walking"
    assert set(asset["rig"]["animation_assets"]) == {"Standing_Idle", "Walking"}
    assert asset["rig"]["animation_assets"]["Walking"]["motion_style"] == "walking"
    assert asset["rig"]["animation_assets"]["Standing_Idle"]["loop"] is True
    assert registry["assets"] == [{
        "asset_id": "human_male_blue_hoodie_0001",
        "asset_class": "human",
        "category": "human",
        "family": "male_blue_hoodie",
        "path": "human/male_blue_hoodie/human_male_blue_hoodie_0001/asset.json",
        "overall_status": "approved",
    }]
