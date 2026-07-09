import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from source_asset_registry import (  # noqa: E402
    approved_assets,
    load_asset,
    load_registry,
    resolve_source_pool,
    resolve_source_pool_entry,
)


def _write_asset_fixture(root: Path, asset: dict, *, status: str) -> None:
    category = asset.get("category", "dog")
    family = asset.get("family", "fixture")
    asset_path = root / category / family / asset["asset_id"] / "asset.json"
    asset_path.parent.mkdir(parents=True)
    asset["review"] = {
        "overall_status": status,
        "appearance_status": status,
        "direction_status": status,
        "texture_status": status,
        "rig_status": status,
        "audio_mapping_status": status,
        "approved_by": None,
        "approved_at": None,
        "notes": None,
    }
    asset_path.write_text(json.dumps(asset), encoding="utf-8")
    (root / "registry.json").write_text(json.dumps({
        "schema_version": "source_assets_v1",
        "assets": [{
            "asset_id": asset["asset_id"],
            "asset_class": asset["asset_class"],
            "category": asset["category"],
            "family": asset["family"],
            "path": f"{category}/{family}/{asset['asset_id']}/asset.json",
            "overall_status": status,
        }],
    }), encoding="utf-8")


def test_load_registry_indexes_initial_approved_assets():
    registry = load_registry()

    ids = {item["asset_id"] for item in registry["assets"]}

    assert {
        "dog_golden_0001",
        "dog_beagle_0002",
        "cat_british_shorthair_0002",
    } <= ids


def test_load_asset_contains_generation_and_measured_colors():
    asset = load_asset("dog_golden_0001")

    assert asset["legacy_tag"] == "dog_golden"
    assert asset["generation"]["text_description"]
    color = asset["appearance"]["dominant_colors"][0]
    assert color["source"] == "measured_from_texture"
    assert color["hex"].startswith("#")
    assert len(color["rgb"]) == 3
    assert len(color["lab"]) == 3


def test_approved_assets_can_filter_by_category():
    dogs = approved_assets(category="dog")

    assert {asset["asset_id"] for asset in dogs} >= {
        "dog_golden_0001",
        "dog_beagle_0002",
    }
    assert all(asset["category"] == "dog" for asset in dogs)
    assert all(asset["review"]["overall_status"] == "approved" for asset in dogs)


def test_resolve_source_pool_entry_uses_asset_default_audio():
    resolved = resolve_source_pool_entry({"asset_id": "dog_beagle_0002"})

    assert resolved["asset_id"] == "dog_beagle_0002"
    assert resolved["tag"] == "dog_beagle_v2"
    assert resolved["audio_lookup"] == "dog_bark"
    assert resolved["asset_class"] == "animal"
    assert resolved["category"] == "dog"
    assert resolved["family"] == "beagle"


def test_resolve_source_pool_entry_allows_audio_override():
    resolved = resolve_source_pool_entry({
        "asset_id": "dog_beagle_0002",
        "audio_lookup": "dog_sharp_bark",
        "placement": "camera_rear",
    })

    assert resolved["tag"] == "dog_beagle_v2"
    assert resolved["audio_lookup"] == "dog_sharp_bark"
    assert resolved["placement"] == "camera_rear"


def test_legacy_source_pool_entry_passes_through():
    original = {"tag": "dog_golden", "audio_lookup": "dog_bark"}

    resolved = resolve_source_pool_entry(original)

    assert resolved == original
    assert resolved is not original


def test_resolve_source_pool_preserves_order():
    resolved = resolve_source_pool([
        {"asset_id": "dog_golden_0001"},
        {"asset_id": "cat_british_shorthair_0002"},
    ])

    assert [entry["tag"] for entry in resolved] == [
        "dog_golden",
        "cat_british_shorthair_v2",
    ]


def test_unapproved_registry_asset_is_rejected(tmp_path):
    asset = {
        "schema_version": "source_asset_v1",
        "asset_id": "dog_pending_0001",
        "legacy_tag": "dog_pending",
        "asset_class": "animal",
        "category": "dog",
        "family": "fixture",
        "variant": {"variant_index": 1},
        "generation": {"text_description": "pending fixture dog"},
        "appearance": {"dominant_colors": []},
        "visual_assets": {},
        "rig": {"skeleton_family": "fixture", "animations": [], "loop_required": True},
        "audio": {"default_lookup": "dog_bark", "allowed_lookups": ["dog_bark"]},
    }
    _write_asset_fixture(tmp_path, asset, status="needs_review")

    with pytest.raises(RuntimeError, match="not approved"):
        resolve_source_pool_entry(
            {"asset_id": "dog_pending_0001"},
            registry_root=tmp_path,
        )


def test_resolve_source_pool_entry_carries_runtime_render_hints(tmp_path):
    asset = {
        "schema_version": "source_asset_v1",
        "asset_id": "human_fixture_0001",
        "legacy_tag": "human_fixture_v1",
        "asset_class": "human",
        "category": "human",
        "family": "fixture",
        "variant": {"variant_index": 1},
        "generation": {"text_description": "fixture human"},
        "appearance": {"dominant_colors": []},
        "visual_assets": {},
        "rig": {
            "skeleton_family": "mixamo_humanoid",
            "animations": ["Standing_Idle", "Walking"],
            "loop_required": True,
            "walking_forward_yaw_offset_deg": 90.0,
            "actor_scale": 1.0,
            "actor_z_lift_cm": 14.0,
        },
        "audio": {"default_lookup": "speech", "allowed_lookups": ["speech"]},
    }
    _write_asset_fixture(tmp_path, asset, status="approved")

    resolved = resolve_source_pool_entry(
        {"asset_id": "human_fixture_0001"},
        registry_root=tmp_path,
    )

    assert resolved["tag"] == "human_fixture_v1"
    assert resolved["audio_lookup"] == "speech"
    assert resolved["actor_scale"] == 1.0
    assert resolved["actor_z_lift_cm"] == 14.0
    assert resolved["walking_forward_yaw_offset_deg"] == 90.0
