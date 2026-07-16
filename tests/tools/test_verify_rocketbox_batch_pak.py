import hashlib
import json
from pathlib import Path

import pytest

from tools.verify_rocketbox_batch_pak import (
    build_expected_entries,
    parse_unreal_pak_list,
    verify_pak_inventory,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": "rocketbox_human_inventory_v1",
                "population": {"total": 1},
                "automatic_checks": {"overall": "passed"},
                "avatars": [
                    {
                        "base_avatar_id": "rocketbox_adults_male_adult_01",
                        "inventory_status": "passed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_root = tmp_path / "manifests"
    tag = "rocketbox_adults_male_adult_01_original_ue_v1"
    manifest = manifest_root / tag / "ue_import_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "rocketbox_batch_native_ue_import_v1",
                "base_avatar_id": "rocketbox_adults_male_adult_01",
                "tag": tag,
                "reload_verification": {"status": "passed"},
                "content": {
                    "blueprint": (
                        "/Game/MyAssets/Audioset/Blueprints/"
                        f"gate_{tag}/BP_gate_{tag}"
                    ),
                    "skeletal_mesh": (
                        "/Game/MyAssets/Audioset/Meshes/"
                        f"gate_{tag}/runtime.runtime"
                    ),
                    "skeleton": (
                        "/Game/MyAssets/Audioset/Meshes/"
                        f"gate_{tag}/runtime_Skeleton.runtime_Skeleton"
                    ),
                    "animations": {
                        "Walking": (
                            "/Game/MyAssets/Audioset/Meshes/"
                            f"gate_{tag}/Walking.Walking"
                        ),
                        "Standing_Idle": (
                            "/Game/MyAssets/Audioset/Meshes/"
                            f"gate_{tag}/Standing_Idle.Standing_Idle"
                        ),
                    },
                    "materials": [
                        "/Game/MyAssets/Audioset/Meshes/"
                        f"gate_{tag}/shirt.shirt"
                    ],
                    "textures": [
                        "/Game/MyAssets/Audioset/Meshes/"
                        f"gate_{tag}/shirt_color.shirt_color"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return inventory, manifest_root


def test_parse_unreal_pak_list_normalizes_mount_paths():
    entries = parse_unreal_pak_list(
        '\n'.join(
            [
                'LogPakFile: Display: "../../../SpearSim/Content/Foo.uasset" offset: 0, size: 1 bytes',
                '"../../../SpearSim/Content/Bar.uexp" offset: 1, size: 2 bytes',
                "2 files (3 bytes)",
            ]
        )
    )

    assert entries == {
        "SpearSim/Content/Bar.uexp",
        "SpearSim/Content/Foo.uasset",
    }


def test_build_expected_entries_uses_manifest_assets_and_required_physics(tmp_path):
    inventory, manifest_root = _write_inputs(tmp_path)

    expected, avatars = build_expected_entries(inventory, manifest_root)

    tag = "rocketbox_adults_male_adult_01_original_ue_v1"
    prefix = f"SpearSim/Content/MyAssets/Audioset/Meshes/gate_{tag}"
    assert f"{prefix}/runtime.uasset" in expected
    assert f"{prefix}/runtime_Skeleton.uasset" in expected
    assert f"{prefix}/runtime_PhysicsAsset.uasset" in expected
    assert f"{prefix}/Walking.uasset" in expected
    assert f"{prefix}/Standing_Idle.uasset" in expected
    assert f"{prefix}/shirt.uasset" in expected
    assert f"{prefix}/shirt_color.uasset" in expected
    assert len(avatars) == 1
    assert avatars[0]["base_avatar_id"] == "rocketbox_adults_male_adult_01"
    assert avatars[0]["expected_asset_count"] == 8


def test_verifier_rejects_even_one_missing_required_asset(tmp_path):
    inventory, manifest_root = _write_inputs(tmp_path)
    expected, _ = build_expected_entries(inventory, manifest_root)
    missing = next(path for path in expected if path.endswith("/Walking.uasset"))

    with pytest.raises(RuntimeError, match="missing 1 expected Rocketbox entries"):
        verify_pak_inventory(expected - {missing}, inventory, manifest_root)


def test_verifier_returns_hash_bound_summary(tmp_path):
    inventory, manifest_root = _write_inputs(tmp_path)
    expected, _ = build_expected_entries(inventory, manifest_root)
    pak = tmp_path / "SpearSim-Linux.pak"
    pak.write_bytes(b"pak fixture")

    report = verify_pak_inventory(
        expected,
        inventory,
        manifest_root,
        pak_path=pak,
    )

    assert report["status"] == "passed"
    assert report["avatar_count"] == 1
    assert report["walking_count"] == 1
    assert report["standing_idle_count"] == 1
    assert report["pak_size_bytes"] == len(b"pak fixture")
    assert report["pak_sha256"] == hashlib.sha256(b"pak fixture").hexdigest()
