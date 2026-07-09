import json
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _write_quaternius_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        base = "Ultimate Animated Character Pack - Nov 2019"
        zf.writestr(f"{base}/FBX/Casual_Male.fbx", b"fbx")
        zf.writestr(f"{base}/OBJ/Casual_Male.obj", "v 0 0 0\n")
        zf.writestr(f"{base}/OBJ/Casual_Male.mtl", "newmtl Shirt\nKd 0.1 0.2 0.3\n")
        zf.writestr(f"{base}/FBX/Casual_Female.fbx", b"fbx")


def test_discover_quaternius_human_assets_from_raw_zip(tmp_path):
    from quaternius_human_assets import discover_quaternius_human_assets

    zip_path = tmp_path / "humans.zip"
    _write_quaternius_zip(zip_path)

    assets = discover_quaternius_human_assets(zip_path)

    ids = [asset.asset_id for asset in assets]
    assert ids == ["human_casual_female_0001", "human_casual_male_0001"]
    male = next(asset for asset in assets if asset.asset_id == "human_casual_male_0001")
    assert male.legacy_tag == "human_casual_male_v1"
    assert male.visual_label == "Casual_Male"
    assert male.fbx_member.endswith("/FBX/Casual_Male.fbx")
    assert male.obj_member.endswith("/OBJ/Casual_Male.obj")
    assert male.mtl_member.endswith("/OBJ/Casual_Male.mtl")


def test_build_quaternius_human_asset_manifest_has_speech_and_loop_metadata(tmp_path):
    from quaternius_human_assets import (
        build_quaternius_human_asset_manifest,
        discover_quaternius_human_assets,
    )

    zip_path = tmp_path / "humans.zip"
    _write_quaternius_zip(zip_path)
    source = next(
        asset for asset in discover_quaternius_human_assets(zip_path)
        if asset.asset_id == "human_casual_male_0001"
    )

    manifest = build_quaternius_human_asset_manifest(
        source,
        created_at="2026-07-09T00:00:00+00:00",
    )

    assert manifest["schema_version"] == "source_asset_v1"
    assert manifest["asset_id"] == "human_casual_male_0001"
    assert manifest["legacy_tag"] == "human_casual_male_v1"
    assert manifest["asset_class"] == "human"
    assert manifest["category"] == "human"
    assert manifest["family"] == "casual_male"
    assert manifest["generation"]["source_pipeline"] == "quaternius_human_pack"
    assert "Casual_Male.fbx" in manifest["visual_assets"]["source_fbx"]
    assert manifest["rig"]["skeleton_family"] == "quaternius_human"
    assert manifest["rig"]["animations"] == ["Idle", "Walk"]
    assert manifest["rig"]["loop_required"] is True
    assert manifest["audio"]["default_lookup"] == "speech"
    assert "speech" in manifest["audio"]["allowed_lookups"]
    assert manifest["review"]["overall_status"] == "needs_review"


def test_write_quaternius_human_asset_manifest(tmp_path):
    from quaternius_human_assets import (
        discover_quaternius_human_assets,
        write_quaternius_human_asset_manifest,
    )

    zip_path = tmp_path / "humans.zip"
    registry_root = tmp_path / "registry"
    _write_quaternius_zip(zip_path)
    source = discover_quaternius_human_assets(zip_path)[0]

    out = write_quaternius_human_asset_manifest(
        source,
        registry_root=registry_root,
        created_at="2026-07-09T00:00:00+00:00",
    )

    assert out == registry_root / "human" / source.family / source.asset_id / "asset.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["asset_id"] == source.asset_id
    assert data["review"]["texture_status"] == "needs_review"
