import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _write_fbx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00")


def test_classify_mixamo_clip_roles_from_uploaded_filenames():
    from mixamo_humanoid_adapter import classify_mixamo_clip_role

    assert classify_mixamo_clip_role(Path("raw/Standing_Idle.fbx")) == "idle"
    assert classify_mixamo_clip_role(Path("raw/Standing Idle.fbx")) == "idle"
    assert classify_mixamo_clip_role(Path("raw/Walking.fbx")) == "walk"
    assert classify_mixamo_clip_role(Path("raw/Running.fbx")) == "run"
    assert classify_mixamo_clip_role(Path("raw/Readme.txt")) is None


def test_build_humanoid_import_plan_uses_idle_and_walk_fbx(tmp_path):
    from mixamo_humanoid_adapter import build_mixamo_humanoid_import_plan

    root = tmp_path / "mixamo"
    _write_fbx(root / "raw" / "Walking.fbx")
    _write_fbx(root / "raw" / "Standing_Idle.fbx")
    _write_fbx(root / "raw" / "Running.fbx")

    plan = build_mixamo_humanoid_import_plan(root)

    assert plan["state"] == "ready"
    assert plan["asset_family"] == "mixamo_humanoid"
    assert plan["skeleton_family"] == "humanoid"
    assert plan["visual_source_class"] == "human"
    assert plan["default_audio_lookup"] == "speech"
    assert set(plan["clips"]) == {"idle", "walk"}
    assert plan["clips"]["idle"]["source_name"] == "Standing_Idle"
    assert plan["clips"]["idle"]["motion_style"] == "stationary"
    assert plan["clips"]["idle"]["ue_asset_path"] == "/Game/Mixamo/Humanoid/Standing_Idle"
    assert plan["clips"]["walk"]["source_name"] == "Walking"
    assert plan["clips"]["walk"]["motion_style"] == "walking"
    assert plan["clips"]["walk"]["ue_asset_path"] == "/Game/Mixamo/Humanoid/Walking"
    assert plan["missing_roles"] == []


def test_humanoid_import_plan_reports_missing_required_roles(tmp_path):
    from mixamo_humanoid_adapter import build_mixamo_humanoid_import_plan

    root = tmp_path / "mixamo"
    _write_fbx(root / "raw" / "Standing_Idle.fbx")

    plan = build_mixamo_humanoid_import_plan(root)

    assert plan["state"] == "missing_required_clips"
    assert plan["missing_roles"] == ["walk"]
    assert set(plan["clips"]) == {"idle"}
    assert "Walking.fbx" in plan["manual_action"]


def test_humanoid_import_plan_rejects_missing_root(tmp_path):
    from mixamo_humanoid_adapter import build_mixamo_humanoid_import_plan

    with pytest.raises(FileNotFoundError, match="Mixamo"):
        build_mixamo_humanoid_import_plan(tmp_path / "missing")
