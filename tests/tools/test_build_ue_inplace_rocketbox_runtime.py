from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from tools.normalize_rocketbox_glb_for_ue import read_glb_bytes


SPEAR_ROOT = Path(__file__).resolve().parents[2]


def _builder():
    return importlib.import_module("tools.build_ue_inplace_rocketbox_runtime")


def test_contracts_pin_v1_sources_and_publish_new_v3_tags():
    builder = _builder()
    assert sorted(builder.CONTRACTS) == [
        "rocketbox_female_adult_01_original_ue_v3",
        "rocketbox_male_adult_01_original_ue_v3",
        "rocketbox_male_adult_01_shirt_blue_ue_v3",
    ]
    original = builder.CONTRACTS["rocketbox_male_adult_01_original_ue_v3"]
    assert original["source_tag"] == "rocketbox_male_adult_01_original_v1"
    assert original["source_relative_root"].endswith(
        "rocketbox_male_adult_01_original_v1"
    )
    assert original["output_relative_root"].endswith(
        "rocketbox_male_adult_01_original_ue_v3"
    )
    female = builder.CONTRACTS["rocketbox_female_adult_01_original_ue_v3"]
    assert female["asset_id"] == "rocketbox_female_adult_01"
    assert female["source_tag"] == "rocketbox_female_adult_01_original_v1"
    assert female["source_manifest"] == "build_manifest.json"


def test_cli_can_run_by_absolute_script_path_outside_repository(tmp_path):
    script = SPEAR_ROOT / "tools/build_ue_inplace_rocketbox_runtime.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--tag" in completed.stdout


def test_v3_bundle_is_in_place_hash_locked_and_readonly(tmp_path):
    builder = _builder()
    tag = "rocketbox_male_adult_01_original_ue_v3"
    contract = builder.CONTRACTS[tag]
    output = tmp_path / tag
    source_root = SPEAR_ROOT / contract["source_relative_root"]
    source_glb = source_root / "runtime.glb"
    source_manifest = source_root / contract["source_manifest"]
    source_glb_before = hashlib.sha256(source_glb.read_bytes()).hexdigest()
    source_manifest_before = hashlib.sha256(source_manifest.read_bytes()).hexdigest()

    manifest_path = builder.publish_bundle(tag, contract, output)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "rocketbox_native_ue_runtime_v3"
    assert manifest["tag"] == tag
    assert manifest["normalization"]["schema"] == (
        "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
    )
    assert manifest["normalization"]["in_place_actions"] == ["Walking"]
    walking = manifest["normalization"]["root_motion"]["Walking"]
    assert abs(walking["horizontal_displacement_before_m"][1]) > 1.4
    assert walking["maximum_horizontal_deviation_after_m"] < 1e-6
    assert walking["maximum_vertical_world_error_m"] < 1e-6
    assert manifest["runtime_motion_contract"] == {
        "horizontal_world_trajectory_authority": "UE_actor_trajectory",
        "walking_embedded_horizontal_root_motion": "removed",
        "walking_vertical_motion": "preserved",
        "dynamic_ground_snap_to_floor_required": True,
    }
    assert manifest["expected_ue_qa"]["actor_scale"] == 1.0
    assert manifest["expected_ue_qa"]["height_range_cm"] == [165.0, 200.0]
    assert manifest["expected_ue_qa"]["ground_snap_max_abs_correction_cm"] == 15.0
    assert manifest["equivalence"]["mesh_position_accessors_unchanged"] is True
    assert manifest["equivalence"]["embedded_image_payloads_unchanged"] is True
    assert manifest["equivalence"]["material_texture_graph_unchanged"] is True
    assert manifest["automatic_checks"]["overall"] == "passed"

    document, _binary = read_glb_bytes((output / "runtime.glb").read_bytes())
    assert {animation["name"] for animation in document["animations"]} == {
        "Walking",
        "Standing_Idle",
    }
    assert hashlib.sha256(source_glb.read_bytes()).hexdigest() == source_glb_before
    assert hashlib.sha256(source_manifest.read_bytes()).hexdigest() == source_manifest_before
    for path in output.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (
            0o444 if path.is_file() else 0o555
        )
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    with pytest.raises(builder.UeInPlaceRuntimeBundleError, match="replace"):
        builder.publish_bundle(tag, contract, output)
