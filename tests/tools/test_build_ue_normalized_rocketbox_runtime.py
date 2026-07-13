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
    return importlib.import_module("tools.build_ue_normalized_rocketbox_runtime")


def test_contracts_pin_two_existing_native_sources_and_new_v2_tags():
    builder = _builder()
    assert builder.CONTRACTS == {
        "rocketbox_male_adult_01_original_ue_v2": {
            "asset_id": "rocketbox_male_adult_01",
            "variant_id": "original_v1",
            "source_tag": "rocketbox_male_adult_01_original_v1",
            "source_relative_root": (
                "tmp/rocketbox_native_runtime_v1/"
                "rocketbox_male_adult_01_original_v1"
            ),
            "source_manifest": "build_manifest.json",
            "source_manifest_schema": "rocketbox_native_runtime_build_v1",
            "output_relative_root": (
                "tmp/rocketbox_native_runtime_ue_v2/"
                "rocketbox_male_adult_01_original_ue_v2"
            ),
        },
        "rocketbox_male_adult_01_shirt_blue_ue_v2": {
            "asset_id": "rocketbox_male_adult_01",
            "variant_id": "shirt_blue_v1",
            "source_tag": "rocketbox_male_adult_01_shirt_blue_v1",
            "source_relative_root": (
                "tmp/rocketbox_native_runtime_v1/"
                "rocketbox_male_adult_01_shirt_blue_v1"
            ),
            "source_manifest": "variant_manifest.json",
            "source_manifest_schema": "rocketbox_native_material_variant_v1",
            "output_relative_root": (
                "tmp/rocketbox_native_runtime_ue_v2/"
                "rocketbox_male_adult_01_shirt_blue_ue_v2"
            ),
        },
    }


def test_cli_can_run_by_absolute_script_path_outside_repository(tmp_path):
    script = SPEAR_ROOT / "tools/build_ue_normalized_rocketbox_runtime.py"
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


def test_bundle_publisher_is_hash_locked_grounded_and_readonly(tmp_path):
    builder = _builder()
    tag = "rocketbox_male_adult_01_original_ue_v2"
    contract = builder.CONTRACTS[tag]
    output = tmp_path / tag
    source_root = SPEAR_ROOT / contract["source_relative_root"]
    source_glb = source_root / "runtime.glb"
    source_manifest = source_root / contract["source_manifest"]
    source_glb_before = hashlib.sha256(source_glb.read_bytes()).hexdigest()
    source_manifest_before = hashlib.sha256(source_manifest.read_bytes()).hexdigest()

    manifest_path = builder.publish_bundle(tag, contract, output)

    assert manifest_path == output / "normalization_manifest.json"
    assert {path.name for path in output.iterdir()} == {
        "runtime.glb",
        "normalization_manifest.json",
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "rocketbox_native_ue_runtime_v2"
    assert manifest["tag"] == tag
    assert manifest["source_tag"] == contract["source_tag"]
    assert manifest["asset_id"] == "rocketbox_male_adult_01"
    assert manifest["variant_id"] == "original_v1"
    assert manifest["usage_scope"] == "research_candidate"
    assert manifest["formal_registration_authorized"] is False
    assert manifest["runtime_glb"]["filename"] == "runtime.glb"
    assert manifest["runtime_glb"]["sha256"] == hashlib.sha256(
        (output / "runtime.glb").read_bytes()
    ).hexdigest()
    assert manifest["source"]["runtime_glb_sha256"] == source_glb_before
    assert manifest["source"]["manifest_sha256"] == source_manifest_before
    assert manifest["normalization"]["schema"] == (
        "rocketbox_ue_grounded_metric_skeleton_normalization_v1"
    )
    assert manifest["normalization"]["normalized_joint_count"] == 80
    assert manifest["normalization"]["static_wrapper_translation_zeroed"] is True
    assert manifest["equivalence"]["mesh_position_accessors_unchanged"] is True
    assert manifest["equivalence"]["embedded_image_payloads_unchanged"] is True
    assert manifest["equivalence"]["material_texture_graph_unchanged"] is True
    assert manifest["equivalence"]["animation_names_unchanged"] is True
    assert manifest["automatic_checks"]["overall"] == "passed"

    document, _binary = read_glb_bytes((output / "runtime.glb").read_bytes())
    mesh_nodes = [
        index
        for index, node in enumerate(document["nodes"])
        if "mesh" in node and "skin" in node
    ]
    assert len(mesh_nodes) == 1
    assert mesh_nodes[0] in document["scenes"][document.get("scene", 0)]["nodes"]
    armature = next(node for node in document["nodes"] if node.get("name") == "Bip01")
    assert armature["scale"] == [1.0, 1.0, 1.0]
    assert armature["translation"] == [0.0, 0.0, 0.0]
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
    with pytest.raises(builder.UeRuntimeBundleError, match="replace"):
        builder.publish_bundle(tag, contract, output)
