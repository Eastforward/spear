"""Contracts and optional Blender smoke for persistent morphotype realization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
AVENGINE_ROOT = ROOT.parents[1]
SCRIPT = ROOT / "tools/blender_realize_quadruped_morphotype.py"
RENDERER = ROOT / "tools/blender_render_glb_animation.py"
PROFILE = (
    ROOT
    / "data/controlled_source_attributes_v1/contracts"
    / "quadruped_morphotype_guide_short_leg_tail_stump_v3.json"
)
WOLF = AVENGINE_ROOT / "assets/mesh_library/quaternius_animalpack/Wolf.glb"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_realizer_is_taxonomy_independent_and_profile_driven():
    text = source()

    assert "build_morphotype_guide_plan" in text
    assert "apply_quadruped_morphotype_guide" in text
    assert "infer_canonical_quadruped" in text
    assert "leg_length_ratio" in text
    assert "tail_length_ratio" in text
    assert "breed" not in text.lower()
    assert "corgi" not in text.lower()


def test_realizer_freezes_geometry_and_rest_pose_without_rerigging():
    text = source()

    assert "bpy.ops.pose.armature_apply(selected=False)" in text
    assert "mesh.data.vertices.foreach_set" in text
    assert "topology_uv_skin_unchanged" in text
    assert "skeleton_hierarchy_unchanged" in text
    assert "rest_skeleton_changed" in text
    assert "blender_action_curves_unchanged_before_export" in text
    assert "read_exported_glb_contract" in text
    assert "use_selection=True" in text
    assert "MAX_FREEZE_RESIDUAL_HEIGHT_RATIO = 2.0e-6" in text


def test_realizer_authenticates_inputs_and_never_authorizes_registration():
    text = source()

    assert "--input-sha256" in text
    assert "--profile-sha256" in text
    assert "os.O_NOFOLLOW" in text
    assert "stage_authenticated_input" in text
    assert "authenticated_bytes_staged_before_blender_import" in text
    assert "refusing to replace" in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert '"new_taxon_geometry_substitution_authorized": False' in text
    assert '"ue_asset_bound_readback": "pending"' in text
    assert '"dynamic_audio": "pending"' in text


def test_renderer_names_the_float32_roundtrip_tolerance_without_weakening_profile():
    text = RENDERER.read_text(encoding="utf-8")

    assert "MAX_MORPHOTYPE_MATRIX_ROUNDTRIP_ERROR = 2.0e-6" in text
    assert "profile.maximum_ground_residual_height_ratio" in text
    assert "MAX_MORPHOTYPE_MATRIX_ROUNDTRIP_ERROR" in text


def test_blender_realizes_wolf_walk_idle_without_changing_authorities(tmp_path):
    blender = shutil.which("blender")
    if blender is None:
        pytest.skip("Blender executable is unavailable on PATH")
    if not WOLF.is_file():
        pytest.skip("repository quadruped fixture is unavailable")
    output = tmp_path / "realized.glb"
    manifest = tmp_path / "realization.json"
    source_hash = sha256(WOLF)
    profile_hash = sha256(PROFILE)
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(SCRIPT),
        "--",
        "--input",
        str(WOLF),
        "--input-sha256",
        source_hash,
        "--profile",
        str(PROFILE),
        "--profile-sha256",
        profile_hash,
        "--candidate-id",
        "generic_quadruped_fixture",
        "--output-glb",
        str(output),
        "--manifest",
        str(manifest),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        env={**os.environ},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    realization = payload["realization"]
    assert payload["schema"] == "avengine_quadruped_morphotype_realization_v1"
    assert payload["source"]["sha256"] == source_hash
    assert payload["profile"]["sha256"] == profile_hash
    assert payload["formal_dataset_registration_authorized"] is False
    assert realization["topology_uv_skin_unchanged"] is True
    assert realization["skeleton_hierarchy_unchanged"] is True
    assert realization["rest_skeleton_changed"] is True
    assert realization["blender_action_curves_unchanged_before_export"] is True
    assert realization["actions"] == ["Idle", "Walking"]
    assert realization["exported_glb_readback"]["passed"] is True
    assert realization["exported_glb_readback"]["mesh_count"] == 1
    assert realization["exported_glb_readback"]["joint_count"] == 34
    assert sorted(
        realization["exported_glb_readback"]["animation_names"]
    ) == ["Idle", "Walking"]
    assert payload["qa"]["glb_readback"] == "passed"
    assert (
        realization["rest_bone_target_residual_height_ratio"]
        <= 2.0e-6
    )
    assert output.is_file()
    assert sha256(WOLF) == source_hash
