from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/blender_build_stable_quadruped_instance.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_builder_uses_semantic_body_frame_not_a_fixed_model_axis():
    text = source()
    assert "semantic_head_torso_axis_v1" not in text
    assert "infer_body_frame" in text
    assert "head_center[:2] - torso_center[:2]" in text
    assert "width_xy" in text


def test_builder_preserves_native_topology_skin_and_actions():
    text = source()
    assert "mesh_contract_sha256" in text
    assert "topology_uv_skin_unchanged" in text
    assert "actions_unchanged" in text
    assert "canonical_walk_idle=True" in text
    assert "skin weight" in text.lower()


def test_builder_edits_only_profile_declared_pbr_materials():
    text = source()
    assert "coat_material_names_csv" in text
    assert "declared coat materials are missing" in text
    assert "solid_material_pbr" in text
    assert "senior_coat_desaturation" in text
    assert "muzzle_gray_mix must be in [0, 1]" in text


def test_builder_keeps_all_meshes_and_scales_the_whole_assembly():
    text = source()
    assert "one or more meshes" in text
    assert "skinned_meshes" in text
    assert "install_instance_scale(" in text
    assert '"mesh_count"' in text


def test_builder_applies_only_the_registry_cardinal_direction_transform():
    text = source()
    assert "template_cardinal_yaw_deg" in text
    assert "template cardinal yaw must be -90/0/90/180" in text
    assert "root.rotation_euler[2]" in text
    assert '"automatic_fine_yaw_inference": False' in text


def test_builder_measures_real_semantic_shape_changes_not_whole_asset_bounds():
    text = source()
    assert '"semantic_measurements"' in text
    assert '"torso_weighted_lateral_rms_after"' in text
    assert '"head_weighted_radius_rms_after"' in text
    assert "torso_lateral_square_after" in text
    assert "head_radius_square_after" in text
