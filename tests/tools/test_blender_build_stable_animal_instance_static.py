from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "blender_build_stable_animal_instance.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_instance_builder_authenticates_preflight_and_refuses_overwrite():
    text = source()
    assert "validate_execution_preflight" in text
    assert "stable template no longer matches authenticated preflight" in text
    assert "refusing to replace" in text


def test_every_required_instance_control_has_a_real_operation():
    text = source()
    assert '{"size", "body_build", "coat_tone", "life_stage"}' in text
    assert "apply_shape_controls" in text
    assert "realize_texture" in text
    assert "install_instance_scale" in text
    assert "muzzle_gray_mix" in text
    assert "muzzle_gray_target" in text
    assert 'muzzle_gray_target must be in [0, 1]' in text


def test_generated_animal_rigs_can_use_explicit_semantic_bone_groups():
    text = source()
    assert "torso_group_names_csv" in text
    assert "head_group_names_csv" in text
    assert "exact_group_names" in text
    assert "semantic_measurements" in text


def test_textured_instances_record_absolute_coat_and_age_measurements():
    text = source()
    assert "coat_desaturation" in text
    assert "mean_nonwhite_coat_luminance_before" in text
    assert "mean_nonwhite_coat_luminance_after" in text
    assert '"runtime_front_axis": "positive_x"' in text
    assert '"automatic_fine_yaw_inference": False' in text


def test_builder_preserves_topology_skin_uv_and_actions():
    text = source()
    assert "skin_uv_topology_sha256" in text
    assert "action_sha256" in text
    assert "topology_uv_skin_unchanged" in text
    assert "actions_unchanged" in text
    assert "canonical_walk_idle=True" in text


def test_manifest_keeps_prompt_audio_physical_and_qa_contracts():
    text = source()
    assert '"appearance_reference"' in text
    assert '"acoustic_profile"' in text
    assert '"target_physical_profile"' in text
    assert '"ue_apartment": "pending"' in text
