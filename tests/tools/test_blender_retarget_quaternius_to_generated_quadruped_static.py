from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_retarget_quaternius_to_generated_quadruped.py"
)


def test_generated_quadruped_retarget_uses_semantics_and_world_space_transfer():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'SCHEMA = "avengine_generated_quadruped_retarget_v1"' in text
    assert "infer_quadruped_semantics" in text
    assert "source_sampling_plan" in text
    assert "target_chain_fraction" in text
    assert "source_first.slerp(source_second" in text
    assert "source_rest_world" in text
    assert "target_rest_world" in text
    assert '"local_bone_roll_copied": False' in text
    assert '"formal_dataset_registration_authorized": False' in text


def test_generated_quadruped_retarget_preserves_target_authority_and_no_replace():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"mesh_pbr_skeleton_and_weights_authority": True' in text
    assert "if path.exists() or path.is_symlink()" in text
    assert "refusing to replace" in text
    assert '"geometry_used": False' in text
    assert '"weights_used": False' in text


def test_generated_quadruped_retarget_supports_only_cardinal_front_axes():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"positive-y"' in text
    assert '"negative-y"' in text
    assert '"negative-y": 90.0' in text
    assert '"positive-y": -90.0' in text
