from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_build_generated_animal_motion_basis_preview.py"
)


def test_preview_is_pre_animation_skeleton_only_gate():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_generated_animal_motion_basis_preview_v1" in text
    assert "CARDINAL_MOTION_BASIS_YAWS" in text
    assert 'for side_chain_mode in ("matched", "swapped")' in text
    assert "mesh_points" in text
    assert "target_rest_segments" in text
    assert "candidate_id" in text
    assert "source_motion_forward" in text
    assert "motion_basis_yaw_deg" in text
    assert "formal_dataset_registration_authorized" in text
    assert "export_scene" not in text
    assert "keyframe_insert" not in text


def test_preview_never_generates_or_exports_target_animation():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "pre_animation_review_only" in text
    assert "target_animation_generated" in text
    assert "False" in text
    assert "source_pose_world" in text
    assert "source_rest_world" in text
    assert "basis_rotation" in text
