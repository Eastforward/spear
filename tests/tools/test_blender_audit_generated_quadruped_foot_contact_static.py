from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_audit_generated_quadruped_foot_contact.py"
)


def test_generated_quadruped_foot_contact_audit_is_semantic_and_fail_closed():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'SCHEMA = "avengine_generated_quadruped_foot_contact_audit_v1"' in text
    assert "infer_quadruped_semantics" in text
    assert "weighted_skin_bone_names" in text
    assert '"excluded_non_skin_control_bones"' in text
    assert '"--reference-audit"' in text
    assert "passed_reference_calibrated_foot_contact_proxy" in text
    assert '"reference_audit"' in text
    assert "foot_vertical_delta_from_own_rest" in text
    assert "evaluated_mesh_floor_delta_from_rest" in text
    assert "rejected_ground_penetration" in text
    assert "rejected_all_feet_airborne" in text
    assert '"automatic_fine_yaw": False' in text
    assert '"formal_dataset_registration_authorized": False' in text


def test_generated_quadruped_foot_contact_audit_refuses_overwrite():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "if output.exists() or output.is_symlink()" in text
    assert "refusing to replace output" in text
