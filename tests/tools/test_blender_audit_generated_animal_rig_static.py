from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools/blender_audit_generated_animal_rig.py"


def test_generated_animal_rig_audit_is_name_independent_and_fail_closed():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'SCHEMA = "avengine_generated_animal_rig_audit_v1"' in text
    assert '"positive-x", "negative-x", "positive-y", "negative-y"' in text
    assert "glTF_not_exported" in text
    assert "per_bone_support" in text
    assert "maximum_weight_sum_error" in text
    assert "low_leaf_endpoint_candidates" in text
    assert '"animation_authorized": False' in text
    assert '"formal_dataset_registration_authorized": False' in text


def test_generated_animal_rig_audit_refuses_overwrite():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "if output.exists() or output.is_symlink()" in text
    assert "refusing to replace output" in text
