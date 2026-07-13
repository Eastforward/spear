from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "blender_transfer_compatible_animal_actions.py"
)


def test_compatible_action_transfer_keeps_target_geometry_and_checks_bones():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_compatible_animal_action_transfer_v1" in text
    assert "missing_target_bones" in text
    assert 'OPTIONAL_MISSING_BONE_PREFIXES = ("Tail",)' in text
    assert "drop_optional_missing_bone_curves" in text
    assert "required_missing_bones" in text
    assert '"geometry_and_weights_authority": True' in text
    assert '"geometry_used": False' in text
    assert '"weights_used": False' in text
    assert 'add_action_track(target_armature, target_idle, "Idle")' in text
    assert 'add_action_track(target_armature, transferred_walk, "Walking")' in text
    assert "use_selection=True" in text
    assert "export_skins=True" in text
    assert "bpy.ops.object.modifier_apply" not in text
