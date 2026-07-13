from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "blender_extract_quaternius_walk_idle_glb.py"
)


def test_native_quaternius_extract_keeps_source_skin_and_only_walk_idle():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_quaternius_native_walk_idle_extract_v1" in text
    assert 'choose_native_action(source_actions, "Walk")' in text
    assert 'choose_native_action(source_actions, "Idle")' in text
    assert 'add_track(armature, walking, "Walking")' in text
    assert 'add_track(armature, idle, "Idle")' in text
    assert "use_selection=True" in text
    assert "export_skins=True" in text
    assert '"license"' in text
    assert "repair_legacy_zero_alpha" in text
    assert 'material.diffuse_color[3] = 1.0' in text
    assert 'alpha.default_value = 1.0' in text
    assert '"material_alpha_repairs"' in text
    assert "bpy.ops.object.modifier_apply" not in text
