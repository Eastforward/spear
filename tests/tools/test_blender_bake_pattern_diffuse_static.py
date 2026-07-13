from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "blender_bake_pattern_diffuse.py"


def test_beagle_pattern_uses_stable_template_material_only():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"beagle_tricolor"' in text
    assert 'pattern_spec.get("mode") == "beagle_tricolor"' in text
    assert "export_skins=True" in text
    assert "export_animations=True" in text
    assert "bpy.ops.object.modifier_apply" not in text
    assert "bpy.ops.object.parent_set" not in text
