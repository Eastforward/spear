from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "blender_recolor_semantic_materials.py"
)


def test_recolor_requires_allowlist_and_preserves_rig_geometry():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_semantic_material_color_variant_v1" in text
    assert "--allowed-material" in text
    assert "material assignments outside allowlist" in text
    assert "srgb_to_linear" in text
    assert '"geometry_modified": False' in text
    assert '"skeleton_modified": False' in text
    assert '"weights_modified": False' in text
    assert "export_skins=True" in text
    assert "bpy.ops.object.modifier_apply" not in text
