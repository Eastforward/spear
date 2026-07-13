from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "blender_inventory_animal_template.py"


def test_template_inventory_is_read_only_and_records_required_evidence():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_animal_template_inventory_v1" in text
    assert '"input_sha256"' in text
    assert '"license"' in text
    assert '"connected_components"' in text
    assert '"vertices_with_weights"' in text
    assert '"has_walk_action"' in text
    assert '"has_idle_action"' in text
    assert '"automatic_fine_yaw_inference": False' in text
    assert "bpy.ops.export_scene" not in text
    assert "bpy.ops.object.modifier_apply" not in text
