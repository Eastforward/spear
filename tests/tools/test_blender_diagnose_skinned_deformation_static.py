from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_diagnose_skinned_deformation.py"
)


def test_diagnostic_is_read_only_and_records_exact_edge_weights():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--input"' in text
    assert '"--output"' in text
    assert '"--front-axis"' in text
    assert "evaluated_geometry" in text
    assert "edge_index" in text
    assert "vertex_index" in text
    assert "influences" in text
    assert "bpy.ops.export_scene" not in text


def test_diagnostic_separates_topology_bridges_from_mixed_weights():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "cross_limb_dominant_edge" in text
    assert "vertices_mixing_multiple_limb_chains_count" in text
    assert "mixed_limb_chains" in text
    assert "topology separation or regeneration" in text
    assert "weight sanitation may be sufficient" in text


def test_diagnostic_uses_geometry_inferred_quadruped_semantics():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "infer_quadruped_semantics" in text
    assert "front_side_negative" in text
    assert "front_side_positive" in text
    assert "hind_side_negative" in text
    assert "hind_side_positive" in text
