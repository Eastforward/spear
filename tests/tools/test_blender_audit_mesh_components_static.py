"""Static checks for the low-memory Blender component auditor."""

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools/blender_audit_mesh_components.py"


def test_component_auditor_uses_edge_union_find_not_triangle_split():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "def _find" in text
    assert "def _union" in text
    assert "for edge in mesh.edges" in text
    assert "trimesh" not in text


def test_component_auditor_reports_small_low_shells_against_global_bbox():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"small_low_components"' in text
    assert '"small_low_component_count"' in text
    assert 'record["vertices"] <= 1024' in text
    assert '"relative_to_asset"' in text
    assert '"largest_component_vertex_fraction"' in text
    assert '"max_z_height_fraction"' in text
