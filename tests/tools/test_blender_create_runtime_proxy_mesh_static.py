"""Static contracts for Pixal runtime LOD generation.

The Blender script imports ``bpy`` and cannot be imported by normal pytest,
so these tests deliberately verify its CLI and material-preservation contract
from source.
"""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "blender_create_runtime_proxy_mesh.py"


def test_runtime_proxy_exposes_opt_in_double_sided_materials():
    text = SCRIPT.read_text()

    assert "--double-sided" in text
    assert "_make_materials_double_sided" in text
    assert "material.use_backface_culling = False" in text
    assert "if args.double_sided:" in text


def test_runtime_proxy_keeps_texture_import_and_glb_export_path():
    text = SCRIPT.read_text()

    assert "bpy.ops.import_scene.gltf" in text
    assert 'export_format="GLB"' in text
    assert "write_runtime_proxy_record" in text


def test_runtime_proxy_welds_gltf_split_vertices_before_decimation():
    text = SCRIPT.read_text()

    weld_call = text.index("    weld = _weld_position_duplicates(meshes)")
    decimate_call = text.index("        _apply_decimate(meshes, ratio)")
    assert "bmesh.ops.remove_doubles" in text
    assert weld_call < decimate_call
    assert 'runtime_topology["boundary_edges"] > welded_topology["boundary_edges"]' in text
