from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "tools" / "blender_audit_rocketbox_avatars.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_blender_audit_script_is_syntax_valid_and_shardable():
    tree = ast.parse(_source())
    names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert {"audit_avatar", "rest_mesh_bounds", "main"} <= names
    assert "--shard-index" in _source()
    assert "--shard-count" in _source()
    assert "rocketbox_blender_audit_shard_v1" in _source()


def test_blender_audit_preserves_authored_scale_and_checks_semantic_skeleton():
    source = _source()
    assert "bpy.ops.object.transform_apply" not in source
    assert "authored_height_cm" in source
    assert "armature_scale" in source
    for bone in (
        "Bip01 Pelvis",
        "Bip01 Spine",
        "Bip01 Head",
        "Bip01 L Hand",
        "Bip01 R Hand",
        "Bip01 L Foot",
        "Bip01 R Foot",
        "Bip01 L Toe0",
        "Bip01 R Toe0",
    ):
        assert bone in source
    assert '"Bip01"' in source
    assert '"Bip02"' in source
    assert "skeleton_family" in source


def test_blender_audit_records_all_meshes_and_skin_weight_failures():
    source = _source()
    assert "mesh_count" in source
    assert "material_slot_names" in source
    assert "nonfinite_weight_count" in source
    assert "unweighted_vertex_count" in source
    assert "unweighted_surface_vertex_count" in source
    assert "loose_unweighted_vertex_count" in source
    assert '"status": "passed"' in source
