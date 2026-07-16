from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2] / "tools" / "blender_build_rocketbox_batch_runtime.py"
)


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _function(name: str) -> str:
    source = _source()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(name)


def test_batch_builder_is_syntax_valid_and_selects_one_inventory_avatar():
    tree = ast.parse(_source())
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert {"load_avatar_contract", "relink_and_pack_textures", "build_runtime", "main"} <= names
    source = _source()
    assert "--inventory-json" in source
    assert "--base-avatar-id" in source
    assert "--output-dir" in source
    assert "rocketbox_human_inventory_v1" in source


def test_batch_builder_uses_gender_matched_walk_idle_and_semantic_bone_families():
    source = _source()
    assert "m_walk_neutral.max.fbx" in source
    assert "f_walk_neutral.max.fbx" in source
    assert "m_idle_neutral_01.max.fbx" in source
    assert "f_idle_neutral_01.max.fbx" in source
    assert '"Bip01"' in source
    assert '"Bip02"' in source
    assert "Walking" in source
    assert "Standing_Idle" in source


def test_texture_relink_is_basename_authenticated_and_preserves_material_graphs():
    relink = _function("relink_and_pack_textures")
    assert ".name" in relink or "basename" in relink
    assert "texture_dir" in relink
    assert "sha256" in relink
    assert ".reload(" in relink
    assert "len(image.pixels)" in relink
    assert "has_data" in relink
    assert "image.size" in relink
    assert ".pack(" in relink
    assert "node.image" in relink
    assert "nodes.clear" not in relink
    assert "materials.clear" not in relink


def test_batch_builder_never_applies_scale_or_changes_geometry_and_is_no_replace():
    source = _source()
    assert "bpy.ops.object.transform_apply" not in source
    assert "authored_height_cm" in source
    assert "actor_scale" in source
    assert "source_mesh_contract" in source
    assert "authored_source_mesh_contract" in source
    assert "post_bake_mesh_contract" in source
    sanitation = _function("sanitize_non_surface_loose_vertices")
    assert "polygons" in sanitation
    assert "vertex.groups" in sanitation
    assert "bmesh.ops.delete" in sanitation
    assert "refusing to replace" in source
    assert "os.replace" in source


def test_batch_builder_requires_two_action_glb_roundtrip_and_hash_provenance():
    source = _source()
    assert "roundtrip_validate_combined" in source
    assert "inspect_runtime_glb" in source
    assert "source_fbx_sha256" in source
    assert "inventory_sha256" in source
    assert "motion_sha256" in source
    assert "automatic_checks" in source
    assert '"research_candidate"' in source
