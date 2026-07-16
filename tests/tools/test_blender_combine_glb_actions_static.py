"""Static contracts for the two-action humanoid GLB combiner."""

import ast
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "blender_combine_glb_actions.py"
)


def _tree():
    return ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))


def _function_source(name):
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {name}")


def test_import_filters_blender_gltf_helpers_from_runtime_meshes():
    importer = _function_source("_import_glb")

    assert "ARMATURE" in importer
    assert "mesh.parent is armature" in importer
    assert "modifier.type == \"ARMATURE\"" in importer
    assert "skinned_meshes" in importer


def test_export_is_selection_only_and_removes_base_and_append_helpers():
    main = _function_source("main")

    assert "base_objects" in main
    assert "_remove_non_runtime_objects" in main
    assert "_select_runtime_objects" in main
    assert "use_selection=True" in main
    assert "export_extra_animations=True" in main


def test_combiner_keeps_exact_requested_action_names():
    stash = _function_source("_stash_action_on_armature")
    clear = _function_source("_clear_imported_nla_tracks")
    main = _function_source("main")

    assert "action.name = name" in stash
    assert "animation_data.nla_tracks.remove(track)" in clear
    assert "_clear_imported_nla_tracks(base_armature)" in main
    assert "args.base_action_name" in main
    assert "args.append_action_name" in main
    assert "base_armature.animation_data.action = None" in main
    assert "export_animation_mode=\"NLA_TRACKS\"" in main
