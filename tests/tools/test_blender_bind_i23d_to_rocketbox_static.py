#!/usr/bin/env python3

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "blender_bind_i23d_to_rocketbox.py"


def script_source() -> str:
    assert SCRIPT.is_file(), f"missing I23D direct binder: {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def function_source(name: str) -> str:
    source = script_source()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {name}")


def module_constant(name: str):
    tree = ast.parse(script_source())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing module constant {name}")


def test_cli_requires_authenticated_i23d_inputs_and_pinned_pixal_contract():
    parser = function_source("parse_args")
    for flag in (
        "--guide-glb",
        "--guide-manifest",
        "--guide-backend",
        "--front-axis",
        "--reference-rgba",
    ):
        assert flag in parser
    assert "validate_i23d_inputs" in script_source()
    assert "i23d_rocketbox_contract" in script_source()


def test_import_keeps_only_the_dominant_textured_mesh_and_its_pbr():
    importer = function_source("import_i23d_target")

    assert "bpy.ops.import_scene.gltf" in importer
    assert "select_primary_textured_mesh" in importer
    assert "Matrix.Rotation" in importer
    assert "front_axis" in importer
    assert "material_slots" in importer
    assert "uv_layers" in importer
    assert "remove" in importer
    assert "apply_direct_bind_decimation" in importer
    assert "move_target_to_armature_space" in importer
    assert "cleanup_target_geometry" in importer


def test_direct_binding_preserves_pixal_runtime_mesh_instead_of_projecting_template():
    build = function_source("build_direct_binding")
    source = script_source()

    assert "imported_target = import_i23d_target" in build
    assert "runtime_mesh = imported_target[0]" in build
    assert "bind_target_mesh" in build
    assert "remove_original_body" in build
    assert "runtime_mesh" in build
    assert "project_template_pbr" not in source
    assert "official_color" not in source
    assert "assign_hunyuan_pbr_material" not in source
    assert module_constant("BINDING_MODE") == "direct_i23d_mesh_to_rocketbox_v1"


def test_direct_bind_is_budgeted_and_publishes_research_license_provenance():
    decimation = function_source("apply_direct_bind_decimation")
    manifest = function_source("build_bind_manifest")

    assert module_constant("DIRECT_BIND_FACE_BUDGET") == 120000
    assert "DECIMATE" in decimation
    assert "uv_layers" in decimation
    assert "material_slots" in decimation
    assert "guide_backend" in manifest
    assert "usage_scope" in manifest
    assert "research_release_ok" in manifest
    assert "permissive_commercial_ok" in manifest


def test_bound_blend_and_glbs_revalidate_pbr_skin_and_actions():
    build = function_source("build_direct_binding")

    assert "validate_i23d_pbr_material" in build
    assert "validate_bound_weights" in build
    assert "validate_two_actions" in build
    assert "roundtrip_validate" in build
    assert "inspect_bound_glb" in build
    assert "bound_walk.glb" in build
    assert "bound_idle.glb" in build
