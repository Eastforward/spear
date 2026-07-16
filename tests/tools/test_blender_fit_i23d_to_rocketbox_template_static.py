#!/usr/bin/env python3

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "blender_fit_hy3d_to_rocketbox_template.py"


def script_source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def function_source(name: str) -> str:
    source = script_source()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {name}")


def test_cli_accepts_one_authenticated_i23d_glb_contract():
    parser = function_source("parse_args")
    for flag in (
        "--guide-glb",
        "--guide-manifest",
        "--guide-backend",
        "--front-axis",
        "--reference-rgba",
    ):
        assert flag in parser
    assert "mutually_exclusive_group" in parser
    assert "validate_i23d_inputs" in script_source()
    assert "i23d_rocketbox_contract" in script_source()


def test_i23d_import_canonicalizes_front_before_height_and_floor_alignment():
    importer = function_source("import_clean_i23d_guide")
    compact = "".join(importer.split())

    assert "bpy.ops.import_scene.gltf" in importer
    assert "front_axis" in importer
    assert "Matrix.Rotation" in importer
    assert "math.pi" in importer
    assert "guide.parent=None" in compact
    assert "move_target_to_armature_space" in importer
    assert "cleanup_target_geometry" in importer
    assert importer.index("Matrix.Rotation") < importer.index("cleanup_target_geometry")
    assert "Armature" not in importer


def test_i23d_pbr_reader_traces_base_color_and_unpacks_gltf_metallic_roughness():
    reader = function_source("guide_pbr_images")

    assert "BSDF_PRINCIPLED" in reader
    assert '"Base Color"' in reader
    assert '"Metallic"' in reader
    assert '"Roughness"' in reader
    assert "SEPARATE_COLOR" in reader
    assert '"Blue"' in reader
    assert '"Green"' in reader
    assert "np.repeat" in reader


def test_i23d_build_keeps_rocketbox_runtime_mesh_and_records_license_provenance():
    build = function_source("build_stable_template")
    manifest = function_source("build_template_manifest")

    assert "import_clean_i23d_guide" in build
    assert "runtime_mesh = source_mesh" in build
    assert "remove_hy3d_guide" in build
    assert "guide_backend" in manifest
    assert "research_release_ok" in manifest
    assert "permissive_commercial_ok" in manifest
    assert "usage_scope" in manifest
    assert "technical_spike_only" not in manifest


def test_i23d_inputs_are_staged_once_and_rehashed_before_publication():
    assert "authenticated_i23d_snapshot_sources" in script_source()
    run = function_source("run_template_fit")
    build = function_source("build_stable_template")

    assert "stage_i23d_input_snapshot" in run
    assert "finally:" in run
    assert "cleanup_input_snapshot" in run
    assert "verify_i23d_source_hashes_current" in build
