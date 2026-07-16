#!/usr/bin/env python3

import ast
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO/"tools"/"blender_fit_hy3d_to_rocketbox_template.py"


def script_source():
    assert SCRIPT.is_file(), f"missing stable-template builder: {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def compact_source():
    return "".join(script_source().split())


def function_source(name):
    source = script_source()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {name}")


def module_constant(name):
    tree = ast.parse(script_source())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing module constant {name}")


def test_builder_has_exact_template_fit_contract_and_no_generated_skin_transfer():
    source = script_source()

    assert module_constant("BINDING_MODE") == "stable_rocketbox_template_fit_v1"
    assert module_constant("USAGE_SCOPE") == "technical_spike_only"
    assert module_constant("FIT_STRENGTH") == 0.35
    assert module_constant("FIT_MAX_HEIGHT_RATIO") == 0.035
    assert module_constant("FIT_SMOOTH_ITERATIONS") >= 1
    assert "import human_template_fit" in source
    assert "import blender_bind_hy3d_to_rocketbox as direct" in source
    assert "source_vertex_regions_from_weights" in source
    assert "target_regions_from_capsules" in source
    assert "bind_target_mesh(" not in source
    assert "transfer_human_weights(" not in source
    assert "collapse_finger_weights_to_palms(" not in source
    assert "automatic_weights" not in source.lower()


def test_hunyuan_guide_uses_corrected_reviewed_axes_and_never_becomes_runtime_mesh():
    source = script_source()
    importer = function_source("import_clean_hy3d_guide")
    build = function_source("build_stable_template")

    assert 'forward_axis="NEGATIVE_Z"' in importer
    assert 'up_axis="Y"' in importer
    assert "validate_import_basis_matrix" in importer
    assert "cleanup_import_ground_artifacts" in importer
    assert "cleanup_target_geometry" in importer
    assert "remove_hy3d_guide" in build
    assert "source_mesh" in build
    assert "runtime_mesh = source_mesh" in build


def test_template_contract_captures_and_rechecks_topology_weights_uvs_and_materials():
    capture = function_source("capture_template_contract")
    validate = function_source("validate_template_contract")

    for token in (
        "vertex_count",
        "polygon_count",
        "loop_vertex_indices_sha256",
        "polygon_vertex_indices_sha256",
        "skin_contract",
        "uv_layers",
        "material_slots",
    ):
        assert token in capture
    assert "before != after" in validate
    assert "stable Rocketbox template contract changed" in validate


def test_surface_fit_is_region_locked_xy_only_bounded_smoothed_and_opacity_fixed():
    fit = function_source("fit_template_surface")
    trees = function_source("region_face_trees")
    compact = "".join(fit.split())

    assert "BVHTree.FromPolygons" in trees
    assert "HumanRegion" in fit
    assert "region_trees" in fit
    assert "find_nearest" in function_source("find_region_nearest")
    assert "clamp_xy_displacements" in fit
    assert "smooth_xy_displacements" in fit
    assert "FIT_MAX_HEIGHT_RATIO" in fit
    assert "FIT_STRENGTH" in fit
    assert "opacity_vertex_mask" in fit
    assert "displacements[:,2]=0.0" in compact
    assert "fitted[:,2]=template_vertices[:,2]" in compact


def test_projection_uses_compatible_region_face_uvs_and_material_scoped_outputs():
    correspondence = function_source("build_region_locked_source_uvs")
    projection = function_source("project_template_pbr")

    assert "template_regions" in correspondence
    assert "guide_regions" in correspondence
    assert "region_trees" in correspondence
    assert "triangle_barycentric_3d" in correspondence
    assert "guide_uv_layer" in correspondence
    assert "source_uvs" in correspondence
    assert "for material_index in" in projection
    assert "rasterize_uv_triangle" in function_source(
        "rasterize_template_region_labels"
    )
    assert "dilate_unpainted" in projection
    assert "diffuse" in projection
    assert module_constant("PBR_ROLES") == ("diffuse", "metallic", "roughness")
    assert "body" in projection and "head" in projection
    assert "opacity" in projection


def test_runtime_diffuse_uses_region_palette_over_official_rocketbox_detail():
    palette = function_source("guide_region_palette")
    labels = function_source("rasterize_template_region_labels")
    projection = function_source("project_template_pbr")

    assert "source_uvs" in palette
    assert "template_regions" in palette
    assert "region_palette_from_uv_samples" in palette
    assert "HumanRegion" in palette
    assert "template_regions" in labels
    assert "material_index" in labels
    assert "regularize_region_labels_by_island" in labels
    assert "official_color" in projection
    assert "recolor_regions_preserve_luminance" in projection
    assert "guide_region_palette" in projection
    assert "del source_uvs" not in projection


def test_opacity_material_keeps_official_alpha_and_all_pbr_images_are_packed():
    material = function_source("install_projected_materials")
    surface = function_source("install_surface_pbr")
    validate = function_source("validate_projected_materials")

    assert "reconnect_official_materials" in material
    assert "official_opacity_color" in material
    assert "Alpha" in script_source()
    assert "Base Color" in surface
    assert "Metallic" in surface
    assert "Roughness" in surface
    assert "normal_path" in surface
    assert "bpy.data.images.load" in surface
    assert "check_existing=False" in surface
    assert ".reload()" in material
    assert ".pixels[0]" in material
    assert ".pixels[0]" in validate
    assert ".pack()" in material
    assert "packed_file" in validate
    assert "has_data" in validate
    assert "material_slot_count" in validate


def test_action_floor_normalization_is_per_action_bilateral_and_hard_capped():
    source = script_source()
    normalize = function_source("normalize_action_floor_contact")

    assert module_constant("MAX_ACTION_FLOOR_OFFSET_M") == 0.05
    assert module_constant("MAX_POST_NORMALIZE_PENETRATION_M") == 0.01
    assert module_constant("FOOT_SUPPORT_TOLERANCE_M") == 0.015
    assert '"Bip01 L Foot"' in source
    assert '"Bip01 L Toe0"' in source
    assert '"Bip01 R Foot"' in source
    assert '"Bip01 R Toe0"' in source
    assert "evaluated_mesh_world_points" in function_source("sample_floor_state")
    assert "for side in" in normalize
    assert "floor_z_m" in normalize
    shift = function_source("shift_action_object_z")
    assert 'data_path="location"' in shift
    assert "array_index == 2" in shift
    assert "index=2" in shift
    assert "MAX_ACTION_FLOOR_OFFSET_M" in normalize
    assert "MAX_POST_NORMALIZE_PENETRATION_M" in normalize
    assert "FOOT_SUPPORT_TOLERANCE_M" in normalize


def test_builder_keeps_exact_walk_adds_only_idle_and_exports_two_action_glbs():
    build = function_source("build_stable_template")

    assert "walk_neutral_retarget" in build
    assert "bake_idle_action" in build
    assert "validate_two_actions" in build
    assert 'output_dir/"bound_walk.glb"' in build
    assert 'output_dir/"bound_idle.glb"' in build
    assert "export_single_action_glb" in build
    assert "roundtrip_validate" in build
    assert "walk_expected_skin" in build
    assert "idle_expected_skin" in build
    assert "walk_expected_skin," in build
    assert "idle_expected_skin," in build


def test_manifest_is_renderer_compatible_hash_locked_and_marks_technical_scope():
    manifest = function_source("build_template_manifest")

    assert '"schema_version":"hy3d_rocketbox_bind_v1"' in "".join(manifest.split())
    assert '"binding_mode":BINDING_MODE' in "".join(manifest.split())
    assert '"usage_scope":USAGE_SCOPE' in "".join(manifest.split())
    for token in (
        "bound.blend",
        "bound_walk.glb",
        "bound_idle.glb",
        "bind_metrics.json",
        "cleaned.obj",
        "reference.png",
        "consumed_inputs",
        "axis_contract",
    ):
        assert token in manifest
    assert "atomic_write_json" in function_source("build_stable_template")


def test_template_contract_is_frozen_before_roundtrip_resets_the_blender_scene():
    build = function_source("build_stable_template")

    capture_index = build.index("template_contract_after = capture_template_contract")
    roundtrip_index = build.index("roundtrip_validate")
    assert capture_index < roundtrip_index
    assert '"template_contract_after": template_contract_after' in build


def test_failure_invalidates_all_readiness_and_input_snapshot_is_always_removed():
    run = function_source("run_template_fit")
    build = function_source("build_stable_template")
    main = function_source("main")

    assert "invalidate_readiness" in main
    assert "stage_input_snapshot" in run
    assert "finally:" in run
    assert "cleanup_input_snapshot" in run
    assert "verify_source_hashes_current" in build
