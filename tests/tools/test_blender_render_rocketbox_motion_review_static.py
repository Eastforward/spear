#!/usr/bin/env python3

#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

import ast
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO/"tools"/"blender_render_rocketbox_motion_review.py"

CANONICAL_MEDIA = {
    "front": "front.mp4",
    "side": "side.mp4",
    "top": "top.mp4",
    "joints": "joints.mp4",
    "feet": "feet.mp4",
    "source_target": "source_target.mp4",
    "contact_sheet": "contact_sheet.png",
}


def renderer_source():
    assert SCRIPT.is_file()
    return SCRIPT.read_text(encoding="utf-8")


def compact_source():
    return "".join(renderer_source().split())


def module_constant(name):
    tree = ast.parse(renderer_source())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing module constant {name}")


def function_source(name):
    source = renderer_source()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {name}")


def pure_function(name, constants):
    namespace = {"np": np}
    namespace.update(constants)
    exec(function_source(name), namespace)
    return namespace[name]


def test_script_exposes_only_the_pinned_task_4_cli():
    source = renderer_source()

    for option in ("--asset-id", "--retarget-dir", "--source-motion-fbx"):
        assert option in source
    for old_option in ("--avatar-fbx", "--texture-dir", "--texture-prefix"):
        assert old_option not in function_source("parse_args")


def test_script_declares_exact_canonical_media_and_video_contract():
    source = renderer_source()

    assert module_constant("CANONICAL_MEDIA") == CANONICAL_MEDIA
    assert module_constant("VIDEO_MEDIA") == tuple(CANONICAL_MEDIA)[:-1]
    assert module_constant("VIDEO_SIZE") == (1280, 720)
    assert module_constant("FPS") == 30
    assert module_constant("GAIT_CYCLE_COUNT") == 2
    assert 'scene.render.ffmpeg.format = "MPEG4"' in source
    assert 'scene.render.ffmpeg.codec = "H264"' in source


def test_script_invalidates_readiness_before_loading_blend_and_stages_unique_media():
    source = renderer_source()
    main = function_source("main")

    load_index = main.index("load_stage_manifest(args)")
    invalidation_index = main.index("invalidate_review_readiness(args.retarget_dir)")
    blend_index = main.index("load_retarget_blend(")
    assert load_index < invalidation_index < blend_index
    assert 'READINESS_FILES = ("retarget_manifest.json", "motion_review.json")' in source
    assert "tempfile.NamedTemporaryFile(" in source
    assert "dir=final_path.parent" in source
    assert "os.replace(" in source
    assert 'with_suffix(final_path.suffix + ".tmp")' not in source
    assert "motion_review.json" not in function_source("publish_manifest")


def test_script_loads_but_never_saves_the_target_only_blend():
    source = renderer_source()

    assert "bpy.ops.wm.open_mainfile(" in source
    assert "retarget.blend" in source
    assert "save_as_mainfile" not in source
    assert "save_mainfile" not in source
    assert "target_initial_matrix" in source
    assert "target_initial_matrix.freeze()" in source
    assert "target.matrix_world =" not in source


def test_scene_setup_creates_the_world_missing_from_the_target_only_blend():
    source = compact_source()

    assert "ifscene.worldisNone:" in source
    assert 'scene.world=bpy.data.worlds.new("rocketbox_motion_review_world")' in source


def test_script_authenticates_glb_and_task_3_invariants_before_rendering():
    source = renderer_source()

    assert 'manifest["immutable_input_hashes"]["retarget_glb"]' in source
    assert "sha256_file(glb_path)" in source
    for field in (
        "mapped_80_of_80",
        "hierarchy_mismatch_count",
        "target_mesh_unchanged",
        "official_textures_attached",
        "glb_roundtrip_passed",
        "glb_skin_weights_preserved",
        "glb_material_bindings_preserved",
        "endpoint_direction_dot_negative_y",
        "maximum_facing_reconstruction_error",
        "normalized_bone_delta_residual",
    ):
        assert field in source
    assert "validate_task3_inputs(" in source


def test_loaded_texture_names_normalize_blender_file_extensions():
    source = function_source("validate_loaded_target")

    assert "Path(image.name).stem" in source


def test_script_renders_path_views_and_root_follow_diagnostics_without_target_keys():
    source = renderer_source()

    assert '"front": "path"' in source
    assert '"side": "path"' in source
    assert '"top": "path"' in source
    assert '"joints": "root_follow"' in source
    assert '"feet": "root_follow"' in source
    assert "Bip01 Pelvis" in source
    assert 'driver_add("location")' in source
    assert "target.keyframe_insert" not in source
    assert "target.location =" not in source
    assert "target.rotation" not in source


def test_diagnostic_sticks_connect_parent_and_child_joint_heads():
    source = compact_source()
    links = module_constant("STICK_LINKS")

    assert len(links) == 21
    assert ("Bip01 Pelvis", "Bip01 Spine") in links
    assert ("Bip01 L Calf", "Bip01 L Foot") in links
    assert ("Bip01 R Forearm", "Bip01 R Hand") in links
    assert "joint_head_world(target,parent_name)" in source
    assert "joint_head_world(target,child_name)" in source
    assert "joint_tail_world(target,name)" not in source


def test_floor_is_visible_in_every_view_and_diagnostic_framing_has_margin():
    source = renderer_source()

    assert 'visible_groups = ("floor",)' in source
    assert "bpy.context.view_layer.update()" in function_source("set_path_camera")
    assert module_constant("PATH_BODY_ORTHO_SCALE") >= 4.0
    assert module_constant("JOINTS_ORTHO_SCALE") >= 2.6
    assert module_constant("SOURCE_TARGET_ORTHO_SCALE") >= 4.0
    camera_source = "".join(function_source("configure_view_camera").split())
    assert "max(camera.data.ortho_scale,PATH_BODY_ORTHO_SCALE)" in camera_source


def test_contact_sheet_labels_are_burned_in_with_ffmpeg_drawtext():
    source = renderer_source()
    burn_in = function_source("burn_in_label")
    render_video = function_source("render_video")
    render_sheet = function_source("render_contact_sheet")

    assert module_constant("LABEL_FONT_PATH").endswith("DejaVuSans-Bold.ttf")
    assert "drawtext=" in burn_in
    assert "fontcolor=0x0dff26" in burn_in
    assert 'x_expression="32"' in burn_in
    assert 'y_expression="24"' in burn_in
    assert "f\"x={x_expression}:y={y_expression}:\"" in burn_in
    assert "-frames:v" in burn_in
    assert "libx264" not in burn_in
    assert "os.replace(" in burn_in
    assert "burn_in_label(" not in render_video
    assert render_sheet.index("generated.save()") < render_sheet.index(
        "burn_in_contact_sheet_labels("
    )
    assert "drawtext=" not in render_video


def test_contact_sheet_burns_one_complete_axis_label_per_tile():
    source = function_source("burn_in_contact_sheet_labels")

    assert "CONTACT_FRAME_COUNT" in source
    assert "CONTACT_TILE_SIZE" in source
    assert "CONTACT {index + 1}/{CONTACT_FRAME_COUNT}" in source
    assert 'x_expression=str(column*width + 18)' in source
    assert "UP +Z" in renderer_source()
    assert "FRONT -Y" in renderer_source()
    assert "burn_in_label(" in source


def test_label_pixel_gate_rejects_blank_and_border_clipped_overlays():
    validate = pure_function(
        "validate_label_pixels",
        {
            "LABEL_GREEN_FLOOR": 150,
            "LABEL_RED_CEILING": 110,
            "LABEL_BLUE_CEILING": 170,
            "LABEL_GREEN_DOMINANCE": 45,
            "LABEL_MIN_PIXELS": 40,
            "LABEL_BORDER_MARGIN_PX": 4,
        },
    )
    pixels = np.full((720, 1280, 3), 30, dtype=np.uint8)

    with pytest.raises(RuntimeError):
        validate(pixels, "blank")

    pixels[28:38, 30:230] = (20, 245, 60)
    result = validate(pixels, "complete")
    assert result["label_pixel_count"] == 2000
    assert result["bounds_xyxy"] == [30, 28, 229, 37]

    pixels = np.full((720, 1280, 3), 30, dtype=np.uint8)
    pixels[28:100, 30:500] = 245
    with pytest.raises(RuntimeError):
        validate(pixels, "white avatar block without label")

    pixels = np.full((720, 1280, 3), 30, dtype=np.uint8)
    pixels[0:10, 0:200] = (20, 245, 60)
    with pytest.raises(RuntimeError):
        validate(pixels, "border clipped")


def test_top_arrow_pixel_gate_requires_visible_red_evidence():
    validate = pure_function(
        "validate_top_arrow_pixels",
        {
            "ARROW_RED_FLOOR": 150,
            "ARROW_RED_DOMINANCE": 80,
            "ARROW_MIN_PIXELS": 80,
        },
    )
    pixels = np.full((720, 1280, 3), 30, dtype=np.uint8)

    with pytest.raises(RuntimeError):
        validate(pixels)

    pixels[100:110, 800:820, 0] = 240
    pixels[100:110, 800:820, 1:] = 100
    assert validate(pixels)["red_pixel_count"] == 200

    pixels = np.full((720, 1280, 3), 30, dtype=np.uint8)
    pixels[100:200, 300:500] = (240, 100, 100)
    with pytest.raises(RuntimeError):
        validate(pixels)


def test_real_media_validation_checks_contact_labels_and_top_arrow():
    source = function_source("validate_staged_media")

    assert "extract_rgb_frame" in source
    assert "validate_top_arrow_pixels" in source
    assert "validate_contact_sheet_labels" in source


def test_script_adds_required_labels_root_path_and_red_front_arrow():
    source = renderer_source()

    assert module_constant("AXIS_LABELS") == ("UP +Z", "FRONT -Y")
    assert "burn_in_contact_sheet_labels(" in source
    assert "add_root_path(" in source
    assert "add_front_arrow(" in source
    assert "FRONT -Y" in source
    assert "(1.0, 0.02, 0.02, 1.0)" in source
    assert "set_view_guides_visibility(" in source
    assert "x = maximum.x + 0.35" in source
    assert "maximum.x + TOP_GUIDE_MARGIN_X" in source


def test_script_scans_evaluated_mesh_over_one_original_cycle_for_one_fixed_floor():
    source = compact_source()

    assert "evaluated_get(depsgraph)" in source
    assert "to_mesh()" in source
    assert "to_mesh_clear()" in source
    assert "forframeinrange(frame_start,frame_end+1)" in source
    assert "review_floor_z=min(" in source
    assert '"reviewed_static_floor_z_m"' in source
    assert '"review_floor_z_m"' in source
    assert '"runtime_constant_z_lift_m"' in source
    assert '"mesh_clearance_m"' in source
    assert '"maximum_penetration_below_review_floor_m"' in source
    assert "vertical_correction" not in source


def test_script_reports_foot_contact_slide_and_visible_evidence_authority():
    source = renderer_source()

    for field in (
        "contact_frame_count",
        "maximum_contact_xy_velocity_m_per_s",
        "accumulated_contact_slide_m",
        "visible_foot_evidence_authoritative",
        "feet.mp4",
        "contact_sheet.png",
    ):
        assert field in source


def test_source_target_uses_synchronized_source_stick_bones_without_helpers_or_nubs():
    source = renderer_source()

    assert "bpy.ops.import_scene.fbx(" in source
    assert "source_frame_for_review_frame(" in source
    assert '"Nub" not in bone.name' in source
    assert "MotionExtractionHelper" in source
    assert "source_stick" in source
    assert "add_source_stick_skeleton(" in source
    assert "source_target" in source


def test_script_validates_ffprobe_and_nonblank_media_before_manifest_publication():
    source = renderer_source()
    main = function_source("main")

    assert "ffprobe" in source
    assert "codec_name" in source
    assert "r_frame_rate" in source
    assert "signalstats" in source
    assert "PIL" not in source
    assert "validate_video(" in source
    assert "validate_png(" in source
    assert main.index("validate_staged_media(") < main.index("publish_manifest(")


def test_final_manifest_preserves_task_1_schema_and_adds_detailed_checks():
    source = renderer_source()

    assert 'manifest["schema_version"] == "rocketbox_retarget_manifest_v1"' in source
    assert 'manifest["media"] = dict(CANONICAL_MEDIA)' in source
    assert 'manifest["automatic_checks"] = automatic_checks' in source
    assert '"overall": "passed"' in source
    assert '"stage": "review_ready"' not in source
    for field in (
        "task3_invariants",
        "retarget_glb_current_hash",
        "direction",
        "rest_pose_mapping_roundtrip",
        "material_binding",
        "weight_binding",
        "fixed_floor",
        "foot_contact_and_slide",
        "media_validation",
    ):
        assert field in source
    assert "immutable_input_hashes" not in function_source("publish_manifest")
    assert "binding" not in function_source("publish_manifest")
    assert "artifacts" not in function_source("publish_manifest")


def test_new_python_files_follow_repository_style_contract():
    copyright_lines = (
        "Copyright (c) 2025 The SPEAR Development Team",
        "Copyright (c) 2022 Intel",
    )
    forbidden_strings = (
        "# " + "noqa",
        "from dataclasses import " + "dataclass",
        "@data" + "class",
        "ArgumentParser(" + "description=",
        "he" + "lp=",
    )
    for path in (SCRIPT, Path(__file__)):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for line in copyright_lines:
            assert line in source
        for forbidden in forbidden_strings:
            assert forbidden not in source
        assert not any(isinstance(node, ast.AnnAssign) for node in ast.walk(tree))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.returns is None
                args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                assert all(arg.annotation is None for arg in args)
