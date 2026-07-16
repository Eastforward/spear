#!/usr/bin/env python3

#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO/"tools"/"blender_render_hy3d_rocketbox_review.py"

REQUIRED_MOTIONS = ("walk", "idle")
REQUIRED_VIEWS = ("front", "side", "feet")
CANONICAL_MEDIA = {
    "walk_front": "walk_front.mp4",
    "walk_side": "walk_side.mp4",
    "walk_feet": "walk_feet.mp4",
    "idle_front": "idle_front.mp4",
    "idle_side": "idle_side.mp4",
    "idle_feet": "idle_feet.mp4",
    "contact_sheet": "bind_contact_sheet.png",
}
CANONICAL_GLBS = {
    "walk": "bound_walk.glb",
    "idle": "bound_idle.glb",
}


def artifact_snapshot(marker):
    sha256 = marker*64
    return {
        "schema_version": "hy3d_rocketbox_artifact_snapshot_v1",
        "asset_id": "rocketbox_male_adult_01",
        "bind_manifest_sha256": sha256,
        "review_manifest_sha256": sha256,
        "bound_blend": {"filename": "bound.blend", "sha256": sha256},
        "glbs": {
            "walk": {"filename": "bound_walk.glb", "sha256": sha256},
            "idle": {"filename": "bound_idle.glb", "sha256": sha256},
        },
        "videos": {
            motion: {
                view: {"filename": f"{motion}_{view}.mp4", "sha256": sha256}
                for view in REQUIRED_VIEWS
            }
            for motion in REQUIRED_MOTIONS
        },
        "bind_metrics": {"filename": "bind_metrics.json", "sha256": sha256},
        "contact_sheet": {"filename": "bind_contact_sheet.png", "sha256": sha256},
    }


def renderer_source():
    assert SCRIPT.is_file(), f"missing Task 4 renderer: {SCRIPT}"
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


def pure_function(name, constants=None):
    namespace = dict(constants or {})
    exec(function_source(name), namespace)
    return namespace[name]


def test_script_exposes_render_and_explicit_direct_attempt_record_cli():
    source = function_source("parse_args")

    assert "--asset-id" in source
    assert "--bind-dir" in source
    assert "--record-direct-attempt" in source
    assert "--pixel-qa-json" in source
    for forbidden in (
        "--retarget-dir",
        "--source-motion-fbx",
        "--avatar-fbx",
        "--texture-dir",
    ):
        assert forbidden not in source


def test_script_requires_authenticated_bound_blend_and_canonical_bound_glbs():
    source = renderer_source()
    load_inputs = function_source("load_bind_inputs")
    descriptor = function_source("validate_file_descriptor")
    load_blend = function_source("load_bound_blend")

    assert "bind_manifest.json" in load_inputs
    assert "bound.blend" in load_inputs
    assert module_constant("CANONICAL_GLBS") == CANONICAL_GLBS
    assert "bound_walk.glb" in source
    assert "bound_idle.glb" in source
    assert "bind_metrics.json" in load_inputs
    assert "hy3d_rocketbox_bind_v1" in load_inputs
    assert "sha256_file" in load_inputs
    assert "ensure_direct_regular_file" in load_inputs
    assert 'set(descriptor) != {"filename", "sha256"}' in descriptor
    assert 'manifest.get("bound_blend")' in load_inputs
    assert 'manifest.get("artifacts")' in load_inputs
    assert 'artifacts.get("bind_metrics")' in load_inputs
    assert "bound_blend_descriptor" not in source
    assert "if descriptor is not None" not in load_inputs
    assert "bpy.ops.wm.open_mainfile(" in source
    assert "expected_sha256" in load_blend
    assert load_blend.count("sha256_file(blend_path)") == 2
    assert "changed before Blender opened it" in load_blend
    assert "changed while Blender opened it" in load_blend
    assert '"bound_blend": {' in function_source("publish_review_manifest")
    assert '"bind_metrics": {' in function_source("publish_review_manifest")
    assert "save_as_mainfile" not in source
    assert "save_mainfile" not in source


def test_i23d_bind_metrics_require_matching_backend_and_research_license_scope():
    load_inputs = function_source("load_bind_inputs")

    assert module_constant("BIND_METRICS_SCHEMAS") == (
        "hy3d_rocketbox_bind_metrics_v1",
        "i23d_rocketbox_bind_metrics_v1",
    )
    assert "i23d_rocketbox_bind_metrics_v1" in load_inputs
    assert 'manifest.get("guide_backend")' in load_inputs
    assert 'manifest.get("usage_scope")' in load_inputs
    assert 'manifest.get("research_release_ok") is not True' in load_inputs
    assert 'manifest.get("permissive_commercial_ok") is not False' in load_inputs


def test_script_declares_exact_motion_view_and_media_contract():
    source = renderer_source()

    assert module_constant("REQUIRED_MOTIONS") == REQUIRED_MOTIONS
    assert module_constant("REQUIRED_VIEWS") == REQUIRED_VIEWS
    assert module_constant("CANONICAL_MEDIA") == CANONICAL_MEDIA
    assert module_constant("VIDEO_MEDIA") == tuple(CANONICAL_MEDIA)[:-1]
    assert module_constant("VIDEO_SIZE") == (1280, 720)
    assert module_constant("FPS") == 30
    assert module_constant("LOOP_CYCLE_COUNT") == 2
    assert "MPEG4" in source
    assert "H264" in source


def test_walk_and_idle_action_names_come_from_the_manifest():
    source = renderer_source()
    resolve = function_source("action_name_from_manifest")
    select = function_source("select_action")

    assert 'manifest.get("action_names")' in resolve
    assert "action_name" in resolve
    assert 'manifest.get("glbs")' not in resolve
    assert "bpy.data.actions" in select
    assert "action.name == action_name" in select
    assert 'action_names["walk"] == action_names["idle"]' in function_source(
        "load_bind_inputs"
    )
    for forbidden in ("Walking", "Standing_Idle", "mixamo", "Mixamo"):
        assert forbidden not in source

    resolve_action = pure_function("action_name_from_manifest")
    manifest = {
        "action_names": {
            "walk": "Bound locomotion clip",
            "idle": "Bound rest clip",
        }
    }
    assert resolve_action(manifest, "walk") == "Bound locomotion clip"
    assert resolve_action(manifest, "idle") == "Bound rest clip"


def test_renderer_is_target_only_without_source_sticks_or_path_arrows():
    source = renderer_source().lower()

    for forbidden in (
        "source_stick",
        "source_target",
        "add_root_path",
        "add_front_arrow",
        "import_scene.fbx",
        "source_motion_fbx",
    ):
        assert forbidden not in source


def test_front_and_side_follow_root_while_feet_follow_pelvis():
    source = renderer_source()
    configure = function_source("configure_view_camera")
    scan = function_source("scan_action_bounds")

    assert module_constant("VIEW_CAMERA_MODES") == {
        "front": "root_follow",
        "side": "root_follow",
        "feet": "pelvis_follow",
    }
    assert 'view_name in ("front", "side")' in configure
    assert 'bones["root"]' in configure
    assert 'bones["pelvis"]' in configure
    assert "configure_follow_camera(" in configure
    assert '"first_minimum"' in scan
    assert '"first_maximum"' in scan
    assert "first_center - root_point" in configure
    assert "feet_center - pelvis_point" in configure
    assert 'driver_add("location")' in source
    assert "bone_target = follow_bone_name" in source


def test_orthographic_camera_scale_accounts_for_16_by_9_vertical_extent():
    calculate_scale = pure_function(
        "orthographic_scale_for_extents",
        {"math": math},
    )
    render_size = (1920, 1080)
    aspect_ratio = render_size[0]/render_size[1]
    body_height_m = 1.826
    body_width_m = 0.80
    margin_ratio = 1.22

    old_scale = max(body_height_m, body_width_m)*margin_ratio
    assert old_scale/aspect_ratio < body_height_m

    new_scale = calculate_scale(
        body_width_m,
        body_height_m,
        render_size,
        margin_ratio,
    )
    assert new_scale == pytest.approx(
        body_height_m*aspect_ratio*margin_ratio
    )
    assert new_scale/aspect_ratio >= body_height_m*margin_ratio


@pytest.mark.parametrize(
    "dimensions",
    (
        (1.0523068905, 1.1236836910, 1.8259371519),
        (1.0768673420, 0.5975890160, 1.8330153227),
        (0.9615193605, 1.1263597012, 1.7894309759),
        (0.9868430495, 0.5471241474, 1.7511972189),
    ),
)
def test_real_male_female_smoke_extents_fit_front_side_and_feet(dimensions):
    calculate_scale = pure_function(
        "orthographic_scale_for_extents",
        {"math": math},
    )
    width_x, width_y, height = dimensions
    render_size = (1280, 720)
    aspect_ratio = render_size[0]/render_size[1]

    front_scale = calculate_scale(width_x, height, render_size, 1.22)
    side_scale = calculate_scale(width_y, height, render_size, 1.22)
    feet_width = math.hypot(width_x, width_y)
    feet_height = height*0.72
    feet_scale = calculate_scale(feet_width, feet_height, render_size, 1.05)

    assert front_scale >= width_x*1.22
    assert front_scale/aspect_ratio >= height*1.22
    assert side_scale >= width_y*1.22
    assert side_scale/aspect_ratio >= height*1.22
    assert feet_scale >= feet_width*1.05
    assert feet_scale/aspect_ratio == pytest.approx(
        max(feet_width/aspect_ratio, feet_height)*1.05
    )

    configure = function_source("configure_view_camera")
    assert "orthographic_scale_for_extents(" in configure
    assert "VIDEO_SIZE" in configure


def test_camera_gate_uses_deformed_mesh_and_weighted_foot_toe_vertices():
    source = renderer_source()
    resolve = function_source("resolve_review_bones")
    bounds = function_source("validate_camera_bounds")
    deformed = function_source("evaluated_mesh_world_points")
    weighted = function_source("foot_toe_weighted_vertex_indices")
    weighted_helper = function_source("weighted_vertex_indices")

    for semantic in (
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
        "left_toe",
        "right_toe",
    ):
        assert semantic in source
    assert "REVIEW_BONE_SEMANTICS" in resolve
    assert "evaluated_get(depsgraph)" in deformed
    assert "to_mesh()" in deformed
    assert "to_mesh_clear()" in deformed
    assert "vertex.co" in deformed
    assert "vertex.groups" in weighted_helper
    assert 'bones[f"{side}_foot"]' in weighted
    assert 'bones[f"{side}_toe"]' in weighted
    assert 'for side in ("left", "right")' in weighted
    assert 'f"{side} Foot/Toe vertex groups have no weighted vertices"' in weighted
    assert "evaluated_mesh_world_points(" in bounds
    assert "foot_toe_indices" in bounds
    assert 'for side in ("left", "right")' in bounds
    assert "foot_toe_indices[side][mesh.name]" in bounds
    assert "world_to_camera_view" in bounds
    assert "CAMERA_BOUND_MARGIN" in bounds
    assert "pose_bone" not in bounds
    assert ".head" not in bounds
    assert ".tail" not in bounds
    assert "validate_camera_bounds(" in function_source("render_motion")


def test_weighted_feet_reject_only_left_foot_weights():
    weighted_indices = pure_function("weighted_vertex_indices")
    membership = SimpleNamespace(group=1, weight=1.0)
    vertex = SimpleNamespace(index=7, groups=(membership,))
    assert weighted_indices((vertex,), {1}) == (7,)
    assert weighted_indices((vertex,), {3, 4}) == ()
    foot_indices = pure_function(
        "foot_toe_weighted_vertex_indices",
        {"weighted_vertex_indices": weighted_indices},
    )
    mesh = SimpleNamespace(
        name="target",
        vertex_groups=(
            SimpleNamespace(index=1, name="LeftFoot"),
            SimpleNamespace(index=2, name="LeftToe"),
            SimpleNamespace(index=3, name="RightFoot"),
            SimpleNamespace(index=4, name="RightToe"),
        ),
        data=SimpleNamespace(vertices=(vertex,)),
    )
    bones = {
        "left_foot": "LeftFoot",
        "left_toe": "LeftToe",
        "right_foot": "RightFoot",
        "right_toe": "RightToe",
    }
    with pytest.raises(RuntimeError, match="right Foot/Toe"):
        foot_indices((mesh,), bones)


def test_one_fixed_floor_and_two_exact_action_cycles_drive_frame_counts():
    source = compact_source()
    loop_range = pure_function("loop_frame_range", {"LOOP_CYCLE_COUNT": 2})
    floor = function_source("fixed_floor_check")

    assert loop_range(7, 36) == (7, 64, 58)
    assert "forframeinrange(frame_start,frame_end+1)" in source
    load_inputs = function_source("load_bind_inputs")
    assert 'metrics.get("floor_z_m")' in load_inputs
    assert 'manifest.get("floor_z_m") != floor_z_m' in load_inputs
    assert "math.isfinite(floor_z_m)" in load_inputs
    assert "review_floor_z=min(" not in source
    assert "floor_z_m" in floor
    assert "add_fixed_floor(" in source
    assert source.count("add_fixed_floor(") == 1
    assert 'target.animation_data.action = details["action"]' in floor
    assert 'modifier=curve.modifiers.new(type="CYCLES")' in source
    assert "maximum_penetration_m" in floor
    assert "minimum_support_distance_m" in floor
    assert "support_frame_count" in floor
    assert 'motion_checks[motion] = floor_motion_metrics(' in floor
    assert 'require_floor_motion_pass(motion, motion_checks[motion])' in floor
    assert 'details["bounds"]["maximum_dimensions"].z' in floor
    assert '"actual_world_height_m"' in floor
    assert '"penetration_tolerance_m"' in floor
    assert '"motions": motion_checks' in floor
    assert "expected_frame_count" in function_source("render_motion")


def test_floor_rejects_idle_without_support_even_when_walk_supports():
    tolerance = pure_function(
        "scale_aware_floor_tolerance_m",
        {
            "math": math,
            "FLOOR_PENETRATION_HEIGHT_RATIO": 0.0125,
            "FLOOR_PENETRATION_TOLERANCE_MIN_M": 0.005,
            "FLOOR_PENETRATION_TOLERANCE_MAX_M": 0.025,
        },
    )
    summarize_floor = pure_function(
        "floor_motion_metrics",
        {
            "FLOOR_SUPPORT_TOLERANCE_M": 0.03,
            "scale_aware_floor_tolerance_m": tolerance,
            "FLOOR_PENETRATION_TOLERANCE_MAX_M": 0.025,
        },
    )
    require_floor = pure_function(
        "require_floor_motion_pass",
        {"FLOOR_PENETRATION_TOLERANCE_MAX_M": 0.025},
    )
    walk = summarize_floor((0.0, 0.0), (0.0, 0.01), 0.0, 1.80)
    idle = summarize_floor((0.20, 0.22), (0.20, 0.22), 0.0, 1.80)
    require_floor("walk", walk)
    assert walk["support_frame_count"] == 2
    assert idle["support_frame_count"] == 0
    with pytest.raises(RuntimeError, match="idle.*never support"):
        require_floor("idle", idle)


def test_scale_aware_floor_tolerance_accepts_measured_lbs_drift_but_caps_gross_penetration():
    assert module_constant("FLOOR_PENETRATION_HEIGHT_RATIO") == 0.0125
    assert module_constant("FLOOR_PENETRATION_TOLERANCE_MIN_M") == 0.005
    assert module_constant("FLOOR_PENETRATION_TOLERANCE_MAX_M") == 0.025
    tolerance = pure_function(
        "scale_aware_floor_tolerance_m",
        {
            "math": math,
            "FLOOR_PENETRATION_HEIGHT_RATIO": 0.0125,
            "FLOOR_PENETRATION_TOLERANCE_MIN_M": 0.005,
            "FLOOR_PENETRATION_TOLERANCE_MAX_M": 0.025,
        },
    )
    summarize_floor = pure_function(
        "floor_motion_metrics",
        {
            "FLOOR_SUPPORT_TOLERANCE_M": 0.03,
            "scale_aware_floor_tolerance_m": tolerance,
            "FLOOR_PENETRATION_TOLERANCE_MAX_M": 0.025,
        },
    )
    require_floor = pure_function(
        "require_floor_motion_pass",
        {"FLOOR_PENETRATION_TOLERANCE_MAX_M": 0.025},
    )

    assert tolerance(0.20) == pytest.approx(0.005)
    assert tolerance(3.00) == pytest.approx(0.025)
    measured = (
        ("male walk", 1.8259371519, 0.002527),
        ("male idle", 1.8330153227, 0.0),
        ("female walk", 1.7894309759, 0.019208),
        ("female idle", 1.7511972189, 0.006059),
    )
    for motion, actual_height, penetration in measured:
        metrics = summarize_floor(
            (-penetration, 0.0),
            (0.0, 0.001),
            0.0,
            actual_height,
        )
        assert metrics["actual_world_height_m"] == pytest.approx(actual_height)
        assert metrics["penetration_tolerance_m"] == pytest.approx(
            tolerance(actual_height)
        )
        require_floor(motion, metrics)

    gross = summarize_floor((-0.025001,), (0.0,), 0.0, 4.0)
    with pytest.raises(RuntimeError, match="gross penetration.*0.025"):
        require_floor("walk", gross)


def test_cycle_repeat_mode_offsets_only_walk_root_motion_location():
    repeat_mode = pure_function("curve_repeat_mode")

    assert repeat_mode("idle", "location", "Bip01") == "REPEAT"
    assert repeat_mode("idle", 'pose.bones["Bip01"].location', "Bip01") == "REPEAT"
    assert repeat_mode("walk", "location", "Bip01") == "REPEAT_OFFSET"
    assert (
        repeat_mode("walk", 'pose.bones["Bip01"].location', "Bip01")
        == "REPEAT_OFFSET"
    )
    assert (
        repeat_mode("walk", 'pose.bones["Bip01 Pelvis"].location', "Bip01")
        == "REPEAT"
    )
    add = function_source("add_loop_modifiers")
    assert "curve_repeat_mode(" in add
    assert "root_bone_name" in add


def test_reference_renderer_helpers_are_reused_for_scene_media_and_validation():
    source = renderer_source()

    assert "import blender_render_rocketbox_motion_review as rocketbox_review" in source
    for helper in (
        "configure_scene",
        "add_lighting",
        "add_fixed_floor",
        "make_camera",
        "configure_video_output",
        "render_contact_sheet",
        "validate_video",
        "validate_png",
        "make_staged_path",
    ):
        assert f"rocketbox_review.{helper}" in source


def test_nonblank_media_and_hashes_are_validated_before_atomic_publication():
    source = renderer_source()
    validate = function_source("validate_staged_media")
    publish = function_source("publish_review_manifest")
    render = function_source("render_review")

    assert "validate_video(" in validate
    assert "validate_png(" in validate
    assert "sampled_luma_range" in source
    assert "pixel_range" in source
    assert '"sha256"' in publish
    assert "sha256_file(" in publish
    assert '"bind_manifest_sha256"' in publish
    assert '"action_names"' in publish
    assert '"automatic_checks"' in publish
    assert render.index("validate_staged_media(") < render.index("publish_media(")
    assert render.index("publish_media(") < render.index("publish_review_manifest(")
    assert "os.replace(" in source


def test_render_start_safely_invalidates_manifest_and_both_direct_attempt_files():
    source = renderer_source()
    invalidate = function_source("invalidate_render_outputs")
    safe_unlink = function_source("safe_unlink_outputs")
    render = function_source("render_review")

    assert module_constant("RENDER_INVALIDATION_FILES") == (
        "direct_attempt_ready.json",
        "direct_attempt_rejected.json",
        "review_manifest.json",
    )
    for filename in module_constant("RENDER_INVALIDATION_FILES"):
        assert filename in invalidate
    assert "os.open(" in safe_unlink
    assert "dir_fd=directory_fd" in safe_unlink
    assert "os.unlink(filename" in safe_unlink
    assert "os.fsync(directory_fd)" in safe_unlink
    assert render.index("invalidate_render_outputs(") < render.index("load_bind_inputs(")
    assert render.index("load_bind_inputs(") < render.index("load_bound_blend(")


def test_direct_attempt_record_is_explicit_hash_locked_and_not_automatic():
    source = renderer_source()
    record = function_source("record_direct_attempt")
    build = function_source("build_direct_attempt_payload")
    snapshot = function_source("capture_artifact_snapshot")
    read_qa = function_source("read_pixel_qa")
    render = function_source("render_review")
    main = function_source("main")

    assert module_constant("DIRECT_ATTEMPT_FILES") == (
        "direct_attempt_ready.json",
        "direct_attempt_rejected.json",
    )
    assert module_constant("DIRECT_ATTEMPT_READY_SCHEMA") == (
        "hy3d_rocketbox_direct_attempt_ready_v1"
    )
    assert module_constant("DIRECT_ATTEMPT_REJECTED_SCHEMA") == (
        "hy3d_rocketbox_direct_attempt_rejected_v1"
    )
    assert module_constant("PIXEL_QA_FILENAME") == "pixel_qa.json"
    assert "ensure_direct_regular_file(" in read_qa
    assert "PIXEL_QA_FILENAME" in read_qa
    assert "invalidate_direct_attempt_outputs(" in record
    assert record.index("invalidate_direct_attempt_outputs(") < record.index(
        "capture_artifact_snapshot("
    )
    assert "validate_pixel_qa(" in record
    assert "validate_expected_artifact_snapshot(" in record
    assert record.index("validate_expected_artifact_snapshot(") < record.index(
        "atomic_write_json("
    )
    assert "pixel_qa_sha256" in record
    assert "bind_manifest_sha256" in build
    assert "review_manifest_sha256" in build
    assert '"videos": snapshot["videos"]' in build
    for required in (
        '"schema_version"',
        '"asset_id"',
        '"status"',
        '"bind_manifest_sha256"',
        '"review_manifest_sha256"',
        '"bound_blend"',
        '"glbs"',
        '"videos"',
    ):
        assert required in build
    assert 'status = "ready"' in build
    assert '"status": status' in build
    assert '"bound_blend": snapshot["bound_blend"]' in build
    assert '"glbs": snapshot["glbs"]' in build
    assert "CANONICAL_GLBS[motion]" in snapshot
    assert '"filename": f"{motion}_{view_name}.mp4"' in snapshot
    assert 'status = "rejected"' in build
    assert "build_direct_attempt_payload(" in record
    assert '"pixel_qa": {' in build
    assert '"filename": PIXEL_QA_FILENAME' in build
    assert '"sha256": pixel_qa_sha256' in build
    assert 'inputs["manifest_sha256"]' in snapshot
    assert 'inputs["manifest"]["bound_blend"]' in snapshot
    assert 'inputs["manifest"]["artifacts"]["bind_metrics"]' in snapshot
    assert 'inputs["manifest"]["glbs"]' in snapshot
    assert "bind-to-review" in snapshot
    assert "atomic_write_json(" in record
    assert "snapshot_before != snapshot_after" in record
    assert "safe_unlink_outputs(" in record
    assert "except BaseException:" in record
    assert record.index("atomic_write_json(") < record.index("except BaseException:")
    for filename in (
        "bind_manifest.json",
        "review_manifest.json",
        "bound.blend",
        "bound_walk.glb",
        "bound_idle.glb",
        "bind_contact_sheet.png",
        "walk_front.mp4",
        "idle_feet.mp4",
    ):
        assert filename in snapshot or filename in source
    assert "record_direct_attempt(" not in render
    assert 'if args.record_direct_attempt is not None:' in main
    assert "record_direct_attempt(" in main


def test_finalize_rejects_old_qa_snapshot_before_writing_ready():
    expected_a = artifact_snapshot("a")
    current_b = artifact_snapshot("b")
    pixel_qa = {"expected_artifact_snapshot": expected_a}
    compare = pure_function("validate_expected_artifact_snapshot")
    with pytest.raises(RuntimeError, match="expected artifact snapshot is stale"):
        compare(pixel_qa, current_b)

    writes = []
    record = pure_function(
        "record_direct_attempt",
        {
            "validate_bind_dir": lambda path: path,
            "invalidate_direct_attempt_outputs": lambda path: None,
            "load_bind_inputs": lambda args, path: {},
            "read_pixel_qa": lambda path, bind_dir: (
                pixel_qa,
                "c"*64,
                Path("pixel_qa.json"),
            ),
            "validate_pixel_qa": lambda *args: None,
            "capture_artifact_snapshot": lambda inputs: current_b,
            "validate_expected_artifact_snapshot": compare,
            "build_direct_attempt_payload": lambda *args: {},
            "rocketbox_review": SimpleNamespace(
                atomic_write_json=lambda *args: writes.append(args)
            ),
            "safe_unlink_outputs": lambda *args: None,
            "DIRECT_ATTEMPT_FILES": (
                "direct_attempt_ready.json",
                "direct_attempt_rejected.json",
            ),
        },
    )
    args = SimpleNamespace(
        asset_id="rocketbox_male_adult_01",
        bind_dir=Path("bind"),
        pixel_qa_json=Path("qa.json"),
        record_direct_attempt="ready",
    )
    with pytest.raises(RuntimeError, match="expected artifact snapshot is stale"):
        record(args)
    assert writes == []


def test_new_python_files_follow_repository_style_contract():
    source = renderer_source()
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

    for line in copyright_lines:
        assert line in source
    for forbidden in forbidden_strings:
        assert forbidden not in source
