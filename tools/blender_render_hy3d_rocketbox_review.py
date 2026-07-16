#!/usr/bin/env python3

#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Render target-only Walk/Idle evidence for a bound Hunyuan avatar."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import blender_render_rocketbox_motion_review as rocketbox_review


EXPECTED_ASSET_IDS = (
    "rocketbox_male_adult_01",
    "rocketbox_female_adult_01",
)
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
VIDEO_MEDIA = (
    "walk_front",
    "walk_side",
    "walk_feet",
    "idle_front",
    "idle_side",
    "idle_feet",
)
VIEW_CAMERA_MODES = {
    "front": "root_follow",
    "side": "root_follow",
    "feet": "pelvis_follow",
}
VIDEO_SIZE = (1280, 720)
FPS = 30
LOOP_CYCLE_COUNT = 2
CAMERA_BOUND_MARGIN = 0.015
BODY_CAMERA_MARGIN_RATIO = 1.22
FEET_CAMERA_HEIGHT_RATIO = 0.72
FEET_CAMERA_MARGIN_RATIO = 1.05
FLOOR_PENETRATION_HEIGHT_RATIO = 0.0125
FLOOR_PENETRATION_TOLERANCE_MIN_M = 0.005
FLOOR_PENETRATION_TOLERANCE_MAX_M = 0.025
FLOOR_SUPPORT_TOLERANCE_M = 0.03
FFMPEG_FORMAT = "MPEG4"
FFMPEG_CODEC = "H264"
REVIEW_MANIFEST_SCHEMA = "hy3d_rocketbox_review_manifest_v1"
BIND_MANIFEST_SCHEMA = "hy3d_rocketbox_bind_v1"
BIND_METRICS_SCHEMAS = (
    "hy3d_rocketbox_bind_metrics_v1",
    "i23d_rocketbox_bind_metrics_v1",
)
PIXEL_QA_SCHEMA = "hy3d_rocketbox_pixel_qa_v1"
PIXEL_QA_FILENAME = "pixel_qa.json"
ARTIFACT_SNAPSHOT_SCHEMA = "hy3d_rocketbox_artifact_snapshot_v1"
DIRECT_ATTEMPT_READY_SCHEMA = "hy3d_rocketbox_direct_attempt_ready_v1"
DIRECT_ATTEMPT_REJECTED_SCHEMA = "hy3d_rocketbox_direct_attempt_rejected_v1"
DIRECT_ATTEMPT_FILES = (
    "direct_attempt_ready.json",
    "direct_attempt_rejected.json",
)
RENDER_INVALIDATION_FILES = (
    "direct_attempt_ready.json",
    "direct_attempt_rejected.json",
    "review_manifest.json",
)
PIXEL_QA_CHECKS = (
    "hands_attached",
    "hands_not_duplicated",
    "pieces_nonblank",
    "arm_torso_regions_clean",
    "thigh_regions_clean",
    "sleeves_seam_free",
    "feet_not_inverted",
    "floor_cards_absent",
    "leg_gap_fans_absent",
    "mesh_explosions_absent",
)
REVIEW_BONE_SEMANTICS = (
    "root",
    "pelvis",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
    "left_toe",
    "right_toe",
)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--bind-dir", type=Path, required=True)
    parser.add_argument("--record-direct-attempt", choices=("ready", "rejected"))
    parser.add_argument("--pixel-qa-json", type=Path)
    args = parser.parse_args(argv)
    if args.asset_id not in EXPECTED_ASSET_IDS:
        parser.error(f"unexpected Rocketbox asset id: {args.asset_id}")
    if (args.record_direct_attempt is None) != (args.pixel_qa_json is None):
        parser.error(
            "--record-direct-attempt and --pixel-qa-json must be provided together"
        )
    return args


def action_name_from_manifest(manifest, motion):
    action_names = manifest.get("action_names")
    if isinstance(action_names, dict):
        action_name = action_names.get(motion)
        if isinstance(action_name, str) and action_name:
            return action_name
    raise RuntimeError(f"bind manifest has no {motion} action name")


def validate_file_descriptor(descriptor, path, expected_filename, description):
    if not isinstance(descriptor, dict) or set(descriptor) != {"filename", "sha256"}:
        raise RuntimeError(f"{description} descriptor must contain filename and sha256")
    if descriptor.get("filename") != expected_filename:
        raise RuntimeError(f"{description} filename is not canonical")
    expected_sha256 = descriptor.get("sha256")
    if expected_sha256 != rocketbox_review.sha256_file(path):
        raise RuntimeError(f"{description} hash does not match current file")


def validate_bind_dir(path):
    absolute = Path(path).absolute()
    if absolute != absolute.resolve() or not absolute.is_dir():
        raise RuntimeError("bind directory must be a direct non-symlink directory")
    return absolute


def safe_unlink_outputs(bind_dir, filenames):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(bind_dir, flags)
    try:
        for filename in filenames:
            try:
                os.unlink(filename, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def invalidate_render_outputs(bind_dir):
    safe_unlink_outputs(
        bind_dir,
        (
            "direct_attempt_ready.json",
            "direct_attempt_rejected.json",
            "review_manifest.json",
        ),
    )


def invalidate_direct_attempt_outputs(bind_dir):
    safe_unlink_outputs(bind_dir, DIRECT_ATTEMPT_FILES)


def load_bind_inputs(args, bind_dir=None):
    bind_dir = validate_bind_dir(args.bind_dir) if bind_dir is None else bind_dir
    manifest_path = rocketbox_review.ensure_direct_regular_file(
        bind_dir/"bind_manifest.json", bind_dir, "bind manifest"
    )
    blend_path = rocketbox_review.ensure_direct_regular_file(
        bind_dir/"bound.blend", bind_dir, "bound blend"
    )
    metrics_path = rocketbox_review.ensure_direct_regular_file(
        bind_dir/"bind_metrics.json", bind_dir, "bind metrics"
    )
    manifest = rocketbox_review.load_json(manifest_path, "bind manifest")
    if manifest.get("schema_version") != "hy3d_rocketbox_bind_v1":
        raise RuntimeError(f"bind manifest schema must be {BIND_MANIFEST_SCHEMA}")
    if manifest.get("asset_id") != args.asset_id:
        raise RuntimeError("bind manifest asset_id does not match --asset-id")

    glbs = manifest.get("glbs")
    if not isinstance(glbs, dict) or set(glbs) != set(REQUIRED_MOTIONS):
        raise RuntimeError("bind manifest GLBs must contain exactly walk and idle")
    glb_paths = {}
    for motion in REQUIRED_MOTIONS:
        filename = CANONICAL_GLBS[motion]
        glb_path = rocketbox_review.ensure_direct_regular_file(
            bind_dir/filename, bind_dir, f"{motion} GLB"
        )
        validate_file_descriptor(glbs[motion], glb_path, filename, f"{motion} GLB")
        glb_paths[motion] = glb_path

    validate_file_descriptor(
        manifest.get("bound_blend"), blend_path, "bound.blend", "bound blend"
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("bind manifest artifacts are missing")
    validate_file_descriptor(
        artifacts.get("bind_metrics"),
        metrics_path,
        "bind_metrics.json",
        "bind metrics",
    )
    metrics = rocketbox_review.load_json(metrics_path, "bind metrics")
    metrics_schema = metrics.get("schema_version")
    if metrics_schema not in BIND_METRICS_SCHEMAS:
        raise RuntimeError("bind metrics schema is invalid")
    if metrics_schema == "i23d_rocketbox_bind_metrics_v1":
        guide_backend = manifest.get("guide_backend")
        if guide_backend not in ("trellis2", "pixal3d"):
            raise RuntimeError("I23D bind manifest guide_backend is invalid")
        if manifest.get("usage_scope") != "noncommercial_research_dataset_candidate":
            raise RuntimeError("I23D bind manifest usage_scope is invalid")
        if manifest.get("research_release_ok") is not True:
            raise RuntimeError("I23D bind manifest must allow research release")
        if manifest.get("permissive_commercial_ok") is not False:
            raise RuntimeError("I23D bind manifest cannot claim permissive commercial use")
        provenance = metrics.get("guide_provenance")
        if not isinstance(provenance, dict) or provenance.get("backend") != guide_backend:
            raise RuntimeError("I23D bind metrics backend does not match the manifest")
    if metrics.get("asset_id") != args.asset_id:
        raise RuntimeError("bind metrics asset_id does not match --asset-id")
    floor_z_m = metrics.get("floor_z_m")
    if not isinstance(floor_z_m, (int, float)) or not math.isfinite(floor_z_m):
        raise RuntimeError("bind metrics floor_z_m must be finite")
    floor_z_m = float(floor_z_m)
    if manifest.get("floor_z_m") != floor_z_m:
        raise RuntimeError("bind manifest floor_z_m does not match bind metrics")
    action_names = {
        motion: action_name_from_manifest(manifest, motion)
        for motion in REQUIRED_MOTIONS
    }
    if action_names["walk"] == action_names["idle"]:
        raise RuntimeError("walk and idle action names must be different")
    declared_action_names = manifest.get("action_names")
    if not isinstance(declared_action_names, dict) or set(
        declared_action_names
    ) != set(REQUIRED_MOTIONS):
        raise RuntimeError("bind manifest action_names must contain exactly walk and idle")
    return {
        "bind_dir": bind_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": rocketbox_review.sha256_file(manifest_path),
        "blend_path": blend_path,
        "blend_sha256": rocketbox_review.sha256_file(blend_path),
        "metrics_path": metrics_path,
        "metrics_sha256": rocketbox_review.sha256_file(metrics_path),
        "floor_z_m": floor_z_m,
        "glb_paths": glb_paths,
        "action_names": action_names,
    }


def load_bound_blend(blend_path, expected_sha256):
    if rocketbox_review.sha256_file(blend_path) != expected_sha256:
        raise RuntimeError("bound blend changed before Blender opened it")
    print(f"Loading blend: {blend_path}")
    result = bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    if "FINISHED" not in result:
        raise RuntimeError("could not load bound blend")
    if rocketbox_review.sha256_file(blend_path) != expected_sha256:
        raise RuntimeError("bound blend changed while Blender opened it")


def target_objects():
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError("bound blend must contain exactly one target armature")
    target = armatures[0]
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and any(
            modifier.type == "ARMATURE" and modifier.object == target
            for modifier in obj.modifiers
        )
    ]
    if not meshes:
        raise RuntimeError("bound blend has no mesh bound to the target armature")
    return target, meshes


def select_action(target, action_name):
    matches = [action for action in bpy.data.actions if action.name == action_name]
    if len(matches) != 1:
        raise RuntimeError(f"bound blend does not contain unique action {action_name!r}")
    target.animation_data_create()
    target.animation_data.action = matches[0]
    return matches[0]


def action_frame_range(action):
    frame_start = int(round(float(action.frame_range[0])))
    frame_end = int(round(float(action.frame_range[1])))
    if frame_end <= frame_start:
        raise RuntimeError(f"action {action.name!r} has an invalid frame range")
    return frame_start, frame_end


def loop_frame_range(frame_start, frame_end):
    cycle_span = frame_end - frame_start
    review_frame_end = frame_start + LOOP_CYCLE_COUNT*cycle_span - 1
    expected_frame_count = review_frame_end - frame_start + 1
    return frame_start, review_frame_end, expected_frame_count


def curve_repeat_mode(motion, data_path, root_bone_name):
    root_bone_location = f'pose.bones["{root_bone_name}"].location'
    if motion == "walk" and data_path in ("location", root_bone_location):
        return "REPEAT_OFFSET"
    return "REPEAT"


def add_loop_modifiers(action, motion, root_bone_name):
    for curve in action.fcurves:
        existing = [item for item in curve.modifiers if item.type == "CYCLES"]
        if existing:
            modifier = existing[0]
        else:
            modifier = curve.modifiers.new(type="CYCLES")
        modifier.mode_before = "REPEAT"
        modifier.mode_after = curve_repeat_mode(
            motion, curve.data_path, root_bone_name
        )


def normalized_bone_tokens(name):
    return tuple(value for value in re.split(r"[^a-z0-9]+", name.lower()) if value)


def side_matches(tokens, side):
    if side == "left":
        return "left" in tokens or "l" in tokens
    return "right" in tokens or "r" in tokens


def find_semantic_bone(target, semantic):
    bones = list(target.data.bones)
    if semantic == "root":
        roots = [bone for bone in bones if bone.parent is None]
        named = [bone for bone in roots if "root" in normalized_bone_tokens(bone.name)]
        choices = named or roots
    elif semantic == "pelvis":
        choices = [
            bone
            for bone in bones
            if {"pelvis", "hips", "hip"}.intersection(normalized_bone_tokens(bone.name))
        ]
    else:
        side, part = semantic.split("_", 1)
        part_tokens = {part}
        if part == "foot":
            part_tokens.add("ankle")
        if part == "hand":
            part_tokens.add("wrist")
        choices = []
        for bone in bones:
            tokens = normalized_bone_tokens(bone.name)
            part_matches = bool(part_tokens.intersection(tokens))
            if part == "toe":
                part_matches = any(token.startswith("toe") for token in tokens)
            if side_matches(tokens, side) and part_matches:
                choices.append(bone)
    if len(choices) != 1:
        raise RuntimeError(f"could not resolve unique {semantic} review bone")
    return choices[0].name


def manifest_review_bones(manifest):
    for key in ("review_bones", "bones"):
        value = manifest.get(key)
        if isinstance(value, dict):
            return value
    rig = manifest.get("rig")
    if isinstance(rig, dict):
        for key in ("review_bones", "bones"):
            value = rig.get(key)
            if isinstance(value, dict):
                return value
    return {}


def resolve_review_bones(target, manifest):
    declared = manifest_review_bones(manifest)
    available = set(target.pose.bones.keys())
    resolved = {}
    for semantic in REVIEW_BONE_SEMANTICS:
        name = declared.get(semantic)
        if name is None:
            name = find_semantic_bone(target, semantic)
        if not isinstance(name, str) or name not in available:
            raise RuntimeError(f"missing hand or foot camera bound: {semantic}")
        resolved[semantic] = name
    return resolved


def evaluated_mesh_bounds(meshes):
    minima = []
    maxima = []
    for mesh in meshes:
        minimum, maximum = rocketbox_review.evaluated_mesh_frame_data(mesh)
        minima.append(minimum)
        maxima.append(maximum)
    return (
        Vector(
            (
                min(value.x for value in minima),
                min(value.y for value in minima),
                min(value.z for value in minima),
            )
        ),
        Vector(
            (
                max(value.x for value in maxima),
                max(value.y for value in maxima),
                max(value.z for value in maxima),
            )
        ),
    )


def scan_action_bounds(meshes, frame_start, frame_end):
    minima = []
    maxima = []
    dimensions = []
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        minimum, maximum = evaluated_mesh_bounds(meshes)
        minima.append(minimum)
        maxima.append(maximum)
        dimensions.append(maximum - minimum)
    return {
        "first_minimum": minima[0].copy(),
        "first_maximum": maxima[0].copy(),
        "minimum": Vector(
            (
                min(value.x for value in minima),
                min(value.y for value in minima),
                min(value.z for value in minima),
            )
        ),
        "maximum": Vector(
            (
                max(value.x for value in maxima),
                max(value.y for value in maxima),
                max(value.z for value in maxima),
            )
        ),
        "maximum_dimensions": Vector(
            (
                max(value.x for value in dimensions),
                max(value.y for value in dimensions),
                max(value.z for value in dimensions),
            )
        ),
    }


def prepare_actions(target, meshes, bones, action_names):
    prepared = {}
    for motion in REQUIRED_MOTIONS:
        action = select_action(target, action_names[motion])
        add_loop_modifiers(action, motion, bones["root"])
        original_start, original_end = action_frame_range(action)
        frame_start, frame_end, expected_frame_count = loop_frame_range(
            original_start, original_end
        )
        bounds = scan_action_bounds(meshes, frame_start, frame_end)
        prepared[motion] = {
            "action": action,
            "action_name": action.name,
            "original_frame_start": original_start,
            "original_frame_end": original_end,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "expected_frame_count": expected_frame_count,
            "bounds": bounds,
        }
    return prepared


def combined_bounds(prepared):
    minima = [prepared[motion]["bounds"]["minimum"] for motion in REQUIRED_MOTIONS]
    maxima = [prepared[motion]["bounds"]["maximum"] for motion in REQUIRED_MOTIONS]
    return (
        Vector(
            (
                min(value.x for value in minima),
                min(value.y for value in minima),
                min(value.z for value in minima),
            )
        ),
        Vector(
            (
                max(value.x for value in maxima),
                max(value.y for value in maxima),
                max(value.z for value in maxima),
            )
        ),
    )


def evaluated_mesh_world_points(mesh, vertex_indices=None):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        if vertex_indices is None:
            vertices = evaluated_mesh.vertices
        else:
            if vertex_indices and max(vertex_indices) >= len(evaluated_mesh.vertices):
                raise RuntimeError("evaluated mesh changed weighted-vertex topology")
            vertices = [evaluated_mesh.vertices[index] for index in vertex_indices]
        return [evaluated.matrix_world @ vertex.co for vertex in vertices]
    finally:
        evaluated.to_mesh_clear()


def weighted_vertex_indices(vertices, group_indices):
    return tuple(
        vertex.index
        for vertex in vertices
        if any(
            membership.group in group_indices and membership.weight > 0.0
            for membership in vertex.groups
        )
    )


def foot_toe_weighted_vertex_indices(meshes, bones):
    result = {"left": {}, "right": {}}
    for side in ("left", "right"):
        bone_names = {bones[f"{side}_foot"], bones[f"{side}_toe"]}
        total = 0
        for mesh in meshes:
            group_indices = {
                group.index for group in mesh.vertex_groups if group.name in bone_names
            }
            indices = weighted_vertex_indices(mesh.data.vertices, group_indices)
            result[side][mesh.name] = indices
            total += len(indices)
        if total == 0:
            raise RuntimeError(
                f"{side} Foot/Toe vertex groups have no weighted vertices"
            )
    return result


def scale_aware_floor_tolerance_m(actual_world_height_m):
    height = float(actual_world_height_m)
    if not math.isfinite(height) or height <= 0.0:
        raise RuntimeError("prepared target world height must be finite and positive")
    return min(
        FLOOR_PENETRATION_TOLERANCE_MAX_M,
        max(
            FLOOR_PENETRATION_TOLERANCE_MIN_M,
            height*FLOOR_PENETRATION_HEIGHT_RATIO,
        ),
    )


def floor_motion_metrics(
    frame_minima,
    support_distances,
    floor_z_m,
    actual_world_height_m,
):
    if not frame_minima or not support_distances:
        raise RuntimeError("floor motion metrics require sampled frames")
    maximum_penetration = max(
        max(0.0, floor_z_m - value) for value in frame_minima
    )
    minimum_support_distance = min(support_distances)
    penetration_tolerance = scale_aware_floor_tolerance_m(actual_world_height_m)
    support_frame_count = sum(
        value <= FLOOR_SUPPORT_TOLERANCE_M for value in support_distances
    )
    return {
        "actual_world_height_m": float(actual_world_height_m),
        "maximum_penetration_m": maximum_penetration,
        "penetration_tolerance_m": penetration_tolerance,
        "gross_penetration_cap_m": FLOOR_PENETRATION_TOLERANCE_MAX_M,
        "minimum_support_distance_m": minimum_support_distance,
        "support_frame_count": support_frame_count,
        "sampled_frame_count": len(frame_minima),
    }


def require_floor_motion_pass(motion, metrics):
    maximum_penetration = metrics["maximum_penetration_m"]
    if maximum_penetration > FLOOR_PENETRATION_TOLERANCE_MAX_M:
        raise RuntimeError(
            f"{motion} gross penetration {maximum_penetration:.6f} m exceeds "
            f"{FLOOR_PENETRATION_TOLERANCE_MAX_M:.3f} m cap"
        )
    if maximum_penetration > metrics["penetration_tolerance_m"]:
        raise RuntimeError(
            f"{motion} deformed target mesh exceeds scale-aware penetration "
            f"tolerance at authenticated fixed floor"
        )
    if metrics["support_frame_count"] == 0:
        raise RuntimeError(
            f"{motion} Foot/Toe vertices never support on authenticated fixed floor"
        )


def fixed_floor_check(target, meshes, prepared, floor_z_m, foot_toe_indices):
    minimum, maximum = combined_bounds(prepared)
    motion_checks = {}
    for motion in REQUIRED_MOTIONS:
        details = prepared[motion]
        target.animation_data.action = details["action"]
        frame_minima = []
        support_distances = []
        for frame in range(details["frame_start"], details["frame_end"] + 1):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            all_points = []
            foot_points = []
            for mesh in meshes:
                all_points.extend(evaluated_mesh_world_points(mesh))
                for side in ("left", "right"):
                    indices = foot_toe_indices[side][mesh.name]
                    if indices:
                        foot_points.extend(evaluated_mesh_world_points(mesh, indices))
            frame_minima.append(min(float(point.z) for point in all_points))
            support_distances.append(
                min(abs(float(point.z) - floor_z_m) for point in foot_points)
            )
        motion_checks[motion] = floor_motion_metrics(
            frame_minima,
            support_distances,
            floor_z_m,
            details["bounds"]["maximum_dimensions"].z,
        )
        require_floor_motion_pass(motion, motion_checks[motion])
    maximum_penetration = max(
        values["maximum_penetration_m"] for values in motion_checks.values()
    )
    minimum_support_distance = min(
        values["minimum_support_distance_m"] for values in motion_checks.values()
    )
    support_frame_count = sum(
        values["support_frame_count"] for values in motion_checks.values()
    )
    actual_world_height = max(
        values["actual_world_height_m"] for values in motion_checks.values()
    )
    penetration_tolerance = max(
        values["penetration_tolerance_m"] for values in motion_checks.values()
    )
    floor = rocketbox_review.add_fixed_floor(minimum, maximum, floor_z_m)
    return floor, {
        "status": "passed",
        "review_floor_z_m": floor_z_m,
        "constant_across_all_media": True,
        "actual_world_height_m": actual_world_height,
        "maximum_penetration_m": maximum_penetration,
        "penetration_tolerance_m": penetration_tolerance,
        "penetration_height_ratio": FLOOR_PENETRATION_HEIGHT_RATIO,
        "penetration_tolerance_min_m": FLOOR_PENETRATION_TOLERANCE_MIN_M,
        "gross_penetration_cap_m": FLOOR_PENETRATION_TOLERANCE_MAX_M,
        "minimum_support_distance_m": minimum_support_distance,
        "support_tolerance_m": FLOOR_SUPPORT_TOLERANCE_M,
        "support_frame_count": support_frame_count,
        "motions": motion_checks,
    }


def clear_camera_drivers(camera):
    try:
        camera.driver_remove("location")
    except TypeError:
        pass


def configure_follow_camera(
    camera,
    target,
    follow_bone_name,
    center_offset,
    direction,
    ortho_scale,
    frame_start,
):
    clear_camera_drivers(camera)
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    follow_point = rocketbox_review.joint_head_world(target, follow_bone_name)
    center = follow_point + Vector(center_offset)
    direction = Vector(direction).normalized()
    camera.location = center + direction*8.0
    rocketbox_review.look_at(camera, center)
    camera.data.ortho_scale = ortho_scale
    offsets = [camera.location[index] - follow_point[index] for index in range(3)]
    driver_curves = camera.driver_add("location")
    transform_types = ("LOC_X", "LOC_Y", "LOC_Z")
    for index, curve in enumerate(driver_curves):
        variable = curve.driver.variables.new()
        variable.name = "follow"
        variable.type = "TRANSFORMS"
        driver_target = variable.targets[0]
        driver_target.id = target
        driver_target.bone_target = follow_bone_name
        driver_target.transform_type = transform_types[index]
        driver_target.transform_space = "WORLD_SPACE"
        curve.driver.expression = f"follow + {offsets[index]:.12f}"


def orthographic_scale_for_extents(
    horizontal_extent_m,
    vertical_extent_m,
    render_size,
    margin_ratio,
):
    horizontal_extent = float(horizontal_extent_m)
    vertical_extent = float(vertical_extent_m)
    render_width, render_height = render_size
    margin = float(margin_ratio)
    values = (horizontal_extent, vertical_extent, render_width, render_height, margin)
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
        raise RuntimeError("camera extents, dimensions, and margin must be positive")
    aspect_ratio = float(render_width)/float(render_height)
    return max(horizontal_extent, vertical_extent*aspect_ratio)*margin


def configure_view_camera(camera, view_name, target, bones, bounds, frame_start):
    dimensions = bounds["maximum_dimensions"]
    height = max(float(dimensions.z), 1.0)
    width_x = max(float(dimensions.x), 0.5)
    width_y = max(float(dimensions.y), 0.5)
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    first_center = (bounds["first_minimum"] + bounds["first_maximum"])/2.0
    root_point = rocketbox_review.joint_head_world(target, bones["root"])
    pelvis_point = rocketbox_review.joint_head_world(target, bones["pelvis"])
    if view_name in ("front", "side"):
        horizontal_extent = width_x if view_name == "front" else width_y
        vertical_extent = height
        margin_ratio = BODY_CAMERA_MARGIN_RATIO
        direction = (
            (0.0, -1.0, 0.04)
            if view_name == "front"
            else (1.0, 0.0, 0.04)
        )
        ortho_scale = orthographic_scale_for_extents(
            horizontal_extent,
            vertical_extent,
            VIDEO_SIZE,
            margin_ratio,
        )
        configure_follow_camera(
            camera,
            target,
            bones["root"],
            first_center - root_point,
            direction,
            ortho_scale,
            frame_start,
        )
    else:
        horizontal_extent = math.hypot(width_x, width_y)
        vertical_extent = height*FEET_CAMERA_HEIGHT_RATIO
        margin_ratio = FEET_CAMERA_MARGIN_RATIO
        ortho_scale = orthographic_scale_for_extents(
            horizontal_extent,
            vertical_extent,
            VIDEO_SIZE,
            margin_ratio,
        )
        feet_center = Vector(
            (
                first_center.x,
                first_center.y,
                bounds["first_minimum"].z + height*0.20,
            )
        )
        configure_follow_camera(
            camera,
            target,
            bones["pelvis"],
            feet_center - pelvis_point,
            (1.0, -1.0, 0.06),
            ortho_scale,
            frame_start,
        )
    aspect_ratio = float(VIDEO_SIZE[0])/float(VIDEO_SIZE[1])
    return {
        "camera_mode": VIEW_CAMERA_MODES[view_name],
        "render_width": VIDEO_SIZE[0],
        "render_height": VIDEO_SIZE[1],
        "render_aspect_ratio": aspect_ratio,
        "horizontal_extent_m": horizontal_extent,
        "vertical_extent_m": vertical_extent,
        "margin_ratio": margin_ratio,
        "orthographic_scale": ortho_scale,
        "vertical_frustum_extent_m": ortho_scale/aspect_ratio,
    }


def validate_camera_bounds(
    scene,
    camera,
    meshes,
    foot_toe_indices,
    view_name,
    frame_start,
    frame_end,
):
    coordinates = []
    side_coordinates = {"left": [], "right": []}
    sampled_vertex_count = 0
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for mesh in meshes:
            if view_name == "feet":
                for side in ("left", "right"):
                    indices = foot_toe_indices[side][mesh.name]
                    if not indices:
                        continue
                    points = evaluated_mesh_world_points(mesh, indices)
                    sampled_vertex_count += len(points)
                    for point in points:
                        projected = world_to_camera_view(scene, camera, point)
                        side_coordinates[side].append(
                            (
                                float(projected.x),
                                float(projected.y),
                                float(projected.z),
                            )
                        )
            else:
                points = evaluated_mesh_world_points(mesh)
                sampled_vertex_count += len(points)
                for point in points:
                    projected = world_to_camera_view(scene, camera, point)
                    coordinates.append(
                        (float(projected.x), float(projected.y), float(projected.z))
                    )
    if view_name == "feet":
        side_checks = {}
        for side in ("left", "right"):
            values = side_coordinates[side]
            if not values or any(
                z <= 0.0
                or x < CAMERA_BOUND_MARGIN
                or x > 1.0 - CAMERA_BOUND_MARGIN
                or y < CAMERA_BOUND_MARGIN
                or y > 1.0 - CAMERA_BOUND_MARGIN
                for x, y, z in values
            ):
                raise RuntimeError(f"{side} Foot/Toe mesh is outside feet camera bounds")
            side_checks[side] = {
                "sampled_vertex_count": len(values),
                "minimum_x": min(value[0] for value in values),
                "maximum_x": max(value[0] for value in values),
                "minimum_y": min(value[1] for value in values),
                "maximum_y": max(value[1] for value in values),
            }
        return {
            "status": "passed",
            "geometry": "bilateral_weighted_foot_toe_vertices",
            "sampled_vertex_count": sampled_vertex_count,
            "sides": side_checks,
        }
    if not coordinates or any(
        z <= 0.0
        or x < CAMERA_BOUND_MARGIN
        or x > 1.0 - CAMERA_BOUND_MARGIN
        or y < CAMERA_BOUND_MARGIN
        or y > 1.0 - CAMERA_BOUND_MARGIN
        for x, y, z in coordinates
    ):
        raise RuntimeError(f"deformed mesh is outside {view_name} camera bounds")
    return {
        "status": "passed",
        "geometry": "all_vertices",
        "sampled_vertex_count": sampled_vertex_count,
        "minimum_x": min(value[0] for value in coordinates),
        "maximum_x": max(value[0] for value in coordinates),
        "minimum_y": min(value[1] for value in coordinates),
        "maximum_y": max(value[1] for value in coordinates),
    }


def stage_media_paths(bind_dir):
    return {
        name: rocketbox_review.make_staged_path(bind_dir/filename)
        for name, filename in CANONICAL_MEDIA.items()
    }


def render_motion(
    scene,
    camera,
    target,
    meshes,
    bones,
    foot_toe_indices,
    motion,
    details,
    staged_paths,
):
    target.animation_data.action = details["action"]
    scene.frame_start = details["frame_start"]
    scene.frame_end = details["frame_end"]
    expected_frame_count = details["expected_frame_count"]
    if scene.frame_end - scene.frame_start + 1 != expected_frame_count:
        raise RuntimeError(f"render frame range is invalid for {motion}")
    scene.frame_set(details["frame_start"])
    bounds_checks = {}
    for view_name in REQUIRED_VIEWS:
        camera_setup = configure_view_camera(
            camera,
            view_name,
            target,
            bones,
            details["bounds"],
            details["frame_start"],
        )
        bounds_check = validate_camera_bounds(
            scene,
            camera,
            meshes,
            foot_toe_indices,
            view_name,
            details["frame_start"],
            details["frame_end"],
        )
        bounds_check["camera_setup"] = camera_setup
        bounds_checks[view_name] = bounds_check
        if any(mesh.hide_render for mesh in meshes):
            raise RuntimeError("bound target mesh must remain visible in every video")
        output_path = staged_paths[f"{motion}_{view_name}"]
        rocketbox_review.configure_video_output(scene, output_path)
        print(f"Writing media: {output_path}")
        bpy.ops.render.render(animation=True)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"video render did not produce media: {motion} {view_name}")
    return bounds_checks


def render_bind_contact_sheet(scene, camera, target, prepared, staged_path):
    details = prepared["walk"]
    target.animation_data.action = details["action"]
    scene.frame_start = details["original_frame_start"]
    contact_frame_end = details["original_frame_end"] - 1
    scene.frame_end = contact_frame_end
    rocketbox_review.render_contact_sheet(
        scene,
        camera,
        staged_path,
        target,
        details["original_frame_start"],
        contact_frame_end,
    )


def validate_staged_media(staged_paths, prepared):
    media = {}
    for motion in REQUIRED_MOTIONS:
        expected_frame_count = prepared[motion]["expected_frame_count"]
        for view_name in REQUIRED_VIEWS:
            name = f"{motion}_{view_name}"
            result = rocketbox_review.validate_video(
                staged_paths[name], expected_frame_count
            )
            if result["sampled_luma_range"] < rocketbox_review.NONBLANK_LUMA_RANGE:
                raise RuntimeError(f"video is blank or nearly blank: {name}")
            media[name] = result
    contact = rocketbox_review.validate_png(staged_paths["contact_sheet"])
    if contact["pixel_range"] < rocketbox_review.NONBLANK_PNG_RANGE:
        raise RuntimeError("bind contact sheet is blank or nearly blank")
    media["contact_sheet"] = contact
    return {"status": "passed", "media": media}


def publish_media(staged_paths, bind_dir):
    for name, filename in CANONICAL_MEDIA.items():
        destination = bind_dir/filename
        os.replace(staged_paths[name], destination)
        print(f"Wrote media: {destination}")


def publish_review_manifest(inputs, automatic_checks):
    bind_dir = inputs["bind_dir"]
    videos = {}
    for motion in REQUIRED_MOTIONS:
        videos[motion] = {}
        for view_name in REQUIRED_VIEWS:
            filename = f"{motion}_{view_name}.mp4"
            path = bind_dir/filename
            videos[motion][view_name] = {
                "filename": filename,
                "sha256": rocketbox_review.sha256_file(path),
            }
    glbs = {}
    for motion in REQUIRED_MOTIONS:
        path = inputs["glb_paths"][motion]
        glbs[motion] = {
            "filename": CANONICAL_GLBS[motion],
            "sha256": rocketbox_review.sha256_file(path),
        }
    contact_path = bind_dir/CANONICAL_MEDIA["contact_sheet"]
    payload = {
        "schema_version": REVIEW_MANIFEST_SCHEMA,
        "asset_id": inputs["manifest"]["asset_id"],
        "bind_manifest_sha256": inputs["manifest_sha256"],
        "bound_blend": {
            "filename": "bound.blend",
            "sha256": inputs["blend_sha256"],
        },
        "bind_metrics": {
            "filename": "bind_metrics.json",
            "sha256": inputs["metrics_sha256"],
        },
        "floor_z_m": inputs["floor_z_m"],
        "action_names": dict(inputs["action_names"]),
        "glbs": glbs,
        "videos": videos,
        "contact_sheet": {
            "filename": CANONICAL_MEDIA["contact_sheet"],
            "sha256": rocketbox_review.sha256_file(contact_path),
        },
        "automatic_checks": automatic_checks,
    }
    rocketbox_review.atomic_write_json(bind_dir/"review_manifest.json", payload)


def automatic_checks(inputs, prepared, floor_check, camera_checks, media_validation):
    action_checks = {}
    for motion in REQUIRED_MOTIONS:
        details = prepared[motion]
        action_checks[motion] = {
            "status": "passed",
            "action_name": details["action_name"],
            "original_frame_start": details["original_frame_start"],
            "original_frame_end": details["original_frame_end"],
            "loop_cycle_count": LOOP_CYCLE_COUNT,
            "rendered_frame_count": details["expected_frame_count"],
        }
    return {
        "overall": "passed",
        "inputs": {
            "status": "passed",
            "bound_blend_sha256": inputs["blend_sha256"],
            "bind_metrics_sha256": inputs["metrics_sha256"],
            "fixed_floor_z_m": inputs["floor_z_m"],
            "manifest_action_names_used": True,
        },
        "actions": action_checks,
        "fixed_floor": floor_check,
        "camera_bounds": camera_checks,
        "media_validation": media_validation,
    }


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_json_bytes(value, description):
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {description}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must contain a JSON object")
    return payload


def validate_snapshot_descriptor(descriptor, filename, value, description):
    if not isinstance(descriptor, dict) or set(descriptor) != {"filename", "sha256"}:
        raise RuntimeError(f"{description} descriptor must contain filename and sha256")
    if descriptor["filename"] != filename:
        raise RuntimeError(f"{description} filename is not canonical")
    if descriptor["sha256"] != sha256_bytes(value):
        raise RuntimeError(f"{description} hash does not match current snapshot")


def capture_artifact_snapshot(inputs):
    bind_dir = inputs["bind_dir"]
    paths = {
        "bind_manifest.json": inputs["manifest_path"],
        "review_manifest.json": rocketbox_review.ensure_direct_regular_file(
            bind_dir/"review_manifest.json", bind_dir, "review manifest"
        ),
        "bound.blend": inputs["blend_path"],
        "bind_metrics.json": inputs["metrics_path"],
        "bound_walk.glb": inputs["glb_paths"]["walk"],
        "bound_idle.glb": inputs["glb_paths"]["idle"],
    }
    for filename in CANONICAL_MEDIA.values():
        paths[filename] = rocketbox_review.ensure_direct_regular_file(
            bind_dir/filename, bind_dir, filename
        )
    data = {filename: path.read_bytes() for filename, path in paths.items()}
    if sha256_bytes(data["bind_manifest.json"]) != inputs["manifest_sha256"]:
        raise RuntimeError("bind manifest changed after input authentication")
    review = load_json_bytes(data["review_manifest.json"], "review manifest")
    if review.get("schema_version") != REVIEW_MANIFEST_SCHEMA:
        raise RuntimeError("review manifest schema is invalid")
    if review.get("asset_id") != inputs["manifest"]["asset_id"]:
        raise RuntimeError("review manifest asset_id is stale")
    if review.get("bind_manifest_sha256") != sha256_bytes(data["bind_manifest.json"]):
        raise RuntimeError("review manifest bind hash is stale")
    if review.get("bound_blend") != inputs["manifest"]["bound_blend"]:
        raise RuntimeError("bound blend bind-to-review descriptor mismatch")
    validate_snapshot_descriptor(
        review.get("bound_blend"), "bound.blend", data["bound.blend"], "bound blend"
    )
    if (
        review.get("bind_metrics")
        != inputs["manifest"]["artifacts"]["bind_metrics"]
    ):
        raise RuntimeError("bind metrics bind-to-review descriptor mismatch")
    validate_snapshot_descriptor(
        review.get("bind_metrics"),
        "bind_metrics.json",
        data["bind_metrics.json"],
        "bind metrics",
    )
    if review.get("floor_z_m") != inputs["floor_z_m"]:
        raise RuntimeError("review manifest fixed floor is stale")
    if review.get("action_names") != inputs["action_names"]:
        raise RuntimeError("review manifest action names are stale")
    glbs = review.get("glbs")
    if not isinstance(glbs, dict) or set(glbs) != set(REQUIRED_MOTIONS):
        raise RuntimeError("review manifest GLB set is invalid")
    if glbs != inputs["manifest"]["glbs"]:
        raise RuntimeError("GLB bind-to-review descriptors mismatch")
    for motion in REQUIRED_MOTIONS:
        filename = CANONICAL_GLBS[motion]
        validate_snapshot_descriptor(glbs[motion], filename, data[filename], f"{motion} GLB")
    videos = review.get("videos")
    if not isinstance(videos, dict) or set(videos) != set(REQUIRED_MOTIONS):
        raise RuntimeError("review manifest video motion set is invalid")
    for motion in REQUIRED_MOTIONS:
        views = videos[motion]
        if not isinstance(views, dict) or set(views) != set(REQUIRED_VIEWS):
            raise RuntimeError(f"review manifest {motion} view set is invalid")
        for view_name in REQUIRED_VIEWS:
            filename = f"{motion}_{view_name}.mp4"
            validate_snapshot_descriptor(
                views[view_name], filename, data[filename], f"{motion} {view_name} video"
            )
    contact_filename = CANONICAL_MEDIA["contact_sheet"]
    validate_snapshot_descriptor(
        review.get("contact_sheet"),
        contact_filename,
        data[contact_filename],
        "bind contact sheet",
    )
    checks = review.get("automatic_checks")
    if not isinstance(checks, dict) or checks.get("overall") != "passed":
        raise RuntimeError("review manifest automatic checks have not passed")
    return {
        "schema_version": ARTIFACT_SNAPSHOT_SCHEMA,
        "asset_id": inputs["manifest"]["asset_id"],
        "bind_manifest_sha256": sha256_bytes(data["bind_manifest.json"]),
        "review_manifest_sha256": sha256_bytes(data["review_manifest.json"]),
        "bound_blend": {
            "filename": "bound.blend",
            "sha256": sha256_bytes(data["bound.blend"]),
        },
        "glbs": {
            motion: {
                "filename": CANONICAL_GLBS[motion],
                "sha256": sha256_bytes(data[CANONICAL_GLBS[motion]]),
            }
            for motion in REQUIRED_MOTIONS
        },
        "videos": {
            motion: {
                view_name: {
                    "filename": f"{motion}_{view_name}.mp4",
                    "sha256": sha256_bytes(data[f"{motion}_{view_name}.mp4"]),
                }
                for view_name in REQUIRED_VIEWS
            }
            for motion in REQUIRED_MOTIONS
        },
        "bind_metrics": {
            "filename": "bind_metrics.json",
            "sha256": sha256_bytes(data["bind_metrics.json"]),
        },
        "contact_sheet": {
            "filename": CANONICAL_MEDIA["contact_sheet"],
            "sha256": sha256_bytes(data[CANONICAL_MEDIA["contact_sheet"]]),
        },
    }


def read_pixel_qa(path, bind_dir):
    canonical = (bind_dir/PIXEL_QA_FILENAME).absolute()
    if Path(path).absolute() != canonical:
        raise RuntimeError(f"pixel QA JSON must be {canonical}")
    absolute = rocketbox_review.ensure_direct_regular_file(
        canonical, bind_dir, "pixel QA JSON"
    )
    value = absolute.read_bytes()
    return load_json_bytes(value, "pixel QA"), sha256_bytes(value), absolute


def validate_pixel_qa(pixel_qa, asset_id, decision):
    if pixel_qa.get("schema_version") != PIXEL_QA_SCHEMA:
        raise RuntimeError("pixel QA schema is invalid")
    if pixel_qa.get("asset_id") != asset_id:
        raise RuntimeError("pixel QA asset_id does not match")
    if pixel_qa.get("decision") != decision:
        raise RuntimeError("pixel QA decision does not match requested record")
    if not isinstance(pixel_qa.get("reviewer"), str) or not pixel_qa["reviewer"].strip():
        raise RuntimeError("pixel QA reviewer is required")
    if not isinstance(pixel_qa.get("reviewed_at"), str) or not pixel_qa["reviewed_at"].strip():
        raise RuntimeError("pixel QA reviewed_at is required")
    if not isinstance(pixel_qa.get("notes", ""), str):
        raise RuntimeError("pixel QA notes must be text")
    if not isinstance(pixel_qa.get("expected_artifact_snapshot"), dict):
        raise RuntimeError("pixel QA expected_artifact_snapshot is required")
    checks = pixel_qa.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(PIXEL_QA_CHECKS):
        raise RuntimeError("pixel QA checks are incomplete")
    if any(not isinstance(value, bool) for value in checks.values()):
        raise RuntimeError("pixel QA check values must be boolean")
    if decision == "ready" and not all(checks.values()):
        raise RuntimeError("ready pixel QA requires every visual check to pass")
    if decision == "rejected" and all(checks.values()):
        raise RuntimeError("rejected pixel QA must identify a failed visual check")


def validate_expected_artifact_snapshot(pixel_qa, current_snapshot):
    expected_snapshot = pixel_qa.get("expected_artifact_snapshot")
    if expected_snapshot != current_snapshot:
        raise RuntimeError("pixel QA expected artifact snapshot is stale")


def build_direct_attempt_payload(
    asset_id,
    decision,
    pixel_qa,
    pixel_qa_sha256,
    pixel_qa_path,
    snapshot,
):
    if decision == "ready":
        schema_version = DIRECT_ATTEMPT_READY_SCHEMA
        status = "ready"
    else:
        schema_version = DIRECT_ATTEMPT_REJECTED_SCHEMA
        status = "rejected"
    return {
        "schema_version": schema_version,
        "asset_id": asset_id,
        "status": status,
        "bind_manifest_sha256": snapshot["bind_manifest_sha256"],
        "review_manifest_sha256": snapshot["review_manifest_sha256"],
        "bound_blend": snapshot["bound_blend"],
        "glbs": snapshot["glbs"],
        "videos": snapshot["videos"],
        "pixel_qa": {
            "filename": PIXEL_QA_FILENAME,
            "sha256": pixel_qa_sha256,
        },
        "pixel_qa_metadata": pixel_qa,
        "pixel_qa_sha256": pixel_qa_sha256,
        "pixel_qa_source": pixel_qa_path.name,
        "bind_metrics": snapshot["bind_metrics"],
        "contact_sheet": snapshot["contact_sheet"],
    }


def record_direct_attempt(args):
    bind_dir = validate_bind_dir(args.bind_dir)
    invalidate_direct_attempt_outputs(bind_dir)
    inputs = load_bind_inputs(args, bind_dir)
    pixel_qa, pixel_qa_sha256, pixel_qa_path = read_pixel_qa(
        args.pixel_qa_json, bind_dir
    )
    validate_pixel_qa(pixel_qa, args.asset_id, args.record_direct_attempt)
    snapshot_before = capture_artifact_snapshot(inputs)
    validate_expected_artifact_snapshot(pixel_qa, snapshot_before)
    destination_name = f"direct_attempt_{args.record_direct_attempt}.json"
    payload = build_direct_attempt_payload(
        args.asset_id,
        args.record_direct_attempt,
        pixel_qa,
        pixel_qa_sha256,
        pixel_qa_path,
        snapshot_before,
    )
    try:
        rocketbox_review.atomic_write_json(bind_dir/destination_name, payload)
        snapshot_after = capture_artifact_snapshot(inputs)
        current_pixel_qa_sha256 = rocketbox_review.sha256_file(pixel_qa_path)
        if (
            snapshot_before != snapshot_after
            or pixel_qa_sha256 != current_pixel_qa_sha256
        ):
            raise RuntimeError(
                "review or pixel QA changed while recording direct attempt"
            )
    except BaseException:
        safe_unlink_outputs(bind_dir, DIRECT_ATTEMPT_FILES)
        raise
    print(
        f"HY3D_ROCKETBOX_DIRECT_ATTEMPT_RECORDED "
        f"asset_id={args.asset_id} decision={args.record_direct_attempt}"
    )
    return 0


def render_review(args):
    bind_dir = validate_bind_dir(args.bind_dir)
    invalidate_render_outputs(bind_dir)
    inputs = load_bind_inputs(args, bind_dir)
    staged_paths = {}
    try:
        load_bound_blend(inputs["blend_path"], inputs["blend_sha256"])
        target, meshes = target_objects()
        bones = resolve_review_bones(target, inputs["manifest"])
        foot_toe_indices = foot_toe_weighted_vertex_indices(meshes, bones)
        prepared = prepare_actions(target, meshes, bones, inputs["action_names"])
        minimum, maximum = combined_bounds(prepared)
        scene = rocketbox_review.configure_scene(0, 1)
        scene.render.fps = FPS
        floor, floor_check = fixed_floor_check(
            target,
            meshes,
            prepared,
            inputs["floor_z_m"],
            foot_toe_indices,
        )
        rocketbox_review.add_lighting(minimum, maximum)
        camera = rocketbox_review.make_camera()
        staged_paths = stage_media_paths(inputs["bind_dir"])
        camera_checks = {}
        for motion in REQUIRED_MOTIONS:
            camera_checks[motion] = render_motion(
                scene,
                camera,
                target,
                meshes,
                bones,
                foot_toe_indices,
                motion,
                prepared[motion],
                staged_paths,
            )
        if floor.hide_render:
            raise RuntimeError("fixed review floor must be visible in every output")
        render_bind_contact_sheet(
            scene, camera, target, prepared, staged_paths["contact_sheet"]
        )
        media_validation = validate_staged_media(staged_paths, prepared)
        checks = automatic_checks(
            inputs, prepared, floor_check, camera_checks, media_validation
        )
        publish_media(staged_paths, inputs["bind_dir"])
        publish_review_manifest(inputs, checks)
    finally:
        rocketbox_review.cleanup_paths(staged_paths.values())
    print(f"HY3D_ROCKETBOX_REVIEW_OK asset_id={args.asset_id}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.record_direct_attempt is not None:
        return record_direct_attempt(args)
    return render_review(args)


if __name__ == "__main__":
    raise SystemExit(main())
