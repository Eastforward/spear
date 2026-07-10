#!/usr/bin/env python3

#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Render canonical bound-avatar Rocketbox neutral-walk review evidence."""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


CANONICAL_MEDIA = {
    "front": "front.mp4",
    "side": "side.mp4",
    "top": "top.mp4",
    "joints": "joints.mp4",
    "feet": "feet.mp4",
    "source_target": "source_target.mp4",
    "contact_sheet": "contact_sheet.png",
}
VIDEO_MEDIA = (
    "front",
    "side",
    "top",
    "joints",
    "feet",
    "source_target",
)
READINESS_FILES = ("retarget_manifest.json", "motion_review.json")
EXPECTED_ASSET_IDS = (
    "rocketbox_male_adult_01",
    "rocketbox_female_adult_01",
)
VIEW_CAMERA_MODES = {
    "front": "path",
    "side": "path",
    "top": "path",
    "joints": "root_follow",
    "feet": "root_follow",
    "source_target": "root_follow",
}
AXIS_LABELS = ("UP +Z", "FRONT -Y")
LABEL_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VIDEO_SIZE = (1280, 720)
CONTACT_TILE_SIZE = (480, 540)
CONTACT_GRID = (4, 2)
FPS = 30
GAIT_CYCLE_COUNT = 2
CONTACT_FRAME_COUNT = 8
SOURCE_STICK_SCALE = 0.74
SOURCE_STICK_OFFSET_X_M = -0.92
MESH_PENETRATION_TOLERANCE_M = 1.0e-5
ROOT_DIRECTION_DOT_FLOOR = 0.999
NORMALIZED_LOOP_RESIDUAL_TOLERANCE = 2.0e-3
REST_ANGLE_LIMIT_DEG = 90.0
NONBLANK_LUMA_RANGE = 12.0
NONBLANK_PNG_RANGE = 0.03
PATH_BODY_ORTHO_SCALE = 4.0
JOINTS_ORTHO_SCALE = 2.65
SOURCE_TARGET_ORTHO_SCALE = 4.0
TOP_GUIDE_MARGIN_X = 0.90
TOP_GUIDE_MARGIN_Y = 0.25
LABEL_GREEN_FLOOR = 150
LABEL_RED_CEILING = 110
LABEL_BLUE_CEILING = 170
LABEL_GREEN_DOMINANCE = 45
LABEL_MIN_PIXELS = 40
LABEL_BORDER_MARGIN_PX = 4
ARROW_RED_FLOOR = 150
ARROW_RED_DOMINANCE = 80
ARROW_MIN_PIXELS = 80

CORE_BONES = (
    "Bip01 Pelvis",
    "Bip01 Spine",
    "Bip01 Spine1",
    "Bip01 Spine2",
    "Bip01 Neck",
    "Bip01 Head",
    "Bip01 L Clavicle",
    "Bip01 L UpperArm",
    "Bip01 L Forearm",
    "Bip01 L Hand",
    "Bip01 R Clavicle",
    "Bip01 R UpperArm",
    "Bip01 R Forearm",
    "Bip01 R Hand",
    "Bip01 L Thigh",
    "Bip01 L Calf",
    "Bip01 L Foot",
    "Bip01 L Toe0",
    "Bip01 R Thigh",
    "Bip01 R Calf",
    "Bip01 R Foot",
    "Bip01 R Toe0",
)
STICK_BONES = CORE_BONES
STICK_LINKS = (
    ("Bip01 Pelvis", "Bip01 Spine"),
    ("Bip01 Spine", "Bip01 Spine1"),
    ("Bip01 Spine1", "Bip01 Spine2"),
    ("Bip01 Spine2", "Bip01 Neck"),
    ("Bip01 Neck", "Bip01 Head"),
    ("Bip01 Spine2", "Bip01 L Clavicle"),
    ("Bip01 L Clavicle", "Bip01 L UpperArm"),
    ("Bip01 L UpperArm", "Bip01 L Forearm"),
    ("Bip01 L Forearm", "Bip01 L Hand"),
    ("Bip01 Spine2", "Bip01 R Clavicle"),
    ("Bip01 R Clavicle", "Bip01 R UpperArm"),
    ("Bip01 R UpperArm", "Bip01 R Forearm"),
    ("Bip01 R Forearm", "Bip01 R Hand"),
    ("Bip01 Pelvis", "Bip01 L Thigh"),
    ("Bip01 L Thigh", "Bip01 L Calf"),
    ("Bip01 L Calf", "Bip01 L Foot"),
    ("Bip01 L Foot", "Bip01 L Toe0"),
    ("Bip01 Pelvis", "Bip01 R Thigh"),
    ("Bip01 R Thigh", "Bip01 R Calf"),
    ("Bip01 R Calf", "Bip01 R Foot"),
    ("Bip01 R Foot", "Bip01 R Toe0"),
)
FOOT_BONES = (
    "Bip01 L Foot",
    "Bip01 L Toe0",
    "Bip01 R Foot",
    "Bip01 R Toe0",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SIGNAL_RE = re.compile(r"lavfi\.signalstats\.(YMIN|YMAX)=([0-9.]+)")


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--retarget-dir", type=Path, required=True)
    parser.add_argument("--source-motion-fbx", type=Path, required=True)
    args = parser.parse_args(argv)
    assert args.asset_id in EXPECTED_ASSET_IDS
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path, description):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must contain a JSON object")
    return value


def ensure_direct_regular_file(path, root, description):
    absolute = path.absolute()
    resolved = absolute.resolve()
    if absolute != resolved or resolved.parent != root.resolve() or not resolved.is_file():
        raise RuntimeError(f"{description} must be a direct non-symlink regular file")
    return resolved


def load_stage_manifest(args):
    absolute = args.retarget_dir.absolute()
    if absolute != absolute.resolve() or not absolute.is_dir():
        raise RuntimeError("retarget directory must be a direct non-symlink directory")
    manifest_path = ensure_direct_regular_file(
        absolute/"retarget_manifest.json", absolute, "retarget manifest"
    )
    manifest = load_json(manifest_path, "retarget manifest")
    if manifest.get("asset_id") != args.asset_id:
        raise RuntimeError("retarget manifest asset_id does not match --asset-id")
    return manifest


def invalidate_review_readiness(retarget_dir):
    for filename in READINESS_FILES:
        path = retarget_dir/filename
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def atomic_write_json(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        print(f"Wrote manifest: {path}")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def make_staged_path(final_path):
    with tempfile.NamedTemporaryFile(
        dir=final_path.parent,
        prefix=f".{final_path.stem}.",
        suffix=final_path.suffix,
        delete=False,
    ) as stream:
        return Path(stream.name)


def stage_media_paths(retarget_dir):
    return {
        name: make_staged_path(retarget_dir/filename)
        for name, filename in CANONICAL_MEDIA.items()
    }


def cleanup_paths(paths):
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def publish_staged_media(staged_paths, retarget_dir):
    for name, filename in CANONICAL_MEDIA.items():
        final_path = retarget_dir/filename
        os.replace(staged_paths[name], final_path)
        print(f"Wrote media: {final_path}")


def publish_manifest(manifest, automatic_checks, manifest_path):
    manifest["media"] = dict(CANONICAL_MEDIA)
    manifest["automatic_checks"] = automatic_checks
    atomic_write_json(manifest_path, manifest)


def require_mapping_value(metrics, key, expected):
    actual = metrics["mapping"].get(key)
    if actual != expected:
        raise RuntimeError(f"Task 3 mapping invariant failed: {key}={actual!r}")


def validate_task3_inputs(args, manifest):
    if not manifest["schema_version"] == "rocketbox_retarget_manifest_v1":
        raise RuntimeError("retarget manifest schema is not canonical")
    if manifest.get("binding") != {
        "target_asset_id": args.asset_id,
        "target_mesh_bound": True,
        "official_textures_attached": True,
    }:
        raise RuntimeError("retarget manifest binding provenance is invalid")
    artifacts = manifest.get("artifacts")
    if artifacts != {
        "blend": "retarget.blend",
        "glb": "retarget.glb",
        "metrics": "retarget_metrics.json",
    }:
        raise RuntimeError("retarget manifest artifacts are not canonical")
    hashes = manifest.get("immutable_input_hashes")
    if not isinstance(hashes, dict) or any(
        _SHA256_RE.fullmatch(value) is None
        for value in hashes.values()
        if isinstance(value, str)
    ):
        raise RuntimeError("retarget manifest hashes are malformed")
    required_hashes = {
        "avatar_fbx",
        "motion_fbx",
        "source_review",
        "body_color_texture",
        "head_color_texture",
        "opacity_color_texture",
        "retarget_glb",
    }
    if not isinstance(hashes, dict) or set(hashes) != required_hashes:
        raise RuntimeError("retarget manifest immutable hash keys are invalid")

    glb_path = ensure_direct_regular_file(
        args.retarget_dir/"retarget.glb", args.retarget_dir, "retarget GLB"
    )
    current_glb_sha256 = sha256_file(glb_path)
    if current_glb_sha256 != manifest["immutable_input_hashes"]["retarget_glb"]:
        raise RuntimeError("current retarget GLB hash does not match Task 3 provenance")
    source_motion = args.source_motion_fbx.resolve()
    if not source_motion.is_file():
        raise RuntimeError("source motion FBX is missing")
    if sha256_file(source_motion) != hashes["motion_fbx"]:
        raise RuntimeError("source motion FBX hash does not match Task 3 provenance")

    metrics_path = ensure_direct_regular_file(
        args.retarget_dir/"retarget_metrics.json", args.retarget_dir, "retarget metrics"
    )
    metrics = load_json(metrics_path, "retarget metrics")
    if metrics.get("schema_version") != "rocketbox_retarget_metrics_v1":
        raise RuntimeError("retarget metrics schema is invalid")
    if metrics.get("asset_id") != args.asset_id:
        raise RuntimeError("retarget metrics asset_id does not match --asset-id")
    source_animation = metrics.get("source_animation", {})
    if source_animation.get("fps") != FPS:
        raise RuntimeError("source animation is not 30 fps")
    frame_start = source_animation.get("frame_start")
    frame_end = source_animation.get("frame_end")
    if not isinstance(frame_start, int) or not isinstance(frame_end, int) or frame_end <= frame_start:
        raise RuntimeError("source animation frame range is invalid")

    invariants = metrics.get("invariants", {})
    if invariants.get("overall") != "passed" or invariants.get("errors") != []:
        raise RuntimeError("Task 3 invariants did not pass")
    required_true = (
        "mapped_80_of_80",
        "target_mesh_unchanged",
        "target_only_blend",
        "target_only_glb",
        "official_textures_attached",
        "glb_roundtrip_passed",
        "glb_skin_weights_preserved",
        "glb_material_bindings_preserved",
    )
    failed_true = [ key for key in required_true if invariants.get(key) is not True ]
    if failed_true or invariants.get("hierarchy_mismatch_count") != 0:
        raise RuntimeError(f"Task 3 invariant flags failed: {failed_true}")
    if invariants.get("space_invariant_max_abs_error", math.inf) > invariants.get(
        "space_invariant_tolerance", -math.inf
    ):
        raise RuntimeError("Task 3 parent-local space invariant failed")

    require_mapping_value(metrics, "mapped_bone_count", 80)
    require_mapping_value(metrics, "hierarchy_mismatches", [])
    mapped_bones = metrics["mapping"].get("mapped_bones", [])
    if len(mapped_bones) != 80 or not set(CORE_BONES).issubset(mapped_bones):
        raise RuntimeError("Task 3 core bone map is incomplete")
    rest = metrics.get("rest_angle_statistics", {})
    if not all(
        math.isfinite(float(rest.get(name, math.inf)))
        for name in ("minimum_deg", "mean_deg", "maximum_deg")
    ) or rest.get("maximum_deg", math.inf) > REST_ANGLE_LIMIT_DEG:
        raise RuntimeError("Task 3 rest-pose invariant failed")

    root = metrics.get("root_alignment", {})
    if root.get("endpoint_direction_dot_negative_y", -math.inf) < ROOT_DIRECTION_DOT_FLOOR:
        raise RuntimeError("Task 3 root direction invariant failed")
    if root.get("target_minimum_facing_travel_dot", -math.inf) < root.get(
        "facing_forward_dot_floor", math.inf
    ):
        raise RuntimeError("Task 3 forward-facing invariant failed")
    if root.get("maximum_facing_reconstruction_error", math.inf) > root.get(
        "facing_reconstruction_tolerance", -math.inf
    ):
        raise RuntimeError("Task 3 facing reconstruction invariant failed")
    loop = metrics.get("loop_residual", {})
    if loop.get("root_residual_after_cycle_displacement_m", math.inf) > 1.0e-6:
        raise RuntimeError("Task 3 root loop residual failed")
    if loop.get("normalized_bone_delta_residual", math.inf) > NORMALIZED_LOOP_RESIDUAL_TOLERANCE:
        raise RuntimeError("Task 3 normalized loop residual failed")

    roundtrip = metrics.get("roundtrip", {})
    if roundtrip.get("passed") is not True:
        raise RuntimeError("Task 3 GLB roundtrip failed")
    if roundtrip.get("maximum_world_joint_error_m", math.inf) > roundtrip.get(
        "joint_tolerance_m", -math.inf
    ):
        raise RuntimeError("Task 3 GLB joint roundtrip tolerance failed")
    weights = roundtrip.get("skin_weight_validation", {})
    if weights.get("passed") is not True or weights.get("errors") != []:
        raise RuntimeError("Task 3 GLB skin-weight invariant failed")
    materials = metrics.get("materials", {}).get("semantic_glb_bindings", {})
    if materials.get("passed") is not True or materials.get("errors") != []:
        raise RuntimeError("Task 3 GLB material invariant failed")
    mesh = metrics.get("mesh_invariants", {})
    if mesh.get("unchanged") is not True or mesh.get("before_retarget") != mesh.get(
        "after_retarget"
    ):
        raise RuntimeError("Task 3 target mesh invariant failed")

    return {
        "metrics": metrics,
        "metrics_sha256": sha256_file(metrics_path),
        "current_glb_sha256": current_glb_sha256,
        "frame_start": frame_start,
        "frame_end": frame_end,
    }


def load_retarget_blend(args):
    blend_path = ensure_direct_regular_file(
        args.retarget_dir/"retarget.blend", args.retarget_dir, "retarget blend"
    )
    print(f"Loading blend: {blend_path}")
    result = bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    if "FINISHED" not in result:
        raise RuntimeError("could not load retarget blend")


def target_objects():
    armatures = [ obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" ]
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH"
        and any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
    ]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError("retarget blend must contain one bound target mesh and armature")
    return armatures[0], meshes[0]


def validate_loaded_target(target, mesh, metrics):
    expected = metrics["mesh_invariants"]["after_retarget"]
    actual = {
        "vertex_count": len(mesh.data.vertices),
        "polygon_count": len(mesh.data.polygons),
        "uv_layer_count": len(mesh.data.uv_layers),
        "material_slot_count": len(mesh.material_slots),
        "material_slot_names": [
            slot.material.name if slot.material is not None else None
            for slot in mesh.material_slots
        ],
        "vertex_group_count": len(mesh.vertex_groups),
        "vertex_group_names": [ group.name for group in mesh.vertex_groups ],
        "bone_count": len(target.data.bones),
    }
    for name, value in actual.items():
        if value != expected[name]:
            raise RuntimeError(f"loaded target invariant failed: {name}")
    modifiers = [
        modifier
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object == target
    ]
    if len(modifiers) != 1:
        raise RuntimeError("loaded target mesh is not bound to the target armature")

    image_names = set()
    for slot in mesh.material_slots:
        material = slot.material
        if material is None or not material.use_nodes or material.node_tree is None:
            raise RuntimeError("loaded target has an invalid official material graph")
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                image = node.image
                path = Path(bpy.path.abspath(image.filepath))
                if image.packed_file is None and not path.is_file():
                    raise RuntimeError(f"loaded target texture is missing: {path}")
                image_names.add(Path(image.name).stem)
    expected_images = set(
        metrics["materials"]["validated_bindings"]["official_color_image_names"]
    )
    if not expected_images.issubset(image_names):
        raise RuntimeError("loaded target is missing official color textures")

    maximum_weight_sum_error = 0.0
    vertices_without_weights = 0
    for vertex in mesh.data.vertices:
        total = sum(influence.weight for influence in vertex.groups)
        if total == 0.0:
            vertices_without_weights += 1
        maximum_weight_sum_error = max(maximum_weight_sum_error, abs(total - 1.0))
    if vertices_without_weights or maximum_weight_sum_error > 1.0e-6:
        raise RuntimeError("loaded target skin weights are invalid")
    return {
        "material_slot_names": actual["material_slot_names"],
        "official_color_images": sorted(expected_images),
        "maximum_weight_sum_error": maximum_weight_sum_error,
        "vertices_without_weights": vertices_without_weights,
    }


def configure_scene(frame_start, frame_end):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.render.image_settings.color_mode = "RGB"
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.sync_mode = "NONE"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("rocketbox_motion_review_world")
    scene.world.color = (0.035, 0.045, 0.055)
    return scene


def evaluated_mesh_frame_data(mesh):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    try:
        points = [ evaluated.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices ]
        minimum = Vector(
            (
                min(point.x for point in points),
                min(point.y for point in points),
                min(point.z for point in points),
            )
        )
        maximum = Vector(
            (
                max(point.x for point in points),
                max(point.y for point in points),
                max(point.z for point in points),
            )
        )
        return minimum, maximum
    finally:
        evaluated.to_mesh_clear()


def joint_head_world(armature, bone_name):
    return armature.matrix_world @ armature.pose.bones[bone_name].head


def joint_tail_world(armature, bone_name):
    return armature.matrix_world @ armature.pose.bones[bone_name].tail


def foot_contact_metrics(target, frame_start, frame_end):
    positions = { name: [] for name in FOOT_BONES }
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for name in FOOT_BONES:
            positions[name].append(joint_head_world(target, name))
    result = {}
    for name, points in positions.items():
        minimum_z = min(point.z for point in points)
        contact_limit = minimum_z + 0.02
        slides = []
        contact_frames = 0
        for index, point in enumerate(points):
            if point.z <= contact_limit:
                contact_frames += 1
            if index + 1 < len(points):
                next_point = points[index + 1]
                if point.z <= contact_limit and next_point.z <= contact_limit:
                    slides.append(float((next_point.xy - point.xy).length))
        result[name] = {
            "minimum_world_z_m": float(minimum_z),
            "contact_height_threshold_m": float(contact_limit),
            "contact_frame_count": contact_frames,
            "contact_step_count": len(slides),
            "maximum_contact_xy_velocity_m_per_s": max(slides)*FPS if slides else 0.0,
            "accumulated_contact_slide_m": sum(slides),
        }
    return result


def scan_floor_calibration(mesh, target, metrics, frame_start, frame_end):
    frame_minimum_z = {}
    original_bounds = []
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        minimum, maximum = evaluated_mesh_frame_data(mesh)
        frame_minimum_z[frame] = float(minimum.z)
        original_bounds.append((minimum, maximum))
    review_floor_z = min(frame_minimum_z.values())
    mesh_clearance_m = {
        str(frame): value - review_floor_z for frame, value in frame_minimum_z.items()
    }
    maximum_penetration = max(
        max(0.0, review_floor_z - value) for value in frame_minimum_z.values()
    )
    if maximum_penetration > MESH_PENETRATION_TOLERANCE_M:
        raise RuntimeError("evaluated target mesh penetrates the fixed review floor")
    reviewed_static_floor = metrics["floor_metrics"]["reviewed_floor_z_m"]
    feet = foot_contact_metrics(target, frame_start, frame_end)
    return {
        "status": "passed",
        "reviewed_static_floor_z_m": reviewed_static_floor,
        "review_floor_z_m": review_floor_z,
        "runtime_constant_z_lift_m": reviewed_static_floor - review_floor_z,
        "mesh_clearance_m": mesh_clearance_m,
        "minimum_mesh_clearance_m": min(mesh_clearance_m.values()),
        "maximum_mesh_clearance_m": max(mesh_clearance_m.values()),
        "maximum_penetration_below_review_floor_m": maximum_penetration,
        "penetration_tolerance_m": MESH_PENETRATION_TOLERANCE_M,
        "foot_bones": feet,
        "visible_foot_evidence_authoritative": True,
        "authoritative_media": ["feet.mp4", "contact_sheet.png"],
        "original_cycle_frame_start": frame_start,
        "original_cycle_frame_end": frame_end,
        "original_bounds": original_bounds,
    }


def add_review_cycle_modifiers(target, frame_start, frame_end):
    if target.animation_data is None or target.animation_data.action is None:
        raise RuntimeError("target armature has no active action")
    action = target.animation_data.action
    action_start, action_end = map(float, action.frame_range)
    if round(action_start) != frame_start or round(action_end) != frame_end:
        raise RuntimeError("target action frame range differs from Task 3 metrics")
    for curve in action.fcurves:
        modifier = curve.modifiers.new(type="CYCLES")
        modifier.mode_before = "REPEAT"
        if curve.data_path == "location":
            modifier.mode_after = "REPEAT_OFFSET"
        else:
            modifier.mode_after = "REPEAT"
    return action


def scan_review_motion(mesh, target, frame_start, review_frame_end):
    minima = []
    maxima = []
    pelvis_positions = {}
    for frame in range(frame_start, review_frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        minimum, maximum = evaluated_mesh_frame_data(mesh)
        minima.append(minimum)
        maxima.append(maximum)
        pelvis_positions[frame] = joint_head_world(target, "Bip01 Pelvis")
    minimum = Vector(
        (
            min(value.x for value in minima),
            min(value.y for value in minima),
            min(value.z for value in minima),
        )
    )
    maximum = Vector(
        (
            max(value.x for value in maxima),
            max(value.y for value in maxima),
            max(value.z for value in maxima),
        )
    )
    return minimum, maximum, pelvis_positions


def simple_material(name, color, roughness=0.7):
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = roughness
    return material


def emission_material(name, color):
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 2.5
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def look_at(obj, target_point):
    obj.rotation_euler = (target_point - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, energy, size, target_point):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    look_at(light, target_point)
    return light


def add_lighting(minimum, maximum):
    center = (minimum + maximum)/2.0
    height = max(2.0, maximum.z - minimum.z)
    add_area_light(
        "review_key_light",
        center + Vector((-2.5, -2.0, height)),
        900.0,
        4.0,
        center,
    )
    add_area_light(
        "review_fill_light",
        center + Vector((2.5, -0.5, height/2.0)),
        600.0,
        3.0,
        center,
    )
    add_area_light(
        "review_rim_light",
        center + Vector((0.0, 2.0, height)),
        800.0,
        3.0,
        center,
    )


def add_fixed_floor(minimum, maximum, review_floor_z):
    center = (minimum + maximum)/2.0
    size_x = max(4.0, maximum.x - minimum.x + 3.0)
    size_y = max(5.0, maximum.y - minimum.y + 3.0)
    bpy.ops.mesh.primitive_plane_add(
        size=2.0, location=(center.x, center.y, review_floor_z)
    )
    floor = bpy.context.object
    floor.name = "rocketbox_motion_review_fixed_floor"
    floor.scale = (size_x/2.0, size_y/2.0, 1.0)
    floor.data.materials.append(
        simple_material("rocketbox_motion_review_floor_material", (0.19, 0.22, 0.24, 1.0))
    )
    return floor


def add_polyline(name, points, material, bevel_depth):
    data = bpy.data.curves.new(name=name, type="CURVE")
    data.dimensions = "3D"
    data.resolution_u = 1
    data.bevel_depth = bevel_depth
    data.bevel_resolution = 2
    spline = data.splines.new(type="POLY")
    spline.points.add(len(points) - 1)
    for point, value in zip(spline.points, points):
        point.co = (*value, 1.0)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def add_root_path(pelvis_positions, review_floor_z):
    material = emission_material(
        "rocketbox_motion_review_path_material", (0.1, 0.8, 1.0, 1.0)
    )
    points = [
        Vector((value.x, value.y, review_floor_z + 0.018))
        for _, value in sorted(pelvis_positions.items())
    ]
    return [ add_polyline("rocketbox_motion_review_root_path", points, material, 0.012) ]


def add_front_arrow(minimum, maximum, review_floor_z):
    material = emission_material(
        "rocketbox_motion_review_front_material", (1.0, 0.02, 0.02, 1.0)
    )
    x = maximum.x + 0.35
    y = maximum.y - 0.15
    z = review_floor_z + 0.022
    start = Vector((x, y, z))
    end = Vector((x, y - 0.7, z))
    left = end + Vector((-0.13, 0.18, 0.0))
    right = end + Vector((0.13, 0.18, 0.0))
    objects = [
        add_polyline("rocketbox_motion_review_front_arrow", (start, end), material, 0.018),
        add_polyline("rocketbox_motion_review_front_arrow_left", (left, end), material, 0.018),
        add_polyline("rocketbox_motion_review_front_arrow_right", (right, end), material, 0.018),
    ]
    data = bpy.data.curves.new("rocketbox_motion_review_front_text", type="FONT")
    data.body = "FRONT -Y"
    data.align_x = "CENTER"
    data.size = 0.14
    text = bpy.data.objects.new("rocketbox_motion_review_front_text", data)
    bpy.context.collection.objects.link(text)
    text.location = start + Vector((0.0, 0.15, 0.01))
    text.data.materials.append(material)
    objects.append(text)
    return objects


def make_segment(name, material):
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=1.0, depth=1.0)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.data.materials.append(material)
    return obj


def make_marker(name, material):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def set_segment_transform(obj, head, tail, radius):
    delta = tail - head
    if delta.length < 1.0e-6:
        delta = Vector((0.0, 0.0, 1.0e-6))
    obj.location = (head + tail)/2.0
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(
        delta.normalized()
    )
    obj.scale = (radius, radius, delta.length)


def keyframe_object_transform(obj, frame):
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    obj.keyframe_insert(data_path="scale", frame=frame)


def linearize_object_action(obj):
    if obj.animation_data is None or obj.animation_data.action is None:
        return
    for curve in obj.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def bake_target_guides(target, frame_start, review_frame_end):
    joint_material = emission_material(
        "rocketbox_motion_review_joint_material", (1.0, 0.74, 0.06, 1.0)
    )
    left_material = emission_material(
        "rocketbox_motion_review_left_foot_material", (0.02, 0.9, 0.75, 1.0)
    )
    right_material = emission_material(
        "rocketbox_motion_review_right_foot_material", (1.0, 0.1, 0.55, 1.0)
    )
    segments = {
        link: make_segment(f"target_joint_{link[0]}_{link[1]}", joint_material)
        for link in STICK_LINKS
    }
    markers = {}
    for name in FOOT_BONES:
        material = left_material if " L " in name else right_material
        markers[name] = make_marker(f"target_foot_marker_{name}", material)
    for frame in range(frame_start, review_frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for (parent_name, child_name), obj in segments.items():
            set_segment_transform(
                obj,
                joint_head_world(target, parent_name),
                joint_head_world(target, child_name),
                0.011,
            )
            keyframe_object_transform(obj, frame)
        for name, obj in markers.items():
            obj.location = joint_head_world(target, name)
            obj.scale = (0.032, 0.032, 0.032)
            keyframe_object_transform(obj, frame)
    for obj in tuple(segments.values()) + tuple(markers.values()):
        linearize_object_action(obj)
    return list(segments.values()), list(markers.values())


def source_frame_for_review_frame(review_frame, frame_start, frame_end):
    cycle_span = frame_end - frame_start
    offset = review_frame - frame_start
    if offset <= cycle_span:
        return frame_start + offset
    else:
        return frame_start + offset - cycle_span


def import_source_motion(source_motion_fbx, frame_start, frame_end):
    before_objects = set(bpy.data.objects)
    print(f"Loading source motion: {source_motion_fbx}")
    result = bpy.ops.import_scene.fbx(filepath=str(source_motion_fbx))
    if "FINISHED" not in result:
        raise RuntimeError("source motion FBX import failed")
    imported = [ obj for obj in bpy.data.objects if obj not in before_objects ]
    armatures = [ obj for obj in imported if obj.type == "ARMATURE" ]
    if len(armatures) != 1:
        raise RuntimeError("source motion FBX must contain exactly one armature")
    source = armatures[0]
    if source.animation_data is None or source.animation_data.action is None:
        raise RuntimeError("source motion armature has no action")
    action_start, action_end = map(round, source.animation_data.action.frame_range)
    if action_start != frame_start or action_end != frame_end:
        raise RuntimeError("source motion action range differs from Task 3 metrics")
    helper_names = sorted(obj.name.split(".")[0] for obj in imported if obj != source)
    if "MotionExtractionHelper" not in helper_names:
        raise RuntimeError("source motion is missing MotionExtractionHelper")
    source_bones = [
        bone
        for bone in source.data.bones
        if "Nub" not in bone.name and bone.name in STICK_BONES
    ]
    if { bone.name for bone in source_bones } != set(STICK_BONES):
        raise RuntimeError("source stick skeleton body map is incomplete")
    for obj in imported:
        obj.hide_render = True
    return source, imported, helper_names


def cache_source_stick(source, frame_start, frame_end):
    cache = {}
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        anchor = joint_head_world(source, "Bip01 Pelvis")
        cache[frame] = {
            name: (joint_head_world(source, name) - anchor)*SOURCE_STICK_SCALE
            for name in STICK_BONES
        }
    return cache


def add_source_stick_skeleton(
    source,
    imported,
    pelvis_positions,
    frame_start,
    frame_end,
    review_frame_end,
):
    cache = cache_source_stick(source, frame_start, frame_end)
    material = emission_material(
        "rocketbox_motion_review_source_stick_material", (0.1, 0.55, 1.0, 1.0)
    )
    segments = {
        link: make_segment(f"source_stick_{link[0]}_{link[1]}", material)
        for link in STICK_LINKS
    }
    for review_frame in range(frame_start, review_frame_end + 1):
        source_frame = source_frame_for_review_frame(
            review_frame, frame_start, frame_end
        )
        anchor = pelvis_positions[review_frame] + Vector(
            (SOURCE_STICK_OFFSET_X_M, 0.0, 0.0)
        )
        for (parent_name, child_name), obj in segments.items():
            relative_head = cache[source_frame][parent_name]
            relative_tail = cache[source_frame][child_name]
            set_segment_transform(
                obj, anchor + relative_head, anchor + relative_tail, 0.009
            )
            keyframe_object_transform(obj, review_frame)
    for obj in segments.values():
        linearize_object_action(obj)
    for obj in imported:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    return list(segments.values())


def make_camera():
    data = bpy.data.cameras.new("rocketbox_motion_review_camera")
    data.type = "ORTHO"
    data.lens = 50.0
    data.dof.use_dof = False
    camera = bpy.data.objects.new("rocketbox_motion_review_camera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def bounds_corners(minimum, maximum):
    return tuple(
        Vector((x, y, z))
        for x in (minimum.x, maximum.x)
        for y in (minimum.y, maximum.y)
        for z in (minimum.z, maximum.z)
    )


def clear_camera_drivers(camera):
    try:
        camera.driver_remove("location")
    except TypeError:
        pass


def set_path_camera(camera, minimum, maximum, direction, margin=1.18):
    clear_camera_drivers(camera)
    center = (minimum + maximum)/2.0
    direction = Vector(direction).normalized()
    distance = max(8.0, (maximum - minimum).length*2.0)
    camera.location = center + direction*distance
    look_at(camera, center)
    bpy.context.view_layer.update()
    inverse = camera.matrix_world.inverted()
    projected = [ inverse @ point for point in bounds_corners(minimum, maximum) ]
    width = max(point.x for point in projected) - min(point.x for point in projected)
    height = max(point.y for point in projected) - min(point.y for point in projected)
    aspect = VIDEO_SIZE[0]/VIDEO_SIZE[1]
    camera.data.ortho_scale = max(height, width/aspect)*margin


def configure_root_follow_camera(
    camera,
    target,
    center_offset,
    direction,
    ortho_scale,
    frame_start,
):
    clear_camera_drivers(camera)
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    root = joint_head_world(target, "Bip01 Pelvis")
    center = root + Vector(center_offset)
    direction = Vector(direction).normalized()
    camera.location = center + direction*6.0
    look_at(camera, center)
    camera.data.ortho_scale = ortho_scale
    offsets = [ camera.location[index] - root[index] for index in range(3) ]
    driver_curves = camera.driver_add("location")
    transform_types = ("LOC_X", "LOC_Y", "LOC_Z")
    for index, curve in enumerate(driver_curves):
        variable = curve.driver.variables.new()
        variable.name = "root"
        variable.type = "TRANSFORMS"
        driver_target = variable.targets[0]
        driver_target.id = target
        driver_target.bone_target = "Bip01 Pelvis"
        driver_target.transform_type = transform_types[index]
        driver_target.transform_space = "WORLD_SPACE"
        curve.driver.expression = f"root + {offsets[index]:.12f}"


def set_view_guides_visibility(view_name, guide_groups):
    for objects in guide_groups.values():
        for obj in objects:
            obj.hide_render = True
    visible_groups = ("floor",)
    visible_groups += {
        "top": ("top",),
        "joints": ("joints",),
        "feet": ("feet",),
        "source_target": ("source_target",),
        "contact_sheet": ("feet",),
    }.get(view_name, ())
    for group_name in visible_groups:
        for obj in guide_groups[group_name]:
            obj.hide_render = False


def configure_view_camera(camera, view_name, target, minimum, maximum, frame_start):
    if VIEW_CAMERA_MODES[view_name] == "path":
        directions = {
            "front": (0.0, -1.0, 0.08),
            "side": (1.0, 0.0, 0.08),
            "top": (0.0, 0.0, 1.0),
        }
        margin = 1.28 if view_name == "top" else 1.20
        view_minimum = minimum.copy()
        view_maximum = maximum.copy()
        if view_name == "top":
            view_maximum.x = maximum.x + TOP_GUIDE_MARGIN_X
            view_maximum.y = maximum.y + TOP_GUIDE_MARGIN_Y
        set_path_camera(
            camera, view_minimum, view_maximum, directions[view_name], margin
        )
        if view_name in ("front", "side"):
            camera.data.ortho_scale = max(
                camera.data.ortho_scale, PATH_BODY_ORTHO_SCALE
            )
    elif view_name == "joints":
        configure_root_follow_camera(
            camera,
            target,
            (0.0, 0.0, 0.15),
            (1.0, -1.0, 0.12),
            JOINTS_ORTHO_SCALE,
            frame_start,
        )
    elif view_name == "feet":
        configure_root_follow_camera(
            camera,
            target,
            (0.0, 0.0, -0.68),
            (1.0, -1.0, 0.08),
            0.78,
            frame_start,
        )
    else:
        configure_root_follow_camera(
            camera,
            target,
            (-0.42, 0.0, 0.12),
            (0.0, -1.0, 0.08),
            SOURCE_TARGET_ORTHO_SCALE,
            frame_start,
        )


def configure_video_output(scene, output_path):
    scene.render.resolution_x, scene.render.resolution_y = VIDEO_SIZE
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.audio_codec = "NONE"


def burn_in_label(
    path,
    label,
    x_expression="32",
    y_expression="24",
    font_size=26,
):
    ffmpeg = shutil.which("ffmpeg")
    font_path = Path(LABEL_FONT_PATH)
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for review labels")
    if not font_path.is_file():
        raise RuntimeError(f"review label font is missing: {font_path}")
    escaped_label = label.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    drawtext = (
        f"drawtext=fontfile={font_path}:text='{escaped_label}':"
        f"fontcolor=0x0dff26:fontsize={font_size}:"
        f"x={x_expression}:y={y_expression}:"
        "box=1:boxcolor=0x11181ecc:boxborderw=7"
    )
    output_path = make_staged_path(path)
    command = [ ffmpeg, "-y", "-v", "error", "-i", str(path), "-vf", drawtext ]
    command.extend(("-frames:v", "1"))
    command.append(str(output_path))
    try:
        run_command(command)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"label burn-in did not produce media: {path}")
        os.replace(output_path, path)
    finally:
        cleanup_paths((output_path,))


def burn_in_contact_sheet_labels(path):
    width, height = CONTACT_TILE_SIZE
    columns, rows = CONTACT_GRID
    for index in range(CONTACT_FRAME_COUNT):
        row = index//columns
        column = index%columns
        label = (
            f"CONTACT {index + 1}/{CONTACT_FRAME_COUNT}   "
            f"{AXIS_LABELS[0]}   {AXIS_LABELS[1]}"
        )
        burn_in_label(
            path,
            label,
            x_expression=str(column*width + 18),
            y_expression=str(row*height + 20),
            font_size=16,
        )


def render_video(
    scene,
    camera,
    view_name,
    output_path,
    target,
    mesh,
    minimum,
    maximum,
    frame_start,
):
    if mesh.hide_render:
        raise RuntimeError("official textured target must remain visible in every MP4")
    configure_view_camera(camera, view_name, target, minimum, maximum, frame_start)
    print(
        f"Configured camera: view={view_name} "
        f"ortho_scale={camera.data.ortho_scale:.6f}"
    )
    configure_video_output(scene, output_path)
    print(f"Writing media: {output_path}")
    bpy.ops.render.render(animation=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"video render did not produce media: {view_name}")


def evenly_spaced_frames(frame_start, frame_end, count):
    return [
        int(round(frame_start + index*(frame_end - frame_start)/(count - 1)))
        for index in range(count)
    ]


def render_contact_sheet(
    scene,
    camera,
    output_path,
    target,
    frame_start,
    frame_end,
):
    configure_root_follow_camera(
        camera,
        target,
        (0.0, 0.0, -0.05),
        (1.0, -1.0, 0.10),
        2.25,
        frame_start,
    )
    width, height = CONTACT_TILE_SIZE
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    still_paths = []
    images = []
    try:
        for index, frame in enumerate(
            evenly_spaced_frames(frame_start, frame_end, CONTACT_FRAME_COUNT)
        ):
            scene.frame_set(frame)
            still_path = make_staged_path(output_path.with_name("contact_tile.png"))
            still_paths.append(still_path)
            scene.render.filepath = str(still_path)
            print(f"Writing still: {still_path}")
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(str(still_path), check_existing=False)
            images.append(image)
        columns, rows = CONTACT_GRID
        sheet = np.zeros((rows*height, columns*width, 4), dtype=np.float32)
        for index, image in enumerate(images):
            pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(
                (height, width, 4)
            )
            row = index//columns
            column = index%columns
            sheet[
                row*height : (row + 1)*height,
                column*width : (column + 1)*width,
                :,
            ] = pixels
        generated = bpy.data.images.new(
            "rocketbox_motion_review_contact_sheet",
            width=columns*width,
            height=rows*height,
            alpha=True,
        )
        generated.pixels.foreach_set(sheet.ravel())
        generated.file_format = "PNG"
        generated.filepath_raw = str(output_path)
        print(f"Writing media: {output_path}")
        generated.save()
        bpy.data.images.remove(generated)
        burn_in_contact_sheet_labels(output_path)
    finally:
        for image in images:
            bpy.data.images.remove(image)
        cleanup_paths(still_paths)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("contact sheet render did not produce media")


def run_command(command):
    print(f"Running command: {shlex.join(command)}")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with status {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


def run_binary_command(command):
    print(f"Running command: {shlex.join(command)}")
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with status {result.returncode}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def extract_rgb_frame(path, width, height, seek_seconds=None):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for pixel validation")
    command = [ ffmpeg, "-v", "error" ]
    if seek_seconds is not None:
        command.extend(("-ss", str(seek_seconds)))
    command.extend(
        (
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        )
    )
    raw = run_binary_command(command)
    expected_size = width*height*3
    if len(raw) != expected_size:
        raise RuntimeError(f"decoded frame size is invalid: {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))


def validate_label_pixels(pixels, description):
    height, width = pixels.shape[:2]
    roi_height = max(1, int(height*0.22))
    roi_width = width
    roi = pixels[:roi_height, :roi_width]
    red = roi[:,:,0].astype(np.int16)
    green = roi[:,:,1].astype(np.int16)
    blue = roi[:,:,2].astype(np.int16)
    mask = (
        (green >= LABEL_GREEN_FLOOR)
        & (red <= LABEL_RED_CEILING)
        & (blue <= LABEL_BLUE_CEILING)
        & (green - red >= LABEL_GREEN_DOMINANCE)
        & (green - blue >= LABEL_GREEN_DOMINANCE)
    )
    coordinates = np.argwhere(mask)
    if len(coordinates) < LABEL_MIN_PIXELS:
        raise RuntimeError(f"camera label is missing: {description}")
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    margin = LABEL_BORDER_MARGIN_PX
    if (
        x_min < margin
        or y_min < margin
        or x_max >= roi_width - margin
        or y_max >= roi_height - margin
    ):
        raise RuntimeError(f"camera label is clipped: {description}")
    return {
        "status": "passed",
        "label_pixel_count": int(len(coordinates)),
        "bounds_xyxy": [ int(x_min), int(y_min), int(x_max), int(y_max) ],
    }


def validate_top_arrow_pixels(pixels):
    height, width = pixels.shape[:2]
    roi = pixels[: int(height*0.45), int(width*0.55) : int(width*0.80)]
    red = roi[:,:,0].astype(np.int16)
    green = roi[:,:,1].astype(np.int16)
    blue = roi[:,:,2].astype(np.int16)
    mask = (
        (red >= ARROW_RED_FLOOR)
        & (red - green >= ARROW_RED_DOMINANCE)
        & (red - blue >= ARROW_RED_DOMINANCE)
    )
    red_pixel_count = int(mask.sum())
    if red_pixel_count < ARROW_MIN_PIXELS:
        raise RuntimeError("top FRONT -Y arrow or text is not visible")
    return {
        "status": "passed",
        "red_pixel_count": red_pixel_count,
    }


def validate_contact_sheet_labels(path):
    width = CONTACT_TILE_SIZE[0]*CONTACT_GRID[0]
    height = CONTACT_TILE_SIZE[1]*CONTACT_GRID[1]
    pixels = extract_rgb_frame(path, width, height)
    columns, rows = CONTACT_GRID
    checks = {}
    for row in range(rows):
        for column in range(columns):
            tile = pixels[
                row*CONTACT_TILE_SIZE[1] : (row + 1)*CONTACT_TILE_SIZE[1],
                column*CONTACT_TILE_SIZE[0] : (column + 1)*CONTACT_TILE_SIZE[0],
            ]
            name = f"row_{row + 1}_column_{column + 1}"
            checks[name] = validate_label_pixels(tile, name)
    return {
        "status": "passed",
        "tiles": checks,
    }


def validate_video(path, expected_frame_count):
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise RuntimeError("ffprobe and ffmpeg are required for media validation")
    probe_output = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    probe = json.loads(probe_output)
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"video has an invalid stream count: {path}")
    stream = streams[0]
    if stream.get("codec_name") != "h264":
        raise RuntimeError(f"video is not H.264: {path}")
    if (stream.get("width"), stream.get("height")) != VIDEO_SIZE:
        raise RuntimeError(f"video resolution is not 1280x720: {path}")
    if Fraction(stream.get("r_frame_rate", "0/1")) != FPS:
        raise RuntimeError(f"video frame rate is not 30 fps: {path}")
    if int(stream.get("nb_frames", 0)) != expected_frame_count:
        raise RuntimeError(f"video frame count is not two gait cycles: {path}")
    signal_output = run_command(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "signalstats,metadata=print:file=-",
            "-frames:v",
            "3",
            "-f",
            "null",
            "-",
        ]
    )
    values = {"YMIN": [], "YMAX": []}
    for name, value in _SIGNAL_RE.findall(signal_output):
        values[name].append(float(value))
    if not values["YMIN"] or not values["YMAX"]:
        raise RuntimeError(f"could not read video signal statistics: {path}")
    luma_range = max(values["YMAX"]) - min(values["YMIN"])
    if luma_range < NONBLANK_LUMA_RANGE:
        raise RuntimeError(f"video is blank or nearly blank: {path}")
    return {
        "status": "passed",
        "sha256": sha256_file(path),
        "codec_name": stream["codec_name"],
        "width": stream["width"],
        "height": stream["height"],
        "r_frame_rate": stream["r_frame_rate"],
        "frame_count": int(stream["nb_frames"]),
        "duration_s": float(stream["duration"]),
        "sampled_luma_range": luma_range,
    }


def validate_png(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        expected_size = (
            CONTACT_TILE_SIZE[0]*CONTACT_GRID[0],
            CONTACT_TILE_SIZE[1]*CONTACT_GRID[1],
        )
        if tuple(image.size) != expected_size:
            raise RuntimeError("contact sheet dimensions are invalid")
        pixels = np.asarray(image.pixels[:], dtype=np.float32)
        pixel_range = float(pixels.max() - pixels.min())
        if pixel_range < NONBLANK_PNG_RANGE:
            raise RuntimeError("contact sheet is blank or nearly blank")
        return {
            "status": "passed",
            "sha256": sha256_file(path),
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "pixel_range": pixel_range,
            "sample_count": CONTACT_FRAME_COUNT,
        }
    finally:
        bpy.data.images.remove(image)


def validate_staged_media(staged_paths, expected_frame_count):
    result = {
        name: validate_video(staged_paths[name], expected_frame_count)
        for name in VIDEO_MEDIA
    }
    result["contact_sheet"] = validate_png(staged_paths["contact_sheet"])
    overlay_pixels = {}
    pixels = extract_rgb_frame(
        staged_paths["top"], VIDEO_SIZE[0], VIDEO_SIZE[1], seek_seconds=1.0
    )
    overlay_pixels["top"] = {
        "front_arrow": validate_top_arrow_pixels(pixels)
    }
    overlay_pixels["contact_sheet"] = validate_contact_sheet_labels(
        staged_paths["contact_sheet"]
    )
    return {
        "status": "passed",
        "media": result,
        "overlay_pixels": overlay_pixels,
    }


def matrix_maximum_error(first, second):
    return max(abs(float(value)) for row in first - second for value in row)


def automatic_checks(task3, loaded_target, floor, media_validation):
    metrics = task3["metrics"]
    root = metrics["root_alignment"]
    loop = metrics["loop_residual"]
    roundtrip = metrics["roundtrip"]
    weights = roundtrip["skin_weight_validation"]
    floor_payload = copy.deepcopy(floor)
    floor_payload.pop("original_bounds")
    feet = {
        "status": "passed",
        "bones": floor_payload.pop("foot_bones"),
        "visible_foot_evidence_authoritative": floor_payload.pop(
            "visible_foot_evidence_authoritative"
        ),
        "authoritative_media": floor_payload.pop("authoritative_media"),
    }
    return {
        "overall": "passed",
        "task3_invariants": {
            "status": "passed",
            "metrics_sha256": task3["metrics_sha256"],
            "metrics_overall": metrics["invariants"]["overall"],
            "space_invariant_max_abs_error": metrics["invariants"][
                "space_invariant_max_abs_error"
            ],
            "space_invariant_tolerance": metrics["invariants"][
                "space_invariant_tolerance"
            ],
        },
        "retarget_glb_current_hash": {
            "status": "passed",
            "sha256": task3["current_glb_sha256"],
        },
        "direction": {
            "status": "passed",
            "endpoint_direction_dot_negative_y": root[
                "endpoint_direction_dot_negative_y"
            ],
            "direction_dot_floor": ROOT_DIRECTION_DOT_FLOOR,
            "target_minimum_facing_travel_dot": root[
                "target_minimum_facing_travel_dot"
            ],
            "facing_forward_dot_floor": root["facing_forward_dot_floor"],
            "maximum_facing_reconstruction_error": root[
                "maximum_facing_reconstruction_error"
            ],
            "facing_reconstruction_tolerance": root[
                "facing_reconstruction_tolerance"
            ],
        },
        "rest_pose_mapping_roundtrip": {
            "status": "passed",
            "rest_angle_maximum_deg": metrics["rest_angle_statistics"][
                "maximum_deg"
            ],
            "rest_angle_limit_deg": REST_ANGLE_LIMIT_DEG,
            "mapped_bone_count": metrics["mapping"]["mapped_bone_count"],
            "core_bone_count": len(CORE_BONES),
            "hierarchy_mismatch_count": len(
                metrics["mapping"]["hierarchy_mismatches"]
            ),
            "root_loop_residual_m": loop[
                "root_residual_after_cycle_displacement_m"
            ],
            "normalized_bone_delta_residual": loop[
                "normalized_bone_delta_residual"
            ],
            "normalized_loop_residual_tolerance": NORMALIZED_LOOP_RESIDUAL_TOLERANCE,
            "maximum_world_joint_error_m": roundtrip[
                "maximum_world_joint_error_m"
            ],
            "joint_tolerance_m": roundtrip["joint_tolerance_m"],
        },
        "material_binding": {
            "status": "passed",
            "material_slot_names": loaded_target["material_slot_names"],
            "official_color_images": loaded_target["official_color_images"],
            "semantic_glb_bindings_passed": metrics["materials"][
                "semantic_glb_bindings"
            ]["passed"],
        },
        "weight_binding": {
            "status": "passed",
            "loaded_maximum_weight_sum_error": loaded_target[
                "maximum_weight_sum_error"
            ],
            "loaded_vertices_without_weights": loaded_target[
                "vertices_without_weights"
            ],
            "roundtrip_maximum_weight_sum_error": weights[
                "maximum_weight_sum_error"
            ],
            "roundtrip_maximum_weight_l1_error": weights[
                "maximum_weight_l1_error"
            ],
        },
        "fixed_floor": floor_payload,
        "foot_contact_and_slide": feet,
        "media_validation": media_validation,
    }


def main(argv=None):
    args = parse_args(argv)
    manifest = load_stage_manifest(args)
    manifest = copy.deepcopy(manifest)
    invalidate_review_readiness(args.retarget_dir)
    staged_paths = {}
    try:
        task3 = validate_task3_inputs(args, manifest)
        load_retarget_blend(args)
        frame_start = task3["frame_start"]
        frame_end = task3["frame_end"]
        target, mesh = target_objects()
        scene = configure_scene(frame_start, frame_end)
        scene.frame_set(frame_start)
        bpy.context.view_layer.update()
        target_initial_matrix = target.matrix_world.copy()
        target_initial_matrix.freeze()
        loaded_target = validate_loaded_target(target, mesh, task3["metrics"])
        floor = scan_floor_calibration(
            mesh, target, task3["metrics"], frame_start, frame_end
        )

        action = add_review_cycle_modifiers(target, frame_start, frame_end)
        cycle_span = frame_end - frame_start
        review_frame_end = frame_start + GAIT_CYCLE_COUNT*cycle_span
        scene.frame_end = review_frame_end
        minimum, maximum, pelvis_positions = scan_review_motion(
            mesh, target, frame_start, review_frame_end
        )
        floor_objects = [
            add_fixed_floor(minimum, maximum, floor["review_floor_z_m"])
        ]
        add_lighting(minimum, maximum)
        top_guides = add_root_path(
            pelvis_positions, floor["review_floor_z_m"]
        ) + add_front_arrow(minimum, maximum, floor["review_floor_z_m"])
        joint_guides, foot_guides = bake_target_guides(
            target, frame_start, review_frame_end
        )
        source, imported, helper_names = import_source_motion(
            args.source_motion_fbx.resolve(), frame_start, frame_end
        )
        source_guides = add_source_stick_skeleton(
            source,
            imported,
            pelvis_positions,
            frame_start,
            frame_end,
            review_frame_end,
        )
        guide_groups = {
            "floor": floor_objects,
            "top": top_guides,
            "joints": joint_guides,
            "feet": foot_guides,
            "source_target": source_guides,
        }
        camera = make_camera()
        staged_paths = stage_media_paths(args.retarget_dir)
        for view_name in VIDEO_MEDIA:
            set_view_guides_visibility(view_name, guide_groups)
            render_video(
                scene,
                camera,
                view_name,
                staged_paths[view_name],
                target,
                mesh,
                minimum,
                maximum,
                frame_start,
            )
            if (
                view_name == VIDEO_MEDIA[0]
                and os.environ.get("ROCKETBOX_MOTION_REVIEW_FAIL_AFTER_FIRST_MEDIA") == "1"
            ):
                raise RuntimeError("injected Rocketbox motion-review render failure")
        set_view_guides_visibility("contact_sheet", guide_groups)
        render_contact_sheet(
            scene,
            camera,
            staged_paths["contact_sheet"],
            target,
            frame_start,
            frame_end,
        )
        scene.frame_set(frame_start)
        bpy.context.view_layer.update()
        if matrix_maximum_error(target_initial_matrix, target.matrix_world) > 1.0e-7:
            raise RuntimeError("review setup changed target transforms")
        if target.animation_data.action != action:
            raise RuntimeError("review setup changed target action ownership")
        expected_frame_count = review_frame_end - frame_start + 1
        media_validation = validate_staged_media(
            staged_paths, expected_frame_count
        )
        media_validation["source_target"] = {
            "source_motion_fbx_sha256": manifest["immutable_input_hashes"][
                "motion_fbx"
            ],
            "excluded_source_helpers": helper_names,
            "excluded_source_nub_geometry": True,
            "synchronized_original_cycle": True,
        }
        checks = automatic_checks(task3, loaded_target, floor, media_validation)
        publish_staged_media(staged_paths, args.retarget_dir)
        publish_manifest(
            manifest, checks, args.retarget_dir/"retarget_manifest.json"
        )
    finally:
        cleanup_paths(staged_paths.values())
    print(f"ROCKETBOX_MOTION_REVIEW_OK asset_id={args.asset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
