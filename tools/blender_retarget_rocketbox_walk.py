#!/usr/bin/env python3

#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Bake a Rocketbox neutral walk onto an approved textured avatar.

Run with Blender 4.2 or newer:

    blender --background --python tools/blender_retarget_rocketbox_walk.py -- \
      --asset-id rocketbox_male_adult_01 --avatar-fbx /absolute/avatar.fbx \
      --texture-dir /absolute/Textures --texture-prefix m002 \
      --motion-fbx /absolute/m_walk_neutral.max.fbx \
      --source-review-json /absolute/source_review.json \
      --output-dir /absolute/output
"""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from statistics import mean

import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.kdtree import KDTree


TOOLS_DIR = Path(__file__).resolve().parent
SPIKE_RLR_DIR = TOOLS_DIR/"spike_rlr"
for import_dir in (TOOLS_DIR, SPIKE_RLR_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from blender_render_rocketbox_source_review import (
    ImportedAvatar,
    import_avatar,
    material_uses_color_as_alpha,
    reconnect_official_materials,
)
from rocketbox_human_review import assert_source_review_approved


TARGET_BONES = (
    "Bip01 Pelvis",
    "Bip01 Spine",
    "Bip01 Spine1",
    "Bip01 Spine2",
    "Bip01 Neck",
    "Bip01 Head",
    "Bip01 REye",
    "Bip01 LEye",
    "Bip01 MJaw",
    "Bip01 MBottomLip",
    "Bip01 MTongue",
    "Bip01 LMouthBottom",
    "Bip01 RMouthBottom",
    "Bip01 RMasseter",
    "Bip01 LMasseter",
    "Bip01 MUpperLip",
    "Bip01 RCaninus",
    "Bip01 LCaninus",
    "Bip01 REyeBlinkBottom",
    "Bip01 LEyeBlinkBottom",
    "Bip01 RUpperlip",
    "Bip01 LUpperlip",
    "Bip01 RMouthCorner",
    "Bip01 LMouthCorner",
    "Bip01 RCheek",
    "Bip01 LCheek",
    "Bip01 REyeBlinkTop",
    "Bip01 LEyeBlinkTop",
    "Bip01 RInnerEyebrow",
    "Bip01 LInnerEyebrow",
    "Bip01 MMiddleEyebrow",
    "Bip01 ROuterEyebrow",
    "Bip01 LOuterEyebrow",
    "Bip01 MNose",
    "Bip01 L Clavicle",
    "Bip01 L UpperArm",
    "Bip01 L Forearm",
    "Bip01 L Hand",
    "Bip01 L Finger0",
    "Bip01 L Finger01",
    "Bip01 L Finger02",
    "Bip01 L Finger1",
    "Bip01 L Finger11",
    "Bip01 L Finger12",
    "Bip01 L Finger2",
    "Bip01 L Finger21",
    "Bip01 L Finger22",
    "Bip01 L Finger3",
    "Bip01 L Finger31",
    "Bip01 L Finger32",
    "Bip01 L Finger4",
    "Bip01 L Finger41",
    "Bip01 L Finger42",
    "Bip01 R Clavicle",
    "Bip01 R UpperArm",
    "Bip01 R Forearm",
    "Bip01 R Hand",
    "Bip01 R Finger0",
    "Bip01 R Finger01",
    "Bip01 R Finger02",
    "Bip01 R Finger1",
    "Bip01 R Finger11",
    "Bip01 R Finger12",
    "Bip01 R Finger2",
    "Bip01 R Finger21",
    "Bip01 R Finger22",
    "Bip01 R Finger3",
    "Bip01 R Finger31",
    "Bip01 R Finger32",
    "Bip01 R Finger4",
    "Bip01 R Finger41",
    "Bip01 R Finger42",
    "Bip01 L Thigh",
    "Bip01 L Calf",
    "Bip01 L Foot",
    "Bip01 L Toe0",
    "Bip01 R Thigh",
    "Bip01 R Calf",
    "Bip01 R Foot",
    "Bip01 R Toe0",
)

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
ABSOLUTE_POSE_BONES = CORE_BONES

IMMUTABLE_HASH_KEYS = (
    "avatar_fbx",
    "motion_fbx",
    "source_review",
    "body_color_texture",
    "head_color_texture",
    "opacity_color_texture",
    "retarget_glb",
)

CONSUMED_TEXTURE_SUFFIXES = (
    "body_color",
    "body_normal",
    "body_specular",
    "head_color",
    "head_normal",
    "head_specular",
    "opacity_color",
)
READINESS_FILES = ("retarget_manifest.json", "motion_review.json")

FOOT_BONES = (
    "Bip01 L Foot",
    "Bip01 L Toe0",
    "Bip01 R Foot",
    "Bip01 R Toe0",
)
AXIS_MAP = Matrix.Identity(3)
ROOT_SCALE = 1.0
FPS = 30
EXPECTED_SOURCE_ONLY_NUB_BONES = 41
MATRIX_TOLERANCE = 1.0e-4
ABSOLUTE_POSE_ROTATION_TOLERANCE_RAD = 1.0e-5
ROUNDTRIP_JOINT_TOLERANCE_M = 1.0e-4
SKIN_POSITION_TOLERANCE_M = 2.0e-6
SKIN_WEIGHT_L1_TOLERANCE = 1.0e-6
SKIN_WEIGHT_SUM_TOLERANCE = 1.0e-6
FACING_RECONSTRUCTION_TOLERANCE = 1.0e-5
# Allows measured gait sway while rejecting sideways or backward reconstruction.
FACING_FORWARD_DOT_FLOOR = 0.90
_BONE_PATH_RE = re.compile(r'^pose\.bones\["(.+)"\]')
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

assert len(TARGET_BONES) == 80
assert len(set(TARGET_BONES)) == 80


class SourceImport:
    def __init__(self, armature, action, helper, imported_objects, imported_actions):
        self.armature = armature
        self.action = action
        self.helper = helper
        self.imported_objects = imported_objects
        self.imported_actions = imported_actions


class CachedFrame:
    def __init__(
        self,
        frame,
        source_location,
        source_quaternion,
        bone_pose_rotations,
        pelvis_translation_delta,
    ):
        self.frame = frame
        self.source_location = source_location
        self.source_quaternion = source_quaternion
        self.bone_pose_rotations = bone_pose_rotations
        self.pelvis_translation_delta = pelvis_translation_delta


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--avatar-fbx", type=Path, required=True)
    parser.add_argument("--texture-dir", type=Path, required=True)
    parser.add_argument("--texture-prefix", required=True)
    parser.add_argument("--motion-fbx", type=Path, required=True)
    parser.add_argument("--source-review-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing JSON: {path}")
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
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def configure_animation_scene():
    scene = bpy.context.scene
    fps = 30
    if fps != FPS:
        raise RuntimeError("Rocketbox FPS constants disagree")
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    scene.sync_mode = "NONE"


def vector_list(value):
    return [ float(component) for component in value ]


def matrix_max_abs(matrix):
    return max(abs(float(value)) for row in matrix for value in row)


def matrix_difference_max_abs(actual, expected):
    return matrix_max_abs(actual - expected)


def shortest_rotation_angle(first, second):
    angle = first.normalized().rotation_difference(second.normalized()).angle
    return min(angle, 2.0*math.pi - angle)


def parent_local_rest(bone):
    if bone.parent is None:
        return bone.matrix_local.copy()
    else:
        return bone.parent.matrix_local.inverted() @ bone.matrix_local


def parent_local_pose(pose_bone):
    if pose_bone.parent is None:
        return pose_bone.matrix.copy()
    else:
        return pose_bone.parent.matrix.inverted() @ pose_bone.matrix


def mesh_data_sha256(mesh):
    digest = hashlib.sha256()
    digest.update(mesh.name.encode("utf-8"))
    for vertex in mesh.data.vertices:
        digest.update(struct.pack("<3d", *map(float, vertex.co)))
        for group in sorted(vertex.groups, key=lambda item: item.group):
            digest.update(struct.pack("<Id", int(group.group), float(group.weight)))
    for polygon in mesh.data.polygons:
        digest.update(struct.pack("<II", len(polygon.vertices), polygon.material_index))
        digest.update(struct.pack(f"<{len(polygon.vertices)}I", *polygon.vertices))
    for uv_layer in mesh.data.uv_layers:
        digest.update(uv_layer.name.encode("utf-8"))
        for uv_loop in uv_layer.data:
            digest.update(struct.pack("<2d", *map(float, uv_loop.uv)))
    for slot in mesh.material_slots:
        digest.update((slot.material.name if slot.material else "").encode("utf-8"))
    for group in mesh.vertex_groups:
        digest.update(group.name.encode("utf-8"))
    return digest.hexdigest()


def mesh_metrics(mesh, armature):
    return {
        "mesh_name": mesh.name,
        "vertex_count": len(mesh.data.vertices),
        "polygon_count": len(mesh.data.polygons),
        "uv_layer_count": len(mesh.data.uv_layers),
        "material_slot_count": len(mesh.material_slots),
        "material_slot_names": [
            slot.material.name if slot.material else None for slot in mesh.material_slots
        ],
        "vertex_group_count": len(mesh.vertex_groups),
        "vertex_group_names": [ group.name for group in mesh.vertex_groups ],
        "bone_count": len(armature.data.bones),
        "bind_mesh_sha256": mesh_data_sha256(mesh),
    }


def reviewed_floor_z(mesh):
    return min(float((mesh.matrix_world @ vertex.co).z) for vertex in mesh.data.vertices)


def validate_source_review(args):
    review = assert_source_review_approved(args.source_review_json)
    if review["asset_id"] != args.asset_id:
        raise RuntimeError(
            f"source review asset_id {review['asset_id']!r} does not match {args.asset_id!r}"
        )
    avatar_sha256 = sha256_file(args.avatar_fbx)
    if avatar_sha256 != review["source_sha256"]:
        raise RuntimeError("avatar FBX does not match the approved source review")
    if not args.motion_fbx.is_file():
        raise FileNotFoundError(f"Rocketbox motion FBX does not exist: {args.motion_fbx}")

    approved_by_basename = {
        Path(record["local_path"]).name: record["sha256"]
        for record in review["official_files"]
        if record.get("role") == "texture"
    }
    for suffix in CONSUMED_TEXTURE_SUFFIXES:
        name = f"{args.texture_prefix}_{suffix}.tga"
        texture_path = args.texture_dir/name
        if not texture_path.is_file():
            raise FileNotFoundError(f"required official texture is missing: {texture_path}")
        if approved_by_basename.get(name) != sha256_file(texture_path):
            raise RuntimeError(f"approved texture hash mismatch: {texture_path}")
    return review


def invalidate_review_readiness(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in READINESS_FILES:
        path = output_dir/filename
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def remove_avatar_helpers(avatar):
    removed = []
    for obj in avatar.imported_objects:
        if obj not in {avatar.armature, avatar.mesh}:
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    return sorted(removed)


def validate_target_binding(avatar):
    modifiers = [
        modifier
        for modifier in avatar.mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object == avatar.armature
    ]
    if len(modifiers) != 1:
        raise RuntimeError("target mesh must have exactly one modifier bound to target armature")
    actual_bones = tuple(bone.name for bone in avatar.armature.data.bones)
    if set(actual_bones) != set(TARGET_BONES):
        missing = sorted(set(TARGET_BONES) - set(actual_bones))
        extra = sorted(set(actual_bones) - set(TARGET_BONES))
        raise RuntimeError(f"target bone contract mismatch: missing={missing} extra={extra}")
    missing_core = sorted(set(CORE_BONES) - set(actual_bones))
    if missing_core:
        raise RuntimeError(f"target is missing required body bones: {missing_core}")


def validate_official_material_bindings(avatar, provenance):
    materials = {
        slot.material.name: slot.material
        for slot in avatar.mesh.material_slots
        if slot.material is not None
    }
    validated = {}
    for material_name, texture_records in provenance.items():
        material = materials.get(material_name)
        if material is None or not material.use_nodes or material.node_tree is None:
            raise RuntimeError(f"official material graph is missing: {material_name}")
        expected_paths = {
            Path(value).resolve()
            for role, value in texture_records.items()
            if role in {"color", "normal", "specular"}
        }
        actual_paths = {
            Path(bpy.path.abspath(node.image.filepath)).resolve()
            for node in material.node_tree.nodes
            if node.type == "TEX_IMAGE" and node.image is not None
        }
        missing_paths = sorted(str(path) for path in expected_paths - actual_paths)
        if missing_paths:
            raise RuntimeError(
                f"official material graph missing images for {material_name}: {missing_paths}"
            )
        validated[material_name] = sorted(str(path) for path in actual_paths)

    opacity_name = next(
        (name for name in provenance if name.endswith("_opacity")), None
    )
    if opacity_name is None or not material_uses_color_as_alpha(materials[opacity_name]):
        raise RuntimeError("official opacity material must use color luminance as alpha")
    official_color_image_names = sorted(
        Path(records["color"]).stem for records in provenance.values()
    )
    return {
        "material_images": validated,
        "official_color_image_names": official_color_image_names,
        "opacity_uses_color_luminance": True,
    }


def import_source_motion(path):
    if not path.is_file():
        raise FileNotFoundError(f"Rocketbox motion FBX does not exist: {path}")
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=str(path))
    imported_objects = tuple(obj for obj in bpy.data.objects if obj not in before_objects)
    imported_actions = tuple(action for action in bpy.data.actions if action not in before_actions)
    armatures = [ obj for obj in imported_objects if obj.type == "ARMATURE" ]
    if len(armatures) != 1:
        raise RuntimeError(f"motion FBX must contain one armature, found {len(armatures)}")
    armature = armatures[0]
    if armature.animation_data is None or armature.animation_data.action is None:
        raise RuntimeError("motion armature has no active action")
    action = armature.animation_data.action
    helpers = [
        obj
        for obj in imported_objects
        if obj.type == "EMPTY" and obj.name.split(".")[0] == "MotionExtractionHelper"
    ]
    if len(helpers) != 1:
        raise RuntimeError(
            f"motion FBX must contain one MotionExtractionHelper, found {len(helpers)}"
        )
    return SourceImport(
        armature=armature,
        action=action,
        helper=helpers[0],
        imported_objects=imported_objects,
        imported_actions=imported_actions,
    )


def action_driven_bones(action):
    driven = set()
    for curve in action.fcurves:
        match = _BONE_PATH_RE.match(curve.data_path)
        if match:
            driven.add(match.group(1))
    return driven


def validate_mapping(source_armature, target_armature):
    source_names = {bone.name for bone in source_armature.data.bones}
    target_names = {bone.name for bone in target_armature.data.bones}
    missing_source = sorted(set(TARGET_BONES) - source_names)
    missing_target = sorted(set(TARGET_BONES) - target_names)
    if missing_source or missing_target:
        raise RuntimeError(
            f"required Rocketbox mapping is incomplete: source={missing_source} target={missing_target}"
        )
    source_only_bones = sorted(source_names - set(TARGET_BONES))
    if len(source_only_bones) != EXPECTED_SOURCE_ONLY_NUB_BONES or any(
        "Nub" not in name for name in source_only_bones
    ):
        raise RuntimeError(
            "source-only bones must be exactly the 41 ignored Nub bones: "
            f"actual={source_only_bones}"
        )

    hierarchy_mismatches = []
    for name in TARGET_BONES:
        source_parent = source_armature.data.bones[name].parent
        target_parent = target_armature.data.bones[name].parent
        source_parent_name = source_parent.name if source_parent else None
        target_parent_name = target_parent.name if target_parent else None
        if source_parent_name != target_parent_name:
            hierarchy_mismatches.append(
                {
                    "bone": name,
                    "source_parent": source_parent_name,
                    "target_parent": target_parent_name,
                }
            )
    if hierarchy_mismatches:
        raise RuntimeError(f"source/target hierarchy mismatch: {hierarchy_mismatches}")
    unmapped_target_bones = sorted(target_names - set(TARGET_BONES))
    return source_only_bones, unmapped_target_bones, hierarchy_mismatches


def parent_first_names(armature):
    target_index = {name: index for index, name in enumerate(TARGET_BONES)}

    def depth(name):
        value = 0
        parent = armature.data.bones[name].parent
        while parent is not None:
            value += 1
            parent = parent.parent
        return value

    parent_first_bones = sorted(TARGET_BONES, key=lambda name: (depth(name), target_index[name]))
    seen = set()
    for name in parent_first_bones:
        parent = armature.data.bones[name].parent
        if parent is not None and parent.name in TARGET_BONES and parent.name not in seen:
            raise RuntimeError(f"parent-first ordering failed at {name}")
        seen.add(name)
    return parent_first_bones


def integer_frame_range(action):
    start_value, end_value = map(float, action.frame_range)
    start = int(round(start_value))
    end = int(round(end_value))
    if abs(start_value - start) > 1.0e-5 or abs(end_value - end) > 1.0e-5:
        raise RuntimeError(f"source action frame range must be integral: {action.frame_range[:]} ")
    if end <= start:
        raise RuntimeError(f"source action frame range is empty: {start}-{end}")
    return start, end


def cache_source_frames(source, frame_start, frame_end):
    scene = bpy.context.scene
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    helper_basis = source.helper.matrix_world.to_quaternion().normalized()
    source_pelvis_rest_translation = source.armature.data.bones[
        "Bip01 Pelvis"
    ].matrix_local.translation.copy()
    cached_frames = []
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        pose_rotations = {}
        for name in ABSOLUTE_POSE_BONES:
            source_pb = source.armature.pose.bones[name]
            source_pose_rotation = source_pb.matrix.to_quaternion().normalized()
            pose_rotations[name] = source_pose_rotation.copy()
        source_pelvis_translation = source.armature.pose.bones[
            "Bip01 Pelvis"
        ].matrix.translation
        cached_frames.append(
            CachedFrame(
                frame=frame,
                source_location=source.armature.matrix_world.translation.copy(),
                source_quaternion=source.armature.matrix_world.to_quaternion().normalized(),
                bone_pose_rotations=pose_rotations,
                pelvis_translation_delta=(
                    source_pelvis_translation - source_pelvis_rest_translation
                ).copy(),
            )
        )
    return cached_frames, helper_basis


def rest_angle_statistics(source_armature, target_armature):
    values = {}
    for name in TARGET_BONES:
        source_rotation = parent_local_rest(
            source_armature.data.bones[name]
        ).to_quaternion()
        target_rotation = parent_local_rest(
            target_armature.data.bones[name]
        ).to_quaternion()
        values[name] = math.degrees(
            shortest_rotation_angle(source_rotation, target_rotation)
        )
    maximum_name = max(values, key=values.get)
    return {
        "minimum_deg": min(values.values()),
        "mean_deg": mean(values.values()),
        "maximum_deg": values[maximum_name],
        "maximum_bone": maximum_name,
        "per_bone_deg": values,
    }


def remove_source_import(source):
    helper_names = sorted(
        obj.name for obj in source.imported_objects if obj.type != "ARMATURE"
    )
    source_armature_data = source.armature.data
    for obj in source.imported_objects:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    for action in source.imported_actions:
        if action.name in bpy.data.actions:
            bpy.data.actions.remove(action)
    if source_armature_data.name in bpy.data.armatures and source_armature_data.users == 0:
        bpy.data.armatures.remove(source_armature_data)
    return helper_names


def create_target_action(target_armature, asset_id):
    target_armature.animation_data_clear()
    target_armature.animation_data_create()
    target_action = bpy.data.actions.new(name=f"{asset_id}_walk_neutral_retarget")
    target_armature.animation_data.action = target_action
    return target_action


def keyframe_transform(target, frame):
    target.keyframe_insert(data_path="location", frame=frame, group=target.name)
    target.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=target.name)
    target.keyframe_insert(data_path="scale", frame=frame, group=target.name)


def joint_head_world(armature, bone_name):
    return armature.matrix_world @ armature.pose.bones[bone_name].head


def bake_target_action(
    target_armature,
    cached_frames,
    helper_basis,
    parent_first_bones,
    target_action,
):
    target_armature.data.pose_position = "POSE"
    target_armature.rotation_mode = "QUATERNION"
    target_base_location = target_armature.matrix_world.translation.copy()
    target_base_quaternion = target_armature.matrix_world.to_quaternion().normalized()
    target_base_scale = target_armature.scale.copy()
    target_rest_locals = {
        name: parent_local_rest(target_armature.data.bones[name]) for name in TARGET_BONES
    }
    source_frame_one_location = cached_frames[0].source_location.copy()
    sampled_positions = {}
    baked_root_locations = {}
    frame_errors = []
    body_pose_rotation_errors = []

    for cached in cached_frames:
        bpy.context.scene.frame_set(cached.frame)
        for target_pb in target_armature.pose.bones:
            target_pb.rotation_mode = "QUATERNION"
            target_pb.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()

        source_location = cached.source_location
        root_displacement = AXIS_MAP @ (
            (source_location - source_frame_one_location)*ROOT_SCALE
        )
        target_armature.location = target_base_location + root_displacement
        source_quaternion = cached.source_quaternion
        root_motion_quaternion = helper_basis.inverted() @ source_quaternion
        root_motion_quaternion.normalize()
        target_armature.rotation_quaternion = (
            target_base_quaternion @ root_motion_quaternion
        ).normalized()
        target_armature.scale = target_base_scale

        requested_armature_matrices = {}
        for name in parent_first_bones:
            if name not in ABSOLUTE_POSE_BONES:
                continue
            target_pb = target_armature.pose.bones[name]
            target_rest_local = target_rest_locals[name]
            source_pose_rotation = cached.bone_pose_rotations[name]
            if target_pb.parent is None:
                target_translation = target_rest_local.translation.copy()
            else:
                target_translation = (
                    target_pb.parent.matrix @ target_rest_local.translation
                )
            if name == "Bip01 Pelvis":
                target_translation += cached.pelvis_translation_delta
            target_armature_matrix = Matrix.LocRotScale(
                target_translation,
                source_pose_rotation,
                Vector((1.0, 1.0, 1.0)),
            )
            target_pb.matrix = target_armature_matrix
            bpy.context.view_layer.update()
            requested_armature_matrices[name] = target_armature_matrix.copy()

        maximum_error = 0.0
        maximum_body_pose_rotation_error = 0.0
        maximum_body_pose_rotation_error_bone = None
        for name in TARGET_BONES:
            target_pb = target_armature.pose.bones[name]
            if name in ABSOLUTE_POSE_BONES:
                actual_transform = target_pb.matrix
                requested_transform = requested_armature_matrices[name]
            else:
                actual_transform = parent_local_pose(target_pb)
                requested_transform = target_rest_locals[name]
            maximum_error = max(
                maximum_error,
                matrix_difference_max_abs(actual_transform, requested_transform),
            )
            if name in ABSOLUTE_POSE_BONES:
                pose_rotation_error = shortest_rotation_angle(
                    target_pb.matrix.to_quaternion(),
                    cached.bone_pose_rotations[name],
                )
                if pose_rotation_error > maximum_body_pose_rotation_error:
                    maximum_body_pose_rotation_error = pose_rotation_error
                    maximum_body_pose_rotation_error_bone = name
        if maximum_error > MATRIX_TOLERANCE:
            raise RuntimeError(
                f"target pose invariant failed at frame {cached.frame}: {maximum_error}"
            )
        if (
            maximum_body_pose_rotation_error
            > ABSOLUTE_POSE_ROTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                "target body pose does not reconstruct the absolute source pose at "
                f"frame {cached.frame}: {maximum_body_pose_rotation_error} rad at "
                f"{maximum_body_pose_rotation_error_bone}"
            )
        frame_errors.append(maximum_error)
        body_pose_rotation_errors.append(maximum_body_pose_rotation_error)

        keyframe_transform(target_armature, cached.frame)
        for name in TARGET_BONES:
            keyframe_transform(target_armature.pose.bones[name], cached.frame)
        sampled_positions[cached.frame] = {
            name: joint_head_world(target_armature, name).copy() for name in TARGET_BONES
        }
        baked_root_locations[cached.frame] = (
            target_armature.matrix_world.translation.copy()
        )

    for curve in target_action.fcurves:
        for keyframe in curve.keyframe_points:
            keyframe.interpolation = "LINEAR"
    bpy.context.scene.frame_set(cached_frames[0].frame)
    bpy.context.view_layer.update()
    return (
        sampled_positions,
        max(frame_errors),
        max(body_pose_rotation_errors),
        vector_list(target_base_scale),
        baked_root_locations,
    )


def validate_action_ownership(target_armature, target_action):
    if target_armature.animation_data is None:
        raise RuntimeError("target armature has no animation data after bake")
    if target_armature.animation_data.action != target_action:
        raise RuntimeError("new target action is not active on the target armature")
    if not target_action.fcurves:
        raise RuntimeError("new target action has no F-curves")
    unresolved = []
    for curve in target_action.fcurves:
        try:
            target_armature.path_resolve(curve.data_path)
        except ValueError:
            unresolved.append(curve.data_path)
    if unresolved:
        raise RuntimeError(f"target action contains unresolved F-curves: {unresolved}")
    return {
        "action_name": target_action.name,
        "fcurve_count": len(target_action.fcurves),
        "keyframe_count": sum(len(curve.keyframe_points) for curve in target_action.fcurves),
        "unresolved_fcurve_paths": unresolved,
    }


def select_target_only(avatar):
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    avatar.armature.hide_set(False)
    avatar.mesh.hide_set(False)
    avatar.armature.select_set(True)
    avatar.mesh.select_set(True)
    bpy.context.view_layer.objects.active = avatar.armature
    selected = set(bpy.context.selected_objects)
    if selected != {avatar.armature, avatar.mesh}:
        raise RuntimeError(f"target-only selection failed: {[ obj.name for obj in selected ]}")


def only_target_objects_remain(avatar):
    return set(bpy.context.scene.objects) == {avatar.armature, avatar.mesh}


def save_target_blend(avatar, path):
    select_target_only(avatar)
    bpy.context.preferences.filepaths.save_version = 0
    for backup in path.parent.glob(f"{path.name}[0-9]*"):
        if backup.is_file():
            backup.unlink()
    print(f"Saving target blend: {path}")
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path))
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"target-only blend save failed: result={result} path={path}")


def export_target_glb(avatar, path):
    select_target_only(avatar)
    print(f"Exporting target GLB: {path}")
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="ACTIVE_ACTIONS",
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"GLB export failed: result={result} path={path}")


def read_glb_json(path):
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise RuntimeError(f"not a GLB file: {path}")
    version, declared_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or declared_length != len(raw):
        raise RuntimeError(
            f"invalid GLB header: version={version} length={declared_length}/{len(raw)}"
        )
    offset = 12
    while offset + 8 <= len(raw):
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            return json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
    raise RuntimeError("GLB has no JSON chunk")


def gltf_texture_image_name(payload, texture_info):
    if not isinstance(texture_info, dict) or not isinstance(texture_info.get("index"), int):
        return None
    texture_index = texture_info["index"]
    textures = payload.get("textures", [])
    if texture_index < 0 or texture_index >= len(textures):
        return None
    texture = textures[texture_index]
    image_index = texture.get("source")
    if image_index is None:
        image_index = texture.get("extensions", {}).get("KHR_texture_basisu", {}).get("source")
    images = payload.get("images", [])
    if not isinstance(image_index, int) or image_index < 0 or image_index >= len(images):
        return None
    return images[image_index].get("name")


def inspect_semantic_material_bindings(payload, texture_prefix):
    material_binding_errors = []
    checks = {}
    materials = {
        material.get("name"): material for material in payload.get("materials", [])
    }
    for suffix in ("body", "head"):
        material_name = f"{texture_prefix}_{suffix}"
        material = materials.get(material_name, {})
        actual = {
            "base_color": gltf_texture_image_name(
                payload,
                material.get("pbrMetallicRoughness", {}).get("baseColorTexture"),
            ),
            "normal": gltf_texture_image_name(payload, material.get("normalTexture")),
            "specular": gltf_texture_image_name(
                payload,
                material.get("extensions", {})
                .get("KHR_materials_specular", {})
                .get("specularTexture"),
            ),
        }
        expected = {
            "base_color": f"{texture_prefix}_{suffix}_color",
            "normal": f"{texture_prefix}_{suffix}_normal",
            "specular": f"{texture_prefix}_{suffix}_specular",
        }
        checks[material_name] = {"actual": actual, "expected": expected}
        for role in expected:
            if actual[role] != expected[role]:
                material_binding_errors.append(
                    f"{material_name}.{role}: actual={actual[role]!r} expected={expected[role]!r}"
                )

    opacity_name = f"{texture_prefix}_opacity"
    opacity = materials.get(opacity_name, {})
    actual_opacity = {
        "base_color": gltf_texture_image_name(
            payload,
            opacity.get("pbrMetallicRoughness", {}).get("baseColorTexture"),
        ),
        "alpha_mode": opacity.get("alphaMode"),
    }
    expected_opacity = {
        "base_color": f"{texture_prefix}_opacity_color",
        "alpha_mode": "BLEND",
    }
    checks[opacity_name] = {"actual": actual_opacity, "expected": expected_opacity}
    for role in expected_opacity:
        if actual_opacity[role] != expected_opacity[role]:
            material_binding_errors.append(
                f"{opacity_name}.{role}: actual={actual_opacity[role]!r} "
                f"expected={expected_opacity[role]!r}"
            )
    return {
        "passed": not material_binding_errors,
        "errors": material_binding_errors,
        "materials": checks,
    }


def inspect_glb_structure(path, texture_prefix):
    payload = read_glb_json(path)
    nodes = payload.get("nodes", [])
    skins = payload.get("skins", [])
    animations = payload.get("animations", [])
    image_names = { image.get("name", "") for image in payload.get("images", []) }
    official_color_image_names = {
        f"{texture_prefix}_body_color",
        f"{texture_prefix}_head_color",
        f"{texture_prefix}_opacity_color",
    }
    names = [ node.get("name", "") for node in nodes ]
    helper_or_nub_names = sorted(
        name
        for name in names
        if "Nub" in name
        or name.startswith("MotionExtractionHelper")
        or name.startswith("ExposeTransformHelper")
        or name.startswith("Bip01 Footsteps")
    )
    if len(payload.get("meshes", [])) != 1:
        raise RuntimeError("target GLB must contain exactly one mesh")
    if len(skins) != 1 or len(skins[0].get("joints", [])) != len(TARGET_BONES):
        raise RuntimeError("target GLB must contain one 80-joint skin")
    joint_names = [ names[index] for index in skins[0]["joints"] ]
    if set(joint_names) != set(TARGET_BONES):
        raise RuntimeError("target GLB skin joint names do not match target contract")
    if helper_or_nub_names:
        raise RuntimeError(f"target GLB leaked source helpers/Nub bones: {helper_or_nub_names}")
    missing_color_images = sorted(official_color_image_names - image_names)
    if len(animations) != 1 or not animations[0].get("channels"):
        raise RuntimeError("target GLB must contain one non-empty animation")
    primitive_count = sum(len(mesh.get("primitives", [])) for mesh in payload["meshes"])
    semantic_material_bindings = inspect_semantic_material_bindings(
        payload, texture_prefix
    )
    semantic_material_bindings["missing_color_images"] = missing_color_images
    if missing_color_images:
        semantic_material_bindings["passed"] = False
        semantic_material_bindings["errors"].append(
            f"missing official color images: {missing_color_images}"
        )
    return {
        "mesh_count": len(payload["meshes"]),
        "mesh_primitive_count": primitive_count,
        "skin_count": len(skins),
        "skin_joint_count": len(skins[0]["joints"]),
        "animation_count": len(animations),
        "animation_channel_count": len(animations[0]["channels"]),
        "helper_or_nub_names": helper_or_nub_names,
        "official_color_image_names": sorted(official_color_image_names),
        "semantic_material_bindings": semantic_material_bindings,
    }


def sample_frames(frame_start, frame_end):
    return frame_start, (frame_start + frame_end)//2, frame_end


def normalized_vertex_weights(mesh, vertex):
    group_names = [ group.name for group in mesh.vertex_groups ]
    weights = {
        group_names[influence.group]: float(influence.weight)
        for influence in vertex.groups
        if influence.weight > 0.0
    }
    total = sum(weights.values())
    if total <= 0.0:
        return {}, total
    else:
        return { name: weight/total for name, weight in weights.items() }, total


def capture_skin_contract(mesh):
    if tuple(group.name for group in mesh.vertex_groups) != TARGET_BONES:
        raise RuntimeError("target mesh vertex groups do not match the exact 80-bone contract")
    vertices = []
    for vertex in mesh.data.vertices:
        weights, total = normalized_vertex_weights(mesh, vertex)
        if not weights or abs(total - 1.0) > SKIN_WEIGHT_SUM_TOLERANCE:
            raise RuntimeError(
                f"target bind vertex {vertex.index} has invalid influence sum {total}"
            )
        vertices.append(
            {
                "position": (mesh.matrix_world @ vertex.co).copy(),
                "weights": weights,
            }
        )
    return {"group_names": TARGET_BONES, "vertices": vertices}


def validate_roundtrip_skin(mesh, expected_skin):
    errors = []
    actual_group_names = tuple(group.name for group in mesh.vertex_groups)
    if actual_group_names != TARGET_BONES:
        errors.append(
            f"vertex groups differ: actual={actual_group_names} expected={TARGET_BONES}"
        )

    expected_vertices = expected_skin["vertices"]
    position_index = KDTree(len(expected_vertices))
    for index, expected in enumerate(expected_vertices):
        position_index.insert(expected["position"], index)
    position_index.balance()

    vertices_without_influences = []
    mapped_original_indices = set()
    maximum_position_error = 0.0
    maximum_weight_l1_error = 0.0
    maximum_weight_sum_error = 0.0
    for vertex in mesh.data.vertices:
        weights, total = normalized_vertex_weights(mesh, vertex)
        if not weights:
            vertices_without_influences.append(vertex.index)
            continue
        maximum_weight_sum_error = max(maximum_weight_sum_error, abs(total - 1.0))
        position = mesh.matrix_world @ vertex.co
        _, original_index, position_error = position_index.find(position)
        mapped_original_indices.add(original_index)
        maximum_position_error = max(maximum_position_error, float(position_error))
        expected_weights = expected_vertices[original_index]["weights"]
        names = set(weights) | set(expected_weights)
        weight_l1_error = sum(
            abs(weights.get(name, 0.0) - expected_weights.get(name, 0.0))
            for name in names
        )
        maximum_weight_l1_error = max(maximum_weight_l1_error, weight_l1_error)

    mapped_original_vertex_count = len(mapped_original_indices)
    if vertices_without_influences:
        errors.append(
            f"vertices without influences: count={len(vertices_without_influences)}"
        )
    if maximum_weight_sum_error > SKIN_WEIGHT_SUM_TOLERANCE:
        errors.append(f"maximum weight sum error: {maximum_weight_sum_error}")
    if maximum_position_error > SKIN_POSITION_TOLERANCE_M:
        errors.append(f"maximum seam position error: {maximum_position_error}")
    if maximum_weight_l1_error > SKIN_WEIGHT_L1_TOLERANCE:
        errors.append(f"maximum normalized weight L1 error: {maximum_weight_l1_error}")
    unmapped_original_vertex_count = len(expected_vertices) - mapped_original_vertex_count
    return {
        "passed": not errors,
        "errors": errors,
        "vertex_group_count": len(actual_group_names),
        "vertex_group_names": list(actual_group_names),
        "serialized_vertex_count": len(mesh.data.vertices),
        "vertices_without_influences": vertices_without_influences,
        "mapped_original_vertex_count": mapped_original_vertex_count,
        "unmapped_original_vertex_count": unmapped_original_vertex_count,
        "expected_original_vertex_count": len(expected_vertices),
        "maximum_position_error_m": maximum_position_error,
        "position_tolerance_m": SKIN_POSITION_TOLERANCE_M,
        "maximum_weight_sum_error": maximum_weight_sum_error,
        "weight_sum_tolerance": SKIN_WEIGHT_SUM_TOLERANCE,
        "maximum_weight_l1_error": maximum_weight_l1_error,
        "weight_l1_tolerance": SKIN_WEIGHT_L1_TOLERANCE,
    }


def roundtrip_validate(
    glb_path,
    expected_mesh,
    expected_positions,
    expected_skin,
    frame_start,
    frame_end,
):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_animation_scene()
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    scene = bpy.context.scene
    if scene.render.fps != FPS:
        raise RuntimeError(f"roundtrip scene FPS changed: {scene.render.fps}")
    armatures = [ obj for obj in scene.objects if obj.type == "ARMATURE" ]
    skinned_meshes = [
        obj
        for obj in scene.objects
        if obj.type == "MESH"
        and any(modifier.type == "ARMATURE" for modifier in obj.modifiers)
    ]
    if len(armatures) != 1 or len(skinned_meshes) != 1:
        raise RuntimeError(
            "GLB roundtrip must contain one armature and one skinned mesh: "
            f"armatures={len(armatures)} meshes={len(skinned_meshes)}"
        )
    armature = armatures[0]
    mesh = skinned_meshes[0]
    actual_bones = {bone.name for bone in armature.data.bones}
    if not set(CORE_BONES).issubset(actual_bones):
        raise RuntimeError("GLB roundtrip is missing required core bones")
    if actual_bones != set(TARGET_BONES) or any("Nub" in name for name in actual_bones):
        raise RuntimeError("GLB roundtrip bone names do not match the 80-bone target")
    if armature.animation_data is None or armature.animation_data.action is None:
        raise RuntimeError("GLB roundtrip armature has no active action")
    action = armature.animation_data.action
    if not action.fcurves:
        raise RuntimeError("GLB roundtrip action is empty")
    actual_start, actual_end = integer_frame_range(action)
    if (actual_start, actual_end) != (frame_start, frame_end):
        raise RuntimeError(
            "GLB roundtrip frame range changed at 30 fps: "
            f"actual={actual_start}-{actual_end} expected={frame_start}-{frame_end}"
        )

    scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    skin_weight_validation = validate_roundtrip_skin(mesh, expected_skin)

    actual_mesh = mesh_metrics(mesh, armature)
    invariant_count_fields = (
        "polygon_count",
        "uv_layer_count",
        "material_slot_count",
        "bone_count",
    )
    changed_counts = {
        field: {"expected": expected_mesh[field], "actual": actual_mesh[field]}
        for field in invariant_count_fields
        if actual_mesh[field] != expected_mesh[field]
    }
    if actual_mesh["material_slot_names"] != expected_mesh["material_slot_names"]:
        changed_counts["material_slot_names"] = {
            "expected": expected_mesh["material_slot_names"],
            "actual": actual_mesh["material_slot_names"],
        }
    if changed_counts:
        raise RuntimeError(f"GLB roundtrip changed mesh/material counts: {changed_counts}")

    maximum_joint_error = 0.0
    per_frame_error = {}
    for frame in sample_frames(frame_start, frame_end):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        frame_error = 0.0
        for name in TARGET_BONES:
            expected = expected_positions[frame][name]
            actual = joint_head_world(armature, name)
            frame_error = max(frame_error, float((actual - expected).length))
        per_frame_error[str(frame)] = frame_error
        maximum_joint_error = max(maximum_joint_error, frame_error)
    if maximum_joint_error >= ROUNDTRIP_JOINT_TOLERANCE_M:
        raise RuntimeError(
            f"GLB roundtrip joint error is too large: {maximum_joint_error} m"
        )
    return {
        "fps": scene.render.fps,
        "armature_count": len(armatures),
        "skinned_mesh_count": len(skinned_meshes),
        "action_name": action.name,
        "action_fcurve_count": len(action.fcurves),
        "frame_start": actual_start,
        "frame_end": actual_end,
        "core_bones_present": True,
        "bone_count": len(actual_bones),
        "nub_bone_count": sum("Nub" in name for name in actual_bones),
        "mesh_counts": {
            field: actual_mesh[field]
            for field in ("vertex_count", *invariant_count_fields)
        },
        "serialized_vertex_count_change": (
            actual_mesh["vertex_count"] - expected_mesh["vertex_count"]
        ),
        "material_slot_names": actual_mesh["material_slot_names"],
        "sampled_world_joint_error_m": per_frame_error,
        "maximum_world_joint_error_m": maximum_joint_error,
        "joint_tolerance_m": ROUNDTRIP_JOINT_TOLERANCE_M,
        "skin_weight_validation": skin_weight_validation,
        "passed": skin_weight_validation["passed"],
    }


def horizontal_unit(vector):
    result = Vector((vector.x, vector.y, 0.0))
    if result.length == 0.0:
        raise RuntimeError("horizontal vector has zero length")
    result.normalize()
    return result


def root_metrics(
    cached_frames,
    helper_basis,
    target_base_quaternion,
    baked_root_locations,
):
    source_frame_one_location = cached_frames[0].source_location
    source_translations = [
        AXIS_MAP @ ((cached.source_location - source_frame_one_location)*ROOT_SCALE)
        for cached in cached_frames
    ]
    baked_frame_one_location = baked_root_locations[cached_frames[0].frame]
    translations = [
        baked_root_locations[cached.frame] - baked_frame_one_location
        for cached in cached_frames
    ]
    travel = translations[-1] - translations[0]
    expected_travel = source_translations[-1] - source_translations[0]
    travel_reconstruction_error = float((travel - expected_travel).length)
    if travel_reconstruction_error > 1.0e-6:
        raise RuntimeError(
            "baked target root travel differs from measured source travel: "
            f"{travel_reconstruction_error} m"
        )
    travel_unit = horizontal_unit(travel)
    negative_y = Vector((0.0, -1.0, 0.0))
    root_direction_dot = float(travel_unit.dot(negative_y))
    if root_direction_dot < 0.999:
        raise RuntimeError(f"root travel is not aligned to reviewed -Y: {root_direction_dot}")
    source_travel_unit = horizontal_unit(expected_travel)
    source_facing_dots = []
    target_facing_dots = []
    for cached in cached_frames:
        source_facing = horizontal_unit(
            cached.source_quaternion @ Vector((1.0, 0.0, 0.0))
        )
        root_motion = helper_basis.inverted() @ cached.source_quaternion
        target_root_quaternion = target_base_quaternion @ root_motion
        target_facing = horizontal_unit(
            target_root_quaternion @ Vector((1.0, 0.0, 0.0))
        )
        source_facing_dots.append(float(source_facing.dot(source_travel_unit)))
        target_facing_dots.append(float(target_facing.dot(travel_unit)))
    maximum_facing_reconstruction_error = max(
        abs(target_dot - source_dot)
        for source_dot, target_dot in zip(source_facing_dots, target_facing_dots)
    )
    if maximum_facing_reconstruction_error > FACING_RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(
            "target root facing does not reconstruct source facing: "
            f"maximum dot error={maximum_facing_reconstruction_error}"
        )
    if min(target_facing_dots) < FACING_FORWARD_DOT_FLOOR:
        raise RuntimeError(
            "reconstructed root facing is not forward: "
            f"minimum dot={min(target_facing_dots)}"
        )

    velocities = [
        (translations[index + 1] - translations[index])*FPS
        for index in range(len(translations) - 1)
    ]
    boundary_velocity_difference = velocities[-1] - velocities[0]
    return {
        "axis_map": [ list(map(float, row)) for row in AXIS_MAP ],
        "root_scale": ROOT_SCALE,
        "source_frame_one_location_m": vector_list(source_frame_one_location),
        "source_travel_vector_m": vector_list(
            cached_frames[-1].source_location - cached_frames[0].source_location
        ),
        "target_travel_vector_m": vector_list(travel),
        "travel_reconstruction_error_m": travel_reconstruction_error,
        "travel_distance_m": float(travel.length),
        "x_drift_m": float(max(item.x for item in translations) - min(item.x for item in translations)),
        "z_bob_range_m": float(max(item.z for item in translations) - min(item.z for item in translations)),
        "endpoint_direction_dot_negative_y": root_direction_dot,
        "source_minimum_facing_travel_dot": min(source_facing_dots),
        "source_maximum_facing_travel_dot": max(source_facing_dots),
        "target_minimum_facing_travel_dot": min(target_facing_dots),
        "target_maximum_facing_travel_dot": max(target_facing_dots),
        "maximum_facing_reconstruction_error": maximum_facing_reconstruction_error,
        "facing_reconstruction_tolerance": FACING_RECONSTRUCTION_TOLERANCE,
        "facing_forward_dot_floor": FACING_FORWARD_DOT_FLOOR,
        "minimum_facing_travel_dot": min(target_facing_dots),
        "maximum_facing_travel_dot": max(target_facing_dots),
        "start_velocity_m_per_s": vector_list(velocities[0]),
        "end_velocity_m_per_s": vector_list(velocities[-1]),
        "boundary_velocity_difference_m_per_s": vector_list(
            boundary_velocity_difference
        ),
    }


def loop_metrics(cached_frames, baked_root_locations):
    start = cached_frames[0]
    end = cached_frames[-1]
    bone_residuals = {
        name: (
            matrix_difference_max_abs(
                end.bone_pose_rotations[name].to_matrix().to_4x4(),
                start.bone_pose_rotations[name].to_matrix().to_4x4(),
            )
            if name in ABSOLUTE_POSE_BONES
            else 0.0
        )
        for name in TARGET_BONES
    }
    maximum_bone = max(bone_residuals, key=bone_residuals.get)
    start_next = cached_frames[1]
    end_previous = cached_frames[-2]
    boundary_motion_residuals = {}
    for name in TARGET_BONES:
        if name in ABSOLUTE_POSE_BONES:
            start_velocity = (
                start.bone_pose_rotations[name].inverted()
                @ start_next.bone_pose_rotations[name]
            )
            end_velocity = (
                end_previous.bone_pose_rotations[name].inverted()
                @ end.bone_pose_rotations[name]
            )
            boundary_motion_residuals[name] = matrix_difference_max_abs(
                start_velocity.to_matrix().to_4x4(),
                end_velocity.to_matrix().to_4x4(),
            )
        else:
            boundary_motion_residuals[name] = 0.0
    maximum_velocity_bone = max(
        boundary_motion_residuals, key=boundary_motion_residuals.get
    )
    expected_cycle_displacement = AXIS_MAP @ (
        (end.source_location - start.source_location)*ROOT_SCALE
    )
    actual_cycle_displacement = (
        baked_root_locations[end.frame] - baked_root_locations[start.frame]
    )
    root_loop_residual = (
        actual_cycle_displacement - expected_cycle_displacement
    ).length
    return {
        "expected_cycle_displacement_m": vector_list(expected_cycle_displacement),
        "actual_cycle_displacement_m": vector_list(actual_cycle_displacement),
        "root_residual_after_cycle_displacement_m": float(root_loop_residual),
        "maximum_bone_delta_residual": bone_residuals[maximum_bone],
        "maximum_bone_delta_residual_bone": maximum_bone,
        "normalized_bone_delta_residual": bone_residuals[maximum_bone],
        "maximum_boundary_motion_residual": boundary_motion_residuals[
            maximum_velocity_bone
        ],
        "maximum_boundary_motion_residual_bone": maximum_velocity_bone,
        "per_bone_delta_residual": bone_residuals,
    }


def floor_metrics(positions, reviewed_floor):
    frames = sorted(positions)
    result = {}
    for name in FOOT_BONES:
        points = [ positions[frame][name] for frame in frames ]
        minimum_z = min(point.z for point in points)
        contact_limit = minimum_z + 0.02
        contact_steps = []
        for index in range(len(points) - 1):
            if points[index].z <= contact_limit and points[index + 1].z <= contact_limit:
                displacement = Vector(
                    (
                        points[index + 1].x - points[index].x,
                        points[index + 1].y - points[index].y,
                    )
                ).length
                contact_steps.append(float(displacement))
        result[name] = {
            "minimum_world_z_m": float(minimum_z),
            "penetration_below_reviewed_floor_m": max(
                0.0, float(reviewed_floor - minimum_z)
            ),
            "contact_height_threshold_m": float(contact_limit),
            "contact_step_count": len(contact_steps),
            "maximum_contact_xy_velocity_m_per_s": (
                max(contact_steps)*FPS if contact_steps else 0.0
            ),
            "accumulated_contact_slide_m": sum(contact_steps),
        }
    return {
        "reviewed_floor_z_m": reviewed_floor,
        "bones": result,
        "maximum_penetration_m": max(
            item["penetration_below_reviewed_floor_m"] for item in result.values()
        ),
    }


def build_manifest(args, texture_paths, glb_path, frame_start, frame_end):
    immutable_input_hashes = {
        "avatar_fbx": sha256_file(args.avatar_fbx),
        "motion_fbx": sha256_file(args.motion_fbx),
        "source_review": sha256_file(args.source_review_json),
        "body_color_texture": sha256_file(texture_paths["body"]),
        "head_color_texture": sha256_file(texture_paths["head"]),
        "opacity_color_texture": sha256_file(texture_paths["opacity"]),
        "retarget_glb": sha256_file(glb_path),
    }
    if set(immutable_input_hashes) != set(IMMUTABLE_HASH_KEYS) or any(
        _SHA256_RE.fullmatch(value) is None
        for value in immutable_input_hashes.values()
    ):
        raise RuntimeError("immutable input hash contract is invalid")
    return {
        "schema_version": "rocketbox_retarget_manifest_v1",
        "stage": "retargeted",
        "asset_id": args.asset_id,
        "immutable_input_hashes": immutable_input_hashes,
        "binding": {
            "target_asset_id": args.asset_id,
            "target_mesh_bound": True,
            "official_textures_attached": True,
        },
        "source_animation": {
            "fps": FPS,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": frame_end - frame_start + 1,
        },
        "artifacts": {
            "blend": "retarget.blend",
            "glb": "retarget.glb",
            "metrics": "retarget_metrics.json",
        },
        "automatic_checks": {
            "overall": "passed",
            "retarget_bake": {"status": "passed"},
            "glb_roundtrip": {"status": "passed"},
        },
    }


def main(argv=None):
    args = parse_args(argv)
    review = validate_source_review(args)
    invalidate_review_readiness(args.output_dir)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_animation_scene()
    avatar = import_avatar(args.avatar_fbx.resolve(), args.texture_prefix)
    configure_animation_scene()
    validate_target_binding(avatar)
    material_provenance = reconnect_official_materials(
        avatar, args.texture_dir, args.texture_prefix
    )
    official_material_bindings = validate_official_material_bindings(
        avatar, material_provenance
    )
    removed_target_helpers = remove_avatar_helpers(avatar)
    target_base_matrix = avatar.armature.matrix_world.copy()
    target_base_scale = avatar.armature.scale.copy()
    pre_retarget_mesh = mesh_metrics(avatar.mesh, avatar.armature)
    floor_z = reviewed_floor_z(avatar.mesh)

    source = import_source_motion(args.motion_fbx.resolve())
    configure_animation_scene()
    source_only_bones, unmapped_target_bones, hierarchy_mismatches = validate_mapping(
        source.armature, avatar.armature
    )
    driven_bones = action_driven_bones(source.action)
    missing_driven_core = sorted(set(CORE_BONES) - driven_bones)
    if missing_driven_core:
        raise RuntimeError(f"source action does not drive core bones: {missing_driven_core}")
    frame_start, frame_end = integer_frame_range(source.action)
    parent_first_bones = parent_first_names(avatar.armature)
    rest_angles = rest_angle_statistics(source.armature, avatar.armature)
    cached_frames, helper_basis = cache_source_frames(source, frame_start, frame_end)
    excluded_source_helpers = remove_source_import(source)

    avatar.armature.matrix_world = target_base_matrix
    avatar.armature.scale = target_base_scale
    target_action = create_target_action(avatar.armature, args.asset_id)
    scene = bpy.context.scene
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    (
        sampled_positions,
        maximum_space_error,
        maximum_body_pose_rotation_error,
        preserved_target_scale,
        baked_root_locations,
    ) = bake_target_action(
        avatar.armature,
        cached_frames,
        helper_basis,
        parent_first_bones,
        target_action,
    )
    action_metrics = validate_action_ownership(avatar.armature, target_action)
    post_retarget_mesh = mesh_metrics(avatar.mesh, avatar.armature)
    if post_retarget_mesh != pre_retarget_mesh:
        raise RuntimeError("retarget changed target mesh, materials, weights, or bind data")
    if not only_target_objects_remain(avatar):
        raise RuntimeError(
            f"retarget scene is not target-only: {[ obj.name for obj in scene.objects ]}"
        )

    texture_paths = {
        "body": args.texture_dir/f"{args.texture_prefix}_body_color.tga",
        "head": args.texture_dir/f"{args.texture_prefix}_head_color.tga",
        "opacity": args.texture_dir/f"{args.texture_prefix}_opacity_color.tga",
    }
    blend_path = args.output_dir/"retarget.blend"
    glb_path = args.output_dir/"retarget.glb"
    metrics_path = args.output_dir/"retarget_metrics.json"
    manifest_path = args.output_dir/"retarget_manifest.json"

    scene.frame_set(frame_start)
    expected_skin = capture_skin_contract(avatar.mesh)
    save_target_blend(avatar, blend_path)
    export_target_glb(avatar, glb_path)
    if os.environ.get("ROCKETBOX_RETARGET_FAIL_AFTER_EXPORT") == "1":
        raise RuntimeError("injected Rocketbox post-export failure")
    glb_structure = inspect_glb_structure(glb_path, args.texture_prefix)
    roundtrip = roundtrip_validate(
        glb_path,
        pre_retarget_mesh,
        sampled_positions,
        expected_skin,
        frame_start,
        frame_end,
    )

    roots = root_metrics(
        cached_frames,
        helper_basis,
        target_base_matrix.to_quaternion(),
        baked_root_locations,
    )
    loops = loop_metrics(cached_frames, baked_root_locations)
    feet = floor_metrics(sampled_positions, floor_z)
    semantic_material_bindings = glb_structure["semantic_material_bindings"]
    validation_errors = (
        semantic_material_bindings["errors"]
        + roundtrip["skin_weight_validation"]["errors"]
    )
    metrics = {
        "schema_version": "rocketbox_retarget_metrics_v1",
        "asset_id": args.asset_id,
        "blender_version": bpy.app.version_string,
        "source_review_schema": review["schema_version"],
        "source_animation": {
            "fps": FPS,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": frame_end - frame_start + 1,
        },
        "mapping": {
            "mapped_bone_count": len(TARGET_BONES),
            "mapped_bones": list(TARGET_BONES),
            "parent_first_bones": parent_first_bones,
            "unmapped_target_bones": unmapped_target_bones,
            "source_only_bone_count": len(source_only_bones),
            "source_only_bones": source_only_bones,
            "hierarchy_mismatches": hierarchy_mismatches,
            "driven_source_bone_count": len(driven_bones & set(TARGET_BONES)),
            "absolute_pose_bones": list(ABSOLUTE_POSE_BONES),
            "target_rest_pose_bones": sorted(
                set(TARGET_BONES) - set(ABSOLUTE_POSE_BONES)
            ),
            "undriven_target_bones_kept_at_rest": sorted(
                set(TARGET_BONES) - driven_bones
            ),
        },
        "rest_angle_statistics": rest_angles,
        "root_alignment": {
            **roots,
            "helper_basis_quaternion_wxyz": vector_list(helper_basis),
            "target_base_matrix": [ list(map(float, row)) for row in target_base_matrix ],
            "target_scale_preserved": preserved_target_scale,
            "displacement_rotated_by_target_base": False,
            "extra_180_degree_rotation": False,
        },
        "loop_residual": loops,
        "floor_metrics": feet,
        "mesh_invariants": {
            "before_retarget": pre_retarget_mesh,
            "after_retarget": post_retarget_mesh,
            "unchanged": pre_retarget_mesh == post_retarget_mesh,
        },
        "materials": {
            "official_texture_graphs": material_provenance,
            "validated_bindings": official_material_bindings,
            "semantic_glb_bindings": semantic_material_bindings,
            "material_slots_cleared": False,
        },
        "action": action_metrics,
        "glb_structure": glb_structure,
        "roundtrip": roundtrip,
        "excluded": {
            "target_helpers": removed_target_helpers,
            "source_helpers": excluded_source_helpers,
            "source_nub_bones": source_only_bones,
        },
        "invariants": {
            "overall": "passed" if not validation_errors else "failed",
            "errors": validation_errors,
            "space_invariant_max_abs_error": maximum_space_error,
            "space_invariant_tolerance": MATRIX_TOLERANCE,
            "absolute_body_pose_max_rotation_error_rad": (
                maximum_body_pose_rotation_error
            ),
            "absolute_body_pose_rotation_tolerance_rad": (
                ABSOLUTE_POSE_ROTATION_TOLERANCE_RAD
            ),
            "mapped_80_of_80": len(TARGET_BONES) == 80,
            "hierarchy_mismatch_count": len(hierarchy_mismatches),
            "target_mesh_unchanged": pre_retarget_mesh == post_retarget_mesh,
            "target_only_blend": True,
            "target_only_glb": True,
            "official_textures_attached": True,
            "glb_roundtrip_passed": roundtrip["passed"],
            "glb_skin_weights_preserved": roundtrip["skin_weight_validation"][
                "passed"
            ],
            "glb_material_bindings_preserved": semantic_material_bindings["passed"],
        },
        "artifacts": {
            "blend": str(blend_path.resolve()),
            "glb": str(glb_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    write_json(metrics_path, metrics)
    if validation_errors:
        raise RuntimeError(f"retarget validation failed: {validation_errors}")
    write_json(
        manifest_path,
        build_manifest(args, texture_paths, glb_path, frame_start, frame_end),
    )
    print(f"ROCKETBOX_RETARGET_OK asset_id={args.asset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
