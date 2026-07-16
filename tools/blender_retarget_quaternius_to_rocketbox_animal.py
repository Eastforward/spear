#!/usr/bin/env python3
"""Retarget Quaternius Walk/Idle onto a native Rocketbox animal skeleton.

The Rocketbox mesh, topology, UVs, skeleton, and native skin weights remain
unchanged.  Quaternius supplies only sampled bone rotations.  This is the
stable breed-template route for Rocketbox animals that ship without actions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

import bpy
from mathutils import Matrix, Vector


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_robust_swap_mesh_keep_rig as robust  # noqa: E402


SCHEMA = "avengine_quaternius_to_rocketbox_animal_retarget_v1"
CANONICAL_YAW_DEGREES = 90.0
TARGET_AUTHORED_FORWARD = Vector((0.0, -1.0, 0.0))
TARGET_AUTHORED_LATERAL = Vector((1.0, 0.0, 0.0))
BONE_PRIMARY_AXIS = Vector((0.0, 1.0, 0.0))
BONE_MAP = {
    "beagle Pelvis": "Bone",
    "beagle Spine": "Bone.001",
    "beagle Neck": "Bone.002",
    "beagle Head": "Bone.003",
    "beagle Tail": "Bone.004",
    "beagle Tail1": "Bone.005",
    "beagle Tail2": "Bone.006",
    "beagle Tail3": "Bone.007",
    "beagle L Thigh": "Bone.008",
    "beagle L Calf": "Bone.009",
    "beagle L Foot": "Bone.010",
    "beagle R Thigh": "Bone.011",
    "beagle R Calf": "Bone.012",
    "beagle R Foot": "Bone.013",
    "beagle R UpperArm": "Bone.014",
    "beagle R Forearm": "Bone.015",
    "beagle R Hand": "Bone.016",
    "beagle L UpperArm": "Bone.017",
    "beagle L Forearm": "Bone.018",
    "beagle L Hand": "Bone.019",
}
LOCOMOTION_LIMB_BONES = {
    "beagle L Thigh",
    "beagle L Calf",
    "beagle L Foot",
    "beagle R Thigh",
    "beagle R Calf",
    "beagle R Foot",
    "beagle L UpperArm",
    "beagle L Forearm",
    "beagle L Hand",
    "beagle R UpperArm",
    "beagle R Forearm",
    "beagle R Hand",
}
ACTION_HINTS = {"Walking": "walk", "Idle": "idle"}


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rig-glb", type=Path, required=True)
    parser.add_argument("--target-fbx", type=Path, required=True)
    parser.add_argument("--color-texture", type=Path, required=True)
    parser.add_argument("--bump-texture", type=Path, required=True)
    parser.add_argument("--specular-texture", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--license-evidence", type=Path, required=True)
    parser.add_argument(
        "--limb-lateral-damping",
        type=float,
        default=1.0,
        help=(
            "Retain this fraction of animated limb-direction displacement away "
            "from each Rocketbox limb bone's authored sagittal plane. 1 keeps "
            "the source motion unchanged; 0 keeps the authored lateral stance "
            "while preserving forward/up walking motion."
        ),
    )
    return parser.parse_args(argv)


def require_file(path: Path, label: str, suffixes=None) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe {label}: {path}")
    if suffixes and path.suffix.lower() not in suffixes:
        raise SystemExit(f"unexpected {label} format: {path}")
    return path


def require_output(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mesh_skin_sha256(mesh) -> str:
    digest = hashlib.sha256()
    for vertex in mesh.data.vertices:
        digest.update(struct.pack("<3d", *map(float, vertex.co)))
        for group in sorted(vertex.groups, key=lambda value: value.group):
            digest.update(struct.pack("<Id", int(group.group), float(group.weight)))
    for polygon in mesh.data.polygons:
        digest.update(struct.pack("<I", len(polygon.vertices)))
        digest.update(struct.pack(f"<{len(polygon.vertices)}I", *polygon.vertices))
    for layer in mesh.data.uv_layers:
        for item in layer.data:
            digest.update(struct.pack("<2d", *map(float, item.uv)))
    for group in mesh.vertex_groups:
        digest.update(group.name.encode("utf-8"))
    return digest.hexdigest()


def import_source(path: Path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = tuple(item for item in bpy.data.objects if item not in before)
    armatures = [item for item in imported if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError("source must contain exactly one armature")
    armature = armatures[0]
    actions = list(bpy.data.actions)
    selected = {}
    for canonical, hint in ACTION_HINTS.items():
        matches = [action for action in actions if hint in action.name.lower()]
        if len(matches) != 1:
            raise RuntimeError(
                f"source {canonical} action is ambiguous: {[item.name for item in matches]}"
            )
        selected[canonical] = matches[0]
    return armature, imported, selected


def import_target(path: Path):
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    imported = tuple(item for item in bpy.data.objects if item not in before_objects)
    unexpected_actions = [item for item in bpy.data.actions if item not in before_actions]
    if unexpected_actions:
        raise RuntimeError("static Rocketbox animal unexpectedly contains actions")
    armatures = [item for item in imported if item.type == "ARMATURE"]
    meshes = [item for item in imported if item.type == "MESH"]
    if len(armatures) != 1 or not meshes:
        raise RuntimeError("target FBX must contain one armature and a skinned mesh")
    armature = armatures[0]
    mesh = max(meshes, key=lambda item: len(item.data.vertices))
    if not any(modifier.type == "ARMATURE" for modifier in mesh.modifiers):
        raise RuntimeError("Rocketbox animal mesh is not skinned")
    return armature, mesh, imported


def validate_mapping(source, target):
    # Iteration yields Bone objects; explicit keys are the lookup namespace.
    source_names = set(source.data.bones.keys())
    target_names = set(target.data.bones.keys())
    missing_source = sorted(set(BONE_MAP.values()) - source_names)
    missing_target = sorted(set(BONE_MAP) - target_names)
    if missing_source or missing_target:
        raise RuntimeError(
            f"incomplete animal retarget map source={missing_source} target={missing_target}"
        )


def parent_local_rest(bone):
    if bone.parent is None:
        return bone.matrix_local.copy()
    return bone.parent.matrix_local.inverted() @ bone.matrix_local


def mapped_parent_first(target):
    def depth(name):
        value = 0
        bone = target.data.bones[name]
        while bone.parent is not None:
            value += 1
            bone = bone.parent
        return value

    return sorted(BONE_MAP, key=lambda name: (depth(name), name))


def output_action_frame_range(action):
    start_value, end_value = map(float, action.frame_range)
    start, end = int(round(start_value)), int(round(end_value))
    if abs(start - start_value) > 1.0e-3 or abs(end - end_value) > 1.0e-3:
        raise RuntimeError(
            "baked animal action must have integral frame bounds: "
            f"name={action.name!r} range=({start_value!r}, {end_value!r})"
        )
    return start, end


def source_action_sample_frames(action):
    mapped_names = set(BONE_MAP.values())
    prefix = 'pose.bones["'
    frames = set()
    for curve in action.fcurves:
        if not curve.data_path.startswith(prefix):
            continue
        bone_name = curve.data_path[len(prefix) :].split('"', 1)[0]
        if bone_name not in mapped_names:
            continue
        frames.update(float(point.co.x) for point in curve.keyframe_points)
    ordered = sorted(frames)
    if len(ordered) < 2:
        raise RuntimeError(f"source action has too few mapped samples: {action.name!r}")
    return ordered


def set_scene_time(value):
    base = math.floor(float(value))
    bpy.context.scene.frame_set(base, subframe=float(value) - base)


def cache_source_action(source, action):
    source.animation_data_create()
    source.animation_data.action = action
    source_frames = source_action_sample_frames(action)
    frames = []
    for output_frame, source_frame in enumerate(source_frames):
        set_scene_time(source_frame)
        bpy.context.view_layer.update()
        frames.append(
            {
                "frame": output_frame,
                "source_frame": source_frame,
                "rotations": {
                    name: source.pose.bones[name].matrix.to_quaternion().normalized()
                    for name in set(BONE_MAP.values())
                },
            }
        )
    return 0, len(source_frames) - 1, frames


def keyframe_pose_bone(pose_bone, frame):
    pose_bone.keyframe_insert(data_path="location", frame=frame, group=pose_bone.name)
    pose_bone.keyframe_insert(
        data_path="rotation_quaternion", frame=frame, group=pose_bone.name
    )
    pose_bone.keyframe_insert(data_path="scale", frame=frame, group=pose_bone.name)


def damp_limb_lateral_direction(rotation, rest_rotation, damping):
    """Damp only in/out limb swing in the target's authored cardinal frame.

    The correction is applied to the global bone direction, not to arbitrary
    Euler channels.  This keeps the transferred fore/aft and vertical gait
    exactly in the source motion plane while preserving the Rocketbox rest
    stance and native skin.  It is deterministic and cannot swap left/right.
    """
    posed_direction = (rotation @ BONE_PRIMARY_AXIS).normalized()
    rest_direction = (rest_rotation @ BONE_PRIMARY_AXIS).normalized()
    posed_lateral = float(posed_direction.dot(TARGET_AUTHORED_LATERAL))
    rest_lateral = float(rest_direction.dot(TARGET_AUTHORED_LATERAL))
    retained_lateral = rest_lateral + float(damping) * (
        posed_lateral - rest_lateral
    )
    sagittal_direction = (
        posed_direction - TARGET_AUTHORED_LATERAL * posed_lateral
    )
    if sagittal_direction.length <= 1.0e-8:
        raise RuntimeError("lateral gait damping collapsed a limb direction")
    sagittal_direction.normalize()
    retained_lateral = max(-0.999999, min(0.999999, retained_lateral))
    sagittal_length = math.sqrt(max(0.0, 1.0 - retained_lateral**2))
    corrected_direction = (
        sagittal_direction * sagittal_length
        + TARGET_AUTHORED_LATERAL * retained_lateral
    ).normalized()
    correction = posed_direction.rotation_difference(corrected_direction)
    corrected_rotation = (correction @ rotation).normalized()
    corrected_lateral = float(
        (corrected_rotation @ BONE_PRIMARY_AXIS).dot(TARGET_AUTHORED_LATERAL)
    )
    return corrected_rotation, {
        "rest_lateral_component": rest_lateral,
        "before_lateral_component": posed_lateral,
        "after_lateral_component": corrected_lateral,
    }


def bake_action(
    source,
    target,
    source_action,
    canonical_name,
    *,
    limb_lateral_damping,
):
    start, end, cached = cache_source_action(source, source_action)
    action = bpy.data.actions.new(name=canonical_name)
    target.animation_data_create()
    target.animation_data.action = action
    target.data.pose_position = "POSE"
    target_rest_locals = {
        bone.name: parent_local_rest(bone) for bone in target.data.bones
    }
    rest_offsets = {}
    for target_name, source_name in BONE_MAP.items():
        target_rest = target.data.bones[target_name].matrix_local.to_quaternion().normalized()
        source_rest = source.data.bones[source_name].matrix_local.to_quaternion().normalized()
        rest_offsets[target_name] = (target_rest @ source_rest.inverted()).normalized()

    parent_first = mapped_parent_first(target)
    maximum_rotation_error = 0.0
    maximum_limb_lateral_delta_before = 0.0
    maximum_limb_lateral_delta_after = 0.0
    for cached_frame in cached:
        frame = cached_frame["frame"]
        bpy.context.scene.frame_set(frame)
        for pose_bone in target.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()

        requested = {}
        for target_name in parent_first:
            source_name = BONE_MAP[target_name]
            pose_bone = target.pose.bones[target_name]
            rest_local = target_rest_locals[target_name]
            if pose_bone.parent is None:
                translation = rest_local.translation.copy()
            else:
                translation = pose_bone.parent.matrix @ rest_local.translation
            rotation = (
                rest_offsets[target_name] @ cached_frame["rotations"][source_name]
            ).normalized()
            if target_name in LOCOMOTION_LIMB_BONES:
                rest_rotation = (
                    target.data.bones[target_name]
                    .matrix_local.to_quaternion()
                    .normalized()
                )
                rotation, lateral_record = damp_limb_lateral_direction(
                    rotation,
                    rest_rotation,
                    limb_lateral_damping,
                )
                maximum_limb_lateral_delta_before = max(
                    maximum_limb_lateral_delta_before,
                    abs(
                        lateral_record["before_lateral_component"]
                        - lateral_record["rest_lateral_component"]
                    ),
                )
                maximum_limb_lateral_delta_after = max(
                    maximum_limb_lateral_delta_after,
                    abs(
                        lateral_record["after_lateral_component"]
                        - lateral_record["rest_lateral_component"]
                    ),
                )
            desired = Matrix.LocRotScale(translation, rotation, Vector((1.0, 1.0, 1.0)))
            pose_bone.matrix = desired
            bpy.context.view_layer.update()
            requested[target_name] = rotation

        for pose_bone in target.pose.bones:
            keyframe_pose_bone(pose_bone, frame)
        for target_name, rotation in requested.items():
            actual = target.pose.bones[target_name].matrix.to_quaternion().normalized()
            # q and -q encode the same rotation.  Use the absolute dot product
            # so an equivalent quaternion sign cannot appear as a 360° error.
            dot = min(1.0, max(-1.0, abs(float(actual.dot(rotation)))))
            error = 2.0 * math.acos(dot)
            maximum_rotation_error = max(maximum_rotation_error, float(error))

    for curve in action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    action.use_fake_user = True
    return action, {
        "source_action": source_action.name,
        "output_action": canonical_name,
        "source_frame_range": [
            cached[0]["source_frame"],
            cached[-1]["source_frame"],
        ],
        "frame_range": [start, end],
        "sampled_frames": end - start + 1,
        "maximum_requested_rotation_error_degrees": math.degrees(
            maximum_rotation_error
        ),
        "limb_lateral_damping": float(limb_lateral_damping),
        "maximum_abs_limb_lateral_direction_delta_before": (
            maximum_limb_lateral_delta_before
        ),
        "maximum_abs_limb_lateral_direction_delta_after": (
            maximum_limb_lateral_delta_after
        ),
    }


def remove_source(imported, actions):
    for item in imported:
        if item.name in bpy.data.objects:
            bpy.data.objects.remove(item, do_unlink=True)
    for action in actions:
        if action.name in bpy.data.actions:
            bpy.data.actions.remove(action)


def remove_export_extras(target, mesh):
    """Remove source helper meshes/empties before selecting the export pair."""
    removed = []
    for item in list(bpy.data.objects):
        if item not in {target, mesh}:
            removed.append(item.name)
            bpy.data.objects.remove(item, do_unlink=True)
    return sorted(removed)


def load_image(path: Path, colorspace: str):
    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = colorspace
    image.pack()
    return image


def install_pbr(mesh, color_path, bump_path, specular_path):
    material = bpy.data.materials.new(name="Rocketbox_Beagle_PBR")
    material.use_nodes = True
    material.use_backface_culling = False
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    color = nodes.new("ShaderNodeTexImage")
    color.name = "Rocketbox Color"
    color.image = load_image(color_path, "sRGB")
    bump = nodes.new("ShaderNodeTexImage")
    bump.name = "Rocketbox Bump"
    bump.image = load_image(bump_path, "Non-Color")
    bump_node = nodes.new("ShaderNodeBump")
    bump_node.inputs["Strength"].default_value = 0.30
    bump_node.inputs["Distance"].default_value = 0.08
    specular = nodes.new("ShaderNodeTexImage")
    specular.name = "Rocketbox Specular"
    specular.image = load_image(specular_path, "Non-Color")
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Roughness"].default_value = 0.72
    links.new(color.outputs["Color"], shader.inputs["Base Color"])
    links.new(bump.outputs["Color"], bump_node.inputs["Height"])
    links.new(bump_node.outputs["Normal"], shader.inputs["Normal"])
    specular_input = shader.inputs.get("Specular IOR Level")
    if specular_input is not None:
        links.new(specular.outputs["Color"], specular_input)
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    mesh.data.materials.clear()
    mesh.data.materials.append(material)
    for polygon in mesh.data.polygons:
        polygon.material_index = 0
    return {
        "material": material.name,
        "color": str(color_path),
        "bump": str(bump_path),
        "specular": str(specular_path),
    }


def add_nla_tracks(target, actions):
    target.animation_data_create()
    target.animation_data.action = None
    while target.animation_data.nla_tracks:
        target.animation_data.nla_tracks.remove(target.animation_data.nla_tracks[0])
    for action in actions:
        start, end = output_action_frame_range(action)
        track = target.animation_data.nla_tracks.new()
        track.name = action.name
        strip = track.strips.new(action.name, start, action)
        strip.name = action.name
        strip.action_frame_start = start
        strip.action_frame_end = end


def export_target(target, mesh, actions, output):
    add_nla_tracks(target, actions)
    target.rotation_mode = "XYZ"
    target.rotation_euler.z += math.radians(CANONICAL_YAW_DEGREES)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_nla_strips=True,
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_image_format="AUTO",
    )
    robust.postprocess_glb_animation_channels(
        output, {"translation", "rotation"}, canonical_walk_idle=True
    )


def main():
    args = parse_argv()
    source_path = require_file(args.source_rig_glb, "source rig", {".glb", ".gltf"})
    target_path = require_file(args.target_fbx, "target FBX", {".fbx"})
    color_path = require_file(args.color_texture, "color texture")
    bump_path = require_file(args.bump_texture, "bump texture")
    specular_path = require_file(args.specular_texture, "specular texture")
    license_path = require_file(args.license_evidence, "license evidence")
    output_path = require_output(args.output_glb, "output GLB")
    manifest_path = require_output(args.manifest, "manifest")
    if not 0.0 <= args.limb_lateral_damping <= 1.0:
        raise SystemExit("--limb-lateral-damping must be in [0, 1]")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    source, source_objects, source_actions = import_source(source_path)
    target, mesh, target_objects = import_target(target_path)
    validate_mapping(source, target)
    mesh_contract_before = mesh_skin_sha256(mesh)
    action_records = []
    output_actions = []
    for canonical in ("Walking", "Idle"):
        action, record = bake_action(
            source,
            target,
            source_actions[canonical],
            canonical,
            limb_lateral_damping=args.limb_lateral_damping,
        )
        output_actions.append(action)
        action_records.append(record)
    remove_source(source_objects, source_actions.values())
    removed_export_extras = remove_export_extras(target, mesh)
    material = install_pbr(mesh, color_path, bump_path, specular_path)
    mesh_contract_after = mesh_skin_sha256(mesh)
    if mesh_contract_before != mesh_contract_after:
        raise RuntimeError("retarget changed Rocketbox mesh topology, UVs, or skin weights")
    export_target(target, mesh, output_actions, output_path)

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "source_motion": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
            "actions": action_records,
        },
        "target_template": {
            "path": str(target_path),
            "sha256": sha256_file(target_path),
            "size_bytes": target_path.stat().st_size,
            "native_vertex_count": len(mesh.data.vertices),
            "native_polygon_count": len(mesh.data.polygons),
            "native_bone_count": len(target.data.bones),
            "native_skin_contract_sha256_before": mesh_contract_before,
            "native_skin_contract_sha256_after": mesh_contract_after,
            "native_skin_unchanged": True,
        },
        "bone_map": BONE_MAP,
        "removed_export_extras": removed_export_extras,
        "axis_contract": {
            "target_authored_front": "negative-y",
            "runtime_front": "positive-x",
            "cardinal_yaw_degrees": CANONICAL_YAW_DEGREES,
            "fine_yaw_inference": False,
        },
        "gait_stabilization": {
            "method": "rest_lateral_component_damping_v1",
            "limb_lateral_damping": float(args.limb_lateral_damping),
            "target_authored_forward": "negative-y",
            "target_authored_lateral": "positive-x",
            "affected_bones": sorted(LOCOMOTION_LIMB_BONES),
            "native_skin_unchanged": True,
        },
        "material": material,
        "textures": {
            "color_sha256": sha256_file(color_path),
            "bump_sha256": sha256_file(bump_path),
            "specular_sha256": sha256_file(specular_path),
        },
        "license": {
            "spdx_id": "MIT",
            "evidence_path": str(license_path),
            "evidence_sha256": sha256_file(license_path),
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
            "actions": ["Walking", "Idle"],
        },
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "QUATERNIUS_TO_ROCKETBOX_ANIMAL_RETARGET_OK "
        f"output={output_path} manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
