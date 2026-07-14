#!/usr/bin/env python3
"""Retarget Quaternius Dog Walk/Idle to a TokenRig-generated quadruped rig.

The generated mesh, PBR material, skeleton, and skin weights remain the target
authority.  Target bone names are never assumed: a reviewed front axis plus
the rest hierarchy identifies the axial, head, tail, and four limb chains.
Source rotations are sampled in world space and smoothly resampled by chain
arc length, which avoids copying unrelated local bone rolls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.generated_quadruped_semantics import (  # noqa: E402
    SemanticRigError,
    infer_quadruped_semantics,
)


SCHEMA = "avengine_generated_quadruped_retarget_v1"
SOURCE_CHAINS = {
    "axial_head": ("Bone.001", "Bone.002", "Bone.003"),
    "tail": ("Bone.004", "Bone.005", "Bone.006", "Bone.007"),
    "front_side_negative": ("Bone.014", "Bone.015", "Bone.016"),
    "front_side_positive": ("Bone.017", "Bone.018", "Bone.019"),
    "hind_side_negative": ("Bone.011", "Bone.012", "Bone.013"),
    "hind_side_positive": ("Bone.008", "Bone.009", "Bone.010"),
}
SOURCE_ROOT = "Bone"
ACTION_HINTS = {"Walking": "walk", "Idle": "idle"}


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-glb", type=Path, required=True)
    parser.add_argument("--source-rig-glb", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--target-front-axis",
        required=True,
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
    )
    return parser.parse_args(argv)


def require_file(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe {label}: {path}")
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


def hidden_objects():
    collection = bpy.data.collections.get("glTF_not_exported")
    return set(collection.objects) if collection is not None else set()


def imported_real_meshes(imported):
    hidden = hidden_objects()
    return [item for item in imported if item.type == "MESH" and item not in hidden]


def linked_armatures(mesh):
    result = set()
    if mesh.parent is not None and mesh.parent.type == "ARMATURE":
        result.add(mesh.parent)
    for modifier in mesh.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            result.add(modifier.object)
    return result


def import_target(path: Path):
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = tuple(item for item in bpy.data.objects if item not in before_objects)
    actions = [item for item in bpy.data.actions if item not in before_actions]
    armatures = [item for item in imported if item.type == "ARMATURE"]
    meshes = imported_real_meshes(imported)
    skinned = [item for item in meshes if linked_armatures(item)]
    if len(armatures) != 1 or len(skinned) != 1:
        raise RuntimeError(
            "target must contain one real skinned mesh and one armature; "
            f"meshes={[item.name for item in meshes]} "
            f"armatures={[item.name for item in armatures]}"
        )
    if actions:
        raise RuntimeError("generated target must not already contain actions")
    armature = armatures[0]
    mesh = skinned[0]
    if armature not in linked_armatures(mesh):
        raise RuntimeError("target mesh is linked to an unexpected armature")
    return armature, mesh, imported


def import_source(path: Path):
    before_objects = set(bpy.data.objects)
    before_actions = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = tuple(item for item in bpy.data.objects if item not in before_objects)
    actions = [item for item in bpy.data.actions if item not in before_actions]
    armatures = [item for item in imported if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError("source rig must contain exactly one armature")
    selected = {}
    for canonical, hint in ACTION_HINTS.items():
        matches = [item for item in actions if hint in item.name.lower()]
        if len(matches) != 1:
            raise RuntimeError(
                f"source action {canonical} is ambiguous: {[item.name for item in matches]}"
            )
        selected[canonical] = matches[0]
    required = {SOURCE_ROOT}
    for chain in SOURCE_CHAINS.values():
        required.update(chain)
    missing = sorted(required - set(armatures[0].data.bones.keys()))
    if missing:
        raise RuntimeError(f"source rig is missing required bones: {missing}")
    return armatures[0], imported, selected


def detach_armature_parent(armature):
    if armature.parent is None:
        return None
    parent_name = armature.parent.name
    world = armature.matrix_world.copy()
    armature.parent = None
    armature.matrix_world = world
    bpy.context.view_layer.update()
    return parent_name


def target_bbox(mesh):
    points = [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    extent = [maximum[index] - minimum[index] for index in range(3)]
    return minimum, maximum, extent


def target_semantic_records(armature):
    records = []
    for bone in armature.data.bones:
        head = armature.matrix_world @ bone.head_local
        tail = armature.matrix_world @ bone.tail_local
        records.append(
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent is not None else None,
                "children": [child.name for child in bone.children],
                "head_world": [float(value) for value in head],
                "tail_world": [float(value) for value in tail],
            }
        )
    return records


def target_chains(armature, mesh, front_axis):
    minimum, _maximum, extent = target_bbox(mesh)
    semantics = infer_quadruped_semantics(
        target_semantic_records(armature),
        bbox_min=minimum,
        bbox_extent=extent,
        front_axis=front_axis,
    )
    axial_head = tuple(semantics.axial[1:]) + tuple(semantics.head_chain)
    if not axial_head:
        raise SemanticRigError("target axial/head chain is empty after root")
    chains = {
        "axial_head": axial_head,
        "tail": semantics.tail_chain,
        "front_side_negative": semantics.front_side_negative,
        "front_side_positive": semantics.front_side_positive,
        "hind_side_negative": semantics.hind_side_negative,
        "hind_side_positive": semantics.hind_side_positive,
    }
    covered = {semantics.root}
    for chain in chains.values():
        covered.update(chain)
    target_names = set(armature.data.bones.keys())
    if covered != target_names:
        raise RuntimeError(
            "target semantics do not cover the complete skeleton: "
            f"missing={sorted(target_names - covered)} extra={sorted(covered - target_names)}"
        )
    return semantics, chains


def bone_world_rotation(armature, matrix) -> Quaternion:
    return (armature.matrix_world.to_quaternion() @ matrix.to_quaternion()).normalized()


def normalized_chain_centers(armature, names):
    lengths = [max(float(armature.data.bones[name].length), 1.0e-9) for name in names]
    total = sum(lengths)
    cursor = 0.0
    centers = []
    for length in lengths:
        centers.append((cursor + 0.5 * length) / total)
        cursor += length
    return centers


def source_sampling_plan(target, source, target_names, source_names):
    target_centers = normalized_chain_centers(target, target_names)
    source_centers = normalized_chain_centers(source, source_names)
    plan = []
    for target_name, fraction in zip(target_names, target_centers):
        if fraction <= source_centers[0]:
            first = second = 0
            blend = 0.0
        elif fraction >= source_centers[-1]:
            first = second = len(source_centers) - 1
            blend = 0.0
        else:
            second = next(
                index for index, value in enumerate(source_centers) if value >= fraction
            )
            first = second - 1
            span = source_centers[second] - source_centers[first]
            blend = (fraction - source_centers[first]) / max(span, 1.0e-12)
        source_first = source_names[first]
        source_second = source_names[second]
        source_rest_first = bone_world_rotation(
            source, source.data.bones[source_first].matrix_local
        )
        source_rest_second = bone_world_rotation(
            source, source.data.bones[source_second].matrix_local
        )
        source_rest = source_rest_first.slerp(source_rest_second, blend)
        target_rest = bone_world_rotation(
            target, target.data.bones[target_name].matrix_local
        )
        plan.append(
            {
                "target": target_name,
                "source_first": source_first,
                "source_second": source_second,
                "blend": float(blend),
                "source_rest_world": source_rest,
                "target_rest_world": target_rest,
                "target_chain_fraction": float(fraction),
            }
        )
    return plan


def full_sampling_plan(target, source, semantics, chains):
    source_root_rest = bone_world_rotation(
        source, source.data.bones[SOURCE_ROOT].matrix_local
    )
    target_root_rest = bone_world_rotation(
        target, target.data.bones[semantics.root].matrix_local
    )
    result = [
        {
            "target": semantics.root,
            "source_first": SOURCE_ROOT,
            "source_second": SOURCE_ROOT,
            "blend": 0.0,
            "source_rest_world": source_root_rest,
            "target_rest_world": target_root_rest,
            "target_chain_fraction": 0.0,
            "semantic_chain": "root",
        }
    ]
    for semantic, target_names in chains.items():
        entries = source_sampling_plan(
            target, source, target_names, SOURCE_CHAINS[semantic]
        )
        for entry in entries:
            entry["semantic_chain"] = semantic
        result.extend(entries)
    if len({entry["target"] for entry in result}) != len(target.data.bones):
        raise RuntimeError("sampling plan must map every target bone exactly once")
    return result


def source_action_sample_frames(action, source_bones):
    prefix = 'pose.bones["'
    frames = set()
    for curve in action.fcurves:
        if not curve.data_path.startswith(prefix):
            continue
        bone_name = curve.data_path[len(prefix) :].split('"', 1)[0]
        if bone_name in source_bones:
            frames.update(float(point.co.x) for point in curve.keyframe_points)
    ordered = sorted(frames)
    if len(ordered) < 2:
        raise RuntimeError(f"source action has too few samples: {action.name}")
    return ordered


def set_scene_time(value):
    base = math.floor(float(value))
    bpy.context.scene.frame_set(base, subframe=float(value) - base)


def cache_source_action(source, action, source_bones):
    source.animation_data_create()
    source.animation_data.action = action
    result = []
    for output_frame, source_frame in enumerate(
        source_action_sample_frames(action, source_bones)
    ):
        set_scene_time(source_frame)
        bpy.context.view_layer.update()
        result.append(
            {
                "frame": output_frame,
                "source_frame": source_frame,
                "rotations": {
                    name: bone_world_rotation(source, source.pose.bones[name].matrix)
                    for name in source_bones
                },
            }
        )
    return result


def target_parent_first(target):
    def depth(name):
        value = 0
        bone = target.data.bones[name]
        while bone.parent is not None:
            value += 1
            bone = bone.parent
        return value

    return sorted(target.data.bones.keys(), key=lambda name: (depth(name), name))


def parent_local_rest(bone):
    if bone.parent is None:
        return bone.matrix_local.copy()
    return bone.parent.matrix_local.inverted() @ bone.matrix_local


def keyframe_pose_bone(pose_bone, frame):
    pose_bone.keyframe_insert(data_path="location", frame=frame, group=pose_bone.name)
    pose_bone.keyframe_insert(
        data_path="rotation_quaternion", frame=frame, group=pose_bone.name
    )
    pose_bone.keyframe_insert(data_path="scale", frame=frame, group=pose_bone.name)


def shortest_quaternion_error(first, second):
    dot = min(1.0, max(-1.0, abs(float(first.normalized().dot(second.normalized())))))
    return 2.0 * math.acos(dot)


def bake_action(source, target, source_action, canonical_name, plan):
    source_bones = sorted(
        {
            entry["source_first"]
            for entry in plan
        }
        | {entry["source_second"] for entry in plan}
    )
    cached = cache_source_action(source, source_action, source_bones)
    action = bpy.data.actions.new(name=canonical_name)
    target.animation_data_create()
    target.animation_data.action = action
    target.data.pose_position = "POSE"
    rest_locals = {
        name: parent_local_rest(target.data.bones[name]) for name in target.data.bones.keys()
    }
    by_target = {entry["target"]: entry for entry in plan}
    order = target_parent_first(target)
    target_object_rotation = target.matrix_world.to_quaternion().normalized()
    max_error = 0.0
    for cached_frame in cached:
        frame = cached_frame["frame"]
        bpy.context.scene.frame_set(frame)
        for pose_bone in target.pose.bones:
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        requested = {}
        for target_name in order:
            entry = by_target[target_name]
            pose_bone = target.pose.bones[target_name]
            rest_local = rest_locals[target_name]
            if pose_bone.parent is None:
                translation = rest_local.translation.copy()
            else:
                translation = pose_bone.parent.matrix @ rest_local.translation
            source_first = cached_frame["rotations"][entry["source_first"]]
            source_second = cached_frame["rotations"][entry["source_second"]]
            source_pose_world = source_first.slerp(source_second, entry["blend"])
            desired_world = (
                entry["target_rest_world"]
                @ entry["source_rest_world"].inverted()
                @ source_pose_world
            ).normalized()
            desired_armature = (
                target_object_rotation.inverted() @ desired_world
            ).normalized()
            desired = Matrix.LocRotScale(
                translation, desired_armature, Vector((1.0, 1.0, 1.0))
            )
            pose_bone.matrix = desired
            bpy.context.view_layer.update()
            requested[target_name] = desired_armature
        for pose_bone in target.pose.bones:
            keyframe_pose_bone(pose_bone, frame)
        for name, desired in requested.items():
            actual = target.pose.bones[name].matrix.to_quaternion()
            max_error = max(max_error, shortest_quaternion_error(actual, desired))
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
        "frame_range": [0, len(cached) - 1],
        "sampled_frames": len(cached),
        "maximum_requested_rotation_error_degrees": math.degrees(max_error),
    }


def remove_source(imported, actions):
    for item in imported:
        if item.name in bpy.data.objects:
            bpy.data.objects.remove(item, do_unlink=True)
    for action in actions:
        if action.name in bpy.data.actions:
            bpy.data.actions.remove(action)


def remove_export_extras(target, mesh):
    removed = []
    for item in list(bpy.data.objects):
        if item not in {target, mesh}:
            removed.append(item.name)
            bpy.data.objects.remove(item, do_unlink=True)
    return sorted(removed)


def add_nla_tracks(target, actions):
    target.animation_data_create()
    target.animation_data.action = None
    while target.animation_data.nla_tracks:
        target.animation_data.nla_tracks.remove(target.animation_data.nla_tracks[0])
    for action in actions:
        start, end = [int(round(value)) for value in action.frame_range]
        track = target.animation_data.nla_tracks.new()
        track.name = action.name
        strip = track.strips.new(action.name, start, action)
        strip.name = action.name
        strip.action_frame_start = start
        strip.action_frame_end = end


def export_target(target, mesh, actions, output, target_front_axis):
    add_nla_tracks(target, actions)
    canonical_yaw = {
        "positive-x": 0.0,
        "negative-x": 180.0,
        "positive-y": -90.0,
        "negative-y": 90.0,
    }[target_front_axis]
    if canonical_yaw:
        target.matrix_world = (
            Matrix.Rotation(math.radians(canonical_yaw), 4, "Z")
            @ target.matrix_world
        )
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
        export_materials="EXPORT",
    )
    return canonical_yaw


def serializable_plan(plan):
    return [
        {
            key: value
            for key, value in entry.items()
            if key not in {"source_rest_world", "target_rest_world"}
        }
        for entry in plan
    ]


def main():
    args = parse_argv()
    target_path = require_file(args.target_glb, "generated target GLB")
    source_path = require_file(args.source_rig_glb, "source rig GLB")
    output = require_output(args.output_glb, "animated output GLB")
    manifest_path = require_output(args.manifest, "retarget manifest")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    target, mesh, target_imported = import_target(target_path)
    detached_parent = detach_armature_parent(target)
    semantics, chains = target_chains(target, mesh, args.target_front_axis)
    source, source_imported, source_actions = import_source(source_path)
    plan = full_sampling_plan(target, source, semantics, chains)
    action_results = []
    output_actions = []
    for canonical in ("Walking", "Idle"):
        action, result = bake_action(
            source,
            target,
            source_actions[canonical],
            canonical,
            plan,
        )
        output_actions.append(action)
        action_results.append(result)
    remove_source(source_imported, source_actions.values())
    removed = remove_export_extras(target, mesh)
    canonical_yaw = export_target(
        target, mesh, output_actions, output, args.target_front_axis
    )
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "path": str(target_path),
            "sha256": sha256_file(target_path),
            "mesh_pbr_skeleton_and_weights_authority": True,
            "reviewed_front_axis": args.target_front_axis,
        },
        "source_motion": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "geometry_used": False,
            "weights_used": False,
        },
        "semantic_inference": {
            "method": "one_root_four_low_leaf_geometry_hierarchy_v1",
            "bone_name_independent_target": True,
            "root": semantics.root,
            "chains": {name: list(value) for name, value in chains.items()},
            "foot_leaves": list(semantics.foot_leaves),
            "complete_target_bone_coverage": True,
        },
        "rotation_transfer": {
            "method": "world_space_rest_offset_chain_arc_length_slerp_v1",
            "local_bone_roll_copied": False,
            "sampling_plan": serializable_plan(plan),
        },
        "actions": action_results,
        "export": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            "canonical_front_axis": "positive-x",
            "canonical_yaw_degrees": canonical_yaw,
            "detached_target_parent": detached_parent,
            "removed_export_extras": removed,
            "action_names": ["Walking", "Idle"],
        },
        "status": "research_candidate_pending_deformation_and_visual_qa",
        "formal_dataset_registration_authorized": False,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_QUADRUPED_RETARGET_OK "
        f"bones={len(target.data.bones)} actions=2 output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
