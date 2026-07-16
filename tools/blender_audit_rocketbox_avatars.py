#!/usr/bin/env python3

"""Measure canonical Rocketbox avatars in Blender without modifying them.

This script is intentionally shardable because 115 FBX imports are independent:

    blender --background --python tools/blender_audit_rocketbox_avatars.py -- \
      --rocketbox-root /data/datasets/rocketbox/Microsoft-Rocketbox \
      --shard-index 0 --shard-count 4 --output /tmp/audit-0.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

import bpy
from mathutils import Vector


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from rocketbox_inventory import AUDIT_SCHEMA, discover_canonical_avatars


REQUIRED_BIP01_BONES = (
    "Bip01 Pelvis",
    "Bip01 Spine",
    "Bip01 Spine1",
    "Bip01 Spine2",
    "Bip01 Neck",
    "Bip01 Head",
    "Bip01 L UpperArm",
    "Bip01 L Forearm",
    "Bip01 L Hand",
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
SKELETON_FAMILIES = ("Bip01", "Bip02")


def required_bones_for_family(family: str) -> tuple[str, ...]:
    if family not in SKELETON_FAMILIES:
        raise RuntimeError(f"unsupported Rocketbox skeleton family: {family}")
    return tuple(name.replace("Bip01", family, 1) for name in REQUIRED_BIP01_BONES)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocketbox-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _vec(value: Vector) -> list[float]:
    return [float(component) for component in value]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rest_mesh_bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    vertex_count = 0
    for mesh in meshes:
        for vertex in mesh.data.vertices:
            world = mesh.matrix_world @ vertex.co
            if not all(math.isfinite(float(component)) for component in world):
                raise RuntimeError(f"non-finite vertex position in {mesh.name}")
            minimum.x = min(minimum.x, world.x)
            minimum.y = min(minimum.y, world.y)
            minimum.z = min(minimum.z, world.z)
            maximum.x = max(maximum.x, world.x)
            maximum.y = max(maximum.y, world.y)
            maximum.z = max(maximum.z, world.z)
            vertex_count += 1
    if vertex_count == 0:
        raise RuntimeError("avatar contains no mesh vertices")
    return minimum, maximum


def _skin_stats(mesh: bpy.types.Object, bone_names: set[str]) -> dict:
    bone_group_indices = {
        group.index for group in mesh.vertex_groups if group.name in bone_names
    }
    nonfinite_weight_count = 0
    negative_weight_count = 0
    unweighted_vertex_count = 0
    unweighted_surface_vertex_count = 0
    loose_unweighted_vertex_count = 0
    unweighted_surface_vertex_indices = []
    loose_unweighted_vertex_indices = []
    maximum_influence_count = 0
    minimum_weight_sum = math.inf
    maximum_weight_sum = -math.inf
    surface_vertices = {
        int(vertex_index)
        for polygon in mesh.data.polygons
        for vertex_index in polygon.vertices
    }
    for vertex in mesh.data.vertices:
        weights = [
            float(group.weight)
            for group in vertex.groups
            if group.group in bone_group_indices
        ]
        nonfinite_weight_count += sum(not math.isfinite(weight) for weight in weights)
        negative_weight_count += sum(weight < 0.0 for weight in weights)
        finite_positive = [
            weight for weight in weights if math.isfinite(weight) and weight > 0.0
        ]
        total = sum(finite_positive)
        if total <= 0.0:
            unweighted_vertex_count += 1
            if vertex.index in surface_vertices:
                unweighted_surface_vertex_count += 1
                unweighted_surface_vertex_indices.append(int(vertex.index))
            else:
                loose_unweighted_vertex_count += 1
                loose_unweighted_vertex_indices.append(int(vertex.index))
        minimum_weight_sum = min(minimum_weight_sum, total)
        maximum_weight_sum = max(maximum_weight_sum, total)
        maximum_influence_count = max(maximum_influence_count, len(finite_positive))
    if not mesh.data.vertices:
        minimum_weight_sum = 0.0
        maximum_weight_sum = 0.0
    return {
        "bone_vertex_group_count": len(bone_group_indices),
        "nonfinite_weight_count": nonfinite_weight_count,
        "negative_weight_count": negative_weight_count,
        "unweighted_vertex_count": unweighted_vertex_count,
        "unweighted_surface_vertex_count": unweighted_surface_vertex_count,
        "loose_unweighted_vertex_count": loose_unweighted_vertex_count,
        "unweighted_surface_vertex_indices": unweighted_surface_vertex_indices,
        "loose_unweighted_vertex_indices": loose_unweighted_vertex_indices,
        "maximum_influence_count": maximum_influence_count,
        "minimum_weight_sum": minimum_weight_sum,
        "maximum_weight_sum": maximum_weight_sum,
    }


def audit_avatar(source: dict) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.fbx(filepath=source["fbx_path"])
    if "FINISHED" not in result:
        raise RuntimeError("Blender FBX import did not finish")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    armatures = [obj for obj in imported if obj.type == "ARMATURE"]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(armatures) != 1:
        raise RuntimeError(f"expected one armature, found {len(armatures)}")
    if not meshes:
        raise RuntimeError("avatar FBX contains no meshes")
    armature = armatures[0]
    armature.data.pose_position = "REST"
    scene.frame_set(1)
    bpy.context.view_layer.update()

    bone_names = {bone.name for bone in armature.data.bones}
    matching_families = [
        family
        for family in SKELETON_FAMILIES
        if set(required_bones_for_family(family)) <= bone_names
    ]
    if len(matching_families) != 1:
        missing_by_family = {
            family: sorted(set(required_bones_for_family(family)) - bone_names)
            for family in SKELETON_FAMILIES
        }
        raise RuntimeError(
            f"semantic skeleton family is ambiguous/incomplete: {missing_by_family}"
        )
    skeleton_family = matching_families[0]
    required_bones = required_bones_for_family(skeleton_family)
    if len(bone_names) != 80:
        raise RuntimeError(f"expected 80 Rocketbox bones, found {len(bone_names)}")

    minimum, maximum = rest_mesh_bounds(meshes)
    dimensions = maximum - minimum
    authored_height_cm = float(dimensions.z * 100.0)
    if not 50.0 <= authored_height_cm <= 250.0:
        raise RuntimeError(f"implausible authored height: {authored_height_cm}cm")

    mesh_records = []
    total_nonfinite = 0
    total_negative = 0
    total_unweighted = 0
    total_unweighted_surface = 0
    total_loose_unweighted = 0
    for mesh in sorted(meshes, key=lambda value: value.name):
        skin = _skin_stats(mesh, bone_names)
        total_nonfinite += skin["nonfinite_weight_count"]
        total_negative += skin["negative_weight_count"]
        total_unweighted += skin["unweighted_vertex_count"]
        total_unweighted_surface += skin["unweighted_surface_vertex_count"]
        total_loose_unweighted += skin["loose_unweighted_vertex_count"]
        mesh_records.append(
            {
                "name": mesh.name,
                "vertex_count": len(mesh.data.vertices),
                "polygon_count": len(mesh.data.polygons),
                "uv_layer_count": len(mesh.data.uv_layers),
                "material_slot_names": [
                    slot.material.name if slot.material is not None else None
                    for slot in mesh.material_slots
                ],
                "armature_modifier_count": sum(
                    modifier.type == "ARMATURE" for modifier in mesh.modifiers
                ),
                "skin": skin,
            }
        )
    if total_nonfinite or total_negative:
        raise RuntimeError(
            f"invalid skin weights: nonfinite={total_nonfinite} negative={total_negative}"
        )
    if total_unweighted_surface:
        raise RuntimeError(
            f"unweighted surface mesh vertices: {total_unweighted_surface}"
        )

    armature_scale = _vec(armature.scale)
    return {
        "base_avatar_id": source["base_avatar_id"],
        "fbx_path": source["fbx_path"],
        "fbx_sha256": source["fbx_sha256"],
        "status": "passed",
        "errors": [],
        "blender_version": bpy.app.version_string,
        "object_count": len(imported),
        "mesh_count": len(meshes),
        "armature_count": len(armatures),
        "armature_name": armature.name,
        "armature_scale": armature_scale,
        "bone_count": len(bone_names),
        "skeleton_family": skeleton_family,
        "required_bones_present": list(required_bones),
        "bounds_m": {"minimum": _vec(minimum), "maximum": _vec(maximum)},
        "bounds_cm": {
            "minimum": [value * 100.0 for value in _vec(minimum)],
            "maximum": [value * 100.0 for value in _vec(maximum)],
            "dimensions": [value * 100.0 for value in _vec(dimensions)],
        },
        "authored_height_cm": authored_height_cm,
        "rest_floor_z_cm": float(minimum.z * 100.0),
        "front_axis": "-Y",
        "up_axis": "+Z",
        "mesh_records": mesh_records,
        "aggregate_skin": {
            "nonfinite_weight_count": total_nonfinite,
            "negative_weight_count": total_negative,
            "unweighted_vertex_count": total_unweighted,
            "unweighted_surface_vertex_count": total_unweighted_surface,
            "loose_unweighted_vertex_count": total_loose_unweighted,
        },
        "authored_geometry_modified": False,
        "authored_scale_modified": False,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.shard_count <= 0:
        raise RuntimeError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("--shard-index must be in [0, shard-count)")
    all_sources = discover_canonical_avatars(args.rocketbox_root)
    sources = [
        source
        for index, source in enumerate(all_sources)
        if index % args.shard_count == args.shard_index
    ]
    records = []
    for ordinal, source in enumerate(sources, start=1):
        print(
            f"AUDIT_ROCKETBOX {ordinal}/{len(sources)} "
            f"{source['base_avatar_id']}",
            flush=True,
        )
        try:
            records.append(audit_avatar(source))
        except Exception as error:
            records.append(
                {
                    "base_avatar_id": source["base_avatar_id"],
                    "fbx_path": source["fbx_path"],
                    "fbx_sha256": source["fbx_sha256"],
                    "status": "failed",
                    "errors": [str(error)],
                    "traceback": traceback.format_exc(),
                    "authored_geometry_modified": False,
                    "authored_scale_modified": False,
                }
            )
    payload = {
        "schema_version": "rocketbox_blender_audit_shard_v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_total": len(all_sources),
        "record_count": len(records),
        "passed_count": sum(record["status"] == "passed" for record in records),
        "failed_count": sum(record["status"] != "passed" for record in records),
        "records": records,
    }
    _atomic_json(args.output, payload)
    print(
        f"ROCKETBOX_BLENDER_AUDIT_DONE shard={args.shard_index} "
        f"passed={payload['passed_count']} failed={payload['failed_count']} "
        f"output={args.output}",
        flush=True,
    )
    return 1 if payload["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
