"""Audit a generated-mesh animal rig before attaching locomotion.

The audit is deliberately independent of bone names and animal taxonomy.  It
records the complete rest hierarchy, per-bone skin support, normalized skin
weights, and low endpoint candidates that a later semantic/IK stage can use.
It never edits or exports the input asset.
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
from mathutils import Vector


SCHEMA = "avengine_generated_animal_rig_audit_v1"


def parse_argv():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--front-axis",
        required=True,
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
        help="Reviewed visible torso-forward axis in Blender coordinates.",
    )
    parser.add_argument("--maximum-influences", type=int, default=4)
    parser.add_argument("--weight-sum-tolerance", type=float, default=1.0e-5)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def real_meshes():
    hidden = bpy.data.collections.get("glTF_not_exported")
    hidden_objects = set(hidden.objects) if hidden is not None else set()
    return [
        item
        for item in bpy.context.scene.objects
        if item.type == "MESH" and item not in hidden_objects
    ]


def linked_armatures(mesh):
    result = set()
    if mesh.parent is not None and mesh.parent.type == "ARMATURE":
        result.add(mesh.parent)
    for modifier in mesh.modifiers:
        if (
            modifier.type == "ARMATURE"
            and modifier.object is not None
            and modifier.object.type == "ARMATURE"
        ):
            result.add(modifier.object)
    return result


def world_point(obj, point):
    value = obj.matrix_world @ Vector(point)
    return [float(value.x), float(value.y), float(value.z)]


def bbox(mesh):
    points = [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    extent = [maximum[index] - minimum[index] for index in range(3)]
    diagonal = math.sqrt(sum(value * value for value in extent))
    return minimum, maximum, extent, diagonal


def normalized_position(point, minimum, extent):
    return [
        (point[index] - minimum[index]) / max(extent[index], 1.0e-12)
        for index in range(3)
    ]


def skin_records(mesh, armature, *, maximum_influences, tolerance):
    bone_names = {bone.name for bone in armature.data.bones}
    group_names = [group.name for group in mesh.vertex_groups]
    group_to_bone = {
        index: name for index, name in enumerate(group_names) if name in bone_names
    }
    support_vertices = {name: 0 for name in bone_names}
    support_mass = {name: 0.0 for name in bone_names}
    maximum_seen = 0
    unweighted = 0
    non_normalized = 0
    maximum_sum_error = 0.0
    for vertex in mesh.data.vertices:
        influences = [
            (group_to_bone[item.group], float(item.weight))
            for item in vertex.groups
            if item.group in group_to_bone and item.weight > 0.0
        ]
        maximum_seen = max(maximum_seen, len(influences))
        total = sum(weight for _, weight in influences)
        if not influences:
            unweighted += 1
            continue
        error = abs(total - 1.0)
        maximum_sum_error = max(maximum_sum_error, error)
        if error > tolerance:
            non_normalized += 1
        for name, weight in influences:
            support_vertices[name] += 1
            support_mass[name] += weight
    hard_failures = []
    if unweighted:
        hard_failures.append("unweighted_vertices")
    if non_normalized:
        hard_failures.append("non_normalized_vertices")
    if maximum_seen > maximum_influences:
        hard_failures.append("too_many_influences_per_vertex")
    return {
        "vertex_count": len(mesh.data.vertices),
        "vertex_group_count": len(group_names),
        "maximum_influences_per_vertex": maximum_seen,
        "maximum_allowed_influences": maximum_influences,
        "unweighted_vertices": unweighted,
        "non_normalized_vertices": non_normalized,
        "weight_sum_tolerance": tolerance,
        "maximum_weight_sum_error": maximum_sum_error,
        "per_bone_support": {
            name: {
                "vertices": support_vertices[name],
                "weight_mass": support_mass[name],
            }
            for name in sorted(bone_names)
        },
        "hard_failures": hard_failures,
        "status": "passed" if not hard_failures else "rejected",
    }


def hierarchy_records(armature, minimum, maximum, extent, front_axis):
    forward_sign = 1.0 if front_axis == "positive-x" else -1.0
    floor = minimum[2]
    height = extent[2]
    low_limit = floor + 0.22 * height
    records = []
    endpoint_candidates = []
    for bone in armature.data.bones:
        head = world_point(armature, bone.head_local)
        tail = world_point(armature, bone.tail_local)
        record = {
            "name": bone.name,
            "parent": bone.parent.name if bone.parent is not None else None,
            "children": [child.name for child in bone.children],
            "head_world": head,
            "tail_world": tail,
            "head_normalized_bbox": normalized_position(head, minimum, extent),
            "tail_normalized_bbox": normalized_position(tail, minimum, extent),
            "length_world": math.dist(head, tail),
        }
        records.append(record)
        if not bone.children and tail[2] <= low_limit:
            forward_coordinate = forward_sign * tail[0]
            endpoint_candidates.append(
                {
                    "bone": bone.name,
                    "tail_world": tail,
                    "tail_normalized_bbox": record["tail_normalized_bbox"],
                    "forward_coordinate": forward_coordinate,
                    "lateral_coordinate": tail[1],
                    "height_above_floor": tail[2] - floor,
                }
            )
    endpoint_candidates.sort(
        key=lambda item: (item["forward_coordinate"], item["lateral_coordinate"])
    )
    roots = [record["name"] for record in records if record["parent"] is None]
    leaves = [record["name"] for record in records if not record["children"]]
    return {
        "bone_count": len(records),
        "root_bones": roots,
        "leaf_bones": leaves,
        "records": records,
        "low_leaf_endpoint_height_fraction": 0.22,
        "low_leaf_endpoint_candidates": endpoint_candidates,
    }


def main():
    args = parse_argv()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if source.suffix.lower() not in {".glb", ".gltf"} or not source.is_file():
        raise SystemExit(f"missing or unsupported input: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")
    if args.maximum_influences < 1 or args.weight_sum_tolerance <= 0.0:
        raise SystemExit("invalid skin audit thresholds")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = real_meshes()
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    skinned = [mesh for mesh in meshes if linked_armatures(mesh)]
    if len(skinned) != 1 or len(armatures) != 1:
        raise SystemExit(
            "expected exactly one real skinned mesh and one armature; "
            f"meshes={[item.name for item in meshes]} "
            f"skinned={[item.name for item in skinned]} "
            f"armatures={[item.name for item in armatures]}"
        )
    mesh = skinned[0]
    armature = armatures[0]
    if armature not in linked_armatures(mesh):
        raise SystemExit("skinned mesh is linked to an unexpected armature")
    minimum, maximum, extent, diagonal = bbox(mesh)
    skin = skin_records(
        mesh,
        armature,
        maximum_influences=args.maximum_influences,
        tolerance=args.weight_sum_tolerance,
    )
    hierarchy = hierarchy_records(
        armature, minimum, maximum, extent, args.front_axis
    )
    hard_failures = list(skin["hard_failures"])
    if len(hierarchy["root_bones"]) != 1:
        hard_failures.append("skeleton_must_have_exactly_one_root")
    if hierarchy["bone_count"] < 5:
        hard_failures.append("skeleton_too_small")
    if len(hierarchy["low_leaf_endpoint_candidates"]) < 2:
        hard_failures.append("insufficient_low_limb_endpoints")
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "coordinate_contract": {
            "up_axis": "positive-z",
            "reviewed_front_axis": args.front_axis,
            "automatic_fine_yaw": False,
        },
        "mesh": {
            "name": mesh.name,
            "vertices": len(mesh.data.vertices),
            "polygons": len(mesh.data.polygons),
            "world_bbox_min": minimum,
            "world_bbox_max": maximum,
            "world_bbox_extent": extent,
            "world_bbox_diagonal": diagonal,
            "materials": [item.name for item in mesh.data.materials if item],
            "uv_layers": [item.name for item in mesh.data.uv_layers],
        },
        "armature": {"name": armature.name, **hierarchy},
        "skin": skin,
        "automatic_checks": {
            "hard_failures": hard_failures,
            "overall": "passed" if not hard_failures else "rejected",
        },
        "animation_authorized": False,
        "next_gate": "semantic_limb_assignment_and_rest_skeleton_visual_review",
        "formal_dataset_registration_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_ANIMAL_RIG_AUDIT_OK "
        f"overall={payload['automatic_checks']['overall']} "
        f"bones={hierarchy['bone_count']} "
        f"low_endpoints={len(hierarchy['low_leaf_endpoint_candidates'])} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
