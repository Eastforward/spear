"""Inventory a rigged animal GLB without changing or exporting it.

The report is intentionally species-independent.  It records topology,
skinning, actions, bounds, and license evidence so a stable template can be
selected before any generated appearance is applied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy


SCHEMA = "avengine_animal_template_inventory_v1"


def parse_argv():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--license-id", default="unknown")
    parser.add_argument("--license-evidence")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def connected_component_count(mesh) -> int:
    adjacency = [[] for _ in mesh.vertices]
    for edge in mesh.edges:
        first, second = edge.vertices
        adjacency[first].append(second)
        adjacency[second].append(first)
    visited = set()
    components = 0
    for vertex in range(len(mesh.vertices)):
        if vertex in visited:
            continue
        components += 1
        stack = [vertex]
        visited.add(vertex)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return components


def edge_incidence(mesh):
    counts = Counter()
    triangles = 0
    for polygon in mesh.polygons:
        indices = list(polygon.vertices)
        triangles += max(0, len(indices) - 2)
        for index, first in enumerate(indices):
            second = indices[(index + 1) % len(indices)]
            counts[tuple(sorted((first, second)))] += 1
    return {
        "triangles_after_fan_triangulation": triangles,
        "boundary_edges": sum(value == 1 for value in counts.values()),
        "manifold_edges": sum(value == 2 for value in counts.values()),
        "nonmanifold_edges": sum(value > 2 for value in counts.values()),
    }


def mesh_record(obj):
    mesh = obj.data
    points = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    influence_counts = [len(vertex.groups) for vertex in mesh.vertices]
    return {
        "name": obj.name,
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "polygons": len(mesh.polygons),
        **edge_incidence(mesh),
        "connected_components": connected_component_count(mesh),
        "uv_layers": [layer.name for layer in mesh.uv_layers],
        "materials": [material.name for material in mesh.materials if material],
        "vertex_groups": len(obj.vertex_groups),
        "vertices_with_weights": sum(count > 0 for count in influence_counts),
        "maximum_influences_per_vertex": max(influence_counts, default=0),
        "armature_modifiers": [
            modifier.object.name
            for modifier in obj.modifiers
            if modifier.type == "ARMATURE" and modifier.object is not None
        ],
        "world_bbox_min": minimum,
        "world_bbox_max": maximum,
        "world_bbox_extent": [maximum[i] - minimum[i] for i in range(3)],
    }


def main():
    args = parse_argv()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if source.suffix.lower() not in {".glb", ".gltf"} or not source.is_file():
        raise SystemExit(f"missing or unsupported input: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")
    license_evidence = None
    if args.license_evidence:
        license_evidence = Path(args.license_evidence).resolve()
        if not license_evidence.is_file():
            raise SystemExit(f"missing license evidence: {license_evidence}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [item for item in bpy.data.objects if item.type == "MESH"]
    armatures = [item for item in bpy.data.objects if item.type == "ARMATURE"]
    if not meshes:
        raise SystemExit("input has no mesh")

    mesh_records = [mesh_record(item) for item in meshes]
    armature_records = [
        {
            "name": item.name,
            "bones": len(item.data.bones),
            "root_bones": [bone.name for bone in item.data.bones if bone.parent is None],
            "bone_names": [bone.name for bone in item.data.bones],
        }
        for item in armatures
    ]
    actions = [
        {
            "name": action.name,
            "frame_range": [float(value) for value in action.frame_range],
            "fcurves": len(action.fcurves),
        }
        for action in bpy.data.actions
    ]
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(source),
        "input_sha256": sha256_file(source),
        "input_size_bytes": source.stat().st_size,
        "license": {
            "spdx_or_policy_id": args.license_id,
            "evidence_path": str(license_evidence) if license_evidence else None,
            "evidence_sha256": sha256_file(license_evidence) if license_evidence else None,
        },
        "meshes": mesh_records,
        "armatures": armature_records,
        "actions": actions,
        "has_walk_action": any("walk" in item["name"].lower() for item in actions),
        "has_idle_action": any("idle" in item["name"].lower() for item in actions),
        "orientation_policy": {
            "automatic_fine_yaw_inference": False,
            "status": "requires_authored_or_manual_cardinal_direction",
        },
        "formal_dataset_registration_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"ANIMAL_TEMPLATE_INVENTORY_OK output={output}")


if __name__ == "__main__":
    main()
