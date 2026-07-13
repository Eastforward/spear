"""Audit disconnected mesh components without constructing triangle adjacency.

Run with Blender so GLB import exactly matches the LOD/binding path::

  blender -b --python tools/blender_audit_mesh_components.py -- \
    --input mesh.glb --output component_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy


SCHEMA = "avengine_blender_mesh_component_audit_v1"


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-limit", type=int, default=200)
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _find(parent, value):
    root = value
    while parent[root] != root:
        root = parent[root]
    while parent[value] != value:
        next_value = parent[value]
        parent[value] = root
        value = next_value
    return root


def _union(parent, size, left, right):
    left = _find(parent, left)
    right = _find(parent, right)
    if left == right:
        return
    if size[left] < size[right]:
        left, right = right, left
    parent[right] = left
    size[left] += size[right]


def audit_object(obj, report_limit):
    mesh = obj.data
    vertex_count = len(mesh.vertices)
    parent = list(range(vertex_count))
    size = [1] * vertex_count
    for edge in mesh.edges:
        _union(parent, size, edge.vertices[0], edge.vertices[1])

    records = {}
    for vertex in mesh.vertices:
        root = _find(parent, vertex.index)
        point = vertex.co
        record = records.get(root)
        if record is None:
            record = {
                "root_vertex": root,
                "vertices": 0,
                "faces": 0,
                "bbox_min": [float(point.x), float(point.y), float(point.z)],
                "bbox_max": [float(point.x), float(point.y), float(point.z)],
            }
            records[root] = record
        record["vertices"] += 1
        for axis in range(3):
            value = float(point[axis])
            record["bbox_min"][axis] = min(record["bbox_min"][axis], value)
            record["bbox_max"][axis] = max(record["bbox_max"][axis], value)
    for polygon in mesh.polygons:
        if polygon.vertices:
            records[_find(parent, polygon.vertices[0])]["faces"] += 1
    components = list(records.values())
    for record in components:
        record["bbox_extent"] = [
            record["bbox_max"][axis] - record["bbox_min"][axis]
            for axis in range(3)
        ]
    components.sort(key=lambda item: (-item["vertices"], item["root_vertex"]))
    asset_min = [
        min(record["bbox_min"][axis] for record in components) for axis in range(3)
    ]
    asset_max = [
        max(record["bbox_max"][axis] for record in components) for axis in range(3)
    ]
    asset_extent = [asset_max[axis] - asset_min[axis] for axis in range(3)]
    height = max(asset_extent[2], 1e-12)
    for record in components:
        record["relative_to_asset"] = {
            "vertex_fraction": record["vertices"] / max(vertex_count, 1),
            "min_z_height_fraction": (record["bbox_min"][2] - asset_min[2])
            / height,
            "max_z_height_fraction": (record["bbox_max"][2] - asset_min[2])
            / height,
        }
    buckets = {}
    for threshold in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
        buckets[f"le_{threshold}_vertices"] = sum(
            record["vertices"] <= threshold for record in components
        )
    small_low = [
        record
        for record in components
        if record["vertices"] <= 1024
        and record["relative_to_asset"]["max_z_height_fraction"] <= 0.10
    ]
    return {
        "object": obj.name,
        "vertices": vertex_count,
        "edges": len(mesh.edges),
        "faces": len(mesh.polygons),
        "component_count": len(components),
        "asset_bbox_min": asset_min,
        "asset_bbox_max": asset_max,
        "asset_bbox_extent": asset_extent,
        "largest_component_vertex_fraction": components[0]["vertices"]
        / max(vertex_count, 1),
        "size_buckets": buckets,
        "largest_components": components[:report_limit],
        "small_low_component_count": len(small_low),
        "small_low_components": small_low[:report_limit],
    }


def main():
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if source.is_symlink() or not source.is_file():
        raise SystemExit(f"input is missing/non-direct: {source}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")
    if not 1 <= args.report_limit <= 1000:
        raise SystemExit("--report-limit must be in [1, 1000]")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise SystemExit("input GLB contains no mesh")
    payload = {
        "schema": SCHEMA,
        "input": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "objects": [audit_object(obj, args.report_limit) for obj in meshes],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"MESH_COMPONENT_AUDIT_OK output={output}", flush=True)


if __name__ == "__main__":
    main()
