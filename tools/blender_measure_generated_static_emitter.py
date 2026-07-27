#!/usr/bin/env python3
"""Measure a reviewed static-object emitter on its finalized mesh surface.

The tool contains no object-class heuristics.  A per-instance, hash-bound
anchor spec selects reviewed surface samples either by exact barycentric
coordinates or by reviewed normalized-bounds targets resolved to the nearest
surface.  The result is expressed in the final scaled asset-root frame:
+X forward, +Y up, +Z anatomical right.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import generated_asset_emitter_contract as contract  # noqa: E402


def parse_argv() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--finalization-manifest", type=Path, required=True)
    parser.add_argument("--anchor-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker-glb", type=Path, required=True)
    return parser.parse_args(argv)


def require_new_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise contract.EmitterContractError(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def scene_meshes() -> list[Any]:
    meshes = sorted(
        (item for item in bpy.context.scene.objects if item.type == "MESH"),
        key=lambda item: item.name,
    )
    if not meshes:
        raise contract.EmitterContractError("finalized static GLB has no mesh")
    if any(
        any(modifier.type == "ARMATURE" for modifier in mesh.modifiers)
        or mesh.vertex_groups
        for mesh in meshes
    ):
        raise contract.EmitterContractError("static emitter input contains rig data")
    if any(item.type == "ARMATURE" for item in bpy.context.scene.objects):
        raise contract.EmitterContractError("static emitter input contains an armature")
    if bpy.data.actions:
        raise contract.EmitterContractError("static emitter input contains animation")
    return meshes


def triangulated_world(mesh: Any) -> tuple[list[Vector], list[tuple[int, int, int]]]:
    vertices = [mesh.matrix_world @ vertex.co for vertex in mesh.data.vertices]
    triangles: list[tuple[int, int, int]] = []
    for polygon in mesh.data.polygons:
        indices = list(polygon.vertices)
        for index in range(1, len(indices) - 1):
            triangles.append((indices[0], indices[index], indices[index + 1]))
    if not triangles:
        raise contract.EmitterContractError(f"mesh has no triangles: {mesh.name}")
    return vertices, triangles


def mesh_surfaces(meshes: list[Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for mesh in meshes:
        if mesh.name in result:
            raise contract.EmitterContractError(f"duplicate mesh name: {mesh.name}")
        vertices, triangles = triangulated_world(mesh)
        result[mesh.name] = {
            "vertices": vertices,
            "triangles": triangles,
            "bvh": BVHTree.FromPolygons(vertices, triangles, all_triangles=True),
        }
    return result


def avengine_bounds(surfaces: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    points = [
        contract.blender_xyz_to_avengine_local(vertex)
        for surface in surfaces.values()
        for vertex in surface["vertices"]
    ]
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    extent = [maximum[axis] - minimum[axis] for axis in range(3)]
    if any(not math.isfinite(value) or value <= 0.0 for value in extent):
        raise contract.EmitterContractError("static mesh has degenerate bounds")
    return {"minimum_m": minimum, "maximum_m": maximum, "extent_m": extent}


def barycentric_point(
    surface: dict[str, Any],
    triangle_index: int,
    barycentric: list[float],
) -> Vector:
    triangles = surface["triangles"]
    if triangle_index >= len(triangles):
        raise contract.EmitterContractError("anchor triangle index is out of range")
    triangle = triangles[triangle_index]
    vertices = surface["vertices"]
    return sum(
        (vertices[vertex_index] * barycentric[index] for index, vertex_index in enumerate(triangle)),
        Vector((0.0, 0.0, 0.0)),
    )


def resolve_barycentric_samples(
    selection: dict[str, Any],
    surfaces: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved = []
    for sample in selection["samples"]:
        mesh_name = sample["mesh_name"]
        if mesh_name not in surfaces:
            raise contract.EmitterContractError(
                f"anchor mesh is not present: {mesh_name}"
            )
        barycentric = [float(value) for value in sample["barycentric"]]
        point = barycentric_point(
            surfaces[mesh_name],
            int(sample["triangle_index"]),
            barycentric,
        )
        resolved.append(
            {
                "mesh_name": mesh_name,
                "triangle_index": int(sample["triangle_index"]),
                "barycentric": barycentric,
                "weight": float(sample["weight"]),
                "surface_point_m": contract.blender_xyz_to_avengine_local(point),
                "reviewed_target_m": None,
                "target_to_surface_distance_m": 0.0,
            }
        )
    return resolved


def resolve_bbox_samples(
    selection: dict[str, Any],
    surfaces: dict[str, dict[str, Any]],
    bounds: dict[str, list[float]],
) -> list[dict[str, Any]]:
    diagonal = math.sqrt(sum(value * value for value in bounds["extent_m"]))
    maximum_distance = diagonal * float(
        selection["maximum_search_distance_fraction"]
    )
    resolved = []
    for sample in selection["samples"]:
        fraction = [float(value) for value in sample["target_fraction_xyz"]]
        target_avengine = [
            bounds["minimum_m"][axis] + bounds["extent_m"][axis] * fraction[axis]
            for axis in range(3)
        ]
        target_blender = Vector(
            contract.avengine_local_to_blender_xyz(target_avengine)
        )
        nearest = None
        for mesh_name, surface in surfaces.items():
            location, _normal, triangle_index, distance = surface["bvh"].find_nearest(
                target_blender
            )
            if (
                location is None
                or triangle_index is None
                or distance is None
                or not math.isfinite(float(distance))
            ):
                continue
            candidate = (
                float(distance),
                mesh_name,
                int(triangle_index),
                location.copy(),
            )
            if nearest is None or candidate[:3] < nearest[:3]:
                nearest = candidate
        if nearest is None:
            raise contract.EmitterContractError(
                "reviewed anchor target did not resolve to a mesh surface"
            )
        distance, mesh_name, triangle_index, location = nearest
        if distance > maximum_distance:
            raise contract.EmitterContractError(
                "reviewed anchor target is too far from the final mesh surface"
            )
        resolved.append(
            {
                "mesh_name": mesh_name,
                "triangle_index": triangle_index,
                "target_fraction_xyz": fraction,
                "weight": float(sample["weight"]),
                "surface_point_m": contract.blender_xyz_to_avengine_local(location),
                "reviewed_target_m": target_avengine,
                "target_to_surface_distance_m": distance,
            }
        )
    return resolved


def weighted_emitter(resolved: list[dict[str, Any]]) -> list[float]:
    total = sum(sample["weight"] for sample in resolved)
    return [
        sum(
            sample["surface_point_m"][axis] * sample["weight"]
            for sample in resolved
        )
        / total
        for axis in range(3)
    ]


def assert_inside_bounds(
    point: list[float],
    bounds: dict[str, list[float]],
) -> None:
    diagonal = math.sqrt(sum(value * value for value in bounds["extent_m"]))
    tolerance = max(diagonal * 1.0e-6, 1.0e-8)
    for axis in range(3):
        if not (
            bounds["minimum_m"][axis] - tolerance
            <= point[axis]
            <= bounds["maximum_m"][axis] + tolerance
        ):
            raise contract.EmitterContractError(
                "measured emitter lies outside finalized mesh bounds"
            )


def publish_marker_glb(
    marker_path: Path,
    point_avengine: list[float],
    bounds: dict[str, list[float]],
) -> tuple[float, dict[str, Any]]:
    diagonal = math.sqrt(sum(value * value for value in bounds["extent_m"]))
    radius = max(diagonal * 0.025, 0.002)
    point_blender = contract.avengine_local_to_blender_xyz(point_avengine)
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=12,
        radius=radius,
        location=point_blender,
    )
    marker = bpy.context.active_object
    marker.name = "AVEngine_Reviewed_Emitter_Marker"
    material = bpy.data.materials.new("AVEngine_Emitter_Marker_Red")
    material.diffuse_color = (1.0, 0.01, 0.01, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (1.0, 0.01, 0.01, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.35
    marker.data.materials.append(material)

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{marker_path.stem}.",
        suffix=".staging.glb",
        dir=marker_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.export_scene.gltf(
            filepath=str(temporary),
            export_format="GLB",
            use_selection=True,
            export_animations=False,
            export_texcoords=True,
            export_normals=True,
            export_materials="EXPORT",
            export_all_vertex_colors=True,
            export_vertex_color="ACTIVE",
        )
        os.link(temporary, marker_path)
    finally:
        temporary.unlink(missing_ok=True)
    return radius, {
        "path": str(marker_path),
        "sha256": contract.sha256_file(marker_path),
        "size_bytes": marker_path.stat().st_size,
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    args = parse_argv()
    input_glb = args.input_glb.resolve()
    finalization_path = args.finalization_manifest.resolve()
    spec_path = args.anchor_spec.resolve()
    output = require_new_file(args.output, "emitter measurement")
    marker_glb = require_new_file(args.marker_glb, "emitter marker GLB")

    finalization = contract.validate_static_finalization(
        finalization_path,
        input_glb,
    )
    spec = contract.validate_static_anchor_spec(
        spec_path,
        finalization_path=finalization_path,
        input_glb=input_glb,
        finalization=finalization,
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    meshes = scene_meshes()
    surfaces = mesh_surfaces(meshes)
    bounds = avengine_bounds(surfaces)
    selection = spec["selection"]
    if selection["method"] == "mesh_surface_barycentric_samples_v1":
        resolved = resolve_barycentric_samples(selection, surfaces)
    elif selection["method"] == "reviewed_bbox_fraction_nearest_surface_v1":
        resolved = resolve_bbox_samples(selection, surfaces, bounds)
    else:
        raise contract.EmitterContractError(
            "unsupported static emitter selection method"
        )
    emitter = weighted_emitter(resolved)
    assert_inside_bounds(emitter, bounds)
    marker_radius, marker_record = publish_marker_glb(
        marker_glb,
        emitter,
        bounds,
    )

    payload = {
        "schema": contract.MEASUREMENT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured_pending_marker_visual_review",
        "asset_class": "static_object",
        "instance_id": finalization["instance_id"],
        "request_sha256": finalization["request_sha256"],
        "profile_sha256": finalization["profile_sha256"],
        "input": {
            "path": str(input_glb),
            "sha256": contract.sha256_file(input_glb),
            "size_bytes": input_glb.stat().st_size,
        },
        "finalization_manifest": {
            "path": str(finalization_path),
            "sha256": contract.sha256_file(finalization_path),
            "size_bytes": finalization_path.stat().st_size,
        },
        "anchor_spec": {
            "path": str(spec_path),
            "sha256": contract.sha256_file(spec_path),
            "size_bytes": spec_path.stat().st_size,
        },
        "coordinate_system": contract.COORDINATE_SYSTEM,
        "asset_bounds": bounds,
        "emitter_anchor": {
            "anchor_id": spec["anchor_id"],
            "anchor_type": spec["anchor_type"],
            "semantic_role": spec["semantic_role"],
            "offset_m": emitter,
            "offset_space": "final_scaled_asset_root",
            "method": selection["method"],
            "aggregation": "weighted_centroid",
            "resolved_surface_samples": resolved,
            "asset_specific_not_class_template": True,
            "animation_required": False,
        },
        "marker_review": {
            "marker_glb": marker_record,
            "marker_radius_m": marker_radius,
            "visual_review": "pending",
        },
        "formal_dataset_registration_authorized": False,
    }
    write_json_exclusive(output, payload)
    print(
        "GENERATED_STATIC_EMITTER_OK "
        f"offset={emitter} output={output} marker={marker_glb}",
        flush=True,
    )


if __name__ == "__main__":
    main()
