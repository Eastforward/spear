"""Independently prove that TokenRig only added one skin and one armature.

This gate is intentionally run in a fresh Blender process.  It imports the
unrigged TokenRig input and the emitted rig GLB separately, then compares the
visible asset surface by logical world-position vertices and by a bijective
triangle-corner signature (world geometry, winding, material slot, and every
UV layer).  Export-time vertex splitting for skin weights is allowed only when
the logical position and triangle-corner sets are unchanged.

The Blender readback is complemented by a direct GLB-container check.  The
container check proves that PBR materials, samplers, textures, and embedded
image bytes are identical, that the input has no skin, and that the output has
exactly one skin and no animation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import bpy
from mathutils.kdtree import KDTree
import numpy as np


SCHEMA = "avengine_generated_animal_tokenrig_binding_readback_v1"
# Two float32 round trips (source export -> import -> TokenRig export -> import)
# can accumulate just under two 1e-7-diagonal ULPs on these assets.  This fixed
# bound is deliberately below the pipeline's 1e-6 rigid-transform tolerance.
POSITION_TOLERANCE_RATIO = 2.0e-7
MINIMUM_POSITION_TOLERANCE = 1.0e-9
UV_ABSOLUTE_TOLERANCE = 1.0e-7
GLB_MAGIC = 0x46546C67
GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942


def parse_argv() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenrig-input", type=Path, required=True)
    parser.add_argument("--tokenrig-output", type=Path, required=True)
    parser.add_argument("--readback-output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_without(payload: dict[str, Any], field: str) -> str:
    return sha256_bytes(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    )


def require_regular_input(path: Path, label: str) -> Path:
    unresolved = path.absolute()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {unresolved}")
    if unresolved.stat().st_size <= 0:
        raise RuntimeError(f"{label} is empty: {unresolved}")
    return unresolved.resolve(strict=True)


def require_new_output(path: Path) -> Path:
    unresolved = path.absolute()
    if unresolved.exists() or unresolved.is_symlink():
        raise RuntimeError(f"refusing to replace TokenRig readback: {unresolved}")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    return unresolved


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def recursive_finite(value: Any, label: str = "payload") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            recursive_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            recursive_finite(child, f"{label}[{index}]")


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise RuntimeError(f"truncated GLB: {path}")
    magic, version, declared_length = struct.unpack_from("<III", raw, 0)
    if magic != GLB_MAGIC or version != 2 or declared_length != len(raw):
        raise RuntimeError(f"invalid GLB header: {path}")
    offset = 12
    json_chunk = None
    binary_chunk = None
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise RuntimeError(f"truncated GLB chunk header: {path}")
        length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        end = offset + length
        if end > len(raw):
            raise RuntimeError(f"truncated GLB chunk payload: {path}")
        chunk = raw[offset:end]
        offset = end
        if chunk_type == GLB_JSON_CHUNK:
            if json_chunk is not None:
                raise RuntimeError(f"multiple GLB JSON chunks: {path}")
            json_chunk = chunk
        elif chunk_type == GLB_BIN_CHUNK:
            if binary_chunk is not None:
                raise RuntimeError(f"multiple GLB BIN chunks: {path}")
            binary_chunk = chunk
    if json_chunk is None or binary_chunk is None:
        raise RuntimeError(f"GLB must contain JSON and BIN chunks: {path}")
    try:
        payload = json.loads(json_chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid GLB JSON chunk: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"GLB JSON root must be an object: {path}")
    recursive_finite(payload, f"GLB {path}")
    return payload, binary_chunk


def image_records(payload: dict[str, Any], binary: bytes) -> list[dict[str, Any]]:
    records = []
    buffer_views = payload.get("bufferViews", [])
    for index, image in enumerate(payload.get("images", [])):
        if not isinstance(image, dict):
            raise RuntimeError(f"GLB image {index} is not an object")
        if "bufferView" not in image or "uri" in image:
            raise RuntimeError("TokenRig binding requires embedded bufferView images")
        view_index = image["bufferView"]
        if (
            isinstance(view_index, bool)
            or not isinstance(view_index, int)
            or not 0 <= view_index < len(buffer_views)
        ):
            raise RuntimeError(f"invalid image bufferView index: {view_index!r}")
        view = buffer_views[view_index]
        if not isinstance(view, dict):
            raise RuntimeError("image bufferView must be an object")
        start = view.get("byteOffset", 0)
        length = view.get("byteLength")
        if (
            isinstance(start, bool)
            or isinstance(length, bool)
            or not isinstance(start, int)
            or not isinstance(length, int)
            or start < 0
            or length <= 0
            or start + length > len(binary)
        ):
            raise RuntimeError("invalid embedded image byte range")
        image_bytes = binary[start : start + length]
        records.append(
            {
                "index": index,
                "name": image.get("name"),
                "mime_type": image.get("mimeType"),
                "size_bytes": len(image_bytes),
                "sha256": sha256_bytes(image_bytes),
            }
        )
    return records


def container_snapshot(path: Path) -> dict[str, Any]:
    payload, binary = read_glb(path)
    meshes = payload.get("meshes", [])
    primitive_count = sum(
        len(mesh.get("primitives", [])) if isinstance(mesh, dict) else 0
        for mesh in meshes
    )
    pbr_payload = {
        "materials": payload.get("materials", []),
        "samplers": payload.get("samplers", []),
        "textures": payload.get("textures", []),
        "images": image_records(payload, binary),
    }
    return {
        "mesh_count": len(meshes),
        "primitive_count": primitive_count,
        "skin_count": len(payload.get("skins", [])),
        "animation_count": len(payload.get("animations", [])),
        "material_count": len(payload.get("materials", [])),
        "sampler_count": len(payload.get("samplers", [])),
        "texture_count": len(payload.get("textures", [])),
        "image_count": len(payload.get("images", [])),
        "pbr_payload": pbr_payload,
        "pbr_payload_sha256": sha256_bytes(canonical_bytes(pbr_payload)),
    }


def hidden_helper_objects() -> set[Any]:
    collection = bpy.data.collections.get("glTF_not_exported")
    return set(collection.objects) if collection is not None else set()


def linked_armatures(mesh: Any) -> set[Any]:
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


def world_vertices(mesh: Any) -> np.ndarray:
    local = np.empty((len(mesh.data.vertices), 3), dtype=np.float64)
    mesh.data.vertices.foreach_get("co", local.ravel())
    matrix = np.asarray(mesh.matrix_world, dtype=np.float64)
    result = local @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(result).all():
        raise RuntimeError("mesh contains non-finite world-space vertices")
    return result


def uv_arrays(mesh: Any) -> dict[str, np.ndarray]:
    result = {}
    for layer in mesh.data.uv_layers:
        values = np.empty((len(layer.data), 2), dtype=np.float64)
        layer.data.foreach_get("uv", values.ravel())
        if not np.isfinite(values).all():
            raise RuntimeError(f"UV layer {layer.name!r} contains non-finite values")
        result[layer.name] = values
    return result


def scene_snapshot(path: Path, *, expect_rig: bool) -> dict[str, Any]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    hidden = hidden_helper_objects()
    visible_meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj not in hidden
    ]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(visible_meshes) != 1:
        raise RuntimeError(
            "expected exactly one visible asset mesh; "
            f"got {[obj.name for obj in visible_meshes]}"
        )
    mesh = visible_meshes[0]
    linked = linked_armatures(mesh)
    if expect_rig:
        if len(armatures) != 1 or linked != {armatures[0]}:
            raise RuntimeError(
                "TokenRig output must contain exactly one linked armature"
            )
        if len(mesh.vertex_groups) == 0:
            raise RuntimeError("TokenRig output has no skin vertex groups")
        bone_names = {bone.name for bone in armatures[0].data.bones}
        group_by_index = {
            group.index: group.name
            for group in mesh.vertex_groups
            if group.name in bone_names
        }
        unweighted = 0
        for vertex in mesh.data.vertices:
            total = sum(
                float(membership.weight)
                for membership in vertex.groups
                if membership.group in group_by_index
                and math.isfinite(float(membership.weight))
                and membership.weight > 0.0
            )
            if total <= 0.0:
                unweighted += 1
        if unweighted:
            raise RuntimeError(
                f"TokenRig output has {unweighted} vertices without skin weights"
            )
    elif armatures or linked or mesh.vertex_groups:
        raise RuntimeError("TokenRig input must be unrigged and unskinned")
    if bpy.data.actions:
        raise RuntimeError("TokenRig binding artifacts must contain no animation")
    polygons = []
    for polygon in mesh.data.polygons:
        if len(polygon.vertices) != 3 or len(polygon.loop_indices) != 3:
            raise RuntimeError("TokenRig binding requires a fully triangulated mesh")
        polygons.append(
            {
                "vertices": tuple(int(index) for index in polygon.vertices),
                "loops": tuple(int(index) for index in polygon.loop_indices),
                "material_index": int(polygon.material_index),
            }
        )
    vertices = world_vertices(mesh)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    extent = maximum - minimum
    diagonal = float(np.linalg.norm(extent))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise RuntimeError("mesh has an invalid world-space bounding box")
    return {
        "mesh_name": mesh.name,
        "vertices_array": vertices,
        "polygons_internal": polygons,
        "uv_arrays_internal": uv_arrays(mesh),
        "material_names": [
            material.name if material is not None else None
            for material in mesh.data.materials
        ],
        "vertex_count": len(mesh.data.vertices),
        "edge_count": len(mesh.data.edges),
        "triangle_count": len(polygons),
        "loop_count": len(mesh.data.loops),
        "uv_layers": [layer.name for layer in mesh.data.uv_layers],
        "world_bbox_min": minimum.tolist(),
        "world_bbox_max": maximum.tolist(),
        "world_bbox_extent": extent.tolist(),
        "world_bbox_diagonal": diagonal,
        "visible_mesh_count": len(visible_meshes),
        "hidden_helper_objects": sorted(
            ({"name": obj.name, "type": obj.type} for obj in hidden),
            key=lambda item: (item["type"], item["name"]),
        ),
        "armature_count": len(armatures),
        "linked_armature_count": len(linked),
        "bone_count": sum(len(armature.data.bones) for armature in armatures),
        "skin_vertex_group_count": len(mesh.vertex_groups),
        "action_count": len(bpy.data.actions),
    }


def quantized(values: np.ndarray, tolerance: float) -> np.ndarray:
    scaled = np.rint(values / tolerance)
    if not np.isfinite(scaled).all():
        raise RuntimeError("quantization produced non-finite values")
    maximum = float(np.abs(scaled).max())
    if maximum > np.iinfo(np.int64).max:
        raise RuntimeError("quantized geometry exceeds int64 range")
    return scaled.astype(np.int64)


def cyclic_triangle_signature(corners: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    rotations = [
        tuple(corners),
        tuple(corners[1:] + corners[:1]),
        tuple(corners[2:] + corners[:2]),
    ]
    return min(rotations)


def topology_signatures(
    snapshot: dict[str, Any],
    *,
    position_class_ids: np.ndarray,
) -> tuple[Counter[Any], set[Any]]:
    uv_arrays = {
        name: quantized(values, UV_ABSOLUTE_TOLERANCE)
        for name, values in snapshot["uv_arrays_internal"].items()
    }
    logical_corner_classes = set()
    triangles: Counter[Any] = Counter()
    for polygon in snapshot["polygons_internal"]:
        corners = []
        for vertex_index, loop_index in zip(
            polygon["vertices"], polygon["loops"], strict=True
        ):
            position_class = int(position_class_ids[vertex_index])
            uvs = tuple(
                (
                    name,
                    tuple(int(value) for value in uv_arrays[name][loop_index]),
                )
                for name in snapshot["uv_layers"]
            )
            corner = (position_class, uvs)
            logical_corner_classes.add(corner)
            corners.append(corner)
        triangles[
            (
                polygon["material_index"],
                cyclic_triangle_signature(corners),
            )
        ] += 1
    return triangles, logical_corner_classes


def source_position_classes(vertices: np.ndarray) -> tuple[np.ndarray, int]:
    """Assign one stable logical class to every exact source position."""
    class_by_position: dict[tuple[float, float, float], int] = {}
    class_ids = np.empty(len(vertices), dtype=np.int64)
    for index, point in enumerate(vertices):
        key = tuple(float(value) for value in point)
        class_id = class_by_position.get(key)
        if class_id is None:
            class_id = len(class_by_position)
            class_by_position[key] = class_id
        class_ids[index] = class_id
    return class_ids, len(class_by_position)


def map_to_source_position_classes(
    source: np.ndarray,
    source_class_ids: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    tree = KDTree(len(source))
    for index, point in enumerate(source):
        tree.insert(point, index)
    tree.balance()
    target_classes = np.empty(len(target), dtype=np.int64)
    maximum = 0.0
    for target_index, point in enumerate(target):
        _nearest, source_index, distance = tree.find(point)
        target_classes[target_index] = source_class_ids[source_index]
        maximum = max(maximum, float(distance))
    return target_classes, maximum


def maximum_nearest_distance(source: np.ndarray, target: np.ndarray) -> float:
    tree = KDTree(len(target))
    for index, point in enumerate(target):
        tree.insert(point, index)
    tree.balance()
    maximum = 0.0
    for point in source:
        _nearest, _index, distance = tree.find(point)
        maximum = max(maximum, float(distance))
    return maximum


def public_scene_record(snapshot: dict[str, Any]) -> dict[str, Any]:
    omitted = {
        "vertices_array",
        "polygons_internal",
        "uv_arrays_internal",
    }
    return {key: value for key, value in snapshot.items() if key not in omitted}


def compare_binding(
    input_scene: dict[str, Any],
    output_scene: dict[str, Any],
    input_container: dict[str, Any],
    output_container: dict[str, Any],
) -> dict[str, Any]:
    diagonal = input_scene["world_bbox_diagonal"]
    position_tolerance = max(
        diagonal * POSITION_TOLERANCE_RATIO, MINIMUM_POSITION_TOLERANCE
    )
    if input_scene["uv_layers"] != output_scene["uv_layers"]:
        raise RuntimeError("TokenRig changed the UV layer inventory")
    if input_scene["material_names"] != output_scene["material_names"]:
        raise RuntimeError("TokenRig changed the visible material-slot inventory")
    input_vertices = input_scene["vertices_array"]
    output_vertices = output_scene["vertices_array"]
    input_class_ids, input_class_count = source_position_classes(input_vertices)
    output_class_ids, reverse_distance = map_to_source_position_classes(
        input_vertices, input_class_ids, output_vertices
    )
    forward_distance = maximum_nearest_distance(input_vertices, output_vertices)
    maximum_distance = max(forward_distance, reverse_distance)
    if maximum_distance > position_tolerance:
        raise RuntimeError(
            "TokenRig changed world geometry beyond the fixed tolerance: "
            f"{maximum_distance} > {position_tolerance}"
        )
    output_covered_classes = set(int(value) for value in output_class_ids)
    if output_covered_classes != set(range(input_class_count)):
        raise RuntimeError(
            "TokenRig output does not cover every logical source vertex class"
        )
    input_triangles, input_corners = topology_signatures(
        input_scene, position_class_ids=input_class_ids
    )
    output_triangles, output_corners = topology_signatures(
        output_scene, position_class_ids=output_class_ids
    )
    if input_corners != output_corners:
        raise RuntimeError("TokenRig changed logical vertex/UV corner mapping")
    if input_triangles != output_triangles:
        raise RuntimeError(
            "TokenRig changed triangle topology, winding, UVs, or material assignment"
        )
    if input_container["skin_count"] != 0:
        raise RuntimeError("TokenRig input GLB unexpectedly contains a skin")
    if output_container["skin_count"] != 1:
        raise RuntimeError("TokenRig output GLB must contain exactly one skin")
    if input_container["animation_count"] or output_container["animation_count"]:
        raise RuntimeError("TokenRig input/output GLBs must contain no animation")
    if input_container["mesh_count"] != 1 or output_container["mesh_count"] != 1:
        raise RuntimeError("TokenRig input/output GLBs must each contain one mesh")
    if (
        input_container["primitive_count"] != 1
        or output_container["primitive_count"] != 1
    ):
        raise RuntimeError(
            "TokenRig input/output GLBs must each contain one mesh primitive"
        )
    if input_container["pbr_payload_sha256"] != output_container["pbr_payload_sha256"]:
        raise RuntimeError(
            "TokenRig changed PBR materials, textures, samplers, or image bytes"
        )
    triangle_digest = sha256_bytes(
        canonical_bytes(
            [
                {"signature": repr(signature), "multiplicity": multiplicity}
                for signature, multiplicity in sorted(
                    input_triangles.items(), key=lambda item: repr(item[0])
                )
            ]
        )
    )
    return {
        "status": "passed",
        "fixed_thresholds": {
            "position_tolerance_ratio_of_input_bbox_diagonal": (
                POSITION_TOLERANCE_RATIO
            ),
            "minimum_position_tolerance": MINIMUM_POSITION_TOLERANCE,
            "effective_world_position_tolerance": position_tolerance,
            "uv_absolute_tolerance": UV_ABSOLUTE_TOLERANCE,
        },
        "world_geometry": {
            "input_to_output_maximum_nearest_vertex_distance": forward_distance,
            "output_to_input_maximum_nearest_vertex_distance": reverse_distance,
            "maximum_bidirectional_nearest_vertex_distance": maximum_distance,
            "logical_world_position_class_count": input_class_count,
            "logical_position_classes_bijective": True,
            "export_split_vertices_allowed_only_with_identical_logical_positions": True,
            "input_raw_vertex_count": input_scene["vertex_count"],
            "output_raw_vertex_count": output_scene["vertex_count"],
            "output_minus_input_raw_vertex_count": (
                output_scene["vertex_count"] - input_scene["vertex_count"]
            ),
        },
        "topology_and_uv": {
            "triangle_count": sum(input_triangles.values()),
            "triangle_correspondence": "bijective_multiset_with_winding_preserved",
            "triangle_correspondence_sha256": triangle_digest,
            "logical_vertex_uv_corner_class_count": len(input_corners),
            "logical_vertex_uv_corner_classes_bijective": True,
            "all_uv_layers_compared": input_scene["uv_layers"],
            "material_assignment_compared_per_triangle": True,
        },
        "pbr": {
            "container_payload_identical": True,
            "payload_sha256": input_container["pbr_payload_sha256"],
            "embedded_image_bytes_compared": True,
        },
        "rig": {
            "input_skin_count": input_container["skin_count"],
            "output_skin_count": output_container["skin_count"],
            "input_armature_count": input_scene["armature_count"],
            "output_armature_count": output_scene["armature_count"],
            "output_linked_armature_count": output_scene["linked_armature_count"],
            "output_bone_count": output_scene["bone_count"],
        },
        "animation": {
            "input_container_animation_count": input_container["animation_count"],
            "output_container_animation_count": output_container["animation_count"],
            "input_blender_action_count": input_scene["action_count"],
            "output_blender_action_count": output_scene["action_count"],
        },
    }


def main() -> int:
    args = parse_argv()
    source = require_regular_input(args.tokenrig_input, "TokenRig input")
    rig = require_regular_input(args.tokenrig_output, "TokenRig output")
    if source == rig:
        raise RuntimeError("TokenRig input and output must be different files")
    output = require_new_output(args.readback_output)
    input_container = container_snapshot(source)
    output_container = container_snapshot(rig)
    input_scene = scene_snapshot(source, expect_rig=False)
    output_scene = scene_snapshot(rig, expect_rig=True)
    comparison = compare_binding(
        input_scene, output_scene, input_container, output_container
    )
    blender_build_hash = bpy.app.build_hash
    if isinstance(blender_build_hash, bytes):
        blender_build_hash = blender_build_hash.decode("ascii", errors="replace")
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": {
            "script": file_record(Path(__file__).resolve()),
            "blender_version": bpy.app.version_string,
            "blender_build_hash": str(blender_build_hash),
        },
        "tokenrig_input": file_record(source),
        "tokenrig_output": file_record(rig),
        "input_scene": public_scene_record(input_scene),
        "output_scene": public_scene_record(output_scene),
        "input_container": input_container,
        "output_container": output_container,
        "comparison": comparison,
        "automatic_checks": {
            "overall": "passed",
            "world_geometry_unchanged": True,
            "logical_vertices_bijective": True,
            "triangles_bijective": True,
            "uvs_unchanged": True,
            "pbr_unchanged": True,
            "exactly_one_output_skin": True,
            "exactly_one_output_armature": True,
            "no_animation": True,
        },
        "claim_boundary": (
            "This proves source-to-TokenRig surface preservation and the presence "
            "of one static skin/armature. It does not qualify semantic rigging, "
            "motion, UE behavior, emitter placement, or dataset registration."
        ),
        "formal_dataset_registration_authorized": False,
    }
    recursive_finite(payload)
    payload["manifest_sha256"] = hash_without(payload, "manifest_sha256")
    output.write_bytes(canonical_bytes(payload) + b"\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "readback": str(output),
                "sha256": sha256_file(output),
                "manifest_sha256": payload["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
