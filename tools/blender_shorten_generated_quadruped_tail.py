#!/usr/bin/env python3
"""Shorten only a reviewed tail corridor on a generated quadruped GLB.

The generated GLB remains the sole geometry and PBR authority.  The operation
does not add, delete, or substitute mesh elements: it resamples tail-shaft
vertex positions along an authenticated reviewed centerline while preserving
the root band, every vertex outside the corridor, topology, UVs, materials,
and packed images.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
from typing import Any

import bmesh
import bpy
from mathutils import Vector


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.generated_quadruped_tail_shortening import (  # noqa: E402
    TailShorteningError,
    load_profile,
    map_point,
)


SCHEMA = "avengine_generated_quadruped_tail_shortening_realization_v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GLB_MAGIC = 0x46546C67
GLB_JSON_CHUNK = 0x4E4F534A


def parse_argv(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_new_output(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def stage_authenticated_input(
    path: Path,
    expected_sha256: str,
    label: str,
    staged_path: Path,
) -> tuple[Path, int]:
    path = path.absolute()
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise RuntimeError(f"invalid authenticated {label} sha256")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"missing or unsafe authenticated {label}: {path}") from error
    digest = hashlib.sha256()
    copied_size = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise RuntimeError(f"authenticated {label} is not a non-empty file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            with staged_path.open("xb") as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    destination.write(chunk)
                    copied_size += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
    finally:
        os.close(descriptor)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} sha256 mismatch: expected={expected_sha256} actual={actual}"
        )
    if copied_size != metadata.st_size:
        raise RuntimeError(f"{label} size changed while staging")
    os.chmod(staged_path, 0o400)
    return path, copied_size


def _hash_records(records) -> str:
    encoded = json.dumps(
        records,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mesh_topology_sha256(mesh) -> str:
    return _hash_records(
        {
            "vertices": len(mesh.vertices),
            "edges": [
                tuple(int(index) for index in edge.vertices)
                for edge in mesh.edges
            ],
            "polygons": [
                {
                    "vertices": tuple(int(index) for index in polygon.vertices),
                    "material_index": int(polygon.material_index),
                }
                for polygon in mesh.polygons
            ],
        }
    )


def uv_sha256(mesh) -> str:
    digest = hashlib.sha256()
    for layer in mesh.uv_layers:
        digest.update(layer.name.encode("utf-8"))
        for item in layer.data:
            digest.update(struct.pack("<2d", float(item.uv.x), float(item.uv.y)))
    return digest.hexdigest()


def material_pbr_sha256(mesh) -> str:
    records: list[dict[str, Any]] = []
    for slot in mesh.materials:
        if slot is None:
            records.append({"material": None})
            continue
        record: dict[str, Any] = {
            "material": slot.name,
            "use_nodes": bool(slot.use_nodes),
            "blend_method": getattr(slot, "surface_render_method", None),
            "backface_culling": bool(slot.use_backface_culling),
            "nodes": [],
            "links": [],
        }
        if slot.use_nodes and slot.node_tree is not None:
            for node in sorted(slot.node_tree.nodes, key=lambda item: item.name):
                node_record = {
                    "name": node.name,
                    "type": node.bl_idname,
                    "image": None,
                }
                image = getattr(node, "image", None)
                if image is not None:
                    packed = image.packed_file
                    node_record["image"] = {
                        "name": image.name,
                        "source": image.source,
                        "packed_sha256": (
                            hashlib.sha256(bytes(packed.data)).hexdigest()
                            if packed is not None
                            else None
                        ),
                        "packed_size": len(packed.data) if packed is not None else None,
                    }
                record["nodes"].append(node_record)
            record["links"] = sorted(
                (
                    link.from_node.name,
                    link.from_socket.name,
                    link.to_node.name,
                    link.to_socket.name,
                )
                for link in slot.node_tree.links
            )
        records.append(record)
    return _hash_records(records)


def topology_stats(mesh) -> dict[str, int]:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        return {
            "vertices": len(bm.verts),
            "edges": len(bm.edges),
            "faces": len(bm.faces),
            "boundary_edges": sum(len(edge.link_faces) == 1 for edge in bm.edges),
            "wire_edges": sum(len(edge.link_faces) == 0 for edge in bm.edges),
            "nonmanifold_edges_over_two_faces": sum(
                len(edge.link_faces) > 2 for edge in bm.edges
            ),
        }
    finally:
        bm.free()


def glb_document(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20:
        raise RuntimeError("exported GLB is truncated")
    magic, version, declared = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2 or declared != len(data):
        raise RuntimeError("exported GLB header is invalid")
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != GLB_JSON_CHUNK:
        raise RuntimeError("exported GLB has no leading JSON chunk")
    try:
        return json.loads(data[20 : 20 + chunk_length].rstrip(b" \t\r\n\x00"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"exported GLB JSON is invalid: {error}") from error


def pbr_readback(path: Path) -> dict[str, Any]:
    document = glb_document(path)
    meshes = document.get("meshes", [])
    materials = document.get("materials", [])
    textures = document.get("textures", [])
    images = document.get("images", [])
    if (
        len(meshes) != 1
        or not materials
        or not textures
        or not images
        or document.get("skins")
        or document.get("animations")
    ):
        raise RuntimeError(
            "tail-shortened raw GLB lost its single unrigged packed-PBR container"
        )
    primitive_count = 0
    used_materials = set()
    for mesh in meshes:
        for primitive in mesh.get("primitives", []):
            primitive_count += 1
            material_index = primitive.get("material")
            position_index = primitive.get("attributes", {}).get("POSITION")
            if (
                not isinstance(material_index, int)
                or not 0 <= material_index < len(materials)
                or not isinstance(position_index, int)
            ):
                raise RuntimeError("exported GLB primitive lost material or POSITION")
            used_materials.add(material_index)
    if primitive_count <= 0:
        raise RuntimeError("exported GLB has no mesh primitive")
    for material_index in used_materials:
        pbr = materials[material_index].get("pbrMetallicRoughness", {})
        if not isinstance(pbr.get("baseColorTexture", {}).get("index"), int):
            raise RuntimeError("exported GLB material lost base-color texture")
        if not isinstance(pbr.get("metallicRoughnessTexture", {}).get("index"), int):
            raise RuntimeError("exported GLB material lost metallic-roughness texture")
    return {
        "passed": True,
        "mesh_count": len(meshes),
        "primitive_count": primitive_count,
        "material_count": len(materials),
        "texture_count": len(textures),
        "image_count": len(images),
        "skin_count": len(document.get("skins", [])),
        "animation_count": len(document.get("animations", [])),
    }


def main(argv=None):
    args = parse_argv(argv)
    output = require_new_output(args.output_glb, "output GLB")
    manifest_path = require_new_output(args.manifest, "manifest")

    with tempfile.TemporaryDirectory(prefix="generated_tail_shortening_") as temporary:
        staging = Path(temporary)
        input_path, input_size = stage_authenticated_input(
            args.input,
            args.input_sha256,
            "generated GLB",
            staging / "source.glb",
        )
        profile_path, profile_size = stage_authenticated_input(
            args.profile,
            args.profile_sha256,
            "tail-shortening profile",
            staging / "profile.json",
        )
        try:
            raw_profile = json.loads(
                (staging / "profile.json").read_text(encoding="utf-8")
            )
            profile = load_profile(raw_profile)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TailShorteningError) as error:
            raise RuntimeError(f"invalid tail-shortening profile: {error}") from error
        source_record = raw_profile["source"]
        if (
            source_record["path"] != str(input_path)
            or source_record["sha256"] != args.input_sha256
            or source_record["size_bytes"] != input_size
        ):
            raise RuntimeError("profile source identity does not match authenticated input")

        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(staging / "source.glb"))
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
        if len(meshes) != 1 or armatures:
            raise RuntimeError(
                "tail shortening requires exactly one unrigged generated mesh"
            )
        mesh_object = meshes[0]
        mesh = mesh_object.data
        if not mesh.materials or not mesh.uv_layers:
            raise RuntimeError("generated mesh has no material or UV authority")

        topology_before = mesh_topology_sha256(mesh)
        uv_before = uv_sha256(mesh)
        material_before = material_pbr_sha256(mesh)
        stats_before = topology_stats(mesh)
        original_local = [tuple(float(value) for value in vertex.co) for vertex in mesh.vertices]
        inverse_world = mesh_object.matrix_world.inverted()

        selected = 0
        moved_indices: list[int] = []
        displacement: list[float] = []
        moved_source_world: list[tuple[float, float, float]] = []
        moved_output_world: list[tuple[float, float, float]] = []
        for vertex in mesh.vertices:
            source_world_vector = mesh_object.matrix_world @ vertex.co
            source_world = tuple(float(value) for value in source_world_vector)
            mapping = map_point(profile, source_world)
            selected += int(mapping.selected)
            if not mapping.moved:
                continue
            output_world_vector = Vector(mapping.output_world)
            vertex.co = inverse_world @ output_world_vector
            moved_indices.append(int(vertex.index))
            moved_source_world.append(mapping.source_world)
            moved_output_world.append(mapping.output_world)
            displacement.append(
                (output_world_vector - source_world_vector).length
            )
        mesh.update()

        moved = len(moved_indices)
        total_vertices = len(mesh.vertices)
        moved_fraction = moved / total_vertices if total_vertices else 1.0
        if moved < profile.minimum_moved_vertices:
            raise RuntimeError(
                f"too few tail vertices moved: {moved} < {profile.minimum_moved_vertices}"
            )
        if moved_fraction > profile.maximum_moved_vertex_fraction:
            raise RuntimeError(
                "tail edit escaped its local bound: "
                f"{moved_fraction:.6f} > {profile.maximum_moved_vertex_fraction:.6f}"
            )
        moved_set = set(moved_indices)
        for vertex, before in zip(mesh.vertices, original_local):
            if vertex.index not in moved_set and tuple(float(value) for value in vertex.co) != before:
                raise RuntimeError("a non-tail vertex position changed")

        topology_after = mesh_topology_sha256(mesh)
        uv_after = uv_sha256(mesh)
        material_after = material_pbr_sha256(mesh)
        stats_after = topology_stats(mesh)
        if (
            topology_after != topology_before
            or uv_after != uv_before
            or material_after != material_before
            or stats_after != stats_before
        ):
            raise RuntimeError("tail shortening changed topology, UV, PBR, or boundaries")

        bpy.ops.export_scene.gltf(filepath=str(output), export_format="GLB")

    readback = pbr_readback(output)
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "technical_spike_only",
        "formal_dataset_registration_authorized": False,
        "source": {
            "path": str(input_path),
            "sha256": args.input_sha256,
            "size_bytes": input_size,
            "authenticated_bytes_staged_before_blender_import": True,
        },
        "profile": {
            "path": str(profile_path),
            "sha256": args.profile_sha256,
            "size_bytes": profile_size,
            "schema": raw_profile["schema"],
        },
        "operation": {
            "method": "reviewed_centerline_free_tail_resampling_v1",
            "generated_input_geometry_authority": True,
            "template_geometry_used": False,
            "mesh_elements_added": 0,
            "mesh_elements_deleted": 0,
            "tail_root_band_preserved": True,
            "rump_surface_opening_introduced": False,
            "selected_vertices": selected,
            "moved_vertices": moved,
            "total_vertices": total_vertices,
            "moved_vertex_fraction": moved_fraction,
            "minimum_displacement_world": min(displacement),
            "maximum_displacement_world": max(displacement),
            "source_moved_bbox_world": {
                "min": [
                    min(point[axis] for point in moved_source_world)
                    for axis in range(3)
                ],
                "max": [
                    max(point[axis] for point in moved_source_world)
                    for axis in range(3)
                ],
            },
            "output_moved_bbox_world": {
                "min": [
                    min(point[axis] for point in moved_output_world)
                    for axis in range(3)
                ],
                "max": [
                    max(point[axis] for point in moved_output_world)
                    for axis in range(3)
                ],
            },
            "initial_free_tail_arc_length_world": profile.initial_free_length_world,
            "final_free_tail_arc_length_world": profile.final_free_length_world,
            "free_tail_length_ratio": profile.free_tail_length_ratio,
        },
        "authority_verification": {
            "non_tail_vertex_positions_preserved_in_memory": True,
            "mesh_topology_preserved_in_memory": topology_after == topology_before,
            "uv_preserved_in_memory": uv_after == uv_before,
            "pbr_material_and_packed_images_preserved_in_memory": (
                material_after == material_before
            ),
            "boundary_and_nonmanifold_counts_preserved_in_memory": (
                stats_after == stats_before
            ),
            "topology_sha256": topology_after,
            "uv_sha256": uv_after,
            "pbr_sha256": material_after,
            "topology_stats": stats_after,
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
            "pbr_glb_readback": readback,
        },
        "next_gate": "render_localization_then_generate_target_native_walk_idle",
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_QUADRUPED_TAIL_SHORTENING_OK "
        f"moved_vertices={moved} moved_fraction={moved_fraction:.6f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
