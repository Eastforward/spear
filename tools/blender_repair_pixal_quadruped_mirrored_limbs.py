"""Repair two missing far-side limb chains without changing Pixel3D identity.

This is a bounded research repair, not a generic mirroring operator.  The
owner-approved image, Pixal manifest, rejected raw GLB, owner review, and
formal rejection are reauthenticated before Blender mutates a copy.  Only two
normalized near-side limb corridors are copied across the sagittal plane.
The tail is segmented from the authenticated source surface without a height
cut, so a low/downward tail is immutable too.  No donor mesh, skeleton, weight,
material, texture, animation, or rig is accepted by the CLI.

Whole-animal voxel remesh, smoothing, and decimation are forbidden here:
those operations silently replace Pixel3D body, head, tail, UV, and material
identity outside the authorized corridors.  This implementation retains every
source triangle that is not wholly mutable, mirrors only wholly-contained
source-side triangles, and attempts only a corridor-local exact weld.  It
fails closed before export unless the source topology outside the corridors is
already closed and the local result is one watertight component.  Export is
then re-imported and exact float32 POSITION/TEXCOORD_0/material triangle
signatures, tail surface, embedded image bytes, and PBR bindings are compared
against the authenticated source.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

import bmesh
import bpy
import numpy as np


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.generated_quadruped_geometry_repair import (  # noqa: E402
    IMPLEMENTATION_CONTRACT,
    SCHEMA,
    RepairSpec,
    audit_expected_surface_signatures,
    audit_immutable_source_topology,
    audit_low_slice_limb_chains,
    canonical_triangle_surface_signatures,
    estimate_sagittal_plane,
    identify_tail_surface,
    limb_corridor_mask,
    mask_record,
    mirror_points_across_y_with_attachment_taper,
    normalized_points,
    repair_vertex_masks,
    robust_bounds,
)


def stage(label: str, started_at: float | None = None, **values: Any) -> float:
    suffix = " ".join(f"{key}={value}" for key, value in values.items())
    if started_at is not None:
        suffix = f"elapsed_s={time.perf_counter() - started_at:.2f} {suffix}".strip()
    print(f"PIXAL_LIMB_REPAIR_STAGE {label} {suffix}".rstrip(), flush=True)
    return time.perf_counter()


def parse_argv() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pixal-manifest", type=Path, required=True)
    parser.add_argument("--approved-reference", type=Path, required=True)
    parser.add_argument("--owner-review", type=Path, required=True)
    parser.add_argument("--static-decision", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-side",
        choices=("positive-y", "negative-y"),
        default="positive-y",
    )
    parser.add_argument(
        "--head-direction",
        choices=("positive-x", "negative-x"),
        default="negative-x",
    )
    parser.add_argument("--front-foot-x-fraction", type=float, default=0.27)
    parser.add_argument("--front-attachment-x-fraction", type=float, default=0.36)
    parser.add_argument("--hind-foot-x-fraction", type=float, default=0.83)
    parser.add_argument("--hind-attachment-x-fraction", type=float, default=0.68)
    parser.add_argument("--attachment-height-fraction", type=float, default=0.58)
    parser.add_argument("--foot-half-width-fraction", type=float, default=0.065)
    parser.add_argument(
        "--attachment-half-width-fraction",
        type=float,
        default=0.12,
    )
    parser.add_argument("--source-side-guard-fraction", type=float, default=0.01)
    parser.add_argument(
        "--central-attachment-bridge-start-fraction",
        type=float,
        default=0.38,
    )
    parser.add_argument(
        "--mirrored-attachment-taper-start-fraction",
        type=float,
        default=0.34,
    )
    parser.add_argument(
        "--mirrored-attachment-top-lateral-scale",
        type=float,
        default=0.35,
    )
    parser.add_argument("--tail-protection-x-fraction", type=float, default=0.78)
    parser.add_argument(
        "--tail-protection-height-fraction",
        type=float,
        default=0.38,
        help=(
            "Legacy evidence value only. Tail selection is height-independent "
            "and this value never limits the protected source surface."
        ),
    )
    parser.add_argument("--low-slice-height-fraction", type=float, default=0.20)
    parser.add_argument(
        "--local-weld-distance-fraction",
        type=float,
        default=1.0e-6,
        help=(
            "Maximum corridor-local weld distance as a source-bbox diagonal "
            "fraction. No global remesh, smoothing, or decimation is allowed."
        ),
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe {label}: {path}")
    return path


def require_output(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return value


def validate_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise SystemExit(f"{label} must be a lowercase SHA-256")
    return normalized


def authenticate_lineage(
    source: Path,
    pixal_manifest_path: Path,
    reference: Path,
    owner_review_path: Path,
    static_decision_path: Path,
    expected_source_sha256: str,
    expected_reference_sha256: str,
) -> dict[str, Any]:
    source_record = file_record(source)
    reference_record = file_record(reference)
    if source_record["sha256"] != validate_sha256(
        expected_source_sha256,
        "--expected-source-sha256",
    ):
        raise SystemExit("Pixal source SHA-256 changed")
    if reference_record["sha256"] != validate_sha256(
        expected_reference_sha256,
        "--expected-reference-sha256",
    ):
        raise SystemExit("owner-approved reference SHA-256 changed")

    pixal = load_json(pixal_manifest_path, "Pixal manifest")
    owner = load_json(owner_review_path, "owner review")
    decision = load_json(static_decision_path, "static decision")
    controlled_request = pixal.get("controlled_request", {})
    instance_id = controlled_request.get("instance_id")
    pixal_output = pixal.get("output", {})
    if (
        pixal.get("backend") != "pixal3d"
        or controlled_request.get("route") != "flux2_pixal3d_animal_v1"
        or pixal_output.get("sha256") != source_record["sha256"]
        or pixal_output.get("bytes") != source_record["size_bytes"]
        or pixal.get("one_shot_execution", {}).get("invocations_allowed") != 1
        or pixal.get("one_shot_execution", {}).get("seed_retry_allowed") is not False
    ):
        raise SystemExit("Pixal manifest/source/one-shot contract changed")
    owner_hard_gates = owner.get("checks", {}).get("hard_gates", {})
    if (
        owner.get("schema") != "avengine_controlled_animal_2d_review_v1"
        or owner.get("decision") != "approved_for_pixal3d"
        or owner.get("instance_id") != instance_id
        or owner.get("candidate", {}).get("sha256") != reference_record["sha256"]
        or owner.get("candidate", {}).get("size_bytes") != reference_record["size_bytes"]
        or owner_hard_gates.get("species_correct_tail") != "passed"
        or owner_hard_gates.get("anatomically_connected_limbs") != "passed"
    ):
        raise SystemExit("owner-approved 2D lineage changed")
    checks = decision.get("checks", {})
    if (
        decision.get("schema") != "avengine_controlled_animal_static_decision_v1"
        or decision.get("decision") != "rejected"
        or decision.get("state_classification") != "rejected"
        or decision.get("formal_dataset_registration_authorized") is not False
        or decision.get("instance_id") != instance_id
        or checks.get("four_limbs_usable") is not False
        or checks.get("pose_riggable") is not False
    ):
        raise SystemExit("formal raw-Pixal rejection contract changed")
    return {
        "instance_id": instance_id,
        "approved_reference": reference_record,
        "owner_review": file_record(owner_review_path),
        "owner_review_decision": owner["decision"],
        "owner_single_tail_gate": owner_hard_gates["species_correct_tail"],
        "pixal_manifest": file_record(pixal_manifest_path),
        "pixal_source": source_record,
        "static_decision": file_record(static_decision_path),
        "static_decision_state": decision["decision"],
        "raw_four_limbs_usable": checks["four_limbs_usable"],
        "raw_pose_riggable": checks["pose_riggable"],
    }


def real_meshes() -> list[bpy.types.Object]:
    return [
        item
        for item in bpy.context.scene.objects
        if item.type == "MESH" and len(item.data.polygons) > 0
    ]


def activate(obj: bpy.types.Object) -> None:
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_transforms(obj: bpy.types.Object) -> None:
    activate(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def copy_mesh_object(source: bpy.types.Object, name: str) -> bpy.types.Object:
    copied = source.copy()
    copied.data = source.data.copy()
    copied.name = name
    bpy.context.collection.objects.link(copied)
    return copied


def mesh_surface_arrays(
    obj: bpy.types.Object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [tuple(vertex.co) for vertex in obj.data.vertices],
        dtype=np.float64,
    )
    if any(len(polygon.vertices) != 3 for polygon in obj.data.polygons):
        raise RuntimeError("bounded repair requires a triangulated Pixel3D source")
    triangles = np.asarray(
        [
            tuple(int(value) for value in polygon.vertices)
            for polygon in obj.data.polygons
        ],
        dtype=np.int64,
    )
    if not obj.data.uv_layers or obj.data.uv_layers.active is None:
        raise RuntimeError("bounded repair requires source TEXCOORD_0")
    uv_data = obj.data.uv_layers.active.data
    corner_uvs = np.asarray(
        [
            [tuple(uv_data[loop_index].uv) for loop_index in polygon.loop_indices]
            for polygon in obj.data.polygons
        ],
        dtype=np.float64,
    )
    material_indices = np.asarray(
        [int(polygon.material_index) for polygon in obj.data.polygons],
        dtype=np.int64,
    )
    return points, triangles, corner_uvs, material_indices


def face_scope_masks(
    triangles: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    tail_faces = np.any(masks["tail_protected"][triangles], axis=1)
    replace_faces = (
        np.all(masks["replace_corridor"][triangles], axis=1)
        & ~tail_faces
    )
    donor_faces = (
        np.all(masks["donor"][triangles], axis=1)
        & ~tail_faces
    )
    return {
        "tail_faces": tail_faces,
        "replace_faces": replace_faces,
        "donor_faces": donor_faces,
        "immutable_faces": ~replace_faces,
    }


def retain_faces(obj: bpy.types.Object, retain: np.ndarray) -> dict[str, int]:
    if len(retain) != len(obj.data.polygons):
        raise RuntimeError("face mask does not match mesh")
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        delete_faces = [
            face for face in bm.faces if not bool(retain[face.index])
        ]
        bmesh.ops.delete(bm, geom=delete_faces, context="FACES")
        loose_vertices = [vertex for vertex in bm.verts if not vertex.link_faces]
        if loose_vertices:
            bmesh.ops.delete(bm, geom=loose_vertices, context="VERTS")
        bm.to_mesh(obj.data)
        obj.data.update()
        return {
            "faces_removed": int(len(delete_faces)),
            "loose_vertices_removed": int(len(loose_vertices)),
        }
    finally:
        bm.free()


def mirror_mesh_across_y(
    obj: bpy.types.Object,
    plane_y: float,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    spec: RepairSpec,
) -> dict[str, Any]:
    points = np.asarray(
        [tuple(vertex.co) for vertex in obj.data.vertices],
        dtype=np.float64,
    )
    mirrored, record = mirror_points_across_y_with_attachment_taper(
        points,
        plane_y,
        source_lower,
        source_upper,
        spec,
    )
    for vertex, point in zip(obj.data.vertices, mirrored):
        vertex.co = tuple(float(value) for value in point)
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
        bm.to_mesh(obj.data)
        obj.data.update()
    finally:
        bm.free()
    return record


def join_meshes(objects: list[bpy.types.Object], name: str) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = objects[0]
    joined.name = name
    return joined


def local_weld(
    obj: bpy.types.Object,
    *,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    spec: RepairSpec,
    protected_position_keys: set[bytes],
    distance: float,
) -> dict[str, Any]:
    """Merge only corridor vertices; source tail positions are never selected."""
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        points = np.asarray(
            [tuple(vertex.co) for vertex in bm.verts],
            dtype=np.float64,
        )
        corridor = limb_corridor_mask(
            points,
            source_lower,
            source_upper,
            spec,
        )
        selected = []
        protected_overlap = 0
        for vertex, in_corridor, point in zip(bm.verts, corridor, points):
            if not in_corridor:
                continue
            key = np.asarray(point, dtype="<f4").tobytes()
            if key in protected_position_keys:
                protected_overlap += 1
                continue
            selected.append(vertex)
        before_vertices = len(bm.verts)
        if selected:
            bmesh.ops.remove_doubles(
                bm,
                verts=selected,
                dist=float(distance),
            )
        after_vertices = len(bm.verts)
        bm.to_mesh(obj.data)
        obj.data.update()
        return {
            "method": "corridor_selected_remove_doubles_only",
            "distance": float(distance),
            "selected_vertex_count": int(len(selected)),
            "protected_tail_vertices_excluded": int(protected_overlap),
            "merged_vertex_count": int(before_vertices - after_vertices),
            "whole_animal_remesh_used": False,
            "smoothing_used": False,
            "decimation_used": False,
        }
    finally:
        bm.free()


def topology_stats(obj: bpy.types.Object) -> dict[str, int]:
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        boundary = 0
        wire = 0
        over_two = 0
        noncontiguous = 0
        for edge in bm.edges:
            linked = len(edge.link_faces)
            if linked == 0:
                wire += 1
            elif linked == 1:
                boundary += 1
            elif linked > 2:
                over_two += 1
            elif not edge.is_contiguous:
                noncontiguous += 1
        return {
            "vertices": len(bm.verts),
            "edges": len(bm.edges),
            "faces": len(bm.faces),
            "boundary_edges": boundary,
            "wire_edges": wire,
            "nonmanifold_edges_over_two_faces": over_two,
            "noncontiguous_two_face_edges": noncontiguous,
        }
    finally:
        bm.free()


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise RuntimeError(f"not a GLB container: {path}")
    _magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if version != 2 or declared_length != len(payload):
        raise RuntimeError(f"invalid GLB header: {path}")
    offset = 12
    document = None
    binary = b""
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise RuntimeError(f"truncated GLB chunk header: {path}")
        length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + length]
        if len(chunk) != length:
            raise RuntimeError(f"truncated GLB chunk: {path}")
        offset += length
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.rstrip(b" \x00").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
    if not isinstance(document, dict):
        raise RuntimeError(f"GLB has no JSON object: {path}")
    return document, binary


def write_glb(path: Path, document: dict[str, Any], binary: bytes) -> None:
    json_bytes = json.dumps(
        document,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary = bytes(binary) + b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_bytes)
    if binary:
        total += 8 + len(binary)
    temporary = path.with_name(f".{path.name}.pbr-contract.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"refusing stale GLB rewrite temporary: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total))
        stream.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        stream.write(json_bytes)
        if binary:
            stream.write(struct.pack("<II", len(binary), 0x004E4942))
            stream.write(binary)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def embedded_images(
    document: dict[str, Any],
    binary: bytes,
) -> list[dict[str, Any]]:
    result = []
    buffer_views = document.get("bufferViews", [])
    for index, image in enumerate(document.get("images", [])):
        if not isinstance(image, dict) or "bufferView" not in image:
            raise RuntimeError("bounded repair requires embedded GLB images")
        view_index = image["bufferView"]
        if (
            not isinstance(view_index, int)
            or view_index < 0
            or view_index >= len(buffer_views)
        ):
            raise RuntimeError("GLB image bufferView is invalid")
        view = buffer_views[view_index]
        start = int(view.get("byteOffset", 0))
        length = int(view["byteLength"])
        content = binary[start : start + length]
        if len(content) != length:
            raise RuntimeError("GLB embedded image is truncated")
        result.append(
            {
                "index": index,
                "buffer_view": view_index,
                "mime_type": image.get("mimeType"),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
    return result


def pbr_contract(
    document: dict[str, Any],
    binary: bytes,
) -> dict[str, Any]:
    images = embedded_images(document, binary)
    payload = {
        "materials": document.get("materials", []),
        "textures": document.get("textures", []),
        "samplers_present": "samplers" in document,
        "samplers": document.get("samplers", []),
        "extensions_used": document.get("extensionsUsed", []),
        "extensions_required": document.get("extensionsRequired", []),
        "images": [
            {
                "mime_type": image["mime_type"],
                "sha256": image["sha256"],
                "size_bytes": image["size_bytes"],
            }
            for image in images
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "embedded_image_bytes_compared": True,
    }


def restore_source_pbr_contract(source: Path, output: Path) -> dict[str, Any]:
    """Restore exact source PBR bindings around Blender's geometry export."""
    source_document, source_binary = read_glb(source)
    output_document, output_binary = read_glb(output)
    source_images = embedded_images(source_document, source_binary)
    output_images = embedded_images(output_document, output_binary)
    if Counter(
        (item["sha256"], item["mime_type"], item["size_bytes"])
        for item in source_images
    ) != Counter(
        (item["sha256"], item["mime_type"], item["size_bytes"])
        for item in output_images
    ):
        raise RuntimeError("export changed embedded Pixel3D image bytes")

    unused_output = set(range(len(output_images)))
    restored_images = []
    for source_index, source_image in enumerate(source_images):
        matches = [
            output_index
            for output_index in unused_output
            if (
                output_images[output_index]["sha256"]
                == source_image["sha256"]
                and output_images[output_index]["mime_type"]
                == source_image["mime_type"]
                and output_images[output_index]["size_bytes"]
                == source_image["size_bytes"]
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "embedded Pixel3D image cannot be mapped bijectively after export"
            )
        output_index = matches[0]
        unused_output.remove(output_index)
        metadata = copy.deepcopy(source_document["images"][source_index])
        metadata["bufferView"] = output_images[output_index]["buffer_view"]
        metadata.pop("uri", None)
        restored_images.append(metadata)
    output_document["images"] = restored_images
    for key in ("materials", "textures"):
        output_document[key] = copy.deepcopy(source_document.get(key, []))
    for key in ("samplers", "extensionsUsed", "extensionsRequired"):
        if key in source_document:
            output_document[key] = copy.deepcopy(source_document[key])
        else:
            output_document.pop(key, None)
    write_glb(output, output_document, output_binary)

    source_contract = pbr_contract(source_document, source_binary)
    restored_document, restored_binary = read_glb(output)
    restored_contract = pbr_contract(restored_document, restored_binary)
    passed = (
        source_contract["payload_sha256"]
        == restored_contract["payload_sha256"]
    )
    return {
        "method": "source_glb_pbr_bindings_and_embedded_bytes_restored_exactly",
        "source_payload_sha256": source_contract["payload_sha256"],
        "output_payload_sha256": restored_contract["payload_sha256"],
        "embedded_image_sha256s": [
            item["sha256"] for item in source_images
        ],
        "passed": passed,
    }


def envelope_audit(
    points: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
) -> dict[str, Any]:
    normalized = normalized_points(points, source_lower, source_upper)
    minimum = normalized.min(axis=0)
    maximum = normalized.max(axis=0)
    passed = bool(np.all(minimum >= -0.06) and np.all(maximum <= 1.06))
    return {
        "normalized_bbox_min": [float(value) for value in minimum],
        "normalized_bbox_max": [float(value) for value in maximum],
        "allowed_range": [-0.06, 1.06],
        "passed": passed,
    }


def export_static(obj: bpy.types.Object, output: Path) -> None:
    activate(obj)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_all_vertex_colors=True,
        export_vertex_color="ACTIVE",
    )


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def import_exported_surface(
    path: Path,
) -> tuple[
    bpy.types.Object,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [
        item
        for item in bpy.context.scene.objects
        if item not in before
        and item.type == "MESH"
        and len(item.data.polygons) > 0
    ]
    if len(imported) != 1:
        raise RuntimeError(
            f"export readback expected one mesh, got {[item.name for item in imported]}"
        )
    obj = imported[0]
    apply_transforms(obj)
    return (obj, *mesh_surface_arrays(obj))


def rejected_payload(
    common: dict[str, Any],
    *,
    checks: dict[str, bool],
    status: str,
    rejection_reasons: list[str],
    output: Path | None = None,
) -> dict[str, Any]:
    payload = copy.deepcopy(common)
    payload["checks"] = checks
    payload["decision"] = {
        "status": status,
        "rejection_reasons": rejection_reasons,
        "cat_semantic_retarget_authorized": False,
        "next_gate": "stop_preserve_failed_repair_evidence",
    }
    payload["output"] = file_record(output) if output is not None else None
    payload["formal_dataset_registration_authorized"] = False
    return payload


def main() -> None:
    args = parse_argv()
    source_path = require_input(args.source, "Pixal source")
    pixal_manifest_path = require_input(args.pixal_manifest, "Pixal manifest")
    reference_path = require_input(args.approved_reference, "approved reference")
    owner_review_path = require_input(args.owner_review, "owner review")
    static_decision_path = require_input(args.static_decision, "static decision")
    output = require_output(args.output, "repair output GLB")
    manifest = require_output(args.manifest, "repair manifest")
    if not 0.0 < args.local_weld_distance_fraction <= 1.0e-3:
        raise SystemExit(
            "--local-weld-distance-fraction must be in (0, 0.001]"
        )
    spec = RepairSpec(
        source_side=args.source_side,
        head_direction=args.head_direction,
        front_foot_x_fraction=args.front_foot_x_fraction,
        front_attachment_x_fraction=args.front_attachment_x_fraction,
        hind_foot_x_fraction=args.hind_foot_x_fraction,
        hind_attachment_x_fraction=args.hind_attachment_x_fraction,
        attachment_height_fraction=args.attachment_height_fraction,
        foot_half_width_fraction=args.foot_half_width_fraction,
        attachment_half_width_fraction=args.attachment_half_width_fraction,
        source_side_guard_fraction=args.source_side_guard_fraction,
        central_attachment_bridge_start_fraction=(
            args.central_attachment_bridge_start_fraction
        ),
        mirrored_attachment_taper_start_fraction=(
            args.mirrored_attachment_taper_start_fraction
        ),
        mirrored_attachment_top_lateral_scale=(
            args.mirrored_attachment_top_lateral_scale
        ),
        tail_protection_x_fraction=args.tail_protection_x_fraction,
        tail_protection_height_fraction=args.tail_protection_height_fraction,
        low_slice_height_fraction=args.low_slice_height_fraction,
    )
    spec.validate()
    lineage = authenticate_lineage(
        source_path,
        pixal_manifest_path,
        reference_path,
        owner_review_path,
        static_decision_path,
        args.expected_source_sha256,
        args.expected_reference_sha256,
    )

    timer = stage("import_start", source=source_path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source_path))
    meshes = real_meshes()
    if len(meshes) != 1:
        raise RuntimeError(
            f"expected exactly one Pixal mesh, got {[item.name for item in meshes]}"
        )
    source = meshes[0]
    apply_transforms(source)
    source.name = "Immutable_Pixal_Geometry_And_Appearance_Authority"
    (
        source_points,
        source_triangles,
        source_uvs,
        source_material_indices,
    ) = mesh_surface_arrays(source)
    source_edges = [
        (int(edge.vertices[0]), int(edge.vertices[1]))
        for edge in source.data.edges
    ]
    if len(source.data.materials) < 1:
        raise RuntimeError("bounded repair requires Pixel3D PBR material slots")
    source_pbr = pbr_contract(*read_glb(source_path))
    source_lower, source_upper = robust_bounds(source_points)
    source_extent = source_upper - source_lower
    if not (
        source_extent[0] > 2.0 * source_extent[1]
        and source_extent[2] > 1.5 * source_extent[1]
    ):
        raise RuntimeError(
            "expected Blender X-longitudinal/Y-lateral/Z-up quadruped bounds"
        )
    sagittal_plane_y, plane_record = estimate_sagittal_plane(
        source_points,
        source_lower,
        source_upper,
    )
    tail_surface, tail_source_record = identify_tail_surface(
        source_points,
        source_edges,
        source_lower,
        source_upper,
        spec,
    )
    if not tail_source_record["passed"]:
        raise RuntimeError(
            f"authenticated source tail segmentation failed: {tail_source_record}"
        )
    masks = repair_vertex_masks(
        source_points,
        source_lower,
        source_upper,
        sagittal_plane_y,
        spec,
        tail_surface=tail_surface,
    )
    normalized = normalized_points(source_points, source_lower, source_upper)
    masks_record = mask_record(source_points, normalized, masks, spec)
    if not masks_record["passed"]:
        raise RuntimeError(f"bounded same-mesh donor mask failed: {masks_record}")
    timer = stage(
        "import_and_mask_done",
        timer,
        vertices=len(source_points),
        plane_y=f"{sagittal_plane_y:.8f}",
        donor_vertices=masks_record["donor"]["vertex_count"],
    )

    face_scopes = face_scope_masks(source_triangles, masks)
    source_topology_preflight = audit_immutable_source_topology(
        source_points,
        source_triangles,
        masks["replace_corridor"],
    )
    common: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_contract": IMPLEMENTATION_CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lineage": lineage,
        "repair_spec": spec.record(),
        "coordinate_contract": {
            "blender_axes": {
                "longitudinal": "x",
                "lateral": "y",
                "up": "positive-z",
            },
            "source_side": spec.source_side,
            "head_direction": spec.head_direction,
            "robust_bbox_min": [float(value) for value in source_lower],
            "robust_bbox_max": [float(value) for value in source_upper],
            "sagittal_plane": plane_record,
        },
        "source_pbr_contract": {
            "payload_sha256": source_pbr["payload_sha256"],
            "embedded_image_bytes_compared": True,
        },
        "tail_source_surface": tail_source_record,
        "mask_audit": masks_record,
        "face_scope": {
            "source_triangle_count": int(len(source_triangles)),
            "mutable_triangle_count": int(
                np.count_nonzero(face_scopes["replace_faces"])
            ),
            "donor_triangle_count": int(
                np.count_nonzero(face_scopes["donor_faces"])
            ),
            "immutable_triangle_count": int(
                np.count_nonzero(face_scopes["immutable_faces"])
            ),
            "tail_triangle_count": int(
                np.count_nonzero(face_scopes["tail_faces"])
            ),
            "mutation_rule": (
                "only_triangles_wholly_inside_limb_corridor_and_not_on_"
                "authenticated_tail_surface"
            ),
        },
        "source_topology_preflight": source_topology_preflight,
        "mutation": {
            "mirrored_geometry_source": "same_authenticated_pixal_mesh_only",
            "external_geometry_inputs": [],
            "external_skeleton_inputs": [],
            "external_weight_inputs": [],
            "external_material_inputs": [],
            "external_texture_inputs": [],
            "animation_inputs": [],
            "tail_geometry_selected_for_mirroring": False,
            "whole_animal_voxel_remesh": False,
            "whole_animal_smoothing": False,
            "whole_animal_decimation": False,
        },
    }
    preflight_checks = {
        "authenticated_same_pixal_mesh_only": True,
        "near_side_front_and_hind_donor_masks_passed": masks_record["passed"],
        "height_independent_source_tail_identified": tail_source_record["passed"],
        "tail_region_never_selected_for_replacement_or_mirroring": (
            masks_record["tail_protected_donor_overlap_vertex_count"] == 0
            and masks_record["tail_protected_replacement_overlap_vertex_count"] == 0
        ),
        "source_topology_outside_corridors_already_closed": (
            source_topology_preflight["passed"]
        ),
    }
    if not all(preflight_checks.values()):
        reasons = [
            label for label, passed in preflight_checks.items() if not passed
        ]
        payload = rejected_payload(
            common,
            checks=preflight_checks,
            status="rejected_bounded_local_repair_preflight",
            rejection_reasons=reasons,
        )
        write_manifest(manifest, payload)
        print(
            "PIXAL_SAME_MESH_LIMB_REPAIR_REJECTED "
            f"stage=preflight reasons={','.join(reasons)} manifest={manifest}",
            flush=True,
        )
        raise SystemExit(2)

    immutable_signatures = canonical_triangle_surface_signatures(
        source_points,
        source_triangles,
        source_uvs,
        source_material_indices,
        selected_faces=face_scopes["immutable_faces"],
    )
    tail_signatures = canonical_triangle_surface_signatures(
        source_points,
        source_triangles,
        source_uvs,
        source_material_indices,
        selected_faces=face_scopes["tail_faces"],
    )
    base = copy_mesh_object(source, "Pixal_Base_Outside_Limb_Corridors")
    donor = copy_mesh_object(source, "Pixal_Near_Side_Limb_Donor")
    base_edit = retain_faces(base, ~face_scopes["replace_faces"])
    donor_edit = retain_faces(donor, face_scopes["donor_faces"])
    mirrored = copy_mesh_object(donor, "Mirrored_Same_Pixal_Limb_Donor")
    mirror_attachment = mirror_mesh_across_y(
        mirrored,
        sagittal_plane_y,
        source_lower,
        source_upper,
        spec,
    )
    composite = join_meshes(
        [base, donor, mirrored],
        "Same_Pixal_Mirrored_Limb_Composite",
    )
    protected_position_keys = {
        np.asarray(point, dtype="<f4").tobytes()
        for point in source_points[tail_surface]
    }
    weld_distance = (
        float(np.linalg.norm(source_extent))
        * args.local_weld_distance_fraction
    )
    weld_record = local_weld(
        composite,
        source_lower=source_lower,
        source_upper=source_upper,
        spec=spec,
        protected_position_keys=protected_position_keys,
        distance=weld_distance,
    )
    composite_topology_raw = topology_stats(composite)
    timer = stage(
        "same_mesh_local_composite_done",
        timer,
        base_faces_removed=base_edit["faces_removed"],
        donor_faces_removed=donor_edit["faces_removed"],
        locally_welded_vertices=weld_record["merged_vertex_count"],
        faces=composite_topology_raw["faces"],
    )

    (
        final_points,
        final_triangles,
        _final_uvs,
        _final_material_indices,
    ) = mesh_surface_arrays(composite)
    final_edges = [
        (int(edge.vertices[0]), int(edge.vertices[1]))
        for edge in composite.data.edges
    ]
    final_logical_topology = audit_immutable_source_topology(
        final_points,
        final_triangles,
        np.zeros(len(final_points), dtype=bool),
    )
    low_slice = audit_low_slice_limb_chains(
        final_points,
        final_edges,
        sagittal_plane_y,
        spec,
    )
    output_envelope = envelope_audit(
        final_points,
        source_lower,
        source_upper,
    )
    local_checks = {
        **preflight_checks,
        "four_independent_low_limb_chains": low_slice["passed"],
        "no_low_cross_limb_membrane": low_slice["checks"][
            "no_low_cross_limb_membrane"
        ],
        "one_connected_output_component": (
            final_logical_topology["logical_component_count"] == 1
        ),
        "watertight_manifold_topology": final_logical_topology["passed"],
        "output_within_authenticated_source_envelope": output_envelope["passed"],
    }
    common["mutation"].update(
        {
            "base_edit": base_edit,
            "donor_edit": donor_edit,
            "mirrored_attachment": mirror_attachment,
            "local_weld": weld_record,
        }
    )
    common["topology"] = {
        "raw_blender_after_local_weld": composite_topology_raw,
        "exact_position_logical_after_local_weld": final_logical_topology,
    }
    common["low_slice_dynamic_geometry_gate"] = low_slice
    common["output_envelope"] = output_envelope
    if not all(local_checks.values()):
        reasons = [
            label for label, passed in local_checks.items() if not passed
        ]
        payload = rejected_payload(
            common,
            checks=local_checks,
            status="rejected_unsafe_or_incomplete_local_weld",
            rejection_reasons=reasons,
        )
        write_manifest(manifest, payload)
        print(
            "PIXAL_SAME_MESH_LIMB_REPAIR_REJECTED "
            f"stage=local_weld reasons={','.join(reasons)} manifest={manifest}",
            flush=True,
        )
        raise SystemExit(2)

    source.hide_render = True
    source.hide_viewport = True
    timer = stage("export_start", timer)
    export_static(composite, output)
    pbr_readback = restore_source_pbr_contract(source_path, output)
    (
        _readback_obj,
        readback_points,
        readback_triangles,
        readback_uvs,
        readback_material_indices,
    ) = import_exported_surface(output)
    readback_signatures = canonical_triangle_surface_signatures(
        readback_points,
        readback_triangles,
        readback_uvs,
        readback_material_indices,
    )
    immutable_readback = audit_expected_surface_signatures(
        immutable_signatures,
        readback_signatures,
    )
    tail_readback = audit_expected_surface_signatures(
        tail_signatures,
        readback_signatures,
    )
    readback_checks = {
        **local_checks,
        "outside_corridor_position_index_uv_material_preserved": (
            immutable_readback["passed"]
        ),
        "authenticated_source_tail_surface_preserved": tail_readback["passed"],
        "embedded_textures_and_pbr_unchanged": pbr_readback["passed"],
    }
    common["export_readback"] = {
        "immutable_outside_corridor_surface": immutable_readback,
        "tail_surface": tail_readback,
        "pbr": pbr_readback,
    }
    stage("export_and_readback_done", timer, output=output)
    passed = all(readback_checks.values())
    rejection_reasons = [
        label for label, value in readback_checks.items() if not value
    ]
    if not passed:
        payload = rejected_payload(
            common,
            checks=readback_checks,
            status="rejected_export_readback_identity_mismatch",
            rejection_reasons=rejection_reasons,
            output=output,
        )
        write_manifest(manifest, payload)
        print(
            "PIXAL_SAME_MESH_LIMB_REPAIR_REJECTED "
            f"stage=export_readback reasons={','.join(rejection_reasons)} "
            f"manifest={manifest}",
            flush=True,
        )
        raise SystemExit(2)

    payload = copy.deepcopy(common)
    payload["checks"] = readback_checks
    payload["decision"] = {
        "status": "passed_automatic_geometry_gate_pending_multiview_review",
        "rejection_reasons": [],
        "cat_semantic_retarget_authorized": False,
        "next_gate": "multiview_one_tail_four_limb_no_stray_visual_review",
    }
    payload["output"] = file_record(output)
    payload["formal_dataset_registration_authorized"] = False
    write_manifest(manifest, payload)
    print(
        "PIXAL_SAME_MESH_LIMB_REPAIR_OK "
        f"status={payload['decision']['status']} "
        f"faces={len(readback_triangles)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
