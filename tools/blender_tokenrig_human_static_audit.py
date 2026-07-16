#!/usr/bin/env python3
"""Fail-closed static gate for a Pixal3D mesh rigged by TokenRig.

The module is intentionally importable without Blender. Pure manifest and GLB
contract helpers are exercised by CPU tests; ``main`` imports ``bpy`` only when
Blender executes the actual static audit.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import uuid
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942
TASK3_SCHEMA = "pixal_tokenrig_canary_v1"
RECOVERY_SCHEMA = "pixal_tokenrig_recovery_v1"
FITTED_SCHEMA = "pixal_tokenrig_fitted_skeleton_v1"
SANITIZED_SCHEMA = "pixal_tokenrig_sanitized_weights_v1"
FITTED_BASE_RUNNER_SHA256 = (
    "ab9b56019acc8491777112ee3979d6d4b4a581f3cb339b0bfba0866c87b8f9b9"
)
SOURCE_FRONT = "positive-y"
CANONICAL_FRONT = "negative-y"
WEIGHT_SUM_TOLERANCE = 1.0e-6
SEAM_POSITION_TOLERANCE_M = 2.0e-6
SEAM_WEIGHT_L1_TOLERANCE = 1.0e-6
OPPOSITE_LIMB_WEIGHT_TOLERANCE = 1.0e-4
FULL_REST_TOLERANCE = 2.0e-6
INVERSE_BIND_MATRIX_TOLERANCE = 2.0e-6
BLENDER_EXPORT_MIN_INFLUENCE = 0.0001
BLENDER_EXPORT_SAFE_WEIGHT_FLOOR = 0.00010000000474974513
EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX = 1.0e-8
SURFACE_POSITION_TOLERANCE_M = 2.0e-6
SURFACE_NORMAL_TOLERANCE = 2.0e-2
SURFACE_UV_TOLERANCE = 2.0e-7
SURFACE_AREA_RELATIVE_TOLERANCE = 1.0e-6
RAW_TRIANGLE_LOSS_RATIO_MAX = 2.0e-5
RAW_REMOVED_AREA_RATIO_MAX = 2.0e-6
RAW_NORMAL_P99_MAX = 1.0e-3
RAW_NORMAL_MAX = 2.0e-2
STATIC_AUDIT_DIRNAME = "static_audit_v1"
REQUIRED_BUNDLE_FILES = (
    "bind_pose.glb",
    "bind_front.png",
    "bind_back.png",
    "bind_side.png",
    "bind_top.png",
    "skeleton_overlay.png",
    "weights_contact.png",
    "texture_compare.png",
    "joint_hierarchy.txt",
    "static_qa.json",
)


class StaticAuditError(RuntimeError):
    """Raised when a candidate cannot satisfy the static readiness contract."""


@dataclass(frozen=True)
class ParsedGLB:
    path: Path
    document: dict[str, Any]
    binary: bytes


@dataclass(frozen=True)
class BoneRecord:
    name: str
    parent: str | None
    head: tuple[float, float, float]


@dataclass(frozen=True)
class SurfaceReference:
    polygon_loop_counts: Sequence[int]
    polygon_material_indices: Sequence[int]
    corner_unique_indices: Sequence[int]
    corner_positions: Sequence[float]
    corner_normals: Sequence[float]
    uv_layers: Sequence[Sequence[float]]
    unique_positions: Sequence[float]
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    surface_area_m2: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, description: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise StaticAuditError(f"{description} must be a direct regular file: {path}")
    return path.resolve()


def read_glb(path: Path) -> ParsedGLB:
    path = _regular_file(path, "GLB")
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise StaticAuditError(f"not a GLB 2.0 file: {path}")
    version, declared_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or declared_length != len(raw):
        raise StaticAuditError(
            f"invalid GLB header: version={version} length={declared_length}/{len(raw)}"
        )

    offset = 12
    json_chunks: list[bytes] = []
    binary_chunks: list[bytes] = []
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise StaticAuditError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(raw):
            raise StaticAuditError("truncated GLB chunk payload")
        payload = raw[offset:end]
        offset = end
        if chunk_type == GLB_JSON_CHUNK:
            json_chunks.append(payload)
        elif chunk_type == GLB_BIN_CHUNK:
            binary_chunks.append(payload)
    if len(json_chunks) != 1:
        raise StaticAuditError(f"GLB must contain one JSON chunk, found {len(json_chunks)}")
    if len(binary_chunks) > 1:
        raise StaticAuditError("GLB must contain at most one BIN chunk")
    try:
        document = json.loads(json_chunks[0].rstrip(b" \t\r\n\x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAuditError(f"invalid GLB JSON chunk: {exc}") from exc
    if not isinstance(document, dict):
        raise StaticAuditError("GLB JSON root must be an object")
    return ParsedGLB(path=path, document=document, binary=binary_chunks[0] if binary_chunks else b"")


def _indexed(values: Any, index: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(values, list) or not isinstance(index, int) or not 0 <= index < len(values):
        raise StaticAuditError(f"invalid {description} index: {index!r}")
    value = values[index]
    if not isinstance(value, dict):
        raise StaticAuditError(f"{description} entry must be an object")
    return value


def _texture_image(document: Mapping[str, Any], texture_info: Any) -> Mapping[str, Any]:
    if not isinstance(texture_info, dict):
        raise StaticAuditError("PBR texture binding must be an object")
    texture = _indexed(document.get("textures"), texture_info.get("index"), "texture")
    sources = []
    if texture.get("source") is not None:
        sources.append(texture.get("source"))
    extensions = texture.get("extensions", {})
    if isinstance(extensions, dict):
        for extension_name in ("EXT_texture_webp", "KHR_texture_basisu"):
            extension = extensions.get(extension_name, {})
            if isinstance(extension, dict) and extension.get("source") is not None:
                sources.append(extension.get("source"))
    if not sources or len(set(sources)) != 1:
        raise StaticAuditError(
            f"PBR texture must resolve to exactly one embedded image source: {sources}"
        )
    source = sources[0]
    return _indexed(document.get("images"), source, "image")


def _embedded_image_payload(parsed: ParsedGLB, image: Mapping[str, Any]) -> bytes:
    if not isinstance(image.get("bufferView"), int):
        raise StaticAuditError("every PBR image must use an embedded bufferView")
    view = _indexed(parsed.document.get("bufferViews"), image["bufferView"], "bufferView")
    if view.get("buffer", 0) != 0:
        raise StaticAuditError("embedded PBR bufferView must reference GLB buffer 0")
    start = view.get("byteOffset", 0)
    length = view.get("byteLength")
    if not isinstance(start, int) or not isinstance(length, int) or start < 0 or length <= 0:
        raise StaticAuditError("embedded PBR bufferView has invalid bounds")
    end = start + length
    if end > len(parsed.binary):
        raise StaticAuditError("embedded PBR bufferView exceeds the GLB BIN chunk")
    return parsed.binary[start:end]


def _primitive_material_slots(document: Mapping[str, Any]) -> tuple[int, ...]:
    meshes = document.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise StaticAuditError("GLB has no mesh primitives for PBR material slots")
    ordered: list[int] = []
    for mesh in meshes:
        if not isinstance(mesh, dict) or not isinstance(mesh.get("primitives"), list):
            raise StaticAuditError("GLB mesh has invalid primitives")
        for primitive in mesh["primitives"]:
            if not isinstance(primitive, dict):
                raise StaticAuditError("GLB primitive must be an object")
            material_index = primitive.get("material")
            if not isinstance(material_index, int):
                raise StaticAuditError("every rendered GLB primitive must bind a material")
            if material_index not in ordered:
                ordered.append(material_index)
    if not ordered:
        raise StaticAuditError("GLB mesh primitives bind no PBR materials")
    return tuple(ordered)


def pbr_payload_contract(parsed: ParsedGLB) -> dict[str, dict[str, Any]]:
    """Hash packed images by stable primitive material-slot and semantic role."""

    result: dict[str, dict[str, Any]] = {}
    materials = parsed.document.get("materials")
    if not isinstance(materials, list) or not materials:
        raise StaticAuditError("GLB has no PBR materials")
    for material_slot, material_index in enumerate(
        _primitive_material_slots(parsed.document)
    ):
        material = _indexed(materials, material_index, "material")
        name = material.get("name")
        pbr = material.get("pbrMetallicRoughness", {})
        if not isinstance(pbr, dict):
            raise StaticAuditError(
                f"material slot {material_slot} ({name!r}) has invalid PBR data"
            )
        bindings = {
            "base_color": pbr.get("baseColorTexture"),
            "metallic_roughness": pbr.get("metallicRoughnessTexture"),
            "normal": material.get("normalTexture"),
            "occlusion": material.get("occlusionTexture"),
            "emissive": material.get("emissiveTexture"),
        }
        for role, texture_info in bindings.items():
            if texture_info is None:
                continue
            image = _texture_image(parsed.document, texture_info)
            payload = _embedded_image_payload(parsed, image)
            result[f"material_slot_{material_slot}:{role}"] = {
                "material_name": name,
                "image_name": image.get("name"),
                "mime_type": image.get("mimeType"),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    if not result or not any(key.endswith(":base_color") for key in result):
        raise StaticAuditError("GLB has no packed base-color PBR texture")
    return result


def compare_pbr_payloads(
    source: Mapping[str, Mapping[str, Any]],
    output: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require identical packed bytes by stable primitive slot and texture role."""

    expected = dict(source)
    actual = dict(output)
    if set(actual) != set(expected):
        raise StaticAuditError(
            "PBR material-role set changed: "
            f"missing={sorted(set(expected) - set(actual))} "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    mismatches = {}
    fields = ("sha256", "size_bytes", "mime_type")
    for role in sorted(expected):
        changed = {
            field: {"source": expected[role].get(field), "output": actual[role].get(field)}
            for field in fields
            if expected[role].get(field) != actual[role].get(field)
        }
        if changed:
            mismatches[role] = changed
    if mismatches:
        raise StaticAuditError(f"packed PBR payload bytes changed: {mismatches}")
    return {"passed": True, "roles": sorted(expected), "role_count": len(expected)}


def raw_glb_triangle_contract(parsed: ParsedGLB) -> dict[str, Any]:
    meshes = parsed.document.get("meshes")
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise StaticAuditError("raw GLB must contain exactly one mesh")
    primitives = meshes[0].get("primitives") if isinstance(meshes[0], dict) else None
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise StaticAuditError("raw GLB mesh must contain exactly one primitive")
    primitive = primitives[0]
    if not isinstance(primitive, dict) or primitive.get("mode", 4) != 4:
        raise StaticAuditError("raw GLB primitive must use indexed TRIANGLES mode")
    attributes = primitive.get("attributes")
    if not isinstance(attributes, dict) or any(
        name not in attributes for name in ("POSITION", "NORMAL", "TEXCOORD_0")
    ):
        raise StaticAuditError("raw GLB primitive lacks position/normal/UV attributes")
    accessors = parsed.document.get("accessors")
    index_accessor = _indexed(accessors, primitive.get("indices"), "index accessor")
    if (
        index_accessor.get("type") != "SCALAR"
        or index_accessor.get("componentType") not in {5121, 5123, 5125}
        or not isinstance(index_accessor.get("count"), int)
        or index_accessor["count"] <= 0
        or index_accessor["count"] % 3
    ):
        raise StaticAuditError("raw GLB triangle index accessor is invalid")
    attribute_counts = {}
    for name in ("POSITION", "NORMAL", "TEXCOORD_0"):
        accessor = _indexed(accessors, attributes[name], f"{name} accessor")
        if not isinstance(accessor.get("count"), int) or accessor["count"] <= 0:
            raise StaticAuditError(f"raw GLB {name} accessor count is invalid")
        attribute_counts[name] = accessor["count"]
    if len(set(attribute_counts.values())) != 1:
        raise StaticAuditError("raw GLB position/normal/UV accessor counts differ")
    material_slot = primitive.get("material")
    _indexed(parsed.document.get("materials"), material_slot, "material")
    return {
        "mesh_count": 1,
        "primitive_count": 1,
        "triangle_count": index_accessor["count"] // 3,
        "index_count": index_accessor["count"],
        "serialized_vertex_count": attribute_counts["POSITION"],
        "material_slot": material_slot,
    }


def compare_raw_triangle_contracts(
    source: Mapping[str, Any], output: Mapping[str, Any]
) -> dict[str, Any]:
    if source.get("triangle_count") != output.get("triangle_count"):
        raise StaticAuditError(
            "raw GLB triangle count changed: "
            f"source={source.get('triangle_count')} output={output.get('triangle_count')}"
        )
    fields = ("mesh_count", "primitive_count", "material_slot")
    changed = {
        field: {"source": source.get(field), "output": output.get(field)}
        for field in fields
        if source.get(field) != output.get(field)
    }
    if changed:
        raise StaticAuditError(f"raw GLB primitive contract changed: {changed}")
    return {
        "passed": True,
        "triangle_count": source["triangle_count"],
        "serialized_vertex_count_change": int(output["serialized_vertex_count"])
        - int(source["serialized_vertex_count"]),
    }


def validate_serialization_equivalence_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if metrics.get("removed_faces_are_reverse_coincident") is not True:
        raise StaticAuditError("removed raw triangles are not reverse-coincident duplicates")
    if metrics.get("unique_undirected_face_sets_equal") is not True:
        raise StaticAuditError("unique undirected raw face sets changed")
    checks = (
        ("triangle_loss_ratio", RAW_TRIANGLE_LOSS_RATIO_MAX, "triangle loss ratio"),
        ("removed_area_ratio", RAW_REMOVED_AREA_RATIO_MAX, "removed area ratio"),
        ("maximum_position_error_m", SURFACE_POSITION_TOLERANCE_M, "position error"),
        ("maximum_uv_error", SURFACE_UV_TOLERANCE, "UV error"),
        ("normal_error_p99", RAW_NORMAL_P99_MAX, "normal p99"),
        ("maximum_normal_error", RAW_NORMAL_MAX, "maximum normal error"),
    )
    for field, limit, description in checks:
        try:
            value = float(metrics[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise StaticAuditError(f"serialization metric is missing: {field}") from exc
        if not math.isfinite(value) or value > limit:
            raise StaticAuditError(
                f"bounded serialization {description} exceeds limit: {value} > {limit}"
            )
    result = dict(metrics)
    result.update(
        {
            "passed": True,
            "qualification": "bounded_serialization_equivalence",
            "exact_topology_unchanged": metrics.get("removed_triangle_count") == 0,
            "exact_normals_unchanged": float(metrics["maximum_normal_error"]) == 0.0,
            "limits": {
                "triangle_loss_ratio": RAW_TRIANGLE_LOSS_RATIO_MAX,
                "removed_area_ratio": RAW_REMOVED_AREA_RATIO_MAX,
                "maximum_position_error_m": SURFACE_POSITION_TOLERANCE_M,
                "maximum_uv_error": SURFACE_UV_TOLERANCE,
                "normal_error_p99": RAW_NORMAL_P99_MAX,
                "maximum_normal_error": RAW_NORMAL_MAX,
            },
        }
    )
    return result


def _numpy_glb_accessor(np: Any, parsed: ParsedGLB, index: int) -> Any:
    accessor = _indexed(parsed.document.get("accessors"), index, "accessor")
    if "sparse" in accessor:
        raise StaticAuditError("sparse GLB accessors are not accepted by the raw gate")
    view = _indexed(
        parsed.document.get("bufferViews"), accessor.get("bufferView"), "bufferView"
    )
    if view.get("buffer", 0) != 0:
        raise StaticAuditError("raw GLB accessor must reference embedded buffer 0")
    component_types = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    component_counts = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }
    if accessor.get("componentType") not in component_types or accessor.get(
        "type"
    ) not in component_counts:
        raise StaticAuditError("raw GLB accessor has an unsupported type")
    data_type = np.dtype(component_types[accessor["componentType"]]).newbyteorder("<")
    width = component_counts[accessor["type"]]
    count = accessor.get("count")
    if not isinstance(count, int) or count <= 0:
        raise StaticAuditError("raw GLB accessor count is invalid")
    byte_offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", data_type.itemsize * width)
    try:
        values = np.ndarray(
            (count, width),
            dtype=data_type,
            buffer=parsed.binary,
            offset=byte_offset,
            strides=(stride, data_type.itemsize),
        ).copy()
    except (TypeError, ValueError) as exc:
        raise StaticAuditError(f"raw GLB accessor bounds are invalid: {exc}") from exc
    return values[:, 0] if width == 1 else values


def analyze_raw_serialization_equivalence(
    source: ParsedGLB, output: ParsedGLB
) -> dict[str, Any]:
    import numpy as np

    source_contract = raw_glb_triangle_contract(source)
    output_contract = raw_glb_triangle_contract(output)
    for field in ("mesh_count", "primitive_count", "material_slot"):
        if source_contract[field] != output_contract[field]:
            raise StaticAuditError(f"raw GLB primitive {field} changed")

    def primitive_arrays(parsed: ParsedGLB) -> dict[str, Any]:
        primitive = parsed.document["meshes"][0]["primitives"][0]
        attributes = primitive["attributes"]
        return {
            "positions": _numpy_glb_accessor(np, parsed, attributes["POSITION"]),
            "normals": _numpy_glb_accessor(np, parsed, attributes["NORMAL"]),
            "uvs": _numpy_glb_accessor(np, parsed, attributes["TEXCOORD_0"]),
            "indices": _numpy_glb_accessor(np, parsed, primitive["indices"]).astype(
                np.int64
            ),
        }

    source_values = primitive_arrays(source)
    output_values = primitive_arrays(output)
    source_triangles = source_values["indices"].reshape(-1, 3)
    output_triangles = output_values["indices"].reshape(-1, 3)
    source_unique, source_inverse = np.unique(
        source_values["positions"], axis=0, return_inverse=True
    )

    def row_keys(values: Any) -> Any:
        contiguous = np.ascontiguousarray(values)
        return contiguous.view(
            np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
        ).reshape(-1)

    output_unique, output_inverse = np.unique(
        output_values["positions"], axis=0, return_inverse=True
    )
    source_unique_mapping, output_unique_mapping, _ = _match_unique_positions(
        source_unique.reshape(-1), output_unique.reshape(-1)
    )
    output_to_source_unique = np.asarray(output_unique_mapping, dtype=np.int64)[
        output_inverse
    ]
    source_to_canonical_unique = np.asarray(
        source_unique_mapping, dtype=np.int64
    )[source_inverse]
    source_ids = source_to_canonical_unique[source_triangles]
    output_ids = output_to_source_unique[output_triangles]

    def canonicalize(ids: Any, values: Any) -> tuple[Any, Any]:
        minimum = np.argmin(ids, axis=1)
        canonical_ids = ids.copy()
        canonical_values = values.copy()
        mask = minimum == 1
        canonical_ids[mask] = ids[mask][:, [1, 2, 0]]
        canonical_values[mask] = values[mask][:, [1, 2, 0]]
        mask = minimum == 2
        canonical_ids[mask] = ids[mask][:, [2, 0, 1]]
        canonical_values[mask] = values[mask][:, [2, 0, 1]]
        return canonical_ids, canonical_values

    source_canonical_ids, source_positions = canonicalize(
        source_ids, source_values["positions"][source_triangles]
    )
    output_canonical_ids, output_positions = canonicalize(
        output_ids, output_values["positions"][output_triangles]
    )
    _, source_normals = canonicalize(
        source_ids, source_values["normals"][source_triangles]
    )
    _, output_normals = canonicalize(
        output_ids, output_values["normals"][output_triangles]
    )
    _, source_uvs = canonicalize(source_ids, source_values["uvs"][source_triangles])
    _, output_uvs = canonicalize(output_ids, output_values["uvs"][output_triangles])
    source_face_keys = row_keys(source_canonical_ids.astype(np.int64))
    output_face_keys = row_keys(output_canonical_ids.astype(np.int64))
    if len(np.unique(source_face_keys)) != len(source_face_keys) or len(
        np.unique(output_face_keys)
    ) != len(output_face_keys):
        raise StaticAuditError("raw oriented face keys are unexpectedly duplicated")
    if np.any(~np.isin(output_face_keys, source_face_keys, assume_unique=True)):
        raise StaticAuditError("raw output contains faces absent from the source")
    removed_mask = ~np.isin(source_face_keys, output_face_keys, assume_unique=True)
    removed_indices = np.flatnonzero(removed_mask)

    reversed_removed_ids = source_canonical_ids[removed_indices][:, [0, 2, 1]]
    reversed_canonical_ids, _ = canonicalize(
        reversed_removed_ids, reversed_removed_ids
    )
    reverse_keys = row_keys(reversed_canonical_ids.astype(np.int64))
    reverse_coincident = bool(
        np.all(np.isin(reverse_keys, output_face_keys, assume_unique=True))
    )
    source_undirected = np.unique(np.sort(source_ids, axis=1), axis=0)
    output_undirected = np.unique(np.sort(output_ids, axis=1), axis=0)
    undirected_equal = bool(
        source_undirected.shape == output_undirected.shape
        and np.array_equal(source_undirected, output_undirected)
    )

    source_order = np.argsort(source_face_keys)
    sorted_source_face_keys = source_face_keys[source_order]
    source_locations = np.searchsorted(sorted_source_face_keys, output_face_keys)
    aligned_source = source_order[source_locations]
    position_errors = np.linalg.norm(
        source_positions[aligned_source] - output_positions, axis=2
    ).reshape(-1)
    normal_errors = np.linalg.norm(
        source_normals[aligned_source] - output_normals, axis=2
    ).reshape(-1)
    uv_errors = np.linalg.norm(
        source_uvs[aligned_source] - output_uvs, axis=2
    ).reshape(-1)
    all_source_positions = source_values["positions"][source_triangles]
    doubled_areas = np.linalg.norm(
        np.cross(
            all_source_positions[:, 1] - all_source_positions[:, 0],
            all_source_positions[:, 2] - all_source_positions[:, 0],
        ),
        axis=1,
    )
    source_area = float(0.5 * doubled_areas.sum())
    removed_area = float(0.5 * doubled_areas[removed_indices].sum())
    removed_centers = all_source_positions[removed_indices].mean(axis=1)
    source_material = source.document["materials"][source_contract["material_slot"]]
    output_material = output.document["materials"][output_contract["material_slot"]]
    source_double_sided = bool(source_material.get("doubleSided", False))
    output_double_sided = bool(output_material.get("doubleSided", False))
    if source_double_sided != output_double_sided:
        raise StaticAuditError("raw PBR material doubleSided behavior changed")
    if source_material.get("alphaMode", "OPAQUE") != output_material.get(
        "alphaMode", "OPAQUE"
    ):
        raise StaticAuditError("raw PBR material alphaMode behavior changed")
    thresholds = (2.0e-5, 1.0e-4, 1.0e-3)
    metrics = {
        "source_triangle_count": len(source_triangles),
        "output_triangle_count": len(output_triangles),
        "removed_triangle_count": len(removed_indices),
        "triangle_loss_ratio": len(removed_indices) / len(source_triangles),
        "removed_faces_are_reverse_coincident": reverse_coincident,
        "unique_undirected_face_sets_equal": undirected_equal,
        "source_surface_area_m2": source_area,
        "removed_surface_area_m2": removed_area,
        "removed_area_ratio": removed_area / source_area,
        "maximum_position_error_m": float(position_errors.max(initial=0.0)),
        "maximum_uv_error": float(uv_errors.max(initial=0.0)),
        "normal_error_p50": float(np.quantile(normal_errors, 0.50)),
        "normal_error_p95": float(np.quantile(normal_errors, 0.95)),
        "normal_error_p99": float(np.quantile(normal_errors, 0.99)),
        "maximum_normal_error": float(normal_errors.max(initial=0.0)),
        "normal_corner_count": int(normal_errors.size),
        "normal_error_counts": {
            str(threshold): int((normal_errors > threshold).sum())
            for threshold in thresholds
        },
        "normal_error_ratios": {
            str(threshold): float((normal_errors > threshold).mean())
            for threshold in thresholds
        },
        "removed_face_center_bounds": (
            {
                "minimum": removed_centers.min(axis=0).tolist(),
                "maximum": removed_centers.max(axis=0).tolist(),
            }
            if len(removed_centers)
            else None
        ),
        "source_double_sided": source_double_sided,
        "output_double_sided": output_double_sided,
        "backface_cull_risk": bool(len(removed_indices) and not source_double_sided),
        "serialized_vertex_count_change": output_contract["serialized_vertex_count"]
        - source_contract["serialized_vertex_count"],
    }
    return validate_serialization_equivalence_metrics(metrics)


def canonical_axis_contract(*, source_front: str, prior_transform_count: int) -> dict[str, Any]:
    if source_front != SOURCE_FRONT:
        raise StaticAuditError(f"source front must be {SOURCE_FRONT!r}")
    if prior_transform_count != 0:
        raise StaticAuditError("FRONT +Y to -Y canonicalization must occur exactly once")
    return {
        "source_front": SOURCE_FRONT,
        "canonical_front": CANONICAL_FRONT,
        "yaw_radians": math.pi,
        "yaw_degrees": 180.0,
        "transform_count": 1,
        "matrix": (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        "canonical_front_vector": (0.0, -1.0, 0.0),
        "canonical_up_vector": (0.0, 0.0, 1.0),
        "determinant": 1.0,
    }


def ground_bind_contract(*, source_floor_z: float, prior_transform_count: int) -> dict[str, Any]:
    if not math.isfinite(source_floor_z):
        raise StaticAuditError("source bind floor must be finite")
    if prior_transform_count != 0:
        raise StaticAuditError("bind-floor grounding must occur exactly once")
    translation = -float(source_floor_z)
    post_floor = float(source_floor_z) + translation
    if abs(post_floor) > 1.0e-12:
        raise StaticAuditError("bind-floor grounding did not produce Z=0")
    return {
        "source_floor_z": float(source_floor_z),
        "ground_translation_z": translation,
        "post_floor_z": 0.0,
        "canonical_floor_z": 0.0,
        "transform_count": 1,
    }


def validate_hierarchy(bones: Sequence[BoneRecord]) -> dict[str, Any]:
    if not bones:
        raise StaticAuditError("rest hierarchy is empty")
    names = [bone.name for bone in bones]
    if any(not isinstance(name, str) or not name for name in names):
        raise StaticAuditError("rest hierarchy contains an invalid bone name")
    if len(set(names)) != len(names):
        raise StaticAuditError("rest hierarchy contains duplicate bone names")
    by_name = {bone.name: bone for bone in bones}
    for bone in bones:
        if len(bone.head) != 3 or any(not math.isfinite(float(value)) for value in bone.head):
            raise StaticAuditError(f"bone {bone.name!r} has a non-finite rest head")
        if bone.parent is not None and bone.parent not in by_name:
            raise StaticAuditError(f"bone {bone.name!r} has missing parent {bone.parent!r}")
        if bone.parent == bone.name:
            raise StaticAuditError(f"bone {bone.name!r} forms a cycle")
    roots = [bone.name for bone in bones if bone.parent is None]
    if len(roots) != 1:
        raise StaticAuditError(f"rest hierarchy must have exactly one root, found {roots}")

    children: dict[str, list[str]] = {name: [] for name in names}
    for bone in bones:
        if bone.parent is not None:
            children[bone.parent].append(bone.name)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise StaticAuditError(f"rest hierarchy contains a cycle at {name!r}")
        if name in visited:
            return
        visiting.add(name)
        for child in children[name]:
            visit(child)
        visiting.remove(name)
        visited.add(name)

    visit(roots[0])
    if visited != set(names):
        missing = sorted(set(names) - visited)
        raise StaticAuditError(f"rest hierarchy is disconnected: {missing}")
    index = {name: value for value, name in enumerate(names)}
    parent_first = all(
        bone.parent is None or index[bone.parent] < index[bone.name] for bone in bones
    )
    if not parent_first:
        raise StaticAuditError("rest hierarchy is not stored parent-first")
    return {
        "root": roots[0],
        "bone_count": len(bones),
        "connected": True,
        "parent_first": True,
    }


def compare_full_rest_contracts(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    expected_bones = expected.get("bones")
    actual_bones = actual.get("bones")
    if not isinstance(expected_bones, list) or not isinstance(actual_bones, list):
        raise StaticAuditError("full rest contract bones are missing")
    if len(expected_bones) != len(actual_bones) or not expected_bones:
        raise StaticAuditError("full rest bone count changed")
    expected_names = [bone.get("name") for bone in expected_bones]
    actual_names = [bone.get("name") for bone in actual_bones]
    if expected_names != actual_names:
        raise StaticAuditError("full rest bone order changed")

    def vector_error(first: Any, second: Any, width: int, description: str) -> float:
        if (
            not isinstance(first, (list, tuple))
            or not isinstance(second, (list, tuple))
            or len(first) != width
            or len(second) != width
        ):
            raise StaticAuditError(f"full rest {description} coverage changed")
        values = [float(a) - float(b) for a, b in zip(first, second)]
        if any(not math.isfinite(value) for value in values):
            raise StaticAuditError(f"full rest {description} is non-finite")
        return math.sqrt(sum(value * value for value in values))

    expected_object = expected.get("armature_object_matrix_world")
    actual_object = actual.get("armature_object_matrix_world")
    if (
        not isinstance(expected_object, (list, tuple))
        or not isinstance(actual_object, (list, tuple))
        or len(expected_object) != 16
        or len(actual_object) != 16
    ):
        raise StaticAuditError("full rest armature object matrix coverage changed")
    object_matrix_error = max(
        abs(float(first) - float(second))
        for first, second in zip(expected_object, actual_object)
    )
    if not math.isfinite(object_matrix_error):
        raise StaticAuditError("full rest armature object matrix is non-finite")

    maximum_head = 0.0
    maximum_tail = 0.0
    maximum_axis = 0.0
    maximum_roll = 0.0
    maximum_matrix = 0.0
    exact_metadata_fields = ("parent", "use_connect", "use_deform", "inherit_scale")
    for first, second in zip(expected_bones, actual_bones):
        if not isinstance(first, dict) or not isinstance(second, dict):
            raise StaticAuditError("full rest bone record is invalid")
        if any(first.get(field) != second.get(field) for field in exact_metadata_fields):
            raise StaticAuditError(
                f"full rest metadata changed for {first.get('name')!r}"
            )
        maximum_head = max(
            maximum_head,
            vector_error(first.get("head_local"), second.get("head_local"), 3, "head"),
        )
        maximum_tail = max(
            maximum_tail,
            vector_error(first.get("tail_local"), second.get("tail_local"), 3, "tail"),
        )
        maximum_axis = max(
            maximum_axis,
            vector_error(first.get("roll_axis"), second.get("roll_axis"), 3, "roll axis"),
        )
        first_roll = float(first.get("roll_radians"))
        second_roll = float(second.get("roll_radians"))
        if not math.isfinite(first_roll) or not math.isfinite(second_roll):
            raise StaticAuditError("full rest roll is non-finite")
        roll_error = abs((first_roll - second_roll + math.pi) % (2.0 * math.pi) - math.pi)
        maximum_roll = max(maximum_roll, roll_error)
        first_matrix = first.get("matrix_local")
        second_matrix = second.get("matrix_local")
        if (
            not isinstance(first_matrix, (list, tuple))
            or not isinstance(second_matrix, (list, tuple))
            or len(first_matrix) != 16
            or len(second_matrix) != 16
        ):
            raise StaticAuditError("full rest matrix coverage changed")
        matrix_error = max(
            abs(float(a) - float(b)) for a, b in zip(first_matrix, second_matrix)
        )
        if not math.isfinite(matrix_error):
            raise StaticAuditError("full rest matrix is non-finite")
        maximum_matrix = max(maximum_matrix, matrix_error)
    if object_matrix_error > FULL_REST_TOLERANCE:
        raise StaticAuditError("full rest armature object matrix changed")
    if maximum_head > FULL_REST_TOLERANCE:
        raise StaticAuditError("full rest head changed beyond tolerance")
    if maximum_tail > FULL_REST_TOLERANCE:
        raise StaticAuditError("full rest tail changed beyond tolerance")
    if maximum_axis > FULL_REST_TOLERANCE:
        raise StaticAuditError("full rest roll axis changed beyond tolerance")
    if maximum_roll > FULL_REST_TOLERANCE:
        raise StaticAuditError("full rest roll changed beyond tolerance")
    if maximum_matrix > FULL_REST_TOLERANCE:
        raise StaticAuditError("full rest matrix changed beyond tolerance")
    return {
        "passed": True,
        "bone_count": len(expected_bones),
        "maximum_object_matrix_element_error": object_matrix_error,
        "maximum_head_error_m": maximum_head,
        "maximum_tail_error_m": maximum_tail,
        "maximum_roll_axis_error": maximum_axis,
        "maximum_roll_error_radians": maximum_roll,
        "maximum_matrix_element_error": maximum_matrix,
        "tolerance": FULL_REST_TOLERANCE,
    }


def compare_inverse_bind_contracts(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    expected_names = expected.get("joint_names")
    actual_names = actual.get("joint_names")
    if expected_names != actual_names or not isinstance(expected_names, list):
        raise StaticAuditError("inverse-bind joint order changed")
    expected_matrices = expected.get("matrices")
    actual_matrices = actual.get("matrices")
    if (
        not isinstance(expected_matrices, list)
        or not isinstance(actual_matrices, list)
        or len(expected_matrices) != len(expected_names)
        or len(actual_matrices) != len(expected_names)
        or not expected_names
    ):
        raise StaticAuditError("inverse-bind matrix coverage changed")
    maximum = 0.0
    exact = True
    for first, second in zip(expected_matrices, actual_matrices):
        if (
            not isinstance(first, (list, tuple))
            or not isinstance(second, (list, tuple))
            or len(first) != 16
            or len(second) != 16
        ):
            raise StaticAuditError("inverse-bind matrix width changed")
        for left, right in zip(first, second):
            error = abs(float(left) - float(right))
            if not math.isfinite(error):
                raise StaticAuditError("inverse-bind matrix is non-finite")
            maximum = max(maximum, error)
            exact = exact and float(left) == float(right)
    if maximum > INVERSE_BIND_MATRIX_TOLERANCE:
        raise StaticAuditError(
            f"inverse-bind matrix changed by {maximum}"
        )
    return {
        "passed": True,
        "joint_count": len(expected_names),
        "joint_order_unchanged": True,
        "exact_matrices_unchanged": exact,
        "maximum_matrix_element_error": maximum,
        "tolerance": INVERSE_BIND_MATRIX_TOLERANCE,
    }


def _path_from_root(name: str, by_name: Mapping[str, BoneRecord]) -> list[str]:
    path = []
    current: str | None = name
    while current is not None:
        path.append(current)
        current = by_name[current].parent
    path.reverse()
    return path


def _common_prefix_length(first: Sequence[str], second: Sequence[str]) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


def _require_nonzero_chain(chain_name: str, chain: Sequence[str], by_name: Mapping[str, BoneRecord]) -> None:
    for parent_name, child_name in zip(chain, chain[1:]):
        first = by_name[parent_name].head
        second = by_name[child_name].head
        distance = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)))
        if distance <= 1.0e-8:
            raise StaticAuditError(f"{chain_name} contains a zero-length rest segment")


def _bilateral_paths(
    paths: Sequence[list[str]],
    by_name: Mapping[str, BoneRecord],
    *,
    chain_type: str,
    minimum_nodes: int,
) -> tuple[list[str], list[str]]:
    if len(paths) != 2:
        raise StaticAuditError(f"ambiguous {chain_type} chains: expected 2, found {len(paths)}")
    positive = [path for path in paths if by_name[path[-1]].head[0] > 0.0]
    negative = [path for path in paths if by_name[path[-1]].head[0] < 0.0]
    if len(positive) != 1 or len(negative) != 1:
        raise StaticAuditError(f"ambiguous {chain_type} bilateral side assignment")
    left, right = positive[0], negative[0]
    if len(left) < minimum_nodes or len(right) < minimum_nodes:
        raise StaticAuditError(f"{chain_type} chains are too short")
    if abs(len(left) - len(right)) > 1:
        raise StaticAuditError(f"{chain_type} chains are not bilaterally compatible")
    left_terminal = by_name[left[-1]].head
    right_terminal = by_name[right[-1]].head
    scale = max(abs(left_terminal[0]), abs(right_terminal[0]), 1.0e-9)
    if abs(abs(left_terminal[0]) - abs(right_terminal[0])) > 0.20 * scale:
        raise StaticAuditError(f"{chain_type} terminal positions are not bilaterally symmetric")
    return left, right


def _common_path(paths: Sequence[Sequence[str]]) -> list[str]:
    if not paths:
        return []
    common = list(paths[0])
    for path in paths[1:]:
        common = common[: _common_prefix_length(common, path)]
    return common


def _cluster_proven_distal_arm_paths(
    paths: Sequence[list[str]],
    by_name: Mapping[str, BoneRecord],
    *,
    body_height: float,
) -> tuple[list[list[str]], list[str]]:
    groups: dict[str, list[list[str]]] = {}
    for path in paths:
        groups.setdefault(path[0], []).append(path)
    reduced: list[list[str]] = []
    ignored: list[str] = []
    for proximal in sorted(groups):
        cluster = groups[proximal]
        if len(cluster) == 1:
            reduced.append(cluster[0])
            continue
        common = _common_path(cluster)
        if len(common) < 4:
            raise StaticAuditError(
                f"ambiguous arm paths below proximal branch {proximal!r}"
            )
        anchor = by_name[common[-1]].head
        distal_names = sorted(
            {name for path in cluster for name in path[len(common) :]}
        )
        if not distal_names:
            raise StaticAuditError(f"arm cluster {proximal!r} has no distal descendants")
        maximum_distance = max(
            math.sqrt(
                sum(
                    (float(value) - float(origin)) ** 2
                    for value, origin in zip(by_name[name].head, anchor)
                )
            )
            for name in distal_names
        )
        if maximum_distance > 0.15 * body_height:
            raise StaticAuditError(
                f"ambiguous arm descendants below {common[-1]!r} are not short fingers"
            )
        reduced.append(common)
        ignored.extend(distal_names)
    return reduced, sorted(set(ignored))


def resolve_five_semantic_chains(bones: Sequence[BoneRecord]) -> dict[str, Any]:
    """Resolve an intentionally strict humanoid map from topology and rest heads.

    The direct TokenRig path emits generic ``bone_N`` names. This resolver
    enumerates paths satisfying hard geometric constraints and rejects any
    non-unique answer rather than guessing an animation mapping.
    """

    validate_hierarchy(bones)
    by_name = {bone.name: bone for bone in bones}
    children: dict[str, list[str]] = {bone.name: [] for bone in bones}
    for bone in bones:
        if bone.parent is not None:
            children[bone.parent].append(bone.name)
    leaves = [name for name, values in children.items() if not values]
    xs = [float(bone.head[0]) for bone in bones]
    zs = [float(bone.head[2]) for bone in bones]
    height = max(zs) - min(zs)
    width = max(xs) - min(xs)
    if height <= 1.0e-8 or width <= 1.0e-8:
        raise StaticAuditError("rest hierarchy has degenerate humanoid bounds")
    root = next(bone for bone in bones if bone.parent is None)
    top_tolerance = max(1.0e-5, 0.02 * height)
    center_tolerance = max(1.0e-5, 0.10 * height)
    axial_terminals = [
        name
        for name in leaves
        if float(by_name[name].head[2]) >= max(zs) - top_tolerance
        and abs(float(by_name[name].head[0]) - float(root.head[0])) <= center_tolerance
    ]
    ignored_head_descendants: list[str] = []
    if len(axial_terminals) == 1:
        axial = _path_from_root(axial_terminals[0], by_name)
        axial_leaf_names = {axial_terminals[0]}
    elif len(axial_terminals) > 1:
        candidate_paths = [_path_from_root(name, by_name) for name in axial_terminals]
        axial = _common_path(candidate_paths)
        if not axial:
            raise StaticAuditError(
                f"ambiguous axial head chain: candidates={sorted(axial_terminals)}"
            )
        head_anchor = by_name[axial[-1]].head
        if max(zs) - float(head_anchor[2]) > 0.05 * height:
            raise StaticAuditError(
                f"ambiguous axial head chain: candidates={sorted(axial_terminals)}"
            )
        ignored_head_descendants = sorted(
            {name for path in candidate_paths for name in path[len(axial) :]}
        )
        maximum_head_distance = max(
            math.sqrt(
                sum(
                    (float(value) - float(origin)) ** 2
                    for value, origin in zip(by_name[name].head, head_anchor)
                )
            )
            for name in ignored_head_descendants
        )
        if maximum_head_distance > 0.10 * height:
            raise StaticAuditError("ambiguous axial descendants are not confined to Head")
        axial_leaf_names = set(axial_terminals)
    else:
        raise StaticAuditError("ambiguous axial head chain: no top-center candidate")
    if len(axial) < 4:
        raise StaticAuditError("axial chain is too short for pelvis/spine/neck/head")

    arm_paths: list[list[str]] = []
    leg_paths: list[list[str]] = []
    for leaf in leaves:
        if leaf in axial_leaf_names:
            continue
        full_path = _path_from_root(leaf, by_name)
        divergence = _common_prefix_length(full_path, axial)
        branch = full_path[divergence:]
        if not branch:
            continue
        terminal = by_name[leaf].head
        proximal = by_name[branch[0]].head
        if (
            divergence <= 1
            and float(terminal[2]) < float(root.head[2]) - 0.25 * height
        ):
            leg_paths.append(branch)
        elif (
            divergence >= max(2, len(axial) - 3)
            and float(proximal[2]) > float(root.head[2]) + 0.20 * height
            and abs(float(proximal[0]) - float(root.head[0])) > 0.03 * width
        ):
            arm_paths.append(branch)
        else:
            raise StaticAuditError(f"unclassified required-looking branch ending at {leaf!r}")

    arm_paths, ignored_distal_descendants = _cluster_proven_distal_arm_paths(
        arm_paths, by_name, body_height=height
    )
    left_arm, right_arm = _bilateral_paths(
        arm_paths, by_name, chain_type="arm", minimum_nodes=4
    )
    left_leg, right_leg = _bilateral_paths(
        leg_paths, by_name, chain_type="leg", minimum_nodes=4
    )
    chains = {
        "axial": axial,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_leg": left_leg,
        "right_leg": right_leg,
    }
    for chain_name, chain in chains.items():
        _require_nonzero_chain(chain_name, chain, by_name)

    semantic = {
        "pelvis": axial[0],
        "spine": axial[1:-2],
        "neck": axial[-2],
        "head": axial[-1],
        "left_clavicle": left_arm[-4],
        "left_upper_arm": left_arm[-3],
        "left_forearm": left_arm[-2],
        "left_hand": left_arm[-1],
        "right_clavicle": right_arm[-4],
        "right_upper_arm": right_arm[-3],
        "right_forearm": right_arm[-2],
        "right_hand": right_arm[-1],
        "left_thigh": left_leg[-4],
        "left_calf": left_leg[-3],
        "left_foot": left_leg[-2],
        "left_toe": left_leg[-1],
        "right_thigh": right_leg[-4],
        "right_calf": right_leg[-3],
        "right_foot": right_leg[-2],
        "right_toe": right_leg[-1],
    }
    return {
        "chains": chains,
        "semantic_bones": semantic,
        "side_basis": {"left": "positive-x", "right": "negative-x"},
        "method": "unique_topology_and_canonical_rest_position_v1",
        "ignored_proven_distal_descendants": ignored_distal_descendants,
        "ignored_proven_head_descendants": ignored_head_descendants,
    }


def validate_vertex_weights(
    vertex_weights: Sequence[Mapping[str, float]],
    *,
    bone_names: set[str],
) -> dict[str, Any]:
    maximum_influences = 0
    maximum_sum_error = 0.0
    for vertex_index, weights in enumerate(vertex_weights):
        if not weights:
            raise StaticAuditError(f"zero-weight vertex {vertex_index}")
        if len(weights) > 4:
            raise StaticAuditError(f"vertex {vertex_index} has more than four influences")
        unknown = sorted(set(weights) - bone_names)
        if unknown:
            raise StaticAuditError(f"vertex {vertex_index} references unknown bone(s): {unknown}")
        values = [float(value) for value in weights.values()]
        if any(not math.isfinite(value) for value in values):
            raise StaticAuditError(f"vertex {vertex_index} has a non-finite weight")
        if any(value <= 0.0 for value in values):
            raise StaticAuditError(f"vertex {vertex_index} has a non-positive weight")
        error = abs(sum(values) - 1.0)
        if error > WEIGHT_SUM_TOLERANCE:
            raise StaticAuditError(f"vertex {vertex_index} weights are not normalized")
        maximum_influences = max(maximum_influences, len(values))
        maximum_sum_error = max(maximum_sum_error, error)
    if not vertex_weights:
        raise StaticAuditError("skinned mesh has no vertices")
    return {
        "vertex_count": len(vertex_weights),
        "maximum_influences": maximum_influences,
        "maximum_weight_sum_error": maximum_sum_error,
        "weight_sum_tolerance": WEIGHT_SUM_TOLERANCE,
    }


def _weight_l1(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    return sum(
        abs(float(first.get(name, 0.0)) - float(second.get(name, 0.0)))
        for name in set(first) | set(second)
    )


def validate_seam_weights(
    positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    if len(positions) != len(vertex_weights):
        raise StaticAuditError("seam positions and weights have different lengths")
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, position in enumerate(positions):
        if len(position) != 3 or any(not math.isfinite(float(value)) for value in position):
            raise StaticAuditError(f"vertex {index} has a non-finite seam position")
        key = tuple(round(float(value) / SEAM_POSITION_TOLERANCE_M) for value in position)
        buckets.setdefault(key, []).append(index)
    duplicates = [indices for indices in buckets.values() if len(indices) > 1]
    maximum_error = 0.0
    for indices in duplicates:
        expected = vertex_weights[indices[0]]
        for index in indices[1:]:
            error = _weight_l1(expected, vertex_weights[index])
            maximum_error = max(maximum_error, error)
            if error > SEAM_WEIGHT_L1_TOLERANCE:
                raise StaticAuditError(
                    f"UV seam duplicate vertex {index} has inconsistent skin weight"
                )
    return {
        "duplicate_position_group_count": len(duplicates),
        "maximum_weight_l1_error": maximum_error,
        "weight_l1_tolerance": SEAM_WEIGHT_L1_TOLERANCE,
        "position_tolerance_m": SEAM_POSITION_TOLERANCE_M,
    }


def compare_mesh_contracts(
    source: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    allow_serialization_splits: bool = False,
) -> dict[str, Any]:
    if allow_serialization_splits:
        fields = (
            "polygon_count",
            "loop_count",
            "uv_layer_count",
            "material_slot_count",
            "uv_sha256",
            "polygon_material_sha256",
        )
    else:
        fields = (
            "vertex_count",
            "polygon_count",
            "loop_count",
            "material_slot_count",
            "position_sha256",
            "topology_sha256",
            "uv_sha256",
            "polygon_material_sha256",
        )
    changed = {
        field: {"source": source.get(field), "output": output.get(field)}
        for field in fields
        if source.get(field) != output.get(field)
    }
    if changed:
        raise StaticAuditError(f"mesh/UV/material contract changed: {changed}")
    surface_area_error = 0.0
    surface_area_tolerance = 0.0
    if allow_serialization_splits:
        try:
            expected_area = float(source["surface_area_m2"])
            actual_area = float(output["surface_area_m2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StaticAuditError("mesh surface area contract is missing") from exc
        if not math.isfinite(expected_area) or not math.isfinite(actual_area):
            raise StaticAuditError("mesh surface area is non-finite")
        surface_area_error = abs(expected_area - actual_area)
        surface_area_tolerance = max(
            1.0e-9, abs(expected_area) * SURFACE_AREA_RELATIVE_TOLERANCE
        )
        if surface_area_error > surface_area_tolerance:
            raise StaticAuditError(
                "mesh surface area changed: "
                f"source={expected_area} output={actual_area} "
                f"tolerance={surface_area_tolerance}"
            )
    return {
        "passed": True,
        "fields": list(fields),
        "serialization_splits_allowed": allow_serialization_splits,
        "serialized_vertex_count_change": (
            int(output.get("vertex_count", 0)) - int(source.get("vertex_count", 0))
        ),
        "surface_area_error_m2": surface_area_error,
        "surface_area_tolerance_m2": surface_area_tolerance,
    }


def _maximum_vector_error(
    expected: Sequence[float],
    actual: Sequence[float],
    *,
    width: int,
    description: str,
) -> float:
    if len(expected) != len(actual) or len(expected) % width:
        raise StaticAuditError(f"surface {description} coverage changed")
    maximum = 0.0
    for offset in range(0, len(expected), width):
        error = math.sqrt(
            sum(
                (float(expected[offset + axis]) - float(actual[offset + axis])) ** 2
                for axis in range(width)
            )
        )
        maximum = max(maximum, error)
    return maximum


def _match_unique_positions(
    expected: Sequence[float], actual: Sequence[float]
) -> tuple[array, array, float]:
    if len(expected) % 3 or len(actual) % 3:
        raise StaticAuditError("surface unique position count changed")
    expected_count = len(expected) // 3
    actual_count = len(actual) // 3
    if expected_count == 0 or actual_count == 0:
        raise StaticAuditError("surface unique position coverage is empty")
    cell = SURFACE_POSITION_TOLERANCE_M
    def cluster(
        values: Sequence[float], count: int, description: str
    ) -> tuple[array, set[int]]:
        representative_buckets: dict[tuple[int, int, int], list[int]] = {}
        mapping = array("I")
        representatives: set[int] = set()
        for index in range(count):
            offset = index * 3
            point = tuple(float(values[offset + axis]) for axis in range(3))
            base = tuple(math.floor(value / cell) for value in point)
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for representative in representative_buckets.get(
                            (base[0] + dx, base[1] + dy, base[2] + dz), ()
                        ):
                            representative_offset = representative * 3
                            distance = math.sqrt(
                                sum(
                                    (
                                        point[axis]
                                        - float(values[representative_offset + axis])
                                    )
                                    ** 2
                                    for axis in range(3)
                                )
                            )
                            if distance <= SURFACE_POSITION_TOLERANCE_M:
                                candidates.append(representative)
            unique_candidates = sorted(set(candidates))
            if len(unique_candidates) > 1:
                raise StaticAuditError(
                    f"ambiguous {description} tolerance clustering"
                )
            if unique_candidates:
                representative = unique_candidates[0]
            else:
                representative = index
                representatives.add(index)
                representative_buckets.setdefault(base, []).append(index)
            mapping.append(representative)
        return mapping, representatives

    expected_mapping, expected_representatives = cluster(
        expected, expected_count, "expected position"
    )
    actual_local_mapping, actual_representatives = cluster(
        actual, actual_count, "actual position"
    )

    expected_buckets: dict[tuple[int, int, int], list[int]] = {}
    for index in range(expected_count):
        offset = index * 3
        key = tuple(
            math.floor(float(expected[offset + axis]) / cell) for axis in range(3)
        )
        expected_buckets.setdefault(key, []).append(index)
    actual_point_targets: list[int] = []
    maximum = 0.0
    for actual_index in range(actual_count):
        actual_offset = actual_index * 3
        point = tuple(float(actual[actual_offset + axis]) for axis in range(3))
        base = tuple(math.floor(value / cell) for value in point)
        best_distance = math.inf
        candidate_representatives: set[int] = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for expected_index in expected_buckets.get(
                        (base[0] + dx, base[1] + dy, base[2] + dz), ()
                    ):
                        expected_offset = expected_index * 3
                        distance = math.sqrt(
                            sum(
                                (
                                    point[axis]
                                    - float(expected[expected_offset + axis])
                                )
                                ** 2
                                for axis in range(3)
                            )
                        )
                        if distance <= SURFACE_POSITION_TOLERANCE_M:
                            candidate_representatives.add(
                                int(expected_mapping[expected_index])
                            )
                            best_distance = min(best_distance, distance)
        if not candidate_representatives:
            raise StaticAuditError(
                "surface unique position coverage changed beyond tolerance"
            )
        if len(candidate_representatives) > 1:
            raise StaticAuditError(
                "ambiguous cross-surface tolerance-cluster match"
            )
        actual_point_targets.append(next(iter(candidate_representatives)))
        maximum = max(maximum, best_distance)

    actual_cluster_targets: dict[int, int] = {}
    for local_representative, target in zip(
        actual_local_mapping, actual_point_targets
    ):
        local = int(local_representative)
        prior = actual_cluster_targets.get(local)
        if prior is not None and prior != target:
            raise StaticAuditError(
                "actual tolerance cluster maps to multiple expected clusters"
            )
        actual_cluster_targets[local] = target
    targets = list(actual_cluster_targets.values())
    if (
        set(actual_cluster_targets) != actual_representatives
        or len(set(targets)) != len(targets)
        or set(targets) != expected_representatives
    ):
        raise StaticAuditError(
            "surface tolerance clusters do not have a one-to-one match"
        )
    actual_mapping = array(
        "I",
        (
            actual_cluster_targets[int(local_representative)]
            for local_representative in actual_local_mapping
        ),
    )
    return expected_mapping, actual_mapping, maximum


def _canonical_face(values: Sequence[int]) -> tuple[tuple[int, ...], int]:
    if len(values) < 3 or len(set(values)) != len(values):
        raise StaticAuditError("surface contains a degenerate polygon")
    rotations = [
        tuple(int(values[(offset + index) % len(values)]) for index in range(len(values)))
        for offset in range(len(values))
    ]
    canonical = min(rotations)
    return canonical, rotations.index(canonical)


def compare_surface_references(
    expected: SurfaceReference, actual: SurfaceReference
) -> dict[str, Any]:
    if len(expected.uv_layers) != len(actual.uv_layers):
        raise StaticAuditError("surface UV layer count changed")
    expected_loop_count = sum(int(value) for value in expected.polygon_loop_counts)
    actual_loop_count = sum(int(value) for value in actual.polygon_loop_counts)
    if (
        len(expected.polygon_loop_counts) != len(actual.polygon_loop_counts)
        or expected_loop_count != actual_loop_count
        or len(expected.polygon_material_indices) != len(expected.polygon_loop_counts)
        or len(actual.polygon_material_indices) != len(actual.polygon_loop_counts)
    ):
        raise StaticAuditError("surface polygon topology changed")
    for reference, description in ((expected, "expected"), (actual, "actual")):
        if (
            len(reference.corner_unique_indices) != expected_loop_count
            or len(reference.corner_positions) != expected_loop_count * 3
            or len(reference.corner_normals) != expected_loop_count * 3
            or any(len(layer) != expected_loop_count * 2 for layer in reference.uv_layers)
        ):
            raise StaticAuditError(f"{description} surface corner coverage changed")

    expected_unique_mapping, actual_unique_mapping, unique_error = _match_unique_positions(
        expected.unique_positions, actual.unique_positions
    )
    actual_faces: dict[bytes, list[int]] = {}
    actual_offset = 0
    for loop_count, material_index in zip(
        actual.polygon_loop_counts, actual.polygon_material_indices
    ):
        count = int(loop_count)
        mapped_ids = [
            int(
                actual_unique_mapping[
                    int(actual.corner_unique_indices[actual_offset + index])
                ]
            )
            for index in range(count)
        ]
        if len(set(mapped_ids)) != count:
            raise StaticAuditError("surface contains a degenerate polygon")
        canonical = tuple(sorted(mapped_ids))
        key = struct.pack(f"<{count + 2}I", int(material_index), count, *canonical)
        actual_faces.setdefault(key, []).append(actual_offset)
        actual_offset += count

    position_error = 0.0
    normal_error = 0.0
    maximum_uv_error = 0.0
    normal_face_error_thresholds = (2.0e-5, 1.0e-4, 5.0e-4, 1.0e-3)
    normal_face_error_counts = {threshold: 0 for threshold in normal_face_error_thresholds}
    expected_offset = 0
    for loop_count, material_index in zip(
        expected.polygon_loop_counts, expected.polygon_material_indices
    ):
        count = int(loop_count)
        expected_ids = [
            int(
                expected_unique_mapping[
                    int(expected.corner_unique_indices[expected_offset + index])
                ]
            )
            for index in range(count)
        ]
        if len(set(expected_ids)) != count:
            raise StaticAuditError("surface contains a degenerate polygon")
        canonical = tuple(sorted(expected_ids))
        key = struct.pack(f"<{count + 2}I", int(material_index), count, *canonical)
        candidates = actual_faces.get(key)
        if not candidates:
            raise StaticAuditError("surface polygon topology/material assignment changed")
        best_candidate_index = 0
        best_errors = (math.inf, math.inf, math.inf)
        best_score = math.inf
        for candidate_index, actual_start in enumerate(candidates):
            actual_corner_by_id = {
                int(
                    actual_unique_mapping[
                        int(actual.corner_unique_indices[actual_start + index])
                    ]
                ): actual_start + index
                for index in range(count)
            }
            candidate_position_error = 0.0
            candidate_normal_error = 0.0
            candidate_uv_error = 0.0
            for index, expected_id in enumerate(expected_ids):
                expected_corner = expected_offset + index
                actual_corner = actual_corner_by_id[expected_id]
                candidate_position_error = max(
                    candidate_position_error,
                    math.sqrt(
                        sum(
                            (
                                float(
                                    expected.corner_positions[
                                        expected_corner * 3 + axis
                                    ]
                                )
                                - float(
                                    actual.corner_positions[actual_corner * 3 + axis]
                                )
                            )
                            ** 2
                            for axis in range(3)
                        )
                    ),
                )
                candidate_normal_error = max(
                    candidate_normal_error,
                    math.sqrt(
                        sum(
                            (
                                float(
                                    expected.corner_normals[
                                        expected_corner * 3 + axis
                                    ]
                                )
                                - float(
                                    actual.corner_normals[actual_corner * 3 + axis]
                                )
                            )
                            ** 2
                            for axis in range(3)
                        )
                    ),
                )
                for expected_uv, actual_uv in zip(
                    expected.uv_layers, actual.uv_layers
                ):
                    candidate_uv_error = max(
                        candidate_uv_error,
                        math.sqrt(
                            sum(
                                (
                                    float(expected_uv[expected_corner * 2 + axis])
                                    - float(actual_uv[actual_corner * 2 + axis])
                                )
                                ** 2
                                for axis in range(2)
                            )
                        ),
                    )
            score = max(
                candidate_position_error / SURFACE_POSITION_TOLERANCE_M,
                candidate_normal_error / SURFACE_NORMAL_TOLERANCE,
                candidate_uv_error / SURFACE_UV_TOLERANCE,
            )
            if score < best_score:
                best_candidate_index = candidate_index
                best_errors = (
                    candidate_position_error,
                    candidate_normal_error,
                    candidate_uv_error,
                )
                best_score = score
        candidates.pop(best_candidate_index)
        if not candidates:
            del actual_faces[key]
        position_error = max(position_error, best_errors[0])
        normal_error = max(normal_error, best_errors[1])
        maximum_uv_error = max(maximum_uv_error, best_errors[2])
        for threshold in normal_face_error_thresholds:
            if best_errors[1] > threshold:
                normal_face_error_counts[threshold] += 1
        expected_offset += count
    if actual_faces:
        raise StaticAuditError("surface contains extra polygons")
    if position_error > SURFACE_POSITION_TOLERANCE_M:
        raise StaticAuditError(f"surface corner position changed by {position_error} m")
    if normal_error > SURFACE_NORMAL_TOLERANCE:
        raise StaticAuditError(
            f"surface corner normal changed by {normal_error}; "
            f"face_error_counts={normal_face_error_counts}"
        )
    if maximum_uv_error > SURFACE_UV_TOLERANCE:
        raise StaticAuditError(f"surface UV corner changed by {maximum_uv_error}")
    bounds_error = _maximum_vector_error(
        tuple(value for point in expected.bounds for value in point),
        tuple(value for point in actual.bounds for value in point),
        width=3,
        description="bounds",
    )
    if bounds_error > SURFACE_POSITION_TOLERANCE_M:
        raise StaticAuditError(f"surface bounds changed by {bounds_error} m")
    expected_area = float(expected.surface_area_m2)
    actual_area = float(actual.surface_area_m2)
    if not math.isfinite(expected_area) or not math.isfinite(actual_area):
        raise StaticAuditError("surface area is non-finite")
    area_error = abs(expected_area - actual_area)
    area_tolerance = max(
        1.0e-9, abs(expected_area) * SURFACE_AREA_RELATIVE_TOLERANCE
    )
    if area_error > area_tolerance:
        raise StaticAuditError(
            f"surface area changed by {area_error} m2 (tolerance {area_tolerance})"
        )
    return {
        "passed": True,
        "polygon_count": len(expected.polygon_loop_counts),
        "loop_count": expected_loop_count,
        "unique_position_count": len(expected.unique_positions) // 3,
        "tolerance_cluster_count": len(set(expected_unique_mapping)),
        "serialized_unique_position_count_change": (
            len(actual.unique_positions) - len(expected.unique_positions)
        )
        // 3,
        "maximum_corner_position_error_m": position_error,
        "maximum_corner_normal_error": normal_error,
        "normal_face_error_counts": {
            str(threshold): count
            for threshold, count in normal_face_error_counts.items()
        },
        "maximum_uv_error": maximum_uv_error,
        "maximum_unique_position_error_m": unique_error,
        "maximum_bounds_error_m": bounds_error,
        "surface_area_error_m2": area_error,
        "position_tolerance_m": SURFACE_POSITION_TOLERANCE_M,
        "normal_tolerance": SURFACE_NORMAL_TOLERANCE,
        "uv_tolerance": SURFACE_UV_TOLERANCE,
        "surface_area_tolerance_m2": area_tolerance,
    }


def validate_bilateral_contamination(
    positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    chains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if len(positions) != len(vertex_weights) or not positions:
        raise StaticAuditError("bilateral contamination inputs are empty or mismatched")
    xs = [float(position[0]) for position in positions]
    if any(not math.isfinite(value) for value in xs):
        raise StaticAuditError("bilateral contamination positions are non-finite")
    center = (min(xs) + max(xs)) * 0.5
    half_width = (max(xs) - min(xs)) * 0.5
    if half_width <= 1.0e-9:
        raise StaticAuditError("bilateral contamination check has degenerate width")
    distal_cutoff = 0.25 * half_width
    left_bones = set(chains["left_arm"]) | set(chains["left_leg"])
    right_bones = set(chains["right_arm"]) | set(chains["right_leg"])
    contaminated = []
    maximum_opposite = 0.0
    considered = 0
    for index, (position, weights) in enumerate(zip(positions, vertex_weights)):
        side_x = float(position[0]) - center
        if abs(side_x) < distal_cutoff:
            continue
        considered += 1
        opposite = right_bones if side_x > 0.0 else left_bones
        opposite_weight = sum(float(weights.get(name, 0.0)) for name in opposite)
        maximum_opposite = max(maximum_opposite, opposite_weight)
        if opposite_weight > OPPOSITE_LIMB_WEIGHT_TOLERANCE:
            contaminated.append(index)
    if contaminated:
        raise StaticAuditError(
            "opposite-limb contamination on distal vertices: "
            f"count={len(contaminated)} maximum={maximum_opposite}"
        )
    return {
        "considered_distal_vertex_count": considered,
        "contaminated_vertex_count": 0,
        "maximum_opposite_limb_weight": maximum_opposite,
        "tolerance": OPPOSITE_LIMB_WEIGHT_TOLERANCE,
    }


def _position_key(position: Sequence[float]) -> tuple[int, int, int]:
    if len(position) != 3 or any(not math.isfinite(float(value)) for value in position):
        raise StaticAuditError("skin contract contains a non-finite position")
    return tuple(round(float(value) / SEAM_POSITION_TOLERANCE_M) for value in position)


def _skin_by_position(
    positions: Sequence[Sequence[float]],
    weights: Sequence[Mapping[str, float]],
    description: str,
) -> dict[tuple[int, int, int], Mapping[str, float]]:
    if len(positions) != len(weights):
        raise StaticAuditError(f"{description} skin positions and weights differ in length")
    result: dict[tuple[int, int, int], Mapping[str, float]] = {}
    for position, vertex_weights in zip(positions, weights):
        key = _position_key(position)
        if key in result and _weight_l1(result[key], vertex_weights) > SEAM_WEIGHT_L1_TOLERANCE:
            if description == "roundtrip":
                raise StaticAuditError(
                    "roundtrip skin weights changed across serialized seam duplicates"
                )
            raise StaticAuditError(f"{description} seam duplicates have inconsistent weights")
        result[key] = vertex_weights
    return result


def compare_skin_by_position(
    expected_positions: Sequence[Sequence[float]],
    expected_weights: Sequence[Mapping[str, float]],
    actual_positions: Sequence[Sequence[float]],
    actual_weights: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    def exact_points(
        positions: Sequence[Sequence[float]],
        weights: Sequence[Mapping[str, float]],
        description: str,
    ) -> tuple[list[tuple[float, float, float]], list[Mapping[str, float]]]:
        if len(positions) != len(weights):
            raise StaticAuditError(
                f"{description} skin positions and weights differ in length"
            )
        points: list[tuple[float, float, float]] = []
        values: list[Mapping[str, float]] = []
        by_point: dict[tuple[float, float, float], int] = {}
        for position, vertex_weights in zip(positions, weights):
            if len(position) != 3 or any(
                not math.isfinite(float(value)) for value in position
            ):
                raise StaticAuditError(
                    f"{description} skin contains a non-finite position"
                )
            point = tuple(float(value) for value in position)
            prior = by_point.get(point)
            if prior is not None:
                if _weight_l1(values[prior], vertex_weights) > SEAM_WEIGHT_L1_TOLERANCE:
                    if description == "roundtrip":
                        raise StaticAuditError(
                            "roundtrip skin weights changed across exact-position duplicates"
                        )
                    raise StaticAuditError(
                        f"{description} exact-position duplicates have inconsistent weights"
                    )
                continue
            by_point[point] = len(points)
            points.append(point)
            values.append(vertex_weights)
        return points, values

    expected_points, expected_point_weights = exact_points(
        expected_positions, expected_weights, "expected"
    )
    actual_points, actual_point_weights = exact_points(
        actual_positions, actual_weights, "roundtrip"
    )
    expected_mapping, actual_mapping, position_error = _match_unique_positions(
        tuple(value for point in expected_points for value in point),
        tuple(value for point in actual_points for value in point),
    )

    def clustered_weights(
        mapping: Sequence[int],
        values: Sequence[Mapping[str, float]],
        description: str,
    ) -> dict[int, Mapping[str, float]]:
        result: dict[int, Mapping[str, float]] = {}
        for representative, weights in zip(mapping, values):
            key = int(representative)
            if (
                key in result
                and _weight_l1(result[key], weights) > SEAM_WEIGHT_L1_TOLERANCE
            ):
                raise StaticAuditError(
                    f"{description} skin tolerance cluster has inconsistent weights"
                )
            result[key] = weights
        return result

    expected = clustered_weights(
        expected_mapping, expected_point_weights, "expected"
    )
    actual = clustered_weights(actual_mapping, actual_point_weights, "roundtrip")
    if set(expected) != set(actual):
        raise StaticAuditError("roundtrip skin tolerance-cluster coverage changed")
    maximum_error = 0.0
    for key in expected:
        error = _weight_l1(expected[key], actual[key])
        maximum_error = max(maximum_error, error)
        if error > SEAM_WEIGHT_L1_TOLERANCE:
            raise StaticAuditError(
                f"roundtrip skin weights changed at {key}: L1={error}"
            )
    return {
        "passed": True,
        "unique_position_count": len(expected_points),
        "tolerance_cluster_count": len(expected),
        "serialized_vertex_count_change": len(actual_positions) - len(expected_positions),
        "serialized_unique_position_count_change": len(actual_points)
        - len(expected_points),
        "maximum_position_error_m": position_error,
        "position_tolerance_m": SURFACE_POSITION_TOLERANCE_M,
        "maximum_weight_l1_error": maximum_error,
        "weight_l1_tolerance": SEAM_WEIGHT_L1_TOLERANCE,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir() or source.is_symlink():
        raise StaticAuditError(f"staging directory is invalid: {source}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise StaticAuditError("atomic no-replace directory rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        value = ctypes.get_errno()
        if value == errno.EEXIST:
            raise StaticAuditError(f"no-replace destination already exists: {destination}")
        raise StaticAuditError(
            f"atomic no-replace rename failed: {os.strerror(value)}"
        )
    _fsync_directory(destination.parent)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> Path:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return path


def write_failure_evidence(
    *,
    output_dir: Path,
    asset_id: str,
    error: BaseException,
    authenticated: Mapping[str, Any] | None,
) -> Path:
    output_dir = Path(output_dir)
    parent = output_dir.parent.resolve()
    if not parent.is_dir() or parent.is_symlink():
        raise StaticAuditError(f"failure-evidence parent is invalid: {parent}")
    evidence = parent / f"{output_dir.name}.failed.{uuid.uuid4().hex}.json"
    payload = {
        "schema": "tokenrig_human_static_attempt_v1",
        "asset_id": asset_id,
        "decision": "rejected",
        "agent_qa_status": "rejected",
        "user_acceptance": "not_requested",
        "readiness_bundle_published": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authenticated": dict(authenticated or {}),
        "failure": {"type": type(error).__name__, "message": str(error)},
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return _write_exclusive(evidence, encoded)


def _canonical_blender_name(name: str | None) -> str | None:
    return None if name is None else re.sub(r"\.\d{3}$", "", name)


def _finite_matrix(matrix: Any, description: str) -> None:
    values = [float(value) for row in matrix for value in row]
    if not values or any(not math.isfinite(value) for value in values):
        raise StaticAuditError(f"{description} contains a non-finite matrix")


def _quantized_digest(rows: Sequence[Sequence[float]], quantum: float) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            if not math.isfinite(float(value)):
                raise StaticAuditError("mesh contract contains a non-finite value")
            digest.update(struct.pack("<q", round(float(value) / quantum)))
    return digest.hexdigest()


def capture_blender_mesh_contract(mesh: Any) -> dict[str, Any]:
    if len(mesh.data.uv_layers) < 1:
        raise StaticAuditError("Pixal runtime mesh has no UV layer")
    if len(mesh.material_slots) < 1 or any(slot.material is None for slot in mesh.material_slots):
        raise StaticAuditError("Pixal runtime mesh has an empty material slot")
    _finite_matrix(mesh.matrix_world, "mesh object transform")
    positions = [tuple(mesh.matrix_world @ vertex.co) for vertex in mesh.data.vertices]
    position_keys = [
        tuple(round(float(value) / 1.0e-7) for value in position)
        for position in positions
    ]
    unique_position_keys = sorted(set(position_keys))
    unique_position_digest = hashlib.sha256()
    for key in unique_position_keys:
        unique_position_digest.update(struct.pack("<3q", *key))
    bounds_quantized = (
        tuple(min(key[axis] for key in position_keys) for axis in range(3)),
        tuple(max(key[axis] for key in position_keys) for axis in range(3)),
    )
    topology = hashlib.sha256()
    corner_positions = hashlib.sha256()
    polygon_material = hashlib.sha256()
    surface_area = 0.0
    for polygon in mesh.data.polygons:
        topology.update(struct.pack("<I", len(polygon.vertices)))
        topology.update(struct.pack(f"<{len(polygon.vertices)}I", *polygon.vertices))
        corner_positions.update(struct.pack("<I", len(polygon.vertices)))
        for vertex_index in polygon.vertices:
            corner_positions.update(struct.pack("<3q", *position_keys[vertex_index]))
        polygon_positions = [positions[index] for index in polygon.vertices]
        for corner in range(1, len(polygon_positions) - 1):
            origin = polygon_positions[0]
            first = tuple(
                float(polygon_positions[corner][axis]) - float(origin[axis])
                for axis in range(3)
            )
            second = tuple(
                float(polygon_positions[corner + 1][axis]) - float(origin[axis])
                for axis in range(3)
            )
            cross = (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )
            surface_area += 0.5 * math.sqrt(sum(value * value for value in cross))
        polygon_material.update(struct.pack("<I", int(polygon.material_index)))
        if polygon.material_index < 0 or polygon.material_index >= len(mesh.material_slots):
            raise StaticAuditError("polygon references an invalid material slot")
    uv_digest = hashlib.sha256()
    for layer in mesh.data.uv_layers:
        if len(layer.data) != len(mesh.data.loops):
            raise StaticAuditError("UV layer loop count differs from mesh loop count")
        for item in layer.data:
            for value in item.uv:
                if not math.isfinite(float(value)):
                    raise StaticAuditError("UV layer contains a non-finite coordinate")
                uv_digest.update(struct.pack("<q", round(float(value) / 1.0e-7)))
    if len(mesh.data.corner_normals) != len(mesh.data.loops):
        raise StaticAuditError("corner normal count differs from mesh loop count")
    normal_matrix = mesh.matrix_world.to_3x3().inverted_safe().transposed()
    corner_normal_digest = hashlib.sha256()
    for item in mesh.data.corner_normals:
        normal = normal_matrix @ item.vector
        if float(normal.length) <= 1.0e-12:
            raise StaticAuditError("mesh contains a zero corner normal")
        normal.normalize()
        for value in normal:
            if not math.isfinite(float(value)):
                raise StaticAuditError("mesh contains a non-finite corner normal")
            corner_normal_digest.update(
                struct.pack("<q", round(float(value) / 1.0e-6))
            )
    return {
        "vertex_count": len(mesh.data.vertices),
        "polygon_count": len(mesh.data.polygons),
        "loop_count": len(mesh.data.loops),
        "uv_layer_count": len(mesh.data.uv_layers),
        "material_slot_count": len(mesh.material_slots),
        "position_sha256": _quantized_digest(positions, 1.0e-7),
        "topology_sha256": topology.hexdigest(),
        "corner_position_sha256": corner_positions.hexdigest(),
        "corner_normal_sha256": corner_normal_digest.hexdigest(),
        "unique_position_count": len(unique_position_keys),
        "unique_position_sha256": unique_position_digest.hexdigest(),
        "bounds_quantized": bounds_quantized,
        "surface_area_m2": surface_area,
        "uv_sha256": uv_digest.hexdigest(),
        "polygon_material_sha256": polygon_material.hexdigest(),
        "material_names": [
            _canonical_blender_name(slot.material.name if slot.material else None)
            for slot in mesh.material_slots
        ],
    }


def capture_blender_surface_reference(mesh: Any) -> SurfaceReference:
    _finite_matrix(mesh.matrix_world, "surface reference mesh transform")
    positions = [tuple(mesh.matrix_world @ vertex.co) for vertex in mesh.data.vertices]
    if not positions:
        raise StaticAuditError("surface reference mesh has no vertices")
    corner_positions = array("f")
    for loop in mesh.data.loops:
        position = positions[loop.vertex_index]
        if any(not math.isfinite(float(value)) for value in position):
            raise StaticAuditError("surface reference has a non-finite corner position")
        corner_positions.extend(float(value) for value in position)

    normal_matrix = mesh.matrix_world.to_3x3().inverted_safe().transposed()
    if len(mesh.data.corner_normals) != len(mesh.data.loops):
        raise StaticAuditError("surface reference corner-normal coverage changed")
    corner_normals = array("f")
    for item in mesh.data.corner_normals:
        normal = normal_matrix @ item.vector
        if float(normal.length) <= 1.0e-12:
            raise StaticAuditError("surface reference has a zero corner normal")
        normal.normalize()
        if any(not math.isfinite(float(value)) for value in normal):
            raise StaticAuditError("surface reference has a non-finite corner normal")
        corner_normals.extend(float(value) for value in normal)

    uv_layers = []
    for layer in mesh.data.uv_layers:
        if len(layer.data) != len(mesh.data.loops):
            raise StaticAuditError("surface reference UV coverage changed")
        values = array("f")
        for item in layer.data:
            if any(not math.isfinite(float(value)) for value in item.uv):
                raise StaticAuditError("surface reference has a non-finite UV")
            values.extend(float(value) for value in item.uv)
        uv_layers.append(values)

    unique = sorted(set(positions))
    unique_index = {position: index for index, position in enumerate(unique)}
    corner_unique_indices = array(
        "I",
        (
            unique_index[positions[loop.vertex_index]]
            for loop in mesh.data.loops
        ),
    )
    unique_positions = array("f")
    for position in unique:
        unique_positions.extend(float(value) for value in position)
    minimum = tuple(min(position[axis] for position in positions) for axis in range(3))
    maximum = tuple(max(position[axis] for position in positions) for axis in range(3))
    surface_area = 0.0
    for polygon in mesh.data.polygons:
        polygon_positions = [positions[index] for index in polygon.vertices]
        for corner in range(1, len(polygon_positions) - 1):
            origin = polygon_positions[0]
            first = tuple(
                float(polygon_positions[corner][axis]) - float(origin[axis])
                for axis in range(3)
            )
            second = tuple(
                float(polygon_positions[corner + 1][axis]) - float(origin[axis])
                for axis in range(3)
            )
            cross = (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )
            surface_area += 0.5 * math.sqrt(sum(value * value for value in cross))
    return SurfaceReference(
        polygon_loop_counts=array(
            "I", (len(polygon.vertices) for polygon in mesh.data.polygons)
        ),
        polygon_material_indices=array(
            "I", (int(polygon.material_index) for polygon in mesh.data.polygons)
        ),
        corner_unique_indices=corner_unique_indices,
        corner_positions=corner_positions,
        corner_normals=corner_normals,
        uv_layers=tuple(uv_layers),
        unique_positions=unique_positions,
        bounds=(minimum, maximum),
        surface_area_m2=surface_area,
    )


def identify_source_mesh(bpy: Any) -> Any:
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render
    ]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(meshes) != 1 or armatures:
        raise StaticAuditError(
            "original Pixal GLB must contain exactly one rendered mesh and no armature"
        )
    return meshes[0]


def _identity_matrix(matrix: Any, tolerance: float = 1.0e-9) -> bool:
    _finite_matrix(matrix, "non-runtime empty transform")
    try:
        return all(
            abs(float(matrix[row][column]) - (1.0 if row == column else 0.0))
            <= tolerance
            for row in range(4)
            for column in range(4)
        )
    except (IndexError, TypeError) as exc:
        raise StaticAuditError("non-runtime empty transform is not a 4x4 matrix") from exc


def validate_proven_runtime_orphans(
    objects: Sequence[Any],
) -> tuple[dict[str, str], ...]:
    records = []
    for obj in objects:
        if obj.type != "EMPTY":
            raise StaticAuditError(
                f"TokenRig GLB contains a non-runtime scene object: {obj.name} ({obj.type})"
            )
        if obj.parent is not None:
            raise StaticAuditError(
                f"non-runtime empty {obj.name!r} must be an orphan root, not parented"
            )
        if tuple(obj.children):
            raise StaticAuditError(
                f"non-runtime empty {obj.name!r} has children and cannot be removed"
            )
        if obj.data is not None:
            raise StaticAuditError(
                f"non-runtime empty {obj.name!r} unexpectedly carries data"
            )
        if not _identity_matrix(obj.matrix_world):
            raise StaticAuditError(
                f"non-runtime empty {obj.name!r} must have an identity transform"
            )
        records.append(
            {
                "name": obj.name,
                "type": "EMPTY",
                "reason": "finite_identity_childless_dataless_root",
            }
        )
    return tuple(records)


def validate_gltf_import_helper_collection(collection: Any) -> dict[str, Any]:
    if collection.name != "glTF_not_exported":
        raise StaticAuditError("unexpected Blender glTF helper collection name")
    if not collection.hide_render or not collection.hide_viewport:
        raise StaticAuditError("Blender glTF helper collection must be hidden")
    objects = tuple(collection.objects)
    if len(objects) != 1:
        raise StaticAuditError(
            "Blender glTF helper collection must contain exactly one joint shape"
        )
    helper = objects[0]
    users_collection = tuple(helper.users_collection)
    if (
        helper.type != "MESH"
        or not re.fullmatch(r"Icosphere(?:\.\d{3})?", helper.name)
        or helper.parent is not None
        or tuple(helper.children)
        or len(users_collection) != 1
        or users_collection[0] is not collection
        or not _identity_matrix(helper.matrix_world)
        or len(helper.data.vertices) != 42
        or len(helper.data.polygons) != 80
    ):
        raise StaticAuditError(
            "Blender glTF non-exported joint shape must be the exact 42-vertex/80-polygon Icosphere"
        )
    return {
        "collection": collection.name,
        "object": helper.name,
        "vertex_count": 42,
        "polygon_count": 80,
        "reason": "blender_gltf_generated_nonexported_joint_shape",
    }


def remove_gltf_import_helpers(bpy: Any) -> list[dict[str, Any]]:
    collection = bpy.data.collections.get("glTF_not_exported")
    if collection is None:
        return []
    record = validate_gltf_import_helper_collection(collection)
    bpy.data.collections.remove(collection)
    bpy.context.view_layer.update()
    if any(obj.name == record["object"] for obj in bpy.context.scene.objects):
        raise StaticAuditError("Blender glTF helper object remained in the runtime scene")
    return [record]


def remove_proven_runtime_orphans(
    bpy: Any, objects: Sequence[Any]
) -> tuple[dict[str, str], ...]:
    records = validate_proven_runtime_orphans(objects)
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()
    return records


def identify_exact_runtime(bpy: Any) -> tuple[Any, Any, tuple[Any, ...]]:
    scene_objects = set(bpy.context.scene.objects)
    armatures = [obj for obj in scene_objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in scene_objects if obj.type == "MESH" and not obj.hide_render]
    if len(armatures) != 1 or len(meshes) != 1:
        raise StaticAuditError(
            "TokenRig GLB must contain exactly one armature and one rendered skinned mesh"
        )
    armature, mesh = armatures[0], meshes[0]
    extras = tuple(sorted(scene_objects - {armature, mesh}, key=lambda obj: obj.name))
    validate_proven_runtime_orphans(extras)
    modifiers = [
        modifier
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object == armature
    ]
    if len(modifiers) != 1:
        raise StaticAuditError(
            "runtime mesh must have exactly one Armature modifier targeting its armature"
        )
    if len(mesh.data.uv_layers) < 1 or len(mesh.material_slots) < 1:
        raise StaticAuditError("runtime Pixal mesh lost UVs or materials")
    _finite_matrix(armature.matrix_world, "armature object transform")
    _finite_matrix(mesh.matrix_world, "mesh object transform")
    return armature, mesh, extras


def runtime_roots(runtime_objects: set[Any]) -> tuple[Any, ...]:
    roots = tuple(obj for obj in runtime_objects if obj.parent not in runtime_objects)
    if not roots:
        raise StaticAuditError("runtime closure has no transform root")
    return roots


def mesh_world_positions(mesh: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in (mesh.matrix_world @ vertex.co)) for vertex in mesh.data.vertices)


def mesh_floor_z(mesh: Any) -> float:
    positions = mesh_world_positions(mesh)
    if not positions:
        raise StaticAuditError("runtime Pixal mesh has no vertices")
    value = min(position[2] for position in positions)
    if not math.isfinite(value):
        raise StaticAuditError("runtime Pixal bind floor is non-finite")
    return value


def bone_records_from_armature(armature: Any) -> tuple[BoneRecord, ...]:
    records = []
    for bone in armature.data.bones:
        _finite_matrix(bone.matrix_local, f"rest matrix for {bone.name}")
        head = armature.matrix_world @ bone.head_local
        records.append(
            BoneRecord(
                name=bone.name,
                parent=bone.parent.name if bone.parent is not None else None,
                head=tuple(float(value) for value in head),
            )
        )
    return tuple(records)


def capture_blender_full_rest_contract(armature: Any) -> dict[str, Any]:
    object_matrix = [float(value) for row in armature.matrix_world for value in row]
    if len(object_matrix) != 16 or any(
        not math.isfinite(value) for value in object_matrix
    ):
        raise StaticAuditError("armature object matrix is invalid")
    bones = []
    for bone in armature.data.bones:
        axis, roll = bone.AxisRollFromMatrix(bone.matrix_local.to_3x3())
        record = {
            "name": bone.name,
            "parent": bone.parent.name if bone.parent is not None else None,
            "head_local": [float(value) for value in bone.head_local],
            "tail_local": [float(value) for value in bone.tail_local],
            "roll_axis": [float(value) for value in axis],
            "roll_radians": float(roll),
            "matrix_local": [
                float(value) for row in bone.matrix_local for value in row
            ],
            "use_connect": bool(bone.use_connect),
            "use_deform": bool(bone.use_deform),
            "inherit_scale": str(bone.inherit_scale),
        }
        numeric = (
            record["head_local"]
            + record["tail_local"]
            + record["roll_axis"]
            + [record["roll_radians"]]
            + record["matrix_local"]
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise StaticAuditError(f"full rest bone {bone.name!r} is non-finite")
        bones.append(record)
    if not bones:
        raise StaticAuditError("full rest contract has no bones")
    return {
        "armature_object_matrix_world": object_matrix,
        "bones": bones,
    }


def extract_inverse_bind_contract(parsed: ParsedGLB) -> dict[str, Any]:
    import numpy as np

    skins = parsed.document.get("skins")
    nodes = parsed.document.get("nodes")
    if (
        not isinstance(skins, list)
        or len(skins) != 1
        or not isinstance(skins[0], dict)
        or not isinstance(nodes, list)
    ):
        raise StaticAuditError("GLB must contain one inverse-bind skin")
    skin = skins[0]
    joints = skin.get("joints")
    accessor = skin.get("inverseBindMatrices")
    if not isinstance(joints, list) or not isinstance(accessor, int):
        raise StaticAuditError("GLB skin has no inverse-bind matrices")
    joint_names = []
    for index in joints:
        if (
            not isinstance(index, int)
            or index < 0
            or index >= len(nodes)
            or not isinstance(nodes[index], dict)
            or not isinstance(nodes[index].get("name"), str)
        ):
            raise StaticAuditError("GLB inverse-bind joint is invalid")
        joint_names.append(nodes[index]["name"])
    if len(set(joint_names)) != len(joint_names):
        raise StaticAuditError("GLB inverse-bind joint names are not unique")
    values = _numpy_glb_accessor(np, parsed, accessor)
    try:
        matrices = values.reshape(len(joints), 16)
    except ValueError as exc:
        raise StaticAuditError("GLB inverse-bind matrix shape changed") from exc
    if not np.isfinite(matrices).all():
        raise StaticAuditError("GLB inverse-bind matrices are non-finite")
    return {
        "joint_names": joint_names,
        "matrices": [
            [float(value) for value in matrix]
            for matrix in matrices
        ],
    }


def extract_vertex_weights(
    mesh: Any, armature: Any
) -> tuple[tuple[dict[str, float], ...], tuple[tuple[float, float, float], ...]]:
    bone_names = {bone.name for bone in armature.data.bones}
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    result = []
    for vertex in mesh.data.vertices:
        weights: dict[str, float] = {}
        for membership in vertex.groups:
            name = group_names.get(membership.group)
            if name is None:
                raise StaticAuditError(f"vertex {vertex.index} references a missing vertex group")
            value = float(membership.weight)
            if value > 0.0:
                weights[name] = value
        result.append(weights)
    values = tuple(result)
    validate_vertex_weights(values, bone_names=bone_names)
    positions = mesh_world_positions(mesh)
    return values, positions


def _select_runtime_only(bpy: Any, armature: Any, mesh: Any) -> None:
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    armature.hide_render = False
    mesh.hide_render = False
    armature.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    if set(bpy.context.selected_objects) != {armature, mesh}:
        raise StaticAuditError("could not select the exact runtime closure")


def export_bind_pose_glb(bpy: Any, armature: Any, mesh: Any, path: Path) -> None:
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action, do_unlink=True)
    if armature.animation_data is not None:
        armature.animation_data_clear()
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    _select_runtime_only(bpy, armature, mesh)
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size <= 0:
        raise StaticAuditError(f"bind-pose GLB export failed: {path}")


def _compare_rest_bones(
    expected: Sequence[BoneRecord], actual: Sequence[BoneRecord]
) -> dict[str, Any]:
    expected_by_name = {bone.name: bone for bone in expected}
    actual_by_name = {bone.name: bone for bone in actual}
    if set(expected_by_name) != set(actual_by_name):
        raise StaticAuditError("GLB roundtrip changed rest bone names")
    maximum_error = 0.0
    for name, first in expected_by_name.items():
        second = actual_by_name[name]
        if first.parent != second.parent:
            raise StaticAuditError(f"GLB roundtrip changed parent of {name!r}")
        error = math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(first.head, second.head))
        )
        maximum_error = max(maximum_error, error)
    if maximum_error > SEAM_POSITION_TOLERANCE_M:
        raise StaticAuditError(f"GLB roundtrip changed rest heads by {maximum_error} m")
    return {
        "bone_count": len(expected),
        "maximum_rest_head_error_m": maximum_error,
        "tolerance_m": SEAM_POSITION_TOLERANCE_M,
    }


def roundtrip_validate_bind(
    *,
    bpy: Any,
    glb_path: Path,
    source_pbr: Mapping[str, Mapping[str, Any]],
    expected_mesh: Mapping[str, Any],
    expected_surface: SurfaceReference,
    expected_bones: Sequence[BoneRecord],
    expected_positions: Sequence[Sequence[float]],
    expected_weights: Sequence[Mapping[str, float]],
    expected_semantics: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.gltf(filepath=str(glb_path))
    if "FINISHED" not in result:
        raise StaticAuditError("could not re-import bind-pose GLB")
    import_helpers = remove_gltf_import_helpers(bpy)
    armature, mesh, orphans = identify_exact_runtime(bpy)
    removed_orphans = remove_proven_runtime_orphans(bpy, orphans)
    actual_mesh = capture_blender_mesh_contract(mesh)
    mesh_validation = compare_mesh_contracts(
        expected_mesh, actual_mesh, allow_serialization_splits=True
    )
    actual_surface = capture_blender_surface_reference(mesh)
    surface_validation = compare_surface_references(expected_surface, actual_surface)
    actual_pbr = pbr_payload_contract(read_glb(glb_path))
    pbr_validation = compare_pbr_payloads(source_pbr, actual_pbr)
    actual_bones = bone_records_from_armature(armature)
    hierarchy = validate_hierarchy(actual_bones)
    semantics = resolve_five_semantic_chains(actual_bones)
    if semantics["chains"] != expected_semantics["chains"]:
        raise StaticAuditError("GLB roundtrip changed the five semantic chains")
    rest_validation = _compare_rest_bones(expected_bones, actual_bones)
    actual_weights, actual_positions = extract_vertex_weights(mesh, armature)
    weight_validation = validate_vertex_weights(
        actual_weights, bone_names={bone.name for bone in actual_bones}
    )
    seam_validation = validate_seam_weights(actual_positions, actual_weights)
    skin_validation = compare_skin_by_position(
        expected_positions,
        expected_weights,
        actual_positions,
        actual_weights,
    )
    contamination = validate_bilateral_contamination(
        actual_positions, actual_weights, semantics["chains"]
    )
    return armature, mesh, {
        "passed": True,
        "mesh": mesh_validation,
        "surface": surface_validation,
        "pbr": pbr_validation,
        "hierarchy": hierarchy,
        "rest": rest_validation,
        "weights": weight_validation,
        "seams": seam_validation,
        "skin": skin_validation,
        "bilateral_contamination": contamination,
        "removed_proven_orphans": list(removed_orphans),
        "removed_gltf_import_helpers": import_helpers,
    }


def _validate_record(path: Path, record: Any, description: str) -> str:
    if not isinstance(record, dict):
        raise StaticAuditError(f"{description} record is missing")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise StaticAuditError(f"{description} record has no SHA-256")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise StaticAuditError(
            f"{description} SHA-256 mismatch: actual={actual_hash} expected={expected_hash}"
        )
    expected_size = record.get("bytes", record.get("size_bytes"))
    if expected_size is not None and expected_size != path.stat().st_size:
        raise StaticAuditError(f"{description} byte size mismatch")
    recorded_path = record.get("path")
    if recorded_path is not None and Path(recorded_path).resolve() != path.resolve():
        raise StaticAuditError(f"{description} path mismatch")
    return actual_hash


def _manifest_relative_file(
    manifest_path: Path, relative_path: Any, description: str
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise StaticAuditError(f"{description} relative path is missing")
    value = Path(relative_path)
    if value.is_absolute() or ".." in value.parts:
        raise StaticAuditError(f"{description} must remain under the fitted output")
    root = manifest_path.parent.resolve()
    path = _regular_file(root / value, description)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StaticAuditError(
            f"{description} escaped the fitted output directory"
        ) from exc
    return path


def _read_json_mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAuditError(f"invalid {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise StaticAuditError(f"{description} root must be an object")
    return value


def _authenticate_fitted_execution(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    conditioning_path: Path,
) -> dict[str, Any]:
    command = manifest.get("command")
    ledger_record = manifest.get("attempt_ledger")
    if not isinstance(ledger_record, dict) or not isinstance(
        ledger_record.get("path"), str
    ):
        raise StaticAuditError("fitted attempt ledger descriptor is missing")
    ledger_path = _regular_file(
        Path(ledger_record["path"]), "fitted attempt ledger"
    )
    ledger_hash = _validate_record(
        ledger_path, ledger_record, "fitted attempt ledger"
    )
    ledger = _read_json_mapping(ledger_path, "fitted attempt ledger")
    if (
        ledger.get("schema") != "pixal_tokenrig_attempt_v1"
        or ledger.get("status") != "succeeded"
        or ledger.get("returncode") != 0
    ):
        raise StaticAuditError(
            "fitted attempt ledger must prove status succeeded with returncode 0"
        )
    if not isinstance(command, list) or command != ledger.get("command"):
        raise StaticAuditError("fitted manifest/ledger command mismatch")
    if (
        command.count("42") != 1
        or command.count("--use_skeleton") != 1
        or command.count("--use_transfer") != 1
        or manifest.get("random_parameters", {}).get("seed") != 42
    ):
        raise StaticAuditError(
            "fitted command must contain exact seed 42, --use_skeleton, and --use_transfer"
        )

    orchestrator = manifest.get("orchestrator")
    if (
        not isinstance(orchestrator, dict)
        or orchestrator.get("provenance_schema") != TASK3_SCHEMA
        or not isinstance(orchestrator.get("runner"), dict)
    ):
        raise StaticAuditError("fitted orchestrator provenance is missing")
    runner_record = orchestrator["runner"]
    if not isinstance(runner_record.get("path"), str):
        raise StaticAuditError("fitted orchestrator runner path is missing")
    runner_path = _regular_file(
        Path(runner_record["path"]), "fitted orchestrator runner"
    )
    runner_hash = _validate_record(
        runner_path, runner_record, "fitted orchestrator runner"
    )
    delegated_record = orchestrator.get("delegated_runner")
    if delegated_record is not None:
        if not isinstance(delegated_record, dict) or not isinstance(
            delegated_record.get("path"), str
        ):
            raise StaticAuditError("fitted delegated runner descriptor is missing")
        delegated_path = _regular_file(
            Path(delegated_record["path"]), "fitted delegated runner"
        )
        delegated_hash = _validate_record(
            delegated_path, delegated_record, "fitted delegated runner"
        )
    else:
        if runner_path.name != "tokenrig_human_fitted_skeleton_fallback.py":
            raise StaticAuditError(
                "legacy fitted provenance cannot infer its delegated runner"
            )
        delegated_path = _regular_file(
            runner_path.with_name("tokenrig_human_canary.py"),
            "fitted delegated runner",
        )
        delegated_hash = sha256_file(delegated_path)
        if delegated_hash != FITTED_BASE_RUNNER_SHA256:
            raise StaticAuditError(
                "fitted delegated runner SHA-256 does not match the local base-runner pin"
            )

    hygiene = manifest.get("server_hygiene")
    if (
        not isinstance(hygiene, dict)
        or hygiene.get("cleans_before_every_bpyparser_load") is not True
        or hygiene.get("mechanism") != "injected_sitecustomize_v1"
    ):
        raise StaticAuditError("fitted server hygiene contract is missing")
    patch_path = _manifest_relative_file(
        manifest_path,
        hygiene.get("relative_path"),
        "fitted server hygiene patch",
    )
    patch_hash = sha256_file(patch_path)
    if hygiene.get("sha256") != patch_hash:
        raise StaticAuditError("fitted server hygiene patch SHA-256 mismatch")

    load_record = hygiene.get("load_audit")
    if not isinstance(load_record, dict):
        raise StaticAuditError("fitted server load-audit descriptor is missing")
    load_path = _manifest_relative_file(
        manifest_path,
        load_record.get("relative_path"),
        "fitted server load-audit",
    )
    _validate_record(
        load_path,
        {**load_record, "path": str(load_path)},
        "fitted server load-audit",
    )
    events = []
    try:
        for line in load_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if not isinstance(event, dict):
                raise StaticAuditError("fitted load-audit event must be an object")
            events.append(event)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAuditError(f"invalid fitted server load-audit: {exc}") from exc
    expected_events = [
        (sequence, phase)
        for sequence in (1, 2)
        for phase in ("before_clean", "after_clean", "after_import")
    ]
    actual_events = [(event.get("sequence"), event.get("phase")) for event in events]
    if len(events) != 6 or actual_events != expected_events:
        raise StaticAuditError(
            "fitted hygiene must preserve exactly six ordered load-audit events"
        )
    for event in events:
        try:
            event_path = Path(str(event.get("filepath"))).resolve()
        except (OSError, RuntimeError) as exc:
            raise StaticAuditError("fitted load-audit filepath is invalid") from exc
        if event_path != conditioning_path.resolve():
            raise StaticAuditError("fitted load-audit input path changed")
        if event.get("phase") == "after_clean":
            inventory = event.get("inventory")
            if (
                not isinstance(inventory, dict)
                or inventory.get("objects") != []
                or any(
                    inventory.get(field) != 0
                    for field in ("mesh_count", "material_count", "image_count")
                )
            ):
                raise StaticAuditError(
                    "fitted load-audit does not prove an empty post-clean scene"
                )
    loads = hygiene.get("loads")
    if (
        not isinstance(loads, list)
        or len(loads) != 2
        or [(item.get("sequence"), item.get("role")) for item in loads]
        != [(1, "source"), (2, "transfer_target")]
        or any(
            Path(str(item.get("filepath"))).resolve() != conditioning_path.resolve()
            for item in loads
        )
    ):
        raise StaticAuditError("fitted hygiene must authenticate its exact two loads")

    processes = hygiene.get("processes")
    if (
        not isinstance(processes, list)
        or len(processes) != 2
        or {item.get("role") for item in processes if isinstance(item, dict)}
        != {"demo", "bpy_server"}
    ):
        raise StaticAuditError(
            "fitted hygiene must authenticate exactly two runtime processes"
        )
    for process in processes:
        marker_record = process.get("marker")
        if not isinstance(marker_record, dict):
            raise StaticAuditError("fitted process marker descriptor is missing")
        marker_path = _manifest_relative_file(
            manifest_path,
            marker_record.get("path"),
            "fitted process marker",
        )
        _validate_record(
            marker_path,
            {**marker_record, "path": str(marker_path)},
            "fitted process marker",
        )
        marker = _read_json_mapping(marker_path, "fitted process marker")
        if (
            marker.get("pid") != process.get("pid")
            or marker.get("seed") != 42
            or marker.get("patch_sha256") != patch_hash
        ):
            raise StaticAuditError("fitted process marker contract changed")
    return {
        "attempt_ledger_sha256": ledger_hash,
        "orchestrator_runner_sha256": runner_hash,
        "delegated_runner_sha256": delegated_hash,
        "server_hygiene_patch_sha256": patch_hash,
        "server_hygiene_load_audit_sha256": sha256_file(load_path),
        "server_hygiene_load_event_count": len(events),
    }


def _same_file_record(first: Any, second: Any) -> bool:
    return isinstance(first, dict) and isinstance(second, dict) and all(
        first.get(field) == second.get(field)
        for field in ("path", "sha256", "size_bytes")
    )


def _authenticate_sanitized_inputs(
    *,
    asset_id: str,
    source_glb: Path,
    tokenrig_glb: Path,
    tokenrig_manifest: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    publication = manifest.get("publication")
    if (
        manifest.get("attempt") != "deterministic_learned_weight_sanitation"
        or manifest.get("algorithm_version")
        != "tokenrig_side_transfer_seam_hybrid_export_floor_v3"
        or manifest.get("inference_used") is not False
        or manifest.get("rocketbox_mesh_used") is not False
        or manifest.get("rocketbox_weights_used") is not False
        or manifest.get("animation_authorized") is not False
        or manifest.get("static_audit_status")
        != "pending_sanitized_static_audit"
        or not isinstance(publication, dict)
        or publication.get("directory_mode") != "0755"
        or publication.get("artifact_mode") != "0444"
        or publication.get("no_replace") is not True
        or "nested static_audit_v1" not in str(
            publication.get("directory_mode_reason", "")
        )
    ):
        raise StaticAuditError(
            "sanitized manifest does not prove deterministic no-inference sanitation"
        )
    input_record = manifest.get("input")
    if not isinstance(input_record, dict):
        raise StaticAuditError("sanitized input provenance is missing")
    original_record = input_record.get("original_source_glb")
    original_manifest_record = input_record.get("original_source_manifest")
    source_hash = _validate_record(
        source_glb, original_record, "sanitized original Pixal GLB"
    )
    if not isinstance(original_manifest_record, dict) or not isinstance(
        original_manifest_record.get("path"), str
    ):
        raise StaticAuditError("sanitized original Pixal manifest record is missing")
    source_manifest = _regular_file(
        Path(original_manifest_record["path"]), "sanitized original Pixal manifest"
    )
    source_manifest_hash = _validate_record(
        source_manifest,
        original_manifest_record,
        "sanitized original Pixal manifest",
    )

    fitted_record = input_record.get("fitted_glb")
    fitted_manifest_record = input_record.get("fitted_manifest")
    if not isinstance(fitted_record, dict) or not isinstance(
        fitted_record.get("path"), str
    ):
        raise StaticAuditError("sanitized fitted GLB descriptor is missing")
    fitted_glb = _regular_file(
        Path(fitted_record["path"]), "sanitized fitted TokenRig GLB"
    )
    fitted_hash = _validate_record(
        fitted_glb, fitted_record, "sanitized fitted TokenRig GLB"
    )
    if not isinstance(fitted_manifest_record, dict) or not isinstance(
        fitted_manifest_record.get("path"), str
    ):
        raise StaticAuditError("sanitized fitted manifest descriptor is missing")
    fitted_manifest = _regular_file(
        Path(fitted_manifest_record["path"]), "sanitized fitted TokenRig manifest"
    )
    _validate_record(
        fitted_manifest,
        fitted_manifest_record,
        "sanitized fitted TokenRig manifest",
    )
    fitted_authentication = authenticate_task3_inputs(
        asset_id=asset_id,
        source_glb=source_glb,
        tokenrig_glb=fitted_glb,
        tokenrig_manifest=fitted_manifest,
    )
    if fitted_authentication.get("manifest_schema") != FITTED_SCHEMA:
        raise StaticAuditError("sanitized input is not the authenticated fitted candidate")
    recorded_fitted_auth = input_record.get("fitted_authentication")
    required_fitted_auth_keys = (
        "source_glb_sha256",
        "source_manifest_sha256",
        "conditioning_glb_sha256",
        "conditioning_manifest_sha256",
        "tokenrig_glb_sha256",
        "tokenrig_manifest_sha256",
        "attempt_ledger_sha256",
        "orchestrator_runner_sha256",
        "delegated_runner_sha256",
        "server_hygiene_patch_sha256",
        "server_hygiene_load_audit_sha256",
        "server_hygiene_load_event_count",
    )
    if not isinstance(recorded_fitted_auth, dict) or any(
        recorded_fitted_auth.get(key) != fitted_authentication.get(key)
        for key in required_fitted_auth_keys
    ):
        raise StaticAuditError("sanitized fitted authentication snapshot changed")

    fitted_payload = _read_json_mapping(
        fitted_manifest, "sanitized fitted TokenRig manifest"
    )
    fitted_input = fitted_payload.get("input")
    fallback = (
        fitted_input.get("fallback_provenance")
        if isinstance(fitted_input, dict)
        else None
    )
    if (
        not isinstance(fitted_input, dict)
        or not isinstance(fallback, dict)
        or not _same_file_record(input_record.get("direct_glb"), fitted_input.get("glb"))
        or not _same_file_record(
            input_record.get("recovery_manifest"), fitted_input.get("manifest")
        )
        or not _same_file_record(
            original_record, fallback.get("original_source_glb")
        )
        or not _same_file_record(
            original_manifest_record, fallback.get("original_source_manifest")
        )
        or input_record.get("direct_failures") != fallback.get("static_failures")
    ):
        raise StaticAuditError("sanitized nested direct/fitted provenance changed")

    fitted_failures = input_record.get("fitted_failures")
    if not isinstance(fitted_failures, list) or len(fitted_failures) != 2:
        raise StaticAuditError("sanitized provenance must contain two fitted failures")
    fitted_failure_hashes = []
    fitted_failure_messages = []
    for index, record in enumerate(fitted_failures):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise StaticAuditError("sanitized fitted failure descriptor is missing")
        path = _regular_file(
            Path(record["path"]), f"sanitized fitted failure {index}"
        )
        fitted_failure_hashes.append(
            _validate_record(path, record, f"sanitized fitted failure {index}")
        )
        if path.stat().st_mode & 0o222:
            raise StaticAuditError("sanitized fitted failure evidence is mutable")
        payload = _read_json_mapping(path, f"sanitized fitted failure {index}")
        failure = payload.get("failure")
        if (
            payload.get("decision") != "rejected"
            or payload.get("readiness_bundle_published") is not False
            or not isinstance(failure, dict)
        ):
            raise StaticAuditError("sanitized fitted failure is not a rejection")
        fitted_failure_messages.append(str(failure.get("message", "")))
    if (
        "surface unique position count changed" not in fitted_failure_messages[0]
        or "UV seam duplicate vertex" not in fitted_failure_messages[1]
    ):
        raise StaticAuditError("sanitized fitted failure sequence changed")
    if input_record.get("fitted_failure_summary") != {
        "obsolete_exact_tuple_import_gate": "rejected",
        "ordered_fitted_skin_gate": "rejected_at_seam",
        "animation_authorized": False,
    }:
        raise StaticAuditError("sanitized fitted failure summary changed")
    prior_sanitation_failures = input_record.get("prior_sanitation_failures")
    if (
        not isinstance(prior_sanitation_failures, list)
        or len(prior_sanitation_failures) != 3
    ):
        raise StaticAuditError(
            "sanitized provenance must preserve its prior roundtrip rejection"
        )
    prior_sanitation_failure_hashes = []
    prior_failure_messages = (
        "surface unique position coverage changed",
        "roundtrip skin position coverage changed",
        "roundtrip skin weights changed",
    )
    for index, record in enumerate(prior_sanitation_failures):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise StaticAuditError("prior sanitation failure descriptor is missing")
        path = _regular_file(
            Path(record["path"]), f"prior sanitation failure {index}"
        )
        prior_sanitation_failure_hashes.append(
            _validate_record(path, record, f"prior sanitation failure {index}")
        )
        if path.stat().st_mode & 0o222:
            raise StaticAuditError("prior sanitation failure evidence is mutable")
        payload = _read_json_mapping(path, f"prior sanitation failure {index}")
        if (
            payload.get("decision") != "rejected"
            or payload.get("readiness_bundle_published") is not False
            or prior_failure_messages[index]
            not in str(payload.get("failure", {}).get("message", ""))
        ):
            raise StaticAuditError("prior sanitation failure sequence changed")

    code = manifest.get("code")
    if not isinstance(code, dict):
        raise StaticAuditError("sanitized code provenance is missing")
    code_hashes: dict[str, str] = {}
    descriptions = {
        "sanitizer": "sanitizer runner",
        "static_audit": "sanitized static-audit runner",
        "fitted_wrapper": "sanitized fitted wrapper",
        "delegated_base_runner": "sanitized delegated runner",
    }
    for key, description in descriptions.items():
        record = code.get(key)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise StaticAuditError(f"{description} descriptor is missing")
        path = _regular_file(Path(record["path"]), description)
        code_hashes[key] = _validate_record(path, record, description)
    if (
        code_hashes["fitted_wrapper"]
        != fitted_authentication.get("orchestrator_runner_sha256")
        or code_hashes["delegated_base_runner"]
        != fitted_authentication.get("delegated_runner_sha256")
    ):
        raise StaticAuditError("sanitized delegated code provenance changed")

    pre = manifest.get("pre_sanitation")
    sanitation = manifest.get("sanitation")
    if (
        not isinstance(pre, dict)
        or "UV seam duplicate vertex" not in str(pre.get("seam_rejection", ""))
        or "opposite-limb contamination" not in str(
            pre.get("bilateral_rejection", "")
        )
        or not isinstance(sanitation, dict)
        or sanitation.get("algorithm_version")
        != "tokenrig_side_transfer_seam_hybrid_export_floor_v3"
        or sanitation.get("inference_used") is not False
    ):
        raise StaticAuditError("sanitized algorithm/pre-rejection contract changed")
    vertex_count = sanitation.get("vertex_count")
    changed_count = sanitation.get("changed_vertex_count")
    changed_ratio = sanitation.get("changed_vertex_ratio")
    transferred_mass = sanitation.get("total_transferred_mass")
    accounting = sanitation.get("per_vertex_l1_accounting")
    seam_group_count = sanitation.get("seam_duplicate_group_count")
    seam_methods = sanitation.get("seam_reconciliation_method_counts")
    if (
        not isinstance(vertex_count, int)
        or not isinstance(changed_count, int)
        or not 0 < changed_count <= vertex_count
        or not isinstance(changed_ratio, (int, float))
        or abs(float(changed_ratio) - changed_count / vertex_count) > 1.0e-12
        or not isinstance(transferred_mass, (int, float))
        or not math.isfinite(float(transferred_mass))
        or float(transferred_mass) <= 0.0
        or not isinstance(sanitation.get("transferred_mass_by_bone_pair"), dict)
        or not sanitation["transferred_mass_by_bone_pair"]
        or accounting
        != {
            "vertex_count": vertex_count,
            "explicit_changed_record_count": changed_count,
            "implicit_unchanged_vertex_count": vertex_count - changed_count,
            "implicit_unchanged_l1_before_after": 0.0,
        }
        or not isinstance(seam_group_count, int)
        or seam_group_count <= 0
        or not isinstance(seam_methods, dict)
        or set(seam_methods) != {"weighted_average", "l1_medoid"}
        or any(not isinstance(value, int) or value < 0 for value in seam_methods.values())
        or sum(seam_methods.values()) != seam_group_count
        or sanitation.get("total_truncated_mass") != 0.0
        or sanitation.get("maximum_truncated_mass") != 0.0
        or not isinstance(
            sanitation.get("total_proposed_average_truncated_mass"),
            (int, float),
        )
        or not isinstance(
            sanitation.get("maximum_proposed_average_truncated_mass"),
            (int, float),
        )
        or float(sanitation["total_proposed_average_truncated_mass"]) < 0.0
        or float(sanitation["maximum_proposed_average_truncated_mass"]) < 0.0
        or not math.isfinite(
            float(sanitation["total_proposed_average_truncated_mass"])
        )
        or not math.isfinite(
            float(sanitation["maximum_proposed_average_truncated_mass"])
        )
    ):
        raise StaticAuditError("sanitized change/mass/truncation statistics are invalid")
    export_projection = sanitation.get("export_floor_projection")
    if not isinstance(export_projection, dict):
        raise StaticAuditError("sanitized export-floor projection is missing")
    projected_vertex_count = export_projection.get("projected_vertex_count")
    projected_component_count = export_projection.get("projected_component_count")
    projection_total = export_projection.get("total_added_mass")
    projection_maximum = export_projection.get("maximum_added_mass")
    projection_minimum_output = export_projection.get("minimum_output_weight")
    projection_minimum_applied = export_projection.get(
        "minimum_applied_blender_weight"
    )
    if (
        export_projection.get("policy")
        != "raise_droppable_support_to_next_float32_and_debit_largest_v1"
        or export_projection.get("blender_min_influence")
        != BLENDER_EXPORT_MIN_INFLUENCE
        or export_projection.get("safe_floor")
        != BLENDER_EXPORT_SAFE_WEIGHT_FLOOR
        or export_projection.get("maximum_added_mass_per_vertex_budget")
        != EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX
        or not isinstance(projected_vertex_count, int)
        or not 0 <= projected_vertex_count <= vertex_count
        or not isinstance(projected_component_count, int)
        or not projected_vertex_count <= projected_component_count <= 3 * projected_vertex_count
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in (
                projection_total,
                projection_maximum,
                projection_minimum_output,
                projection_minimum_applied,
            )
        )
        or float(projection_total) < 0.0
        or float(projection_maximum) < 0.0
        or float(projection_maximum) > EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX
        or float(projection_total)
        > projected_vertex_count * EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX
        or float(projection_minimum_output) <= BLENDER_EXPORT_MIN_INFLUENCE
        or float(projection_minimum_applied) <= BLENDER_EXPORT_MIN_INFLUENCE
    ):
        raise StaticAuditError("sanitized export-floor projection is invalid")
    for name in ("l1_all_vertices", "l1_changed_vertices"):
        distribution = sanitation.get(name)
        if not isinstance(distribution, dict):
            raise StaticAuditError("sanitized L1 distributions are missing")
        values = [distribution.get(key) for key in ("p50", "p95", "p99", "maximum")]
        if (
            any(not isinstance(value, (int, float)) for value in values)
            or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values)
            or any(float(left) > float(right) for left, right in zip(values, values[1:]))
        ):
            raise StaticAuditError("sanitized L1 distribution is invalid")
    seam = sanitation.get("seam_validation")
    if (
        not isinstance(seam, dict)
        or seam.get("weight_l1_tolerance") != SEAM_WEIGHT_L1_TOLERANCE
        or not isinstance(seam.get("maximum_weight_l1_error"), (int, float))
        or float(seam["maximum_weight_l1_error"]) > SEAM_WEIGHT_L1_TOLERANCE
    ):
        raise StaticAuditError("sanitized seam validation is not clean")
    bilateral = sanitation.get("bilateral_validation")
    if (
        not isinstance(bilateral, dict)
        or bilateral.get("contaminated_vertex_count") != 0
        or bilateral.get("tolerance") != OPPOSITE_LIMB_WEIGHT_TOLERANCE
        or not isinstance(
            bilateral.get("maximum_opposite_limb_weight"), (int, float)
        )
        or float(bilateral["maximum_opposite_limb_weight"])
        > OPPOSITE_LIMB_WEIGHT_TOLERANCE
    ):
        raise StaticAuditError("sanitized bilateral validation is not clean")

    validation = manifest.get("validation")
    required_passed = (
        "input_pbr",
        "input_raw_surface",
        "in_scene_mesh",
        "in_scene_surface",
        "in_scene_full_rest",
        "roundtrip_full_rest",
        "inverse_bind",
        "output_pbr",
        "output_raw_surface",
        "roundtrip",
    )
    if (
        not isinstance(validation, dict)
        or any(
            not isinstance(validation.get(key), dict)
            or validation[key].get("passed") is not True
            for key in required_passed
        )
        or not isinstance(
            validation.get("restored_root_matrix_maximum_error"), (int, float)
        )
        or abs(float(validation["restored_root_matrix_maximum_error"])) > 1.0e-12
        or not isinstance(validation.get("in_scene_rest"), dict)
        or validation["in_scene_rest"].get("bone_count") != 52
        or validation["in_scene_full_rest"].get("bone_count") != 52
        or validation["roundtrip_full_rest"].get("bone_count") != 52
        or validation["inverse_bind"].get("joint_count") != 52
    ):
        raise StaticAuditError("sanitized mesh/PBR/rest/roundtrip validation is incomplete")
    full_rest_metric_names = (
        "maximum_object_matrix_element_error",
        "maximum_head_error_m",
        "maximum_tail_error_m",
        "maximum_roll_axis_error",
        "maximum_roll_error_radians",
        "maximum_matrix_element_error",
    )
    for name in ("in_scene_full_rest", "roundtrip_full_rest"):
        contract = validation[name]
        metrics = [contract.get(metric) for metric in full_rest_metric_names]
        if (
            contract.get("tolerance") != FULL_REST_TOLERANCE
            or any(not isinstance(value, (int, float)) for value in metrics)
            or any(
                not math.isfinite(float(value))
                or float(value) < 0.0
                or float(value) > FULL_REST_TOLERANCE
                for value in metrics
            )
        ):
            raise StaticAuditError("sanitized full-rest validation is invalid")
    inverse_bind = validation["inverse_bind"]
    inverse_error = inverse_bind.get("maximum_matrix_element_error")
    if (
        inverse_bind.get("joint_order_unchanged") is not True
        or inverse_bind.get("tolerance") != INVERSE_BIND_MATRIX_TOLERANCE
        or not isinstance(inverse_bind.get("exact_matrices_unchanged"), bool)
        or not isinstance(inverse_error, (int, float))
        or not math.isfinite(float(inverse_error))
        or float(inverse_error) < 0.0
        or float(inverse_error) > INVERSE_BIND_MATRIX_TOLERANCE
    ):
        raise StaticAuditError("sanitized inverse-bind validation is invalid")

    artifacts = manifest.get("artifacts")
    change_record = (
        artifacts.get("weight_changes") if isinstance(artifacts, dict) else None
    )
    if not isinstance(change_record, dict) or not isinstance(
        change_record.get("path"), str
    ):
        raise StaticAuditError("sanitized weight changes descriptor is missing")
    changes_path = _regular_file(
        Path(change_record["path"]), "sanitized weight changes"
    )
    changes_hash = _validate_record(
        changes_path, change_record, "sanitized weight changes"
    )
    if (
        changes_path.parent != tokenrig_manifest.parent
        or changes_path.stat().st_mode & 0o222
    ):
        raise StaticAuditError(
            "sanitized weight changes must be immutable in the output directory"
        )
    observed_l1 = []
    observed_transferred_mass = 0.0
    observed_export_floor_mass = 0.0
    observed_export_floor_maximum = 0.0
    observed_export_floor_components = 0
    observed_export_floor_vertices = 0
    observed_export_floor_l1 = []
    previous_vertex_index = -1

    def validate_recorded_weights(value: Any, description: str) -> Mapping[str, Any]:
        if not isinstance(value, dict) or not 1 <= len(value) <= 4:
            raise StaticAuditError(
                f"sanitized weight changes {description} influence count is invalid"
            )
        weights = list(value.values())
        if (
            any(not isinstance(item, (int, float)) for item in weights)
            or any(not math.isfinite(float(item)) or float(item) <= 0.0 for item in weights)
            or abs(sum(float(item) for item in weights) - 1.0) > WEIGHT_SUM_TOLERANCE
        ):
            raise StaticAuditError(
                f"sanitized weight changes {description} weights are invalid"
            )
        return value

    try:
        change_lines = changes_path.read_text(encoding="utf-8").splitlines()
        for line in change_lines:
            change = json.loads(line)
            if not isinstance(change, dict):
                raise StaticAuditError("sanitized weight changes record is not an object")
            vertex_index = change.get("vertex_index")
            if (
                not isinstance(vertex_index, int)
                or vertex_index <= previous_vertex_index
                or vertex_index < 0
                or vertex_index >= vertex_count
            ):
                raise StaticAuditError(
                    "sanitized weight changes vertex indices are not unique and ordered"
                )
            previous_vertex_index = vertex_index
            before = validate_recorded_weights(change.get("before"), "before")
            after = validate_recorded_weights(change.get("after"), "after")
            recorded_l1 = change.get("l1_before_after")
            transferred = change.get("transferred_mass")
            floor_mass = change.get("export_floor_added_mass")
            floor_components = change.get("export_floor_component_count")
            computed_l1 = _weight_l1(before, after)
            if (
                not isinstance(recorded_l1, (int, float))
                or not math.isfinite(float(recorded_l1))
                or float(recorded_l1) <= 1.0e-15
                or abs(float(recorded_l1) - computed_l1) > 1.0e-12
                or not isinstance(transferred, (int, float))
                or not math.isfinite(float(transferred))
                or float(transferred) < 0.0
                or float(transferred) > 1.0 + WEIGHT_SUM_TOLERANCE
                or not isinstance(floor_mass, (int, float))
                or not math.isfinite(float(floor_mass))
                or not 0.0 <= float(floor_mass)
                <= EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX
                or not isinstance(floor_components, int)
                or not 0 <= floor_components <= 3
                or (floor_components == 0) != (float(floor_mass) == 0.0)
            ):
                raise StaticAuditError(
                    "sanitized weight changes L1/mass/export-floor is invalid"
                )
            observed_l1.append(float(recorded_l1))
            observed_transferred_mass += float(transferred)
            observed_export_floor_mass += float(floor_mass)
            observed_export_floor_maximum = max(
                observed_export_floor_maximum, float(floor_mass)
            )
            observed_export_floor_components += floor_components
            observed_export_floor_vertices += int(floor_components > 0)
            observed_export_floor_l1.append(2.0 * float(floor_mass))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAuditError(f"invalid sanitized weight changes: {exc}") from exc
    if len(change_lines) != changed_count:
        raise StaticAuditError("sanitized weight changes line count changed")

    def observed_distribution(values: Sequence[float]) -> dict[str, float]:
        ordered = sorted(float(value) for value in values)

        def percentile(fraction: float) -> float:
            if not ordered:
                return 0.0
            position = (len(ordered) - 1) * fraction
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return ordered[lower]
            remainder = position - lower
            return ordered[lower] * (1.0 - remainder) + ordered[upper] * remainder

        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "maximum": ordered[-1] if ordered else 0.0,
        }

    observed_changed_distribution = observed_distribution(observed_l1)
    observed_all_distribution = observed_distribution(
        [0.0] * (vertex_count - changed_count) + observed_l1
    )
    for recorded_name, observed in (
        ("l1_changed_vertices", observed_changed_distribution),
        ("l1_all_vertices", observed_all_distribution),
    ):
        recorded = sanitation.get(recorded_name)
        if not isinstance(recorded, dict) or any(
            abs(float(recorded.get(key, math.nan)) - value) > 1.0e-12
            for key, value in observed.items()
        ):
            raise StaticAuditError("sanitized weight changes L1 aggregate changed")
    observed_export_distribution = observed_distribution(
        [0.0] * (vertex_count - changed_count) + observed_export_floor_l1
    )
    recorded_export_distribution = export_projection.get("l1_all_vertices")
    if (
        not isinstance(recorded_export_distribution, dict)
        or any(
            abs(float(recorded_export_distribution.get(key, math.nan)) - value)
            > 1.0e-15
            for key, value in observed_export_distribution.items()
        )
        or observed_export_floor_vertices != projected_vertex_count
        or observed_export_floor_components != projected_component_count
        or abs(observed_export_floor_mass - float(projection_total)) > 1.0e-15
        or abs(observed_export_floor_maximum - float(projection_maximum)) > 1.0e-15
    ):
        raise StaticAuditError(
            "sanitized weight changes export-floor aggregate changed"
        )
    pair_masses = sanitation.get("transferred_mass_by_bone_pair")
    if (
        abs(observed_transferred_mass - float(transferred_mass)) > 1.0e-9
        or not isinstance(pair_masses, dict)
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in pair_masses.values()
        )
        or abs(
            sum(float(value) for value in pair_masses.values())
            - float(transferred_mass)
        )
        > 1.0e-9
    ):
        raise StaticAuditError("sanitized weight changes transferred mass changed")
    seam_group_record = (
        artifacts.get("seam_groups") if isinstance(artifacts, dict) else None
    )
    if not isinstance(seam_group_record, dict) or not isinstance(
        seam_group_record.get("path"), str
    ):
        raise StaticAuditError("sanitized seam-group audit descriptor is missing")
    seam_group_path = _regular_file(
        Path(seam_group_record["path"]), "sanitized seam-group audit"
    )
    seam_group_hash = _validate_record(
        seam_group_path, seam_group_record, "sanitized seam-group audit"
    )
    if (
        seam_group_path.parent != tokenrig_manifest.parent
        or seam_group_path.stat().st_mode & 0o222
    ):
        raise StaticAuditError(
            "sanitized seam-group audit must be immutable in the output directory"
        )
    observed_methods = {"weighted_average": 0, "l1_medoid": 0}
    observed_proposed_total = 0.0
    observed_proposed_maximum = 0.0
    try:
        seam_group_lines = seam_group_path.read_text(encoding="utf-8").splitlines()
        for expected_index, line in enumerate(seam_group_lines):
            group = json.loads(line)
            if not isinstance(group, dict):
                raise StaticAuditError("sanitized seam-group record is not an object")
            method = group.get("method")
            method_reason = group.get("method_reason")
            proposed = group.get("proposed_average_truncated_mass")
            applied = group.get("applied_truncated_mass")
            representative_index = group.get("representative_vertex_index")
            medoid_index = group.get("medoid_vertex_index")
            union_count = group.get("union_influence_count")
            maximum_member_l1 = group.get("maximum_member_l1_to_reconciled")
            total_member_l1 = group.get("total_member_l1_to_reconciled")
            if (
                group.get("group_index") != expected_index
                or method not in observed_methods
                or not isinstance(group.get("vertex_count"), int)
                or group["vertex_count"] < 2
                or not isinstance(union_count, int)
                or union_count < 1
                or not isinstance(representative_index, int)
                or not 0 <= representative_index < vertex_count
                or not isinstance(proposed, (int, float))
                or not math.isfinite(float(proposed))
                or float(proposed) < 0.0
                or applied != 0.0
                or not isinstance(maximum_member_l1, (int, float))
                or not isinstance(total_member_l1, (int, float))
                or not math.isfinite(float(maximum_member_l1))
                or not math.isfinite(float(total_member_l1))
                or float(maximum_member_l1) < 0.0
                or float(total_member_l1) < float(maximum_member_l1)
                or (
                    method == "l1_medoid"
                    and (
                        not isinstance(medoid_index, int)
                        or not 0 <= medoid_index < vertex_count
                        or (
                            union_count > 4
                            and method_reason != "influence_union_exceeds_four"
                        )
                        or (
                            union_count <= 4
                            and method_reason != "export_floor_guard"
                        )
                    )
                )
                or (
                    method == "weighted_average"
                    and (
                        union_count > 4
                        or medoid_index is not None
                        or method_reason != "influence_union_within_four"
                    )
                )
            ):
                raise StaticAuditError("sanitized seam-group audit record is invalid")
            observed_methods[method] += 1
            observed_proposed_total += float(proposed)
            observed_proposed_maximum = max(
                observed_proposed_maximum, float(proposed)
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAuditError(f"invalid sanitized seam-group audit: {exc}") from exc
    aggregate_tolerance = 1.0e-12 * max(1.0, observed_proposed_total)
    if (
        len(seam_group_lines) != seam_group_count
        or observed_methods != seam_methods
        or abs(
            observed_proposed_total
            - float(sanitation["total_proposed_average_truncated_mass"])
        )
        > aggregate_tolerance
        or abs(
            observed_proposed_maximum
            - float(sanitation["maximum_proposed_average_truncated_mass"])
        )
        > aggregate_tolerance
    ):
        raise StaticAuditError("sanitized seam-group audit aggregates changed")
    tokenrig_hash = _validate_record(
        tokenrig_glb, manifest.get("output"), "sanitized TokenRig output"
    )
    read_glb(source_glb)
    read_glb(tokenrig_glb)
    return {
        "asset_id": asset_id,
        "attempt": manifest.get("attempt"),
        "manifest_schema": SANITIZED_SCHEMA,
        "task3_gate_status": "failed",
        "recovered_candidate": False,
        "fitted_candidate": False,
        "sanitized_candidate": True,
        "prior_direct_static_failure_sha256": fitted_authentication.get(
            "prior_direct_static_failure_sha256"
        ),
        "prior_fitted_static_failure_sha256": fitted_failure_hashes,
        "prior_sanitation_failure_sha256": prior_sanitation_failure_hashes,
        "source_glb_sha256": source_hash,
        "source_manifest_sha256": source_manifest_hash,
        "conditioning_glb_sha256": fitted_hash,
        "conditioning_manifest_sha256": fitted_authentication.get(
            "tokenrig_manifest_sha256"
        ),
        "tokenrig_glb_sha256": tokenrig_hash,
        "tokenrig_manifest_sha256": sha256_file(tokenrig_manifest),
        "sanitizer_runner_sha256": code_hashes["sanitizer"],
        "static_audit_runner_sha256": code_hashes["static_audit"],
        "orchestrator_runner_sha256": code_hashes["fitted_wrapper"],
        "delegated_runner_sha256": code_hashes["delegated_base_runner"],
        "weight_changes_sha256": changes_hash,
        "seam_groups_sha256": seam_group_hash,
        "source_front": SOURCE_FRONT,
        "canonical_front": CANONICAL_FRONT,
    }


def authenticate_task3_inputs(
    *,
    asset_id: str,
    source_glb: Path,
    tokenrig_glb: Path,
    tokenrig_manifest: Path,
) -> dict[str, Any]:
    source_glb = _regular_file(source_glb, "original Pixal PBR GLB")
    tokenrig_glb = _regular_file(tokenrig_glb, "TokenRig output GLB")
    tokenrig_manifest = _regular_file(tokenrig_manifest, "TokenRig manifest")
    try:
        manifest = json.loads(tokenrig_manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAuditError(f"invalid TokenRig manifest: {exc}") from exc
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    if schema not in {TASK3_SCHEMA, RECOVERY_SCHEMA, FITTED_SCHEMA, SANITIZED_SCHEMA}:
        raise StaticAuditError(
            "TokenRig manifest schema must be one of "
            f"{TASK3_SCHEMA!r}, {RECOVERY_SCHEMA!r}, {FITTED_SCHEMA!r}, "
            f"or {SANITIZED_SCHEMA!r}"
        )
    if manifest.get("asset_id") != asset_id:
        raise StaticAuditError("TokenRig manifest asset_id mismatch")
    if manifest.get("source_front") != SOURCE_FRONT or manifest.get("canonical_front") != CANONICAL_FRONT:
        raise StaticAuditError("TokenRig manifest front-axis contract mismatch")
    if schema == SANITIZED_SCHEMA:
        canonical_axis_contract(
            source_front=manifest["source_front"], prior_transform_count=0
        )
        return _authenticate_sanitized_inputs(
            asset_id=asset_id,
            source_glb=source_glb,
            tokenrig_glb=tokenrig_glb,
            tokenrig_manifest=tokenrig_manifest,
            manifest=manifest,
        )
    recovered_candidate = schema == RECOVERY_SCHEMA
    fitted_candidate = schema == FITTED_SCHEMA
    fitted_failure_hashes: list[str] | None = None
    fitted_execution: dict[str, Any] = {}
    if recovered_candidate:
        recovery = manifest.get("recovery")
        failed_evidence = recovery.get("failed_evidence") if isinstance(recovery, dict) else None
        failed_files = failed_evidence.get("files") if isinstance(failed_evidence, dict) else None
        failed_output = (
            failed_files.get("tokenrig_transfer.glb")
            if isinstance(failed_files, dict)
            else None
        )
        output_record = manifest.get("output")
        required_recovery = (
            manifest.get("state_classification")
            == "research_candidate_recovered_from_hygiene_assertion"
            and manifest.get("task3_gate_status") == "failed"
            and manifest.get("pbr_validation_status") == "pending_static_audit"
            and isinstance(recovery, dict)
            and recovery.get("task3_passed") is False
            and recovery.get("returncode") == 0
            and isinstance(recovery.get("failure_stage"), str)
            and bool(recovery.get("failure_stage"))
            and isinstance(recovery.get("error"), dict)
            and isinstance(recovery.get("upstream_clean_bpy"), dict)
            and recovery["upstream_clean_bpy"].get(
                "bpyparser_load_calls_clean_before_import"
            )
            is True
            and isinstance(failed_output, dict)
            and isinstance(output_record, dict)
            and all(
                failed_output.get(field) == output_record.get(field)
                for field in ("path", "sha256", "size_bytes")
            )
        )
        if not required_recovery:
            raise StaticAuditError(
                "recovery manifest does not prove an honest failed Task 3 gate"
            )
    if fitted_candidate:
        parameters = manifest.get("inference_parameters")
        fitted = manifest.get("fitted_skeleton")
        input_value = manifest.get("input")
        fallback_provenance = (
            input_value.get("fallback_provenance")
            if isinstance(input_value, dict)
            else None
        )
        if (
            manifest.get("base_runner_schema") != TASK3_SCHEMA
            or manifest.get("attempt") != "fitted_skeleton_transfer"
            or manifest.get("task3_direct_gate_status") != "failed"
            or manifest.get("static_audit_status") != "pending_fitted_static_audit"
            or manifest.get("pbr_validation_status") != "pending_static_audit"
            or manifest.get("animation_authorized") is not False
            or not isinstance(parameters, dict)
            or parameters.get("use_skeleton") is not True
            or parameters.get("use_transfer") is not True
            or not isinstance(fitted, dict)
            or fitted.get("use_skeleton_input") is not True
            or not isinstance(fallback_provenance, dict)
            or fallback_provenance.get("animation_authorized") is not False
        ):
            raise StaticAuditError(
                "fitted manifest does not prove the forced skeleton fallback"
            )
    prior_count = manifest.get("axis_transform_count", 0)
    if manifest.get("axis_transform_applied") is True:
        prior_count = max(1, prior_count)
    canonical_axis_contract(source_front=manifest["source_front"], prior_transform_count=prior_count)

    input_record = manifest.get("input", {})
    input_glb_record = input_record.get("glb") if isinstance(input_record, dict) else None
    input_manifest_record = (
        input_record.get("manifest") if isinstance(input_record, dict) else None
    )
    if fitted_candidate:
        fallback_provenance = input_record["fallback_provenance"]
        source_input_record = fallback_provenance.get("original_source_glb")
        source_manifest_input_record = fallback_provenance.get(
            "original_source_manifest"
        )
        conditioning_record = input_glb_record
        conditioning_manifest_record = input_manifest_record
        fitted_source = manifest["fitted_skeleton"].get("conditioning_source")
        if (
            not isinstance(conditioning_record, dict)
            or not isinstance(fitted_source, dict)
            or any(
                conditioning_record.get(field) != fitted_source.get(field)
                for field in ("path", "sha256", "size_bytes")
            )
        ):
            raise StaticAuditError("fitted conditioning source descriptor changed")
        conditioning_path = _regular_file(
            Path(conditioning_record.get("path", "")),
            "fitted conditioning TokenRig GLB",
        )
        conditioning_hash = _validate_record(
            conditioning_path, conditioning_record, "fitted conditioning TokenRig GLB"
        )
        fitted_execution = _authenticate_fitted_execution(
            manifest=manifest,
            manifest_path=tokenrig_manifest,
            conditioning_path=conditioning_path,
        )
        if not isinstance(conditioning_manifest_record, dict):
            raise StaticAuditError("fitted conditioning recovery manifest is missing")
        conditioning_manifest_path = _regular_file(
            Path(conditioning_manifest_record.get("path", "")),
            "fitted conditioning recovery manifest",
        )
        conditioning_manifest_hash = _validate_record(
            conditioning_manifest_path,
            conditioning_manifest_record,
            "fitted conditioning recovery manifest",
        )
        failure_records = fallback_provenance.get("static_failures")
        if not isinstance(failure_records, list) or len(failure_records) != 2:
            raise StaticAuditError("fitted fallback must authenticate two static failures")
        failure_messages = []
        fitted_failure_hashes = []
        for index, record in enumerate(failure_records):
            if not isinstance(record, dict):
                raise StaticAuditError("fitted static failure descriptor is missing")
            failure_path = _regular_file(
                Path(record.get("path", "")), f"fitted static failure {index}"
            )
            fitted_failure_hashes.append(
                _validate_record(
                    failure_path, record, f"fitted static failure {index}"
                )
            )
            if failure_path.stat().st_mode & 0o222:
                raise StaticAuditError("fitted static failure evidence is mutable")
            try:
                failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StaticAuditError(f"fitted static failure is invalid: {exc}")
            if (
                not isinstance(failure_payload, dict)
                or failure_payload.get("decision") != "rejected"
                or failure_payload.get("readiness_bundle_published") is not False
                or not isinstance(failure_payload.get("failure"), dict)
            ):
                raise StaticAuditError("fitted static failure is not a rejection")
            failure_messages.append(str(failure_payload["failure"].get("message", "")))
        if (
            "raw GLB triangle count changed" not in failure_messages[0]
            or "opposite-limb contamination" not in failure_messages[1]
        ):
            raise StaticAuditError("fitted fallback static failure sequence changed")
        input_glb_record = source_input_record
        input_manifest_record = source_manifest_input_record
    else:
        conditioning_hash = None
        conditioning_manifest_hash = None
    source_hash = _validate_record(source_glb, input_glb_record, "original Pixal PBR input")
    if not isinstance(input_manifest_record, dict) or not isinstance(
        input_manifest_record.get("path"), str
    ):
        raise StaticAuditError("original Pixal input manifest record is missing")
    source_manifest = _regular_file(
        Path(input_manifest_record["path"]), "original Pixal input manifest"
    )
    if source_manifest.parent != source_glb.parent:
        raise StaticAuditError("original Pixal GLB and manifest must share one directory")
    source_manifest_hash = _validate_record(
        source_manifest,
        input_manifest_record,
        "original Pixal input manifest",
    )
    try:
        source_manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAuditError(f"invalid original Pixal input manifest: {exc}") from exc
    if not isinstance(source_manifest_payload, dict):
        raise StaticAuditError("original Pixal input manifest root must be an object")
    tokenrig_hash = _validate_record(tokenrig_glb, manifest.get("output"), "TokenRig output")
    # Parse both now so malformed GLBs fail before Blender or staging starts.
    read_glb(source_glb)
    read_glb(tokenrig_glb)
    return {
        "asset_id": asset_id,
        "attempt": manifest.get("attempt"),
        "manifest_schema": schema,
        "task3_gate_status": manifest.get(
            "task3_gate_status",
            manifest.get("task3_direct_gate_status", "not_reported"),
        ),
        "recovered_candidate": recovered_candidate,
        "fitted_candidate": fitted_candidate,
        "sanitized_candidate": False,
        "prior_direct_static_failure_sha256": fitted_failure_hashes,
        "source_glb_sha256": source_hash,
        "source_manifest_sha256": source_manifest_hash,
        "conditioning_glb_sha256": conditioning_hash,
        "conditioning_manifest_sha256": conditioning_manifest_hash,
        "tokenrig_glb_sha256": tokenrig_hash,
        "tokenrig_manifest_sha256": sha256_file(tokenrig_manifest),
        "source_front": SOURCE_FRONT,
        "canonical_front": CANONICAL_FRONT,
        **fitted_execution,
    }


def _mesh_bounds(mesh: Any) -> dict[str, tuple[float, float, float]]:
    positions = mesh_world_positions(mesh)
    if not positions:
        raise StaticAuditError("cannot render an empty runtime mesh")
    minimum = tuple(min(position[axis] for position in positions) for axis in range(3))
    maximum = tuple(max(position[axis] for position in positions) for axis in range(3))
    dimensions = tuple(maximum[axis] - minimum[axis] for axis in range(3))
    if any(not math.isfinite(value) for value in (*minimum, *maximum, *dimensions)):
        raise StaticAuditError("runtime render bounds are non-finite")
    if max(dimensions) <= 1.0e-8:
        raise StaticAuditError("runtime render bounds are degenerate")
    center = tuple((minimum[axis] + maximum[axis]) * 0.5 for axis in range(3))
    return {"minimum": minimum, "maximum": maximum, "dimensions": dimensions, "center": center}


def _look_at(obj: Any, target: Any) -> None:
    delta = target - obj.location
    if float(delta.length) <= 1.0e-9:
        raise StaticAuditError("review camera/light coincides with its target")
    obj.rotation_euler = delta.to_track_quat("-Z", "Y").to_euler()


def _review_material(bpy: Any, name: str, color: Sequence[float]) -> Any:
    rgba = tuple(float(value) for value in color)
    if len(rgba) != 4 or any(not math.isfinite(value) for value in rgba):
        raise StaticAuditError(f"invalid review material color for {name!r}")
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = rgba
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is None:
        raise StaticAuditError("Blender Principled BSDF node is unavailable")
    shader.inputs["Base Color"].default_value = rgba
    shader.inputs["Roughness"].default_value = 0.72
    emission = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    if emission is not None:
        emission.default_value = rgba
    emission_strength = shader.inputs.get("Emission Strength")
    if emission_strength is not None:
        emission_strength.default_value = 0.18
    return material


def _configure_review_scene(bpy: Any, mesh: Any) -> tuple[Any, dict[str, Any]]:
    from mathutils import Vector

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    if scene.world is None:
        scene.world = bpy.data.worlds.new("tokenrig_static_world")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is None:
        raise StaticAuditError("Blender world Background node is unavailable")
    background.inputs["Color"].default_value = (0.025, 0.030, 0.040, 1.0)
    background.inputs["Strength"].default_value = 0.22

    bounds = _mesh_bounds(mesh)
    center = Vector(bounds["center"])
    span = max(bounds["dimensions"])
    camera_data = bpy.data.cameras.new("tokenrig_static_camera")
    camera_data.type = "ORTHO"
    camera_data.lens = 52.0
    camera_data.clip_start = max(1.0e-4, span * 0.002)
    camera_data.clip_end = max(100.0, span * 20.0)
    camera = bpy.data.objects.new("tokenrig_static_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    light_specs = (
        ("key", (-1.8, -2.2, 2.4), 850.0, 4.0),
        ("fill", (2.0, -1.0, 1.3), 520.0, 3.0),
        ("rim", (0.7, 2.2, 2.0), 700.0, 3.0),
    )
    for name, direction, energy, size in light_specs:
        data = bpy.data.lights.new(f"tokenrig_static_{name}", type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = max(0.25, span * size)
        light = bpy.data.objects.new(f"tokenrig_static_{name}", data)
        bpy.context.collection.objects.link(light)
        light.location = center + Vector(direction).normalized() * max(2.0, span * 2.5)
        _look_at(light, center)

    floor_size = max(2.0, span * 2.8)
    bpy.ops.mesh.primitive_plane_add(
        size=floor_size,
        location=(bounds["center"][0], bounds["center"][1], bounds["minimum"][2] - 0.003),
    )
    floor = bpy.context.object
    floor.name = "tokenrig_static_floor"
    floor.data.materials.append(
        _review_material(bpy, "tokenrig_static_floor_material", (0.12, 0.14, 0.18, 1.0))
    )
    return camera, bounds


def _render_view(
    *,
    bpy: Any,
    camera: Any,
    bounds: Mapping[str, Sequence[float]],
    direction: Sequence[float],
    frame_span: float,
    output_path: Path,
) -> dict[str, Any]:
    from mathutils import Vector

    center = Vector(bounds["center"])
    vector = Vector(tuple(float(value) for value in direction))
    if float(vector.length) <= 1.0e-9:
        raise StaticAuditError("review view direction is zero")
    span = max(float(value) for value in bounds["dimensions"])
    camera.location = center + vector.normalized() * max(3.0, span * 4.0)
    _look_at(camera, center)
    camera.data.ortho_scale = max(float(frame_span) * 1.24, span * 0.12, 1.0e-3)
    bpy.context.scene.render.filepath = str(output_path)
    result = bpy.ops.render.render(write_still=True)
    if "FINISHED" not in result or not output_path.is_file() or output_path.stat().st_size <= 8:
        raise StaticAuditError(f"static evidence render failed: {output_path.name}")
    if output_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise StaticAuditError(f"static evidence is not a PNG: {output_path.name}")
    return {
        "filename": output_path.name,
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "direction": tuple(float(value) for value in direction),
        "orthographic_scale": float(camera.data.ortho_scale),
    }


def _projected_skeleton_guides(
    bpy: Any,
    armature: Any,
    semantics: Mapping[str, Any],
    bounds: Mapping[str, Sequence[float]],
) -> list[Any]:
    from mathutils import Vector

    colors = {
        "axial": (1.0, 0.78, 0.10, 1.0),
        "left_arm": (0.95, 0.18, 0.12, 1.0),
        "right_arm": (0.12, 0.45, 1.0, 1.0),
        "left_leg": (0.24, 0.92, 0.30, 1.0),
        "right_leg": (0.90, 0.22, 0.86, 1.0),
        "other": (0.92, 0.92, 0.96, 1.0),
    }
    materials = {
        name: _review_material(bpy, f"tokenrig_skeleton_{name}", color)
        for name, color in colors.items()
    }
    bone_to_chain = {
        bone_name: chain_name
        for chain_name, bone_names in semantics["chains"].items()
        for bone_name in bone_names
    }
    front_plane = float(bounds["minimum"][1]) - max(bounds["dimensions"]) * 0.035
    radius = max(bounds["dimensions"]) * 0.0055
    guides = []
    for bone in armature.data.bones:
        head_world = armature.matrix_world @ bone.head_local
        tail_world = armature.matrix_world @ bone.tail_local
        start = Vector((float(head_world.x), front_plane, float(head_world.z)))
        end = Vector((float(tail_world.x), front_plane, float(tail_world.z)))
        segment = end - start
        length = float(segment.length)
        if length <= 1.0e-7:
            continue
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=8,
            radius=radius,
            depth=length,
            location=(start + end) * 0.5,
        )
        guide = bpy.context.object
        guide.name = f"tokenrig_skeleton_guide_{bone.name}"
        guide.rotation_mode = "QUATERNION"
        guide.rotation_quaternion = segment.to_track_quat("Z", "Y")
        guide.data.materials.append(materials[bone_to_chain.get(bone.name, "other")])
        guides.append(guide)
    if not guides:
        raise StaticAuditError("could not construct projected skeleton evidence")
    return guides


def _render_weight_evidence(
    *,
    bpy: Any,
    mesh: Any,
    armature: Any,
    semantics: Mapping[str, Any],
    camera: Any,
    bounds: Mapping[str, Sequence[float]],
    output_path: Path,
) -> dict[str, Any]:
    chain_order = ("axial", "left_arm", "right_arm", "left_leg", "right_leg")
    colors = (
        (1.0, 0.78, 0.10, 1.0),
        (0.95, 0.18, 0.12, 1.0),
        (0.12, 0.45, 1.0, 1.0),
        (0.24, 0.92, 0.30, 1.0),
        (0.90, 0.22, 0.86, 1.0),
    )
    weights, _ = extract_vertex_weights(mesh, armature)
    chain_bones = [set(semantics["chains"][name]) for name in chain_order]
    materials = [
        _review_material(bpy, f"tokenrig_weights_{name}", color)
        for name, color in zip(chain_order, colors)
    ]
    original_materials = list(mesh.data.materials)
    original_indices = [int(polygon.material_index) for polygon in mesh.data.polygons]
    try:
        mesh.data.materials.clear()
        for material in materials:
            mesh.data.materials.append(material)
        for polygon in mesh.data.polygons:
            scores = [
                sum(
                    float(weights[vertex_index].get(bone_name, 0.0))
                    for vertex_index in polygon.vertices
                    for bone_name in bones
                )
                for bones in chain_bones
            ]
            polygon.material_index = max(range(len(scores)), key=scores.__getitem__)
        bpy.context.view_layer.update()
        return _render_view(
            bpy=bpy,
            camera=camera,
            bounds=bounds,
            direction=(0.0, -1.0, 0.0),
            frame_span=max(bounds["dimensions"][0], bounds["dimensions"][2]),
            output_path=output_path,
        )
    finally:
        mesh.data.materials.clear()
        for material in original_materials:
            mesh.data.materials.append(material)
        for polygon, material_index in zip(mesh.data.polygons, original_indices):
            polygon.material_index = material_index
        bpy.context.view_layer.update()


def render_static_evidence(
    *,
    bpy: Any,
    armature: Any,
    mesh: Any,
    semantics: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    camera, bounds = _configure_review_scene(bpy, mesh)
    dimensions = bounds["dimensions"]
    views = {}
    specifications = (
        ("bind_front.png", (0.0, -1.0, 0.0), max(dimensions[0], dimensions[2])),
        ("bind_back.png", (0.0, 1.0, 0.0), max(dimensions[0], dimensions[2])),
        ("bind_side.png", (1.0, 0.0, 0.0), max(dimensions[1], dimensions[2])),
        ("bind_top.png", (0.0, 0.0, 1.0), max(dimensions[0], dimensions[1])),
        ("texture_compare.png", (0.65, -1.0, 0.18), max(dimensions[0], dimensions[2])),
    )
    for filename, direction, frame_span in specifications:
        views[filename] = _render_view(
            bpy=bpy,
            camera=camera,
            bounds=bounds,
            direction=direction,
            frame_span=frame_span,
            output_path=output_dir / filename,
        )

    guides = _projected_skeleton_guides(bpy, armature, semantics, bounds)
    try:
        views["skeleton_overlay.png"] = _render_view(
            bpy=bpy,
            camera=camera,
            bounds=bounds,
            direction=(0.0, -1.0, 0.0),
            frame_span=max(dimensions[0], dimensions[2]),
            output_path=output_dir / "skeleton_overlay.png",
        )
    finally:
        for guide in guides:
            bpy.data.objects.remove(guide, do_unlink=True)
    views["weights_contact.png"] = _render_weight_evidence(
        bpy=bpy,
        mesh=mesh,
        armature=armature,
        semantics=semantics,
        camera=camera,
        bounds=bounds,
        output_path=output_dir / "weights_contact.png",
    )
    return {
        "passed": True,
        "renderer": bpy.context.scene.render.engine,
        "resolution": [
            bpy.context.scene.render.resolution_x,
            bpy.context.scene.render.resolution_y,
        ],
        "views": views,
    }


def write_joint_hierarchy(
    *,
    path: Path,
    bones: Sequence[BoneRecord],
    semantics: Mapping[str, Any],
) -> None:
    semantic_labels: dict[str, list[str]] = {}
    for label, value in semantics["semantic_bones"].items():
        values = value if isinstance(value, list) else [value]
        for index, bone_name in enumerate(values):
            suffix = f"[{index}]" if len(values) > 1 else ""
            semantic_labels.setdefault(bone_name, []).append(f"{label}{suffix}")
    lines = [
        "TokenRig static joint hierarchy",
        f"root={next(bone.name for bone in bones if bone.parent is None)}",
        f"bone_count={len(bones)}",
        "format: bone<TAB>parent<TAB>world_rest_head_xyz<TAB>semantic_labels",
    ]
    for bone in bones:
        head = ",".join(f"{float(value):.9f}" for value in bone.head)
        labels = ",".join(sorted(semantic_labels.get(bone.name, []))) or "-"
        lines.append(f"{bone.name}\t{bone.parent or '-'}\t{head}\t{labels}")
    _write_exclusive(path, ("\n".join(lines) + "\n").encode("utf-8"))


def run_blender_audit(
    *,
    source_glb: Path,
    tokenrig_glb: Path,
    staging_dir: Path,
) -> dict[str, Any]:
    import bpy
    from mathutils import Matrix

    source_parsed = read_glb(source_glb)
    tokenrig_parsed = read_glb(tokenrig_glb)
    raw_triangle_validation = analyze_raw_serialization_equivalence(
        source_parsed, tokenrig_parsed
    )
    source_pbr = pbr_payload_contract(source_parsed)
    tokenrig_pbr = pbr_payload_contract(tokenrig_parsed)
    input_pbr_validation = compare_pbr_payloads(source_pbr, tokenrig_pbr)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    imported = bpy.ops.import_scene.gltf(filepath=str(source_glb))
    if "FINISHED" not in imported:
        raise StaticAuditError("could not import the original Pixal PBR GLB")
    source_import_helpers = remove_gltf_import_helpers(bpy)
    source_mesh = identify_source_mesh(bpy)
    source_mesh_contract = capture_blender_mesh_contract(source_mesh)
    source_surface_reference = capture_blender_surface_reference(source_mesh)
    original_source_floor_z = mesh_floor_z(source_mesh)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    imported = bpy.ops.import_scene.gltf(filepath=str(tokenrig_glb))
    if "FINISHED" not in imported:
        raise StaticAuditError("could not import the TokenRig output GLB")
    tokenrig_import_helpers = remove_gltf_import_helpers(bpy)
    armature, mesh, orphans = identify_exact_runtime(bpy)
    removed_orphans = remove_proven_runtime_orphans(bpy, orphans)
    tokenrig_mesh_contract = capture_blender_mesh_contract(mesh)
    tokenrig_surface_reference = capture_blender_surface_reference(mesh)
    input_mesh_validation = compare_mesh_contracts(
        source_mesh_contract,
        tokenrig_mesh_contract,
        allow_serialization_splits=True,
    )
    input_surface_validation = compare_surface_references(
        source_surface_reference, tokenrig_surface_reference
    )
    del source_surface_reference, tokenrig_surface_reference
    source_floor_z = mesh_floor_z(mesh)
    if abs(source_floor_z - original_source_floor_z) > SEAM_POSITION_TOLERANCE_M:
        raise StaticAuditError(
            "TokenRig changed the ungrounded Pixal bind floor: "
            f"source={original_source_floor_z} output={source_floor_z}"
        )

    axis = canonical_axis_contract(source_front=SOURCE_FRONT, prior_transform_count=0)
    grounding = ground_bind_contract(source_floor_z=source_floor_z, prior_transform_count=0)
    if axis["transform_count"] != 1 or grounding["transform_count"] != 1:
        raise StaticAuditError("canonical yaw and grounding must each occur exactly once")
    roots = runtime_roots({armature, mesh})
    canonical_yaw = Matrix.Rotation(math.pi, 4, "Z")
    ground_translation = Matrix.Translation((0.0, 0.0, grounding["ground_translation_z"]))
    canonical_ground = ground_translation @ canonical_yaw
    for root in roots:
        root.matrix_world = canonical_ground @ root.matrix_world
    bpy.context.view_layer.update()
    post_floor_z = mesh_floor_z(mesh)
    if abs(post_floor_z) > SEAM_POSITION_TOLERANCE_M:
        raise StaticAuditError(
            f"runtime closure grounding did not produce floor Z=0: {post_floor_z}"
        )
    grounding = dict(grounding)
    grounding.update(
        {
            "post_floor_z": post_floor_z,
            "runtime_root_count": len(roots),
            "runtime_root_names": sorted(root.name for root in roots),
            "closure_operation_count": 2,
        }
    )

    canonical_mesh_contract = capture_blender_mesh_contract(mesh)
    canonical_surface_reference = capture_blender_surface_reference(mesh)
    bones = bone_records_from_armature(armature)
    hierarchy = validate_hierarchy(bones)
    semantics = resolve_five_semantic_chains(bones)
    weights, positions = extract_vertex_weights(mesh, armature)
    weight_validation = validate_vertex_weights(
        weights, bone_names={bone.name for bone in bones}
    )
    seam_validation = validate_seam_weights(positions, weights)
    bilateral_validation = validate_bilateral_contamination(
        positions, weights, semantics["chains"]
    )

    bind_path = staging_dir / "bind_pose.glb"
    export_bind_pose_glb(bpy, armature, mesh, bind_path)
    exported_pbr_validation = compare_pbr_payloads(
        source_pbr, pbr_payload_contract(read_glb(bind_path))
    )
    armature, mesh, roundtrip = roundtrip_validate_bind(
        bpy=bpy,
        glb_path=bind_path,
        source_pbr=source_pbr,
        expected_mesh=canonical_mesh_contract,
        expected_surface=canonical_surface_reference,
        expected_bones=bones,
        expected_positions=positions,
        expected_weights=weights,
        expected_semantics=semantics,
    )
    roundtrip_bones = bone_records_from_armature(armature)
    roundtrip_semantics = resolve_five_semantic_chains(roundtrip_bones)
    renders = render_static_evidence(
        bpy=bpy,
        armature=armature,
        mesh=mesh,
        semantics=roundtrip_semantics,
        output_dir=staging_dir,
    )
    write_joint_hierarchy(
        path=staging_dir / "joint_hierarchy.txt",
        bones=roundtrip_bones,
        semantics=roundtrip_semantics,
    )
    return {
        "automatic_static_checks": "passed",
        "raw_triangle_contract": raw_triangle_validation,
        "input_pbr": input_pbr_validation,
        "input_mesh": input_mesh_validation,
        "input_surface": input_surface_validation,
        "source_mesh_contract": source_mesh_contract,
        "source_removed_gltf_import_helpers": source_import_helpers,
        "tokenrig_mesh_contract_before_canonical_transform": tokenrig_mesh_contract,
        "tokenrig_removed_gltf_import_helpers": tokenrig_import_helpers,
        "removed_proven_orphans": list(removed_orphans),
        "axis_canonicalization": axis,
        "grounding": grounding,
        "canonical_mesh_contract": canonical_mesh_contract,
        "hierarchy": hierarchy,
        "semantic_mapping": semantics,
        "weights": weight_validation,
        "seams": seam_validation,
        "bilateral_contamination": bilateral_validation,
        "exported_pbr": exported_pbr_validation,
        "glb_roundtrip": roundtrip,
        "renders": renders,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise StaticAuditError(f"required staged artifact is missing or empty: {path.name}")
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_staged_bundle(staging_dir: Path, *, include_manifest: bool) -> None:
    names = REQUIRED_BUNDLE_FILES if include_manifest else REQUIRED_BUNDLE_FILES[:-1]
    for name in names:
        path = staging_dir / name
        _artifact_record(path)
        if name.endswith(".png") and path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise StaticAuditError(f"required staged artifact is not PNG: {name}")
    read_glb(staging_dir / "bind_pose.glb")


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_static_audit(
    *,
    asset_id: str,
    source_glb: Path,
    tokenrig_glb: Path,
    tokenrig_manifest: Path,
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    parent_argument = output_dir.parent
    if parent_argument.is_symlink() or not parent_argument.is_dir():
        raise StaticAuditError(
            f"static-audit output parent must be a direct directory: {parent_argument}"
        )
    parent = parent_argument.resolve()
    output_dir = parent / output_dir.name
    authenticated: Mapping[str, Any] | None = None
    staging_dir: Path | None = None
    prior_strict_failure_evidence: list[dict[str, Any]] = []
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", asset_id):
            raise StaticAuditError(f"invalid asset_id: {asset_id!r}")
        if output_dir.name != STATIC_AUDIT_DIRNAME:
            raise StaticAuditError(
                f"output directory must be named {STATIC_AUDIT_DIRNAME!r}"
            )
        if output_dir.exists() or output_dir.is_symlink():
            raise StaticAuditError(f"immutable readiness bundle already exists: {output_dir}")
        authenticated = authenticate_task3_inputs(
            asset_id=asset_id,
            source_glb=source_glb,
            tokenrig_glb=tokenrig_glb,
            tokenrig_manifest=tokenrig_manifest,
        )
        source_glb = Path(source_glb).resolve()
        tokenrig_glb = Path(tokenrig_glb).resolve()
        tokenrig_manifest = Path(tokenrig_manifest).resolve()
        if tokenrig_manifest.parent != parent:
            raise StaticAuditError(
                "Task 3 manifest and Task 4 bundle must share one canonical asset tree"
            )
        if tokenrig_glb.parent != parent:
            expected_failed_parent = (
                parent.parent / f"{asset_id}.tokenrig_failed_attempt"
            )
            if (
                authenticated.get("recovered_candidate") is not True
                or tokenrig_glb.parent != expected_failed_parent
            ):
                raise StaticAuditError(
                    "external TokenRig GLB is allowed only as the exact authenticated "
                    "sibling tokenrig_failed_attempt recovery evidence"
                )
        for path in sorted(parent.glob(f"{STATIC_AUDIT_DIRNAME}.failed.*.json")):
            if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
                raise StaticAuditError(f"prior static failure evidence is mutable: {path}")
            try:
                prior_payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StaticAuditError(f"prior static failure evidence is invalid: {exc}")
            if (
                not isinstance(prior_payload, dict)
                or prior_payload.get("decision") != "rejected"
                or prior_payload.get("readiness_bundle_published") is not False
            ):
                raise StaticAuditError("prior static failure evidence is not a rejection")
            prior_strict_failure_evidence.append(
                {
                    **_artifact_record(path),
                    "failure": prior_payload.get("failure"),
                }
            )
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{STATIC_AUDIT_DIRNAME}.",
                suffix=".staging",
                dir=str(parent),
            )
        )
        checks = run_blender_audit(
            source_glb=source_glb,
            tokenrig_glb=tokenrig_glb,
            staging_dir=staging_dir,
        )
        _validate_staged_bundle(staging_dir, include_manifest=False)
        artifacts = {
            name: _artifact_record(staging_dir / name)
            for name in REQUIRED_BUNDLE_FILES[:-1]
        }
        raw_equivalence = checks["raw_triangle_contract"]
        qa = {
            "schema": "tokenrig_human_static_qa_v1",
            "asset_id": asset_id,
            "decision": "automatic_static_checks_passed",
            "agent_qa_status": "pending_agent_visual_qa",
            "user_acceptance": "pending_user_review",
            "readiness_bundle_published": True,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "authenticated": dict(authenticated),
            "qualification": {
                "basis": "bounded_serialization_equivalence_not_exact",
                "exact_topology_unchanged": raw_equivalence[
                    "exact_topology_unchanged"
                ],
                "exact_normals_unchanged": raw_equivalence[
                    "exact_normals_unchanged"
                ],
                "backface_cull_risk": raw_equivalence["backface_cull_risk"],
                "warning": (
                    "Raw topology and normals are not exact; continuation is bounded "
                    "by reverse-coincident face-loss, area, position, UV, and normal limits."
                ),
            },
            "prior_strict_failure_evidence": prior_strict_failure_evidence,
            "checks": checks,
            "artifacts": artifacts,
        }
        _write_exclusive(
            staging_dir / "static_qa.json",
            (json.dumps(qa, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        _validate_staged_bundle(staging_dir, include_manifest=True)
        for path in sorted(staging_dir.iterdir()):
            _fsync_file(path)
            path.chmod(0o444)
        _fsync_directory(staging_dir)
        staging_dir.chmod(0o555)
        rename_directory_noreplace(staging_dir, output_dir)
        staging_dir = None
        return output_dir
    except BaseException as error:
        if staging_dir is not None and staging_dir.exists():
            staging_dir.chmod(0o700)
            for path in staging_dir.iterdir():
                if path.is_dir():
                    path.chmod(0o700)
                else:
                    path.chmod(0o600)
            shutil.rmtree(staging_dir)
        try:
            evidence = write_failure_evidence(
                output_dir=output_dir,
                asset_id=asset_id,
                error=error,
                authenticated=authenticated,
            )
        except BaseException as evidence_error:
            raise StaticAuditError(
                f"static audit failed ({error}); failure evidence also failed ({evidence_error})"
            ) from error
        raise StaticAuditError(
            f"static audit rejected: {error}; failure_evidence={evidence}"
        ) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--tokenrig-glb", type=Path, required=True)
    parser.add_argument("--tokenrig-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _blender_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    output = run_static_audit(
        asset_id=args.asset_id,
        source_glb=args.source_glb,
        tokenrig_glb=args.tokenrig_glb,
        tokenrig_manifest=args.tokenrig_manifest,
        output_dir=args.output_dir,
    )
    print(f"TOKENRIG_HUMAN_STATIC_AUDIT_OK {output}")
    return output


if __name__ == "__main__":
    main(_blender_argv())
