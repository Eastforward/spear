"""Prepare the MP3D 17DRP5sb8fy GLB for a fair Unreal comparison.

Matterport's raw GLB is Z-up.  Habitat loads that source with the proper
rotation, so feeding the raw bytes directly to Unreal would compare two
different world frames.  This tool bakes the same source-to-Habitat rotation
into every geometric float accessor before UE Interchange sees the file::

    source S = (x, y, z) -> Habitat-canonical H = (x, z, -y)

Only POSITION, NORMAL, and TANGENT payloads are changed.  Indices, UVs,
materials, textures, images, and scene topology remain byte/logically
identical.  The implementation is intentionally standard-library only and
handles interleaved buffer views through ``byteStride``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "avengine_mp3d_ue_prepared_scene_v1"
SCENE_ID = "17DRP5sb8fy"
EXPECTED_RAW_SHA256 = (
    "334456925e056c83a9a7a5c768b3d37cdd23425d8ca20743bfce015be3f56b04"
)
EXPECTED_ROOT_MESH_COUNT = 71
JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942
FLOAT_COMPONENT_TYPE = 5126
REFERENCE_RAW_BOUNDS = {
    "minimum": [-11.593440055847168, -2.886620044708252, -0.12755300104618073],
    "maximum": [4.757026195526123, 5.392021179199219, 2.6787829399108887],
}
REFERENCE_CANONICAL_BOUNDS = {
    "minimum": [-11.593440055847168, -0.12755300104618073, -5.392021179199219],
    "maximum": [4.757026195526123, 2.6787829399108887, 2.886620044708252],
}
REFERENCE_BOUNDS_ABS_TOLERANCE = 1.0e-5
_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass(frozen=True)
class GlbPayload:
    document: dict[str, Any]
    binary: bytes


@dataclass(frozen=True)
class AccessorLayout:
    accessor_index: int
    count: int
    component_count: int
    element_size: int
    stride: int
    first_offset: int
    view_end: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_glb(path: Path) -> GlbPayload:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError(f"GLB is too short: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise ValueError(
            "expected a complete GLB 2.0 container with a truthful byte length"
        )

    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > len(payload):
            raise ValueError("truncated GLB chunk payload")
        chunks.append((chunk_type, payload[offset:chunk_end]))
        offset = chunk_end
    if offset != len(payload):
        raise ValueError("GLB chunks do not consume the declared container")
    if [chunk_type for chunk_type, _ in chunks] != [
        JSON_CHUNK_TYPE,
        BIN_CHUNK_TYPE,
    ]:
        raise ValueError("expected exactly one JSON chunk followed by one BIN chunk")

    json_bytes = chunks[0][1].rstrip(b" \t\r\n\x00")
    try:
        document = json.loads(json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid GLB JSON document") from error
    if not isinstance(document, dict):
        raise ValueError("GLB JSON root must be an object")
    binary = chunks[1][1]
    buffers = document.get("buffers")
    if (
        not isinstance(buffers, list)
        or len(buffers) != 1
        or buffers[0].get("uri") is not None
        or int(buffers[0].get("byteLength", -1)) > len(binary)
        or len(binary) - int(buffers[0].get("byteLength", -1)) not in range(4)
    ):
        raise ValueError("expected one embedded BIN buffer with at most 3 padding bytes")
    return GlbPayload(document=document, binary=binary)


def build_glb(document: dict[str, Any], binary: bytes) -> bytes:
    json_bytes = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary_padded = bytes(binary) + b"\x00" * ((-len(binary)) % 4)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary_padded)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<II", len(json_bytes), JSON_CHUNK_TYPE),
            json_bytes,
            struct.pack("<II", len(binary_padded), BIN_CHUNK_TYPE),
            binary_padded,
        )
    )


def source_to_habitat_canonical(vector: Sequence[float]) -> tuple[float, ...]:
    if len(vector) not in (3, 4):
        raise ValueError("geometric vector must be VEC3 or VEC4")
    x, y, z = (float(vector[0]), float(vector[1]), float(vector[2]))
    transformed = (x, z, -y)
    if len(vector) == 4:
        return (*transformed, float(vector[3]))
    return transformed


def _accessor_layout(
    document: dict[str, Any],
    accessor_index: int,
    *,
    expected_type: str,
) -> AccessorLayout:
    accessors = document.get("accessors", [])
    buffer_views = document.get("bufferViews", [])
    try:
        accessor = accessors[accessor_index]
    except (IndexError, TypeError) as error:
        raise ValueError(f"invalid accessor index {accessor_index}") from error
    if accessor.get("sparse") is not None:
        raise ValueError(f"sparse geometric accessor {accessor_index} is unsupported")
    if accessor.get("componentType") != FLOAT_COMPONENT_TYPE:
        raise ValueError(f"geometric accessor {accessor_index} is not float32")
    if accessor.get("type") != expected_type:
        raise ValueError(
            f"geometric accessor {accessor_index} type {accessor.get('type')} "
            f"!= {expected_type}"
        )
    if "bufferView" not in accessor:
        raise ValueError(f"geometric accessor {accessor_index} has no bufferView")
    view_index = int(accessor["bufferView"])
    try:
        view = buffer_views[view_index]
    except (IndexError, TypeError) as error:
        raise ValueError(f"invalid bufferView {view_index}") from error
    if int(view.get("buffer", 0)) != 0:
        raise ValueError("geometric accessor does not reference embedded buffer 0")

    component_count = _COMPONENT_COUNTS[expected_type]
    element_size = component_count * 4
    stride = int(view.get("byteStride", element_size))
    if stride < element_size or stride % 4 != 0:
        raise ValueError(f"invalid float accessor byteStride {stride}")
    count = int(accessor.get("count", -1))
    if count <= 0:
        raise ValueError(f"invalid accessor count {count}")
    view_start = int(view.get("byteOffset", 0))
    view_length = int(view.get("byteLength", -1))
    if view_start < 0 or view_length < 0:
        raise ValueError("invalid bufferView byte range")
    first_offset = view_start + int(accessor.get("byteOffset", 0))
    view_end = view_start + view_length
    last_end = first_offset if count == 0 else first_offset + (count - 1) * stride + element_size
    if first_offset < view_start or last_end > view_end:
        raise ValueError(f"accessor {accessor_index} escapes its bufferView")
    return AccessorLayout(
        accessor_index=accessor_index,
        count=count,
        component_count=component_count,
        element_size=element_size,
        stride=stride,
        first_offset=first_offset,
        view_end=view_end,
    )


def _semantic_accessors(
    document: dict[str, Any],
) -> dict[str, list[int]]:
    result: dict[str, set[int]] = {
        "POSITION": set(),
        "NORMAL": set(),
        "TANGENT": set(),
    }
    meshes = document.get("meshes", [])
    for mesh_index, mesh in enumerate(meshes):
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise ValueError(f"mesh {mesh_index} has no primitives")
        for primitive_index, primitive in enumerate(primitives):
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict) or "POSITION" not in attributes:
                raise ValueError(
                    f"mesh {mesh_index} primitive {primitive_index} has no POSITION"
                )
            for semantic in result:
                if semantic in attributes:
                    result[semantic].add(int(attributes[semantic]))
    owners: dict[int, str] = {}
    for semantic, indices in result.items():
        for index in indices:
            previous = owners.setdefault(index, semantic)
            if previous != semantic:
                raise ValueError(
                    f"accessor {index} is aliased by {previous} and {semantic}"
                )
    return {semantic: sorted(indices) for semantic, indices in result.items()}


def validate_root_mesh_identity(
    document: dict[str, Any], expected_count: int = EXPECTED_ROOT_MESH_COUNT
) -> dict[str, Any]:
    scenes = document.get("scenes")
    scene_index = int(document.get("scene", 0))
    nodes = document.get("nodes")
    meshes = document.get("meshes")
    if (
        not isinstance(scenes, list)
        or len(scenes) != 1
        or scene_index != 0
        or not isinstance(nodes, list)
        or not isinstance(meshes, list)
    ):
        raise ValueError("expected one active scene with explicit nodes and meshes")
    expected_indices = list(range(int(expected_count)))
    if (
        len(nodes) != expected_count
        or len(meshes) != expected_count
        or scenes[0].get("nodes") != expected_indices
    ):
        raise ValueError(
            f"expected {expected_count} one-to-one root nodes/meshes in index order"
        )
    transform_keys = {"matrix", "translation", "rotation", "scale", "children"}
    for node_index, node in enumerate(nodes):
        if node.get("mesh") != node_index:
            raise ValueError(f"root node {node_index} does not own mesh {node_index}")
        unexpected = sorted(transform_keys.intersection(node))
        if unexpected:
            raise ValueError(
                f"root node {node_index} is not identity-only: {unexpected}"
            )
    primitive_count = sum(len(mesh.get("primitives", [])) for mesh in meshes)
    if primitive_count != expected_count:
        raise ValueError(
            f"expected {expected_count} primitives, observed {primitive_count}"
        )
    return {
        "status": "passed",
        "active_scene_index": 0,
        "root_node_count": len(nodes),
        "mesh_count": len(meshes),
        "primitive_count": primitive_count,
        "root_node_indices": expected_indices,
        "node_mesh_index_identity": True,
        "all_root_transforms_identity": True,
    }


def _read_vectors(binary: bytes, layout: AccessorLayout) -> Iterable[tuple[float, ...]]:
    fmt = f"<{layout.component_count}f"
    for element_index in range(layout.count):
        offset = layout.first_offset + element_index * layout.stride
        values = struct.unpack_from(fmt, binary, offset)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"accessor {layout.accessor_index} contains a non-finite value"
            )
        yield values


def _bounds(vectors: Iterable[Sequence[float]]) -> dict[str, list[float]]:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    count = 0
    for vector in vectors:
        count += 1
        for axis in range(3):
            minimum[axis] = min(minimum[axis], float(vector[axis]))
            maximum[axis] = max(maximum[axis], float(vector[axis]))
    if count == 0:
        raise ValueError("scene has no position vertices")
    return {"minimum": minimum, "maximum": maximum}


def _bounds_close(
    actual: dict[str, list[float]],
    expected: dict[str, list[float]],
    tolerance: float = REFERENCE_BOUNDS_ABS_TOLERANCE,
) -> bool:
    return all(
        abs(float(actual[key][axis]) - float(expected[key][axis])) <= tolerance
        for key in ("minimum", "maximum")
        for axis in range(3)
    )


def _document_without_geometric_bounds(
    document: dict[str, Any], semantic_accessors: dict[str, list[int]]
) -> dict[str, Any]:
    stripped = copy.deepcopy(document)
    for accessor_indices in semantic_accessors.values():
        for accessor_index in accessor_indices:
            stripped["accessors"][accessor_index].pop("min", None)
            stripped["accessors"][accessor_index].pop("max", None)
    return stripped


def transform_document_and_binary(
    document: dict[str, Any],
    binary: bytes,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    transformed_document = copy.deepcopy(document)
    transformed_binary = bytearray(binary)
    semantic_accessors = _semantic_accessors(document)
    expected_types = {
        "POSITION": "VEC3",
        "NORMAL": "VEC3",
        "TANGENT": "VEC4",
    }
    layouts: dict[str, list[AccessorLayout]] = {}
    for semantic, accessor_indices in semantic_accessors.items():
        layouts[semantic] = [
            _accessor_layout(
                document,
                accessor_index,
                expected_type=expected_types[semantic],
            )
            for accessor_index in accessor_indices
        ]

    raw_positions = (
        vector
        for layout in layouts["POSITION"]
        for vector in _read_vectors(binary, layout)
    )
    raw_bounds = _bounds(raw_positions)
    transformed_ranges: list[dict[str, int | str]] = []
    element_counts: dict[str, int] = {}
    recomputed_bounds: dict[str, list[int]] = {
        semantic: [] for semantic in expected_types
    }
    for semantic in ("POSITION", "NORMAL", "TANGENT"):
        element_counts[semantic] = 0
        for layout in layouts[semantic]:
            fmt = f"<{layout.component_count}f"
            source_accessor = document["accessors"][layout.accessor_index]
            has_minimum = "min" in source_accessor
            has_maximum = "max" in source_accessor
            if has_minimum != has_maximum:
                raise ValueError(
                    f"accessor {layout.accessor_index} has only one bounds endpoint"
                )
            write_bounds = semantic == "POSITION" or has_minimum
            accessor_minimum = [math.inf] * layout.component_count
            accessor_maximum = [-math.inf] * layout.component_count
            for element_index, vector in enumerate(_read_vectors(binary, layout)):
                transformed = source_to_habitat_canonical(vector)
                offset = layout.first_offset + element_index * layout.stride
                struct.pack_into(fmt, transformed_binary, offset, *transformed)
                if write_bounds:
                    for axis in range(layout.component_count):
                        accessor_minimum[axis] = min(
                            accessor_minimum[axis], transformed[axis]
                        )
                        accessor_maximum[axis] = max(
                            accessor_maximum[axis], transformed[axis]
                        )
                transformed_ranges.append(
                    {
                        "semantic": semantic,
                        "accessor_index": layout.accessor_index,
                        "byte_offset": offset,
                        "byte_length": layout.element_size,
                    }
                )
                element_counts[semantic] += 1
            if write_bounds:
                accessor = transformed_document["accessors"][layout.accessor_index]
                accessor["min"] = accessor_minimum
                accessor["max"] = accessor_maximum
                recomputed_bounds[semantic].append(layout.accessor_index)

    canonical_positions = (
        vector
        for layout in layouts["POSITION"]
        for vector in _read_vectors(bytes(transformed_binary), layout)
    )
    canonical_bounds = _bounds(canonical_positions)
    expected_from_raw = {
        "minimum": [
            raw_bounds["minimum"][0],
            raw_bounds["minimum"][2],
            -raw_bounds["maximum"][1],
        ],
        "maximum": [
            raw_bounds["maximum"][0],
            raw_bounds["maximum"][2],
            -raw_bounds["minimum"][1],
        ],
    }
    if not _bounds_close(canonical_bounds, expected_from_raw, tolerance=2.0e-6):
        raise AssertionError("canonical position readback disagrees with raw rotation")
    if _document_without_geometric_bounds(
        document, semantic_accessors
    ) != _document_without_geometric_bounds(
        transformed_document, semantic_accessors
    ):
        raise AssertionError(
            "preparation changed GLB JSON beyond geometric accessor min/max"
        )

    metadata = {
        "semantic_accessor_indices": semantic_accessors,
        "semantic_element_counts": element_counts,
        "recomputed_bounds_accessor_indices": recomputed_bounds,
        "raw_bounds": raw_bounds,
        "canonical_bounds": canonical_bounds,
        "canonical_bounds_derived_from_raw": expected_from_raw,
        "interleaved_accessor_count": sum(
            layout.stride > layout.element_size
            for semantic_layouts in layouts.values()
            for layout in semantic_layouts
        ),
        "transformed_element_range_count": len(transformed_ranges),
        "transformed_element_ranges_sha256": canonical_json_sha256(
            transformed_ranges
        ),
    }
    return transformed_document, bytes(transformed_binary), metadata


def _publish_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_mp3d_scene(
    *,
    input_glb: Path,
    output_glb: Path,
    manifest_path: Path,
    scene_id: str = SCENE_ID,
    expected_root_mesh_count: int = EXPECTED_ROOT_MESH_COUNT,
    enforce_reference_bounds: bool = True,
) -> dict[str, Any]:
    input_glb = input_glb.resolve()
    output_glb = output_glb.resolve()
    manifest_path = manifest_path.resolve()
    if not input_glb.is_file() or input_glb.is_symlink():
        raise FileNotFoundError(f"input GLB is missing or not a direct file: {input_glb}")
    if output_glb == input_glb or manifest_path in (input_glb, output_glb):
        raise ValueError("input, output, and manifest paths must be distinct")
    if output_glb.exists() or output_glb.is_symlink():
        raise FileExistsError(f"refusing to replace prepared GLB: {output_glb}")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"refusing to replace manifest: {manifest_path}")

    source_sha256 = sha256_file(input_glb)
    if enforce_reference_bounds and source_sha256 != EXPECTED_RAW_SHA256:
        raise ValueError(
            f"raw {SCENE_ID} SHA-256 changed: {source_sha256}"
        )

    loaded = load_glb(input_glb)
    identity = validate_root_mesh_identity(
        loaded.document, expected_count=expected_root_mesh_count
    )
    transformed_document, transformed_binary, transform = (
        transform_document_and_binary(loaded.document, loaded.binary)
    )
    if enforce_reference_bounds:
        if scene_id != SCENE_ID:
            raise ValueError("reference bounds may only be enforced for 17DRP5sb8fy")
        if not _bounds_close(transform["raw_bounds"], REFERENCE_RAW_BOUNDS):
            raise ValueError(
                f"raw {SCENE_ID} bounds changed: {transform['raw_bounds']}"
            )
        if not _bounds_close(
            transform["canonical_bounds"], REFERENCE_CANONICAL_BOUNDS
        ):
            raise ValueError(
                f"canonical {SCENE_ID} bounds changed: "
                f"{transform['canonical_bounds']}"
            )

    output_bytes = build_glb(transformed_document, transformed_binary)
    preserved_sections = {
        key: loaded.document.get(key)
        for key in (
            "asset",
            "scene",
            "scenes",
            "nodes",
            "meshes",
            "materials",
            "textures",
            "images",
            "samplers",
        )
        if key in loaded.document
    }
    manifest = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scene_id": scene_id,
        "source": {
            "path": str(input_glb),
            "sha256": source_sha256,
            "size_bytes": input_glb.stat().st_size,
        },
        "prepared": {
            "path": str(output_glb),
            "sha256": hashlib.sha256(output_bytes).hexdigest(),
            "size_bytes": len(output_bytes),
        },
        "coordinate_contract": {
            "source_axis_description": "Matterport raw GLB Z-up",
            "canonical_axis_description": "Habitat world Y-up",
            "source_to_canonical": "H=(S.x,S.z,-S.y)",
            "matrix_row_major": [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            "determinant": 1.0,
            "transformed_semantics": ["POSITION", "NORMAL", "TANGENT"],
            "tangent_w_preserved": True,
        },
        "root_mesh_identity": identity,
        "geometry": transform,
        "reference_bounds_validation": {
            "enforced": bool(enforce_reference_bounds),
            "absolute_tolerance": REFERENCE_BOUNDS_ABS_TOLERANCE,
            "expected_raw_bounds": (
                REFERENCE_RAW_BOUNDS if enforce_reference_bounds else None
            ),
            "expected_canonical_bounds": (
                REFERENCE_CANONICAL_BOUNDS if enforce_reference_bounds else None
            ),
            "status": "passed" if enforce_reference_bounds else "not_applicable",
        },
        "preservation": {
            "document_changes_limited_to_geometric_accessor_min_max": True,
            "indices_unchanged": True,
            "uvs_unchanged": True,
            "materials_unchanged": True,
            "textures_and_images_unchanged": True,
            "preserved_sections_sha256": canonical_json_sha256(
                preserved_sections
            ),
        },
        "claim_boundary": (
            "This artifact equalizes the MP3D world coordinate basis for UE. "
            "It does not prove UE import, cook, material fidelity, collision, "
            "lighting, visibility, or render equivalence."
        ),
    }
    _publish_no_replace(output_glb, output_bytes)
    try:
        if sha256_file(input_glb) != source_sha256:
            raise AssertionError("raw MP3D source changed during preparation")
        readback = load_glb(output_glb)
        if readback.document != transformed_document:
            raise AssertionError("published GLB JSON readback changed")
        if readback.binary[: len(transformed_binary)] != transformed_binary:
            raise AssertionError("published GLB BIN readback changed")
        if sha256_file(output_glb) != manifest["prepared"]["sha256"]:
            raise AssertionError("published GLB hash readback changed")
        manifest_bytes = (
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        _publish_no_replace(manifest_path, manifest_bytes)
    except BaseException:
        output_glb.unlink(missing_ok=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-id", default=SCENE_ID, choices=[SCENE_ID])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_mp3d_scene(
        input_glb=args.input_glb,
        output_glb=args.output_glb,
        manifest_path=args.manifest,
        scene_id=args.scene_id,
    )
    print(
        "PREPARE_MP3D_UE_SCENE_OK "
        f"scene={manifest['scene_id']} "
        f"source_sha256={manifest['source']['sha256']} "
        f"prepared_sha256={manifest['prepared']['sha256']} "
        f"manifest={args.manifest.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
