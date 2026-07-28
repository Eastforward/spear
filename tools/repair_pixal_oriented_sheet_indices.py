#!/usr/bin/env python3
"""Remove only exact-position-degenerate Pixel3D index triangles.

This CPU-only repair is deliberately narrower than a mesh cleanup.  It accepts
one static GLB mesh primitive only when every directed exact-position edge
occurrence among the retained triangles has an oppositely oriented partner.
IEEE signed zero is canonicalized for that comparison.  The only authorized
mutation is filtering exact-position-degenerate index triplets and updating
the index accessor count.  Vertex attributes, materials, PBR bindings,
embedded images, and every BIN byte outside the index accessor remain exact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_quadruped_i23d_geometry as geometry_audit
from tools import controlled_source_asset_schema as contracts
from tools import review_controlled_animal_pixal_static_candidates as static_decisions
from tools import run_controlled_animal_static_reviews as static_reviews


SCHEMA = "avengine_pixal_oriented_sheet_index_repair_v1"
IMPLEMENTATION_CONTRACT = "bounded_oriented_sheet_identity_v1"
TOPOLOGY_METHOD = (
    "exact_position_directed_edge_occurrence_reverse_pairing_signed_zero_v1"
)
MUTATION_METHOD = "exact_position_degenerate_triangle_index_filter_v1"
STATIC_DECISION_SCHEMA = "avengine_controlled_animal_static_decision_v1"
APPROVED = "approved_for_lod_and_binding"
RESEARCH = "research_candidate"
NEXT_GATE = "derived_static_multiview_review"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942
_INDEX_COMPONENTS = {
    5121: ("<B", 1),
    5123: ("<H", 2),
    5125: ("<I", 4),
}
_SAMPLER_MAG_FILTERS = frozenset({9728, 9729})
_SAMPLER_MIN_FILTERS = frozenset({9728, 9729, 9984, 9985, 9986, 9987})
_SAMPLER_WRAPS = frozenset({33071, 33648, 10497})
_CHECK_FIELDS = frozenset(
    {
        "source_glb_sha256_authenticated",
        "canonical_raw_static_decision_batch_replayed",
        "single_mesh_single_primitive_authenticated",
        "required_position_normal_uv_attributes_authenticated",
        "signed_zero_canonicalized_for_exact_position_identity",
        "all_retained_directed_edge_occurrences_reverse_paired",
        "only_exact_position_degenerate_triangles_removed",
        "all_nondegenerate_index_triplets_preserved_in_order",
        "index_accessor_count_matches_retained_triplets",
        "all_non_index_bin_bytes_unchanged",
        "position_normal_uv_accessor_bytes_unchanged",
        "material_pbr_bindings_unchanged",
        "embedded_image_bytes_unchanged",
        "output_glb_readback_reauthenticated",
        "no_external_geometry_skeleton_weight_material_texture_or_animation_input",
        "formal_dataset_registration_not_authorized",
    }
)
_TOPOLOGY_FIELDS = frozenset(
    {
        "method",
        "evaluated_triangle_count",
        "exact_position_degenerate_triangle_count",
        "unique_exact_positions",
        "directed_edge_occurrences",
        "undirected_edge_count",
        "maximum_edge_face_multiplicity",
        "balanced_oriented_multicover_edges_over_two_faces",
        "unpaired_oriented_edges",
        "unpaired_oriented_edge_occurrences",
        "all_directed_edge_occurrences_reverse_paired",
    }
)


class OrientedSheetRepairError(ValueError):
    """Fail-closed oriented-sheet repair error."""


def _canonical(value: Any) -> str:
    try:
        return contracts.canonical_json(value)
    except contracts.ContractError as error:
        raise OrientedSheetRepairError(str(error)) from error


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    payload = {
        name: copy.deepcopy(item) for name, item in value.items() if name != field
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise OrientedSheetRepairError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrientedSheetRepairError(f"{label} must be non-empty text")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _text(value, label)
    if not _ID_RE.fullmatch(value):
        raise OrientedSheetRepairError(f"{label} is not a canonical identifier")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrientedSheetRepairError(f"{label} must be an object")
    return value


def _exact(
    value: Any,
    fields: set[str] | frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    value = _mapping(value, label)
    if set(value) != set(fields):
        missing = sorted(set(fields) - set(value))
        extra = sorted(set(value) - set(fields))
        raise OrientedSheetRepairError(
            f"{label} fields are invalid: missing={missing} extra={extra}"
        )
    return value


def _nonnegative(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OrientedSheetRepairError(f"{label} must be a nonnegative integer")
    return value


def _positive(value: Any, label: str) -> int:
    value = _nonnegative(value, label)
    if value == 0:
        raise OrientedSheetRepairError(f"{label} must be positive")
    return value


def _file_record(value: Any, label: str) -> dict[str, Any]:
    value = _exact(value, {"path", "sha256", "size_bytes"}, label)
    path = _text(value["path"], f"{label}.path")
    if not Path(path).is_absolute():
        raise OrientedSheetRepairError(f"{label}.path must be absolute")
    _sha256(value["sha256"], f"{label}.sha256")
    _positive(value["size_bytes"], f"{label}.size_bytes")
    return copy.deepcopy(dict(value))


def _regular_file(path: Path, expected_sha256: str, label: str) -> tuple[Path, bytes]:
    expected_sha256 = _sha256(expected_sha256, f"{label} expected SHA-256")
    literal = Path(path).absolute()
    if literal.is_symlink():
        raise OrientedSheetRepairError(f"{label} cannot be a leaf symlink")
    resolved = literal.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise OrientedSheetRepairError(f"{label} is missing or empty: {literal}")
    data = resolved.read_bytes()
    if _sha256_bytes(data) != expected_sha256:
        raise OrientedSheetRepairError(f"{label} changed from its expected SHA-256")
    return resolved, data


def _record(path: Path, data: bytes | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if data is None:
        data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": _sha256_bytes(data),
        "size_bytes": len(data),
    }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise OrientedSheetRepairError(
                f"strict JSON contains duplicate key {name!r}"
            )
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise OrientedSheetRepairError(
        f"strict JSON contains non-finite constant {value!r}"
    )


def _strict_json_loads(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise OrientedSheetRepairError(
            f"{label} is not readable strict JSON: {error}"
        ) from error
    if not isinstance(value, dict):
        raise OrientedSheetRepairError(f"{label} must contain a JSON object")
    return value


def _parse_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OrientedSheetRepairError(
            f"{label} is not readable strict JSON: {error}"
        ) from error
    return _strict_json_loads(text, label)


def _strict_json_file(
    literal: Path,
    label: str,
    *,
    containment_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    literal = Path(literal).absolute()
    if literal.is_symlink():
        raise OrientedSheetRepairError(f"{label} cannot be a leaf symlink")
    resolved = literal.resolve()
    if containment_root is not None:
        try:
            resolved.relative_to(containment_root.resolve())
        except ValueError as error:
            raise OrientedSheetRepairError(
                f"{label} escaped its authority root"
            ) from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise OrientedSheetRepairError(f"{label} is missing or empty")
    return resolved, _parse_json_bytes(resolved.read_bytes(), label)


def _strict_authority_chain_json(
    decision_batch_path: Path,
    decision_batch: Mapping[str, Any],
) -> None:
    review_descriptor = _mapping(
        decision_batch.get("static_review_batch"),
        "canonical static review batch descriptor",
    )
    review_value = review_descriptor.get("path")
    if not isinstance(review_value, str) or not Path(review_value).is_absolute():
        raise OrientedSheetRepairError(
            "canonical static review batch path must be absolute"
        )
    review_batch_path, review_batch = _strict_json_file(
        Path(review_value),
        "canonical static review batch",
    )
    reviews = review_batch.get("reviews")
    if not isinstance(reviews, list):
        raise OrientedSheetRepairError(
            "canonical static review batch reviews are invalid"
        )
    for index, entry in enumerate(reviews):
        descriptor = (
            entry.get("review") if isinstance(entry, Mapping) else None
        )
        value = descriptor.get("path") if isinstance(descriptor, Mapping) else None
        if (
            not isinstance(value, str)
            or Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise OrientedSheetRepairError(
                f"canonical static review record {index} path is invalid"
            )
        _strict_json_file(
            review_batch_path.parent / value,
            f"canonical static review record {index}",
            containment_root=review_batch_path.parent,
        )
    decisions = decision_batch.get("decisions")
    if not isinstance(decisions, list):
        raise OrientedSheetRepairError(
            "canonical static decision records are invalid"
        )
    decision_root = decision_batch_path.parent.resolve()
    for index, entry in enumerate(decisions):
        descriptor = (
            entry.get("record") if isinstance(entry, Mapping) else None
        )
        value = descriptor.get("path") if isinstance(descriptor, Mapping) else None
        if (
            not isinstance(value, str)
            or Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise OrientedSheetRepairError(
                f"canonical static decision record {index} path is invalid"
            )
        _strict_json_file(
            decision_root / value,
            f"canonical static decision record {index}",
            containment_root=decision_root,
        )


def _parse_glb(data: bytes, label: str) -> tuple[dict[str, Any], bytes]:
    if len(data) < 28 or data[:4] != b"glTF":
        raise OrientedSheetRepairError(f"{label} is not a GLB 2.0 file")
    version, declared_length = struct.unpack_from("<II", data, 4)
    if version != 2 or declared_length != len(data):
        raise OrientedSheetRepairError(f"{label} GLB header is invalid")
    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise OrientedSheetRepairError(f"{label} has a truncated chunk header")
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + length
        if length % 4 or end > len(data):
            raise OrientedSheetRepairError(f"{label} has an invalid GLB chunk")
        chunks.append((kind, data[offset:end]))
        offset = end
    if len(chunks) != 2 or [kind for kind, _ in chunks] != [
        _JSON_CHUNK,
        _BIN_CHUNK,
    ]:
        raise OrientedSheetRepairError(
            f"{label} must contain exactly one JSON chunk followed by one BIN chunk"
        )
    try:
        document_text = chunks[0][1].rstrip(b" \t\r\n\x00").decode("utf-8")
    except UnicodeDecodeError as error:
        raise OrientedSheetRepairError(
            f"{label} JSON chunk is invalid: {error}"
        ) from error
    document = _strict_json_loads(document_text, f"{label} GLB JSON chunk")
    binary = chunks[1][1]
    asset = document.get("asset")
    if not isinstance(asset, Mapping) or asset.get("version") != "2.0":
        raise OrientedSheetRepairError(f"{label} asset.version must be 2.0")
    buffers = document.get("buffers")
    if (
        not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], Mapping)
        or "uri" in buffers[0]
        or isinstance(buffers[0].get("byteLength"), bool)
        or not isinstance(buffers[0].get("byteLength"), int)
        or buffers[0]["byteLength"] <= 0
        or buffers[0]["byteLength"] > len(binary)
        or len(binary) - buffers[0]["byteLength"] > 3
    ):
        raise OrientedSheetRepairError(
            f"{label} must contain one valid embedded buffer"
        )
    return document, binary


def _build_glb(document: Mapping[str, Any], binary: bytes) -> bytes:
    json_payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    if len(binary) % 4:
        raise OrientedSheetRepairError("source BIN chunk alignment changed")
    total = 12 + 8 + len(json_payload) + 8 + len(binary)
    return b"".join(
        (
            b"glTF",
            struct.pack("<II", 2, total),
            struct.pack("<II", len(json_payload), _JSON_CHUNK),
            json_payload,
            struct.pack("<II", len(binary), _BIN_CHUNK),
            binary,
        )
    )


def _list(document: Mapping[str, Any], name: str, label: str) -> list[Any]:
    value = document.get(name)
    if not isinstance(value, list) or not value:
        raise OrientedSheetRepairError(f"{label}.{name} must be a non-empty list")
    return value


def _declared_buffer_length(document: Mapping[str, Any], label: str) -> int:
    buffers = document.get("buffers")
    if (
        not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], Mapping)
        or isinstance(buffers[0].get("byteLength"), bool)
        or not isinstance(buffers[0].get("byteLength"), int)
        or buffers[0]["byteLength"] <= 0
    ):
        raise OrientedSheetRepairError(
            f"{label} declared buffer length is invalid"
        )
    return buffers[0]["byteLength"]


def _buffer_view_interval(
    document: Mapping[str, Any],
    binary: bytes,
    index: Any,
    label: str,
) -> tuple[int, int]:
    views = _list(document, "bufferViews", label)
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(views)
    ):
        raise OrientedSheetRepairError(f"{label} bufferView index is invalid")
    view = _mapping(views[index], f"{label} bufferView")
    if view.get("buffer", 0) != 0:
        raise OrientedSheetRepairError(f"{label} bufferView must use buffer zero")
    start = view.get("byteOffset", 0)
    length = view.get("byteLength")
    declared_length = _declared_buffer_length(document, label)
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or start + length > declared_length
        or declared_length > len(binary)
    ):
        raise OrientedSheetRepairError(f"{label} bufferView bounds are invalid")
    return start, start + length


def _validate_nonoverlapping_views(
    document: Mapping[str, Any],
    binary: bytes,
    label: str,
) -> None:
    views = _list(document, "bufferViews", label)
    intervals = [
        (*_buffer_view_interval(document, binary, index, label), index)
        for index in range(len(views))
    ]
    intervals.sort()
    for (_, previous_end, previous), (start, _, current) in zip(
        intervals, intervals[1:]
    ):
        if start < previous_end:
            raise OrientedSheetRepairError(
                f"{label} bufferViews overlap: {previous} and {current}"
            )


def _accessor(
    document: Mapping[str, Any],
    index: Any,
    label: str,
) -> Mapping[str, Any]:
    accessors = _list(document, "accessors", label)
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(accessors)
    ):
        raise OrientedSheetRepairError(f"{label} accessor index is invalid")
    accessor = _mapping(accessors[index], f"{label} accessor")
    if "sparse" in accessor:
        raise OrientedSheetRepairError(f"{label} sparse accessor is unsupported")
    return accessor


def _float32_equivalent(left: Any, right: float) -> bool:
    if (
        isinstance(left, bool)
        or not isinstance(left, (int, float))
        or not math.isfinite(float(left))
    ):
        return False
    if float(left) == 0.0 and right == 0.0:
        return True
    try:
        return struct.pack("<f", float(left)) == struct.pack("<f", right)
    except (OverflowError, struct.error):
        return False


def _validate_optional_float_extrema(
    accessor: Mapping[str, Any],
    values: Sequence[Sequence[float]],
    components: int,
    label: str,
) -> None:
    extrema = {
        "min": [min(value[index] for value in values) for index in range(components)],
        "max": [max(value[index] for value in values) for index in range(components)],
    }
    for name, expected in extrema.items():
        if name not in accessor:
            continue
        declared = accessor[name]
        if (
            not isinstance(declared, list)
            or len(declared) != components
            or any(
                not _float32_equivalent(item, actual)
                for item, actual in zip(declared, expected)
            )
        ):
            raise OrientedSheetRepairError(
                f"{label} accessor {name} does not match actual extrema"
            )


def _validate_optional_index_extrema(
    accessor: Mapping[str, Any],
    values: Sequence[int],
    label: str,
) -> None:
    for name, expected in (("min", min(values)), ("max", max(values))):
        if name not in accessor:
            continue
        declared = accessor[name]
        if (
            not isinstance(declared, list)
            or len(declared) != 1
            or isinstance(declared[0], bool)
            or not isinstance(declared[0], int)
            or declared[0] != expected
        ):
            raise OrientedSheetRepairError(
                f"{label} accessor {name} does not match actual extrema"
            )


def _float_attribute_values(
    document: Mapping[str, Any],
    binary: bytes,
    accessor_index: int,
    *,
    semantic: str,
    expected_type: str,
    components: int,
    expected_count: int | None = None,
) -> tuple[list[tuple[float, ...]], dict[str, Any]]:
    label = f"source GLB {semantic}"
    accessor = _accessor(document, accessor_index, label)
    if (
        accessor.get("componentType") != 5126
        or accessor.get("type") != expected_type
        or accessor.get("normalized", False) is not False
    ):
        raise OrientedSheetRepairError(
            f"{label} must be a non-normalized float32 {expected_type}"
        )
    count = accessor.get("count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or (expected_count is not None and count != expected_count)
    ):
        raise OrientedSheetRepairError(f"{label} count is invalid")
    view_index = accessor.get("bufferView")
    view_start, view_end = _buffer_view_interval(
        document,
        binary,
        view_index,
        label,
    )
    view = _mapping(
        document["bufferViews"][view_index],
        f"{label} bufferView",
    )
    element_width = components * 4
    stride = view.get("byteStride", element_width)
    relative = accessor.get("byteOffset", 0)
    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride < element_width
        or stride > 252
        or stride % 4
        or isinstance(relative, bool)
        or not isinstance(relative, int)
        or relative < 0
        or (view_start + relative) % 4
    ):
        raise OrientedSheetRepairError(f"{label} layout is invalid")
    start = view_start + relative
    end = start + (count - 1) * stride + element_width
    if end > view_end:
        raise OrientedSheetRepairError(
            f"{label} accessor exceeds its bufferView"
        )
    format_string = "<" + "f" * components
    values = [
        tuple(
            float(item)
            for item in struct.unpack_from(
                format_string,
                binary,
                start + index * stride,
            )
        )
        for index in range(count)
    ]
    if any(not all(math.isfinite(item) for item in value) for value in values):
        raise OrientedSheetRepairError(f"{label} contains non-finite values")
    _validate_optional_float_extrema(
        accessor,
        values,
        components,
        label,
    )
    return values, {
        "accessor_index": accessor_index,
        "buffer_view_index": view_index,
        "count": count,
        "start": start,
        "end": end,
        "stride": stride,
    }


def _index_layout(
    document: Mapping[str, Any],
    binary: bytes,
    accessor_index: int,
    label: str,
) -> dict[str, Any]:
    accessor = _accessor(document, accessor_index, label)
    component = accessor.get("componentType")
    if component not in _INDEX_COMPONENTS or accessor.get("type") != "SCALAR":
        raise OrientedSheetRepairError(
            f"{label} indices must be an unsigned scalar accessor"
        )
    if accessor.get("normalized", False) is not False:
        raise OrientedSheetRepairError(f"{label} index accessor cannot be normalized")
    count = accessor.get("count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or count % 3
    ):
        raise OrientedSheetRepairError(
            f"{label} index accessor count must be positive and divisible by three"
        )
    view_index = accessor.get("bufferView")
    view_start, view_end = _buffer_view_interval(
        document, binary, view_index, label
    )
    views = document["bufferViews"]
    view = _mapping(views[view_index], f"{label} index bufferView")
    if "byteStride" in view:
        raise OrientedSheetRepairError(
            f"{label} index bufferView cannot have byteStride"
        )
    fmt, width = _INDEX_COMPONENTS[component]
    relative = accessor.get("byteOffset", 0)
    if (
        isinstance(relative, bool)
        or not isinstance(relative, int)
        or relative < 0
        or relative % width
    ):
        raise OrientedSheetRepairError(f"{label} index byteOffset is invalid")
    start = view_start + relative
    end = start + count * width
    if start % width or end > view_end:
        raise OrientedSheetRepairError(
            f"{label} index accessor absolute offset/bounds are invalid"
        )
    return {
        "accessor": accessor,
        "accessor_index": accessor_index,
        "buffer_view_index": view_index,
        "component_type": component,
        "format": fmt,
        "width": width,
        "count": count,
        "start": start,
        "end": end,
    }


def _position_values(
    document: Mapping[str, Any],
    binary: bytes,
    accessor_index: int,
    label: str,
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    values, layout = _float_attribute_values(
        document,
        binary,
        accessor_index,
        semantic="POSITION",
        expected_type="VEC3",
        components=3,
    )
    return [
        (value[0], value[1], value[2]) for value in values
    ], layout


def _unpack_indices(binary: bytes, layout: Mapping[str, Any]) -> list[int]:
    fmt = layout["format"]
    width = layout["width"]
    start = layout["start"]
    return [
        int(struct.unpack_from(fmt, binary, start + index * width)[0])
        for index in range(layout["count"])
    ]


def _pack_indices(values: Sequence[int], component_type: int) -> bytes:
    fmt, _ = _INDEX_COMPONENTS[component_type]
    try:
        return b"".join(struct.pack(fmt, value) for value in values)
    except struct.error as error:
        raise OrientedSheetRepairError(
            "retained index is outside its source component type"
        ) from error


def _triplets(values: Sequence[int]) -> list[tuple[int, int, int]]:
    return [
        (int(values[index]), int(values[index + 1]), int(values[index + 2]))
        for index in range(0, len(values), 3)
    ]


def _canonical_position(
    value: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(0.0 if coordinate == 0.0 else coordinate for coordinate in value)


def _partition_triangles(
    positions: Sequence[tuple[float, float, float]],
    triangles: Sequence[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int]], list[int]]:
    retained: list[tuple[int, int, int]] = []
    removed: list[int] = []
    canonical = [_canonical_position(point) for point in positions]
    for ordinal, triangle in enumerate(triangles):
        if any(index < 0 or index >= len(canonical) for index in triangle):
            raise OrientedSheetRepairError("index accessor escapes POSITION domain")
        keys = [canonical[index] for index in triangle]
        if len(set(keys)) != 3:
            removed.append(ordinal)
        else:
            retained.append(triangle)
    if not retained:
        raise OrientedSheetRepairError(
            "exact-position filtering would remove every triangle"
        )
    return retained, removed


def _triplet_sha256(values: Sequence[tuple[int, int, int]]) -> str:
    digest = hashlib.sha256()
    for triangle in values:
        digest.update(struct.pack("<III", *triangle))
    return digest.hexdigest()


def _ordinal_sha256(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<Q", value))
    return digest.hexdigest()


def _topology(
    positions: Sequence[tuple[float, float, float]],
    triangles: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    result = geometry_audit.position_indexed_topology(
        np.asarray(positions, dtype=np.float64),
        np.asarray(triangles, dtype=np.int64),
    )
    return {
        "method": TOPOLOGY_METHOD,
        "evaluated_triangle_count": int(result["position_indexed_triangles"]),
        "exact_position_degenerate_triangle_count": int(
            result["degenerate_triangles_after_position_indexing"]
        ),
        "unique_exact_positions": int(result["position_unique_vertices"]),
        "directed_edge_occurrences": int(
            result["paired_oriented_sheet_edge_occurrences"]
            + result["unpaired_oriented_edge_occurrences"]
        ),
        "undirected_edge_count": int(
            result["boundary_edges"]
            + result["manifold_two_face_edges"]
            + result["two_face_orientation_mismatch_edges"]
            + result["balanced_oriented_multicover_edges_over_two_faces"]
            + result["unbalanced_edges_over_two_faces"]
        ),
        "maximum_edge_face_multiplicity": int(
            result["maximum_edge_face_multiplicity"]
        ),
        "balanced_oriented_multicover_edges_over_two_faces": int(
            result["balanced_oriented_multicover_edges_over_two_faces"]
        ),
        "unpaired_oriented_edges": int(result["unpaired_oriented_edges"]),
        "unpaired_oriented_edge_occurrences": int(
            result["unpaired_oriented_edge_occurrences"]
        ),
        "all_directed_edge_occurrences_reverse_paired": bool(
            result["unpaired_oriented_edges"] == 0
            and result["unpaired_oriented_edge_occurrences"] == 0
        ),
    }


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _pbr_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: copy.deepcopy(document.get(name, []))
        for name in (
            "materials",
            "textures",
            "samplers",
            "images",
            "extensionsUsed",
            "extensionsRequired",
        )
    }


def _embedded_image_payloads(
    document: Mapping[str, Any],
    binary: bytes,
) -> list[bytes]:
    images = _list(document, "images", "GLB")
    payloads: list[bytes] = []
    for index, image_value in enumerate(images):
        image = _mapping(image_value, f"GLB image {index}")
        if "uri" in image:
            raise OrientedSheetRepairError(
                "all Pixel3D images must be embedded bufferViews"
            )
        start, end = _buffer_view_interval(
            document, binary, image.get("bufferView"), f"GLB image {index}"
        )
        if image.get("mimeType") not in {
            "image/png",
            "image/jpeg",
            "image/webp",
        }:
            raise OrientedSheetRepairError(
                f"GLB image {index} MIME type is unsupported"
            )
        payloads.append(binary[start:end])
    return payloads


def _texture_image_source(
    document: Mapping[str, Any],
    texture: Mapping[str, Any],
    image_count: int,
    label: str,
) -> int:
    """Resolve either the core glTF source or Pixel3D's WebP extension source."""

    source = texture.get("source")
    extension_source: Any = None
    extensions = texture.get("extensions")
    if isinstance(extensions, Mapping):
        webp = extensions.get("EXT_texture_webp")
        if isinstance(webp, Mapping):
            if "EXT_texture_webp" not in document.get("extensionsUsed", []):
                raise OrientedSheetRepairError(
                    f"{label} uses undeclared EXT_texture_webp"
                )
            extension_source = webp.get("source")
    if source is None:
        source = extension_source
    elif extension_source is not None and extension_source != source:
        raise OrientedSheetRepairError(
            f"{label} core and EXT_texture_webp image references disagree"
        )
    if (
        isinstance(source, bool)
        or not isinstance(source, int)
        or not 0 <= source < image_count
    ):
        raise OrientedSheetRepairError(f"{label} image source is invalid")
    return source


def _validate_pbr_bindings(
    document: Mapping[str, Any],
    binary: bytes,
    primitive: Mapping[str, Any],
) -> tuple[int, list[bytes]]:
    materials = _list(document, "materials", "source GLB")
    textures = _list(document, "textures", "source GLB")
    images = _embedded_image_payloads(document, binary)
    samplers_value = document.get("samplers", [])
    if not isinstance(samplers_value, list):
        raise OrientedSheetRepairError("source GLB samplers must be a list")
    samplers = samplers_value
    if len(materials) != 1 or len(textures) != 2 or len(images) != 2:
        raise OrientedSheetRepairError(
            "source GLB requires one material and exactly two Pixel3D "
            "textures/images"
        )
    for sampler_index, sampler_value in enumerate(samplers):
        sampler = _mapping(
            sampler_value,
            f"source GLB sampler {sampler_index}",
        )
        for name, allowed in (
            ("magFilter", _SAMPLER_MAG_FILTERS),
            ("minFilter", _SAMPLER_MIN_FILTERS),
            ("wrapS", _SAMPLER_WRAPS),
            ("wrapT", _SAMPLER_WRAPS),
        ):
            if name in sampler and (
                isinstance(sampler[name], bool)
                or not isinstance(sampler[name], int)
                or sampler[name] not in allowed
            ):
                raise OrientedSheetRepairError(
                    f"source GLB sampler {sampler_index} {name} is invalid"
                )
    texture_sources: list[int] = []
    for texture_index, texture_value in enumerate(textures):
        texture = _mapping(
            texture_value,
            f"source GLB texture {texture_index}",
        )
        source = _texture_image_source(
            document,
            texture,
            len(images),
            f"source GLB texture {texture_index}",
        )
        texture_sources.append(source)
        if "sampler" in texture:
            sampler = texture["sampler"]
            if (
                isinstance(sampler, bool)
                or not isinstance(sampler, int)
                or not 0 <= sampler < len(samplers)
            ):
                raise OrientedSheetRepairError(
                    f"source GLB texture {texture_index} sampler is invalid"
                )
    material_index = primitive.get("material")
    if (
        isinstance(material_index, bool)
        or not isinstance(material_index, int)
        or material_index != 0
    ):
        raise OrientedSheetRepairError(
            "source GLB primitive requires the sole material"
        )
    material = _mapping(materials[material_index], "source GLB material")
    pbr = _mapping(
        material.get("pbrMetallicRoughness"),
        "source GLB pbrMetallicRoughness",
    )
    bound_textures: list[int] = []
    for name in ("baseColorTexture", "metallicRoughnessTexture"):
        binding = _mapping(pbr.get(name), f"source GLB {name}")
        texture_index = binding.get("index")
        texcoord = binding.get("texCoord", 0)
        if (
            isinstance(texture_index, bool)
            or not isinstance(texture_index, int)
            or not 0 <= texture_index < len(textures)
            or isinstance(texcoord, bool)
            or not isinstance(texcoord, int)
            or texcoord != 0
        ):
            raise OrientedSheetRepairError(
                f"source GLB {name} binding is invalid"
            )
        bound_textures.append(texture_index)
    if (
        set(bound_textures) != {0, 1}
        or set(texture_sources[index] for index in bound_textures) != {0, 1}
    ):
        raise OrientedSheetRepairError(
            "source GLB PBR bindings do not cover both Pixel3D images exactly"
        )
    return material_index, images


def _referenced_position_extrema(
    positions: Sequence[tuple[float, float, float]],
    triangles: Sequence[tuple[int, int, int]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    referenced = sorted({index for triangle in triangles for index in triangle})
    return (
        tuple(min(positions[index][axis] for index in referenced) for axis in range(3)),
        tuple(max(positions[index][axis] for index in referenced) for axis in range(3)),
    )


def _authenticate_primitive(
    document: Mapping[str, Any],
    binary: bytes,
) -> dict[str, Any]:
    _validate_nonoverlapping_views(document, binary, "source GLB")
    meshes = _list(document, "meshes", "source GLB")
    if len(meshes) != 1:
        raise OrientedSheetRepairError("source GLB must contain exactly one mesh")
    mesh = _mapping(meshes[0], "source GLB mesh")
    primitives = mesh.get("primitives")
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise OrientedSheetRepairError(
            "source GLB must contain exactly one mesh primitive"
        )
    primitive = _mapping(primitives[0], "source GLB primitive")
    if primitive.get("mode", 4) != 4:
        raise OrientedSheetRepairError("source GLB primitive must use TRIANGLES mode")
    attributes = _mapping(
        primitive.get("attributes"), "source GLB primitive attributes"
    )
    if not {"POSITION", "NORMAL", "TEXCOORD_0"} <= set(attributes):
        raise OrientedSheetRepairError(
            "source GLB primitive requires POSITION, NORMAL, and TEXCOORD_0"
        )
    position_values, position_layout = _position_values(
        document, binary, attributes["POSITION"], "source GLB"
    )
    normal_values, normal_layout = _float_attribute_values(
        document,
        binary,
        attributes["NORMAL"],
        semantic="NORMAL",
        expected_type="VEC3",
        components=3,
        expected_count=position_layout["count"],
    )
    texcoord_values, texcoord_layout = _float_attribute_values(
        document,
        binary,
        attributes["TEXCOORD_0"],
        semantic="TEXCOORD_0",
        expected_type="VEC2",
        components=2,
        expected_count=position_layout["count"],
    )
    index_accessor = primitive.get("indices")
    if isinstance(index_accessor, bool) or not isinstance(index_accessor, int):
        raise OrientedSheetRepairError("source GLB primitive requires indices")
    index_layout = _index_layout(
        document, binary, index_accessor, "source GLB"
    )
    index_view = index_layout["buffer_view_index"]
    accessors = document["accessors"]
    for other_index, accessor_value in enumerate(accessors):
        accessor = _mapping(accessor_value, f"source GLB accessor {other_index}")
        if (
            other_index != index_accessor
            and accessor.get("bufferView") == index_view
        ):
            raise OrientedSheetRepairError(
                "index bufferView is shared by a protected accessor"
            )
    material_index, images = _validate_pbr_bindings(
        document,
        binary,
        primitive,
    )
    for image_index, image in enumerate(document["images"]):
        if image.get("bufferView") == index_view:
            raise OrientedSheetRepairError(
                f"index bufferView is shared by embedded image {image_index}"
            )
    indices = _unpack_indices(binary, index_layout)
    _validate_optional_index_extrema(
        index_layout["accessor"],
        indices,
        "source GLB indices",
    )
    triangles = _triplets(indices)
    retained, removed = _partition_triangles(position_values, triangles)
    retained_indices = [value for triangle in retained for value in triangle]
    if (
        (min(indices), max(indices))
        != (min(retained_indices), max(retained_indices))
        or _referenced_position_extrema(position_values, triangles)
        != _referenced_position_extrema(position_values, retained)
    ):
        raise OrientedSheetRepairError(
            "exact-position filtering would change accessor/geometry extrema"
        )
    source_topology = _topology(position_values, triangles)
    if (
        source_topology["exact_position_degenerate_triangle_count"]
        != len(removed)
    ):
        raise OrientedSheetRepairError(
            "exact-position degenerate accounting disagrees with audit v4"
        )
    if not source_topology["all_directed_edge_occurrences_reverse_paired"]:
        raise OrientedSheetRepairError(
            "retained exact-position directed edges are not completely reverse-paired"
        )
    return {
        "primitive": primitive,
        "attributes": dict(attributes),
        "material_index": material_index,
        "position_values": position_values,
        "position_layout": position_layout,
        "normal_values": normal_values,
        "normal_layout": normal_layout,
        "texcoord_0_values": texcoord_values,
        "texcoord_0_layout": texcoord_layout,
        "index_layout": index_layout,
        "source_indices": indices,
        "source_triangles": triangles,
        "retained_triangles": retained,
        "removed_ordinals": removed,
        "source_topology": source_topology,
        "embedded_images": images,
    }


def _authenticate_pixal_manifest(
    payload: Mapping[str, Any],
    *,
    instance_id: str,
    source_path: Path,
    source_record: Mapping[str, Any],
) -> None:
    controlled = _mapping(
        payload.get("controlled_request"), "Pixel3D manifest controlled_request"
    )
    if controlled.get("instance_id") != instance_id:
        raise OrientedSheetRepairError(
            "Pixel3D manifest instance identity changed"
        )
    output = _mapping(payload.get("output"), "Pixel3D manifest output")
    output_size = output.get("size_bytes", output.get("bytes"))
    raw_path = output.get("path")
    if (
        not isinstance(raw_path, str)
        or output.get("sha256") != source_record["sha256"]
        or output_size != source_record["size_bytes"]
    ):
        raise OrientedSheetRepairError(
            "Pixel3D manifest no longer binds the source GLB"
        )


def _authenticate_static_decision(
    payload: Mapping[str, Any],
    *,
    instance_id: str,
    source_record: Mapping[str, Any],
    controlled_request: Mapping[str, Any],
) -> None:
    checks = _mapping(payload.get("checks"), "raw static decision checks")
    review_record = _file_record(
        payload.get("review"),
        "raw static decision review",
    )
    review_path = Path(_text(review_record["path"], "raw static decision review.path"))
    if not review_path.is_absolute() or review_path.is_symlink():
        raise OrientedSheetRepairError(
            "raw static decision review must be an absolute direct file"
        )
    review_path = review_path.resolve()
    if (
        not review_path.is_file()
        or review_path.stat().st_size != review_record["size_bytes"]
        or _sha256_file(review_path) != review_record["sha256"]
    ):
        raise OrientedSheetRepairError(
            "raw static decision review descriptor changed"
        )
    review = _parse_json_bytes(
        review_path.read_bytes(), "raw static decision review"
    )
    review_output = _mapping(
        review.get("pixal_output"), "raw static decision review pixal_output"
    )
    request_sha256 = controlled_request.get("request_sha256")
    if (
        payload.get("schema") != STATIC_DECISION_SCHEMA
        or payload.get("instance_id") != instance_id
        or payload.get("decision") != APPROVED
        or payload.get("state_classification") != RESEARCH
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("next_gate") != "lod_then_species_rig_binding"
        or set(checks) != static_decisions.CHECK_FIELDS
        or any(value is not True for value in checks.values())
        or not isinstance(payload.get("decision_sha256"), str)
        or payload["decision_sha256"] != _hash_without(payload, "decision_sha256")
        or review.get("schema") != static_reviews.REVIEW_SCHEMA
        or review.get("instance_id") != instance_id
        or payload.get("review_sha256") != review.get("review_sha256")
        or review.get("review_sha256") != _hash_without(review, "review_sha256")
        or review.get("automatic_checks", {}).get("overall") != "passed"
        or review_output.get("sha256") != source_record["sha256"]
        or review_output.get("size_bytes") != source_record["size_bytes"]
        or (
            isinstance(request_sha256, str)
            and review.get("request_sha256") != request_sha256
        )
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet repair requires an exact approved raw static decision "
            "with all visual checks passed and a review bound to this Pixel3D GLB"
        )


def _readback(
    *,
    source_document: Mapping[str, Any],
    source_binary: bytes,
    output_data: bytes,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    output_document, output_binary = _parse_glb(output_data, "output GLB")
    output = _authenticate_primitive(output_document, output_binary)
    source_index = source["index_layout"]
    output_index = output["index_layout"]
    if output_index["accessor_index"] != source_index["accessor_index"]:
        raise OrientedSheetRepairError("output index accessor identity changed")
    expected_triangles = source["retained_triangles"]
    if output["source_triangles"] != expected_triangles:
        raise OrientedSheetRepairError(
            "output did not preserve retained index triplets exactly in order"
        )
    restored_document = copy.deepcopy(output_document)
    restored_document["accessors"][source_index["accessor_index"]]["count"] = (
        source_index["count"]
    )
    if restored_document != source_document:
        raise OrientedSheetRepairError(
            "output JSON changed outside the index accessor count"
        )
    if len(output_binary) != len(source_binary):
        raise OrientedSheetRepairError("output BIN length changed")
    start, end = source_index["start"], source_index["end"]
    source_non_index = source_binary[:start] + source_binary[end:]
    output_non_index = output_binary[:start] + output_binary[end:]
    if source_non_index != output_non_index:
        raise OrientedSheetRepairError("output changed BIN bytes outside indices")
    source_images = source["embedded_images"]
    output_images = output["embedded_images"]
    if output_images != source_images:
        raise OrientedSheetRepairError("output embedded image bytes changed")
    pbr_source = _canonical_hash(_pbr_payload(source_document))
    pbr_output = _canonical_hash(_pbr_payload(output_document))
    if pbr_output != pbr_source:
        raise OrientedSheetRepairError("output material/PBR payload changed")
    attributes = source["attributes"]
    attribute_hashes: dict[str, str] = {}
    for name in ("POSITION", "NORMAL", "TEXCOORD_0"):
        view_index = source_document["accessors"][attributes[name]]["bufferView"]
        source_bounds = _buffer_view_interval(
            source_document, source_binary, view_index, f"source {name}"
        )
        output_bounds = _buffer_view_interval(
            output_document, output_binary, view_index, f"output {name}"
        )
        source_payload = source_binary[slice(*source_bounds)]
        output_payload = output_binary[slice(*output_bounds)]
        if source_payload != output_payload:
            raise OrientedSheetRepairError(f"output {name} bytes changed")
        attribute_hashes[name] = _sha256_bytes(source_payload)
    topology = output["source_topology"]
    if (
        topology["exact_position_degenerate_triangle_count"] != 0
        or not topology["all_directed_edge_occurrences_reverse_paired"]
    ):
        raise OrientedSheetRepairError(
            "output topology readback did not close the oriented-sheet gate"
        )
    return {
        "document_change": (
            "none_byte_identical"
            if not source["removed_ordinals"]
            else f"accessors[{source_index['accessor_index']}].count_only"
        ),
        "source_non_index_bin_sha256": _sha256_bytes(source_non_index),
        "output_non_index_bin_sha256": _sha256_bytes(output_non_index),
        "attribute_buffer_view_sha256": attribute_hashes,
        "source_pbr_payload_sha256": pbr_source,
        "output_pbr_payload_sha256": pbr_output,
        "source_embedded_image_sha256s": [
            _sha256_bytes(value) for value in source_images
        ],
        "output_embedded_image_sha256s": [
            _sha256_bytes(value) for value in output_images
        ],
        "retained_index_triplets_preserved_exactly_in_order": True,
        "output_topology": topology,
        "passed": True,
    }


def validate_repair_manifest(value: Any) -> dict[str, Any]:
    """Validate a bounded oriented-sheet manifest without filesystem I/O."""

    manifest = _exact(
        value,
        {
            "schema",
            "implementation_contract",
            "created_at",
            "formal_dataset_registration_authorized",
            "lineage",
            "source_contract",
            "mutation",
            "topology",
            "readback",
            "checks",
            "decision",
            "output",
        },
        "oriented-sheet repair manifest",
    )
    if (
        manifest["schema"] != SCHEMA
        or manifest["implementation_contract"] != IMPLEMENTATION_CONTRACT
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet repair schema/implementation changed"
        )
    _text(manifest["created_at"], "oriented-sheet created_at")
    if manifest["formal_dataset_registration_authorized"] is not False:
        raise OrientedSheetRepairError(
            "oriented-sheet repair cannot authorize formal registration"
        )
    lineage = _exact(
        manifest["lineage"],
        {
            "instance_id",
            "pixal_source",
            "pixal_manifest",
            "static_decision_batch",
            "static_decision",
            "static_decision_value",
            "static_decision_state",
            "raw_four_limbs_usable",
            "raw_pose_riggable",
        },
        "oriented-sheet lineage",
    )
    _identifier(lineage["instance_id"], "oriented-sheet lineage.instance_id")
    for name in (
        "pixal_source",
        "pixal_manifest",
        "static_decision_batch",
        "static_decision",
    ):
        _file_record(lineage[name], f"oriented-sheet lineage.{name}")
    if (
        lineage["static_decision_value"] != APPROVED
        or lineage["static_decision_state"] != RESEARCH
        or lineage["raw_four_limbs_usable"] is not True
        or lineage["raw_pose_riggable"] is not True
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet lineage did not preserve the approved raw decision"
        )
    source = _exact(
        manifest["source_contract"],
        {
            "glb_version",
            "mesh_count",
            "primitive_count",
            "primitive_mode",
            "position_accessor_index",
            "normal_accessor_index",
            "texcoord_0_accessor_index",
            "index_accessor_index",
            "index_component_type",
            "source_index_count",
            "source_triangle_count",
            "source_exact_position_degenerate_triangle_count",
            "source_nondegenerate_triangle_count",
        },
        "oriented-sheet source contract",
    )
    for name in (
        "position_accessor_index",
        "normal_accessor_index",
        "texcoord_0_accessor_index",
        "index_accessor_index",
    ):
        _nonnegative(source[name], f"oriented-sheet source.{name}")
    source_index_count = _positive(
        source["source_index_count"], "oriented-sheet source.source_index_count"
    )
    source_triangles = _positive(
        source["source_triangle_count"],
        "oriented-sheet source.source_triangle_count",
    )
    degenerates = _nonnegative(
        source["source_exact_position_degenerate_triangle_count"],
        "oriented-sheet source.source_exact_position_degenerate_triangle_count",
    )
    retained = _positive(
        source["source_nondegenerate_triangle_count"],
        "oriented-sheet source.source_nondegenerate_triangle_count",
    )
    if (
        source["glb_version"] != 2
        or source["mesh_count"] != 1
        or source["primitive_count"] != 1
        or source["primitive_mode"] != 4
        or source["index_component_type"] not in _INDEX_COMPONENTS
        or source_index_count != source_triangles * 3
        or retained + degenerates != source_triangles
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet source accounting changed"
        )
    mutation = _exact(
        manifest["mutation"],
        {
            "method",
            "canonical_signed_zero",
            "authorized_json_changes",
            "authorized_bin_byte_range",
            "external_geometry_inputs",
            "external_skeleton_inputs",
            "external_weight_inputs",
            "external_material_inputs",
            "external_texture_inputs",
            "animation_inputs",
            "removed_triangle_count",
            "removed_triangle_ordinals_sha256",
            "source_nondegenerate_index_triplets_sha256",
            "output_index_triplets_sha256",
            "output_index_count",
            "output_byte_identical_to_source",
        },
        "oriented-sheet mutation",
    )
    if (
        mutation["method"] != MUTATION_METHOD
        or mutation["canonical_signed_zero"] is not True
        or any(
            mutation[name] != []
            for name in (
                "external_geometry_inputs",
                "external_skeleton_inputs",
                "external_weight_inputs",
                "external_material_inputs",
                "external_texture_inputs",
                "animation_inputs",
            )
        )
        or mutation["removed_triangle_count"] != degenerates
        or mutation["output_index_count"] != retained * 3
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet mutation exceeded the bounded index filter"
        )
    for name in (
        "removed_triangle_ordinals_sha256",
        "source_nondegenerate_index_triplets_sha256",
        "output_index_triplets_sha256",
    ):
        _sha256(mutation[name], f"oriented-sheet mutation.{name}")
    if (
        mutation["source_nondegenerate_index_triplets_sha256"]
        != mutation["output_index_triplets_sha256"]
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet retained index triplets changed"
        )
    expected_changes = (
        []
        if degenerates == 0
        else [f"accessors[{source['index_accessor_index']}].count"]
    )
    byte_range = _exact(
        mutation["authorized_bin_byte_range"],
        {"offset", "source_length", "output_used_length"},
        "oriented-sheet authorized BIN range",
    )
    offset = _nonnegative(byte_range["offset"], "oriented-sheet BIN offset")
    source_length = _positive(
        byte_range["source_length"], "oriented-sheet BIN source_length"
    )
    output_length = _positive(
        byte_range["output_used_length"], "oriented-sheet BIN output length"
    )
    if (
        mutation["authorized_json_changes"] != expected_changes
        or source_length != source_index_count * _INDEX_COMPONENTS[
            source["index_component_type"]
        ][1]
        or output_length != mutation["output_index_count"] * _INDEX_COMPONENTS[
            source["index_component_type"]
        ][1]
        or output_length > source_length
        or offset < 0
        or mutation["output_byte_identical_to_source"] is not (degenerates == 0)
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet authorized byte range/accounting changed"
        )
    topology = _exact(
        manifest["topology"],
        _TOPOLOGY_FIELDS,
        "oriented-sheet topology",
    )
    unique_positions = _positive(
        topology["unique_exact_positions"],
        "oriented-sheet topology.unique_exact_positions",
    )
    directed_occurrences = _positive(
        topology["directed_edge_occurrences"],
        "oriented-sheet topology.directed_edge_occurrences",
    )
    undirected_edges = _positive(
        topology["undirected_edge_count"],
        "oriented-sheet topology.undirected_edge_count",
    )
    maximum_multiplicity = _positive(
        topology["maximum_edge_face_multiplicity"],
        "oriented-sheet topology.maximum_edge_face_multiplicity",
    )
    balanced_multicover = _nonnegative(
        topology["balanced_oriented_multicover_edges_over_two_faces"],
        (
            "oriented-sheet topology."
            "balanced_oriented_multicover_edges_over_two_faces"
        ),
    )
    if (
        topology.get("method") != TOPOLOGY_METHOD
        or topology.get("evaluated_triangle_count") != retained
        or topology.get("exact_position_degenerate_triangle_count") != 0
        or unique_positions < 3
        or directed_occurrences != retained * 3
        or undirected_edges * 2 > directed_occurrences
        or maximum_multiplicity < 2
        or maximum_multiplicity > directed_occurrences
        or balanced_multicover > undirected_edges
        or topology.get("unpaired_oriented_edges") != 0
        or topology.get("unpaired_oriented_edge_occurrences") != 0
        or topology.get("all_directed_edge_occurrences_reverse_paired") is not True
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet topology does not prove complete reverse pairing"
        )
    readback = _exact(
        manifest["readback"],
        {
            "document_change",
            "source_non_index_bin_sha256",
            "output_non_index_bin_sha256",
            "attribute_buffer_view_sha256",
            "source_pbr_payload_sha256",
            "output_pbr_payload_sha256",
            "source_embedded_image_sha256s",
            "output_embedded_image_sha256s",
            "retained_index_triplets_preserved_exactly_in_order",
            "output_topology",
            "passed",
        },
        "oriented-sheet readback",
    )
    for name in (
        "source_non_index_bin_sha256",
        "output_non_index_bin_sha256",
        "source_pbr_payload_sha256",
        "output_pbr_payload_sha256",
    ):
        _sha256(readback[name], f"oriented-sheet readback.{name}")
    attributes = _exact(
        readback["attribute_buffer_view_sha256"],
        {"POSITION", "NORMAL", "TEXCOORD_0"},
        "oriented-sheet readback attribute hashes",
    )
    for name, digest in attributes.items():
        _sha256(digest, f"oriented-sheet readback attribute {name}")
    for name in (
        "source_embedded_image_sha256s",
        "output_embedded_image_sha256s",
    ):
        values = readback[name]
        if not isinstance(values, list) or not values:
            raise OrientedSheetRepairError(
                f"oriented-sheet readback.{name} must be a non-empty list"
            )
        for index, digest in enumerate(values):
            _sha256(digest, f"oriented-sheet readback.{name}[{index}]")
    expected_document_change = (
        "none_byte_identical"
        if degenerates == 0
        else f"accessors[{source['index_accessor_index']}].count_only"
    )
    if (
        readback["document_change"] != expected_document_change
        or readback["source_non_index_bin_sha256"]
        != readback["output_non_index_bin_sha256"]
        or readback["source_pbr_payload_sha256"]
        != readback["output_pbr_payload_sha256"]
        or readback["source_embedded_image_sha256s"]
        != readback["output_embedded_image_sha256s"]
        or readback["retained_index_triplets_preserved_exactly_in_order"] is not True
        or readback["output_topology"] != topology
        or readback["passed"] is not True
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet output readback identity changed"
        )
    checks = _exact(
        manifest["checks"], _CHECK_FIELDS, "oriented-sheet checks"
    )
    if any(item is not True for item in checks.values()):
        raise OrientedSheetRepairError(
            "all oriented-sheet automatic checks must pass"
        )
    decision = _exact(
        manifest["decision"],
        {"status", "rejection_reasons", "next_gate"},
        "oriented-sheet decision",
    )
    expected_status = (
        "passed_byte_identical_noop_pending_derived_static_review"
        if degenerates == 0
        else "passed_bounded_index_repair_pending_derived_static_review"
    )
    if (
        decision["status"] != expected_status
        or decision["rejection_reasons"] != []
        or decision["next_gate"] != NEXT_GATE
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet decision is not a passed bounded result"
        )
    output = _file_record(manifest["output"], "oriented-sheet output")
    if (
        degenerates == 0
        and output["sha256"] != lineage["pixal_source"]["sha256"]
    ):
        raise OrientedSheetRepairError(
            "oriented-sheet no-op output is not byte-identical to its source"
        )
    return copy.deepcopy(dict(manifest))


def repair(
    *,
    instance_id: str,
    source_path: Path,
    expected_source_sha256: str,
    pixal_manifest_path: Path,
    expected_pixal_manifest_sha256: str,
    static_decision_batch_path: Path,
    expected_static_decision_batch_sha256: str,
    static_decision_path: Path,
    expected_static_decision_sha256: str,
    output_path: Path,
    manifest_path: Path,
) -> Path:
    """Run one fail-closed oriented-sheet repair and publish its manifest."""

    instance_id = _identifier(instance_id, "instance_id")
    source_path, source_data = _regular_file(
        source_path, expected_source_sha256, "source Pixel3D GLB"
    )
    pixal_manifest_path, pixal_manifest_bytes = _regular_file(
        pixal_manifest_path,
        expected_pixal_manifest_sha256,
        "Pixel3D manifest",
    )
    static_decision_batch_path, static_decision_batch_bytes = _regular_file(
        static_decision_batch_path,
        expected_static_decision_batch_sha256,
        "canonical raw static decision batch",
    )
    static_decision_path, static_decision_bytes = _regular_file(
        static_decision_path,
        expected_static_decision_sha256,
        "raw static decision",
    )
    source_record = _record(source_path, source_data)
    pixal_manifest = _parse_json_bytes(
        pixal_manifest_bytes, "Pixel3D manifest"
    )
    static_decision_batch = _parse_json_bytes(
        static_decision_batch_bytes,
        "canonical raw static decision batch",
    )
    static_decision = _parse_json_bytes(
        static_decision_bytes, "raw static decision"
    )
    _strict_authority_chain_json(
        static_decision_batch_path,
        static_decision_batch,
    )
    # A self-hashed standalone decision is not an authority.  Replay the
    # canonical batch, its static-review batch, and the indexed record before
    # accepting the selected approval.
    from tools import register_controlled_animal_source_assets as source_registry

    try:
        (
            authenticated_batch_path,
            _decision_batch,
            canonical_decisions,
        ) = source_registry.load_decision_batch(static_decision_batch_path)
    except contracts.ContractError as error:
        raise OrientedSheetRepairError(
            f"canonical raw static decision batch is invalid: {error}"
        ) from error
    canonical = canonical_decisions.get(instance_id)
    if (
        authenticated_batch_path != static_decision_batch_path
        or not isinstance(canonical, Mapping)
        or canonical.get("path") != static_decision_path
        or _canonical(canonical.get("payload")) != _canonical(static_decision)
    ):
        raise OrientedSheetRepairError(
            "raw static decision is not the selected canonical batch record"
        )
    _authenticate_pixal_manifest(
        pixal_manifest,
        instance_id=instance_id,
        source_path=source_path,
        source_record=source_record,
    )
    controlled_request = _mapping(
        pixal_manifest.get("controlled_request"),
        "Pixel3D manifest controlled_request",
    )
    _authenticate_static_decision(
        static_decision,
        instance_id=instance_id,
        source_record=source_record,
        controlled_request=controlled_request,
    )
    source_document, source_binary = _parse_glb(
        source_data, "source Pixel3D GLB"
    )
    source = _authenticate_primitive(source_document, source_binary)
    retained = source["retained_triangles"]
    removed = source["removed_ordinals"]
    index = source["index_layout"]
    if removed:
        output_document = copy.deepcopy(source_document)
        output_document["accessors"][index["accessor_index"]]["count"] = (
            len(retained) * 3
        )
        output_binary_mutable = bytearray(source_binary)
        packed = _pack_indices(
            [item for triangle in retained for item in triangle],
            index["component_type"],
        )
        output_binary_mutable[index["start"] : index["start"] + len(packed)] = packed
        output_binary = bytes(output_binary_mutable)
        output_data = _build_glb(output_document, output_binary)
    else:
        output_data = source_data
    readback = _readback(
        source_document=source_document,
        source_binary=source_binary,
        output_data=output_data,
        source=source,
    )
    output_record = {
        "path": str(Path(output_path).absolute().resolve(strict=False)),
        "sha256": _sha256_bytes(output_data),
        "size_bytes": len(output_data),
    }
    source_triangles = source["source_triangles"]
    mutation = {
        "method": MUTATION_METHOD,
        "canonical_signed_zero": True,
        "authorized_json_changes": (
            []
            if not removed
            else [f"accessors[{index['accessor_index']}].count"]
        ),
        "authorized_bin_byte_range": {
            "offset": index["start"],
            "source_length": index["count"] * index["width"],
            "output_used_length": len(retained) * 3 * index["width"],
        },
        "external_geometry_inputs": [],
        "external_skeleton_inputs": [],
        "external_weight_inputs": [],
        "external_material_inputs": [],
        "external_texture_inputs": [],
        "animation_inputs": [],
        "removed_triangle_count": len(removed),
        "removed_triangle_ordinals_sha256": _ordinal_sha256(removed),
        "source_nondegenerate_index_triplets_sha256": _triplet_sha256(retained),
        "output_index_triplets_sha256": _triplet_sha256(retained),
        "output_index_count": len(retained) * 3,
        "output_byte_identical_to_source": not removed,
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "implementation_contract": IMPLEMENTATION_CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_dataset_registration_authorized": False,
        "lineage": {
            "instance_id": instance_id,
            "pixal_source": source_record,
            "pixal_manifest": _record(
                pixal_manifest_path, pixal_manifest_bytes
            ),
            "static_decision_batch": _record(
                static_decision_batch_path, static_decision_batch_bytes
            ),
            "static_decision": _record(
                static_decision_path, static_decision_bytes
            ),
            "static_decision_value": APPROVED,
            "static_decision_state": RESEARCH,
            "raw_four_limbs_usable": True,
            "raw_pose_riggable": True,
        },
        "source_contract": {
            "glb_version": 2,
            "mesh_count": 1,
            "primitive_count": 1,
            "primitive_mode": 4,
            "position_accessor_index": source["attributes"]["POSITION"],
            "normal_accessor_index": source["attributes"]["NORMAL"],
            "texcoord_0_accessor_index": source["attributes"]["TEXCOORD_0"],
            "index_accessor_index": index["accessor_index"],
            "index_component_type": index["component_type"],
            "source_index_count": index["count"],
            "source_triangle_count": len(source_triangles),
            "source_exact_position_degenerate_triangle_count": len(removed),
            "source_nondegenerate_triangle_count": len(retained),
        },
        "mutation": mutation,
        "topology": readback["output_topology"],
        "readback": readback,
        "checks": {name: True for name in sorted(_CHECK_FIELDS)},
        "decision": {
            "status": (
                "passed_byte_identical_noop_pending_derived_static_review"
                if not removed
                else "passed_bounded_index_repair_pending_derived_static_review"
            ),
            "rejection_reasons": [],
            "next_gate": NEXT_GATE,
        },
        "output": output_record,
    }
    validate_repair_manifest(manifest)
    output_literal = Path(output_path).absolute()
    manifest_literal = Path(manifest_path).absolute()
    if output_literal == source_path or output_literal == manifest_literal:
        raise OrientedSheetRepairError(
            "source, output, and manifest paths must be distinct"
        )
    if output_literal.exists() or output_literal.is_symlink():
        raise OrientedSheetRepairError(f"output already exists: {output_literal}")
    if manifest_literal.exists() or manifest_literal.is_symlink():
        raise OrientedSheetRepairError(
            f"manifest already exists: {manifest_literal}"
        )
    if not output_literal.parent.is_dir() or not manifest_literal.parent.is_dir():
        raise OrientedSheetRepairError("output and manifest parents must exist")
    with output_literal.open("xb") as stream:
        stream.write(output_data)
        stream.flush()
        os.fsync(stream.fileno())
    if (
        output_literal.stat().st_size != len(output_data)
        or _sha256_file(output_literal) != output_record["sha256"]
    ):
        raise OrientedSheetRepairError("published output failed exact readback")
    manifest["output"] = _record(output_literal)
    validate_repair_manifest(manifest)
    payload = (
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with manifest_literal.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return manifest_literal.resolve()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--pixal-manifest", required=True, type=Path)
    parser.add_argument("--expected-pixal-manifest-sha256", required=True)
    parser.add_argument("--static-decision-batch", required=True, type=Path)
    parser.add_argument(
        "--expected-static-decision-batch-sha256",
        required=True,
    )
    parser.add_argument("--static-decision", required=True, type=Path)
    parser.add_argument("--expected-static-decision-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = repair(
            instance_id=args.instance_id,
            source_path=args.source,
            expected_source_sha256=args.expected_source_sha256,
            pixal_manifest_path=args.pixal_manifest,
            expected_pixal_manifest_sha256=args.expected_pixal_manifest_sha256,
            static_decision_batch_path=args.static_decision_batch,
            expected_static_decision_batch_sha256=(
                args.expected_static_decision_batch_sha256
            ),
            static_decision_path=args.static_decision,
            expected_static_decision_sha256=args.expected_static_decision_sha256,
            output_path=args.output,
            manifest_path=args.manifest,
        )
        payload = validate_repair_manifest(
            _parse_json_bytes(manifest.read_bytes(), "published manifest")
        )
    except (OSError, OrientedSheetRepairError) as error:
        print(f"PIXAL_ORIENTED_SHEET_INDEX_REPAIR_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "PIXAL_ORIENTED_SHEET_INDEX_REPAIR_OK "
        f"status={payload['decision']['status']} "
        f"removed={payload['mutation']['removed_triangle_count']} "
        f"output={payload['output']['path']} manifest={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
