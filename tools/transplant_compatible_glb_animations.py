#!/usr/bin/env python3
"""Replace GLB animations without re-exporting target geometry or skin data.

The source and target must expose the same animated node names, parent names,
and numerically compatible local rest transforms.  Only animation accessors and
their buffer views are appended to the target BIN chunk.  Meshes, nodes, skins,
materials, textures, images, and the complete original target BIN prefix remain
byte-for-byte authoritative.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence


SCHEMA = "avengine_compatible_glb_animation_transplant_v1"
GLB_JSON_CHUNK = 0x4E4F534A
GLB_BIN_CHUNK = 0x004E4942
PRESERVED_DOCUMENT_KEYS = (
    "asset",
    "scene",
    "scenes",
    "nodes",
    "meshes",
    "skins",
    "materials",
    "textures",
    "images",
    "samplers",
    "cameras",
    "extensionsUsed",
    "extensionsRequired",
)


class AnimationTransplantError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_input(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise AnimationTransplantError(f"missing or unsafe {label}: {path}")
    return path


def require_output(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise AnimationTransplantError(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise AnimationTransplantError(f"truncated GLB: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(raw):
        raise AnimationTransplantError(f"invalid GLB header: {path}")
    offset = 12
    chunks: dict[int, bytes] = {}
    order: list[int] = []
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise AnimationTransplantError(f"truncated GLB chunk header: {path}")
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        if length < 0 or offset + length > len(raw) or kind in chunks:
            raise AnimationTransplantError(f"invalid GLB chunk table: {path}")
        chunks[kind] = raw[offset : offset + length]
        order.append(kind)
        offset += length
    if order != [GLB_JSON_CHUNK, GLB_BIN_CHUNK]:
        raise AnimationTransplantError(
            f"GLB must contain exactly JSON then BIN chunks: {path}"
        )
    try:
        document = json.loads(
            chunks[GLB_JSON_CHUNK].rstrip(b" \t\r\n\x00").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnimationTransplantError(f"invalid GLB JSON: {path}: {error}") from error
    buffers = document.get("buffers")
    if (
        not isinstance(document, dict)
        or not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], dict)
        or "uri" in buffers[0]
    ):
        raise AnimationTransplantError(f"GLB must use one embedded buffer: {path}")
    try:
        declared_binary = int(buffers[0]["byteLength"])
    except (KeyError, TypeError, ValueError) as error:
        raise AnimationTransplantError(f"GLB buffer length is invalid: {path}") from error
    binary_chunk = chunks[GLB_BIN_CHUNK]
    if declared_binary < 0 or declared_binary > len(binary_chunk):
        raise AnimationTransplantError(f"GLB BIN chunk is truncated: {path}")
    return document, binary_chunk[:declared_binary]


def encode_glb(document: Mapping[str, Any], binary: bytes) -> bytes:
    value = deepcopy(dict(document))
    buffers = value.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise AnimationTransplantError("output document must retain one buffer")
    buffers[0]["byteLength"] = len(binary)
    encoded_json = json.dumps(value, separators=(",", ":")).encode("utf-8")
    encoded_json += b" " * ((4 - len(encoded_json) % 4) % 4)
    encoded_binary = binary + b"\x00" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(encoded_json) + 8 + len(encoded_binary)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<II", len(encoded_json), GLB_JSON_CHUNK),
            encoded_json,
            struct.pack("<II", len(encoded_binary), GLB_BIN_CHUNK),
            encoded_binary,
        )
    )


def write_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def parent_indices(document: Mapping[str, Any]) -> dict[int, int]:
    nodes = document.get("nodes", [])
    result: dict[int, int] = {}
    for parent, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise AnimationTransplantError("GLB node must be an object")
        for child in node.get("children", []):
            if not isinstance(child, int) or not 0 <= child < len(nodes):
                raise AnimationTransplantError("GLB node child index is invalid")
            if child in result:
                raise AnimationTransplantError("GLB node has multiple parents")
            result[child] = parent
    return result


def unique_node_names(document: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, node in enumerate(document.get("nodes", [])):
        name = node.get("name") if isinstance(node, Mapping) else None
        if not isinstance(name, str) or not name:
            continue
        if name in result:
            duplicates.add(name)
        else:
            result[name] = index
    for name in duplicates:
        result.pop(name, None)
    return result


def default_trs(node: Mapping[str, Any]) -> tuple[list[float], list[float], list[float]]:
    if "matrix" in node:
        raise AnimationTransplantError(
            "animated compatibility requires explicit TRS nodes, not matrix nodes"
        )
    translation = [float(value) for value in node.get("translation", [0, 0, 0])]
    rotation = [float(value) for value in node.get("rotation", [0, 0, 0, 1])]
    scale = [float(value) for value in node.get("scale", [1, 1, 1])]
    if len(translation) != 3 or len(rotation) != 4 or len(scale) != 3:
        raise AnimationTransplantError("animated node TRS has an invalid shape")
    if not all(math.isfinite(value) for value in translation + rotation + scale):
        raise AnimationTransplantError("animated node TRS is not finite")
    return translation, rotation, scale


def maximum_abs_difference(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(first, second))


def quaternion_difference(first: Sequence[float], second: Sequence[float]) -> float:
    direct = math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))
    negated = math.sqrt(sum((a + b) ** 2 for a, b in zip(first, second)))
    return min(direct, negated)


def choose_animations(
    document: Mapping[str, Any], requested: Sequence[str]
) -> list[tuple[str, Mapping[str, Any]]]:
    animations = document.get("animations", [])
    if not isinstance(animations, list):
        raise AnimationTransplantError("source animations must be a list")
    selected = []
    for canonical in requested:
        lowered = canonical.lower()
        matches = [
            animation
            for animation in animations
            if isinstance(animation, Mapping)
            and (
                str(animation.get("name", "")).lower() == lowered
                or str(animation.get("name", "")).lower().startswith(lowered + "_")
            )
        ]
        if len(matches) != 1:
            raise AnimationTransplantError(
                f"source action {canonical!r} did not resolve uniquely: "
                f"{[item.get('name') for item in animations if isinstance(item, Mapping)]}"
            )
        selected.append((canonical, matches[0]))
    return selected


def animated_source_nodes(animations: Sequence[tuple[str, Mapping[str, Any]]]) -> set[int]:
    result: set[int] = set()
    for _canonical, animation in animations:
        if animation.get("extensions"):
            raise AnimationTransplantError("animation extensions are unsupported")
        for channel in animation.get("channels", []):
            if not isinstance(channel, Mapping) or channel.get("extensions"):
                raise AnimationTransplantError("animation channel is unsupported")
            target = channel.get("target")
            if not isinstance(target, Mapping) or not isinstance(target.get("node"), int):
                raise AnimationTransplantError("animation channel target node is invalid")
            if target.get("path") not in {"translation", "rotation", "scale", "weights"}:
                raise AnimationTransplantError("animation channel target path is unsupported")
            result.add(target["node"])
    return result


def compatible_node_map(
    target: Mapping[str, Any],
    source: Mapping[str, Any],
    source_indices: set[int],
    *,
    tolerance: float,
) -> tuple[dict[int, int], dict[str, Any]]:
    if not 0.0 < tolerance <= 1.0e-3:
        raise AnimationTransplantError("rest transform tolerance must be in (0, 1e-3]")
    target_nodes = target.get("nodes", [])
    source_nodes = source.get("nodes", [])
    target_names = unique_node_names(target)
    source_parents = parent_indices(source)
    target_parents = parent_indices(target)
    mapping: dict[int, int] = {}
    errors = {"translation": 0.0, "rotation": 0.0, "scale": 0.0}
    records = []
    for source_index in sorted(source_indices):
        if not 0 <= source_index < len(source_nodes):
            raise AnimationTransplantError("animation references an invalid source node")
        source_node = source_nodes[source_index]
        name = source_node.get("name") if isinstance(source_node, Mapping) else None
        if not isinstance(name, str) or name not in target_names:
            raise AnimationTransplantError(
                f"animated source node lacks a unique target name: {name!r}"
            )
        target_index = target_names[name]
        target_node = target_nodes[target_index]
        source_parent = source_parents.get(source_index)
        target_parent = target_parents.get(target_index)
        source_parent_name = (
            source_nodes[source_parent].get("name") if source_parent is not None else None
        )
        target_parent_name = (
            target_nodes[target_parent].get("name") if target_parent is not None else None
        )
        if source_parent_name != target_parent_name:
            raise AnimationTransplantError(
                f"animated node parent mismatch for {name}: "
                f"source={source_parent_name!r} target={target_parent_name!r}"
            )
        source_trs = default_trs(source_node)
        target_trs = default_trs(target_node)
        translation_error = maximum_abs_difference(source_trs[0], target_trs[0])
        rotation_error = quaternion_difference(source_trs[1], target_trs[1])
        scale_error = maximum_abs_difference(source_trs[2], target_trs[2])
        errors["translation"] = max(errors["translation"], translation_error)
        errors["rotation"] = max(errors["rotation"], rotation_error)
        errors["scale"] = max(errors["scale"], scale_error)
        if max(translation_error, rotation_error, scale_error) > tolerance:
            raise AnimationTransplantError(
                f"animated node rest transform mismatch for {name}: "
                f"translation={translation_error:.9g} rotation={rotation_error:.9g} "
                f"scale={scale_error:.9g} tolerance={tolerance:.9g}"
            )
        mapping[source_index] = target_index
        records.append(
            {
                "name": name,
                "source_node": source_index,
                "target_node": target_index,
                "parent_name": source_parent_name,
            }
        )
    return mapping, {
        "animated_node_count": len(mapping),
        "exact_parent_name_mapping": True,
        "rest_transform_tolerance": tolerance,
        "maximum_translation_component_error": errors["translation"],
        "maximum_sign_invariant_quaternion_error": errors["rotation"],
        "maximum_scale_component_error": errors["scale"],
        "nodes": records,
    }


def preserved_subset(document: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(document[key]) for key in PRESERVED_DOCUMENT_KEYS if key in document}


def transplant_animations(
    target: Mapping[str, Any],
    target_binary: bytes,
    source: Mapping[str, Any],
    source_binary: bytes,
    requested: Sequence[str],
    *,
    rest_tolerance: float = 1.0e-5,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    selected = choose_animations(source, requested)
    source_nodes = animated_source_nodes(selected)
    node_map, compatibility = compatible_node_map(
        target, source, source_nodes, tolerance=rest_tolerance
    )
    output = deepcopy(dict(target))
    output.setdefault("bufferViews", [])
    output.setdefault("accessors", [])
    source_views = source.get("bufferViews", [])
    source_accessors = source.get("accessors", [])
    if not isinstance(source_views, list) or not isinstance(source_accessors, list):
        raise AnimationTransplantError("source accessors and bufferViews must be lists")
    binary = bytearray(target_binary)
    copied_views: dict[int, int] = {}
    copied_accessors: dict[int, int] = {}

    def copy_view(index: int) -> int:
        if index in copied_views:
            return copied_views[index]
        if not isinstance(index, int) or not 0 <= index < len(source_views):
            raise AnimationTransplantError("animation accessor bufferView is invalid")
        source_view = source_views[index]
        if (
            not isinstance(source_view, Mapping)
            or source_view.get("buffer", 0) != 0
            or source_view.get("extensions")
        ):
            raise AnimationTransplantError("animation bufferView is unsupported")
        start = int(source_view.get("byteOffset", 0))
        length = int(source_view.get("byteLength", -1))
        if start < 0 or length < 0 or start + length > len(source_binary):
            raise AnimationTransplantError("animation bufferView exceeds source BIN")
        while len(binary) % 4:
            binary.append(0)
        copied = deepcopy(dict(source_view))
        copied["buffer"] = 0
        copied["byteOffset"] = len(binary)
        output["bufferViews"].append(copied)
        copied_index = len(output["bufferViews"]) - 1
        binary.extend(source_binary[start : start + length])
        copied_views[index] = copied_index
        return copied_index

    def copy_accessor(index: int) -> int:
        if index in copied_accessors:
            return copied_accessors[index]
        if not isinstance(index, int) or not 0 <= index < len(source_accessors):
            raise AnimationTransplantError("animation sampler accessor is invalid")
        source_accessor = source_accessors[index]
        if not isinstance(source_accessor, Mapping) or source_accessor.get("extensions"):
            raise AnimationTransplantError("animation accessor is unsupported")
        copied = deepcopy(dict(source_accessor))
        if "bufferView" in copied:
            copied["bufferView"] = copy_view(int(copied["bufferView"]))
        sparse = copied.get("sparse")
        if sparse is not None:
            if not isinstance(sparse, dict):
                raise AnimationTransplantError("animation sparse accessor is invalid")
            for key in ("indices", "values"):
                record = sparse.get(key)
                if not isinstance(record, dict) or "bufferView" not in record:
                    raise AnimationTransplantError("animation sparse accessor is invalid")
                record["bufferView"] = copy_view(int(record["bufferView"]))
        output["accessors"].append(copied)
        copied_index = len(output["accessors"]) - 1
        copied_accessors[index] = copied_index
        return copied_index

    transplanted = []
    for canonical, source_animation in selected:
        animation = deepcopy(dict(source_animation))
        animation["name"] = canonical
        samplers = animation.get("samplers")
        channels = animation.get("channels")
        if not isinstance(samplers, list) or not isinstance(channels, list):
            raise AnimationTransplantError("source animation structure is invalid")
        for sampler in samplers:
            if not isinstance(sampler, dict) or sampler.get("extensions"):
                raise AnimationTransplantError("animation sampler is unsupported")
            sampler["input"] = copy_accessor(int(sampler["input"]))
            sampler["output"] = copy_accessor(int(sampler["output"]))
        for channel in channels:
            source_node = int(channel["target"]["node"])
            channel["target"]["node"] = node_map[source_node]
        transplanted.append(animation)
    output["animations"] = transplanted
    output["buffers"][0]["byteLength"] = len(binary)

    target_subset = preserved_subset(target)
    output_subset = preserved_subset(output)
    if canonical_json(target_subset) != canonical_json(output_subset):
        raise AnimationTransplantError("target non-animation document fields changed")
    if bytes(binary[: len(target_binary)]) != target_binary:
        raise AnimationTransplantError("target BIN prefix changed")
    record = {
        "requested_actions": list(requested),
        "source_action_names": [animation.get("name") for _, animation in selected],
        "output_action_names": [animation["name"] for animation in transplanted],
        "compatibility": compatibility,
        "preservation": {
            "preserved_document_keys": list(target_subset),
            "preserved_document_sha256_before": json_sha256(target_subset),
            "preserved_document_sha256_after": json_sha256(output_subset),
            "target_binary_prefix_size_bytes": len(target_binary),
            "target_binary_prefix_sha256_before": sha256_bytes(target_binary),
            "target_binary_prefix_sha256_after": sha256_bytes(
                bytes(binary[: len(target_binary)])
            ),
            "target_binary_prefix_unchanged": True,
            "target_accessor_count": len(target.get("accessors", [])),
            "appended_accessor_count": len(output["accessors"])
            - len(target.get("accessors", [])),
            "target_buffer_view_count": len(target.get("bufferViews", [])),
            "appended_buffer_view_count": len(output["bufferViews"])
            - len(target.get("bufferViews", [])),
        },
    }
    return output, bytes(binary), record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-glb", type=Path, required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--rest-tolerance", type=float, default=1.0e-5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        target_path = require_input(args.target_glb, "target GLB")
        source_path = require_input(args.source_glb, "source GLB")
        output_path = require_output(args.output_glb, "output GLB")
        manifest_path = require_output(args.manifest, "output manifest")
        if output_path == manifest_path:
            raise AnimationTransplantError("GLB and manifest outputs must differ")
        actions = args.action or ["Walking", "Idle"]
        if len(actions) != len(set(actions)) or not all(actions):
            raise AnimationTransplantError("requested actions must be unique and non-empty")
        target, target_binary = read_glb(target_path)
        source, source_binary = read_glb(source_path)
        output, output_binary, transplant = transplant_animations(
            target,
            target_binary,
            source,
            source_binary,
            actions,
            rest_tolerance=args.rest_tolerance,
        )
        encoded = encode_glb(output, output_binary)
        write_exclusive(output_path, encoded)
        observed, observed_binary = read_glb(output_path)
        if (
            [item.get("name") for item in observed.get("animations", [])] != actions
            or observed_binary[: len(target_binary)] != target_binary
            or canonical_json(preserved_subset(observed))
            != canonical_json(preserved_subset(target))
        ):
            raise AnimationTransplantError("written GLB failed readback authentication")
        manifest = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": {
                "path": str(target_path),
                "sha256": sha256_file(target_path),
                "size_bytes": target_path.stat().st_size,
                "geometry_skin_material_authority": True,
            },
            "animation_source": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "size_bytes": source_path.stat().st_size,
                "geometry_used": False,
                "skin_used": False,
                "material_used": False,
            },
            "transplant": transplant,
            "output": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "size_bytes": output_path.stat().st_size,
                "readback_passed": True,
            },
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
        }
        write_json_exclusive(manifest_path, manifest)
    except (AnimationTransplantError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"COMPATIBLE_GLB_ANIMATION_TRANSPLANT_FAILED {error}", flush=True)
        return 2
    print(
        f"COMPATIBLE_GLB_ANIMATION_TRANSPLANT_OK output={output_path} "
        f"actions={actions}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
