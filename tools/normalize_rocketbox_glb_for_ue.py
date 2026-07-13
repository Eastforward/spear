#!/usr/bin/env python3
"""Normalize a native Rocketbox GLB so its skinned mesh is a scene root."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np


class RocketboxGlbNormalizationError(RuntimeError):
    """Raised when the expected native Rocketbox node topology is absent."""


_FLOAT_COMPONENT_TYPE = 5126
_ACCESSOR_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def read_glb_bytes(payload: bytes) -> tuple[dict[str, object], bytes]:
    if len(payload) < 28 or payload[:4] != b"glTF":
        raise RocketboxGlbNormalizationError("input is not a GLB file")
    version, declared_length = struct.unpack_from("<II", payload, 4)
    if version != 2 or declared_length != len(payload):
        raise RocketboxGlbNormalizationError("GLB header is invalid")
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise RocketboxGlbNormalizationError("GLB chunk header is truncated")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(payload):
            raise RocketboxGlbNormalizationError("GLB chunk is truncated")
        chunks.append((chunk_type, payload[offset:end]))
        offset = end
    if len(chunks) != 2 or chunks[0][0] != 0x4E4F534A or chunks[1][0] != 0x004E4942:
        raise RocketboxGlbNormalizationError("GLB must contain one JSON and one BIN chunk")
    try:
        document = json.loads(chunks[0][1].rstrip(b" \t\r\n\0").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RocketboxGlbNormalizationError("GLB JSON is invalid") from error
    if not isinstance(document, dict):
        raise RocketboxGlbNormalizationError("GLB JSON must be an object")
    return document, chunks[1][1]


def _float_accessor_layout(
    document: dict[str, object], binary: bytes | bytearray, index: int
) -> tuple[dict[str, object], int, int, int, int]:
    accessors = document.get("accessors")
    views = document.get("bufferViews")
    if (
        not isinstance(accessors, list)
        or not isinstance(views, list)
        or not isinstance(index, int)
        or index < 0
        or index >= len(accessors)
        or not isinstance(accessors[index], dict)
    ):
        raise RocketboxGlbNormalizationError("float accessor is invalid")
    accessor = accessors[index]
    if (
        accessor.get("componentType") != _FLOAT_COMPONENT_TYPE
        or accessor.get("type") not in _ACCESSOR_COMPONENT_COUNTS
        or "sparse" in accessor
    ):
        raise RocketboxGlbNormalizationError(
            "normalization requires a non-sparse float accessor"
        )
    view_index = accessor.get("bufferView")
    if (
        not isinstance(view_index, int)
        or view_index < 0
        or view_index >= len(views)
        or not isinstance(views[view_index], dict)
    ):
        raise RocketboxGlbNormalizationError("float accessor bufferView is invalid")
    view = views[view_index]
    if view.get("buffer", 0) != 0:
        raise RocketboxGlbNormalizationError("only the embedded GLB buffer is supported")
    count = accessor.get("count")
    if not isinstance(count, int) or count <= 0:
        raise RocketboxGlbNormalizationError("float accessor count is invalid")
    components = _ACCESSOR_COMPONENT_COUNTS[accessor["type"]]
    item_size = components * 4
    stride = view.get("byteStride", item_size)
    if not isinstance(stride, int) or stride < item_size or stride % 4:
        raise RocketboxGlbNormalizationError("float accessor stride is invalid")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    end = start + (count - 1) * stride + item_size
    view_start = int(view.get("byteOffset", 0))
    view_end = view_start + int(view.get("byteLength", 0))
    if start < view_start or end > view_end or end > len(binary):
        raise RocketboxGlbNormalizationError("float accessor exceeds its bufferView")
    return accessor, start, count, components, stride


def read_float_accessor(
    document: dict[str, object], binary: bytes | bytearray, index: int
) -> np.ndarray:
    _accessor, start, count, components, stride = _float_accessor_layout(
        document, binary, index
    )
    raw = np.frombuffer(binary, dtype=np.uint8)
    return np.ndarray(
        shape=(count, components),
        dtype="<f4",
        buffer=raw,
        offset=start,
        strides=(stride, 4),
    ).copy()


def _write_float_accessor(
    document: dict[str, object], binary: bytearray, index: int, values: np.ndarray
) -> None:
    accessor, start, count, components, stride = _float_accessor_layout(
        document, binary, index
    )
    array = np.asarray(values, dtype="<f4")
    if array.shape != (count, components) or not np.isfinite(array).all():
        raise RocketboxGlbNormalizationError("replacement accessor values are invalid")
    raw = np.frombuffer(binary, dtype=np.uint8)
    target = np.ndarray(
        shape=(count, components),
        dtype="<f4",
        buffer=raw,
        offset=start,
        strides=(stride, 4),
    )
    target[:] = array
    if "min" in accessor:
        accessor["min"] = array.min(axis=0).astype(float).tolist()
    if "max" in accessor:
        accessor["max"] = array.max(axis=0).astype(float).tolist()


def _accessor_storage_bytes(
    document: dict[str, object], binary: bytes | bytearray, index: int
) -> bytes:
    _accessor, start, count, components, stride = _float_accessor_layout(
        document, binary, index
    )
    item_size = components * 4
    return b"".join(
        bytes(binary[start + row * stride : start + row * stride + item_size])
        for row in range(count)
    )


def _quaternion_matrix(values: list[float]) -> np.ndarray:
    if not isinstance(values, list) or len(values) != 4:
        raise RocketboxGlbNormalizationError("node quaternion is invalid")
    x, y, z, w = (float(value) for value in values)
    norm = x * x + y * y + z * z + w * w
    if not np.isfinite(norm) or norm <= 1.0e-20:
        raise RocketboxGlbNormalizationError("node quaternion is degenerate")
    scale = 2.0 / norm
    return np.asarray(
        [
            [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _node_local_matrix(node: dict[str, object]) -> np.ndarray:
    if "matrix" in node:
        raise RocketboxGlbNormalizationError("metric normalization requires node TRS")
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    if translation.shape != (3,) or scale.shape != (3,) or not (
        np.isfinite(translation).all() and np.isfinite(scale).all()
    ):
        raise RocketboxGlbNormalizationError("node TRS is invalid")
    rotation = _quaternion_matrix(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _build_parent_map(nodes: list[dict[str, object]]) -> dict[int, int]:
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        for child in node.get("children", []):
            if child in parents:
                raise RocketboxGlbNormalizationError("GLB node has multiple parents")
            parents[child] = parent_index
    return parents


def _global_node_matrix(
    nodes: list[dict[str, object]], parents: dict[int, int], index: int
) -> np.ndarray:
    chain = []
    seen = set()
    current = index
    while True:
        if current in seen:
            raise RocketboxGlbNormalizationError("GLB node graph contains a cycle")
        seen.add(current)
        chain.append(current)
        if current not in parents:
            break
        current = parents[current]
    matrix = np.eye(4, dtype=np.float64)
    for node_index in reversed(chain):
        matrix = matrix @ _node_local_matrix(nodes[node_index])
    return matrix


def promote_skinned_mesh_to_scene_root(
    document: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    nodes = document.get("nodes")
    scenes = document.get("scenes")
    scene_index = document.get("scene", 0)
    if (
        not isinstance(nodes, list)
        or not isinstance(scenes, list)
        or not isinstance(scene_index, int)
        or scene_index < 0
        or scene_index >= len(scenes)
        or not isinstance(scenes[scene_index], dict)
    ):
        raise RocketboxGlbNormalizationError("GLB scene graph is invalid")
    mesh_nodes = [
        index
        for index, node in enumerate(nodes)
        if isinstance(node, dict) and "mesh" in node and "skin" in node
    ]
    if len(mesh_nodes) != 1:
        raise RocketboxGlbNormalizationError(
            "GLB must contain exactly one skinned mesh node"
        )
    mesh_index = mesh_nodes[0]
    mesh_node = nodes[mesh_index]
    if any(key in mesh_node for key in ("matrix", "translation", "rotation", "scale")):
        raise RocketboxGlbNormalizationError(
            "skinned mesh node must have an identity transform"
        )

    parents = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise RocketboxGlbNormalizationError("GLB node is not an object")
        children = node.get("children", [])
        if not isinstance(children, list) or any(
            not isinstance(child, int) or child < 0 or child >= len(nodes)
            for child in children
        ):
            raise RocketboxGlbNormalizationError("GLB node children are invalid")
        if mesh_index in children:
            parents.append(index)
    if len(parents) != 1:
        raise RocketboxGlbNormalizationError(
            "skinned mesh node must have exactly one parent before normalization"
        )
    armature_index = parents[0]
    scene = scenes[scene_index]
    roots = scene.get("nodes")
    if not isinstance(roots, list) or armature_index not in roots or mesh_index in roots:
        raise RocketboxGlbNormalizationError(
            "skinned mesh parent must be a scene root before normalization"
        )
    armature_node = nodes[armature_index]
    old_roots = list(roots)
    old_children = list(armature_node.get("children", []))
    new_children = [child for child in old_children if child != mesh_index]
    armature_node["children"] = new_children
    roots.append(mesh_index)
    evidence = {
        "schema": "rocketbox_ue_skinned_mesh_root_normalization_v1",
        "scene_index": scene_index,
        "armature_node_index": armature_index,
        "armature_node_name": armature_node.get("name"),
        "mesh_node_index": mesh_index,
        "mesh_node_name": mesh_node.get("name"),
        "old_scene_roots": old_roots,
        "new_scene_roots": list(roots),
        "old_armature_children": old_children,
        "new_armature_children": new_children,
        "mesh_binary_payload_unchanged": True,
    }
    return document, evidence


def normalize_metric_glb_bytes(
    payload: bytes, *, expected_unit_scale: float = 0.01
) -> tuple[bytes, dict[str, object]]:
    document, source_binary = read_glb_bytes(payload)
    document, root_evidence = promote_skinned_mesh_to_scene_root(document)
    nodes = document.get("nodes")
    skins = document.get("skins")
    animations = document.get("animations", [])
    if (
        not isinstance(nodes, list)
        or not isinstance(skins, list)
        or len(skins) != 1
        or not isinstance(skins[0], dict)
        or not isinstance(animations, list)
    ):
        raise RocketboxGlbNormalizationError("metric skin contract is invalid")
    armature_index = root_evidence["armature_node_index"]
    armature = nodes[armature_index]
    root_scale = np.asarray(armature.get("scale"), dtype=np.float64)
    if (
        root_scale.shape != (3,)
        or not np.allclose(
            root_scale,
            np.full(3, expected_unit_scale),
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        raise RocketboxGlbNormalizationError(
            "armature root does not contain the expected centimeter unit scale"
        )
    skin = skins[0]
    joints = skin.get("joints")
    inverse_bind_accessor = skin.get("inverseBindMatrices")
    if (
        not isinstance(joints, list)
        or not joints
        or len(set(joints)) != len(joints)
        or any(
            not isinstance(index, int) or index < 0 or index >= len(nodes)
            for index in joints
        )
        or not isinstance(inverse_bind_accessor, int)
    ):
        raise RocketboxGlbNormalizationError("skin joints or inverse binds are invalid")
    inverse_binds_before = read_float_accessor(
        document, source_binary, inverse_bind_accessor
    )
    if inverse_binds_before.shape != (len(joints), 16):
        raise RocketboxGlbNormalizationError("inverse bind accessor shape changed")

    mesh_position_accessors = sorted(
        {
            primitive.get("attributes", {}).get("POSITION")
            for mesh in document.get("meshes", [])
            if isinstance(mesh, dict)
            for primitive in mesh.get("primitives", [])
            if isinstance(primitive, dict)
            and isinstance(primitive.get("attributes", {}).get("POSITION"), int)
        }
    )
    if not mesh_position_accessors:
        raise RocketboxGlbNormalizationError("GLB has no mesh POSITION accessors")
    position_bytes_before = {
        index: _accessor_storage_bytes(document, source_binary, index)
        for index in mesh_position_accessors
    }

    for joint_index in joints:
        joint = nodes[joint_index]
        if "matrix" in joint:
            raise RocketboxGlbNormalizationError("joint matrix cannot be metric-normalized")
        translation = np.asarray(
            joint.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise RocketboxGlbNormalizationError("joint translation is invalid")
        joint["translation"] = (translation * expected_unit_scale).tolist()
    armature["scale"] = [1.0, 1.0, 1.0]

    binary = bytearray(source_binary)
    accessor_operations: dict[int, str] = {}
    joint_translation_accessors = set()
    root_scale_accessors = set()
    for animation in animations:
        if not isinstance(animation, dict):
            raise RocketboxGlbNormalizationError("animation is not an object")
        samplers = animation.get("samplers")
        channels = animation.get("channels")
        if not isinstance(samplers, list) or not isinstance(channels, list):
            raise RocketboxGlbNormalizationError("animation channels are invalid")
        for channel in channels:
            if not isinstance(channel, dict) or not isinstance(channel.get("target"), dict):
                raise RocketboxGlbNormalizationError("animation channel is invalid")
            sampler_index = channel.get("sampler")
            target = channel["target"]
            node_index = target.get("node")
            path = target.get("path")
            if (
                not isinstance(sampler_index, int)
                or sampler_index < 0
                or sampler_index >= len(samplers)
                or not isinstance(samplers[sampler_index], dict)
            ):
                raise RocketboxGlbNormalizationError("animation sampler is invalid")
            output_accessor = samplers[sampler_index].get("output")
            if not isinstance(output_accessor, int):
                raise RocketboxGlbNormalizationError("animation output is invalid")
            operation = None
            if node_index in joints and path == "translation":
                operation = "joint_translation"
                joint_translation_accessors.add(output_accessor)
            elif node_index == armature_index and path == "scale":
                operation = "root_scale"
                root_scale_accessors.add(output_accessor)
            if operation is None:
                continue
            previous = accessor_operations.setdefault(output_accessor, operation)
            if previous != operation:
                raise RocketboxGlbNormalizationError(
                    "animation accessor has conflicting metric semantics"
                )

    for accessor_index in sorted(joint_translation_accessors):
        values = read_float_accessor(document, binary, accessor_index)
        if values.shape[1] != 3:
            raise RocketboxGlbNormalizationError(
                "joint translation output must be VEC3"
            )
        _write_float_accessor(
            document, binary, accessor_index, values * expected_unit_scale
        )
    for accessor_index in sorted(root_scale_accessors):
        values = read_float_accessor(document, binary, accessor_index)
        if values.shape[1] != 3 or not np.allclose(
            values,
            np.full_like(values, expected_unit_scale),
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise RocketboxGlbNormalizationError(
                "animated armature scale is not the fixed unit conversion"
            )
        _write_float_accessor(document, binary, accessor_index, np.ones_like(values))

    parents = _build_parent_map(nodes)
    for joint_index in joints:
        current = joint_index
        seen = set()
        while current != armature_index:
            if current in seen or current not in parents:
                raise RocketboxGlbNormalizationError(
                    "skin joint is not a descendant of the armature root"
                )
            seen.add(current)
            current = parents[current]
    inverse_binds = np.empty((len(joints), 16), dtype="<f4")
    for offset, joint_index in enumerate(joints):
        global_matrix = _global_node_matrix(nodes, parents, joint_index)
        if abs(float(np.linalg.det(global_matrix))) <= 1.0e-12:
            raise RocketboxGlbNormalizationError("joint rest matrix is singular")
        inverse_binds[offset] = np.linalg.inv(global_matrix).reshape(16, order="F")
    _write_float_accessor(
        document, binary, inverse_bind_accessor, inverse_binds
    )

    unchanged_positions = []
    for index in mesh_position_accessors:
        if _accessor_storage_bytes(document, binary, index) != position_bytes_before[index]:
            raise RocketboxGlbNormalizationError("metric normalization changed mesh positions")
        unchanged_positions.append(index)
    if len(binary) != len(source_binary):
        raise RocketboxGlbNormalizationError("metric normalization resized the BIN chunk")
    encoded = json.dumps(
        document, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    output = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded) + 8 + len(binary))
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + bytes(binary)
    )
    evidence = {
        **root_evidence,
        "schema": "rocketbox_ue_metric_skeleton_normalization_v1",
        "source_unit_scale": expected_unit_scale,
        "armature_translation_preserved": True,
        "armature_rotation_preserved": True,
        "normalized_joint_count": len(joints),
        "normalized_joint_translation_accessor_count": len(
            joint_translation_accessors
        ),
        "normalized_root_scale_accessor_count": len(root_scale_accessors),
        "recomputed_inverse_bind_matrix_count": len(joints),
        "inverse_bind_accessor": inverse_bind_accessor,
        "mesh_position_accessors_unchanged": unchanged_positions,
        "binary_chunk_size_unchanged": True,
    }
    return output, evidence


def normalize_grounded_metric_glb_bytes(
    payload: bytes, *, expected_unit_scale: float = 0.01
) -> tuple[bytes, dict[str, object]]:
    metric_payload, metric_evidence = normalize_metric_glb_bytes(
        payload, expected_unit_scale=expected_unit_scale
    )
    document, metric_binary = read_glb_bytes(metric_payload)
    binary = bytearray(metric_binary)
    nodes = document["nodes"]
    armature_index = metric_evidence["armature_node_index"]
    armature = nodes[armature_index]
    joints = document["skins"][0]["joints"]
    inverse_bind_accessor = document["skins"][0]["inverseBindMatrices"]
    parents = _build_parent_map(nodes)
    root_joints = [index for index in joints if parents.get(index) == armature_index]
    if len(root_joints) != 1:
        raise RocketboxGlbNormalizationError(
            "metric skeleton must have one joint directly below the armature wrapper"
        )
    pelvis_index = root_joints[0]
    pelvis = nodes[pelvis_index]
    wrapper_translation = np.asarray(
        armature.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    wrapper_rotation = _quaternion_matrix(
        armature.get("rotation", [0.0, 0.0, 0.0, 1.0])
    )
    pelvis_translation = np.asarray(
        pelvis.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    if wrapper_translation.shape != (3,) or pelvis_translation.shape != (3,):
        raise RocketboxGlbNormalizationError("wrapper or pelvis translation is invalid")
    static_global_before = _global_node_matrix(nodes, parents, pelvis_index)
    pelvis["translation"] = (
        wrapper_rotation.T @ wrapper_translation + pelvis_translation
    ).tolist()
    armature["translation"] = [0.0, 0.0, 0.0]
    static_global_after = _global_node_matrix(nodes, parents, pelvis_index)
    if not np.allclose(static_global_after, static_global_before, rtol=0.0, atol=1.0e-6):
        raise RocketboxGlbNormalizationError(
            "static wrapper translation absorption changed the pelvis transform"
        )

    absorbed_animation_count = 0
    removed_pelvis_channels = 0
    maximum_pelvis_translation_world_error = 0.0
    for animation in document.get("animations", []):
        channels = animation.get("channels", [])
        samplers = animation.get("samplers", [])
        wrapper_channels = [
            channel
            for channel in channels
            if channel.get("target")
            == {"node": armature_index, "path": "translation"}
        ]
        pelvis_channels = [
            channel
            for channel in channels
            if channel.get("target")
            == {"node": pelvis_index, "path": "translation"}
        ]
        rotation_channels = [
            channel
            for channel in channels
            if channel.get("target")
            == {"node": armature_index, "path": "rotation"}
        ]
        if len(wrapper_channels) != 1 or len(pelvis_channels) != 1:
            raise RocketboxGlbNormalizationError(
                "each action must animate wrapper and pelvis translation exactly once"
            )
        if len(rotation_channels) > 1:
            raise RocketboxGlbNormalizationError(
                "action contains duplicate wrapper rotation channels"
            )
        wrapper_channel = wrapper_channels[0]
        pelvis_channel = pelvis_channels[0]
        wrapper_sampler = samplers[wrapper_channel["sampler"]]
        pelvis_sampler = samplers[pelvis_channel["sampler"]]
        pelvis_interpolation = pelvis_sampler.get("interpolation", "LINEAR")
        if wrapper_sampler.get("interpolation", "LINEAR") != "LINEAR" or (
            pelvis_interpolation not in {"LINEAR", "STEP"}
        ):
            raise RocketboxGlbNormalizationError(
                "wrapper/pelvis translation interpolation is unsupported"
            )
        wrapper_times = read_float_accessor(
            document, binary, wrapper_sampler["input"]
        ).reshape(-1)
        wrapper_values = read_float_accessor(
            document, binary, wrapper_sampler["output"]
        )
        pelvis_times = read_float_accessor(
            document, binary, pelvis_sampler["input"]
        ).reshape(-1)
        pelvis_values = read_float_accessor(
            document, binary, pelvis_sampler["output"]
        )
        if (
            wrapper_values.shape != (len(wrapper_times), 3)
            or pelvis_values.shape != (len(pelvis_times), 3)
            or np.any(np.diff(wrapper_times) < 0.0)
            or np.any(np.diff(pelvis_times) < 0.0)
            or wrapper_times[0] < pelvis_times[0] - 1.0e-6
            or wrapper_times[-1] > pelvis_times[-1] + 1.0e-6
        ):
            raise RocketboxGlbNormalizationError(
                "wrapper/pelvis translation time domains are incompatible"
            )
        if pelvis_interpolation == "STEP":
            sample_indices = np.searchsorted(
                pelvis_times, wrapper_times, side="right"
            ) - 1
            sample_indices = np.clip(sample_indices, 0, len(pelvis_times) - 1)
            sampled_pelvis = pelvis_values[sample_indices]
        else:
            sampled_pelvis = np.column_stack(
                [
                    np.interp(wrapper_times, pelvis_times, pelvis_values[:, axis])
                    for axis in range(3)
                ]
            )
        if rotation_channels:
            rotation_sampler = samplers[rotation_channels[0]["sampler"]]
            rotation_times = read_float_accessor(
                document, binary, rotation_sampler["input"]
            ).reshape(-1)
            rotations = read_float_accessor(
                document, binary, rotation_sampler["output"]
            )
            if not np.array_equal(rotation_times, wrapper_times) or rotations.shape != (
                len(wrapper_times),
                4,
            ):
                raise RocketboxGlbNormalizationError(
                    "wrapper rotation and translation samples must share exact times"
                )
            rotation_matrices = [_quaternion_matrix(row.tolist()) for row in rotations]
        else:
            rotation_matrices = [wrapper_rotation] * len(wrapper_times)
        absorbed = np.empty_like(wrapper_values, dtype=np.float64)
        for index, rotation in enumerate(rotation_matrices):
            absorbed[index] = (
                rotation.T @ wrapper_values[index] + sampled_pelvis[index]
            )
            world_before = wrapper_values[index] + rotation @ sampled_pelvis[index]
            world_after = rotation @ absorbed[index]
            maximum_pelvis_translation_world_error = max(
                maximum_pelvis_translation_world_error,
                float(np.max(np.abs(world_before - world_after))),
            )
        if maximum_pelvis_translation_world_error > 1.0e-6:
            raise RocketboxGlbNormalizationError(
                "animated wrapper translation absorption changed pelvis motion"
            )
        _write_float_accessor(
            document, binary, wrapper_sampler["output"], absorbed
        )
        wrapper_channel["target"] = {
            "node": pelvis_index,
            "path": "translation",
        }
        channels.remove(pelvis_channel)
        absorbed_animation_count += 1
        removed_pelvis_channels += 1

    parents = _build_parent_map(nodes)
    inverse_binds = np.empty((len(joints), 16), dtype="<f4")
    for offset, joint_index in enumerate(joints):
        inverse_binds[offset] = np.linalg.inv(
            _global_node_matrix(nodes, parents, joint_index)
        ).reshape(16, order="F")
    _write_float_accessor(document, binary, inverse_bind_accessor, inverse_binds)
    for index in metric_evidence["mesh_position_accessors_unchanged"]:
        if _accessor_storage_bytes(document, binary, index) != _accessor_storage_bytes(
            *read_glb_bytes(metric_payload), index
        ):
            raise RocketboxGlbNormalizationError(
                "grounding changed a mesh POSITION accessor"
            )
    encoded = json.dumps(
        document, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    output = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded) + 8 + len(binary))
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + bytes(binary)
    )
    return output, {
        **metric_evidence,
        "schema": "rocketbox_ue_grounded_metric_skeleton_normalization_v1",
        "pelvis_node_index": pelvis_index,
        "pelvis_node_name": pelvis.get("name"),
        "absorbed_static_wrapper_translation": wrapper_translation.tolist(),
        "static_wrapper_translation_zeroed": True,
        "absorbed_wrapper_translation_animation_count": absorbed_animation_count,
        "removed_duplicate_pelvis_translation_channel_count": removed_pelvis_channels,
        "maximum_pelvis_translation_world_error_m": (
            maximum_pelvis_translation_world_error
        ),
    }


def normalize_in_place_grounded_metric_glb_bytes(
    payload: bytes,
    *,
    expected_unit_scale: float = 0.01,
    in_place_actions: tuple[str, ...] = ("Walking",),
) -> tuple[bytes, dict[str, object]]:
    """Normalize units/ground origin and remove duplicate horizontal root motion.

    Apartment trajectories move the Unreal actor in world space.  Native
    Rocketbox Walking also advances its pelvis by roughly one stride cycle, so
    retaining both motions makes the skinned body jump backwards every time
    the animation loops.  This transform keeps the pelvis world-space vertical
    curve and every joint rotation, while pinning only the requested actions'
    pelvis world X/Z coordinates to their first sample.
    """
    grounded_payload, grounded_evidence = normalize_grounded_metric_glb_bytes(
        payload, expected_unit_scale=expected_unit_scale
    )
    document, grounded_binary = read_glb_bytes(grounded_payload)
    binary = bytearray(grounded_binary)
    nodes = document.get("nodes")
    animations = document.get("animations")
    skins = document.get("skins")
    if (
        not isinstance(nodes, list)
        or not isinstance(animations, list)
        or not isinstance(skins, list)
        or len(skins) != 1
        or not isinstance(skins[0], dict)
    ):
        raise RocketboxGlbNormalizationError(
            "in-place normalization requires one grounded metric skin"
        )
    requested = tuple(in_place_actions)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(not isinstance(name, str) or not name for name in requested)
    ):
        raise RocketboxGlbNormalizationError("in-place action names are invalid")

    armature_index = grounded_evidence["armature_node_index"]
    pelvis_index = grounded_evidence["pelvis_node_index"]
    armature = nodes[armature_index]
    if not np.allclose(
        np.asarray(armature.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
        np.zeros(3),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RocketboxGlbNormalizationError(
            "in-place normalization requires a zeroed armature translation"
        )
    static_rotation = _quaternion_matrix(
        armature.get("rotation", [0.0, 0.0, 0.0, 1.0])
    )
    found = set()
    root_motion = {}
    changed_accessors = []
    for animation in animations:
        if not isinstance(animation, dict):
            raise RocketboxGlbNormalizationError("animation is not an object")
        action_name = animation.get("name")
        if action_name not in requested:
            continue
        if action_name in found:
            raise RocketboxGlbNormalizationError(
                f"duplicate in-place action: {action_name}"
            )
        found.add(action_name)
        channels = animation.get("channels")
        samplers = animation.get("samplers")
        if not isinstance(channels, list) or not isinstance(samplers, list):
            raise RocketboxGlbNormalizationError(
                f"{action_name} animation channels are invalid"
            )
        pelvis_channels = [
            channel
            for channel in channels
            if isinstance(channel, dict)
            and channel.get("target")
            == {"node": pelvis_index, "path": "translation"}
        ]
        rotation_channels = [
            channel
            for channel in channels
            if isinstance(channel, dict)
            and channel.get("target")
            == {"node": armature_index, "path": "rotation"}
        ]
        if len(pelvis_channels) != 1 or len(rotation_channels) > 1:
            raise RocketboxGlbNormalizationError(
                f"{action_name} must have one pelvis translation and at most "
                "one armature rotation"
            )
        pelvis_sampler = samplers[pelvis_channels[0]["sampler"]]
        if pelvis_sampler.get("interpolation", "LINEAR") != "LINEAR":
            raise RocketboxGlbNormalizationError(
                f"{action_name} pelvis translation must use LINEAR interpolation"
            )
        times = read_float_accessor(
            document, binary, pelvis_sampler["input"]
        ).reshape(-1)
        pelvis_values = read_float_accessor(
            document, binary, pelvis_sampler["output"]
        )
        if (
            pelvis_values.shape != (len(times), 3)
            or len(times) < 2
            or np.any(np.diff(times) <= 0.0)
        ):
            raise RocketboxGlbNormalizationError(
                f"{action_name} pelvis samples are invalid"
            )
        if rotation_channels:
            rotation_sampler = samplers[rotation_channels[0]["sampler"]]
            if rotation_sampler.get("interpolation", "LINEAR") != "LINEAR":
                raise RocketboxGlbNormalizationError(
                    f"{action_name} armature rotation must use LINEAR interpolation"
                )
            rotation_times = read_float_accessor(
                document, binary, rotation_sampler["input"]
            ).reshape(-1)
            rotations = read_float_accessor(
                document, binary, rotation_sampler["output"]
            )
            if not np.array_equal(rotation_times, times) or rotations.shape != (
                len(times),
                4,
            ):
                raise RocketboxGlbNormalizationError(
                    f"{action_name} rotation and pelvis samples must share exact times"
                )
            rotation_matrices = [
                _quaternion_matrix(row.tolist()) for row in rotations
            ]
        else:
            rotation_matrices = [static_rotation] * len(times)

        world_before = np.asarray(
            [rotation @ value for rotation, value in zip(rotation_matrices, pelvis_values)],
            dtype=np.float64,
        )
        world_after = world_before.copy()
        world_after[:, 0] = world_before[0, 0]
        world_after[:, 2] = world_before[0, 2]
        local_after = np.asarray(
            [rotation.T @ value for rotation, value in zip(rotation_matrices, world_after)],
            dtype=np.float64,
        )
        output_accessor = pelvis_sampler["output"]
        _write_float_accessor(document, binary, output_accessor, local_after)
        changed_accessors.append(output_accessor)

        stored_local = read_float_accessor(document, binary, output_accessor)
        stored_world = np.asarray(
            [rotation @ value for rotation, value in zip(rotation_matrices, stored_local)],
            dtype=np.float64,
        )
        vertical_error = float(
            np.max(np.abs(stored_world[:, 1] - world_before[:, 1]))
        )
        horizontal_deviation = float(
            np.max(
                np.linalg.norm(
                    stored_world[:, (0, 2)] - stored_world[0, (0, 2)], axis=1
                )
            )
        )
        if vertical_error > 1.0e-6 or horizontal_deviation > 1.0e-6:
            raise RocketboxGlbNormalizationError(
                f"{action_name} in-place root-motion verification failed"
            )
        root_motion[action_name] = {
            "sample_count": len(times),
            "time_range_seconds": [float(times[0]), float(times[-1])],
            "horizontal_displacement_before_m": [
                float(world_before[-1, 0] - world_before[0, 0]),
                float(world_before[-1, 2] - world_before[0, 2]),
            ],
            "horizontal_path_span_before_m": [
                float(np.ptp(world_before[:, 0])),
                float(np.ptp(world_before[:, 2])),
            ],
            "maximum_horizontal_deviation_after_m": horizontal_deviation,
            "maximum_vertical_world_error_m": vertical_error,
            "pelvis_translation_accessor": output_accessor,
        }

    missing = sorted(set(requested) - found)
    if missing:
        raise RocketboxGlbNormalizationError(
            f"requested in-place actions are missing: {missing}"
        )
    for index in grounded_evidence["mesh_position_accessors_unchanged"]:
        if _accessor_storage_bytes(document, binary, index) != _accessor_storage_bytes(
            *read_glb_bytes(grounded_payload), index
        ):
            raise RocketboxGlbNormalizationError(
                "in-place normalization changed a mesh POSITION accessor"
            )
    if len(binary) != len(grounded_binary):
        raise RocketboxGlbNormalizationError(
            "in-place normalization resized the BIN chunk"
        )
    encoded = json.dumps(
        document, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    output = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded) + 8 + len(binary))
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + bytes(binary)
    )
    return output, {
        **grounded_evidence,
        "schema": (
            "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
        ),
        "in_place_actions": list(requested),
        "root_motion": root_motion,
        "in_place_pelvis_translation_accessors": sorted(changed_accessors),
    }


def normalize_glb_bytes(payload: bytes) -> tuple[bytes, dict[str, object]]:
    document, binary = read_glb_bytes(payload)
    normalized, evidence = promote_skinned_mesh_to_scene_root(document)
    encoded = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    total_length = 12 + 8 + len(encoded) + 8 + len(binary)
    output = (
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )
    return output, evidence


def _publish_transformed_glb(
    source: Path,
    output: Path,
    transform,
    *,
    require_binary_unchanged: bool,
) -> dict[str, object]:
    source = Path(source).absolute()
    output = Path(output).absolute()
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise RocketboxGlbNormalizationError(
            f"source GLB is not a direct regular file: {source}"
        )
    if output.exists() or output.is_symlink():
        raise RocketboxGlbNormalizationError(
            f"refusing to replace normalized GLB: {output}"
        )
    source_payload = source.read_bytes()
    output_payload, evidence = transform(source_payload)
    _document, source_binary = read_glb_bytes(source_payload)
    _normalized_document, output_binary = read_glb_bytes(output_payload)
    if require_binary_unchanged and output_binary != source_binary:
        raise RocketboxGlbNormalizationError("normalization changed the BIN chunk")
    if len(output_binary) != len(source_binary):
        raise RocketboxGlbNormalizationError("normalization resized the BIN chunk")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(output_payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise RocketboxGlbNormalizationError(
                f"refusing to replace normalized GLB: {output}"
            ) from error
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    if output.read_bytes() != output_payload:
        output.unlink(missing_ok=True)
        raise RocketboxGlbNormalizationError("published GLB readback changed")
    return {
        **evidence,
        "source_glb": str(source),
        "source_glb_sha256": hashlib.sha256(source_payload).hexdigest(),
        "source_size_bytes": len(source_payload),
        "output_glb": str(output),
        "output_glb_sha256": hashlib.sha256(output_payload).hexdigest(),
        "output_size_bytes": len(output_payload),
        "binary_chunk_sha256": hashlib.sha256(source_binary).hexdigest(),
        "source_binary_chunk_sha256": hashlib.sha256(source_binary).hexdigest(),
        "output_binary_chunk_sha256": hashlib.sha256(output_binary).hexdigest(),
    }


def publish_normalized_glb(source: Path, output: Path) -> dict[str, object]:
    return _publish_transformed_glb(
        source,
        output,
        normalize_glb_bytes,
        require_binary_unchanged=True,
    )


def publish_metric_normalized_glb(
    source: Path, output: Path
) -> dict[str, object]:
    return _publish_transformed_glb(
        source,
        output,
        normalize_metric_glb_bytes,
        require_binary_unchanged=False,
    )


def publish_grounded_metric_normalized_glb(
    source: Path, output: Path
) -> dict[str, object]:
    return _publish_transformed_glb(
        source,
        output,
        normalize_grounded_metric_glb_bytes,
        require_binary_unchanged=False,
    )


def publish_in_place_grounded_metric_normalized_glb(
    source: Path, output: Path
) -> dict[str, object]:
    return _publish_transformed_glb(
        source,
        output,
        normalize_in_place_grounded_metric_glb_bytes,
        require_binary_unchanged=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metric", action="store_true")
    mode.add_argument("--grounded-metric", action="store_true")
    mode.add_argument("--in-place-grounded-metric", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            (
                publish_in_place_grounded_metric_normalized_glb(
                    arguments.input, arguments.output
                )
                if arguments.in_place_grounded_metric
                else (
                    publish_grounded_metric_normalized_glb(
                        arguments.input, arguments.output
                    )
                    if arguments.grounded_metric
                    else (
                        publish_metric_normalized_glb(arguments.input, arguments.output)
                        if arguments.metric
                        else publish_normalized_glb(arguments.input, arguments.output)
                    )
                )
            ),
            sort_keys=True,
        )
    )
