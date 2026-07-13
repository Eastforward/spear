from __future__ import annotations

import copy
import hashlib
import importlib
import json
import struct

import numpy as np
import pytest


def _module():
    return importlib.import_module("tools.normalize_rocketbox_glb_for_ue")


def _nested_document():
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [3]}],
        "nodes": [
            {"name": "Pelvis", "children": [1]},
            {"name": "Spine"},
            {"name": "Body", "mesh": 0, "skin": 0},
            {
                "name": "Bip01",
                "children": [2, 0],
                "translation": [0.0, 0.895, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [0.01, 0.01, 0.01],
            },
        ],
        "meshes": [{"name": "Mesh", "primitives": []}],
        "skins": [{"name": "Bip01", "joints": [0, 1]}],
        "animations": [{"name": "Walking", "channels": [], "samplers": []}],
        "materials": [{"name": "m002_body"}],
        "images": [{"name": "m002_body_color", "bufferView": 0}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 4}],
        "buffers": [{"byteLength": 4}],
    }


def _glb(document, binary=b"DATA"):
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    binary += b"\0" * ((-len(binary)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + binary
    )


def _metric_fixture_glb(*, pelvis_translation_values=None):
    chunks = []

    def add(values):
        array = np.asarray(values, dtype="<f4")
        offset = sum(len(chunk) for chunk in chunks)
        chunks.append(array.tobytes())
        return offset, array.nbytes

    identity = np.eye(4, dtype="<f4").reshape(1, 16, order="F")
    old_ibm = np.repeat(identity, 2, axis=0)
    ibm_offset, ibm_size = add(old_ibm)
    position_offset, position_size = add([[0.0, 0.0, 0.0], [0.0, 1.8, 0.0]])
    time_offset, time_size = add([[0.0], [1.0]])
    child_translation_offset, child_translation_size = add(
        [[100.0, 0.0, 0.0], [110.0, 0.0, 0.0]]
    )
    root_translation_offset, root_translation_size = add(
        [[0.0, 0.9, 0.0], [0.0, 1.1, 0.0]]
    )
    if pelvis_translation_values is None:
        pelvis_translation_values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    pelvis_translation_offset, pelvis_translation_size = add(
        pelvis_translation_values
    )
    binary = b"".join(chunks)
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [3]}],
        "nodes": [
            {"name": "Pelvis", "children": [1], "translation": [0.0, 0.0, 0.0]},
            {"name": "Spine", "translation": [100.0, 0.0, 0.0]},
            {"name": "Body", "mesh": 0, "skin": 0},
            {
                "name": "Bip01",
                "children": [2, 0],
                "translation": [0.0, 0.9, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [0.01, 0.01, 0.01],
            },
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 1}}]}
        ],
        "skins": [{"joints": [0, 1], "inverseBindMatrices": 0}],
        "animations": [
            {
                "name": "Walking",
                "samplers": [
                    {"input": 2, "output": 3, "interpolation": "LINEAR"},
                    {"input": 2, "output": 4, "interpolation": "LINEAR"},
                    {"input": 2, "output": 5, "interpolation": "STEP"},
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": 1, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 3, "path": "translation"}},
                    {"sampler": 2, "target": {"node": 0, "path": "translation"}},
                ],
            }
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 2, "type": "MAT4"},
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 2,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [0.0, 1.8, 0.0],
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 2,
                "type": "SCALAR",
            },
            {
                "bufferView": 3,
                "componentType": 5126,
                "count": 2,
                "type": "VEC3",
                "min": [100.0, 0.0, 0.0],
                "max": [110.0, 0.0, 0.0],
            },
            {
                "bufferView": 4,
                "componentType": 5126,
                "count": 2,
                "type": "VEC3",
                "min": [0.0, 0.9, 0.0],
                "max": [0.0, 1.1, 0.0],
            },
            {
                "bufferView": 5,
                "componentType": 5126,
                "count": 2,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [0.0, 0.0, 0.0],
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": ibm_offset, "byteLength": ibm_size},
            {"buffer": 0, "byteOffset": position_offset, "byteLength": position_size},
            {"buffer": 0, "byteOffset": time_offset, "byteLength": time_size},
            {
                "buffer": 0,
                "byteOffset": child_translation_offset,
                "byteLength": child_translation_size,
            },
            {
                "buffer": 0,
                "byteOffset": root_translation_offset,
                "byteLength": root_translation_size,
            },
            {
                "buffer": 0,
                "byteOffset": pelvis_translation_offset,
                "byteLength": pelvis_translation_size,
            },
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    return _glb(document, binary), position_offset, position_size


def test_promotes_only_skinned_mesh_to_scene_root_without_touching_armature_trs():
    module = _module()
    original = _nested_document()
    normalized, evidence = module.promote_skinned_mesh_to_scene_root(
        copy.deepcopy(original)
    )

    assert normalized["scenes"][0]["nodes"] == [3, 2]
    assert normalized["nodes"][3]["children"] == [0]
    assert normalized["nodes"][3]["translation"] == [0.0, 0.895, 0.0]
    assert normalized["nodes"][3]["scale"] == [0.01, 0.01, 0.01]
    assert normalized["nodes"][2] == original["nodes"][2]
    assert normalized["meshes"] == original["meshes"]
    assert normalized["skins"] == original["skins"]
    assert normalized["animations"] == original["animations"]
    assert evidence == {
        "schema": "rocketbox_ue_skinned_mesh_root_normalization_v1",
        "scene_index": 0,
        "armature_node_index": 3,
        "armature_node_name": "Bip01",
        "mesh_node_index": 2,
        "mesh_node_name": "Body",
        "old_scene_roots": [3],
        "new_scene_roots": [3, 2],
        "old_armature_children": [2, 0],
        "new_armature_children": [0],
        "mesh_binary_payload_unchanged": True,
    }


def test_glb_normalization_preserves_bin_and_every_non_node_json_section():
    module = _module()
    original_document = _nested_document()
    original_glb = _glb(original_document, b"ABCD")

    normalized_glb, evidence = module.normalize_glb_bytes(original_glb)
    before_document, before_bin = module.read_glb_bytes(original_glb)
    after_document, after_bin = module.read_glb_bytes(normalized_glb)

    assert before_bin == after_bin == b"ABCD"
    assert evidence["mesh_binary_payload_unchanged"] is True
    for key in before_document:
        if key not in {"nodes", "scenes"}:
            assert after_document[key] == before_document[key], key
    assert after_document["scenes"][0]["nodes"] == [3, 2]
    assert after_document["nodes"][3]["children"] == [0]


def test_rejects_mesh_with_transform_or_multiple_parentage():
    module = _module()
    transformed = _nested_document()
    transformed["nodes"][2]["scale"] = [2.0, 2.0, 2.0]
    with pytest.raises(module.RocketboxGlbNormalizationError, match="identity"):
        module.promote_skinned_mesh_to_scene_root(transformed)

    multiply_parented = _nested_document()
    multiply_parented["nodes"][0].setdefault("children", []).append(2)
    with pytest.raises(module.RocketboxGlbNormalizationError, match="exactly one"):
        module.promote_skinned_mesh_to_scene_root(multiply_parented)


def test_atomic_publisher_is_no_replace_and_hash_locks_both_containers(tmp_path):
    module = _module()
    source = tmp_path / "nested.glb"
    output = tmp_path / "normalized.glb"
    source.write_bytes(_glb(_nested_document(), b"ABCD"))

    evidence = module.publish_normalized_glb(source, output)

    assert output.is_file()
    assert evidence["source_glb_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert evidence["output_glb_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert evidence["source_glb_sha256"] != evidence["output_glb_sha256"]
    assert evidence["source_size_bytes"] == source.stat().st_size
    assert evidence["output_size_bytes"] == output.stat().st_size
    assert evidence["binary_chunk_sha256"] == hashlib.sha256(b"ABCD").hexdigest()
    with pytest.raises(module.RocketboxGlbNormalizationError, match="replace"):
        module.publish_normalized_glb(source, output)


def test_metric_normalization_bakes_centimeter_bones_and_keeps_meter_mesh():
    module = _module()
    source_glb, position_offset, position_size = _metric_fixture_glb()
    source_document, source_binary = module.read_glb_bytes(source_glb)

    normalized_glb, evidence = module.normalize_metric_glb_bytes(source_glb)
    document, binary = module.read_glb_bytes(normalized_glb)

    assert document["scenes"][0]["nodes"] == [3, 2]
    assert document["nodes"][3]["children"] == [0]
    assert document["nodes"][3]["scale"] == [1.0, 1.0, 1.0]
    assert document["nodes"][3]["translation"] == [0.0, 0.9, 0.0]
    assert document["nodes"][0]["translation"] == [0.0, 0.0, 0.0]
    assert document["nodes"][1]["translation"] == [1.0, 0.0, 0.0]
    assert binary[position_offset : position_offset + position_size] == (
        source_binary[position_offset : position_offset + position_size]
    )
    assert np.allclose(
        module.read_float_accessor(document, binary, 3),
        [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
        atol=1e-6,
    )
    assert np.array_equal(
        module.read_float_accessor(document, binary, 4),
        module.read_float_accessor(source_document, source_binary, 4),
    )
    inverse_binds = module.read_float_accessor(document, binary, 0).reshape(
        2, 4, 4, order="F"
    )
    expected_first = np.eye(4)
    expected_first[1, 3] = -0.9
    expected_second = np.eye(4)
    expected_second[0, 3] = -1.0
    expected_second[1, 3] = -0.9
    assert np.allclose(inverse_binds[0], expected_first, atol=1e-6)
    assert np.allclose(inverse_binds[1], expected_second, atol=1e-6)
    assert evidence["source_unit_scale"] == 0.01
    assert evidence["normalized_joint_count"] == 2
    assert evidence["normalized_joint_translation_accessor_count"] == 2
    assert evidence["recomputed_inverse_bind_matrix_count"] == 2
    assert evidence["mesh_position_accessors_unchanged"] == [1]


def test_metric_publisher_uses_metric_transform_and_is_no_replace(tmp_path):
    module = _module()
    source = tmp_path / "nested_metric.glb"
    output = tmp_path / "normalized_metric.glb"
    source.write_bytes(_metric_fixture_glb()[0])

    evidence = module.publish_metric_normalized_glb(source, output)

    document, _binary = module.read_glb_bytes(output.read_bytes())
    assert evidence["schema"] == "rocketbox_ue_metric_skeleton_normalization_v1"
    assert document["nodes"][3]["scale"] == [1.0, 1.0, 1.0]
    assert document["nodes"][1]["translation"] == [1.0, 0.0, 0.0]
    with pytest.raises(module.RocketboxGlbNormalizationError, match="replace"):
        module.publish_metric_normalized_glb(source, output)


def test_grounded_metric_normalization_absorbs_wrapper_translation_into_pelvis():
    module = _module()
    source_glb, position_offset, position_size = _metric_fixture_glb()
    _source_document, source_binary = module.read_glb_bytes(source_glb)

    normalized_glb, evidence = module.normalize_grounded_metric_glb_bytes(source_glb)
    document, binary = module.read_glb_bytes(normalized_glb)

    assert document["nodes"][3]["translation"] == [0.0, 0.0, 0.0]
    assert np.allclose(document["nodes"][0]["translation"], [0.0, 0.9, 0.0])
    channels = document["animations"][0]["channels"]
    wrapper_translation = [
        channel
        for channel in channels
        if channel["target"] == {"node": 3, "path": "translation"}
    ]
    pelvis_translation = [
        channel
        for channel in channels
        if channel["target"] == {"node": 0, "path": "translation"}
    ]
    assert wrapper_translation == []
    assert len(pelvis_translation) == 1
    sampler = document["animations"][0]["samplers"][
        pelvis_translation[0]["sampler"]
    ]
    assert np.allclose(
        module.read_float_accessor(document, binary, sampler["output"]),
        [[0.0, 0.9, 0.0], [0.0, 1.1, 0.0]],
        atol=1e-6,
    )
    assert binary[position_offset : position_offset + position_size] == (
        source_binary[position_offset : position_offset + position_size]
    )
    assert evidence["absorbed_wrapper_translation_animation_count"] == 1
    assert evidence["removed_duplicate_pelvis_translation_channel_count"] == 1
    assert evidence["static_wrapper_translation_zeroed"] is True
    assert evidence["mesh_position_accessors_unchanged"] == [1]


def test_grounded_metric_publisher_is_no_replace(tmp_path):
    module = _module()
    source = tmp_path / "nested_ground.glb"
    output = tmp_path / "normalized_ground.glb"
    source.write_bytes(_metric_fixture_glb()[0])

    evidence = module.publish_grounded_metric_normalized_glb(source, output)

    document, _binary = module.read_glb_bytes(output.read_bytes())
    assert evidence["schema"] == (
        "rocketbox_ue_grounded_metric_skeleton_normalization_v1"
    )
    assert document["nodes"][3]["translation"] == [0.0, 0.0, 0.0]
    with pytest.raises(module.RocketboxGlbNormalizationError, match="replace"):
        module.publish_grounded_metric_normalized_glb(source, output)


def test_in_place_grounded_metric_strips_only_walking_horizontal_root_motion():
    module = _module()
    source_glb, position_offset, position_size = _metric_fixture_glb(
        pelvis_translation_values=[
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 50.0],
        ]
    )
    _source_document, source_binary = module.read_glb_bytes(source_glb)

    normalized_glb, evidence = module.normalize_in_place_grounded_metric_glb_bytes(
        source_glb
    )
    document, binary = module.read_glb_bytes(normalized_glb)

    animation = document["animations"][0]
    pelvis_channel = next(
        channel
        for channel in animation["channels"]
        if channel["target"] == {"node": 0, "path": "translation"}
    )
    sampler = animation["samplers"][pelvis_channel["sampler"]]
    translations = module.read_float_accessor(document, binary, sampler["output"])
    assert np.allclose(
        translations,
        [[0.0, 0.9, 0.0], [0.0, 1.1, 0.0]],
        atol=1e-6,
    )
    assert binary[position_offset : position_offset + position_size] == (
        source_binary[position_offset : position_offset + position_size]
    )
    assert evidence["schema"] == (
        "rocketbox_ue_in_place_grounded_metric_skeleton_normalization_v1"
    )
    assert evidence["in_place_actions"] == ["Walking"]
    walking = evidence["root_motion"]["Walking"]
    assert walking["horizontal_displacement_before_m"] == pytest.approx(
        [1.0, 0.5]
    )
    assert walking["maximum_horizontal_deviation_after_m"] < 1e-6
    assert walking["maximum_vertical_world_error_m"] < 1e-6
    assert evidence["mesh_position_accessors_unchanged"] == [1]
