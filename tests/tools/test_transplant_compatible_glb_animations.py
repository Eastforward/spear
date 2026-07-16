from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import struct

import pytest

from tools import transplant_compatible_glb_animations as transplant


def document_and_binary(*, parent_name="Armature", bone_translation=1.0):
    binary = bytes(range(64))
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": parent_name, "children": [1]},
            {
                "name": "bone_0",
                "translation": [bone_translation, 0, 0],
                "rotation": [0, 0, 0, 1],
                "children": [2],
            },
            {
                "name": "bone_1",
                "translation": [0, 1, 0],
                "rotation": [0, 0, 0, 1],
            },
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "skins": [{"joints": [1, 2]}],
        "materials": [{"name": "PBR"}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 16},
            {"buffer": 0, "byteOffset": 16, "byteLength": 16},
            {"buffer": 0, "byteOffset": 32, "byteLength": 16},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC4"},
        ],
        "buffers": [{"byteLength": len(binary)}],
        "animations": [
            {
                "name": "Walking",
                "samplers": [{"input": 1, "output": 2, "interpolation": "LINEAR"}],
                "channels": [
                    {"sampler": 0, "target": {"node": 2, "path": "rotation"}}
                ],
            },
            {
                "name": "Idle",
                "samplers": [{"input": 1, "output": 2}],
                "channels": [
                    {"sampler": 0, "target": {"node": 2, "path": "rotation"}}
                ],
            },
        ],
    }
    return document, binary


def write_glb(path: Path, document, binary):
    path.write_bytes(transplant.encode_glb(document, binary))


def test_transplant_preserves_target_prefix_and_non_animation_fields():
    target, target_binary = document_and_binary()
    source, source_binary = document_and_binary(bone_translation=1.0 + 1.0e-7)
    target_before = deepcopy(target)
    output, output_binary, record = transplant.transplant_animations(
        target,
        target_binary,
        source,
        source_binary,
        ["Walking", "Idle"],
    )
    assert output_binary[: len(target_binary)] == target_binary
    assert output["meshes"] == target_before["meshes"]
    assert output["skins"] == target_before["skins"]
    assert output["materials"] == target_before["materials"]
    assert output["nodes"] == target_before["nodes"]
    assert [item["name"] for item in output["animations"]] == ["Walking", "Idle"]
    assert output["animations"][0]["channels"][0]["target"]["node"] == 2
    assert len(output["accessors"]) == len(target["accessors"]) + 2
    assert len(output["bufferViews"]) == len(target["bufferViews"]) + 2
    assert record["preservation"]["target_binary_prefix_unchanged"] is True
    assert record["compatibility"]["animated_node_count"] == 1


def test_transplant_remaps_animation_node_by_unique_name():
    target, target_binary = document_and_binary()
    source, source_binary = document_and_binary()
    source["nodes"] = [source["nodes"][0], source["nodes"][2], source["nodes"][1]]
    source["nodes"][0]["children"] = [2]
    source["nodes"][2]["children"] = [1]
    for animation in source["animations"]:
        animation["channels"][0]["target"]["node"] = 1
    output, _binary, _record = transplant.transplant_animations(
        target, target_binary, source, source_binary, ["Walking"]
    )
    assert output["animations"][0]["channels"][0]["target"]["node"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda source: source["nodes"][2].update(translation=[0.1, 1, 0]), "rest transform mismatch"),
        (lambda source: source["nodes"][1].update(name="DifferentParent"), "parent mismatch"),
    ],
)
def test_transplant_rejects_incompatible_skeleton(mutation, message):
    target, target_binary = document_and_binary()
    source, source_binary = document_and_binary()
    mutation(source)
    with pytest.raises(transplant.AnimationTransplantError, match=message):
        transplant.transplant_animations(
            target, target_binary, source, source_binary, ["Walking"]
        )


def test_main_writes_authenticated_glb_and_refuses_overwrite(tmp_path):
    target, target_binary = document_and_binary()
    source, source_binary = document_and_binary()
    target_path = tmp_path / "target.glb"
    source_path = tmp_path / "source.glb"
    output_path = tmp_path / "output.glb"
    manifest_path = tmp_path / "manifest.json"
    write_glb(target_path, target, target_binary)
    write_glb(source_path, source, source_binary)
    arguments = [
        "--target-glb",
        str(target_path),
        "--source-glb",
        str(source_path),
        "--output-glb",
        str(output_path),
        "--manifest",
        str(manifest_path),
    ]
    assert transplant.main(arguments) == 0
    output, output_binary = transplant.read_glb(output_path)
    manifest = json.loads(manifest_path.read_text())
    assert output_binary[: len(target_binary)] == target_binary
    assert [item["name"] for item in output["animations"]] == ["Walking", "Idle"]
    assert manifest["output"]["readback_passed"] is True
    assert transplant.main(arguments) == 2
