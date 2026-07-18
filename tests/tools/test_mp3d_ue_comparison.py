"""Focused contracts for the fair Habitat/UE MP3D comparison."""

from __future__ import annotations

import ast
import json
import struct
from pathlib import Path

import pytest

from tools import prepare_mp3d_ue_scene as prepare
from tools.spike_rlr import run_mp3d_ue_comparison as comparison


REPO = Path(__file__).resolve().parents[2]
IMPORTER = REPO / "tools" / "import_mp3d_scene_editor.py"
RUNNER = REPO / "tools" / "spike_rlr" / "run_mp3d_ue_comparison.py"


def _fixture_glb() -> tuple[dict, bytes, dict[str, tuple[int, int]]]:
    binary = bytearray()
    ranges: dict[str, tuple[int, int]] = {}

    def append(name: str, payload: bytes) -> tuple[int, int]:
        while len(binary) % 4:
            binary.append(0)
        start = len(binary)
        binary.extend(payload)
        ranges[name] = (start, len(payload))
        return start, len(payload)

    # Every geometric view is interleaved with a sentinel float that must not
    # be touched.  TANGENT keeps its handedness W and has a fifth sentinel.
    position_offset, position_length = append(
        "position",
        struct.pack("<4f4f", 1.0, 2.0, 3.0, 101.0, -4.0, 5.0, -6.0, 102.0),
    )
    normal_offset, normal_length = append(
        "normal",
        struct.pack("<4f4f", 0.0, 1.0, 0.0, 201.0, 0.0, 0.0, 1.0, 202.0),
    )
    tangent_offset, tangent_length = append(
        "tangent",
        struct.pack(
            "<5f5f",
            1.0,
            0.0,
            0.0,
            -1.0,
            301.0,
            0.0,
            1.0,
            0.0,
            1.0,
            302.0,
        ),
    )
    uv_offset, uv_length = append("uv", struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))
    index_offset, index_length = append("indices", struct.pack("<2H", 0, 1))
    document = {
        "asset": {"version": "2.0", "generator": "unit-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "mesh0", "mesh": 0}],
        "meshes": [
            {
                "name": "mesh0",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TANGENT": 2,
                            "TEXCOORD_0": 3,
                        },
                        "indices": 4,
                        "material": 0,
                    }
                ],
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 2,
                "type": "VEC3",
                "min": [-4.0, 2.0, -6.0],
                "max": [1.0, 5.0, 3.0],
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 2,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [0.0, 1.0, 1.0],
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 2,
                "type": "VEC4",
                "min": [0.0, 0.0, 0.0, -1.0],
                "max": [1.0, 1.0, 0.0, 1.0],
            },
            {"bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC2"},
            {"bufferView": 4, "componentType": 5123, "count": 2, "type": "SCALAR"},
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": position_offset,
                "byteLength": position_length,
                "byteStride": 16,
            },
            {
                "buffer": 0,
                "byteOffset": normal_offset,
                "byteLength": normal_length,
                "byteStride": 16,
            },
            {
                "buffer": 0,
                "byteOffset": tangent_offset,
                "byteLength": tangent_length,
                "byteStride": 20,
            },
            {"buffer": 0, "byteOffset": uv_offset, "byteLength": uv_length},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": index_length},
        ],
        "buffers": [{"byteLength": len(binary)}],
        "materials": [{"name": "material0"}],
        "textures": [{"source": 0}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
    }
    return document, bytes(binary), ranges


def _vectors(document: dict, binary: bytes, accessor_index: int) -> list[tuple]:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    component_count = {"VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    element_size = component_count * 4
    stride = view.get("byteStride", element_size)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    return [
        struct.unpack_from(f"<{component_count}f", binary, start + index * stride)
        for index in range(accessor["count"])
    ]


def test_prepare_rotates_every_interleaved_geometry_accessor_and_preserves_payloads(
    tmp_path: Path,
):
    document, binary, ranges = _fixture_glb()
    source = tmp_path / "source.glb"
    source.write_bytes(prepare.build_glb(document, binary))
    output = tmp_path / "canonical.glb"
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare.prepare_mp3d_scene(
        input_glb=source,
        output_glb=output,
        manifest_path=manifest_path,
        scene_id="fixture",
        expected_root_mesh_count=1,
        enforce_reference_bounds=False,
    )
    readback = prepare.load_glb(output)

    assert _vectors(readback.document, readback.binary, 0) == [
        pytest.approx((1.0, 3.0, -2.0)),
        pytest.approx((-4.0, -6.0, -5.0)),
    ]
    assert _vectors(readback.document, readback.binary, 1) == [
        pytest.approx((0.0, 0.0, -1.0)),
        pytest.approx((0.0, 1.0, -0.0)),
    ]
    assert _vectors(readback.document, readback.binary, 2) == [
        pytest.approx((1.0, 0.0, -0.0, -1.0)),
        pytest.approx((0.0, 0.0, -1.0, 1.0)),
    ]
    assert readback.document["accessors"][0]["min"] == pytest.approx(
        [-4.0, -6.0, -5.0]
    )
    assert readback.document["accessors"][0]["max"] == pytest.approx(
        [1.0, 3.0, -2.0]
    )
    assert readback.document["accessors"][1]["min"] == pytest.approx(
        [0.0, 0.0, -1.0]
    )
    assert readback.document["accessors"][1]["max"] == pytest.approx(
        [0.0, 1.0, 0.0]
    )
    assert readback.document["accessors"][2]["min"] == pytest.approx(
        [0.0, 0.0, -1.0, -1.0]
    )
    assert readback.document["accessors"][2]["max"] == pytest.approx(
        [1.0, 0.0, 0.0, 1.0]
    )
    # UV and index payloads stay byte-identical.
    for name in ("uv", "indices"):
        offset, length = ranges[name]
        assert readback.binary[offset : offset + length] == binary[offset : offset + length]
    # Interleaved sentinels stay byte-identical too.
    assert struct.unpack_from("<f", readback.binary, ranges["position"][0] + 12)[0] == 101.0
    assert struct.unpack_from("<f", readback.binary, ranges["normal"][0] + 12)[0] == 201.0
    assert struct.unpack_from("<f", readback.binary, ranges["tangent"][0] + 16)[0] == 301.0
    assert manifest["geometry"]["interleaved_accessor_count"] == 3
    assert manifest["geometry"]["recomputed_bounds_accessor_indices"] == {
        "POSITION": [0],
        "NORMAL": [1],
        "TANGENT": [2],
    }
    assert manifest["geometry"]["raw_bounds"] == {
        "minimum": [-4.0, 2.0, -6.0],
        "maximum": [1.0, 5.0, 3.0],
    }
    assert manifest["geometry"]["canonical_bounds"] == {
        "minimum": [-4.0, -6.0, -5.0],
        "maximum": [1.0, 3.0, -2.0],
    }
    assert json.loads(manifest_path.read_text())["prepared"]["sha256"] == (
        prepare.sha256_file(output)
    )


def test_prepare_fails_closed_on_nonidentity_root_or_existing_output(tmp_path: Path):
    document, binary, _ = _fixture_glb()
    document["nodes"][0]["translation"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="not identity-only"):
        prepare.validate_root_mesh_identity(document, expected_count=1)

    document["nodes"][0].pop("translation")
    source = tmp_path / "source.glb"
    source.write_bytes(prepare.build_glb(document, binary))
    output = tmp_path / "output.glb"
    output.write_bytes(b"owned")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        prepare.prepare_mp3d_scene(
            input_glb=source,
            output_glb=output,
            manifest_path=tmp_path / "manifest.json",
            scene_id="fixture",
            expected_root_mesh_count=1,
            enforce_reference_bounds=False,
        )


def test_habitat_to_ue_mapping_camera_routes_and_asset_yaws_are_exact():
    assert comparison.habitat_to_ue_cm(comparison.CAMERA_HABITAT_M) == pytest.approx(
        comparison.CAMERA_UE_CM
    )
    assert comparison.habitat_to_ue_cm(
        comparison.HUMAN_START_HABITAT_M
    ) == pytest.approx(comparison.HUMAN_START_UE_CM)
    assert comparison.habitat_to_ue_cm(
        comparison.DOG_END_HABITAT_M
    ) == pytest.approx(comparison.DOG_END_UE_CM)
    config = comparison.validate_configuration(comparison.default_configuration())
    for actor_id in ("human0", "dog0"):
        assert len(config["actors"][actor_id]["route_ue_cm"]) == 270
        assert comparison.route_yaw_ue_deg(
            config["actors"][actor_id]["route_ue_cm"]
        ) == pytest.approx(-90.0)
    assert config["actors"]["human0"]["walking_local_forward_axis_ue"] == "+Y"
    assert config["actors"]["human0"]["walking_forward_yaw_offset_deg"] == -90.0
    assert config["actors"]["human0"]["actor_yaw_ue_deg"] == -180.0
    assert config["actors"]["dog0"]["anatomical_forward_axis"] == "+X"
    assert config["actors"]["dog0"]["actor_yaw_ue_deg"] == -90.0
    assert config["camera"]["ue_yaw_deg"] == -90.0
    assert comparison.animation_phase_seconds("human0", 0) == pytest.approx(1 / 30)
    assert comparison.animation_phase_seconds("human0", 16) == pytest.approx(1 / 30)
    assert comparison.animation_phase_seconds("dog0", 0) == 0.0
    assert comparison.animation_phase_seconds("dog0", 25) == 0.0


def test_configuration_rejects_mirrored_camera_or_wrong_actor_yaw():
    config = comparison.default_configuration()
    config["camera"]["ue_yaw_deg"] = 90.0
    with pytest.raises(ValueError, match="camera yaw"):
        comparison.validate_configuration(config)
    config = comparison.default_configuration()
    config["actors"]["dog0"]["actor_yaw_ue_deg"] = 90.0
    with pytest.raises(ValueError, match="dog0 actor yaw"):
        comparison.validate_configuration(config)


def _valid_import_manifest() -> dict:
    scene_root = "/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy"
    return {
        "schema": comparison.IMPORT_SCHEMA,
        "status": "passed",
        "scene_id": comparison.SCENE_ID,
        "source": {"sha256": comparison.EXPECTED_RAW_MP3D_SHA256},
        "reload_verification": {"status": "passed"},
        "scene_content": {
            "status": "passed",
            "static_mesh_count": 71,
            "static_meshes": [f"{scene_root}/mesh_{index}.mesh_{index}" for index in range(71)],
            "ue_bounds": {
                "status": "passed",
                "minimum_cm": comparison.EXPECTED_UE_SCENE_BOUNDS_CM["minimum"],
                "maximum_cm": comparison.EXPECTED_UE_SCENE_BOUNDS_CM["maximum"],
                "maximum_absolute_error_cm": 0.01,
                "expected": comparison.EXPECTED_UE_SCENE_BOUNDS_CM,
                "tolerance_cm": comparison.UE_SCENE_BOUNDS_TOLERANCE_CM,
            },
        },
        "m2_beagle": {
            "source": {"sha256": comparison.EXPECTED_M2_BEAGLE_SHA256},
            "exact_habitat_m2_runtime": True,
            "content": {
                "status": "passed",
                "blueprint_package_path": comparison.EXACT_BEAGLE_BP_PACKAGE_PATH,
                "blueprint_class_path": comparison.EXACT_BEAGLE_BP_CLASS_PATH,
                "animations": {
                    "Idle": "/Game/M2/Idle.Idle",
                    "Walking": "/Game/M2/Walking.Walking",
                },
            },
        },
    }


def test_import_manifest_requires_reload_verified_exact_m2_bp_and_hash():
    manifest = _valid_import_manifest()
    assert comparison.validate_ue_import_manifest_payload(manifest) is manifest
    manifest = _valid_import_manifest()
    manifest["m2_beagle"]["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not a reload-verified exact"):
        comparison.validate_ue_import_manifest_payload(manifest)
    manifest = _valid_import_manifest()
    manifest["scene_content"]["ue_bounds"].pop("minimum_cm")
    with pytest.raises(ValueError, match="vector"):
        comparison.validate_ue_import_manifest_payload(manifest)
    manifest = _valid_import_manifest()
    manifest["reload_verification"]["status"] = "pending"
    with pytest.raises(ValueError, match="not a reload-verified exact"):
        comparison.validate_ue_import_manifest_payload(manifest)


def _valid_habitat_gate() -> dict:
    return {
        "schema": comparison.HABITAT_GATE_SCHEMA,
        "status": "pass",
        "route_id": comparison.ROUTE_ID,
        "frame_count": 270,
        "frame_rate_hz": 15,
        "gates": [{"gate_id": "all", "status": "pass"}],
        "pathfinder": {
            "center_navigation_semantics": "actor_root_center_only",
            "declared_navmesh_loaded": True,
            "routes": {
                actor_id: {
                    "all_frames_navigable": True,
                    "frame_count": 270,
                    "navigable_frame_count": 270,
                    "maximum_snap_error_m": 1.0e-7,
                    "required_maximum_snap_error_m": 1.0e-5,
                    "start_m": list(
                        comparison.HUMAN_START_HABITAT_M
                        if actor_id == "human0"
                        else comparison.DOG_START_HABITAT_M
                    ),
                    "end_m": list(
                        comparison.HUMAN_END_HABITAT_M
                        if actor_id == "human0"
                        else comparison.DOG_END_HABITAT_M
                    ),
                    "trajectory_sha256": comparison.EXPECTED_TRAJECTORY_SHA256[
                        actor_id
                    ],
                }
                for actor_id in ("human0", "dog0")
            },
        },
    }


def test_habitat_center_navmesh_is_authority_and_never_full_body_claim():
    evidence = _valid_habitat_gate()
    assert comparison.validate_habitat_navmesh_authority_payload(evidence) is evidence
    evidence["pathfinder"]["routes"]["dog0"]["navigable_frame_count"] = 269
    with pytest.raises(ValueError, match="not 270/270"):
        comparison.validate_habitat_navmesh_authority_payload(evidence)
    config = comparison.default_configuration()["navigation_and_collision"]
    assert config["scene_mesh_collision_in_ue"] == "NoCollision"
    assert config["ue_raytrace_clearance_claim"] is False
    assert config["full_body_clearance_claim"] is False


def _valid_habitat_capture() -> dict:
    return {
        "schema": comparison.HABITAT_CAPTURE_SCHEMA,
        "status": "pass",
        "frame_count": 270,
        "frame_rate_hz": 15,
        "camera": {
            "horizontal_fov_deg": 90.0,
            "position_m": list(comparison.CAMERA_HABITAT_M),
            "rotation_xyzw": [0, 0, 0, 1],
        },
        "inputs": {
            "human_runtime_glb": {"sha256": comparison.EXPECTED_HUMAN_SHA256},
            "beagle_manifest": {
                "sha256": comparison.EXPECTED_BEAGLE_MANIFEST_SHA256
            },
            "beagle_m2_request": {
                "sha256": comparison.EXPECTED_BEAGLE_M2_REQUEST_SHA256
            },
            "m1_request": {"sha256": comparison.EXPECTED_M1_REQUEST_SHA256},
            "room_manifest": {
                "sha256": comparison.EXPECTED_ROOM_MANIFEST_SHA256
            },
            "route_provenance": {
                "route_id": comparison.ROUTE_ID,
                "path_generation": "linear_endpoint_interpolation_v1",
                "path_consumption": (
                    "derived_once_from_manifest_endpoints_then_verbatim"
                ),
                "human_trajectory_sha256": comparison.EXPECTED_TRAJECTORY_SHA256[
                    "human0"
                ],
                "dog_trajectory_sha256": comparison.EXPECTED_TRAJECTORY_SHA256[
                    "dog0"
                ],
            }
        },
    }


def _valid_habitat_delivery() -> dict:
    return {
        "schema": comparison.HABITAT_DELIVERY_SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "review_media": {
            "annotated_mp3d": {
                "schema": "avengine_m5_1_annotated_review_v1",
                "audio_muxed": True,
                "topdown_is_qa_only": True,
                "width": 1280,
                "height": 480,
                "frame_count": 270,
                "frame_rate_hz": 15,
                "duration_seconds": 18,
            }
        },
    }


def test_habitat_delivery_binds_exact_camera_fov_routes_and_review_contract():
    assert comparison.validate_habitat_capture_payload(_valid_habitat_capture())
    assert comparison.validate_habitat_delivery_payload(_valid_habitat_delivery())
    capture = _valid_habitat_capture()
    capture["camera"]["horizontal_fov_deg"] = 70
    with pytest.raises(ValueError, match="camera/route"):
        comparison.validate_habitat_capture_payload(capture)
    delivery = _valid_habitat_delivery()
    delivery["review_media"]["annotated_mp3d"]["topdown_is_qa_only"] = False
    with pytest.raises(ValueError, match="delivery/review"):
        comparison.validate_habitat_delivery_payload(delivery)


def test_editor_importer_owns_only_exact_dirs_and_imports_exact_m2_with_reload_gate():
    source = IMPORTER.read_text(encoding="utf-8")
    assert comparison.EXPECTED_M2_BEAGLE_SHA256 in source
    assert comparison.EXPECTED_RAW_MP3D_SHA256 in source
    module = ast.parse(source)
    literal_constants = {
        target.id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"BEAGLE_BP_CLASS_PATH"}
    }
    assert literal_constants["BEAGLE_BP_CLASS_PATH"] == (
        comparison.EXACT_BEAGLE_BP_CLASS_PATH
    )
    assert 'EXPECTED_BEAGLE_ANIMATIONS = {"Idle", "Walking"}' in source
    assert "EXPECTED_SCENE_STATIC_MESH_COUNT = 71" in source
    assert "MP3D_UE_VERIFY_ONLY" in source
    assert '"status": "pending"' in source
    assert '"status": "passed"' in source
    assert "_validate_managed_directory(path)" in source
    assert "unreal.SystemLibrary.quit_editor()" in source
    assert "validation never recreates it" in source
    assert "stable_dog_beagle" not in source
    assert "dd428da8c82a" not in source
    for directory in (
        "/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy",
        "/Game/MyAssets/Audioset/Meshes/gate_m2_beagle_v7_world_contact_r5",
        "/Game/MyAssets/Audioset/Blueprints/gate_m2_beagle_v7_world_contact_r5",
    ):
        assert directory in source


def test_runner_owns_fixed_step_entry_configuration_without_patching_example():
    source = RUNNER.read_text(encoding="utf-8")
    assert "def _configure_mp3d_instance(" in source
    assert "configure_gpurir_instance" not in source
    assert '"/Engine/Maps/Entry"' in source
    assert "COMMAND_LINE_ARGS.renderoffscreen = None" in source
    assert "INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME" in source
    assert 'config.SPEAR.INSTANCE.TEMP_DIR = settings["temp_dir"]' in source
    assert "config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID" in source
    assert "directional_component.SetCastShadows(bNewValue=True)" in source
    assert "directional_component.bCastDynamicShadow" not in source
    assert "directional_component.bCastStaticShadow" not in source
    assert "frame = read_frame(capture).copy()" in source


def test_unreal_struct_adapter_accepts_editor_and_packaged_return_shapes():
    location = {"x": 1.25, "y": -2.5, "z": 3.75}
    assert comparison._xyz_dict(location, ("x", "y", "z")) == pytest.approx(
        [1.25, -2.5, 3.75]
    )
    assert comparison._xyz_dict(
        {"ReturnValue": location}, ("x", "y", "z")
    ) == pytest.approx([1.25, -2.5, 3.75])
    assert comparison._xyz_dict(
        {"ReturnValue": {"ReturnValue": {"Roll": 1, "Pitch": 2, "Yaw": 3}}},
        ("roll", "pitch", "yaw"),
    ) == pytest.approx([1, 2, 3])


def test_unreal_struct_adapter_does_not_discard_named_outputs():
    with pytest.raises(RuntimeError, match="missing component x"):
        comparison._xyz_dict(
            {"ReturnValue": {"x": 1, "y": 2, "z": 3}, "WasSwept": False},
            ("x", "y", "z"),
        )


def _valid_review_probe(width: int, height: int) -> dict:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": width,
                "height": height,
                "avg_frame_rate": "15/1",
                "nb_read_frames": "270",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "sample_rate": "16000",
                "duration": "18.000000",
                "nb_read_frames": "282",
            },
        ],
        "format": {"duration": "18.000000"},
    }


def test_habitat_review_probe_requires_exact_av_contract():
    payload = _valid_review_probe(1280, 480)
    validated = comparison.validate_review_probe_payload(
        payload,
        expected_width=comparison.HABITAT_REVIEW_WIDTH,
        expected_height=comparison.HABITAT_REVIEW_HEIGHT,
        owner="test Habitat review",
    )
    assert validated["video"]["nb_read_frames"] == "270"
    assert validated["audio"]["channels"] == 2

    tampered = _valid_review_probe(1280, 480)
    tampered["streams"][1]["channels"] = 1
    with pytest.raises(ValueError, match="media contract changed"):
        comparison.validate_review_probe_payload(
            tampered,
            expected_width=1280,
            expected_height=480,
            owner="test Habitat review",
        )


def test_triptych_command_keeps_habitat_1280_panel_and_copies_its_audio():
    command = comparison.build_triptych_ffmpeg_command(
        ue_video_path=Path("ue.mp4"),
        habitat_review_path=Path("habitat.mp4"),
        output_path=Path("triptych.mp4"),
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "[0:v:0]scale=640:480:flags=lanczos" in graph
    assert "[1:v:0]setpts=PTS-STARTPTS,setsar=1[habitat]" in graph
    assert "[ue][habitat]hstack=inputs=2[comparison]" in graph
    assert command[command.index("-map") + 1] == "[comparison]"
    second_map = command.index("-map", command.index("-map") + 1)
    assert command[second_map + 1] == "1:a:0"
    assert command[command.index("-c:a") + 1] == "copy"
    assert command[command.index("-frames:v") + 1] == "270"
    assert command[command.index("-vsync") + 1] == "cfr"
    assert "-fps_mode" not in command
    panels = comparison.triptych_panel_definition()
    assert [(panel["panel_id"], panel["x"], panel["width"]) for panel in panels] == [
        ("ue_main", 0, 640),
        ("habitat_main", 640, 640),
        ("habitat_topdown", 1280, 640),
    ]
    assert panels[-1]["qa_only"] is True
