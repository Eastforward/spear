"""Render a fair UE/Habitat comparison in the exact Habitat MP3D room.

The authoritative geometry is the prepared/imported 17DRP5sb8fy scene, and
the authoritative actor-center routes are the 270/270 Habitat Pathfinder gate.
UE scene-mesh collision is disabled because that imported render geometry is
not a second navigation authority.  This runner never claims full-body or
runtime-OBB clearance.

Heavy UE rendering is deliberately isolated in :func:`run`; coordinate,
manifest, route, yaw, and configuration contracts remain importable in an
ordinary Python unit-test process.

The formal review artifact is a 1920x480 triptych: a 640x480 upscale of the
UE 320x240 main view followed by the unmodified 1280x480 Habitat main+topdown
review.  Its two-channel binaural stream is copied from that Habitat review;
the UE renderer does not synthesize a second or visually gated soundtrack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
TOOLS_DIR = REPO_ROOT / "tools"
SPIKE_RLR_DIR = TOOLS_DIR / "spike_rlr"
SCHEMA = "avengine_mp3d_ue_visual_comparison_v1"
IMPORT_SCHEMA = "avengine_mp3d_ue_import_result_v1"
HABITAT_GATE_SCHEMA = "avengine_m5_1_mp3d_mixed_visual_gate_v1"
HABITAT_CAPTURE_SCHEMA = "avengine_m5_1_human_beagle_capture_v1"
HABITAT_DELIVERY_SCHEMA = "avengine_m5_1_mp3d_delivery_v1"
SCENE_ID = "17DRP5sb8fy"
ROUTE_ID = "m5_1_mp3d_human_beagle_parallel_18s_v1"
FRAME_COUNT = 270
FPS = 15
WIDTH = 320
HEIGHT = 240
HFOV_DEG = 90.0
HABITAT_REVIEW_WIDTH = 1280
HABITAT_REVIEW_HEIGHT = 480
TRIPTYCH_WIDTH = 1920
TRIPTYCH_HEIGHT = 480
REVIEW_AUDIO_SAMPLE_RATE_HZ = 16_000
REVIEW_AUDIO_PACKET_COUNT = 282
DURATION_SECONDS = FRAME_COUNT / FPS
CAMERA_HABITAT_M = (-4.1499128342, 1.572447, -1.2454376221)
CAMERA_UE_CM = (-414.99128342, -124.54376221, 157.2447)
CAMERA_UE_YAW_DEG = -90.0
HUMAN_START_HABITAT_M = (-4.6, 0.072447, -2.7)
HUMAN_END_HABITAT_M = (-4.6, 0.072447, -3.8)
DOG_START_HABITAT_M = (-3.7, 0.072447, -2.7)
DOG_END_HABITAT_M = (-3.7, 0.072447, -3.8)
EXPECTED_TRAJECTORY_SHA256 = {
    "human0": "95303909b5d634dd04860d04ea62ff293499ec823837a98003794a0f395100b6",
    "dog0": "337c6af1ec905081a3d8bd3d731267328c1c1eb25f6b9ba7f939955c9050daf6",
}
HUMAN_START_UE_CM = (-460.0, -270.0, 7.2447)
HUMAN_END_UE_CM = (-460.0, -380.0, 7.2447)
DOG_START_UE_CM = (-370.0, -270.0, 7.2447)
DOG_END_UE_CM = (-370.0, -380.0, 7.2447)
ROUTE_MOTION_YAW_UE_DEG = -90.0
HUMAN_WALKING_LOCAL_FORWARD_AXIS_UE = "+Y"
HUMAN_WALKING_LOCAL_FORWARD_YAW_UE_DEG = 90.0
HUMAN_WALKING_FORWARD_YAW_OFFSET_DEG = -90.0
HUMAN_ACTOR_YAW_UE_DEG = -180.0
DOG_LOCAL_ANATOMICAL_FORWARD = "+X"
DOG_ACTOR_YAW_UE_DEG = -90.0
HUMAN_WALKING_CLIP_START_SECONDS = 1.0 / 30.0
HUMAN_WALKING_SAMPLE_COUNT = 16
DOG_WALKING_CLIP_START_SECONDS = 0.0
DOG_WALKING_SAMPLE_COUNT = 25
ANIMATION_POSITION_READBACK_TOLERANCE_SECONDS = 1.0e-4
EXPECTED_RAW_MP3D_SHA256 = (
    "334456925e056c83a9a7a5c768b3d37cdd23425d8ca20743bfce015be3f56b04"
)
EXPECTED_UE_SCENE_BOUNDS_CM = {
    "minimum": [-1159.3440055847168, -539.2021179199219, -12.755300104618073],
    "maximum": [475.7026195526123, 288.6620044708252, 267.87829399108887],
}
UE_SCENE_BOUNDS_TOLERANCE_CM = 0.5
EXPECTED_M2_BEAGLE_SHA256 = (
    "788a667537f7660bac5e128c38c2182453d1d4a9a4f8380343e7a9fa1947538c"
)
EXPECTED_HUMAN_SHA256 = (
    "02c13940227bd724d03aa91861e6acefde5457f0d2df9f808d427e88a3e94059"
)
EXPECTED_BEAGLE_MANIFEST_SHA256 = (
    "ad5df61f4b955980f6ab3d6d00e44f31942cb31ce3d2f2a3c1a1f353c307e240"
)
EXPECTED_BEAGLE_M2_REQUEST_SHA256 = (
    "8f77e6034b5ad4389f7b5828e7cd81049e8e50184f46f271bc6273b9fd63b5cc"
)
EXPECTED_M1_REQUEST_SHA256 = (
    "5f46923e60a87cb874444cc569d227aab51c2bae0e6a1c9930d5a0caba8af151"
)
EXPECTED_ROOM_MANIFEST_SHA256 = (
    "22ed412154222379464de7601d50a40c23df9e70c77b7b5f67da9d6acf3ed7d8"
)
HUMAN_TAG = "rocketbox_male_adult_01_original_ue_v3"
HUMAN_BP_PACKAGE_PATH = (
    "/Game/MyAssets/Audioset/Blueprints/"
    f"gate_{HUMAN_TAG}/BP_gate_{HUMAN_TAG}"
)
HUMAN_BP_CLASS_PATH = (
    f"{HUMAN_BP_PACKAGE_PATH}.BP_gate_{HUMAN_TAG}_C"
)
EXACT_BEAGLE_TAG = "m2_beagle_v7_world_contact_r5"
EXACT_BEAGLE_BP_PACKAGE_PATH = (
    "/Game/MyAssets/Audioset/Blueprints/"
    f"gate_{EXACT_BEAGLE_TAG}/BP_gate_{EXACT_BEAGLE_TAG}"
)
EXACT_BEAGLE_BP_CLASS_PATH = (
    f"{EXACT_BEAGLE_BP_PACKAGE_PATH}.BP_gate_{EXACT_BEAGLE_TAG}_C"
)
DEFAULT_HUMAN_UE_MANIFEST = (
    REPO_ROOT
    / "tmp"
    / "rocketbox_native_ue_import_v3"
    / HUMAN_TAG
    / "ue_import_manifest.json"
)
MAIN_LIGHT = {
    "kind": "movable_directional_light",
    "yaw_deg": -45.0,
    "pitch_deg": -50.0,
    "intensity_lux": 10.0,
    "cast_dynamic_shadows": True,
}
AMBIENT_LIGHT = {
    "kind": "native_sky_atmosphere_skylight_fog",
    "skylight_intensity": 0.35,
    "cast_shadows": False,
}
POSITION_READBACK_TOLERANCE_CM = 0.02
YAW_READBACK_TOLERANCE_DEG = 0.02
FOV_READBACK_TOLERANCE_DEG = 1.0e-4


def triptych_panel_definition() -> list[dict[str, Any]]:
    """Return the fixed spatial meaning of the formal comparison artifact."""

    return [
        {
            "panel_id": "ue_main",
            "x": 0,
            "y": 0,
            "width": 640,
            "height": 480,
            "source": "ue_main_320x240",
            "transform": "scale_2x_lanczos",
        },
        {
            "panel_id": "habitat_main",
            "x": 640,
            "y": 0,
            "width": 640,
            "height": 480,
            "source": "habitat_review_left_half",
            "transform": "none",
        },
        {
            "panel_id": "habitat_topdown",
            "x": 1280,
            "y": 0,
            "width": 640,
            "height": 480,
            "source": "habitat_review_right_half",
            "transform": "none",
            "qa_only": True,
        },
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} is missing or not a direct file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} root must be an object")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path.resolve()),
        "size_bytes": path.resolve().stat().st_size,
    }


def habitat_to_ue_cm(point_habitat_m: Sequence[float]) -> tuple[float, float, float]:
    """Map canonical Habitat H to UE U after UE glTF import.

    UE Interchange maps canonical glTF ``(x, y, z)`` to ``(x, z, y)`` and
    meters to centimeters, hence ``U=(H.x,H.z,H.y)*100``.
    """

    if len(point_habitat_m) != 3:
        raise ValueError("Habitat point must contain exactly three coordinates")
    x, y, z = (float(value) for value in point_habitat_m)
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("Habitat point contains a non-finite coordinate")
    return (x * 100.0, z * 100.0, y * 100.0)


def linear_route(
    start: Sequence[float], end: Sequence[float], frame_count: int = FRAME_COUNT
) -> list[tuple[float, float, float]]:
    if len(start) != 3 or len(end) != 3 or int(frame_count) < 2:
        raise ValueError("linear route needs two 3D endpoints and at least two frames")
    start_values = tuple(float(value) for value in start)
    end_values = tuple(float(value) for value in end)
    return [
        tuple(
            start_values[axis]
            + (end_values[axis] - start_values[axis]) * frame / (frame_count - 1)
            for axis in range(3)
        )
        for frame in range(frame_count)
    ]


def route_yaw_ue_deg(route_ue_cm: Sequence[Sequence[float]]) -> float:
    if len(route_ue_cm) < 2:
        raise ValueError("route needs at least two points")
    dx = float(route_ue_cm[-1][0]) - float(route_ue_cm[0][0])
    dy = float(route_ue_cm[-1][1]) - float(route_ue_cm[0][1])
    if math.hypot(dx, dy) <= 1.0e-9:
        raise ValueError("route has no horizontal displacement")
    return math.degrees(math.atan2(dy, dx))


def animation_phase_seconds(actor_id: str, frame_index: int) -> float:
    """Map a formal frame to the exact endpoint-exclusive Habitat sample."""

    if isinstance(frame_index, bool) or not 0 <= int(frame_index) < FRAME_COUNT:
        raise ValueError("animation frame index must be in the formal 270-frame range")
    if actor_id == "human0":
        start = HUMAN_WALKING_CLIP_START_SECONDS
        count = HUMAN_WALKING_SAMPLE_COUNT
    elif actor_id == "dog0":
        start = DOG_WALKING_CLIP_START_SECONDS
        count = DOG_WALKING_SAMPLE_COUNT
    else:
        raise ValueError(f"unknown comparison actor: {actor_id}")
    return start + (int(frame_index) % count) / FPS


def wrap_yaw_difference_deg(observed: float, expected: float) -> float:
    return (float(observed) - float(expected) + 180.0) % 360.0 - 180.0


def default_configuration() -> dict[str, Any]:
    human_h = linear_route(HUMAN_START_HABITAT_M, HUMAN_END_HABITAT_M)
    dog_h = linear_route(DOG_START_HABITAT_M, DOG_END_HABITAT_M)
    return {
        "schema": "avengine_mp3d_ue_comparison_config_v1",
        "scene_id": SCENE_ID,
        "route_id": ROUTE_ID,
        "render": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FPS,
            "width": WIDTH,
            "height": HEIGHT,
            "horizontal_fov_deg": HFOV_DEG,
            "streaming_warmup_frames": 120,
            "camera_warmup_frames": 40,
        },
        "coordinate_contract": {
            "matterport_source_to_habitat": "H=(S.x,S.z,-S.y)",
            "canonical_gltf_to_ue": "U=(H.x,H.z,H.y)*100cm",
        },
        "camera": {
            "habitat_position_m": list(CAMERA_HABITAT_M),
            "ue_position_cm": list(CAMERA_UE_CM),
            "habitat_forward": "-Z",
            "ue_forward": "-Y",
            "ue_yaw_deg": CAMERA_UE_YAW_DEG,
        },
        "actors": {
            "human0": {
                "route_habitat_m": [list(point) for point in human_h],
                "route_ue_cm": [list(habitat_to_ue_cm(point)) for point in human_h],
                "motion_yaw_ue_deg": ROUTE_MOTION_YAW_UE_DEG,
                "walking_local_forward_axis_ue": (
                    HUMAN_WALKING_LOCAL_FORWARD_AXIS_UE
                ),
                "walking_local_forward_yaw_ue_deg": (
                    HUMAN_WALKING_LOCAL_FORWARD_YAW_UE_DEG
                ),
                "walking_forward_yaw_offset_deg": (
                    HUMAN_WALKING_FORWARD_YAW_OFFSET_DEG
                ),
                "actor_yaw_ue_deg": HUMAN_ACTOR_YAW_UE_DEG,
                "actor_scale": 1.0,
                "animation": "Walking",
                "animation_phase": {
                    "clip_start_seconds": HUMAN_WALKING_CLIP_START_SECONDS,
                    "sample_count": HUMAN_WALKING_SAMPLE_COUNT,
                    "sample_rate_hz": FPS,
                    "playback": "explicit_fixed_state_modulo_loop",
                },
            },
            "dog0": {
                "route_habitat_m": [list(point) for point in dog_h],
                "route_ue_cm": [list(habitat_to_ue_cm(point)) for point in dog_h],
                "motion_yaw_ue_deg": ROUTE_MOTION_YAW_UE_DEG,
                "anatomical_forward_axis": DOG_LOCAL_ANATOMICAL_FORWARD,
                "actor_yaw_ue_deg": DOG_ACTOR_YAW_UE_DEG,
                "actor_scale": 1.0,
                "animation": "Walking",
                "animation_phase": {
                    "clip_start_seconds": DOG_WALKING_CLIP_START_SECONDS,
                    "sample_count": DOG_WALKING_SAMPLE_COUNT,
                    "sample_rate_hz": FPS,
                    "playback": "explicit_fixed_state_modulo_loop",
                },
            },
        },
        "lighting": {
            "main": dict(MAIN_LIGHT),
            "ambient": dict(AMBIENT_LIGHT),
            "purpose": (
                "UE visual-engine comparison with an explicit movable shadow "
                "key plus bounded ambient; not an attempt to reproduce baked "
                "Matterport capture illumination."
            ),
        },
        "navigation_and_collision": {
            "planning_authority": "Habitat Pathfinder 270/270 actor centers",
            "center_navigation_only": True,
            "scene_mesh_collision_in_ue": "NoCollision",
            "actor_spawn_collision_override": "AlwaysSpawn",
            "ue_raytrace_clearance_claim": False,
            "full_body_clearance_claim": False,
        },
    }


def _assert_vector_close(
    actual: Sequence[float], expected: Sequence[float], tolerance: float = 1.0e-6
) -> None:
    if len(actual) != len(expected) or any(
        abs(float(a) - float(e)) > tolerance for a, e in zip(actual, expected)
    ):
        raise ValueError(f"vector {list(actual)} != expected {list(expected)}")


def validate_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    if (
        configuration.get("schema") != "avengine_mp3d_ue_comparison_config_v1"
        or configuration.get("scene_id") != SCENE_ID
        or configuration.get("route_id") != ROUTE_ID
    ):
        raise ValueError("MP3D UE comparison configuration identity changed")
    render = configuration.get("render", {})
    if render != {
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "horizontal_fov_deg": HFOV_DEG,
        "streaming_warmup_frames": 120,
        "camera_warmup_frames": 40,
    }:
        raise ValueError("fixed UE comparison render contract changed")
    camera = configuration.get("camera", {})
    _assert_vector_close(camera.get("habitat_position_m", []), CAMERA_HABITAT_M)
    _assert_vector_close(camera.get("ue_position_cm", []), CAMERA_UE_CM)
    _assert_vector_close(
        habitat_to_ue_cm(camera["habitat_position_m"]), camera["ue_position_cm"]
    )
    if camera.get("ue_yaw_deg") != CAMERA_UE_YAW_DEG:
        raise ValueError("fixed camera yaw changed")

    expected = {
        "human0": (
            HUMAN_START_HABITAT_M,
            HUMAN_END_HABITAT_M,
            HUMAN_START_UE_CM,
            HUMAN_END_UE_CM,
            HUMAN_ACTOR_YAW_UE_DEG,
        ),
        "dog0": (
            DOG_START_HABITAT_M,
            DOG_END_HABITAT_M,
            DOG_START_UE_CM,
            DOG_END_UE_CM,
            DOG_ACTOR_YAW_UE_DEG,
        ),
    }
    for actor_id, values in expected.items():
        actor = configuration.get("actors", {}).get(actor_id, {})
        route_h = actor.get("route_habitat_m", [])
        route_ue = actor.get("route_ue_cm", [])
        if len(route_h) != FRAME_COUNT or len(route_ue) != FRAME_COUNT:
            raise ValueError(f"{actor_id} route does not contain 270 frames")
        _assert_vector_close(route_h[0], values[0])
        _assert_vector_close(route_h[-1], values[1])
        _assert_vector_close(route_ue[0], values[2])
        _assert_vector_close(route_ue[-1], values[3])
        for point_h, point_ue in zip(route_h, route_ue):
            _assert_vector_close(habitat_to_ue_cm(point_h), point_ue)
        yaw = route_yaw_ue_deg(route_ue)
        if abs(wrap_yaw_difference_deg(yaw, ROUTE_MOTION_YAW_UE_DEG)) > 1.0e-8:
            raise ValueError(f"{actor_id} UE route yaw changed")
        if actor.get("actor_yaw_ue_deg") != values[4]:
            raise ValueError(f"{actor_id} actor yaw changed")
    human = configuration["actors"]["human0"]
    if (
        human.get("walking_local_forward_axis_ue") != "+Y"
        or human.get("walking_local_forward_yaw_ue_deg")
        != HUMAN_WALKING_LOCAL_FORWARD_YAW_UE_DEG
        or human.get("walking_forward_yaw_offset_deg")
        != HUMAN_WALKING_FORWARD_YAW_OFFSET_DEG
        or human["motion_yaw_ue_deg"]
        + human["walking_forward_yaw_offset_deg"]
        != human["actor_yaw_ue_deg"]
    ):
        raise ValueError("Rocketbox human walking yaw binding changed")
    dog = configuration["actors"]["dog0"]
    if (
        dog.get("anatomical_forward_axis") != "+X"
        or dog["motion_yaw_ue_deg"] != dog["actor_yaw_ue_deg"]
    ):
        raise ValueError("M2 Beagle +X forward yaw binding changed")
    expected_phases = {
        "human0": {
            "clip_start_seconds": HUMAN_WALKING_CLIP_START_SECONDS,
            "sample_count": HUMAN_WALKING_SAMPLE_COUNT,
            "sample_rate_hz": FPS,
            "playback": "explicit_fixed_state_modulo_loop",
        },
        "dog0": {
            "clip_start_seconds": DOG_WALKING_CLIP_START_SECONDS,
            "sample_count": DOG_WALKING_SAMPLE_COUNT,
            "sample_rate_hz": FPS,
            "playback": "explicit_fixed_state_modulo_loop",
        },
    }
    for actor_id, expected_phase in expected_phases.items():
        if configuration["actors"][actor_id].get("animation_phase") != expected_phase:
            raise ValueError(f"{actor_id} animation phase contract changed")
    if configuration.get("lighting", {}).get("main") != MAIN_LIGHT:
        raise ValueError("explicit UE main-light contract changed")
    if configuration.get("lighting", {}).get("ambient") != AMBIENT_LIGHT:
        raise ValueError("explicit UE ambient-light contract changed")
    collision = configuration.get("navigation_and_collision", {})
    if (
        collision.get("planning_authority")
        != "Habitat Pathfinder 270/270 actor centers"
        or collision.get("scene_mesh_collision_in_ue") != "NoCollision"
        or collision.get("ue_raytrace_clearance_claim") is not False
        or collision.get("full_body_clearance_claim") is not False
    ):
        raise ValueError("center-navigation/UE-collision claim boundary changed")
    return configuration


def validate_ue_import_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    scene = manifest.get("scene_content", {})
    beagle = manifest.get("m2_beagle", {})
    beagle_content = beagle.get("content", {})
    static_meshes = scene.get("static_meshes", [])
    bounds = scene.get("ue_bounds", {})
    if (
        manifest.get("schema") != IMPORT_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("scene_id") != SCENE_ID
        or manifest.get("source", {}).get("sha256")
        != EXPECTED_RAW_MP3D_SHA256
        or manifest.get("reload_verification", {}).get("status") != "passed"
        or scene.get("status") != "passed"
        or scene.get("static_mesh_count") != 71
        or len(static_meshes) != 71
        or len(set(static_meshes)) != 71
        or bounds.get("status") != "passed"
        or bounds.get("expected") != EXPECTED_UE_SCENE_BOUNDS_CM
        or bounds.get("tolerance_cm") != UE_SCENE_BOUNDS_TOLERANCE_CM
        or float(bounds.get("maximum_absolute_error_cm", math.inf))
        > UE_SCENE_BOUNDS_TOLERANCE_CM
        or any(
            not path.startswith(
                "/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy/"
            )
            for path in static_meshes
        )
        or beagle.get("source", {}).get("sha256")
        != EXPECTED_M2_BEAGLE_SHA256
        or beagle.get("exact_habitat_m2_runtime") is not True
        or beagle_content.get("status") != "passed"
        or beagle_content.get("blueprint_package_path")
        != EXACT_BEAGLE_BP_PACKAGE_PATH
        or beagle_content.get("blueprint_class_path") != EXACT_BEAGLE_BP_CLASS_PATH
        or set(beagle_content.get("animations", {})) != {"Idle", "Walking"}
    ):
        raise ValueError("UE import manifest is not a reload-verified exact comparison")
    validate_ue_scene_bounds(
        bounds.get("minimum_cm", []),
        bounds.get("maximum_cm", []),
        owner="reload-verified UE editor StaticMesh bounds",
    )
    return manifest


def load_and_validate_ue_import_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    manifest = validate_ue_import_manifest_payload(
        _load_json(path, "MP3D UE import manifest")
    )
    for key in ("source", "prepared"):
        record = manifest[key]
        artifact = Path(record["path"]).resolve()
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or _sha256(artifact) != record.get("sha256")
            or artifact.stat().st_size != record.get("size_bytes")
        ):
            raise ValueError(f"UE import {key} artifact authentication failed")
    beagle_source = manifest["m2_beagle"]["source"]
    beagle_path = Path(beagle_source["path"]).resolve()
    if (
        not beagle_path.is_file()
        or beagle_path.is_symlink()
        or _sha256(beagle_path) != EXPECTED_M2_BEAGLE_SHA256
        or beagle_path.stat().st_size != beagle_source.get("size_bytes")
    ):
        raise ValueError("exact M2 Beagle artifact authentication failed")
    return manifest


def validate_habitat_navmesh_authority_payload(
    evidence: dict[str, Any]
) -> dict[str, Any]:
    pathfinder = evidence.get("pathfinder", {})
    routes = pathfinder.get("routes", {})
    if (
        evidence.get("schema") != HABITAT_GATE_SCHEMA
        or evidence.get("status") != "pass"
        or evidence.get("route_id") != ROUTE_ID
        or evidence.get("frame_count") != FRAME_COUNT
        or evidence.get("frame_rate_hz") != FPS
        or pathfinder.get("center_navigation_semantics")
        != "actor_root_center_only"
        or pathfinder.get("declared_navmesh_loaded") is not True
    ):
        raise ValueError("Habitat MP3D navmesh authority contract changed")
    endpoints = {
        "human0": (HUMAN_START_HABITAT_M, HUMAN_END_HABITAT_M),
        "dog0": (DOG_START_HABITAT_M, DOG_END_HABITAT_M),
    }
    for actor_id in ("human0", "dog0"):
        route = routes.get(actor_id, {})
        if (
            route.get("all_frames_navigable") is not True
            or route.get("navigable_frame_count") != FRAME_COUNT
            or route.get("frame_count") != FRAME_COUNT
            or route.get("trajectory_sha256")
            != EXPECTED_TRAJECTORY_SHA256[actor_id]
            or float(route.get("maximum_snap_error_m", math.inf))
            > float(route.get("required_maximum_snap_error_m", -math.inf))
        ):
            raise ValueError(f"Habitat route is not 270/270 navigable: {actor_id}")
        _assert_vector_close(route.get("start_m", []), endpoints[actor_id][0])
        _assert_vector_close(route.get("end_m", []), endpoints[actor_id][1])
    if any(gate.get("status") != "pass" for gate in evidence.get("gates", [])):
        raise ValueError("Habitat MP3D gate contains a non-pass entry")
    return evidence


def validate_habitat_capture_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    camera = evidence.get("camera", {})
    inputs = evidence.get("inputs", {})
    provenance = inputs.get("route_provenance", {})
    if (
        evidence.get("schema") != HABITAT_CAPTURE_SCHEMA
        or evidence.get("status") != "pass"
        or evidence.get("frame_count") != FRAME_COUNT
        or evidence.get("frame_rate_hz") != FPS
        or camera.get("horizontal_fov_deg") != HFOV_DEG
        or camera.get("rotation_xyzw") != [0, 0, 0, 1]
        or provenance.get("route_id") != ROUTE_ID
        or provenance.get("path_generation")
        != "linear_endpoint_interpolation_v1"
        or provenance.get("path_consumption")
        != "derived_once_from_manifest_endpoints_then_verbatim"
        or provenance.get("human_trajectory_sha256")
        != EXPECTED_TRAJECTORY_SHA256["human0"]
        or provenance.get("dog_trajectory_sha256")
        != EXPECTED_TRAJECTORY_SHA256["dog0"]
        or inputs.get("human_runtime_glb", {}).get("sha256")
        != EXPECTED_HUMAN_SHA256
        or inputs.get("beagle_manifest", {}).get("sha256")
        != EXPECTED_BEAGLE_MANIFEST_SHA256
        or inputs.get("beagle_m2_request", {}).get("sha256")
        != EXPECTED_BEAGLE_M2_REQUEST_SHA256
        or inputs.get("m1_request", {}).get("sha256")
        != EXPECTED_M1_REQUEST_SHA256
        or inputs.get("room_manifest", {}).get("sha256")
        != EXPECTED_ROOM_MANIFEST_SHA256
    ):
        raise ValueError("Habitat capture camera/route contract changed")
    _assert_vector_close(camera.get("position_m", []), CAMERA_HABITAT_M)
    return evidence


def validate_habitat_delivery_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    review = evidence.get("review_media", {}).get("annotated_mp3d", {})
    if (
        evidence.get("schema") != HABITAT_DELIVERY_SCHEMA
        or evidence.get("status") != "pass"
        or evidence.get("research_only") is not True
        or evidence.get("qualification_claim") is not False
        or review.get("schema") != "avengine_m5_1_annotated_review_v1"
        or review.get("audio_muxed") is not True
        or review.get("topdown_is_qa_only") is not True
        or review.get("width") != HABITAT_REVIEW_WIDTH
        or review.get("height") != HABITAT_REVIEW_HEIGHT
        or review.get("frame_count") != FRAME_COUNT
        or review.get("frame_rate_hz") != FPS
        or float(review.get("duration_seconds", -1)) != DURATION_SECONDS
    ):
        raise ValueError("Habitat delivery/review manifest contract changed")
    return evidence


def _authenticate_declared_artifact(
    *,
    record: dict[str, Any],
    expected_path: Path | None,
    base_dir: Path | None = None,
    owner: str,
) -> Path:
    declared_path = Path(record.get("path", ""))
    if not declared_path.is_absolute():
        if base_dir is None:
            raise ValueError(f"{owner} uses an unanchored relative path")
        declared_path = base_dir / declared_path
    path = declared_path.resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{owner} declared path does not match the CLI input")
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256(path) != record.get("sha256")
        or path.stat().st_size != record.get("byte_size")
    ):
        raise ValueError(f"{owner} artifact authentication failed: {path}")
    return path


def load_and_validate_habitat_delivery(
    *,
    delivery_path: Path,
    gate_path: Path,
    review_video_path: Path,
) -> dict[str, Any]:
    delivery_path = delivery_path.resolve()
    delivery = validate_habitat_delivery_payload(
        _load_json(delivery_path, "Habitat MP3D delivery evidence")
    )
    inputs = delivery.get("inputs", {})
    gate_record = inputs.get("mp3d_gate_evidence", {})
    _authenticate_declared_artifact(
        record=gate_record,
        expected_path=gate_path.resolve(),
        owner="Habitat delivery MP3D gate",
    )
    capture_record = inputs.get("capture_evidence", {})
    capture_path = _authenticate_declared_artifact(
        record=capture_record,
        expected_path=None,
        owner="Habitat delivery capture evidence",
    )
    capture = validate_habitat_capture_payload(
        _load_json(capture_path, "Habitat MP3D capture evidence")
    )
    review_record = delivery["review_media"]["annotated_mp3d"]
    declared_review = (delivery_path.parent / review_record["path"]).resolve()
    if declared_review != review_video_path.resolve():
        raise ValueError("Habitat delivery review video differs from CLI input")
    _authenticate_declared_artifact(
        record=review_record,
        expected_path=review_video_path.resolve(),
        base_dir=delivery_path.parent,
        owner="Habitat annotated review video",
    )
    return {
        "delivery": delivery,
        "capture": capture,
        "capture_path": capture_path,
        "review_record": review_record,
    }


def validate_human_ue_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    content = manifest.get("content", {})
    if (
        manifest.get("schema") != "rocketbox_native_ue_import_v3"
        or manifest.get("tag") != HUMAN_TAG
        or manifest.get("source_glb_sha256") != EXPECTED_HUMAN_SHA256
        or manifest.get("reload_verification", {}).get("status") != "passed"
        or content.get("blueprint") != HUMAN_BP_PACKAGE_PATH
        or "Walking" not in content.get("animations", {})
    ):
        raise ValueError("Rocketbox human UE binding manifest changed")
    return manifest


def _unwrap_return_value(value: Any) -> Any:
    """Unwrap SPEAR's packaged-build representation of a return value.

    Depending on whether the bridge is talking to the editor or a packaged
    build, an Unreal struct return can be serialized either as the struct
    mapping itself or as ``{"ReturnValue": <struct>}``.  Only unwrap a mapping
    whose sole field is that wrapper so genuine multi-output functions keep
    all of their named outputs.
    """

    while (
        isinstance(value, dict)
        and len(value) == 1
        and "ReturnValue" in value
    ):
        value = value["ReturnValue"]
    return value


def _xyz_dict(value: dict[str, Any], names: Sequence[str]) -> list[float]:
    value = _unwrap_return_value(value)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected Unreal struct mapping, got {value!r}")
    result = []
    for name in names:
        candidates = (name, name.lower(), name.upper(), name.capitalize())
        for candidate in candidates:
            if candidate in value:
                result.append(float(value[candidate]))
                break
        else:
            raise RuntimeError(f"missing component {name} in {value}")
    return result


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _set_collision_disabled(component: Any) -> None:
    try:
        component.SetCollisionEnabled(NewType="NoCollision")
    except Exception:
        component.set_property_value(
            property_name="CollisionEnabled", property_value="NoCollision"
        )


def validate_ue_scene_bounds(
    minimum_cm: Sequence[float],
    maximum_cm: Sequence[float],
    *,
    owner: str,
) -> dict[str, Any]:
    _assert_vector_close(
        minimum_cm,
        EXPECTED_UE_SCENE_BOUNDS_CM["minimum"],
        tolerance=UE_SCENE_BOUNDS_TOLERANCE_CM,
    )
    _assert_vector_close(
        maximum_cm,
        EXPECTED_UE_SCENE_BOUNDS_CM["maximum"],
        tolerance=UE_SCENE_BOUNDS_TOLERANCE_CM,
    )
    maximum_error = max(
        abs(float(observed) - EXPECTED_UE_SCENE_BOUNDS_CM[key][axis])
        for key, values in (("minimum", minimum_cm), ("maximum", maximum_cm))
        for axis, observed in enumerate(values)
    )
    return {
        "status": "passed",
        "owner": owner,
        "minimum_cm": [float(value) for value in minimum_cm],
        "maximum_cm": [float(value) for value in maximum_cm],
        "expected": EXPECTED_UE_SCENE_BOUNDS_CM,
        "maximum_absolute_error_cm": maximum_error,
        "tolerance_cm": UE_SCENE_BOUNDS_TOLERANCE_CM,
    }


def _component_local_bounds(component: Any) -> tuple[list[float], list[float]]:
    result = component.GetLocalBounds(
        Min={"X": 0.0, "Y": 0.0, "Z": 0.0},
        Max={"X": 0.0, "Y": 0.0, "Z": 0.0},
        as_dict=True,
    )
    if isinstance(result.get("ReturnValue"), dict):
        result = result["ReturnValue"]
    minimum = result.get("Min", result.get("min"))
    maximum = result.get("Max", result.get("max"))
    if not isinstance(minimum, dict) or not isinstance(maximum, dict):
        raise RuntimeError(f"unexpected UStaticMeshComponent local bounds: {result}")
    return _xyz_dict(minimum, ("x", "y", "z")), _xyz_dict(
        maximum, ("x", "y", "z")
    )


def _spawn_scene_meshes(
    game: Any, mesh_paths: Iterable[str]
) -> tuple[list[Any], dict[str, Any]]:
    actors = []
    per_mesh_bounds = []
    aggregate_minimum = [math.inf] * 3
    aggregate_maximum = [-math.inf] * 3
    for index, mesh_path in enumerate(mesh_paths):
        mesh = game.unreal_service.load_object(uclass="UStaticMesh", name=mesh_path)
        actor = game.unreal_service.spawn_actor(
            uclass="AStaticMeshActor",
            location={"X": 0.0, "Y": 0.0, "Z": 0.0},
            spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
        )
        component = game.unreal_service.get_component_by_class(
            actor=actor, uclass="UStaticMeshComponent"
        )
        component.SetMobility(NewMobility="Movable")
        component.SetStaticMesh(NewMesh=mesh)
        component.SetCastShadow(NewCastShadow=True)
        _set_collision_disabled(component)
        minimum, maximum = _component_local_bounds(component)
        for axis in range(3):
            aggregate_minimum[axis] = min(aggregate_minimum[axis], minimum[axis])
            aggregate_maximum[axis] = max(aggregate_maximum[axis], maximum[axis])
        per_mesh_bounds.append(
            {
                "object_path": mesh_path,
                "minimum_cm": minimum,
                "maximum_cm": maximum,
            }
        )
        game.unreal_service.set_stable_name_for_actor(
            actor=actor, stable_name=f"MP3D/{SCENE_ID}/mesh_{index:03d}"
        )
        actors.append(actor)
    if len(actors) != 71:
        raise RuntimeError(f"spawned MP3D mesh actor count {len(actors)} != 71")
    bounds = validate_ue_scene_bounds(
        aggregate_minimum,
        aggregate_maximum,
        owner="packaged runtime UStaticMeshComponent local bounds",
    )
    bounds["per_mesh"] = per_mesh_bounds
    return actors, bounds


def _spawn_lighting(game: Any, spawn_sky: Any, spawn_directional_light: Any) -> dict[str, Any]:
    sky = spawn_sky(game=game)
    directional = spawn_directional_light(
        game=game,
        yaw_deg=MAIN_LIGHT["yaw_deg"],
        pitch_deg=MAIN_LIGHT["pitch_deg"],
        intensity_lux=MAIN_LIGHT["intensity_lux"],
    )
    directional_component = game.unreal_service.get_component_by_class(
        actor=directional, uclass="UDirectionalLightComponent"
    )
    directional_component.SetCastShadows(bNewValue=True)
    # Do not assign C++ member-style names such as ``bCastDynamicShadow``
    # through UnrealObject.__setattr__.  The packaged SPEAR reflection bridge
    # resolves the exported property name (without the C++ ``b`` prefix), and
    # a failed property lookup puts the whole RPC service into an error state.
    # A movable directional light with CastShadows enabled already follows the
    # dynamic-shadow path; SetCastShadows is the engine-backed, packaged-safe
    # API that the existing light helpers also use.
    skylight_readback = None
    skylight = sky.get("ASkyLight")
    if skylight is not None:
        skylight_component = game.unreal_service.get_component_by_class(
            actor=skylight, uclass="USkyLightComponent"
        )
        skylight_component.SetIntensity(NewIntensity=AMBIENT_LIGHT["skylight_intensity"])
        try:
            skylight_component.SetCastShadows(bNewValue=False)
        except Exception:
            pass
        try:
            skylight_readback = float(
                skylight_component.get_property_value("Intensity")
            )
        except Exception:
            skylight_readback = AMBIENT_LIGHT["skylight_intensity"]
    return {
        "status": "configured",
        "main": {
            **MAIN_LIGHT,
            "mobility_requested": "Movable",
            "intensity_readback_lux": float(
                directional_component.get_property_value("Intensity")
            ),
            "cast_shadows_readback": bool(
                directional_component.get_property_value("CastShadows")
            ),
        },
        "ambient": {
            **AMBIENT_LIGHT,
            "spawned_native_actor_classes": sorted(sky),
            "skylight_intensity_readback": skylight_readback,
        },
        "claim_boundary": (
            "Movable UE key-light shadowing is configured; this is not a "
            "reconstruction of unknown Matterport capture lights."
        ),
    }


def _spawn_skeletal_actor(
    *,
    game: Any,
    blueprint_class_path: str,
    animation_path: str,
    position_cm: Sequence[float],
    yaw_deg: float,
    scale: float,
) -> tuple[Any, Any, float, float]:
    if str(blueprint_class_path).startswith(EXACT_BEAGLE_BP_PACKAGE_PATH):
        expected_class = EXACT_BEAGLE_BP_CLASS_PATH
    else:
        expected_class = HUMAN_BP_CLASS_PATH
    if blueprint_class_path != expected_class:
        raise RuntimeError(f"unapproved comparison actor BP: {blueprint_class_path}")
    blueprint = game.unreal_service.load_class(
        uclass="AActor", name=blueprint_class_path
    )
    actor = game.unreal_service.spawn_actor(
        uclass=blueprint,
        location={"X": position_cm[0], "Y": position_cm[1], "Z": position_cm[2]},
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    actor.SetActorEnableCollision(bNewActorEnableCollision=False)
    actor.SetActorTickEnabled(bEnabled=True)
    actor.SetActorScale3D(NewScale3D={"X": scale, "Y": scale, "Z": scale})
    sys.path.insert(0, str(SPIKE_RLR_DIR))
    from rig_direction_check import select_skeletal_mesh_component

    component = select_skeletal_mesh_component(
        unreal_service=game.unreal_service, actor=actor
    )
    if component is None:
        raise RuntimeError(f"no populated skeletal component for {blueprint_class_path}")
    component.SetComponentTickEnabled(bEnabled=True)
    component.SetCastShadow(NewCastShadow=True)
    animation = game.unreal_service.load_object(
        uclass="UAnimationAsset", name=animation_path
    )
    animation_length_seconds = float(animation.GetPlayLength())
    if not math.isfinite(animation_length_seconds) or animation_length_seconds <= 0.0:
        raise RuntimeError(
            f"animation length is invalid for {blueprint_class_path}: "
            f"{animation_length_seconds}"
        )
    component.PlayAnimation(NewAnimToPlay=animation, bLooping=True)
    component.set_property_value(
        property_name="GlobalAnimRateScale", property_value=1.0
    )
    play_rate = float(component.get_property_value("GlobalAnimRateScale"))
    if abs(play_rate - 1.0) > 1.0e-6:
        raise RuntimeError(f"animation rate readback {play_rate} != 1.0")
    actor.K2_SetActorLocationAndRotation(
        NewLocation={"X": position_cm[0], "Y": position_cm[1], "Z": position_cm[2]},
        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": yaw_deg},
        bSweep=False,
        bTeleport=True,
    )
    return actor, component, play_rate, animation_length_seconds


def _set_animation_phase(
    component: Any,
    *,
    actor_id: str,
    frame_index: int,
) -> dict[str, Any]:
    requested = animation_phase_seconds(actor_id, frame_index)
    component.SetPosition(InPos=requested, bFireNotifies=False)
    observed = float(component.GetPosition())
    error = abs(observed - requested)
    if error > ANIMATION_POSITION_READBACK_TOLERANCE_SECONDS:
        raise RuntimeError(
            f"{actor_id} animation phase readback failed at frame {frame_index}: "
            f"requested={requested} observed={observed}"
        )
    return {
        "frame_index": frame_index,
        "requested_position_seconds": requested,
        "readback_position_seconds": observed,
        "absolute_error_seconds": error,
    }


def _animation_phase_gate(
    records: Sequence[dict[str, Any]], *, actor_id: str
) -> dict[str, Any]:
    if len(records) != FRAME_COUNT:
        raise RuntimeError(f"{actor_id} animation phase gate requires 270 records")
    maximum_error = max(float(record["absolute_error_seconds"]) for record in records)
    for frame_index, record in enumerate(records):
        if record.get("frame_index") != frame_index:
            raise RuntimeError(f"{actor_id} animation phase frame order changed")
        expected = animation_phase_seconds(actor_id, frame_index)
        if abs(float(record["requested_position_seconds"]) - expected) > 1.0e-12:
            raise RuntimeError(f"{actor_id} animation phase request changed")
    if maximum_error > ANIMATION_POSITION_READBACK_TOLERANCE_SECONDS:
        raise RuntimeError(f"{actor_id} animation phase readback tolerance failed")
    return {
        "status": "passed",
        "playback": "explicit_fixed_state_modulo_loop",
        "frame_count": len(records),
        "maximum_absolute_error_seconds": maximum_error,
        "tolerance_seconds": ANIMATION_POSITION_READBACK_TOLERANCE_SECONDS,
        "per_frame": list(records),
    }


def _actor_readback(actor: Any, frame_index: int) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "location_cm": _xyz_dict(
            actor.K2_GetActorLocation(as_dict=True), ("x", "y", "z")
        ),
        "rotation_deg": _xyz_dict(
            actor.K2_GetActorRotation(as_dict=True), ("roll", "pitch", "yaw")
        ),
    }


def _readback_gate(
    readbacks: Sequence[dict[str, Any]],
    expected_route: Sequence[Sequence[float]],
    expected_yaw_deg: float,
) -> dict[str, Any]:
    if len(readbacks) != FRAME_COUNT or len(expected_route) != FRAME_COUNT:
        raise RuntimeError("root readback gate requires exactly 270 frames")
    position_errors = []
    yaw_errors = []
    roll_pitch_errors = []
    for frame, (readback, expected) in enumerate(zip(readbacks, expected_route)):
        if readback.get("frame_index") != frame:
            raise RuntimeError("root readback frame order changed")
        location = readback["location_cm"]
        rotation = readback["rotation_deg"]
        position_errors.append(
            max(abs(float(location[axis]) - float(expected[axis])) for axis in range(3))
        )
        yaw_errors.append(
            abs(wrap_yaw_difference_deg(rotation[2], expected_yaw_deg))
        )
        roll_pitch_errors.append(max(abs(rotation[0]), abs(rotation[1])))
    maximum_position_error = max(position_errors)
    maximum_yaw_error = max(yaw_errors)
    maximum_roll_pitch = max(roll_pitch_errors)
    status = (
        "passed"
        if maximum_position_error <= POSITION_READBACK_TOLERANCE_CM
        and maximum_yaw_error <= YAW_READBACK_TOLERANCE_DEG
        and maximum_roll_pitch <= YAW_READBACK_TOLERANCE_DEG
        else "failed"
    )
    if status != "passed":
        raise RuntimeError(
            "UE actor root readback failed: "
            f"position={maximum_position_error} yaw={maximum_yaw_error} "
            f"roll_pitch={maximum_roll_pitch}"
        )
    return {
        "status": status,
        "frame_count": len(readbacks),
        "maximum_absolute_position_error_cm": maximum_position_error,
        "maximum_absolute_yaw_error_deg": maximum_yaw_error,
        "maximum_absolute_roll_or_pitch_deg": maximum_roll_pitch,
        "position_tolerance_cm": POSITION_READBACK_TOLERANCE_CM,
        "yaw_tolerance_deg": YAW_READBACK_TOLERANCE_DEG,
        "per_frame": list(readbacks),
    }


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("ffprobe did not find one video stream")
    stream = streams[0]
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", -1)) != WIDTH
        or int(stream.get("height", -1)) != HEIGHT
        or stream.get("avg_frame_rate") != f"{FPS}/1"
        or int(stream.get("nb_read_frames", -1)) != FRAME_COUNT
    ):
        raise RuntimeError(f"UE MP3D video readback changed: {stream}")
    return {"status": "passed", **stream, **_file_record(path)}


def _probe_av(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,width,height,avg_frame_rate,"
            "nb_read_frames,duration,sample_rate,channels,channel_layout:"
            "format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"ffprobe returned no JSON object for {path}")
    return payload


def validate_review_probe_payload(
    payload: dict[str, Any],
    *,
    expected_width: int,
    expected_height: int,
    owner: str,
) -> dict[str, Any]:
    """Validate a formal 18-second H.264 + two-channel AAC review."""

    streams = payload.get("streams", [])
    video_streams = [
        stream for stream in streams if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream for stream in streams if stream.get("codec_type") == "audio"
    ]
    if len(video_streams) != 1 or len(audio_streams) != 1:
        raise ValueError(f"{owner} must contain exactly one video and one audio stream")
    video = video_streams[0]
    audio = audio_streams[0]
    try:
        format_duration = float(payload.get("format", {}).get("duration"))
        audio_duration = float(audio.get("duration"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{owner} has no numeric container duration") from error
    if (
        video.get("codec_name") != "h264"
        or int(video.get("width", -1)) != expected_width
        or int(video.get("height", -1)) != expected_height
        or video.get("avg_frame_rate") != f"{FPS}/1"
        or int(video.get("nb_read_frames", -1)) != FRAME_COUNT
        or abs(format_duration - DURATION_SECONDS) > 1.0 / FPS
        or abs(audio_duration - DURATION_SECONDS) > 1.0 / FPS
        or audio.get("codec_name") != "aac"
        or int(audio.get("channels", -1)) != 2
        or int(audio.get("sample_rate", -1)) != REVIEW_AUDIO_SAMPLE_RATE_HZ
        or int(audio.get("nb_read_frames", -1)) != REVIEW_AUDIO_PACKET_COUNT
    ):
        raise ValueError(f"{owner} media contract changed: {payload}")
    return {
        "status": "passed",
        "video": video,
        "audio": audio,
        "format": payload.get("format", {}),
    }


def _audio_packet_sha256(path: Path) -> str:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-c",
        "copy",
        "-f",
        "streamhash",
        "-hash",
        "sha256",
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    prefix = "0,a,SHA256="
    if len(lines) != 1 or not lines[0].startswith(prefix):
        raise RuntimeError(f"unexpected audio streamhash output for {path}: {lines}")
    digest = lines[0][len(prefix) :].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"invalid audio streamhash digest for {path}: {digest}")
    return digest


def probe_habitat_review_video(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Habitat review is missing or not a direct file: {path}")
    return {
        **_file_record(path),
        "readback": validate_review_probe_payload(
            _probe_av(path),
            expected_width=HABITAT_REVIEW_WIDTH,
            expected_height=HABITAT_REVIEW_HEIGHT,
            owner="Habitat main+topdown review",
        ),
        "panel_contract": {
            "left_half": "habitat_main",
            "right_half": "habitat_topdown_qa_only",
        },
        "audio_semantics": "Habitat 2ch binaural mixture (L,R)",
        "audio_packet_sha256": _audio_packet_sha256(path),
    }


def build_triptych_ffmpeg_command(
    *,
    ue_video_path: Path,
    habitat_review_path: Path,
    output_path: Path,
) -> list[str]:
    """Build the fixed, testable triptych encode and audio-copy command."""

    filter_graph = (
        "[0:v:0]scale=640:480:flags=lanczos,setsar=1[ue];"
        "[1:v:0]setpts=PTS-STARTPTS,setsar=1[habitat];"
        "[ue][habitat]hstack=inputs=2[comparison]"
    )
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(ue_video_path),
        "-i",
        str(habitat_review_path),
        "-filter_complex",
        filter_graph,
        "-map",
        "[comparison]",
        "-map",
        "1:a:0",
        "-frames:v",
        str(FRAME_COUNT),
        "-r",
        str(FPS),
        # Ubuntu 22.04 ships FFmpeg 4.4, which predates -fps_mode.  -vsync cfr
        # is the equivalent output contract on that supported host version.
        "-vsync",
        "cfr",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-c:a",
        "copy",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def probe_comparison_video(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"comparison video is missing or not a direct file: {path}")
    return {
        **_file_record(path),
        "readback": validate_review_probe_payload(
            _probe_av(path),
            expected_width=TRIPTYCH_WIDTH,
            expected_height=TRIPTYCH_HEIGHT,
            owner="UE/Habitat triptych comparison",
        ),
        "panels": triptych_panel_definition(),
        "audio_source": "byte-stream copied from Habitat review audio stream 0",
        "audio_packet_sha256": _audio_packet_sha256(path),
    }


def _configure_mp3d_instance(
    *,
    rpc_port: int,
    graphics_adapter: int | None,
    executable: str,
    parallel_instance_settings: Any,
) -> Any:
    """Configure an isolated fixed-step Entry-map process from HEAD helpers."""

    import spear

    settings = parallel_instance_settings(
        rpc_port, graphics_adapter=graphics_adapter
    )
    config = spear.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.GAME_EXECUTABLE = executable
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = (
        "/Engine/Maps/Entry"
    )
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = settings["rpc_port"]
    config.SPEAR.INSTANCE.TEMP_DIR = settings["temp_dir"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log = settings["log"]
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = settings[
        "shared_memory_initial_unique_id"
    ]
    if settings["graphics_adapter"] is not None:
        config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.graphicsadapter = settings[
            "graphics_adapter"
        ]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.renderoffscreen = None
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME = 1.0 / FPS
    config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = (
        "/etc/vulkan/icd.d/nvidia_icd.json"
    )
    config.freeze()
    spear.configure_system(config=config)
    return spear.Instance(config=config)


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to replace output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir()

    configuration = validate_configuration(default_configuration())
    import_manifest_path = args.ue_import_manifest.resolve()
    import_manifest = load_and_validate_ue_import_manifest(import_manifest_path)
    habitat_gate_path = args.habitat_gate_evidence.resolve()
    habitat_gate = validate_habitat_navmesh_authority_payload(
        _load_json(habitat_gate_path, "Habitat MP3D gate evidence")
    )
    habitat_review_path = args.habitat_review_video.resolve()
    habitat_review_evidence = probe_habitat_review_video(habitat_review_path)
    habitat_delivery_path = args.habitat_delivery_evidence.resolve()
    habitat_delivery_binding = load_and_validate_habitat_delivery(
        delivery_path=habitat_delivery_path,
        gate_path=habitat_gate_path,
        review_video_path=habitat_review_path,
    )
    human_manifest_path = args.human_ue_manifest.resolve()
    human_manifest = validate_human_ue_manifest_payload(
        _load_json(human_manifest_path, "Rocketbox human UE manifest")
    )
    human_source_path = Path(human_manifest["source_glb"]).resolve()
    if (
        not human_source_path.is_file()
        or human_source_path.is_symlink()
        or _sha256(human_source_path) != EXPECTED_HUMAN_SHA256
    ):
        raise ValueError("Rocketbox human runtime source authentication failed")

    sys.path.insert(0, str(EXAMPLES_DIR))
    from render_in_apartment import (
        parallel_instance_settings,
        read_frame,
        spawn_camera,
    )
    from render_in_gpurir_room import (
        SPEARSIM_EXECUTABLE,
        spawn_directional_light,
        spawn_sky,
    )

    instance = _configure_mp3d_instance(
        rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter,
        executable=SPEARSIM_EXECUTABLE,
        parallel_instance_settings=parallel_instance_settings,
    )
    game = instance.get_game()
    human_readbacks: list[dict[str, Any]] = []
    dog_readbacks: list[dict[str, Any]] = []
    camera_readbacks: list[dict[str, Any]] = []
    human_animation_readbacks: list[dict[str, Any]] = []
    dog_animation_readbacks: list[dict[str, Any]] = []
    lighting_evidence: dict[str, Any]
    try:
        with instance.begin_frame():
            room_actors, runtime_scene_bounds = _spawn_scene_meshes(
                game, import_manifest["scene_content"]["static_meshes"]
            )
            lighting_evidence = _spawn_lighting(
                game, spawn_sky, spawn_directional_light
            )
            camera, capture = spawn_camera(game=game, width=WIDTH, height=HEIGHT)
            capture.set_property_value(property_name="FOVAngle", property_value=HFOV_DEG)
            human_route = configuration["actors"]["human0"]["route_ue_cm"]
            dog_route = configuration["actors"]["dog0"]["route_ue_cm"]
            (
                human_actor,
                human_component,
                human_play_rate,
                human_animation_length,
            ) = _spawn_skeletal_actor(
                game=game,
                blueprint_class_path=HUMAN_BP_CLASS_PATH,
                animation_path=human_manifest["content"]["animations"]["Walking"],
                position_cm=human_route[0],
                yaw_deg=HUMAN_ACTOR_YAW_UE_DEG,
                scale=1.0,
            )
            (
                dog_actor,
                dog_component,
                dog_play_rate,
                dog_animation_length,
            ) = _spawn_skeletal_actor(
                game=game,
                blueprint_class_path=EXACT_BEAGLE_BP_CLASS_PATH,
                animation_path=import_manifest["m2_beagle"]["content"][
                    "animations"
                ]["Walking"],
                position_cm=dog_route[0],
                yaw_deg=DOG_ACTOR_YAW_UE_DEG,
                scale=1.0,
            )
            camera.K2_SetActorLocationAndRotation(
                NewLocation={
                    "X": CAMERA_UE_CM[0],
                    "Y": CAMERA_UE_CM[1],
                    "Z": CAMERA_UE_CM[2],
                },
                NewRotation={
                    "Roll": 0.0,
                    "Pitch": 0.0,
                    "Yaw": CAMERA_UE_YAW_DEG,
                },
                bSweep=False,
                bTeleport=True,
            )
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        if len(room_actors) != 71:
            raise RuntimeError("runtime MP3D scene mesh count changed")
        instance.step(num_frames=configuration["render"]["streaming_warmup_frames"])
        with instance.begin_frame():
            camera.K2_SetActorLocationAndRotation(
                NewLocation={
                    "X": CAMERA_UE_CM[0],
                    "Y": CAMERA_UE_CM[1],
                    "Z": CAMERA_UE_CM[2],
                },
                NewRotation={
                    "Roll": 0.0,
                    "Pitch": 0.0,
                    "Yaw": CAMERA_UE_YAW_DEG,
                },
                bSweep=False,
                bTeleport=True,
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=configuration["render"]["camera_warmup_frames"])

        maximum_human_phase = max(
            animation_phase_seconds("human0", frame) for frame in range(FRAME_COUNT)
        )
        maximum_dog_phase = max(
            animation_phase_seconds("dog0", frame) for frame in range(FRAME_COUNT)
        )
        if (
            maximum_human_phase > human_animation_length + 1.0e-6
            or maximum_dog_phase > dog_animation_length + 1.0e-6
        ):
            raise RuntimeError(
                "UE Walking animation is shorter than its Habitat phase contract"
            )
        with instance.begin_frame():
            # Warmup must not choose the formal frame-0 phase.  Stop automatic
            # advancement; the capture loop applies each exact Habitat sample.
            human_component.Stop()
            dog_component.Stop()
            camera_fov_readback = float(
                capture.get_property_value(property_name="FOVAngle")
            )
            if abs(camera_fov_readback - HFOV_DEG) > FOV_READBACK_TOLERANCE_DEG:
                raise RuntimeError(
                    "UE camera HFOV readback failed: "
                    f"requested={HFOV_DEG} observed={camera_fov_readback}"
                )
        with instance.end_frame():
            pass

        import cv2

        for frame_index in range(FRAME_COUNT):
            human_position = human_route[frame_index]
            dog_position = dog_route[frame_index]
            with instance.begin_frame():
                human_actor.K2_SetActorLocationAndRotation(
                    NewLocation={
                        "X": human_position[0],
                        "Y": human_position[1],
                        "Z": human_position[2],
                    },
                    NewRotation={
                        "Roll": 0.0,
                        "Pitch": 0.0,
                        "Yaw": HUMAN_ACTOR_YAW_UE_DEG,
                    },
                    bSweep=False,
                    bTeleport=True,
                )
                dog_actor.K2_SetActorLocationAndRotation(
                    NewLocation={
                        "X": dog_position[0],
                        "Y": dog_position[1],
                        "Z": dog_position[2],
                    },
                    NewRotation={
                        "Roll": 0.0,
                        "Pitch": 0.0,
                        "Yaw": DOG_ACTOR_YAW_UE_DEG,
                    },
                    bSweep=False,
                    bTeleport=True,
                )
                human_animation_readbacks.append(
                    _set_animation_phase(
                        human_component,
                        actor_id="human0",
                        frame_index=frame_index,
                    )
                )
                dog_animation_readbacks.append(
                    _set_animation_phase(
                        dog_component,
                        actor_id="dog0",
                        frame_index=frame_index,
                    )
                )
                camera.K2_SetActorLocationAndRotation(
                    NewLocation={
                        "X": CAMERA_UE_CM[0],
                        "Y": CAMERA_UE_CM[1],
                        "Z": CAMERA_UE_CM[2],
                    },
                    NewRotation={
                        "Roll": 0.0,
                        "Pitch": 0.0,
                        "Yaw": CAMERA_UE_YAW_DEG,
                    },
                    bSweep=False,
                    bTeleport=True,
                )
                human_readbacks.append(_actor_readback(human_actor, frame_index))
                dog_readbacks.append(_actor_readback(dog_actor, frame_index))
                camera_readbacks.append(_actor_readback(camera, frame_index))
            with instance.end_frame():
                # Detach the image from SPEAR's shared-memory allocation before
                # the instance closes.  The copied pixels are identical, while
                # the allocation can be released without an exported view.
                frame = read_frame(capture).copy()
                frame_path = frames_dir / f"frame_{frame_index:04d}.png"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"failed to write UE frame {frame_path}")
            if frame_index % FPS == 0:
                print(f"[mp3d-ue] frame {frame_index}/{FRAME_COUNT}", flush=True)
    finally:
        instance.close(force=True)

    video_path = output_dir / "mp3d_17DRP5sb8fy_ue_view0.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(video_path),
        ],
        check=True,
    )
    human_gate = _readback_gate(
        human_readbacks,
        configuration["actors"]["human0"]["route_ue_cm"],
        HUMAN_ACTOR_YAW_UE_DEG,
    )
    dog_gate = _readback_gate(
        dog_readbacks,
        configuration["actors"]["dog0"]["route_ue_cm"],
        DOG_ACTOR_YAW_UE_DEG,
    )
    camera_gate = _readback_gate(
        camera_readbacks,
        [CAMERA_UE_CM] * FRAME_COUNT,
        CAMERA_UE_YAW_DEG,
    )
    human_animation_gate = _animation_phase_gate(
        human_animation_readbacks, actor_id="human0"
    )
    dog_animation_gate = _animation_phase_gate(
        dog_animation_readbacks, actor_id="dog0"
    )
    video_evidence = _probe_video(video_path)
    comparison_video_path = (
        output_dir
        / "mp3d_17DRP5sb8fy_ue_vs_habitat_triptych_binaural.mp4"
    )
    subprocess.run(
        build_triptych_ffmpeg_command(
            ue_video_path=video_path,
            habitat_review_path=habitat_review_path,
            output_path=comparison_video_path,
        ),
        check=True,
    )
    comparison_video_evidence = probe_comparison_video(comparison_video_path)
    if (
        comparison_video_evidence["audio_packet_sha256"]
        != habitat_review_evidence["audio_packet_sha256"]
    ):
        raise RuntimeError(
            "triptych remux changed or truncated the Habitat binaural packets"
        )
    evidence = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scene_id": SCENE_ID,
        "route_id": ROUTE_ID,
        "configuration": configuration,
        "inputs": {
            "ue_import_manifest": _file_record(import_manifest_path),
            "habitat_navmesh_authority": _file_record(habitat_gate_path),
            "habitat_delivery_evidence": _file_record(habitat_delivery_path),
            "habitat_capture_evidence": _file_record(
                habitat_delivery_binding["capture_path"]
            ),
            "habitat_review_video": habitat_review_evidence,
            "human_ue_manifest": _file_record(human_manifest_path),
            "exact_m2_beagle": import_manifest["m2_beagle"]["source"],
            "human_runtime": {
                "path": str(human_source_path),
                "sha256": EXPECTED_HUMAN_SHA256,
                "size_bytes": human_source_path.stat().st_size,
            },
        },
        "asset_binding": {
            "human": {
                "blueprint_class_path": HUMAN_BP_CLASS_PATH,
                "source_sha256": EXPECTED_HUMAN_SHA256,
                "walking_forward_yaw_offset_deg": (
                    HUMAN_WALKING_FORWARD_YAW_OFFSET_DEG
                ),
                "motion_yaw_ue_deg": ROUTE_MOTION_YAW_UE_DEG,
                "actor_yaw_ue_deg": HUMAN_ACTOR_YAW_UE_DEG,
                "binding_authority": (
                    "reload-verified native Rocketbox UE v3 import manifest"
                ),
            },
            "dog": {
                "blueprint_class_path": EXACT_BEAGLE_BP_CLASS_PATH,
                "source_sha256": EXPECTED_M2_BEAGLE_SHA256,
                "exact_habitat_m2_runtime": True,
                "anatomical_forward_axis_actor": "+X",
                "motion_yaw_ue_deg": ROUTE_MOTION_YAW_UE_DEG,
                "actor_yaw_ue_deg": DOG_ACTOR_YAW_UE_DEG,
                "binding_authority": (
                    "reload-verified exact M2 visual.glb UE import manifest"
                ),
            },
        },
        "runtime": {
            "map": "/Engine/Maps/Entry",
            "rpc_port": args.rpc_port,
            "graphics_adapter": args.graphics_adapter,
            "render_offscreen": True,
            "spawned_scene_mesh_count": 71,
            "scene_mesh_transform": {
                "location_cm": [0.0, 0.0, 0.0],
                "rotation_deg": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "scene_mesh_collision": "NoCollision",
            "scene_mesh_bounds": runtime_scene_bounds,
            "camera_horizontal_fov_deg": {
                "status": "passed",
                "requested": HFOV_DEG,
                "readback": camera_fov_readback,
                "tolerance": FOV_READBACK_TOLERANCE_DEG,
            },
            "human_animation_play_rate_readback": human_play_rate,
            "dog_animation_play_rate_readback": dog_play_rate,
            "human_animation_length_seconds": human_animation_length,
            "dog_animation_length_seconds": dog_animation_length,
            "animation_playback": {
                "automatic_advancement_during_formal_capture": False,
                "human0": human_animation_gate,
                "dog0": dog_animation_gate,
            },
            "lighting": lighting_evidence,
        },
        "navigation_authority": {
            "source_schema": habitat_gate["schema"],
            "source_status": habitat_gate["status"],
            "semantics": habitat_gate["pathfinder"][
                "center_navigation_semantics"
            ],
            "routes": {
                actor_id: habitat_gate["pathfinder"]["routes"][actor_id]
                for actor_id in ("human0", "dog0")
            },
            "ue_scene_collision_disabled": True,
            "ue_raytrace_performed": False,
            "full_body_or_obb_clearance_claim": False,
        },
        "root_readback": {
            "human0": human_gate,
            "dog0": dog_gate,
            "camera": camera_gate,
        },
        "videos": {
            "ue_main": video_evidence,
            "formal_triptych": comparison_video_evidence,
        },
        "comparison_contract": {
            "resolution": [TRIPTYCH_WIDTH, TRIPTYCH_HEIGHT],
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FPS,
            "duration_seconds": DURATION_SECONDS,
            "panels": triptych_panel_definition(),
            "audio": {
                "source": "Habitat review stream 1:a:0",
                "codec_operation": "copy/remux",
                "channels": ["left", "right"],
                "semantics": "binaural",
                "sample_rate_hz": REVIEW_AUDIO_SAMPLE_RATE_HZ,
                "not_visual_fov_gated": True,
            },
        },
        "claim_boundary": (
            "Fair visual-engine canary using the same MP3D source geometry, "
            "camera, actor-center routes, exact M2 Beagle source, and native "
            "Rocketbox human. Habitat Pathfinder proves actor centers only; "
            "this evidence does not claim full-body clearance, acoustic "
            "equivalence, unknown Matterport light reconstruction, or dataset "
            "admission."
        ),
    }
    evidence_path = output_dir / "evidence.json"
    _atomic_write_json(evidence_path, evidence)
    print(
        "MP3D_UE_COMPARISON_OK "
        f"video={comparison_video_path} evidence={evidence_path}",
        flush=True,
    )
    return evidence_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ue-import-manifest", type=Path, required=True)
    parser.add_argument("--habitat-gate-evidence", type=Path, required=True)
    parser.add_argument("--habitat-delivery-evidence", type=Path, required=True)
    parser.add_argument(
        "--habitat-review-video",
        type=Path,
        required=True,
        help=(
            "validated 1280x480 Habitat main+topdown MP4 whose 2ch binaural "
            "audio is copied into the formal triptych"
        ),
    )
    parser.add_argument(
        "--human-ue-manifest", type=Path, default=DEFAULT_HUMAN_UE_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39271)
    parser.add_argument("--graphics-adapter", type=int)
    args = parser.parse_args()
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024, 65535]")
    if args.graphics_adapter is not None and args.graphics_adapter < 0:
        parser.error("--graphics-adapter must be non-negative")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
