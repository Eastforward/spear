from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import struct
import subprocess
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "blender_render_tokenrig_human_review.py"
)
SPEC = importlib.util.spec_from_file_location("tokenrig_dynamic_renderer", SCRIPT)
assert SPEC and SPEC.loader
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, filename: str | None = None) -> dict:
    result = {"sha256": _sha(path), "size_bytes": path.stat().st_size}
    if filename is not None:
        result["filename"] = filename
    return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_glb(path: Path, action_name: str | None = None) -> None:
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "bone_0"}, {"name": "body", "mesh": 0, "skin": 0}],
        "meshes": [{"name": "body", "primitives": [{"attributes": {"POSITION": 0}}]}],
        "skins": [{"joints": [0]}],
    }
    if action_name is not None:
        document["animations"] = [{"name": action_name, "channels": [], "samplers": []}]
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    raw = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(payload))
    raw += struct.pack("<II", len(payload), 0x4E4F534A) + payload
    path.write_bytes(raw)


@pytest.fixture()
def upstream(tmp_path: Path) -> dict[str, Path]:
    static_dir = tmp_path / "static_audit_v1"
    static_dir.mkdir()
    bind = static_dir / "bind_pose.glb"
    _write_glb(bind)
    static_names = (
        "bind_front.png",
        "bind_back.png",
        "bind_side.png",
        "bind_top.png",
        "skeleton_overlay.png",
        "weights_contact.png",
        "texture_compare.png",
        "joint_hierarchy.txt",
    )
    for name in static_names:
        (static_dir / name).write_bytes((name + " evidence").encode())
    artifacts = {
        "bind_pose.glb": {
            "filename": "bind_pose.glb",
            **_record(bind),
        }
    }
    for name in static_names:
        artifacts[name] = {"filename": name, **_record(static_dir / name)}
    static_qa = static_dir / "static_qa.json"
    _write_json(
        static_qa,
        {
            "schema": "tokenrig_human_static_qa_v1",
            "asset_id": "person_01",
            "decision": "automatic_static_checks_passed",
            "agent_qa_status": "pending_agent_visual_qa",
            "user_acceptance": "pending_user_review",
            "checks": {
                "axis_canonicalization": {
                    "canonical_front": "negative-y",
                    "transform_count": 1,
                },
                "grounding": {
                    "canonical_floor_z": 0.0,
                    "post_floor_z": 0.0,
                    "transform_count": 1,
                },
                "exported_pbr": {"passed": True},
                "semantic_mapping": {
                    "method": "test_fixture_v1",
                    "semantic_bones": {
                        "pelvis": "bone_0",
                        "head": "bone_5",
                        "left_hand": "bone_9",
                        "right_hand": "bone_28",
                        "left_foot": "bone_46",
                        "left_toe": "bone_47",
                        "right_foot": "bone_50",
                        "right_toe": "bone_51",
                    },
                },
            },
            "artifacts": artifacts,
        },
    )

    retarget_dir = tmp_path / "retarget_v1"
    retarget_dir.mkdir()
    walking = retarget_dir / "walking.glb"
    idle = retarget_dir / "standing_idle.glb"
    _write_glb(walking, "Walking")
    _write_glb(idle, "Standing_Idle")
    metrics = retarget_dir / "retarget_metrics.json"
    _write_json(
        metrics,
        {
            "schema": "tokenrig_rocketbox_retarget_metrics_v1",
            "automatic_checks": "passed",
            "actions": {
                "Walking": {"action_name": "Walking"},
                "Standing_Idle": {"action_name": "Standing_Idle"},
            },
        },
    )
    retarget = retarget_dir / "retarget_manifest.json"
    _write_json(
        retarget,
        {
            "schema": "tokenrig_rocketbox_retarget_v1",
            "asset_id": "person_01",
            "state_classification": "research_candidate",
            "canonical_front": "negative-y",
            "canonical_up": "positive-z",
            "automatic_checks": "passed",
            "user_acceptance": "pending_user_review",
            "environment": {"blender_version": "4.2.1", "fps": 30},
            "authenticated_inputs": {
                "static": {
                    "bind_pose": _record(bind),
                    "static_qa": _record(static_qa),
                    "floor_z_m": 0.0,
                    "axis_map_3x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                }
            },
            "actions": {
                "Walking": {"status": "passed", "action_name": "Walking"},
                "Standing_Idle": {"status": "passed", "action_name": "Standing_Idle"},
            },
            "artifacts": {
                "walking.glb": _record(walking, filename=None) | {"path": "walking.glb"},
                "standing_idle.glb": _record(idle, filename=None) | {"path": "standing_idle.glb"},
                "retarget_metrics.json": _record(metrics, filename=None) | {"path": "retarget_metrics.json"},
            },
        },
    )
    return {
        "static_qa": static_qa,
        "retarget_manifest": retarget,
        "walking": walking,
        "standing_idle": idle,
        "bind_pose": bind,
    }


def _authenticate(upstream: dict[str, Path]):
    return renderer.authenticate_review_inputs(
        asset_id="person_01",
        static_qa_json=upstream["static_qa"],
        retarget_manifest=upstream["retarget_manifest"],
        walking_glb=upstream["walking"],
        standing_idle_glb=upstream["standing_idle"],
    )


def test_module_is_importable_without_blender_and_defines_exact_media_matrix():
    assert renderer.MOTIONS == {"walking": "Walking", "standing_idle": "Standing_Idle"}
    assert renderer.VIEWS == ("front", "side", "top", "feet", "skeleton")
    assert renderer.VIDEO_SIZE == (1280, 720)
    assert renderer.FPS == 30


def test_authenticates_static_retarget_glbs_and_all_static_evidence(upstream):
    authenticated = _authenticate(upstream)
    assert authenticated["asset_id"] == "person_01"
    assert authenticated["static_qa"]["sha256"] == _sha(upstream["static_qa"])
    assert authenticated["retarget_manifest"]["sha256"] == _sha(upstream["retarget_manifest"])
    assert authenticated["glbs"]["walking"]["sha256"] == _sha(upstream["walking"])
    assert set(authenticated["static_evidence"]) == {
        "bind_front.png",
        "bind_back.png",
        "bind_side.png",
        "bind_top.png",
        "skeleton_overlay.png",
        "weights_contact.png",
        "texture_compare.png",
        "joint_hierarchy.txt",
    }


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("schema", "static QA schema"),
        ("decision", "static QA decision"),
        ("asset_id", "static QA asset_id"),
    ],
)
def test_rejects_unpassed_or_mismatched_static_gate(upstream, field, message):
    payload = json.loads(upstream["static_qa"].read_text())
    payload[field] = "wrong"
    _write_json(upstream["static_qa"], payload)
    with pytest.raises(renderer.ReviewRenderError, match=message):
        _authenticate(upstream)


def test_rejects_static_evidence_tamper(upstream):
    (upstream["static_qa"].parent / "texture_compare.png").write_bytes(b"tampered")
    with pytest.raises(renderer.ReviewRenderError, match="texture_compare.png SHA-256"):
        _authenticate(upstream)


def test_rejects_retarget_that_does_not_bind_exact_static_snapshot(upstream):
    payload = json.loads(upstream["retarget_manifest"].read_text())
    payload["authenticated_inputs"]["static"]["static_qa"]["sha256"] = "0" * 64
    _write_json(upstream["retarget_manifest"], payload)
    with pytest.raises(renderer.ReviewRenderError, match="retarget static QA hash"):
        _authenticate(upstream)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong", "retarget manifest schema"),
        ("automatic_checks", "failed", "retarget automatic checks"),
        ("canonical_front", "positive-y", "FRONT -Y"),
        ("canonical_up", "negative-z", r"UP \+Z"),
    ],
)
def test_rejects_bad_retarget_gate(upstream, field, value, message):
    payload = json.loads(upstream["retarget_manifest"].read_text())
    payload[field] = value
    _write_json(upstream["retarget_manifest"], payload)
    with pytest.raises(renderer.ReviewRenderError, match=message):
        _authenticate(upstream)


def test_rejects_glb_hash_tamper_even_when_it_still_parses(upstream):
    _write_glb(upstream["walking"], "WalkingChanged")
    with pytest.raises(renderer.ReviewRenderError, match="walking.glb SHA-256"):
        _authenticate(upstream)


def test_rejects_wrong_or_extra_animation_name(upstream):
    _write_glb(upstream["walking"], "NotWalking")
    payload = json.loads(upstream["retarget_manifest"].read_text())
    payload["artifacts"]["walking.glb"] = _record(upstream["walking"]) | {"path": "walking.glb"}
    _write_json(upstream["retarget_manifest"], payload)
    with pytest.raises(renderer.ReviewRenderError, match="exactly one Walking animation"):
        _authenticate(upstream)


def test_rejects_symlinked_upstream_artifact(upstream):
    real = upstream["walking"].with_suffix(".real.glb")
    upstream["walking"].rename(real)
    upstream["walking"].symlink_to(real.name)
    with pytest.raises(renderer.ReviewRenderError, match="direct regular file"):
        _authenticate(upstream)


def test_validate_ffprobe_payload_requires_exact_codec_geometry_fps_frames_and_duration():
    payload = {
        "streams": [
            {
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "nb_frames": "61",
                "nb_read_frames": "61",
                "duration": str(61 / 30),
            }
        ]
    }
    result = renderer.validate_ffprobe_payload(payload, expected_frame_count=61)
    assert result["frame_count"] == 61
    assert result["duration_s"] == pytest.approx(61 / 30)
    for field, bad in (("codec_name", "vp9"), ("width", 640), ("r_frame_rate", "24/1"), ("avg_frame_rate", "24/1"), ("nb_frames", "60")):
        changed = json.loads(json.dumps(payload))
        changed["streams"][0][field] = bad
        with pytest.raises(renderer.ReviewRenderError):
            renderer.validate_ffprobe_payload(changed, expected_frame_count=61)
    duplicate = json.loads(json.dumps(payload))
    duplicate["streams"].append(dict(duplicate["streams"][0]))
    with pytest.raises(renderer.ReviewRenderError, match="exactly one video stream"):
        renderer.validate_ffprobe_payload(duplicate, expected_frame_count=61)


def test_validate_luma_ranges_requires_every_decoded_frame_and_nonblank_signal():
    result = renderer.validate_luma_ranges([(10.0, 220.0)] * 9, expected_frame_count=9)
    assert result["decoded_frame_count"] == 9
    assert result["minimum_luma_span"] == 210.0
    with pytest.raises(renderer.ReviewRenderError, match="decoded frame count"):
        renderer.validate_luma_ranges([(0.0, 255.0)], expected_frame_count=9)
    with pytest.raises(renderer.ReviewRenderError, match="blank or nearly blank"):
        renderer.validate_luma_ranges([(10.0, 220.0)] * 8 + [(30.0, 32.0)], expected_frame_count=9)


def _gray_frame(width: int, height: int, *, subject_x: int | None, border_value: int = 20) -> bytes:
    pixels = bytearray([border_value] * (width * height))
    if subject_x is not None:
        for y in range(1, height - 1):
            pixels[y * width + subject_x] = 210
            pixels[y * width + min(subject_x + 1, width - 2)] = 150
    return bytes(pixels)


def test_temporal_gate_rejects_nonblank_but_identical_walking_and_idle_frames():
    frames = [_gray_frame(12, 8, subject_x=5)] * 20
    for motion in renderer.MOTIONS:
        with pytest.raises(renderer.ReviewRenderError, match="temporally frozen"):
            renderer.validate_temporal_frames(
                frames,
                width=12,
                height=8,
                expected_frame_count=20,
                motion=motion,
                view="front",
            )


def test_ffmpeg_identical_nonblank_mp4_is_rejected_by_real_probe(tmp_path):
    output = tmp_path / "frozen_nonblank.mp4"
    command = [
        "ffmpeg",
        "-v", "error",
        "-f", "lavfi",
        "-i", "color=c=gray:s=1280x720:r=30:d=1,drawbox=x=320:y=120:w=320:h=480:color=white:t=fill",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-frames:v", "30",
        str(output),
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    with pytest.raises(renderer.ReviewRenderError, match="temporally frozen"):
        renderer.probe_video(
            output,
            expected_frame_count=30,
            motion="walking",
            view="front",
        )


def test_temporal_gate_requires_subject_roi_motion_not_only_border_flicker():
    frames = []
    for index in range(20):
        pixels = bytearray(_gray_frame(12, 8, subject_x=5, border_value=20))
        for x in range(12):
            pixels[x] = 20 + index
            pixels[(8 - 1) * 12 + x] = 20 + index
        frames.append(bytes(pixels))
    with pytest.raises(renderer.ReviewRenderError, match="subject ROI has no motion"):
        renderer.validate_temporal_frames(
            frames,
            width=12,
            height=8,
            expected_frame_count=20,
            motion="standing_idle",
            view="front",
        )


def test_temporal_gate_passes_idle_and_skeleton_motion_with_unique_frame_evidence():
    frames = [_gray_frame(12, 8, subject_x=3 + index % 5) for index in range(20)]
    result = renderer.validate_temporal_frames(
        frames,
        width=12,
        height=8,
        expected_frame_count=20,
        motion="standing_idle",
        view="skeleton",
    )
    assert result["unique_frame_count"] >= 5
    assert result["maximum_subject_roi_delta"] > 0.0
    assert result["skeleton_motion_passed"] is True


def test_walking_temporal_gate_rejects_two_frame_toggle_as_low_unique_motion():
    two = [_gray_frame(12, 8, subject_x=4), _gray_frame(12, 8, subject_x=6)]
    with pytest.raises(renderer.ReviewRenderError, match="unique-frame ratio"):
        renderer.validate_temporal_frames(
            [two[index % 2] for index in range(30)],
            width=12,
            height=8,
            expected_frame_count=30,
            motion="walking",
            view="side",
        )


def _rgb_skeleton_frame(width: int, height: int, *, skeleton_x: int, body_value: int) -> bytes:
    pixels = bytearray([body_value, body_value, body_value] * (width * height))
    for y in range(1, height - 1):
        index = (y * width + skeleton_x) * 3
        pixels[index : index + 3] = bytes((5, 245, 220))
    return bytes(pixels)


def test_skeleton_overlay_motion_gate_rejects_frozen_cyan_skeleton_over_moving_body():
    frames = [
        _rgb_skeleton_frame(12, 8, skeleton_x=5, body_value=20 + index)
        for index in range(20)
    ]
    with pytest.raises(renderer.ReviewRenderError, match="cyan skeleton overlay is frozen"):
        renderer.validate_skeleton_overlay_motion(
            frames, width=12, height=8, expected_frame_count=20
        )


def test_skeleton_overlay_motion_gate_tracks_cyan_mask_and_centroid():
    frames = [
        _rgb_skeleton_frame(12, 8, skeleton_x=3 + index % 5, body_value=20)
        for index in range(20)
    ]
    result = renderer.validate_skeleton_overlay_motion(
        frames, width=12, height=8, expected_frame_count=20
    )
    assert result["minimum_cyan_pixels"] >= 6
    assert result["unique_cyan_mask_count"] >= 5
    assert result["maximum_centroid_displacement_px"] >= 1.0


def test_validate_destination_is_no_replace_and_rejects_symlinked_parent(tmp_path):
    destination = tmp_path / "dynamic_review_v1"
    assert renderer.validate_destination(destination) == destination.absolute()
    destination.mkdir()
    with pytest.raises(renderer.ReviewRenderError, match="already exists"):
        renderer.validate_destination(destination)
    destination.rmdir()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    link_parent = tmp_path / "link"
    link_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(renderer.ReviewRenderError, match="real directory"):
        renderer.validate_destination(link_parent / "review")


def test_atomic_directory_publication_never_replaces_existing_bundle(tmp_path):
    source = tmp_path / ".review.first.staging"
    destination = tmp_path / "dynamic_review_v1"
    source.mkdir()
    (source / "sentinel").write_text("first", encoding="utf-8")
    renderer.rename_directory_noreplace(source, destination)
    assert (destination / "sentinel").read_text() == "first"
    second = tmp_path / ".review.second.staging"
    second.mkdir()
    (second / "sentinel").write_text("second", encoding="utf-8")
    with pytest.raises(renderer.ReviewRenderError, match="already exists"):
        renderer.rename_directory_noreplace(second, destination)
    assert (destination / "sentinel").read_text() == "first"


def test_failure_evidence_is_exclusive_rejected_and_preserves_authenticated_snapshot(tmp_path):
    destination = tmp_path / "dynamic_review_v1"
    authenticated = {"static_qa": {"sha256": "a" * 64}}
    first = renderer.write_failure_evidence(
        destination=destination,
        asset_id="person_01",
        error=RuntimeError("render failed"),
        authenticated=authenticated,
    )
    second = renderer.write_failure_evidence(
        destination=destination,
        asset_id="person_01",
        error=RuntimeError("second failure"),
        authenticated=authenticated,
    )
    assert first != second
    payload = json.loads(first.read_text())
    assert payload["decision"] == "rejected"
    assert payload["authenticated_inputs"] == authenticated
    assert "render failed" in payload["error"]
    assert first.stat().st_mode & 0o777 == 0o444
    assert second.stat().st_mode & 0o777 == 0o444
    assert not any(path.name.endswith(".tmp") for path in tmp_path.iterdir())


def test_build_review_manifest_never_claims_agent_or_user_approval(upstream):
    authenticated = _authenticate(upstream)
    execution = renderer.authenticate_execution_environment()
    media = {}
    for motion in renderer.MOTIONS:
        media[motion] = {}
        for view in renderer.VIEWS:
            media[motion][view] = {
                "mp4": {"filename": f"{motion}_{view}.mp4", "sha256": "1" * 64, "size_bytes": 1},
                "png": {"filename": f"{motion}_{view}.png", "sha256": "2" * 64, "size_bytes": 1},
            }
    manifest = renderer.build_review_manifest(
        asset_id="person_01",
        display_label="Person 01",
        instance_kind="male_canary",
        authenticated=authenticated,
        actions={
            "walking": {"action_name": "Walking", "frame_start": 1, "frame_end": 61, "frame_count": 61},
            "standing_idle": {"action_name": "Standing_Idle", "frame_start": 1, "frame_end": 31, "frame_count": 31},
        },
        media=media,
        media_qa_record={"filename": "media_qa.json", "sha256": "3" * 64, "size_bytes": 1},
        execution=execution,
        blender_version="4.2.1",
        command=["blender", "--background"],
    )
    encoded = json.dumps(manifest)
    assert manifest["agent_visual_qa"] == "pending_agent_visual_qa"
    assert manifest["user_acceptance"] == "pending_user_review"
    assert "user_approved" not in encoded
    assert set(manifest["actions"]) == {"walking", "standing_idle"}
    assert all(set(action["views"]) == set(renderer.VIEWS) for action in manifest["actions"].values())
    assert manifest["execution"]["renderer"]["sha256"] == _sha(renderer.SCRIPT_PATH)
    assert manifest["execution"]["ffmpeg"]["version"]
    assert manifest["execution"]["ffprobe"]["version"]


def test_execution_environment_pins_renderer_and_media_binary_bytes_and_versions():
    execution = renderer.authenticate_execution_environment()
    assert execution["renderer"]["path"] == str(renderer.SCRIPT_PATH)
    assert execution["renderer"]["sha256"] == _sha(renderer.SCRIPT_PATH)
    for name in ("ffmpeg", "ffprobe"):
        descriptor = execution[name]
        path = Path(descriptor["path"])
        assert path.is_file() and not path.is_symlink() and path.resolve() == path
        assert descriptor["sha256"] == _sha(path)
        assert descriptor["size_bytes"] == path.stat().st_size
        assert descriptor["version"].lower().startswith(name)


def test_execution_contract_rejects_renderer_or_tool_change_during_render():
    before = renderer.authenticate_execution_environment()
    after = json.loads(json.dumps(before))
    after["renderer"]["sha256"] = "0" * 64
    with pytest.raises(renderer.ReviewRenderError, match="changed during rendering"):
        renderer.assert_execution_unchanged(before, after)


def test_published_media_descriptor_never_contains_staging_path(tmp_path):
    staging = tmp_path / ".dynamic_review_v1.deadbeef.staging"
    staging.mkdir()
    media = staging / "walking_front.mp4"
    media.write_bytes(b"video")
    record = renderer.file_record(media, filename=media.name)
    assert record == {
        "filename": "walking_front.mp4",
        "sha256": _sha(media),
        "size_bytes": 5,
    }
    assert ".staging" not in json.dumps(record)


def test_blender_runtime_is_pinned_to_4_2_lts():
    assert renderer.validate_blender_version((4, 2, 1)) == "4.2.1"
    with pytest.raises(renderer.ReviewRenderError, match="Blender 4.2"):
        renderer.validate_blender_version((5, 0, 0))


def _animated_bounds():
    return [
        {
            "minimum": (-0.9, -0.35, 0.0),
            "maximum": (0.9, 0.35, 2.2),
            "center": (0.0, 0.0, 1.1),
        },
        {
            "minimum": (-0.7, -1.25, -0.01),
            "maximum": (1.1, -0.55, 2.25),
            "center": (0.2, -0.9, 1.12),
        },
    ]


@pytest.mark.parametrize("view", renderer.VIEWS)
def test_camera_plan_projects_every_animated_bound_corner_inside_safe_ndc(view):
    margin = 0.08
    plans = renderer.plan_camera_keyframes(
        view=view,
        bounds=_animated_bounds(),
        angle_x=1.05,
        angle_y=0.68,
        ndc_margin=margin,
        clip_start=0.05,
    )
    assert len(plans) == 2
    assert plans[0]["target"] != plans[1]["target"]
    for sample, plan in zip(_animated_bounds(), plans):
        projected = renderer.project_bound_corners_to_ndc(
            sample,
            location=plan["location"],
            target=plan["target"],
            view=view,
            angle_x=1.05,
            angle_y=0.68,
        )
        assert len(projected) == 8
        assert all(margin <= point[0] <= 1.0 - margin for point in projected)
        assert all(margin <= point[1] <= 1.0 - margin for point in projected)
        assert all(point[2] > 0.05 for point in projected)


def test_camera_distance_uses_vertical_fov_not_old_fixed_height_multiplier():
    sample = _animated_bounds()[0]
    plans = renderer.plan_camera_keyframes(
        view="front",
        bounds=[sample],
        angle_x=1.05,
        angle_y=0.50,
        ndc_margin=0.08,
        clip_start=0.05,
    )
    body_height = sample["maximum"][2] - sample["minimum"][2]
    assert plans[0]["distance"] > body_height * 1.35
    assert plans[0]["angle_y"] == pytest.approx(0.50)


def test_feet_review_bounds_are_a_true_lower_leg_roi_not_full_body_bounds():
    samples = [
        [(0.12, -0.3, 0.08), (0.10, -0.18, 0.04), (-0.11, 0.1, 0.08), (-0.09, 0.22, 0.04)],
        [(0.11, -0.4, 0.09), (0.09, -0.28, 0.05), (-0.12, 0.0, 0.08), (-0.10, 0.12, 0.04)],
    ]
    bounds = renderer.build_feet_review_bounds(samples, body_height=0.80)
    assert len(bounds) == 2
    assert all(
        value["maximum"][2] - value["minimum"][2] == pytest.approx(0.256)
        for value in bounds
    )
    assert all(value["maximum"][2] < 0.30 for value in bounds)
    assert bounds[0]["minimum"][1] < -0.3
    assert bounds[0]["maximum"][1] > 0.22


def test_render_motion_sets_import_fps_normalizes_blender_suffix_and_uses_feet_roi():
    source = inspect.getsource(renderer._render_motion)
    assert source.index("scene.render.fps = FPS") < source.index("bpy.ops.import_scene.gltf")
    assert "foot_bounds = _foot_joint_bounds" in source
    assert "view_bounds=foot_bounds if view == \"feet\" else bounds" in source
    scene_objects = inspect.getsource(renderer._scene_objects)
    assert 'f"{expected_action}_Armature"' in scene_objects


def test_skeleton_review_hides_opaque_body_instead_of_occluding_cyan_rig():
    source = inspect.getsource(renderer._render_view)
    skeleton_branch = source[source.index('if view == "skeleton":') : source.index("png_path")]
    assert "mesh.hide_render = True" in skeleton_branch
    assert "_create_skeleton_body_copy" not in skeleton_branch


def test_skeleton_overlay_keyframes_pose_bone_cylinders_without_skin_modifier():
    source = inspect.getsource(renderer._create_skeleton_overlay)
    assert "bpy.ops.mesh.primitive_cylinder_add" in source
    assert "pose_bone.head" in source
    assert 'keyframe_insert(data_path="location"' in source
    assert 'keyframe_insert(data_path="rotation_quaternion"' in source
    assert 'modifiers.new("Skin", "SKIN")' not in source


def test_blender_camera_projection_smoke_keeps_all_animated_corners_in_margin():
    blender = Path("/data/jzy/blender/blender-4.2.1-linux-x64/blender")
    expression = """
import bpy, importlib.util, itertools, pathlib
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
p=pathlib.Path('/data/jzy/code/AVEngine/external/SPEAR/tools/blender_render_tokenrig_human_review.py')
s=importlib.util.spec_from_file_location('review_renderer',p)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
scene=bpy.context.scene
scene.frame_start=1
scene.frame_end=2
scene.render.resolution_x=1280
scene.render.resolution_y=720
bounds=[
 {'minimum':(-0.9,-0.35,0.0),'maximum':(0.9,0.35,2.2),'center':(0,0,1.1)},
 {'minimum':(-0.7,-1.25,-0.01),'maximum':(1.1,-0.55,2.25),'center':(0.2,-0.9,1.12)},
]
for view in m.VIEWS:
 camera=m._create_camera(bpy,view,bounds,2.26)
 for frame,sample in enumerate(bounds,start=1):
  scene.frame_set(frame)
  bpy.context.view_layer.update()
  corners=itertools.product(*zip(sample['minimum'],sample['maximum']))
  for corner in corners:
   ndc=world_to_camera_view(scene,camera,Vector(corner))
   assert 0.079 <= ndc.x <= 0.921,(view,frame,corner,tuple(ndc))
   assert 0.079 <= ndc.y <= 0.921,(view,frame,corner,tuple(ndc))
   assert ndc.z > camera.data.clip_start,(view,frame,corner,tuple(ndc))
 bpy.data.objects.remove(camera,do_unlink=True)
print('TOKENRIG_CAMERA_PROJECTION_SMOKE_OK')
"""
    result = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-expr", expression],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert "TOKENRIG_CAMERA_PROJECTION_SMOKE_OK" in result.stdout, result.stdout


def test_source_contains_real_blender_render_skeleton_and_atomic_failure_paths():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import bpy" not in "\n".join(source.splitlines()[:40])
    for token in (
        "BLENDER_EEVEE_NEXT",
        "bpy.ops.import_scene.gltf",
        "bpy.ops.render.render(animation=True)",
        "Armature",
        "primitive_cylinder_add",
        "renameat2",
        "render_failed_attempt",
        "ffprobe",
        "signalstats",
        "pose_bone.tail",
    ):
        assert token in source


def test_skeleton_overlay_refreshes_pose_before_sampling_bone_endpoints():
    source = inspect.getsource(renderer._create_skeleton_overlay)
    assert "scene.frame_set(frame)\n        bpy.context.view_layer.update()" in source


def test_blender_linked_alpha_skeleton_copy_preserves_original_pbr_graph(tmp_path):
    blender = Path("/data/jzy/blender/blender-4.2.1-linux-x64/blender")
    expression = """
import bpy, importlib.util, pathlib
p=pathlib.Path('/data/jzy/code/AVEngine/external/SPEAR/tools/blender_render_tokenrig_human_review.py')
s=importlib.util.spec_from_file_location('review_renderer',p)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
mesh_data=bpy.data.meshes.new('BodyMesh')
mesh_data.from_pydata([(0,0,0),(1,0,0),(0,1,0)],[],[(0,1,2)])
body=bpy.data.objects.new('Body',mesh_data)
bpy.context.scene.collection.objects.link(body)
material=bpy.data.materials.new('LinkedPBR')
material.use_nodes=True
bsdf=material.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Alpha'].default_value=0.73
image=bpy.data.images.new('LinkedAlphaImage',width=1,height=1,alpha=True)
image.pixels[:]=(0.2,0.7,0.4,0.91)
texture=material.node_tree.nodes.new('ShaderNodeTexImage')
texture.name='LinkedAlphaTexture'
texture.image=image
material.node_tree.links.new(texture.outputs['Alpha'],bsdf.inputs['Alpha'])
assert bsdf.inputs['Alpha'].is_linked
mesh_data.materials.append(material)
mesh_data.materials.append(material)
before=m.material_graph_hash(body)
clone, owned=m._create_skeleton_body_copy(bpy,body)
m._set_copied_body_transparency(clone)
after=m.material_graph_hash(body)
assert before == after
assert clone.data.materials[0] is clone.data.materials[1]
assert clone.data.materials[0] is not material
assert abs(bsdf.inputs['Alpha'].default_value-0.73) < 1e-6
assert bsdf.inputs['Alpha'].is_linked
clone_bsdf=clone.data.materials[0].node_tree.nodes.get('Principled BSDF')
assert abs(clone_bsdf.inputs['Alpha'].default_value-0.22) < 1e-6
assert not clone_bsdf.inputs['Alpha'].is_linked
transparency=m.validate_copied_body_transparency(clone)
assert transparency['passed'] is True
assert transparency['linked_alpha_inputs_removed'] == 1
assert transparency['effective_alpha'] == 0.22
assert transparency['surface_render_method'] == 'DITHERED'
m._delete_skeleton_body_copy(bpy,clone,owned)
assert m.material_graph_hash(body) == before
print('TOKENRIG_LINKED_ALPHA_SMOKE_OK',before)
"""
    result = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-expr", expression],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    assert "TOKENRIG_LINKED_ALPHA_SMOKE_OK" in result.stdout


def test_cli_requires_all_hash_locked_upstreams_and_metadata():
    args = renderer.parse_args(
        [
            "--asset-id", "person_01",
            "--display-label", "Person 01",
            "--instance-kind", "male_canary",
            "--static-qa-json", "/tmp/static_qa.json",
            "--retarget-manifest", "/tmp/retarget_manifest.json",
            "--walking-glb", "/tmp/walking.glb",
            "--standing-idle-glb", "/tmp/standing_idle.glb",
            "--output-dir", "/tmp/dynamic_review_v1",
        ]
    )
    assert args.asset_id == "person_01"
    assert args.instance_kind == "male_canary"
    assert args.output_dir == Path("/tmp/dynamic_review_v1")
