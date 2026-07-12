#!/usr/bin/env python3
"""Render a top-down facing/travel diagnostic for rejected retarget attempt two."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_DIR = SCRIPT_PATH.parent
SPIKE_DIR = TOOLS_DIR / "spike_rlr"
for directory in (TOOLS_DIR, SPIKE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import blender_render_tokenrig_human_review as review
from second_retarget_facing_review import (
    CANONICAL_FRONT,
    FacingReviewError,
    authenticate_second_attempt,
    build_review_html,
    compute_facing_samples,
    sha256_file,
)


FPS = 30
FRAME_COUNT = 33
VIDEO_SIZE = (640, 360)
OUTPUT_SCHEMA = "second_retarget_facing_review_render_v1"


class FacingRenderError(RuntimeError):
    """The derived facing render failed a pinned invariant."""


def validate_blender_version(version: Sequence[int]) -> str:
    value = tuple(int(component) for component in version[:3])
    if value != (4, 2, 1):
        raise FacingRenderError(f"Blender must be exactly 4.2.1, got {value}")
    return ".".join(str(component) for component in value)


def aggregate_top_bounds(
    bounds: Sequence[Mapping[str, Sequence[float]]], *, padding: float
) -> dict[str, tuple[float, float, float]]:
    if not bounds or not math.isfinite(padding) or padding < 0.0:
        raise FacingRenderError("top camera bounds/padding are invalid")
    minimum = tuple(
        min(float(sample["minimum"][axis]) for sample in bounds) - padding
        for axis in range(3)
    )
    maximum = tuple(
        max(float(sample["maximum"][axis]) for sample in bounds) + padding
        for axis in range(3)
    )
    if any(not math.isfinite(value) for value in minimum + maximum):
        raise FacingRenderError("top camera bounds contain a non-finite value")
    return {
        "minimum": minimum,
        "maximum": maximum,
        "center": tuple((minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _record(path: Path, *, filename: str | None = None) -> dict[str, Any]:
    result = {
        "path" if filename is None else "filename": (
            str(path.resolve()) if filename is None else filename
        ),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    return result


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _scene_objects(bpy: Any) -> tuple[Any, Any, Any]:
    armatures = [value for value in bpy.context.scene.objects if value.type == "ARMATURE"]
    meshes = [
        value
        for value in bpy.context.scene.objects
        if value.type == "MESH"
        and any(modifier.type == "ARMATURE" for modifier in value.modifiers)
    ]
    if len(armatures) != 1 or len(meshes) != 1:
        raise FacingRenderError("diagnostic import must contain one armature and one skinned mesh")
    armature, mesh = armatures[0], meshes[0]
    action = armature.animation_data.action if armature.animation_data else None
    if action is None or action.name not in {"Walking", "Walking_Armature"}:
        raise FacingRenderError("diagnostic import has no authenticated Walking armature action")
    if action.name == "Walking_Armature":
        action.name = "Walking"
    if action.name != "Walking":
        raise FacingRenderError("Walking action normalization failed")
    return armature, mesh, action


def _import_scene(bpy: Any, glb_path: Path) -> tuple[Any, Any, Any]:
    review._clear_scene(bpy)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    armature, mesh, action = _scene_objects(bpy)
    frame_start, frame_end = review._integer_frame_range(action)
    if (frame_start, frame_end, frame_end - frame_start + 1) != (1, 33, FRAME_COUNT):
        raise FacingRenderError(
            f"Walking must import as frames 1..33, got {frame_start}..{frame_end}"
        )
    scene.frame_start, scene.frame_end = frame_start, frame_end
    return armature, mesh, action


def _sample_semantic_points(
    bpy: Any,
    armature: Any,
    semantic_bones: Mapping[str, str],
    *,
    frame_start: int,
    frame_end: int,
) -> tuple[dict[str, tuple[float, float, float]], list[dict[str, tuple[float, float, float]]]]:
    names = {
        "pelvis": semantic_bones["pelvis"],
        "left_clavicle": semantic_bones["left_clavicle"],
        "right_clavicle": semantic_bones["right_clavicle"],
        "left_thigh": semantic_bones["left_thigh"],
        "right_thigh": semantic_bones["right_thigh"],
    }
    rest = {}
    for role, name in names.items():
        bone = armature.data.bones.get(name)
        if bone is None:
            raise FacingRenderError(f"rest semantic bone is missing: {role}={name}")
        rest[role] = tuple(armature.matrix_world @ bone.head_local)
    samples = []
    scene = bpy.context.scene
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        sample = {}
        for role, name in names.items():
            pose_bone = armature.pose.bones.get(name)
            if pose_bone is None:
                raise FacingRenderError(f"pose semantic bone is missing: {role}={name}")
            sample[role] = tuple(armature.matrix_world @ pose_bone.head)
        samples.append(sample)
    return rest, samples


def _emission_material(bpy: Any, name: str, color: Sequence[float]) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = tuple(float(value) for value in color) + (1.0,)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _create_arrow(bpy: Any, name: str, material: Any, width: float) -> Any:
    vertices = (
        (0.0, -width * 0.22, 0.0),
        (0.62, -width * 0.22, 0.0),
        (0.62, -width * 0.52, 0.0),
        (1.0, 0.0, 0.0),
        (0.62, width * 0.52, 0.0),
        (0.62, width * 0.22, 0.0),
        (0.0, width * 0.22, 0.0),
    )
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], [tuple(range(len(vertices)))])
    mesh.update()
    arrow = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(arrow)
    arrow.data.materials.append(material)
    return arrow


def _set_arrow(
    arrow: Any,
    *,
    origin: Sequence[float],
    direction: Sequence[float],
    length: float,
    z: float,
    frame: int,
) -> None:
    arrow.location = (float(origin[0]), float(origin[1]), float(z))
    arrow.rotation_euler = (0.0, 0.0, math.atan2(float(direction[1]), float(direction[0])))
    arrow.scale = (float(length), float(length), 1.0)
    arrow.keyframe_insert(data_path="location", frame=frame)
    arrow.keyframe_insert(data_path="rotation_euler", frame=frame)
    arrow.keyframe_insert(data_path="scale", frame=frame)


def _keyframe_direction_arrows(
    bpy: Any,
    metrics: Mapping[str, Any],
    *,
    overlay_z: float,
    arrow_length: float,
) -> dict[str, Any]:
    body = _create_arrow(
        bpy,
        "BodyForward_BLUE",
        _emission_material(bpy, "BodyForwardBlue", (0.0, 0.18, 0.9)),
        0.34,
    )
    travel = _create_arrow(
        bpy,
        "TravelDirection_RED",
        _emission_material(bpy, "TravelDirectionRed", (0.9, 0.02, 0.0)),
        0.34,
    )
    canonical = _create_arrow(
        bpy,
        "CanonicalFront_GREY",
        _emission_material(bpy, "CanonicalFrontGrey", (0.38, 0.38, 0.38)),
        0.34,
    )
    for sample in metrics["frames"]:
        frame = int(sample["frame"])
        pelvis = sample["pelvis_position"]
        right = sample["body_right"]
        body_origin = (
            pelvis[0] - right[0] * arrow_length * 0.18,
            pelvis[1] - right[1] * arrow_length * 0.18,
        )
        travel_origin = (
            pelvis[0] + right[0] * arrow_length * 0.18,
            pelvis[1] + right[1] * arrow_length * 0.18,
        )
        canonical_origin = (
            pelvis[0] + arrow_length * 0.75,
            pelvis[1] + arrow_length * 0.75,
        )
        _set_arrow(
            body,
            origin=body_origin,
            direction=sample["body_forward"],
            length=arrow_length,
            z=overlay_z,
            frame=frame,
        )
        travel.hide_render = sample["travel_direction"] is None
        if sample["travel_direction"] is not None:
            _set_arrow(
                travel,
                origin=travel_origin,
                direction=sample["travel_direction"],
                length=arrow_length,
                z=overlay_z + 0.003,
                frame=frame,
            )
        else:
            _set_arrow(
                travel,
                origin=travel_origin,
                direction=CANONICAL_FRONT,
                length=arrow_length,
                z=overlay_z + 0.003,
                frame=frame,
            )
        travel.keyframe_insert(data_path="hide_render", frame=frame)
        _set_arrow(
            canonical,
            origin=canonical_origin,
            direction=CANONICAL_FRONT,
            length=arrow_length * 0.72,
            z=overlay_z + 0.006,
            frame=frame,
        )
    for arrow in (body, travel, canonical):
        if arrow.animation_data and arrow.animation_data.action:
            for curve in arrow.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "CONSTANT" if curve.data_path == "hide_render" else "LINEAR"
    return {"body": body, "travel": travel, "canonical": canonical}


def _create_root_trail(
    bpy: Any, metrics: Mapping[str, Any], *, overlay_z: float, radius: float
) -> Any:
    curve = bpy.data.curves.new("PelvisRootTrailCurve", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    points = metrics["frames"]
    spline.points.add(len(points) - 1)
    for target, sample in zip(spline.points, points):
        pelvis = sample["pelvis_position"]
        target.co = (float(pelvis[0]), float(pelvis[1]), float(overlay_z), 1.0)
    trail = bpy.data.objects.new("PelvisRootTrail_YELLOW", curve)
    bpy.context.scene.collection.objects.link(trail)
    trail.data.materials.append(
        _emission_material(bpy, "PelvisRootTrailYellow", (0.95, 0.55, 0.0))
    )
    return trail


def _render_top(
    bpy: Any,
    *,
    staging: Path,
    mesh: Any,
    bounds: Sequence[Mapping[str, Any]],
    body_height: float,
    frame_start: int,
    frame_end: int,
) -> dict[str, Any]:
    aggregate = aggregate_top_bounds(bounds, padding=body_height * 0.20)
    fixed_bounds = [aggregate for _ in range(frame_start, frame_end + 1)]
    camera = review._create_camera(bpy, "top", fixed_bounds, body_height)
    scene = bpy.context.scene
    png_path = staging / "top_facing.png"
    mp4_path = staging / "top_facing.mp4"
    scene.frame_set(frame_start + (frame_end - frame_start) // 2)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(png_path)
    bpy.ops.render.render(write_still=True)
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(mp4_path)
    bpy.ops.render.render(animation=True)
    bpy.data.objects.remove(camera, do_unlink=True)
    return {
        "png": review.probe_png(png_path),
        "mp4": review.probe_video(
            mp4_path,
            expected_frame_count=frame_end - frame_start + 1,
            motion="walking",
            view="top",
        ),
        "fixed_camera_bounds": aggregate,
    }


def _render_bundle(
    bpy: Any,
    *,
    authenticated: Mapping[str, Any],
    staging: Path,
    blender_version: str,
    command: Sequence[str],
) -> None:
    glb_path = Path(authenticated["glb"]["path"])
    armature, mesh, action = _import_scene(bpy, glb_path)
    frame_start, frame_end = review._integer_frame_range(action)
    semantic_bones = authenticated["semantic_bones"]
    rest, samples = _sample_semantic_points(
        bpy,
        armature,
        semantic_bones,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    metrics = compute_facing_samples(rest, samples, fps=FPS)
    bounds = review._world_bounds(bpy, mesh, frame_start, frame_end)
    body_height = max(sample["maximum"][2] for sample in bounds) - min(
        sample["minimum"][2] for sample in bounds
    )
    if not math.isfinite(body_height) or body_height <= 0.2:
        raise FacingRenderError("animated body height is invalid")
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("SecondFacingDiagnosticWorld")
    review.VIDEO_SIZE = VIDEO_SIZE
    review._setup_scene(bpy, body_height)
    bpy.context.scene.eevee.taa_render_samples = 4
    bpy.context.scene.eevee.taa_samples = 4
    overlay_z = max(sample["maximum"][2] for sample in bounds) + body_height * 0.07
    _create_root_trail(
        bpy, metrics, overlay_z=overlay_z - 0.008, radius=body_height * 0.004
    )
    _keyframe_direction_arrows(
        bpy,
        metrics,
        overlay_z=overlay_z,
        arrow_length=body_height * 0.28,
    )
    media_qa = _render_top(
        bpy,
        staging=staging,
        mesh=mesh,
        bounds=bounds,
        body_height=body_height,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    metrics_path = staging / "facing_metrics.json"
    _write_json_exclusive(metrics_path, metrics)
    _write_bytes_exclusive(staging / "review.html", build_review_html(metrics))
    current = authenticate_second_attempt(Path(authenticated["manifest"]["path"]).parent)
    if current != authenticated:
        raise FacingRenderError("second-attempt source changed during facing render")
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "asset_id": authenticated["asset_id"],
        "classification": "technical_diagnostic_only",
        "decision": "rejected",
        "formal_dataset_asset": False,
        "readiness_bundle_published": False,
        "user_approval": "not_requested_for_rejected_diagnostic",
        "purpose": "human_authoritative_facing_vs_travel_review",
        "canonical_front": "negative-y",
        "canonical_up": "positive-z",
        "source": json.loads(json.dumps(authenticated)),
        "metrics_summary": metrics["summary"],
        "derived_artifacts": {
            "top_facing.png": _record(
                staging / "top_facing.png", filename="top_facing.png"
            ),
            "top_facing.mp4": _record(
                staging / "top_facing.mp4", filename="top_facing.mp4"
            ),
            "facing_metrics.json": _record(
                metrics_path, filename="facing_metrics.json"
            ),
            "review.html": _record(
                staging / "review.html", filename="review.html"
            ),
        },
        "media_qa": media_qa,
        "environment": {
            "blender_version": blender_version,
            "fps": FPS,
            "frame_count": FRAME_COUNT,
            "resolution": list(VIDEO_SIZE),
        },
        "execution": {
            "renderer": _record(SCRIPT_PATH),
            "command": [str(value) for value in command],
        },
    }
    if "user_approved" in json.dumps(manifest, sort_keys=True):
        raise FacingRenderError("facing render may not claim user approval")
    _write_json_exclusive(staging / "facing_review_manifest.json", manifest)


def run(
    *, diagnostic_dir: Path, output_dir: Path, command: Sequence[str] | None = None
) -> Path:
    destination = review.validate_destination(output_dir)
    authenticated = authenticate_second_attempt(diagnostic_dir)
    try:
        import bpy
    except ImportError as error:
        raise FacingRenderError("renderer must run inside Blender") from error
    blender_version = validate_blender_version(tuple(bpy.app.version))
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".staging",
            dir=str(destination.parent),
        )
    )
    try:
        _render_bundle(
            bpy,
            authenticated=authenticated,
            staging=staging,
            blender_version=blender_version,
            command=list(command if command is not None else sys.argv),
        )
        expected = {
            "top_facing.png",
            "top_facing.mp4",
            "facing_metrics.json",
            "facing_review_manifest.json",
            "review.html",
        }
        if {path.name for path in staging.iterdir()} != expected:
            raise FacingRenderError("staged facing render inventory is incomplete")
        for path in staging.iterdir():
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise FacingRenderError(f"invalid staged artifact: {path.name}")
            review._fsync_file(path)
            path.chmod(0o444)
        review._fsync_directory(staging)
        staging.chmod(0o555)
        review.rename_directory_noreplace(staging, destination)
        staging = None
        review._fsync_directory(destination.parent)
        return destination / "facing_review_manifest.json"
    finally:
        if staging is not None and staging.exists():
            staging.chmod(0o700)
            for path in staging.iterdir():
                path.chmod(0o600)
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(
        diagnostic_dir=args.diagnostic_dir,
        output_dir=args.output_dir,
        command=sys.argv,
    )
    print(f"SECOND_RETARGET_FACING_RENDER_OK {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
