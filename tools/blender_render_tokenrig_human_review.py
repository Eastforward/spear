#!/usr/bin/env python3
"""Render hash-locked five-view Walk/Idle media for a TokenRig human.

The pure authentication and media-validation helpers intentionally import
without Blender.  Run the renderer with Blender 4.2 using ``--background
--python ... -- <arguments>``.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


MOTIONS = {"walking": "Walking", "standing_idle": "Standing_Idle"}
VIEWS = ("front", "side", "top", "feet", "skeleton")
VIDEO_SIZE = (1280, 720)
FPS = 30
MINIMUM_LUMA_SPAN = 8.0
STATIC_QA_SCHEMA = "tokenrig_human_static_qa_v1"
STATIC_QA_DECISION = "automatic_static_checks_passed"
RETARGET_SCHEMA = "tokenrig_rocketbox_retarget_v1"
RETARGET_METRICS_SCHEMA = "tokenrig_rocketbox_retarget_metrics_v1"
MEDIA_QA_SCHEMA = "tokenrig_human_media_qa_v1"
REVIEW_MANIFEST_SCHEMA = "tokenrig_human_dynamic_review_v1"
CANONICAL_FRONT = "negative-y"
CANONICAL_UP = "positive-z"
STATIC_EVIDENCE = (
    "bind_front.png",
    "bind_back.png",
    "bind_side.png",
    "bind_top.png",
    "skeleton_overlay.png",
    "weights_contact.png",
    "texture_compare.png",
    "joint_hierarchy.txt",
)
_ASSET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SIGNAL_RE = re.compile(r"lavfi\.signalstats\.(YMIN|YMAX)=(-?[0-9]+(?:\.[0-9]+)?)")
SCRIPT_PATH = Path(__file__).resolve()


class ReviewRenderError(RuntimeError):
    """An authenticated rendering or media invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _real_directory(path: Path, description: str) -> Path:
    path = _absolute(path)
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ReviewRenderError(f"{description} must be a direct real directory: {path}")
    return path


def _regular_file(path: Path, root: Path, description: str) -> Path:
    path = _absolute(path)
    root = _real_directory(root, f"{description} root")
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ReviewRenderError(f"{description} must be a direct regular file: {path}")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ReviewRenderError(f"{description} is outside its authenticated root") from error
    if path.stat().st_size <= 0:
        raise ReviewRenderError(f"{description} is empty")
    return path


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewRenderError(f"{description} is not readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise ReviewRenderError(f"{description} must be a JSON object")
    return value


def file_record(path: Path, *, filename: str | None = None) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ReviewRenderError(f"artifact is missing, empty, or symlinked: {path}")
    result: dict[str, Any] = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if filename is not None:
        result["filename"] = filename
    else:
        result["path"] = str(path.resolve())
    return result


def validate_blender_version(version: Sequence[int]) -> str:
    values = tuple(int(item) for item in version[:3])
    if len(values) < 2 or values[:2] != (4, 2):
        raise ReviewRenderError(
            f"dynamic review requires Blender 4.2 LTS, got {values}"
        )
    padded = values + (0,) * (3 - len(values))
    return ".".join(str(item) for item in padded[:3])


def _media_tool_descriptor(name: str) -> dict[str, Any]:
    located = shutil.which(name)
    if not located:
        raise ReviewRenderError(f"required media tool is missing: {name}")
    path = Path(located).resolve()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ReviewRenderError(f"{name} must resolve to a direct regular executable")
    result = subprocess.run(
        [str(path), "-version"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines or not lines[0].lower().startswith(name):
        raise ReviewRenderError(f"could not authenticate {name} version")
    return {**file_record(path), "version": lines[0]}


def authenticate_execution_environment() -> dict[str, Any]:
    return {
        "renderer": file_record(SCRIPT_PATH),
        "ffmpeg": _media_tool_descriptor("ffmpeg"),
        "ffprobe": _media_tool_descriptor("ffprobe"),
    }


def assert_execution_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    if before != after:
        raise ReviewRenderError(
            "renderer or FFmpeg/FFprobe execution contract changed during rendering"
        )


def _validate_record(path: Path, record: Any, description: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ReviewRenderError(f"{description} has no authenticated record")
    if record.get("sha256") != sha256_file(path):
        raise ReviewRenderError(f"{description} SHA-256 does not match its authenticated record")
    if record.get("size_bytes") != path.stat().st_size:
        raise ReviewRenderError(f"{description} size does not match its authenticated record")
    return file_record(path)


def read_glb_document(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise ReviewRenderError(f"not a GLB 2.0 file: {path}")
    version, declared = struct.unpack_from("<II", raw, 4)
    if version != 2 or declared != len(raw):
        raise ReviewRenderError(f"invalid GLB header: {path}")
    offset = 12
    documents: list[bytes] = []
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ReviewRenderError(f"truncated GLB chunk header: {path}")
        length, kind = struct.unpack_from("<II", raw, offset)
        offset += 8
        end = offset + length
        if end > len(raw):
            raise ReviewRenderError(f"truncated GLB chunk: {path}")
        if kind == 0x4E4F534A:
            documents.append(raw[offset:end])
        offset = end
    if len(documents) != 1:
        raise ReviewRenderError(f"GLB must contain exactly one JSON chunk: {path}")
    try:
        result = json.loads(documents[0].rstrip(b" \t\r\n\0").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewRenderError(f"invalid GLB JSON: {path}: {error}") from error
    if not isinstance(result, dict):
        raise ReviewRenderError(f"GLB JSON root must be an object: {path}")
    return result


def _validate_one_action_glb(path: Path, expected_action: str) -> dict[str, Any]:
    document = read_glb_document(path)
    animations = document.get("animations")
    names = [item.get("name") for item in animations] if isinstance(animations, list) else []
    if names != [expected_action]:
        raise ReviewRenderError(f"{path.name} must contain exactly one {expected_action} animation")
    if not isinstance(document.get("meshes"), list) or not document["meshes"]:
        raise ReviewRenderError(f"{path.name} has no mesh")
    if not isinstance(document.get("skins"), list) or len(document["skins"]) != 1:
        raise ReviewRenderError(f"{path.name} must contain exactly one skin")
    return {
        "animation_name": expected_action,
        "mesh_count": len(document["meshes"]),
        "skin_joint_count": len(document["skins"][0].get("joints", [])),
    }


def authenticate_review_inputs(
    *,
    asset_id: str,
    static_qa_json: Path,
    retarget_manifest: Path,
    walking_glb: Path,
    standing_idle_glb: Path,
) -> dict[str, Any]:
    """Authenticate the exact static and retarget snapshots consumed by rendering."""

    if not _ASSET_RE.fullmatch(asset_id):
        raise ReviewRenderError(f"invalid asset_id: {asset_id!r}")
    static_qa_json = _absolute(static_qa_json)
    if static_qa_json.name != "static_qa.json":
        raise ReviewRenderError("static QA input must be named static_qa.json")
    static_root = _real_directory(static_qa_json.parent, "static QA bundle")
    static_path = _regular_file(static_qa_json, static_root, "static_qa.json")
    static = _load_json(static_path, "static QA")
    if static.get("schema") != STATIC_QA_SCHEMA:
        raise ReviewRenderError(f"static QA schema must be {STATIC_QA_SCHEMA}")
    if static.get("decision") != STATIC_QA_DECISION:
        raise ReviewRenderError("static QA decision is not passed")
    if static.get("asset_id") != asset_id:
        raise ReviewRenderError("static QA asset_id does not match")
    if "user_approved" in json.dumps(static):
        raise ReviewRenderError("static QA must not synthesize user approval")
    checks = static.get("checks")
    if not isinstance(checks, Mapping):
        raise ReviewRenderError("static QA checks are missing")
    axis = checks.get("axis_canonicalization")
    if not isinstance(axis, Mapping) or axis.get("canonical_front") != CANONICAL_FRONT or axis.get("transform_count") != 1:
        raise ReviewRenderError("static QA does not prove exactly-once FRONT -Y")
    ground = checks.get("grounding")
    if not isinstance(ground, Mapping) or ground.get("canonical_floor_z") != 0.0 or ground.get("post_floor_z") != 0.0 or ground.get("transform_count") != 1:
        raise ReviewRenderError("static QA does not prove one Z=0 floor")
    pbr = checks.get("exported_pbr")
    if not isinstance(pbr, Mapping) or pbr.get("passed") is not True:
        raise ReviewRenderError("static QA PBR preservation is not passed")
    semantic_mapping = checks.get("semantic_mapping")
    semantic_bones = (
        semantic_mapping.get("semantic_bones")
        if isinstance(semantic_mapping, Mapping)
        else None
    )
    required_semantic_bones = {
        "pelvis",
        "head",
        "left_hand",
        "right_hand",
        "left_foot",
        "left_toe",
        "right_foot",
        "right_toe",
    }
    if not isinstance(semantic_bones, Mapping) or any(
        not isinstance(semantic_bones.get(role), str)
        or not semantic_bones[role]
        for role in required_semantic_bones
    ):
        raise ReviewRenderError("static QA semantic bone mapping is incomplete")
    artifacts = static.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ReviewRenderError("static QA artifacts are missing")
    bind = _regular_file(static_root / "bind_pose.glb", static_root, "bind_pose.glb")
    bind_record = _validate_record(bind, artifacts.get("bind_pose.glb"), "bind_pose.glb")
    static_evidence: dict[str, Any] = {}
    for filename in STATIC_EVIDENCE:
        path = _regular_file(static_root / filename, static_root, filename)
        static_evidence[filename] = _validate_record(path, artifacts.get(filename), filename)

    retarget_manifest = _absolute(retarget_manifest)
    if retarget_manifest.name != "retarget_manifest.json":
        raise ReviewRenderError("retarget input must be named retarget_manifest.json")
    retarget_root = _real_directory(retarget_manifest.parent, "retarget bundle")
    retarget_path = _regular_file(retarget_manifest, retarget_root, "retarget_manifest.json")
    retarget = _load_json(retarget_path, "retarget manifest")
    if retarget.get("schema") != RETARGET_SCHEMA:
        raise ReviewRenderError(f"retarget manifest schema must be {RETARGET_SCHEMA}")
    if retarget.get("asset_id") != asset_id:
        raise ReviewRenderError("retarget asset_id does not match")
    if retarget.get("automatic_checks") != "passed":
        raise ReviewRenderError("retarget automatic checks are not passed")
    if retarget.get("canonical_front") != CANONICAL_FRONT:
        raise ReviewRenderError("retarget does not preserve FRONT -Y")
    if retarget.get("canonical_up") != CANONICAL_UP:
        raise ReviewRenderError("retarget does not preserve UP +Z")
    environment = retarget.get("environment")
    if not isinstance(environment, Mapping) or environment.get("fps") != FPS:
        raise ReviewRenderError("retarget FPS is not the required 30")
    actions = retarget.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != set(MOTIONS.values()):
        raise ReviewRenderError("retarget must authenticate exactly Walking and Standing_Idle")
    for action_name in MOTIONS.values():
        action = actions.get(action_name)
        if not isinstance(action, Mapping) or action.get("status") != "passed" or action.get("action_name") != action_name:
            raise ReviewRenderError(f"retarget action is not passed: {action_name}")
    retarget_static = retarget.get("authenticated_inputs", {}).get("static")
    if not isinstance(retarget_static, Mapping):
        raise ReviewRenderError("retarget manifest has no authenticated static input")
    if retarget_static.get("static_qa", {}).get("sha256") != sha256_file(static_path):
        raise ReviewRenderError("retarget static QA hash does not match current static snapshot")
    if retarget_static.get("bind_pose", {}).get("sha256") != sha256_file(bind):
        raise ReviewRenderError("retarget bind-pose hash does not match current static snapshot")
    if retarget_static.get("floor_z_m") != 0.0:
        raise ReviewRenderError("retarget static floor is not Z=0")

    supplied = {"walking": _absolute(walking_glb), "standing_idle": _absolute(standing_idle_glb)}
    expected_names = {"walking": "walking.glb", "standing_idle": "standing_idle.glb"}
    retarget_artifacts = retarget.get("artifacts")
    if not isinstance(retarget_artifacts, Mapping):
        raise ReviewRenderError("retarget artifact records are missing")
    glb_records: dict[str, Any] = {}
    glb_readback: dict[str, Any] = {}
    for motion, path in supplied.items():
        filename = expected_names[motion]
        if path.name != filename or path.parent != retarget_root:
            raise ReviewRenderError(f"{filename} must be the canonical retarget artifact")
        checked = _regular_file(path, retarget_root, filename)
        record = retarget_artifacts.get(filename)
        if isinstance(record, Mapping) and record.get("path") not in (None, filename):
            raise ReviewRenderError(f"retarget {filename} path is not canonical")
        glb_records[motion] = _validate_record(checked, record, filename)
        glb_readback[motion] = _validate_one_action_glb(checked, MOTIONS[motion])

    metrics_path = _regular_file(retarget_root / "retarget_metrics.json", retarget_root, "retarget_metrics.json")
    metrics_record = _validate_record(metrics_path, retarget_artifacts.get("retarget_metrics.json"), "retarget_metrics.json")
    metrics = _load_json(metrics_path, "retarget metrics")
    if metrics.get("schema") != RETARGET_METRICS_SCHEMA or metrics.get("automatic_checks") != "passed":
        raise ReviewRenderError("retarget metrics are not passed under the pinned schema")
    if set(metrics.get("actions", {})) != set(MOTIONS.values()):
        raise ReviewRenderError("retarget metrics do not cover exactly Walking and Standing_Idle")
    return {
        "asset_id": asset_id,
        "canonical_front": CANONICAL_FRONT,
        "canonical_up": CANONICAL_UP,
        "static_qa": file_record(static_path),
        "bind_pose": bind_record,
        "static_evidence": static_evidence,
        "retarget_manifest": file_record(retarget_path),
        "retarget_metrics": metrics_record,
        "glbs": glb_records,
        "glb_readback": glb_readback,
        "semantic_mapping": json.loads(json.dumps(semantic_mapping, sort_keys=True)),
    }


def validate_ffprobe_payload(payload: Mapping[str, Any], *, expected_frame_count: int) -> dict[str, Any]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], Mapping):
        raise ReviewRenderError("media must contain exactly one video stream")
    stream = streams[0]
    if stream.get("codec_name") != "h264":
        raise ReviewRenderError("video codec is not H.264")
    if (stream.get("width"), stream.get("height")) != VIDEO_SIZE:
        raise ReviewRenderError("video resolution is not 1280x720")
    try:
        rate = Fraction(str(stream.get("r_frame_rate", "0/1")))
        average_rate = Fraction(str(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError) as error:
        raise ReviewRenderError("video frame rate is malformed") from error
    if rate != FPS or average_rate != FPS:
        raise ReviewRenderError("video r_frame_rate and avg_frame_rate must both be 30 fps")
    raw_count = stream.get("nb_frames") or stream.get("nb_read_frames")
    try:
        frame_count = int(raw_count)
    except (TypeError, ValueError) as error:
        raise ReviewRenderError("video frame count is unavailable") from error
    if frame_count != expected_frame_count:
        raise ReviewRenderError(f"video frame count is {frame_count}, expected {expected_frame_count}")
    try:
        duration = float(stream.get("duration"))
    except (TypeError, ValueError) as error:
        raise ReviewRenderError("video duration is unavailable") from error
    expected_duration = expected_frame_count / FPS
    if not math.isfinite(duration) or abs(duration - expected_duration) > 1.0 / FPS:
        raise ReviewRenderError("video duration is outside one-frame tolerance")
    return {
        "codec_name": "h264",
        "width": VIDEO_SIZE[0],
        "height": VIDEO_SIZE[1],
        "r_frame_rate": str(stream.get("r_frame_rate")),
        "avg_frame_rate": str(stream.get("avg_frame_rate")),
        "frame_count": frame_count,
        "duration_s": duration,
    }


def validate_luma_ranges(ranges: Sequence[tuple[float, float]], *, expected_frame_count: int) -> dict[str, Any]:
    if len(ranges) != expected_frame_count:
        raise ReviewRenderError(f"decoded frame count is {len(ranges)}, expected {expected_frame_count}")
    spans = [float(maximum) - float(minimum) for minimum, maximum in ranges]
    if any(not math.isfinite(value) or value < MINIMUM_LUMA_SPAN for value in spans):
        raise ReviewRenderError("one or more video frames are blank or nearly blank")
    return {
        "decoded_frame_count": len(ranges),
        "minimum_luma_span": min(spans),
        "maximum_luma_span": max(spans),
    }


def _mean_absolute_byte_delta(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise ReviewRenderError("temporal frames have inconsistent dimensions")
    return sum(abs(int(a) - int(b)) for a, b in zip(left, right)) / len(left)


def _subject_roi(frame: bytes, *, width: int, height: int) -> bytes:
    if len(frame) != width * height:
        raise ReviewRenderError("decoded temporal frame has an invalid byte count")
    x_start = max(0, int(math.floor(width * 0.15)))
    x_end = min(width, int(math.ceil(width * 0.85)))
    y_start = max(0, int(math.floor(height * 0.15)))
    y_end = min(height, int(math.ceil(height * 0.85)))
    if x_end <= x_start or y_end <= y_start:
        raise ReviewRenderError("subject ROI is empty")
    return b"".join(
        frame[y * width + x_start : y * width + x_end]
        for y in range(y_start, y_end)
    )


def validate_temporal_frames(
    frames: Sequence[bytes],
    *,
    width: int,
    height: int,
    expected_frame_count: int,
    motion: str,
    view: str,
) -> dict[str, Any]:
    if motion not in MOTIONS or view not in VIEWS:
        raise ReviewRenderError("temporal QA motion or view is invalid")
    if len(frames) != expected_frame_count or expected_frame_count < 2:
        raise ReviewRenderError(
            f"temporal decoded frame count is {len(frames)}, expected {expected_frame_count}"
        )
    if width <= 2 or height <= 2:
        raise ReviewRenderError("temporal QA dimensions are too small")
    if any(len(frame) != width * height for frame in frames):
        raise ReviewRenderError("decoded temporal frame has an invalid byte count")
    unique_count = len({hashlib.sha256(frame).digest() for frame in frames})
    if unique_count < 2:
        raise ReviewRenderError("video is nonblank but temporally frozen")
    minimum_unique = (
        max(3, int(math.ceil(expected_frame_count * 0.15)))
        if motion == "walking"
        else 2
    )
    if unique_count < minimum_unique:
        raise ReviewRenderError(
            f"video unique-frame ratio is too low: {unique_count}/{expected_frame_count}"
        )
    full_deltas = [
        _mean_absolute_byte_delta(frames[index - 1], frames[index])
        for index in range(1, len(frames))
    ]
    rois = [_subject_roi(frame, width=width, height=height) for frame in frames]
    roi_deltas = [
        _mean_absolute_byte_delta(rois[index - 1], rois[index])
        for index in range(1, len(rois))
    ]
    mean_full = sum(full_deltas) / len(full_deltas)
    mean_roi = sum(roi_deltas) / len(roi_deltas)
    maximum_roi = max(roi_deltas)
    minimum_full_delta = 0.05 if motion == "walking" else 0.005
    minimum_roi_delta = 0.20 if motion == "walking" else 0.02
    if mean_full < minimum_full_delta:
        raise ReviewRenderError("video has no meaningful cross-frame difference")
    if maximum_roi < minimum_roi_delta or mean_roi <= 0.0:
        raise ReviewRenderError("video subject ROI has no motion")
    skeleton_motion_passed = True
    if view == "skeleton" and maximum_roi < 0.05:
        skeleton_motion_passed = False
        raise ReviewRenderError("skeleton overlay has no measurable motion")
    return {
        "decoded_frame_count": len(frames),
        "unique_frame_count": unique_count,
        "unique_frame_ratio": unique_count / len(frames),
        "mean_adjacent_luma_delta": mean_full,
        "mean_subject_roi_delta": mean_roi,
        "maximum_subject_roi_delta": maximum_roi,
        "subject_roi_xyxy": [
            int(math.floor(width * 0.15)),
            int(math.floor(height * 0.15)),
            int(math.ceil(width * 0.85)),
            int(math.ceil(height * 0.85)),
        ],
        "skeleton_motion_passed": skeleton_motion_passed,
    }


def validate_skeleton_overlay_motion(
    frames: Sequence[bytes],
    *,
    width: int,
    height: int,
    expected_frame_count: int,
) -> dict[str, Any]:
    if len(frames) != expected_frame_count or expected_frame_count < 2:
        raise ReviewRenderError("skeleton RGB frame count is incomplete")
    expected_bytes = width * height * 3
    if any(len(frame) != expected_bytes for frame in frames):
        raise ReviewRenderError("skeleton RGB frame has an invalid byte count")
    masks: list[bytes] = []
    centroids: list[tuple[float, float]] = []
    counts: list[int] = []
    minimum_required = max(3, int(math.ceil(width * height * 0.001)))
    for frame in frames:
        mask = bytearray(width * height)
        xs: list[int] = []
        ys: list[int] = []
        for pixel in range(width * height):
            red, green, blue = frame[pixel * 3 : pixel * 3 + 3]
            if (
                green >= 120
                and blue >= 100
                and int(green) - int(red) >= 45
                and int(blue) - int(red) >= 35
            ):
                mask[pixel] = 1
                xs.append(pixel % width)
                ys.append(pixel // width)
        count = len(xs)
        if count < minimum_required:
            raise ReviewRenderError(
                f"cyan skeleton overlay is missing or too small: {count} pixels"
            )
        masks.append(bytes(mask))
        counts.append(count)
        centroids.append((sum(xs) / count, sum(ys) / count))
    unique_masks = len({hashlib.sha256(mask).digest() for mask in masks})
    if unique_masks < 2:
        raise ReviewRenderError("cyan skeleton overlay is frozen")
    centroid_displacements = [
        math.hypot(
            centroids[index][0] - centroids[index - 1][0],
            centroids[index][1] - centroids[index - 1][1],
        )
        for index in range(1, len(centroids))
    ]
    mask_deltas = [
        sum(left != right for left, right in zip(masks[index - 1], masks[index]))
        / len(masks[index])
        for index in range(1, len(masks))
    ]
    maximum_centroid = max(centroid_displacements)
    maximum_mask_delta = max(mask_deltas)
    if maximum_centroid < 0.05 and maximum_mask_delta < 0.0005:
        raise ReviewRenderError("cyan skeleton overlay has no measurable motion")
    return {
        "decoded_frame_count": len(frames),
        "minimum_cyan_pixels": min(counts),
        "maximum_cyan_pixels": max(counts),
        "unique_cyan_mask_count": unique_masks,
        "maximum_centroid_displacement_px": maximum_centroid,
        "maximum_mask_change_ratio": maximum_mask_delta,
        "passed": True,
    }


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ReviewRenderError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout[-4000:]}")
    return result.stdout


def _run_bytes(command: Sequence[str]) -> bytes:
    result = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise ReviewRenderError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{message}"
        )
    return result.stdout


def probe_video(
    path: Path,
    *,
    expected_frame_count: int,
    motion: str,
    view: str,
) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise ReviewRenderError("ffprobe and ffmpeg are required for media QA")
    probe = json.loads(
        _run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "v",
                "-count_frames",
                "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
                "-of", "json",
                str(path),
            ]
        )
    )
    result = validate_ffprobe_payload(probe, expected_frame_count=expected_frame_count)
    signal = _run(
        [
            ffmpeg,
            "-v", "error",
            "-i", str(path),
            "-vf", "signalstats,metadata=print:file=-",
            "-an", "-f", "null", "-",
        ]
    )
    values: list[tuple[float, float]] = []
    pending: dict[str, float] = {}
    for name, raw in _SIGNAL_RE.findall(signal):
        pending[name] = float(raw)
        if set(pending) == {"YMIN", "YMAX"}:
            values.append((pending["YMIN"], pending["YMAX"]))
            pending = {}
    result["nonblank_frames"] = validate_luma_ranges(values, expected_frame_count=expected_frame_count)
    temporal_width, temporal_height = 160, 90
    raw_frames = _run_bytes(
        [
            ffmpeg,
            "-v", "error",
            "-i", str(path),
            "-vf", f"scale={temporal_width}:{temporal_height}:flags=area,format=gray",
            "-an", "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
    )
    frame_bytes = temporal_width * temporal_height
    if len(raw_frames) % frame_bytes != 0:
        raise ReviewRenderError("decoded temporal byte stream is truncated")
    frames = [
        raw_frames[offset : offset + frame_bytes]
        for offset in range(0, len(raw_frames), frame_bytes)
    ]
    result["temporal_motion"] = validate_temporal_frames(
        frames,
        width=temporal_width,
        height=temporal_height,
        expected_frame_count=expected_frame_count,
        motion=motion,
        view=view,
    )
    if view == "skeleton":
        rgb_frames_raw = _run_bytes(
            [
                ffmpeg,
                "-v", "error",
                "-i", str(path),
                "-vf", f"scale={temporal_width}:{temporal_height}:flags=area,format=rgb24",
                "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ]
        )
        rgb_frame_bytes = temporal_width * temporal_height * 3
        if len(rgb_frames_raw) % rgb_frame_bytes != 0:
            raise ReviewRenderError("decoded skeleton RGB stream is truncated")
        rgb_frames = [
            rgb_frames_raw[offset : offset + rgb_frame_bytes]
            for offset in range(0, len(rgb_frames_raw), rgb_frame_bytes)
        ]
        result["skeleton_overlay_motion"] = validate_skeleton_overlay_motion(
            rgb_frames,
            width=temporal_width,
            height=temporal_height,
            expected_frame_count=expected_frame_count,
        )
    result.update({"sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return result


def probe_png(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise ReviewRenderError("ffprobe and ffmpeg are required for PNG QA")
    payload = json.loads(
        _run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(path)])
    )
    streams = payload.get("streams", [])
    if len(streams) != 1 or (streams[0].get("width"), streams[0].get("height")) != VIDEO_SIZE:
        raise ReviewRenderError("representative PNG resolution is not 1280x720")
    signal = _run([ffmpeg, "-v", "error", "-i", str(path), "-vf", "signalstats,metadata=print:file=-", "-frames:v", "1", "-f", "null", "-"])
    values = {name: float(raw) for name, raw in _SIGNAL_RE.findall(signal)}
    if set(values) != {"YMIN", "YMAX"} or values["YMAX"] - values["YMIN"] < MINIMUM_LUMA_SPAN:
        raise ReviewRenderError("representative PNG is blank or nearly blank")
    return {
        "width": VIDEO_SIZE[0],
        "height": VIDEO_SIZE[1],
        "luma_span": values["YMAX"] - values["YMIN"],
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_destination(destination: Path) -> Path:
    destination = _absolute(destination)
    if os.path.lexists(destination):
        raise ReviewRenderError(f"output directory already exists: {destination}")
    _real_directory(destination.parent, "output parent")
    return destination


def build_review_manifest(
    *,
    asset_id: str,
    display_label: str,
    instance_kind: str,
    authenticated: Mapping[str, Any],
    actions: Mapping[str, Mapping[str, Any]],
    media: Mapping[str, Mapping[str, Any]],
    media_qa_record: Mapping[str, Any],
    execution: Mapping[str, Any],
    blender_version: str,
    command: Sequence[str],
) -> dict[str, Any]:
    if not _ASSET_RE.fullmatch(asset_id) or authenticated.get("asset_id") != asset_id:
        raise ReviewRenderError("manifest asset identity is invalid")
    if not display_label.strip() or not _ASSET_RE.fullmatch(instance_kind):
        raise ReviewRenderError("display label and instance kind are required")
    if set(actions) != set(MOTIONS) or set(media) != set(MOTIONS):
        raise ReviewRenderError("review manifest must contain exactly Walk and Idle")
    if not isinstance(execution, Mapping) or set(execution) != {
        "renderer",
        "ffmpeg",
        "ffprobe",
    }:
        raise ReviewRenderError("review manifest execution contract is incomplete")
    for name in ("renderer", "ffmpeg", "ffprobe"):
        descriptor = execution[name]
        if (
            not isinstance(descriptor, Mapping)
            or not isinstance(descriptor.get("path"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("sha256", "")))
            or not isinstance(descriptor.get("size_bytes"), int)
        ):
            raise ReviewRenderError(f"execution descriptor is invalid: {name}")
    for name in ("ffmpeg", "ffprobe"):
        if not isinstance(execution[name].get("version"), str) or not execution[name][
            "version"
        ].lower().startswith(name):
            raise ReviewRenderError(f"execution tool version is invalid: {name}")
    action_payload: dict[str, Any] = {}
    for motion, action_name in MOTIONS.items():
        details = dict(actions[motion])
        if details.get("action_name") != action_name:
            raise ReviewRenderError(f"action metadata is stale: {motion}")
        views = media[motion]
        if not isinstance(views, Mapping) or set(views) != set(VIEWS):
            raise ReviewRenderError(f"media matrix is incomplete: {motion}")
        action_payload[motion] = {**details, "views": json.loads(json.dumps(views))}
    result = {
        "schema": REVIEW_MANIFEST_SCHEMA,
        "asset_id": asset_id,
        "display_label": display_label.strip(),
        "instance_kind": instance_kind,
        "state_classification": "research_candidate",
        "canonical_front": CANONICAL_FRONT,
        "canonical_up": CANONICAL_UP,
        "fixed_floor_z_m": 0.0,
        "upstream": json.loads(json.dumps(authenticated)),
        "actions": action_payload,
        "media_qa": dict(media_qa_record),
        "execution": json.loads(json.dumps(execution)),
        "automatic_checks": "passed",
        "agent_visual_qa": "pending_agent_visual_qa",
        "user_acceptance": "pending_user_review",
        "environment": {"blender_version": str(blender_version), "fps": FPS, "resolution": list(VIDEO_SIZE)},
        "command": [str(item) for item in command],
    }
    if "user_approved" in json.dumps(result):
        raise ReviewRenderError("review manifest may not claim user approval")
    return result


def _write_exclusive(path: Path, payload: bytes) -> None:
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_path_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise ReviewRenderError("atomic renameat2 no-replace is unavailable") from error
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise ReviewRenderError(f"no-replace destination already exists: {destination}")
    raise ReviewRenderError(f"atomic no-replace publication failed: {os.strerror(number)}")


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    try:
        rename_path_noreplace(source, destination)
    except ReviewRenderError as error:
        if "no-replace destination already exists" in str(error):
            raise ReviewRenderError(f"output directory already exists: {destination}") from error
        raise


def write_failure_evidence(*, destination: Path, asset_id: str, error: BaseException, authenticated: Mapping[str, Any] | None) -> Path:
    parent = _real_directory(_absolute(destination).parent, "failure evidence parent")
    path = parent / f"{Path(destination).name}.render_failed_attempt.{uuid.uuid4().hex}.json"
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema": "tokenrig_human_render_failed_attempt_v1",
        "asset_id": asset_id,
        "decision": "rejected",
        "error_type": type(error).__name__,
        "error": str(error),
        "authenticated_inputs": dict(authenticated or {}),
    }
    try:
        _write_json_exclusive(temporary, payload)
        temporary.chmod(0o444)
        _fsync_file(temporary)
        _fsync_directory(parent)
        rename_path_noreplace(temporary, path)
        _fsync_directory(parent)
        return path
    except BaseException:
        if temporary.exists():
            temporary.chmod(0o600)
            temporary.unlink()
            _fsync_directory(parent)
        raise


def _clear_scene(bpy: Any) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.actions, bpy.data.armatures, bpy.data.meshes, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _scene_objects(bpy: Any, expected_action: str) -> tuple[Any, Any, Any]:
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    meshes = [item for item in bpy.context.scene.objects if item.type == "MESH" and any(mod.type == "ARMATURE" for mod in item.modifiers)]
    actions = list(bpy.data.actions)
    if len(armatures) != 1 or len(meshes) != 1:
        raise ReviewRenderError("render import must contain one armature and one skinned mesh")
    allowed_action_names = {expected_action, f"{expected_action}_Armature"}
    matching = [item for item in actions if item.name in allowed_action_names]
    if len(actions) != 1 or len(matching) != 1:
        raise ReviewRenderError(f"render import must contain exactly one {expected_action} action")
    armature, mesh, action = armatures[0], meshes[0], matching[0]
    if action.name == f"{expected_action}_Armature":
        action.name = expected_action
    if action.name != expected_action:
        raise ReviewRenderError(f"render import action normalization failed: {action.name}")
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    return armature, mesh, action


def _integer_frame_range(action: Any) -> tuple[int, int]:
    start_raw, end_raw = map(float, action.frame_range)
    start, end = int(round(start_raw)), int(round(end_raw))
    if abs(start_raw - start) > 1.0e-4 or abs(end_raw - end) > 1.0e-4 or end < start:
        raise ReviewRenderError(f"action has a non-integral or empty frame range: {action.frame_range[:]}")
    return start, end


def _world_bounds(bpy: Any, mesh: Any, frame_start: int, frame_end: int) -> list[dict[str, tuple[float, float, float]]]:
    from mathutils import Vector

    result = []
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        evaluated = mesh.evaluated_get(depsgraph)
        corners = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
        minimum = tuple(min(point[index] for point in corners) for index in range(3))
        maximum = tuple(max(point[index] for point in corners) for index in range(3))
        center = tuple((minimum[index] + maximum[index]) * 0.5 for index in range(3))
        result.append({"minimum": minimum, "maximum": maximum, "center": center})
    return result


def build_feet_review_bounds(
    joint_samples: Sequence[Sequence[Sequence[float]]], *, body_height: float
) -> list[dict[str, tuple[float, float, float]]]:
    """Build a per-frame feet/lower-leg ROI from authenticated foot joints."""

    if not math.isfinite(body_height) or body_height <= 0.2:
        raise ReviewRenderError("feet review body height is invalid")
    xy_padding = body_height * 0.08
    lower_padding = body_height * 0.04
    roi_height = body_height * 0.32
    result: list[dict[str, tuple[float, float, float]]] = []
    for sample in joint_samples:
        points = [tuple(float(value) for value in point) for point in sample]
        if len(points) != 4 or any(
            len(point) != 3 or any(not math.isfinite(value) for value in point)
            for point in points
        ):
            raise ReviewRenderError("feet review requires four finite foot/toe joints per frame")
        minimum_z = max(0.0, min(point[2] for point in points) - lower_padding)
        minimum = (
            min(point[0] for point in points) - xy_padding,
            min(point[1] for point in points) - xy_padding,
            minimum_z,
        )
        maximum = (
            max(point[0] for point in points) + xy_padding,
            max(point[1] for point in points) + xy_padding,
            minimum_z + roi_height,
        )
        result.append(
            {
                "minimum": minimum,
                "maximum": maximum,
                "center": tuple(
                    (minimum[index] + maximum[index]) * 0.5 for index in range(3)
                ),
            }
        )
    if not result:
        raise ReviewRenderError("feet review has no animated joint samples")
    return result


def _foot_joint_bounds(
    bpy: Any,
    armature: Any,
    action: Any,
    semantic_bones: Mapping[str, Any],
    body_height: float,
) -> list[dict[str, tuple[float, float, float]]]:
    roles = ("left_foot", "left_toe", "right_foot", "right_toe")
    names = [semantic_bones.get(role) for role in roles]
    if any(not isinstance(name, str) or not name for name in names):
        raise ReviewRenderError("feet review semantic bone mapping is incomplete")
    pose_bones = []
    for name in names:
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            raise ReviewRenderError(f"feet review pose bone is missing: {name}")
        pose_bones.append(pose_bone)
    frame_start, frame_end = _integer_frame_range(action)
    scene = bpy.context.scene
    samples = []
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        samples.append(
            [tuple(armature.matrix_world @ pose_bone.head) for pose_bone in pose_bones]
        )
    return build_feet_review_bounds(samples, body_height=body_height)


def _look_at(camera: Any, target: Any) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _v_add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[index]) + float(right[index]) for index in range(3))


def _v_sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))


def _v_scale(value: Sequence[float], scale: float) -> tuple[float, float, float]:
    return tuple(float(item) * float(scale) for item in value)


def _v_dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _v_cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _v_normalized(value: Sequence[float]) -> tuple[float, float, float]:
    length = math.sqrt(_v_dot(value, value))
    if not math.isfinite(length) or length <= 1.0e-12:
        raise ReviewRenderError("camera basis contains a zero-length vector")
    return _v_scale(value, 1.0 / length)


def camera_basis(view: str) -> dict[str, tuple[float, float, float]]:
    if view not in VIEWS:
        raise ReviewRenderError(f"unknown review camera view: {view}")
    if view in {"front", "feet"}:
        forward = (0.0, 1.0, 0.0)
        reference_up = (0.0, 0.0, 1.0)
    elif view == "side":
        forward = (-1.0, 0.0, 0.0)
        reference_up = (0.0, 0.0, 1.0)
    elif view == "top":
        forward = (0.0, 0.0, -1.0)
        reference_up = (0.0, 1.0, 0.0)
    else:
        forward = _v_normalized((-0.42, 1.0, -0.06))
        reference_up = (0.0, 0.0, 1.0)
    forward = _v_normalized(forward)
    right = _v_normalized(_v_cross(forward, reference_up))
    up = _v_normalized(_v_cross(right, forward))
    return {"forward": forward, "right": right, "up": up}


def _bound_corners(sample: Mapping[str, Any]) -> tuple[tuple[float, float, float], ...]:
    minimum = sample.get("minimum")
    maximum = sample.get("maximum")
    if (
        not isinstance(minimum, Sequence)
        or not isinstance(maximum, Sequence)
        or len(minimum) != 3
        or len(maximum) != 3
    ):
        raise ReviewRenderError("animated bound sample is missing min/max corners")
    values = tuple(float(item) for item in (*minimum, *maximum))
    if any(not math.isfinite(item) for item in values):
        raise ReviewRenderError("animated bound contains a non-finite coordinate")
    if any(values[index] >= values[index + 3] for index in range(3)):
        raise ReviewRenderError("animated bound is empty or inverted")
    return tuple(
        (x, y, z)
        for x in (values[0], values[3])
        for y in (values[1], values[4])
        for z in (values[2], values[5])
    )


def project_bound_corners_to_ndc(
    sample: Mapping[str, Any],
    *,
    location: Sequence[float],
    target: Sequence[float],
    view: str,
    angle_x: float,
    angle_y: float,
) -> list[tuple[float, float, float]]:
    if not 0.0 < float(angle_x) < math.pi or not 0.0 < float(angle_y) < math.pi:
        raise ReviewRenderError("camera field of view is invalid")
    basis = camera_basis(view)
    expected_forward = _v_normalized(_v_sub(target, location))
    if _v_dot(expected_forward, basis["forward"]) < 1.0 - 1.0e-8:
        raise ReviewRenderError("camera location/target does not match the canonical view basis")
    tan_x = math.tan(float(angle_x) * 0.5)
    tan_y = math.tan(float(angle_y) * 0.5)
    projected = []
    for corner in _bound_corners(sample):
        relative = _v_sub(corner, location)
        depth = _v_dot(relative, basis["forward"])
        if depth <= 0.0:
            raise ReviewRenderError("animated bound corner is behind the review camera")
        horizontal = _v_dot(relative, basis["right"])
        vertical = _v_dot(relative, basis["up"])
        projected.append(
            (
                0.5 + horizontal / (2.0 * depth * tan_x),
                0.5 + vertical / (2.0 * depth * tan_y),
                depth,
            )
        )
    return projected


def plan_camera_keyframes(
    *,
    view: str,
    bounds: Sequence[Mapping[str, Any]],
    angle_x: float,
    angle_y: float,
    ndc_margin: float = 0.08,
    clip_start: float = 0.05,
) -> list[dict[str, Any]]:
    if not bounds:
        raise ReviewRenderError("camera planning requires animated bounds")
    if not 0.0 < ndc_margin < 0.25:
        raise ReviewRenderError("camera NDC margin must be between zero and 0.25")
    if not math.isfinite(clip_start) or clip_start <= 0.0:
        raise ReviewRenderError("camera clip_start must be positive")
    basis = camera_basis(view)
    safe_fraction = 1.0 - 2.0 * ndc_margin
    tan_x = math.tan(float(angle_x) * 0.5)
    tan_y = math.tan(float(angle_y) * 0.5)
    if tan_x <= 0.0 or tan_y <= 0.0:
        raise ReviewRenderError("camera field of view is invalid")
    plans = []
    for sample in bounds:
        corners = _bound_corners(sample)
        target = tuple(
            (float(sample["minimum"][index]) + float(sample["maximum"][index])) * 0.5
            for index in range(3)
        )
        distance = clip_start * 1.5
        for corner in corners:
            relative = _v_sub(corner, target)
            forward_offset = _v_dot(relative, basis["forward"])
            horizontal = abs(_v_dot(relative, basis["right"]))
            vertical = abs(_v_dot(relative, basis["up"]))
            distance = max(
                distance,
                horizontal / (tan_x * safe_fraction) - forward_offset,
                vertical / (tan_y * safe_fraction) - forward_offset,
                clip_start * 1.5 - forward_offset,
            )
        distance += max(1.0e-4, distance * 1.0e-5)
        location = _v_sub(target, _v_scale(basis["forward"], distance))
        projected = project_bound_corners_to_ndc(
            sample,
            location=location,
            target=target,
            view=view,
            angle_x=angle_x,
            angle_y=angle_y,
        )
        tolerance = 2.0e-6
        if any(
            point[0] < ndc_margin - tolerance
            or point[0] > 1.0 - ndc_margin + tolerance
            or point[1] < ndc_margin - tolerance
            or point[1] > 1.0 - ndc_margin + tolerance
            or point[2] <= clip_start
            for point in projected
        ):
            raise ReviewRenderError("camera plan clips an animated bound corner")
        plans.append(
            {
                "location": location,
                "target": target,
                "distance": distance,
                "angle_x": float(angle_x),
                "angle_y": float(angle_y),
                "ndc_margin": ndc_margin,
                "projected_ndc_bounds": {
                    "minimum": [min(point[index] for point in projected) for index in range(2)],
                    "maximum": [max(point[index] for point in projected) for index in range(2)],
                    "minimum_depth": min(point[2] for point in projected),
                },
            }
        )
    return plans


def _create_skeleton_overlay(
    bpy: Any,
    armature: Any,
    scale: float,
    semantic_bones: Mapping[str, Any],
) -> tuple[Any, ...]:
    selected_names: list[str] = []
    for value in semantic_bones.values():
        candidates = value if isinstance(value, list) else [value]
        for name in candidates:
            if isinstance(name, str) and name and name not in selected_names:
                selected_names.append(name)
    if not selected_names:
        raise ReviewRenderError("skeleton review has no authenticated semantic bones")

    material = bpy.data.materials.new("TokenRigSkeletonCyan")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.0, 0.8, 0.8, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    radius = max(scale * 0.012, 0.008)
    overlays = []
    pose_bones = []
    for name in selected_names:
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            raise ReviewRenderError(f"skeleton review pose bone is missing: {name}")
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=radius, depth=1.0)
        overlay = bpy.context.object
        overlay.name = f"TokenRigSkeleton_{name}"
        overlay.data.materials.append(material)
        overlay.rotation_mode = "QUATERNION"
        overlay.hide_render = True
        overlays.append(overlay)
        pose_bones.append(pose_bone)

    scene = bpy.context.scene
    previous_quaternions = [None] * len(overlays)
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for index, (overlay, pose_bone) in enumerate(zip(overlays, pose_bones)):
            head = armature.matrix_world @ pose_bone.head
            tail = armature.matrix_world @ pose_bone.tail
            direction = tail - head
            length = direction.length
            if not math.isfinite(length) or length <= 1.0e-8:
                raise ReviewRenderError(
                    f"skeleton review bone has invalid pose length: {pose_bone.name}"
                )
            quaternion = direction.to_track_quat("Z", "Y")
            previous = previous_quaternions[index]
            if previous is not None and previous.dot(quaternion) < 0.0:
                quaternion.negate()
            previous_quaternions[index] = quaternion.copy()
            overlay.location = (head + tail) * 0.5
            overlay.rotation_quaternion = quaternion
            overlay.scale = (1.0, 1.0, length)
            overlay.keyframe_insert(data_path="location", frame=frame)
            overlay.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            overlay.keyframe_insert(data_path="scale", frame=frame)
    for overlay in overlays:
        if overlay.animation_data and overlay.animation_data.action:
            for curve in overlay.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
    return tuple(overlays)


def _stable_blender_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReviewRenderError("material graph contains a non-finite value")
        return round(value, 12)
    if hasattr(value, "to_tuple"):
        return [_stable_blender_value(item) for item in value.to_tuple()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_blender_value(item) for item in value]
    if hasattr(value, "name"):
        result = {
            "id_type": type(value).__name__,
            "name": str(value.name),
        }
        if hasattr(value, "filepath_raw"):
            result["filepath_raw"] = str(value.filepath_raw)
        if hasattr(value, "source"):
            result["source"] = str(value.source)
        return result
    return {"type": type(value).__name__}


def _material_graph_payload(material: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": str(material.name),
        "use_nodes": bool(material.use_nodes),
        "diffuse_color": _stable_blender_value(material.diffuse_color),
    }
    if hasattr(material, "surface_render_method"):
        result["surface_render_method"] = str(material.surface_render_method)
    elif hasattr(material, "blend_method"):
        result["blend_method"] = str(material.blend_method)
    if not material.use_nodes or material.node_tree is None:
        result.update({"nodes": [], "links": []})
        return result
    nodes = []
    for node in sorted(material.node_tree.nodes, key=lambda item: item.name):
        inputs = []
        for socket in node.inputs:
            entry = {
                "name": str(socket.name),
                "identifier": str(socket.identifier),
                "linked": bool(socket.is_linked),
            }
            if hasattr(socket, "default_value"):
                entry["default_value"] = _stable_blender_value(socket.default_value)
            inputs.append(entry)
        node_payload = {
            "name": str(node.name),
            "label": str(node.label),
            "type": str(node.bl_idname),
            "mute": bool(node.mute),
            "inputs": inputs,
        }
        if hasattr(node, "image"):
            node_payload["image"] = _stable_blender_value(node.image)
        if hasattr(node, "node_tree"):
            node_payload["node_tree"] = _stable_blender_value(node.node_tree)
        nodes.append(node_payload)
    links = sorted(
        (
            str(link.from_node.name),
            str(link.from_socket.identifier),
            str(link.to_node.name),
            str(link.to_socket.identifier),
        )
        for link in material.node_tree.links
    )
    result.update({"nodes": nodes, "links": links})
    return result


def material_graph_hash(mesh: Any) -> str:
    material_ids: dict[int, str] = {}
    payloads: dict[str, Any] = {}
    slots = []
    for material in mesh.data.materials:
        if material is None:
            slots.append(None)
            continue
        pointer = int(material.as_pointer())
        if pointer not in material_ids:
            identifier = f"material_{len(material_ids)}"
            material_ids[pointer] = identifier
            payloads[identifier] = _material_graph_payload(material)
        slots.append(material_ids[pointer])
    payload = {"slots": slots, "materials": payloads}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _create_skeleton_body_copy(bpy: Any, mesh: Any) -> tuple[Any, list[Any]]:
    clone = mesh.copy()
    clone.data = mesh.data.copy()
    clone.name = f"{mesh.name}_SkeletonReviewCopy"
    bpy.context.scene.collection.objects.link(clone)
    copied_by_pointer: dict[int, Any] = {}
    for index, material in enumerate(list(clone.data.materials)):
        if material is None:
            continue
        pointer = int(material.as_pointer())
        copied = copied_by_pointer.get(pointer)
        if copied is None:
            copied = material.copy()
            copied.name = f"{material.name}_SkeletonReviewCopy"
            copied_by_pointer[pointer] = copied
        clone.data.materials[index] = copied
    return clone, list(copied_by_pointer.values())


def validate_copied_body_transparency(clone: Any) -> dict[str, Any]:
    materials = {item for item in clone.data.materials if item is not None}
    if not materials:
        raise ReviewRenderError("skeleton body copy has no PBR materials")
    alpha_inputs = 0
    for material in materials:
        if not material.name.endswith("_SkeletonReviewCopy"):
            raise ReviewRenderError("skeleton body copy contains a non-copied PBR material")
        if not material.use_nodes or material.node_tree is None:
            raise ReviewRenderError("copied PBR material has no node graph")
        material_alpha_inputs = 0
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED" or node.inputs.get("Alpha") is None:
                continue
            alpha = node.inputs["Alpha"]
            material_alpha_inputs += 1
            alpha_inputs += 1
            if alpha.is_linked:
                raise ReviewRenderError("copied Principled Alpha remains texture-linked")
            if not math.isclose(float(alpha.default_value), 0.22, abs_tol=1.0e-6):
                raise ReviewRenderError("copied Principled Alpha is not effectively fixed at 0.22")
        if material_alpha_inputs == 0:
            raise ReviewRenderError("copied PBR material has no Principled Alpha path")
        if hasattr(material, "surface_render_method"):
            if material.surface_render_method != "DITHERED":
                raise ReviewRenderError("copied PBR material is not DITHERED")
        elif not hasattr(material, "blend_method") or material.blend_method != "BLEND":
            raise ReviewRenderError("copied PBR material has no transparent surface mode")
    links_removed = clone.get("_tokenrig_review_alpha_links_removed")
    if not isinstance(links_removed, int) or isinstance(links_removed, bool) or links_removed < 0:
        raise ReviewRenderError("copied PBR material is missing the Alpha-link removal audit")
    return {
        "passed": True,
        "material_count": len(materials),
        "principled_alpha_input_count": alpha_inputs,
        "linked_alpha_inputs_removed": links_removed,
        "effective_alpha": 0.22,
        "surface_render_method": (
            "DITHERED" if all(hasattr(item, "surface_render_method") for item in materials) else "BLEND"
        ),
    }


def _set_copied_body_transparency(clone: Any) -> dict[str, Any]:
    linked_alpha_inputs_removed = 0
    for material in {item for item in clone.data.materials if item is not None}:
        if not material.name.endswith("_SkeletonReviewCopy"):
            raise ReviewRenderError("refusing to modify a non-copied PBR material")
        if material.use_nodes and material.node_tree is not None:
            for node in material.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED" and node.inputs.get("Alpha") is not None:
                    alpha = node.inputs["Alpha"]
                    incoming = list(alpha.links)
                    linked_alpha_inputs_removed += len(incoming)
                    for link in incoming:
                        material.node_tree.links.remove(link)
                    alpha.default_value = 0.22
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        elif hasattr(material, "blend_method"):
            material.blend_method = "BLEND"
    clone["_tokenrig_review_alpha_links_removed"] = linked_alpha_inputs_removed
    return validate_copied_body_transparency(clone)


def _delete_skeleton_body_copy(
    bpy: Any, clone: Any, copied_materials: Sequence[Any]
) -> None:
    mesh_data = clone.data
    bpy.data.objects.remove(clone, do_unlink=True)
    if mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)
    for material in copied_materials:
        if material.users == 0:
            bpy.data.materials.remove(material)


def _setup_scene(bpy: Any, body_height: float) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x, scene.render.resolution_y = VIDEO_SIZE
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.world.color = (0.055, 0.065, 0.08)
    bpy.ops.mesh.primitive_plane_add(size=max(200.0, body_height * 30.0), location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "FixedFloorZ0"
    material = bpy.data.materials.new("FixedFloorMaterial")
    material.diffuse_color = (0.12, 0.14, 0.16, 1.0)
    floor.data.materials.append(material)
    for location, energy, size in (
        ((-3.0, -4.0, 6.0), 1100.0, 4.0),
        ((4.0, 1.0, 4.0), 750.0, 3.0),
        ((0.0, 5.0, 3.0), 450.0, 2.0),
    ):
        data = bpy.data.lights.new(f"ReviewArea{len(bpy.data.lights)}", "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(data.name, data)
        light.location = location
        scene.collection.objects.link(light)


def _create_camera(bpy: Any, view: str, bounds: Sequence[Mapping[str, Any]], body_height: float) -> Any:
    from mathutils import Vector

    data = bpy.data.cameras.new(f"ReviewCamera_{view}")
    data.lens = 55.0 if view != "feet" else 72.0
    data.sensor_fit = "VERTICAL"
    data.clip_start = 0.05
    camera = bpy.data.objects.new(data.name, data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    plans = plan_camera_keyframes(
        view=view,
        bounds=bounds,
        angle_x=float(data.angle_x),
        angle_y=float(data.angle_y),
        ndc_margin=0.08,
        clip_start=float(data.clip_start),
    )
    for frame, plan in zip(
        range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1), plans
    ):
        target = Vector(plan["target"])
        camera.location = Vector(plan["location"])
        _look_at(camera, target)
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
    if camera.animation_data and camera.animation_data.action:
        for curve in camera.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
    return camera


def _render_view(bpy: Any, *, staging: Path, motion: str, view: str, frame_start: int, frame_end: int, view_bounds: Sequence[Mapping[str, Any]], body_height: float, mesh: Any, skeleton: Sequence[Any]) -> dict[str, Any]:
    scene = bpy.context.scene
    camera = _create_camera(bpy, view, view_bounds, body_height)
    for overlay in skeleton:
        overlay.hide_render = view != "skeleton"
    original_material_graph = material_graph_hash(mesh)
    original_hide_render = bool(mesh.hide_render)
    skeleton_body_copy = None
    copied_materials: list[Any] = []
    copied_body_transparency = None
    if view == "skeleton":
        mesh.hide_render = True
    png_path = staging / f"{motion}_{view}.png"
    mp4_path = staging / f"{motion}_{view}.mp4"
    try:
        scene.frame_set(frame_start + (frame_end - frame_start) // 2)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(png_path)
        bpy.ops.render.render(write_still=True)
        scene.frame_start = frame_start
        scene.frame_end = frame_end
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.audio_codec = "NONE"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.filepath = str(mp4_path)
        bpy.ops.render.render(animation=True)
    finally:
        mesh.hide_render = original_hide_render
        if skeleton_body_copy is not None:
            _delete_skeleton_body_copy(bpy, skeleton_body_copy, copied_materials)
        material_graph_after_cleanup = material_graph_hash(mesh)
        bpy.data.objects.remove(camera, do_unlink=True)
        if material_graph_after_cleanup != original_material_graph:
            raise ReviewRenderError(
                f"original PBR material graph changed while rendering {motion}/{view}"
            )
    if not png_path.is_file() or not mp4_path.is_file():
        raise ReviewRenderError(f"Blender did not publish both PNG and MP4 for {motion}/{view}")
    frame_count = frame_end - frame_start + 1
    return {
        "png": probe_png(png_path),
        "mp4": probe_video(
            mp4_path,
            expected_frame_count=frame_count,
            motion=motion,
            view=view,
        ),
        "original_pbr_material_graph": {
            "before_sha256": original_material_graph,
            "after_sha256": material_graph_hash(mesh),
            "unchanged": True,
        },
        "copied_body_transparency": copied_body_transparency,
    }


def _render_motion(bpy: Any, *, glb_path: Path, motion: str, staging: Path, semantic_bones: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _clear_scene(bpy)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    armature, mesh, action = _scene_objects(bpy, MOTIONS[motion])
    frame_start, frame_end = _integer_frame_range(action)
    scene.frame_start, scene.frame_end = frame_start, frame_end
    bounds = _world_bounds(bpy, mesh, frame_start, frame_end)
    body_height = max(sample["maximum"][2] for sample in bounds) - min(sample["minimum"][2] for sample in bounds)
    if not math.isfinite(body_height) or body_height <= 0.2:
        raise ReviewRenderError("animated body height is invalid")
    foot_bounds = _foot_joint_bounds(
        bpy, armature, action, semantic_bones, body_height
    )
    _setup_scene(bpy, body_height)
    skeleton = _create_skeleton_overlay(
        bpy, armature, body_height, semantic_bones
    )
    material_graph_before = material_graph_hash(mesh)
    media_qa = {}
    for view in VIEWS:
        media_qa[view] = _render_view(
            bpy,
            staging=staging,
            motion=motion,
            view=view,
            frame_start=frame_start,
            frame_end=frame_end,
            view_bounds=foot_bounds if view == "feet" else bounds,
            body_height=body_height,
            mesh=mesh,
            skeleton=skeleton,
        )
    material_graph_after = material_graph_hash(mesh)
    if material_graph_after != material_graph_before:
        raise ReviewRenderError("original PBR material graph changed across review views")
    return (
        {
            "action_name": MOTIONS[motion],
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": frame_end - frame_start + 1,
            "fps": FPS,
            "duration_s": (frame_end - frame_start + 1) / FPS,
            "pbr_material_graph_sha256": material_graph_before,
            "pbr_material_graph_unchanged": True,
        },
        media_qa,
    )


def run_review_render(
    *,
    asset_id: str,
    display_label: str,
    instance_kind: str,
    static_qa_json: Path,
    retarget_manifest: Path,
    walking_glb: Path,
    standing_idle_glb: Path,
    output_dir: Path,
    command: Sequence[str] | None = None,
) -> Path:
    destination = validate_destination(output_dir)
    authenticated: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    staging: Path | None = None
    try:
        authenticated = authenticate_review_inputs(
            asset_id=asset_id,
            static_qa_json=static_qa_json,
            retarget_manifest=retarget_manifest,
            walking_glb=walking_glb,
            standing_idle_glb=standing_idle_glb,
        )
        execution = authenticate_execution_environment()
        import bpy

        blender_version = validate_blender_version(tuple(bpy.app.version))

        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".staging", dir=str(destination.parent)))
        actions: dict[str, Any] = {}
        media_qa_actions: dict[str, Any] = {}
        glbs = {"walking": Path(walking_glb), "standing_idle": Path(standing_idle_glb)}
        for motion in MOTIONS:
            actions[motion], media_qa_actions[motion] = _render_motion(
                bpy,
                glb_path=glbs[motion],
                motion=motion,
                staging=staging,
                semantic_bones=authenticated["semantic_mapping"]["semantic_bones"],
            )
        current = authenticate_review_inputs(
            asset_id=asset_id,
            static_qa_json=static_qa_json,
            retarget_manifest=retarget_manifest,
            walking_glb=walking_glb,
            standing_idle_glb=standing_idle_glb,
        )
        current_execution = authenticate_execution_environment()
        if current != authenticated:
            raise ReviewRenderError("upstream static or retarget snapshot changed during rendering")
        assert_execution_unchanged(execution, current_execution)
        media: dict[str, Any] = {}
        for motion in MOTIONS:
            media[motion] = {}
            for view in VIEWS:
                media[motion][view] = {
                    kind: file_record(staging / f"{motion}_{view}.{kind}", filename=f"{motion}_{view}.{kind}")
                    for kind in ("png", "mp4")
                }
        media_qa = {
            "schema": MEDIA_QA_SCHEMA,
            "asset_id": asset_id,
            "fps": FPS,
            "resolution": list(VIDEO_SIZE),
            "actions": media_qa_actions,
            "automatic_checks": "passed",
        }
        media_qa_path = staging / "media_qa.json"
        _write_json_exclusive(media_qa_path, media_qa)
        manifest = build_review_manifest(
            asset_id=asset_id,
            display_label=display_label,
            instance_kind=instance_kind,
            authenticated=authenticated,
            actions=actions,
            media=media,
            media_qa_record=file_record(media_qa_path, filename="media_qa.json"),
            execution=execution,
            blender_version=blender_version,
            command=list(command if command is not None else sys.argv),
        )
        _write_json_exclusive(staging / "review_manifest.json", manifest)
        expected = {"media_qa.json", "review_manifest.json"}
        expected.update(f"{motion}_{view}.{kind}" for motion in MOTIONS for view in VIEWS for kind in ("png", "mp4"))
        if {item.name for item in staging.iterdir()} != expected:
            raise ReviewRenderError("staged dynamic review bundle has missing or unexpected files")
        for path in sorted(staging.iterdir()):
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise ReviewRenderError(f"invalid staged media artifact: {path.name}")
            _fsync_file(path)
            path.chmod(0o444)
        _fsync_directory(staging)
        staging.chmod(0o555)
        rename_directory_noreplace(staging, destination)
        staging = None
        _fsync_directory(destination.parent)
        return destination / "review_manifest.json"
    except BaseException as error:
        if staging is not None and staging.exists():
            staging.chmod(0o700)
            for path in staging.iterdir():
                path.chmod(0o600)
            shutil.rmtree(staging)
        evidence = write_failure_evidence(
            destination=destination,
            asset_id=asset_id,
            error=error,
            authenticated={
                "upstream": authenticated,
                "execution": execution,
            },
        )
        raise ReviewRenderError(f"dynamic media render rejected; evidence={evidence}: {error}") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--display-label", required=True)
    parser.add_argument("--instance-kind", required=True)
    parser.add_argument("--static-qa-json", type=Path, required=True)
    parser.add_argument("--retarget-manifest", type=Path, required=True)
    parser.add_argument("--walking-glb", type=Path, required=True)
    parser.add_argument("--standing-idle-glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_review_render(
        asset_id=args.asset_id,
        display_label=args.display_label,
        instance_kind=args.instance_kind,
        static_qa_json=args.static_qa_json,
        retarget_manifest=args.retarget_manifest,
        walking_glb=args.walking_glb,
        standing_idle_glb=args.standing_idle_glb,
        output_dir=args.output_dir,
    )
    print(f"TOKENRIG_HUMAN_DYNAMIC_REVIEW_OK {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
