#!/usr/bin/env python3
"""Compose one authenticated v4 quadruped review into a fixed six-view video.

This is a presentation-only bridge.  It authenticates the exact v4 review
bytes, every authoritative review MP4, and every render/encode receipt before
and after FFmpeg runs.  It never records an owner decision and never grants
formal dataset registration authority.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
import errno
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import stat
import struct
import subprocess
import sys
from typing import Any


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.controlled_source_asset_schema import (  # noqa: E402
    StrictJSONError,
    strict_json_loads,
)
from tools.run_target_native_generated_quadruped_review import (  # noqa: E402
    REVIEW_MEDIA_FPS,
    REVIEW_MEDIA_HEIGHT,
    REVIEW_MEDIA_SPECS,
    REVIEW_MEDIA_WIDTH,
    SCHEMA as REVIEW_SCHEMA,
    file_record,
    render_frame_set,
    require_encode_manifest,
    require_render_manifest,
)


PRESENTATION_SCHEMA = (
    "avengine_target_native_generated_quadruped_owner_review_presentation_v1"
)
PRESENTATION_TOOL_VERSION = "1"
PRESENTATION_STATUS = "presentation_ready_pending_owner_review"
CONTENT_READBACK_SCHEMA = (
    "avengine_target_native_generated_quadruped_cell_content_readback_v1"
)
OUTPUT_VIDEO_NAME = "owner_review_six_view.mp4"
RECEIPT_NAME = "presentation_receipt.json"
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
LOGICAL_TMP_ROOT = SPEAR_ROOT / "tmp"
PHYSICAL_TMP_ROOT = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp"
)

MEDIA_LAYOUT = (
    ("walking_side", "Walking", "side", 0.0, "WALKING / SIDE", 0, 0),
    ("walking_front", "Walking", "front", 0.0, "WALKING / FRONT", 0, 1),
    ("walking_rear", "Walking", "front", 180.0, "WALKING / REAR", 0, 2),
    ("idle_side", "Idle", "side", 0.0, "IDLE / SIDE", 1, 0),
    ("idle_front", "Idle", "front", 0.0, "IDLE / FRONT", 1, 1),
    ("idle_rear", "Idle", "front", 180.0, "IDLE / REAR", 1, 2),
)
MEDIA_LABELS = tuple(item[0] for item in MEDIA_LAYOUT)
XSTACK_LAYOUT = "0_0|512_0|1024_0|0_384|512_384|1024_384"
OUTPUT_WIDTH = REVIEW_MEDIA_WIDTH * 3
OUTPUT_HEIGHT = REVIEW_MEDIA_HEIGHT * 2
REVIEW_FRAME_COUNT = 8
CONTENT_PIXEL_FORMAT = "gray"
CONTENT_MEAN_ABSOLUTE_ERROR_MAX = 4.0
CONTENT_ROOT_MEAN_SQUARE_ERROR_MAX = 8.0
CONTENT_MAX_ABSOLUTE_ERROR_MAX = 80
FILE_RECORD_FIELDS = frozenset({"path", "sha256", "size_bytes"})
VIDEO_RECORD_FIELDS = FILE_RECORD_FIELDS | frozenset(
    {
        "codec",
        "width",
        "height",
        "frame_count",
        "frame_rate",
        "duration_seconds",
    }
)
REVIEW_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "created_at",
        "status",
        "formal_dataset_registration_authorized",
        "forward_contract",
        "inputs",
        "pipeline_order",
        "automatic_admission_gates",
        "outputs",
        "timings_seconds",
    }
)
REVIEW_GATE_FIELDS = frozenset(
    {
        "heading",
        "rig",
        "support_plane",
        "retarget_export_front_axis",
        "gait_initial",
        "deformation_initial",
        "weight_repair_policy",
        "weight_repair_triggered",
        "weight_repair",
        "weight_repair_strategy",
        "weight_repair_branch",
        "weight_repair_attempts",
        "weight_repair_final_artifact",
        "gait_final",
        "deformation_final",
        "all_automatic_gates_passed",
    }
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PROBE_STREAM_FIELDS = frozenset(
    {
        "index",
        "codec_name",
        "codec_type",
        "width",
        "height",
        "sample_aspect_ratio",
        "pix_fmt",
        "r_frame_rate",
        "avg_frame_rate",
        "nb_frames",
        "nb_read_frames",
    }
)
FILE_GUARD_FIELDS = frozenset(
    {
        "path",
        "device",
        "inode",
        "mode",
        "link_count",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
    }
)
AUTOMATIC_CHECK_FIELDS = frozenset(
    {
        "external_source_review_sha256_authenticated",
        "review_schema_status_and_automatic_gates_authenticated",
        "six_source_media_and_lineage_receipts_authenticated",
        "all_render_frame_bindings_authenticated",
        "source_unique_video_streams_authenticated",
        "source_counted_frames_authenticated",
        "source_full_decodes_authenticated",
        "private_snapshot_hashes_authenticated",
        "private_snapshot_counted_frames_authenticated",
        "private_snapshot_full_decodes_authenticated",
        "private_snapshots_unchanged_after_composition",
        "source_graph_unchanged_after_snapshot",
        "source_graph_unchanged_after_composition",
        "source_restore_race_guards_unchanged",
        "fixed_source_order_authenticated",
        "toolchain_identity_and_versions_unchanged",
        "output_unique_video_stream_authenticated",
        "output_counted_frames_authenticated",
        "output_full_decode_authenticated",
        "all_output_frame_cell_content_authenticated",
        "source_review_unmodified",
        "formal_dataset_registration_authority_not_granted",
    }
)
CONTENT_READBACK_FIELDS = frozenset(
    {
        "schema",
        "method",
        "reference_argv",
        "observed_argv",
        "reference_gray_frames_sha256",
        "observed_gray_frames_sha256",
        "cells",
        "all_cells_all_frames_passed",
        "content_readback_sha256",
    }
)
RECEIPT_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "created_at",
        "status",
        "authority",
        "expected_source_review_sha256",
        "source_review",
        "reviewed_animation",
        "authenticated_inputs",
        "source_authority_guards",
        "source_authority_guard_sha256",
        "source_order",
        "source_order_sha256",
        "source_set",
        "source_set_sha256",
        "private_composition_inputs",
        "frame_cell_content_readback",
        "presentation_contract",
        "automatic_checks",
        "toolchain",
        "command",
        "output",
        "receipt_sha256",
    }
)


class PresentationContractError(RuntimeError):
    """The v4 review cannot safely be presented."""


@dataclass
class HeldDirectory:
    """One directory inode held open across every security-sensitive operation."""

    path: Path
    name: str
    descriptor: int
    device: int
    inode: int
    owner_uid: int
    owner_gid: int
    mode: int
    closed: bool = False

    def close(self) -> None:
        if not self.closed:
            os.close(self.descriptor)
            self.closed = True


@dataclass
class OutputLocation:
    root: Path
    output_name: str
    parent: HeldDirectory
    review_root: Path


@dataclass
class StagingInventory:
    """Exact known staging entries; anything else forces quarantine."""

    top_level_files: dict[str, dict] = dataclass_field(default_factory=dict)
    snapshot_directory: dict | None = None
    snapshot_files: dict[str, dict] = dataclass_field(default_factory=dict)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-run", type=Path, required=True)
    parser.add_argument(
        "--expected-review-run-sha256",
        required=True,
        help="External SHA-256 authority for the exact review_run.json bytes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="A new directory; existing paths and symlinks are rejected.",
    )
    return parser.parse_args(argv)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def hash_without(payload: dict, field: str) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_components(path: Path) -> list[Path]:
    absolute = _lexical_absolute(path)
    components = []
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        components.append(current)
    return components


def _allowed_workspace_tmp_link(component: Path) -> bool:
    if component != LOGICAL_TMP_ROOT or not component.is_symlink():
        return False
    try:
        target = Path(os.readlink(component))
    except OSError:
        return False
    if not target.is_absolute():
        target = component.parent / target
    try:
        return os.path.samefile(target, PHYSICAL_TMP_ROOT)
    except OSError:
        return False


def _reject_unsafe_symlinks(path: Path, label: str, *, file_leaf: bool) -> None:
    components = _path_components(path)
    for index, component in enumerate(components):
        if not component.is_symlink():
            continue
        is_leaf = index == len(components) - 1
        if _allowed_workspace_tmp_link(component) and not (file_leaf and is_leaf):
            continue
        raise PresentationContractError(
            f"{label} contains an unsafe symlink component: {component}"
        )


def _require_tmp_bridge_containment(lexical: Path, resolved: Path, label: str) -> None:
    if LOGICAL_TMP_ROOT not in _path_components(lexical):
        return
    try:
        resolved.relative_to(PHYSICAL_TMP_ROOT)
    except ValueError as error:
        raise PresentationContractError(
            f"{label} escaped the exact SPEAR/tmp bridge"
        ) from error


def resolve_regular_file(path: Path, label: str) -> Path:
    lexical = _lexical_absolute(path)
    _reject_unsafe_symlinks(lexical, label, file_leaf=True)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise PresentationContractError(f"missing {label}: {lexical}") from error
    _require_tmp_bridge_containment(lexical, resolved, label)
    resolved_stat = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISREG(resolved_stat.st_mode) or resolved_stat.st_size <= 0:
        raise PresentationContractError(f"missing or empty {label}: {resolved}")
    return resolved


def resolve_directory(path: Path, label: str) -> Path:
    lexical = _lexical_absolute(path)
    _reject_unsafe_symlinks(lexical, label, file_leaf=False)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise PresentationContractError(f"missing {label}: {lexical}") from error
    _require_tmp_bridge_containment(lexical, resolved, label)
    resolved_stat = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISDIR(resolved_stat.st_mode):
        raise PresentationContractError(f"missing {label}: {resolved}")
    return resolved


def file_guard(path: Path, label: str) -> dict:
    path = resolve_regular_file(path, label)
    current = os.stat(path, follow_symlinks=False)
    return {
        "path": str(path),
        "device": current.st_dev,
        "inode": current.st_ino,
        "mode": stat.S_IMODE(current.st_mode),
        "link_count": current.st_nlink,
        "size_bytes": current.st_size,
        "mtime_ns": current.st_mtime_ns,
        "ctime_ns": current.st_ctime_ns,
    }


def require_same_guard(expected: dict, path: Path, label: str) -> None:
    observed = file_guard(path, label)
    if observed != expected:
        raise PresentationContractError(f"{label} identity changed during use")


def resolve_named_executable(name: str) -> Path:
    resolved = shutil.which(name)
    if resolved is None:
        raise PresentationContractError(f"required executable is missing: {name}")
    path = resolve_regular_file(Path(resolved), f"{name} executable")
    if not os.access(path, os.X_OK):
        raise PresentationContractError(
            f"required executable is not executable: {path}"
        )
    return path


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise PresentationContractError(
            f"{label} must be one lowercase 64-character SHA-256 digest"
        )
    return value


def require_exact_file_record(
    value: Any,
    path: Path,
    label: str,
) -> dict:
    if not isinstance(value, dict) or set(value) != FILE_RECORD_FIELDS:
        raise PresentationContractError(f"{label} file record fields changed")
    before = file_guard(path, label)
    expected = file_record(path)
    require_same_guard(before, path, label)
    if value != expected:
        raise PresentationContractError(f"{label} file binding changed")
    return expected


def read_stable_bytes(path: Path, label: str) -> tuple[bytes, dict]:
    path = resolve_regular_file(path, label)
    before = file_guard(path, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before["device"]
            or opened.st_ino != before["inode"]
            or opened.st_size != before["size_bytes"]
        ):
            raise PresentationContractError(f"{label} changed before it was opened")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        encoded = b"".join(chunks)
        closed = os.fstat(descriptor)
        if (
            closed.st_dev != before["device"]
            or closed.st_ino != before["inode"]
            or closed.st_size != before["size_bytes"]
        ):
            raise PresentationContractError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    require_same_guard(before, path, label)
    return encoded, before


def read_exact_review(
    path: Path,
    expected_sha256: str,
) -> tuple[dict, dict]:
    expected_sha256 = require_sha256(expected_sha256, "expected review-run SHA-256")
    path = resolve_regular_file(path, "v4 review run")
    try:
        encoded, guard = read_stable_bytes(path, "v4 review run")
    except OSError as error:
        raise PresentationContractError(
            f"could not read v4 review run: {path}"
        ) from error
    observed_sha256 = sha256_bytes(encoded)
    if observed_sha256 != expected_sha256:
        raise PresentationContractError(
            "v4 review run failed external SHA-256 authentication"
        )
    try:
        payload = strict_json_loads(encoded)
    except StrictJSONError as error:
        raise PresentationContractError(
            f"v4 review run is not strict JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise PresentationContractError("v4 review run must be a JSON object")
    return payload, {
        "path": str(path),
        "sha256": observed_sha256,
        "size_bytes": len(encoded),
        "file_guard": guard,
    }


def build_ffprobe_argv(ffprobe: Path, video: Path) -> list[str]:
    return [
        str(ffprobe),
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,pix_fmt,width,height,"
            "sample_aspect_ratio,r_frame_rate,avg_frame_rate,nb_frames,"
            "nb_read_frames:format=duration"
        ),
        "-of",
        "json",
        str(video),
    ]


def build_decode_argv(ffmpeg: Path, video: Path) -> list[str]:
    return [
        str(ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]


def probe_video(
    path: Path,
    *,
    ffprobe: Path,
    ffmpeg: Path,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
    expected_fps: int,
) -> dict:
    path = resolve_regular_file(path, "review video")
    guard = file_guard(path, "review video")
    argv = build_ffprobe_argv(ffprobe, path)
    result = subprocess.run(
        argv,
        cwd=SPEAR_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = strict_json_loads(result.stdout)
    except StrictJSONError as error:
        raise PresentationContractError(
            f"FFprobe returned ambiguous JSON for {path}"
        ) from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"programs", "streams", "format"}
        or payload.get("programs") != []
        or not isinstance(payload.get("streams"), list)
        or len(payload["streams"]) != 1
        or not isinstance(payload["streams"][0], dict)
        or not isinstance(payload.get("format"), dict)
        or set(payload["format"]) != {"duration"}
    ):
        raise PresentationContractError(
            f"FFprobe stream coverage is not exact for {path}"
        )
    stream = payload["streams"][0]
    if set(stream) != PROBE_STREAM_FIELDS:
        raise PresentationContractError(
            f"FFprobe video stream fields changed for {path}"
        )
    try:
        width = int(stream["width"])
        height = int(stream["height"])
        frame_count = int(stream["nb_frames"])
        decoded_frame_count = int(stream["nb_read_frames"])
        frame_rate = Fraction(str(stream["r_frame_rate"]))
        average_rate = Fraction(str(stream["avg_frame_rate"]))
        duration = float(payload["format"]["duration"])
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise PresentationContractError(
            f"FFprobe video metadata is invalid for {path}"
        ) from error
    if (
        stream.get("index") != 0
        or stream.get("codec_type") != "video"
        or stream.get("codec_name") != "h264"
        or stream.get("pix_fmt") != "yuv420p"
        or stream.get("sample_aspect_ratio") != "1:1"
        or width != expected_width
        or height != expected_height
        or frame_count != expected_frames
        or decoded_frame_count != expected_frames
        or frame_rate != expected_fps
        or average_rate != expected_fps
        or not math.isfinite(duration)
        or abs(duration - expected_frames / expected_fps) > 1.0 / expected_fps
    ):
        raise PresentationContractError(f"FFprobe video readback failed for {path}")
    decode_argv = build_decode_argv(ffmpeg, path)
    subprocess.run(
        decode_argv,
        cwd=SPEAR_ROOT,
        check=True,
        capture_output=True,
    )
    require_same_guard(guard, path, "review video")
    video = {
        **file_record(path),
        "codec": "h264",
        "width": expected_width,
        "height": expected_height,
        "frame_count": expected_frames,
        "frame_rate": stream["r_frame_rate"],
        "duration_seconds": duration,
    }
    require_same_guard(guard, path, "review video")
    return {
        "video": video,
        "readback": {
            "stream_count": 1,
            "stream_index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "1:1",
            "width": width,
            "height": height,
            "r_frame_rate": stream["r_frame_rate"],
            "avg_frame_rate": stream["avg_frame_rate"],
            "nb_frames": frame_count,
            "nb_read_frames": decoded_frame_count,
            "duration_seconds": duration,
        },
        "ffprobe_argv": argv,
        "full_decode": {
            "argv": decode_argv,
            "passed": True,
        },
        "file_guard": guard,
    }


def _review_media_root(review_path: Path) -> Path:
    return review_path.parent / "05_review"


def _require_review_shape(review: dict) -> tuple[dict, dict, dict]:
    if set(review) != REVIEW_TOP_LEVEL_FIELDS:
        raise PresentationContractError("v4 review top-level fields changed")
    gates = review.get("automatic_admission_gates")
    outputs = review.get("outputs")
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("status") != "research_candidate_pending_human_review"
        or review.get("formal_dataset_registration_authorized") is not False
        or not isinstance(review.get("created_at"), str)
        or not isinstance(review.get("forward_contract"), dict)
        or not isinstance(review.get("inputs"), dict)
        or not isinstance(review.get("pipeline_order"), list)
        or not isinstance(review.get("timings_seconds"), dict)
        or not isinstance(gates, dict)
        or set(gates) != REVIEW_GATE_FIELDS
        or gates.get("all_automatic_gates_passed") is not True
        or not isinstance(outputs, dict)
    ):
        raise PresentationContractError(
            "v4 review is not an admitted candidate pending human review"
        )
    media = outputs.get("media")
    lineage = outputs.get("media_lineage")
    if (
        not isinstance(media, dict)
        or set(media) != set(MEDIA_LABELS)
        or not isinstance(lineage, dict)
        or set(lineage) != set(MEDIA_LABELS)
    ):
        raise PresentationContractError(
            "v4 review does not contain exactly six authoritative media entries"
        )
    return outputs, media, lineage


def authenticate_review(
    review_path: Path,
    expected_review_sha256: str,
    *,
    ffprobe: Path,
    ffmpeg: Path,
) -> dict:
    review, review_record = read_exact_review(review_path, expected_review_sha256)
    review_guard = review_record.pop("file_guard")
    review_path = Path(review_record["path"])
    outputs, media, lineage = _require_review_shape(review)
    authority_guards = {"review_run": review_guard}
    animated_record = outputs.get("animated_glb")
    if (
        not isinstance(animated_record, dict)
        or set(animated_record) != FILE_RECORD_FIELDS
        or not isinstance(animated_record.get("path"), str)
    ):
        raise PresentationContractError("v4 review animated GLB descriptor changed")
    animated_glb = resolve_regular_file(
        Path(animated_record["path"]), "reviewed animated GLB"
    )
    animated_guard = file_guard(animated_glb, "reviewed animated GLB")
    authenticated_animated = require_exact_file_record(
        animated_record, animated_glb, "reviewed animated GLB"
    )
    authority_guards["animated_glb"] = animated_guard

    expected_specs = tuple(
        (label, action, view, float(yaw))
        for label, action, view, yaw in REVIEW_MEDIA_SPECS
    )
    presentation_specs = tuple(
        (label, action, view, float(yaw))
        for label, action, view, yaw, _title, _row, _column in MEDIA_LAYOUT
    )
    if expected_specs != presentation_specs:
        raise PresentationContractError(
            "runner review-media contract no longer matches presentation layout"
        )

    review_media_root = resolve_directory(
        _review_media_root(review_path), "v4 review media root"
    )
    authenticated_media = {}
    for (
        label,
        action,
        view,
        yaw,
        title,
        row,
        column,
    ) in MEDIA_LAYOUT:
        media_record = media[label]
        lineage_record = lineage[label]
        if (
            not isinstance(media_record, dict)
            or set(media_record) != VIDEO_RECORD_FIELDS
            or not isinstance(media_record.get("path"), str)
            or not isinstance(lineage_record, dict)
            or set(lineage_record) != {"render_manifest", "encode_manifest"}
        ):
            raise PresentationContractError(
                f"v4 review media descriptor fields changed for {label}"
            )
        video = resolve_regular_file(Path(media_record["path"]), f"{label} review MP4")
        render_record = lineage_record["render_manifest"]
        encode_record = lineage_record["encode_manifest"]
        if (
            not isinstance(render_record, dict)
            or set(render_record) != FILE_RECORD_FIELDS
            or not isinstance(render_record.get("path"), str)
            or not isinstance(encode_record, dict)
            or set(encode_record) != FILE_RECORD_FIELDS
            or not isinstance(encode_record.get("path"), str)
        ):
            raise PresentationContractError(
                f"v4 review receipt descriptors changed for {label}"
            )
        render_manifest = resolve_regular_file(
            Path(render_record["path"]), f"{label} render receipt"
        )
        encode_manifest = resolve_regular_file(
            Path(encode_record["path"]), f"{label} encode receipt"
        )
        frame_dir = resolve_directory(
            review_media_root / f"{label}_frames", f"{label} frame directory"
        )
        expected_paths = {
            "video": review_media_root / f"{label}.mp4",
            "render": review_media_root / f"{label}_render_manifest.json",
            "encode": review_media_root / f"{label}_encode_manifest.json",
        }
        if (
            video != expected_paths["video"]
            or render_manifest != expected_paths["render"]
            or encode_manifest != expected_paths["encode"]
        ):
            raise PresentationContractError(
                f"v4 review media paths changed for {label}"
            )
        render_guard = file_guard(render_manifest, f"{label} render receipt")
        encode_guard = file_guard(encode_manifest, f"{label} encode receipt")
        frame_guards = [
            file_guard(
                frame_dir / f"frame_{index:04d}.png",
                f"{label} frame {index}",
            )
            for index in range(REVIEW_FRAME_COUNT)
        ]
        require_exact_file_record(
            render_record, render_manifest, f"{label} render receipt"
        )
        require_exact_file_record(
            encode_record, encode_manifest, f"{label} encode receipt"
        )
        render_payload = require_render_manifest(
            render_manifest,
            input_glb=animated_glb,
            frame_dir=frame_dir,
            action=action,
            view=view,
            asset_yaw_deg=yaw,
            n_frames=REVIEW_FRAME_COUNT,
        )
        video_readback = probe_video(
            video,
            ffprobe=ffprobe,
            ffmpeg=ffmpeg,
            expected_width=REVIEW_MEDIA_WIDTH,
            expected_height=REVIEW_MEDIA_HEIGHT,
            expected_frames=REVIEW_FRAME_COUNT,
            expected_fps=REVIEW_MEDIA_FPS,
        )
        if video_readback["video"] != media_record:
            raise PresentationContractError(
                f"v4 review MP4 descriptor contradicts FFprobe for {label}"
            )
        encode_payload = require_encode_manifest(
            encode_manifest,
            label=label,
            render_manifest_path=render_manifest,
            render_payload=render_payload,
            frame_dir=frame_dir,
            video_path=video,
            video_record=video_readback["video"],
            action=action,
            view=view,
            asset_yaw_deg=yaw,
            n_frames=REVIEW_FRAME_COUNT,
        )
        require_same_guard(render_guard, render_manifest, f"{label} render receipt")
        require_same_guard(encode_guard, encode_manifest, f"{label} encode receipt")
        for index, guard in enumerate(frame_guards):
            require_same_guard(
                guard,
                frame_dir / f"frame_{index:04d}.png",
                f"{label} frame {index}",
            )
        authority_guards[f"media.{label}.video"] = video_readback["file_guard"]
        authority_guards[f"media.{label}.render_manifest"] = render_guard
        authority_guards[f"media.{label}.encode_manifest"] = encode_guard
        for index, guard in enumerate(frame_guards):
            authority_guards[f"media.{label}.frame.{index:04d}"] = guard
        authenticated_media[label] = {
            "title": title,
            "row": row,
            "column": column,
            "video": video_readback["video"],
            "render_manifest": dict(render_record),
            "encode_manifest": dict(encode_record),
            "frame_set": render_frame_set(render_payload),
            "source_encode_ffmpeg": encode_payload["ffmpeg"],
            "video_probe": {
                "readback": video_readback["readback"],
                "ffprobe_argv": video_readback["ffprobe_argv"],
                "full_decode": video_readback["full_decode"],
            },
        }
    require_same_guard(review_guard, review_path, "v4 review run")
    require_same_guard(animated_guard, animated_glb, "reviewed animated GLB")
    return {
        "review_run": review_record,
        "animated_glb": authenticated_animated,
        "media": authenticated_media,
        "authority_guards": authority_guards,
        "authority_guard_sha256": canonical_json_sha256(authority_guards),
    }


def build_filter_complex(font: Path) -> str:
    filters = []
    for index, item in enumerate(MEDIA_LAYOUT):
        title = item[4]
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,"
            "drawbox=x=0:y=0:w=iw:h=44:color=black@0.68:t=fill,"
            f"drawtext=fontfile={font}:text='{title}':"
            "x=12:y=8:fontsize=24:fontcolor=white[v"
            f"{index}]"
        )
    inputs = "".join(f"[v{index}]" for index in range(len(MEDIA_LAYOUT)))
    filters.append(
        f"{inputs}xstack=inputs=6:layout={XSTACK_LAYOUT}:fill=black:shortest=1[outv]"
    )
    return ";".join(filters)


def presentation_contract() -> dict:
    return {
        "layout": "fixed_three_columns_by_two_rows",
        "xstack_layout": XSTACK_LAYOUT,
        "tile": {
            "width": REVIEW_MEDIA_WIDTH,
            "height": REVIEW_MEDIA_HEIGHT,
        },
        "canvas": {
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
        },
        "frame_count": REVIEW_FRAME_COUNT,
        "fps": REVIEW_MEDIA_FPS,
        "cells": [
            {
                "label": label,
                "title": title,
                "row": row,
                "column": column,
                "x": column * REVIEW_MEDIA_WIDTH,
                "y": row * REVIEW_MEDIA_HEIGHT,
            }
            for (
                label,
                _action,
                _view,
                _yaw,
                title,
                row,
                column,
            ) in MEDIA_LAYOUT
        ],
        "input_isolation": "private_authenticated_snapshots_only",
        "publication": "sealed_staging_atomic_rename_noreplace",
    }


def build_ffmpeg_argv(
    *,
    ffmpeg: Path,
    font: Path,
    videos: list[Path],
    output: Path,
) -> list[str]:
    if len(videos) != len(MEDIA_LAYOUT):
        raise PresentationContractError(
            "presentation requires exactly six ordered videos"
        )
    argv = [str(ffmpeg), "-nostdin", "-n", "-loglevel", "error"]
    for video in videos:
        argv.extend(["-i", str(video)])
    argv.extend(
        [
            "-filter_complex",
            build_filter_complex(font),
            "-map",
            "[outv]",
            "-an",
            "-frames:v",
            str(REVIEW_FRAME_COUNT),
            "-r",
            str(REVIEW_MEDIA_FPS),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            str(output),
        ]
    )
    return argv


def build_reference_rawvideo_argv(
    *,
    ffmpeg: Path,
    font: Path,
    videos: list[Path],
) -> list[str]:
    if len(videos) != len(MEDIA_LAYOUT):
        raise PresentationContractError(
            "content reference requires exactly six ordered videos"
        )
    argv = [str(ffmpeg), "-nostdin", "-v", "error", "-xerror"]
    for video in videos:
        argv.extend(["-i", str(video)])
    argv.extend(
        [
            "-filter_complex",
            build_filter_complex(font),
            "-map",
            "[outv]",
            "-frames:v",
            str(REVIEW_FRAME_COUNT),
            "-pix_fmt",
            CONTENT_PIXEL_FORMAT,
            "-f",
            "rawvideo",
            "-",
        ]
    )
    return argv


def build_observed_rawvideo_argv(
    *,
    ffmpeg: Path,
    video: Path,
) -> list[str]:
    return [
        str(ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-frames:v",
        str(REVIEW_FRAME_COUNT),
        "-vf",
        f"format={CONTENT_PIXEL_FORMAT}",
        "-pix_fmt",
        CONTENT_PIXEL_FORMAT,
        "-f",
        "rawvideo",
        "-",
    ]


def _run_exact_rawvideo(argv: list[str]) -> bytes:
    result = subprocess.run(
        argv,
        cwd=SPEAR_ROOT,
        check=True,
        capture_output=True,
    )
    expected_size = OUTPUT_WIDTH * OUTPUT_HEIGHT * REVIEW_FRAME_COUNT
    if not isinstance(result.stdout, bytes) or len(result.stdout) != expected_size:
        raise PresentationContractError(
            "decoded content readback byte coverage is not exact"
        )
    return result.stdout


def _gray_tile(frame: bytes, row: int, column: int) -> bytes:
    rows = []
    x = column * REVIEW_MEDIA_WIDTH
    y = row * REVIEW_MEDIA_HEIGHT
    for tile_y in range(REVIEW_MEDIA_HEIGHT):
        start = (y + tile_y) * OUTPUT_WIDTH + x
        rows.append(frame[start : start + REVIEW_MEDIA_WIDTH])
    tile = b"".join(rows)
    if len(tile) != REVIEW_MEDIA_WIDTH * REVIEW_MEDIA_HEIGHT:
        raise PresentationContractError("decoded cell byte coverage is not exact")
    return tile


def _cell_difference(reference: bytes, observed: bytes) -> dict:
    if len(reference) != len(observed) or not reference:
        raise PresentationContractError("decoded cell samples are inconsistent")
    absolute_sum = 0
    squared_sum = 0
    maximum = 0
    for expected, actual in zip(reference, observed):
        difference = abs(expected - actual)
        absolute_sum += difference
        squared_sum += difference * difference
        maximum = max(maximum, difference)
    count = len(reference)
    mean_absolute_error = round(absolute_sum / count, 6)
    root_mean_square_error = round(math.sqrt(squared_sum / count), 6)
    passed = (
        mean_absolute_error <= CONTENT_MEAN_ABSOLUTE_ERROR_MAX
        and root_mean_square_error <= CONTENT_ROOT_MEAN_SQUARE_ERROR_MAX
        and maximum <= CONTENT_MAX_ABSOLUTE_ERROR_MAX
    )
    return {
        "mean_absolute_error": mean_absolute_error,
        "root_mean_square_error": root_mean_square_error,
        "max_absolute_error": maximum,
        "passed": passed,
    }


def content_readback_method() -> dict:
    return {
        "reference": (
            "fixed_filter_graph_decoded_from_authenticated_private_snapshots"
        ),
        "observed": "full_output_decode",
        "pixel_format": CONTENT_PIXEL_FORMAT,
        "frame_count": REVIEW_FRAME_COUNT,
        "canvas": {
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
        },
        "tile": {
            "width": REVIEW_MEDIA_WIDTH,
            "height": REVIEW_MEDIA_HEIGHT,
        },
        "thresholds": {
            "mean_absolute_error_max": CONTENT_MEAN_ABSOLUTE_ERROR_MAX,
            "root_mean_square_error_max": (CONTENT_ROOT_MEAN_SQUARE_ERROR_MAX),
            "max_absolute_error_max": CONTENT_MAX_ABSOLUTE_ERROR_MAX,
        },
    }


def audit_frame_cell_content(
    *,
    ffmpeg: Path,
    font: Path,
    snapshots: list[Path],
    output_video: Path,
    snapshot_bindings: list[dict],
) -> dict:
    if len(snapshots) != len(MEDIA_LAYOUT) or len(snapshot_bindings) != len(
        MEDIA_LAYOUT
    ):
        raise PresentationContractError(
            "cell-content readback source set is incomplete"
        )
    output_guard = file_guard(output_video, "six-view presentation")
    for index, (snapshot, binding) in enumerate(zip(snapshots, snapshot_bindings)):
        if (
            binding.get("ordinal") != index
            or binding.get("label") != MEDIA_LABELS[index]
        ):
            raise PresentationContractError(
                "cell-content readback source order changed"
            )
        require_same_guard(
            binding["private_snapshot_file_guard"],
            snapshot,
            f"{MEDIA_LABELS[index]} private snapshot",
        )

    reference_argv = build_reference_rawvideo_argv(
        ffmpeg=ffmpeg,
        font=font,
        videos=snapshots,
    )
    observed_argv = build_observed_rawvideo_argv(
        ffmpeg=ffmpeg,
        video=output_video,
    )
    reference = _run_exact_rawvideo(reference_argv)
    observed = _run_exact_rawvideo(observed_argv)

    require_same_guard(output_guard, output_video, "six-view presentation")
    for index, (snapshot, binding) in enumerate(zip(snapshots, snapshot_bindings)):
        require_same_guard(
            binding["private_snapshot_file_guard"],
            snapshot,
            f"{MEDIA_LABELS[index]} private snapshot",
        )

    frame_size = OUTPUT_WIDTH * OUTPUT_HEIGHT
    cell_results = []
    all_passed = True
    for label, _action, _view, _yaw, title, row, column in MEDIA_LAYOUT:
        frames = []
        for frame_index in range(REVIEW_FRAME_COUNT):
            frame_start = frame_index * frame_size
            frame_end = frame_start + frame_size
            reference_tile = _gray_tile(reference[frame_start:frame_end], row, column)
            observed_tile = _gray_tile(observed[frame_start:frame_end], row, column)
            difference = _cell_difference(reference_tile, observed_tile)
            all_passed = all_passed and difference["passed"]
            frames.append(
                {
                    "frame_index": frame_index,
                    "reference_gray_sha256": sha256_bytes(reference_tile),
                    "observed_gray_sha256": sha256_bytes(observed_tile),
                    **difference,
                }
            )
        cell_results.append(
            {
                "label": label,
                "title": title,
                "row": row,
                "column": column,
                "frames": frames,
                "all_frames_passed": all(frame["passed"] for frame in frames),
            }
        )
    if not all_passed:
        failed = [
            f"{cell['label']}:{frame['frame_index']}"
            for cell in cell_results
            for frame in cell["frames"]
            if not frame["passed"]
        ]
        raise PresentationContractError(
            "frame/cell content readback failed: " + ",".join(failed)
        )
    result = {
        "schema": CONTENT_READBACK_SCHEMA,
        "method": content_readback_method(),
        "reference_argv": reference_argv,
        "observed_argv": observed_argv,
        "reference_gray_frames_sha256": sha256_bytes(reference),
        "observed_gray_frames_sha256": sha256_bytes(observed),
        "cells": cell_results,
        "all_cells_all_frames_passed": True,
        "content_readback_sha256": None,
    }
    result["content_readback_sha256"] = hash_without(result, "content_readback_sha256")
    return result


def _versioned_executable(executable: Path, label: str) -> dict:
    guard = file_guard(executable, f"{label} executable")
    argv = [str(executable), "-version"]
    result = subprocess.run(
        argv,
        cwd=SPEAR_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    require_same_guard(guard, executable, f"{label} executable")
    combined = result.stdout + result.stderr
    nonempty_lines = [line.strip() for line in combined.splitlines() if line.strip()]
    if not nonempty_lines:
        raise PresentationContractError(f"{label} returned no version identity")
    return {
        "executable": file_record(executable),
        "file_guard": guard,
        "version_argv": argv,
        "version_first_line": nonempty_lines[0],
        "version_output_sha256": sha256_bytes(combined.encode("utf-8")),
    }


def _decode_ttf_name(platform_id: int, encoded: bytes) -> str:
    if platform_id in {0, 3}:
        return encoded.decode("utf-16-be")
    if platform_id == 1:
        return encoded.decode("mac_roman")
    return encoded.decode("latin-1")


def _font_identity(font: Path) -> dict:
    encoded, guard = read_stable_bytes(font, "presentation font")
    if len(encoded) < 12:
        raise PresentationContractError("presentation font has no SFNT header")
    sfnt_version = encoded[:4].hex()
    try:
        table_count = struct.unpack_from(">H", encoded, 4)[0]
    except struct.error as error:
        raise PresentationContractError(
            "presentation font SFNT header is invalid"
        ) from error
    name_offset = None
    name_length = None
    for index in range(table_count):
        record_offset = 12 + index * 16
        if record_offset + 16 > len(encoded):
            raise PresentationContractError(
                "presentation font table directory is truncated"
            )
        tag, _checksum, offset, length = struct.unpack_from(
            ">4sIII", encoded, record_offset
        )
        if tag == b"name":
            name_offset = offset
            name_length = length
    if (
        name_offset is None
        or name_length is None
        or name_offset + name_length > len(encoded)
        or name_length < 6
    ):
        raise PresentationContractError("presentation font has no valid name table")
    try:
        _format, name_count, string_offset = struct.unpack_from(
            ">HHH", encoded, name_offset
        )
    except struct.error as error:
        raise PresentationContractError(
            "presentation font name table is invalid"
        ) from error
    storage = name_offset + string_offset
    candidates = {}
    for index in range(name_count):
        record_offset = name_offset + 6 + index * 12
        if record_offset + 12 > name_offset + name_length:
            raise PresentationContractError(
                "presentation font name records are truncated"
            )
        platform_id, _encoding_id, language_id, name_id, length, offset = (
            struct.unpack_from(">HHHHHH", encoded, record_offset)
        )
        if name_id not in {1, 2, 5, 6}:
            continue
        start = storage + offset
        end = start + length
        if start < storage or end > name_offset + name_length:
            raise PresentationContractError(
                "presentation font name string escaped its table"
            )
        try:
            value = _decode_ttf_name(platform_id, encoded[start:end]).strip("\x00 ")
        except UnicodeError:
            continue
        if not value:
            continue
        preference = (
            0 if platform_id == 3 and language_id in {0x0409, 0} else 1,
            platform_id,
            language_id,
        )
        previous = candidates.get(name_id)
        if previous is None or preference < previous[0]:
            candidates[name_id] = (preference, value)
    if set(candidates) != {1, 2, 5, 6}:
        raise PresentationContractError(
            "presentation font identity/version names are incomplete"
        )
    require_same_guard(guard, font, "presentation font")
    return {
        "file": file_record(font),
        "file_guard": guard,
        "sfnt_version_hex": sfnt_version,
        "family": candidates[1][1],
        "subfamily": candidates[2][1],
        "version": candidates[5][1],
        "postscript_name": candidates[6][1],
    }


def capture_runtime_identity(
    *,
    ffmpeg: Path,
    ffprobe: Path,
    font: Path,
) -> dict:
    tool = resolve_regular_file(Path(__file__), "presentation tool")
    tool_guard = file_guard(tool, "presentation tool")
    tool_record = file_record(tool)
    require_same_guard(tool_guard, tool, "presentation tool")
    return {
        "presentation_tool": {
            "version": PRESENTATION_TOOL_VERSION,
            "file": tool_record,
            "file_guard": tool_guard,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "ffmpeg": _versioned_executable(ffmpeg, "ffmpeg"),
        "ffprobe": _versioned_executable(ffprobe, "ffprobe"),
        "font": _font_identity(font),
    }


def _opened_file_matches_guard(opened: os.stat_result, expected: dict) -> bool:
    return (
        opened.st_dev == expected["device"]
        and opened.st_ino == expected["inode"]
        and stat.S_IMODE(opened.st_mode) == expected["mode"]
        and opened.st_nlink == expected["link_count"]
        and opened.st_size == expected["size_bytes"]
        and opened.st_mtime_ns == expected["mtime_ns"]
        and opened.st_ctime_ns == expected["ctime_ns"]
    )


def copy_private_snapshot(
    source: Path,
    destination: Path,
    *,
    expected_source_guard: dict,
    expected_source_video: dict,
) -> dict:
    source = resolve_regular_file(source, "source review MP4")
    require_same_guard(expected_source_guard, source, "source review MP4")
    destination = _lexical_absolute(destination)
    _reject_unsafe_symlinks(destination, "private snapshot", file_leaf=True)
    if os.path.lexists(destination):
        raise PresentationContractError(
            f"refusing to replace private snapshot: {destination}"
        )
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_descriptor = os.open(source, source_flags)
    try:
        if not _opened_file_matches_guard(
            os.fstat(source_descriptor), expected_source_guard
        ):
            raise PresentationContractError(
                "source review MP4 changed before snapshot copy"
            )
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        try:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                offset = 0
                while offset < len(chunk):
                    offset += os.write(destination_descriptor, chunk[offset:])
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
        if not _opened_file_matches_guard(
            os.fstat(source_descriptor), expected_source_guard
        ):
            raise PresentationContractError(
                "source review MP4 changed during snapshot copy"
            )
    finally:
        os.close(source_descriptor)
    require_same_guard(expected_source_guard, source, "source review MP4")
    destination.chmod(0o400)
    snapshot = file_record(resolve_regular_file(destination, "private snapshot"))
    if (
        snapshot["sha256"] != expected_source_video["sha256"]
        or snapshot["size_bytes"] != expected_source_video["size_bytes"]
    ):
        raise PresentationContractError(
            "private snapshot does not match its authenticated source MP4"
        )
    return snapshot


def _video_content_contract(video: dict) -> dict:
    return {key: value for key, value in video.items() if key != "path"}


def snapshot_authenticated_media(
    authenticated: dict,
    snapshot_root: Path,
    *,
    ffmpeg: Path,
    ffprobe: Path,
) -> tuple[list[Path], list[dict]]:
    snapshot_root.mkdir(mode=0o700)
    videos = []
    source_order = []
    for index, label in enumerate(MEDIA_LABELS):
        source_entry = authenticated["media"][label]
        source_video = source_entry["video"]
        source_path = Path(source_video["path"])
        snapshot_path = snapshot_root / f"{index:02d}_{label}.mp4"
        snapshot_record = copy_private_snapshot(
            source_path,
            snapshot_path,
            expected_source_guard=authenticated["authority_guards"][
                f"media.{label}.video"
            ],
            expected_source_video=source_video,
        )
        snapshot_probe = probe_video(
            snapshot_path,
            ffprobe=ffprobe,
            ffmpeg=ffmpeg,
            expected_width=REVIEW_MEDIA_WIDTH,
            expected_height=REVIEW_MEDIA_HEIGHT,
            expected_frames=REVIEW_FRAME_COUNT,
            expected_fps=REVIEW_MEDIA_FPS,
        )
        if _video_content_contract(snapshot_probe["video"]) != _video_content_contract(
            source_video
        ):
            raise PresentationContractError(
                f"private snapshot video contract changed for {label}"
            )
        if snapshot_probe["video"] != {
            **snapshot_record,
            **{
                key: value
                for key, value in source_video.items()
                if key not in FILE_RECORD_FIELDS
            },
        }:
            raise PresentationContractError(
                f"private snapshot binding changed for {label}"
            )
        videos.append(snapshot_path)
        source_order.append(
            {
                "ordinal": index,
                "label": label,
                "source": source_video,
                "source_probe": source_entry["video_probe"],
                "private_snapshot": snapshot_probe["video"],
                "private_snapshot_probe": {
                    "readback": snapshot_probe["readback"],
                    "ffprobe_argv": snapshot_probe["ffprobe_argv"],
                    "full_decode": snapshot_probe["full_decode"],
                },
                "private_snapshot_file_guard": snapshot_probe["file_guard"],
            }
        )
    return videos, source_order


def reauthenticate_private_snapshots(
    videos: list[Path],
    snapshot_bindings: list[dict],
    *,
    ffmpeg: Path,
    ffprobe: Path,
) -> None:
    if len(videos) != len(MEDIA_LAYOUT) or len(snapshot_bindings) != len(MEDIA_LAYOUT):
        raise PresentationContractError(
            "private snapshot authentication set is incomplete"
        )
    for index, (video, binding) in enumerate(zip(videos, snapshot_bindings)):
        label = MEDIA_LABELS[index]
        if (
            binding.get("ordinal") != index
            or binding.get("label") != label
            or binding.get("private_snapshot", {}).get("path") != str(video)
        ):
            raise PresentationContractError(
                f"private snapshot order changed for {label}"
            )
        observed = probe_video(
            video,
            ffprobe=ffprobe,
            ffmpeg=ffmpeg,
            expected_width=REVIEW_MEDIA_WIDTH,
            expected_height=REVIEW_MEDIA_HEIGHT,
            expected_frames=REVIEW_FRAME_COUNT,
            expected_fps=REVIEW_MEDIA_FPS,
        )
        if (
            observed["video"] != binding["private_snapshot"]
            or observed["readback"] != binding["private_snapshot_probe"]["readback"]
            or observed["file_guard"] != binding["private_snapshot_file_guard"]
        ):
            raise PresentationContractError(
                f"private snapshot changed during composition for {label}"
            )


def write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_ffmpeg(argv: list[str]) -> None:
    subprocess.run(argv, cwd=SPEAR_ROOT, check=True)


def _require_child_name(name: str, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    ):
        raise PresentationContractError(f"{label} is not one safe child name")
    return name


def _directory_guard(current: os.stat_result) -> dict:
    if not stat.S_ISDIR(current.st_mode):
        raise PresentationContractError("held object is not a directory")
    return {
        "device": current.st_dev,
        "inode": current.st_ino,
        "owner_uid": current.st_uid,
        "owner_gid": current.st_gid,
        "mode": stat.S_IMODE(current.st_mode),
    }


def _held_directory(
    *,
    path: Path,
    name: str,
    descriptor: int,
    expected: os.stat_result,
    label: str,
) -> HeldDirectory:
    opened = os.fstat(descriptor)
    expected_guard = _directory_guard(expected)
    opened_guard = _directory_guard(opened)
    if opened_guard != expected_guard:
        raise PresentationContractError(f"{label} changed while it was opened")
    return HeldDirectory(
        path=path,
        name=name,
        descriptor=descriptor,
        device=opened_guard["device"],
        inode=opened_guard["inode"],
        owner_uid=opened_guard["owner_uid"],
        owner_gid=opened_guard["owner_gid"],
        mode=opened_guard["mode"],
    )


def _require_held_directory(
    directory: HeldDirectory,
    label: str,
    *,
    expected_mode: int | None = None,
) -> os.stat_result:
    if directory.closed:
        raise PresentationContractError(f"{label} descriptor is closed")
    current = os.fstat(directory.descriptor)
    guard = _directory_guard(current)
    if (
        guard["device"] != directory.device
        or guard["inode"] != directory.inode
        or guard["owner_uid"] != directory.owner_uid
        or guard["owner_gid"] != directory.owner_gid
        or (expected_mode is not None and guard["mode"] != expected_mode)
    ):
        raise PresentationContractError(f"{label} descriptor identity changed")
    return current


def _require_secure_output_parent(current: os.stat_result) -> None:
    mode = stat.S_IMODE(current.st_mode)
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_uid != os.geteuid()
        or current.st_gid != os.getegid()
        or mode & 0o700 != 0o700
        or mode & 0o002
    ):
        raise PresentationContractError("output parent owner/mode policy failed")


def _require_outside_review_tree(root: Path, review_root: Path) -> None:
    if root == review_root or review_root in root.parents:
        raise PresentationContractError(
            "presentation output root must be outside the immutable review run"
        )


def _require_parent_path_binding(location: OutputLocation) -> None:
    parent = resolve_directory(location.parent.path, "output parent")
    current = os.stat(parent, follow_symlinks=False)
    _require_secure_output_parent(current)
    expected = _require_held_directory(
        location.parent,
        "output parent",
        expected_mode=location.parent.mode,
    )
    if _directory_guard(current) != _directory_guard(expected):
        raise PresentationContractError("output parent path identity changed")
    _require_outside_review_tree(
        parent / location.output_name,
        location.review_root,
    )


def _directory_entry_matches(
    parent_descriptor: int,
    name: str,
    directory: HeldDirectory,
) -> bool:
    try:
        current = os.stat(
            _require_child_name(name, "directory entry"),
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    return (
        stat.S_ISDIR(current.st_mode)
        and current.st_dev == directory.device
        and current.st_ino == directory.inode
        and current.st_uid == directory.owner_uid
        and current.st_gid == directory.owner_gid
    )


def _file_guard_from_stat(path: Path, current: os.stat_result) -> dict:
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_size <= 0
        or current.st_nlink != 1
    ):
        raise PresentationContractError(
            f"staging artifact is not one private regular file: {path}"
        )
    return {
        "path": str(path),
        "device": current.st_dev,
        "inode": current.st_ino,
        "mode": stat.S_IMODE(current.st_mode),
        "link_count": current.st_nlink,
        "size_bytes": current.st_size,
        "mtime_ns": current.st_mtime_ns,
        "ctime_ns": current.st_ctime_ns,
    }


def _file_guard_at(
    directory_descriptor: int,
    name: str,
    path: Path,
    label: str,
) -> dict:
    name = _require_child_name(name, label)
    before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    guard = _file_guard_from_stat(path, before)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        opened = _file_guard_from_stat(path, os.fstat(descriptor))
        if opened != guard:
            raise PresentationContractError(f"{label} changed while it was opened")
    finally:
        os.close(descriptor)
    after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if _file_guard_from_stat(path, after) != guard:
        raise PresentationContractError(f"{label} changed during authentication")
    return guard


def _require_file_guard_at(
    directory_descriptor: int,
    name: str,
    path: Path,
    expected: dict,
    label: str,
) -> None:
    observed = _file_guard_at(directory_descriptor, name, path, label)
    if observed != expected:
        raise PresentationContractError(f"{label} identity changed")


def _read_stable_bytes_at(
    directory_descriptor: int,
    name: str,
    path: Path,
    label: str,
) -> tuple[bytes, dict]:
    name = _require_child_name(name, label)
    before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    guard = _file_guard_from_stat(path, before)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        opened = _file_guard_from_stat(path, os.fstat(descriptor))
        if opened != guard:
            raise PresentationContractError(f"{label} changed before it was read")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        closed = _file_guard_from_stat(path, os.fstat(descriptor))
        if closed != guard:
            raise PresentationContractError(f"{label} changed while it was read")
    finally:
        os.close(descriptor)
    _require_file_guard_at(directory_descriptor, name, path, guard, label)
    return b"".join(chunks), guard


def _file_record_at(
    directory_descriptor: int,
    name: str,
    path: Path,
    label: str,
) -> dict:
    encoded, _guard = _read_stable_bytes_at(
        directory_descriptor,
        name,
        path,
        label,
    )
    return {
        "path": str(path),
        "sha256": sha256_bytes(encoded),
        "size_bytes": len(encoded),
    }


def prepare_output_location(path: Path, review_path: Path) -> OutputLocation:
    lexical = _lexical_absolute(path)
    _reject_unsafe_symlinks(lexical, "output root", file_leaf=False)
    if os.path.lexists(lexical):
        raise PresentationContractError(f"refusing to replace output root: {lexical}")
    output_name = _require_child_name(lexical.name, "output root name")
    review_root = review_path.resolve(strict=True).parent
    _require_outside_review_tree(lexical.resolve(strict=False), review_root)

    # Requiring an existing parent avoids a check-then-create write through an
    # ancestor that can be replaced between lexical validation and mkdir().
    parent = resolve_directory(lexical.parent, "output parent")
    root = parent / output_name
    _require_outside_review_tree(root, review_root)
    parent_before = os.stat(parent, follow_symlinks=False)
    _require_secure_output_parent(parent_before)
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(parent, flags)
    try:
        held_parent = _held_directory(
            path=parent,
            name=parent.name,
            descriptor=descriptor,
            expected=parent_before,
            label="output parent",
        )
        location = OutputLocation(
            root=root,
            output_name=output_name,
            parent=held_parent,
            review_root=review_root,
        )
        _require_parent_path_binding(location)
        try:
            os.stat(
                output_name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return location
        raise PresentationContractError(f"refusing to replace output root: {root}")
    except Exception:
        os.close(descriptor)
        raise


def create_private_staging(location: OutputLocation) -> HeldDirectory:
    parent = location.parent
    _require_parent_path_binding(location)
    _require_held_directory(parent, "output parent", expected_mode=parent.mode)
    for _attempt in range(128):
        name = f".{location.output_name}.{secrets.token_hex(16)}.staging"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        break
    else:
        raise PresentationContractError("could not allocate unique private staging")

    descriptor = None
    staging = None
    try:
        before = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, flags, dir_fd=parent.descriptor)
        staging = _held_directory(
            path=parent.path / name,
            name=name,
            descriptor=descriptor,
            expected=before,
            label="private staging root",
        )
        os.fchmod(descriptor, 0o700)
        staging.mode = 0o700
        _require_held_directory(
            staging,
            "private staging root",
            expected_mode=0o700,
        )
        if not _directory_entry_matches(parent.descriptor, name, staging):
            raise PresentationContractError(
                "private staging parent entry identity changed"
            )
        os.fsync(parent.descriptor)
        return staging
    except Exception:
        if descriptor is not None and staging is not None:
            try:
                if _directory_entry_matches(
                    parent.descriptor, name, staging
                ) and not os.listdir(descriptor):
                    os.rmdir(name, dir_fd=parent.descriptor)
            except OSError:
                pass
            finally:
                os.close(descriptor)
        elif descriptor is not None:
            os.close(descriptor)
        raise


def _snapshot_directory_guard(
    staging: HeldDirectory,
    name: str,
) -> tuple[dict, int]:
    name = _require_child_name(name, "private snapshot directory")
    current = os.stat(
        name,
        dir_fd=staging.descriptor,
        follow_symlinks=False,
    )
    guard = _directory_guard(current)
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=staging.descriptor)
    if _directory_guard(os.fstat(descriptor)) != guard:
        os.close(descriptor)
        raise PresentationContractError(
            "private snapshot directory changed while it was opened"
        )
    return guard, descriptor


def register_snapshot_tree(
    staging: HeldDirectory,
    inventory: StagingInventory,
    snapshot_root: Path,
    videos: list[Path],
    snapshot_bindings: list[dict],
) -> None:
    if (
        inventory.snapshot_directory is not None
        or inventory.snapshot_files
        or len(videos) != len(MEDIA_LAYOUT)
        or len(snapshot_bindings) != len(MEDIA_LAYOUT)
    ):
        raise PresentationContractError("private snapshot registration set is invalid")
    directory_name = _require_child_name(
        snapshot_root.name,
        "private snapshot directory",
    )
    directory_guard, descriptor = _snapshot_directory_guard(
        staging,
        directory_name,
    )
    try:
        expected_names = {video.name for video in videos}
        if set(os.listdir(descriptor)) != expected_names:
            raise PresentationContractError("private snapshot artifact set changed")
        registered = {}
        for video, binding in zip(videos, snapshot_bindings):
            expected = binding["private_snapshot_file_guard"]
            observed = _file_guard_at(
                descriptor,
                video.name,
                video,
                "private snapshot",
            )
            if observed != expected:
                raise PresentationContractError("private snapshot fd binding changed")
            registered[video.name] = observed
    finally:
        os.close(descriptor)
    inventory.snapshot_directory = {
        "name": directory_name,
        **directory_guard,
    }
    inventory.snapshot_files = registered


def register_top_level_file(
    staging: HeldDirectory,
    inventory: StagingInventory,
    name: str,
    path: Path,
    label: str,
    *,
    expected: dict | None = None,
) -> dict:
    if name in inventory.top_level_files:
        raise PresentationContractError(f"{label} was registered twice")
    guard = _file_guard_at(staging.descriptor, name, path, label)
    if expected is not None and guard != expected:
        raise PresentationContractError(f"{label} fd binding changed")
    inventory.top_level_files[name] = guard
    return guard


def remove_private_snapshots(
    staging: HeldDirectory,
    inventory: StagingInventory,
) -> None:
    directory_guard = inventory.snapshot_directory
    if directory_guard is None or len(inventory.snapshot_files) != len(MEDIA_LAYOUT):
        raise PresentationContractError(
            "private snapshot cleanup inventory is incomplete"
        )
    name = directory_guard["name"]
    observed_guard, descriptor = _snapshot_directory_guard(staging, name)
    try:
        if observed_guard != {
            key: directory_guard[key]
            for key in ("device", "inode", "owner_uid", "owner_gid", "mode")
        }:
            raise PresentationContractError(
                "private snapshot directory identity changed"
            )
        if set(os.listdir(descriptor)) != set(inventory.snapshot_files):
            raise PresentationContractError(
                "private snapshot cleanup found unknown artifacts"
            )
        for filename, guard in inventory.snapshot_files.items():
            _require_file_guard_at(
                descriptor,
                filename,
                Path(guard["path"]),
                guard,
                "private snapshot",
            )
        for filename, guard in inventory.snapshot_files.items():
            _require_file_guard_at(
                descriptor,
                filename,
                Path(guard["path"]),
                guard,
                "private snapshot",
            )
            os.unlink(filename, dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.rmdir(name, dir_fd=staging.descriptor)
    os.fsync(staging.descriptor)
    inventory.snapshot_directory = None
    inventory.snapshot_files.clear()


def write_json_exclusive_at(
    staging: HeldDirectory,
    inventory: StagingInventory,
    name: str,
    path: Path,
    payload: dict,
) -> dict:
    name = _require_child_name(name, "presentation receipt")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=staging.descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return register_top_level_file(
        staging,
        inventory,
        name,
        path,
        "presentation receipt",
    )


def require_receipt_self_hash_at(
    staging: HeldDirectory,
    name: str,
    path: Path,
    expected: dict,
) -> None:
    encoded, guard = _read_stable_bytes_at(
        staging.descriptor,
        name,
        path,
        "presentation receipt",
    )
    try:
        observed = strict_json_loads(encoded)
    except StrictJSONError as error:
        raise PresentationContractError(
            "presentation receipt is not strict JSON"
        ) from error
    if observed != expected:
        raise PresentationContractError(
            "presentation receipt changed after serialization"
        )
    validate_presentation_receipt(observed)
    _require_file_guard_at(
        staging.descriptor,
        name,
        path,
        guard,
        "presentation receipt",
    )


def seal_readonly_tree(
    staging: HeldDirectory,
    inventory: StagingInventory,
) -> None:
    _require_held_directory(
        staging,
        "publication staging root",
        expected_mode=0o700,
    )
    if inventory.snapshot_directory is not None or inventory.snapshot_files:
        raise PresentationContractError(
            "private snapshots remain in publication staging"
        )
    if set(inventory.top_level_files) != {OUTPUT_VIDEO_NAME, RECEIPT_NAME}:
        raise PresentationContractError("publication artifact inventory changed")
    if set(os.listdir(staging.descriptor)) != set(inventory.top_level_files):
        raise PresentationContractError("publication artifact set changed")
    for name, guard in inventory.top_level_files.items():
        _require_file_guard_at(
            staging.descriptor,
            name,
            Path(guard["path"]),
            guard,
            "publication artifact",
        )
    for name, guard in tuple(inventory.top_level_files.items()):
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=staging.descriptor)
        try:
            opened = _file_guard_from_stat(Path(guard["path"]), os.fstat(descriptor))
            if opened != guard:
                raise PresentationContractError(
                    "publication artifact changed before sealing"
                )
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
        inventory.top_level_files[name] = _file_guard_at(
            staging.descriptor,
            name,
            Path(guard["path"]),
            "publication artifact",
        )
    os.fsync(staging.descriptor)
    os.fchmod(staging.descriptor, 0o555)
    staging.mode = 0o555
    os.fsync(staging.descriptor)
    _require_held_directory(
        staging,
        "publication staging root",
        expected_mode=0o555,
    )


def require_readonly_publication_fd(
    staging: HeldDirectory,
    inventory: StagingInventory,
) -> None:
    _require_held_directory(
        staging,
        "sealed publication",
        expected_mode=0o555,
    )
    if (
        inventory.snapshot_directory is not None
        or inventory.snapshot_files
        or set(inventory.top_level_files) != {OUTPUT_VIDEO_NAME, RECEIPT_NAME}
        or set(os.listdir(staging.descriptor)) != set(inventory.top_level_files)
    ):
        raise PresentationContractError("publication artifact set changed")
    for name, guard in inventory.top_level_files.items():
        observed = _file_guard_at(
            staging.descriptor,
            name,
            Path(guard["path"]),
            "sealed publication artifact",
        )
        if observed != guard or observed["mode"] != 0o444:
            raise PresentationContractError(
                "publication artifact is not sealed read-only"
            )


def require_readonly_publication(root: Path) -> None:
    root = resolve_directory(root, "sealed publication")
    if stat.S_IMODE(os.stat(root, follow_symlinks=False).st_mode) != 0o555:
        raise PresentationContractError("publication root is not read-only")
    expected_names = {OUTPUT_VIDEO_NAME, RECEIPT_NAME}
    children = list(root.iterdir())
    if {child.name for child in children} != expected_names:
        raise PresentationContractError("publication artifact set changed")
    for child in children:
        child = resolve_regular_file(child, "sealed publication artifact")
        current = os.stat(child, follow_symlinks=False)
        if stat.S_IMODE(current.st_mode) != 0o444 or current.st_nlink != 1:
            raise PresentationContractError(
                f"publication artifact is not sealed read-only: {child}"
            )


def _renameat2_no_replace(
    source_directory_descriptor: int,
    source_name: str,
    target_directory_descriptor: int,
    target_name: str,
    *,
    target_display: Path,
) -> None:
    source_name = _require_child_name(source_name, "rename source")
    target_name = _require_child_name(target_name, "rename target")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise PresentationContractError(
            "atomic no-replace directory publication is unavailable"
        ) from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    rename_noreplace = 1
    result = renameat2(
        source_directory_descriptor,
        os.fsencode(source_name),
        target_directory_descriptor,
        os.fsencode(target_name),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise PresentationContractError(
            f"refusing to replace output root: {target_display}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise PresentationContractError(
            "atomic no-replace directory publication is unsupported"
        )
    raise OSError(error_number, os.strerror(error_number), str(target_display))


def atomic_publish_no_replace(
    location: OutputLocation,
    staging: HeldDirectory,
) -> None:
    _require_held_directory(
        location.parent,
        "output parent",
        expected_mode=location.parent.mode,
    )
    _require_held_directory(
        staging,
        "publication staging root",
        expected_mode=0o555,
    )
    if not _directory_entry_matches(
        location.parent.descriptor,
        staging.name,
        staging,
    ):
        raise PresentationContractError("private staging parent entry identity changed")
    _renameat2_no_replace(
        location.parent.descriptor,
        staging.name,
        location.parent.descriptor,
        location.output_name,
        target_display=location.root,
    )


def _find_held_directory_entry(
    parent: HeldDirectory,
    directory: HeldDirectory,
) -> str | None:
    matches = []
    for name in os.listdir(parent.descriptor):
        try:
            current = os.stat(
                name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        if (
            stat.S_ISDIR(current.st_mode)
            and current.st_dev == directory.device
            and current.st_ino == directory.inode
        ):
            matches.append(name)
    if len(matches) > 1:
        raise PresentationContractError(
            "held staging directory has ambiguous parent entries"
        )
    return matches[0] if matches else None


def quarantine_staging(
    location: OutputLocation,
    staging: HeldDirectory,
) -> str:
    _require_held_directory(staging, "quarantined staging")
    os.fchmod(staging.descriptor, 0o700)
    staging.mode = 0o700
    entry_name = _find_held_directory_entry(location.parent, staging)
    if entry_name is None:
        return "quarantined_detached_inode"
    for _attempt in range(128):
        quarantine_name = f".quarantine.{location.output_name}.{secrets.token_hex(16)}"
        try:
            _renameat2_no_replace(
                location.parent.descriptor,
                entry_name,
                location.parent.descriptor,
                quarantine_name,
                target_display=location.parent.path / quarantine_name,
            )
        except PresentationContractError as error:
            if "refusing to replace output root" in str(error):
                continue
            raise
        staging.name = quarantine_name
        staging.path = location.parent.path / quarantine_name
        os.fsync(location.parent.descriptor)
        return "quarantined"
    return "quarantined_name_allocation_failed"


def _preflight_known_staging_cleanup(
    staging: HeldDirectory,
    inventory: StagingInventory,
) -> None:
    expected_names = set(inventory.top_level_files)
    if inventory.snapshot_directory is not None:
        expected_names.add(inventory.snapshot_directory["name"])
    if set(os.listdir(staging.descriptor)) != expected_names:
        raise PresentationContractError(
            "staging cleanup found unknown or missing artifacts"
        )
    for name, guard in inventory.top_level_files.items():
        _require_file_guard_at(
            staging.descriptor,
            name,
            Path(guard["path"]),
            guard,
            "known staging artifact",
        )
    if inventory.snapshot_directory is None:
        if inventory.snapshot_files:
            raise PresentationContractError(
                "staging cleanup snapshot inventory is inconsistent"
            )
        return
    name = inventory.snapshot_directory["name"]
    observed_guard, descriptor = _snapshot_directory_guard(staging, name)
    try:
        expected_guard = {
            key: inventory.snapshot_directory[key]
            for key in ("device", "inode", "owner_uid", "owner_gid", "mode")
        }
        if observed_guard != expected_guard:
            raise PresentationContractError("known snapshot directory identity changed")
        if set(os.listdir(descriptor)) != set(inventory.snapshot_files):
            raise PresentationContractError(
                "snapshot cleanup found unknown or missing artifacts"
            )
        for filename, guard in inventory.snapshot_files.items():
            _require_file_guard_at(
                descriptor,
                filename,
                Path(guard["path"]),
                guard,
                "known private snapshot",
            )
    finally:
        os.close(descriptor)


def cleanup_unpublished_staging(
    location: OutputLocation,
    staging: HeldDirectory,
    inventory: StagingInventory,
) -> str:
    """Remove only a completely authenticated known tree, otherwise quarantine."""

    try:
        _require_held_directory(staging, "unpublished staging")
        entry_name = _find_held_directory_entry(location.parent, staging)
        if entry_name != staging.name:
            return quarantine_staging(location, staging)
        _preflight_known_staging_cleanup(staging, inventory)
        os.fchmod(staging.descriptor, 0o700)
        staging.mode = 0o700
        if inventory.snapshot_directory is not None:
            remove_private_snapshots(staging, inventory)
        for name, guard in tuple(inventory.top_level_files.items()):
            _require_file_guard_at(
                staging.descriptor,
                name,
                Path(guard["path"]),
                guard,
                "known staging artifact",
            )
            os.unlink(name, dir_fd=staging.descriptor)
            del inventory.top_level_files[name]
        os.fsync(staging.descriptor)
        if os.listdir(staging.descriptor):
            return quarantine_staging(location, staging)
        if not _directory_entry_matches(
            location.parent.descriptor,
            staging.name,
            staging,
        ):
            return quarantine_staging(location, staging)
        os.rmdir(staging.name, dir_fd=location.parent.descriptor)
        os.fsync(location.parent.descriptor)
        return "removed"
    except (OSError, PresentationContractError):
        try:
            return quarantine_staging(location, staging)
        except (OSError, PresentationContractError):
            return "quarantined_in_place"


def require_published_binding(
    location: OutputLocation,
    staging: HeldDirectory,
) -> None:
    _require_held_directory(
        location.parent,
        "output parent",
        expected_mode=location.parent.mode,
    )
    _require_held_directory(
        staging,
        "published output root",
        expected_mode=0o555,
    )
    if not _directory_entry_matches(
        location.parent.descriptor,
        location.output_name,
        staging,
    ):
        raise PresentationContractError(
            "published output parent entry identity changed"
        )
    _require_parent_path_binding(location)
    published_path = resolve_directory(location.root, "published output root")
    current = os.stat(published_path, follow_symlinks=False)
    if (
        current.st_dev != staging.device
        or current.st_ino != staging.inode
        or current.st_uid != staging.owner_uid
        or current.st_gid != staging.owner_gid
        or stat.S_IMODE(current.st_mode) != 0o555
    ):
        raise PresentationContractError("published output path identity changed")


def _require_record_shape(value: Any, fields: frozenset, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise PresentationContractError(f"{label} fields changed")
    if (
        not isinstance(value.get("path"), str)
        or not value["path"]
        or not Path(value["path"]).is_absolute()
        or require_sha256(value.get("sha256"), f"{label} SHA-256") != value["sha256"]
        or isinstance(value.get("size_bytes"), bool)
        or not isinstance(value.get("size_bytes"), int)
        or value["size_bytes"] <= 0
    ):
        raise PresentationContractError(f"{label} is invalid")
    return value


def _require_file_guard_shape(value: Any, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != FILE_GUARD_FIELDS:
        raise PresentationContractError(f"{label} guard fields changed")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise PresentationContractError(f"{label} guard path is invalid")
    for field in FILE_GUARD_FIELDS - {"path"}:
        if isinstance(value.get(field), bool) or not isinstance(value.get(field), int):
            raise PresentationContractError(f"{label} guard is invalid")
    if (
        value["device"] < 0
        or value["inode"] <= 0
        or value["mode"] < 0
        or value["link_count"] <= 0
        or value["size_bytes"] <= 0
        or value["mtime_ns"] <= 0
        or value["ctime_ns"] <= 0
    ):
        raise PresentationContractError(f"{label} guard is invalid")
    return value


def _require_rate(value: Any, expected: int, label: str) -> Fraction:
    try:
        observed = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise PresentationContractError(f"{label} is invalid") from error
    if observed != expected:
        raise PresentationContractError(f"{label} changed")
    return observed


def _require_video_record_shape(
    value: Any,
    *,
    width: int,
    height: int,
    label: str,
) -> dict:
    video = _require_record_shape(value, VIDEO_RECORD_FIELDS, label)
    duration = video.get("duration_seconds")
    if (
        video.get("codec") != "h264"
        or video.get("width") != width
        or video.get("height") != height
        or video.get("frame_count") != REVIEW_FRAME_COUNT
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or abs(duration - REVIEW_FRAME_COUNT / REVIEW_MEDIA_FPS)
        > 1.0 / REVIEW_MEDIA_FPS
    ):
        raise PresentationContractError(f"{label} video contract changed")
    _require_rate(video.get("frame_rate"), REVIEW_MEDIA_FPS, f"{label} frame rate")
    return video


def _require_video_readback_shape(
    value: Any,
    *,
    width: int,
    height: int,
    label: str,
) -> dict:
    expected_fields = {
        "stream_count",
        "stream_index",
        "codec_type",
        "codec_name",
        "pix_fmt",
        "sample_aspect_ratio",
        "width",
        "height",
        "r_frame_rate",
        "avg_frame_rate",
        "nb_frames",
        "nb_read_frames",
        "duration_seconds",
    }
    duration = value.get("duration_seconds") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("stream_count") != 1
        or value.get("stream_index") != 0
        or value.get("codec_type") != "video"
        or value.get("codec_name") != "h264"
        or value.get("pix_fmt") != "yuv420p"
        or value.get("sample_aspect_ratio") != "1:1"
        or value.get("width") != width
        or value.get("height") != height
        or value.get("nb_frames") != REVIEW_FRAME_COUNT
        or value.get("nb_read_frames") != REVIEW_FRAME_COUNT
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or abs(duration - REVIEW_FRAME_COUNT / REVIEW_MEDIA_FPS)
        > 1.0 / REVIEW_MEDIA_FPS
    ):
        raise PresentationContractError(f"{label} readback changed")
    _require_rate(value.get("r_frame_rate"), REVIEW_MEDIA_FPS, f"{label} readback rate")
    _require_rate(
        value.get("avg_frame_rate"),
        REVIEW_MEDIA_FPS,
        f"{label} average readback rate",
    )
    return value


def _require_probe_evidence_shape(
    value: Any,
    *,
    width: int,
    height: int,
    label: str,
) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {"readback", "ffprobe_argv", "full_decode"}
        or not isinstance(value.get("ffprobe_argv"), list)
        or not value["ffprobe_argv"]
        or not isinstance(value.get("full_decode"), dict)
        or set(value["full_decode"]) != {"argv", "passed"}
        or value["full_decode"].get("passed") is not True
        or not isinstance(value["full_decode"].get("argv"), list)
        or not value["full_decode"]["argv"]
    ):
        raise PresentationContractError(f"{label} probe evidence changed")
    _require_video_readback_shape(
        value["readback"], width=width, height=height, label=label
    )
    return value


def validate_frame_cell_content_readback(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != CONTENT_READBACK_FIELDS:
        raise PresentationContractError("frame/cell content readback fields changed")
    if (
        value.get("schema") != CONTENT_READBACK_SCHEMA
        or value.get("method") != content_readback_method()
        or value.get("all_cells_all_frames_passed") is not True
        or require_sha256(
            value.get("reference_gray_frames_sha256"),
            "reference gray-frame SHA-256",
        )
        != value["reference_gray_frames_sha256"]
        or require_sha256(
            value.get("observed_gray_frames_sha256"),
            "observed gray-frame SHA-256",
        )
        != value["observed_gray_frames_sha256"]
        or not isinstance(value.get("reference_argv"), list)
        or not value["reference_argv"]
        or not all(isinstance(item, str) for item in value["reference_argv"])
        or not isinstance(value.get("observed_argv"), list)
        or not value["observed_argv"]
        or not all(isinstance(item, str) for item in value["observed_argv"])
    ):
        raise PresentationContractError("frame/cell content readback is invalid")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != len(MEDIA_LAYOUT):
        raise PresentationContractError(
            "frame/cell content readback cell coverage changed"
        )
    for expected, cell in zip(MEDIA_LAYOUT, cells):
        label, _action, _view, _yaw, title, row, column = expected
        if (
            not isinstance(cell, dict)
            or set(cell)
            != {
                "label",
                "title",
                "row",
                "column",
                "frames",
                "all_frames_passed",
            }
            or cell.get("label") != label
            or cell.get("title") != title
            or cell.get("row") != row
            or cell.get("column") != column
            or cell.get("all_frames_passed") is not True
            or not isinstance(cell.get("frames"), list)
            or len(cell["frames"]) != REVIEW_FRAME_COUNT
        ):
            raise PresentationContractError(
                f"frame/cell content readback changed for {label}"
            )
        for frame_index, frame in enumerate(cell["frames"]):
            if (
                not isinstance(frame, dict)
                or set(frame)
                != {
                    "frame_index",
                    "reference_gray_sha256",
                    "observed_gray_sha256",
                    "mean_absolute_error",
                    "root_mean_square_error",
                    "max_absolute_error",
                    "passed",
                }
                or frame.get("frame_index") != frame_index
                or frame.get("passed") is not True
                or require_sha256(
                    frame.get("reference_gray_sha256"),
                    f"{label} frame {frame_index} reference SHA-256",
                )
                != frame["reference_gray_sha256"]
                or require_sha256(
                    frame.get("observed_gray_sha256"),
                    f"{label} frame {frame_index} observed SHA-256",
                )
                != frame["observed_gray_sha256"]
            ):
                raise PresentationContractError(
                    f"frame/cell content evidence changed for {label} "
                    f"frame {frame_index}"
                )
            mean = frame.get("mean_absolute_error")
            root_mean_square = frame.get("root_mean_square_error")
            maximum = frame.get("max_absolute_error")
            if (
                isinstance(mean, bool)
                or not isinstance(mean, (int, float))
                or not math.isfinite(mean)
                or mean < 0
                or mean > CONTENT_MEAN_ABSOLUTE_ERROR_MAX
                or isinstance(root_mean_square, bool)
                or not isinstance(root_mean_square, (int, float))
                or not math.isfinite(root_mean_square)
                or root_mean_square < 0
                or root_mean_square > CONTENT_ROOT_MEAN_SQUARE_ERROR_MAX
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or maximum < 0
                or maximum > CONTENT_MAX_ABSOLUTE_ERROR_MAX
            ):
                raise PresentationContractError(
                    f"frame/cell content error exceeded its contract for "
                    f"{label} frame {frame_index}"
                )
    if value.get("content_readback_sha256") != hash_without(
        value, "content_readback_sha256"
    ):
        raise PresentationContractError(
            "frame/cell content readback canonical self-hash failed"
        )
    return value


def validate_presentation_receipt(
    payload: Any,
    *,
    expected_source_review_sha256=None,
) -> dict:
    if not isinstance(payload, dict) or set(payload) != RECEIPT_TOP_LEVEL_FIELDS:
        raise PresentationContractError("presentation receipt fields changed")
    expected_review = require_sha256(
        payload.get("expected_source_review_sha256"),
        "receipt expected source-review SHA-256",
    )
    if expected_source_review_sha256 is not None and expected_review != require_sha256(
        expected_source_review_sha256,
        "external expected source-review SHA-256",
    ):
        raise PresentationContractError(
            "presentation receipt source-review authority changed"
        )
    try:
        created_at = datetime.fromisoformat(payload.get("created_at"))
    except (TypeError, ValueError) as error:
        raise PresentationContractError(
            "presentation receipt timestamp is invalid"
        ) from error
    if (
        payload.get("schema") != PRESENTATION_SCHEMA
        or payload.get("status") != PRESENTATION_STATUS
        or created_at.tzinfo is None
        or payload.get("authority")
        != {
            "purpose": "owner_animation_review_presentation_only",
            "decision_authority": "none",
            "user_decision_recorded": False,
            "source_review_modified": False,
            "formal_dataset_registration_authorized": False,
        }
    ):
        raise PresentationContractError(
            "presentation receipt authority/status is invalid"
        )
    source_review = _require_record_shape(
        payload.get("source_review"),
        FILE_RECORD_FIELDS,
        "source review",
    )
    if source_review["sha256"] != expected_review:
        raise PresentationContractError(
            "presentation receipt source-review hash is inconsistent"
        )
    _require_record_shape(
        payload.get("reviewed_animation"),
        FILE_RECORD_FIELDS,
        "reviewed animation",
    )
    automatic_checks = payload.get("automatic_checks")
    if (
        not isinstance(automatic_checks, dict)
        or set(automatic_checks) != AUTOMATIC_CHECK_FIELDS
        or not all(item is True for item in automatic_checks.values())
    ):
        raise PresentationContractError("presentation receipt automatic checks changed")
    guards = payload.get("source_authority_guards")
    if not isinstance(guards, dict) or not guards:
        raise PresentationContractError(
            "presentation receipt source authority guards are missing"
        )
    for label, guard in guards.items():
        if not isinstance(label, str) or not label:
            raise PresentationContractError(
                "presentation receipt source authority guard label is invalid"
            )
        _require_file_guard_shape(guard, f"source authority {label}")
    if payload.get("source_authority_guard_sha256") != canonical_json_sha256(guards):
        raise PresentationContractError(
            "presentation receipt source authority guard hash failed"
        )

    media = payload.get("authenticated_inputs")
    source_order = payload.get("source_order")
    source_set = payload.get("source_set")
    snapshots = payload.get("private_composition_inputs")
    if (
        not isinstance(media, dict)
        or set(media) != set(MEDIA_LABELS)
        or not isinstance(source_order, list)
        or len(source_order) != len(MEDIA_LAYOUT)
        or not isinstance(source_set, list)
        or len(source_set) != len(MEDIA_LAYOUT)
        or not isinstance(snapshots, list)
        or len(snapshots) != len(MEDIA_LAYOUT)
    ):
        raise PresentationContractError(
            "presentation receipt fixed source coverage changed"
        )
    for index, label in enumerate(MEDIA_LABELS):
        expected_layout = MEDIA_LAYOUT[index]
        entry = media[label]
        order = source_order[index]
        source = source_set[index]
        snapshot = snapshots[index]
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "title",
                "row",
                "column",
                "video",
                "render_manifest",
                "encode_manifest",
                "frame_set",
                "source_encode_ffmpeg",
                "video_probe",
            }
            or not isinstance(order, dict)
            or order
            != {
                "ordinal": index,
                "label": label,
                "video": entry["video"],
            }
            or not isinstance(source, dict)
            or source
            != {
                "ordinal": index,
                "label": label,
                "video": entry["video"],
                "render_manifest": entry["render_manifest"],
                "encode_manifest": entry["encode_manifest"],
                "frame_set": entry["frame_set"],
            }
            or not isinstance(snapshot, dict)
            or set(snapshot)
            != {
                "ordinal",
                "label",
                "source",
                "source_probe",
                "private_snapshot",
                "private_snapshot_probe",
                "private_snapshot_file_guard",
            }
            or snapshot.get("ordinal") != index
            or snapshot.get("label") != label
            or snapshot.get("source") != entry["video"]
            or snapshot.get("source_probe") != entry["video_probe"]
            or entry.get("title") != expected_layout[4]
            or entry.get("row") != expected_layout[5]
            or entry.get("column") != expected_layout[6]
        ):
            raise PresentationContractError(
                f"presentation receipt source binding changed for {label}"
            )
        _require_video_record_shape(
            entry["video"],
            width=REVIEW_MEDIA_WIDTH,
            height=REVIEW_MEDIA_HEIGHT,
            label=f"{label} source video",
        )
        _require_record_shape(
            entry["render_manifest"],
            FILE_RECORD_FIELDS,
            f"{label} render manifest",
        )
        _require_record_shape(
            entry["encode_manifest"],
            FILE_RECORD_FIELDS,
            f"{label} encode manifest",
        )
        _require_probe_evidence_shape(
            entry["video_probe"],
            width=REVIEW_MEDIA_WIDTH,
            height=REVIEW_MEDIA_HEIGHT,
            label=f"{label} source video",
        )
        _require_video_record_shape(
            snapshot["private_snapshot"],
            width=REVIEW_MEDIA_WIDTH,
            height=REVIEW_MEDIA_HEIGHT,
            label=f"{label} private snapshot",
        )
        _require_probe_evidence_shape(
            snapshot["private_snapshot_probe"],
            width=REVIEW_MEDIA_WIDTH,
            height=REVIEW_MEDIA_HEIGHT,
            label=f"{label} private snapshot",
        )
        _require_file_guard_shape(
            snapshot["private_snapshot_file_guard"],
            f"{label} private snapshot",
        )
    if payload.get("source_order_sha256") != canonical_json_sha256(
        source_order
    ) or payload.get("source_set_sha256") != canonical_json_sha256(source_set):
        raise PresentationContractError(
            "presentation receipt fixed source canonical hash failed"
        )

    content_readback = validate_frame_cell_content_readback(
        payload.get("frame_cell_content_readback")
    )
    if payload.get("presentation_contract") != presentation_contract():
        raise PresentationContractError(
            "presentation receipt layout contract is missing"
        )
    toolchain = payload.get("toolchain")
    if not isinstance(toolchain, dict) or set(toolchain) != {
        "presentation_tool",
        "python",
        "ffmpeg",
        "ffprobe",
        "font",
    }:
        raise PresentationContractError("presentation receipt toolchain fields changed")
    presentation_tool = toolchain["presentation_tool"]
    python_identity = toolchain["python"]
    if (
        not isinstance(presentation_tool, dict)
        or set(presentation_tool) != {"version", "file", "file_guard"}
        or presentation_tool.get("version") != PRESENTATION_TOOL_VERSION
        or not isinstance(python_identity, dict)
        or set(python_identity) != {"implementation", "version"}
        or not all(
            isinstance(python_identity.get(field), str) and python_identity[field]
            for field in ("implementation", "version")
        )
    ):
        raise PresentationContractError("presentation receipt tool identity changed")
    _require_record_shape(
        presentation_tool["file"],
        FILE_RECORD_FIELDS,
        "presentation tool",
    )
    _require_file_guard_shape(presentation_tool["file_guard"], "presentation tool")
    for executable_label in ("ffmpeg", "ffprobe"):
        executable = toolchain[executable_label]
        if (
            not isinstance(executable, dict)
            or set(executable)
            != {
                "executable",
                "file_guard",
                "version_argv",
                "version_first_line",
                "version_output_sha256",
            }
            or not isinstance(executable.get("version_argv"), list)
            or len(executable["version_argv"]) != 2
            or executable["version_argv"][0]
            != executable.get("executable", {}).get("path")
            or executable["version_argv"][1] != "-version"
            or not isinstance(executable.get("version_first_line"), str)
            or not executable["version_first_line"]
        ):
            raise PresentationContractError(
                f"presentation receipt {executable_label} identity changed"
            )
        _require_record_shape(
            executable["executable"],
            FILE_RECORD_FIELDS,
            f"{executable_label} executable",
        )
        _require_file_guard_shape(
            executable["file_guard"], f"{executable_label} executable"
        )
        require_sha256(
            executable.get("version_output_sha256"),
            f"{executable_label} version-output SHA-256",
        )
    font = toolchain["font"]
    if (
        not isinstance(font, dict)
        or set(font)
        != {
            "file",
            "file_guard",
            "sfnt_version_hex",
            "family",
            "subfamily",
            "version",
            "postscript_name",
        }
        or not all(
            isinstance(font.get(field), str) and font[field]
            for field in (
                "sfnt_version_hex",
                "family",
                "subfamily",
                "version",
                "postscript_name",
            )
        )
    ):
        raise PresentationContractError("presentation receipt font identity changed")
    _require_record_shape(font["file"], FILE_RECORD_FIELDS, "presentation font")
    _require_file_guard_shape(font["file_guard"], "presentation font")
    command = payload.get("command")
    if (
        not isinstance(command, dict)
        or set(command)
        != {
            "cwd",
            "ffmpeg_argv",
            "output_probe_argv",
            "output_full_decode_argv",
        }
        or command.get("cwd") != str(SPEAR_ROOT)
        or not all(
            isinstance(command.get(field), list) and command[field]
            for field in (
                "ffmpeg_argv",
                "output_probe_argv",
                "output_full_decode_argv",
            )
        )
    ):
        raise PresentationContractError("presentation receipt command fields changed")
    ffmpeg_path = Path(toolchain["ffmpeg"]["executable"]["path"])
    ffprobe_path = Path(toolchain["ffprobe"]["executable"]["path"])
    font_path = Path(toolchain["font"]["file"]["path"])
    snapshot_paths = [
        Path(binding["private_snapshot"]["path"]) for binding in snapshots
    ]
    staged_output = Path(command["ffmpeg_argv"][-1])
    if (
        command["ffmpeg_argv"]
        != build_ffmpeg_argv(
            ffmpeg=ffmpeg_path,
            font=font_path,
            videos=snapshot_paths,
            output=staged_output,
        )
        or command["output_probe_argv"]
        != build_ffprobe_argv(ffprobe_path, staged_output)
        or command["output_full_decode_argv"]
        != build_decode_argv(ffmpeg_path, staged_output)
        or content_readback["reference_argv"]
        != build_reference_rawvideo_argv(
            ffmpeg=ffmpeg_path,
            font=font_path,
            videos=snapshot_paths,
        )
        or content_readback["observed_argv"]
        != build_observed_rawvideo_argv(
            ffmpeg=ffmpeg_path,
            video=staged_output,
        )
    ):
        raise PresentationContractError(
            "presentation receipt command/source binding changed"
        )
    output = payload.get("output")
    if (
        not isinstance(output, dict)
        or set(output) != VIDEO_RECORD_FIELDS | {"readback", "full_decode_passed"}
        or output.get("full_decode_passed") is not True
    ):
        raise PresentationContractError("presentation receipt output fields changed")
    _require_video_record_shape(
        {key: output[key] for key in VIDEO_RECORD_FIELDS},
        width=OUTPUT_WIDTH,
        height=OUTPUT_HEIGHT,
        label="presentation output",
    )
    if (
        output.get("codec") != "h264"
        or output.get("width") != OUTPUT_WIDTH
        or output.get("height") != OUTPUT_HEIGHT
        or output.get("frame_count") != REVIEW_FRAME_COUNT
    ):
        raise PresentationContractError("presentation receipt output contract changed")
    _require_video_readback_shape(
        output.get("readback"),
        width=OUTPUT_WIDTH,
        height=OUTPUT_HEIGHT,
        label="presentation output",
    )
    if payload.get("receipt_sha256") != hash_without(payload, "receipt_sha256"):
        raise PresentationContractError(
            "presentation receipt canonical self-hash failed"
        )
    return payload


def load_presentation_receipt(
    path: Path,
    expected_receipt_file_sha256: str,
    *,
    expected_source_review_sha256=None,
) -> tuple[dict, dict]:
    expected_receipt_file_sha256 = require_sha256(
        expected_receipt_file_sha256,
        "expected presentation-receipt file SHA-256",
    )
    receipt_path = resolve_regular_file(path, "presentation receipt")
    if receipt_path.name != RECEIPT_NAME:
        raise PresentationContractError(
            f"presentation receipt must be named {RECEIPT_NAME}"
        )
    require_readonly_publication(receipt_path.parent)
    encoded, guard = read_stable_bytes(receipt_path, "presentation receipt")
    observed_sha256 = sha256_bytes(encoded)
    if observed_sha256 != expected_receipt_file_sha256:
        raise PresentationContractError(
            "presentation receipt failed external SHA-256 authentication"
        )
    try:
        payload = strict_json_loads(encoded)
    except StrictJSONError as error:
        raise PresentationContractError(
            "presentation receipt is not strict JSON"
        ) from error
    payload = validate_presentation_receipt(
        payload,
        expected_source_review_sha256=expected_source_review_sha256,
    )
    expected_video_path = receipt_path.parent / OUTPUT_VIDEO_NAME
    output = payload["output"]
    output_path = resolve_regular_file(
        Path(output["path"]),
        "presentation output video",
    )
    if output_path != expected_video_path:
        raise PresentationContractError(
            "presentation output video must be next to its receipt"
        )
    video_bytes, video_guard = read_stable_bytes(
        output_path,
        "presentation output video",
    )
    expected_output_record = {
        "path": str(output_path),
        "sha256": sha256_bytes(video_bytes),
        "size_bytes": len(video_bytes),
    }
    observed_output_record = {key: output[key] for key in FILE_RECORD_FIELDS}
    if observed_output_record != expected_output_record:
        raise PresentationContractError(
            "presentation output video failed file-record authentication"
        )
    require_same_guard(
        video_guard,
        output_path,
        "presentation output video",
    )
    require_same_guard(guard, receipt_path, "presentation receipt")
    require_readonly_publication(receipt_path.parent)
    return payload, {
        "path": guard["path"],
        "sha256": observed_sha256,
        "size_bytes": len(encoded),
    }


def require_receipt_self_hash(path: Path, expected: dict) -> None:
    encoded, _guard = read_stable_bytes(path, "presentation receipt")
    try:
        observed = strict_json_loads(encoded)
    except StrictJSONError as error:
        raise PresentationContractError(
            "presentation receipt is not strict JSON"
        ) from error
    if observed != expected:
        raise PresentationContractError(
            "presentation receipt changed after serialization"
        )
    validate_presentation_receipt(observed)


def main(argv=None):
    args = parse_args(argv)
    expected_review_sha256 = require_sha256(
        args.expected_review_run_sha256,
        "expected review-run SHA-256",
    )
    review_path = resolve_regular_file(args.review_run, "v4 review run")
    ffmpeg = resolve_named_executable("ffmpeg")
    ffprobe = resolve_named_executable("ffprobe")
    font = resolve_regular_file(FONT_PATH, "presentation font")
    runtime_identity = capture_runtime_identity(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        font=font,
    )
    initial = authenticate_review(
        review_path,
        expected_review_sha256,
        ffprobe=ffprobe,
        ffmpeg=ffmpeg,
    )
    location = prepare_output_location(args.output_root, review_path)
    staging = None
    try:
        staging = create_private_staging(location)
        inventory = StagingInventory()
        published = False
        try:
            snapshot_root = staging.path / ".private_input_snapshots"
            videos, snapshot_bindings = snapshot_authenticated_media(
                initial,
                snapshot_root,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )
            register_snapshot_tree(
                staging,
                inventory,
                snapshot_root,
                videos,
                snapshot_bindings,
            )

            # This catches mutations, replacements, and byte-restoration races during
            # snapshotting because the authenticated graph includes ctime/inode guards.
            after_snapshot = authenticate_review(
                review_path,
                expected_review_sha256,
                ffprobe=ffprobe,
                ffmpeg=ffmpeg,
            )
            if after_snapshot != initial:
                raise PresentationContractError(
                    "v4 review authority graph changed during private snapshotting"
                )

            staged_video = staging.path / OUTPUT_VIDEO_NAME
            ffmpeg_argv = build_ffmpeg_argv(
                ffmpeg=ffmpeg,
                font=font,
                videos=videos,
                output=staged_video,
            )
            run_ffmpeg(ffmpeg_argv)
            staged_video = resolve_regular_file(
                staged_video,
                "six-view presentation",
            )

            output_readback = probe_video(
                staged_video,
                ffprobe=ffprobe,
                ffmpeg=ffmpeg,
                expected_width=OUTPUT_WIDTH,
                expected_height=OUTPUT_HEIGHT,
                expected_frames=REVIEW_FRAME_COUNT,
                expected_fps=REVIEW_MEDIA_FPS,
            )
            register_top_level_file(
                staging,
                inventory,
                OUTPUT_VIDEO_NAME,
                staged_video,
                "six-view presentation",
                expected=output_readback["file_guard"],
            )
            content_readback = audit_frame_cell_content(
                ffmpeg=ffmpeg,
                font=font,
                snapshots=videos,
                output_video=staged_video,
                snapshot_bindings=snapshot_bindings,
            )
            reauthenticate_private_snapshots(
                videos,
                snapshot_bindings,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )

            # FFmpeg only reads private snapshots, but the full original authority
            # graph must still remain byte- and identity-stable until composition ends.
            final_inputs = authenticate_review(
                review_path,
                expected_review_sha256,
                ffprobe=ffprobe,
                ffmpeg=ffmpeg,
            )
            if final_inputs != initial:
                raise PresentationContractError(
                    "v4 review authority graph changed during composition"
                )
            final_runtime_identity = capture_runtime_identity(
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                font=font,
            )
            if final_runtime_identity != runtime_identity:
                raise PresentationContractError(
                    "presentation toolchain identity changed during composition"
                )

            source_order = [
                {
                    "ordinal": index,
                    "label": label,
                    "video": initial["media"][label]["video"],
                }
                for index, label in enumerate(MEDIA_LABELS)
            ]
            source_set = [
                {
                    "ordinal": index,
                    "label": label,
                    "video": initial["media"][label]["video"],
                    "render_manifest": initial["media"][label]["render_manifest"],
                    "encode_manifest": initial["media"][label]["encode_manifest"],
                    "frame_set": initial["media"][label]["frame_set"],
                }
                for index, label in enumerate(MEDIA_LABELS)
            ]
            published_video = dict(output_readback["video"])
            published_video["path"] = str(location.root / OUTPUT_VIDEO_NAME)
            receipt = {
                "schema": PRESENTATION_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": PRESENTATION_STATUS,
                "authority": {
                    "purpose": "owner_animation_review_presentation_only",
                    "decision_authority": "none",
                    "user_decision_recorded": False,
                    "source_review_modified": False,
                    "formal_dataset_registration_authorized": False,
                },
                "expected_source_review_sha256": expected_review_sha256,
                "source_review": initial["review_run"],
                "reviewed_animation": initial["animated_glb"],
                "authenticated_inputs": initial["media"],
                "source_authority_guards": initial["authority_guards"],
                "source_authority_guard_sha256": initial["authority_guard_sha256"],
                "source_order": source_order,
                "source_order_sha256": canonical_json_sha256(source_order),
                "source_set": source_set,
                "source_set_sha256": canonical_json_sha256(source_set),
                "private_composition_inputs": snapshot_bindings,
                "frame_cell_content_readback": content_readback,
                "presentation_contract": presentation_contract(),
                "automatic_checks": {
                    "external_source_review_sha256_authenticated": True,
                    "review_schema_status_and_automatic_gates_authenticated": True,
                    "six_source_media_and_lineage_receipts_authenticated": True,
                    "all_render_frame_bindings_authenticated": True,
                    "source_unique_video_streams_authenticated": True,
                    "source_counted_frames_authenticated": True,
                    "source_full_decodes_authenticated": True,
                    "private_snapshot_hashes_authenticated": True,
                    "private_snapshot_counted_frames_authenticated": True,
                    "private_snapshot_full_decodes_authenticated": True,
                    "private_snapshots_unchanged_after_composition": True,
                    "source_graph_unchanged_after_snapshot": True,
                    "source_graph_unchanged_after_composition": True,
                    "source_restore_race_guards_unchanged": True,
                    "fixed_source_order_authenticated": True,
                    "toolchain_identity_and_versions_unchanged": True,
                    "output_unique_video_stream_authenticated": True,
                    "output_counted_frames_authenticated": True,
                    "output_full_decode_authenticated": True,
                    "all_output_frame_cell_content_authenticated": True,
                    "source_review_unmodified": True,
                    "formal_dataset_registration_authority_not_granted": True,
                },
                "toolchain": runtime_identity,
                "command": {
                    "cwd": str(SPEAR_ROOT),
                    "ffmpeg_argv": ffmpeg_argv,
                    "output_probe_argv": output_readback["ffprobe_argv"],
                    "output_full_decode_argv": output_readback["full_decode"]["argv"],
                },
                "output": {
                    **published_video,
                    "readback": output_readback["readback"],
                    "full_decode_passed": output_readback["full_decode"]["passed"],
                },
                "receipt_sha256": None,
            }
            receipt["receipt_sha256"] = hash_without(receipt, "receipt_sha256")
            receipt_path = staging.path / RECEIPT_NAME
            write_json_exclusive_at(
                staging,
                inventory,
                RECEIPT_NAME,
                receipt_path,
                receipt,
            )
            require_receipt_self_hash_at(
                staging,
                RECEIPT_NAME,
                receipt_path,
                receipt,
            )
            _require_file_guard_at(
                staging.descriptor,
                OUTPUT_VIDEO_NAME,
                staged_video,
                inventory.top_level_files[OUTPUT_VIDEO_NAME],
                "six-view presentation",
            )
            staged_record = _file_record_at(
                staging.descriptor,
                OUTPUT_VIDEO_NAME,
                staged_video,
                "six-view presentation",
            )
            if any(
                staged_record[key] != published_video[key]
                for key in ("sha256", "size_bytes")
            ):
                raise PresentationContractError(
                    "presentation changed after its receipt was written"
                )

            remove_private_snapshots(staging, inventory)
            seal_readonly_tree(staging, inventory)
            require_readonly_publication_fd(staging, inventory)
            _require_parent_path_binding(location)
            try:
                os.stat(
                    location.output_name,
                    dir_fd=location.parent.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise PresentationContractError(
                    f"refusing to replace output root: {location.root}"
                )
            atomic_publish_no_replace(location, staging)
            published = True

            # From this point onward the final is an immutable publication.  A
            # post-publication validation or fsync failure must never delete it.
            os.fsync(location.parent.descriptor)
            require_published_binding(location, staging)
            require_readonly_publication_fd(staging, inventory)
            final_video = location.root / OUTPUT_VIDEO_NAME
            final_receipt_path = location.root / RECEIPT_NAME
            require_receipt_self_hash_at(
                staging,
                RECEIPT_NAME,
                final_receipt_path,
                receipt,
            )
            if _file_record_at(
                staging.descriptor,
                OUTPUT_VIDEO_NAME,
                final_video,
                "published presentation",
            ) != {key: published_video[key] for key in FILE_RECORD_FIELDS}:
                raise PresentationContractError(
                    "published presentation does not match its receipt"
                )
            print(
                "GENERATED_QUADRUPED_OWNER_REVIEW_PRESENTATION_OK "
                f"video={final_video} receipt={final_receipt_path}",
                flush=True,
            )
            return 0
        except Exception:
            if not published:
                cleanup_unpublished_staging(location, staging, inventory)
            raise
        finally:
            staging.close()
    finally:
        location.parent.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        PresentationContractError,
        StrictJSONError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(
            f"GENERATED_QUADRUPED_OWNER_REVIEW_PRESENTATION_FAILED {error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
