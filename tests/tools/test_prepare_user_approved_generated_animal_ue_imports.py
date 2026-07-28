import copy
import hashlib
import io
import json
import os
import shutil
import stat
import struct
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_source_asset_execution as execution_preparation
from tools import prepare_user_approved_generated_animal_ue_imports as preparation
from tools import register_controlled_animal_source_assets as source_registry
from tools import transcode_glb_webp_to_png as transcode_tool

PROFILE = (
    Path(__file__).resolve().parents[2]
    / "data/controlled_source_attributes_v1/candidate_profiles/animal"
    / "dog_shiba_inu_four_limb_rest_side_clay_v1.json"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path):
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_record(path, root):
    record = _record(path)
    record["path"] = path.resolve().relative_to(root.resolve()).as_posix()
    return record


def _root_record(root_id, path, root):
    return {
        "root_id": root_id,
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _inject_duplicate_schema(path):
    payload = contracts.load_json(path)
    encoded = json.dumps(payload, ensure_ascii=False)
    duplicate = json.dumps(payload["schema"], ensure_ascii=False)
    path.write_text(
        f'{{"schema":{duplicate},' + encoded[1:] + "\n",
        encoding="utf-8",
    )


def _write_glb(path):
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"mesh": 0, "skin": 0},
            {"name": "root"},
            {"name": "joint_1"},
            {"name": "joint_2"},
            {"name": "joint_3"},
            {"name": "joint_4"},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "JOINTS_0": 1,
                            "WEIGHTS_0": 2,
                        }
                    }
                ]
            }
        ],
        "skins": [{"joints": [1, 2, 3, 4, 5], "inverseBindMatrices": 3}],
        "animations": [
            {
                "name": name,
                "samplers": [{"input": 4, "output": 5}],
                "channels": [
                    {
                        "sampler": 0,
                        "target": {"node": 1, "path": "rotation"},
                    }
                ],
            }
            for name in ("Idle", "Walking")
        ],
        "buffers": [{"byteLength": 512}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 512}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 1,
                "type": "VEC3",
            },
            {
                "bufferView": 0,
                "componentType": 5123,
                "count": 1,
                "type": "VEC4",
            },
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 1,
                "type": "VEC4",
            },
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 5,
                "type": "MAT4",
            },
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 1,
                "type": "SCALAR",
            },
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 1,
                "type": "VEC4",
            },
        ],
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    binary_chunk = bytes(512)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary_chunk), 0x004E4942)
        + binary_chunk
    )


def _encoded_image(format_name, color):
    encoded = io.BytesIO()
    Image.new("RGBA", (2, 2), color).save(
        encoded,
        format=format_name,
        lossless=True,
    )
    return encoded.getvalue()


def _write_mixed_webp_glb(path):
    webp = _encoded_image("WEBP", (20, 80, 140, 255))
    preserved_png = _encoded_image("PNG", (160, 120, 40, 255))
    binary = bytearray(512)
    webp_offset = len(binary)
    binary.extend(webp)
    binary.extend(b"\x00" * ((-len(binary)) % 4))
    png_offset = len(binary)
    binary.extend(preserved_png)
    document = {
        "asset": {"version": "2.0", "generator": "transcode-auth-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"mesh": 0, "skin": 0},
            {"name": "root"},
            {"name": "joint_1"},
            {"name": "joint_2"},
            {"name": "joint_3"},
            {"name": "joint_4"},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "JOINTS_0": 1,
                            "WEIGHTS_0": 2,
                        },
                        "material": 0,
                    }
                ]
            }
        ],
        "skins": [{"joints": [1, 2, 3, 4, 5], "inverseBindMatrices": 3}],
        "animations": [
            {
                "name": name,
                "samplers": [{"input": 4, "output": 5}],
                "channels": [
                    {
                        "sampler": 0,
                        "target": {"node": 1, "path": "rotation"},
                    }
                ],
            }
            for name in ("Idle", "Walking")
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 512},
            {
                "buffer": 0,
                "byteOffset": webp_offset,
                "byteLength": len(webp),
            },
            {
                "buffer": 0,
                "byteOffset": png_offset,
                "byteLength": len(preserved_png),
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": component,
                "count": count,
                "type": kind,
            }
            for component, count, kind in (
                (5126, 1, "VEC3"),
                (5123, 1, "VEC4"),
                (5126, 1, "VEC4"),
                (5126, 5, "MAT4"),
                (5126, 1, "SCALAR"),
                (5126, 1, "VEC4"),
            )
        ],
        "images": [
            {"name": "coat", "bufferView": 1, "mimeType": "image/webp"},
            {"name": "normal", "bufferView": 2, "mimeType": "image/png"},
        ],
        "samplers": [{"magFilter": 9729, "minFilter": 9987}],
        "textures": [
            {
                "name": "coat",
                "sampler": 0,
                "source": 1,
                "extensions": {"EXT_texture_webp": {"source": 0}},
            },
            {"name": "normal", "sampler": 0, "source": 1},
        ],
        "materials": [
            {
                "name": "pbr_coat",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.7,
                },
                "normalTexture": {"index": 1},
            }
        ],
        "extensionsUsed": ["KHR_materials_unlit", "EXT_texture_webp"],
        "extensionsRequired": ["EXT_texture_webp"],
    }
    path.write_bytes(transcode_tool.encode_glb(document, bytes(binary)))
    return path


def _write_texture_transcode_fixture(root, *, source_path=None):
    source = source_path or root / "reviewed_mixed_textures.glb"
    _write_mixed_webp_glb(source)
    output = root / "ue_compatible.glb"
    manifest_path = root / "texture_transcode_manifest.json"
    document, binary = transcode_tool.read_glb(source)
    rewritten, rewritten_binary, records = transcode_tool.transcode(
        document,
        binary,
    )
    output.write_bytes(transcode_tool.encode_glb(rewritten, rewritten_binary))
    manifest = {
        "schema": preparation.TEXTURE_TRANSCODE_SCHEMA,
        "purpose": preparation.TEXTURE_TRANSCODE_PURPOSE,
        "geometry_skin_animation_byte_graph_changed": False,
        "input": _record(source),
        "output": _record(output),
        "images": records,
    }
    _write_json(manifest_path, manifest)
    return source, output, manifest_path


def _write_fake_unskinned_glb(path):
    document = {
        "asset": {"version": "2.0"},
        "nodes": [{"mesh": 0, "skin": 0}],
        "meshes": [{"primitives": [{}]}],
        "skins": [{"joints": [0]}],
        "animations": [{"name": "Idle"}, {"name": "Walking"}],
    }
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total_length = 12 + 8 + len(payload)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
    )


def _build_frozen_preflight(tmp_path):
    profiles = input_builder.load_profiles([PROFILE])
    roots = execution_preparation.default_artifact_roots()
    authentication = {
        profile["profile_schema_id"]: input_builder.authenticate_profile_artifacts(
            profile, roots
        )
        for profile in profiles
    }
    files = input_builder.compile_inputs(
        profiles=profiles,
        count_per_profile=1,
        seed=20260728,
        plan_id="approved_generated_animal_bridge_test_v1",
        split_salt="approved_generated_animal_bridge_test_v1",
        max_qa_pairs_per_split=None,
        artifact_authentication=authentication,
    )
    files["execution_jobs.json"] = input_builder.build_execution_jobs(
        files["instance_requests.json"],
        route_names={
            "flux2_pixal3d_animal_v1",
            "stable_animal_template_v1",
            "rocketbox_material_v1",
        },
    )
    input_dir = tmp_path / "inputs"
    input_builder.publish_output(input_dir, files)
    preflight = execution_preparation.build_execution_preflight(input_dir, roots)
    preflight["routes"].pop("flux2_pixal3d_static_v1")
    preflight["execution_summary"].pop("static_object_job_count")
    preflight["preflight_sha256"] = execution_preparation.preflight_sha256(preflight)
    preflight_path = tmp_path / "execution_preflight.json"
    contracts.write_json_no_replace(preflight_path, preflight)
    request = files["instance_requests.json"]["requests"][0]
    return preflight_path, preflight, profiles[0], request


def _make_video(path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=512x384:r=8",
            "-frames:v",
            str(preparation.REQUIRED_REVIEW_FRAMES),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _build_media_lineage(root, *, input_glb, media):
    media_root = root / "05_review"
    media_root.mkdir(parents=True, exist_ok=True)
    lineage = {}
    for label, action, view, yaw in preparation.generated_review.REVIEW_MEDIA_SPECS:
        paths = preparation.generated_review.review_media_paths(
            {"review_root": media_root}, label
        )
        frame_dir = paths["frame_dir"]
        frame_dir.mkdir(exist_ok=True)
        frame_range = [1.0, 9.0]
        frames = []
        for index in range(preparation.REQUIRED_REVIEW_FRAMES):
            frame = frame_dir / f"frame_{index:04d}.png"
            frame.write_bytes(f"{label}-png-{index}".encode())
            fraction = index / (preparation.REQUIRED_REVIEW_FRAMES - 1)
            frames.append(
                {
                    "index": index,
                    "sample_fraction": fraction,
                    "source_action_frame": float(
                        int(
                            round(
                                frame_range[0]
                                + (frame_range[1] - frame_range[0]) * fraction
                            )
                        )
                    ),
                    "artifact": _record(frame),
                }
            )
        render_payload = {
            "schema": preparation.generated_review.RENDER_MANIFEST_SCHEMA,
            "status": "frames_rendered",
            "formal_dataset_registration_authorized": False,
            "input_glb": _record(input_glb),
            "request": {
                "action": action,
                "resolved_action": action,
                "rest_pose": False,
                "view": view,
                "asset_yaw_deg": yaw,
                "n_frames": preparation.REQUIRED_REVIEW_FRAMES,
                "resolution": {
                    "width": preparation.generated_review.REVIEW_MEDIA_WIDTH,
                    "height": preparation.generated_review.REVIEW_MEDIA_HEIGHT,
                },
                "fps": preparation.generated_review.REVIEW_MEDIA_FPS,
                "output_dir": str(frame_dir.resolve()),
            },
            "render_config": copy.deepcopy(
                preparation.generated_review.REVIEW_RENDER_CONFIG
            ),
            "action_frame_range": frame_range,
            "frames": frames,
        }
        _write_json(paths["render_manifest"], render_payload)
        encode_payload = {
            "schema": preparation.generated_review.ENCODE_MANIFEST_SCHEMA,
            "status": "video_encoded_and_probed",
            "formal_dataset_registration_authorized": False,
            "media_identity": {
                "label": label,
                "action": action,
                "view": view,
                "asset_yaw_deg": yaw,
            },
            "render_manifest": _record(paths["render_manifest"]),
            "frame_set": preparation.generated_review.render_frame_set(render_payload),
            "ffmpeg": preparation.generated_review.expected_review_ffmpeg_config(
                frame_dir,
                preparation.REQUIRED_REVIEW_FRAMES,
            ),
            "video": copy.deepcopy(media[label]),
        }
        _write_json(paths["encode_manifest"], encode_payload)
        lineage[label] = {
            "render_manifest": _record(paths["render_manifest"]),
            "encode_manifest": _record(paths["encode_manifest"]),
        }
    return lineage


def _guard(path):
    current = os.stat(path, follow_symlinks=False)
    return {
        "path": str(path.resolve()),
        "device": current.st_dev,
        "inode": current.st_ino,
        "mode": stat.S_IMODE(current.st_mode),
        "link_count": current.st_nlink,
        "size_bytes": current.st_size,
        "mtime_ns": current.st_mtime_ns,
        "ctime_ns": current.st_ctime_ns,
    }


def _video_record(path, *, width, height):
    return {
        **_record(path),
        "codec": "h264",
        "width": width,
        "height": height,
        "frame_count": preparation.REQUIRED_REVIEW_FRAMES,
        "frame_rate": "8/1",
        "duration_seconds": 1.0,
    }


def _video_readback(*, width, height):
    return {
        "stream_count": 1,
        "stream_index": 0,
        "codec_type": "video",
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "sample_aspect_ratio": "1:1",
        "width": width,
        "height": height,
        "r_frame_rate": "8/1",
        "avg_frame_rate": "8/1",
        "nb_frames": preparation.REQUIRED_REVIEW_FRAMES,
        "nb_read_frames": preparation.REQUIRED_REVIEW_FRAMES,
        "duration_seconds": 1.0,
    }


def _probe_evidence(path, *, width, height, ffmpeg, ffprobe):
    return {
        "readback": _video_readback(width=width, height=height),
        "ffprobe_argv": preparation.presentation.build_ffprobe_argv(ffprobe, path),
        "full_decode": {
            "argv": preparation.presentation.build_decode_argv(ffmpeg, path),
            "passed": True,
        },
    }


def _write_presentation_bundle(
    tmp_path,
    *,
    review_path,
    review,
    bundle_name="owner_review_presentation",
):
    presentation_root = tmp_path / bundle_name
    presentation_root.mkdir()
    output_video = presentation_root / preparation.presentation.OUTPUT_VIDEO_NAME
    output_video.write_bytes(b"authenticated-owner-review-video")
    output_record = {
        **_video_record(
            output_video,
            width=preparation.presentation.OUTPUT_WIDTH,
            height=preparation.presentation.OUTPUT_HEIGHT,
        ),
        "readback": _video_readback(
            width=preparation.presentation.OUTPUT_WIDTH,
            height=preparation.presentation.OUTPUT_HEIGHT,
        ),
        "full_decode_passed": True,
    }

    ffmpeg = Path(shutil.which("ffmpeg")).resolve()
    ffprobe = Path(shutil.which("ffprobe")).resolve()
    font = preparation.presentation.FONT_PATH.resolve()
    snapshots = []
    authenticated_inputs = {}
    private_inputs = []
    source_order = []
    source_set = []
    for ordinal, (
        label,
        _action,
        _view,
        _yaw,
        title,
        row,
        column,
    ) in enumerate(preparation.presentation.MEDIA_LAYOUT):
        source_video = copy.deepcopy(review["outputs"]["media"][label])
        lineage = review["outputs"]["media_lineage"][label]
        render_payload = contracts.load_json(Path(lineage["render_manifest"]["path"]))
        encode_payload = contracts.load_json(Path(lineage["encode_manifest"]["path"]))
        source_probe = _probe_evidence(
            Path(source_video["path"]),
            width=preparation.presentation.REVIEW_MEDIA_WIDTH,
            height=preparation.presentation.REVIEW_MEDIA_HEIGHT,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
        entry = {
            "title": title,
            "row": row,
            "column": column,
            "video": source_video,
            "render_manifest": copy.deepcopy(lineage["render_manifest"]),
            "encode_manifest": copy.deepcopy(lineage["encode_manifest"]),
            "frame_set": preparation.generated_review.render_frame_set(render_payload),
            "source_encode_ffmpeg": encode_payload["ffmpeg"],
            "video_probe": source_probe,
        }
        authenticated_inputs[label] = entry
        source_order.append(
            {
                "ordinal": ordinal,
                "label": label,
                "video": source_video,
            }
        )
        source_set.append(
            {
                "ordinal": ordinal,
                "label": label,
                "video": source_video,
                "render_manifest": entry["render_manifest"],
                "encode_manifest": entry["encode_manifest"],
                "frame_set": entry["frame_set"],
            }
        )
        snapshot_path = (
            tmp_path / ".removed_private_snapshots" / f"{ordinal:02d}_{label}.mp4"
        ).resolve()
        snapshots.append(snapshot_path)
        snapshot_video = copy.deepcopy(source_video)
        snapshot_video["path"] = str(snapshot_path)
        snapshot_guard = {
            "path": str(snapshot_path),
            "device": 1,
            "inode": ordinal + 1,
            "mode": 0o400,
            "link_count": 1,
            "size_bytes": snapshot_video["size_bytes"],
            "mtime_ns": ordinal + 1,
            "ctime_ns": ordinal + 1,
        }
        private_inputs.append(
            {
                "ordinal": ordinal,
                "label": label,
                "source": source_video,
                "source_probe": source_probe,
                "private_snapshot": snapshot_video,
                "private_snapshot_probe": _probe_evidence(
                    snapshot_path,
                    width=preparation.presentation.REVIEW_MEDIA_WIDTH,
                    height=preparation.presentation.REVIEW_MEDIA_HEIGHT,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                ),
                "private_snapshot_file_guard": snapshot_guard,
            }
        )

    staged_output = (
        tmp_path
        / ".owner_review_presentation.staging"
        / preparation.presentation.OUTPUT_VIDEO_NAME
    ).resolve()
    ffmpeg_argv = preparation.presentation.build_ffmpeg_argv(
        ffmpeg=ffmpeg,
        font=font,
        videos=snapshots,
        output=staged_output,
    )
    reference_argv = preparation.presentation.build_reference_rawvideo_argv(
        ffmpeg=ffmpeg,
        font=font,
        videos=snapshots,
    )
    observed_argv = preparation.presentation.build_observed_rawvideo_argv(
        ffmpeg=ffmpeg,
        video=staged_output,
    )
    content = {
        "schema": preparation.presentation.CONTENT_READBACK_SCHEMA,
        "method": preparation.presentation.content_readback_method(),
        "reference_argv": reference_argv,
        "observed_argv": observed_argv,
        "reference_gray_frames_sha256": "1" * 64,
        "observed_gray_frames_sha256": "2" * 64,
        "cells": [
            {
                "label": label,
                "title": title,
                "row": row,
                "column": column,
                "frames": [
                    {
                        "frame_index": frame_index,
                        "reference_gray_sha256": "3" * 64,
                        "observed_gray_sha256": "4" * 64,
                        "mean_absolute_error": 0.0,
                        "root_mean_square_error": 0.0,
                        "max_absolute_error": 0,
                        "passed": True,
                    }
                    for frame_index in range(preparation.REQUIRED_REVIEW_FRAMES)
                ],
                "all_frames_passed": True,
            }
            for (
                label,
                _action,
                _view,
                _yaw,
                title,
                row,
                column,
            ) in preparation.presentation.MEDIA_LAYOUT
        ],
        "all_cells_all_frames_passed": True,
        "content_readback_sha256": None,
    }
    content["content_readback_sha256"] = preparation.presentation.hash_without(
        content, "content_readback_sha256"
    )
    tool_path = Path(preparation.presentation.__file__).resolve()
    authority_guards = {"review_run": _guard(review_path)}
    receipt = {
        "schema": preparation.presentation.PRESENTATION_SCHEMA,
        "created_at": "2026-07-28T00:00:00+00:00",
        "status": preparation.presentation.PRESENTATION_STATUS,
        "authority": {
            "purpose": "owner_animation_review_presentation_only",
            "decision_authority": "none",
            "user_decision_recorded": False,
            "source_review_modified": False,
            "formal_dataset_registration_authorized": False,
        },
        "expected_source_review_sha256": _sha256(review_path),
        "source_review": _record(review_path),
        "reviewed_animation": copy.deepcopy(review["outputs"]["animated_glb"]),
        "authenticated_inputs": authenticated_inputs,
        "source_authority_guards": authority_guards,
        "source_authority_guard_sha256": (
            preparation.presentation.canonical_json_sha256(authority_guards)
        ),
        "source_order": source_order,
        "source_order_sha256": preparation.presentation.canonical_json_sha256(
            source_order
        ),
        "source_set": source_set,
        "source_set_sha256": preparation.presentation.canonical_json_sha256(source_set),
        "private_composition_inputs": private_inputs,
        "frame_cell_content_readback": content,
        "presentation_contract": preparation.presentation.presentation_contract(),
        "automatic_checks": {
            name: True for name in preparation.presentation.AUTOMATIC_CHECK_FIELDS
        },
        "toolchain": {
            "presentation_tool": {
                "version": preparation.presentation.PRESENTATION_TOOL_VERSION,
                "file": _record(tool_path),
                "file_guard": _guard(tool_path),
            },
            "python": {"implementation": "CPython", "version": "3.9.test"},
            "ffmpeg": {
                "executable": _record(ffmpeg),
                "file_guard": _guard(ffmpeg),
                "version_argv": [str(ffmpeg), "-version"],
                "version_first_line": "ffmpeg version test",
                "version_output_sha256": "5" * 64,
            },
            "ffprobe": {
                "executable": _record(ffprobe),
                "file_guard": _guard(ffprobe),
                "version_argv": [str(ffprobe), "-version"],
                "version_first_line": "ffprobe version test",
                "version_output_sha256": "6" * 64,
            },
            "font": {
                "file": _record(font),
                "file_guard": _guard(font),
                "sfnt_version_hex": "00010000",
                "family": "DejaVu Sans",
                "subfamily": "Bold",
                "version": "Version test",
                "postscript_name": "DejaVuSans-Bold",
            },
        },
        "command": {
            "cwd": str(preparation.presentation.SPEAR_ROOT),
            "ffmpeg_argv": ffmpeg_argv,
            "output_probe_argv": preparation.presentation.build_ffprobe_argv(
                ffprobe, staged_output
            ),
            "output_full_decode_argv": (
                preparation.presentation.build_decode_argv(ffmpeg, staged_output)
            ),
        },
        "output": output_record,
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = preparation.presentation.hash_without(
        receipt, "receipt_sha256"
    )
    receipt_path = presentation_root / preparation.presentation.RECEIPT_NAME
    _write_json(receipt_path, receipt)
    output_video.chmod(0o444)
    receipt_path.chmod(0o444)
    presentation_root.chmod(0o555)
    preparation.presentation.load_presentation_receipt(
        receipt_path,
        _sha256(receipt_path),
        expected_source_review_sha256=_sha256(review_path),
    )
    return {
        "presentation_receipt": _record(receipt_path),
        "expected_presentation_receipt_file_sha256": _sha256(receipt_path),
        "presentation_receipt_sha256": receipt["receipt_sha256"],
        "output_video": output_record,
    }


def _set_presentation_writable(fixture):
    evidence = fixture["presentation_evidence"]
    receipt = Path(evidence["presentation_receipt"]["path"])
    output = Path(evidence["output_video"]["path"])
    receipt.parent.chmod(0o755)
    receipt.chmod(0o644)
    output.chmod(0o644)


def _reseal_presentation(fixture):
    evidence = fixture["presentation_evidence"]
    receipt = Path(evidence["presentation_receipt"]["path"])
    output = Path(evidence["output_video"]["path"])
    receipt.chmod(0o444)
    output.chmod(0o444)
    receipt.parent.chmod(0o555)


def _rebind_presentation_receipt_file(fixture):
    evidence = fixture["presentation_evidence"]
    receipt = Path(evidence["presentation_receipt"]["path"])
    evidence["presentation_receipt"] = _record(receipt)
    evidence["expected_presentation_receipt_file_sha256"] = _sha256(receipt)


def _refresh_presentation_evidence(fixture, review):
    bundle_name = f"owner_review_presentation_{_sha256(fixture['review_path'])[:12]}"
    try:
        evidence = _write_presentation_bundle(
            fixture["receipt_path"].parent,
            review_path=fixture["review_path"],
            review=review,
            bundle_name=bundle_name,
        )
    except (KeyError, TypeError, contracts.ContractError):
        return
    fixture["presentation_evidence"] = evidence


def _fake_tokenrig_lineage(
    root,
    *,
    raw_pixal_glb,
    target_rig_glb,
    tokenrig_input_glb=None,
    upstream_kind="watertight_runtime_proxy",
    bounded_geometry_closure=None,
    bounded_geometry_closure_manifest_sha256=None,
    watertight_proxy_manifest=None,
    watertight_proxy_geometry_audit=None,
    raw_static_decision_batch=None,
):
    tokenrig_input_glb = tokenrig_input_glb or raw_pixal_glb
    closure_root = root / "tokenrig_closure"
    closure_root.mkdir(exist_ok=True)
    closure_manifest = closure_root / "manifest.json"
    if upstream_kind == "bounded_watertight_runtime_proxy":
        if (
            bounded_geometry_closure is None
            or bounded_geometry_closure_manifest_sha256 is None
            or watertight_proxy_manifest is None
            or watertight_proxy_geometry_audit is None
            or raw_static_decision_batch is None
        ):
            raise ValueError("composite TokenRig fixture authority is missing")
        closure_payload = {
            "lineage": {
                "composite_upstream_authority": {
                    "schema": (
                        "bounded_watertight_runtime_proxy_authority_v1"
                    ),
                    "bounded_geometry_closure_file_sha256": _sha256(
                        bounded_geometry_closure
                    ),
                    "bounded_geometry_closure_manifest_sha256": (
                        bounded_geometry_closure_manifest_sha256
                    ),
                    "watertight_proxy_manifest_file_sha256": _sha256(
                        watertight_proxy_manifest
                    ),
                    "watertight_proxy_geometry_audit_file_sha256": _sha256(
                        watertight_proxy_geometry_audit
                    ),
                    "watertight_proxy_correspondence": {
                        "schema": (
                            "avengine_bounded_watertight_vertex_"
                            "correspondence_v1"
                        ),
                        "normalization": (
                            "repaired_axis_aligned_bbox_diagonal"
                        ),
                        "repaired_imported_vertices": 100,
                        "proxy_unique_vertices": 100,
                        "proxy_to_repaired": {
                            "p99_ratio": 0.001,
                            "max_ratio": 0.002,
                        },
                        "repaired_to_proxy": {
                            "p99_ratio": 0.010,
                            "max_ratio": 0.020,
                        },
                        "thresholds": {
                            "proxy_to_repaired_p99_ratio_max": 0.003,
                            "proxy_to_repaired_max_ratio_max": 0.006,
                            "repaired_to_proxy_p99_ratio_max": 0.030,
                            "repaired_to_proxy_max_ratio_max": 0.060,
                        },
                    },
                    "raw_static_decision_batch_sha256": _sha256(
                        raw_static_decision_batch
                    ),
                }
            },
            "evidence": {
                "upstream_manifest": {
                    "original": _record(watertight_proxy_manifest),
                },
                "extra_upstream_manifests": [
                    {"original": _record(bounded_geometry_closure)},
                    {"original": _record(watertight_proxy_geometry_audit)},
                    {"original": _record(watertight_proxy_geometry_audit)},
                    {"original": _record(watertight_proxy_geometry_audit)},
                    {"original": _record(raw_static_decision_batch)},
                ],
            },
        }
    else:
        closure_payload = {"fixture": "closure"}
    _write_json(closure_manifest, closure_payload)
    readback = closure_root / "geometry_readback.json"
    readback.write_text('{"fixture":"readback"}\n', encoding="utf-8")
    descriptor = {
        "schema": "avengine_generated_animal_tokenrig_closure_descriptor_v1",
        "status": "passed",
        "asset_id": "fixture_workspace",
        "evidence_mode": "complete_load_audit_v1",
        "closure_manifest": {
            **_record(closure_manifest),
            "manifest_sha256": "1" * 64,
        },
        "lineage": {
            "raw_pixal_glb": _record(raw_pixal_glb),
            "tokenrig_input": _record(tokenrig_input_glb),
            "tokenrig_output": _record(target_rig_glb),
            "upstream_kind": upstream_kind,
        },
        "geometry_readback": {
            **_record(readback),
            "manifest_sha256": "2" * 64,
        },
        "execution": {
            "seed": 42,
            "exact_argv_sha256": "3" * 64,
            "model_checkpoint_sha256": "4" * 64,
            "model_snapshot_revision": "5" * 40,
            "runtime_patch_sha256": "6" * 64,
            "skintokens_revision": "7" * 40,
        },
        "formal_dataset_registration_authorized": False,
    }
    descriptor["descriptor_sha256"] = preparation._hash_without(
        descriptor, "descriptor_sha256"
    )
    return descriptor


@pytest.fixture
def approved_generated_animal(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    candidate = artifact_root / "candidate.bin"
    candidate.write_bytes(b"authenticated source candidate")
    license_path = artifact_root / "LICENSE.txt"
    license_path.write_bytes(b"fixture research license")
    preflight_path, preflight, profile, request = _build_frozen_preflight(tmp_path)

    source_asset = contracts.build_source_asset_v2(
        request,
        artifacts={
            "pixal_raw_glb": _root_record("fixture_root", candidate, artifact_root)
        },
        physical_measurements={"status": "pending"},
        provenance={
            "attempt_id": "fixture_approved_generated_animal_v1",
            "request_sha256": request["request_sha256"],
            "models": copy.deepcopy(request["generation_plan"]["model_revisions"]),
        },
        rights={
            "status": "review_required",
            "licenses": [_root_record("fixture_root", license_path, artifact_root)],
            "blockers": ["research_candidate_only"],
        },
        qa={
            "reference_2d": "passed",
            "static_mesh": "passed",
            "binding": "pending",
            "walking": "pending",
            "idle": "pending",
            "ue_import_readback": "pending",
            "apartment_media": "pending",
            "audio": "pending",
        },
        state_classification="research_candidate",
    )
    registry_root = tmp_path / "source_registry"
    source_dir = registry_root / "source_assets"
    source_dir.mkdir(parents=True)
    source_path = source_dir / f"{source_asset['asset_id']}.json"
    contracts.write_json_no_replace(source_path, source_asset)
    attribute_evidence = {
        name: (
            "deferred_to_metric_3d"
            if name == source_asset["target_physical_profile"]["control_attribute"]
            else "passed_static_visual"
        )
        for name in source_asset["sampled_attributes"]
    }
    source_index = {
        "asset_id": source_asset["asset_id"],
        "profile_schema_id": source_asset["profile_schema_id"],
        "request_sha256": source_asset["request_sha256"],
        "sampled_attributes": copy.deepcopy(source_asset["sampled_attributes"]),
        "attribute_evidence": attribute_evidence,
        "source_asset": {
            "path": source_path.relative_to(registry_root).as_posix(),
            "sha256": _sha256(source_path),
            "size_bytes": source_path.stat().st_size,
        },
        "state_classification": "research_candidate",
        "next_gate": "lod_then_species_rig_binding",
    }
    pixal_inputs_path = tmp_path / "pixal_inputs_manifest.json"
    pixal_inputs = {
        "manifest_sha256": "5" * 64,
        "jobs": [{"fixture": source_asset["asset_id"]}],
    }
    contracts.write_json_no_replace(pixal_inputs_path, pixal_inputs)
    pixal_batch = {
        "schema": source_registry.pixal_runner.BATCH_SCHEMA,
        "status": "passed_generation_and_glb_readback",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "pixal_inputs": {
            "path": str(pixal_inputs_path.resolve()),
            "sha256": _sha256(pixal_inputs_path),
            "manifest_sha256": pixal_inputs["manifest_sha256"],
        },
        "automatic_checks": copy.deepcopy(preparation.PIXAL_BATCH_AUTOMATIC_CHECKS),
    }
    pixal_batch["batch_sha256"] = source_registry._hash_without(
        pixal_batch, "batch_sha256"
    )
    pixal_batch_path = tmp_path / "pixal_batch_manifest.json"
    contracts.write_json_no_replace(pixal_batch_path, pixal_batch)

    static_decision_batch = {
        "schema": source_registry.static_decisions.DECISION_BATCH_SCHEMA,
        "status": "completed",
        "automatic_checks": copy.deepcopy(
            preparation.STATIC_DECISION_BATCH_AUTOMATIC_CHECKS
        ),
    }
    static_decision_batch["decision_batch_sha256"] = source_registry._hash_without(
        static_decision_batch, "decision_batch_sha256"
    )
    static_decision_batch_path = tmp_path / "static_decision_batch_manifest.json"
    contracts.write_json_no_replace(static_decision_batch_path, static_decision_batch)

    input_jobs = {source_asset["asset_id"]: {"fixture": "input"}}
    attempts = {source_asset["asset_id"]: {"instance_id": source_asset["asset_id"]}}
    decisions = {
        source_asset["asset_id"]: {
            "payload": {
                "decision": "approved_for_lod_and_binding",
                "attribute_evidence": copy.deepcopy(attribute_evidence),
            }
        }
    }

    def load_pixal_inputs(path):
        assert path == pixal_inputs_path.resolve()
        return path, copy.deepcopy(pixal_inputs)

    def validate_pixal_request_identity(batch, manifest, requests):
        assert batch["batch_sha256"] == pixal_batch["batch_sha256"]
        assert manifest["manifest_sha256"] == pixal_inputs["manifest_sha256"]
        assert source_asset["asset_id"] in requests
        return copy.deepcopy(input_jobs), copy.deepcopy(attempts)

    def load_decision_batch(path):
        assert path == static_decision_batch_path.resolve()
        return path, copy.deepcopy(static_decision_batch), copy.deepcopy(decisions)

    monkeypatch.setattr(
        source_registry.pixal_runner, "load_pixal_inputs", load_pixal_inputs
    )
    monkeypatch.setattr(
        source_registry,
        "validate_pixal_request_identity",
        validate_pixal_request_identity,
    )
    monkeypatch.setattr(source_registry, "load_decision_batch", load_decision_batch)

    registry = {
        "schema": source_registry.REGISTRY_SCHEMA,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": _sha256(preflight_path),
            "preflight_sha256": preflight["preflight_sha256"],
            "validation_mode": "frozen_historical_preflight_v1",
        },
        "pixal_batch": {
            "path": str(pixal_batch_path.resolve()),
            "sha256": _sha256(pixal_batch_path),
            "batch_sha256": pixal_batch["batch_sha256"],
        },
        "static_decision_batch": {
            "path": str(static_decision_batch_path.resolve()),
            "sha256": _sha256(static_decision_batch_path),
            "decision_batch_sha256": static_decision_batch["decision_batch_sha256"],
        },
        "source_asset_count": 1,
        "source_assets": [source_index],
        "automatic_checks": copy.deepcopy(preparation.REGISTRY_AUTOMATIC_CHECKS),
    }
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    registry_path = registry_root / "registry_manifest.json"
    contracts.write_json_no_replace(registry_path, registry)

    animated_glb = tmp_path / "target_animated.glb"
    _write_glb(animated_glb)
    input_glb = tmp_path / "target_rig.glb"
    _write_glb(input_glb)
    target_rig_lineage = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=candidate,
        target_rig_glb=input_glb,
    )
    heading_evidence = tmp_path / "forward_declaration.json"
    heading_evidence.write_text('{"fixture":true}\n', encoding="utf-8")
    source_motion = tmp_path / "source_motion.glb"
    _write_glb(source_motion)
    stage_records = {}
    for name in preparation.REQUIRED_REVIEW_OUTPUTS - {
        "animated_glb",
        "retargeted_animated_glb",
    }:
        evidence = tmp_path / f"{name}.json"
        evidence.write_text(
            json.dumps({"status": "passed", "name": name}), encoding="utf-8"
        )
        stage_records[name] = _record(evidence)
    stage_records["animated_glb"] = _record(animated_glb)
    stage_records["retargeted_animated_glb"] = _record(animated_glb)
    stage_records["gait_direction_audit"] = copy.deepcopy(
        stage_records["gait_direction_audit_initial"]
    )
    stage_records["deformation_audit"] = copy.deepcopy(
        stage_records["deformation_audit_initial"]
    )
    stage_records["gait_direction_status"] = "pass"
    stage_records["deformation_overall"] = "passed"

    review_media_root = tmp_path / "05_review"
    review_media_root.mkdir()
    first_video = review_media_root / "walking_side.mp4"
    _make_video(first_video)
    media = {}
    for name in preparation.REQUIRED_MEDIA:
        path = review_media_root / f"{name}.mp4"
        if path != first_video:
            shutil.copyfile(first_video, path)
        media[name] = preparation.generated_review.verify_video(
            path, preparation.REQUIRED_REVIEW_FRAMES
        )
    stage_records["media"] = media
    stage_records["media_lineage"] = _build_media_lineage(
        tmp_path,
        input_glb=animated_glb,
        media=media,
    )

    pipeline = preparation._review_pipeline_order(repair_branch="not_needed")
    review = {
        "schema": preparation.BRANCHED_GENERATED_REVIEW_SCHEMA,
        "created_at": "2026-07-28T00:00:00+00:00",
        "status": "research_candidate_pending_human_review",
        "formal_dataset_registration_authorized": False,
        "forward_contract": {},
        "pipeline_order": pipeline,
        "automatic_admission_gates": {
            "heading": "positive-x",
            "rig": "passed",
            "support_plane": "mesh-foot-bottoms",
            "retarget_export_front_axis": "positive-x",
            "gait_initial": "pass",
            "deformation_initial": "passed",
            "weight_repair_policy": "auto",
            "weight_repair_triggered": False,
            "weight_repair": "not_needed",
            "weight_repair_strategy": "not_needed",
            "weight_repair_branch": "not_needed",
            "weight_repair_attempts": [],
            "weight_repair_final_artifact": (
                preparation.generated_review.weight_repair_final_artifact("not_needed")
            ),
            "gait_final": "pass",
            "deformation_final": "passed",
            "all_automatic_gates_passed": True,
        },
        "inputs": {
            "target_rig_glb": _record(input_glb),
            "target_rig_lineage": target_rig_lineage,
            "heading_review_evidence": _record(heading_evidence),
            "source_motion_glb": _record(source_motion),
        },
        "outputs": stage_records,
        "timings_seconds": {name: 0.1 for name in pipeline},
    }
    review_path = tmp_path / "review_run.json"
    contracts.write_json_no_replace(review_path, review)

    decision = {
        "schema": preparation.DECISION_SCHEMA,
        "asset_id": source_asset["asset_id"],
        "review_sha256": _sha256(review_path),
        "decision": "approved_for_ue_apartment",
        "checks": {name: True for name in preparation.DECISION_CHECK_FIELDS},
        "caveats": [],
        "notes": "User approved all authenticated Walk and Idle views.",
        "review": _record(review_path),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": "ue_import_metric_trajectory_audio_and_apartment_media",
    }
    decision["decision_sha256"] = preparation._hash_without(decision, "decision_sha256")
    decision_path = tmp_path / "animation_decision.json"
    contracts.write_json_no_replace(decision_path, decision)

    def fixture_lineage(**kwargs):
        return kwargs["authenticated"]["output:animated_glb"]

    monkeypatch.setattr(preparation, "_validate_review_lineage", fixture_lineage)
    fixture = {
        "artifact_root": artifact_root,
        "registry_path": registry_path,
        "source_asset": source_asset,
        "source_path": source_path,
        "review_path": review_path,
        "decision_path": decision_path,
        "receipt_path": tmp_path / "decision_freeze_receipt.json",
        "media_path": Path(next(iter(media.values()))["path"]),
    }
    fixture["presentation_evidence"] = _write_presentation_bundle(
        tmp_path,
        review_path=review_path,
        review=review,
    )
    _rewrite_freeze_receipt(fixture)
    return fixture


def _prepare(
    fixture,
    output,
    *,
    expected_registry_sha256=None,
    expected_decision_sha256=None,
    expected_receipt_sha256=None,
):
    return preparation.prepare_import(
        source_registry_manifest_path=fixture["registry_path"],
        expected_source_registry_sha256=(
            expected_registry_sha256 or _sha256(fixture["registry_path"])
        ),
        source_asset_path=fixture["source_path"],
        animation_review_path=fixture["review_path"],
        animation_decision_path=fixture["decision_path"],
        expected_animation_decision_sha256=(
            expected_decision_sha256 or _sha256(fixture["decision_path"])
        ),
        animation_decision_freeze_receipt_path=fixture["receipt_path"],
        expected_animation_decision_freeze_receipt_sha256=(
            expected_receipt_sha256 or _sha256(fixture["receipt_path"])
        ),
        output_root=output,
        artifact_roots={"fixture_root": fixture["artifact_root"]},
    )


def test_derived_registry_cannot_fall_back_to_a_raw_only_source_asset(
    approved_generated_animal,
    monkeypatch,
):
    source_asset = copy.deepcopy(approved_generated_animal["source_asset"])
    monkeypatch.setattr(
        preparation.contracts,
        "validate_source_asset_v2",
        lambda value, **_kwargs: copy.deepcopy(source_asset),
    )

    with pytest.raises(
        contracts.ContractError,
        match="without complete repair authority",
    ):
        preparation.load_source_asset(
            approved_generated_animal["source_path"],
            {"fixture_root": approved_generated_animal["artifact_root"]},
            request={},
            profile={},
            require_derived_authority=True,
        )


def _derived_source_reader_fixture(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    paths = {}
    for role in (
        "pixal_raw_glb",
        "raw_static_decision",
        "derived_repaired_glb",
        "derived_geometry_closure",
        "derived_repair_manifest",
        "derived_geometry_audit",
        "derived_static_review_manifest",
        "derived_static_decision",
    ):
        suffix = ".glb" if role.endswith("glb") else ".json"
        path = artifact_root / f"{role}{suffix}"
        if suffix == ".json":
            _write_json(path, {"role": role})
        else:
            path.write_bytes(role.encode("utf-8"))
        paths[role] = path
    decision_batch = tmp_path / "raw_static_decision_batch.json"
    _write_json(decision_batch, {"decision_batch_sha256": "c" * 64})
    request_sha256 = "a" * 64
    asset_id = f"dog_fixture_{request_sha256[:12]}"
    source_path = tmp_path / "source_asset_v2.json"
    _write_json(source_path, {"schema": contracts.SOURCE_ASSET_SCHEMA})
    source_asset = {
        "asset_id": asset_id,
        "request_sha256": request_sha256,
        "asset_class": "animal",
        "state_classification": "research_candidate",
        "rig": {"actions": ["Idle", "Walking"]},
        "artifacts": {
            role: _root_record("fixture_root", path, artifact_root)
            for role, path in paths.items()
        },
        "rights": {"licenses": []},
    }
    raw_authority = {
        "decision": "approved_for_lod_and_binding",
        "file": _record(paths["raw_static_decision"]),
    }
    review = {
        "review_sha256": "b" * 64,
        "instance_identity": {"instance_id": asset_id},
        "source_authorities": {
            "raw_pixal_glb": _record(paths["pixal_raw_glb"]),
            "raw_static_decision": raw_authority,
            "raw_static_decision_batch": {
                "file": _record(decision_batch),
                "decision_batch_sha256": "c" * 64,
            },
        },
        "derived_geometry": {
            "repaired_glb": _record(paths["derived_repaired_glb"]),
            "geometry_closure": _record(paths["derived_geometry_closure"]),
            "repair_manifest": _record(paths["derived_repair_manifest"]),
            "independent_geometry_audit": _record(
                paths["derived_geometry_audit"]
            ),
            "repair_method": (
                source_registry.derived_review_contract
                .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
            ),
        },
    }
    decision = {
        "decision": source_registry.derived_static_decisions.APPROVED,
        "instance_id": asset_id,
        "review_binding": {
            "review_file": {
                "sha256": _sha256(
                    paths["derived_static_review_manifest"]
                )
            },
            "internal_review_sha256": review["review_sha256"],
        },
    }
    repair = {
        "implementation_contract": (
            source_registry.derived_review_contract
            .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "pixal_source": _record(paths["pixal_raw_glb"]),
            "static_decision": _record(paths["raw_static_decision"]),
            "static_decision_batch": _record(decision_batch),
        },
        "output": _record(paths["derived_repaired_glb"]),
    }
    monkeypatch.setattr(
        preparation.contracts,
        "validate_source_asset_v2",
        lambda _value, **_kwargs: copy.deepcopy(source_asset),
    )
    monkeypatch.setattr(
        source_registry.derived_static_decisions,
        "validate_decision",
        lambda _value: copy.deepcopy(decision),
    )
    monkeypatch.setattr(
        source_registry.derived_review_contract,
        "validate_review",
        lambda _value: copy.deepcopy(review),
    )
    monkeypatch.setattr(
        source_registry.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _value: copy.deepcopy(repair),
    )
    return {
        "artifact_root": artifact_root,
        "source_path": source_path,
        "source_asset": source_asset,
        "review": review,
        "repair": repair,
        "decision_batch": {
            "path": str(decision_batch.resolve()),
            "sha256": _sha256(decision_batch),
            "decision_batch_sha256": "c" * 64,
        },
    }


def test_ue_reader_accepts_oriented_sheet_approved_raw_decision(
    tmp_path,
    monkeypatch,
):
    fixture = _derived_source_reader_fixture(tmp_path, monkeypatch)

    _path, payload, authenticated = preparation.load_source_asset(
        fixture["source_path"],
        {"fixture_root": fixture["artifact_root"]},
        request={},
        profile={},
        require_derived_authority=True,
        expected_raw_static_decision_batch=fixture["decision_batch"],
    )

    assert payload["asset_id"] == fixture["source_asset"]["asset_id"]
    assert "artifact:derived_repaired_glb" in authenticated


def test_ue_reader_rejects_cross_mode_raw_decision(
    tmp_path,
    monkeypatch,
):
    fixture = _derived_source_reader_fixture(tmp_path, monkeypatch)
    mirror = (
        source_registry.derived_review_contract.REPAIR_IMPLEMENTATION_CONTRACT
    )
    fixture["repair"]["implementation_contract"] = mirror
    fixture["review"]["derived_geometry"]["repair_method"] = mirror

    with pytest.raises(
        contracts.ContractError,
        match="decision/review identity changed",
    ):
        preparation.load_source_asset(
            fixture["source_path"],
            {"fixture_root": fixture["artifact_root"]},
            request={},
            profile={},
            require_derived_authority=True,
            expected_raw_static_decision_batch=fixture["decision_batch"],
        )


def test_ue_reader_rejects_oriented_raw_decision_batch_path_rebind(
    tmp_path,
    monkeypatch,
):
    fixture = _derived_source_reader_fixture(tmp_path, monkeypatch)
    canonical_batch = Path(fixture["decision_batch"]["path"])
    rebound_batch = tmp_path / "rebound_raw_static_decision_batch.json"
    rebound_batch.write_bytes(canonical_batch.read_bytes())
    fixture["repair"]["lineage"]["static_decision_batch"] = _record(
        rebound_batch
    )

    with pytest.raises(
        contracts.ContractError,
        match="oriented repair raw decision batch was rebound",
    ):
        preparation.load_source_asset(
            fixture["source_path"],
            {"fixture_root": fixture["artifact_root"]},
            request={},
            profile={},
            require_derived_authority=True,
            expected_raw_static_decision_batch=fixture["decision_batch"],
        )


def test_legacy_derived_registry_requires_complete_derived_source_authority(
    tmp_path,
    monkeypatch,
):
    registry_path = tmp_path / "legacy_derived_registry.json"
    source_path = tmp_path / "source_asset_v2.json"
    decision_batch = {
        "path": str((tmp_path / "raw_decisions.json").resolve()),
        "sha256": "a" * 64,
        "decision_batch_sha256": "b" * 64,
    }

    monkeypatch.setattr(
        preparation,
        "load_source_registry_anchor",
        lambda *_args, **_kwargs: (
            registry_path.resolve(),
            {
                "schema": source_registry.LEGACY_DERIVED_REGISTRY_SCHEMA,
                "static_decision_batch": decision_batch,
            },
            {},
            {},
            "frozen_preflight_v1",
        ),
    )

    class DerivedAuthorityObserved(Exception):
        pass

    def load_source_asset(
        _path,
        _roots,
        *,
        request,
        profile,
        require_derived_authority,
        require_direct_geometry_authority,
        expected_raw_static_decision_batch,
    ):
        assert request == {}
        assert profile == {}
        assert require_derived_authority is True
        assert require_direct_geometry_authority is False
        assert expected_raw_static_decision_batch == decision_batch
        raise DerivedAuthorityObserved

    monkeypatch.setattr(preparation, "load_source_asset", load_source_asset)

    with pytest.raises(DerivedAuthorityObserved):
        preparation._authenticate_import_authority(
            source_registry_manifest_path=registry_path,
            expected_source_registry_sha256="c" * 64,
            source_asset_path=source_path,
            animation_review_path=tmp_path / "animation_review.json",
            animation_decision_path=tmp_path / "animation_decision.json",
            expected_animation_decision_sha256="d" * 64,
            animation_decision_freeze_receipt_path=(
                tmp_path / "animation_decision_freeze_receipt.json"
            ),
            expected_animation_decision_freeze_receipt_sha256="e" * 64,
            roots={},
        )


def _direct_registry_reader_fixture(tmp_path, monkeypatch):
    artifact_root = tmp_path / "direct_artifacts"
    artifact_root.mkdir()
    raw_glb = artifact_root / "pixal_raw.glb"
    raw_glb.write_bytes(b"direct adopted Pixel3D GLB")
    license_path = artifact_root / "LICENSE"
    license_path.write_bytes(b"research license")
    _preflight_path, _preflight, _profile, request = (
        _build_frozen_preflight(tmp_path)
    )
    source_asset = contracts.build_source_asset_v2(
        request,
        artifacts={
            "pixal_raw_glb": _root_record(
                "direct_fixture_root",
                raw_glb,
                artifact_root,
            )
        },
        physical_measurements={"status": "pending"},
        provenance={
            "attempt_id": "direct_fixture_attempt_v1",
            "request_sha256": request["request_sha256"],
            "models": copy.deepcopy(
                request["generation_plan"]["model_revisions"]
            ),
        },
        rights={
            "status": "review_required",
            "licenses": [
                _root_record(
                    "direct_fixture_root",
                    license_path,
                    artifact_root,
                )
            ],
            "blockers": ["research_candidate_only"],
        },
        qa={
            "reference_2d": "passed",
            "static_mesh": "passed",
            "binding": "pending",
            "walking": "pending",
            "idle": "pending",
            "ue_import_readback": "pending",
            "apartment_media": "pending",
            "audio": "pending",
        },
        state_classification="research_candidate",
    )
    direct_asset_id = "dog_direct_adopted_noncanonical_v1"
    direct_request_sha256 = "1" * 64
    source_asset["asset_id"] = direct_asset_id
    source_asset["request_sha256"] = direct_request_sha256
    source_asset["provenance"]["request_sha256"] = direct_request_sha256

    registry_root = tmp_path / "direct_registry"
    source_root = registry_root / "source_assets"
    source_root.mkdir(parents=True)
    source_path = source_root / f"{direct_asset_id}.json"
    _write_json(source_path, source_asset)

    pixal_path = artifact_root / "direct_adopted_pixal_batch.json"
    _write_json(pixal_path, {"fixture": "authenticated by adopter loader"})
    batch_sha256 = "2" * 64
    adopted_batch = {
        "batch_sha256": batch_sha256,
        "models": copy.deepcopy(source_asset["provenance"]["models"]),
    }
    attempt = {
        "instance_id": direct_asset_id,
        "execution_job_id": "direct_fixture_execution_v1",
        "profile_schema_id": source_asset["profile_schema_id"],
        "request_sha256": direct_request_sha256,
        "sampled_attributes": copy.deepcopy(
            source_asset["sampled_attributes"]
        ),
        "target_physical_profile": copy.deepcopy(
            source_asset["target_physical_profile"]
        ),
    }
    authority_sha256 = "3" * 64
    authority_path = artifact_root / "direct_source_authority.json"
    _write_json(authority_path, {"fixture": "authenticated direct authority"})
    source_spec_path = artifact_root / "adoption_spec.json"
    _write_json(source_spec_path, {"fixture": "source spec"})
    authority = {
        "schema": preparation.direct_adopter.SOURCE_AUTHORITY_SCHEMA,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "adopted_batch": {
            "file": _record(pixal_path),
            "batch_sha256": batch_sha256,
        },
        "source_spec": {"fixture": "loader-authenticated"},
        "instance_id": direct_asset_id,
        "profile_schema_id": source_asset["profile_schema_id"],
        "profile_sha256": source_asset["profile_sha256"],
        "request_sha256": direct_request_sha256,
        "taxonomy": copy.deepcopy(source_asset["taxonomy"]),
        "fixed_attributes": copy.deepcopy(source_asset["fixed_attributes"]),
        "lineage_group_id": source_asset["lineage_group_id"],
        "acoustic_profile": copy.deepcopy(
            source_asset["acoustic_profile"]
        ),
        "authority_sha256": authority_sha256,
    }
    context = {
        "adopted_batch_path": pixal_path.resolve(),
        "adopted_batch": adopted_batch,
        "attempt": attempt,
        "source_spec_path": source_spec_path.resolve(),
        "source_spec": {"fixture": "source spec"},
        "adoption_context": {
            "controlled": {
                "rig_profile": copy.deepcopy(source_asset["rig"])
            },
            "job": {"fixture": "direct job"},
        },
    }
    source_asset["artifacts"].update(
        {
            "direct_source_authority": _root_record(
                "direct_fixture_root",
                authority_path,
                artifact_root,
            ),
            "adopted_pixal_batch": _root_record(
                "direct_fixture_root",
                pixal_path,
                artifact_root,
            ),
            "direct_adoption_spec": _root_record(
                "direct_fixture_root",
                source_spec_path,
                artifact_root,
            ),
        }
    )
    _write_json(source_path, source_asset)

    def load_direct_authority(path, *, expected_sha256=None):
        assert Path(path) == authority_path.resolve()
        assert expected_sha256 == _sha256(authority_path)
        return (
            authority_path.resolve(),
            copy.deepcopy(authority),
            copy.deepcopy(context),
        )

    monkeypatch.setattr(
        preparation.direct_adopter,
        "load_direct_source_authority",
        load_direct_authority,
    )

    static_batch = {
        "schema": source_registry.static_decisions.DECISION_BATCH_SCHEMA,
        "status": "completed",
        "automatic_checks": copy.deepcopy(
            preparation.STATIC_DECISION_BATCH_AUTOMATIC_CHECKS
        ),
    }
    static_batch["decision_batch_sha256"] = source_registry._hash_without(
        static_batch,
        "decision_batch_sha256",
    )
    static_batch_path = tmp_path / "direct_static_decision_batch.json"
    _write_json(static_batch_path, static_batch)
    attribute_evidence = {
        name: "passed_static_visual"
        for name in source_asset["sampled_attributes"]
    }
    raw_decisions = {
        direct_asset_id: {
            "payload": {
                "decision": "approved_for_lod_and_binding",
                "attribute_evidence": copy.deepcopy(attribute_evidence),
            }
        }
    }

    def load_decision_batch(path):
        assert path == static_batch_path.resolve()
        return (
            path,
            copy.deepcopy(static_batch),
            copy.deepcopy(raw_decisions),
        )

    monkeypatch.setattr(
        source_registry,
        "load_decision_batch",
        load_decision_batch,
    )
    derived_decision_path = tmp_path / "derived_static_decision.json"
    _write_json(derived_decision_path, {"fixture": "derived decision"})
    derived_decision = {
        "decision": source_registry.derived_static_decisions.APPROVED,
        "instance_id": direct_asset_id,
        "decision_sha256": "4" * 64,
        "attribute_evidence": copy.deepcopy(attribute_evidence),
    }
    monkeypatch.setattr(
        source_registry.derived_static_decisions,
        "validate_decision",
        lambda _value: copy.deepcopy(derived_decision),
    )
    direct_checks = copy.deepcopy(
        source_registry.DIRECT_DERIVED_REGISTRY_AUTOMATIC_CHECKS
    )
    source_index = {
        "asset_id": direct_asset_id,
        "profile_schema_id": source_asset["profile_schema_id"],
        "request_sha256": direct_request_sha256,
        "sampled_attributes": copy.deepcopy(
            source_asset["sampled_attributes"]
        ),
        "attribute_evidence": copy.deepcopy(attribute_evidence),
        "source_asset": _relative_record(source_path, registry_root),
        "state_classification": "research_candidate",
        "next_gate": "lod_then_species_rig_binding",
    }
    registry = {
        "schema": source_registry.DERIVED_REGISTRY_SCHEMA,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "direct_source_authority": {
            **_record(authority_path),
            "authority_sha256": authority_sha256,
        },
        "pixal_batch": {
            "path": str(pixal_path.resolve()),
            "sha256": _sha256(pixal_path),
            "batch_sha256": batch_sha256,
        },
        "static_decision_batch": {
            "path": str(static_batch_path.resolve()),
            "sha256": _sha256(static_batch_path),
            "decision_batch_sha256": static_batch[
                "decision_batch_sha256"
            ],
        },
        "derived_static_decisions": [
            {
                "path": str(derived_decision_path.resolve()),
                "sha256": _sha256(derived_decision_path),
                "decision_sha256": derived_decision["decision_sha256"],
            }
        ],
        "source_asset_count": 1,
        "source_assets": [source_index],
        "automatic_checks": copy.deepcopy(direct_checks),
    }
    registry["registry_sha256"] = source_registry._hash_without(
        registry,
        "registry_sha256",
    )
    registry_path = registry_root / "registry_manifest.json"
    _write_json(registry_path, registry)
    return {
        "artifact_root": artifact_root,
        "authority": authority,
        "context": context,
        "pixal_path": pixal_path,
        "registry_path": registry_path,
        "source_asset": source_asset,
        "source_path": source_path,
        "canonical_request": request,
        "request_batch_path": tmp_path / "inputs" / "instance_requests.json",
    }


def _direct_geometry_v4_reader_fixture(tmp_path, monkeypatch):
    from tools import (
        publish_generated_animal_geometry_closure as geometry_closures,
    )

    fixture = _direct_registry_reader_fixture(tmp_path, monkeypatch)
    artifact_root = fixture["artifact_root"]
    source_path = fixture["source_path"]
    context = fixture["context"]
    instance_id = fixture["authority"]["instance_id"]

    def artifact(name, payload):
        path = artifact_root / name
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            _write_json(path, payload)
        return path

    original_raw = artifact_root / "pixal_raw.glb"
    adopted_raw = artifact("adopted_raw.glb", original_raw.read_bytes())
    original_manifest = artifact(
        "original_attempt.json",
        {"fixture": "original Pixel3D attempt"},
    )
    adopted_manifest = artifact(
        "adopted_attempt.json",
        original_manifest.read_bytes(),
    )
    original_input = artifact("original_input.png", b"Pixel3D RGBA bytes")
    adopted_input = artifact("adopted_input.png", original_input.read_bytes())
    review_reference = artifact(
        "raw_review_reference.png",
        original_input.read_bytes(),
    )
    review_batch = artifact("raw_static_review_batch.json", {"fixture": "batch"})
    raw_review = artifact("raw_static_review.json", {"fixture": "review"})
    raw_contact = artifact("raw_static_contact.png", b"raw contact sheet")
    raw_decision = artifact("raw_static_decision.json", {"fixture": "decision"})
    decision_batch = artifact(
        "raw_static_decision_batch.json",
        {"fixture": "decision batch"},
    )
    repaired = artifact("repaired.glb", b"bounded repaired Pixel3D geometry")
    repair_manifest = artifact("repair_manifest.json", {"fixture": "repair"})
    geometry_audit = artifact("geometry_audit.json", {"fixture": "audit"})
    clay_manifest = artifact("clay_manifest.json", {"fixture": "clay"})
    clay_contact = artifact("clay_contact.png", b"clay contact sheet")
    geometry_closure = artifact(
        "geometry_closure.json",
        {"fixture": "geometry closure"},
    )

    context["attempt"].update(
        {
            "output": _relative_record(adopted_raw, artifact_root),
            "attempt_manifest": _relative_record(
                adopted_manifest,
                artifact_root,
            ),
            "pixal_input": _relative_record(adopted_input, artifact_root),
        }
    )
    context["source_spec"]["instance_id"] = instance_id
    context["adoption_context"]["controlled"].update(
        {
            "request_sha256": fixture["authority"]["request_sha256"],
            "profile_schema_id": fixture["authority"]["profile_schema_id"],
        }
    )
    context["adoption_context"]["bundle_sources"] = {
        "pixal_raw_glb": original_raw,
        "pixal_attempt_manifest": original_manifest,
        "pixal_input_rgba": original_input,
    }
    checks = {
        name: True for name in source_registry.static_decisions.CHECK_FIELDS
    }
    attribute_evidence = {
        name: "passed_static_visual"
        for name in fixture["source_asset"]["sampled_attributes"]
    }
    decision_payload = {
        "decision": "approved_for_lod_and_binding",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": "lod_then_species_rig_binding",
        "decision_sha256": "8" * 64,
        "checks": checks,
        "attribute_evidence": copy.deepcopy(attribute_evidence),
    }
    review_payload = {
        "instance_id": instance_id,
        "request_sha256": fixture["authority"]["request_sha256"],
        "profile_schema_id": fixture["authority"]["profile_schema_id"],
        "sampled_attributes": copy.deepcopy(
            context["attempt"]["sampled_attributes"]
        ),
        "target_physical_profile": copy.deepcopy(
            context["attempt"]["target_physical_profile"]
        ),
        "pixal_output": copy.deepcopy(context["attempt"]["output"]),
        "reference_rgba": _relative_record(
            review_reference,
            artifact_root,
        ),
        "contact_sheet": _relative_record(raw_contact, artifact_root),
    }
    decisions = {
        instance_id: {
            "path": raw_decision.resolve(),
            "payload": decision_payload,
            "static_review": {
                "path": raw_review.resolve(),
                "payload": review_payload,
            },
        }
    }
    decision_batch_payload = {
        "schema": source_registry.static_decisions.DECISION_BATCH_SCHEMA,
        "status": "completed",
        "static_review_batch": {"path": str(review_batch.resolve())},
        "automatic_checks": copy.deepcopy(
            preparation.STATIC_DECISION_BATCH_AUTOMATIC_CHECKS
        ),
    }
    decision_batch_payload["decision_batch_sha256"] = (
        source_registry._hash_without(
            decision_batch_payload,
            "decision_batch_sha256",
        )
    )
    _write_json(decision_batch, decision_batch_payload)

    def load_decision_batch(path):
        assert Path(path) == decision_batch.resolve()
        return (
            decision_batch.resolve(),
            copy.deepcopy(decision_batch_payload),
            copy.deepcopy(decisions),
        )

    monkeypatch.setattr(
        source_registry,
        "load_decision_batch",
        load_decision_batch,
    )
    internal_manifest_sha256 = "9" * 64
    replay = {
        "manifest": {
            "schema": geometry_closures.SCHEMA,
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "manifest_sha256": internal_manifest_sha256,
            "bounded_repair": {
                "implementation_contract": (
                    geometry_closures.oriented_repair.IMPLEMENTATION_CONTRACT
                ),
                "mutation_class": geometry_closures.MUTATION_CLASS,
                "removed_exact_position_degenerate_triangle_count": 1,
                "output_byte_identical_to_raw": False,
            },
            "inherited_static_judgment": {
                "authority": "canonical_raw_static_approval_v1",
                "decision_sha256": decision_payload["decision_sha256"],
                "checks": copy.deepcopy(checks),
                "inheritance_scope": geometry_closures.INHERITANCE_SCOPE,
                "new_human_approval_created": False,
                "clay_render_human_approval_claimed": False,
            },
            "downstream": {
                "tokenrig_entry_authorized": True,
                "tokenrig_execution_performed": False,
                "formal_dataset_registration_authorized": False,
            },
        },
        "paths": {
            "raw_pixal_glb": original_raw.resolve(),
            "pixal_manifest": original_manifest.resolve(),
            "source_reference": review_reference.resolve(),
            "raw_static_decision_batch": decision_batch.resolve(),
            "raw_static_decision": raw_decision.resolve(),
            "raw_static_review": raw_review.resolve(),
            "repair_manifest": repair_manifest.resolve(),
            "repaired_glb": repaired.resolve(),
            "geometry_audit": geometry_audit.resolve(),
            "clay_render_manifest": clay_manifest.resolve(),
            "clay_contact_sheet": clay_contact.resolve(),
        },
        "repair": {"fixture": "strictly replayed repair"},
        "audit": {"fixture": "strictly replayed audit"},
    }
    closure_calls = []

    def load_geometry_closure(path, **kwargs):
        closure_calls.append((Path(path), copy.deepcopy(kwargs)))
        return copy.deepcopy(replay)

    monkeypatch.setattr(
        geometry_closures,
        "load_geometry_closure_v2",
        load_geometry_closure,
    )

    source_asset = copy.deepcopy(fixture["source_asset"])
    source_asset["provenance"]["attempt_id"] = (
        f"direct_geometry_{context['attempt']['execution_job_id']}"
    )
    source_asset["provenance"]["models"].update(
        {
            source_registry.DIRECT_GEOMETRY_RAW_DECISION_PROVENANCE_MODEL: (
                decision_payload["decision_sha256"]
            ),
            source_registry.DIRECT_GEOMETRY_CLOSURE_PROVENANCE_MODEL: (
                internal_manifest_sha256
            ),
        }
    )
    artifact_paths = {
        "source_reference_2d": review_reference,
        "pixal_input_rgba": adopted_input,
        "pixal_raw_glb": original_raw,
        "pixal_attempt_manifest": original_manifest,
        "raw_static_review_manifest": raw_review,
        "raw_static_contact_sheet": raw_contact,
        "raw_static_decision": raw_decision,
        "raw_static_decision_batch": decision_batch,
        "derived_repaired_glb": repaired,
        "derived_geometry_closure": geometry_closure,
        "derived_repair_manifest": repair_manifest,
        "derived_geometry_audit": geometry_audit,
        "derived_clay_render_manifest": clay_manifest,
        "derived_clay_contact_sheet": clay_contact,
        "direct_source_authority": (
            artifact_root / "direct_source_authority.json"
        ),
        "adopted_pixal_batch": fixture["pixal_path"],
        "direct_adoption_spec": artifact_root / "adoption_spec.json",
    }
    source_asset["artifacts"] = {
        role: _root_record(
            "direct_fixture_root",
            path,
            artifact_root,
        )
        for role, path in artifact_paths.items()
    }
    _write_json(source_path, source_asset)

    registry_root = fixture["registry_path"].parent
    source_index = {
        "asset_id": instance_id,
        "profile_schema_id": source_asset["profile_schema_id"],
        "request_sha256": source_asset["request_sha256"],
        "sampled_attributes": copy.deepcopy(source_asset["sampled_attributes"]),
        "attribute_evidence": copy.deepcopy(attribute_evidence),
        "source_asset": _relative_record(source_path, registry_root),
        "state_classification": "research_candidate",
        "next_gate": "lod_then_species_rig_binding",
    }
    registry = {
        "schema": source_registry.DIRECT_GEOMETRY_REGISTRY_SCHEMA,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "direct_source_authority": {
            **_record(artifact_root / "direct_source_authority.json"),
            "authority_sha256": fixture["authority"]["authority_sha256"],
        },
        "pixal_batch": {
            "path": str(fixture["pixal_path"].resolve()),
            "sha256": _sha256(fixture["pixal_path"]),
            "batch_sha256": context["adopted_batch"]["batch_sha256"],
        },
        "static_decision_batch": {
            "path": str(decision_batch.resolve()),
            "sha256": _sha256(decision_batch),
            "decision_batch_sha256": decision_batch_payload[
                "decision_batch_sha256"
            ],
        },
        "geometry_closure": {
            "path": str(geometry_closure.resolve()),
            "sha256": _sha256(geometry_closure),
            "manifest_sha256": internal_manifest_sha256,
        },
        "source_asset_count": 1,
        "source_assets": [source_index],
        "automatic_checks": copy.deepcopy(
            source_registry.DIRECT_GEOMETRY_REGISTRY_AUTOMATIC_CHECKS
        ),
    }
    registry["registry_sha256"] = source_registry._hash_without(
        registry,
        "registry_sha256",
    )
    _write_json(fixture["registry_path"], registry)
    fixture.update(
        {
            "source_asset": source_asset,
            "artifact_paths": artifact_paths,
            "decision_batch": decision_batch,
            "geometry_closure": geometry_closure,
            "closure_calls": closure_calls,
            "decision_payload": decision_payload,
            "decision_batch_payload": decision_batch_payload,
            "decisions": decisions,
            "replay": replay,
        }
    )
    return fixture


def _load_direct_geometry_v4_fixture(fixture):
    anchor = preparation.load_source_registry_anchor(
        fixture["registry_path"],
        fixture["source_path"],
        expected_file_sha256=_sha256(fixture["registry_path"]),
    )
    source = preparation.load_source_asset(
        fixture["source_path"],
        {"direct_fixture_root": fixture["artifact_root"]},
        request=anchor[2],
        profile=anchor[3],
        require_direct_geometry_authority=True,
        expected_raw_static_decision_batch=anchor[1][
            "static_decision_batch"
        ],
    )
    return anchor, source


def test_direct_geometry_v4_registry_round_trip_replays_closure_and_source_asset(
    tmp_path,
    monkeypatch,
):
    fixture = _direct_geometry_v4_reader_fixture(tmp_path, monkeypatch)

    anchor, (_source_path, _source_asset, authenticated) = (
        _load_direct_geometry_v4_fixture(fixture)
    )

    assert anchor[1]["schema"] == (
        source_registry.DIRECT_GEOMETRY_REGISTRY_SCHEMA
    )
    assert anchor[4] == "direct_geometry_source_authority_v1"
    assert len(fixture["closure_calls"]) == 1
    closure_path, closure_kwargs = fixture["closure_calls"][0]
    assert closure_path == fixture["geometry_closure"].resolve()
    assert closure_kwargs["expected_raw_pixal_glb"] == (
        fixture["artifact_paths"]["pixal_raw_glb"].resolve()
    )
    assert closure_kwargs["expected_raw_static_decision_batch"] == (
        fixture["decision_batch"].resolve()
    )
    assert authenticated["artifact:derived_repaired_glb"] == (
        fixture["artifact_paths"]["derived_repaired_glb"].resolve()
    )
    assert set(anchor[3]["direct_geometry_authority"]["source_artifacts"]) == (
        preparation.DIRECT_GEOMETRY_SOURCE_ARTIFACT_ROLES
    )


def test_direct_geometry_v4_reauthenticates_physical_profile_authority(
    tmp_path,
    monkeypatch,
):
    fixture = _direct_geometry_v4_reader_fixture(tmp_path, monkeypatch)
    role = source_registry.PHYSICAL_PROFILE_AUTHORITY_ARTIFACT_ROLE
    model = source_registry.PHYSICAL_PROFILE_AUTHORITY_REQUEST_MODEL
    authority_batch = fixture["artifact_root"] / "physical_authority_batch.json"
    shutil.copyfile(fixture["request_batch_path"], authority_batch)
    fixture["context"]["attempt"]["target_physical_profile"].pop(
        "reference_provenance"
    )
    fixture["decisions"][fixture["authority"]["instance_id"]][
        "static_review"
    ]["payload"]["target_physical_profile"].pop("reference_provenance")
    fixture["source_asset"]["artifacts"][role] = _root_record(
        "direct_fixture_root",
        authority_batch,
        fixture["artifact_root"],
    )
    fixture["source_asset"]["provenance"]["models"][model] = fixture[
        "canonical_request"
    ]["request_sha256"]
    fixture["artifact_paths"][role] = authority_batch
    _write_json(fixture["source_path"], fixture["source_asset"])
    registry = contracts.load_json(fixture["registry_path"])
    registry["source_assets"][0]["source_asset"] = _relative_record(
        fixture["source_path"],
        fixture["registry_path"].parent,
    )
    registry["registry_sha256"] = source_registry._hash_without(
        registry,
        "registry_sha256",
    )
    _write_json(fixture["registry_path"], registry)

    anchor, (_path, source_asset, authenticated) = (
        _load_direct_geometry_v4_fixture(fixture)
    )

    assert source_asset["target_physical_profile"]["reference_provenance"]
    assert authenticated[f"artifact:{role}"] == authority_batch.resolve()
    assert set(anchor[3]["direct_geometry_authority"]["source_artifacts"]) == (
        preparation.DIRECT_GEOMETRY_SOURCE_ARTIFACT_ROLES
    )


def test_direct_geometry_v4_actual_producer_round_trip_into_ue_bridge(
    tmp_path,
    monkeypatch,
):
    fixture = _direct_geometry_v4_reader_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(source_registry, "SPEAR_ROOT", tmp_path)
    monkeypatch.setattr(
        source_registry.direct_adopter,
        "load_adopted_batch",
        lambda _path: (
            fixture["pixal_path"].resolve(),
            copy.deepcopy(fixture["context"]["adopted_batch"]),
        ),
    )
    produced_root = tmp_path / "actual_v4_registry"
    produced_registry = source_registry.register_direct_geometry(
        fixture["artifact_root"] / "direct_source_authority.json",
        _sha256(
            fixture["artifact_root"] / "direct_source_authority.json"
        ),
        fixture["pixal_path"],
        fixture["decision_batch"],
        fixture["geometry_closure"],
        _sha256(fixture["geometry_closure"]),
        produced_root,
    )
    produced_source = (
        produced_root
        / "source_assets"
        / f"{fixture['authority']['instance_id']}.json"
    )

    anchor = preparation.load_source_registry_anchor(
        produced_registry,
        produced_source,
        expected_file_sha256=_sha256(produced_registry),
    )
    _source_path, source_asset, authenticated = preparation.load_source_asset(
        produced_source,
        {
            "spear_repo": tmp_path,
            "models_root": Path("/data/models"),
        },
        request=anchor[2],
        profile=anchor[3],
        require_direct_geometry_authority=True,
        expected_raw_static_decision_batch=anchor[1][
            "static_decision_batch"
        ],
    )

    assert source_asset["provenance"]["attempt_id"].startswith(
        "direct_geometry_"
    )
    assert source_asset["provenance"]["models"][
        source_registry.DIRECT_GEOMETRY_RAW_DECISION_PROVENANCE_MODEL
    ] == fixture["decision_payload"]["decision_sha256"]
    assert authenticated["artifact:pixal_raw_glb"] == (
        fixture["artifact_paths"]["pixal_raw_glb"].resolve()
    )
    assert anchor[4] == "direct_geometry_source_authority_v1"


def test_direct_geometry_v4_registry_rejects_resealed_internal_closure_hash(
    tmp_path,
    monkeypatch,
):
    fixture = _direct_geometry_v4_reader_fixture(tmp_path, monkeypatch)
    registry = contracts.load_json(fixture["registry_path"])
    registry["geometry_closure"]["manifest_sha256"] = "a" * 64
    registry["registry_sha256"] = source_registry._hash_without(
        registry,
        "registry_sha256",
    )
    _write_json(fixture["registry_path"], registry)

    with pytest.raises(
        contracts.ContractError,
        match="closure replay result is invalid",
    ):
        preparation.load_source_registry_anchor(
            fixture["registry_path"],
            fixture["source_path"],
            expected_file_sha256=_sha256(fixture["registry_path"]),
        )


@pytest.mark.parametrize("mutation", ("automatic_checks", "schema_downgrade"))
def test_direct_geometry_v4_registry_rejects_check_or_schema_masquerade(
    tmp_path,
    monkeypatch,
    mutation,
):
    fixture = _direct_geometry_v4_reader_fixture(tmp_path, monkeypatch)
    registry = contracts.load_json(fixture["registry_path"])
    if mutation == "automatic_checks":
        registry["automatic_checks"][
            "exact_geometry_closure_replayed"
        ] = False
    else:
        registry["schema"] = source_registry.DERIVED_REGISTRY_SCHEMA
    registry["registry_sha256"] = source_registry._hash_without(
        registry,
        "registry_sha256",
    )
    _write_json(fixture["registry_path"], registry)

    with pytest.raises(
        contracts.ContractError,
        match="automatic checks are invalid|contract/hash is invalid",
    ):
        preparation.load_source_registry_anchor(
            fixture["registry_path"],
            fixture["source_path"],
            expected_file_sha256=_sha256(fixture["registry_path"]),
        )


def test_direct_geometry_v4_source_asset_rejects_arbitrary_repaired_body(
    tmp_path,
    monkeypatch,
):
    fixture = _direct_geometry_v4_reader_fixture(tmp_path, monkeypatch)
    anchor = preparation.load_source_registry_anchor(
        fixture["registry_path"],
        fixture["source_path"],
        expected_file_sha256=_sha256(fixture["registry_path"]),
    )
    arbitrary = fixture["artifact_root"] / "arbitrary_template_body.glb"
    arbitrary.write_bytes(b"Rocketbox or Quaternius substitute body")
    source_asset = contracts.load_json(fixture["source_path"])
    source_asset["artifacts"]["derived_repaired_glb"] = _root_record(
        "direct_fixture_root",
        arbitrary,
        fixture["artifact_root"],
    )
    _write_json(fixture["source_path"], source_asset)

    with pytest.raises(
        contracts.ContractError,
        match="derived_repaired_glb changed from registry closure replay",
    ):
        preparation.load_source_asset(
            fixture["source_path"],
            {"direct_fixture_root": fixture["artifact_root"]},
            request=anchor[2],
            profile=anchor[3],
            require_direct_geometry_authority=True,
            expected_raw_static_decision_batch=anchor[1][
                "static_decision_batch"
            ],
        )


def test_direct_registry_reader_accepts_noncanonical_adopted_identity(
    tmp_path,
    monkeypatch,
):
    fixture = _direct_registry_reader_fixture(tmp_path, monkeypatch)

    (
        registry_path,
        registry,
        source_authority,
        source_context,
        validation_mode,
    ) = preparation.load_source_registry_anchor(
        fixture["registry_path"],
        fixture["source_path"],
        expected_file_sha256=_sha256(fixture["registry_path"]),
    )
    source_path, source_asset, artifacts = preparation.load_source_asset(
        fixture["source_path"],
        {"direct_fixture_root": fixture["artifact_root"]},
        request=source_authority,
        profile=source_context,
    )

    assert registry_path == fixture["registry_path"].resolve()
    assert "preflight" not in registry
    assert source_authority == fixture["authority"]
    assert source_context["attempt"] == fixture["context"]["attempt"]
    assert validation_mode == "direct_source_authority_v1"
    assert source_path == fixture["source_path"].resolve()
    assert source_asset["asset_id"] == "dog_direct_adopted_noncanonical_v1"
    assert not source_asset["asset_id"].endswith(
        source_asset["request_sha256"][:12]
    )
    assert artifacts["artifact:pixal_raw_glb"] == (
        fixture["artifact_root"] / "pixal_raw.glb"
    )


def test_direct_registry_reader_rejects_adopted_batch_path_rebind(
    tmp_path,
    monkeypatch,
):
    fixture = _direct_registry_reader_fixture(tmp_path, monkeypatch)
    rebound = tmp_path / "rebound_direct_adopted_pixal_batch.json"
    rebound.write_bytes(fixture["pixal_path"].read_bytes())
    registry = contracts.load_json(fixture["registry_path"])
    registry["pixal_batch"]["path"] = str(rebound.resolve())
    registry["pixal_batch"]["sha256"] = _sha256(rebound)
    registry["registry_sha256"] = source_registry._hash_without(
        registry,
        "registry_sha256",
    )
    _write_json(fixture["registry_path"], registry)

    with pytest.raises(
        contracts.ContractError,
        match="direct adopted Pixal batch identity changed",
    ):
        preparation.load_source_registry_anchor(
            fixture["registry_path"],
            fixture["source_path"],
            expected_file_sha256=_sha256(fixture["registry_path"]),
        )


@pytest.mark.parametrize("mutation", ["missing", "rebound"])
def test_direct_source_asset_requires_registry_bound_lineage_artifacts(
    tmp_path,
    monkeypatch,
    mutation,
):
    fixture = _direct_registry_reader_fixture(tmp_path, monkeypatch)
    source_asset = contracts.load_json(fixture["source_path"])
    if mutation == "missing":
        source_asset["artifacts"].pop("direct_adoption_spec")
    else:
        replacement = fixture["artifact_root"] / "replacement_authority.json"
        replacement.write_bytes(
            (
                fixture["artifact_root"] / "direct_source_authority.json"
            ).read_bytes()
        )
        source_asset["artifacts"]["direct_source_authority"] = _root_record(
            "direct_fixture_root",
            replacement,
            fixture["artifact_root"],
        )
    _write_json(fixture["source_path"], source_asset)
    registry = contracts.load_json(fixture["registry_path"])
    registry["source_assets"][0]["source_asset"] = _relative_record(
        fixture["source_path"],
        fixture["registry_path"].parent,
    )
    registry["registry_sha256"] = source_registry._hash_without(
        registry,
        "registry_sha256",
    )
    _write_json(fixture["registry_path"], registry)
    (
        _registry_path,
        _registry,
        source_authority,
        source_context,
        _validation_mode,
    ) = preparation.load_source_registry_anchor(
        fixture["registry_path"],
        fixture["source_path"],
        expected_file_sha256=_sha256(fixture["registry_path"]),
    )

    with pytest.raises(
        contracts.ContractError,
        match="direct source asset lineage artifacts",
    ):
        preparation.load_source_asset(
            fixture["source_path"],
            {"direct_fixture_root": fixture["artifact_root"]},
            request=source_authority,
            profile=source_context,
        )


def _rewrite_review_and_rebind_decision(fixture, review):
    _write_json(fixture["review_path"], review)
    decision = contracts.load_json(fixture["decision_path"])
    decision["review"] = _record(fixture["review_path"])
    decision["review_sha256"] = decision["review"]["sha256"]
    decision["decision_sha256"] = preparation._hash_without(decision, "decision_sha256")
    _write_json(fixture["decision_path"], decision)
    _refresh_presentation_evidence(fixture, review)
    _rewrite_freeze_receipt(fixture, review=review)


def _rewrite_freeze_receipt(fixture, *, review=None):
    registry = contracts.load_json(fixture["registry_path"])
    if review is None:
        review = contracts.load_json(fixture["review_path"])
    decision = contracts.load_json(fixture["decision_path"])
    gates = review["automatic_admission_gates"]
    legacy = review["schema"] == preparation.LEGACY_GENERATED_REVIEW_SCHEMA
    try:
        branch = preparation._repair_branch_contract(
            gates,
            review_schema=review["schema"],
        )[1]
        descriptor_names = set(preparation.REQUIRED_REVIEW_OUTPUTS)
        descriptor_names.update(
            preparation._repair_output_names(legacy=legacy, branch=branch)
        )
        review_artifact_count = (
            len(preparation.REVIEW_FILE_INPUTS)
            + len(descriptor_names)
            + len(preparation.REQUIRED_MEDIA)
            + (2 * len(preparation.REQUIRED_MEDIA) if not legacy else 0)
        )
    except contracts.ContractError:
        if not fixture["receipt_path"].is_file():
            raise
        review_artifact_count = contracts.load_json(fixture["receipt_path"])[
            "authenticated_review_artifact_count"
        ]
    presentation_evidence = copy.deepcopy(fixture["presentation_evidence"])
    if (
        presentation_evidence.get("mode")
        == preparation.MOTION_STYLE_AND_CURRENT_READBACK_MODE
    ):
        user_instruction_binding = {
            "decision": "approved_for_ue_apartment",
            "motion_style_approval_file_sha256": presentation_evidence[
                "motion_style_approval"
            ]["sha256"],
            "current_asset_short_readback_file_sha256": presentation_evidence[
                "current_asset_short_readback"
            ]["sha256"],
            "current_asset_readback_is_machine_gate": True,
        }
    else:
        user_instruction_binding = {
            "decision": "approved_for_ue_apartment",
            "review_sha256": _sha256(fixture["review_path"]),
            "all_six_checks_explicit": True,
            "presentation_receipt_file_sha256": presentation_evidence[
                "expected_presentation_receipt_file_sha256"
            ],
        }
    receipt = {
        "schema": preparation.DECISION_FREEZE_RECEIPT_SCHEMA,
        "status": "frozen",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "source_asset_registry": _record(fixture["registry_path"]),
        "expected_source_asset_registry_file_sha256": _sha256(fixture["registry_path"]),
        "source_asset_registry_validation_mode": registry["preflight"][
            "validation_mode"
        ],
        "source_asset": _record(fixture["source_path"]),
        "animation_review": _record(fixture["review_path"]),
        "expected_animation_review_file_sha256": _sha256(fixture["review_path"]),
        "user_instruction_binding": user_instruction_binding,
        "user_instruction_authority": copy.deepcopy(
            preparation.USER_INSTRUCTION_AUTHORITY
        ),
        "authenticated_review_artifact_count": review_artifact_count,
        "animation_decision": _relative_record(
            fixture["decision_path"],
            fixture["receipt_path"].parent,
        ),
        "decision_sha256": decision["decision_sha256"],
        "presentation_evidence": presentation_evidence,
    }
    receipt["receipt_sha256"] = preparation._hash_without(
        receipt,
        "receipt_sha256",
    )
    _write_json(fixture["receipt_path"], receipt)


def _use_compact_approval_evidence(
    fixture,
    *,
    rebind_current_readback_to_style_video=False,
):
    style_video = Path(
        fixture["presentation_evidence"]["output_video"]["path"]
    )
    style_approval = {
        "schema": preparation.MOTION_STYLE_APPROVAL_SCHEMA,
        "status": "approved_for_idle_walking_motion_style",
        "actions": copy.deepcopy(preparation.ANIMATION_ACTIONS),
        "evidence_video": _record(style_video),
    }
    style_approval["approval_sha256"] = preparation._hash_without(
        style_approval,
        "approval_sha256",
    )
    style_path = fixture["receipt_path"].parent / "motion_style_approval.json"
    _write_json(style_path, style_approval)
    review = contracts.load_json(fixture["review_path"])
    reviewed_glb = Path(review["outputs"]["animated_glb"]["path"])
    review_media = review["outputs"]["media"]
    action_readbacks = {
        "Idle": {
            name: review_media["idle_side"][name]
            for name in ("path", "sha256", "size_bytes")
        },
        "Walking": {
            name: review_media["walking_side"][name]
            for name in ("path", "sha256", "size_bytes")
        },
    }
    if rebind_current_readback_to_style_video:
        action_readbacks["Walking"] = _record(style_video)
    short_readback = {
        "schema": preparation.CURRENT_ASSET_SHORT_READBACK_SCHEMA,
        "status": "passed_current_asset_geometry_and_actions",
        "asset_id": fixture["source_asset"]["asset_id"],
        "animation_review": _record(fixture["review_path"]),
        "reviewed_animated_glb": _record(reviewed_glb),
        "actions": copy.deepcopy(preparation.ANIMATION_ACTIONS),
        "action_readbacks": action_readbacks,
        "checks": copy.deepcopy(
            preparation.CURRENT_ASSET_SHORT_READBACK_CHECKS
        ),
    }
    short_readback["receipt_sha256"] = preparation._hash_without(
        short_readback,
        "receipt_sha256",
    )
    short_readback_path = (
        fixture["receipt_path"].parent / "current_asset_short_readback.json"
    )
    _write_json(short_readback_path, short_readback)
    fixture["presentation_evidence"] = {
        "mode": preparation.MOTION_STYLE_AND_CURRENT_READBACK_MODE,
        "motion_style_approval": {
            **_record(style_path),
            "approval_sha256": style_approval["approval_sha256"],
        },
        "current_asset_short_readback": {
            **_record(short_readback_path),
            "receipt_sha256": short_readback["receipt_sha256"],
        },
    }
    _rewrite_freeze_receipt(fixture, review=review)


def _assert_failed_without_preparation_output(output):
    assert not output.exists()
    assert not output.is_symlink()
    assert not list(output.parent.glob(f".{output.name}.*.staging"))


def _reauthenticate_registry_source(fixture):
    registry = contracts.load_json(fixture["registry_path"])
    descriptor = registry["source_assets"][0]["source_asset"]
    descriptor["sha256"] = _sha256(fixture["source_path"])
    descriptor["size_bytes"] = fixture["source_path"].stat().st_size
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    _write_json(fixture["registry_path"], registry)
    _rewrite_freeze_receipt(fixture)


def _write_weight_repair_manifest(
    path,
    *,
    stage,
    input_glb,
    output_glb,
    status,
):
    parameters = copy.deepcopy(preparation._weight_repair_parameters(stage))
    authority = {
        name: True for name in preparation.generated_review.REPAIR_AUTHORITY_FIELDS
    }
    authority.update(
        {
            "source_animation_fingerprints": {"Walking": "fixture"},
            "post_repair_animation_fingerprints": {"Walking": "fixture"},
            "source_animation_curve_stats": {"Walking": {"curves": 1}},
            "post_repair_animation_curve_stats": {"Walking": {"curves": 1}},
            "rest_geometry_topology_fingerprint_before": "fixture-topology",
            "rest_geometry_topology_fingerprint_after": "fixture-topology",
            "maximum_rest_geometry_delta": 0.0,
            "maximum_allowed_rest_geometry_delta": 0.0,
        }
    )
    ready = status == preparation.generated_review.WEIGHT_REPAIR_READY_STATUS
    payload = {
        "schema": preparation.generated_review.WEIGHT_REPAIR_SCHEMA,
        "status": status,
        "input": _record(input_glb),
        "output": _record(output_glb),
        "front_axis": "positive-x",
        "parameters": parameters,
        "formal_dataset_registration_authorized": False,
        "authority_contract": authority,
        "final_measurements": {
            "maximum_extension_ratio_of_rest_diagonal": (0.019 if ready else 0.03),
            "remaining_seed_edge_count": 0 if ready else 2,
        },
    }
    if parameters["cross_limb_authority"] == "low-slice-components":
        chains = (
            "front_side_negative",
            "front_side_positive",
            "hind_side_negative",
            "hind_side_positive",
        )
        payload["cross_limb_preclean"] = {
            "entries_after": 0,
            "weight_mass_after": 0,
            "final_forbidden_entries": 0,
            "final_forbidden_weight_mass": 0,
            "skipped_for_residual_pass": parameters["skip_cross_limb_preclean"],
            "spatial_authority": {
                "method": "four_largest_disconnected_low_slice_components",
                "immutable_through_motion_repair": True,
                "blender_up_axis": "positive-z",
                "height_fraction": parameters["limb_slice_height_fraction"],
                "substantial_component_sizes_descending": [4, 3, 2, 1],
                "assignment_counts": {name: 1 for name in chains},
                "components": [{"semantic_chain": name} for name in chains],
            },
        }
    _write_json(path, payload)
    return payload


def _build_weight_repair_lineage(tmp_path, branch):
    retargeted = tmp_path / "retargeted.glb"
    _write_glb(retargeted)
    statuses = {
        "primary": {
            "primary": preparation.generated_review.WEIGHT_REPAIR_READY_STATUS,
        },
        "fallback_a": {
            "primary": preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
            "fallback_a": preparation.generated_review.WEIGHT_REPAIR_READY_STATUS,
        },
        "fallback_a_b": {
            "primary": preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
            "fallback_a": preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
            "fallback_b": preparation.generated_review.WEIGHT_REPAIR_READY_STATUS,
        },
        "fallback_a_b_c": {
            "primary": preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
            "fallback_a": preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
            "fallback_b": preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
            "fallback_c": preparation.generated_review.WEIGHT_REPAIR_READY_STATUS,
        },
    }[branch]
    outputs = {}
    authenticated = {}
    attempts = []
    payloads = {}
    stage_outputs = {}
    stage_manifests = {}
    for stage in preparation.generated_review.WEIGHT_REPAIR_BRANCH_STAGES[branch]:
        contract = preparation.generated_review.WEIGHT_REPAIR_STAGE_CONTRACTS[stage]
        output_path = tmp_path / f"{stage}.glb"
        _write_glb(output_path)
        manifest_path = tmp_path / f"{stage}.json"
        input_descriptor = contract["input_glb_output_descriptor"]
        input_path = (
            retargeted
            if input_descriptor == "retargeted_animated_glb"
            else stage_outputs[
                {
                    "weight_repair_primary_glb": "primary",
                    "weight_repair_fallback_a_glb": "fallback_a",
                    "weight_repair_fallback_b_glb": "fallback_b",
                }[input_descriptor]
            ]
        )
        payload = _write_weight_repair_manifest(
            manifest_path,
            stage=stage,
            input_glb=input_path,
            output_glb=output_path,
            status=statuses[stage],
        )
        output_name = contract["output_glb_output_descriptor"]
        manifest_name = contract["manifest_output_descriptor"]
        outputs[output_name] = _record(output_path)
        outputs[manifest_name] = _record(manifest_path)
        authenticated[f"output:{output_name}"] = output_path.resolve()
        authenticated[f"output:{manifest_name}"] = manifest_path.resolve()
        stage_outputs[stage] = output_path
        stage_manifests[stage] = manifest_path
        payloads[stage] = payload
        attempts.append(
            preparation.generated_review.weight_repair_attempt_record(
                stage,
                payload,
                manifest_path,
                output_path,
            )
        )
    final_artifact = preparation.generated_review.weight_repair_final_artifact(branch)
    final_stage = preparation.generated_review.WEIGHT_REPAIR_BRANCH_FINAL_STAGE[branch]
    final_glb = stage_outputs[final_stage]
    final_manifest = stage_manifests[final_stage]
    outputs["animated_glb"] = copy.deepcopy(
        outputs[final_artifact["glb_output_descriptor"]]
    )
    outputs["weight_repair_manifest"] = copy.deepcopy(
        outputs[final_artifact["manifest_output_descriptor"]]
    )
    authenticated["output:animated_glb"] = final_glb.resolve()
    authenticated["output:weight_repair_manifest"] = final_manifest.resolve()
    gates = {
        "heading": "positive-x",
        "rig": "passed",
        "support_plane": "mesh-foot-bottoms",
        "retarget_export_front_axis": "positive-x",
        "gait_initial": "pass",
        "deformation_initial": "rejected",
        "weight_repair_policy": "auto",
        "weight_repair_triggered": True,
        "weight_repair": preparation.generated_review.WEIGHT_REPAIR_READY_STATUS,
        "weight_repair_strategy": (
            preparation.generated_review.WEIGHT_REPAIR_BRANCH_STRATEGY[branch]
        ),
        "weight_repair_branch": branch,
        "weight_repair_attempts": (
            preparation.generated_review.weight_repair_gate_attempts(attempts)
        ),
        "weight_repair_final_artifact": final_artifact,
        "gait_final": "pass",
        "deformation_final": "passed",
        "all_automatic_gates_passed": True,
    }
    return {
        "gates": gates,
        "outputs": outputs,
        "authenticated": authenticated,
        "retargeted": retargeted.resolve(),
        "final_glb": final_glb.resolve(),
        "payloads": payloads,
        "stage_outputs": stage_outputs,
        "stage_manifests": stage_manifests,
    }


def _validate_weight_repair_fixture(fixture):
    legacy, branch = preparation._repair_branch_contract(
        fixture["gates"],
        review_schema=preparation.BRANCHED_GENERATED_REVIEW_SCHEMA,
    )
    preparation._validate_weight_repair_lineage(
        gates=fixture["gates"],
        outputs=fixture["outputs"],
        authenticated=fixture["authenticated"],
        retargeted_glb=fixture["retargeted"],
        final_glb=fixture["final_glb"],
        legacy=legacy,
        branch=branch,
    )


def _v4_not_needed_review(fixture):
    review = contracts.load_json(fixture["review_path"])
    review["schema"] = preparation.BRANCHED_GENERATED_REVIEW_SCHEMA
    review["automatic_admission_gates"].update(
        {
            "weight_repair_strategy": "not_needed",
            "weight_repair_branch": "not_needed",
            "weight_repair_attempts": [],
            "weight_repair_final_artifact": (
                preparation.generated_review.weight_repair_final_artifact("not_needed")
            ),
        }
    )
    review["outputs"]["media_lineage"] = _build_media_lineage(
        fixture["review_path"].parent,
        input_glb=Path(review["outputs"]["animated_glb"]["path"]),
        media=review["outputs"]["media"],
    )
    return review


def test_prepares_fresh_canonical_job_and_does_not_rewrite_old_job(
    approved_generated_animal, tmp_path
):
    old_job = tmp_path / "historical_ue_import_jobs.json"
    old_bytes = b'{"sampled_attributes":{"coat_tone":"red"}}\n'
    old_job.write_bytes(old_bytes)

    manifest_path = _prepare(
        approved_generated_animal, tmp_path / "new_canonical_ue_import"
    )

    manifest = contracts.load_json(manifest_path)
    jobs = contracts.load_json(manifest_path.parent / "ue_import_jobs.json")
    source = approved_generated_animal["source_asset"]
    job = jobs["jobs"][0]
    assert set(jobs) == {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "job_type",
        "job_count",
        "jobs",
        "non_destructive_policy",
        "batch_sha256",
    }
    assert jobs["schema"] == preparation.IMPORT_SCHEMA
    assert jobs["status"] == "ready_for_new_ue_import"
    assert jobs["state_classification"] == "research_candidate"
    assert jobs["formal_dataset_registration_authorized"] is False
    assert jobs["job_type"] == preparation.IMPORT_JOB_TYPE
    assert jobs["job_count"] == 1
    assert jobs["batch_sha256"] == preparation._hash_without(jobs, "batch_sha256")
    assert set(job) == {
        "job_type",
        "asset_id",
        "legacy_tag",
        "tag",
        "profile_schema_id",
        "sampled_attributes",
        "expected_actions",
        "rigged_glb",
        "rigged_glb_sha256",
        "source_registry_sha256",
        "source_asset_sha256",
        "request_sha256",
        "animation_decision_file_sha256",
        "animation_decision_sha256",
    }
    assert job["job_type"] == preparation.IMPORT_JOB_TYPE
    assert job["legacy_tag"] == source["asset_id"]
    assert job["expected_actions"] == ["Idle", "Walking"]
    assert manifest["automatic_checks"]["overall"] == "passed"
    assert manifest["automatic_checks"]["source_registry_and_preflight_reauthenticated"]
    assert manifest["automatic_checks"][
        "source_registry_matched_external_expected_sha256"
    ]
    assert manifest["automatic_checks"][
        "human_animation_approval_matched_external_expected_sha256"
    ]
    assert manifest["schema"] == preparation.SCHEMA
    assert manifest["schema"].endswith("_v3")
    assert (
        manifest["presentation_evidence"]
        == approved_generated_animal["presentation_evidence"]
    )
    for check, expected in preparation.PRESENTATION_AUTOMATIC_CHECKS.items():
        assert manifest["automatic_checks"][check] is expected
    assert manifest["animation_decision_freeze_receipt"] == _record(
        approved_generated_animal["receipt_path"]
    )
    assert manifest[
        "expected_animation_decision_freeze_receipt_file_sha256"
    ] == _sha256(approved_generated_animal["receipt_path"])
    assert manifest["user_instruction_authority"] == (
        preparation.USER_INSTRUCTION_AUTHORITY
    )
    assert (
        manifest["user_instruction_authority"]["cryptographic_user_identity_verified"]
        is False
    )
    assert manifest["automatic_checks"][
        "animation_decision_freeze_receipt_reauthenticated"
    ]
    assert manifest["automatic_checks"][
        "user_instruction_authority_preserved_without_cryptographic_upgrade"
    ]
    assert set(manifest["ue_import_jobs"]) == {"path", "sha256", "size_bytes"}
    assert manifest["ue_import_jobs"]["path"] == "ue_import_jobs.json"
    assert manifest["ue_import_jobs"]["sha256"] == _sha256(
        manifest_path.parent / "ue_import_jobs.json"
    )
    assert job["asset_id"] == source["asset_id"]
    assert job["tag"] == f"pixal_{source['asset_id']}"
    assert job["profile_schema_id"] == source["profile_schema_id"]
    assert job["sampled_attributes"] == source["sampled_attributes"]
    assert job["request_sha256"] == source["request_sha256"]
    assert old_job.read_bytes() == old_bytes


def test_prepares_from_motion_style_baseline_and_current_asset_short_readback(
    approved_generated_animal,
    tmp_path,
):
    _use_compact_approval_evidence(approved_generated_animal)

    manifest_path = _prepare(
        approved_generated_animal,
        tmp_path / "compact_approval_ue_import",
    )
    manifest = contracts.load_json(manifest_path)

    assert manifest["presentation_evidence"] == (
        approved_generated_animal["presentation_evidence"]
    )
    assert manifest["presentation_evidence"]["mode"] == (
        preparation.MOTION_STYLE_AND_CURRENT_READBACK_MODE
    )
    for name, expected in (
        preparation.MOTION_STYLE_AND_CURRENT_READBACK_AUTOMATIC_CHECKS.items()
    ):
        assert manifest["automatic_checks"][name] is expected
    assert not (
        set(preparation.PRESENTATION_AUTOMATIC_CHECKS)
        & set(manifest["automatic_checks"])
    )
    assert manifest["reviewed_animated_glb"] == _record(
        Path(
            contracts.load_json(approved_generated_animal["review_path"])[
                "outputs"
            ]["animated_glb"]["path"]
        )
    )


def test_prepare_rejects_style_video_as_current_review_action_readback(
    approved_generated_animal,
    tmp_path,
):
    _use_compact_approval_evidence(
        approved_generated_animal,
        rebind_current_readback_to_style_video=True,
    )
    output = tmp_path / "rejected_style_video_current_geometry"

    with pytest.raises(
        contracts.ContractError,
        match="current geometry/action binding is invalid",
    ):
        _prepare(approved_generated_animal, output)
    _assert_failed_without_preparation_output(output)


def test_texture_transcode_authenticates_full_graph_and_preserves_non_webp(
    tmp_path,
):
    source, output, manifest_path = _write_texture_transcode_fixture(tmp_path)

    authenticated = preparation._authenticate_texture_transcode(
        reviewed_glb=source,
        ue_compatible_glb_path=output,
        texture_transcode_manifest_path=manifest_path,
    )

    source_document, _ = transcode_tool.read_glb(source)
    output_document, _ = transcode_tool.read_glb(output)
    assert authenticated["ue_compatible_glb"] == output.resolve()
    assert output_document["images"][1] == source_document["images"][1]
    assert output_document["textures"][1] == source_document["textures"][1]


def test_texture_transcode_rejects_resealed_geometry_bytes(tmp_path):
    source, output, manifest_path = _write_texture_transcode_fixture(tmp_path)
    document, binary = transcode_tool.read_glb(output)
    tampered = bytearray(binary)
    tampered[0] ^= 1
    output.write_bytes(transcode_tool.encode_glb(document, bytes(tampered)))
    manifest = contracts.load_json(manifest_path)
    manifest["output"] = _record(output)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        contracts.ContractError,
        match="original GLB byte graph",
    ):
        preparation._authenticate_texture_transcode(
            reviewed_glb=source,
            ue_compatible_glb_path=output,
            texture_transcode_manifest_path=manifest_path,
        )


def test_texture_transcode_rejects_resealed_pbr_routing(tmp_path):
    source, output, manifest_path = _write_texture_transcode_fixture(tmp_path)
    document, binary = transcode_tool.read_glb(output)
    document["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"] = 0.1
    output.write_bytes(transcode_tool.encode_glb(document, binary))
    manifest = contracts.load_json(manifest_path)
    manifest["output"] = _record(output)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        contracts.ContractError,
        match="structure or PBR routing",
    ):
        preparation._authenticate_texture_transcode(
            reviewed_glb=source,
            ue_compatible_glb_path=output,
            texture_transcode_manifest_path=manifest_path,
        )


def test_texture_transcode_rejects_resealed_png_with_different_rgba(tmp_path):
    source, output, manifest_path = _write_texture_transcode_fixture(tmp_path)
    document, binary = transcode_tool.read_glb(output)
    image = document["images"][0]
    view = document["bufferViews"][image["bufferView"]]
    start = view["byteOffset"]
    replacement = _encoded_image("PNG", (220, 20, 30, 255))
    view["byteLength"] = len(replacement)
    output.write_bytes(
        transcode_tool.encode_glb(
            document,
            binary[:start] + replacement,
        )
    )
    with Image.open(io.BytesIO(replacement)) as opened:
        replacement_rgba = opened.convert("RGBA").tobytes()
    manifest = contracts.load_json(manifest_path)
    manifest["output"] = _record(output)
    manifest["images"][0].update(
        {
            "pixel_size": [2, 2],
            "rgba_sha256": hashlib.sha256(replacement_rgba).hexdigest(),
            "png_sha256": hashlib.sha256(replacement).hexdigest(),
            "png_size_bytes": len(replacement),
        }
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(
        contracts.ContractError,
        match="bytes/pixels",
    ):
        preparation._authenticate_texture_transcode(
            reviewed_glb=source,
            ue_compatible_glb_path=output,
            texture_transcode_manifest_path=manifest_path,
        )


def test_transcoded_job_binds_manifest_path_sha_and_size(tmp_path):
    source, output, manifest_path = _write_texture_transcode_fixture(tmp_path)
    authenticated = preparation._authenticate_texture_transcode(
        reviewed_glb=source,
        ue_compatible_glb_path=output,
        texture_transcode_manifest_path=manifest_path,
    )
    job_fields = preparation._texture_transcode_job_fields(authenticated)

    assert job_fields["texture_transcode_manifest"] == str(manifest_path.resolve())
    assert job_fields["texture_transcode_manifest_sha256"] == _sha256(manifest_path)
    assert (
        job_fields["texture_transcode_manifest_size_bytes"]
        == manifest_path.stat().st_size
    )


def test_rejects_source_registry_without_external_expected_file_hash(
    approved_generated_animal, tmp_path
):
    with pytest.raises(contracts.ContractError, match="externally expected SHA-256"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_unanchored_registry",
            expected_registry_sha256="f" * 64,
        )


def test_rejects_vacuous_source_registry_automatic_checks(
    approved_generated_animal, tmp_path
):
    registry = contracts.load_json(approved_generated_animal["registry_path"])
    registry["automatic_checks"] = {"overall": "passed"}
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    _write_json(approved_generated_animal["registry_path"], registry)

    with pytest.raises(contracts.ContractError, match="automatic checks are invalid"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_vacuous_registry_checks",
        )


@pytest.mark.parametrize(
    ("authority_name", "internal_sha_name"),
    (
        ("pixal_batch", "batch_sha256"),
        ("static_decision_batch", "decision_batch_sha256"),
    ),
)
def test_rejects_missing_rebound_source_registry_authority(
    approved_generated_animal,
    tmp_path,
    authority_name,
    internal_sha_name,
):
    registry = contracts.load_json(approved_generated_animal["registry_path"])
    registry[authority_name] = {
        "path": str((tmp_path / f"missing_{authority_name}.json").resolve()),
        "sha256": "a" * 64,
        internal_sha_name: "b" * 64,
    }
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    _write_json(approved_generated_animal["registry_path"], registry)

    with pytest.raises(contracts.ContractError, match="missing or unsafe"):
        _prepare(
            approved_generated_animal,
            tmp_path / f"rejected_missing_{authority_name}",
        )


@pytest.mark.parametrize(
    ("authority_name", "internal_sha_name"),
    (
        ("pixal_batch", "batch_sha256"),
        ("static_decision_batch", "decision_batch_sha256"),
    ),
)
def test_rejects_rebound_source_registry_internal_authority_hash(
    approved_generated_animal,
    tmp_path,
    authority_name,
    internal_sha_name,
):
    registry = contracts.load_json(approved_generated_animal["registry_path"])
    registry[authority_name][internal_sha_name] = "b" * 64
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    _write_json(approved_generated_animal["registry_path"], registry)

    with pytest.raises(contracts.ContractError, match="contract/hash is invalid"):
        _prepare(
            approved_generated_animal,
            tmp_path / f"rejected_internal_{authority_name}",
        )


@pytest.mark.parametrize(
    "branch",
    ("primary", "fallback_a", "fallback_a_b", "fallback_a_b_c"),
)
def test_accepts_exact_authenticated_weight_repair_branch(tmp_path, branch):
    fixture = _build_weight_repair_lineage(tmp_path, branch)

    _validate_weight_repair_fixture(fixture)


def test_target_rig_lineage_binds_source_raw_pixal_and_exact_target(
    tmp_path, monkeypatch
):
    spear_root = tmp_path / "SPEAR"
    workspace = spear_root / "tmp/new_animal_assets/fixture_workspace"
    workspace.mkdir(parents=True)
    raw_pixal = workspace / "pixal_raw.glb"
    raw_pixal.write_bytes(b"raw Pixal geometry")
    target_rig = workspace / "tokenrig_output.glb"
    target_rig.write_bytes(b"TokenRig output")
    descriptor = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=raw_pixal,
        target_rig_glb=target_rig,
    )

    def validate(path, *, expected_manifest_sha256, expected_target_rig_glb):
        assert path == Path(descriptor["closure_manifest"]["path"])
        assert expected_manifest_sha256 == descriptor["closure_manifest"]["sha256"]
        assert expected_target_rig_glb == target_rig
        return copy.deepcopy(descriptor)

    monkeypatch.setattr(preparation, "SPEAR_ROOT", spear_root)
    monkeypatch.setattr(preparation, "validate_tokenrig_closure_manifest", validate)
    authenticated = {}

    observed = preparation._validate_target_rig_lineage(
        descriptor,
        target_rig_glb=target_rig,
        source_asset={"asset_class": "animal"},
        source_artifacts={"artifact:pixal_raw_glb": raw_pixal},
        authenticated=authenticated,
    )

    assert observed == descriptor
    assert set(authenticated) == {
        "target_rig_lineage:closure_manifest",
        "target_rig_lineage:geometry_readback",
    }


def test_target_rig_lineage_rejects_rebound_raw_pixal(tmp_path, monkeypatch):
    spear_root = tmp_path / "SPEAR"
    workspace = spear_root / "tmp/new_animal_assets/fixture_workspace"
    workspace.mkdir(parents=True)
    canonical_raw = workspace / "canonical_raw.glb"
    canonical_raw.write_bytes(b"canonical raw Pixal geometry")
    rebound_raw = workspace / "rebound_raw.glb"
    rebound_raw.write_bytes(b"other raw Pixal geometry")
    target_rig = workspace / "tokenrig_output.glb"
    target_rig.write_bytes(b"TokenRig output")
    descriptor = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=rebound_raw,
        target_rig_glb=target_rig,
    )
    monkeypatch.setattr(preparation, "SPEAR_ROOT", spear_root)
    monkeypatch.setattr(
        preparation,
        "validate_tokenrig_closure_manifest",
        lambda *args, **kwargs: copy.deepcopy(descriptor),
    )

    with pytest.raises(contracts.ContractError, match="raw Pixal source rejected"):
        preparation._validate_target_rig_lineage(
            descriptor,
            target_rig_glb=target_rig,
            source_asset={"asset_class": "animal"},
            source_artifacts={"artifact:pixal_raw_glb": canonical_raw},
            authenticated={},
        )


def test_target_rig_lineage_rejects_plain_watertight_proxy_with_derived_artifact(
    tmp_path,
    monkeypatch,
):
    spear_root = tmp_path / "SPEAR"
    workspace = spear_root / "tmp/new_animal_assets/fixture_workspace"
    workspace.mkdir(parents=True)
    raw_pixal = workspace / "pixal_raw.glb"
    raw_pixal.write_bytes(b"raw Pixal geometry")
    repaired = workspace / "bounded_repaired.glb"
    repaired.write_bytes(b"bounded repaired Pixel3D geometry")
    watertight_proxy = workspace / "watertight_proxy.glb"
    watertight_proxy.write_bytes(b"authenticated watertight Pixel3D proxy")
    target_rig = workspace / "tokenrig_output.glb"
    target_rig.write_bytes(b"TokenRig output")
    descriptor = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=raw_pixal,
        target_rig_glb=target_rig,
        tokenrig_input_glb=watertight_proxy,
    )
    monkeypatch.setattr(preparation, "SPEAR_ROOT", spear_root)
    monkeypatch.setattr(
        preparation,
        "validate_tokenrig_closure_manifest",
        lambda *args, **kwargs: copy.deepcopy(descriptor),
    )

    with pytest.raises(
        contracts.ContractError,
        match="cannot bypass its source-registry repair authority",
    ):
        preparation._validate_target_rig_lineage(
            descriptor,
            target_rig_glb=target_rig,
            source_asset={"asset_class": "animal"},
            source_artifacts={
                "artifact:pixal_raw_glb": raw_pixal,
                "artifact:derived_repaired_glb": repaired,
            },
            authenticated={},
        )


def _bounded_watertight_target_lineage_fixture(tmp_path, monkeypatch):
    spear_root = tmp_path / "SPEAR"
    workspace = spear_root / "tmp/new_animal_assets/fixture_workspace"
    workspace.mkdir(parents=True)
    raw_pixal = workspace / "pixal_raw.glb"
    raw_pixal.write_bytes(b"raw Pixel3D geometry")
    repaired = workspace / "bounded_repaired.glb"
    repaired.write_bytes(b"bounded repaired Pixel3D geometry")
    geometry_closure = workspace / "geometry_closure.json"
    _write_json(geometry_closure, {"fixture": "bounded geometry closure"})
    raw_decision_batch = workspace / "raw_static_decision_batch.json"
    _write_json(raw_decision_batch, {"fixture": "raw decision batch"})
    watertight_proxy_manifest = workspace / "watertight_proxy_manifest.json"
    _write_json(
        watertight_proxy_manifest,
        {"fixture": "authenticated watertight proxy manifest"},
    )
    watertight_proxy_geometry_audit = (
        workspace / "watertight_proxy_geometry_audit.json"
    )
    _write_json(
        watertight_proxy_geometry_audit,
        {"fixture": "authenticated watertight proxy geometry audit"},
    )
    watertight_proxy = workspace / "watertight_proxy.glb"
    watertight_proxy.write_bytes(
        b"authenticated watertight proxy of bounded Pixel3D geometry"
    )
    target_rig = workspace / "tokenrig_output.glb"
    target_rig.write_bytes(b"TokenRig output")
    geometry_closure_manifest_sha256 = "b" * 64
    descriptor = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=raw_pixal,
        target_rig_glb=target_rig,
        tokenrig_input_glb=watertight_proxy,
        upstream_kind="bounded_watertight_runtime_proxy",
        bounded_geometry_closure=geometry_closure,
        bounded_geometry_closure_manifest_sha256=(
            geometry_closure_manifest_sha256
        ),
        watertight_proxy_manifest=watertight_proxy_manifest,
        watertight_proxy_geometry_audit=watertight_proxy_geometry_audit,
        raw_static_decision_batch=raw_decision_batch,
    )
    source_asset = {
        "asset_class": "animal",
        "provenance": {
            "models": {
                (
                    source_registry
                    .DIRECT_GEOMETRY_RAW_DECISION_PROVENANCE_MODEL
                ): "a" * 64,
                (
                    source_registry
                    .DIRECT_GEOMETRY_CLOSURE_PROVENANCE_MODEL
                ): geometry_closure_manifest_sha256,
            }
        },
    }
    source_artifacts = {
        "artifact:pixal_raw_glb": raw_pixal,
        "artifact:derived_repaired_glb": repaired,
        "artifact:derived_geometry_closure": geometry_closure,
        "artifact:raw_static_decision_batch": raw_decision_batch,
    }
    monkeypatch.setattr(preparation, "SPEAR_ROOT", spear_root)
    monkeypatch.setattr(
        preparation,
        "validate_tokenrig_closure_manifest",
        lambda *args, **kwargs: copy.deepcopy(descriptor),
    )
    return {
        "descriptor": descriptor,
        "target_rig": target_rig,
        "raw_pixal": raw_pixal,
        "watertight_proxy": watertight_proxy,
        "geometry_closure": geometry_closure,
        "geometry_closure_manifest_sha256": (
            geometry_closure_manifest_sha256
        ),
        "watertight_proxy_manifest": watertight_proxy_manifest,
        "watertight_proxy_geometry_audit": (
            watertight_proxy_geometry_audit
        ),
        "source_asset": source_asset,
        "source_artifacts": source_artifacts,
    }


def test_target_rig_lineage_accepts_exact_bounded_watertight_composite_kind(
    tmp_path,
    monkeypatch,
):
    fixture = _bounded_watertight_target_lineage_fixture(
        tmp_path,
        monkeypatch,
    )

    observed = preparation._validate_target_rig_lineage(
        fixture["descriptor"],
        target_rig_glb=fixture["target_rig"],
        source_asset=fixture["source_asset"],
        source_artifacts=fixture["source_artifacts"],
        authenticated={},
    )

    assert observed["lineage"]["upstream_kind"] == (
        "bounded_watertight_runtime_proxy"
    )
    assert observed["lineage"]["tokenrig_input"] == _record(
        fixture["watertight_proxy"]
    )
    assert _sha256(fixture["geometry_closure"]) != (
        fixture["geometry_closure_manifest_sha256"]
    )
    assert _sha256(fixture["watertight_proxy_manifest"]) != _sha256(
        fixture["watertight_proxy_geometry_audit"]
    )


@pytest.mark.parametrize(
    "confusion",
    (
        "authority_file_uses_internal_manifest_sha",
        "provenance_internal_manifest_uses_file_sha",
        "proxy_manifest_authority_uses_audit_sha",
        "proxy_audit_authority_uses_manifest_sha",
    ),
)
def test_target_rig_lineage_rejects_bounded_watertight_hash_semantic_confusion(
    tmp_path,
    monkeypatch,
    confusion,
):
    fixture = _bounded_watertight_target_lineage_fixture(
        tmp_path,
        monkeypatch,
    )
    geometry_file_sha256 = _sha256(fixture["geometry_closure"])
    geometry_manifest_sha256 = fixture[
        "geometry_closure_manifest_sha256"
    ]
    if confusion in {
        "authority_file_uses_internal_manifest_sha",
        "proxy_manifest_authority_uses_audit_sha",
        "proxy_audit_authority_uses_manifest_sha",
    }:
        descriptor = fixture["descriptor"]
        closure_path = Path(descriptor["closure_manifest"]["path"])
        closure_payload = contracts.load_json(closure_path)
        composite_authority = closure_payload["lineage"][
            "composite_upstream_authority"
        ]
        if confusion == "authority_file_uses_internal_manifest_sha":
            composite_authority[
                "bounded_geometry_closure_file_sha256"
            ] = geometry_manifest_sha256
        elif confusion == "proxy_manifest_authority_uses_audit_sha":
            composite_authority[
                "watertight_proxy_manifest_file_sha256"
            ] = _sha256(fixture["watertight_proxy_geometry_audit"])
        else:
            composite_authority[
                "watertight_proxy_geometry_audit_file_sha256"
            ] = _sha256(fixture["watertight_proxy_manifest"])
        _write_json(closure_path, closure_payload)
        internal_closure_sha256 = descriptor["closure_manifest"][
            "manifest_sha256"
        ]
        descriptor["closure_manifest"] = {
            **_record(closure_path),
            "manifest_sha256": internal_closure_sha256,
        }
        descriptor["descriptor_sha256"] = preparation._hash_without(
            descriptor,
            "descriptor_sha256",
        )
    else:
        fixture["source_asset"]["provenance"]["models"][
            source_registry.DIRECT_GEOMETRY_CLOSURE_PROVENANCE_MODEL
        ] = geometry_file_sha256

    with pytest.raises(
        contracts.ContractError,
        match="composite authority changed",
    ):
        preparation._validate_target_rig_lineage(
            fixture["descriptor"],
            target_rig_glb=fixture["target_rig"],
            source_asset=fixture["source_asset"],
            source_artifacts=fixture["source_artifacts"],
            authenticated={},
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "schema",
        "normalization",
        "counts",
        "metrics",
        "thresholds",
    ),
)
def test_target_rig_lineage_rejects_watertight_correspondence_tamper(
    tmp_path,
    monkeypatch,
    tamper,
):
    fixture = _bounded_watertight_target_lineage_fixture(
        tmp_path,
        monkeypatch,
    )
    descriptor = fixture["descriptor"]
    closure_path = Path(descriptor["closure_manifest"]["path"])
    closure_payload = contracts.load_json(closure_path)
    correspondence = closure_payload["lineage"][
        "composite_upstream_authority"
    ]["watertight_proxy_correspondence"]
    if tamper == "schema":
        correspondence["schema"] = "wrong_correspondence_v1"
    elif tamper == "normalization":
        correspondence["normalization"] = "unit_cube"
    elif tamper == "counts":
        correspondence["proxy_unique_vertices"] = True
    elif tamper == "metrics":
        correspondence["proxy_to_repaired"]["unexpected"] = 0.0
    else:
        correspondence["thresholds"][
            "proxy_to_repaired_p99_ratio_max"
        ] = 0.004
    _write_json(closure_path, closure_payload)
    internal_closure_sha256 = descriptor["closure_manifest"][
        "manifest_sha256"
    ]
    descriptor["closure_manifest"] = {
        **_record(closure_path),
        "manifest_sha256": internal_closure_sha256,
    }
    descriptor["descriptor_sha256"] = preparation._hash_without(
        descriptor,
        "descriptor_sha256",
    )

    with pytest.raises(contracts.ContractError, match="correspondence"):
        preparation._validate_target_rig_lineage(
            descriptor,
            target_rig_glb=fixture["target_rig"],
            source_asset=fixture["source_asset"],
            source_artifacts=fixture["source_artifacts"],
            authenticated={},
        )


def test_target_rig_lineage_rejects_bounded_watertight_on_old_direct_source(
    tmp_path,
    monkeypatch,
):
    fixture = _bounded_watertight_target_lineage_fixture(
        tmp_path,
        monkeypatch,
    )

    with pytest.raises(
        contracts.ContractError,
        match="lacks its exact v4 direct-geometry registry authority",
    ):
        preparation._validate_target_rig_lineage(
            fixture["descriptor"],
            target_rig_glb=fixture["target_rig"],
            source_asset={"asset_class": "animal"},
            source_artifacts={
                "artifact:pixal_raw_glb": fixture["raw_pixal"],
            },
            authenticated={},
        )


def test_target_rig_lineage_rejects_old_kind_relabelled_as_bounded_watertight(
    tmp_path,
    monkeypatch,
):
    fixture = _bounded_watertight_target_lineage_fixture(
        tmp_path,
        monkeypatch,
    )
    old_closure_descriptor = copy.deepcopy(fixture["descriptor"])
    old_closure_descriptor["lineage"][
        "upstream_kind"
    ] = "watertight_runtime_proxy"
    old_closure_descriptor["descriptor_sha256"] = preparation._hash_without(
        old_closure_descriptor,
        "descriptor_sha256",
    )
    monkeypatch.setattr(
        preparation,
        "validate_tokenrig_closure_manifest",
        lambda *args, **kwargs: copy.deepcopy(old_closure_descriptor),
    )

    with pytest.raises(
        contracts.ContractError,
        match="contradicts its authenticated closure",
    ):
        preparation._validate_target_rig_lineage(
            fixture["descriptor"],
            target_rig_glb=fixture["target_rig"],
            source_asset=fixture["source_asset"],
            source_artifacts=fixture["source_artifacts"],
            authenticated={},
        )


def test_target_rig_lineage_rejects_arbitrary_proxy_not_returned_by_closure(
    tmp_path,
    monkeypatch,
):
    spear_root = tmp_path / "SPEAR"
    workspace = spear_root / "tmp/new_animal_assets/fixture_workspace"
    workspace.mkdir(parents=True)
    raw_pixal = workspace / "pixal_raw.glb"
    raw_pixal.write_bytes(b"raw Pixel3D geometry")
    authenticated_proxy = workspace / "authenticated_proxy.glb"
    authenticated_proxy.write_bytes(b"closure authenticated watertight proxy")
    arbitrary_proxy = workspace / "arbitrary_proxy.glb"
    arbitrary_proxy.write_bytes(b"arbitrary substitute geometry")
    target_rig = workspace / "tokenrig_output.glb"
    target_rig.write_bytes(b"TokenRig output")
    closure_descriptor = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=raw_pixal,
        target_rig_glb=target_rig,
        tokenrig_input_glb=authenticated_proxy,
    )
    review_descriptor = copy.deepcopy(closure_descriptor)
    review_descriptor["lineage"]["tokenrig_input"] = _record(arbitrary_proxy)
    review_descriptor["descriptor_sha256"] = preparation._hash_without(
        review_descriptor,
        "descriptor_sha256",
    )
    monkeypatch.setattr(preparation, "SPEAR_ROOT", spear_root)
    monkeypatch.setattr(
        preparation,
        "validate_tokenrig_closure_manifest",
        lambda *args, **kwargs: copy.deepcopy(closure_descriptor),
    )

    with pytest.raises(
        contracts.ContractError,
        match="contradicts its authenticated closure",
    ):
        preparation._validate_target_rig_lineage(
            review_descriptor,
            target_rig_glb=target_rig,
            source_asset={"asset_class": "animal"},
            source_artifacts={"artifact:pixal_raw_glb": raw_pixal},
            authenticated={},
        )


def test_target_rig_lineage_accepts_exact_bounded_source_registry_repair(
    tmp_path,
    monkeypatch,
):
    spear_root = tmp_path / "SPEAR"
    workspace = spear_root / "tmp/new_animal_assets/fixture_workspace"
    workspace.mkdir(parents=True)
    raw_pixal = workspace / "pixal_raw.glb"
    raw_pixal.write_bytes(b"raw Pixel3D geometry")
    repaired = workspace / "bounded_repaired.glb"
    repaired.write_bytes(b"bounded repaired Pixel3D geometry")
    target_rig = workspace / "tokenrig_output.glb"
    target_rig.write_bytes(b"TokenRig output")
    descriptor = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=raw_pixal,
        target_rig_glb=target_rig,
        tokenrig_input_glb=repaired,
        upstream_kind="bounded_geometry_closure",
    )
    monkeypatch.setattr(preparation, "SPEAR_ROOT", spear_root)
    monkeypatch.setattr(
        preparation,
        "validate_tokenrig_closure_manifest",
        lambda *args, **kwargs: copy.deepcopy(descriptor),
    )

    observed = preparation._validate_target_rig_lineage(
        descriptor,
        target_rig_glb=target_rig,
        source_asset={"asset_class": "animal"},
        source_artifacts={
            "artifact:pixal_raw_glb": raw_pixal,
            "artifact:derived_repaired_glb": repaired,
        },
        authenticated={},
    )

    assert observed["lineage"]["upstream_kind"] == "bounded_geometry_closure"


def test_target_rig_lineage_rejects_bounded_closure_without_registry_repair(
    tmp_path,
    monkeypatch,
):
    spear_root = tmp_path / "SPEAR"
    workspace = spear_root / "tmp/new_animal_assets/fixture_workspace"
    workspace.mkdir(parents=True)
    raw_pixal = workspace / "pixal_raw.glb"
    raw_pixal.write_bytes(b"raw Pixel3D geometry")
    repaired = workspace / "bounded_repaired.glb"
    repaired.write_bytes(b"bounded repaired Pixel3D geometry")
    target_rig = workspace / "tokenrig_output.glb"
    target_rig.write_bytes(b"TokenRig output")
    descriptor = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=raw_pixal,
        target_rig_glb=target_rig,
        tokenrig_input_glb=repaired,
        upstream_kind="bounded_geometry_closure",
    )
    monkeypatch.setattr(preparation, "SPEAR_ROOT", spear_root)
    monkeypatch.setattr(
        preparation,
        "validate_tokenrig_closure_manifest",
        lambda *args, **kwargs: copy.deepcopy(descriptor),
    )

    with pytest.raises(
        contracts.ContractError,
        match="lacks its source-registry repair authority",
    ):
        preparation._validate_target_rig_lineage(
            descriptor,
            target_rig_glb=target_rig,
            source_asset={"asset_class": "animal"},
            source_artifacts={"artifact:pixal_raw_glb": raw_pixal},
            authenticated={},
        )


@pytest.mark.parametrize(
    "branch",
    ("primary", "fallback_a", "fallback_a_b", "fallback_a_b_c"),
)
def test_v4_review_accepts_only_exact_branch_pipeline_and_outputs(
    approved_generated_animal,
    tmp_path,
    branch,
):
    repair_root = tmp_path / f"repair_{branch}"
    repair_root.mkdir()
    repair = _build_weight_repair_lineage(repair_root, branch)
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["schema"] = preparation.BRANCHED_GENERATED_REVIEW_SCHEMA
    review["automatic_admission_gates"] = repair["gates"]
    review["pipeline_order"] = preparation._review_pipeline_order(repair_branch=branch)
    review["timings_seconds"] = {name: 0.1 for name in review["pipeline_order"]}
    review["outputs"].update(repair["outputs"])
    review["outputs"]["media_lineage"] = _build_media_lineage(
        approved_generated_animal["review_path"].parent,
        input_glb=repair["final_glb"],
        media=review["outputs"]["media"],
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    manifest_path = _prepare(
        approved_generated_animal,
        tmp_path / f"accepted_v4_{branch}",
    )

    manifest = contracts.load_json(manifest_path)
    assert manifest["animation_review_schema"] == (
        preparation.BRANCHED_GENERATED_REVIEW_SCHEMA
    )
    assert manifest["reviewed_animated_glb"]["sha256"] == _sha256(repair["final_glb"])


def test_primary_branch_rejects_compatibility_manifest_not_equal_to_final(
    tmp_path,
):
    fixture = _build_weight_repair_lineage(tmp_path, "primary")
    unrelated = tmp_path / "unrelated_manifest.json"
    unrelated.write_text('{"unrelated":true}\n', encoding="utf-8")
    fixture["outputs"]["weight_repair_manifest"] = _record(unrelated)
    fixture["authenticated"]["output:weight_repair_manifest"] = unrelated.resolve()

    with pytest.raises(contracts.ContractError, match="final weight-repair lineage"):
        _validate_weight_repair_fixture(fixture)


def test_fallback_a_branch_rejects_a_input_other_than_retarget(tmp_path):
    fixture = _build_weight_repair_lineage(tmp_path, "fallback_a")
    payload = fixture["payloads"]["fallback_a"]
    payload["input"] = _record(fixture["stage_outputs"]["primary"])
    _write_json(fixture["stage_manifests"]["fallback_a"], payload)

    with pytest.raises(
        contracts.ContractError, match="fallback_a weight repair manifest rejected"
    ):
        _validate_weight_repair_fixture(fixture)


def test_fallback_a_b_branch_rejects_b_input_other_than_a_output(tmp_path):
    fixture = _build_weight_repair_lineage(tmp_path, "fallback_a_b")
    payload = fixture["payloads"]["fallback_b"]
    payload["input"] = _record(fixture["retargeted"])
    _write_json(fixture["stage_manifests"]["fallback_b"], payload)

    with pytest.raises(
        contracts.ContractError, match="fallback_b weight repair manifest rejected"
    ):
        _validate_weight_repair_fixture(fixture)


def test_fallback_a_b_c_requires_exact_residual_parameters_and_b_to_c_input(
    tmp_path,
):
    assert preparation._weight_repair_parameters("fallback_c") == (
        preparation.generated_review.WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS
    )
    assert preparation._weight_repair_parameters("fallback_c") == (
        preparation._weight_repair_parameters("fallback_b")
    )
    with pytest.raises(
        contracts.ContractError, match="stage is unsupported"
    ):
        preparation._weight_repair_parameters("fallback_d")

    wrong_input_root = tmp_path / "wrong_input"
    wrong_input_root.mkdir()
    wrong_input = _build_weight_repair_lineage(
        wrong_input_root, "fallback_a_b_c"
    )
    payload = wrong_input["payloads"]["fallback_c"]
    payload["input"] = _record(wrong_input["stage_outputs"]["fallback_a"])
    _write_json(wrong_input["stage_manifests"]["fallback_c"], payload)
    with pytest.raises(
        contracts.ContractError,
        match="fallback_c weight repair manifest rejected",
    ):
        _validate_weight_repair_fixture(wrong_input)

    wrong_parameters_root = tmp_path / "wrong_parameters"
    wrong_parameters_root.mkdir()
    wrong_parameters = _build_weight_repair_lineage(
        wrong_parameters_root, "fallback_a_b_c"
    )
    payload = wrong_parameters["payloads"]["fallback_c"]
    payload["parameters"]["maximum_passes"] += 1
    _write_json(wrong_parameters["stage_manifests"]["fallback_c"], payload)
    with pytest.raises(
        contracts.ContractError,
        match="fallback_c weight repair manifest rejected",
    ):
        _validate_weight_repair_fixture(wrong_parameters)


def test_fallback_a_b_c_forbids_c_after_ready_b_and_rejects_incomplete_c(
    tmp_path,
):
    unnecessary_c_root = tmp_path / "unnecessary_c"
    unnecessary_c_root.mkdir()
    unnecessary_c = _build_weight_repair_lineage(
        unnecessary_c_root, "fallback_a_b_c"
    )
    unnecessary_c["gates"]["weight_repair_attempts"][2]["status"] = (
        preparation.generated_review.WEIGHT_REPAIR_READY_STATUS
    )
    with pytest.raises(
        contracts.ContractError,
        match="weight-repair branch consistency rejected.*contradictory",
    ):
        _validate_weight_repair_fixture(unnecessary_c)

    incomplete_c_root = tmp_path / "incomplete_c"
    incomplete_c_root.mkdir()
    incomplete_c = _build_weight_repair_lineage(
        incomplete_c_root, "fallback_a_b_c"
    )
    c_manifest = incomplete_c["stage_manifests"]["fallback_c"]
    _write_weight_repair_manifest(
        c_manifest,
        stage="fallback_c",
        input_glb=incomplete_c["stage_outputs"]["fallback_b"],
        output_glb=incomplete_c["stage_outputs"]["fallback_c"],
        status=preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
    )
    with pytest.raises(
        contracts.ContractError,
        match="fallback_c weight repair manifest rejected",
    ):
        _validate_weight_repair_fixture(incomplete_c)


def test_fallback_a_b_c_rejects_rebound_final_artifact_or_descriptors(
    tmp_path,
):
    wrong_final_root = tmp_path / "wrong_final"
    wrong_final_root.mkdir()
    wrong_final = _build_weight_repair_lineage(
        wrong_final_root, "fallback_a_b_c"
    )
    wrong_final["gates"]["weight_repair_final_artifact"] = (
        preparation.generated_review.weight_repair_final_artifact(
            "fallback_a_b"
        )
    )
    with pytest.raises(
        contracts.ContractError,
        match="branch authority contradicts itself",
    ):
        _validate_weight_repair_fixture(wrong_final)

    rebound_output_root = tmp_path / "rebound_output"
    rebound_output_root.mkdir()
    rebound_output = _build_weight_repair_lineage(
        rebound_output_root, "fallback_a_b_c"
    )
    rebound_output["outputs"]["animated_glb"] = copy.deepcopy(
        rebound_output["outputs"]["weight_repair_fallback_b_glb"]
    )
    rebound_output["authenticated"]["output:animated_glb"] = (
        rebound_output["stage_outputs"]["fallback_b"].resolve()
    )
    rebound_output["final_glb"] = rebound_output["stage_outputs"][
        "fallback_b"
    ].resolve()
    with pytest.raises(
        contracts.ContractError,
        match="final weight-repair lineage is inconsistent",
    ):
        _validate_weight_repair_fixture(rebound_output)


def test_v4_fallback_a_b_c_rejects_fallback_b_pipeline_order(
    approved_generated_animal,
    tmp_path,
):
    repair_root = tmp_path / "repair_fallback_a_b_c_wrong_pipeline"
    repair_root.mkdir()
    repair = _build_weight_repair_lineage(
        repair_root, "fallback_a_b_c"
    )
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["schema"] = preparation.BRANCHED_GENERATED_REVIEW_SCHEMA
    review["automatic_admission_gates"] = repair["gates"]
    review["pipeline_order"] = preparation._review_pipeline_order(
        repair_branch="fallback_a_b"
    )
    review["timings_seconds"] = {
        name: 0.1 for name in review["pipeline_order"]
    }
    review["outputs"].update(repair["outputs"])
    review["outputs"]["media_lineage"] = _build_media_lineage(
        approved_generated_animal["review_path"].parent,
        input_glb=repair["final_glb"],
        media=review["outputs"]["media"],
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(
        contracts.ContractError,
        match="pipeline/timings are invalid",
    ):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_fallback_a_b_c_wrong_pipeline",
        )


def test_rejects_v3_review_with_v4_repair_fields(approved_generated_animal, tmp_path):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["schema"] = preparation.LEGACY_GENERATED_REVIEW_SCHEMA
    review["automatic_admission_gates"].update(
        {
            "weight_repair_strategy": "not_needed",
            "weight_repair_branch": "not_needed",
            "weight_repair_attempts": [],
            "weight_repair_final_artifact": (
                preparation.generated_review.weight_repair_final_artifact("not_needed")
            ),
        }
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="legacy v3.*fields"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_v3_v4_field_union",
        )


def test_rejects_v4_review_with_legacy_repair_fields(
    approved_generated_animal, tmp_path
):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["schema"] = preparation.BRANCHED_GENERATED_REVIEW_SCHEMA
    for name in (
        "weight_repair_strategy",
        "weight_repair_branch",
        "weight_repair_attempts",
        "weight_repair_final_artifact",
    ):
        review["automatic_admission_gates"].pop(name)
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="branched v4.*fields"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_v4_legacy_fields",
        )


def test_legacy_v3_review_is_audit_only_and_cannot_publish_v2_batch(
    approved_generated_animal, tmp_path
):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["schema"] = preparation.LEGACY_GENERATED_REVIEW_SCHEMA
    for name in (
        "weight_repair_strategy",
        "weight_repair_branch",
        "weight_repair_attempts",
        "weight_repair_final_artifact",
    ):
        review["automatic_admission_gates"].pop(name)
    review["outputs"].pop("media_lineage")
    review["inputs"].pop("target_rig_lineage")
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="legacy v3 is audit-only"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_v3_prepare",
        )


def test_v4_not_needed_branch_rejects_unrun_repair_output(
    approved_generated_animal, tmp_path
):
    review = _v4_not_needed_review(approved_generated_animal)
    review["outputs"]["weight_repair_primary_glb"] = copy.deepcopy(
        review["outputs"]["retargeted_animated_glb"]
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="review I/O is invalid"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_unrun_repair_output",
        )


def test_v4_review_rejects_missing_media_lineage(approved_generated_animal, tmp_path):
    review = _v4_not_needed_review(approved_generated_animal)
    review["outputs"].pop("media_lineage")
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="review I/O is invalid"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_missing_media_lineage",
        )


def test_v4_review_rejects_tampered_render_frame_artifact(
    approved_generated_animal, tmp_path
):
    review = _v4_not_needed_review(approved_generated_animal)
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)
    frame = (
        approved_generated_animal["review_path"].parent
        / "05_review/walking_side_frames/frame_0003.png"
    )
    frame.write_bytes(b"tampered render frame")

    with pytest.raises(
        contracts.ContractError,
        match="review render/encode media lineage rejected",
    ):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_tampered_render_frame",
        )


def test_v4_review_rejects_swapped_encode_media_identity(
    approved_generated_animal, tmp_path
):
    review = _v4_not_needed_review(approved_generated_animal)
    encode_manifest = (
        approved_generated_animal["review_path"].parent
        / "05_review/walking_side_encode_manifest.json"
    )
    encode = contracts.load_json(encode_manifest)
    encode["media_identity"]["action"] = "Idle"
    _write_json(encode_manifest, encode)
    review["outputs"]["media_lineage"]["walking_side"]["encode_manifest"] = _record(
        encode_manifest
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(
        contracts.ContractError,
        match="review render/encode media lineage rejected",
    ):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_swapped_encode_identity",
        )


def test_rejects_empty_required_gate_descriptor(approved_generated_animal, tmp_path):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["outputs"]["support_plane_manifest"] = {}
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="descriptor"):
        _prepare(approved_generated_animal, tmp_path / "rejected_empty_gate")


def test_rejects_tampered_animation_review_media(approved_generated_animal, tmp_path):
    approved_generated_animal["media_path"].write_bytes(b"tampered")

    with pytest.raises(contracts.ContractError, match="review media .* changed"):
        _prepare(approved_generated_animal, tmp_path / "rejected_tamper")


def test_rejects_external_nonmedia_masquerading_as_h264(
    approved_generated_animal, tmp_path
):
    review = contracts.load_json(approved_generated_animal["review_path"])
    hosts = Path("/etc/hosts")
    fake = {
        **_record(hosts),
        "codec": "h264",
        "width": 512,
        "height": 384,
        "frame_count": preparation.REQUIRED_REVIEW_FRAMES,
        "frame_rate": "8/1",
        "duration_seconds": 1,
    }
    review["outputs"]["media"]["walking_side"] = fake
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="escaped its artifact root"):
        _prepare(approved_generated_animal, tmp_path / "rejected_fake_media")


def test_rejects_minimal_fake_glb_without_real_skin_accessors(
    approved_generated_animal, tmp_path
):
    fake = tmp_path / "fake_minimal.glb"
    _write_fake_unskinned_glb(fake)
    review = contracts.load_json(approved_generated_animal["review_path"])
    fake_record = _record(fake)
    review["outputs"]["animated_glb"] = fake_record
    review["outputs"]["retargeted_animated_glb"] = copy.deepcopy(fake_record)
    review["outputs"]["media_lineage"] = _build_media_lineage(
        approved_generated_animal["review_path"].parent,
        input_glb=fake,
        media=review["outputs"]["media"],
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(
        contracts.ContractError, match="complete embedded skinned-mesh payload"
    ):
        _prepare(approved_generated_animal, tmp_path / "rejected_fake_glb")


def test_rejects_self_claimed_source_request_and_profile_hashes(
    approved_generated_animal, tmp_path
):
    source = contracts.load_json(approved_generated_animal["source_path"])
    source["request_sha256"] = "a" * 64
    source["profile_sha256"] = "b" * 64
    source["provenance"]["request_sha256"] = "a" * 64
    _write_json(approved_generated_animal["source_path"], source)
    _reauthenticate_registry_source(approved_generated_animal)

    with pytest.raises(contracts.ContractError, match="does not match its request"):
        _prepare(approved_generated_animal, tmp_path / "rejected_self_claim")


def test_rejects_nonfinite_anywhere_in_review(approved_generated_animal, tmp_path):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["timings_seconds"]["heading"] = float("nan")
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="non-finite"):
        _prepare(approved_generated_animal, tmp_path / "rejected_nan")


def test_rejects_decision_without_external_expected_file_hash(
    approved_generated_animal, tmp_path
):
    with pytest.raises(contracts.ContractError, match="external expected SHA-256"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_unanchored_decision",
            expected_decision_sha256="f" * 64,
        )


def test_rejects_decision_bound_only_to_review_self_claimed_hash(
    approved_generated_animal, tmp_path
):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["review_sha256"] = "f" * 64
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)
    decision = contracts.load_json(approved_generated_animal["decision_path"])
    decision["review_sha256"] = "f" * 64
    decision["decision_sha256"] = preparation._hash_without(decision, "decision_sha256")
    _write_json(approved_generated_animal["decision_path"], decision)

    with pytest.raises(contracts.ContractError, match="generated animation review"):
        _prepare(approved_generated_animal, tmp_path / "rejected_self_hash")


def test_rejects_existing_output(approved_generated_animal, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(contracts.ContractError, match="refusing to replace output"):
        _prepare(approved_generated_animal, output)


def test_rejects_freeze_receipt_without_external_expected_file_hash(
    approved_generated_animal,
    tmp_path,
):
    with pytest.raises(contracts.ContractError, match="external expected SHA-256"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_unanchored_receipt",
            expected_receipt_sha256="f" * 64,
        )


def test_cannot_upgrade_caller_assertion_to_cryptographic_identity(
    approved_generated_animal,
    tmp_path,
):
    receipt = contracts.load_json(approved_generated_animal["receipt_path"])
    receipt["user_instruction_authority"]["cryptographic_user_identity_verified"] = True
    receipt["receipt_sha256"] = preparation._hash_without(
        receipt,
        "receipt_sha256",
    )
    _write_json(approved_generated_animal["receipt_path"], receipt)

    with pytest.raises(contracts.ContractError, match="authority contract"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_crypto_upgrade",
        )


def test_rejects_tampered_owner_review_video_bytes_without_output(
    approved_generated_animal,
    tmp_path,
):
    output = tmp_path / "rejected_tampered_presentation_video"
    _set_presentation_writable(approved_generated_animal)
    video = Path(
        approved_generated_animal["presentation_evidence"]["output_video"]["path"]
    )
    video.write_bytes(video.read_bytes() + b"-tampered")
    _reseal_presentation(approved_generated_animal)

    with pytest.raises(contracts.ContractError, match="output video"):
        _prepare(approved_generated_animal, output)

    _assert_failed_without_preparation_output(output)


def test_rejects_tampered_presentation_receipt_raw_file_without_output(
    approved_generated_animal,
    tmp_path,
):
    output = tmp_path / "rejected_tampered_presentation_receipt"
    _set_presentation_writable(approved_generated_animal)
    receipt = Path(
        approved_generated_animal["presentation_evidence"]["presentation_receipt"][
            "path"
        ]
    )
    receipt.write_bytes(receipt.read_bytes() + b" ")
    _reseal_presentation(approved_generated_animal)

    with pytest.raises(contracts.ContractError, match="external SHA-256"):
        _prepare(approved_generated_animal, output)

    _assert_failed_without_preparation_output(output)


def test_rejects_rebound_presentation_receipt_with_invalid_internal_self_hash(
    approved_generated_animal,
    tmp_path,
):
    output = tmp_path / "rejected_presentation_internal_hash"
    _set_presentation_writable(approved_generated_animal)
    receipt_path = Path(
        approved_generated_animal["presentation_evidence"]["presentation_receipt"][
            "path"
        ]
    )
    receipt = contracts.load_json(receipt_path)
    receipt["created_at"] = "2026-07-28T00:00:01+00:00"
    _write_json(receipt_path, receipt)
    _rebind_presentation_receipt_file(approved_generated_animal)
    _reseal_presentation(approved_generated_animal)
    _rewrite_freeze_receipt(approved_generated_animal)

    with pytest.raises(contracts.ContractError, match="canonical self-hash"):
        _prepare(approved_generated_animal, output)

    _assert_failed_without_preparation_output(output)


def test_rejects_presentation_receipt_bound_to_prior_v4_review(
    approved_generated_animal,
    tmp_path,
):
    output = tmp_path / "rejected_stale_presentation_review_binding"
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["created_at"] = "2026-07-28T00:00:02+00:00"
    _write_json(approved_generated_animal["review_path"], review)
    decision = contracts.load_json(approved_generated_animal["decision_path"])
    decision["review"] = _record(approved_generated_animal["review_path"])
    decision["review_sha256"] = decision["review"]["sha256"]
    decision["decision_sha256"] = preparation._hash_without(
        decision,
        "decision_sha256",
    )
    _write_json(approved_generated_animal["decision_path"], decision)
    _rewrite_freeze_receipt(approved_generated_animal, review=review)

    with pytest.raises(contracts.ContractError, match="source-review authority"):
        _prepare(approved_generated_animal, output)

    _assert_failed_without_preparation_output(output)


def test_rejects_freeze_presentation_output_path_mismatch_without_output(
    approved_generated_animal,
    tmp_path,
):
    output = tmp_path / "rejected_presentation_path_mismatch"
    evidence = approved_generated_animal["presentation_evidence"]
    evidence["output_video"]["path"] = f"{evidence['output_video']['path']}.substituted"
    _rewrite_freeze_receipt(approved_generated_animal)

    with pytest.raises(contracts.ContractError, match="authority binding changed"):
        _prepare(approved_generated_animal, output)

    _assert_failed_without_preparation_output(output)


def test_rejects_symlinked_presentation_video_without_output(
    approved_generated_animal,
    tmp_path,
):
    output = tmp_path / "rejected_symlinked_presentation_video"
    _set_presentation_writable(approved_generated_animal)
    video = Path(
        approved_generated_animal["presentation_evidence"]["output_video"]["path"]
    )
    moved_video = tmp_path / "moved_owner_review_six_view.mp4"
    video.rename(moved_video)
    video.symlink_to(moved_video)
    _reseal_presentation(approved_generated_animal)

    with pytest.raises(contracts.ContractError, match="symlink"):
        _prepare(approved_generated_animal, output)

    _assert_failed_without_preparation_output(output)


def test_rejects_duplicate_json_key_in_rebound_presentation_receipt(
    approved_generated_animal,
    tmp_path,
):
    output = tmp_path / "rejected_duplicate_presentation_receipt"
    _set_presentation_writable(approved_generated_animal)
    receipt = Path(
        approved_generated_animal["presentation_evidence"]["presentation_receipt"][
            "path"
        ]
    )
    _inject_duplicate_schema(receipt)
    _rebind_presentation_receipt_file(approved_generated_animal)
    _reseal_presentation(approved_generated_animal)
    _rewrite_freeze_receipt(approved_generated_animal)

    with pytest.raises(contracts.ContractError, match="strict JSON"):
        _prepare(approved_generated_animal, output)

    _assert_failed_without_preparation_output(output)


def test_rejects_legacy_v1_freeze_receipt_without_presentation_evidence(
    approved_generated_animal,
    tmp_path,
):
    output = tmp_path / "rejected_legacy_v1_freeze"
    receipt = contracts.load_json(approved_generated_animal["receipt_path"])
    receipt["schema"] = next(iter(preparation.LEGACY_DECISION_FREEZE_RECEIPT_SCHEMAS))
    receipt.pop("presentation_evidence")
    receipt["user_instruction_binding"].pop("presentation_receipt_file_sha256")
    receipt["receipt_sha256"] = preparation._hash_without(
        receipt,
        "receipt_sha256",
    )
    _write_json(approved_generated_animal["receipt_path"], receipt)

    with pytest.raises(contracts.ContractError, match="audit-only"):
        _prepare(approved_generated_animal, output)

    _assert_failed_without_preparation_output(output)


def test_rejects_v2_freeze_receipt_missing_presentation_evidence(
    approved_generated_animal,
    tmp_path,
):
    output = tmp_path / "rejected_missing_presentation_evidence"
    receipt = contracts.load_json(approved_generated_animal["receipt_path"])
    receipt.pop("presentation_evidence")
    receipt["receipt_sha256"] = preparation._hash_without(
        receipt,
        "receipt_sha256",
    )
    _write_json(approved_generated_animal["receipt_path"], receipt)

    with pytest.raises(contracts.ContractError, match="authority contract"):
        _prepare(approved_generated_animal, output)

    _assert_failed_without_preparation_output(output)


def test_rejects_restore_race_on_presentation_video_and_removes_staging(
    approved_generated_animal,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "rejected_presentation_restore_race"
    video = Path(
        approved_generated_animal["presentation_evidence"]["output_video"]["path"]
    )
    original_bytes = video.read_bytes()
    original_stat = video.stat()
    original_write = preparation._write_json_at
    raced = False

    def racing_write(directory_fd, name, payload):
        nonlocal raced
        result = original_write(directory_fd, name, payload)
        if name == "ue_import_jobs.json" and not raced:
            raced = True
            video.chmod(0o644)
            video.write_bytes(original_bytes + b"-transient-race")
            video.write_bytes(original_bytes)
            os.utime(
                video,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            video.chmod(0o444)
        return result

    monkeypatch.setattr(preparation, "_write_json_at", racing_write)

    with pytest.raises(contracts.ContractError, match="authority graph changed"):
        _prepare(approved_generated_animal, output)

    assert raced
    _assert_failed_without_preparation_output(output)


def test_atomic_publication_race_preserves_concurrent_output_and_removes_staging(
    approved_generated_animal,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "concurrent_preparation"
    marker = output / "other_writer"
    original_publish = preparation._atomic_publish_no_replace_at

    def concurrent_publish(parent_fd, staging_name, output_name, **kwargs):
        os.mkdir(output_name, dir_fd=parent_fd)
        marker_fd = os.open(
            f"{output_name}/other_writer",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            os.write(marker_fd, b"concurrent")
        finally:
            os.close(marker_fd)
        original_publish(parent_fd, staging_name, output_name, **kwargs)

    monkeypatch.setattr(
        preparation,
        "_atomic_publish_no_replace_at",
        concurrent_publish,
    )

    with pytest.raises(contracts.ContractError, match="refusing to replace"):
        _prepare(approved_generated_animal, output)

    assert marker.read_text(encoding="utf-8") == "concurrent"
    assert not list(output.parent.glob(f".{output.name}.*.staging"))


def test_dirfd_publication_preserves_the_authenticated_staging_inode(
    approved_generated_animal,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "dirfd_published_preparation"
    observed = {}
    original_publish = preparation._atomic_publish_no_replace_at

    def recording_publish(parent_fd, staging_name, output_name, **kwargs):
        before = os.stat(
            staging_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        original_publish(parent_fd, staging_name, output_name, **kwargs)
        after = os.stat(
            output_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        observed["before"] = (before.st_dev, before.st_ino)
        observed["after"] = (after.st_dev, after.st_ino)

    monkeypatch.setattr(
        preparation,
        "_atomic_publish_no_replace_at",
        recording_publish,
    )

    result = _prepare(approved_generated_animal, output)

    assert result == output / "ue_import_preparation_manifest.json"
    assert observed["before"] == observed["after"]
    assert not list(output.parent.glob(f".{output.name}.*.staging"))


def test_staging_byte_swap_inside_publication_hook_fails_before_ready(
    approved_generated_animal,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "staging_byte_swap_preparation"
    original_publish = preparation._atomic_publish_no_replace_at
    tampered = False

    def tampering_publish(parent_fd, staging_name, output_name, **kwargs):
        nonlocal tampered
        staging_fd = kwargs["staging_fd"]
        artifact_fd = os.open(
            "ue_import_jobs.json",
            os.O_RDONLY,
            dir_fd=staging_fd,
        )
        try:
            os.fchmod(artifact_fd, 0o644)
        finally:
            os.close(artifact_fd)
        artifact_fd = os.open(
            "ue_import_jobs.json",
            os.O_WRONLY | os.O_TRUNC,
            dir_fd=staging_fd,
        )
        try:
            os.write(artifact_fd, b"tampered")
            os.fsync(artifact_fd)
        finally:
            os.close(artifact_fd)
        tampered = True
        original_publish(
            parent_fd,
            staging_name,
            output_name,
            **kwargs,
        )

    monkeypatch.setattr(
        preparation,
        "_atomic_publish_no_replace_at",
        tampering_publish,
    )

    with pytest.raises(contracts.ContractError, match="staging artifact"):
        _prepare(approved_generated_animal, output)

    assert tampered
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.staging"))


def test_parent_swap_after_precommit_check_cannot_publish_a_ready_output(
    approved_generated_animal,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "parent_swap_preparation"
    moved_parent = tmp_path.with_name(f"{tmp_path.name}_moved")
    original_publish = preparation._atomic_publish_no_replace_at
    swapped = False

    def parent_swapping_publish(parent_fd, staging_name, output_name, **kwargs):
        nonlocal swapped
        tmp_path.rename(moved_parent)
        tmp_path.symlink_to(moved_parent, target_is_directory=True)
        swapped = True
        original_publish(parent_fd, staging_name, output_name, **kwargs)

    monkeypatch.setattr(
        preparation,
        "_atomic_publish_no_replace_at",
        parent_swapping_publish,
    )
    try:
        with pytest.raises(contracts.ContractError, match="held directory"):
            _prepare(approved_generated_animal, output)

        assert swapped
        assert not (moved_parent / output.name).exists()
    finally:
        if tmp_path.is_symlink():
            tmp_path.unlink()
        if moved_parent.exists():
            moved_parent.rename(tmp_path)


def test_fd_cleanup_root_swap_never_touches_an_external_tree(
    tmp_path,
    monkeypatch,
):
    parent_fd = preparation._open_directory_no_follow(tmp_path, "test parent")
    staging_name = ".owned.staging"
    staging = tmp_path / staging_name
    staging.mkdir()
    (staging / "ue_import_jobs.json").write_text("owned", encoding="utf-8")
    identity_stat = staging.stat()
    identity = (identity_stat.st_dev, identity_stat.st_ino)
    moved = tmp_path / ".owned.moved"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    external.chmod(0o755)
    original_listdir = os.listdir
    swapped = False

    def swapping_listdir(path):
        nonlocal swapped
        if isinstance(path, int) and not swapped:
            staging.rename(moved)
            staging.symlink_to(external, target_is_directory=True)
            swapped = True
        return original_listdir(path)

    monkeypatch.setattr(os, "listdir", swapping_listdir)
    try:
        removed = preparation._remove_owned_staging_at(
            parent_fd,
            staging_name,
            identity,
            frozenset(
                {
                    "ue_import_jobs.json",
                    "ue_import_preparation_manifest.json",
                }
            ),
        )
    finally:
        os.close(parent_fd)

    assert swapped
    assert removed is False
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert stat.S_IMODE(external.stat().st_mode) == 0o755

    staging.unlink()
    moved.rmdir()


@pytest.mark.parametrize(
    "artifact",
    (
        "registry",
        "source_asset",
        "pixal_batch",
        "static_decision_batch",
        "preflight",
        "review",
        "decision",
        "receipt",
    ),
)
def test_rejects_duplicate_keys_across_security_critical_json_entrypoints(
    approved_generated_animal,
    tmp_path,
    artifact,
):
    fixture = approved_generated_animal
    registry = contracts.load_json(fixture["registry_path"])
    paths = {
        "registry": fixture["registry_path"],
        "source_asset": fixture["source_path"],
        "pixal_batch": Path(registry["pixal_batch"]["path"]),
        "static_decision_batch": Path(registry["static_decision_batch"]["path"]),
        "preflight": Path(registry["preflight"]["path"]),
        "review": fixture["review_path"],
        "decision": fixture["decision_path"],
        "receipt": fixture["receipt_path"],
    }
    target = paths[artifact]
    _inject_duplicate_schema(target)
    if artifact == "source_asset":
        registry["source_assets"][0]["source_asset"].update(
            {
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
            }
        )
    elif artifact in {"pixal_batch", "static_decision_batch", "preflight"}:
        registry_field = {
            "pixal_batch": "pixal_batch",
            "static_decision_batch": "static_decision_batch",
            "preflight": "preflight",
        }[artifact]
        registry[registry_field]["sha256"] = _sha256(target)
    if artifact in {
        "source_asset",
        "pixal_batch",
        "static_decision_batch",
        "preflight",
    }:
        registry["registry_sha256"] = source_registry._hash_without(
            registry,
            "registry_sha256",
        )
        _write_json(fixture["registry_path"], registry)
        _rewrite_freeze_receipt(fixture)

    with pytest.raises(contracts.ContractError, match="duplicate JSON object key"):
        _prepare(
            fixture,
            tmp_path / f"rejected_duplicate_{artifact}",
        )


def test_direct_files_reject_arbitrary_symlink_parent_and_allow_only_exact_tmp_bridge(
    tmp_path,
    monkeypatch,
):
    real_root = tmp_path / "real"
    real_root.mkdir()
    evidence = real_root / "evidence.json"
    evidence.write_text('{"status":"passed"}\n', encoding="utf-8")
    arbitrary = tmp_path / "arbitrary"
    arbitrary.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(contracts.ContractError, match="symlink path component"):
        preparation._direct_file(arbitrary / evidence.name, "arbitrary alias")

    trusted_bridge = tmp_path / "tmp"
    trusted_bridge.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(preparation, "SPEAR_TMP_BRIDGE", trusted_bridge.absolute())
    assert (
        preparation._direct_file(
            trusted_bridge / evidence.name,
            "trusted tmp evidence",
        )
        == evidence.resolve()
    )

    nested_real = real_root / "nested_real"
    nested_real.mkdir()
    nested_evidence = nested_real / "nested.json"
    nested_evidence.write_text('{"status":"passed"}\n', encoding="utf-8")
    nested_alias = real_root / "nested_alias"
    nested_alias.symlink_to(nested_real, target_is_directory=True)
    with pytest.raises(contracts.ContractError, match="symlink path component"):
        preparation._direct_file(
            trusted_bridge / "nested_alias" / nested_evidence.name,
            "nested alias",
        )


def test_output_and_artifact_roots_reject_arbitrary_symlink_parents(
    tmp_path,
):
    real_root = tmp_path / "real"
    real_root.mkdir()
    evidence = real_root / "evidence.bin"
    evidence.write_bytes(b"evidence")
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(contracts.ContractError, match="symlink path component"):
        preparation._new_output_path(alias / "new_output", "output")
    with pytest.raises(contracts.ContractError, match="symlink path component"):
        preparation._resolve_root_artifact(
            _root_record("custom", evidence, real_root),
            {"custom": alias},
            "custom rooted evidence",
        )
