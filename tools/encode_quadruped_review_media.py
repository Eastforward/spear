#!/usr/bin/env python3
"""Encode one authenticated quadruped review frame set without replacement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.run_target_native_generated_quadruped_review import (  # noqa: E402
    ENCODE_MANIFEST_SCHEMA,
    expected_review_ffmpeg_config,
    file_record,
    render_frame_set,
    require_render_manifest,
    verify_video,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--view", choices=("side", "front", "quarter"), required=True)
    parser.add_argument("--asset-yaw-deg", type=float, required=True)
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    args = parse_args(argv)
    raw_output = Path(os.path.abspath(args.output))
    raw_manifest = Path(os.path.abspath(args.manifest))
    if os.path.lexists(raw_output):
        raise RuntimeError(f"refusing to replace review video: {raw_output}")
    if os.path.lexists(raw_manifest):
        raise RuntimeError(
            f"refusing to replace encode manifest: {raw_manifest}"
        )
    args.input_glb = args.input_glb.resolve()
    args.render_manifest = args.render_manifest.resolve()
    args.frame_dir = args.frame_dir.resolve()
    args.output = raw_output
    args.manifest = raw_manifest
    if (args.width, args.height, args.fps) != (512, 384, 8):
        raise RuntimeError(
            "quadruped review encoding requires 512x384 at 8 fps"
        )
    render_payload = require_render_manifest(
        args.render_manifest,
        input_glb=args.input_glb,
        frame_dir=args.frame_dir,
        action=args.action,
        view=args.view,
        asset_yaw_deg=args.asset_yaw_deg,
        n_frames=args.n_frames,
    )
    ffmpeg = expected_review_ffmpeg_config(args.frame_dir, args.n_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-n",
            "-loglevel",
            ffmpeg["loglevel"],
            "-framerate",
            str(ffmpeg["input_framerate"]),
            "-start_number",
            str(ffmpeg["start_number"]),
            "-i",
            ffmpeg["input_pattern"],
            "-frames:v",
            str(ffmpeg["frame_count"]),
            "-c:v",
            ffmpeg["video_codec"],
            "-crf",
            str(ffmpeg["crf"]),
            "-pix_fmt",
            ffmpeg["pixel_format"],
            "-movflags",
            ffmpeg["movflags"],
            str(args.output),
        ],
        cwd=SPEAR_ROOT,
        check=True,
    )
    video = verify_video(
        args.output,
        args.n_frames,
        expected_width=args.width,
        expected_height=args.height,
        expected_fps=args.fps,
    )
    # Re-read the render lineage after FFmpeg so a concurrent frame or input
    # mutation cannot be hidden behind a previously loaded JSON object.
    render_readback = require_render_manifest(
        args.render_manifest,
        input_glb=args.input_glb,
        frame_dir=args.frame_dir,
        action=args.action,
        view=args.view,
        asset_yaw_deg=args.asset_yaw_deg,
        n_frames=args.n_frames,
    )
    if render_readback != render_payload:
        raise RuntimeError(
            "render manifest changed while its video was encoded"
        )
    current_video = file_record(args.output)
    if any(
        video[name] != current_video[name]
        for name in ("path", "sha256", "size_bytes")
    ):
        raise RuntimeError(
            "review video changed before its encode manifest was written"
        )
    payload = {
        "schema": ENCODE_MANIFEST_SCHEMA,
        "status": "video_encoded_and_probed",
        "formal_dataset_registration_authorized": False,
        "media_identity": {
            "label": args.label,
            "action": args.action,
            "view": args.view,
            "asset_yaw_deg": float(args.asset_yaw_deg),
        },
        "render_manifest": file_record(args.render_manifest),
        "frame_set": render_frame_set(render_payload),
        "ffmpeg": ffmpeg,
        "video": video,
    }
    write_json_exclusive(args.manifest, payload)
    print(
        f"QUADRUPED_REVIEW_ENCODE_OK output={args.output} "
        f"manifest={args.manifest}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"QUADRUPED_REVIEW_ENCODE_FAILED {error}", file=sys.stderr)
        raise SystemExit(2)
