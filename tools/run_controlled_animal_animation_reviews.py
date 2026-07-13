#!/usr/bin/env python3
"""Render authenticated Walk/Idle videos for controlled animal runtimes."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_lod_binding as lod_binding


BATCH_SCHEMA = "avengine_controlled_animal_animation_review_batch_v1"
REVIEW_SCHEMA = "avengine_controlled_animal_animation_review_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/data/jzy/.local/bin/blender")
RENDERER = SPEAR_ROOT / "tools/blender_render_glb_animation.py"
FFMPEG = Path("/usr/bin/ffmpeg")
FFPROBE = Path("/usr/bin/ffprobe")
REVIEW_SPECS = (
    ("walking_side", "Walking", "side"),
    ("walking_front", "Walking", "front"),
    ("idle_side", "Idle", "side"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_lod_binding_batch(
    path: Path, asset_ids: Sequence[str] = ()
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"LOD/binding batch is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != lod_binding.BATCH_SCHEMA
        or payload.get("batch_sha256") != _hash_without(payload, "batch_sha256")
        or payload.get("job_count") != len(payload.get("attempts", []))
        or payload.get("passed_count") + payload.get("failed_count")
        != payload.get("job_count")
    ):
        raise contracts.ContractError("LOD/binding batch contract/hash is invalid")
    selected = set(asset_ids)
    known = {item.get("asset_id") for item in payload["attempts"]}
    if selected - known:
        raise contracts.ContractError(
            f"unknown requested assets: {sorted(selected - known)}"
        )
    attempts = []
    for item in payload["attempts"]:
        if selected and item["asset_id"] not in selected:
            continue
        if item.get("status") != "passed_lod_binding_glb_readback":
            raise contracts.ContractError(
                f"selected asset did not pass binding: {item.get('asset_id')}"
            )
        artifact = item.get("artifacts", {}).get("rigged_glb", {})
        rigged = (path.parent / artifact.get("path", "")).resolve()
        try:
            rigged.relative_to(path.parent.resolve())
        except ValueError as error:
            raise contracts.ContractError("rigged runtime escaped batch root") from error
        if (
            rigged.is_symlink()
            or not rigged.is_file()
            or rigged.stat().st_size != artifact.get("size_bytes")
            or _sha256_file(rigged) != artifact.get("sha256")
            or item.get("rigged_runtime_readback", {}).get("animation_names")
            != ["Idle", "Walking"]
        ):
            raise contracts.ContractError(
                f"rigged runtime changed: {item.get('asset_id')}"
            )
        attempts.append({**copy.deepcopy(item), "rigged_path": rigged})
    if not attempts:
        raise contracts.ContractError("animation review selection is empty")
    return path, payload, attempts


def build_render_command(
    input_glb: Path,
    output_dir: Path,
    *,
    action: str,
    view: str,
    frames: int = 24,
    width: int = 640,
    height: int = 480,
) -> list[str]:
    return [
        str(BLENDER),
        "-b",
        "--python",
        str(RENDERER),
        "--",
        "--input",
        str(input_glb),
        "--action",
        action,
        "--output-dir",
        str(output_dir),
        "--n-frames",
        str(frames),
        "--width",
        str(width),
        "--height",
        str(height),
        "--view",
        view,
        "--camera-distance-multiplier",
        "2.0",
        "--ground-plane",
    ]


def build_encode_command(frame_dir: Path, video: Path, fps: int = 12) -> list[str]:
    return [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(video),
    ]


def _run_logged(command: Sequence[str], log_path: Path, timeout: int) -> float:
    started = time.monotonic()
    with log_path.open("xb") as log:
        completed = subprocess.run(
            list(command),
            cwd=SPEAR_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise contracts.ContractError(
            f"command failed ({completed.returncode}); see {log_path}"
        )
    return time.monotonic() - started


def _probe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,duration,pix_fmt",
            "-of",
            "json",
            str(path),
        ],
        cwd=SPEAR_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise contracts.ContractError(f"ffprobe failed: {path}")
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise contracts.ContractError(f"video stream readback failed: {path}")
    stream = streams[0]
    if (
        stream.get("codec_name") != "h264"
        or stream.get("pix_fmt") != "yuv420p"
        or int(stream.get("width", 0)) != 640
        or int(stream.get("height", 0)) != 480
        or int(stream.get("nb_frames", 0)) != 24
        or not 1.9 <= float(stream.get("duration", 0)) <= 2.1
    ):
        raise contracts.ContractError(f"video contract changed: {stream}")
    return {
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_frames"]),
        "duration_seconds": float(stream["duration"]),
    }


def _labeled_panel(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as opened:
        opened.load()
        panel = opened.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((5, 5, bounds[2] + 13, bounds[3] + 13), fill=(0, 0, 0))
    draw.text((9, 9), label, font=font, fill=(255, 255, 255))
    return panel


def build_contact_sheet(review_root: Path, output: Path) -> None:
    samples = []
    for index in (0, 3, 6, 9, 12, 15, 18, 21):
        samples.append(
            (
                review_root / "walking_side_frames" / f"frame_{index:04d}.png",
                f"Walk {index:02d}",
            )
        )
    for index in (0, 8, 16, 23):
        samples.append(
            (
                review_root / "idle_side_frames" / f"frame_{index:04d}.png",
                f"Idle {index:02d}",
            )
        )
    canvas = Image.new("RGB", (1280, 720), (22, 22, 22))
    for index, (path, label) in enumerate(samples):
        canvas.paste(
            _labeled_panel(path, label, (320, 240)),
            ((index % 4) * 320, (index // 4) * 240),
        )
    canvas.save(output, format="PNG", optimize=False, compress_level=6)


def _run_one(attempt: Mapping[str, Any], staging: Path) -> dict[str, Any]:
    asset_id = attempt["asset_id"]
    root = staging / "reviews" / asset_id
    root.mkdir(parents=True, exist_ok=False)
    videos = {}
    timings = {}
    try:
        for stem, action, view in REVIEW_SPECS:
            frame_dir = root / f"{stem}_frames"
            frame_dir.mkdir()
            render_log = root / f"{stem}_render.log"
            encode_log = root / f"{stem}_encode.log"
            video = root / f"{stem}.mp4"
            timings[f"{stem}_render_seconds"] = _run_logged(
                build_render_command(
                    attempt["rigged_path"],
                    frame_dir,
                    action=action,
                    view=view,
                ),
                render_log,
                timeout=900,
            )
            timings[f"{stem}_encode_seconds"] = _run_logged(
                build_encode_command(frame_dir, video), encode_log, timeout=120
            )
            videos[stem] = {
                "action": action,
                "view": view,
                "video": _relative_artifact(video, staging),
                "render_log": _relative_artifact(render_log, staging),
                "encode_log": _relative_artifact(encode_log, staging),
                "readback": _probe_video(video),
            }
        contact = root / "walk_idle_contact_sheet.png"
        build_contact_sheet(root, contact)
        review: dict[str, Any] = {
            "schema": REVIEW_SCHEMA,
            "asset_id": asset_id,
            "profile_schema_id": attempt["profile_schema_id"],
            "request_sha256": attempt["request_sha256"],
            "sampled_attributes": attempt["sampled_attributes"],
            "rigged_runtime": copy.deepcopy(attempt["artifacts"]["rigged_glb"]),
            "videos": videos,
            "contact_sheet": _relative_artifact(contact, staging),
            "timings": timings,
            "automatic_checks": {
                "rigged_runtime_reauthenticated": True,
                "walking_side_rendered": True,
                "walking_front_rendered": True,
                "idle_side_rendered": True,
                "rest_pose_ground_plane_rendered": True,
                "all_videos_h264_yuv420p_readback": True,
                "overall": "passed",
            },
            "visual_qa": "pending",
            "next_gate": "visual_limb_direction_contact_and_deformation_decision",
        }
        review["review_sha256"] = _hash_without(review, "review_sha256")
        review_path = root / "animation_review_manifest.json"
        contracts.write_json_no_replace(review_path, review)
        return {
            "asset_id": asset_id,
            "profile_schema_id": attempt["profile_schema_id"],
            "status": "rendered_pending_visual_qa",
            "review": _relative_artifact(review_path, staging),
            "review_sha256": review["review_sha256"],
            "contact_sheet": review["contact_sheet"],
            "videos": {
                name: value["video"] for name, value in videos.items()
            },
            "timings": timings,
        }
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        return {
            "asset_id": asset_id,
            "profile_schema_id": attempt["profile_schema_id"],
            "status": "failed",
            "error": str(error),
        }


def build_overview(results: Sequence[Mapping[str, Any]], staging: Path) -> Path:
    passed = [item for item in results if item["status"] == "rendered_pending_visual_qa"]
    columns = 5
    rows = math.ceil(len(passed) / columns)
    canvas = Image.new("RGB", (columns * 320, rows * 200), (18, 18, 18))
    for index, item in enumerate(passed):
        source = staging / item["contact_sheet"]["path"]
        panel = _labeled_panel(source, item["asset_id"], (320, 180))
        canvas.paste(panel, ((index % columns) * 320, (index // columns) * 200))
    output = staging / "walk_idle_overview.png"
    canvas.save(output, format="PNG", optimize=False, compress_level=6)
    return output


def run_reviews(
    lod_binding_batch_path: Path,
    output_root: Path,
    *,
    workers: int = 8,
    asset_ids: Sequence[str] = (),
) -> Path:
    if not 1 <= workers <= 16:
        raise contracts.ContractError("workers must be between 1 and 16")
    if not all(path.is_file() for path in (BLENDER, RENDERER, FFMPEG, FFPROBE)):
        raise contracts.ContractError("review renderer or video tools are missing")
    batch_path, batch, attempts = load_lod_binding_batch(
        lod_binding_batch_path, asset_ids
    )
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    started_at = _utc_now()
    started = time.monotonic()
    try:
        results = []
        with ThreadPoolExecutor(max_workers=min(workers, len(attempts))) as executor:
            futures = {
                executor.submit(_run_one, attempt, staging): attempt["asset_id"]
                for attempt in attempts
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    "CONTROLLED_ANIMAL_ANIMATION_REVIEW_DONE "
                    f"asset={result['asset_id']} status={result['status']}",
                    flush=True,
                )
        results.sort(key=lambda item: item["asset_id"])
        passed = sum(
            item["status"] == "rendered_pending_visual_qa" for item in results
        )
        failed = len(results) - passed
        overview = build_overview(results, staging) if passed else None
        manifest: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "status": "rendered_pending_visual_qa"
            if failed == 0
            else "completed_with_render_failures",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "wall_seconds": time.monotonic() - started,
            "lod_binding_batch": {
                "path": str(batch_path),
                "sha256": _sha256_file(batch_path),
                "batch_sha256": batch["batch_sha256"],
            },
            "renderer": {
                "path": RENDERER.relative_to(SPEAR_ROOT).as_posix(),
                "sha256": _sha256_file(RENDERER),
                "size_bytes": RENDERER.stat().st_size,
            },
            "parameters": {
                "frames_per_video": 24,
                "fps": 12,
                "resolution": [640, 480],
                "views": [list(item) for item in REVIEW_SPECS],
                "ground_plane": "rest_pose_min_z",
                "workers": min(workers, len(attempts)),
            },
            "review_count": len(results),
            "passed_render_count": passed,
            "failed_render_count": failed,
            "reviews": results,
            "overview": _relative_artifact(overview, staging) if overview else None,
            "automatic_checks": {
                "lod_binding_batch_reauthenticated": True,
                "all_selected_runtimes_reauthenticated": True,
                "all_successful_reviews_have_walk_and_idle_videos": passed > 0,
                "all_successful_videos_read_back": passed > 0,
                "visual_qa_pending": True,
                "overall": "passed" if failed == 0 else "needs_render_failure_review",
            },
        }
        manifest["batch_sha256"] = _hash_without(manifest, "batch_sha256")
        contracts.write_json_no_replace(
            staging / "animation_review_batch_manifest.json", manifest
        )
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("animation review output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "animation_review_batch_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lod-binding-batch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--asset-id", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = run_reviews(
            args.lod_binding_batch,
            args.output_root,
            workers=args.workers,
            asset_ids=args.asset_id,
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_ANIMATION_REVIEW_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_ANIMATION_REVIEW_OK "
        f"rendered={manifest['passed_render_count']} "
        f"failed={manifest['failed_render_count']} output={manifest_path}"
    )
    return 0 if manifest["failed_render_count"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
