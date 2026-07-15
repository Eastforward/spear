#!/usr/bin/env python3
"""Render authenticated Walking/Idle videos for a nine-instance animal OFAT."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import controlled_source_asset_schema as contracts  # noqa: E402
from tools import render_stable_quadruped_ofat_contact_sheets as static_review  # noqa: E402


SCHEMA = "avengine_stable_animal_ofat_animation_review_v1"
ACTIONS = ("Walking", "Idle")
BLENDER = Path("/data/jzy/.local/bin/blender")
FFMPEG = Path("/usr/bin/ffmpeg")
RENDERER = SPEAR_ROOT / "tools/blender_render_glb_animation.py"


class AnimationReviewError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise AnimationReviewError(f"missing or unsafe artifact: {path}")
    return {
        "absolute_path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def published_artifact(path: Path, staging: Path, output: Path) -> dict[str, Any]:
    value = artifact(path)
    relative = path.resolve().relative_to(staging.resolve())
    value["absolute_path"] = str((output / relative).resolve())
    return value


def camera_references(static_manifest: Path, batch_status: Path) -> dict[str, float]:
    value = static_review.load_json(static_manifest, "static OFAT review")
    if (
        value.get("schema") != static_review.SCHEMA
        or value.get("manifest_sha256") != contracts.manifest_sha256(value)
        or value.get("source_batch", {}).get("sha256") != sha256_file(batch_status)
    ):
        raise AnimationReviewError("static review does not authenticate this batch")
    result = {
        str(item["profile_schema_id"]): float(item["camera_reference_diagonal"])
        for item in value.get("profiles", [])
    }
    if not result or any(number <= 0.0 for number in result.values()):
        raise AnimationReviewError("static review has no positive camera references")
    return result


def build_render_command(
    *,
    blender: Path,
    glb: Path,
    action: str,
    output_dir: Path,
    camera_reference_diagonal: float,
    frames: int,
    width: int,
    height: int,
    samples: int,
) -> list[str]:
    if action not in ACTIONS:
        raise AnimationReviewError(f"unsupported action: {action}")
    return [
        str(blender),
        "--background",
        "--python",
        str(RENDERER),
        "--",
        "--input",
        str(glb),
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
        "--samples",
        str(samples),
        "--view",
        "side",
        "--orthographic",
        "--camera-reference-diagonal",
        f"{camera_reference_diagonal:.9f}",
        "--ground-plane",
    ]


def run_one(
    entry: Mapping[str, Any],
    action: str,
    destination: Path,
    camera_reference: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    glb = Path(entry["artifacts"]["glb"]["path"]).resolve()
    if sha256_file(glb) != entry["artifacts"]["glb"]["sha256"]:
        raise AnimationReviewError(f"instance GLB changed: {glb}")
    render_log = destination / "render.log"
    render_command = build_render_command(
        blender=args.blender,
        glb=glb,
        action=action,
        output_dir=destination,
        camera_reference_diagonal=camera_reference,
        frames=args.frames,
        width=args.width,
        height=args.height,
        samples=args.samples,
    )
    started = time.monotonic()
    rendered = subprocess.run(
        render_command,
        cwd=SPEAR_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    render_log.write_text(rendered.stdout + rendered.stderr, encoding="utf-8")
    frames = sorted(destination.glob("frame_*.png"))
    if (
        rendered.returncode != 0
        or "RENDER_GLB_ANIM_OK" not in rendered.stdout
        or len(frames) != args.frames
    ):
        raise AnimationReviewError(
            f"animation render failed: {entry['instance_id']} {action} log={render_log}"
        )
    video = destination / f"{action.lower()}_side.mp4"
    encode_log = destination / "encode.log"
    encode_command = [
        str(args.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-framerate",
        str(args.preview_fps),
        "-i",
        str(destination / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    encoded = subprocess.run(
        encode_command,
        cwd=SPEAR_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    encode_log.write_text(
        "COMMAND " + json.dumps(encode_command, ensure_ascii=False) + "\n"
        + encoded.stdout
        + encoded.stderr,
        encoding="utf-8",
    )
    if encoded.returncode != 0 or not video.is_file() or video.stat().st_size <= 0:
        raise AnimationReviewError(
            f"animation encode failed: {entry['instance_id']} {action} log={encode_log}"
        )
    return {
        "instance_id": entry["instance_id"],
        "label": entry["label"],
        "action": action,
        "sampled_attributes": entry["sampled_attributes"],
        "strict_deformation": entry.get("deformation_gate"),
        "frame_count": len(frames),
        "preview_fps": args.preview_fps,
        "elapsed_seconds": time.monotonic() - started,
        "video_path": video,
        "render_log_path": render_log,
        "encode_log_path": encode_log,
    }


def run(args: argparse.Namespace) -> Path:
    batch_path = args.batch_status.resolve()
    batch, groups = static_review.validate_batch(batch_path)
    references = camera_references(args.static_review.resolve(), batch_path)
    if set(references) != set(groups):
        raise AnimationReviewError("static and animation profile sets differ")
    if not all(path.is_file() for path in (args.blender, args.ffmpeg, RENDERER)):
        raise AnimationReviewError("Blender, ffmpeg, or renderer is unavailable")
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise AnimationReviewError(f"refusing to replace output: {output}")
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)
    started = time.monotonic()
    try:
        jobs = []
        for profile_id, entries in sorted(groups.items()):
            for entry in entries:
                for action in ACTIONS:
                    jobs.append(
                        (
                            profile_id,
                            entry,
                            action,
                            staging
                            / "profiles"
                            / profile_id
                            / "instances"
                            / entry["label"]
                            / action.lower(),
                            references[profile_id],
                        )
                    )
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one, entry, action, destination, camera, args): (
                    profile_id,
                    entry["instance_id"],
                    action,
                )
                for profile_id, entry, action, destination, camera in jobs
            }
            completed = 0
            for future in as_completed(futures):
                profile_id, instance_id, action = futures[future]
                results.append((profile_id, future.result()))
                completed += 1
                print(
                    f"STABLE_OFAT_ANIMATION_PROGRESS completed={completed}/{len(jobs)} "
                    f"profile={profile_id} instance={instance_id} action={action}",
                    flush=True,
                )
        profile_records = []
        for profile_id in sorted(groups):
            clips = []
            for _profile, item in sorted(
                (value for value in results if value[0] == profile_id),
                key=lambda value: (value[1]["label"], value[1]["action"]),
            ):
                clips.append(
                    {
                        key: value
                        for key, value in item.items()
                        if not key.endswith("_path")
                    }
                    | {
                        "video": published_artifact(
                            item["video_path"], staging, output
                        ),
                        "render_log": published_artifact(
                            item["render_log_path"], staging, output
                        ),
                        "encode_log": published_artifact(
                            item["encode_log_path"], staging, output
                        ),
                    }
                )
            profile_records.append(
                {
                    "profile_schema_id": profile_id,
                    "camera_reference_diagonal": references[profile_id],
                    "clip_count": len(clips),
                    "clips": clips,
                }
            )
        manifest = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state_classification": "research_candidate_animation_rendered_pending_lenient_visual_review",
            "formal_dataset_registration_authorized": False,
            "source_batch": artifact(batch_path),
            "source_batch_manifest_sha256": batch["manifest_sha256"],
            "source_static_review": artifact(args.static_review.resolve()),
            "profile_count": len(profile_records),
            "instance_count": sum(len(entries) for entries in groups.values()),
            "clip_count": len(jobs),
            "actions": list(ACTIONS),
            "elapsed_seconds": time.monotonic() - started,
            "profiles": profile_records,
            "automatic_checks": {
                "all_source_glbs_authenticated": True,
                "all_instances_have_walking_and_idle": True,
                "shared_camera_reference_used": True,
                "strict_deformation_records_retained": True,
                "lenient_visual_review": "pending",
            },
        }
        manifest["manifest_sha256"] = contracts.manifest_sha256(manifest)
        with (staging / "review_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(staging, output)
        published = output / "review_manifest.json"
        observed = static_review.load_json(published, "animation review")
        if observed["manifest_sha256"] != contracts.manifest_sha256(observed):
            raise AnimationReviewError("published animation manifest hash failed")
        return published
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--batch-status", type=Path, required=True)
    result.add_argument("--static-review", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--workers", type=int, default=8, choices=range(1, 17))
    result.add_argument("--frames", type=int, default=20)
    result.add_argument("--preview-fps", type=int, default=12)
    result.add_argument("--width", type=int, default=640)
    result.add_argument("--height", type=int, default=480)
    result.add_argument("--samples", type=int, default=8)
    result.add_argument("--blender", type=Path, default=BLENDER)
    result.add_argument("--ffmpeg", type=Path, default=FFMPEG)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = run(args)
    except (AnimationReviewError, static_review.ReviewError, OSError, ValueError) as error:
        print(f"STABLE_OFAT_ANIMATION_FAILED {error}", file=sys.stderr)
        return 2
    print(f"STABLE_OFAT_ANIMATION_OK manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
