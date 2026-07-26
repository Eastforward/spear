#!/usr/bin/env python3
"""Run the reviewed post-TokenRig quadruped normalization and QA sequence.

This runner deliberately starts from a target-native, unanimated rig.  FLUX,
Pixel3D and topology/PBR repair keep their own review gates and artifacts.  It
then runs, in order: rigid heading normalization, rig audit, four-foot support
plane leveling, semantic motion retarget, rotation-invariant Walk/Idle
deformation audit, six-view rendering, encoding and media readback.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SPEAR_ROOT = Path(__file__).resolve().parents[1]
TOOLS = SPEAR_ROOT / "tools"
SCHEMA = "avengine_target_native_generated_quadruped_review_run_v1"

if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.generated_animal_forward_contract import (  # noqa: E402
    assert_declared_motion_basis,
    load_forward_declaration,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-rig-glb", type=Path, required=True)
    parser.add_argument(
        "--forward-declaration",
        type=Path,
        help=(
            "Authenticated single-point forward declaration.  Supplies the "
            "reviewed source front yaw, the canonical target axis and the "
            "constant motion basis of the declared donor; the legacy "
            "per-asset heading/motion-basis flags are then forbidden."
        ),
    )
    parser.add_argument("--heading-review-evidence", type=Path)
    parser.add_argument("--reviewed-source-front-yaw-deg", type=float)
    parser.add_argument(
        "--target-front-axis",
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
        default="positive-x",
    )
    parser.add_argument("--source-motion-glb", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", default=os.environ.get("AVENGINE_BLENDER", "blender"))
    parser.add_argument("--motion-amplitude", type=float, default=1.0)
    parser.add_argument(
        "--motion-basis-yaw-deg", type=int, choices=(-90, 0, 90, 180)
    )
    parser.add_argument("--side-chain-mode", choices=("matched", "swapped"))
    parser.add_argument("--deformation-samples", type=int, default=24)
    parser.add_argument("--review-frames", type=int, default=8)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help=(
            "Stop after retarget, gait-direction audit and two cheap Walking "
            "renders so direction mistakes surface in minutes instead of "
            "after the full deformation audit and six-view render."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or unsafe {label}: {path}")
    return path


def executable(value: str) -> str:
    candidate = Path(value)
    if candidate.parent != Path("."):
        candidate = candidate.resolve()
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"missing or unsafe Blender executable: {candidate}")
        return str(candidate)
    resolved = shutil.which(value)
    if resolved is None:
        raise ValueError(f"Blender executable not found on PATH: {value}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    path = regular_file(path, "output artifact")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def output_paths(root: Path) -> dict[str, Path]:
    heading = root / "01_heading"
    rig = root / "02_rig_audit"
    level = root / "03_support_plane"
    motion = root / "04_motion"
    review = root / "05_review"
    return {
        "heading_glb": heading / "target_heading.glb",
        "heading_manifest": heading / "manifest.json",
        "rig_audit": rig / "generated_rig_audit.json",
        "leveled_glb": level / "target_heading_leveled.glb",
        "level_manifest": level / "manifest.json",
        "animated_glb": motion / "target_animated.glb",
        "retarget_manifest": motion / "retarget_manifest.json",
        "gait_audit": motion / "gait_direction_audit.json",
        "deformation_audit": motion / "skinned_deformation_walk_idle.json",
        "review_root": review,
        "result": root / "review_run.json",
        "preview_result": root / "preview_run.json",
    }


def blender_command(blender: str, script: str, arguments: list[str]) -> list[str]:
    # Blender otherwise reports some uncaught Python-script exceptions with a
    # successful process return code and lets the outer pipeline continue with
    # missing artifacts.  New-asset stages must fail closed at the first error.
    return [
        blender,
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        str(TOOLS / script),
        "--",
        *arguments,
    ]


def build_commands(args, paths: dict[str, Path], blender: str) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    commands.append(
        (
            "heading",
            blender_command(
                blender,
                "blender_normalize_generated_animal_heading.py",
                [
                    "--input", str(args.target_rig_glb),
                    "--output", str(paths["heading_glb"]),
                    "--manifest", str(paths["heading_manifest"]),
                    "--reviewed-source-front-yaw-deg",
                    str(args.reviewed_source_front_yaw_deg),
                    "--target-front-axis", args.target_front_axis,
                    "--review-evidence", str(args.heading_review_evidence),
                ],
            ),
        )
    )
    commands.append(
        (
            "rig_audit",
            blender_command(
                blender,
                "blender_audit_generated_animal_rig.py",
                [
                    "--input", str(paths["heading_glb"]),
                    "--output", str(paths["rig_audit"]),
                    "--front-axis", args.target_front_axis,
                ],
            ),
        )
    )
    commands.append(
        (
            "support_plane",
            blender_command(
                blender,
                "blender_level_generated_animal_support_plane.py",
                [
                    "--input", str(paths["heading_glb"]),
                    "--output", str(paths["leveled_glb"]),
                    "--manifest", str(paths["level_manifest"]),
                    "--front-axis", args.target_front_axis,
                    "--review-evidence", str(paths["rig_audit"]),
                    "--maximum-tilt-deg", "30",
                    "--maximum-foot-plane-residual-ratio", "0.02",
                ],
            ),
        )
    )
    commands.append(
        (
            "retarget",
            blender_command(
                blender,
                "blender_retarget_quaternius_to_generated_quadruped.py",
                [
                    "--target-glb", str(paths["leveled_glb"]),
                    "--source-rig-glb", str(args.source_motion_glb),
                    "--output-glb", str(paths["animated_glb"]),
                    "--manifest", str(paths["retarget_manifest"]),
                    "--technical-spike-only",
                    "--target-front-axis", args.target_front_axis,
                    "--motion-amplitude", str(args.motion_amplitude),
                    "--rotation-transfer-mode", "world-left-delta-v2",
                    "--pose-transfer-mode", "world-rotation-retarget-v2",
                    "--motion-basis-yaw-deg", str(args.motion_basis_yaw_deg),
                    "--side-chain-mode", args.side_chain_mode,
                ],
            ),
        )
    )
    commands.append(
        (
            "gait_direction",
            blender_command(
                blender,
                "blender_audit_gait_direction.py",
                [
                    "--input", str(paths["animated_glb"]),
                    "--output", str(paths["gait_audit"]),
                    "--action", "Walking",
                ],
            ),
        )
    )
    commands.append(
        (
            "deformation",
            blender_command(
                blender,
                "blender_audit_skinned_deformation.py",
                [
                    "--input", str(paths["animated_glb"]),
                    "--output", str(paths["deformation_audit"]),
                    "--action", "Walking",
                    "--action", "Idle",
                    "--samples", str(args.deformation_samples),
                ],
            ),
        )
    )
    views = (
        ("walking_side", "Walking", "side", "0"),
        ("walking_front", "Walking", "front", "0"),
        ("walking_rear", "Walking", "front", "180"),
        ("idle_side", "Idle", "side", "0"),
        ("idle_front", "Idle", "front", "0"),
        ("idle_rear", "Idle", "front", "180"),
    )
    for label, action, view, yaw in views:
        frame_dir = paths["review_root"] / f"{label}_frames"
        video = paths["review_root"] / f"{label}.mp4"
        commands.append(
            (
                f"render_{label}",
                blender_command(
                    blender,
                    "blender_render_glb_animation.py",
                    [
                        "--input", str(paths["animated_glb"]),
                        "--action", action,
                        "--output-dir", str(frame_dir),
                        "--n-frames", str(args.review_frames),
                        "--width", "512",
                        "--height", "384",
                        "--samples", "16",
                        "--view", view,
                        "--asset-yaw-deg", yaw,
                        "--trajectory-distance-ratio", "0",
                        "--ground-plane",
                        "--engine", "BLENDER_EEVEE_NEXT",
                    ],
                ),
            )
        )
        commands.append(
            (
                f"encode_{label}",
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", "8",
                    "-i", str(frame_dir / "frame_%04d.png"),
                    "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(video),
                ],
            )
        )
    return commands


def verify_video(path: Path, expected_frames: int) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", 0)) != 512
        or int(stream.get("height", 0)) != 384
        or int(stream.get("nb_frames", 0)) != expected_frames
    ):
        raise RuntimeError(f"review video readback failed: {path}")
    return {
        **file_record(path),
        "codec": "h264",
        "width": 512,
        "height": 384,
        "frame_count": expected_frames,
        "frame_rate": stream.get("r_frame_rate"),
        "duration_seconds": float(payload["format"]["duration"]),
    }


def main(argv=None):
    args = parse_args(argv)
    args.target_rig_glb = regular_file(args.target_rig_glb, "target rig GLB")
    if args.forward_declaration is not None:
        if (
            args.heading_review_evidence is not None
            or args.reviewed_source_front_yaw_deg is not None
            or args.motion_basis_yaw_deg is not None
            or args.side_chain_mode is not None
        ):
            raise ValueError(
                "--forward-declaration replaces --heading-review-evidence, "
                "--reviewed-source-front-yaw-deg, --motion-basis-yaw-deg and "
                "--side-chain-mode; the anatomical front is declared exactly "
                "once and the motion basis is the donor constant"
            )
        declaration_path = regular_file(
            args.forward_declaration, "forward declaration"
        )
        declaration = load_forward_declaration(declaration_path)
        basis = declaration["expected_motion_basis"]
        assert_declared_motion_basis(
            declaration["motion_donor_tag"],
            basis["motion_basis_yaw_deg"],
            basis["side_chain_mode"],
        )
        args.heading_review_evidence = declaration_path
        args.reviewed_source_front_yaw_deg = declaration[
            "reviewed_source_front_yaw_deg"
        ]
        args.target_front_axis = declaration["target_front_axis"]
        args.motion_basis_yaw_deg = basis["motion_basis_yaw_deg"]
        args.side_chain_mode = basis["side_chain_mode"]
        forward_contract = {
            "mode": "forward_declaration_v1",
            "declaration": file_record(declaration_path),
            "motion_donor_tag": declaration["motion_donor_tag"],
            "derived_motion_basis": dict(basis),
        }
    else:
        if (
            args.heading_review_evidence is None
            or args.reviewed_source_front_yaw_deg is None
            or args.motion_basis_yaw_deg is None
            or args.side_chain_mode is None
        ):
            raise ValueError(
                "legacy mode requires --heading-review-evidence, "
                "--reviewed-source-front-yaw-deg, --motion-basis-yaw-deg and "
                "--side-chain-mode together; prefer --forward-declaration"
            )
        args.heading_review_evidence = regular_file(
            args.heading_review_evidence, "heading review evidence"
        )
        forward_contract = {
            "mode": "legacy_free_parameters_deprecated",
            "deprecation": (
                "per-asset motion-basis and heading parameters are a known "
                "orientation-bug source; migrate to --forward-declaration"
            ),
        }
    args.source_motion_glb = regular_file(args.source_motion_glb, "source motion GLB")
    if not 0.0 <= args.motion_amplitude <= 1.0:
        raise ValueError("--motion-amplitude must be in [0, 1]")
    if not 4 <= args.deformation_samples <= 120:
        raise ValueError("--deformation-samples must be in [4, 120]")
    if not 4 <= args.review_frames <= 120:
        raise ValueError("--review-frames must be in [4, 120]")
    root = args.output_root.resolve()
    if root.exists() or root.is_symlink():
        raise ValueError(f"refusing to replace output root: {root}")
    blender = executable(args.blender)
    paths = output_paths(root)
    commands = build_commands(args, paths, blender)
    preview_labels = (
        "heading", "rig_audit", "support_plane", "retarget", "gait_direction",
        "render_walking_side", "encode_walking_side",
        "render_walking_front", "encode_walking_front",
    )
    if args.preview_only:
        commands = [item for item in commands if item[0] in preview_labels]
    if args.validate_only:
        print(json.dumps({"schema": SCHEMA, "commands": commands}, indent=2))
        return 0

    root.mkdir(parents=True)
    timings = {}
    for label, command in commands:
        started = time.monotonic()
        subprocess.run(command, cwd=SPEAR_ROOT, check=True)
        timings[label] = time.monotonic() - started

    gait = json.loads(paths["gait_audit"].read_text(encoding="utf-8"))
    if args.preview_only:
        media = {
            label: verify_video(
                paths["review_root"] / f"{label}.mp4", args.review_frames
            )
            for label in ("walking_side", "walking_front")
        }
        result = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "preview_only_pending_full_review",
            "formal_dataset_registration_authorized": False,
            "forward_contract": forward_contract,
            "pipeline_order": [label for label, _command in commands],
            "inputs": {
                "target_rig_glb": file_record(args.target_rig_glb),
                "heading_review_evidence": file_record(args.heading_review_evidence),
                "source_motion_glb": file_record(args.source_motion_glb),
            },
            "outputs": {
                "heading_manifest": file_record(paths["heading_manifest"]),
                "rig_audit": file_record(paths["rig_audit"]),
                "support_plane_manifest": file_record(paths["level_manifest"]),
                "animated_glb": file_record(paths["animated_glb"]),
                "retarget_manifest": file_record(paths["retarget_manifest"]),
                "gait_direction_audit": file_record(paths["gait_audit"]),
                "gait_direction_status": gait.get("status"),
                "media": media,
            },
            "timings_seconds": timings,
        }
        with paths["preview_result"].open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            "TARGET_NATIVE_QUADRUPED_PREVIEW_OK "
            f"output={paths['preview_result']}"
        )
        return 0

    deformation = json.loads(paths["deformation_audit"].read_text(encoding="utf-8"))
    media = {
        label: verify_video(paths["review_root"] / f"{label}.mp4", args.review_frames)
        for label in (
            "walking_side", "walking_front", "walking_rear",
            "idle_side", "idle_front", "idle_rear",
        )
    }
    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_candidate_pending_human_review",
        "formal_dataset_registration_authorized": False,
        "forward_contract": forward_contract,
        "pipeline_order": [label for label, _command in commands],
        "inputs": {
            "target_rig_glb": file_record(args.target_rig_glb),
            "heading_review_evidence": file_record(args.heading_review_evidence),
            "source_motion_glb": file_record(args.source_motion_glb),
        },
        "outputs": {
            "heading_manifest": file_record(paths["heading_manifest"]),
            "rig_audit": file_record(paths["rig_audit"]),
            "support_plane_manifest": file_record(paths["level_manifest"]),
            "animated_glb": file_record(paths["animated_glb"]),
            "retarget_manifest": file_record(paths["retarget_manifest"]),
            "gait_direction_audit": file_record(paths["gait_audit"]),
            "gait_direction_status": gait.get("status"),
            "deformation_audit": file_record(paths["deformation_audit"]),
            "deformation_overall": deformation.get("overall"),
            "media": media,
        },
        "timings_seconds": timings,
    }
    with paths["result"].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"TARGET_NATIVE_QUADRUPED_REVIEW_OK output={paths['result']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"TARGET_NATIVE_QUADRUPED_REVIEW_FAILED {error}", file=sys.stderr)
        raise SystemExit(2)
