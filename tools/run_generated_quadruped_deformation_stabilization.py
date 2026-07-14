#!/usr/bin/env python3
"""Batch the approved generated-quadruped deformation stabilization stage.

This runner starts only after a human-approved motion-basis/cardinal-direction
decision and an animated fitted rig already exist.  It never infers direction,
never edits its inputs, and never promotes outputs to formal dataset assets.

For every immutable job it creates a topology-safe 200k PBR runtime mesh,
transfers the approved rig with the accelerated exact-triangle BVH backend,
repairs only vertex weights against Walking/Idle, performs a 41-frame GLB
readback, renders both actions, and publishes hashes plus timing evidence.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts


SCHEMA = "avengine_generated_quadruped_stabilization_jobs_v1"
RESULT_SCHEMA = "avengine_generated_quadruped_stabilization_batch_v1"
DECISION_SCHEMA = "generated_animal_motion_basis_manual_decision_v1"
DECISION_STATUS = "motion_basis_approved"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = Path("/data/jzy/.local/bin/blender")
DEFAULT_TARGET_FACES = 200_000
CARDINAL_YAWS = {-90.0, 0.0, 90.0, 180.0}
ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

RUNTIME_SCRIPT = SPEAR_ROOT / "tools/blender_create_runtime_proxy_mesh.py"
BIND_SCRIPT = SPEAR_ROOT / "tools/blender_robust_swap_mesh_keep_rig.py"
DIAGNOSTIC_SCRIPT = SPEAR_ROOT / "tools/blender_diagnose_skinned_deformation.py"
REPAIR_SCRIPT = SPEAR_ROOT / "tools/blender_repair_animated_quadruped_weight_stretch.py"
RENDER_SCRIPT = SPEAR_ROOT / "tools/blender_render_glb_animation.py"


class StabilizationError(RuntimeError):
    """Raised when a job violates an immutable or QA contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def canonical_hash_without(value: Mapping[str, Any], key: str) -> str:
    body = {name: item for name, item in value.items() if name != key}
    return hashlib.sha256(contracts.canonical_json(body).encode("utf-8")).hexdigest()


def resolve_pinned_file(record: Mapping[str, Any], *, label: str) -> Path:
    try:
        raw_path = Path(str(record["path"]))
        expected_hash = str(record["sha256"])
    except (KeyError, TypeError) as error:
        raise StabilizationError(f"{label} needs path and sha256") from error
    if not raw_path.is_absolute():
        raise StabilizationError(f"{label} path must be absolute: {raw_path}")
    path = raw_path.resolve()
    if raw_path.is_symlink() or not path.is_file():
        raise StabilizationError(f"{label} is missing, relative, or symlinked: {path}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise StabilizationError(
            f"{label} hash changed: expected {expected_hash}, got {actual}: {path}"
        )
    expected_size = record.get("size_bytes")
    if expected_size is not None and int(expected_size) != path.stat().st_size:
        raise StabilizationError(f"{label} size changed: {path}")
    return path


def validate_motion_basis_decision(path: Path, *, asset_id: str) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise StabilizationError(f"motion-basis decision missing: {path}")
    decision = load_json(path)
    if not isinstance(decision, dict):
        raise StabilizationError("motion-basis decision must be a JSON object")
    if decision.get("schema") != DECISION_SCHEMA:
        raise StabilizationError("motion-basis decision schema changed")
    if decision.get("asset_id") != asset_id:
        raise StabilizationError("motion-basis decision asset_id changed")
    if decision.get("status") != DECISION_STATUS:
        raise StabilizationError("motion basis is not approved")
    if decision.get("human_approved") is not True:
        raise StabilizationError("motion basis lacks explicit human approval")
    if decision.get("target_animation_generation_authorized") is not True:
        raise StabilizationError("motion-basis decision does not authorize animation")
    if decision.get("formal_dataset_registration_authorized") is not False:
        raise StabilizationError("motion-basis decision improperly authorizes formal use")
    if decision.get("decision_sha256") != canonical_hash_without(
        decision, "decision_sha256"
    ):
        raise StabilizationError("motion-basis decision hash is invalid")
    try:
        yaw = float(decision["manual_cardinal_motion_basis_yaw_deg"])
    except (KeyError, TypeError, ValueError) as error:
        raise StabilizationError("motion-basis decision has no cardinal yaw") from error
    if yaw not in CARDINAL_YAWS:
        raise StabilizationError(f"fine-yaw decisions are forbidden: {yaw}")
    if decision.get("target", {}).get("reviewed_front_axis") != "positive-x":
        raise StabilizationError("only reviewed positive-x quadruped targets are supported")
    return decision


def validate_job(job: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(job, Mapping):
        raise StabilizationError("each job must be an object")
    asset_id = str(job.get("asset_id", ""))
    if not ASSET_ID_RE.fullmatch(asset_id):
        raise StabilizationError(f"invalid asset_id: {asset_id!r}")
    raw_mesh = resolve_pinned_file(job.get("raw_pbr_glb", {}), label="raw_pbr_glb")
    animated_rig = resolve_pinned_file(
        job.get("approved_animated_rig_glb", {}),
        label="approved_animated_rig_glb",
    )
    decision_record = job.get("motion_basis_decision", {})
    decision_path = resolve_pinned_file(
        decision_record, label="motion_basis_decision"
    )
    decision = validate_motion_basis_decision(decision_path, asset_id=asset_id)

    orientation = job.get("target_preprocess", {})
    if not isinstance(orientation, Mapping):
        raise StabilizationError("target_preprocess must be an object")
    flip_x = orientation.get("flip_x")
    if not isinstance(flip_x, bool):
        raise StabilizationError("target_preprocess.flip_x must be explicit boolean")
    try:
        rotate_z = float(orientation.get("rotate_z_deg", 0.0))
    except (TypeError, ValueError) as error:
        raise StabilizationError("target_preprocess.rotate_z_deg must be numeric") from error
    if rotate_z not in CARDINAL_YAWS:
        raise StabilizationError(f"target preprocessing must be cardinal: {rotate_z}")
    if orientation.get("automatic_orientation_inference_used") is not False:
        raise StabilizationError("automatic orientation inference is forbidden")
    provenance = orientation.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise StabilizationError("target_preprocess.provenance is required")

    if Path(raw_mesh).suffix.lower() not in {".glb", ".gltf"}:
        raise StabilizationError("raw_pbr_glb must be glTF/GLB")
    return {
        "asset_id": asset_id,
        "raw_mesh": raw_mesh,
        "animated_rig": animated_rig,
        "decision_path": decision_path,
        "decision": decision,
        "flip_x": flip_x,
        "rotate_z": rotate_z,
        "preprocess_provenance": provenance,
    }


def load_jobs_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.resolve()
    payload = load_json(path)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise StabilizationError(f"invalid jobs manifest schema: {path}")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise StabilizationError("jobs manifest must contain at least one job")
    jobs = [validate_job(job) for job in raw_jobs]
    asset_ids = [job["asset_id"] for job in jobs]
    if len(set(asset_ids)) != len(asset_ids):
        raise StabilizationError("jobs manifest contains duplicate asset_id values")
    return payload, jobs


def blender_command(blender: Path, script: Path, arguments: Sequence[str]) -> list[str]:
    return [str(blender), "-b", "--python", str(script), "--", *map(str, arguments)]


def build_commands(
    job: Mapping[str, Any],
    job_root: Path,
    *,
    blender: Path = DEFAULT_BLENDER,
    target_faces: int = DEFAULT_TARGET_FACES,
) -> dict[str, list[str]]:
    runtime_dir = job_root / "runtime"
    binding_dir = job_root / "binding"
    diagnostics_dir = job_root / "diagnostics"
    repair_dir = job_root / "repair"
    review_dir = job_root / "review"
    runtime_glb = runtime_dir / f"mesh_runtime_{target_faces}.glb"
    runtime_json = runtime_dir / f"mesh_runtime_{target_faces}.json"
    bound_glb = binding_dir / "animated_walk_idle_bound.glb"
    repaired_glb = repair_dir / "animated_walk_idle_repaired.glb"

    bind_args = [
        "--rig-glb", str(job["animated_rig"]),
        "--new-mesh", str(runtime_glb),
        "--output", str(bound_glb),
        "--auto-align", "yes",
        "--align-mode", "uniform",
        "--semantic-forward-axis", "positive-x",
        "--weight-mode", "nearest",
        "--nearest-backend", "bvh",
        "--top-k", "4",
        "--max-distance-ratio", "0.10",
        "--dampen-foot-rotations", "1.0",
        "--dampen-head-rotations", "1.0",
        "--dampen-tail-rotations", "1.0",
        "--animation-export-mode", "actions",
        "--export-action-policy", "walk-idle",
        "--ue-safe-animation-channels", "no",
        "--remove-ground-artifacts", "no",
        "--remove-limb-bridges", "no",
        "--backface-culling", "no",
    ]
    if job["flip_x"]:
        bind_args.append("--flip-x")
    if float(job["rotate_z"]):
        bind_args.extend(["--target-rotate-z-deg", str(float(job["rotate_z"]))])

    commands = {
        "runtime": blender_command(
            blender,
            RUNTIME_SCRIPT,
            [
                "--source", str(job["raw_mesh"]),
                "--output", str(runtime_glb),
                "--metadata", str(runtime_json),
                "--target-faces", str(target_faces),
                "--double-sided",
            ],
        ),
        "binding": blender_command(blender, BIND_SCRIPT, bind_args),
        "diagnostic_before": blender_command(
            blender,
            DIAGNOSTIC_SCRIPT,
            [
                "--input", str(bound_glb),
                "--output", str(diagnostics_dir / "walking_before.json"),
                "--action", "Walking",
                "--samples", "41",
                "--front-axis", "positive-x",
            ],
        ),
        "repair": blender_command(
            blender,
            REPAIR_SCRIPT,
            [
                "--input", str(bound_glb),
                "--output", str(repaired_glb),
                "--manifest", str(repair_dir / "weight_repair_manifest.json"),
                "--front-axis", "positive-x",
                "--walking-samples", "11",
                "--idle-samples", "5",
                "--maximum-passes", "12",
                "--inner-iterations", "6",
                "--extension-threshold", "0.008",
                "--minimum-stretch-ratio", "1.8",
                "--maximum-rest-edge-ratio", "0.04",
                "--blend", "0.9",
                "--repair-mode", "edge-average",
                "--top-k", "4",
                "--maximum-seed-edges", "8192",
            ],
        ),
        "diagnostic_after": blender_command(
            blender,
            DIAGNOSTIC_SCRIPT,
            [
                "--input", str(repaired_glb),
                "--output", str(diagnostics_dir / "walking_after.json"),
                "--action", "Walking",
                "--samples", "41",
                "--front-axis", "positive-x",
            ],
        ),
    }
    for action in ("Walking", "Idle"):
        name = action.lower()
        commands[f"render_{name}"] = blender_command(
            blender,
            RENDER_SCRIPT,
            [
                "--input", str(repaired_glb),
                "--action", action,
                "--output-dir", str(review_dir / f"{name}_frames"),
                "--n-frames", "24",
                "--width", "960",
                "--height", "720",
                "--samples", "8",
                "--view", "side",
                "--ground-plane",
            ],
        )
    return commands


def run_command(command: Sequence[str], *, label: str) -> float:
    print(f"[{label}] {shlex.join(map(str, command))}", flush=True)
    started = time.perf_counter()
    subprocess.run(list(map(str, command)), cwd=SPEAR_ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(f"[{label}] completed elapsed_seconds={elapsed:.3f}", flush=True)
    return elapsed


def diagnostic_max(path: Path) -> float:
    value = load_json(path).get("worst_edge_summary", {}).get(
        "maximum_extension_ratio_of_rest_diagonal"
    )
    if not isinstance(value, (int, float)):
        raise StabilizationError(f"diagnostic maximum is missing: {path}")
    return float(value)


def encode_video(frames: Path, output: Path) -> float:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", "12", "-i", "frame_%04d.png",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    print(f"[encode] {shlex.join(command)}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=frames, check=True)
    return time.perf_counter() - started


def verify_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_name,width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    probe = json.loads(result.stdout)
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise StabilizationError(f"video stream count changed: {path}")
    stream = streams[0]
    duration = float(probe.get("format", {}).get("duration", 0.0))
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", 0)) != 960
        or int(stream.get("height", 0)) != 720
        or stream.get("avg_frame_rate") != "12/1"
        or int(stream.get("nb_frames", 0)) != 24
        or abs(duration - 2.0) > 1.0e-6
    ):
        raise StabilizationError(f"video readback contract failed: {path}: {probe}")
    return probe


def run_job(
    job: Mapping[str, Any],
    output_root: Path,
    *,
    blender: Path,
    target_faces: int,
    max_extension: float,
    render: bool,
) -> dict[str, Any]:
    asset_id = str(job["asset_id"])
    job_root = output_root / asset_id
    job_root.mkdir(parents=True, exist_ok=False)
    commands = build_commands(
        job, job_root, blender=blender, target_faces=target_faces
    )
    timings: dict[str, float] = {}
    for stage in (
        "runtime", "binding", "diagnostic_before", "repair", "diagnostic_after"
    ):
        timings[stage] = run_command(commands[stage], label=f"{asset_id}:{stage}")

    diagnostics = job_root / "diagnostics"
    repair = job_root / "repair"
    before = diagnostic_max(diagnostics / "walking_before.json")
    after = diagnostic_max(diagnostics / "walking_after.json")
    repair_manifest = load_json(repair / "weight_repair_manifest.json")
    authority = repair_manifest.get("authority_contract", {})
    if (
        authority.get("native_mesh_topology_preserved") is not True
        or authority.get("pbr_material_preserved") is not True
        or authority.get("fitted_skeleton_rest_matrices_preserved") is not True
        or authority.get("approved_animation_curves_preserved") is not True
        or authority.get("only_vertex_weights_modified_in_memory") is not True
    ):
        raise StabilizationError(f"repair authority contract failed: {asset_id}")
    if after > before + 1.0e-9 or after > max_extension:
        raise StabilizationError(
            f"deformation gate failed for {asset_id}: before={before}, "
            f"after={after}, limit={max_extension}"
        )

    media: dict[str, Any] = {}
    if render:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    run_command,
                    commands[f"render_{action}"],
                    label=f"{asset_id}:render_{action}",
                ): action
                for action in ("walking", "idle")
            }
            for future in as_completed(futures):
                action = futures[future]
                timings[f"render_{action}"] = future.result()
        review = job_root / "review"
        for action in ("walking", "idle"):
            frames = review / f"{action}_frames"
            output = review / f"{action}_side_pbr.mp4"
            timings[f"encode_{action}"] = encode_video(frames, output)
            probe = verify_video(output)
            media[action] = {**file_record(output), "ffprobe": probe}

    result = {
        "asset_id": asset_id,
        "status": "research_candidate_pending_human_visual_review",
        "formal_dataset_registration_authorized": False,
        "inputs": {
            "raw_pbr_glb": file_record(Path(job["raw_mesh"])),
            "approved_animated_rig_glb": file_record(Path(job["animated_rig"])),
            "motion_basis_decision": file_record(Path(job["decision_path"])),
        },
        "target_preprocess": {
            "flip_x": bool(job["flip_x"]),
            "rotate_z_deg": float(job["rotate_z"]),
            "automatic_orientation_inference_used": False,
            "provenance": job["preprocess_provenance"],
        },
        "deformation": {
            "before_max_edge_extension_ratio_of_rest_diagonal": before,
            "after_max_edge_extension_ratio_of_rest_diagonal": after,
            "gate_limit": max_extension,
            "improved": after <= before,
        },
        "outputs": {
            "runtime_glb": file_record(
                job_root / "runtime" / f"mesh_runtime_{target_faces}.glb"
            ),
            "bound_glb": file_record(
                job_root / "binding" / "animated_walk_idle_bound.glb"
            ),
            "repaired_glb": file_record(
                repair / "animated_walk_idle_repaired.glb"
            ),
            "repair_manifest": file_record(
                repair / "weight_repair_manifest.json"
            ),
            "walking_readback": file_record(
                diagnostics / "walking_after.json"
            ),
            "media": media,
        },
        "timings_seconds": timings,
        "commands": commands,
    }
    write_json(job_root / "stabilization_result.json", result)
    return result


def run_batch(
    jobs_manifest: Path,
    output_root: Path,
    *,
    blender: Path = DEFAULT_BLENDER,
    target_faces: int = DEFAULT_TARGET_FACES,
    max_extension: float = 0.04,
    workers: int = 2,
    render: bool = True,
    validate_only: bool = False,
    asset_ids: Sequence[str] = (),
) -> dict[str, Any]:
    source_manifest, jobs = load_jobs_manifest(jobs_manifest)
    requested = set(asset_ids)
    if requested:
        available = {job["asset_id"] for job in jobs}
        missing = requested - available
        if missing:
            raise StabilizationError(
                f"requested asset_id values are absent: {sorted(missing)}"
            )
        jobs = [job for job in jobs if job["asset_id"] in requested]
    blender = blender.resolve()
    if not validate_only and (not blender.is_file() or blender.is_symlink()):
        raise StabilizationError(f"Blender executable missing: {blender}")
    if target_faces < 50_000:
        raise StabilizationError("target_faces below 50k is not authorized")
    if not 0.0 < max_extension <= 0.08:
        raise StabilizationError("max_extension must be in (0, 0.08]")
    if workers <= 0:
        raise StabilizationError("workers must be positive")
    if validate_only:
        return {
            "schema": RESULT_SCHEMA,
            "status": "validated_only",
            "job_count": len(jobs),
            "asset_ids": [job["asset_id"] for job in jobs],
            "formal_dataset_registration_authorized": False,
        }

    output_root = output_root.resolve()
    if output_root.exists():
        raise StabilizationError(f"immutable output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    state_path = output_root / "batch_state.json"
    state = {
        "schema": RESULT_SCHEMA,
        "status": "running",
        "created_at": utc_now(),
        "jobs_manifest": file_record(jobs_manifest),
        "target_faces": target_faces,
        "max_extension": max_extension,
        "workers": workers,
        "render": render,
        "formal_dataset_registration_authorized": False,
    }
    write_json(state_path, state)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            future_jobs = {
                executor.submit(
                    run_job,
                    job,
                    output_root,
                    blender=blender,
                    target_faces=target_faces,
                    max_extension=max_extension,
                    render=render,
                ): job["asset_id"]
                for job in jobs
            }
            for future in as_completed(future_jobs):
                results.append(future.result())
    except Exception as error:
        state.update(
            {
                "status": "failed_preserved_as_evidence",
                "failed_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(state_path, state)
        raise

    results.sort(key=lambda item: item["asset_id"])
    manifest = {
        **state,
        "status": "research_candidate_batch_complete_pending_human_visual_review",
        "completed_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "source_manifest_schema": source_manifest["schema"],
        "asset_count": len(results),
        "assets": results,
    }
    write_json(output_root / "batch_manifest.json", manifest)
    state.update(
        {
            "status": "complete",
            "completed_at": manifest["completed_at"],
            "batch_manifest": file_record(output_root / "batch_manifest.json"),
        }
    )
    write_json(state_path, state)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--target-faces", type=int, default=DEFAULT_TARGET_FACES)
    parser.add_argument("--max-extension", type=float, default=0.04)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_batch(
        args.jobs_manifest,
        args.output_root,
        blender=args.blender,
        target_faces=args.target_faces,
        max_extension=args.max_extension,
        workers=args.workers,
        render=not args.skip_render,
        validate_only=args.validate_only,
        asset_ids=args.asset_id,
    )
    print(
        "GENERATED_QUADRUPED_STABILIZATION_OK "
        f"status={result['status']} assets={result.get('asset_count', result.get('job_count'))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
