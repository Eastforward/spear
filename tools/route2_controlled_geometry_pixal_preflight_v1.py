#!/usr/bin/env python3
"""Atomically execute Route-2 v3 controlled references with pinned Pixal3D."""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import human_attribute_pixal_contract as pixal_validation
from tools import route2_controlled_geometry_references_v3 as geometry
from tools import route2_human_contract_common as common
from tools import route2_human_instance_contract as route2_instance


SCHEMA = "route2_controlled_geometry_pixal_execution_contract_v1"
MANIFEST_SCHEMA = "route2_controlled_geometry_pixal_candidate_v1"
ATTEMPT_SCHEMA = "route2_controlled_geometry_pixal_attempt_v1"
RUNNER_PATH = Path(__file__).resolve()
SPEAR_ROOT = RUNNER_PATH.parents[1]
SOURCE_ROOT = SPEAR_ROOT / "tmp/route2_controlled_geometry_references_v3"
SOURCE_PIXAL_JOBS = SOURCE_ROOT / "review_summary_v1/pixal_jobs_v1.json"
SOURCE_PIXAL_JOBS_SHA256 = "82b4aefc9d575f22c1983e60ceac1338bababc99123b38536f67c857670ff45b"
OUTPUT_ROOT = SPEAR_ROOT / "tmp/i23d_controlled_geometry_v3/pixal3d"
PIXAL_WRAPPER = SPEAR_ROOT / "tools/i23d_human_bakeoff.py"
PIXAL_WRAPPER_SHA256 = "6291e42a4f3ca6957beba4e2cd5749c264347657c98b9e067b66c2b2012fc799"
PIXAL_PYTHON = Path("/data/jzy/miniconda3/envs/avengine-3dgen/bin/python3.10")
PIXAL_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
DINO_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
PARAMETERS = {"seed": 42, "manual_fov": 0.2, "resolution": 1024, "low_vram": True}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class ControlledPixalError(RuntimeError):
    """Raised when a controlled Pixal preflight cannot be authenticated."""


def sha256_file(path: Path) -> str:
    return common.sha256_file(Path(path))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _regular_file(path: Path, description: str, *, mode: int | None = None) -> Path:
    path = Path(path).absolute()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or not stat.S_ISREG(os.lstat(path).st_mode)
        or path.stat().st_size <= 0
    ):
        raise ControlledPixalError(f"{description} must be a direct nonempty file: {path}")
    if mode is not None and stat.S_IMODE(path.stat().st_mode) != mode:
        raise ControlledPixalError(f"{description} must have mode {mode:04o}")
    return path


def _record(path: Path, *, public_path: Path | None = None) -> dict[str, Any]:
    path = _regular_file(path, "artifact")
    return {
        "path": str(public_path if public_path is not None else path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _require_record(path: Path, value: Any, description: str) -> Path:
    path = _regular_file(path, description)
    expected = _record(path)
    if not isinstance(value, Mapping) or any(
        value.get(key) != expected[key] for key in ("path", "sha256", "size_bytes")
    ):
        raise ControlledPixalError(f"{description} descriptor changed")
    return path


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise ControlledPixalError("atomic no-replace publication requires renameat2")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(destination)
    raise OSError(number, os.strerror(number), destination)


def _readonly_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ControlledPixalError(f"output tree contains a symlink: {path}")
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        path.chmod(0o755)
    root.chmod(0o755)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(path: Path, description: str, *, mode: int | None = None) -> dict[str, Any]:
    path = _regular_file(path, description, mode=mode)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlledPixalError(f"{description} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ControlledPixalError(f"{description} must contain an object")
    return value


def _source_jobs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(SOURCE_PIXAL_JOBS, "controlled reference Pixal jobs", mode=0o444)
    if sha256_file(SOURCE_PIXAL_JOBS) != SOURCE_PIXAL_JOBS_SHA256:
        raise ControlledPixalError("controlled reference Pixal jobs SHA-256 changed")
    jobs = payload.get("jobs")
    expected_ids = [f"route2_v3_{case_id}" for case_id in geometry.CASE_BY_ID]
    if (
        payload.get("schema") != geometry.PIXAL_JOBS_SCHEMA
        or payload.get("state_classification") != "research_candidate_preflight"
        or payload.get("formal_registration_authorized") is not False
        or not isinstance(jobs, list)
        or [job.get("asset_id") for job in jobs if isinstance(job, Mapping)] != expected_ids
    ):
        raise ControlledPixalError("controlled reference Pixal jobs schema/order changed")
    validated = []
    for job in jobs:
        if not isinstance(job, dict):
            raise ControlledPixalError("controlled Pixal job is not an object")
        case_id = str(job["asset_id"])[len("route2_v3_") :]
        candidate, decision = geometry._load_decision(case_id)
        expected_output = OUTPUT_ROOT / str(job["asset_id"])
        if (
            decision.get("status") != "agent_2d_passed"
            or decision.get("pixal_authorized") is not True
            or job.get("base_asset_id") != candidate.get("base_asset_id")
            or job.get("geometry_attribute") != candidate.get("geometry_attribute")
            or job.get("state_classification") != "research_candidate"
            or job.get("input_rgba") != candidate.get("artifacts", {}).get("candidate_rgba.png")
            or job.get("reference_manifest")
            != _record(SOURCE_ROOT / "cases" / case_id / "candidate_manifest.json")
            or job.get("reference_decision")
            != _record(SOURCE_ROOT / "cases" / case_id / "agent_2d_visual_qa.json")
            or job.get("model") != {"name": "TencentARC/Pixal3D", "revision": PIXAL_REVISION}
            or job.get("parameters") != PARAMETERS
            or Path(str(job.get("output_dir"))).absolute() != expected_output
            or job.get("execution_status")
            != "ready_for_pixal_preflight_not_formal_registration"
        ):
            raise ControlledPixalError(f"controlled Pixal job lineage changed: {case_id}")
        rgba = Path(job["input_rgba"]["path"])
        _require_record(rgba, job["input_rgba"], f"{case_id} accepted RGBA")
        validated.append(dict(job))
    return payload, validated


def _model_evidence() -> dict[str, Any]:
    try:
        return {
            "pixal": route2_instance.model_snapshot_evidence(PIXAL_REVISION),
            "dino": route2_instance.model_snapshot_evidence(DINO_REVISION),
        }
    except route2_instance.InstanceContractError as error:
        raise ControlledPixalError(f"Pixal/DINO model authentication failed: {error}") from error


def _runtime_environment(gpu: str) -> dict[str, str]:
    if gpu not in {"0", "1", "2", "3"}:
        raise ControlledPixalError("GPU must be one of 0,1,2,3")
    environment = dict(os.environ)
    environment.update(
        {
            "ATTN_BACKEND": "sdpa",
            "CUDA_VISIBLE_DEVICES": gpu,
            "HF_HUB_CACHE": "/data/models/hub",
            "HF_HUB_OFFLINE": "1",
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_HOME": "/data/models/torch",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


RUNTIME_PROBE = """
import json, platform, torch
import o_voxel
print(json.dumps({
  'python_version': platform.python_version(),
  'cuda_available': torch.cuda.is_available(),
  'logical_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
  'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'torch': torch.__version__,
  'o_voxel_imported': True,
}, sort_keys=True))
""".strip()


def probe_runtime(gpu: str) -> dict[str, Any]:
    _regular_file(PIXAL_PYTHON, "Pixal Python executable")
    result = subprocess.run(
        [str(PIXAL_PYTHON), "-c", RUNTIME_PROBE],
        env=_runtime_environment(gpu),
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError as error:
        raise ControlledPixalError(f"Pixal runtime probe returned invalid JSON: {result.stderr}") from error
    if (
        result.returncode != 0
        or payload.get("cuda_available") is not True
        or payload.get("logical_device") != 0
        or payload.get("o_voxel_imported") is not True
    ):
        raise ControlledPixalError(f"Pixal runtime probe failed: {payload} {result.stderr}")
    return {
        **payload,
        "physical_gpu": gpu,
        "python": _record(PIXAL_PYTHON),
        "environment": {
            key: _runtime_environment(gpu)[key]
            for key in (
                "ATTN_BACKEND",
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_CACHE",
                "HF_HUB_OFFLINE",
                "OPENCV_IO_ENABLE_OPENEXR",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TORCH_HOME",
                "TRANSFORMERS_OFFLINE",
            )
        },
    }


def prepare() -> Path:
    output = OUTPUT_ROOT.absolute()
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or output.parent.resolve() != output.parent:
        raise ControlledPixalError("controlled Pixal parent must be a direct real directory")
    _, jobs = _source_jobs()
    if sha256_file(PIXAL_WRAPPER) != PIXAL_WRAPPER_SHA256:
        raise ControlledPixalError("pinned Pixal wrapper SHA-256 changed")
    _regular_file(PIXAL_PYTHON, "pinned Pixal Python")
    payload = {
        "schema": SCHEMA,
        "state_classification": "research_candidate_preflight",
        "formal_registration_authorized": False,
        "output_root": str(output),
        "source_pixal_jobs": _record(SOURCE_PIXAL_JOBS),
        "geometry_runner": _record(geometry.RUNNER_PATH),
        "runner": _record(RUNNER_PATH),
        "pixal_wrapper": _record(PIXAL_WRAPPER),
        "pixal_python": _record(PIXAL_PYTHON),
        "models": _model_evidence(),
        "parameters": PARAMETERS,
        "jobs": jobs,
        "created_at_utc": _utc_now(),
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    try:
        (staging / ".attempts").mkdir()
        (staging / ".failed_attempts").mkdir()
        (staging / "execution_contract_v1.json").write_bytes(_json_bytes(payload))
        _readonly_tree(staging)
        _fsync_tree(staging)
        _rename_noreplace(staging, output)
        return output / "execution_contract_v1.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_contract() -> dict[str, Any]:
    path = OUTPUT_ROOT / "execution_contract_v1.json"
    payload = _load_json(path, "controlled Pixal execution contract", mode=0o444)
    _, jobs = _source_jobs()
    if (
        payload.get("schema") != SCHEMA
        or payload.get("state_classification") != "research_candidate_preflight"
        or payload.get("formal_registration_authorized") is not False
        or payload.get("output_root") != str(OUTPUT_ROOT)
        or payload.get("source_pixal_jobs") != _record(SOURCE_PIXAL_JOBS)
        or payload.get("geometry_runner") != _record(geometry.RUNNER_PATH)
        or payload.get("runner") != _record(RUNNER_PATH)
        or payload.get("pixal_wrapper") != _record(PIXAL_WRAPPER)
        or payload.get("pixal_python") != _record(PIXAL_PYTHON)
        or payload.get("parameters") != PARAMETERS
        or payload.get("jobs") != jobs
    ):
        raise ControlledPixalError("controlled Pixal execution contract changed")
    current_models = _model_evidence()
    recorded = payload.get("models")
    if not isinstance(recorded, Mapping) or any(
        recorded.get(name, {}).get(key) != current_models[name].get(key)
        for name in ("pixal", "dino")
        for key in ("path", "revision", "file_count", "inventory_sha256", "license")
    ):
        raise ControlledPixalError("controlled Pixal/DINO model snapshot changed")
    return payload


def _job(contract: Mapping[str, Any], asset_id: str) -> dict[str, Any]:
    matches = [job for job in contract["jobs"] if job["asset_id"] == asset_id]
    if len(matches) != 1:
        raise ControlledPixalError(f"asset is missing/duplicated in execution contract: {asset_id}")
    return dict(matches[0])


def _command(job: Mapping[str, Any], staged_glb: Path, gpu: str) -> list[str]:
    return [
        str(PIXAL_PYTHON),
        str(PIXAL_WRAPPER),
        "--backend",
        "pixal3d",
        "--image",
        str(Path(job["input_rgba"]["path"])),
        "--output",
        str(staged_glb),
        "--gpu",
        gpu,
        "--seed",
        "42",
        "--resolution",
        "1024",
        "--manual-fov",
        "0.2",
        "--low-vram",
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, _json_bytes(payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _failure(job: Mapping[str, Any], staging: Path, stage: str, error: BaseException) -> Path:
    root = OUTPUT_ROOT / ".failed_attempts" / str(job["asset_id"])
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"attempt_{uuid.uuid4().hex}"
    _write_json(
        staging / "failure.json",
        {
            "schema": "route2_controlled_geometry_pixal_failure_v1",
            "asset_id": job["asset_id"],
            "base_asset_id": job["base_asset_id"],
            "state_classification": "rejected",
            "failure_stage": stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "recorded_at_utc": _utc_now(),
        },
    )
    _readonly_tree(staging)
    _fsync_tree(staging)
    _rename_noreplace(staging, destination)
    return destination


def run(asset_id: str, gpu: str) -> Path:
    before = load_contract()
    job = _job(before, asset_id)
    public_root = Path(job["output_dir"]).absolute()
    if public_root != OUTPUT_ROOT / asset_id or os.path.lexists(public_root):
        raise ControlledPixalError("controlled Pixal output is not an unused canonical root")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{asset_id}.attempt_", suffix=".staging", dir=OUTPUT_ROOT)
    )
    stage = "runtime_probe"
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        runtime = probe_runtime(gpu)
        staged_glb = staging / "canary_1024_seed42.glb"
        staged_manifest = staging / "canary_1024_seed42.manifest.json"
        command = _command(job, staged_glb, gpu)
        started = _utc_now()
        stage = "pixal_subprocess"
        completed = subprocess.run(
            command,
            cwd=str(SPEAR_ROOT),
            env=_runtime_environment(gpu),
            check=False,
            capture_output=True,
            text=True,
        )
        _write_json(
            staging / "execution_log.json",
            {
                "schema": "route2_controlled_geometry_pixal_execution_log_v1",
                "asset_id": asset_id,
                "returncode": completed.returncode,
                "command": command,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
        if completed.returncode != 0:
            raise ControlledPixalError(f"Pixal subprocess returned {completed.returncode}")
        if not staged_glb.is_file() or not staged_manifest.is_file():
            raise ControlledPixalError("Pixal subprocess did not create GLB and manifest")
        stage = "pbr_glb_readback"
        glb_document, glb_file = pixal_validation.validate_staged_pixal_glb(
            staged_glb,
            staging=staging,
            input_rgba=Path(job["input_rgba"]["path"]),
        )
        position_counts = []
        primitive_count = 0
        for mesh in glb_document["meshes"]:
            for primitive in mesh["primitives"]:
                primitive_count += 1
                position_counts.append(
                    int(
                        glb_document["accessors"][
                            primitive["attributes"]["POSITION"]
                        ]["count"]
                    )
                )
        pbr_validation = {
            "passed": True,
            "mesh_count": len(glb_document["meshes"]),
            "primitive_count": primitive_count,
            "position_accessor_counts": position_counts,
            "material_count": len(glb_document["materials"]),
            "texture_count": len(glb_document["textures"]),
            "image_count": len(glb_document["images"]),
            "packed_pbr": True,
        }
        generated = _load_json(staged_manifest, "generated Pixal manifest")
        if (
            generated.get("backend") != "pixal3d"
            or generated.get("input", {}).get("path") != job["input_rgba"]["path"]
            or generated.get("input", {}).get("sha256") != job["input_rgba"]["sha256"]
            or generated.get("output", {}).get("path") != str(staged_glb)
            or generated.get("output", {}).get("sha256") != glb_file["sha256"]
            or generated.get("output", {}).get("bytes") != glb_file["size_bytes"]
            or generated.get("model", {}).get("revision") != PIXAL_REVISION
            or generated.get("dino", {}).get("revision") != DINO_REVISION
            or generated.get("parameters") != PARAMETERS
        ):
            raise ControlledPixalError("generated Pixal manifest readback changed")
        stage = "postflight_reauthentication"
        after = load_contract()
        if common.canonical_json(after) != common.canonical_json(before):
            raise ControlledPixalError("controlled Pixal contract changed during inference")
        final_manifest = {
            "schema": MANIFEST_SCHEMA,
            "backend": "pixal3d",
            "asset_id": asset_id,
            "base_asset_id": job["base_asset_id"],
            "geometry_attribute": job["geometry_attribute"],
            "state_classification": "research_candidate",
            "formal_registration_authorized": False,
            "source_pixal_job": job,
            "source_pixal_jobs_manifest": _record(SOURCE_PIXAL_JOBS),
            "input_rgba": job["input_rgba"],
            "reference_manifest": job["reference_manifest"],
            "reference_decision": job["reference_decision"],
            "model": before["models"]["pixal"],
            "dino": before["models"]["dino"],
            "parameters": PARAMETERS,
            "runtime": runtime,
            "runner": _record(RUNNER_PATH),
            "wrapper": _record(PIXAL_WRAPPER),
            "command": command,
            "output": {
                "path": str(public_root / staged_glb.name),
                "sha256": glb_file["sha256"],
                "size_bytes": glb_file["size_bytes"],
            },
            "pbr_glb_readback": pbr_validation,
            "created_at_utc": _utc_now(),
        }
        staged_manifest.unlink()
        _write_json(staged_manifest, final_manifest)
        attempt = {
            "schema": ATTEMPT_SCHEMA,
            "asset_id": asset_id,
            "status": "succeeded_pending_static_multiview_qa",
            "state_classification": "research_candidate",
            "formal_registration_authorized": False,
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "physical_gpu": gpu,
            "command": command,
            "returncode": completed.returncode,
            "execution_contract": _record(OUTPUT_ROOT / "execution_contract_v1.json"),
            "manifest": _record(
                staged_manifest,
                public_path=public_root / staged_manifest.name,
            ),
            "glb": {
                "path": str(public_root / staged_glb.name),
                "sha256": glb_file["sha256"],
                "size_bytes": glb_file["size_bytes"],
            },
            "static_multiview_qa": "pending",
        }
        _write_json(staging / "pixal_attempt.json", attempt)
        _readonly_tree(staging)
        _fsync_tree(staging)
        stage = "atomic_publication"
        _rename_noreplace(staging, public_root)
        return public_root / "pixal_attempt.json"
    except BaseException as error:
        if staging.exists():
            evidence = _failure(job, staging, stage, error)
            raise ControlledPixalError(
                f"controlled Pixal rejected at {stage}; evidence={evidence}"
            ) from error
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--asset-id",
        choices=tuple(f"route2_v3_{case_id}" for case_id in geometry.CASE_BY_ID),
        required=True,
    )
    run_parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        print(f"ROUTE2_CONTROLLED_PIXAL_PREPARED {prepare()}")
    elif args.command == "run":
        print(f"ROUTE2_CONTROLLED_PIXAL_PUBLISHED {run(args.asset_id, args.gpu)}")
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
