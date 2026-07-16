"""Reload Rocketbox UE assets in isolated editor processes.

UE 5.5 can stall when many skeletal meshes are reloaded serially from one
Python commandlet.  This host-side runner keeps the stronger second-process
contract while isolating each avatar in its own editor process.  Workers read
shared Content only; every process writes a unique manifest and absolute log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UE_EDITOR = Path("/data/UE_5.5/Engine/Binaries/Linux/UnrealEditor")
DEFAULT_PROJECT = SPEAR_ROOT / "cpp/unreal_projects/SpearSim/SpearSim.uproject"
DEFAULT_GATE_SCRIPT = SPEAR_ROOT / "tools/import_gate_rocketbox_native_editor.py"


@dataclass(frozen=True)
class VerificationJob:
    base_avatar_id: str
    tag: str
    source_glb: Path
    source_manifest: Path
    ue_manifest: Path
    log_path: Path
    environment: dict[str, str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_inventory(path: Path) -> dict:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Rocketbox inventory is not a direct file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "rocketbox_human_inventory_v1"
        or payload.get("population", {}).get("total") != 115
        or payload.get("automatic_checks", {}).get("overall") != "passed"
        or not isinstance(payload.get("avatars"), list)
    ):
        raise RuntimeError("Rocketbox inventory is not UE verification ready")
    return payload


def build_jobs(
    inventory_path: Path,
    normalized_root: Path,
    manifest_root: Path,
    log_root: Path,
) -> list[VerificationJob]:
    inventory_path = inventory_path.resolve()
    normalized_root = normalized_root.resolve()
    manifest_root = manifest_root.resolve()
    log_root = log_root.resolve()
    inventory = _load_inventory(inventory_path)
    jobs = []
    for avatar in sorted(
        inventory["avatars"], key=lambda item: item["base_avatar_id"]
    ):
        avatar_id = avatar["base_avatar_id"]
        if avatar.get("inventory_status") != "passed":
            raise RuntimeError(f"inventory avatar is not ready: {avatar_id}")
        tag = f"{avatar_id}_original_ue_v1"
        source_root = normalized_root / tag
        source_glb = source_root / "runtime.glb"
        source_manifest = source_root / "normalization_manifest.json"
        ue_manifest = manifest_root / tag / "ue_import_manifest.json"
        for path, description in (
            (source_glb, "normalized GLB"),
            (source_manifest, "normalization manifest"),
            (ue_manifest, "UE import manifest"),
        ):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"missing direct {description}: {path}")
        log_path = log_root / f"{tag}.log"
        environment = {
            "ROCKETBOX_NATIVE_ENABLE_DYNAMIC_BATCH": "1",
            "ROCKETBOX_NATIVE_BATCH_NORMALIZED_ROOT": str(normalized_root),
            "ROCKETBOX_NATIVE_BATCH_UE_MANIFEST_ROOT": str(manifest_root),
            "ROCKETBOX_NATIVE_INVENTORY_JSON": str(inventory_path),
            "ROCKETBOX_NATIVE_TAG": tag,
            "ROCKETBOX_NATIVE_GLB": str(source_glb),
            "ROCKETBOX_NATIVE_SOURCE_MANIFEST": str(source_manifest),
            "ROCKETBOX_NATIVE_UE_MANIFEST": str(ue_manifest),
            "ROCKETBOX_NATIVE_VERIFY_ONLY": "1",
        }
        jobs.append(
            VerificationJob(
                base_avatar_id=avatar_id,
                tag=tag,
                source_glb=source_glb,
                source_manifest=source_manifest,
                ue_manifest=ue_manifest,
                log_path=log_path,
                environment=environment,
            )
        )
    return jobs


def build_command(
    job: VerificationJob,
    *,
    ue_editor: Path,
    project: Path,
    gate_script: Path,
) -> list[str]:
    return [
        str(ue_editor),
        str(project),
        "-RenderOffscreen",
        "-NoAssetRegistryCacheWrite",
        f"-AbsLog={job.log_path}",
        "-unattended",
        "-nop4",
        "-nosplash",
        "-NoSound",
        "-run=pythonscript",
        f"-script={gate_script}",
    ]


def validate_completed_job(job: VerificationJob) -> dict:
    payload = json.loads(job.ue_manifest.read_text(encoding="utf-8"))
    bounds = payload.get("runtime_contract", {}).get("bounds", {})
    if (
        payload.get("schema") != "rocketbox_batch_native_ue_import_v1"
        or payload.get("base_avatar_id") != job.base_avatar_id
        or payload.get("tag") != job.tag
        or payload.get("reload_verification", {}).get("status") != "passed"
        or payload.get("runtime_contract", {}).get("bone_count") != 80
        or payload.get("runtime_contract", {}).get("actor_scale") != 1
        or bounds.get("height_passed") is not True
        or bounds.get("authored_height_preserved") is not True
        or bounds.get("ground_passed") is not True
    ):
        raise RuntimeError(f"reload contract failed for {job.base_avatar_id}")
    return payload


def select_jobs_for_resume(
    jobs: list[VerificationJob],
    failed_avatar_ids: set[str] | None = None,
) -> list[VerificationJob]:
    failed_avatar_ids = failed_avatar_ids or set()
    selected = []
    for job in jobs:
        if job.base_avatar_id in failed_avatar_ids:
            selected.append(job)
            continue
        try:
            payload = json.loads(job.ue_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            selected.append(job)
            continue
        if payload.get("reload_verification", {}).get("status") != "passed":
            selected.append(job)
    return selected


def run_command_with_timeout(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    console_log: Path,
    timeout_seconds: float,
) -> int:
    """Run one command in an isolated process group and bound all descendants."""
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with console_log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise TimeoutError(
                f"command timed out after {timeout_seconds} seconds"
            ) from error


def aggregate_verification_results(
    jobs: list[VerificationJob],
    *,
    current_results: dict[str, dict],
    current_failures: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    aggregate_results = []
    aggregate_failures = []
    for job in jobs:
        if job.base_avatar_id in current_failures:
            aggregate_failures.append(current_failures[job.base_avatar_id])
            continue
        try:
            manifest = validate_completed_job(job)
        except BaseException as error:
            aggregate_failures.append(
                {
                    "base_avatar_id": job.base_avatar_id,
                    "tag": job.tag,
                    "error": str(error),
                    "ue_log": str(job.log_path),
                }
            )
            continue
        record = current_results.get(job.base_avatar_id)
        if record is None:
            bounds = manifest["runtime_contract"]["bounds"]
            record = {
                "base_avatar_id": job.base_avatar_id,
                "tag": job.tag,
                "status": "passed",
                "verification_reused": True,
                "ue_manifest": str(job.ue_manifest),
                "ue_manifest_sha256": _sha256(job.ue_manifest),
                "height_cm": bounds["height_cm"],
                "authored_height_cm": bounds.get("authored_height_cm"),
            }
        aggregate_results.append(record)
    aggregate_results.sort(key=lambda item: item["base_avatar_id"])
    aggregate_failures.sort(key=lambda item: item["base_avatar_id"])
    return aggregate_results, aggregate_failures


def _run_job(
    job: VerificationJob,
    *,
    ue_editor: Path,
    project: Path,
    gate_script: Path,
    timeout_seconds: float,
) -> dict:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        job,
        ue_editor=ue_editor,
        project=project,
        gate_script=gate_script,
    )
    environment = os.environ.copy()
    environment.update(job.environment)
    started = time.monotonic()
    console_log = job.log_path.with_suffix(".console.log")
    returncode = run_command_with_timeout(
        command,
        cwd=SPEAR_ROOT,
        environment=environment,
        console_log=console_log,
        timeout_seconds=timeout_seconds,
    )
    elapsed = time.monotonic() - started
    if returncode != 0:
        tail = "\n".join(
            console_log.read_text(errors="replace").splitlines()[-40:]
        )
        raise RuntimeError(
            f"UE reload returned {returncode} for "
            f"{job.base_avatar_id}:\n{tail}"
        )
    manifest = validate_completed_job(job)
    return {
        "base_avatar_id": job.base_avatar_id,
        "tag": job.tag,
        "status": "passed",
        "elapsed_seconds": elapsed,
        "ue_manifest": str(job.ue_manifest),
        "ue_manifest_sha256": _sha256(job.ue_manifest),
        "ue_log": str(job.log_path),
        "ue_log_sha256": _sha256(job.log_path),
        "console_log": str(console_log),
        "console_log_sha256": _sha256(console_log),
        "height_cm": manifest["runtime_contract"]["bounds"]["height_cm"],
        "authored_height_cm": manifest["runtime_contract"]["bounds"].get(
            "authored_height_cm"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--ue-editor", type=Path, default=DEFAULT_UE_EDITOR)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--gate-script", type=Path, default=DEFAULT_GATE_SCRIPT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise RuntimeError("workers must be positive")
    for path, description in (
        (args.ue_editor.resolve(), "UnrealEditor"),
        (args.project.resolve(), "SpearSim project"),
        (args.gate_script.resolve(), "Rocketbox gate script"),
    ):
        if not path.is_file():
            raise RuntimeError(f"missing {description}: {path}")
    jobs = build_jobs(
        args.inventory,
        args.normalized_root,
        args.manifest_root,
        args.log_root,
    )
    if len(jobs) != 115:
        raise RuntimeError(f"expected 115 Rocketbox verification jobs, got {len(jobs)}")

    status_path = args.manifest_root.resolve() / "batch_process_verify_status.json"
    previous_failed_avatar_ids = set()
    if args.resume and status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        previous_failed = previous.get("automatic_checks", {}).get(
            "failed_avatar_ids", []
        )
        if not isinstance(previous_failed, list):
            raise RuntimeError("previous process verification failure list is invalid")
        previous_failed_avatar_ids = set(previous_failed)
    selected_jobs = (
        select_jobs_for_resume(jobs, previous_failed_avatar_ids)
        if args.resume
        else jobs
    )

    results = []
    failures = []
    started_at = _utc_now()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_job,
                job,
                ue_editor=args.ue_editor.resolve(),
                project=args.project.resolve(),
                gate_script=args.gate_script.resolve(),
                timeout_seconds=args.timeout_seconds,
            ): job
            for job in selected_jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"ROCKETBOX_UE_PROCESS_VERIFY_OK "
                    f"{len(results) + len(failures)}/{len(selected_jobs)} "
                    f"{job.base_avatar_id}",
                    flush=True,
                )
            except BaseException as error:
                failures.append(
                    {
                        "base_avatar_id": job.base_avatar_id,
                        "tag": job.tag,
                        "error": str(error),
                        "ue_log": str(job.log_path),
                    }
                )
                print(
                    f"ROCKETBOX_UE_PROCESS_VERIFY_FAILED "
                    f"{job.base_avatar_id}: {error}",
                    flush=True,
                )

    current_results = {item["base_avatar_id"]: item for item in results}
    current_failures = {item["base_avatar_id"]: item for item in failures}
    aggregate_results, aggregate_failures = aggregate_verification_results(
        jobs,
        current_results=current_results,
        current_failures=current_failures,
    )
    status = {
        "schema_version": "rocketbox_batch_ue_process_verify_v1",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "worker_count": args.workers,
        "job_count": len(jobs),
        "selected_job_count": len(selected_jobs),
        "resume": args.resume,
        "passed_count": len(aggregate_results),
        "failed_count": len(aggregate_failures),
        "process_isolation": "one_fresh_ue_editor_process_per_avatar",
        "shared_content_access": "read_only",
        "results": aggregate_results,
        "failures": aggregate_failures,
        "automatic_checks": {
            "overall": "passed" if not aggregate_failures else "failed",
            "failed_avatar_ids": [
                item["base_avatar_id"] for item in aggregate_failures
            ],
        },
    }
    _atomic_json(status_path, status)
    if aggregate_failures:
        raise RuntimeError(
            f"Rocketbox UE process verification has "
            f"{len(aggregate_failures)} failures"
        )
    print(
        f"ROCKETBOX_UE_PROCESS_VERIFY_ALL_OK total={len(aggregate_results)} "
        f"workers={args.workers}",
        flush=True,
    )


if __name__ == "__main__":
    main()
