#!/usr/bin/env python3

"""Run the one-avatar Blender builder across the authenticated inventory."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
BUILDER = SPEAR_ROOT / "tools/blender_build_rocketbox_batch_runtime.py"
DEFAULT_BLENDER = Path("/data/jzy/.local/bin/blender")
OUTPUT_SCHEMA = "rocketbox_batch_native_runtime_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_is_verified(output: Path, base_avatar_id: str) -> bool:
    manifest_path = Path(output) / "build_manifest.json"
    runtime_path = Path(output) / "runtime.glb"
    if not manifest_path.is_file() or not runtime_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    runtime = manifest.get("runtime_glb", {})
    return (
        manifest.get("schema") == OUTPUT_SCHEMA
        and manifest.get("base_avatar_id") == base_avatar_id
        and manifest.get("automatic_checks", {}).get("overall") == "passed"
        and runtime.get("filename") == "runtime.glb"
        and runtime.get("size_bytes") == runtime_path.stat().st_size
        and runtime.get("sha256") == sha256_file(runtime_path)
    )


def plan_jobs(inventory: dict, output_root: Path) -> tuple[list[dict], list[str]]:
    if (
        inventory.get("schema_version") != "rocketbox_human_inventory_v1"
        or inventory.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise RuntimeError("inventory is not batch-ready")
    output_root = Path(output_root)
    jobs = []
    skipped = []
    records = sorted(inventory.get("avatars", []), key=lambda item: item["base_avatar_id"])
    for avatar in records:
        avatar_id = avatar["base_avatar_id"]
        if avatar.get("inventory_status") != "passed":
            raise RuntimeError(f"inventory avatar failed: {avatar_id}")
        output = output_root / f"{avatar_id}_original_v1"
        if output.exists() or output.is_symlink():
            if output_is_verified(output, avatar_id):
                skipped.append(avatar_id)
                continue
            raise RuntimeError(f"unverified existing output blocks no-replace job: {output}")
        jobs.append({"base_avatar_id": avatar_id, "output": output})
    return jobs, skipped


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run_job(job: dict, *, blender: Path, inventory_path: Path, log_root: Path) -> dict:
    avatar_id = job["base_avatar_id"]
    command = [
        str(blender),
        "--background",
        "--python",
        str(BUILDER),
        "--",
        "--inventory-json",
        str(inventory_path),
        "--base-avatar-id",
        avatar_id,
        "--output-dir",
        str(job["output"]),
    ]
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=SPEAR_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    finished = datetime.now(timezone.utc)
    log_path = log_root / f"{avatar_id}.log"
    _atomic_text(log_path, completed.stdout)
    verified = output_is_verified(job["output"], avatar_id)
    marker = f"ROCKETBOX_BATCH_RUNTIME_OK base_avatar_id={avatar_id}" in completed.stdout
    status = "passed" if verified and marker else "failed"
    return {
        "base_avatar_id": avatar_id,
        "status": status,
        "subprocess_returncode": completed.returncode,
        "success_marker_present": marker,
        "output_verified": verified,
        "output": str(job["output"]),
        "log": str(log_path),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.jobs <= 0 or args.jobs > 32:
        raise RuntimeError("--jobs must be between 1 and 32")
    inventory_path = args.inventory_json.resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs, skipped = plan_jobs(inventory, output_root)
    log_root = output_root / "_logs"
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_id = {
            executor.submit(
                _run_job,
                job,
                blender=args.blender.resolve(),
                inventory_path=inventory_path,
                log_root=log_root,
            ): job["base_avatar_id"]
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_id):
            result = future.result()
            results.append(result)
            print(
                f"ROCKETBOX_BATCH_JOB {result['status']} "
                f"{result['base_avatar_id']} {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    results.sort(key=lambda item: item["base_avatar_id"])
    failed = [item for item in results if item["status"] != "passed"]
    status = {
        "schema_version": "rocketbox_batch_native_runtime_status_v1",
        "inventory": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "output_root": str(output_root),
        "requested_job_count": len(jobs),
        "skipped_verified_count": len(skipped),
        "skipped_verified_avatar_ids": skipped,
        "passed_count": sum(item["status"] == "passed" for item in results),
        "failed_count": len(failed),
        "results": results,
        "automatic_checks": {
            "overall": "passed" if not failed else "failed",
            "failed_avatar_ids": [item["base_avatar_id"] for item in failed],
        },
    }
    _atomic_json(output_root / "batch_status.json", status)
    if failed:
        raise RuntimeError(f"{len(failed)} Rocketbox batch runtimes failed")
    print(
        f"ROCKETBOX_BATCH_ALL_OK total={len(results) + len(skipped)} "
        f"built={len(results)} skipped={len(skipped)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
