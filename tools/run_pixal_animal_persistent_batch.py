"""Run resumable Pixal animal generation with one persistent model per GPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    from .run_pixal_animal_replacement_batch import (
        DEFAULT_OUT_ROOT,
        _atomic_json,
        build_jobs,
        job_is_complete,
    )
except ImportError:
    from run_pixal_animal_replacement_batch import (
        DEFAULT_OUT_ROOT,
        _atomic_json,
        build_jobs,
        job_is_complete,
    )


SPEAR_ROOT = Path(__file__).resolve().parents[1]
WORKER = SPEAR_ROOT / "tools/pixal_animal_persistent_worker.py"
PYTHON = Path("/data/jzy/miniconda3/envs/avengine-3dgen/bin/python3.10")


def partition_jobs(jobs, gpus):
    partitions = {int(gpu): [] for gpu in gpus}
    for index, job in enumerate(jobs):
        partitions[int(gpus[index % len(gpus)])].append(job)
    return partitions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--gpu", type=int, action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gpus = args.gpu or [0, 1, 2, 3]
    if not 1 <= len(gpus) <= 4 or len(set(gpus)) != len(gpus):
        raise ValueError("provide one to four unique GPUs")
    out_root = args.out_root.resolve()
    jobs = build_jobs(out_root, set(args.tag) if args.tag else None)
    selected = [job for job in jobs if not (args.resume and job_is_complete(job))]
    partitions = {
        gpu: bucket for gpu, bucket in partition_jobs(selected, gpus).items() if bucket
    }
    started_at = datetime.now(timezone.utc).isoformat()

    def run_partition(gpu, bucket):
        worker_dir = out_root / "persistent_workers"
        jobs_path = worker_dir / f"gpu_{gpu}_jobs.json"
        status_path = worker_dir / f"gpu_{gpu}_status.json"
        _atomic_json(jobs_path, bucket)
        log_path = worker_dir / f"gpu_{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            result = subprocess.run(
                [
                    str(PYTHON),
                    str(WORKER),
                    "--jobs", str(jobs_path),
                    "--gpu", str(gpu),
                    "--status", str(status_path),
                ],
                cwd=SPEAR_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=10800,
            )
        return {
            "gpu": gpu,
            "returncode": result.returncode,
            "jobs": str(jobs_path),
            "status": str(status_path),
            "log": str(log_path),
        }

    workers = []
    with ThreadPoolExecutor(max_workers=len(partitions) or 1) as executor:
        futures = {
            executor.submit(run_partition, gpu, bucket): gpu
            for gpu, bucket in partitions.items()
        }
        for future in as_completed(futures):
            result = future.result()
            workers.append(result)
            print(
                f"PIXAL_PERSISTENT_WORKER_DONE gpu={result['gpu']} "
                f"returncode={result['returncode']}",
                flush=True,
            )

    completed = [job for job in jobs if job_is_complete(job)]
    missing = [job for job in jobs if not job_is_complete(job)]
    status = {
        "schema": "pixal_animal_persistent_batch_v1",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "job_count": len(jobs),
        "selected_count": len(selected),
        "completed_count": len(completed),
        "missing_count": len(missing),
        "gpus": gpus,
        "workers": sorted(workers, key=lambda item: item["gpu"]),
        "completed_tags": sorted(job["legacy_tag"] for job in completed),
        "missing_tags": sorted(job["legacy_tag"] for job in missing),
    }
    _atomic_json(out_root / "persistent_batch_status.json", status)
    return 1 if missing or any(item["returncode"] for item in workers) else 0


if __name__ == "__main__":
    raise SystemExit(main())
