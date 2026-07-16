"""One-GPU Pixal3D worker that loads the model once and executes many jobs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = SPEAR_ROOT / "tools"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload):
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


def claim_id(legacy_tag: str) -> str:
    return hashlib.sha256(legacy_tag.encode("utf-8")).hexdigest()


def claim_job(claim_dir: Path, job, gpu: int):
    """Atomically claim one shared-queue job, or return None if already owned."""
    identifier = claim_id(job["legacy_tag"])
    path = Path(claim_dir) / f"{identifier}.claim.json"
    payload = {
        "schema": "pixal_dynamic_work_claim_v1",
        "claim_sha256": identifier,
        "legacy_tag": job["legacy_tag"],
        "gpu": int(gpu),
        "claimed_at": _utc_now(),
    }
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        return None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--claim-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.gpu < 0:
        raise ValueError("gpu must be non-negative")
    # This must happen before importing Pixal/Torch.
    os.environ.update(
        {
            "ATTN_BACKEND": "sdpa",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "HF_HUB_CACHE": "/data/models/hub",
            "HF_HUB_OFFLINE": "1",
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_HOME": "/data/models/torch",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    sys.path.insert(0, str(TOOLS_ROOT))
    from tools import controlled_animal_one_shot_policy as one_shot
    import i23d_human_bakeoff as wrapper

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("persistent worker jobs must be a non-empty list")
    for job in jobs:
        one_shot.validate_pixal_job(job)
    claim_dir = None
    if args.claim_dir is not None:
        if args.claim_dir.is_symlink() or not args.claim_dir.is_dir():
            raise ValueError("claim-dir must be an existing direct directory")
        claim_dir = args.claim_dir.resolve()
    status = {
        "schema": "pixal_animal_persistent_worker_v1",
        "gpu": args.gpu,
        "started_at": _utc_now(),
        "model_load_seconds": None,
        "scheduling_mode": (
            "shared_claim_queue_v1" if claim_dir is not None else "fixed_partition_v1"
        ),
        "jobs": [],
    }
    _atomic_json(args.status, status)

    def claimable_jobs():
        for job in jobs:
            claim_path = (
                claim_job(claim_dir, job, args.gpu)
                if claim_dir is not None
                else None
            )
            if claim_dir is None or claim_path is not None:
                yield job, claim_path

    selected_jobs = claimable_jobs()
    first = next(selected_jobs, None)
    if first is None:
        status.update(
            {
                "finished_at": _utc_now(),
                "passed_count": 0,
                "failed_count": 0,
            }
        )
        _atomic_json(args.status, status)
        return 0

    assets = wrapper.resolve_backend_assets("pixal3d")
    runtime = wrapper._import_pixal_runtime()
    wrapper.patch_pixal_conditioning(
        runtime.rembg_module,
        runtime.inference.IMAGE_COND_CONFIGS,
        assets["dino"],
    )
    load_started = time.perf_counter()
    pipeline = runtime.inference.init_pipeline(
        str(assets["model"]), low_vram=True
    )
    status["model_load_seconds"] = time.perf_counter() - load_started
    _atomic_json(args.status, status)

    def jobs_after_model_load():
        yield first
        yield from selected_jobs

    for job, claim_path in jobs_after_model_load():
        started = time.perf_counter()
        output = Path(job["output"]).resolve()
        manifest_path = Path(
            job.get("manifest", output.with_suffix(".manifest.json"))
        ).resolve()
        public_output = str(job.get("public_output", output))
        public_manifest = str(job.get("public_manifest", manifest_path))
        image = Path(job["reference"]["pixal_input"]["path"]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "legacy_tag": job["legacy_tag"],
            "candidate_tag": job["candidate_tag"],
            "seed": int(job["seed"]),
            "attempt_ordinal": int(job["attempt_ordinal"]),
            "started_at": _utc_now(),
        }
        if claim_path is not None:
            record["claim"] = str(claim_path)
        try:
            input_metadata = wrapper.inspect_rgba_input(image)
            runtime.inference.run_inference(
                image_path=str(image),
                output_path=str(output),
                seed=int(job["seed"]),
                manual_fov=0.2,
                model_path=str(assets["model"]),
                low_vram=True,
                resolution=1024,
                pipeline=pipeline,
            )
            if not output.is_file() or output.stat().st_size <= 0:
                raise RuntimeError("persistent Pixal inference produced no GLB")
            manifest = {
                "backend": "pixal3d",
                "input": {"path": str(image), **input_metadata},
                "output": {
                    "bytes": output.stat().st_size,
                    "path": public_output,
                    "sha256": _sha256(output),
                },
                "model": {
                    "snapshot": str(assets["model"]),
                    "revision": wrapper.MODEL_SPECS["pixal3d"]["revision"],
                },
                "dino": {
                    "snapshot": str(assets["dino"]),
                    "revision": wrapper.DINO_SPEC["revision"],
                },
                "parameters": {
                    "low_vram": True,
                    "manual_fov": 0.2,
                    "resolution": 1024,
                    "seed": int(job["seed"]),
                },
                "one_shot_execution": job["one_shot_execution"],
                "timings": {
                    "persistent_worker_model_load_seconds": status[
                        "model_load_seconds"
                    ],
                    "inference_and_export_seconds": time.perf_counter() - started,
                    "model_reused": True,
                },
            }
            if "controlled_request" in job:
                manifest["controlled_request"] = job["controlled_request"]
            _atomic_json(manifest_path, manifest)
            record.update(
                {
                    "status": "passed",
                    "finished_at": _utc_now(),
                    "wall_seconds": time.perf_counter() - started,
                    "output": public_output,
                    "manifest": public_manifest,
                    "output_sha256": manifest["output"]["sha256"],
                }
            )
        except BaseException as error:
            record.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "wall_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        status["jobs"].append(record)
        _atomic_json(args.status, status)
        gc.collect()
        runtime.inference.torch.cuda.empty_cache()
        print(
            f"PIXAL_PERSISTENT_JOB {job['legacy_tag']} {record['status']} "
            f"wall={record['wall_seconds']:.1f}s",
            flush=True,
        )

    status["finished_at"] = _utc_now()
    status["passed_count"] = sum(
        item["status"] == "passed" for item in status["jobs"]
    )
    status["failed_count"] = sum(
        item["status"] == "failed" for item in status["jobs"]
    )
    _atomic_json(args.status, status)
    return 1 if status["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
