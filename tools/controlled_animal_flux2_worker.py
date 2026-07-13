#!/usr/bin/env python3
"""One-GPU persistent FLUX.2 worker for controlled animal reference jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from PIL import Image


MODEL_ROOT = Path("/data/models/hub/models--black-forest-labs--FLUX.2-klein-4B")
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
PARTITION_SCHEMA = "avengine_controlled_animal_flux2_partition_v1"
WORKER_STATUS_SCHEMA = "avengine_controlled_animal_flux2_worker_v1"
CANDIDATE_SCHEMA = "avengine_controlled_animal_flux2_candidate_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _copy_no_replace(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to replace candidate artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.rename(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _save_png_no_replace(image: Image.Image, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to replace candidate artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            image.save(handle, format="PNG", optimize=False, compress_level=6)
            handle.flush()
            os.fsync(handle.fileno())
        with Image.open(temporary) as opened:
            opened.verify()
        os.rename(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_partition(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != PARTITION_SCHEMA:
        raise ValueError(f"partition schema must be {PARTITION_SCHEMA}")
    expected_hash = _json_sha256(
        {key: item for key, item in value.items() if key != "partition_sha256"}
    )
    if value.get("partition_sha256") != expected_hash:
        raise ValueError("partition hash mismatch")
    model = value.get("model")
    if model != {
        "name": "black-forest-labs/FLUX.2-klein-4B",
        "root": str(MODEL_ROOT),
        "revision": MODEL_REVISION,
        "local_files_only": True,
    }:
        raise ValueError("partition model pin changed")
    parameters = value.get("parameters")
    if parameters != {
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 28,
        "guidance_scale": 1.0,
        "max_sequence_length": 512,
        "output_mode": "rgb_pending_segmentation",
    }:
        raise ValueError("partition inference parameters changed")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("partition contains no jobs")
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("partition job must be an object")
        if job.get("execution_job_id") in seen:
            raise ValueError("partition contains duplicate execution_job_id")
        seen.add(job.get("execution_job_id"))
        if (
            job.get("generation_plan", {}).get("route")
            != "flux2_pixal3d_animal_v1"
            or job.get("generation_plan", {}).get("flux_invocations") != 1
            or job.get("generation_plan", {}).get("model_revisions", {}).get("flux2")
            != MODEL_REVISION
            or len(job.get("consumer_requests", [])) != 1
        ):
            raise ValueError("partition animal job contract changed")
    return value


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def run_worker(partition: Mapping[str, Any], gpu: int, output_root: Path, status_path: Path) -> int:
    partition = validate_partition(dict(partition))
    if gpu < 0:
        raise ValueError("gpu must be non-negative")
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_CACHE": "/data/models/hub",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    import torch
    from diffusers import Flux2KleinPipeline

    snapshot = MODEL_ROOT / "snapshots" / MODEL_REVISION
    if snapshot.is_symlink() or not snapshot.is_dir() or not (snapshot / "model_index.json").is_file():
        raise RuntimeError(f"pinned FLUX.2 snapshot is missing: {snapshot}")
    status: dict[str, Any] = {
        "schema": WORKER_STATUS_SCHEMA,
        "gpu": gpu,
        "partition_sha256": partition["partition_sha256"],
        "model_load_seconds": None,
        "jobs": [],
        "status": "running",
    }
    _atomic_json(status_path, status)
    load_started = time.perf_counter()
    pipeline = Flux2KleinPipeline.from_pretrained(
        str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    status["model_load_seconds"] = time.perf_counter() - load_started
    _atomic_json(status_path, status)
    parameters = partition["parameters"]

    for job in partition["jobs"]:
        started = time.perf_counter()
        job_id = job["execution_job_id"]
        destination = output_root / "candidates" / job_id
        record: dict[str, Any] = {
            "execution_job_id": job_id,
            "instance_id": job["consumer_requests"][0]["instance_id"],
            "profile_schema_id": job["profile_schema_id"],
            "status": "running",
        }
        try:
            if destination.exists() or destination.is_symlink():
                raise RuntimeError(f"candidate output already exists: {destination}")
            destination.mkdir(parents=True)
            reference = Path(job["reference"]["resolved_path"]).resolve()
            if (
                reference.is_symlink()
                or not reference.is_file()
                or reference.stat().st_size != job["reference"]["size_bytes"]
                or _sha256_file(reference) != job["reference"]["sha256"]
            ):
                raise RuntimeError("animal reference changed before FLUX.2")
            copied_source = destination / "source.png"
            _copy_no_replace(reference, copied_source)
            with Image.open(copied_source) as opened:
                opened.load()
                if opened.size != (1024, 1024):
                    raise RuntimeError("controlled animal source canvas must be 1024x1024")
                source = opened.convert("RGB")
            generation = job["generation_plan"]
            effective_prompt = (
                f"{generation['prompt']} Avoid: {generation['negative_prompt']}."
            )
            generator = torch.Generator("cuda").manual_seed(
                int(generation["generation_seed"])
            )
            result = pipeline(
                image=source,
                prompt=effective_prompt,
                width=parameters["width"],
                height=parameters["height"],
                num_inference_steps=parameters["num_inference_steps"],
                guidance_scale=parameters["guidance_scale"],
                generator=generator,
                max_sequence_length=parameters["max_sequence_length"],
            )
            if not getattr(result, "images", None):
                raise RuntimeError("FLUX.2 returned no animal candidate")
            candidate = result.images[0].convert("RGB")
            if candidate.size != (1024, 1024):
                raise RuntimeError("FLUX.2 animal output canvas changed")
            candidate_path = destination / "candidate.png"
            _save_png_no_replace(candidate, candidate_path)
            if _sha256_file(reference) != job["reference"]["sha256"]:
                raise RuntimeError("animal reference changed during FLUX.2")
            manifest: dict[str, Any] = {
                "schema": CANDIDATE_SCHEMA,
                "status": "pending_2d_review",
                "state_classification": "research_candidate",
                "formal_dataset_registration_authorized": False,
                "execution_preflight_sha256": partition["execution_preflight_sha256"],
                "execution_job_id": job_id,
                "instance_id": job["consumer_requests"][0]["instance_id"],
                "request_sha256": job["consumer_requests"][0]["request_sha256"],
                "profile_schema_id": job["profile_schema_id"],
                "profile_sha256": job["profile_sha256"],
                "lineage_group_id": job["lineage_group_id"],
                "taxonomy": job["taxonomy"],
                "fixed_attributes": job["fixed_attributes"],
                "sampled_attributes": job["sampled_attributes"],
                "input": _file_record(copied_source, root=output_root),
                "output": _file_record(candidate_path, root=output_root),
                "generation": {
                    "prompt": generation["prompt"],
                    "negative_prompt": generation["negative_prompt"],
                    "effective_prompt": effective_prompt,
                    "seed": generation["generation_seed"],
                    "flux_invocations": 1,
                    "model": partition["model"],
                    "parameters": parameters,
                },
                "downstream_gate": {
                    "status": "blocked_pending_2d_review",
                    "required_review": "approved_for_exact_candidate_sha256",
                    "next_stage": "foreground_segmentation_then_pixal3d",
                },
                "timings": {
                    "persistent_worker_model_load_seconds": status[
                        "model_load_seconds"
                    ],
                    "inference_and_publish_seconds": time.perf_counter() - started,
                    "model_reused": True,
                },
                "automatic_checks": {
                    "reference_hash_before_after_stable": True,
                    "one_flux_invocation": True,
                    "canvas_1024_rgb": True,
                    "visual_attributes_verified": False,
                    "overall": "pending_2d_review",
                },
            }
            manifest["manifest_sha256"] = _json_sha256(manifest)
            _atomic_json(destination / "candidate_manifest.json", manifest)
            record.update(
                {
                    "status": "passed_pending_2d_review",
                    "candidate": _file_record(candidate_path, root=output_root),
                    "manifest": _file_record(
                        destination / "candidate_manifest.json", root=output_root
                    ),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        except BaseException as error:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        status["jobs"].append(record)
        _atomic_json(status_path, status)
        torch.cuda.empty_cache()
        print(
            f"CONTROLLED_ANIMAL_FLUX2_JOB {job_id} {record['status']} "
            f"wall={record['wall_seconds']:.1f}s",
            flush=True,
        )

    status["passed_count"] = sum(
        item["status"] == "passed_pending_2d_review" for item in status["jobs"]
    )
    status["failed_count"] = sum(item["status"] == "failed" for item in status["jobs"])
    status["status"] = "passed" if status["failed_count"] == 0 else "failed"
    _atomic_json(status_path, status)
    return 1 if status["failed_count"] else 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    partition = validate_partition(json.loads(args.partition.read_text(encoding="utf-8")))
    return run_worker(partition, args.gpu, args.output_root.resolve(), args.status.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
