#!/usr/bin/env python3
"""Execute authenticated controlled-animal Pixal3D jobs on persistent GPUs."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_mesh_efficiency
from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_animal_pixal_inputs as pixal_inputs
from tools import rocketbox_native_material_canary as immutable


BATCH_SCHEMA = "avengine_controlled_animal_pixal_batch_v1"
WORKER_SCHEMA = "pixal_animal_persistent_worker_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
WORKER = SPEAR_ROOT / "tools/pixal_animal_persistent_worker.py"
PYTHON = Path("/data/jzy/miniconda3/envs/avengine-3dgen/bin/python3.10")


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


def load_pixal_inputs(path: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"Pixal input manifest is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != pixal_inputs.PIXAL_INPUT_SCHEMA
        or payload.get("status") != "ready_for_pixal3d"
        or payload.get("manifest_sha256") != _hash_without(payload, "manifest_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("Pixal input manifest contract/hash is invalid")
    try:
        one_shot.validate_stage_record(payload.get("one_shot_execution"), "pixal3d")
        one_shot.validate_upstream_flux_evidence(
            payload.get("upstream_flux_one_shot_evidence")
        )
    except one_shot.PolicyError as error:
        raise contracts.ContractError(str(error)) from error
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or payload.get("job_count") != len(jobs) or not jobs:
        raise contracts.ContractError("Pixal job count is invalid")
    identifiers = [job.get("controlled_request", {}).get("instance_id") for job in jobs]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(jobs):
        raise contracts.ContractError("Pixal jobs contain missing/duplicate instance IDs")
    root = path.parent
    expected_output_root = Path(payload["pixal_output_root"]).resolve()
    for job in jobs:
        try:
            one_shot.validate_pixal_job(job)
        except one_shot.PolicyError as error:
            raise contracts.ContractError(str(error)) from error
        instance_id = job["controlled_request"]["instance_id"]
        if job.get("legacy_tag") != instance_id:
            raise contracts.ContractError("Pixal legacy tag must equal controlled instance ID")
        if job.get("model_revisions") != {
            "pixal3d": pixal_inputs.PIXAL_MODEL_REVISION,
            "dino": pixal_inputs.DINO_REVISION,
        }:
            raise contracts.ContractError("Pixal/DINO revisions changed")
        if job.get("parameters") != {
            "resolution": 1024,
            "manual_fov": 0.2,
            "low_vram": False,
        }:
            raise contracts.ContractError("Pixal parameters changed")
        image_record = job.get("reference", {}).get("pixal_input", {})
        image = Path(image_record.get("path", "")).resolve()
        try:
            image.relative_to(root)
        except ValueError as error:
            raise contracts.ContractError("Pixal RGBA input escaped its immutable root") from error
        if (
            image.is_symlink()
            or not image.is_file()
            or image.stat().st_size != image_record.get("size_bytes")
            or _sha256_file(image) != image_record.get("sha256")
        ):
            raise contracts.ContractError(f"Pixal RGBA input changed: {instance_id}")
        expected_output = expected_output_root / instance_id / "pixal_raw_1024.glb"
        if Path(job.get("output", "")).resolve() != expected_output:
            raise contracts.ContractError("Pixal output path is not deterministic")
        if Path(job.get("manifest", "")).resolve() != expected_output.with_suffix(
            ".manifest.json"
        ):
            raise contracts.ContractError("Pixal attempt manifest path is not deterministic")
    return path, payload


def partition_jobs(
    jobs: Sequence[dict[str, Any]], gpus: Sequence[int]
) -> dict[int, list[dict[str, Any]]]:
    if not 1 <= len(gpus) <= 4 or len(set(gpus)) != len(gpus):
        raise contracts.ContractError("provide one to four unique GPUs")
    partitions = {int(gpu): [] for gpu in gpus}
    for index, job in enumerate(jobs):
        partitions[int(gpus[index % len(gpus)])].append(job)
    return partitions


def build_worker_orders(
    jobs: Sequence[dict[str, Any]], gpus: Sequence[int]
) -> dict[int, list[dict[str, Any]]]:
    """Give each active GPU the full queue with a distinct starting offset."""
    partition_jobs(jobs, gpus)  # Reuse the strict GPU contract validation.
    active_gpus = list(gpus[: min(len(gpus), len(jobs))])
    orders = {}
    for index, gpu in enumerate(active_gpus):
        offset = index % len(jobs)
        orders[int(gpu)] = list(jobs[offset:]) + list(jobs[:offset])
    return orders


def build_worker_job(
    job: dict[str, Any], staging: Path, public_root: Path
) -> dict[str, Any]:
    instance_id = job["controlled_request"]["instance_id"]
    public_output = public_root / instance_id / "pixal_raw_1024.glb"
    physical_output = staging / instance_id / "pixal_raw_1024.glb"
    public_manifest = public_output.with_suffix(".manifest.json")
    physical_manifest = physical_output.with_suffix(".manifest.json")
    return {
        "legacy_tag": job["legacy_tag"],
        "candidate_tag": job["candidate_tag"],
        "seed": job["seed"],
        "attempt_ordinal": job["attempt_ordinal"],
        "one_shot_execution": job["one_shot_execution"],
        "reference": job["reference"],
        "output": str(physical_output),
        "manifest": str(physical_manifest),
        "public_output": str(public_output),
        "public_manifest": str(public_manifest),
        "controlled_request": job["controlled_request"],
    }


def _run_partition(
    gpu: int,
    jobs: Sequence[dict[str, Any]],
    staging: Path,
    claim_dir: Path,
) -> dict[str, Any]:
    workers = staging / "worker_evidence"
    jobs_path = workers / f"gpu_{gpu}_jobs.json"
    status_path = workers / f"gpu_{gpu}_status.json"
    log_path = workers / f"gpu_{gpu}.log"
    workers.mkdir(parents=True, exist_ok=True)
    contracts.write_json_no_replace(jobs_path, list(jobs))
    with log_path.open("xb") as log:
        completed = subprocess.run(
            [
                str(PYTHON),
                str(WORKER),
                "--jobs",
                str(jobs_path),
                "--gpu",
                str(gpu),
                "--status",
                str(status_path),
                "--claim-dir",
                str(claim_dir),
            ],
            cwd=SPEAR_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=10800,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    return {
        "gpu": gpu,
        "returncode": completed.returncode,
        "jobs_path": jobs_path,
        "status_path": status_path,
        "log_path": log_path,
    }


def _relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_outputs(
    source_jobs: Sequence[dict[str, Any]],
    worker_results: Sequence[dict[str, Any]],
    staging: Path,
    public_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    status_by_id: dict[str, tuple[int, dict[str, Any], dict[str, Any]]] = {}
    workers: list[dict[str, Any]] = []
    for worker in sorted(worker_results, key=lambda item: item["gpu"]):
        if worker["returncode"] != 0:
            raise contracts.ContractError(f"Pixal worker failed on GPU {worker['gpu']}")
        status = contracts.load_json(worker["status_path"])
        if (
            not isinstance(status, dict)
            or status.get("schema") != WORKER_SCHEMA
            or status.get("scheduling_mode") != "shared_claim_queue_v1"
            or status.get("failed_count") != 0
        ):
            raise contracts.ContractError("Pixal worker status is invalid")
        if status.get("passed_count") != len(status.get("jobs", [])):
            raise contracts.ContractError("Pixal worker passed count is incomplete")
        for record in status["jobs"]:
            instance_id = record.get("legacy_tag")
            if instance_id in status_by_id or record.get("status") != "passed":
                raise contracts.ContractError("Pixal worker result coverage is invalid")
            status_by_id[instance_id] = (worker["gpu"], record, status)
        workers.append(
            {
                "gpu": worker["gpu"],
                "returncode": worker["returncode"],
                "model_load_seconds": status.get("model_load_seconds"),
                "passed_count": status.get("passed_count"),
                "failed_count": status.get("failed_count"),
                "status": _relative_artifact(worker["status_path"], staging),
                "jobs": _relative_artifact(worker["jobs_path"], staging),
                "log": _relative_artifact(worker["log_path"], staging),
            }
        )
    if set(status_by_id) != {
        job["controlled_request"]["instance_id"] for job in source_jobs
    }:
        raise contracts.ContractError("Pixal worker result coverage changed")

    claim_root = staging / "worker_evidence" / "claims"
    claim_paths = sorted(claim_root.glob("*.claim.json"))
    if len(claim_paths) != len(source_jobs):
        raise contracts.ContractError("Pixal dynamic claim coverage is incomplete")
    claims: list[dict[str, Any]] = []
    claim_by_id: dict[str, Path] = {}
    for claim_path in claim_paths:
        if claim_path.is_symlink() or not claim_path.is_file():
            raise contracts.ContractError("Pixal dynamic claim is not a direct file")
        claim = contracts.load_json(claim_path)
        instance_id = claim.get("legacy_tag")
        expected_id = hashlib.sha256(str(instance_id).encode("utf-8")).hexdigest()
        if (
            claim.get("schema") != "pixal_dynamic_work_claim_v1"
            or claim.get("claim_sha256") != expected_id
            or claim_path.name != f"{expected_id}.claim.json"
            or instance_id not in status_by_id
            or instance_id in claim_by_id
            or claim.get("gpu") != status_by_id[instance_id][0]
        ):
            raise contracts.ContractError("Pixal dynamic claim identity changed")
        worker_record = status_by_id[instance_id][1]
        if Path(worker_record.get("claim", "")).resolve() != claim_path.resolve():
            raise contracts.ContractError("Pixal worker/claim evidence differs")
        claim_by_id[instance_id] = claim_path
        claims.append(
            {
                "instance_id": instance_id,
                "gpu": claim["gpu"],
                "claim": _relative_artifact(claim_path, staging),
            }
        )
    if set(claim_by_id) != set(status_by_id):
        raise contracts.ContractError("Pixal dynamic claim IDs are incomplete")
    for worker in workers:
        worker["claimed_count"] = sum(
            item["gpu"] == worker["gpu"] for item in claims
        )

    attempts: list[dict[str, Any]] = []
    for job in sorted(
        source_jobs, key=lambda item: item["controlled_request"]["instance_id"]
    ):
        controlled = job["controlled_request"]
        instance_id = controlled["instance_id"]
        gpu, worker_record, worker_status = status_by_id[instance_id]
        physical_output = staging / instance_id / "pixal_raw_1024.glb"
        physical_manifest = physical_output.with_suffix(".manifest.json")
        if not physical_output.is_file() or not physical_manifest.is_file():
            raise contracts.ContractError(f"Pixal output is missing: {instance_id}")
        output_hash = _sha256_file(physical_output)
        model_manifest = contracts.load_json(physical_manifest)
        public_output = public_root / instance_id / "pixal_raw_1024.glb"
        if (
            output_hash != worker_record.get("output_sha256")
            or model_manifest.get("backend") != "pixal3d"
            or model_manifest.get("output", {}).get("sha256") != output_hash
            or Path(model_manifest.get("output", {}).get("path", "")).resolve()
            != public_output
            or model_manifest.get("model", {}).get("revision")
            != pixal_inputs.PIXAL_MODEL_REVISION
            or model_manifest.get("dino", {}).get("revision")
            != pixal_inputs.DINO_REVISION
            or model_manifest.get("parameters") != {
                "low_vram": False,
                "manual_fov": 0.2,
                "resolution": 1024,
                "seed": int(job["seed"]),
            }
            or model_manifest.get("controlled_request") != controlled
            or model_manifest.get("one_shot_execution")
            != one_shot.stage_record("pixal3d")
        ):
            raise contracts.ContractError(f"Pixal attempt manifest mismatch: {instance_id}")
        stats = audit_mesh_efficiency.mesh_stats(physical_output)
        if (
            not stats
            or not stats.get("exists")
            or stats.get("triangles", 0) <= 0
            or stats.get("vertices", 0) <= 0
            or stats.get("materials", 0) <= 0
            or stats.get("textures", 0) <= 0
            or stats.get("skins") != 0
            or stats.get("animations") != 0
        ):
            raise contracts.ContractError(f"Pixal GLB readback failed: {instance_id}")
        attempts.append(
            {
                "instance_id": instance_id,
                "execution_job_id": controlled["execution_job_id"],
                "request_sha256": controlled["request_sha256"],
                "profile_schema_id": controlled["profile_schema_id"],
                "sampled_attributes": controlled["sampled_attributes"],
                "target_physical_profile": controlled["target_physical_profile"],
                "gpu": gpu,
                "seed": int(job["seed"]),
                "attempt_ordinal": int(job["attempt_ordinal"]),
                "one_shot_execution": job["one_shot_execution"],
                "pixal_input": job["reference"]["pixal_input"],
                "output": _relative_artifact(physical_output, staging),
                "attempt_manifest": _relative_artifact(physical_manifest, staging),
                "mesh_readback": {
                    key: value for key, value in stats.items() if key not in {"path", "exists"}
                },
                "timings": {
                    "model_load_seconds": worker_status.get("model_load_seconds"),
                    "inference_and_export_seconds": model_manifest["timings"][
                        "inference_and_export_seconds"
                    ],
                    "model_reused": model_manifest["timings"]["model_reused"],
                },
                "status": "passed_generation_and_glb_readback",
                "next_gate": "static_visual_qa",
            }
        )
    return attempts, workers, sorted(claims, key=lambda item: item["instance_id"])


def run_batch(
    input_manifest_path: Path,
    output_root: Path,
    gpus: Sequence[int],
) -> Path:
    input_manifest_path, payload = load_pixal_inputs(input_manifest_path)
    output_root = Path(output_root).absolute()
    expected_root = Path(payload["pixal_output_root"]).resolve()
    if output_root.resolve() != expected_root:
        raise contracts.ContractError("output root differs from authenticated Pixal plan")
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output directory: {output_root}")
    if not PYTHON.is_file() or not WORKER.is_file():
        raise contracts.ContractError("pinned Pixal Python/worker is missing")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    started_at = _utc_now()
    try:
        worker_jobs = [
            build_worker_job(job, staging, output_root) for job in payload["jobs"]
        ]
        claims = staging / "worker_evidence" / "claims"
        claims.mkdir(parents=True, exist_ok=False)
        worker_orders = build_worker_orders(worker_jobs, gpus)
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(worker_orders)) as executor:
            futures = {
                executor.submit(_run_partition, gpu, order, staging, claims): gpu
                for gpu, order in worker_orders.items()
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    "CONTROLLED_ANIMAL_PIXAL_WORKER_DONE "
                    f"gpu={result['gpu']} returncode={result['returncode']}",
                    flush=True,
                )
        attempts, workers, work_claims = _validate_outputs(
            payload["jobs"], results, staging, output_root
        )
        manifest: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "status": "passed_generation_and_glb_readback",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "one_shot_execution": one_shot.stage_record("pixal3d"),
            "upstream_flux_one_shot_evidence": payload[
                "upstream_flux_one_shot_evidence"
            ],
            "started_at": started_at,
            "finished_at": _utc_now(),
            "pixal_inputs": {
                "path": str(input_manifest_path),
                "sha256": _sha256_file(input_manifest_path),
                "manifest_sha256": payload["manifest_sha256"],
            },
            "models": {
                "pixal3d_revision": pixal_inputs.PIXAL_MODEL_REVISION,
                "dino_revision": pixal_inputs.DINO_REVISION,
            },
            "parameters": {
                "resolution": 1024,
                "manual_fov": 0.2,
                "low_vram": False,
            },
            "gpus": list(gpus),
            "job_count": len(attempts),
            "passed_count": len(attempts),
            "failed_count": 0,
            "attempts": attempts,
            "workers": workers,
            "scheduling": {
                "mode": "shared_claim_queue_v1",
                "claim_count": len(work_claims),
                "claims": work_claims,
            },
            "automatic_checks": {
                "all_inputs_reauthenticated": True,
                "all_model_revisions_pinned": True,
                "all_jobs_have_unique_request": True,
                "all_jobs_claimed_once_by_dynamic_queue": True,
                "one_pixal_invocation_per_frozen_request": True,
                "seed_retry_forbidden": True,
                "candidate_ranking_or_best_of_n_forbidden": True,
                "all_outputs_glb2_readable": True,
                "all_outputs_have_pbr_material_and_texture": True,
                "no_generated_asset_has_been_registered": True,
                "overall": "passed",
            },
        }
        manifest["batch_sha256"] = _hash_without(manifest, "batch_sha256")
        contracts.write_json_no_replace(staging / "pixal_batch_manifest.json", manifest)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("Pixal output root appeared during execution")
        os.rename(staging, output_root)
        return output_root / "pixal_batch_manifest.json"
    except Exception:
        # A model failure is evidence, not disposable scratch state.  Preserve
        # the exact worker order, claim, status (when present), and log under a
        # sibling immutable directory so the public output path remains
        # fail-closed and may be retried without overwriting anything.
        failure_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        failure_root = output_root.parent / (
            f"{output_root.name}.failed_{failure_stamp}_{os.getpid()}"
        )
        immutable._seal_readonly_tree(staging)
        os.rename(staging, failure_root)
        print(
            "CONTROLLED_ANIMAL_PIXAL_FAILURE_EVIDENCE "
            f"output={failure_root}",
            file=sys.stderr,
            flush=True,
        )
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixal-inputs", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu", action="append", type=int, default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = run_batch(
            args.pixal_inputs,
            args.output_root,
            args.gpu or [0, 1, 2, 3],
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_PIXAL_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_PIXAL_OK "
        f"jobs={manifest['job_count']} output={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
