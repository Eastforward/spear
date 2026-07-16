#!/usr/bin/env python3
"""Run normalized animal FLUX.2 jobs with one persistent model per GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_animal_flux2_worker as worker
from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as material_execution
from tools import prepare_controlled_source_asset_execution as preparation


BATCH_SCHEMA = "avengine_controlled_animal_flux2_batch_v1"
PYTHON = Path("/data/jzy/miniconda3/envs/avengine-imagegen/bin/python")
WORKER = Path(__file__).resolve().parent / "controlled_animal_flux2_worker.py"
PARAMETERS = {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 1.0,
    "max_sequence_length": 512,
    "output_mode": "rgb_pending_segmentation",
}
MODEL = {
    "name": "black-forest-labs/FLUX.2-klein-4B",
    "root": str(worker.MODEL_ROOT),
    "revision": worker.MODEL_REVISION,
    "local_files_only": True,
}


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json_no_replace(path: Path, value: Any) -> None:
    contracts.write_json_no_replace(path, value)


def select_qa_canary_jobs(
    preflight: Mapping[str, Any],
    *,
    profile_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    preflight = preparation.validate_execution_preflight(preflight)
    jobs = preflight["routes"]["flux2_pixal3d_animal_v1"]
    by_instance = {
        job["consumer_requests"][0]["instance_id"]: job for job in jobs
    }
    generation_plan_path = Path(preflight["source_bundle"]["input_dir"]) / "generation_plan.json"
    generation_plan = contracts.load_json(generation_plan_path)
    selected_pairs = []
    selected_ids = set()
    seen_profiles = set()
    for pair in sorted(generation_plan["qa_pairs"], key=lambda item: item["pair_id"]):
        profile_id = pair["profile_schema_id"]
        if profile_ids is not None and profile_id not in profile_ids:
            continue
        if profile_id in seen_profiles:
            continue
        if pair["instance_a"] not in by_instance or pair["instance_b"] not in by_instance:
            continue
        if len(pair["different_attributes"]) != 1:
            continue
        seen_profiles.add(profile_id)
        selected_pairs.append(copy.deepcopy(pair))
        selected_ids.update((pair["instance_a"], pair["instance_b"]))
    available_profiles = {
        job["profile_schema_id"]
        for job in jobs
        if profile_ids is None or job["profile_schema_id"] in profile_ids
    }
    if seen_profiles != available_profiles:
        missing = sorted(available_profiles - seen_profiles)
        raise contracts.ContractError(
            f"no single-attribute planned QA pair for profiles: {missing}"
        )
    selected_jobs = sorted(
        (by_instance[instance_id] for instance_id in selected_ids),
        key=lambda item: item["execution_job_id"],
    )
    return selected_jobs, selected_pairs


def select_jobs(
    preflight: Mapping[str, Any],
    *,
    profile_ids: set[str] | None,
    execution_job_ids: set[str] | None,
    qa_pair_canary: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = preflight["routes"]["flux2_pixal3d_animal_v1"]
    known_profiles = {job["profile_schema_id"] for job in jobs}
    if profile_ids and not profile_ids.issubset(known_profiles):
        raise contracts.ContractError(
            f"unknown animal profile IDs: {sorted(profile_ids - known_profiles)}"
        )
    if qa_pair_canary:
        selected, pairs = select_qa_canary_jobs(
            preflight, profile_ids=profile_ids
        )
    else:
        selected = [
            copy.deepcopy(job)
            for job in jobs
            if profile_ids is None or job["profile_schema_id"] in profile_ids
        ]
        pairs = []
    if execution_job_ids is not None:
        known_ids = {job["execution_job_id"] for job in selected}
        unknown = execution_job_ids - known_ids
        if unknown:
            raise contracts.ContractError(f"unknown selected execution job IDs: {sorted(unknown)}")
        selected = [job for job in selected if job["execution_job_id"] in execution_job_ids]
        pairs = [
            pair
            for pair in pairs
            if {
                pair["instance_a"],
                pair["instance_b"],
            }.issubset(
                {
                    job["consumer_requests"][0]["instance_id"] for job in selected
                }
            )
        ]
    if not selected:
        raise contracts.ContractError("animal FLUX.2 selection is empty")
    return sorted(selected, key=lambda item: item["execution_job_id"]), pairs


def _partition_jobs(jobs: Sequence[dict[str, Any]], gpus: Sequence[int]):
    partitions = {gpu: [] for gpu in gpus}
    for index, job in enumerate(jobs):
        partitions[gpus[index % len(gpus)]].append(job)
    return {gpu: values for gpu, values in partitions.items() if values}


def run_jobs(
    *,
    preflight_path: Path,
    output_root: Path,
    gpus: Sequence[int],
    profile_ids: set[str] | None,
    execution_job_ids: set[str] | None,
    qa_pair_canary: bool,
) -> Path:
    if not gpus or len(gpus) > 4 or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise contracts.ContractError("provide one to four unique non-negative GPUs")
    preflight = material_execution._load_preflight(preflight_path)
    jobs, selected_pairs = select_jobs(
        preflight,
        profile_ids=profile_ids,
        execution_job_ids=execution_job_ids,
        qa_pair_canary=qa_pair_canary,
    )
    try:
        for job in jobs:
            one_shot.validate_flux_job(job)
    except one_shot.PolicyError as error:
        raise contracts.ContractError(str(error)) from error
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(
            f"refusing to replace existing output directory: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        partitions = _partition_jobs(jobs, list(gpus))
        partition_records = {}
        for gpu, gpu_jobs in partitions.items():
            payload: dict[str, Any] = {
                "schema": worker.PARTITION_SCHEMA,
                "execution_preflight_sha256": preflight["preflight_sha256"],
                "one_shot_execution": one_shot.stage_record("flux2"),
                "model": MODEL,
                "parameters": PARAMETERS,
                "jobs": gpu_jobs,
            }
            payload["partition_sha256"] = _json_sha256(payload)
            worker.validate_partition(payload)
            partition_path = staging / "workers" / f"gpu_{gpu}_partition.json"
            partition_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json_no_replace(partition_path, payload)
            partition_records[gpu] = partition_path

        def execute(gpu: int, partition_path: Path):
            status_path = staging / "workers" / f"gpu_{gpu}_status.json"
            log_path = staging / "workers" / f"gpu_{gpu}.log"
            command = [
                str(PYTHON),
                str(WORKER),
                "--partition",
                str(partition_path),
                "--gpu",
                str(gpu),
                "--output-root",
                str(staging),
                "--status",
                str(status_path),
            ]
            with log_path.open("xb") as log:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    check=False,
                )
                log.flush()
                os.fsync(log.fileno())
            return {
                "gpu": gpu,
                "returncode": completed.returncode,
                "command": command,
                "partition": _record(partition_path, root=staging),
                "status": _record(status_path, root=staging),
                "log": _record(log_path, root=staging),
            }

        workers = []
        with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
            futures = {
                executor.submit(execute, gpu, path): gpu
                for gpu, path in partition_records.items()
            }
            for future in as_completed(futures):
                workers.append(future.result())
        workers.sort(key=lambda item: item["gpu"])
        if any(item["returncode"] != 0 for item in workers):
            raise contracts.ContractError("one or more FLUX.2 workers failed")

        results = []
        for job in jobs:
            candidate_dir = staging / "candidates" / job["execution_job_id"]
            manifest_path = candidate_dir / "candidate_manifest.json"
            candidate_path = candidate_dir / "candidate.png"
            source_path = candidate_dir / "source.png"
            if not all(path.is_file() and not path.is_symlink() for path in (manifest_path, candidate_path, source_path)):
                raise contracts.ContractError(
                    f"FLUX.2 candidate bundle is incomplete: {job['execution_job_id']}"
                )
            manifest = contracts.load_json(manifest_path)
            if (
                manifest.get("schema") != worker.CANDIDATE_SCHEMA
                or manifest.get("execution_job_id") != job["execution_job_id"]
                or manifest.get("request_sha256")
                != job["consumer_requests"][0]["request_sha256"]
                or manifest.get("output", {}).get("sha256")
                != _sha256_file(candidate_path)
                or manifest.get("automatic_checks", {}).get("overall")
                != "pending_2d_review"
            ):
                raise contracts.ContractError("FLUX.2 candidate manifest readback failed")
            try:
                one_shot.validate_stage_record(
                    manifest.get("one_shot_execution"), "flux2"
                )
            except one_shot.PolicyError as error:
                raise contracts.ContractError(str(error)) from error
            results.append(
                {
                    "execution_job_id": job["execution_job_id"],
                    "instance_id": job["consumer_requests"][0]["instance_id"],
                    "profile_schema_id": job["profile_schema_id"],
                    "sampled_attributes": job["sampled_attributes"],
                    "status": "pending_2d_review",
                    "candidate": _record(candidate_path, root=staging),
                    "candidate_manifest": _record(manifest_path, root=staging),
                    "source": _record(source_path, root=staging),
                }
            )
        postflight = preparation.build_execution_preflight(
            Path(preflight["source_bundle"]["input_dir"]),
            {key: Path(value) for key, value in preflight["artifact_roots"].items()},
        )
        if contracts.canonical_json(postflight) != contracts.canonical_json(preflight):
            raise contracts.ContractError("execution inputs changed during FLUX.2 batch")
        batch: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "status": "pending_2d_review",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "one_shot_execution": one_shot.stage_record("flux2"),
            "execution_preflight": {
                "path": str(Path(preflight_path).resolve()),
                "sha256": _sha256_file(Path(preflight_path)),
                "preflight_sha256": preflight["preflight_sha256"],
            },
            "selection": {
                "semantics": "predeclared_request_subset_only_not_output_ranking",
                "profile_ids": sorted(profile_ids) if profile_ids else "all_animal_profiles",
                "execution_job_ids": sorted(execution_job_ids) if execution_job_ids else None,
                "qa_pair_canary": qa_pair_canary,
                "planned_qa_pairs": selected_pairs,
            },
            "model": MODEL,
            "parameters": PARAMETERS,
            "candidate_count": len(results),
            "candidates": sorted(results, key=lambda item: item["execution_job_id"]),
            "workers": workers,
            "automatic_checks": {
                "preflight_reauthenticated_before_and_after": True,
                "one_model_load_per_worker": True,
                "one_flux_invocation_per_candidate": True,
                "one_flux_image_per_candidate": True,
                "seed_retry_forbidden": True,
                "candidate_ranking_or_best_of_n_forbidden": True,
                "all_selected_requests_count_in_batch_outcome": True,
                "all_candidates_pending_visual_review": True,
                "pixal3d_not_started_before_review": True,
                "overall": "pending_2d_review",
            },
        }
        batch["batch_sha256"] = _json_sha256(batch)
        _write_json_no_replace(staging / "flux2_batch_manifest.json", batch)
        material_execution.native._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                f"refusing to replace concurrently-created output: {output_root}"
            )
        os.rename(staging, output_root)
        return output_root / "flux2_batch_manifest.json"
    except Exception:
        material_execution.native._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu", action="append", type=int, default=[])
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--execution-job-id", action="append", default=[])
    parser.add_argument("--qa-pair-canary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = run_jobs(
            preflight_path=args.preflight,
            output_root=args.output_root,
            gpus=args.gpu or [0, 1, 2, 3],
            profile_ids=set(args.profile) if args.profile else None,
            execution_job_ids=(
                set(args.execution_job_id) if args.execution_job_id else None
            ),
            qa_pair_canary=args.qa_pair_canary,
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_FLUX2_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_FLUX2_OK "
        f"candidates={manifest['candidate_count']} "
        f"workers={len(manifest['workers'])} status={manifest['status']} "
        f"output={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
