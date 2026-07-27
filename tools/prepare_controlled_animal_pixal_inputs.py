#!/usr/bin/env python3
"""Segment approved FLUX.2 source assets and compile authenticated Pixal3D jobs.

Animal and static-object requests share the same image segmentation and
Pixal3D inference implementation, but retain different downstream contracts.
The authenticated FLUX batch and execution preflight select that contract; a
static object is never represented as an animated-transfer animal.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_animal_isnet_worker as isnet
from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as material_execution
from tools import review_controlled_animal_flux2_candidates as review


PIXAL_INPUT_SCHEMA = "avengine_controlled_animal_pixal_inputs_v1"
ISNET_EXECUTION_RECEIPT_SCHEMA = "avengine_controlled_isnet_execution_receipt_v1"
ISNET_PYTHON = Path("/data/jzy/miniconda3/envs/hunyuan3d/bin/python")
ISNET_WORKER = Path(__file__).resolve().parent / "controlled_animal_isnet_worker.py"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
PIXAL_MODEL_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
DINO_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
PIXAL_ROUTE_CONTRACTS = {
    "flux2_pixal3d_animal_v1": {
        "asset_class": "animal",
        "generation_schema": "flux2_pixal3d_generation_plan_v1",
        "job_prefix": "animal_",
        "rig_mode": "animated_transfer",
        "base_template_kind": "reference_image",
        "rig_required": True,
    },
    "flux2_pixal3d_static_v1": {
        "asset_class": "static_object",
        "generation_schema": "flux2_pixal3d_static_generation_plan_v1",
        "job_prefix": "static_",
        "base_template_kind": "text_prompt_only",
        "rig_required": False,
    },
}


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _sha256_file(path: Path) -> str:
    return review._sha256_file(path)


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _absolute_file_record(
    path: Path,
    *,
    label: str,
    allow_symlink: bool = False,
    executable: bool = False,
) -> dict[str, Any]:
    path = Path(path).absolute()
    if (
        (path.is_symlink() and not allow_symlink)
        or not path.is_file()
        or (executable and not os.access(path, os.X_OK))
    ):
        raise contracts.ContractError(f"{label} is missing or invalid: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _reauthenticate_absolute_file_record(
    record: Mapping[str, Any],
    *,
    label: str,
    allow_symlink: bool = False,
    executable: bool = False,
) -> None:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise contracts.ContractError(f"{label} file record is invalid")
    path_value = record.get("path")
    path = Path(path_value) if isinstance(path_value, str) else Path()
    if (
        not isinstance(path_value, str)
        or not path.is_absolute()
        or (path.is_symlink() and not allow_symlink)
        or not path.is_file()
        or (executable and not os.access(path, os.X_OK))
        or path.stat().st_size != record.get("size_bytes")
        or _sha256_file(path) != record.get("sha256")
    ):
        raise contracts.ContractError(f"{label} changed")


def _copy_file_no_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    destination.chmod(0o444)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _seal_runtime_directory(path: Path) -> None:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise contracts.ContractError(f"ISNet runtime directory is unsafe: {path}")
    path.chmod(0o555)
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one directory without replacing any peer's root."""

    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise contracts.ContractError(
            "atomic no-replace Pixal input publication requires Linux renameat2"
        )
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
        raise FileExistsError(
            number,
            "refusing to replace concurrently-created Pixal input root",
            destination,
        )
    raise OSError(number, os.strerror(number), destination)


def _rebind_command_root(
    command: Sequence[str], *, source_root: Path, destination_root: Path
) -> list[str]:
    source = str(Path(source_root).absolute())
    destination = str(Path(destination_root).absolute())
    prefix = source + os.sep
    return [
        destination + argument[len(source) :]
        if argument == source or argument.startswith(prefix)
        else argument
        for argument in command
    ]


def _prepare_isnet_execution(
    *,
    staging: Path,
    output_root: Path,
    jobs_path: Path,
    status_path: Path,
) -> dict[str, Any]:
    configured_python = _absolute_file_record(
        ISNET_PYTHON,
        label="configured ISNet Python",
        allow_symlink=True,
        executable=True,
    )
    try:
        resolved_python_path = Path(configured_python["path"]).resolve(strict=True)
    except OSError as error:
        raise contracts.ContractError("configured ISNet Python cannot be resolved") from error
    resolved_python = _absolute_file_record(
        resolved_python_path,
        label="resolved ISNet Python",
        executable=True,
    )
    if (
        configured_python["sha256"] != resolved_python["sha256"]
        or configured_python["size_bytes"] != resolved_python["size_bytes"]
    ):
        raise contracts.ContractError(
            "configured and resolved ISNet Python identities differ"
        )

    worker_source = _absolute_file_record(
        ISNET_WORKER, label="ISNet worker source"
    )
    model = _absolute_file_record(isnet.MODEL_PATH, label="pinned ISNet model")
    if model["sha256"] != isnet.MODEL_SHA256:
        raise contracts.ContractError("pinned ISNet model hash changed")

    staged_worker = (
        staging / ".runtime_commands" / "controlled_animal_isnet_worker.py"
    )
    _copy_file_no_replace(Path(worker_source["path"]), staged_worker)
    staged_worker_record = _absolute_file_record(
        staged_worker, label="frozen ISNet worker"
    )
    if (
        staged_worker_record["sha256"] != worker_source["sha256"]
        or staged_worker_record["size_bytes"] != worker_source["size_bytes"]
    ):
        raise contracts.ContractError("frozen ISNet worker differs from its source")
    _seal_runtime_directory(staged_worker.parent)

    staged_jobs_record = _absolute_file_record(jobs_path, label="ISNet jobs")
    published_worker = (
        output_root / ".runtime_commands" / "controlled_animal_isnet_worker.py"
    )
    published_jobs = output_root / "isnet_jobs.json"
    published_status = output_root / "isnet_status.json"
    command = [
        resolved_python["path"],
        str(published_worker),
        "--jobs",
        str(published_jobs),
        "--status",
        str(published_status),
    ]
    executed_command = _rebind_command_root(
        command, source_root=output_root, destination_root=staging
    )
    expected_executed_command = [
        resolved_python["path"],
        str(staged_worker),
        "--jobs",
        str(jobs_path),
        "--status",
        str(status_path),
    ]
    if executed_command != expected_executed_command:
        raise contracts.ContractError("ISNet command root rebinding is ambiguous")

    receipt = {
        "schema": ISNET_EXECUTION_RECEIPT_SCHEMA,
        "model": model,
        "python": {
            "configured": configured_python,
            "resolved": resolved_python,
        },
        "worker": {
            "source": worker_source,
            "executed": {
                "path": str(published_worker),
                "sha256": staged_worker_record["sha256"],
                "size_bytes": staged_worker_record["size_bytes"],
            },
        },
        "jobs": {
            "path": str(published_jobs),
            "sha256": staged_jobs_record["sha256"],
            "size_bytes": staged_jobs_record["size_bytes"],
        },
        "working_directory": str(SPEAR_ROOT),
        "command": command,
        "command_sha256": _json_sha256(command),
        "executed_command": executed_command,
        "executed_command_sha256": _json_sha256(executed_command),
        "path_rebinding": {
            "staging_root": str(staging),
            "published_root": str(output_root),
        },
    }
    return {
        "receipt": receipt,
        "staged_worker": staged_worker,
        "jobs_path": jobs_path,
    }


def _reauthenticate_isnet_execution(execution: Mapping[str, Any]) -> None:
    receipt = execution["receipt"]
    python = receipt["python"]
    worker = receipt["worker"]
    _reauthenticate_absolute_file_record(
        python["configured"],
        label="configured ISNet Python",
        allow_symlink=True,
        executable=True,
    )
    _reauthenticate_absolute_file_record(
        python["resolved"], label="resolved ISNet Python", executable=True
    )
    try:
        current_resolved_python = Path(python["configured"]["path"]).resolve(
            strict=True
        )
    except OSError as error:
        raise contracts.ContractError("configured ISNet Python changed") from error
    if current_resolved_python != Path(python["resolved"]["path"]):
        raise contracts.ContractError("configured ISNet Python target changed")
    if (
        python["configured"]["sha256"] != python["resolved"]["sha256"]
        or python["configured"]["size_bytes"] != python["resolved"]["size_bytes"]
    ):
        raise contracts.ContractError(
            "configured and resolved ISNet Python identities changed"
        )
    _reauthenticate_absolute_file_record(
        worker["source"], label="ISNet worker source"
    )
    staged_worker_record = {
        **worker["executed"],
        "path": str(execution["staged_worker"]),
    }
    _reauthenticate_absolute_file_record(
        staged_worker_record, label="frozen ISNet worker"
    )
    runtime_directory = Path(execution["staged_worker"]).parent
    if (
        runtime_directory.is_symlink()
        or not runtime_directory.is_dir()
        or stat.S_IMODE(runtime_directory.stat().st_mode) != 0o555
    ):
        raise contracts.ContractError("frozen ISNet worker directory changed")
    if (
        staged_worker_record["sha256"] != worker["source"]["sha256"]
        or staged_worker_record["size_bytes"] != worker["source"]["size_bytes"]
    ):
        raise contracts.ContractError("frozen ISNet worker identity changed")
    _reauthenticate_absolute_file_record(
        receipt["model"], label="pinned ISNet model"
    )
    if receipt["model"]["sha256"] != isnet.MODEL_SHA256:
        raise contracts.ContractError("pinned ISNet model identity changed")
    staged_jobs_record = {**receipt["jobs"], "path": str(execution["jobs_path"])}
    _reauthenticate_absolute_file_record(staged_jobs_record, label="ISNet jobs")

    rebinding = receipt["path_rebinding"]
    if (
        rebinding
        != {
            "staging_root": str(Path(rebinding["staging_root"]).absolute()),
            "published_root": str(Path(rebinding["published_root"]).absolute()),
        }
        or receipt["command_sha256"] != _json_sha256(receipt["command"])
        or receipt["executed_command_sha256"]
        != _json_sha256(receipt["executed_command"])
        or receipt["executed_command"]
        != _rebind_command_root(
            receipt["command"],
            source_root=Path(rebinding["published_root"]),
            destination_root=Path(rebinding["staging_root"]),
        )
    ):
        raise contracts.ContractError("ISNet command execution receipt changed")


def _flux_one_shot_evidence(
    flux_batch: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Authenticate native policy evidence or explicitly label a legacy batch.

    Existing approved artifacts predate the machine policy and must not be
    rewritten.  They may continue only when their sealed manifests prove one
    recorded invocation and one output per frozen request; the evidence remains
    labelled legacy and cannot qualify a profile by itself.
    """

    record = flux_batch.get("one_shot_execution")
    if record is not None:
        try:
            one_shot.validate_stage_record(record, "flux2")
            for candidate in candidates.values():
                one_shot.validate_stage_record(
                    candidate["manifest"].get("one_shot_execution"), "flux2"
                )
        except one_shot.PolicyError as error:
            raise contracts.ContractError(str(error)) from error
        evidence = {
            "mode": "native_policy_enforced_before_inference",
            "policy": one_shot.policy_record(),
            "flux_batch_sha256": flux_batch["batch_sha256"],
            "profile_qualification_authorized": True,
        }
        one_shot.validate_upstream_flux_evidence(evidence)
        return evidence

    checks = flux_batch.get("automatic_checks", {})
    manifests = [candidate["manifest"] for candidate in candidates.values()]
    if (
        checks.get("one_flux_invocation_per_candidate") is not True
        or flux_batch.get("candidate_count") != len(manifests)
        or len(
            {
                candidate["index"].get("execution_job_id")
                for candidate in candidates.values()
            }
        )
        != len(manifests)
        or any(
            manifest.get("generation", {}).get("flux_invocations") != 1
            for manifest in manifests
        )
    ):
        raise contracts.ContractError("legacy FLUX batch lacks one-shot evidence")
    evidence = {
        "mode": "legacy_sealed_manifest_attestation",
        "policy": one_shot.policy_record(),
        "flux_batch_sha256": flux_batch["batch_sha256"],
        "recorded_flux_invocations_per_candidate": 1,
        "recorded_candidates_per_request": 1,
        "cross_batch_seed_lottery_exclusion_proven": False,
        "profile_qualification_authorized": False,
    }
    one_shot.validate_upstream_flux_evidence(evidence)
    return evidence


def load_review_batch(path: Path):
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"2D review batch is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema")
        not in {review.BATCH_REVIEW_SCHEMA, review.STATIC_BATCH_REVIEW_SCHEMA}
        or payload.get("review_batch_sha256")
        != _hash_without(payload, "review_batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("2D review batch contract/hash is invalid")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise contracts.ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _load_authenticated_preflight(flux_batch: Mapping[str, Any]) -> dict[str, Any]:
    record = flux_batch.get("execution_preflight")
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "preflight_sha256",
    }:
        raise contracts.ContractError("FLUX batch execution preflight record is invalid")
    path = Path(record["path"]).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError("FLUX batch execution preflight is missing")
    if _sha256_file(path) != _require_sha256(
        record["sha256"], "execution preflight file hash"
    ):
        raise contracts.ContractError("FLUX batch execution preflight file changed")
    preflight = material_execution._load_preflight(path)
    if preflight.get("preflight_sha256") != _require_sha256(
        record["preflight_sha256"], "execution preflight content hash"
    ):
        raise contracts.ContractError(
            "FLUX batch execution preflight content hash changed"
        )
    return preflight


def _validate_route_job(
    job: Any, route: str, route_contract: Mapping[str, Any]
) -> tuple[str, str]:
    if not isinstance(job, Mapping):
        raise contracts.ContractError(f"{route} execution job must be an object")
    plan = job.get("generation_plan")
    consumers = job.get("consumer_requests")
    if not isinstance(plan, Mapping) or not isinstance(consumers, list):
        raise contracts.ContractError(f"{route} execution job is incomplete")
    if (
        plan.get("route") != route
        or plan.get("schema") != route_contract["generation_schema"]
    ):
        raise contracts.ContractError(f"{route} generation plan route/schema changed")
    if len(consumers) != 1 or not isinstance(consumers[0], Mapping):
        raise contracts.ContractError(
            f"{route} execution job must bind exactly one request"
        )
    instance_id = consumers[0].get("instance_id")
    request_sha256 = _require_sha256(
        consumers[0].get("request_sha256"), f"{route} request hash"
    )
    if not isinstance(instance_id, str) or not instance_id:
        raise contracts.ContractError(f"{route} execution job has no instance ID")
    expected_job_id = f"{route_contract['job_prefix']}{request_sha256[:16]}"
    if job.get("execution_job_id") != expected_job_id:
        raise contracts.ContractError(
            f"{route} execution job ID does not match its request hash"
        )
    _require_sha256(job.get("profile_sha256"), f"{route} profile hash")
    if not isinstance(job.get("profile_schema_id"), str) or not job[
        "profile_schema_id"
    ]:
        raise contracts.ContractError(f"{route} execution job has no profile ID")

    base_template = plan.get("base_template")
    if (
        not isinstance(base_template, Mapping)
        or base_template.get("kind") != route_contract["base_template_kind"]
    ):
        raise contracts.ContractError(f"{route} base-template contract changed")
    rig_profile = job.get("rig_profile")
    if route_contract["rig_required"]:
        rig_actions = (
            rig_profile.get("actions")
            if isinstance(rig_profile, Mapping)
            else None
        )
        if (
            not isinstance(rig_profile, Mapping)
            or not isinstance(rig_actions, list)
            or len(rig_actions) != 2
            or set(rig_actions) != {"Walking", "Idle"}
        ):
            raise contracts.ContractError(
                "animal Pixal job must retain its Walking/Idle rig profile"
            )
        try:
            one_shot.validate_base_acquisition_record(
                plan.get("base_acquisition_policy")
            )
        except one_shot.PolicyError as error:
            raise contracts.ContractError(str(error)) from error
    else:
        if rig_profile is not None or base_template.get("artifact") is not None:
            raise contracts.ContractError(
                "static-object Pixal job must have no rig or reference-image binding"
            )
        try:
            one_shot.validate_static_base_acquisition_record(
                plan.get("base_acquisition_policy")
            )
        except one_shot.PolicyError as error:
            raise contracts.ContractError(str(error)) from error
    return instance_id, request_sha256


def _validate_candidate_job_binding(
    candidate: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    route: str,
    route_contract: Mapping[str, Any],
) -> None:
    index = candidate.get("index")
    manifest = candidate.get("manifest")
    files = candidate.get("files")
    if not all(isinstance(value, Mapping) for value in (index, manifest, files)):
        raise contracts.ContractError("FLUX candidate bundle is incomplete")
    consumer = job["consumer_requests"][0]
    expected = {
        "instance_id": consumer["instance_id"],
        "execution_job_id": job["execution_job_id"],
        "profile_schema_id": job["profile_schema_id"],
        "sampled_attributes": job["sampled_attributes"],
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise contracts.ContractError(
                f"FLUX candidate {key} does not match authenticated {route} job"
            )
    manifest_expected = {
        **expected,
        "profile_sha256": job["profile_sha256"],
        "request_sha256": consumer["request_sha256"],
    }
    for key, value in manifest_expected.items():
        if manifest.get(key) != value:
            raise contracts.ContractError(
                f"FLUX candidate manifest {key} does not match authenticated "
                f"{route} job"
            )
    has_source = "source" in files or "source" in index
    manifest_has_source = manifest.get("input") is not None
    if route_contract["asset_class"] == "static_object":
        if has_source or manifest_has_source:
            raise contracts.ContractError(
                "static-object Pixal candidate must not fabricate a source image"
            )
    elif not has_source or not manifest_has_source:
        raise contracts.ContractError(
            "animal Pixal candidate must retain its authenticated source image"
        )


def _authenticated_route_jobs(
    flux_batch: Mapping[str, Any],
    preflight: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    """Resolve one homogeneous Pixal route from sealed batch/preflight evidence."""

    if not candidates:
        raise contracts.ContractError("FLUX batch contains no candidates")
    routes = preflight.get("routes")
    if not isinstance(routes, Mapping):
        raise contracts.ContractError("execution preflight routes are invalid")
    jobs_by_route: dict[str, dict[str, Mapping[str, Any]]] = {}
    for route, route_contract in PIXAL_ROUTE_CONTRACTS.items():
        jobs = routes.get(route, [])
        if not isinstance(jobs, list):
            raise contracts.ContractError(f"execution preflight {route} jobs are invalid")
        indexed: dict[str, Mapping[str, Any]] = {}
        for job in jobs:
            instance_id, _request_sha256 = _validate_route_job(
                job, route, route_contract
            )
            if instance_id in indexed:
                raise contracts.ContractError(
                    f"execution preflight repeats {route} instance {instance_id}"
                )
            indexed[instance_id] = job
        jobs_by_route[route] = indexed

    resolved_routes: set[str] = set()
    selected_jobs: dict[str, Mapping[str, Any]] = {}
    for instance_id, candidate in candidates.items():
        matches = [
            (route, jobs[instance_id])
            for route, jobs in jobs_by_route.items()
            if instance_id in jobs
        ]
        if len(matches) != 1:
            raise contracts.ContractError(
                f"FLUX candidate {instance_id} does not resolve to one Pixal route"
            )
        route, job = matches[0]
        if instance_id in selected_jobs:
            raise contracts.ContractError(f"duplicate FLUX candidate: {instance_id}")
        _validate_candidate_job_binding(
            candidate,
            job,
            route=route,
            route_contract=PIXAL_ROUTE_CONTRACTS[route],
        )
        resolved_routes.add(route)
        selected_jobs[instance_id] = job
    if len(resolved_routes) != 1:
        raise contracts.ContractError("one Pixal input batch cannot mix asset routes")
    route = next(iter(resolved_routes))

    selection = flux_batch.get("selection")
    if not isinstance(selection, Mapping):
        raise contracts.ContractError("FLUX batch selection contract is invalid")
    declared_route = selection.get("route")
    if declared_route is None:
        # Animal batches sealed before static-object support did not record a
        # route. Their authenticated preflight/job bindings remain sufficient
        # to recover the only then-supported route.
        if route != "flux2_pixal3d_animal_v1":
            raise contracts.ContractError(
                "static-object FLUX batches must explicitly declare their route"
            )
    elif declared_route not in PIXAL_ROUTE_CONTRACTS:
        raise contracts.ContractError("FLUX batch declares an unsupported Pixal route")
    elif declared_route != route:
        raise contracts.ContractError(
            "FLUX batch route differs from its authenticated execution jobs"
        )
    return route, PIXAL_ROUTE_CONTRACTS[route], selected_jobs


def _controlled_request(
    controlled_job: Mapping[str, Any],
    *,
    instance_id: str,
    route: str,
    route_contract: Mapping[str, Any],
) -> dict[str, Any]:
    generation = controlled_job["generation_plan"]
    return {
        "execution_job_id": controlled_job["execution_job_id"],
        "instance_id": instance_id,
        "request_sha256": controlled_job["consumer_requests"][0]["request_sha256"],
        "generation_seed": int(generation["generation_seed"]),
        "profile_schema_id": controlled_job["profile_schema_id"],
        "profile_sha256": controlled_job["profile_sha256"],
        "asset_class": route_contract["asset_class"],
        "route": route,
        "sampled_attributes": copy.deepcopy(controlled_job["sampled_attributes"]),
        "target_physical_profile": copy.deepcopy(
            controlled_job["target_physical_profile"]
        ),
        "rig_profile": copy.deepcopy(controlled_job["rig_profile"]),
    }


def prepare_pixal_inputs(
    review_batch_path: Path,
    output_root: Path,
    pixal_output_root: Path,
) -> Path:
    review_batch = load_review_batch(review_batch_path)
    flux_batch_path = Path(review_batch["flux2_batch"]["path"])
    flux_root, flux_batch, candidates = review.load_flux_batch(flux_batch_path)
    route = flux_batch.get("selection", {}).get(
        "route", "flux2_pixal3d_animal_v1"
    )
    expected_review_batch_schema = (
        review.STATIC_BATCH_REVIEW_SCHEMA
        if route == "flux2_pixal3d_static_v1"
        else review.BATCH_REVIEW_SCHEMA
    )
    expected_review_domain = (
        "static_object" if route == "flux2_pixal3d_static_v1" else "animal"
    )
    if (
        review_batch.get("schema") != expected_review_batch_schema
        or review_batch.get("review_domain", expected_review_domain)
        != expected_review_domain
    ):
        raise contracts.ContractError("2D review domain differs from FLUX route")
    upstream_one_shot = _flux_one_shot_evidence(flux_batch, candidates)
    if flux_batch["batch_sha256"] != review_batch["flux2_batch"]["batch_sha256"]:
        raise contracts.ContractError("review and FLUX.2 batch hashes differ")
    approved_reviews = {}
    review_root = Path(review_batch_path).resolve().parent
    expected_review_schema = (
        review.STATIC_REVIEW_SCHEMA
        if route == "flux2_pixal3d_static_v1"
        else review.REVIEW_SCHEMA
    )
    for item in review_batch["reviews"]:
        record = item["review"]
        path = (review_root / record["path"]).resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise contracts.ContractError("2D review artifact changed")
        payload = contracts.load_json(path)
        if (
            payload.get("schema") != expected_review_schema
            or payload.get("instance_id") != item["instance_id"]
            or payload.get("candidate", {}).get("sha256") != item["candidate_sha256"]
            or payload.get("review_sha256") != _hash_without(payload, "review_sha256")
        ):
            raise contracts.ContractError("2D review record contract/hash is invalid")
        if payload["decision"] == "approved_for_pixal3d":
            approved_reviews[payload["instance_id"]] = payload
    if len(approved_reviews) != review_batch["approved_count"]:
        raise contracts.ContractError("approved 2D review count changed")
    if not approved_reviews:
        raise contracts.ContractError("no approved candidates for Pixal3D")

    preflight = _load_authenticated_preflight(flux_batch)
    route, route_contract, controlled_jobs = _authenticated_route_jobs(
        flux_batch, preflight, candidates
    )
    output_root = Path(output_root).absolute()
    pixal_output_root = Path(pixal_output_root).absolute()
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
        segmentation_jobs = []
        for instance_id, reviewed in sorted(approved_reviews.items()):
            candidate = candidates[instance_id]
            if reviewed["candidate"]["sha256"] != candidate["index"]["candidate"]["sha256"]:
                raise contracts.ContractError("approved candidate hash changed")
            destination = staging / "segmentation" / instance_id
            segmentation_jobs.append(
                {
                    "instance_id": instance_id,
                    "candidate_path": str(candidate["files"]["candidate"]),
                    "candidate_sha256": reviewed["candidate"]["sha256"],
                    "alpha_path": str(destination / "alpha_isnet.png"),
                    "rgba_path": str(destination / "input_rgba_isnet.png"),
                }
            )
        jobs_payload = {"schema": isnet.JOBS_SCHEMA, "jobs": segmentation_jobs}
        jobs_path = staging / "isnet_jobs.json"
        contracts.write_json_no_replace(jobs_path, jobs_payload)
        status_path = staging / "isnet_status.json"
        log_path = staging / "isnet.log"
        isnet_execution = _prepare_isnet_execution(
            staging=staging,
            output_root=output_root,
            jobs_path=jobs_path,
            status_path=status_path,
        )
        _reauthenticate_isnet_execution(isnet_execution)
        try:
            with log_path.open("xb") as log:
                completed = subprocess.run(
                    isnet_execution["receipt"]["executed_command"],
                    cwd=SPEAR_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    check=False,
                )
                log.flush()
                os.fsync(log.fileno())
        finally:
            _reauthenticate_isnet_execution(isnet_execution)
        if completed.returncode != 0:
            raise contracts.ContractError("ISNet worker failed")
        status = contracts.load_json(status_path)
        expected_status_model = {
            "path": isnet_execution["receipt"]["model"]["path"],
            "sha256": isnet_execution["receipt"]["model"]["sha256"],
            "name": "isnet-general-use",
        }
        if (
            status.get("schema") != isnet.STATUS_SCHEMA
            or status.get("status") != "passed"
            or status.get("passed_count") != len(segmentation_jobs)
            or status.get("failed_count") != 0
            or status.get("model") != expected_status_model
        ):
            raise contracts.ContractError("ISNet worker status is incomplete")
        status_by_id = {item["instance_id"]: item for item in status["jobs"]}
        pixal_jobs = []
        segmentations = []
        for job in segmentation_jobs:
            instance_id = job["instance_id"]
            status_item = status_by_id[instance_id]
            alpha_path = Path(job["alpha_path"])
            rgba_path = Path(job["rgba_path"])
            if (
                _sha256_file(alpha_path) != status_item["alpha_sha256"]
                or _sha256_file(rgba_path) != status_item["rgba_sha256"]
            ):
                raise contracts.ContractError("ISNet output hash changed")
            with Image.open(rgba_path) as opened:
                opened.load()
                if opened.mode != "RGBA" or opened.size != (1024, 1024):
                    raise contracts.ContractError("Pixal input RGBA contract changed")
            if instance_id not in controlled_jobs:
                raise contracts.ContractError(
                    f"approved candidate has no authenticated {route} job: {instance_id}"
                )
            controlled_job = controlled_jobs[instance_id]
            generation = controlled_job["generation_plan"]
            if (
                generation["model_revisions"]["pixal3d"] != PIXAL_MODEL_REVISION
                or generation["model_revisions"]["dino"] != DINO_REVISION
            ):
                raise contracts.ContractError("Pixal/DINO model revision changed")
            public_rgba = (
                output_root / "segmentation" / instance_id / "input_rgba_isnet.png"
            )
            pixal_output = pixal_output_root / instance_id / "pixal_raw_1024.glb"
            candidate_record = candidates[instance_id]["index"]["candidate"]
            pixal_job = {
                "legacy_tag": instance_id,
                "candidate_tag": f"{instance_id}_pixal_v1",
                "asset_class": route_contract["asset_class"],
                "route": route,
                "seed": int(generation["generation_seed"]),
                "attempt_ordinal": 0,
                "one_shot_execution": one_shot.stage_record("pixal3d"),
                "reference": {
                    "source": {
                        "path": str(candidates[instance_id]["files"]["candidate"]),
                        "sha256": candidate_record["sha256"],
                        "size_bytes": candidate_record["size_bytes"],
                    },
                    "pixal_input": {
                        "path": str(public_rgba),
                        "sha256": _sha256_file(rgba_path),
                        "size_bytes": rgba_path.stat().st_size,
                    },
                    "normalization": "pinned_isnet_general_use_alpha_v1",
                },
                "output": str(pixal_output),
                "manifest": str(pixal_output.with_suffix(".manifest.json")),
                "controlled_request": _controlled_request(
                    controlled_job,
                    instance_id=instance_id,
                    route=route,
                    route_contract=route_contract,
                ),
                "model_revisions": {
                    "pixal3d": PIXAL_MODEL_REVISION,
                    "dino": DINO_REVISION,
                },
                "parameters": {
                    "resolution": 1024,
                    "manual_fov": 0.2,
                    "low_vram": False,
                },
            }
            if route_contract["rig_required"]:
                pixal_job["rig_mode"] = "animated_transfer"
            pixal_jobs.append(pixal_job)
            segmentations.append(
                {
                    "instance_id": instance_id,
                    "candidate_sha256": candidates[instance_id]["index"]["candidate"][
                        "sha256"
                    ],
                    "alpha": _relative_record(alpha_path, staging),
                    "rgba": _relative_record(rgba_path, staging),
                    "foreground_fraction_at_128": status_item[
                        "foreground_fraction_at_128"
                    ],
                    "foreground_bbox_xyxy": status_item["foreground_bbox_xyxy"],
                    "status": "passed",
                }
            )
        _reauthenticate_isnet_execution(isnet_execution)
        isnet_receipt = copy.deepcopy(isnet_execution["receipt"])
        isnet_receipt["status"] = _relative_record(status_path, staging)
        isnet_receipt["log"] = _relative_record(log_path, staging)
        pixal_payload: dict[str, Any] = {
            "schema": PIXAL_INPUT_SCHEMA,
            "status": "ready_for_pixal3d",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "asset_class": route_contract["asset_class"],
            "route": route,
            "one_shot_execution": one_shot.stage_record("pixal3d"),
            "upstream_flux_one_shot_evidence": upstream_one_shot,
            "review_batch": {
                "path": str(Path(review_batch_path).resolve()),
                "sha256": _sha256_file(Path(review_batch_path)),
                "review_batch_sha256": review_batch["review_batch_sha256"],
            },
            "isnet": isnet_receipt,
            "pixal_output_root": str(pixal_output_root),
            "job_count": len(pixal_jobs),
            "jobs": pixal_jobs,
            "segmentations": segmentations,
            "automatic_checks": {
                "all_2d_reviews_approved": True,
                "all_candidate_hashes_reauthenticated": True,
                "all_isnet_segmentations_passed": True,
                "all_pixal_inputs_rgba_1024": True,
                "pixal_and_dino_revisions_pinned": True,
                "route_and_profile_bindings_reauthenticated": True,
                "isnet_runtime_inputs_reauthenticated_before_and_after_execution": True,
                "isnet_worker_executed_from_frozen_published_copy": True,
                "static_jobs_have_no_rig_or_animation_binding": (
                    route_contract["asset_class"] != "static_object"
                    or all(
                        "rig_mode" not in job
                        and job["controlled_request"]["rig_profile"] is None
                        for job in pixal_jobs
                    )
                ),
                "one_pixal_invocation_per_frozen_request": True,
                "seed_retry_forbidden": True,
                "candidate_ranking_or_best_of_n_forbidden": True,
                "overall": "passed",
            },
        }
        pixal_payload["manifest_sha256"] = _json_sha256(pixal_payload)
        contracts.write_json_no_replace(staging / "pixal_inputs_manifest.json", pixal_payload)
        material_execution.native._seal_readonly_tree(staging)
        _rename_noreplace(staging, output_root)
        return output_root / "pixal_inputs_manifest.json"
    except Exception:
        material_execution.native._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-batch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pixal-output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = prepare_pixal_inputs(
            args.review_batch, args.output_root, args.pixal_output_root
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_PIXAL_INPUT_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_PIXAL_INPUT_OK "
        f"jobs={manifest['job_count']} output={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
