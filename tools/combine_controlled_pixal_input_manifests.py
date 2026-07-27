#!/usr/bin/env python3
"""Combine sealed homogeneous Pixal input manifests without weakening lineage.

The combiner exists to amortize one persistent Pixal3D model load across
multiple independently reviewed input batches.  It does not mutate or flatten
the parent evidence.  Every load of the combined manifest reopens the parents,
checks their file and canonical content hashes, revalidates every controlled
job and source image, and authenticates a private byte-copy of each RGBA input.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as material_execution
from tools import prepare_controlled_animal_pixal_inputs as preparation


COMBINED_PIXAL_INPUT_SCHEMA = "avengine_controlled_pixal_inputs_combined_v1"
COMBINED_UPSTREAM_EVIDENCE_SCHEMA = (
    "avengine_combined_upstream_flux_one_shot_evidence_v1"
)
PARENT_INPUT_SCHEMAS = frozenset({preparation.PIXAL_INPUT_SCHEMA})
_INSTANCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_MODELS = {
    "pixal3d": preparation.PIXAL_MODEL_REVISION,
    "dino": preparation.DINO_REVISION,
}
_EXPECTED_PARAMETERS = {
    "resolution": 1024,
    "manual_fov": 0.2,
    "low_vram": False,
}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


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


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise contracts.ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_file_record_at(
    value: Any,
    physical_path: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise contracts.ContractError(f"{label} file record is invalid")
    raw_path = value.get("path")
    size = value.get("size_bytes")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise contracts.ContractError(f"{label} file record is invalid")
    unresolved = Path(physical_path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(f"{label} file is missing or unsafe")
    path = unresolved.resolve()
    digest = _require_sha256(value.get("sha256"), f"{label} hash")
    if path.stat().st_size != size or _sha256_file(path) != digest:
        raise contracts.ContractError(f"{label} file changed")
    return path, copy.deepcopy(dict(value))


def _require_file_record(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    raw_path = value.get("path") if isinstance(value, Mapping) else None
    if not isinstance(raw_path, str) or not raw_path:
        raise contracts.ContractError(f"{label} file record is invalid")
    return _require_file_record_at(value, Path(raw_path), label)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one directory and fail if any destination exists."""

    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise contracts.ContractError(
            "atomic no-replace Pixal publication requires Linux renameat2"
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
            "refusing to replace concurrently-created combined Pixal root",
            destination,
        )
    raise OSError(number, os.strerror(number), destination)


def _validate_rgba_1024(path: Path, label: str) -> None:
    try:
        with Image.open(path) as opened:
            opened.load()
            if opened.mode != "RGBA" or opened.size != (1024, 1024):
                raise contracts.ContractError(
                    f"{label} must be a decoded 1024x1024 RGBA image"
                )
    except (OSError, UnidentifiedImageError) as error:
        raise contracts.ContractError(f"{label} is not a readable RGBA image") from error


def _load_parent_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    """Use the production runner's base-manifest validator.

    This import is intentionally local.  The runner lazily dispatches the new
    combined schema back to this module, while every parent must remain an
    independently sealed base schema rather than a recursively flattened
    combined manifest.
    """

    from tools import run_controlled_animal_pixal_jobs as pixal_runner

    loaded_path, payload = pixal_runner.load_pixal_inputs(path)
    if payload.get("schema") not in PARENT_INPUT_SCHEMAS:
        raise contracts.ContractError(
            "combined Pixal parents must be independently sealed base manifests"
        )
    return loaded_path, payload


def _validate_controlled_request(
    value: Any,
    *,
    route: str,
    asset_class: str,
) -> tuple[str, str]:
    expected_fields = {
        "execution_job_id",
        "instance_id",
        "request_sha256",
        "generation_seed",
        "profile_schema_id",
        "profile_sha256",
        "asset_class",
        "route",
        "sampled_attributes",
        "target_physical_profile",
        "rig_profile",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise contracts.ContractError("Pixal controlled request contract changed")
    instance_id = value.get("instance_id")
    if (
        not isinstance(instance_id, str)
        or _INSTANCE_ID_RE.fullmatch(instance_id) is None
    ):
        raise contracts.ContractError("Pixal controlled instance ID is invalid")
    request_sha256 = _require_sha256(
        value.get("request_sha256"), f"{instance_id} request hash"
    )
    _require_sha256(value.get("profile_sha256"), f"{instance_id} profile hash")
    if (
        value.get("route") != route
        or value.get("asset_class") != asset_class
        or not isinstance(value.get("profile_schema_id"), str)
        or not value["profile_schema_id"]
        or not isinstance(value.get("sampled_attributes"), Mapping)
        or not isinstance(value.get("target_physical_profile"), Mapping)
    ):
        raise contracts.ContractError(
            f"Pixal controlled request route/profile changed: {instance_id}"
        )
    seed = value.get("generation_seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < (1 << 63)
    ):
        raise contracts.ContractError(
            f"Pixal controlled request seed is invalid: {instance_id}"
        )
    expected_prefix = "static_" if asset_class == "static_object" else "animal_"
    expected_execution_id = f"{expected_prefix}{request_sha256[:16]}"
    if value.get("execution_job_id") != expected_execution_id:
        raise contracts.ContractError(
            f"Pixal controlled execution job changed: {instance_id}"
        )
    rig_profile = value.get("rig_profile")
    if asset_class == "static_object":
        if route != "flux2_pixal3d_static_v1" or rig_profile is not None:
            raise contracts.ContractError(
                "static Pixal controlled request has an animal/rig binding"
            )
    elif asset_class == "animal":
        actions = rig_profile.get("actions") if isinstance(rig_profile, Mapping) else None
        if (
            route != "flux2_pixal3d_animal_v1"
            or not isinstance(actions, list)
            or set(actions) != {"Walking", "Idle"}
            or len(actions) != 2
        ):
            raise contracts.ContractError(
                "animal Pixal controlled request lost its Walking/Idle rig binding"
            )
    else:
        raise contracts.ContractError("unsupported Pixal asset class")
    return instance_id, str(value["execution_job_id"])


def _validate_parent_job(
    job: Any,
    *,
    parent_root: Path,
    route: str,
    asset_class: str,
) -> tuple[str, str, Path, dict[str, Any]]:
    if not isinstance(job, Mapping):
        raise contracts.ContractError("Pixal parent job must be an object")
    try:
        one_shot.validate_pixal_job(job)
    except one_shot.PolicyError as error:
        raise contracts.ContractError(str(error)) from error
    controlled = job.get("controlled_request")
    instance_id, execution_job_id = _validate_controlled_request(
        controlled, route=route, asset_class=asset_class
    )
    if (
        job.get("legacy_tag") != instance_id
        or job.get("asset_class") != asset_class
        or job.get("route") != route
        or job.get("model_revisions") != _EXPECTED_MODELS
        or job.get("parameters") != _EXPECTED_PARAMETERS
        or job.get("reference", {}).get("normalization")
        != "pinned_isnet_general_use_alpha_v1"
    ):
        raise contracts.ContractError(f"Pixal parent job contract changed: {instance_id}")
    if asset_class == "static_object":
        if "rig_mode" in job:
            raise contracts.ContractError("static Pixal parent job carries rig_mode")
    elif job.get("rig_mode") != "animated_transfer":
        raise contracts.ContractError("animal Pixal parent job lost animated_transfer")

    reference = job.get("reference")
    if not isinstance(reference, Mapping) or set(reference) != {
        "source",
        "pixal_input",
        "normalization",
    }:
        raise contracts.ContractError(
            f"Pixal parent reference contract changed: {instance_id}"
        )
    _source_path, source_record = _require_file_record(
        reference["source"], f"{instance_id} source"
    )
    rgba_path, rgba_record = _require_file_record(
        reference["pixal_input"], f"{instance_id} RGBA"
    )
    _validate_rgba_1024(rgba_path, f"{instance_id} RGBA")
    try:
        rgba_path.relative_to(parent_root)
    except ValueError as error:
        raise contracts.ContractError(
            f"Pixal parent RGBA escaped its sealed root: {instance_id}"
        ) from error
    expected_parent_output = (
        Path(job.get("output", "")).resolve().parent / "pixal_raw_1024.glb"
    )
    if Path(job.get("output", "")).resolve() != expected_parent_output:
        raise contracts.ContractError(
            f"Pixal parent output filename changed: {instance_id}"
        )
    if Path(job.get("manifest", "")).resolve() != expected_parent_output.with_suffix(
        ".manifest.json"
    ):
        raise contracts.ContractError(
            f"Pixal parent attempt manifest changed: {instance_id}"
        )
    return instance_id, execution_job_id, rgba_path, {
        "source": source_record,
        "pixal_input": rgba_record,
    }


def _parent_receipt(path: Path, payload: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": payload["schema"],
        "content_sha256": payload["manifest_sha256"],
        "asset_class": payload["asset_class"],
        "route": payload["route"],
        "job_count": payload["job_count"],
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_root_isolation(
    *,
    combined_input_root: Path,
    pixal_output_root: Path,
    parent_paths: Sequence[Path],
    parent_payloads: Sequence[Mapping[str, Any]],
) -> None:
    """Keep a combined plan and its execution root outside every parent plan.

    This is deliberately run both before staging and whenever a combined
    manifest is loaded.  A canonical JSON hash detects accidental changes but
    is not a signature, so a rehashed manifest must not be able to redirect
    execution into a parent input directory or an old parent Pixal plan.
    """

    if _paths_overlap(combined_input_root, pixal_output_root):
        raise contracts.ContractError(
            "combined input root and Pixal output root must be disjoint"
        )
    for parent_path, parent_payload in zip(parent_paths, parent_payloads):
        parent_manifest_root = parent_path.parent.resolve()
        parent_pixal_root = Path(parent_payload["pixal_output_root"]).resolve()
        for root, label in (
            (combined_input_root, "combined input root"),
            (pixal_output_root, "Pixal output root"),
        ):
            if _paths_overlap(root, parent_manifest_root):
                raise contracts.ContractError(
                    f"{label} must not overlap a parent manifest directory"
                )
            if _paths_overlap(root, parent_pixal_root):
                raise contracts.ContractError(
                    f"{label} must bind a new output root, not overlap a parent plan"
                )


def _validate_combined_upstream_evidence(
    value: Any,
    *,
    parent_receipts: Sequence[Mapping[str, Any]],
    parent_payloads: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "policy",
        "parent_count",
        "parents",
    }:
        raise contracts.ContractError("combined upstream one-shot evidence is invalid")
    if (
        value.get("schema") != COMBINED_UPSTREAM_EVIDENCE_SCHEMA
        or value.get("parent_count") != len(parent_payloads)
        or not isinstance(value.get("parents"), list)
        or len(value["parents"]) != len(parent_payloads)
    ):
        raise contracts.ContractError("combined upstream one-shot evidence is incomplete")
    try:
        one_shot.validate_policy_record(value.get("policy"))
    except one_shot.PolicyError as error:
        raise contracts.ContractError(str(error)) from error
    expected_parent_evidence = []
    for receipt, payload in zip(parent_receipts, parent_payloads):
        evidence = payload.get("upstream_flux_one_shot_evidence")
        try:
            one_shot.validate_upstream_flux_evidence(evidence)
        except one_shot.PolicyError as error:
            raise contracts.ContractError(str(error)) from error
        if evidence["policy"] != value["policy"]:
            raise contracts.ContractError("parent Pixal one-shot policies differ")
        expected_parent_evidence.append(
            {
                "parent_content_sha256": receipt["content_sha256"],
                "evidence": copy.deepcopy(evidence),
            }
        )
    if value["parents"] != expected_parent_evidence:
        raise contracts.ContractError("combined parent one-shot evidence changed")


def _load_parent_receipts(
    receipts: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    if not isinstance(receipts, list) or len(receipts) < 2:
        raise contracts.ContractError("combine at least two Pixal parent manifests")
    payloads: list[dict[str, Any]] = []
    paths: list[Path] = []
    normalized_receipts: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for ordinal, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "ordinal",
            "path",
            "sha256",
            "schema",
            "content_sha256",
            "asset_class",
            "route",
            "job_count",
        }:
            raise contracts.ContractError("combined Pixal parent receipt is invalid")
        if receipt.get("ordinal") != ordinal:
            raise contracts.ContractError("combined Pixal parent ordering changed")
        raw_path = receipt.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise contracts.ContractError("combined Pixal parent path is invalid")
        unresolved = Path(raw_path)
        if unresolved.is_symlink() or not unresolved.is_file():
            raise contracts.ContractError("combined Pixal parent is missing or unsafe")
        path = unresolved.resolve()
        if path in seen_paths:
            raise contracts.ContractError("duplicate Pixal parent manifest")
        seen_paths.add(path)
        if _sha256_file(path) != _require_sha256(
            receipt.get("sha256"), "parent manifest file hash"
        ):
            raise contracts.ContractError("parent Pixal manifest file changed")
        loaded_path, payload = _load_parent_manifest(path)
        if loaded_path != path:
            raise contracts.ContractError("parent Pixal manifest path resolution changed")
        expected = _parent_receipt(path, payload, ordinal)
        if dict(receipt) != expected:
            raise contracts.ContractError("parent Pixal manifest receipt changed")
        normalized_receipts.append(expected)
        payloads.append(payload)
        paths.append(path)
    return normalized_receipts, payloads, paths


def _validate_homogeneous_parents(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    route = payloads[0].get("route")
    asset_class = payloads[0].get("asset_class")
    stage_record = payloads[0].get("one_shot_execution")
    models: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    for payload in payloads:
        if (
            payload.get("route") != route
            or payload.get("asset_class") != asset_class
            or payload.get("formal_dataset_registration_authorized") is not False
            or payload.get("state_classification") != "research_candidate"
        ):
            raise contracts.ContractError(
                "Pixal parents mix routes, asset classes, or registration states"
            )
        try:
            one_shot.validate_stage_record(payload.get("one_shot_execution"), "pixal3d")
        except one_shot.PolicyError as error:
            raise contracts.ContractError(str(error)) from error
        if payload.get("one_shot_execution") != stage_record:
            raise contracts.ContractError("Pixal parent one-shot stage policies differ")
        for job in payload["jobs"]:
            job_models = job.get("model_revisions")
            job_parameters = job.get("parameters")
            if models is None:
                models = copy.deepcopy(job_models)
                parameters = copy.deepcopy(job_parameters)
            elif job_models != models or job_parameters != parameters:
                raise contracts.ContractError(
                    "Pixal parent model revisions or parameters differ"
                )
    if models != _EXPECTED_MODELS or parameters != _EXPECTED_PARAMETERS:
        raise contracts.ContractError("Pixal model revisions or parameters changed")
    if not isinstance(route, str) or not isinstance(asset_class, str):
        raise contracts.ContractError("Pixal parent route/asset class is invalid")
    return (
        route,
        asset_class,
        copy.deepcopy(dict(stage_record)),
        copy.deepcopy(models),
        copy.deepcopy(parameters),
    )


def _collect_parent_jobs(
    parent_payloads: Sequence[Mapping[str, Any]],
    parent_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}
    instances: set[str] = set()
    execution_ids: set[str] = set()
    output_paths: set[Path] = set()
    attempt_manifest_paths: set[Path] = set()
    route = str(parent_payloads[0]["route"])
    asset_class = str(parent_payloads[0]["asset_class"])
    for parent_ordinal, (payload, path) in enumerate(
        zip(parent_payloads, parent_paths)
    ):
        for job in payload["jobs"]:
            instance_id, execution_id, rgba_path, reference = _validate_parent_job(
                job,
                parent_root=path.parent,
                route=route,
                asset_class=asset_class,
            )
            parent_output = Path(job["output"]).resolve()
            parent_attempt_manifest = Path(job["manifest"]).resolve()
            if instance_id in instances:
                raise contracts.ContractError(
                    f"duplicate Pixal instance across parents: {instance_id}"
                )
            if execution_id in execution_ids:
                raise contracts.ContractError(
                    f"duplicate Pixal execution job across parents: {execution_id}"
                )
            if parent_output in output_paths or parent_attempt_manifest in attempt_manifest_paths:
                raise contracts.ContractError(
                    "duplicate Pixal output/attempt path across parents"
                )
            instances.add(instance_id)
            execution_ids.add(execution_id)
            output_paths.add(parent_output)
            attempt_manifest_paths.add(parent_attempt_manifest)
            copied_job = copy.deepcopy(dict(job))
            jobs.append(copied_job)
            bindings[instance_id] = {
                "parent_ordinal": parent_ordinal,
                "parent_content_sha256": payload["manifest_sha256"],
                "parent_job_sha256": _json_sha256(job),
                "source": reference["source"],
                "parent_pixal_input": reference["pixal_input"],
                "_parent_rgba_path": rgba_path,
                "_parent_job": copied_job,
            }
    return jobs, bindings


def _expected_combined_job(
    parent_job: Mapping[str, Any],
    copied_rgba: Mapping[str, Any],
    *,
    pixal_output_root: Path,
) -> dict[str, Any]:
    instance_id = parent_job["controlled_request"]["instance_id"]
    job = copy.deepcopy(dict(parent_job))
    job["reference"]["pixal_input"] = copy.deepcopy(dict(copied_rgba))
    output = pixal_output_root / instance_id / "pixal_raw_1024.glb"
    job["output"] = str(output)
    job["manifest"] = str(output.with_suffix(".manifest.json"))
    return job


def _validate_combined_payload(
    payload: Any,
    *,
    manifest_path: Path,
    physical_root: Path,
    public_root: Path,
) -> None:
    """Validate one payload while allowing staged bytes to record final paths."""

    manifest_path = Path(manifest_path).resolve()
    physical_root = Path(physical_root).resolve()
    public_root = Path(public_root).resolve()
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != COMBINED_PIXAL_INPUT_SCHEMA
        or payload.get("status") != "ready_for_pixal3d"
        or payload.get("state_classification") != "research_candidate"
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("manifest_sha256")
        != _hash_without(payload, "manifest_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("combined Pixal manifest contract/hash is invalid")
    recorded_root = payload.get("combined_input_root")
    if (
        not isinstance(recorded_root, str)
        or not Path(recorded_root).is_absolute()
        or Path(recorded_root).resolve() != public_root
        or manifest_path != physical_root / "pixal_inputs_manifest.json"
    ):
        raise contracts.ContractError(
            "combined Pixal physical/public manifest roots changed"
        )
    receipts, parent_payloads, parent_paths = _load_parent_receipts(
        payload.get("parents")
    )
    route, asset_class, stage_record, models, parameters = (
        _validate_homogeneous_parents(parent_payloads)
    )
    if (
        payload.get("parent_count") != len(receipts)
        or payload.get("route") != route
        or payload.get("asset_class") != asset_class
        or payload.get("one_shot_execution") != stage_record
        or payload.get("model_revisions") != models
        or payload.get("parameters") != parameters
    ):
        raise contracts.ContractError("combined Pixal homogeneous contract changed")
    _validate_combined_upstream_evidence(
        payload.get("upstream_flux_one_shot_evidence"),
        parent_receipts=receipts,
        parent_payloads=parent_payloads,
    )
    _validate_root_isolation(
        combined_input_root=public_root,
        pixal_output_root=Path(payload.get("pixal_output_root", "")).resolve(),
        parent_paths=parent_paths,
        parent_payloads=parent_payloads,
    )
    _jobs, parent_bindings = _collect_parent_jobs(parent_payloads, parent_paths)
    jobs = payload.get("jobs")
    copies = payload.get("input_copies")
    if (
        not isinstance(jobs, list)
        or not isinstance(copies, list)
        or not jobs
        or payload.get("job_count") != len(jobs)
        or len(jobs) != len(parent_bindings)
        or len(copies) != len(jobs)
    ):
        raise contracts.ContractError("combined Pixal job/input-copy count is invalid")
    pixal_output_root = Path(payload.get("pixal_output_root", "")).resolve()
    copy_by_id: dict[str, Mapping[str, Any]] = {}
    for binding in copies:
        if not isinstance(binding, Mapping):
            raise contracts.ContractError("combined Pixal input-copy binding is invalid")
        instance_id = binding.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or instance_id in copy_by_id
            or instance_id not in parent_bindings
        ):
            raise contracts.ContractError(
                "combined Pixal input-copy identity/coverage changed"
            )
        expected_parent = parent_bindings[instance_id]
        if set(binding) != {
            "instance_id",
            "parent_ordinal",
            "parent_content_sha256",
            "parent_job_sha256",
            "source",
            "parent_pixal_input",
            "copied_pixal_input",
        }:
            raise contracts.ContractError("combined Pixal input-copy fields changed")
        for key in (
            "parent_ordinal",
            "parent_content_sha256",
            "parent_job_sha256",
            "source",
            "parent_pixal_input",
        ):
            if binding.get(key) != expected_parent[key]:
                raise contracts.ContractError(
                    f"combined Pixal parent binding changed: {instance_id}"
                )
        copied_record_value = binding["copied_pixal_input"]
        recorded_copy_path = (
            Path(copied_record_value.get("path", "")).resolve()
            if isinstance(copied_record_value, Mapping)
            else None
        )
        expected_public_copy_path = (
            public_root / "segmentation" / instance_id / "input_rgba_isnet.png"
        )
        expected_physical_copy_path = (
            physical_root / "segmentation" / instance_id / "input_rgba_isnet.png"
        )
        if recorded_copy_path != expected_public_copy_path:
            raise contracts.ContractError(
                f"combined Pixal RGBA record escaped its public root: {instance_id}"
            )
        copied_path, copied_record = _require_file_record_at(
            copied_record_value,
            expected_physical_copy_path,
            f"{instance_id} copied RGBA",
        )
        _validate_rgba_1024(copied_path, f"{instance_id} copied RGBA")
        if copied_path != expected_physical_copy_path:
            raise contracts.ContractError(
                f"combined Pixal RGBA bytes escaped their physical root: {instance_id}"
            )
        if (
            copied_record["sha256"]
            != expected_parent["parent_pixal_input"]["sha256"]
            or copied_record["size_bytes"]
            != expected_parent["parent_pixal_input"]["size_bytes"]
        ):
            raise contracts.ContractError(
                f"combined Pixal RGBA copy differs from parent: {instance_id}"
            )
        copy_by_id[instance_id] = binding

    seen_instances: set[str] = set()
    seen_execution_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    seen_manifests: set[Path] = set()
    for job in jobs:
        controlled = job.get("controlled_request") if isinstance(job, Mapping) else None
        instance_id = (
            controlled.get("instance_id") if isinstance(controlled, Mapping) else None
        )
        if not isinstance(instance_id, str) or instance_id not in copy_by_id:
            raise contracts.ContractError("combined Pixal job coverage changed")
        binding = copy_by_id[instance_id]
        parent_job = parent_bindings[instance_id]["_parent_job"]
        expected_job = _expected_combined_job(
            parent_job,
            binding["copied_pixal_input"],
            pixal_output_root=pixal_output_root,
        )
        if contracts.canonical_json(job) != contracts.canonical_json(expected_job):
            raise contracts.ContractError(
                f"combined Pixal job differs from authenticated parent: {instance_id}"
            )
        execution_id = controlled["execution_job_id"]
        output = Path(job["output"]).resolve()
        attempt_manifest = Path(job["manifest"]).resolve()
        if (
            instance_id in seen_instances
            or execution_id in seen_execution_ids
            or output in seen_outputs
            or attempt_manifest in seen_manifests
        ):
            raise contracts.ContractError("combined Pixal job identity/path is duplicate")
        seen_instances.add(instance_id)
        seen_execution_ids.add(execution_id)
        seen_outputs.add(output)
        seen_manifests.add(attempt_manifest)
    if seen_instances != set(parent_bindings):
        raise contracts.ContractError("combined Pixal job coverage is incomplete")


def _validate_staged_combined_manifest(
    manifest_path: Path,
    public_root: Path,
) -> dict[str, Any]:
    """Fully authenticate staged bytes before their atomic publication."""

    unresolved = Path(manifest_path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(
            f"staged combined Pixal manifest is missing: {unresolved}"
        )
    manifest_path = unresolved.resolve()
    payload = contracts.load_json(manifest_path)
    _validate_combined_payload(
        payload,
        manifest_path=manifest_path,
        physical_root=manifest_path.parent,
        public_root=public_root,
    )
    return payload


def load_combined_pixal_inputs(path: Path) -> tuple[Path, dict[str, Any]]:
    """Authenticate a combined manifest and all of its live parent/input bytes."""

    unresolved = Path(path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(f"combined Pixal manifest is missing: {unresolved}")
    path = unresolved.resolve()
    payload = contracts.load_json(path)
    recorded_root = payload.get("combined_input_root") if isinstance(payload, dict) else None
    if (
        not isinstance(recorded_root, str)
        or not Path(recorded_root).is_absolute()
        or Path(recorded_root).resolve() != path.parent
    ):
        raise contracts.ContractError(
            "combined Pixal manifest is not at its authenticated public root"
        )
    _validate_combined_payload(
        payload,
        manifest_path=path,
        physical_root=path.parent,
        public_root=path.parent,
    )
    return path, payload


def combine_pixal_inputs(
    parent_manifest_paths: Sequence[Path],
    output_root: Path,
    pixal_output_root: Path,
) -> Path:
    if len(parent_manifest_paths) < 2:
        raise contracts.ContractError("combine at least two Pixal parent manifests")
    normalized_paths = sorted(Path(item).resolve() for item in parent_manifest_paths)
    if len(set(normalized_paths)) != len(normalized_paths):
        raise contracts.ContractError("duplicate Pixal parent manifest")
    parent_payloads: list[dict[str, Any]] = []
    for path in normalized_paths:
        loaded_path, payload = _load_parent_manifest(path)
        if loaded_path != path:
            raise contracts.ContractError("parent Pixal manifest path resolution changed")
        parent_payloads.append(payload)
    receipts = [
        _parent_receipt(path, payload, ordinal)
        for ordinal, (path, payload) in enumerate(
            zip(normalized_paths, parent_payloads)
        )
    ]
    route, asset_class, stage_record, models, parameters = (
        _validate_homogeneous_parents(parent_payloads)
    )
    _parent_jobs, parent_bindings = _collect_parent_jobs(
        parent_payloads, normalized_paths
    )

    output_root = Path(output_root).resolve()
    pixal_output_root = Path(pixal_output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(
            f"refusing to replace combined Pixal input root: {output_root}"
        )
    if pixal_output_root.exists() or pixal_output_root.is_symlink():
        raise contracts.ContractError(
            f"refusing to target existing Pixal output root: {pixal_output_root}"
        )
    _validate_root_isolation(
        combined_input_root=output_root,
        pixal_output_root=pixal_output_root,
        parent_paths=normalized_paths,
        parent_payloads=parent_payloads,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        combined_jobs: list[dict[str, Any]] = []
        input_copies: list[dict[str, Any]] = []
        for instance_id in sorted(parent_bindings):
            binding = parent_bindings[instance_id]
            source_path = binding["_parent_rgba_path"]
            staged_copy = (
                staging / "segmentation" / instance_id / "input_rgba_isnet.png"
            )
            staged_copy.parent.mkdir(parents=True, exist_ok=False)
            with source_path.open("rb") as source, staged_copy.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            copied_record = {
                "path": str(
                    output_root
                    / "segmentation"
                    / instance_id
                    / "input_rgba_isnet.png"
                ),
                "sha256": _sha256_file(staged_copy),
                "size_bytes": staged_copy.stat().st_size,
            }
            if (
                copied_record["sha256"]
                != binding["parent_pixal_input"]["sha256"]
                or copied_record["size_bytes"]
                != binding["parent_pixal_input"]["size_bytes"]
            ):
                raise contracts.ContractError(
                    f"Pixal RGBA copy changed during publication: {instance_id}"
                )
            combined_jobs.append(
                _expected_combined_job(
                    binding["_parent_job"],
                    copied_record,
                    pixal_output_root=pixal_output_root,
                )
            )
            input_copies.append(
                {
                    key: copy.deepcopy(value)
                    for key, value in binding.items()
                    if not key.startswith("_")
                }
                | {"instance_id": instance_id, "copied_pixal_input": copied_record}
            )

        combined_upstream = {
            "schema": COMBINED_UPSTREAM_EVIDENCE_SCHEMA,
            "policy": one_shot.policy_record(),
            "parent_count": len(receipts),
            "parents": [
                {
                    "parent_content_sha256": receipt["content_sha256"],
                    "evidence": copy.deepcopy(
                        payload["upstream_flux_one_shot_evidence"]
                    ),
                }
                for receipt, payload in zip(receipts, parent_payloads)
            ],
        }
        _validate_combined_upstream_evidence(
            combined_upstream,
            parent_receipts=receipts,
            parent_payloads=parent_payloads,
        )
        payload: dict[str, Any] = {
            "schema": COMBINED_PIXAL_INPUT_SCHEMA,
            "status": "ready_for_pixal3d",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "asset_class": asset_class,
            "route": route,
            "one_shot_execution": stage_record,
            "upstream_flux_one_shot_evidence": combined_upstream,
            "model_revisions": models,
            "parameters": parameters,
            "combined_input_root": str(output_root),
            "parent_count": len(receipts),
            "parents": receipts,
            "pixal_output_root": str(pixal_output_root),
            "job_count": len(combined_jobs),
            "jobs": combined_jobs,
            "input_copies": input_copies,
            "automatic_checks": {
                "all_parent_paths_file_hashes_and_content_hashes_reauthenticated": True,
                "all_parent_jobs_and_controlled_requests_reauthenticated": True,
                "all_source_images_reauthenticated": True,
                "all_rgba_inputs_copied_byte_exactly_into_immutable_root": True,
                "all_routes_asset_classes_models_and_parameters_homogeneous": True,
                "all_instance_job_and_output_paths_globally_unique": True,
                "static_jobs_have_no_rig_or_animation_binding": True,
                "one_pixal_invocation_per_frozen_request": True,
                "seed_retry_forbidden": True,
                "candidate_ranking_or_best_of_n_forbidden": True,
                "no_parent_directory_modified": True,
                "no_generated_asset_has_been_registered": True,
                "overall": "passed",
            },
        }
        payload["manifest_sha256"] = _hash_without(payload, "manifest_sha256")
        contracts.write_json_no_replace(
            staging / "pixal_inputs_manifest.json", payload
        )
        material_execution.native._seal_readonly_tree(staging)
        _validate_staged_combined_manifest(
            staging / "pixal_inputs_manifest.json",
            output_root,
        )
        _rename_noreplace(staging, output_root)
        manifest_path = output_root / "pixal_inputs_manifest.json"
        return manifest_path
    except Exception:
        if staging.exists():
            material_execution.native._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        action="append",
        required=True,
        type=Path,
        help="sealed base Pixal input manifest; provide at least twice",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pixal-output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = combine_pixal_inputs(
            args.parent, args.output_root, args.pixal_output_root
        )
        payload = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError) as error:
        print(f"CONTROLLED_PIXAL_INPUT_COMBINE_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_PIXAL_INPUT_COMBINE_OK "
        f"parents={payload['parent_count']} jobs={payload['job_count']} "
        f"output={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
