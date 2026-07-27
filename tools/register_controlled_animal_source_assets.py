#!/usr/bin/env python3
"""Register static-qualified controlled animal outputs as source_asset_v2 candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as preflight_tools
from tools import prepare_controlled_source_asset_execution as preparation
from tools import review_controlled_animal_pixal_static_candidates as static_decisions
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_pixal_jobs as pixal_runner


REGISTRY_SCHEMA = "avengine_controlled_animal_source_asset_registry_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path("/data/models")
LICENSE_SPECS = (
    {
        "path": "hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md",
        "sha256": "ca02bc51900ab07789d1b70283329e7137f5af98f5161c23a1c81fc38a4af1fe",
        "size_bytes": 9584,
    },
    {
        "path": "hub/models--TencentARC--Pixal3D/snapshots/0b31f9160aa400719af409098bff7936a932f726/LICENSE",
        "sha256": "31d37e9c4fee1e0cd2196bccd592e8a2c30bfa17ea177d70ad25f977ba6bd9c0",
        "size_bytes": 1064,
    },
    {
        "path": "hub/models--camenduru--dinov3-vitl16-pretrain-lvd1689m/snapshots/3c276edd87d6f6e569ff0c4400e086807d0f3881/LICENSE.md",
        "sha256": "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e",
        "size_bytes": 7503,
    },
    {
        "path": "rembg/isnet-general-use/rembg-2.0.69-LICENSE.txt",
        "sha256": "90a3215072968fd304669c5389f04f1274a587abdd0507d99dead0f5511f8999",
        "size_bytes": 1069,
    },
)


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


def spear_artifact(path: Path) -> dict[str, Any]:
    literal = Path(path).absolute()
    path = literal.resolve()
    try:
        relative = path.relative_to(SPEAR_ROOT.resolve())
    except ValueError:
        # SPEAR/tmp is intentionally a logical repository path backed by the
        # external workspace symlink. Preserve that logical root in durable
        # artifact records while authenticating the resolved target bytes.
        try:
            tmp_relative = path.relative_to((SPEAR_ROOT / "tmp").resolve())
        except ValueError as error:
            raise contracts.ContractError(
                f"artifact is outside SPEAR root: {path}"
            ) from error
        relative = Path("tmp") / tmp_relative
    if literal.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise contracts.ContractError(f"artifact is missing/non-direct: {path}")
    return {
        "root_id": "spear_repo",
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def license_records() -> list[dict[str, Any]]:
    models_root = MODELS_ROOT.absolute()
    if models_root.is_symlink() or not models_root.is_dir():
        raise contracts.ContractError("/data/models root is missing or unsafe")
    resolved_root = models_root.resolve()
    records = []
    for spec in LICENSE_SPECS:
        logical = models_root / spec["path"]
        resolved = logical.resolve(strict=True)
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError as error:
            raise contracts.ContractError("model license escaped /data/models") from error
        if (
            not resolved.is_file()
            or resolved.stat().st_size != spec["size_bytes"]
            or _sha256_file(resolved) != spec["sha256"]
        ):
            raise contracts.ContractError(f"model license changed: {logical}")
        # Hugging Face snapshot license entries are normally symlink leaves.
        # Preserve the pinned bytes while recording their direct content-addressed
        # blob path so downstream consumers never need to weaken symlink checks.
        direct = resolved_root / relative
        if direct.is_symlink() or direct.resolve(strict=True) != resolved:
            raise contracts.ContractError(
                f"resolved model license is not a direct file: {resolved}"
            )
        records.append(
            {
                "root_id": "models_root",
                "path": relative.as_posix(),
                "sha256": spec["sha256"],
                "size_bytes": spec["size_bytes"],
            }
        )
    return records


def load_decision_batch(path: Path):
    literal = Path(path).absolute()
    path = literal.resolve()
    if literal.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise contracts.ContractError("static decision batch is missing or unsafe")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != static_decisions.DECISION_BATCH_SCHEMA
        or payload.get("status") != "completed"
        or payload.get("decision_batch_sha256")
        != _hash_without(payload, "decision_batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
        or payload.get("decision_count") != len(payload.get("decisions", []))
    ):
        raise contracts.ContractError("static decision batch contract/hash is invalid")

    review_batch_descriptor = payload.get("static_review_batch")
    if (
        not isinstance(review_batch_descriptor, Mapping)
        or set(review_batch_descriptor)
        != {"path", "sha256", "review_batch_sha256"}
        or not isinstance(review_batch_descriptor.get("path"), str)
        or not Path(review_batch_descriptor["path"]).is_absolute()
        or not isinstance(review_batch_descriptor.get("sha256"), str)
        or len(review_batch_descriptor["sha256"]) != 64
        or not isinstance(review_batch_descriptor.get("review_batch_sha256"), str)
        or len(review_batch_descriptor["review_batch_sha256"]) != 64
    ):
        raise contracts.ContractError("static review batch descriptor is invalid")
    review_batch_literal = Path(review_batch_descriptor["path"])
    review_batch_path = review_batch_literal.resolve()
    if (
        review_batch_literal.is_symlink()
        or not review_batch_path.is_file()
        or _sha256_file(review_batch_path) != review_batch_descriptor["sha256"]
    ):
        raise contracts.ContractError("static review batch changed")
    (
        authenticated_review_batch_path,
        authenticated_review_batch,
        authenticated_reviews,
    ) = static_decisions.load_review_batch(review_batch_path)
    if (
        authenticated_review_batch_path != review_batch_path
        or authenticated_review_batch["review_batch_sha256"]
        != review_batch_descriptor["review_batch_sha256"]
    ):
        raise contracts.ContractError("static review batch identity changed")

    records = {}
    root = path.parent
    for index in payload["decisions"]:
        expected_index_fields = {
            "instance_id",
            "decision",
            "decision_sha256",
            "record",
        }
        record_descriptor = index.get("record") if isinstance(index, Mapping) else None
        if (
            not isinstance(index, Mapping)
            or set(index) != expected_index_fields
            or index.get("decision")
            not in {"approved_for_lod_and_binding", "rejected"}
            or not isinstance(record_descriptor, Mapping)
            or set(record_descriptor) != {"path", "sha256", "size_bytes"}
            or not isinstance(record_descriptor.get("path"), str)
            or Path(record_descriptor["path"]).is_absolute()
            or ".." in Path(record_descriptor["path"]).parts
            or not isinstance(record_descriptor.get("sha256"), str)
            or len(record_descriptor["sha256"]) != 64
            or isinstance(record_descriptor.get("size_bytes"), bool)
            or not isinstance(record_descriptor.get("size_bytes"), int)
            or record_descriptor["size_bytes"] <= 0
        ):
            raise contracts.ContractError("static decision index is invalid")
        record_literal = root / record_descriptor["path"]
        record_path = record_literal.resolve()
        try:
            record_path.relative_to(root.resolve())
        except ValueError as error:
            raise contracts.ContractError(
                "static decision record escaped its batch root"
            ) from error
        if (
            record_literal.is_symlink()
            or not record_path.is_file()
            or record_path.stat().st_size != record_descriptor["size_bytes"]
            or _sha256_file(record_path) != record_descriptor["sha256"]
        ):
            raise contracts.ContractError("static decision record changed")
        record = contracts.load_json(record_path)
        if not isinstance(record, Mapping):
            raise contracts.ContractError("static decision record is invalid")
        review = authenticated_reviews.get(index["instance_id"])
        review_descriptor = record.get("review")
        checks = record.get("checks")
        evidence = record.get("attribute_evidence")
        expected_state = (
            "research_candidate"
            if index["decision"] == "approved_for_lod_and_binding"
            else "rejected"
        )
        expected_next_gate = (
            "lod_then_species_rig_binding"
            if index["decision"] == "approved_for_lod_and_binding"
            else "stop"
        )
        if (
            record.get("schema") != "avengine_controlled_animal_static_decision_v1"
            or record.get("instance_id") != index["instance_id"]
            or record.get("decision") != index["decision"]
            or record.get("decision_sha256") != index["decision_sha256"]
            or record.get("decision_sha256")
            != _hash_without(record, "decision_sha256")
            or review is None
            or not isinstance(review_descriptor, Mapping)
            or set(review_descriptor) != {"path", "sha256", "size_bytes"}
            or Path(str(review_descriptor.get("path", ""))).resolve()
            != review["path"].resolve()
            or review_descriptor.get("sha256") != _sha256_file(review["path"])
            or review_descriptor.get("size_bytes") != review["path"].stat().st_size
            or record.get("review_sha256")
            != review["payload"].get("review_sha256")
            or record.get("state_classification") != expected_state
            or record.get("formal_dataset_registration_authorized") is not False
            or record.get("next_gate") != expected_next_gate
            or not isinstance(checks, Mapping)
            or set(checks) != static_decisions.CHECK_FIELDS
            or any(not isinstance(value, bool) for value in checks.values())
            or (
                index["decision"] == "approved_for_lod_and_binding"
                and not all(checks.values())
            )
            or (
                index["decision"] == "rejected"
                and all(checks.values())
            )
            or not isinstance(evidence, Mapping)
            or set(evidence) != set(review["payload"]["sampled_attributes"])
            or evidence.get(
                review["payload"]["target_physical_profile"]["control_attribute"]
            )
            != "deferred_to_metric_3d"
            or (
                index["decision"] == "approved_for_lod_and_binding"
                and any(
                    status != "passed_static_visual"
                    for attribute, status in evidence.items()
                    if attribute
                    != review["payload"]["target_physical_profile"][
                        "control_attribute"
                    ]
                )
            )
        ):
            raise contracts.ContractError("static decision identity/hash changed")
        records[index["instance_id"]] = {
            "payload": record,
            "path": record_path,
            "static_review": review,
        }
    approved_count = sum(
        value["payload"]["decision"] == "approved_for_lod_and_binding"
        for value in records.values()
    )
    if (
        len(records) != payload["decision_count"]
        or set(records) != set(authenticated_reviews)
        or payload.get("approved_count") != approved_count
        or payload.get("rejected_count") != len(records) - approved_count
    ):
        raise contracts.ContractError("duplicate static decision records")
    return path, payload, records


def _verify_frozen_source_file(
    input_dir: Path, name: str, descriptor: Any
) -> Path:
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != {"path", "sha256", "size_bytes"}
        or not isinstance(descriptor.get("path"), str)
        or not Path(descriptor["path"]).is_absolute()
        or not isinstance(descriptor.get("sha256"), str)
        or len(descriptor["sha256"]) != 64
        or isinstance(descriptor.get("size_bytes"), bool)
        or not isinstance(descriptor.get("size_bytes"), int)
        or descriptor["size_bytes"] <= 0
    ):
        raise contracts.ContractError(f"frozen preflight descriptor is invalid: {name}")
    literal = Path(descriptor["path"])
    path = literal.resolve()
    if path != (input_dir / name).resolve():
        raise contracts.ContractError(
            f"frozen preflight source path changed: {name}"
        )
    if (
        literal.is_symlink()
        or not path.is_file()
        or path.stat().st_size != descriptor["size_bytes"]
        or _sha256_file(path) != descriptor["sha256"]
    ):
        raise contracts.ContractError(f"frozen preflight source file changed: {name}")
    return path


def _load_frozen_historical_preflight(
    preflight_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reauthenticate a sealed older preflight without regenerating it.

    Historical evidence is accepted only when every field that existed at
    publication still equals the corresponding current deterministic rebuild.
    Current rebuild fields that were not present historically may be additive;
    no historical bytes are ever rewritten or silently normalized.
    """

    literal = Path(preflight_path).absolute()
    path = literal.resolve()
    if literal.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise contracts.ContractError(f"preflight is missing or unsafe: {path}")
    preflight = preparation.validate_execution_preflight(contracts.load_json(path))
    expected_preflight_fields = {
        "schema",
        "source_bundle",
        "artifact_roots",
        "profile_artifact_authentication",
        "routes",
        "execution_summary",
        "automatic_checks",
        "preflight_sha256",
    }
    if set(preflight) != expected_preflight_fields:
        raise contracts.ContractError("frozen preflight fields are invalid")

    source_bundle = preflight.get("source_bundle")
    expected_source_fields = {
        "input_dir",
        "files",
        "profile_snapshot_sha256",
        "request_batch_id",
        "request_batch_sha256",
        "execution_jobs_sha256",
        "generation_plan_id",
        "generation_plan_sha256",
        "qa_pair_plan_sha256",
        "profile_count",
        "request_count",
        "planned_job_count",
    }
    if (
        not isinstance(source_bundle, Mapping)
        or set(source_bundle) != expected_source_fields
        or not isinstance(source_bundle.get("input_dir"), str)
        or not Path(source_bundle["input_dir"]).is_absolute()
    ):
        raise contracts.ContractError("frozen preflight source bundle is invalid")
    input_dir = Path(source_bundle["input_dir"]).resolve()
    if not input_dir.is_dir():
        raise contracts.ContractError("frozen preflight input directory is missing")
    descriptors = source_bundle.get("files")
    if (
        not isinstance(descriptors, Mapping)
        or set(descriptors) != set(preparation.REQUIRED_INPUT_FILES)
    ):
        raise contracts.ContractError("frozen preflight source file set changed")
    source_paths = {
        name: _verify_frozen_source_file(input_dir, name, descriptors[name])
        for name in preparation.REQUIRED_INPUT_FILES
    }
    bundle = {
        name: contracts.load_json(source_path)
        for name, source_path in source_paths.items()
    }
    if any(not isinstance(value, dict) for value in bundle.values()):
        raise contracts.ContractError("frozen preflight source must contain JSON objects")

    roots_value = preflight.get("artifact_roots")
    if (
        not isinstance(roots_value, Mapping)
        or not roots_value
        or any(
            not isinstance(root_id, str)
            or not root_id
            or not isinstance(root_path, str)
            or not Path(root_path).is_absolute()
            for root_id, root_path in roots_value.items()
        )
    ):
        raise contracts.ContractError("frozen preflight artifact roots are invalid")
    roots = {
        root_id: Path(root_path).resolve()
        for root_id, root_path in roots_value.items()
    }

    snapshot = bundle["profile_snapshot.json"]
    if (
        snapshot.get("schema") != input_builder.PROFILE_SNAPSHOT_SCHEMA
        or set(snapshot) != {"schema", "profiles", "snapshot_sha256"}
        or snapshot.get("snapshot_sha256")
        != _hash_without(snapshot, "snapshot_sha256")
        or not isinstance(snapshot.get("profiles"), list)
        or not snapshot["profiles"]
    ):
        raise contracts.ContractError("frozen profile snapshot contract/hash is invalid")
    profiles_list: list[dict[str, Any]] = []
    for entry in snapshot["profiles"]:
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "profile_schema_id",
                "profile_sha256",
                "profile",
                "artifact_authentication",
            }
        ):
            raise contracts.ContractError("frozen profile snapshot entry is invalid")
        profile = contracts.validate_attribute_profile(entry["profile"])
        if (
            entry["profile_schema_id"] != profile["profile_schema_id"]
            or entry["profile_sha256"] != contracts.profile_sha256(profile)
        ):
            raise contracts.ContractError("frozen profile identity/hash changed")
        profiles_list.append(profile)
    profiles_list.sort(key=lambda item: item["profile_schema_id"])
    profile_ids = [item["profile_schema_id"] for item in profiles_list]
    if len(profile_ids) != len(set(profile_ids)):
        raise contracts.ContractError("frozen profile snapshot contains duplicates")
    fresh_authentication = {
        profile["profile_schema_id"]: input_builder.authenticate_profile_artifacts(
            profile, roots
        )
        for profile in profiles_list
    }
    rebuilt_snapshot = input_builder.build_profile_snapshot(
        profiles_list, fresh_authentication
    )
    if contracts.canonical_json(snapshot) != contracts.canonical_json(
        rebuilt_snapshot
    ):
        raise contracts.ContractError(
            "frozen profile snapshot no longer matches authenticated artifacts"
        )

    request_batch = bundle["instance_requests.json"]
    try:
        contracts.validate_request_batch(request_batch, profiles_list)
    except contracts.ContractError as error:
        raise contracts.ContractError(
            "frozen instance requests no longer match deterministic sampling"
        ) from error
    profiles = {item["profile_schema_id"]: item for item in profiles_list}
    requests: dict[str, dict[str, Any]] = {}
    for request in request_batch["requests"]:
        profile = profiles.get(request.get("profile_schema_id"))
        if profile is None:
            raise contracts.ContractError("frozen request profile is missing")
        validated = contracts.validate_instance_request(request, profile)
        instance_id = validated["instance_id"]
        if instance_id in requests:
            raise contracts.ContractError("frozen requests contain duplicate identities")
        requests[instance_id] = validated

    execution_jobs = bundle["execution_jobs.json"]
    if (
        execution_jobs.get("schema") != input_builder.EXECUTION_JOBS_SCHEMA
        or not isinstance(execution_jobs.get("routes"), dict)
    ):
        raise contracts.ContractError("frozen execution jobs contract is invalid")
    rebuilt_jobs = input_builder.build_execution_jobs(
        request_batch, route_names=execution_jobs["routes"]
    )
    if contracts.canonical_json(execution_jobs) != contracts.canonical_json(
        rebuilt_jobs
    ):
        raise contracts.ContractError(
            "frozen execution jobs no longer match authenticated requests"
        )
    preparation._validate_planning_sidecars(
        request_batch,
        bundle["generation_plan.json"],
        bundle["qa_pair_plan.json"],
    )

    expected_source_values = {
        "profile_snapshot_sha256": snapshot["snapshot_sha256"],
        "request_batch_id": request_batch["batch_id"],
        "request_batch_sha256": request_batch["batch_sha256"],
        "execution_jobs_sha256": execution_jobs["jobs_sha256"],
        "generation_plan_id": bundle["generation_plan.json"]["plan_id"],
        "generation_plan_sha256": bundle["generation_plan.json"]["manifest_sha256"],
        "qa_pair_plan_sha256": bundle["qa_pair_plan.json"]["pair_plan_sha256"],
        "profile_count": len(profiles_list),
        "request_count": len(requests),
        "planned_job_count": execution_jobs["job_count"],
    }
    if any(
        source_bundle.get(name) != expected
        for name, expected in expected_source_values.items()
    ):
        raise contracts.ContractError("frozen preflight source identity/count changed")
    if set(preflight["routes"]) != set(execution_jobs["routes"]):
        raise contracts.ContractError(
            "frozen preflight and execution job route sets differ"
        )

    rebuilt = preparation.build_execution_preflight(input_dir, roots)
    for name in expected_preflight_fields - {
        "routes",
        "execution_summary",
        "preflight_sha256",
    }:
        if contracts.canonical_json(preflight[name]) != contracts.canonical_json(
            rebuilt[name]
        ):
            raise contracts.ContractError(
                f"frozen preflight field no longer matches current rebuild: {name}"
            )
    for route, jobs in preflight["routes"].items():
        if route not in rebuilt["routes"] or contracts.canonical_json(
            jobs
        ) != contracts.canonical_json(rebuilt["routes"][route]):
            raise contracts.ContractError(
                f"frozen preflight route no longer matches current rebuild: {route}"
            )
    historical_summary = preflight["execution_summary"]
    if (
        not isinstance(historical_summary, Mapping)
        or not set(historical_summary).issubset(rebuilt["execution_summary"])
        or any(
            rebuilt["execution_summary"][name] != value
            for name, value in historical_summary.items()
        )
    ):
        raise contracts.ContractError(
            "frozen preflight summary no longer matches current rebuild"
        )
    return preflight, requests, profiles


def load_source_contract(
    preflight_path: Path, *, frozen_historical_preflight: bool = False
):
    if frozen_historical_preflight:
        return _load_frozen_historical_preflight(preflight_path)
    preflight = preflight_tools._load_preflight(preflight_path)
    source_files = preflight["source_bundle"]["files"]
    requests_path = Path(source_files["instance_requests.json"]["path"])
    profiles_path = Path(source_files["profile_snapshot.json"]["path"])
    for name, path in (
        ("instance_requests.json", requests_path),
        ("profile_snapshot.json", profiles_path),
    ):
        expected = source_files[name]
        if (
            not path.is_file()
            or path.stat().st_size != expected["size_bytes"]
            or _sha256_file(path) != expected["sha256"]
        ):
            raise contracts.ContractError(f"preflight source file changed: {name}")
    request_batch = contracts.load_json(requests_path)
    profile_snapshot = contracts.load_json(profiles_path)
    requests = {
        request["instance_id"]: contracts.validate_request_integrity(request)
        for request in request_batch["requests"]
    }
    profiles = {
        item["profile_schema_id"]: contracts.validate_attribute_profile(
            item["profile"]
        )
        for item in profile_snapshot["profiles"]
    }
    return preflight, requests, profiles


def validate_pixal_request_identity(
    pixal_batch: Mapping[str, Any],
    pixal_inputs_manifest: Mapping[str, Any],
    requests: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind every Pixal input and attempt back to its canonical request."""

    jobs = pixal_inputs_manifest.get("jobs")
    attempts_value = pixal_batch.get("attempts")
    if (
        not isinstance(jobs, list)
        or not jobs
        or pixal_inputs_manifest.get("job_count") != len(jobs)
        or not isinstance(attempts_value, list)
        or not attempts_value
        or pixal_batch.get("job_count") != len(attempts_value)
        or pixal_batch.get("passed_count") != len(attempts_value)
        or pixal_batch.get("failed_count") != 0
        or pixal_batch.get("status") != "passed_generation_and_glb_readback"
    ):
        raise contracts.ContractError("Pixal job/attempt coverage is invalid")
    input_jobs: dict[str, Any] = {}
    for job in jobs:
        controlled = job.get("controlled_request")
        if not isinstance(controlled, Mapping):
            raise contracts.ContractError("Pixal controlled request is missing")
        instance_id = controlled.get("instance_id")
        request = requests.get(instance_id)
        if request is None or request["asset_class"] != "animal":
            raise contracts.ContractError("Pixal input has no canonical animal request")
        expected = {
            "execution_job_id": f"animal_{request['request_sha256'][:16]}",
            "instance_id": request["instance_id"],
            "request_sha256": request["request_sha256"],
            "generation_seed": int(request["generation_plan"]["generation_seed"]),
            "profile_schema_id": request["profile_schema_id"],
            "sampled_attributes": request["sampled_attributes"],
            "target_physical_profile": request["target_physical_profile"],
        }
        optional_expected = {
            "profile_sha256": request["profile_sha256"],
            "asset_class": "animal",
            "route": "flux2_pixal3d_animal_v1",
            "rig_profile": request["rig_profile"],
        }
        if any(
            contracts.canonical_json(controlled.get(name))
            != contracts.canonical_json(value)
            for name, value in expected.items()
        ) or any(
            name in controlled
            and contracts.canonical_json(controlled[name])
            != contracts.canonical_json(value)
            for name, value in optional_expected.items()
        ):
            raise contracts.ContractError(
                "Pixal input/canonical request identity changed"
            )
        if (
            instance_id in input_jobs
            or job.get("legacy_tag") != instance_id
            or job.get("seed") != expected["generation_seed"]
        ):
            raise contracts.ContractError("Pixal input identity is duplicate/invalid")
        input_jobs[instance_id] = job

    attempts: dict[str, Any] = {}
    for attempt in attempts_value:
        if not isinstance(attempt, Mapping):
            raise contracts.ContractError("Pixal attempt is invalid")
        instance_id = attempt.get("instance_id")
        request = requests.get(instance_id)
        input_job = input_jobs.get(instance_id)
        if request is None or input_job is None:
            raise contracts.ContractError("Pixal attempt lacks its frozen input/request")
        expected = {
            "execution_job_id": f"animal_{request['request_sha256'][:16]}",
            "request_sha256": request["request_sha256"],
            "profile_schema_id": request["profile_schema_id"],
            "sampled_attributes": request["sampled_attributes"],
            "target_physical_profile": request["target_physical_profile"],
            "seed": int(request["generation_plan"]["generation_seed"]),
            "attempt_ordinal": input_job["attempt_ordinal"],
            "pixal_input": input_job["reference"]["pixal_input"],
        }
        if any(
            contracts.canonical_json(attempt.get(name))
            != contracts.canonical_json(value)
            for name, value in expected.items()
        ):
            raise contracts.ContractError(
                "Pixal attempt/canonical request identity changed"
            )
        if instance_id in attempts:
            raise contracts.ContractError("duplicate Pixal attempt identity")
        attempts[instance_id] = attempt
    if set(input_jobs) != set(attempts):
        raise contracts.ContractError("Pixal input/attempt identity coverage differs")
    return input_jobs, attempts


def approved_attempt_ids(
    decisions: Mapping[str, Any], attempts: Mapping[str, Any]
) -> set[str]:
    """Require a decision for every Pixal attempt and return the approved subset."""
    if set(decisions) != set(attempts):
        raise contracts.ContractError(
            "static decision/Pixal attempt coverage differs"
        )
    approved = {
        instance_id
        for instance_id, value in decisions.items()
        if value["payload"]["decision"] == "approved_for_lod_and_binding"
    }
    if not approved:
        raise contracts.ContractError(
            "source registry requires at least one approved static candidate"
        )
    return approved


def register(
    preflight_path: Path,
    pixal_batch_path: Path,
    decision_batch_path: Path,
    output_root: Path,
    *,
    frozen_historical_preflight: bool = False,
) -> Path:
    preflight_literal = Path(preflight_path).absolute()
    preflight, requests, profiles = load_source_contract(
        preflight_literal,
        frozen_historical_preflight=frozen_historical_preflight,
    )
    preflight_path = preflight_literal.resolve()
    pixal_batch_path = Path(pixal_batch_path).resolve()
    if pixal_batch_path.is_symlink() or not pixal_batch_path.is_file():
        raise contracts.ContractError("Pixal batch is missing")
    pixal_payload = contracts.load_json(pixal_batch_path)
    if (
        pixal_payload.get("schema") != pixal_runner.BATCH_SCHEMA
        or pixal_payload.get("batch_sha256")
        != _hash_without(pixal_payload, "batch_sha256")
        or pixal_payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("Pixal batch contract/hash is invalid")
    pixal_batch = pixal_payload
    pixal_inputs_path = Path(pixal_batch["pixal_inputs"]["path"]).resolve()
    if (
        not pixal_inputs_path.is_file()
        or _sha256_file(pixal_inputs_path)
        != pixal_batch["pixal_inputs"]["sha256"]
    ):
        raise contracts.ContractError("Pixal inputs manifest changed")
    _pixal_inputs_path, pixal_inputs_manifest = pixal_runner.load_pixal_inputs(
        pixal_inputs_path
    )
    if (
        pixal_inputs_manifest.get("manifest_sha256")
        != pixal_batch["pixal_inputs"].get("manifest_sha256")
    ):
        raise contracts.ContractError("Pixal input manifest identity changed")
    input_jobs, attempts = validate_pixal_request_identity(
        pixal_batch, pixal_inputs_manifest, requests
    )
    decision_batch_path, decision_batch, decisions = load_decision_batch(
        decision_batch_path
    )
    approved_ids = approved_attempt_ids(decisions, attempts)
    model_licenses = license_records()

    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        entries = []
        for instance_id in sorted(approved_ids):
            request = requests.get(instance_id)
            if request is None or request["asset_class"] != "animal":
                raise contracts.ContractError("approved animal request is missing")
            profile = profiles[request["profile_schema_id"]]
            attempt = attempts[instance_id]
            decision = decisions[instance_id]
            input_job = input_jobs.get(instance_id)
            if input_job is None:
                raise contracts.ContractError(
                    "approved instance lacks a Pixal input job"
                )
            if (
                attempt["request_sha256"] != request["request_sha256"]
                or attempt["sampled_attributes"] != request["sampled_attributes"]
            ):
                raise contracts.ContractError("animal request/attempt identity changed")
            pixal_root = pixal_batch_path.parent
            output_path = (pixal_root / attempt["output"]["path"]).resolve()
            attempt_manifest_path = (
                pixal_root / attempt["attempt_manifest"]["path"]
            ).resolve()
            if (
                not output_path.is_file()
                or output_path.stat().st_size != attempt["output"]["size_bytes"]
                or _sha256_file(output_path) != attempt["output"]["sha256"]
                or not attempt_manifest_path.is_file()
                or attempt_manifest_path.stat().st_size
                != attempt["attempt_manifest"]["size_bytes"]
                or _sha256_file(attempt_manifest_path)
                != attempt["attempt_manifest"]["sha256"]
            ):
                raise contracts.ContractError("Pixal attempt artifacts changed")
            attempt_manifest = contracts.load_json(attempt_manifest_path)
            if (
                attempt_manifest.get("controlled_request")
                != input_job["controlled_request"]
                or attempt_manifest.get("output", {}).get("sha256")
                != attempt["output"]["sha256"]
            ):
                raise contracts.ContractError(
                    "Pixal attempt manifest/request binding changed"
                )
            static_review_path = decision["static_review"]["path"]
            static_review = decision["static_review"]["payload"]
            if (
                static_review.get("instance_id") != request["instance_id"]
                or static_review.get("request_sha256") != request["request_sha256"]
                or static_review.get("profile_schema_id")
                != request["profile_schema_id"]
                or contracts.canonical_json(static_review.get("sampled_attributes"))
                != contracts.canonical_json(request["sampled_attributes"])
                or contracts.canonical_json(
                    static_review.get("target_physical_profile")
                )
                != contracts.canonical_json(request["target_physical_profile"])
                or static_review.get("pixal_output", {}).get("sha256")
                != attempt["output"]["sha256"]
            ):
                raise contracts.ContractError(
                    "static review/canonical request identity changed"
                )
            contact_path = (
                static_review_path.parents[1] / static_review["contact_sheet"]["path"]
            ).resolve()
            pixal_input_path = Path(attempt["pixal_input"]["path"]).resolve()
            candidate_path = Path(input_job["reference"]["source"]["path"]).resolve()
            artifacts = {
                "flux2_candidate_image": spear_artifact(candidate_path),
                "pixal_input_rgba": spear_artifact(pixal_input_path),
                "pixal_inputs_manifest": spear_artifact(pixal_inputs_path),
                "pixal_raw_glb": spear_artifact(output_path),
                "pixal_attempt_manifest": spear_artifact(attempt_manifest_path),
                "static_review_manifest": spear_artifact(static_review_path),
                "static_contact_sheet": spear_artifact(contact_path),
                "static_decision": spear_artifact(decision["path"]),
            }
            source_asset = contracts.build_source_asset_v2(
                request,
                artifacts=artifacts,
                physical_measurements={"status": "pending"},
                provenance={
                    "attempt_id": f"static_{attempt['execution_job_id']}",
                    "request_sha256": request["request_sha256"],
                    # This must remain byte-for-byte equivalent to the immutable
                    # request contract. ISNet preprocessing is authenticated by
                    # pixal_inputs_manifest instead of being injected here.
                    "models": copy.deepcopy(
                        request["generation_plan"]["model_revisions"]
                    ),
                },
                rights={
                    "status": "review_required",
                    "licenses": copy.deepcopy(model_licenses),
                    "blockers": [
                        "legacy_reference_provenance_unknown",
                        "physical_target_reference_provisional",
                        "pixal_research_dependency_export_review_required",
                        "dino_snapshot_origin_review_required",
                    ],
                },
                qa={
                    "reference_2d": "passed",
                    "static_mesh": "passed",
                    "binding": "pending",
                    "walking": "pending",
                    "idle": "pending",
                    "ue_import_readback": "pending",
                    "apartment_media": "pending",
                    "audio": "pending",
                },
                state_classification="research_candidate",
            )
            contracts.validate_source_asset_v2(
                source_asset, request=request, profile=profile
            )
            destination = staging / "source_assets" / f"{instance_id}.json"
            contracts.write_json_no_replace(destination, source_asset)
            entries.append(
                {
                    "asset_id": instance_id,
                    "profile_schema_id": request["profile_schema_id"],
                    "request_sha256": request["request_sha256"],
                    "sampled_attributes": request["sampled_attributes"],
                    "attribute_evidence": decision["payload"]["attribute_evidence"],
                    "source_asset": {
                        "path": destination.relative_to(staging).as_posix(),
                        "sha256": _sha256_file(destination),
                        "size_bytes": destination.stat().st_size,
                    },
                    "state_classification": "research_candidate",
                    "next_gate": "lod_then_species_rig_binding",
                }
            )
        registry: dict[str, Any] = {
            "schema": REGISTRY_SCHEMA,
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "preflight": {
                "path": str(preflight_path),
                "sha256": _sha256_file(preflight_path),
                "preflight_sha256": preflight["preflight_sha256"],
                "validation_mode": (
                    "frozen_historical_preflight_v1"
                    if frozen_historical_preflight
                    else "current_exact_rebuild"
                ),
            },
            "pixal_batch": {
                "path": str(pixal_batch_path),
                "sha256": _sha256_file(pixal_batch_path),
                "batch_sha256": pixal_batch["batch_sha256"],
            },
            "static_decision_batch": {
                "path": str(decision_batch_path),
                "sha256": _sha256_file(decision_batch_path),
                "decision_batch_sha256": decision_batch["decision_batch_sha256"],
            },
            "source_asset_count": len(entries),
            "source_assets": entries,
            "automatic_checks": {
                "all_requests_reauthenticated": True,
                "all_pixal_input_attempt_request_identities_reauthenticated": True,
                "all_pixal_outputs_reauthenticated": True,
                "all_static_decisions_reauthenticated": True,
                "all_source_asset_v2_validated_against_request_and_profile": True,
                "all_physical_measurements_pending": True,
                "all_animation_ue_audio_qa_pending": True,
                "all_rights_blockers_preserved": True,
                "overall": "passed",
            },
        }
        registry["registry_sha256"] = _hash_without(registry, "registry_sha256")
        contracts.write_json_no_replace(staging / "registry_manifest.json", registry)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("animal registry output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "registry_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--pixal-batch", required=True, type=Path)
    parser.add_argument("--static-decision-batch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--frozen-historical-preflight",
        action="store_true",
        help=(
            "Validate an immutable older preflight as frozen evidence while "
            "allowing only additive fields in the current deterministic rebuild."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = register(
            args.preflight,
            args.pixal_batch,
            args.static_decision_batch,
            args.output_root,
            frozen_historical_preflight=args.frozen_historical_preflight,
        )
        payload = contracts.load_json(manifest)
    except (contracts.ContractError, OSError) as error:
        print(f"CONTROLLED_ANIMAL_SOURCE_ASSET_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_SOURCE_ASSET_OK "
        f"assets={payload['source_asset_count']} output={manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
