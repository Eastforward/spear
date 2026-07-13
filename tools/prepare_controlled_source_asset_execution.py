#!/usr/bin/env python3
"""Reauthenticate normalized controlled-asset inputs before model execution.

This command is deliberately model-free.  It proves that the profile snapshot,
request batch, route jobs, planned QA, and generation manifest still form the
exact bundle produced by ``build_controlled_source_asset_inputs.py``.  It then
deduplicates deterministic Rocketbox material requests while retaining every
consumer request ID.  Animal generation requests remain one job per request.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts


PREFLIGHT_SCHEMA = "avengine_controlled_execution_preflight_v1"
REQUIRED_INPUT_FILES = (
    "profile_snapshot.json",
    "instance_requests.json",
    "execution_jobs.json",
    "qa_pair_plan.json",
    "generation_plan.json",
)
KNOWN_ROCKETBOX_RUNTIME_ASSET_IDS = {
    "rocketbox_adults_male_adult_01": "rocketbox_male_adult_01",
}


def default_artifact_roots() -> dict[str, Path]:
    return input_builder.default_artifact_roots()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], *keys: str) -> str:
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key not in keys}
    return _json_sha256(payload)


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _load_required_bundle(input_dir: Path) -> tuple[Path, dict[str, Any]]:
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise contracts.ContractError(f"input directory does not exist: {input_dir}")
    payloads: dict[str, Any] = {}
    for name in REQUIRED_INPUT_FILES:
        path = input_dir / name
        if path.parent != input_dir or path.is_symlink() or not path.is_file():
            raise contracts.ContractError(f"required input is missing or unsafe: {path}")
        payload = contracts.load_json(path)
        if not isinstance(payload, dict):
            raise contracts.ContractError(f"required input must be a JSON object: {path}")
        payloads[name] = payload
    return input_dir, payloads


def _validate_planning_sidecars(
    request_batch: Mapping[str, Any],
    generation_plan: Mapping[str, Any],
    pair_plan: Mapping[str, Any],
) -> None:
    if generation_plan.get("schema") != contracts.GENERATION_PLAN_SCHEMA:
        raise contracts.ContractError("generation_plan.json schema is invalid")
    if generation_plan.get("manifest_sha256") != contracts.manifest_sha256(
        generation_plan
    ):
        raise contracts.ContractError("generation_plan.json hash is invalid")
    if pair_plan.get("schema") != contracts.PAIR_PLAN_SCHEMA:
        raise contracts.ContractError("qa_pair_plan.json schema is invalid")
    if pair_plan.get("pair_plan_sha256") != _hash_without(
        pair_plan, "pair_plan_sha256"
    ):
        raise contracts.ContractError("qa_pair_plan.json hash is invalid")
    if (
        pair_plan.get("request_batch_id") != request_batch.get("batch_id")
        or pair_plan.get("request_batch_sha256") != request_batch.get("batch_sha256")
        or pair_plan.get("generation_plan_id") != generation_plan.get("plan_id")
        or pair_plan.get("generation_plan_sha256")
        != generation_plan.get("manifest_sha256")
        or pair_plan.get("pairs") != generation_plan.get("qa_pairs")
        or pair_plan.get("pair_count") != len(generation_plan.get("qa_pairs", []))
        or pair_plan.get("evidence_state") != "planned"
        or pair_plan.get("answer_policy")
        != "pending_until_source_asset_v2_visual_qa_passes"
    ):
        raise contracts.ContractError("planned QA sidecar does not match the generation plan")


def _resolve_artifact(
    artifact: Mapping[str, Any], artifact_roots: Mapping[str, Path]
) -> dict[str, Any]:
    root_id = artifact["root_id"]
    if root_id not in artifact_roots:
        raise contracts.ContractError(f"unknown artifact root in preflight: {root_id}")
    root = Path(artifact_roots[root_id]).resolve()
    relative = Path(artifact["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise contracts.ContractError("artifact path must remain root relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise contracts.ContractError("artifact path escapes its named root") from error
    return {
        **copy.deepcopy(dict(artifact)),
        "resolved_path": str(resolved),
    }


def _consumer(request: Mapping[str, Any]) -> dict[str, str]:
    return {
        "instance_id": str(request["instance_id"]),
        "request_sha256": str(request["request_sha256"]),
    }


def _animal_execution_job(
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    artifact_roots: Mapping[str, Path],
) -> dict[str, Any]:
    plan = request["generation_plan"]
    return {
        "execution_job_id": f"animal_{request['request_sha256'][:16]}",
        "profile_schema_id": request["profile_schema_id"],
        "profile_sha256": request["profile_sha256"],
        "lineage_group_id": request["lineage_group_id"],
        "state_classification": request["state_classification"],
        "taxonomy": copy.deepcopy(request["taxonomy"]),
        "fixed_attributes": copy.deepcopy(request["fixed_attributes"]),
        "sampled_attributes": copy.deepcopy(request["sampled_attributes"]),
        "consumer_requests": [_consumer(request)],
        "reference": _resolve_artifact(
            profile["base_template"]["artifact"], artifact_roots
        ),
        "generation_plan": copy.deepcopy(plan),
        "target_physical_profile": copy.deepcopy(request["target_physical_profile"]),
        "rig_profile": copy.deepcopy(request["rig_profile"]),
        "acoustic_profile": copy.deepcopy(request["acoustic_profile"]),
        "execution_gate": {
            "before_flux2": "authenticated_preflight_passed",
            "before_pixal3d": "approved_2d_review_for_exact_candidate_sha256",
            "before_source_asset_v2": "all_required_static_animation_ue_audio_qa_passed",
        },
    }


def _material_variant_core(
    request: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    plan = copy.deepcopy(request["generation_plan"])
    plan.pop("generation_seed", None)
    return {
        "profile_schema_id": request["profile_schema_id"],
        "profile_sha256": request["profile_sha256"],
        "lineage_group_id": request["lineage_group_id"],
        "base_avatar_id": plan["base_avatar_id"],
        "base_template": copy.deepcopy(plan["base_template"]),
        "sampled_attributes": copy.deepcopy(request["sampled_attributes"]),
        "edits": copy.deepcopy(plan["edits"]),
        "flux_texture_detail": copy.deepcopy(plan["flux_texture_detail"]),
        "geometry_changes_allowed": plan["geometry_changes_allowed"],
        "profile_revision": profile["profile_revision"],
    }


def _material_execution_jobs(
    requests: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, Mapping[str, Any]],
    artifact_roots: Mapping[str, Path],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for request in requests:
        profile = profiles[request["profile_schema_id"]]
        core = _material_variant_core(request, profile)
        variant_key = _json_sha256(core)
        if variant_key not in grouped:
            grouped[variant_key] = {
                "execution_job_id": f"material_{variant_key[:16]}",
                "variant_key": variant_key,
                "variant_id": f"controlled_material_{variant_key[:12]}",
                "profile_schema_id": request["profile_schema_id"],
                "profile_sha256": request["profile_sha256"],
                "lineage_group_id": request["lineage_group_id"],
                "state_classification": request["state_classification"],
                "base_avatar_id": core["base_avatar_id"],
                "sampled_attributes": copy.deepcopy(request["sampled_attributes"]),
                "consumer_requests": [],
                "base_template": {
                    **copy.deepcopy(core["base_template"]),
                    "artifact": _resolve_artifact(
                        core["base_template"]["artifact"], artifact_roots
                    ),
                },
                "material_edit_plan": {
                    key: value
                    for key, value in copy.deepcopy(request["generation_plan"]).items()
                    if key != "generation_seed"
                },
                "target_physical_profile": copy.deepcopy(
                    request["target_physical_profile"]
                ),
                "rig_profile": copy.deepcopy(request["rig_profile"]),
                "acoustic_profile": copy.deepcopy(request["acoustic_profile"]),
                "execution_gate": {
                    "before_material_edit": "authenticated_preflight_passed",
                    "before_runtime_build": "mask_and_non_target_pixel_qa_passed",
                    "before_source_asset_v2": "all_required_runtime_ue_audio_qa_passed",
                },
            }
        else:
            existing = grouped[variant_key]
            if (
                existing["profile_schema_id"] != request["profile_schema_id"]
                or existing["sampled_attributes"] != request["sampled_attributes"]
                or existing["material_edit_plan"]
                != {
                    key: value
                    for key, value in request["generation_plan"].items()
                    if key != "generation_seed"
                }
            ):
                raise contracts.ContractError("material variant hash collision")
        grouped[variant_key]["consumer_requests"].append(_consumer(request))
    for job in grouped.values():
        job["consumer_requests"] = sorted(
            job["consumer_requests"], key=lambda item: item["instance_id"]
        )
    return sorted(grouped.values(), key=lambda item: item["variant_key"])


def preflight_sha256(preflight: Mapping[str, Any]) -> str:
    return _hash_without(preflight, "preflight_sha256")


def build_execution_preflight(
    input_dir: Path,
    artifact_roots: Mapping[str, Path],
) -> dict[str, Any]:
    input_dir, bundle = _load_required_bundle(input_dir)
    snapshot = bundle["profile_snapshot.json"]
    request_batch = bundle["instance_requests.json"]
    execution_jobs = bundle["execution_jobs.json"]
    pair_plan = bundle["qa_pair_plan.json"]
    generation_plan = bundle["generation_plan.json"]

    if snapshot.get("schema") != input_builder.PROFILE_SNAPSHOT_SCHEMA:
        raise contracts.ContractError("profile_snapshot.json schema is invalid")
    if set(snapshot) != {"schema", "profiles", "snapshot_sha256"}:
        raise contracts.ContractError("profile_snapshot.json fields are invalid")
    entries = snapshot.get("profiles")
    if not isinstance(entries, list) or not entries:
        raise contracts.ContractError("profile snapshot contains no profiles")
    profiles: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "profile_schema_id",
            "profile_sha256",
            "profile",
            "artifact_authentication",
        }:
            raise contracts.ContractError("profile snapshot entry fields are invalid")
        profile = contracts.validate_attribute_profile(entry["profile"])
        if (
            entry["profile_schema_id"] != profile["profile_schema_id"]
            or entry["profile_sha256"] != contracts.profile_sha256(profile)
        ):
            raise contracts.ContractError("profile snapshot identity/hash mismatch")
        profiles.append(profile)
    profiles.sort(key=lambda item: item["profile_schema_id"])
    fresh_authentication = {
        profile["profile_schema_id"]: input_builder.authenticate_profile_artifacts(
            profile, dict(artifact_roots)
        )
        for profile in profiles
    }
    rebuilt_snapshot = input_builder.build_profile_snapshot(
        profiles, fresh_authentication
    )
    if contracts.canonical_json(snapshot) != contracts.canonical_json(rebuilt_snapshot):
        raise contracts.ContractError(
            "profile_snapshot.json does not exactly match authenticated profiles"
        )

    if request_batch.get("schema") != contracts.REQUEST_BATCH_SCHEMA:
        raise contracts.ContractError("instance_requests.json schema is invalid")
    sampler = request_batch.get("sampler")
    if not isinstance(sampler, dict):
        raise contracts.ContractError("request batch sampler is invalid")
    rebuilt_batch = contracts.build_request_batch(
        profiles,
        count_per_profile=sampler.get("count_per_profile"),
        batch_seed=sampler.get("batch_seed"),
    )
    if contracts.canonical_json(request_batch) != contracts.canonical_json(rebuilt_batch):
        raise contracts.ContractError(
            "instance_requests.json does not exactly match deterministic sampling"
        )

    if execution_jobs.get("schema") != input_builder.EXECUTION_JOBS_SCHEMA:
        raise contracts.ContractError("execution_jobs.json schema is invalid")
    rebuilt_jobs = input_builder.build_execution_jobs(request_batch)
    if contracts.canonical_json(execution_jobs) != contracts.canonical_json(rebuilt_jobs):
        raise contracts.ContractError(
            "execution_jobs.json does not exactly match the authenticated request batch"
        )
    _validate_planning_sidecars(request_batch, generation_plan, pair_plan)

    profiles_by_id = {profile["profile_schema_id"]: profile for profile in profiles}
    animal_requests: list[dict[str, Any]] = []
    material_requests: list[dict[str, Any]] = []
    for request in request_batch["requests"]:
        profile = profiles_by_id[request["profile_schema_id"]]
        validated = contracts.validate_instance_request(request, profile)
        route = validated["generation_plan"]["route"]
        if route == "flux2_pixal3d_animal_v1":
            animal_requests.append(validated)
        elif route == "rocketbox_material_v1":
            material_requests.append(validated)
        else:  # pragma: no cover - profile validation currently makes this unreachable.
            raise contracts.ContractError(f"unsupported execution route: {route}")

    animal_jobs = [
        _animal_execution_job(request, profiles_by_id[request["profile_schema_id"]], artifact_roots)
        for request in animal_requests
    ]
    animal_jobs.sort(key=lambda item: item["execution_job_id"])
    material_jobs = _material_execution_jobs(
        material_requests, profiles_by_id, artifact_roots
    )
    summary = {
        "animal_job_count": len(animal_jobs),
        "deterministic_material_job_count": len(material_jobs),
        "material_request_count": len(material_requests),
        "material_requests_deduplicated": len(material_requests) - len(material_jobs),
        "unique_execution_job_count": len(animal_jobs) + len(material_jobs),
    }
    preflight: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "source_bundle": {
            "input_dir": str(input_dir),
            "files": {
                name: _file_record(input_dir / name) for name in REQUIRED_INPUT_FILES
            },
            "profile_snapshot_sha256": snapshot["snapshot_sha256"],
            "request_batch_id": request_batch["batch_id"],
            "request_batch_sha256": request_batch["batch_sha256"],
            "execution_jobs_sha256": execution_jobs["jobs_sha256"],
            "generation_plan_id": generation_plan["plan_id"],
            "generation_plan_sha256": generation_plan["manifest_sha256"],
            "qa_pair_plan_sha256": pair_plan["pair_plan_sha256"],
            "profile_count": len(profiles),
            "request_count": len(request_batch["requests"]),
            "planned_job_count": execution_jobs["job_count"],
        },
        "artifact_roots": {
            key: str(Path(value).resolve()) for key, value in sorted(artifact_roots.items())
        },
        "profile_artifact_authentication": fresh_authentication,
        "routes": {
            "flux2_pixal3d_animal_v1": animal_jobs,
            "rocketbox_material_v1": material_jobs,
        },
        "execution_summary": summary,
        "automatic_checks": {
            "all_profile_artifacts_authenticated": all(
                record["status"] == "passed"
                for records in fresh_authentication.values()
                for record in records
            ),
            "all_requests_profile_validated": True,
            "execution_jobs_exactly_rebuilt": True,
            "material_jobs_deduplicated_by_absolute_plan": True,
            "overall": "passed",
        },
    }
    preflight["preflight_sha256"] = preflight_sha256(preflight)
    return preflight


def validate_execution_preflight(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != PREFLIGHT_SCHEMA:
        raise contracts.ContractError(f"preflight schema must be {PREFLIGHT_SCHEMA}")
    if value.get("preflight_sha256") != preflight_sha256(value):
        raise contracts.ContractError("execution preflight hash mismatch")
    checks = value.get("automatic_checks")
    if not isinstance(checks, dict) or checks.get("overall") != "passed":
        raise contracts.ContractError("execution preflight automatic checks did not pass")
    routes = value.get("routes")
    if not isinstance(routes, dict) or set(routes) != set(contracts.ROUTES):
        raise contracts.ContractError("execution preflight routes are invalid")
    return copy.deepcopy(value)


def _runtime_asset_id(base_avatar_id: str) -> str:
    try:
        return KNOWN_ROCKETBOX_RUNTIME_ASSET_IDS[base_avatar_id]
    except KeyError as error:
        raise contracts.ContractError(
            f"no audited native runtime adapter for {base_avatar_id}"
        ) from error


def build_rocketbox_runtime_variant_request(
    job: Mapping[str, Any], texture: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(job, Mapping) or not isinstance(texture, Mapping):
        raise contracts.ContractError("material job and texture record must be objects")
    for key in ("path", "sha256", "size_bytes"):
        if key not in texture:
            raise contracts.ContractError(f"texture record is missing {key}")
    return {
        "schema_version": "rocketbox_native_body_color_variant_v1",
        "asset_id": _runtime_asset_id(str(job["base_avatar_id"])),
        "variant_id": str(job["variant_id"]),
        "tag": f"{_runtime_asset_id(str(job['base_avatar_id']))}_{job['variant_id']}",
        "target_image_name": "m002_body_color",
        "body_color_texture_sha256": texture["sha256"],
        "body_color_texture_size_bytes": texture["size_bytes"],
        "controlled_source": {
            "profile_schema_id": job["profile_schema_id"],
            "profile_sha256": job["profile_sha256"],
            "lineage_group_id": job["lineage_group_id"],
            "variant_key": job["variant_key"],
            "sampled_attributes": copy.deepcopy(job["sampled_attributes"]),
            "consumer_requests": copy.deepcopy(job["consumer_requests"]),
            "material_edit_plan": copy.deepcopy(job["material_edit_plan"]),
            "texture_path": texture["path"],
        },
    }


def publish_preflight(output_dir: Path, preflight: Mapping[str, Any]) -> Path:
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise contracts.ContractError(
            f"refusing to replace existing output directory: {output_dir}"
        ) from error
    try:
        path = contracts.write_json_no_replace(
            output_dir / "execution_preflight.json", preflight
        )
        observed = validate_execution_preflight(contracts.load_json(path))
        if contracts.canonical_json(observed) != contracts.canonical_json(preflight):
            raise contracts.ContractError("published preflight readback mismatch")
        return path
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument(
        "--artifact-root", action="append", default=[], metavar="ROOT_ID=PATH"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        roots = input_builder.parse_artifact_roots(args.artifact_root)
        preflight = build_execution_preflight(args.input_dir, roots)
        path = publish_preflight(args.output_dir, preflight)
    except contracts.ContractError as error:
        print(f"CONTROLLED_EXECUTION_PREFLIGHT_FAILED {error}", file=sys.stderr)
        return 2
    summary = preflight["execution_summary"]
    print(
        "CONTROLLED_EXECUTION_PREFLIGHT_OK "
        f"animal={summary['animal_job_count']} "
        f"material_unique={summary['deterministic_material_job_count']} "
        f"material_requests={summary['material_request_count']} "
        f"output={path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
