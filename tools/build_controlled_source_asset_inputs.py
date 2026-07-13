#!/usr/bin/env python3
"""Compile controlled attribute profiles into immutable generation inputs.

The command only prepares normalized inputs.  It does not invoke FLUX.2,
Pixal3D, Blender, Unreal, or an audio renderer.  Planned QA answers remain
explicitly pending until matching ``source_asset_v2`` records exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts


PROFILE_SNAPSHOT_SCHEMA = "avengine_attribute_profile_snapshot_v1"
EXECUTION_JOBS_SCHEMA = "avengine_controlled_execution_jobs_v1"


def default_artifact_roots() -> dict[str, Path]:
    spear_root = Path(__file__).resolve().parents[1]
    return {
        "spear_repo": spear_root,
        "rocketbox_0943055": Path(
            os.environ.get(
                "AVENGINE_ROCKETBOX_ROOT",
                "/data/datasets/rocketbox/Microsoft-Rocketbox",
            )
        ),
    }


def parse_artifact_roots(values: Sequence[str]) -> dict[str, Path]:
    roots = default_artifact_roots()
    for value in values:
        if "=" not in value:
            raise contracts.ContractError(
                f"artifact root must be ROOT_ID=PATH: {value!r}"
            )
        root_id, raw_path = value.split("=", 1)
        if not root_id or not raw_path:
            raise contracts.ContractError(
                f"artifact root must be ROOT_ID=PATH: {value!r}"
            )
        # Validate root_id using a minimal synthetic artifact contract through
        # the public profile validator would be excessive; match the same ID
        # grammar here.
        if (
            not (root_id[0].islower() or root_id[0].isdigit())
            or not all(
                char.islower() or char.isdigit() or char in "_.-"
                for char in root_id
            )
        ):
            raise contracts.ContractError(f"invalid artifact root ID: {root_id!r}")
        roots[root_id] = Path(raw_path).resolve()
    return roots


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_artifacts(profile: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    artifacts = [("base_template", profile["base_template"]["artifact"])]
    physical = profile["target_physical_profiles"]["reference_provenance"]["artifact"]
    if physical is not None:
        artifacts.append(("target_physical_reference", physical))
    if profile["generation_contract"]["route"] == "rocketbox_material_v1":
        for attribute, binding in profile["generation_contract"]["material_bindings"].items():
            artifacts.append((f"material_mask_registry:{attribute}", binding["mask_registry"]))
    return artifacts


def authenticate_artifact_record(
    artifact: dict[str, Any],
    artifact_roots: dict[str, Path],
    *,
    role: str,
    owner: str,
) -> dict[str, Any]:
    root_id = artifact["root_id"]
    if root_id not in artifact_roots:
        raise contracts.ContractError(
            f"{owner} uses unknown artifact root {root_id}"
        )
    relative = Path(artifact["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise contracts.ContractError(
            f"artifact must be a safe root-relative path: {artifact['path']}"
        )
    root = artifact_roots[root_id].resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise contracts.ContractError(
            f"artifact escapes root {root_id}: {artifact['path']}"
        ) from error
    if not path.is_file():
        raise contracts.ContractError(f"artifact is missing: {path}")
    observed_size = path.stat().st_size
    if observed_size != artifact["size_bytes"]:
        raise contracts.ContractError(
            f"artifact size mismatch for {path}: "
            f"expected={artifact['size_bytes']} observed={observed_size}"
        )
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != artifact["sha256"]:
        raise contracts.ContractError(
            f"artifact hash mismatch for {path}: "
            f"expected={artifact['sha256']} observed={observed_sha256}"
        )
    return {
        "role": role,
        "root_id": root_id,
        "path": artifact["path"],
        "sha256": observed_sha256,
        "size_bytes": observed_size,
        "status": "passed",
    }


def authenticate_profile_artifacts(
    profile: dict[str, Any], artifact_roots: dict[str, Path]
) -> list[dict[str, Any]]:
    return [
        authenticate_artifact_record(
            artifact,
            artifact_roots,
            role=role,
            owner=f"profile {profile['profile_schema_id']}",
        )
        for role, artifact in _profile_artifacts(profile)
    ]


def _profile_files(inputs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise contracts.ContractError(f"profile input does not exist: {path}")
    if not files:
        raise contracts.ContractError("no profile JSON files were found")
    unique = sorted({path.resolve() for path in files})
    return unique


def load_profiles(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in _profile_files(inputs):
        payload = contracts.load_json(path)
        if isinstance(payload, dict) and payload.get("schema") == contracts.PROFILE_SCHEMA:
            profiles.append(contracts.validate_attribute_profile(payload))
            continue
        # Directories may contain README support JSON or unrelated manifests;
        # fail closed instead of silently ignoring an unexpected contract.
        raise contracts.ContractError(
            f"profile file {path} is not a {contracts.PROFILE_SCHEMA} object"
        )
    identifiers = [profile["profile_schema_id"] for profile in profiles]
    if len(identifiers) != len(set(identifiers)):
        raise contracts.ContractError("profile inputs contain duplicate profile_schema_id")
    return sorted(profiles, key=lambda item: item["profile_schema_id"])


def build_profile_snapshot(
    profiles: Sequence[dict[str, Any]],
    artifact_authentication: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "schema": PROFILE_SNAPSHOT_SCHEMA,
        "profiles": [
            {
                "profile_schema_id": profile["profile_schema_id"],
                "profile_sha256": contracts.profile_sha256(profile),
                "profile": profile,
                "artifact_authentication": artifact_authentication[
                    profile["profile_schema_id"]
                ],
            }
            for profile in profiles
        ],
    }
    snapshot["snapshot_sha256"] = contracts.manifest_sha256(snapshot)
    return snapshot


def build_execution_jobs(request_batch: dict[str, Any]) -> dict[str, Any]:
    routes = {route: [] for route in sorted(contracts.ROUTES)}
    for request in request_batch["requests"]:
        route = request["generation_plan"]["route"]
        routes[route].append(
            {
                "instance_id": request["instance_id"],
                "request_sha256": request["request_sha256"],
                "profile_schema_id": request["profile_schema_id"],
                "lineage_group_id": request["lineage_group_id"],
                "generation_plan": request["generation_plan"],
                "target_physical_profile": request["target_physical_profile"],
                "rig_profile": request["rig_profile"],
                "acoustic_profile": request["acoustic_profile"],
            }
        )
    payload: dict[str, Any] = {
        "schema": EXECUTION_JOBS_SCHEMA,
        "request_batch_id": request_batch["batch_id"],
        "request_batch_sha256": request_batch["batch_sha256"],
        "routes": routes,
        "job_count": sum(len(jobs) for jobs in routes.values()),
    }
    payload["jobs_sha256"] = contracts.manifest_sha256(payload)
    return payload


def build_qa_pair_plan(
    generation_plan: dict[str, Any], request_batch: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": contracts.PAIR_PLAN_SCHEMA,
        "generation_plan_id": generation_plan["plan_id"],
        "generation_plan_sha256": generation_plan["manifest_sha256"],
        "request_batch_id": request_batch["batch_id"],
        "request_batch_sha256": request_batch["batch_sha256"],
        "evidence_state": "planned",
        "answer_policy": "pending_until_source_asset_v2_visual_qa_passes",
        "pairs": generation_plan["qa_pairs"],
        "pair_count": len(generation_plan["qa_pairs"]),
    }
    payload["pair_plan_sha256"] = contracts.manifest_sha256(payload)
    return payload


def compile_inputs(
    *,
    profiles: Sequence[dict[str, Any]],
    count_per_profile: int,
    seed: int,
    plan_id: str,
    split_salt: str,
    max_qa_pairs_per_split: int | None,
    artifact_authentication: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    profile_map = {profile["profile_schema_id"]: profile for profile in profiles}
    request_batch = contracts.build_request_batch(
        profiles,
        count_per_profile=count_per_profile,
        batch_seed=seed,
    )
    generation_plan = contracts.build_generation_plan_manifest(
        request_batch["requests"],
        profile_map,
        plan_id=plan_id,
        split_salt=split_salt,
        single_attribute_qa_only=True,
        max_qa_pairs_per_split=max_qa_pairs_per_split,
    )
    return {
        "profile_snapshot.json": build_profile_snapshot(
            profiles, artifact_authentication
        ),
        "instance_requests.json": request_batch,
        "execution_jobs.json": build_execution_jobs(request_batch),
        "qa_pair_plan.json": build_qa_pair_plan(generation_plan, request_batch),
        "generation_plan.json": generation_plan,
    }


def publish_output(output_dir: Path, files: dict[str, Any]) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise contracts.ContractError(
            f"refusing to replace existing output directory: {output_dir}"
        ) from error
    try:
        for name, payload in files.items():
            contracts.write_json_no_replace(output_dir / name, payload)
        # Read every artifact back before declaring publication successful.
        for name, expected in files.items():
            observed = contracts.load_json(output_dir / name)
            if contracts.canonical_json(observed) != contracts.canonical_json(expected):
                raise contracts.ContractError(f"published readback mismatch: {name}")
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        type=Path,
        help="Profile JSON file or directory; repeat for multiple roots.",
    )
    parser.add_argument("--count-per-profile", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--max-qa-pairs-per-split", type=int)
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        metavar="ROOT_ID=PATH",
        help="Add or override a named artifact root.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        profiles = load_profiles(args.profile)
        artifact_roots = parse_artifact_roots(args.artifact_root)
        artifact_authentication = {
            profile["profile_schema_id"]: authenticate_profile_artifacts(
                profile, artifact_roots
            )
            for profile in profiles
        }
        files = compile_inputs(
            profiles=profiles,
            count_per_profile=args.count_per_profile,
            seed=args.seed,
            plan_id=args.plan_id,
            split_salt=args.split_salt,
            max_qa_pairs_per_split=args.max_qa_pairs_per_split,
            artifact_authentication=artifact_authentication,
        )
        publish_output(args.output_dir, files)
    except contracts.ContractError as error:
        print(f"CONTROLLED_SOURCE_INPUT_FAILED {error}", file=sys.stderr)
        return 2

    request_batch = files["instance_requests.json"]
    pair_plan = files["qa_pair_plan.json"]
    execution_jobs = files["execution_jobs.json"]
    route_counts = ",".join(
        f"{route}:{len(jobs)}"
        for route, jobs in execution_jobs["routes"].items()
    )
    print(
        "CONTROLLED_SOURCE_INPUT_OK "
        f"profiles={len(profiles)} requests={len(request_batch['requests'])} "
        f"qa_pairs={pair_plan['pair_count']} routes={route_counts} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
