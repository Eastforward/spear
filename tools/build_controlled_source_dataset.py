#!/usr/bin/env python3
"""Build a lineage-safe dataset and realized QA pairs from source_asset_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_controlled_source_asset_inputs as input_builder
from tools import build_controlled_source_dataset_input_manifest as dataset_inputs
from tools import controlled_source_asset_schema as contracts


QA_DATASET_SCHEMA = "avengine_realized_qa_dataset_v1"
SOURCE_POOL_SCHEMA = "avengine_controlled_scene_source_pool_v1"
SCENE_REQUIRED_QA = (
    "reference_2d",
    "static_mesh",
    "binding",
    "walking",
    "idle",
    "ue_import_readback",
    "apartment_media",
    "audio",
)


def _json_files(inputs: Sequence[Path], label: str) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise contracts.ContractError(f"{label} input does not exist: {path}")
    if not files:
        raise contracts.ContractError(f"no {label} JSON files were found")
    return sorted({path.resolve() for path in files})


def load_source_assets(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for path in _json_files(inputs, "source asset"):
        payload = contracts.load_json(path)
        if not isinstance(payload, dict) or payload.get("schema") != contracts.SOURCE_ASSET_SCHEMA:
            raise contracts.ContractError(
                f"source asset file {path} is not a {contracts.SOURCE_ASSET_SCHEMA} object"
            )
        assets.append(contracts.validate_source_asset_v2(payload))
    identifiers = [asset["asset_id"] for asset in assets]
    if len(identifiers) != len(set(identifiers)):
        raise contracts.ContractError("source asset inputs contain duplicate asset_id")
    return sorted(assets, key=lambda item: item["asset_id"])


def authenticate_source_assets(
    assets: Sequence[dict[str, Any]], artifact_roots: dict[str, Path]
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for asset in assets:
        records: list[dict[str, Any]] = []
        for role, artifact in sorted(asset["artifacts"].items()):
            records.append(
                input_builder.authenticate_artifact_record(
                    artifact,
                    artifact_roots,
                    role=f"asset:{role}",
                    owner=f"source asset {asset['asset_id']}",
                )
            )
        for index, artifact in enumerate(asset["rights"]["licenses"]):
            records.append(
                input_builder.authenticate_artifact_record(
                    artifact,
                    artifact_roots,
                    role=f"license:{index}",
                    owner=f"source asset {asset['asset_id']}",
                )
            )
        results[asset["asset_id"]] = records
    return results


def build_realized_qa_dataset(dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    formal_only = dataset_manifest["allowed_states"] == ["formal_dataset_asset"]
    payload: dict[str, Any] = {
        "schema": QA_DATASET_SCHEMA,
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "evidence_state": "realized",
        "pairs": dataset_manifest["qa_pairs"],
        "pair_count": len(dataset_manifest["qa_pairs"]),
        "question_count": sum(
            len(pair["questions"]) for pair in dataset_manifest["qa_pairs"]
        ),
        "answer_policy": (
            "absolute_attributes_backed_by_formal_source_asset_v2"
            if formal_only
            else "absolute_attributes_backed_by_realized_candidate_pending_formal_acceptance"
        ),
    }
    payload["qa_dataset_sha256"] = contracts.manifest_sha256(payload)
    return payload


def build_scene_source_pool(dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in dataset_manifest["assets"]:
        qa = row["qa"]
        blocking_qa = [
            field
            for field in SCENE_REQUIRED_QA
            if not (
                qa[field] == "passed"
                or (field == "reference_2d" and qa[field] == "not_applicable")
            )
        ]
        rights_ready = (
            row["rights"]["status"] == "cleared"
            and not row["rights"]["blockers"]
        )
        scene_readiness = {
            "eligible_for_apartment_generation": not blocking_qa and rights_ready,
            "blocking_qa": blocking_qa,
            "rights_ready": rights_ready,
            "policy": "all_scene_qa_passed_and_rights_cleared_v1",
        }
        splits[row["split"]].append(
            {
                "asset_id": row["asset_id"],
                "asset_manifest_sha256": row["asset_manifest_sha256"],
                "profile_schema_id": row["profile_schema_id"],
                "lineage_group_id": row["lineage_group_id"],
                "asset_class": row["asset_class"],
                "state_classification": row["state_classification"],
                "taxonomy": row["taxonomy"],
                "semantic_attributes": row["semantic_attributes"],
                "sampled_attributes": row["sampled_attributes"],
                "qa_evidence_attributes": row["qa_evidence_attributes"],
                "physical_measurements": row["physical_measurements"],
                "artifacts": row["artifacts"],
                "rig": row["rig"],
                "acoustic_profile": row["acoustic_profile"],
                "acoustic_profile_id": row["acoustic_profile_id"],
                "qa": qa,
                "rights": row["rights"],
                "scene_readiness": scene_readiness,
            }
        )
    eligible_count = sum(
        source["scene_readiness"]["eligible_for_apartment_generation"]
        for sources in splits.values()
        for source in sources
    )
    payload: dict[str, Any] = {
        "schema": SOURCE_POOL_SCHEMA,
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "splits": splits,
        "source_count": sum(len(items) for items in splits.values()),
        "eligible_source_count": eligible_count,
        "ineligible_source_count": sum(len(items) for items in splits.values())
        - eligible_count,
        "selection_rule": (
            "scene generators must select only records whose "
            "scene_readiness.eligible_for_apartment_generation is true"
        ),
    }
    payload["source_pool_sha256"] = contracts.manifest_sha256(payload)
    return payload


def build_artifact_audit(
    dataset_manifest: dict[str, Any],
    profile_authentication: dict[str, list[dict[str, Any]]],
    asset_authentication: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "avengine_controlled_dataset_artifact_audit_v1",
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "profiles": [
            {
                "profile_schema_id": profile_id,
                "artifacts": records,
                "status": "passed",
            }
            for profile_id, records in sorted(profile_authentication.items())
        ],
        "assets": [
            {
                "asset_id": asset_id,
                "artifacts": records,
                "status": "passed",
            }
            for asset_id, records in sorted(asset_authentication.items())
        ],
        "automatic_checks": {
            "profile_count": len(profile_authentication),
            "asset_count": len(asset_authentication),
            "all_artifacts_authenticated": all(
                record["status"] == "passed"
                for records in list(profile_authentication.values())
                + list(asset_authentication.values())
                for record in records
            ),
        },
    }
    payload["audit_sha256"] = contracts.manifest_sha256(payload)
    return payload


def build_receipt(
    input_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    qa_dataset: dict[str, Any],
    source_pool: dict[str, Any],
    artifact_audit: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "avengine_controlled_dataset_build_receipt_v1",
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_input_manifest_sha256": input_manifest["manifest_sha256"],
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "qa_dataset_sha256": qa_dataset["qa_dataset_sha256"],
        "source_pool_sha256": source_pool["source_pool_sha256"],
        "artifact_audit_sha256": artifact_audit["audit_sha256"],
        "asset_count": len(dataset_manifest["assets"]),
        "request_lineage_status": "passed",
        "artifact_authentication_status": "passed",
    }
    payload["manifest_sha256"] = contracts.manifest_sha256(payload)
    return payload


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
        "--input-manifest",
        type=Path,
        help=(
            "Canonical mode: consume an authenticated normalized profile/request/asset "
            "manifest. Raw selectors remain available for legacy candidate builds."
        ),
    )
    parser.add_argument("--profile", action="append", type=Path)
    parser.add_argument("--asset", action="append", type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--split-salt")
    parser.add_argument(
        "--allow-state",
        action="append",
        choices=sorted(contracts.STATE_CLASSIFICATIONS),
        default=None,
        help="Defaults to formal_dataset_asset only.",
    )
    parser.add_argument("--max-qa-pairs-per-split", type=int)
    parser.add_argument(
        "--artifact-root", action="append", default=[], metavar="ROOT_ID=PATH"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        roots = input_builder.parse_artifact_roots(args.artifact_root)
        input_manifest: dict[str, Any] | None = None
        if args.input_manifest is not None:
            raw_selectors = {
                "--profile": args.profile,
                "--asset": args.asset,
                "--dataset-id": args.dataset_id,
                "--split-salt": args.split_salt,
                "--allow-state": args.allow_state,
                "--max-qa-pairs-per-split": args.max_qa_pairs_per_split,
            }
            supplied = [name for name, value in raw_selectors.items() if value is not None]
            if supplied:
                raise contracts.ContractError(
                    "--input-manifest cannot be combined with raw dataset selectors: "
                    + ", ".join(supplied)
                )
            input_manifest, profiles, assets = (
                dataset_inputs.load_and_validate_dataset_input_manifest(
                    args.input_manifest, roots
                )
            )
            dataset_config = input_manifest["dataset"]
            dataset_id = dataset_config["dataset_id"]
            split_salt = dataset_config["split_salt"]
            allowed_states = set(dataset_config["allowed_states"])
            max_qa_pairs_per_split = dataset_config["max_qa_pairs_per_split"]
        else:
            missing = [
                name
                for name, value in (
                    ("--profile", args.profile),
                    ("--asset", args.asset),
                    ("--dataset-id", args.dataset_id),
                    ("--split-salt", args.split_salt),
                )
                if not value
            ]
            if missing:
                raise contracts.ContractError(
                    "legacy raw dataset build is missing required selectors: "
                    + ", ".join(missing)
                )
            profiles = input_builder.load_profiles(args.profile)
            assets = load_source_assets(args.asset)
            dataset_id = args.dataset_id
            split_salt = args.split_salt
            allowed_states = set(args.allow_state or ["formal_dataset_asset"])
            max_qa_pairs_per_split = args.max_qa_pairs_per_split
        profile_map = {profile["profile_schema_id"]: profile for profile in profiles}
        profile_authentication = {
            profile["profile_schema_id"]: input_builder.authenticate_profile_artifacts(
                profile, roots
            )
            for profile in profiles
        }
        asset_authentication = authenticate_source_assets(assets, roots)
        dataset_manifest = contracts.build_dataset_manifest(
            assets,
            profile_map,
            dataset_id=dataset_id,
            split_salt=split_salt,
            allowed_states=allowed_states,
            max_qa_pairs_per_split=max_qa_pairs_per_split,
        )
        qa_dataset = build_realized_qa_dataset(dataset_manifest)
        source_pool = build_scene_source_pool(dataset_manifest)
        artifact_audit = build_artifact_audit(
            dataset_manifest, profile_authentication, asset_authentication
        )
        files = {
            "dataset_manifest.json": dataset_manifest,
            "qa_dataset.json": qa_dataset,
            "scene_source_pool.json": source_pool,
            "artifact_audit.json": artifact_audit,
        }
        if input_manifest is not None:
            files["dataset_input_manifest.json"] = input_manifest
            files["build_receipt.json"] = build_receipt(
                input_manifest,
                dataset_manifest,
                qa_dataset,
                source_pool,
                artifact_audit,
            )
        publish_output(args.output_dir, files)
    except contracts.ContractError as error:
        print(f"CONTROLLED_SOURCE_DATASET_FAILED {error}", file=sys.stderr)
        return 2

    print(
        "CONTROLLED_SOURCE_DATASET_OK "
        f"assets={len(dataset_manifest['assets'])} "
        f"qa_pairs={len(dataset_manifest['qa_pairs'])} "
        f"questions={files['qa_dataset.json']['question_count']} "
        f"request_lineage={'verified' if input_manifest is not None else 'legacy_unverified'} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
