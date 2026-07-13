#!/usr/bin/env python3
"""Freeze normalized profiles, requests, and realized assets for a dataset build."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts


DATASET_INPUT_SCHEMA = "avengine_controlled_dataset_input_manifest_v1"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_ARTIFACT_FIELDS = frozenset({"root_id", "path", "sha256", "size_bytes"})
_DATASET_FIELDS = frozenset(
    {
        "dataset_id",
        "split_salt",
        "allowed_states",
        "max_qa_pairs_per_split",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {"schema", "dataset", "inputs", "request_lineage", "manifest_sha256"}
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_artifact(path: Path, artifact_roots: Mapping[str, Path]) -> dict[str, Any]:
    resolved = path.resolve()
    candidates: list[tuple[int, str, Path]] = []
    for root_id, raw_root in artifact_roots.items():
        root = raw_root.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        candidates.append((len(root.parts), root_id, relative))
    if not candidates:
        raise contracts.ContractError(
            f"normalized dataset input is outside every artifact root: {resolved}"
        )
    _, root_id, relative = max(candidates, key=lambda item: item[0])
    if not resolved.is_file():
        raise contracts.ContractError(f"normalized dataset input is missing: {resolved}")
    return {
        "root_id": root_id,
        "path": relative.as_posix(),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_artifact(
    artifact: Mapping[str, Any], artifact_roots: Mapping[str, Path], *, role: str
) -> Path:
    if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_FIELDS:
        raise contracts.ContractError(f"{role} artifact fields are invalid")
    input_builder.authenticate_artifact_record(
        dict(artifact), dict(artifact_roots), role=role, owner="dataset input manifest"
    )
    return (artifact_roots[artifact["root_id"]] / artifact["path"]).resolve()


def load_source_asset_inputs(
    inputs: Sequence[Path],
) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in _json_files(inputs, "source asset"):
        payload = contracts.load_json(path)
        if not isinstance(payload, dict) or payload.get("schema") != contracts.SOURCE_ASSET_SCHEMA:
            raise contracts.ContractError(
                f"source asset file {path} is not a {contracts.SOURCE_ASSET_SCHEMA} object"
            )
        records.append((path, contracts.validate_source_asset_v2(payload)))
    asset_ids = [asset["asset_id"] for _, asset in records]
    if len(asset_ids) != len(set(asset_ids)):
        raise contracts.ContractError("source asset inputs contain duplicate asset_id")
    return sorted(records, key=lambda item: item[1]["asset_id"])


def load_request_batch_inputs(
    inputs: Sequence[Path], profiles_by_id: Mapping[str, dict[str, Any]]
) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    seen_batch_ids: set[str] = set()
    seen_request_hashes: set[str] = set()
    seen_instance_ids: set[str] = set()
    for path in _json_files(inputs, "request batch"):
        payload = contracts.load_json(path)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != contracts.REQUEST_BATCH_SCHEMA
        ):
            raise contracts.ContractError(
                f"request batch file {path} is not a {contracts.REQUEST_BATCH_SCHEMA} object"
            )
        references = payload.get("profiles")
        if not isinstance(references, list) or not references:
            raise contracts.ContractError(f"request batch has no profile references: {path}")
        profile_ids: list[str] = []
        for reference in references:
            if not isinstance(reference, dict):
                raise contracts.ContractError(
                    f"request batch profile reference is invalid: {path}"
                )
            profile_id = reference.get("profile_schema_id")
            if profile_id not in profiles_by_id:
                raise contracts.ContractError(
                    f"request batch references missing profile {profile_id}: {path}"
                )
            profile_ids.append(profile_id)
        if len(profile_ids) != len(set(profile_ids)):
            raise contracts.ContractError(f"request batch repeats a profile: {path}")
        batch = contracts.validate_request_batch(
            payload, [profiles_by_id[profile_id] for profile_id in profile_ids]
        )
        if batch["batch_id"] in seen_batch_ids:
            raise contracts.ContractError("request batch inputs contain duplicate batch_id")
        seen_batch_ids.add(batch["batch_id"])
        for request in batch["requests"]:
            if request["request_sha256"] in seen_request_hashes:
                raise contracts.ContractError(
                    "request batch inputs contain duplicate request_sha256"
                )
            if request["instance_id"] in seen_instance_ids:
                raise contracts.ContractError(
                    "request batch inputs contain duplicate instance_id"
                )
            seen_request_hashes.add(request["request_sha256"])
            seen_instance_ids.add(request["instance_id"])
        records.append((path, batch))
    return sorted(records, key=lambda item: item[1]["batch_id"])


def validate_asset_request_bindings(
    source_assets: Sequence[tuple[Path, dict[str, Any]]],
    request_batches: Sequence[tuple[Path, dict[str, Any]]],
    profiles_by_id: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    requests_by_hash: dict[str, tuple[str, dict[str, Any]]] = {}
    for _, batch in request_batches:
        for request in batch["requests"]:
            requests_by_hash[request["request_sha256"]] = (batch["batch_id"], request)
    bindings: list[dict[str, Any]] = []
    used_requests: set[str] = set()
    for _, asset in source_assets:
        request_hash = asset["request_sha256"]
        if request_hash not in requests_by_hash:
            raise contracts.ContractError(
                f"source asset {asset['asset_id']} has no matching normalized request"
            )
        batch_id, request = requests_by_hash[request_hash]
        profile_id = asset["profile_schema_id"]
        if profile_id not in profiles_by_id:
            raise contracts.ContractError(
                f"source asset {asset['asset_id']} has no matching normalized profile"
            )
        contracts.validate_source_asset_v2(
            asset, request=request, profile=profiles_by_id[profile_id]
        )
        used_requests.add(request_hash)
        bindings.append(
            {
                "asset_id": asset["asset_id"],
                "asset_manifest_sha256": contracts.manifest_sha256(asset),
                "request_batch_id": batch_id,
                "request_instance_id": request["instance_id"],
                "request_sha256": request_hash,
                "profile_schema_id": profile_id,
                "profile_sha256": asset["profile_sha256"],
                "status": "passed",
            }
        )
    return (
        sorted(bindings, key=lambda item: item["asset_id"]),
        len(requests_by_hash) - len(used_requests),
    )


def build_dataset_input_manifest(
    *,
    profile_inputs: Sequence[Path],
    request_batch_inputs: Sequence[Path],
    source_asset_inputs: Sequence[Path],
    dataset_id: str,
    split_salt: str,
    allowed_states: Sequence[str],
    max_qa_pairs_per_split: int | None,
    artifact_roots: Mapping[str, Path],
) -> dict[str, Any]:
    if not isinstance(dataset_id, str) or not _ID_RE.fullmatch(dataset_id):
        raise contracts.ContractError("dataset_id is invalid")
    if not isinstance(split_salt, str) or not split_salt.strip():
        raise contracts.ContractError("split_salt must be non-empty text")
    states = sorted(set(allowed_states))
    if not states or any(state not in contracts.STATE_CLASSIFICATIONS for state in states):
        raise contracts.ContractError("allowed_states are invalid")
    if max_qa_pairs_per_split is not None and (
        isinstance(max_qa_pairs_per_split, bool)
        or not isinstance(max_qa_pairs_per_split, int)
        or max_qa_pairs_per_split <= 0
    ):
        raise contracts.ContractError("max_qa_pairs_per_split must be positive")

    profile_files = _json_files(profile_inputs, "profile")
    profiles = input_builder.load_profiles(profile_files)
    profiles_by_id = {profile["profile_schema_id"]: profile for profile in profiles}
    request_records = load_request_batch_inputs(request_batch_inputs, profiles_by_id)
    asset_records = load_source_asset_inputs(source_asset_inputs)
    bindings, unused_request_count = validate_asset_request_bindings(
        asset_records, request_records, profiles_by_id
    )
    manifest: dict[str, Any] = {
        "schema": DATASET_INPUT_SCHEMA,
        "dataset": {
            "dataset_id": dataset_id,
            "split_salt": split_salt,
            "allowed_states": states,
            "max_qa_pairs_per_split": max_qa_pairs_per_split,
        },
        "inputs": {
            "profile_files": [
                _file_artifact(path, artifact_roots) for path in profile_files
            ],
            "request_batch_files": [
                _file_artifact(path, artifact_roots) for path, _ in request_records
            ],
            "source_asset_files": [
                _file_artifact(path, artifact_roots) for path, _ in asset_records
            ],
        },
        "request_lineage": {
            "bindings": bindings,
            "profile_count": len(profiles),
            "request_batch_count": len(request_records),
            "request_count": sum(len(batch["requests"]) for _, batch in request_records),
            "realized_asset_count": len(asset_records),
            "matched_asset_count": len(bindings),
            "unused_request_count": unused_request_count,
            "all_realized_assets_match_exactly_one_request": len(bindings) == len(asset_records),
        },
    }
    manifest["manifest_sha256"] = contracts.manifest_sha256(manifest)
    return manifest


def load_and_validate_dataset_input_manifest(
    path: Path, artifact_roots: Mapping[str, Path]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = contracts.load_json(path)
    if not isinstance(manifest, dict) or set(manifest) != _TOP_LEVEL_FIELDS:
        raise contracts.ContractError("dataset input manifest fields are invalid")
    if manifest.get("schema") != DATASET_INPUT_SCHEMA:
        raise contracts.ContractError(f"dataset input schema must be {DATASET_INPUT_SCHEMA}")
    if manifest.get("manifest_sha256") != contracts.manifest_sha256(manifest):
        raise contracts.ContractError("dataset input manifest hash is invalid")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != _DATASET_FIELDS:
        raise contracts.ContractError("dataset input configuration fields are invalid")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "profile_files",
        "request_batch_files",
        "source_asset_files",
    }:
        raise contracts.ContractError("dataset input file groups are invalid")
    resolved: dict[str, list[Path]] = {}
    for group, artifacts in inputs.items():
        if not isinstance(artifacts, list) or not artifacts:
            raise contracts.ContractError(f"dataset input {group} must be a non-empty list")
        resolved[group] = [
            _resolve_artifact(artifact, artifact_roots, role=f"{group}:{index}")
            for index, artifact in enumerate(artifacts)
        ]
    rebuilt = build_dataset_input_manifest(
        profile_inputs=resolved["profile_files"],
        request_batch_inputs=resolved["request_batch_files"],
        source_asset_inputs=resolved["source_asset_files"],
        dataset_id=dataset["dataset_id"],
        split_salt=dataset["split_salt"],
        allowed_states=dataset["allowed_states"],
        max_qa_pairs_per_split=dataset["max_qa_pairs_per_split"],
        artifact_roots=artifact_roots,
    )
    if contracts.canonical_json(manifest) != contracts.canonical_json(rebuilt):
        raise contracts.ContractError(
            "dataset input manifest does not match authenticated normalized inputs"
        )
    profiles = input_builder.load_profiles(resolved["profile_files"])
    assets = [asset for _, asset in load_source_asset_inputs(resolved["source_asset_files"])]
    return manifest, profiles, assets


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", required=True, type=Path)
    parser.add_argument("--request-batch", action="append", required=True, type=Path)
    parser.add_argument("--asset", action="append", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument(
        "--allow-state",
        action="append",
        choices=sorted(contracts.STATE_CLASSIFICATIONS),
        default=None,
    )
    parser.add_argument("--max-qa-pairs-per-split", type=int)
    parser.add_argument("--artifact-root", action="append", default=[], metavar="ROOT_ID=PATH")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        roots = input_builder.parse_artifact_roots(args.artifact_root)
        manifest = build_dataset_input_manifest(
            profile_inputs=args.profile,
            request_batch_inputs=args.request_batch,
            source_asset_inputs=args.asset,
            dataset_id=args.dataset_id,
            split_salt=args.split_salt,
            allowed_states=args.allow_state or ["formal_dataset_asset"],
            max_qa_pairs_per_split=args.max_qa_pairs_per_split,
            artifact_roots=roots,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        contracts.write_json_no_replace(args.output, manifest)
        observed, _, _ = load_and_validate_dataset_input_manifest(args.output, roots)
        if contracts.canonical_json(observed) != contracts.canonical_json(manifest):
            raise contracts.ContractError("published dataset input manifest readback mismatch")
    except contracts.ContractError as error:
        print(f"CONTROLLED_DATASET_INPUT_MANIFEST_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_DATASET_INPUT_MANIFEST_OK "
        f"profiles={manifest['request_lineage']['profile_count']} "
        f"requests={manifest['request_lineage']['request_count']} "
        f"assets={manifest['request_lineage']['realized_asset_count']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
