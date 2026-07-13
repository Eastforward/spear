#!/usr/bin/env python3
"""Resolve calibrated animal Apartment records without deleting old evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hash_without(value: dict, key: str) -> str:
    return _canonical_hash({k: copy.deepcopy(v) for k, v in value.items() if k != key})


def _file_artifact(path: Path) -> dict:
    path = path.resolve()
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _load_manifest(path: Path) -> dict:
    path = path.resolve()
    value = json.loads(path.read_text())
    records = value.get("records", []) if isinstance(value, dict) else []
    if (
        value.get("schema") != SCHEMA
        or value.get("avatar_count") != len(records)
        or value.get("clip_count") != len(records) * 2
        or value.get("manifest_sha256") != _hash_without(value, "manifest_sha256")
    ):
        raise ValueError(f"invalid controlled animal Apartment manifest: {path}")
    seen = set()
    for record in records:
        asset_id = record.get("asset_id")
        if (
            not asset_id
            or record.get("base_avatar_id") != asset_id
            or set(record.get("actions", {})) != {"Walking", "Idle"}
            or asset_id in seen
        ):
            raise ValueError(f"invalid or duplicate Apartment record: {asset_id}")
        seen.add(asset_id)
    return value


def _same_identity(base: dict, replacement: dict) -> bool:
    keys = (
        "asset_id",
        "base_avatar_id",
        "profile_schema_id",
        "sampled_attributes",
        "source_glb",
        "tag",
    )
    return all(base.get(key) == replacement.get(key) for key in keys)


def resolve_manifests(
    *,
    base_manifest: Path,
    replacement_manifests: Sequence[Path],
    output_path: Path,
) -> Path:
    base_manifest = base_manifest.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    base = _load_manifest(base_manifest)
    records = {item["asset_id"]: copy.deepcopy(item) for item in base["records"]}
    superseded: list[str] = []
    replacement_inputs = []
    for manifest_path in replacement_manifests:
        manifest_path = manifest_path.resolve()
        replacement = _load_manifest(manifest_path)
        replacement_inputs.append(_file_artifact(manifest_path))
        for record in replacement["records"]:
            asset_id = record["asset_id"]
            if asset_id not in records:
                raise ValueError(f"replacement is not present in base manifest: {asset_id}")
            if not _same_identity(records[asset_id], record):
                raise ValueError(f"replacement identity changed: {asset_id}")
            if asset_id in superseded:
                raise ValueError(f"asset replaced more than once: {asset_id}")
            records[asset_id] = copy.deepcopy(record)
            superseded.append(asset_id)
    if not superseded:
        raise ValueError("at least one replacement record is required")

    resolved_records = [records[key] for key in sorted(records)]
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "usage_scope": base.get("usage_scope", "research_candidate"),
        "formal_registration_authorized": False,
        "avatar_count": len(resolved_records),
        "clip_count": len(resolved_records) * 2,
        "audio_policy": base.get("audio_policy"),
        "trajectory_policy": base.get("trajectory_policy"),
        "physical_scale_policy": (
            "base records with explicit immutable measurement-calibrated replacements"
        ),
        "inputs": {
            "base_manifest": _file_artifact(base_manifest),
            "replacement_manifests": replacement_inputs,
        },
        "resolution": {
            "policy": "replacement records supersede matching asset ids only",
            "superseded_asset_ids": sorted(superseded),
            "old_media_deleted": False,
        },
        "records": resolved_records,
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--replacement-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = resolve_manifests(
            base_manifest=args.base_manifest,
            replacement_manifests=args.replacement_manifest,
            output_path=args.output,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"CONTROLLED_ANIMAL_APARTMENT_RESOLUTION_FAILED {error}")
        return 2
    value = json.loads(output.read_text())
    print(
        "CONTROLLED_ANIMAL_APARTMENT_RESOLUTION_OK "
        f"assets={value['avatar_count']} replacements="
        f"{len(value['resolution']['superseded_asset_ids'])} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
