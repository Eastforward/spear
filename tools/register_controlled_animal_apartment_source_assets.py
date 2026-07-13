#!/usr/bin/env python3
"""Publish post-Apartment revisions of controlled animal source_asset_v2 records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import register_controlled_animal_source_assets as static_registry
from tools import rocketbox_native_material_canary as immutable


REGISTRY_SCHEMA = "avengine_controlled_animal_apartment_source_asset_registry_v1"
APARTMENT_SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
MEASUREMENT_BATCH_SCHEMA = "controlled_animal_physical_measurement_batch_v1"
MEASUREMENT_SCHEMA = "controlled_animal_physical_measurement_v1"
APARTMENT_REGISTRY_SCHEMA = (
    "controlled_animal_apartment_research_candidate_registry_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _descriptor_file(value: Mapping[str, Any], label: str) -> Path:
    try:
        path = Path(value["path"]).resolve()
        expected_sha = value["sha256"]
        expected_size = value["size_bytes"]
    except (KeyError, TypeError) as error:
        raise contracts.ContractError(f"invalid {label} descriptor") from error
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != expected_size
        or _sha256(path) != expected_sha
    ):
        raise contracts.ContractError(f"{label} artifact changed: {path}")
    return path


def _load_source_assets(roots: Sequence[Path]) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in sorted(Path(root).resolve().rglob("*.json")):
            try:
                payload = contracts.load_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("schema") != contracts.SOURCE_ASSET_SCHEMA:
                continue
            asset = contracts.validate_source_asset_v2(payload)
            asset_id = asset["asset_id"]
            if asset_id in assets:
                raise contracts.ContractError(f"duplicate source_asset_v2: {asset_id}")
            assets[asset_id] = asset
    if not assets:
        raise contracts.ContractError("no source_asset_v2 records found")
    return assets


def _load_apartment_records(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = contracts.load_json(Path(path).resolve())
        items = payload.get("records", []) if isinstance(payload, dict) else []
        if (
            payload.get("schema") != APARTMENT_SCHEMA
            or payload.get("avatar_count") != len(items)
            or payload.get("clip_count") != len(items) * 2
            or payload.get("manifest_sha256") != _hash_without(payload, "manifest_sha256")
        ):
            raise contracts.ContractError(f"invalid Apartment manifest: {path}")
        for record in items:
            asset_id = record.get("base_avatar_id")
            if not asset_id or asset_id in records:
                raise contracts.ContractError(f"duplicate Apartment asset: {asset_id}")
            records[asset_id] = record
    return records


def _load_measurements(path: Path) -> dict[str, dict[str, Any]]:
    path = Path(path).resolve()
    payload = contracts.load_json(path)
    items = payload.get("measurements", []) if isinstance(payload, dict) else []
    if (
        payload.get("schema") != MEASUREMENT_BATCH_SCHEMA
        or payload.get("asset_count") != len(items)
        or payload.get("batch_sha256") != _hash_without(payload, "batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("invalid physical measurement batch")
    records = {}
    for index in items:
        record_path = _descriptor_file(index["record"], "physical measurement")
        record = contracts.load_json(record_path)
        asset_id = record.get("asset_id")
        if (
            record.get("schema") != MEASUREMENT_SCHEMA
            or asset_id != index.get("asset_id")
            or record.get("physical_measurements") != index.get("physical_measurements")
            or asset_id in records
        ):
            raise contracts.ContractError("physical measurement identity changed")
        for descriptor in record.get("evidence", {}).values():
            _descriptor_file(descriptor, "physical measurement evidence")
        records[asset_id] = {"payload": record, "path": record_path}
    return records


def _validate_audio(action_record: Mapping[str, Any], tag: str) -> dict[str, Path]:
    output = Path(action_record["output_dir"]).resolve()
    audio = output / "binaural.wav"
    schedule_path = output / "binaural_source_schedule.json"
    if audio.is_symlink() or schedule_path.is_symlink():
        raise contracts.ContractError(f"audio evidence cannot be symlinked: {tag}")
    schedule = contracts.load_json(schedule_path)
    source = schedule.get("sources", {}).get(tag, {})
    if (
        schedule.get("schema") != "rlr_audio_source_schedules_v1"
        or set(schedule.get("sources", {})) != {tag}
        or source.get("schema") != "animal_audio_event_schedule_v1"
        or source.get("tag") != tag
        or source.get("adaptive_repeat_short_calls") is not True
        or source.get("short_call_detected") is not True
        or source.get("mode") != "repeated_events_with_silence_gaps"
        or int(source.get("event_count", 0)) <= 1
        or float(source.get("minimum_silence_gap_s", 0.0)) < 0.85
    ):
        raise contracts.ContractError(f"animal audio schedule failed: {tag}")
    try:
        with wave.open(str(audio), "rb") as stream:
            channels = stream.getnchannels()
            rate = stream.getframerate()
            duration = stream.getnframes() / float(rate)
    except (OSError, EOFError, wave.Error) as error:
        raise contracts.ContractError(f"cannot read binaural audio: {audio}") from error
    if channels != 2 or rate != 16000 or abs(duration - 18.0) > 1.0 / rate:
        raise contracts.ContractError(f"binaural audio contract changed: {audio}")
    return {"audio": audio, "schedule": schedule_path}


def _validate_apartment_registry(record: Mapping[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    tag = record["tag"]
    walking_output = Path(record["actions"]["Walking"]["output_dir"]).resolve()
    registry_path = walking_output.parent / "registry" / f"{tag}.json"
    registry = contracts.load_json(registry_path)
    if (
        registry.get("schema_version") != APARTMENT_REGISTRY_SCHEMA
        or registry.get("usage_scope") != "research_candidate"
        or registry.get("formal_registry_promotion") is not False
        or registry.get("tag") != tag
        or registry.get("asset_id") != record["base_avatar_id"]
        or registry.get("sampled_attributes") != record["sampled_attributes"]
        or set(registry.get("clips", {})) != {"Walking", "Idle"}
    ):
        raise contracts.ContractError(f"Apartment registry identity changed: {tag}")
    for descriptor_name in ("animation_decision", "ue_import_result"):
        _descriptor_file(registry[descriptor_name], descriptor_name)
    decision = contracts.load_json(Path(registry["animation_decision"]["path"]))
    if (
        decision.get("asset_id") != record["base_avatar_id"]
        or decision.get("decision") != "approved_for_ue_apartment"
        or not decision.get("checks")
        or not all(decision["checks"].values())
    ):
        raise contracts.ContractError(f"animation decision not approved: {tag}")
    imported = contracts.load_json(Path(registry["ue_import_result"]["path"]))
    imports = [item for item in imported.get("results", []) if item.get("tag") == tag]
    if (
        len(imports) != 1
        or imports[0].get("status") != "passed"
        or set(imports[0].get("actions", [])) != {"Walking", "Idle"}
        or imports[0].get("source_sha256") != registry.get("ue_source_sha256")
    ):
        raise contracts.ContractError(f"UE import readback failed: {tag}")

    extra: dict[str, Path] = {
        "apartment_registry": registry_path,
        "animation_decision": Path(registry["animation_decision"]["path"]).resolve(),
        "ue_import_result": Path(registry["ue_import_result"]["path"]).resolve(),
    }
    for action in ("Walking", "Idle"):
        clip = registry["clips"][action]
        action_record = record["actions"][action]
        if clip.get("clip_id") != action_record["clip_id"]:
            raise contracts.ContractError(f"Apartment clip identity changed: {tag}/{action}")
        for name, descriptor in clip.items():
            if name == "clip_id":
                continue
            extra[f"apartment_{action.lower()}_{name}"] = _descriptor_file(
                descriptor, f"{tag}/{action}/{name}"
            )
        audio = _validate_audio(action_record, tag)
        extra[f"apartment_{action.lower()}_binaural_audio"] = audio["audio"]
        extra[f"apartment_{action.lower()}_audio_schedule"] = audio["schedule"]
    return registry_path, registry, extra


def upgrade_source_asset(
    source_asset: Mapping[str, Any],
    *,
    physical_measurements: Mapping[str, Any],
    added_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    asset = contracts.validate_source_asset_v2(source_asset)
    if asset["state_classification"] != "research_candidate":
        raise contracts.ContractError("Apartment revision requires a research candidate")
    upgraded = copy.deepcopy(asset)
    overlap = set(upgraded["artifacts"]) & set(added_artifacts)
    if overlap:
        raise contracts.ContractError(f"Apartment artifact roles already exist: {sorted(overlap)}")
    upgraded["artifacts"].update(copy.deepcopy(dict(added_artifacts)))
    upgraded["physical_measurements"] = copy.deepcopy(dict(physical_measurements))
    upgraded["qa"].update(
        {
            "binding": "passed",
            "walking": "passed",
            "idle": "passed",
            "ue_import_readback": "passed",
            "apartment_media": "passed",
            "audio": "passed",
        }
    )
    upgraded["provenance"]["attempt_id"] = (
        f"apartment_{upgraded['request_sha256'][:16]}"
    )
    return contracts.validate_source_asset_v2(upgraded)


def register(
    *,
    source_asset_roots: Sequence[Path],
    apartment_manifests: Sequence[Path],
    measurement_batch: Path,
    output_root: Path,
) -> Path:
    output_root = Path(output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"output already exists: {output_root}")
    source_assets = _load_source_assets(source_asset_roots)
    apartment = _load_apartment_records(apartment_manifests)
    measurements = _load_measurements(measurement_batch)
    if set(apartment) != set(measurements):
        raise contracts.ContractError("Apartment and measurement asset sets differ")
    missing = set(apartment) - set(source_assets)
    if missing:
        raise contracts.ContractError(f"Apartment assets lack static source_asset_v2: {sorted(missing)}")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent)
    )
    try:
        entries = []
        for asset_id, record in sorted(apartment.items()):
            source = source_assets[asset_id]
            if (
                source["profile_schema_id"] != record["profile_schema_id"]
                or source["sampled_attributes"] != record["sampled_attributes"]
                or record["source_glb"]["sha256"]
                != _sha256(Path(record["source_glb"]["path"]))
            ):
                raise contracts.ContractError(f"Apartment/source identity changed: {asset_id}")
            _registry_path, _registry, evidence_paths = _validate_apartment_registry(record)
            measurement = measurements[asset_id]
            added = {
                "rigged_walk_idle_glb": static_registry.spear_artifact(
                    Path(record["source_glb"]["path"])
                ),
                "physical_measurement": static_registry.spear_artifact(
                    measurement["path"]
                ),
                **{
                    role: static_registry.spear_artifact(path)
                    for role, path in sorted(evidence_paths.items())
                },
            }
            upgraded = upgrade_source_asset(
                source,
                physical_measurements=measurement["payload"]["physical_measurements"],
                added_artifacts=added,
            )
            destination = staging / "source_assets" / f"{asset_id}.json"
            contracts.write_json_no_replace(destination, upgraded)
            entries.append(
                {
                    "asset_id": asset_id,
                    "profile_schema_id": upgraded["profile_schema_id"],
                    "sampled_attributes": upgraded["sampled_attributes"],
                    "physical_measurements": upgraded["physical_measurements"],
                    "qa": upgraded["qa"],
                    "rights": upgraded["rights"],
                    "source_asset": {
                        "path": f"source_assets/{asset_id}.json",
                        "sha256": _sha256(destination),
                        "size_bytes": destination.stat().st_size,
                    },
                }
            )
        manifest: dict[str, Any] = {
            "schema": REGISTRY_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "source_asset_count": len(entries),
            "inputs": {
                "source_asset_roots": [str(Path(path).resolve()) for path in source_asset_roots],
                "apartment_manifests": [
                    static_registry.spear_artifact(Path(path)) for path in apartment_manifests
                ],
                "physical_measurement_batch": static_registry.spear_artifact(
                    measurement_batch
                ),
            },
            "source_assets": entries,
            "automatic_checks": {
                "all_static_source_assets_revalidated": True,
                "all_physical_measurements_observed": True,
                "all_animation_decisions_approved": True,
                "all_ue_imports_read_back": True,
                "all_walk_idle_apartment_media_passed": True,
                "all_species_audio_schedules_passed": True,
                "all_rights_blockers_preserved": all(
                    item["rights"]["status"] == "review_required"
                    and bool(item["rights"]["blockers"])
                    for item in entries
                ),
                "overall": "passed",
            },
        }
        manifest["registry_sha256"] = _hash_without(manifest, "registry_sha256")
        contracts.write_json_no_replace(staging / "registry_manifest.json", manifest)
        immutable._seal_readonly_tree(staging)
        os.rename(staging, output_root)
        return output_root / "registry_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-asset-root", action="append", required=True, type=Path)
    parser.add_argument("--apartment-manifest", action="append", required=True, type=Path)
    parser.add_argument("--measurement-batch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = register(
            source_asset_roots=args.source_asset_root,
            apartment_manifests=args.apartment_manifest,
            measurement_batch=args.measurement_batch,
            output_root=args.output_root,
        )
        payload = contracts.load_json(manifest)
    except (contracts.ContractError, OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"CONTROLLED_ANIMAL_APARTMENT_REGISTRATION_FAILED {error}", flush=True)
        return 2
    print(
        f"CONTROLLED_ANIMAL_APARTMENT_REGISTRATION_OK assets={payload['source_asset_count']} "
        f"output={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
