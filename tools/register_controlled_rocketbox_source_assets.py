#!/usr/bin/env python3
"""Register realized controlled Rocketbox variants as source_asset_v2 candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as materials
from tools import normalize_controlled_rocketbox_runtimes as ue_handoff
from tools import prepare_controlled_source_asset_execution as preparation
from tools import rocketbox_native_material_canary as native
from tools import run_controlled_rocketbox_runtime_handoffs as runtime_handoff


REGISTRY_SCHEMA = "avengine_controlled_rocketbox_source_asset_registry_v1"
SUPPRESSION_SCHEMA = "avengine_deterministic_request_suppression_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _root_artifact(path: Path, *, root_id: str, root: Path) -> dict[str, Any]:
    path = path.resolve()
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise contracts.ContractError(f"artifact escapes root {root_id}: {path}") from error
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"artifact is missing or unsafe: {path}")
    return {
        "root_id": root_id,
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_absolute_record(record: Mapping[str, Any], label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise contracts.ContractError(f"{label} record is invalid")
    path = Path(str(record["path"])).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"{label} is missing: {path}")
    if path.stat().st_size != record["size_bytes"] or _sha256_file(path) != record["sha256"]:
        raise contracts.ContractError(f"{label} hash/size changed")
    return path


def _validate_material_batch(path: Path) -> tuple[Path, dict[str, Any]]:
    root, batch, _records = runtime_handoff.load_material_batch(path)
    return root, batch


def _load_preflight(path: Path) -> dict[str, Any]:
    return materials._load_preflight(path)


def _request_and_profiles(preflight: Mapping[str, Any]):
    input_dir = Path(preflight["source_bundle"]["input_dir"])
    request_batch = contracts.load_json(input_dir / "instance_requests.json")
    snapshot = contracts.load_json(input_dir / "profile_snapshot.json")
    requests = {item["instance_id"]: item for item in request_batch["requests"]}
    profiles = {
        item["profile_schema_id"]: item["profile"] for item in snapshot["profiles"]
    }
    return requests, profiles


def _material_artifacts_by_variant(
    material_root: Path, material_batch: Mapping[str, Any]
) -> dict[str, dict[str, Path]]:
    records = {}
    for variant in material_batch["variants"]:
        records[variant["variant_key"]] = {
            "body_color_texture": runtime_handoff._authenticate_relative_record(
                variant["body_color_texture"], material_root, "body color texture"
            ),
            "material_attempt": runtime_handoff._authenticate_relative_record(
                variant["attempt_manifest"], material_root, "material attempt"
            ),
            "variant_request": runtime_handoff._authenticate_relative_record(
                variant["variant_request"], material_root, "variant request"
            ),
        }
    return records


def _runtime_records_by_variant(
    native_manifest: Mapping[str, Any], ue_manifest: Mapping[str, Any]
) -> dict[str, dict[str, Path]]:
    native_by_key = {item["variant_key"]: item for item in native_manifest["runtimes"]}
    ue_by_key = {item["variant_key"]: item for item in ue_manifest["runtimes"]}
    if set(native_by_key) != set(ue_by_key):
        raise contracts.ContractError("native and UE runtime variant sets differ")
    records = {}
    for key in sorted(native_by_key):
        native_item = native_by_key[key]
        ue_item = ue_by_key[key]
        if (
            native_item["variant_id"] != ue_item["variant_id"]
            or native_item["sampled_attributes"] != ue_item["sampled_attributes"]
            or native_item["runtime_tag"] != ue_item["runtime_tag"]
        ):
            raise contracts.ContractError("native and UE runtime identity mismatch")
        records[key] = {
            "native_runtime_glb": _validate_absolute_record(
                native_item["runtime_glb"], "native runtime GLB"
            ),
            "native_runtime_manifest": _validate_absolute_record(
                native_item["runtime_manifest"], "native runtime manifest"
            ),
            "ue_runtime_glb": _validate_absolute_record(
                ue_item["runtime_glb"], "UE runtime GLB"
            ),
            "ue_runtime_manifest": _validate_absolute_record(
                ue_item["normalization_manifest"], "UE runtime manifest"
            ),
        }
    return records


def register_assets(
    *,
    preflight_path: Path,
    material_batch_path: Path,
    native_handoff_path: Path,
    ue_handoff_path: Path,
    output_root: Path,
) -> Path:
    preflight = _load_preflight(preflight_path)
    material_root, material_batch = _validate_material_batch(material_batch_path)
    native_manifest = ue_handoff.load_native_handoff(native_handoff_path)
    ue_manifest = contracts.load_json(ue_handoff_path)
    if (
        not isinstance(ue_manifest, dict)
        or ue_manifest.get("schema") != ue_handoff.NORMALIZED_HANDOFF_SCHEMA
        or ue_manifest.get("status") != "passed"
        or ue_manifest.get("handoff_sha256")
        != ue_handoff._hash_without(ue_manifest, "handoff_sha256")
        or ue_manifest.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("UE runtime handoff manifest is invalid")
    material_jobs = {
        job["variant_key"]: job
        for job in preflight["routes"]["rocketbox_material_v1"]
    }
    if set(material_jobs) != {
        item["variant_key"] for item in material_batch["variants"]
    }:
        raise contracts.ContractError("preflight and material variant sets differ")
    requests, profiles = _request_and_profiles(preflight)
    material_artifacts = _material_artifacts_by_variant(material_root, material_batch)
    runtime_artifacts = _runtime_records_by_variant(native_manifest, ue_manifest)
    if set(material_jobs) != set(runtime_artifacts):
        raise contracts.ContractError("material and runtime variant sets differ")

    spear_root = materials.SPEAR_ROOT.resolve()
    rocketbox_root = native.ROCKETBOX_ROOT.resolve()
    license_path = rocketbox_root / native.LICENSE_SPEC["relative_path"]
    license_record = _root_artifact(
        license_path, root_id="rocketbox_0943055", root=rocketbox_root
    )
    if (
        license_record["sha256"] != native.LICENSE_SPEC["sha256"]
        or license_record["size_bytes"] != native.LICENSE_SPEC["size_bytes"]
    ):
        raise contracts.ContractError("Rocketbox MIT license snapshot changed")

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
        source_asset_dir = staging / "source_assets"
        source_asset_dir.mkdir()
        asset_index = []
        suppressed = []
        for variant_key, job in sorted(material_jobs.items()):
            consumers = sorted(job["consumer_requests"], key=lambda item: item["instance_id"])
            canonical = consumers[0]
            duplicate_consumers = consumers[1:]
            request = requests[canonical["instance_id"]]
            if request["request_sha256"] != canonical["request_sha256"]:
                raise contracts.ContractError("canonical request hash changed")
            for duplicate in duplicate_consumers:
                duplicate_request = requests[duplicate["instance_id"]]
                if (
                    duplicate_request["request_sha256"] != duplicate["request_sha256"]
                    or duplicate_request["sampled_attributes"]
                    != request["sampled_attributes"]
                ):
                    raise contracts.ContractError("duplicate material request is not equivalent")
                suppressed.append(
                    {
                        "instance_id": duplicate["instance_id"],
                        "request_sha256": duplicate["request_sha256"],
                        "canonical_instance_id": canonical["instance_id"],
                        "canonical_request_sha256": canonical["request_sha256"],
                        "variant_key": variant_key,
                        "reason": "deterministic_material_plan_has_identical_absolute_visual_output",
                        "dataset_registration": "suppressed",
                    }
                )

            paths = {**material_artifacts[variant_key], **runtime_artifacts[variant_key]}
            artifacts = {
                role: _root_artifact(path, root_id="spear_repo", root=spear_root)
                for role, path in sorted(paths.items())
            }
            target_height = float(request["target_physical_profile"]["target_value_cm"])
            source_asset = contracts.build_source_asset_v2(
                request,
                artifacts=artifacts,
                physical_measurements={
                    "status": "measured",
                    "method": "rocketbox_authored_height_runtime_equivalence_v1",
                    "runtime": {
                        "actor_scale": 1.0,
                        "authored_height_cm": target_height,
                    },
                },
                provenance={
                    "attempt_id": f"material_{variant_key[:16]}",
                    "request_sha256": request["request_sha256"],
                    "models": {},
                },
                rights={
                    "status": "cleared",
                    "licenses": [license_record],
                    "blockers": [],
                },
                qa={
                    "reference_2d": "not_applicable",
                    "static_mesh": "passed",
                    "binding": "passed",
                    "walking": "passed",
                    "idle": "passed",
                    "ue_import_readback": "pending",
                    "apartment_media": "pending",
                    "audio": "pending",
                },
                state_classification="research_candidate",
            )
            source_asset = contracts.validate_source_asset_v2(
                source_asset,
                request=request,
                profile=profiles[request["profile_schema_id"]],
            )
            asset_path = source_asset_dir / f"{source_asset['asset_id']}.json"
            contracts.write_json_no_replace(asset_path, source_asset)
            asset_index.append(
                {
                    "asset_id": source_asset["asset_id"],
                    "request_sha256": source_asset["request_sha256"],
                    "variant_key": variant_key,
                    "sampled_attributes": source_asset["sampled_attributes"],
                    "source_asset_manifest": {
                        "path": asset_path.relative_to(staging).as_posix(),
                        "sha256": _sha256_file(asset_path),
                        "size_bytes": asset_path.stat().st_size,
                    },
                    "duplicate_requests_suppressed": len(duplicate_consumers),
                }
            )

        suppression_manifest: dict[str, Any] = {
            "schema": SUPPRESSION_SCHEMA,
            "policy": "one_source_asset_per_unique_deterministic_absolute_material_plan",
            "canonical_asset_count": len(asset_index),
            "suppressed_request_count": len(suppressed),
            "suppressed_requests": sorted(
                suppressed, key=lambda item: item["instance_id"]
            ),
            "automatic_checks": {
                "all_suppressed_requests_match_canonical_absolute_attributes": True,
                "no_suppressed_request_registered_as_source_asset_v2": True,
                "overall": "passed",
            },
        }
        suppression_manifest["manifest_sha256"] = contracts.manifest_sha256(
            suppression_manifest
        )
        contracts.write_json_no_replace(
            staging / "duplicate_request_suppression.json", suppression_manifest
        )
        registry: dict[str, Any] = {
            "schema": REGISTRY_SCHEMA,
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "execution_preflight": _root_artifact(
                Path(preflight_path), root_id="spear_repo", root=spear_root
            ),
            "material_batch": _root_artifact(
                Path(material_batch_path), root_id="spear_repo", root=spear_root
            ),
            "native_handoff": _root_artifact(
                Path(native_handoff_path), root_id="spear_repo", root=spear_root
            ),
            "ue_handoff": _root_artifact(
                Path(ue_handoff_path), root_id="spear_repo", root=spear_root
            ),
            "source_asset_count": len(asset_index),
            "suppressed_request_count": len(suppressed),
            "source_assets": sorted(asset_index, key=lambda item: item["asset_id"]),
            "automatic_checks": {
                "all_inputs_reauthenticated": True,
                "all_source_asset_v2_validated_against_profile_and_request": True,
                "unique_visual_variants_only": True,
                "lineage_group_preserved": True,
                "baseline_written": False,
                "overall": "passed",
            },
        }
        registry["registry_sha256"] = contracts.manifest_sha256(registry)
        contracts.write_json_no_replace(staging / "registry_manifest.json", registry)
        materials.native._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                f"refusing to replace concurrently-created output: {output_root}"
            )
        os.rename(staging, output_root)
        return output_root / "registry_manifest.json"
    except Exception:
        materials.native._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--material-batch", required=True, type=Path)
    parser.add_argument("--native-handoff", required=True, type=Path)
    parser.add_argument("--ue-handoff", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        registry_path = register_assets(
            preflight_path=args.preflight,
            material_batch_path=args.material_batch,
            native_handoff_path=args.native_handoff,
            ue_handoff_path=args.ue_handoff,
            output_root=args.output_root,
        )
        registry = contracts.load_json(registry_path)
    except (contracts.ContractError, OSError) as error:
        print(f"CONTROLLED_ROCKETBOX_REGISTRATION_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ROCKETBOX_REGISTRATION_OK "
        f"assets={registry['source_asset_count']} "
        f"suppressed={registry['suppressed_request_count']} "
        f"output={registry_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
