#!/usr/bin/env python3
"""Register static-qualified controlled animal outputs as source_asset_v2 candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as preflight_tools
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
    path = Path(path).resolve()
    try:
        relative = path.relative_to(SPEAR_ROOT.resolve())
    except ValueError as error:
        raise contracts.ContractError(f"artifact is outside SPEAR root: {path}") from error
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise contracts.ContractError(f"artifact is missing/non-direct: {path}")
    return {
        "root_id": "spear_repo",
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def license_records() -> list[dict[str, Any]]:
    records = []
    for spec in LICENSE_SPECS:
        logical = MODELS_ROOT / spec["path"]
        resolved = logical.resolve()
        try:
            resolved.relative_to(MODELS_ROOT.resolve())
        except ValueError as error:
            raise contracts.ContractError("model license escaped /data/models") from error
        if (
            not resolved.is_file()
            or resolved.stat().st_size != spec["size_bytes"]
            or _sha256_file(resolved) != spec["sha256"]
        ):
            raise contracts.ContractError(f"model license changed: {logical}")
        records.append({"root_id": "models_root", **copy.deepcopy(spec)})
    return records


def load_decision_batch(path: Path):
    path = Path(path).resolve()
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
    records = {}
    root = path.parent
    for index in payload["decisions"]:
        record_path = (root / index["record"]["path"]).resolve()
        if (
            not record_path.is_file()
            or record_path.stat().st_size != index["record"]["size_bytes"]
            or _sha256_file(record_path) != index["record"]["sha256"]
        ):
            raise contracts.ContractError("static decision record changed")
        record = contracts.load_json(record_path)
        if (
            record.get("instance_id") != index["instance_id"]
            or record.get("decision") != index["decision"]
            or record.get("decision_sha256") != index["decision_sha256"]
            or record.get("decision_sha256")
            != _hash_without(record, "decision_sha256")
        ):
            raise contracts.ContractError("static decision identity/hash changed")
        records[index["instance_id"]] = {"payload": record, "path": record_path}
    if len(records) != payload["decision_count"]:
        raise contracts.ContractError("duplicate static decision records")
    return path, payload, records


def load_source_contract(preflight_path: Path):
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


def register(
    preflight_path: Path,
    pixal_batch_path: Path,
    decision_batch_path: Path,
    output_root: Path,
) -> Path:
    preflight_path = Path(preflight_path).resolve()
    preflight, requests, profiles = load_source_contract(preflight_path)
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
    pixal_inputs_manifest = contracts.load_json(pixal_inputs_path)
    input_jobs = {
        job["controlled_request"]["instance_id"]: job
        for job in pixal_inputs_manifest["jobs"]
    }
    if len(input_jobs) != len(pixal_inputs_manifest["jobs"]):
        raise contracts.ContractError("duplicate instances in Pixal inputs manifest")
    decision_batch_path, decision_batch, decisions = load_decision_batch(
        decision_batch_path
    )
    approved_ids = {
        instance_id
        for instance_id, value in decisions.items()
        if value["payload"]["decision"] == "approved_for_lod_and_binding"
    }
    attempts = {item["instance_id"]: item for item in pixal_batch["attempts"]}
    if approved_ids != set(attempts):
        raise contracts.ContractError("approved static decisions/Pixal attempts differ")
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
            static_review_path = Path(decision["payload"]["review"]["path"]).resolve()
            static_review = contracts.load_json(static_review_path)
            contact_path = (
                static_review_path.parents[1] / static_review["contact_sheet"]["path"]
            ).resolve()
            pixal_input_path = Path(attempt["pixal_input"]["path"]).resolve()
            input_job = input_jobs.get(instance_id)
            if input_job is None:
                raise contracts.ContractError("approved instance lacks a Pixal input job")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = register(
            args.preflight,
            args.pixal_batch,
            args.static_decision_batch,
            args.output_root,
        )
        payload = contracts.load_json(manifest)
    except (contracts.ContractError, OSError, json.JSONDecodeError) as error:
        print(f"CONTROLLED_ANIMAL_SOURCE_ASSET_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_SOURCE_ASSET_OK "
        f"assets={payload['source_asset_count']} output={manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
