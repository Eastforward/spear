#!/usr/bin/env python3
"""Authenticate and freeze decisions for controlled static-object reviews."""

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

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_static_object_reviews as static_reviews


DECISIONS_SCHEMA = "avengine_controlled_static_object_review_decisions_v1"
DECISION_SCHEMA = "avengine_controlled_static_object_decision_v1"
DECISION_BATCH_SCHEMA = "avengine_controlled_static_object_decision_batch_v1"
APPROVED = "approved_for_watertight_finalization"
REJECTED = "rejected"
CHECK_FIELDS = {
    "silhouette_and_category_identity",
    "emitter_feature_visible",
    "material_and_declared_attributes",
    "physically_plausible_construction",
    "no_disconnected_or_floating_parts",
}
ATTRIBUTE_EVIDENCE_VALUES = {
    "passed_raw_pbr_visual",
    "failed_raw_pbr_visual",
    "not_visually_assessable",
}


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


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_review_artifact(record: Any, root: Path, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise contracts.ContractError(f"{label} record is invalid")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise contracts.ContractError(f"{label} path is invalid")
    unresolved = root / raw_path
    if Path(raw_path).is_absolute():
        unresolved = Path(raw_path)
    if unresolved.is_symlink():
        raise contracts.ContractError(f"{label} must not be a symlink")
    path = unresolved.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise contracts.ContractError(f"{label} escaped its review root") from error
    if (
        not path.is_file()
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] < 0
        or path.stat().st_size != record["size_bytes"]
        or _sha256_file(path) != record.get("sha256")
    ):
        raise contracts.ContractError(f"{label} changed")
    return path


def _review_artifact_records(review: Mapping[str, Any]) -> list[tuple[str, Any]]:
    records: list[tuple[str, Any]] = [
        ("raw PBR render manifest", review.get("raw_pbr_render_manifest")),
        ("raw PBR Blender log", review.get("raw_pbr_blender_log")),
        ("contact sheet", review.get("contact_sheet")),
    ]
    raw_views = review.get("raw_pbr_views")
    if not isinstance(raw_views, Mapping) or set(raw_views) != set(
        static_reviews.VIEW_RECORD_KEYS
    ):
        raise contracts.ContractError("raw PBR five-view evidence is incomplete")
    records.extend((f"raw PBR {name}", value) for name, value in raw_views.items())

    clay = review.get("clay_geometry")
    if not isinstance(clay, Mapping) or clay.get("status") not in {
        "included",
        "not_requested",
    }:
        raise contracts.ContractError("clay geometry evidence contract is invalid")
    if clay["status"] == "not_requested":
        if set(clay) != {"status"}:
            raise contracts.ContractError("unrequested clay evidence must be empty")
    else:
        if set(clay) != {"status", "render_manifest", "views", "blender_log"}:
            raise contracts.ContractError("included clay evidence fields are invalid")
        views = clay["views"]
        if not isinstance(views, Mapping) or set(views) != set(
            static_reviews.VIEW_RECORD_KEYS
        ):
            raise contracts.ContractError("clay five-view evidence is incomplete")
        records.extend(
            [
                ("clay render manifest", clay["render_manifest"]),
                ("clay Blender log", clay["blender_log"]),
                *((f"clay {name}", value) for name, value in views.items()),
            ]
        )
    return records


def _validate_render_evidence(
    review: Mapping[str, Any], root: Path, pixal_output: Path
) -> None:
    resolved = {
        label: _resolve_review_artifact(record, root, label)
        for label, record in _review_artifact_records(review)
    }
    raw_manifest = contracts.load_json(resolved["raw PBR render manifest"])
    if (
        not isinstance(raw_manifest, dict)
        or Path(raw_manifest.get("input", "")).resolve() != pixal_output
        or raw_manifest.get("front_axis") != "negative-y"
        or set(raw_manifest.get("views", {})) != set(static_reviews.RENDER_VIEWS)
        or raw_manifest.get("resolution") != [480, 480]
        or raw_manifest.get("material_preview", {}).get("mode")
        != static_reviews.RAW_PBR_MODE
    ):
        raise contracts.ContractError("raw PBR render evidence contract changed")

    clay = review["clay_geometry"]
    if clay["status"] == "included":
        clay_manifest = contracts.load_json(resolved["clay render manifest"])
        if (
            not isinstance(clay_manifest, dict)
            or Path(clay_manifest.get("input", "")).resolve() != pixal_output
            or clay_manifest.get("front_axis") != "negative-y"
            or set(clay_manifest.get("views", {}))
            != set(static_reviews.RENDER_VIEWS)
            or clay_manifest.get("resolution") != [480, 480]
            or clay_manifest.get("material_preview", {}).get("mode")
            != static_reviews.CLAY_MODE
        ):
            raise contracts.ContractError("clay render evidence contract changed")

    image_labels = [
        label
        for label in resolved
        if label.startswith("raw PBR orbit_") or label.startswith("clay orbit_")
    ]
    try:
        for label in image_labels:
            with Image.open(resolved[label]) as opened:
                opened.load()
                if opened.size != (480, 480):
                    raise contracts.ContractError(
                        "static-object review image resolution changed"
                    )
        expected_sheet_size = (
            (960, 1280) if clay["status"] == "included" else (960, 640)
        )
        with Image.open(resolved["contact sheet"]) as opened:
            opened.load()
            if opened.size != expected_sheet_size:
                raise contracts.ContractError(
                    "static-object contact sheet dimensions changed"
                )
    except OSError as error:
        raise contracts.ContractError(
            "static-object review image evidence is unreadable"
        ) from error


def load_review_batch(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    unresolved = Path(path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(f"static-object review batch is missing: {path}")
    path = unresolved.resolve()
    batch = contracts.load_json(path)
    render_contract = batch.get("render_contract") if isinstance(batch, dict) else None
    if (
        not isinstance(batch, dict)
        or batch.get("schema") != static_reviews.REVIEW_BATCH_SCHEMA
        or batch.get("status") != "rendered_pending_visual_qa"
        or batch.get("state_classification") != "research_candidate"
        or batch.get("formal_dataset_registration_authorized") is not False
        or batch.get("asset_class") != static_reviews.STATIC_ASSET_CLASS
        or batch.get("route") != static_reviews.STATIC_ROUTE
        or batch.get("orientation") != static_reviews._orientation_contract()
        or not isinstance(render_contract, Mapping)
        or set(render_contract)
        != {"reference_rgba_included", "raw_pbr_views", "clay_geometry"}
        or render_contract.get("reference_rgba_included") is not True
        or render_contract.get("raw_pbr_views")
        != list(static_reviews.VIEW_RECORD_KEYS)
        or render_contract.get("clay_geometry")
        not in {"included", "not_requested"}
        or batch.get("review_batch_sha256")
        != _hash_without(batch, "review_batch_sha256")
        or batch.get("automatic_checks", {}).get("overall") != "passed"
        or not isinstance(batch.get("reviews"), list)
        or batch.get("review_count") != len(batch["reviews"])
        or not batch["reviews"]
    ):
        raise contracts.ContractError(
            "static-object review batch contract/hash is invalid"
        )

    pixal_record = batch.get("pixal_batch")
    if not isinstance(pixal_record, Mapping) or set(pixal_record) != {
        "path",
        "sha256",
        "batch_sha256",
    }:
        raise contracts.ContractError("review/Pixal batch binding is invalid")
    pixal_path_value = pixal_record.get("path")
    if not isinstance(pixal_path_value, str) or not pixal_path_value:
        raise contracts.ContractError("review/Pixal batch path is invalid")
    pixal_path, pixal_batch, pixal_bindings = static_reviews.load_static_pixal_batch(
        Path(pixal_path_value)
    )
    if (
        _sha256_file(pixal_path) != pixal_record.get("sha256")
        or pixal_batch["batch_sha256"] != pixal_record.get("batch_sha256")
    ):
        raise contracts.ContractError("review/Pixal batch hash binding changed")

    root = path.parent
    reviews: dict[str, dict[str, Any]] = {}
    for index in batch["reviews"]:
        if not isinstance(index, Mapping):
            raise contracts.ContractError("static-object review index is invalid")
        instance_id = index.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or instance_id in reviews
            or instance_id not in pixal_bindings
        ):
            raise contracts.ContractError("static-object review coverage changed")
        review_path = _resolve_review_artifact(
            index.get("review"), root, f"{instance_id} review"
        )
        review = contracts.load_json(review_path)
        binding = pixal_bindings[instance_id]
        attempt = binding["attempt"]
        controlled = binding["controlled_request"]
        if (
            not isinstance(review, dict)
            or review.get("schema") != static_reviews.REVIEW_SCHEMA
            or review.get("review_sha256") != index.get("review_sha256")
            or review.get("review_sha256")
            != _hash_without(review, "review_sha256")
            or review.get("instance_id") != instance_id
            or review.get("execution_job_id") != attempt["execution_job_id"]
            or review.get("request_sha256") != attempt["request_sha256"]
            or review.get("profile_schema_id") != attempt["profile_schema_id"]
            or review.get("profile_sha256") != controlled["profile_sha256"]
            or review.get("asset_class") != static_reviews.STATIC_ASSET_CLASS
            or review.get("route") != static_reviews.STATIC_ROUTE
            or review.get("sampled_attributes") != attempt["sampled_attributes"]
            or review.get("target_physical_profile")
            != attempt["target_physical_profile"]
            or review.get("physical_scale")
            != static_reviews._physical_scale_contract(
                attempt["target_physical_profile"]
            )
            or review.get("orientation") != static_reviews._orientation_contract()
            or review.get("pixal_output") != attempt["output"]
            or review.get("mesh_readback") != binding["mesh_readback"]
            or review.get("reference_rgba") != attempt["pixal_input"]
            or review.get("state_classification") != "research_candidate"
            or review.get("formal_dataset_registration_authorized") is not False
            or review.get("automatic_checks", {}).get("overall") != "passed"
            or review.get("visual_qa") != "pending"
            or review.get("next_gate") != "static_object_visual_decision"
        ):
            raise contracts.ContractError(
                "static-object review record contract/hash is invalid"
            )
        if review["clay_geometry"].get("status") != render_contract[
            "clay_geometry"
        ]:
            raise contracts.ContractError(
                "static-object batch/review clay contract differs"
            )
        if (
            review["mesh_readback"].get("skins") != 0
            or review["mesh_readback"].get("animations") != 0
        ):
            raise contracts.ContractError(
                "static-object review cannot contain skins or animations"
            )
        _validate_render_evidence(review, root, binding["glb"])
        reviews[instance_id] = {
            "payload": review,
            "path": review_path,
            "pixal_output_path": binding["glb"],
        }
    if set(reviews) != set(pixal_bindings):
        raise contracts.ContractError(
            "static-object reviews do not cover every Pixal attempt"
        )
    return path, batch, reviews


def load_decisions(
    path: Path,
    review_batch: Mapping[str, Any],
    reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    unresolved = Path(path)
    if unresolved.is_symlink() or not unresolved.is_file():
        raise contracts.ContractError(f"static-object decisions are missing: {path}")
    path = unresolved.resolve()
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"schema", "static_object_review_batch_sha256", "decisions"}
        or payload.get("schema") != DECISIONS_SCHEMA
        or payload.get("static_object_review_batch_sha256")
        != review_batch["review_batch_sha256"]
        or not isinstance(payload.get("decisions"), list)
    ):
        raise contracts.ContractError(
            "static-object decisions contract/batch hash is invalid"
        )

    expected_fields = {
        "instance_id",
        "review_sha256",
        "decision",
        "checks",
        "attribute_evidence",
        "caveats",
        "notes",
    }
    decisions: dict[str, dict[str, Any]] = {}
    for decision in payload["decisions"]:
        if not isinstance(decision, dict) or set(decision) != expected_fields:
            raise contracts.ContractError("static-object decision fields are invalid")
        instance_id = decision.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or instance_id in decisions
            or instance_id not in reviews
        ):
            raise contracts.ContractError(
                "static-object decision coverage/identity is invalid"
            )
        review = reviews[instance_id]["payload"]
        if decision.get("review_sha256") != review["review_sha256"]:
            raise contracts.ContractError("static-object decision review hash changed")
        if decision.get("decision") not in {APPROVED, REJECTED}:
            raise contracts.ContractError("invalid static-object decision")

        checks = decision.get("checks")
        if (
            not isinstance(checks, dict)
            or set(checks) != CHECK_FIELDS
            or any(not isinstance(value, bool) for value in checks.values())
        ):
            raise contracts.ContractError("static-object decision checks are invalid")
        evidence = decision.get("attribute_evidence")
        if (
            not isinstance(evidence, dict)
            or set(evidence) != set(review["sampled_attributes"])
            or any(
                value not in ATTRIBUTE_EVIDENCE_VALUES
                for value in evidence.values()
            )
        ):
            raise contracts.ContractError(
                "static-object declared-attribute evidence is invalid"
            )
        if decision["decision"] == APPROVED:
            if not all(checks.values()):
                raise contracts.ContractError(
                    "approved static-object decision has a failed check"
                )
            if any(value != "passed_raw_pbr_visual" for value in evidence.values()):
                raise contracts.ContractError(
                    "approved static-object attribute evidence is incomplete"
                )
        elif all(checks.values()):
            raise contracts.ContractError(
                "rejected static-object decision needs a failed check"
            )

        caveats = decision.get("caveats")
        notes = decision.get("notes")
        if (
            not isinstance(caveats, list)
            or len(caveats) != len(set(caveats))
            or any(not isinstance(item, str) or not item.strip() for item in caveats)
            or not isinstance(notes, str)
            or not notes.strip()
        ):
            raise contracts.ContractError(
                "static-object decision notes/caveats are invalid"
            )
        decisions[instance_id] = copy.deepcopy(decision)
    if set(decisions) != set(reviews):
        raise contracts.ContractError("static-object decision coverage is incomplete")
    return payload


def publish_decisions(
    review_batch_path: Path,
    decisions_path: Path,
    output_root: Path,
) -> Path:
    review_batch_path, review_batch, reviews = load_review_batch(review_batch_path)
    decisions_payload = load_decisions(decisions_path, review_batch, reviews)
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
        records = []
        for decision in sorted(
            decisions_payload["decisions"], key=lambda item: item["instance_id"]
        ):
            review = reviews[decision["instance_id"]]
            approved = decision["decision"] == APPROVED
            record: dict[str, Any] = {
                "schema": DECISION_SCHEMA,
                **copy.deepcopy(decision),
                "asset_class": static_reviews.STATIC_ASSET_CLASS,
                "route": static_reviews.STATIC_ROUTE,
                "request_sha256": review["payload"]["request_sha256"],
                "profile_sha256": review["payload"]["profile_sha256"],
                "target_physical_profile": copy.deepcopy(
                    review["payload"]["target_physical_profile"]
                ),
                "pixal_output": _record(review["pixal_output_path"]),
                "review": _record(review["path"]),
                "physical_scale": copy.deepcopy(
                    review["payload"]["physical_scale"]
                ),
                "canonical_heading": copy.deepcopy(
                    review["payload"]["orientation"]["canonical_heading"]
                ),
                "state_classification": (
                    "research_candidate" if approved else "rejected"
                ),
                "formal_dataset_registration_authorized": False,
                "next_gate": (
                    "watertight_then_static_finalization" if approved else "stop"
                ),
            }
            record["decision_sha256"] = _hash_without(record, "decision_sha256")
            destination = (
                staging
                / decision["instance_id"]
                / "static_object_decision.json"
            )
            contracts.write_json_no_replace(destination, record)
            records.append(
                {
                    "instance_id": decision["instance_id"],
                    "request_sha256": record["request_sha256"],
                    "profile_sha256": record["profile_sha256"],
                    "pixal_output_sha256": record["pixal_output"]["sha256"],
                    "decision": decision["decision"],
                    "decision_sha256": record["decision_sha256"],
                    "record": {
                        "path": destination.relative_to(staging).as_posix(),
                        "sha256": _sha256_file(destination),
                        "size_bytes": destination.stat().st_size,
                    },
                }
            )

        approved_count = sum(item["decision"] == APPROVED for item in records)
        batch: dict[str, Any] = {
            "schema": DECISION_BATCH_SCHEMA,
            "status": "completed",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "asset_class": static_reviews.STATIC_ASSET_CLASS,
            "route": static_reviews.STATIC_ROUTE,
            "static_object_review_batch": {
                "path": str(review_batch_path),
                "sha256": _sha256_file(review_batch_path),
                "review_batch_sha256": review_batch["review_batch_sha256"],
            },
            "decision_input": _record(Path(decisions_path)),
            "decision_count": len(records),
            "approved_count": approved_count,
            "rejected_count": len(records) - approved_count,
            "decisions": records,
            "automatic_checks": {
                "all_review_and_pixal_hashes_reauthenticated": True,
                "all_instances_have_one_decision": True,
                "all_physical_scale_checks_deferred_to_finalization": True,
                "approved_assets_route_to_watertight_then_static_finalization": True,
                "no_formal_registration_authorized": True,
                "overall": "passed",
            },
        }
        batch["decision_batch_sha256"] = _hash_without(
            batch, "decision_batch_sha256"
        )
        destination = staging / "static_object_decision_batch_manifest.json"
        contracts.write_json_no_replace(destination, batch)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                "static-object decision output appeared concurrently"
            )
        os.rename(staging, output_root)
        return output_root / destination.name
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-object-review-batch", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = publish_decisions(
            args.static_object_review_batch,
            args.decisions,
            args.output_root,
        )
        payload = contracts.load_json(manifest)
    except (contracts.ContractError, OSError, json.JSONDecodeError) as error:
        print(f"CONTROLLED_STATIC_OBJECT_DECISION_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_STATIC_OBJECT_DECISION_OK "
        f"approved={payload['approved_count']} rejected={payload['rejected_count']} "
        f"output={manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
