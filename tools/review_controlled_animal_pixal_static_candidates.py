#!/usr/bin/env python3
"""Authenticate and freeze visual decisions for controlled animal Pixal reviews."""

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
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_static_reviews as static_reviews


DECISIONS_SCHEMA = "avengine_controlled_animal_static_decisions_v1"
DECISION_BATCH_SCHEMA = "avengine_controlled_animal_static_decision_batch_v1"
CHECK_FIELDS = {
    "complete_silhouette",
    "four_limbs_usable",
    "texture_coherent",
    "pose_riggable",
    "no_large_holes",
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
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_review_batch(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"static review batch is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != static_reviews.REVIEW_BATCH_SCHEMA
        or payload.get("status") != "rendered_pending_visual_qa"
        or payload.get("review_batch_sha256")
        != _hash_without(payload, "review_batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
        or payload.get("review_count") != len(payload.get("reviews", []))
    ):
        raise contracts.ContractError("static review batch contract/hash is invalid")
    reviews = {}
    root = path.parent
    for index in payload["reviews"]:
        review_path = (root / index["review"]["path"]).resolve()
        try:
            review_path.relative_to(root)
        except ValueError as error:
            raise contracts.ContractError("static review escaped its root") from error
        if (
            review_path.is_symlink()
            or not review_path.is_file()
            or review_path.stat().st_size != index["review"]["size_bytes"]
            or _sha256_file(review_path) != index["review"]["sha256"]
        ):
            raise contracts.ContractError("static review artifact changed")
        review = contracts.load_json(review_path)
        if (
            review.get("schema") != static_reviews.REVIEW_SCHEMA
            or review.get("instance_id") != index["instance_id"]
            or review.get("review_sha256") != index["review_sha256"]
            or review.get("review_sha256") != _hash_without(review, "review_sha256")
            or review.get("automatic_checks", {}).get("overall") != "passed"
        ):
            raise contracts.ContractError("static review record contract/hash is invalid")
        for artifact in [
            review["render_manifest"],
            review["contact_sheet"],
            review["blender_log"],
            *review["views"].values(),
        ]:
            artifact_path = (root / artifact["path"]).resolve()
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != artifact["size_bytes"]
                or _sha256_file(artifact_path) != artifact["sha256"]
            ):
                raise contracts.ContractError("static review evidence changed")
        reviews[review["instance_id"]] = {"payload": review, "path": review_path}
    if len(reviews) != payload["review_count"]:
        raise contracts.ContractError("duplicate static review instance ID")
    return path, payload, reviews


def load_decisions(
    path: Path, review_batch: dict[str, Any], reviews: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "static_review_batch_sha256", "decisions"}
        or payload.get("schema") != DECISIONS_SCHEMA
        or payload.get("static_review_batch_sha256")
        != review_batch["review_batch_sha256"]
        or not isinstance(payload.get("decisions"), list)
    ):
        raise contracts.ContractError("static decisions contract/batch hash is invalid")
    decisions = {}
    expected_fields = {
        "instance_id",
        "review_sha256",
        "decision",
        "checks",
        "attribute_evidence",
        "caveats",
        "notes",
    }
    for decision in payload["decisions"]:
        if not isinstance(decision, dict) or set(decision) != expected_fields:
            raise contracts.ContractError("static decision fields are invalid")
        instance_id = decision["instance_id"]
        if instance_id in decisions or instance_id not in reviews:
            raise contracts.ContractError("static decision coverage/identity is invalid")
        review = reviews[instance_id]["payload"]
        if decision["review_sha256"] != review["review_sha256"]:
            raise contracts.ContractError("static decision review hash changed")
        if decision["decision"] not in {"approved_for_lod_and_binding", "rejected"}:
            raise contracts.ContractError("invalid static decision")
        if (
            not isinstance(decision["checks"], dict)
            or set(decision["checks"]) != CHECK_FIELDS
            or any(not isinstance(value, bool) for value in decision["checks"].values())
        ):
            raise contracts.ContractError("static decision checks are invalid")
        expected_attributes = set(review["sampled_attributes"])
        evidence = decision["attribute_evidence"]
        if not isinstance(evidence, dict) or set(evidence) != expected_attributes:
            raise contracts.ContractError("static attribute evidence coverage is invalid")
        control = review["target_physical_profile"]["control_attribute"]
        if evidence.get(control) != "deferred_to_metric_3d":
            raise contracts.ContractError("animal physical control must remain deferred")
        if decision["decision"] == "approved_for_lod_and_binding":
            if not all(decision["checks"].values()):
                raise contracts.ContractError("approved static decision has a failed check")
            if any(
                status != "passed_static_visual"
                for attribute, status in evidence.items()
                if attribute != control
            ):
                raise contracts.ContractError("approved appearance evidence is incomplete")
        elif not any(not value for value in decision["checks"].values()):
            raise contracts.ContractError("rejected static decision needs a failed check")
        if (
            not isinstance(decision["caveats"], list)
            or len(decision["caveats"]) != len(set(decision["caveats"]))
            or any(not isinstance(item, str) or not item for item in decision["caveats"])
            or not isinstance(decision["notes"], str)
            or not decision["notes"].strip()
        ):
            raise contracts.ContractError("static decision notes/caveats are invalid")
        decisions[instance_id] = copy.deepcopy(decision)
    if set(decisions) != set(reviews):
        raise contracts.ContractError("static decision coverage is incomplete")
    return payload


def publish_decisions(
    review_batch_path: Path, decisions_path: Path, output_root: Path
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
            record = {
                "schema": "avengine_controlled_animal_static_decision_v1",
                **copy.deepcopy(decision),
                "review": _record(review["path"]),
                "state_classification": (
                    "research_candidate"
                    if decision["decision"] == "approved_for_lod_and_binding"
                    else "rejected"
                ),
                "formal_dataset_registration_authorized": False,
                "next_gate": (
                    "lod_then_species_rig_binding"
                    if decision["decision"] == "approved_for_lod_and_binding"
                    else "stop"
                ),
            }
            record["decision_sha256"] = _hash_without(record, "decision_sha256")
            destination = staging / decision["instance_id"] / "static_decision.json"
            contracts.write_json_no_replace(destination, record)
            records.append(
                {
                    "instance_id": decision["instance_id"],
                    "decision": decision["decision"],
                    "decision_sha256": record["decision_sha256"],
                    "record": {
                        "path": destination.relative_to(staging).as_posix(),
                        "sha256": _sha256_file(destination),
                        "size_bytes": destination.stat().st_size,
                    },
                }
            )
        approved = sum(
            item["decision"] == "approved_for_lod_and_binding" for item in records
        )
        batch: dict[str, Any] = {
            "schema": DECISION_BATCH_SCHEMA,
            "status": "completed",
            "static_review_batch": {
                "path": str(review_batch_path),
                "sha256": _sha256_file(review_batch_path),
                "review_batch_sha256": review_batch["review_batch_sha256"],
            },
            "decision_input": _record(Path(decisions_path)),
            "decision_count": len(records),
            "approved_count": approved,
            "rejected_count": len(records) - approved,
            "decisions": records,
            "automatic_checks": {
                "all_review_hashes_reauthenticated": True,
                "all_multiview_artifacts_reauthenticated": True,
                "all_instances_have_one_decision": True,
                "all_metric_size_evidence_deferred": True,
                "no_formal_registration_authorized": True,
                "overall": "passed",
            },
        }
        batch["decision_batch_sha256"] = _hash_without(
            batch, "decision_batch_sha256"
        )
        contracts.write_json_no_replace(staging / "static_decision_batch_manifest.json", batch)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("static decision output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "static_decision_batch_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-review-batch", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = publish_decisions(
            args.static_review_batch, args.decisions, args.output_root
        )
        payload = contracts.load_json(manifest)
    except (contracts.ContractError, OSError, json.JSONDecodeError) as error:
        print(f"CONTROLLED_ANIMAL_STATIC_DECISION_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_STATIC_DECISION_OK "
        f"approved={payload['approved_count']} rejected={payload['rejected_count']} "
        f"output={manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
