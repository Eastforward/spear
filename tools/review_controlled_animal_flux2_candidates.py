#!/usr/bin/env python3
"""Validate and publish exact-hash 2D reviews for controlled animal candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_animal_flux2_worker as worker
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as material_execution
from tools import run_controlled_animal_flux2_jobs as flux_runner


DECISIONS_SCHEMA = "avengine_controlled_animal_2d_review_decisions_v1"
REVIEW_SCHEMA = "avengine_controlled_animal_2d_review_v1"
BATCH_REVIEW_SCHEMA = "avengine_controlled_animal_2d_review_batch_v1"
ATTRIBUTE_STATUSES = {
    "passed",
    "deferred_to_3d_physical_scale",
    "rejected",
}


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _sha256_file(path: Path) -> str:
    return flux_runner._sha256_file(path)


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_flux_batch(path: Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"FLUX.2 batch manifest is missing: {path}")
    root = path.parent
    batch = contracts.load_json(path)
    if (
        not isinstance(batch, dict)
        or batch.get("schema") != flux_runner.BATCH_SCHEMA
        or batch.get("status") != "pending_2d_review"
        or batch.get("batch_sha256") != _hash_without(batch, "batch_sha256")
        or batch.get("automatic_checks", {}).get("overall") != "pending_2d_review"
    ):
        raise contracts.ContractError("FLUX.2 batch manifest contract/hash is invalid")
    candidates = batch.get("candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != batch.get("candidate_count")
        or len({item.get("instance_id") for item in candidates}) != len(candidates)
    ):
        raise contracts.ContractError("FLUX.2 candidate index is invalid")
    by_instance = {}
    for candidate in candidates:
        files = {}
        for role in ("candidate", "candidate_manifest", "source"):
            record = candidate.get(role)
            if not isinstance(record, dict) or set(record) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise contracts.ContractError(f"candidate {role} record is invalid")
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise contracts.ContractError(f"candidate {role} path is unsafe")
            artifact = (root / relative).resolve()
            try:
                artifact.relative_to(root)
            except ValueError as error:
                raise contracts.ContractError(f"candidate {role} escaped root") from error
            if artifact.is_symlink() or not artifact.is_file():
                raise contracts.ContractError(f"candidate {role} is missing")
            if (
                artifact.stat().st_size != record["size_bytes"]
                or _sha256_file(artifact) != record["sha256"]
            ):
                raise contracts.ContractError(f"candidate {role} hash/size changed")
            files[role] = artifact
        manifest = contracts.load_json(files["candidate_manifest"])
        if (
            manifest.get("schema") != worker.CANDIDATE_SCHEMA
            or manifest.get("status") != "pending_2d_review"
            or manifest.get("instance_id") != candidate["instance_id"]
            or manifest.get("sampled_attributes") != candidate["sampled_attributes"]
            or manifest.get("output", {}).get("sha256")
            != candidate["candidate"]["sha256"]
            or manifest.get("manifest_sha256")
            != _hash_without(manifest, "manifest_sha256")
        ):
            raise contracts.ContractError("candidate manifest contract/hash is invalid")
        by_instance[candidate["instance_id"]] = {
            "index": candidate,
            "manifest": manifest,
            "files": files,
        }
    return root, batch, by_instance


def load_decisions(path: Path, batch: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"review decisions are missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "flux2_batch_sha256", "reviewer", "decisions"}
        or payload.get("schema") != DECISIONS_SCHEMA
        or payload.get("flux2_batch_sha256") != batch["batch_sha256"]
        or not isinstance(payload.get("reviewer"), str)
        or not payload["reviewer"].strip()
        or not isinstance(payload.get("decisions"), list)
    ):
        raise contracts.ContractError("review decisions contract is invalid")
    decisions = {}
    for decision in payload["decisions"]:
        fields = {
            "instance_id",
            "candidate_sha256",
            "decision",
            "species_breed",
            "anatomy",
            "pose_and_limb_separation",
            "background",
            "sampled_attribute_checks",
            "notes",
        }
        if not isinstance(decision, dict) or set(decision) != fields:
            raise contracts.ContractError("review decision fields are invalid")
        instance_id = decision["instance_id"]
        if instance_id in decisions:
            raise contracts.ContractError(f"duplicate review decision: {instance_id}")
        if decision["decision"] not in {"approved_for_pixal3d", "rejected"}:
            raise contracts.ContractError("review decision value is invalid")
        for check in ("species_breed", "anatomy", "pose_and_limb_separation", "background"):
            if decision[check] not in {"passed", "rejected"}:
                raise contracts.ContractError(f"review check is invalid: {check}")
        attribute_checks = decision["sampled_attribute_checks"]
        if not isinstance(attribute_checks, dict) or any(
            value not in ATTRIBUTE_STATUSES for value in attribute_checks.values()
        ):
            raise contracts.ContractError("sampled attribute review checks are invalid")
        if any(
            value == "deferred_to_3d_physical_scale" and key != "size"
            for key, value in attribute_checks.items()
        ):
            raise contracts.ContractError("only size may defer to physical 3D scaling")
        rejected = any(
            decision[key] == "rejected"
            for key in ("species_breed", "anatomy", "pose_and_limb_separation", "background")
        ) or "rejected" in attribute_checks.values()
        if (decision["decision"] == "rejected") != rejected:
            raise contracts.ContractError("review decision disagrees with its checks")
        if not isinstance(decision["notes"], str):
            raise contracts.ContractError("review notes must be text")
        decisions[instance_id] = decision
    return decisions


def publish_reviews(
    flux_batch_path: Path, decisions_path: Path, output_root: Path
) -> Path:
    flux_root, batch, candidates = load_flux_batch(flux_batch_path)
    decisions = load_decisions(decisions_path, batch)
    if set(decisions) != set(candidates):
        raise contracts.ContractError(
            "review decisions must cover every candidate exactly once"
        )
    for instance_id, candidate in candidates.items():
        decision = decisions[instance_id]
        if decision["candidate_sha256"] != candidate["index"]["candidate"]["sha256"]:
            raise contracts.ContractError(
                f"review decision candidate hash mismatch: {instance_id}"
            )
        if set(decision["sampled_attribute_checks"]) != set(
            candidate["index"]["sampled_attributes"]
        ):
            raise contracts.ContractError(
                f"review attribute checks do not cover request: {instance_id}"
            )

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
        reviews_dir = staging / "reviews"
        reviews_dir.mkdir()
        review_index = []
        for instance_id in sorted(candidates):
            candidate = candidates[instance_id]
            decision = decisions[instance_id]
            review: dict[str, Any] = {
                "schema": REVIEW_SCHEMA,
                "instance_id": instance_id,
                "request_sha256": candidate["manifest"]["request_sha256"],
                "profile_schema_id": candidate["manifest"]["profile_schema_id"],
                "sampled_attributes": candidate["manifest"]["sampled_attributes"],
                "candidate": {
                    "path": str(candidate["files"]["candidate"]),
                    "sha256": decision["candidate_sha256"],
                    "size_bytes": candidate["files"]["candidate"].stat().st_size,
                },
                "candidate_manifest": {
                    "path": str(candidate["files"]["candidate_manifest"]),
                    "sha256": _sha256_file(candidate["files"]["candidate_manifest"]),
                    "size_bytes": candidate["files"]["candidate_manifest"].stat().st_size,
                },
                "reviewer": contracts.load_json(decisions_path)["reviewer"],
                "decision": decision["decision"],
                "checks": {
                    "species_breed": decision["species_breed"],
                    "anatomy": decision["anatomy"],
                    "pose_and_limb_separation": decision[
                        "pose_and_limb_separation"
                    ],
                    "background": decision["background"],
                    "sampled_attributes": decision["sampled_attribute_checks"],
                },
                "notes": decision["notes"],
                "downstream_gate": (
                    "approved_for_segmentation_and_pixal3d"
                    if decision["decision"] == "approved_for_pixal3d"
                    else "blocked_rejected"
                ),
            }
            review["review_sha256"] = _json_sha256(review)
            review_path = reviews_dir / f"{instance_id}.json"
            contracts.write_json_no_replace(review_path, review)
            review_index.append(
                {
                    "instance_id": instance_id,
                    "profile_schema_id": review["profile_schema_id"],
                    "decision": review["decision"],
                    "candidate_sha256": review["candidate"]["sha256"],
                    "review": _relative_record(review_path, staging),
                }
            )

        qa_pair_eligibility = []
        for pair in batch["selection"]["planned_qa_pairs"]:
            left = decisions[pair["instance_a"]]
            right = decisions[pair["instance_b"]]
            attribute = next(iter(pair["different_attributes"]))
            if left["decision"] != "approved_for_pixal3d" or right["decision"] != "approved_for_pixal3d":
                status = "blocked_by_rejected_2d_candidate"
            elif (
                left["sampled_attribute_checks"][attribute]
                == "deferred_to_3d_physical_scale"
                or right["sampled_attribute_checks"][attribute]
                == "deferred_to_3d_physical_scale"
            ):
                status = "deferred_until_3d_physical_measurement"
            else:
                status = "eligible_after_pixal3d_static_qa"
            qa_pair_eligibility.append(
                {
                    "pair_id": pair["pair_id"],
                    "instance_a": pair["instance_a"],
                    "instance_b": pair["instance_b"],
                    "different_attribute": attribute,
                    "status": status,
                }
            )
        approved = sum(item["decision"] == "approved_for_pixal3d" for item in review_index)
        batch_review: dict[str, Any] = {
            "schema": BATCH_REVIEW_SCHEMA,
            "status": "passed" if approved == len(review_index) else "completed_with_rejections",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "flux2_batch": {
                "path": str(Path(flux_batch_path).resolve()),
                "sha256": _sha256_file(Path(flux_batch_path)),
                "batch_sha256": batch["batch_sha256"],
            },
            "decisions_input": {
                "path": str(Path(decisions_path).resolve()),
                "sha256": _sha256_file(Path(decisions_path)),
                "size_bytes": Path(decisions_path).stat().st_size,
            },
            "candidate_count": len(review_index),
            "approved_count": approved,
            "rejected_count": len(review_index) - approved,
            "reviews": review_index,
            "qa_pair_eligibility": qa_pair_eligibility,
            "automatic_checks": {
                "all_candidate_hashes_pinned": True,
                "all_candidates_reviewed_exactly_once": True,
                "all_sampled_attributes_reviewed": True,
                "size_deferment_only_to_3d_physical_scale": True,
                "pixal3d_not_started_before_review": True,
                "overall": "passed",
            },
        }
        batch_review["review_batch_sha256"] = _json_sha256(batch_review)
        contracts.write_json_no_replace(staging / "review_batch_manifest.json", batch_review)
        material_execution.native._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                f"refusing to replace concurrently-created output: {output_root}"
            )
        os.rename(staging, output_root)
        return output_root / "review_batch_manifest.json"
    except Exception:
        material_execution.native._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-batch", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = publish_reviews(
            args.flux_batch, args.decisions, args.output_root
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError) as error:
        print(f"CONTROLLED_ANIMAL_2D_REVIEW_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_2D_REVIEW_OK "
        f"approved={manifest['approved_count']} rejected={manifest['rejected_count']} "
        f"output={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
