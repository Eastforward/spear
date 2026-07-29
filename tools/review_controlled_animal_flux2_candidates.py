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
from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as material_execution
from tools import run_controlled_animal_flux2_jobs as flux_runner


DECISIONS_SCHEMA = "avengine_controlled_animal_2d_review_decisions_v1"
DECISIONS_SCHEMA_V2 = "avengine_controlled_animal_2d_review_decisions_v2"
STATIC_DECISIONS_SCHEMA = "avengine_controlled_static_object_2d_review_decisions_v1"
REVIEW_SCHEMA = "avengine_controlled_animal_2d_review_v1"
BATCH_REVIEW_SCHEMA = "avengine_controlled_animal_2d_review_batch_v1"
STATIC_REVIEW_SCHEMA = "avengine_controlled_static_object_2d_review_v1"
STATIC_BATCH_REVIEW_SCHEMA = "avengine_controlled_static_object_2d_review_batch_v1"
ATTRIBUTE_STATUSES = {
    "passed",
    "deferred_to_3d_physical_scale",
    "rejected",
}
HARD_GATE_FIELDS = {
    "single_subject",
    "photorealistic_pbr_style",
    "species_correct_tail",
    "anatomically_connected_limbs",
    "complete_extremities",
    "closed_body_surface",
    "pose_and_camera_preserved",
    "target_attribute_only",
}
HARD_GATE_STATUSES = {"passed", "rejected", "not_applicable"}
STATIC_CHECK_FIELDS = {
    "category_identity",
    "construction",
    "stable_product_pose",
    "background",
}
STATIC_HARD_GATE_FIELDS = {
    "single_subject",
    "photorealistic_pbr_style",
    "category_distinctive_features",
    "emitter_feature_visible",
    "physically_connected_construction",
    "complete_object",
    "stable_rest_or_mount",
    "target_attribute_only",
}
STATIC_HARD_GATE_STATUSES = {"passed", "rejected"}


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
        # Static text-to-image candidates deliberately have no source pose
        # guide; the candidate manifest cross-check below enforces that the
        # source record is present if and only if the worker consumed one.
        roles = ("candidate", "candidate_manifest", "source")
        if "source" not in candidate:
            roles = ("candidate", "candidate_manifest")
        for role in roles:
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
        if (manifest.get("input") is None) != ("source" not in candidate):
            raise contracts.ContractError(
                "candidate source record does not match the manifest input"
            )
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
    route = batch.get("selection", {}).get(
        "route", "flux2_pixal3d_animal_v1"
    )
    if route == "flux2_pixal3d_static_v1":
        allowed_schemas = {STATIC_DECISIONS_SCHEMA}
        check_fields = STATIC_CHECK_FIELDS
        hard_gate_fields = STATIC_HARD_GATE_FIELDS
        hard_gate_statuses = STATIC_HARD_GATE_STATUSES
    elif route == "flux2_pixal3d_animal_v1":
        allowed_schemas = {DECISIONS_SCHEMA, DECISIONS_SCHEMA_V2}
        check_fields = {
            "species_breed",
            "anatomy",
            "pose_and_limb_separation",
            "background",
        }
        hard_gate_fields = HARD_GATE_FIELDS
        hard_gate_statuses = HARD_GATE_STATUSES
    else:
        raise contracts.ContractError("review batch route is unsupported")
    bounded_animal = (
        route == "flux2_pixal3d_animal_v1"
        and batch.get("selection", {}).get("bounded_exploration") is not None
    )
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "flux2_batch_sha256", "reviewer", "decisions"}
        or payload.get("schema") not in allowed_schemas
        or payload.get("flux2_batch_sha256") != batch["batch_sha256"]
        or not isinstance(payload.get("reviewer"), str)
        or not payload["reviewer"].strip()
        or not isinstance(payload.get("decisions"), list)
    ):
        raise contracts.ContractError("review decisions contract is invalid")
    if (
        bounded_animal
        and payload["schema"] != DECISIONS_SCHEMA_V2
    ):
        raise contracts.ContractError(
            "bounded exploration requires v2 animal hard-gate decisions"
        )
    decisions = {}
    decisions_schema = payload["schema"]
    for decision in payload["decisions"]:
        fields = {
            "instance_id",
            "candidate_sha256",
            "decision",
            "sampled_attribute_checks",
            "notes",
        } | check_fields
        if decisions_schema in {DECISIONS_SCHEMA_V2, STATIC_DECISIONS_SCHEMA}:
            fields.add("hard_gates")
        if not isinstance(decision, dict) or set(decision) != fields:
            raise contracts.ContractError("review decision fields are invalid")
        instance_id = decision["instance_id"]
        if instance_id in decisions:
            raise contracts.ContractError(f"duplicate review decision: {instance_id}")
        if decision["decision"] not in {"approved_for_pixal3d", "rejected"}:
            raise contracts.ContractError("review decision value is invalid")
        for check in check_fields:
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
        hard_gates = decision.get("hard_gates", {})
        if decisions_schema in {DECISIONS_SCHEMA_V2, STATIC_DECISIONS_SCHEMA} and (
            not isinstance(hard_gates, dict)
            or set(hard_gates) != hard_gate_fields
            or any(value not in hard_gate_statuses for value in hard_gates.values())
        ):
            raise contracts.ContractError("review hard gates are invalid")
        rejected = any(
            decision[key] == "rejected"
            for key in check_fields
        ) or "rejected" in attribute_checks.values() or "rejected" in hard_gates.values()
        if (decision["decision"] == "rejected") != rejected:
            raise contracts.ContractError("review decision disagrees with its checks")
        if (
            bounded_animal
            and decision["decision"] == "approved_for_pixal3d"
            and any(value != "passed" for value in hard_gates.values())
        ):
            raise contracts.ContractError(
                "bounded exploration approval requires every hard gate to pass"
            )
        if not isinstance(decision["notes"], str):
            raise contracts.ContractError("review notes must be text")
        decisions[instance_id] = decision
    return decisions


def _validate_bounded_exploration_preflight_binding(
    batch: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> None:
    declaration = batch.get("selection", {}).get("bounded_exploration")
    if declaration is None:
        return
    record = batch.get("execution_preflight")
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "preflight_sha256",
    }:
        raise contracts.ContractError(
            "bounded exploration execution preflight record is invalid"
        )
    preflight_path = Path(record["path"]).resolve()
    if (
        preflight_path.is_symlink()
        or not preflight_path.is_file()
        or _sha256_file(preflight_path) != record["sha256"]
    ):
        raise contracts.ContractError(
            "bounded exploration execution preflight changed"
        )
    preflight = material_execution._load_preflight(preflight_path)
    source_bundle = preflight.get("source_bundle")
    routes = preflight.get("routes")
    jobs = (
        routes.get("flux2_pixal3d_animal_v1")
        if isinstance(routes, Mapping)
        else None
    )
    if (
        preflight.get("preflight_sha256") != record["preflight_sha256"]
        or not isinstance(source_bundle, Mapping)
        or not isinstance(jobs, list)
    ):
        raise contracts.ContractError(
            "bounded exploration execution preflight identity changed"
        )

    declared_groups = declaration.get("groups")
    try:
        validated_groups = [
            one_shot.validate_bounded_exploration_group(group)
            for group in declared_groups
        ]
    except (one_shot.PolicyError, TypeError) as error:
        raise contracts.ContractError(
            "bounded exploration declaration groups are invalid"
        ) from error
    if not isinstance(declared_groups, list) or not validated_groups:
        raise contracts.ContractError(
            "bounded exploration declaration groups are invalid"
        )
    declared_profiles = {
        group.get("profile_schema_id")
        for group in validated_groups
    }
    expected_instances = set(candidates)
    profile_jobs = [
        job
        for job in jobs
        if isinstance(job, Mapping)
        and job.get("profile_schema_id") in declared_profiles
    ]
    selected_jobs = []
    seen_instances = set()
    for job in profile_jobs:
        consumers = job.get("consumer_requests") if isinstance(job, Mapping) else None
        if (
            not isinstance(consumers, list)
            or len(consumers) != 1
            or not isinstance(consumers[0], Mapping)
        ):
            continue
        instance_id = consumers[0].get("instance_id")
        if instance_id not in expected_instances:
            raise contracts.ContractError(
                "bounded exploration omitted a preflight job from a declared profile"
            )
        if instance_id in seen_instances:
            raise contracts.ContractError(
                "bounded exploration preflight repeats an instance"
            )
        seen_instances.add(instance_id)
        candidate = candidates[instance_id]
        index = candidate["index"]
        manifest = candidate["manifest"]
        expected = {
            "execution_job_id": job.get("execution_job_id"),
            "profile_schema_id": job.get("profile_schema_id"),
        }
        manifest_expected = {
            **expected,
            "profile_sha256": job.get("profile_sha256"),
            "request_sha256": consumers[0].get("request_sha256"),
        }
        if any(index.get(key) != value for key, value in expected.items()) or any(
            manifest.get(key) != value for key, value in manifest_expected.items()
        ):
            raise contracts.ContractError(
                "bounded exploration candidate differs from authenticated "
                "preflight job"
            )
        selected_jobs.append(job)
    if seen_instances != expected_instances:
        raise contracts.ContractError(
            "bounded exploration candidates do not resolve to preflight jobs"
        )
    try:
        one_shot.validate_bounded_exploration_declaration(
            declaration,
            execution_preflight_sha256=preflight["preflight_sha256"],
            request_batch_sha256=source_bundle.get("request_batch_sha256"),
            jobs=selected_jobs,
        )
    except one_shot.PolicyError as error:
        raise contracts.ContractError(str(error)) from error


def _build_bounded_exploration_freezes(
    batch: Mapping[str, Any], review_index: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    exploration = batch.get("selection", {}).get("bounded_exploration")
    if exploration is None:
        return None
    if (
        not isinstance(exploration, Mapping)
        or set(exploration) != {"policy", "groups"}
        or not isinstance(exploration.get("groups"), list)
        or not exploration["groups"]
    ):
        raise contracts.ContractError("bounded exploration declaration is invalid")
    try:
        policy = one_shot.validate_bounded_exploration_policy_record(
            exploration["policy"]
        )
        groups = [
            one_shot.validate_bounded_exploration_group(group)
            for group in exploration["groups"]
        ]
        if (
            len({group["exploration_group_sha256"] for group in groups})
            != len(groups)
            or len({group["profile_schema_id"] for group in groups}) != len(groups)
        ):
            raise contracts.ContractError(
                "bounded exploration declaration repeats a group/profile"
            )
        declared_instances = {
            candidate["instance_id"]
            for group in groups
            for candidate in group["candidates"]
        }
        batch_instances = {
            candidate["instance_id"] for candidate in batch["candidates"]
        }
        if (
            declared_instances != batch_instances
            or sum(group["declared_candidate_count"] for group in groups)
            != batch["candidate_count"]
        ):
            raise contracts.ContractError(
                "bounded exploration declaration differs from the FLUX batch"
            )
        freezes = []
        for group in groups:
            group_reviews = [
                item
                for item in review_index
                if item["profile_schema_id"] == group["profile_schema_id"]
            ]
            freezes.append(
                one_shot.build_bounded_exploration_freeze(
                    group=group,
                    candidate_reviews=group_reviews,
                    flux_batch_sha256=batch["batch_sha256"],
                )
            )
    except one_shot.PolicyError as error:
        raise contracts.ContractError(str(error)) from error
    return {
        "policy": policy,
        "freeze_receipts": freezes,
    }


def publish_reviews(
    flux_batch_path: Path, decisions_path: Path, output_root: Path
) -> Path:
    flux_root, batch, candidates = load_flux_batch(flux_batch_path)
    _validate_bounded_exploration_preflight_binding(batch, candidates)
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
        batch_route = batch.get("selection", {}).get(
            "route", "flux2_pixal3d_animal_v1"
        )
        for instance_id in sorted(candidates):
            candidate = candidates[instance_id]
            decision = decisions[instance_id]
            check_fields = (
                STATIC_CHECK_FIELDS
                if batch_route == "flux2_pixal3d_static_v1"
                else {
                    "species_breed",
                    "anatomy",
                    "pose_and_limb_separation",
                    "background",
                }
            )
            review_checks = {
                field: decision[field] for field in sorted(check_fields)
            }
            review_checks["sampled_attributes"] = decision[
                "sampled_attribute_checks"
            ]
            review: dict[str, Any] = {
                "schema": (
                    STATIC_REVIEW_SCHEMA
                    if batch_route == "flux2_pixal3d_static_v1"
                    else REVIEW_SCHEMA
                ),
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
                "checks": review_checks,
                "notes": decision["notes"],
                "downstream_gate": (
                    (
                        "frozen_exact_candidate_for_segmentation_and_pixal3d"
                        if batch["selection"].get("bounded_exploration") is not None
                        else "approved_for_segmentation_and_pixal3d"
                    )
                    if decision["decision"] == "approved_for_pixal3d"
                    else "blocked_rejected"
                ),
            }
            if "hard_gates" in decision:
                review["checks"]["hard_gates"] = copy.deepcopy(
                    decision["hard_gates"]
                )
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
        bounded_exploration = _build_bounded_exploration_freezes(
            batch, review_index
        )
        exhausted = (
            bounded_exploration is not None
            and any(
                item["state"] == "exploration_exhausted"
                for item in bounded_exploration["freeze_receipts"]
            )
        )
        batch_review: dict[str, Any] = {
            "schema": (
                STATIC_BATCH_REVIEW_SCHEMA
                if batch_route == "flux2_pixal3d_static_v1"
                else BATCH_REVIEW_SCHEMA
            ),
            "status": (
                "completed_with_exhausted_groups"
                if exhausted
                else (
                    "passed"
                    if approved == len(review_index)
                    else "completed_with_rejections"
                )
            ),
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "review_domain": (
                "static_object"
                if batch_route == "flux2_pixal3d_static_v1"
                else "animal"
            ),
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
                "v2_hard_gates_enforced": all(
                    "hard_gates" not in decision
                    or decision["decision"] == "rejected"
                    or "rejected" not in decision["hard_gates"].values()
                    for decision in decisions.values()
                ),
                "overall": "passed",
            },
        }
        if bounded_exploration is not None:
            batch_review["bounded_exploration"] = bounded_exploration
            batch_review["automatic_checks"][
                "bounded_exploration_exactly_one_or_exhausted_per_profile"
            ] = True
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
