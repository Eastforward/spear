#!/usr/bin/env python3
"""Machine checks for the controlled-animal no-seed-lottery policy."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    REPO_ROOT
    / "data/controlled_source_attributes_v1/contracts/animal_one_shot_no_seed_lottery_v1.json"
)
POLICY_SCHEMA = "avengine_controlled_animal_one_shot_policy_v1"
POLICY_ID = "animal_one_shot_no_seed_lottery_v1"
POLICY_RECORD_SCHEMA = "avengine_controlled_animal_one_shot_policy_record_v1"
BOUNDED_EXPLORATION_POLICY_SCHEMA = (
    "avengine_controlled_animal_bounded_exploration_policy_v1"
)
BOUNDED_EXPLORATION_GROUP_SCHEMA = (
    "avengine_controlled_animal_bounded_exploration_group_v1"
)
BOUNDED_EXPLORATION_FREEZE_SCHEMA = (
    "avengine_controlled_animal_bounded_exploration_freeze_v1"
)
BOUNDED_EXPLORATION_MAX_CANDIDATES = 10
BASE_ACQUISITION_POLICY = {
    "policy_id": POLICY_ID,
    "acquisition_unit": "one_frozen_base_asset",
    "sampled_domains_must_be_singleton": True,
    "downstream_instance_route": "stable_animal_template_v1",
    "profile_validation": "all_predeclared_requests_count_zero_hidden_failures",
}
# Statics skip rigging, so every sampled attribute combination is its own
# frozen one-shot request; the no-seed-lottery cardinality rules still apply.
STATIC_BASE_ACQUISITION_POLICY = {
    "policy_id": "static_object_per_request_one_shot_v1",
    "acquisition_unit": "one_frozen_asset_per_request",
    "sampled_domains_must_be_singleton": False,
    "downstream_instance_route": "flux2_pixal3d_static_v1",
    "profile_validation": "all_predeclared_requests_count_zero_hidden_failures",
}


class PolicyError(ValueError):
    """Raised when one-shot execution evidence violates the frozen policy."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise PolicyError(f"one-shot policy is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PolicyError("one-shot policy must be an object")
    if value.get("schema") != POLICY_SCHEMA or value.get("policy_id") != POLICY_ID:
        raise PolicyError("one-shot policy identity changed")
    request = value.get("request_freeze", {})
    cardinality = value.get("per_request_cardinality", {})
    failure = value.get("failure_policy", {})
    qualification = value.get("profile_qualification", {})
    production = value.get("production_instance_policy", {})
    required = {
        "request_freeze.seed_override_after_generation_started_allowed": request.get(
            "seed_override_after_generation_started_allowed"
        )
        is False,
        "request_freeze.request_replacement_after_observing_output_allowed": request.get(
            "request_replacement_after_observing_output_allowed"
        )
        is False,
        "per_request_cardinality.flux_invocations": cardinality.get(
            "flux_invocations"
        )
        == 1,
        "per_request_cardinality.flux_images_per_invocation": cardinality.get(
            "flux_images_per_invocation"
        )
        == 1,
        "per_request_cardinality.pixal3d_invocations": cardinality.get(
            "pixal3d_invocations"
        )
        == 1,
        "per_request_cardinality.seed_retry_allowed": cardinality.get(
            "seed_retry_allowed"
        )
        is False,
        "per_request_cardinality.candidate_ranking_or_best_of_n_allowed": cardinality.get(
            "candidate_ranking_or_best_of_n_allowed"
        )
        is False,
        "failure_policy.failed_output_may_be_hidden_from_profile_metrics": failure.get(
            "failed_output_may_be_hidden_from_profile_metrics"
        )
        is False,
        "profile_qualification.all_predeclared_requests_count": qualification.get(
            "all_predeclared_requests_count"
        )
        is True,
        "profile_qualification.required_pass_fraction": qualification.get(
            "required_pass_fraction"
        )
        == 1.0,
        "production_instance_policy.rerun_flux_or_pixal_for_each_color_or_size_instance": production.get(
            "rerun_flux_or_pixal_for_each_color_or_size_instance"
        )
        is False,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise PolicyError(f"one-shot policy weakened: {failed}")
    return copy.deepcopy(value)


def policy_record(path: Path = POLICY_PATH) -> dict[str, Any]:
    path = Path(path).resolve()
    policy = load_policy(path)
    return {
        "schema": POLICY_RECORD_SCHEMA,
        "policy_id": policy["policy_id"],
        "policy_schema": policy["schema"],
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def bounded_exploration_policy_record() -> dict[str, Any]:
    """Return the exact batch-level policy layered over one-shot requests."""

    return {
        "schema": BOUNDED_EXPLORATION_POLICY_SCHEMA,
        "request_policy": policy_record(),
        "minimum_candidates_per_profile": 2,
        "maximum_candidates_per_profile": BOUNDED_EXPLORATION_MAX_CANDIDATES,
        "candidate_set_frozen_before_inference": True,
        "all_candidates_require_exact_hash_review": True,
        "accepted_candidates_per_profile": 1,
        "selected_candidate_reused_without_regeneration": True,
        "downstream_before_freeze_allowed": False,
    }


def validate_bounded_exploration_policy_record(value: Any) -> dict[str, Any]:
    expected = bounded_exploration_policy_record()
    if not isinstance(value, dict) or _canonical_json(value) != _canonical_json(
        expected
    ):
        raise PolicyError("bounded exploration policy record changed")
    return copy.deepcopy(value)


def build_bounded_exploration_group(
    *,
    profile_schema_id: str,
    profile_sha256: str,
    execution_preflight_sha256: str,
    request_batch_sha256: str,
    jobs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        not isinstance(profile_schema_id, str)
        or not profile_schema_id
        or not _is_sha256(profile_sha256)
        or not _is_sha256(execution_preflight_sha256)
        or not _is_sha256(request_batch_sha256)
        or not 2 <= len(jobs) <= BOUNDED_EXPLORATION_MAX_CANDIDATES
    ):
        raise PolicyError("bounded exploration group identity/count is invalid")
    candidates = []
    for ordinal, job in enumerate(
        sorted(jobs, key=lambda item: item.get("execution_job_id", ""))
    ):
        consumers = job.get("consumer_requests")
        if (
            job.get("profile_schema_id") != profile_schema_id
            or job.get("profile_sha256") != profile_sha256
            or not isinstance(consumers, list)
            or len(consumers) != 1
        ):
            raise PolicyError("bounded exploration job differs from its profile")
        consumer = consumers[0]
        execution_job_id = job.get("execution_job_id")
        instance_id = consumer.get("instance_id")
        request_sha256 = consumer.get("request_sha256")
        if (
            not isinstance(execution_job_id, str)
            or not execution_job_id
            or not isinstance(instance_id, str)
            or not instance_id
            or not _is_sha256(request_sha256)
        ):
            raise PolicyError("bounded exploration candidate identity is invalid")
        candidates.append(
            {
                "ordinal": ordinal,
                "execution_job_id": execution_job_id,
                "instance_id": instance_id,
                "request_sha256": request_sha256,
            }
        )
    group = {
        "schema": BOUNDED_EXPLORATION_GROUP_SCHEMA,
        "profile_schema_id": profile_schema_id,
        "profile_sha256": profile_sha256,
        "execution_preflight_sha256": execution_preflight_sha256,
        "request_batch_sha256": request_batch_sha256,
        "declared_candidate_count": len(candidates),
        "candidates": candidates,
    }
    group["exploration_group_sha256"] = hashlib.sha256(
        _canonical_json(group).encode("utf-8")
    ).hexdigest()
    return validate_bounded_exploration_group(group)


def validate_bounded_exploration_group(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "profile_schema_id",
        "profile_sha256",
        "execution_preflight_sha256",
        "request_batch_sha256",
        "declared_candidate_count",
        "candidates",
        "exploration_group_sha256",
    }:
        raise PolicyError("bounded exploration group fields are invalid")
    candidates = value.get("candidates")
    count = value.get("declared_candidate_count")
    if (
        value.get("schema") != BOUNDED_EXPLORATION_GROUP_SCHEMA
        or not isinstance(value.get("profile_schema_id"), str)
        or not value["profile_schema_id"]
        or not _is_sha256(value.get("profile_sha256"))
        or not _is_sha256(value.get("execution_preflight_sha256"))
        or not _is_sha256(value.get("request_batch_sha256"))
        or isinstance(count, bool)
        or not isinstance(count, int)
        or not 2 <= count <= BOUNDED_EXPLORATION_MAX_CANDIDATES
        or not isinstance(candidates, list)
        or len(candidates) != count
    ):
        raise PolicyError("bounded exploration group identity/count is invalid")
    expected_ordinals = list(range(count))
    ordinals = []
    execution_job_ids = []
    instance_ids = []
    request_sha256s = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {
            "ordinal",
            "execution_job_id",
            "instance_id",
            "request_sha256",
        }:
            raise PolicyError("bounded exploration candidate fields are invalid")
        ordinal = candidate["ordinal"]
        execution_job_id = candidate["execution_job_id"]
        instance_id = candidate["instance_id"]
        request_sha256 = candidate["request_sha256"]
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not isinstance(execution_job_id, str)
            or not execution_job_id
            or not isinstance(instance_id, str)
            or not instance_id
            or not _is_sha256(request_sha256)
        ):
            raise PolicyError("bounded exploration candidate identity is invalid")
        ordinals.append(ordinal)
        execution_job_ids.append(execution_job_id)
        instance_ids.append(instance_id)
        request_sha256s.append(request_sha256)
    if (
        ordinals != expected_ordinals
        or len(set(execution_job_ids)) != count
        or len(set(instance_ids)) != count
        or len(set(request_sha256s)) != count
    ):
        raise PolicyError("bounded exploration candidate set is not unique/contiguous")
    expected_hash = hashlib.sha256(
        _canonical_json(
            {
                name: copy.deepcopy(item)
                for name, item in value.items()
                if name != "exploration_group_sha256"
            }
        ).encode("utf-8")
    ).hexdigest()
    if value.get("exploration_group_sha256") != expected_hash:
        raise PolicyError("bounded exploration group hash changed")
    return copy.deepcopy(value)


def validate_bounded_exploration_declaration(
    value: Any,
    *,
    execution_preflight_sha256: str,
    request_batch_sha256: str,
    jobs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild every declared group from authenticated preflight jobs."""

    if (
        not isinstance(value, dict)
        or set(value) != {"policy", "groups"}
        or not isinstance(value.get("groups"), list)
        or not value["groups"]
        or not _is_sha256(execution_preflight_sha256)
        or not _is_sha256(request_batch_sha256)
        or not isinstance(jobs, list)
        or not jobs
    ):
        raise PolicyError("bounded exploration declaration is invalid")
    validate_bounded_exploration_policy_record(value["policy"])

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    execution_job_ids: list[str] = []
    instance_ids: list[str] = []
    for job in jobs:
        if not isinstance(job, Mapping):
            raise PolicyError("bounded exploration preflight job is invalid")
        profile_schema_id = job.get("profile_schema_id")
        execution_job_id = job.get("execution_job_id")
        consumers = job.get("consumer_requests")
        if (
            not isinstance(profile_schema_id, str)
            or not profile_schema_id
            or not isinstance(execution_job_id, str)
            or not execution_job_id
            or not isinstance(consumers, list)
            or len(consumers) != 1
            or not isinstance(consumers[0], Mapping)
            or not isinstance(consumers[0].get("instance_id"), str)
            or not consumers[0]["instance_id"]
        ):
            raise PolicyError("bounded exploration preflight job identity is invalid")
        execution_job_ids.append(execution_job_id)
        instance_ids.append(consumers[0]["instance_id"])
        grouped.setdefault(profile_schema_id, []).append(job)
    if (
        len(set(execution_job_ids)) != len(execution_job_ids)
        or len(set(instance_ids)) != len(instance_ids)
    ):
        raise PolicyError("bounded exploration preflight jobs are not unique")

    expected_groups = []
    for profile_schema_id, profile_jobs in sorted(grouped.items()):
        profile_sha256 = profile_jobs[0].get("profile_sha256")
        if not _is_sha256(profile_sha256) or any(
            job.get("profile_sha256") != profile_sha256
            for job in profile_jobs
        ):
            raise PolicyError("bounded exploration profile revision is not unique")
        expected_groups.append(
            build_bounded_exploration_group(
                profile_schema_id=profile_schema_id,
                profile_sha256=profile_sha256,
                execution_preflight_sha256=execution_preflight_sha256,
                request_batch_sha256=request_batch_sha256,
                jobs=profile_jobs,
            )
        )
    expected = {
        "policy": bounded_exploration_policy_record(),
        "groups": expected_groups,
    }
    if _canonical_json(value) != _canonical_json(expected):
        raise PolicyError(
            "bounded exploration declaration differs from authenticated preflight jobs"
        )
    return copy.deepcopy(value)


def _build_bounded_exploration_freeze_unchecked(
    *,
    group: Mapping[str, Any],
    candidate_reviews: list[Mapping[str, Any]],
    flux_batch_sha256: str,
) -> dict[str, Any]:
    group = validate_bounded_exploration_group(group)
    if not _is_sha256(flux_batch_sha256):
        raise PolicyError("bounded exploration FLUX batch hash is invalid")
    expected_instances = {
        candidate["instance_id"] for candidate in group["candidates"]
    }
    by_instance = {}
    for item in candidate_reviews:
        if not isinstance(item, Mapping):
            raise PolicyError("bounded exploration review is invalid")
        instance_id = item.get("instance_id")
        candidate_sha256 = item.get("candidate_sha256")
        decision = item.get("decision")
        review = item.get("review")
        if (
            instance_id in by_instance
            or instance_id not in expected_instances
            or item.get("profile_schema_id") != group["profile_schema_id"]
            or decision not in {"approved_for_pixal3d", "rejected"}
            or not _is_sha256(candidate_sha256)
            or not isinstance(review, Mapping)
            or not _is_sha256(review.get("sha256"))
        ):
            raise PolicyError("bounded exploration review identity/decision is invalid")
        by_instance[instance_id] = {
            "instance_id": instance_id,
            "candidate_sha256": candidate_sha256,
            "decision": decision,
            "review_sha256": review["sha256"],
        }
    if set(by_instance) != expected_instances:
        raise PolicyError("bounded exploration requires every candidate review")
    approved = [
        item
        for item in by_instance.values()
        if item["decision"] == "approved_for_pixal3d"
    ]
    if len(approved) > 1:
        raise PolicyError(
            "bounded exploration requires at most one approved candidate per profile"
        )
    selected = approved[0] if approved else None
    receipt = {
        "schema": BOUNDED_EXPLORATION_FREEZE_SCHEMA,
        "exploration_group_sha256": group["exploration_group_sha256"],
        "profile_schema_id": group["profile_schema_id"],
        "profile_sha256": group["profile_sha256"],
        "execution_preflight_sha256": group["execution_preflight_sha256"],
        "request_batch_sha256": group["request_batch_sha256"],
        "flux_batch_sha256": flux_batch_sha256,
        "declared_candidate_count": group["declared_candidate_count"],
        "reviewed_candidate_count": len(by_instance),
        "rejected_candidate_count": len(by_instance) - len(approved),
        "state": "frozen" if selected else "exploration_exhausted",
        "selection_rule": "exactly_one_candidate_passes_all_declared_hard_gates",
        "selected_instance_id": selected["instance_id"] if selected else None,
        "selected_candidate_sha256": (
            selected["candidate_sha256"] if selected else None
        ),
        "candidate_outcomes": [
            by_instance[candidate["instance_id"]]
            for candidate in group["candidates"]
        ],
    }
    receipt["freeze_receipt_sha256"] = hashlib.sha256(
        _canonical_json(receipt).encode("utf-8")
    ).hexdigest()
    return receipt


def build_bounded_exploration_freeze(
    *,
    group: Mapping[str, Any],
    candidate_reviews: list[Mapping[str, Any]],
    flux_batch_sha256: str,
) -> dict[str, Any]:
    receipt = _build_bounded_exploration_freeze_unchecked(
        group=group,
        candidate_reviews=candidate_reviews,
        flux_batch_sha256=flux_batch_sha256,
    )
    return validate_bounded_exploration_freeze(
        receipt,
        group=group,
        candidate_reviews=candidate_reviews,
        flux_batch_sha256=flux_batch_sha256,
    )


def validate_bounded_exploration_freeze_record(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "exploration_group_sha256",
        "profile_schema_id",
        "profile_sha256",
        "execution_preflight_sha256",
        "request_batch_sha256",
        "flux_batch_sha256",
        "declared_candidate_count",
        "reviewed_candidate_count",
        "rejected_candidate_count",
        "state",
        "selection_rule",
        "selected_instance_id",
        "selected_candidate_sha256",
        "candidate_outcomes",
        "freeze_receipt_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema") != BOUNDED_EXPLORATION_FREEZE_SCHEMA
        or any(
            not _is_sha256(value.get(name))
            for name in (
                "exploration_group_sha256",
                "profile_sha256",
                "execution_preflight_sha256",
                "request_batch_sha256",
                "flux_batch_sha256",
                "freeze_receipt_sha256",
            )
        )
        or value.get("selection_rule")
        != "exactly_one_candidate_passes_all_declared_hard_gates"
        or value.get("state") not in {"frozen", "exploration_exhausted"}
        or not isinstance(value.get("candidate_outcomes"), list)
    ):
        raise PolicyError("bounded exploration freeze receipt fields are invalid")
    outcomes = value["candidate_outcomes"]
    approved = []
    for item in outcomes:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "instance_id",
                "candidate_sha256",
                "decision",
                "review_sha256",
            }
            or not isinstance(item["instance_id"], str)
            or not item["instance_id"]
            or not _is_sha256(item["candidate_sha256"])
            or not _is_sha256(item["review_sha256"])
            or item["decision"] not in {"approved_for_pixal3d", "rejected"}
        ):
            raise PolicyError("bounded exploration candidate outcome is invalid")
        if item["decision"] == "approved_for_pixal3d":
            approved.append(item)
    count = len(outcomes)
    if (
        len({item["instance_id"] for item in outcomes}) != count
        or value.get("declared_candidate_count") != count
        or value.get("reviewed_candidate_count") != count
        or value.get("rejected_candidate_count") != count - len(approved)
        or len(approved) > 1
        or (value["state"] == "frozen") != (len(approved) == 1)
        or value.get("selected_instance_id")
        != (approved[0]["instance_id"] if approved else None)
        or value.get("selected_candidate_sha256")
        != (approved[0]["candidate_sha256"] if approved else None)
    ):
        raise PolicyError("bounded exploration freeze receipt counts/state changed")
    expected_hash = hashlib.sha256(
        _canonical_json(
            {
                name: copy.deepcopy(item)
                for name, item in value.items()
                if name != "freeze_receipt_sha256"
            }
        ).encode("utf-8")
    ).hexdigest()
    if value["freeze_receipt_sha256"] != expected_hash:
        raise PolicyError("bounded exploration freeze receipt hash changed")
    return copy.deepcopy(value)


def validate_bounded_exploration_freeze(
    value: Any,
    *,
    group: Mapping[str, Any],
    candidate_reviews: list[Mapping[str, Any]],
    flux_batch_sha256: str,
) -> dict[str, Any]:
    validate_bounded_exploration_freeze_record(value)
    expected = _build_bounded_exploration_freeze_unchecked(
        group=group,
        candidate_reviews=candidate_reviews,
        flux_batch_sha256=flux_batch_sha256,
    )
    if _canonical_json(value) != _canonical_json(expected):
        raise PolicyError("bounded exploration freeze receipt changed")
    return copy.deepcopy(value)


def validate_policy_record(value: Any) -> dict[str, Any]:
    expected = policy_record()
    if not isinstance(value, dict) or _canonical_json(value) != _canonical_json(expected):
        raise PolicyError("one-shot policy record/path/hash changed")
    return copy.deepcopy(value)


def stage_record(stage: str) -> dict[str, Any]:
    if stage not in {"flux2", "pixal3d"}:
        raise PolicyError(f"unsupported one-shot stage: {stage}")
    return {
        "policy": policy_record(),
        "stage": stage,
        "invocation_ordinal": 0,
        "invocations_allowed": 1,
        "seed_retry_allowed": False,
        "candidate_ranking_allowed": False,
        "failure_action": "preserve_evidence_and_reject_instance",
    }


def base_acquisition_record() -> dict[str, Any]:
    return copy.deepcopy(BASE_ACQUISITION_POLICY)


def validate_base_acquisition_record(value: Any) -> dict[str, Any]:
    expected = base_acquisition_record()
    if not isinstance(value, dict) or _canonical_json(value) != _canonical_json(expected):
        raise PolicyError(
            "FLUX/Pixal profile must acquire one frozen base; instance variants "
            "must use stable_animal_template_v1"
        )
    return copy.deepcopy(value)


def static_base_acquisition_record() -> dict[str, Any]:
    return copy.deepcopy(STATIC_BASE_ACQUISITION_POLICY)


def validate_static_base_acquisition_record(value: Any) -> dict[str, Any]:
    expected = static_base_acquisition_record()
    if not isinstance(value, dict) or _canonical_json(value) != _canonical_json(expected):
        raise PolicyError(
            "static_object FLUX profile must generate one frozen asset per "
            "predeclared request on flux2_pixal3d_static_v1"
        )
    return copy.deepcopy(value)


def validate_stage_record(value: Any, stage: str) -> dict[str, Any]:
    expected = stage_record(stage)
    if not isinstance(value, dict) or _canonical_json(value) != _canonical_json(expected):
        raise PolicyError(f"{stage} one-shot execution record changed")
    return copy.deepcopy(value)


def validate_upstream_flux_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError("upstream FLUX one-shot evidence must be an object")
    validate_policy_record(value.get("policy"))
    digest = value.get("flux_batch_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise PolicyError("upstream FLUX batch hash is invalid")
    mode = value.get("mode")
    if mode == "native_policy_enforced_before_inference":
        if set(value) != {
            "mode",
            "policy",
            "flux_batch_sha256",
            "profile_qualification_authorized",
        } or value.get("profile_qualification_authorized") is not True:
            raise PolicyError("native one-shot evidence fields changed")
    elif mode == "native_bounded_exploration_frozen":
        receipts = value.get("freeze_receipt_sha256s")
        if (
            set(value)
            != {
                "mode",
                "policy",
                "bounded_exploration_policy",
                "flux_batch_sha256",
                "freeze_receipt_sha256s",
                "profile_qualification_authorized",
            }
            or validate_bounded_exploration_policy_record(
                value.get("bounded_exploration_policy")
            )
            != value.get("bounded_exploration_policy")
            or not isinstance(receipts, list)
            or not receipts
            or receipts != sorted(set(receipts))
            or any(not _is_sha256(item) for item in receipts)
            or value.get("profile_qualification_authorized") is not True
        ):
            raise PolicyError("native bounded exploration evidence fields changed")
    elif mode == "legacy_sealed_manifest_attestation":
        if (
            set(value)
            != {
                "mode",
                "policy",
                "flux_batch_sha256",
                "recorded_flux_invocations_per_candidate",
                "recorded_candidates_per_request",
                "cross_batch_seed_lottery_exclusion_proven",
                "profile_qualification_authorized",
            }
            or value.get("recorded_flux_invocations_per_candidate") != 1
            or value.get("recorded_candidates_per_request") != 1
            or value.get("cross_batch_seed_lottery_exclusion_proven") is not False
            or value.get("profile_qualification_authorized") is not False
        ):
            raise PolicyError("legacy one-shot evidence fields changed")
    else:
        raise PolicyError("unsupported upstream FLUX one-shot evidence mode")
    return copy.deepcopy(value)


def validate_flux_job(job: Mapping[str, Any]) -> None:
    generation = job.get("generation_plan")
    consumers = job.get("consumer_requests")
    if not isinstance(generation, Mapping) or not isinstance(consumers, list):
        raise PolicyError("FLUX job is missing generation/consumer evidence")
    seed = generation.get("generation_seed")
    route = generation.get("route")
    if (
        route not in {"flux2_pixal3d_animal_v1", "flux2_pixal3d_static_v1"}
        or generation.get("flux_invocations") != 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < (1 << 63)
        or len(consumers) != 1
    ):
        raise PolicyError("FLUX job violates one-request/one-seed/one-invocation policy")
    if route == "flux2_pixal3d_static_v1":
        validate_static_base_acquisition_record(
            generation.get("base_acquisition_policy")
        )
    else:
        validate_base_acquisition_record(generation.get("base_acquisition_policy"))


def validate_pixal_job(job: Mapping[str, Any]) -> None:
    controlled = job.get("controlled_request")
    seed = job.get("seed")
    if not isinstance(controlled, Mapping):
        raise PolicyError("Pixal job is missing controlled request evidence")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed != controlled.get("generation_seed")
        or job.get("attempt_ordinal") != 0
    ):
        raise PolicyError("Pixal job seed/attempt differs from the frozen request")
    validate_stage_record(job.get("one_shot_execution"), "pixal3d")
