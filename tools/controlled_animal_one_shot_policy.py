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
BASE_ACQUISITION_POLICY = {
    "policy_id": POLICY_ID,
    "acquisition_unit": "one_frozen_base_asset",
    "sampled_domains_must_be_singleton": True,
    "downstream_instance_route": "stable_animal_template_v1",
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
    if (
        generation.get("route") != "flux2_pixal3d_animal_v1"
        or generation.get("flux_invocations") != 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < (1 << 63)
        or len(consumers) != 1
    ):
        raise PolicyError("FLUX job violates one-request/one-seed/one-invocation policy")
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
