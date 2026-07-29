import copy
import json
from pathlib import Path

import pytest

from tools import controlled_animal_one_shot_policy as policy
from tools import controlled_animal_flux2_worker as flux_worker
from tools import controlled_source_asset_schema as source_schema
from tools import run_controlled_animal_flux2_jobs as flux_runner


REPO = Path(__file__).resolve().parents[2]


def test_checked_in_policy_forbids_seed_retry_and_best_of_n():
    value = policy.load_policy()

    assert value["per_request_cardinality"] == {
        "flux_invocations": 1,
        "flux_images_per_invocation": 1,
        "pixal3d_invocations": 1,
        "accepted_2d_candidates_maximum": 1,
        "accepted_3d_candidates_maximum": 1,
        "seed_retry_allowed": False,
        "candidate_ranking_or_best_of_n_allowed": False,
        "backend_fallback_within_the_same_request_allowed": False,
    }
    assert value["profile_qualification"]["required_pass_fraction"] == 1.0
    assert (
        value["production_instance_policy"]
        ["rerun_flux_or_pixal_for_each_color_or_size_instance"]
        is False
    )


def test_new_base_profile_is_singleton_and_compiles_policy_into_request():
    profile = json.loads(
        (
            REPO
            / "data/controlled_source_attributes_v1/profiles/animal/dog_beagle_open_tricolor_photorealistic_recolor_canary_v3.json"
        ).read_text(encoding="utf-8")
    )
    request = source_schema.sample_instance_requests(
        profile, count=1, batch_seed=20260716
    )[0]

    assert request["generation_plan"]["base_acquisition_policy"] == (
        policy.base_acquisition_record()
    )
    expanded = copy.deepcopy(profile)
    expanded["sampled_attribute_domains"]["size"] = ["small", "medium"]
    expanded["generation_contract"]["value_labels"]["size"]["small"] = (
        "small-sized"
    )
    with pytest.raises(source_schema.ContractError, match="requires singleton"):
        source_schema.validate_attribute_profile(expanded)


def test_policy_record_is_hash_authenticated_and_cannot_be_weakened():
    record = policy.policy_record()
    assert policy.validate_policy_record(record) == record

    tampered = copy.deepcopy(record)
    tampered["sha256"] = "0" * 64
    with pytest.raises(policy.PolicyError, match="record/path/hash changed"):
        policy.validate_policy_record(tampered)


def test_flux_job_rejects_a_seed_sweep_contract():
    job = {
        "generation_plan": {
            "route": "flux2_pixal3d_animal_v1",
            "generation_seed": 42,
            "flux_invocations": 1,
            "base_acquisition_policy": policy.base_acquisition_record(),
        },
        "consumer_requests": [{"instance_id": "dog_x", "request_sha256": "a" * 64}],
    }
    policy.validate_flux_job(job)

    job["generation_plan"]["flux_invocations"] = 4
    with pytest.raises(policy.PolicyError, match="one-request/one-seed"):
        policy.validate_flux_job(job)


def test_flux_partition_rejects_policy_tampering_before_worker_start():
    execution_jobs = json.loads(
        (
            REPO
            / "tmp/controlled_source_asset_input_v1/all_profiles_20260713_v3/execution_jobs.json"
        ).read_text(encoding="utf-8")
    )
    job = execution_jobs["routes"]["flux2_pixal3d_animal_v1"][0]
    job["generation_plan"]["base_acquisition_policy"] = (
        policy.base_acquisition_record()
    )
    partition = {
        "schema": flux_worker.PARTITION_SCHEMA,
        "execution_preflight_sha256": "a" * 64,
        "one_shot_execution": policy.stage_record("flux2"),
        "model": flux_runner.MODEL,
        "parameters": flux_runner.PARAMETERS,
        "jobs": [job],
    }
    partition["one_shot_execution"]["seed_retry_allowed"] = True
    partition["partition_sha256"] = flux_runner._json_sha256(partition)

    with pytest.raises(ValueError, match="one-shot execution record changed"):
        flux_worker.validate_partition(partition)


def test_pixal_job_must_use_same_frozen_seed_and_attempt_zero():
    job = {
        "seed": 42,
        "attempt_ordinal": 0,
        "controlled_request": {"generation_seed": 42},
        "one_shot_execution": policy.stage_record("pixal3d"),
    }
    policy.validate_pixal_job(job)

    changed = copy.deepcopy(job)
    changed["seed"] = 43
    with pytest.raises(policy.PolicyError, match="frozen request"):
        policy.validate_pixal_job(changed)


def test_legacy_evidence_cannot_claim_profile_qualification():
    evidence = {
        "mode": "legacy_sealed_manifest_attestation",
        "policy": policy.policy_record(),
        "flux_batch_sha256": "a" * 64,
        "recorded_flux_invocations_per_candidate": 1,
        "recorded_candidates_per_request": 1,
        "cross_batch_seed_lottery_exclusion_proven": False,
        "profile_qualification_authorized": False,
    }
    assert policy.validate_upstream_flux_evidence(evidence) == evidence

    evidence["profile_qualification_authorized"] = True
    with pytest.raises(policy.PolicyError, match="legacy one-shot evidence"):
        policy.validate_upstream_flux_evidence(evidence)


def _exploration_jobs(count=3):
    return [
        {
            "execution_job_id": f"animal_{index:02d}",
            "profile_schema_id": "dog_short_leg_v1",
            "profile_sha256": "1" * 64,
            "consumer_requests": [
                {
                    "instance_id": f"dog_short_leg_{index:02d}",
                    "request_sha256": f"{index + 2:x}" * 64,
                }
            ],
        }
        for index in range(count)
    ]


def _exploration_group(count=3):
    return policy.build_bounded_exploration_group(
        profile_schema_id="dog_short_leg_v1",
        profile_sha256="1" * 64,
        execution_preflight_sha256="a" * 64,
        request_batch_sha256="b" * 64,
        jobs=_exploration_jobs(count),
    )


def _exploration_reviews(decisions):
    return [
        {
            "instance_id": f"dog_short_leg_{index:02d}",
            "profile_schema_id": "dog_short_leg_v1",
            "candidate_sha256": f"{index + 8:x}" * 64,
            "decision": decision,
            "review": {"sha256": f"{index + 11:x}" * 64},
        }
        for index, decision in enumerate(decisions)
    ]


def test_bounded_exploration_group_is_predeclared_and_hash_authenticated():
    group = _exploration_group()
    assert group["declared_candidate_count"] == 3
    assert [item["ordinal"] for item in group["candidates"]] == [0, 1, 2]

    expanded = copy.deepcopy(group)
    expanded["declared_candidate_count"] = 4
    with pytest.raises(policy.PolicyError, match="identity/count"):
        policy.validate_bounded_exploration_group(expanded)

    changed = copy.deepcopy(group)
    changed["candidates"][0]["instance_id"] = "replacement_after_preview"
    with pytest.raises(policy.PolicyError, match="group hash changed"):
        policy.validate_bounded_exploration_group(changed)


def test_bounded_exploration_declaration_rebuilds_exact_preflight_groups():
    jobs = _exploration_jobs()
    declaration = {
        "policy": policy.bounded_exploration_policy_record(),
        "groups": [_exploration_group()],
    }
    assert policy.validate_bounded_exploration_declaration(
        declaration,
        execution_preflight_sha256="a" * 64,
        request_batch_sha256="b" * 64,
        jobs=jobs,
    ) == declaration

    changed_jobs = copy.deepcopy(jobs)
    changed_jobs[0]["execution_job_id"] = "replacement"
    changed = {
        "policy": policy.bounded_exploration_policy_record(),
        "groups": [
            policy.build_bounded_exploration_group(
                profile_schema_id="dog_short_leg_v1",
                profile_sha256="1" * 64,
                execution_preflight_sha256="a" * 64,
                request_batch_sha256="b" * 64,
                jobs=changed_jobs,
            )
        ],
    }
    with pytest.raises(
        policy.PolicyError, match="differs from authenticated preflight jobs"
    ):
        policy.validate_bounded_exploration_declaration(
            changed,
            execution_preflight_sha256="a" * 64,
            request_batch_sha256="b" * 64,
            jobs=jobs,
        )


def test_bounded_exploration_freezes_exactly_one_reviewed_hash():
    group = _exploration_group()
    reviews = _exploration_reviews(
        ["rejected", "approved_for_pixal3d", "rejected"]
    )
    receipt = policy.build_bounded_exploration_freeze(
        group=group,
        candidate_reviews=reviews,
        flux_batch_sha256="f" * 64,
    )

    assert receipt["state"] == "frozen"
    assert receipt["selected_instance_id"] == "dog_short_leg_01"
    assert receipt["selected_candidate_sha256"] == "9" * 64
    assert receipt["rejected_candidate_count"] == 2
    assert [item["decision"] for item in receipt["candidate_outcomes"]] == [
        "rejected",
        "approved_for_pixal3d",
        "rejected",
    ]

    changed = copy.deepcopy(receipt)
    changed["selected_instance_id"] = "dog_short_leg_02"
    with pytest.raises(policy.PolicyError, match="counts/state changed"):
        policy.validate_bounded_exploration_freeze(
            changed,
            group=group,
            candidate_reviews=reviews,
            flux_batch_sha256="f" * 64,
        )

    changed = copy.deepcopy(receipt)
    changed["candidate_outcomes"][0]["candidate_sha256"] = "e" * 64
    with pytest.raises(policy.PolicyError, match="freeze receipt hash changed"):
        policy.validate_bounded_exploration_freeze_record(changed)


def test_bounded_exploration_exhausts_zero_and_rejects_multiple_approvals():
    group = _exploration_group()
    exhausted = policy.build_bounded_exploration_freeze(
        group=group,
        candidate_reviews=_exploration_reviews(["rejected"] * 3),
        flux_batch_sha256="f" * 64,
    )
    assert exhausted["state"] == "exploration_exhausted"
    assert exhausted["selected_instance_id"] is None

    with pytest.raises(policy.PolicyError, match="at most one approved"):
        policy.build_bounded_exploration_freeze(
            group=group,
            candidate_reviews=_exploration_reviews(
                ["approved_for_pixal3d", "approved_for_pixal3d", "rejected"]
            ),
            flux_batch_sha256="f" * 64,
        )
