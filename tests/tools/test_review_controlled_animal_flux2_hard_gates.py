import json

import pytest

from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import review_controlled_animal_flux2_candidates as review


def _decision(*, decision="approved_for_pixal3d", rejected_gate=None):
    hard_gates = {field: "passed" for field in review.HARD_GATE_FIELDS}
    if rejected_gate is not None:
        hard_gates[rejected_gate] = "rejected"
    return {
        "instance_id": "beagle_canary",
        "candidate_sha256": "a" * 64,
        "decision": decision,
        "species_breed": "passed",
        "anatomy": "passed",
        "pose_and_limb_separation": "passed",
        "background": "passed",
        "sampled_attribute_checks": {
            "body_build": "passed",
            "coat_tone": "passed",
            "size": "deferred_to_3d_physical_scale",
        },
        "hard_gates": hard_gates,
        "notes": "test",
    }


def _write(tmp_path, decision, *, schema=review.DECISIONS_SCHEMA_V2):
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "flux2_batch_sha256": "batch-sha",
                "reviewer": "test",
                "decisions": [decision],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_v2_approval_requires_complete_hard_gate_set(tmp_path):
    decisions = review.load_decisions(
        _write(tmp_path, _decision()), {"batch_sha256": "batch-sha"}
    )
    assert decisions["beagle_canary"]["decision"] == "approved_for_pixal3d"


def test_v2_rejects_approval_when_style_gate_fails(tmp_path):
    path = _write(
        tmp_path,
        _decision(rejected_gate="photorealistic_pbr_style"),
    )
    with pytest.raises(
        contracts.ContractError,
        match="decision disagrees",
    ):
        review.load_decisions(path, {"batch_sha256": "batch-sha"})


def test_v2_can_publish_rejection_from_tail_gate(tmp_path):
    decisions = review.load_decisions(
        _write(
            tmp_path,
            _decision(decision="rejected", rejected_gate="species_correct_tail"),
        ),
        {"batch_sha256": "batch-sha"},
    )
    assert decisions["beagle_canary"]["hard_gates"]["species_correct_tail"] == "rejected"


def test_v2_rejects_missing_hard_gate(tmp_path):
    decision = _decision()
    decision["hard_gates"].pop("target_attribute_only")
    with pytest.raises(contracts.ContractError, match="hard gates"):
        review.load_decisions(
            _write(tmp_path, decision), {"batch_sha256": "batch-sha"}
        )


def test_bounded_exploration_rejects_legacy_review_without_hard_gates(tmp_path):
    batch = {
        "batch_sha256": "batch-sha",
        "selection": {
            "route": "flux2_pixal3d_animal_v1",
            "bounded_exploration": {},
        },
    }
    with pytest.raises(contracts.ContractError, match="requires v2"):
        review.load_decisions(
            _write(tmp_path, _decision(), schema=review.DECISIONS_SCHEMA),
            batch,
        )


def test_bounded_exploration_approval_requires_passed_not_na_hard_gates(
    tmp_path,
):
    decision = _decision()
    decision["hard_gates"] = {
        field: "not_applicable" for field in review.HARD_GATE_FIELDS
    }
    batch = {
        "batch_sha256": "batch-sha",
        "selection": {
            "route": "flux2_pixal3d_animal_v1",
            "bounded_exploration": {},
        },
    }
    with pytest.raises(contracts.ContractError, match="every hard gate to pass"):
        review.load_decisions(_write(tmp_path, decision), batch)


def _static_decision(*, decision="approved_for_pixal3d", rejected_gate=None):
    hard_gates = {
        field: "passed" for field in review.STATIC_HARD_GATE_FIELDS
    }
    if rejected_gate is not None:
        hard_gates[rejected_gate] = "rejected"
    return {
        "instance_id": "doorbell_canary",
        "candidate_sha256": "b" * 64,
        "decision": decision,
        "category_identity": "passed",
        "construction": "passed",
        "stable_product_pose": "passed",
        "background": "passed",
        "sampled_attribute_checks": {"body_color": "passed"},
        "hard_gates": hard_gates,
        "notes": "test static route",
    }


def _static_batch():
    return {
        "batch_sha256": "batch-sha",
        "selection": {"route": "flux2_pixal3d_static_v1"},
    }


def test_static_route_requires_its_own_decision_schema(tmp_path):
    with pytest.raises(contracts.ContractError, match="contract is invalid"):
        review.load_decisions(
            _write(tmp_path, _decision()),
            _static_batch(),
        )


def test_static_route_accepts_complete_static_hard_gates(tmp_path):
    decisions = review.load_decisions(
        _write(
            tmp_path,
            _static_decision(),
            schema=review.STATIC_DECISIONS_SCHEMA,
        ),
        _static_batch(),
    )
    assert decisions["doorbell_canary"]["decision"] == "approved_for_pixal3d"


def test_static_route_rejects_not_applicable_emitter_gate(tmp_path):
    decision = _static_decision()
    decision["hard_gates"]["emitter_feature_visible"] = "not_applicable"
    with pytest.raises(contracts.ContractError, match="hard gates"):
        review.load_decisions(
            _write(
                tmp_path,
                decision,
                schema=review.STATIC_DECISIONS_SCHEMA,
            ),
            _static_batch(),
        )


def test_static_route_can_publish_objective_rejection(tmp_path):
    decisions = review.load_decisions(
        _write(
            tmp_path,
            _static_decision(
                decision="rejected",
                rejected_gate="emitter_feature_visible",
            ),
            schema=review.STATIC_DECISIONS_SCHEMA,
        ),
        _static_batch(),
    )
    assert (
        decisions["doorbell_canary"]["hard_gates"]["emitter_feature_visible"]
        == "rejected"
    )


def _bounded_exploration_batch():
    jobs = [
        {
            "execution_job_id": f"animal_{index}",
            "profile_schema_id": "dog_short_leg_v1",
            "profile_sha256": "1" * 64,
            "consumer_requests": [
                {
                    "instance_id": f"dog_short_leg_{index}",
                    "request_sha256": f"{index + 2:x}" * 64,
                }
            ],
        }
        for index in range(2)
    ]
    group = one_shot.build_bounded_exploration_group(
        profile_schema_id="dog_short_leg_v1",
        profile_sha256="1" * 64,
        execution_preflight_sha256="2" * 64,
        request_batch_sha256="3" * 64,
        jobs=jobs,
    )
    return {
        "batch_sha256": "4" * 64,
        "candidate_count": 2,
        "candidates": [
            {"instance_id": f"dog_short_leg_{index}"} for index in range(2)
        ],
        "selection": {
            "bounded_exploration": {
                "policy": one_shot.bounded_exploration_policy_record(),
                "groups": [group],
            }
        },
    }


def _bounded_review_index(decisions):
    return [
        {
            "instance_id": f"dog_short_leg_{index}",
            "profile_schema_id": "dog_short_leg_v1",
            "candidate_sha256": f"{index + 8:x}" * 64,
            "decision": decision,
            "review": {"sha256": f"{index + 11:x}" * 64},
        }
        for index, decision in enumerate(decisions)
    ]


def test_bounded_exploration_review_freezes_one_exact_hash():
    frozen = review._build_bounded_exploration_freezes(
        _bounded_exploration_batch(),
        _bounded_review_index(["rejected", "approved_for_pixal3d"]),
    )

    assert frozen is not None
    receipt = frozen["freeze_receipts"][0]
    assert receipt["state"] == "frozen"
    assert receipt["selected_instance_id"] == "dog_short_leg_1"
    assert receipt["selected_candidate_sha256"] == "9" * 64


def test_bounded_exploration_review_fails_closed_on_two_winners():
    with pytest.raises(
        contracts.ContractError,
        match="at most one approved candidate",
    ):
        review._build_bounded_exploration_freezes(
            _bounded_exploration_batch(),
            _bounded_review_index(
                ["approved_for_pixal3d", "approved_for_pixal3d"]
            ),
        )
