import json

import pytest

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


def _write(tmp_path, decision):
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps(
            {
                "schema": review.DECISIONS_SCHEMA_V2,
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
