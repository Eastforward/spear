import copy

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import review_controlled_animal_animation_candidates as decisions


def _payload(decision="approved_for_ue_apartment"):
    return {
        "schema": decisions.DECISIONS_SCHEMA,
        "animation_review_batch_sha256": "b" * 64,
        "reviewer": "test_reviewer",
        "decisions": [
            {
                "asset_id": "dog",
                "review_sha256": "a" * 64,
                "decision": decision,
                "checks": {field: True for field in decisions.CHECK_FIELDS},
                "caveats": ["minor_sliding_within_dataset_tolerance"],
                "notes": "fixture decision",
            }
        ],
    }


def _load(tmp_path, payload):
    path = tmp_path / "decisions.json"
    path.write_text(contracts.canonical_json(payload), encoding="utf-8")
    batch = {"batch_sha256": "b" * 64}
    reviews = {"dog": {"payload": {"review_sha256": "a" * 64}}}
    return decisions.load_decisions(path, batch, reviews)


def test_approved_animation_requires_every_visual_check(tmp_path):
    payload = _payload()
    payload["decisions"][0]["checks"]["walking_ground_contact"] = False

    with pytest.raises(contracts.ContractError, match="approved animation"):
        _load(tmp_path, payload)


def test_rejected_animation_requires_a_failed_check(tmp_path):
    payload = _payload("rejected")

    with pytest.raises(contracts.ContractError, match="needs a failed check"):
        _load(tmp_path, payload)

    fixed = copy.deepcopy(payload)
    fixed["decisions"][0]["checks"]["detached_geometry_absent"] = False
    loaded = _load(tmp_path, fixed)
    assert loaded["decisions"][0]["decision"] == "rejected"
