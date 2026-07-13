import copy

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import review_controlled_animal_pixal_static_candidates as decisions


def test_size_evidence_cannot_be_approved_from_static_views(tmp_path):
    review = {
        "review_sha256": "a" * 64,
        "sampled_attributes": {"size": "large", "coat_color": "fawn"},
        "target_physical_profile": {"control_attribute": "size"},
    }
    reviews = {"dog": {"payload": review}}
    batch = {"review_batch_sha256": "b" * 64}
    payload = {
        "schema": decisions.DECISIONS_SCHEMA,
        "static_review_batch_sha256": "b" * 64,
        "decisions": [
            {
                "instance_id": "dog",
                "review_sha256": "a" * 64,
                "decision": "approved_for_lod_and_binding",
                "checks": {field: True for field in decisions.CHECK_FIELDS},
                "attribute_evidence": {
                    "size": "passed_static_visual",
                    "coat_color": "passed_static_visual",
                },
                "caveats": [],
                "notes": "fixture",
            }
        ],
    }
    path = tmp_path / "decisions.json"
    path.write_text(contracts.canonical_json(payload), encoding="utf-8")

    with pytest.raises(contracts.ContractError, match="physical control"):
        decisions.load_decisions(path, batch, reviews)

    fixed = copy.deepcopy(payload)
    fixed["decisions"][0]["attribute_evidence"]["size"] = "deferred_to_metric_3d"
    path.write_text(contracts.canonical_json(fixed), encoding="utf-8")
    loaded = decisions.load_decisions(path, batch, reviews)
    assert loaded["decisions"][0]["decision"] == "approved_for_lod_and_binding"
