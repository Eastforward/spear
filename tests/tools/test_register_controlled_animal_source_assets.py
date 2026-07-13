from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import register_controlled_animal_source_assets as registry


def test_spear_artifact_rejects_paths_outside_repo(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"fixture")

    with pytest.raises(contracts.ContractError, match="outside SPEAR"):
        registry.spear_artifact(path)


def test_pinned_model_license_snapshots_are_present_and_hashed():
    records = registry.license_records()

    assert len(records) == 4
    assert {record["root_id"] for record in records} == {"models_root"}
    assert all(len(record["sha256"]) == 64 for record in records)


def test_approved_attempt_ids_keeps_approved_subset_after_complete_review():
    decisions = {
        "animal_approved": {
            "payload": {"decision": "approved_for_lod_and_binding"}
        },
        "animal_rejected": {"payload": {"decision": "rejected"}},
    }
    attempts = {
        "animal_approved": {"instance_id": "animal_approved"},
        "animal_rejected": {"instance_id": "animal_rejected"},
    }

    assert registry.approved_attempt_ids(decisions, attempts) == {
        "animal_approved"
    }


def test_approved_attempt_ids_requires_decisions_for_every_attempt():
    decisions = {
        "animal_approved": {
            "payload": {"decision": "approved_for_lod_and_binding"}
        }
    }
    attempts = {
        "animal_approved": {"instance_id": "animal_approved"},
        "animal_missing": {"instance_id": "animal_missing"},
    }

    with pytest.raises(contracts.ContractError, match="coverage"):
        registry.approved_attempt_ids(decisions, attempts)
