from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import review_controlled_static_object_candidates as decisions
from tools import run_controlled_static_object_reviews as reviews


INSTANCE_ID = "kitchen_appliance_stovetop_kettle_test_0001"
REQUEST_SHA256 = "1" * 64
PROFILE_SHA256 = "2" * 64
REVIEW_SHA256 = "3" * 64
BATCH_SHA256 = "4" * 64


def _target_physical_profile() -> dict:
    return {
        "profile_id": "kettle_physical_candidate_v1",
        "control_attribute": None,
        "selected_value": "fixed",
        "measurement": "height_cm",
        "mode": "absolute_measurement",
        "reference_value_cm": 24,
        "reference_provenance": {
            "status": "provisional",
            "source_id": "fixture",
            "artifact": None,
            "notes": "fixture",
        },
        "target_value_cm": 24,
        "tolerance_cm": 5,
    }


def _review_fixture(tmp_path: Path):
    review_path = tmp_path / "review.json"
    review_path.write_text("{}\n", encoding="utf-8")
    pixal_output = tmp_path / "pixal_raw_1024.glb"
    pixal_output.write_bytes(b"fixture glb")
    target = _target_physical_profile()
    review = {
        "review_sha256": REVIEW_SHA256,
        "request_sha256": REQUEST_SHA256,
        "profile_sha256": PROFILE_SHA256,
        "sampled_attributes": {"body_color": "red"},
        "target_physical_profile": target,
        "physical_scale": {
            "status": "deferred_to_finalization",
            "control_attribute": None,
            "measurement": "height_cm",
            "target_physical_profile": copy.deepcopy(target),
        },
        "orientation": reviews._orientation_contract(),
    }
    review_batch = {"review_batch_sha256": BATCH_SHA256}
    loaded_reviews = {
        INSTANCE_ID: {
            "payload": review,
            "path": review_path,
            "pixal_output_path": pixal_output,
        }
    }
    return review_batch, loaded_reviews


def _decision_payload(*, decision=decisions.APPROVED) -> dict:
    checks = {field: True for field in decisions.CHECK_FIELDS}
    if decision == decisions.REJECTED:
        checks["physically_plausible_construction"] = False
    return {
        "schema": decisions.DECISIONS_SCHEMA,
        "static_object_review_batch_sha256": BATCH_SHA256,
        "decisions": [
            {
                "instance_id": INSTANCE_ID,
                "review_sha256": REVIEW_SHA256,
                "decision": decision,
                "checks": checks,
                "attribute_evidence": {
                    "body_color": (
                        "passed_raw_pbr_visual"
                        if decision == decisions.APPROVED
                        else "not_visually_assessable"
                    )
                },
                "caveats": [],
                "notes": "Exact reference and raw-PBR five-view review completed.",
            }
        ],
    }


def _write_decisions(path: Path, payload: dict) -> None:
    path.write_text(contracts.canonical_json(payload) + "\n", encoding="utf-8")


def test_control_attribute_null_is_legal_and_scale_remains_deferred(tmp_path):
    review_batch, loaded_reviews = _review_fixture(tmp_path)
    payload = _decision_payload()
    path = tmp_path / "decisions.json"
    _write_decisions(path, payload)

    loaded = decisions.load_decisions(path, review_batch, loaded_reviews)

    assert loaded["decisions"][0]["decision"] == (
        "approved_for_watertight_finalization"
    )
    assert loaded_reviews[INSTANCE_ID]["payload"]["physical_scale"][
        "control_attribute"
    ] is None
    assert loaded_reviews[INSTANCE_ID]["payload"]["physical_scale"]["status"] == (
        "deferred_to_finalization"
    )


def test_only_static_finalization_or_rejection_decisions_are_allowed(tmp_path):
    review_batch, loaded_reviews = _review_fixture(tmp_path)
    payload = _decision_payload()
    payload["decisions"][0]["decision"] = "approved_for_lod_and_binding"
    path = tmp_path / "decisions.json"
    _write_decisions(path, payload)

    with pytest.raises(contracts.ContractError, match="invalid static-object decision"):
        decisions.load_decisions(path, review_batch, loaded_reviews)

    failed_approval = _decision_payload()
    failed_approval["decisions"][0]["checks"][
        "no_disconnected_or_floating_parts"
    ] = False
    _write_decisions(path, failed_approval)
    with pytest.raises(contracts.ContractError, match="failed check"):
        decisions.load_decisions(path, review_batch, loaded_reviews)


def test_publish_approved_decision_is_finalizer_ready_and_fail_closed(
    tmp_path, monkeypatch
):
    review_batch, loaded_reviews = _review_fixture(tmp_path)
    review_batch_path = tmp_path / "review_batch.json"
    review_batch_path.write_text("{}\n", encoding="utf-8")
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path, _decision_payload())
    monkeypatch.setattr(
        decisions,
        "load_review_batch",
        lambda _path: (
            review_batch_path.resolve(),
            copy.deepcopy(review_batch),
            loaded_reviews,
        ),
    )
    monkeypatch.setattr(decisions.immutable, "_seal_readonly_tree", lambda _path: None)
    output_root = tmp_path / "published"

    manifest_path = decisions.publish_decisions(
        review_batch_path, decisions_path, output_root
    )
    manifest = contracts.load_json(manifest_path)
    index = manifest["decisions"][0]
    record_path = output_root / index["record"]["path"]
    record = contracts.load_json(record_path)

    assert record["schema"] == "avengine_controlled_static_object_decision_v1"
    assert record["decision"] == "approved_for_watertight_finalization"
    assert record["next_gate"] == "watertight_then_static_finalization"
    assert record["instance_id"] == INSTANCE_ID
    assert record["request_sha256"] == REQUEST_SHA256
    assert record["profile_sha256"] == PROFILE_SHA256
    assert record["target_physical_profile"]["measurement"] == "height_cm"
    assert record["target_physical_profile"]["target_value_cm"] == 24
    assert record["target_physical_profile"]["tolerance_cm"] == 5
    assert record["target_physical_profile"]["control_attribute"] is None
    assert Path(record["pixal_output"]["path"]).resolve() == loaded_reviews[
        INSTANCE_ID
    ]["pixal_output_path"].resolve()
    assert record["pixal_output"]["sha256"] == index["pixal_output_sha256"]
    assert record["physical_scale"]["status"] == "deferred_to_finalization"
    assert record["canonical_heading"]["status"] == (
        "deferred_to_static_finalization"
    )
    assert record["formal_dataset_registration_authorized"] is False
    assert record["decision_sha256"] == decisions._hash_without(
        record, "decision_sha256"
    )
    assert index["record"]["sha256"] == decisions._sha256_file(record_path)
    assert index["request_sha256"] == REQUEST_SHA256
    assert index["profile_sha256"] == PROFILE_SHA256
    assert manifest["formal_dataset_registration_authorized"] is False
    serialized = contracts.canonical_json(record)
    assert "pose_riggable" not in serialized
    assert "species_rig" not in serialized
    assert "limb" not in serialized

    with pytest.raises(contracts.ContractError, match="refusing to replace"):
        decisions.publish_decisions(review_batch_path, decisions_path, output_root)


def test_rejected_decision_stops_without_opening_finalization(tmp_path, monkeypatch):
    review_batch, loaded_reviews = _review_fixture(tmp_path)
    review_batch_path = tmp_path / "review_batch.json"
    review_batch_path.write_text("{}\n", encoding="utf-8")
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(
        decisions_path, _decision_payload(decision=decisions.REJECTED)
    )
    monkeypatch.setattr(
        decisions,
        "load_review_batch",
        lambda _path: (
            review_batch_path.resolve(),
            copy.deepcopy(review_batch),
            loaded_reviews,
        ),
    )
    monkeypatch.setattr(decisions.immutable, "_seal_readonly_tree", lambda _path: None)

    manifest_path = decisions.publish_decisions(
        review_batch_path, decisions_path, tmp_path / "rejected"
    )
    manifest = contracts.load_json(manifest_path)
    record = contracts.load_json(
        manifest_path.parent / manifest["decisions"][0]["record"]["path"]
    )

    assert record["decision"] == "rejected"
    assert record["state_classification"] == "rejected"
    assert record["next_gate"] == "stop"
    assert manifest["approved_count"] == 0
    assert manifest["rejected_count"] == 1
