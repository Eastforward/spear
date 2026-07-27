from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import freeze_target_native_generated_animal_animation_decision as freezer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def authenticated_review(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry_manifest.json"
    registry_path.write_text('{"fixture":"registry"}\n', encoding="utf-8")
    source_path = tmp_path / "source_asset.json"
    source_path.write_text('{"fixture":"source"}\n', encoding="utf-8")
    review_path = tmp_path / "review_run.json"
    review_path.write_text('{"fixture":"review-v4"}\n', encoding="utf-8")
    animated_glb = tmp_path / "animated.glb"
    animated_glb.write_bytes(b"fixture animated GLB")
    review_payload = {
        "schema": freezer.bridge.BRANCHED_GENERATED_REVIEW_SCHEMA,
        "status": "research_candidate_pending_human_review",
        "formal_dataset_registration_authorized": False,
        "automatic_admission_gates": {
            "all_automatic_gates_passed": True,
        },
    }
    source_asset = {
        "asset_id": "dog_fixture_generated_animal_0123456789ab",
        "asset_class": "animal",
    }

    def load_registry(path, selected_source, *, expected_file_sha256):
        assert Path(path) == registry_path
        assert Path(selected_source) == source_path
        assert expected_file_sha256 == _sha256(registry_path)
        return (
            registry_path,
            {"fixture": "registry"},
            {"fixture": "request"},
            {"fixture": "profile"},
            "current_exact_rebuild",
        )

    def load_source(path, roots, *, request, profile):
        assert Path(path) == source_path
        assert roots
        assert request == {"fixture": "request"}
        assert profile == {"fixture": "profile"}
        return source_path, copy.deepcopy(source_asset), {"artifact:raw": animated_glb}

    def load_review(path, *, source_asset, source_artifacts):
        assert Path(path) == review_path
        assert source_asset["asset_id"] == "dog_fixture_generated_animal_0123456789ab"
        assert source_artifacts == {"artifact:raw": animated_glb}
        return (
            review_path,
            copy.deepcopy(review_payload),
            animated_glb,
            {"output:animated_glb": animated_glb},
        )

    monkeypatch.setattr(freezer.bridge, "load_source_registry_anchor", load_registry)
    monkeypatch.setattr(freezer.bridge, "load_source_asset", load_source)
    monkeypatch.setattr(freezer.bridge, "load_animation_review", load_review)
    return {
        "registry_path": registry_path,
        "registry_sha256": _sha256(registry_path),
        "source_path": source_path,
        "review_path": review_path,
        "review_sha256": _sha256(review_path),
        "review_payload": review_payload,
        "source_asset": source_asset,
    }


def _checks(*, failed: str | None = None) -> dict[str, bool]:
    return {name: name != failed for name in freezer.bridge.DECISION_CHECK_FIELDS}


def _freeze(authority, output_root, **overrides):
    values = {
        "source_registry_manifest_path": authority["registry_path"],
        "expected_source_registry_sha256": authority["registry_sha256"],
        "source_asset_path": authority["source_path"],
        "animation_review_path": authority["review_path"],
        "expected_animation_review_sha256": authority["review_sha256"],
        "decision": freezer.APPROVED,
        "checks": _checks(),
        "caveats": [],
        "notes": "The user explicitly approved all six authenticated views.",
        "user_explicit_decision": freezer.APPROVED,
        "user_explicit_review_sha256": authority["review_sha256"],
        "output_root": output_root,
    }
    values.update(overrides)
    return freezer.freeze_decision(**values)


def test_freezes_exact_approved_record_and_external_hash_receipt(
    authenticated_review, tmp_path
):
    decision_path = _freeze(authenticated_review, tmp_path / "frozen_approved_decision")

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    receipt_path = decision_path.parent / "decision_freeze_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(decision) == {
        "schema",
        "asset_id",
        "review_sha256",
        "decision",
        "checks",
        "caveats",
        "notes",
        "review",
        "state_classification",
        "formal_dataset_registration_authorized",
        "next_gate",
        "decision_sha256",
    }
    assert decision["schema"] == freezer.bridge.DECISION_SCHEMA
    assert decision["asset_id"] == authenticated_review["source_asset"]["asset_id"]
    assert decision["review_sha256"] == authenticated_review["review_sha256"]
    assert decision["decision"] == freezer.APPROVED
    assert all(decision["checks"].values())
    assert decision["state_classification"] == "research_candidate"
    assert decision["formal_dataset_registration_authorized"] is False
    assert decision["next_gate"] == (
        "ue_import_metric_trajectory_audio_and_apartment_media"
    )
    assert decision["decision_sha256"] == freezer.bridge._hash_without(
        decision, "decision_sha256"
    )
    assert receipt["schema"] == freezer.RECEIPT_SCHEMA
    assert receipt["source_asset_registry"] == freezer.bridge._absolute_record(
        authenticated_review["registry_path"]
    )
    assert (
        receipt["expected_source_asset_registry_file_sha256"]
        == authenticated_review["registry_sha256"]
    )
    assert receipt["source_asset_registry_validation_mode"] == "current_exact_rebuild"
    assert receipt["user_instruction_authority"] == {
        "mode": "caller_assertion_v1",
        "cryptographic_user_identity_verified": False,
        "policy": (
            "the caller is responsible for invoking this tool only after "
            "an explicit user instruction"
        ),
    }
    assert receipt["animation_decision"]["path"] == "animation_decision.json"
    assert receipt["animation_decision"]["sha256"] == _sha256(decision_path)
    assert (
        receipt["expected_animation_review_file_sha256"]
        == (authenticated_review["review_sha256"])
    )
    assert receipt["authenticated_review_artifact_count"] == 1
    assert receipt["receipt_sha256"] == freezer.bridge._hash_without(
        receipt, "receipt_sha256"
    )
    assert decision_path.stat().st_mode & 0o222 == 0
    assert receipt_path.stat().st_mode & 0o222 == 0


def test_freezes_explicit_rejection_without_authorizing_next_gate(
    authenticated_review, tmp_path
):
    decision_path = _freeze(
        authenticated_review,
        tmp_path / "frozen_rejection",
        decision=freezer.REJECTED,
        checks=_checks(failed="walking_limb_deformation"),
        user_explicit_decision=freezer.REJECTED,
        notes="The user explicitly rejected the observed limb deformation.",
    )

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == freezer.REJECTED
    assert decision["state_classification"] == "rejected"
    assert decision["next_gate"] == "stop"
    assert decision["formal_dataset_registration_authorized"] is False


def test_rejects_review_changed_after_external_hash_was_recorded(
    authenticated_review, tmp_path
):
    authenticated_review["review_path"].write_text(
        '{"fixture":"tampered"}\n', encoding="utf-8"
    )

    with pytest.raises(contracts.ContractError, match="external expected SHA-256"):
        _freeze(authenticated_review, tmp_path / "rejected_tampered_review")


def test_rejects_legacy_v3_review_instead_of_reusing_old_approval(
    authenticated_review, tmp_path
):
    authenticated_review["review_payload"]["schema"] = (
        freezer.bridge.LEGACY_GENERATED_REVIEW_SCHEMA
    )

    with pytest.raises(contracts.ContractError, match="target-native v4"):
        _freeze(authenticated_review, tmp_path / "rejected_v3")


def test_rejects_approval_with_any_failed_check(authenticated_review, tmp_path):
    with pytest.raises(contracts.ContractError, match="all six checks"):
        _freeze(
            authenticated_review,
            tmp_path / "rejected_failed_approval_check",
            checks=_checks(failed="walking_ground_contact"),
        )


def test_allows_explicit_subjective_rejection_with_all_checks_passing(
    authenticated_review, tmp_path
):
    decision_path = _freeze(
        authenticated_review,
        tmp_path / "subjective_rejection",
        decision=freezer.REJECTED,
        user_explicit_decision=freezer.REJECTED,
        notes="The user rejected the asset on final subjective product fit.",
    )

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == freezer.REJECTED
    assert all(decision["checks"].values())
    assert decision["next_gate"] == "stop"


def test_rejects_verdict_not_matching_explicit_user_instruction(
    authenticated_review, tmp_path
):
    with pytest.raises(contracts.ContractError, match="user's explicit decision"):
        _freeze(
            authenticated_review,
            tmp_path / "rejected_instruction_mismatch",
            user_explicit_decision=freezer.REJECTED,
        )


def test_rejects_user_instruction_bound_to_another_review(
    authenticated_review, tmp_path
):
    with pytest.raises(contracts.ContractError, match="not bound"):
        _freeze(
            authenticated_review,
            tmp_path / "rejected_instruction_review_mismatch",
            user_explicit_review_sha256="f" * 64,
        )


def test_rejects_output_below_arbitrary_symlink_parent(
    authenticated_review,
    tmp_path,
):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(contracts.ContractError, match="symlink path component"):
        _freeze(
            authenticated_review,
            alias / "frozen_decision",
        )


def test_rejects_existing_output_without_replacing_it(authenticated_review, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(contracts.ContractError, match="refusing to replace"):
        _freeze(authenticated_review, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
