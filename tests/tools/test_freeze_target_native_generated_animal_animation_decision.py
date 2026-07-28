from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import freeze_target_native_generated_animal_animation_decision as freezer


PRODUCTION_SHIBA_PRESENTATION_RECEIPT = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp/"
    "new_animal_assets/shiba_inu_20260726_01/"
    "owner_review_presentation_v1_20260728_codex1/presentation_receipt.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unseal_presentation(authority) -> None:
    authority["presentation_root"].chmod(0o755)
    authority["presentation_receipt"].chmod(0o644)
    authority["presentation_video"].chmod(0o644)


def _seal_presentation(authority) -> None:
    authority["presentation_receipt"].chmod(0o444)
    authority["presentation_video"].chmod(0o444)
    authority["presentation_root"].chmod(0o555)


def _rewrite_presentation_receipt(authority, payload) -> str:
    _unseal_presentation(authority)
    authority["presentation_receipt"].write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _seal_presentation(authority)
    return _sha256(authority["presentation_receipt"])


@pytest.fixture
def authenticated_review(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry_manifest.json"
    registry_path.write_text('{"fixture":"registry"}\n', encoding="utf-8")
    source_path = tmp_path / "source_asset.json"
    source_path.write_text('{"fixture":"source"}\n', encoding="utf-8")
    review_path = tmp_path / "review_run.json"
    review_path.write_text('{"fixture":"review-v4"}\n', encoding="utf-8")
    source_artifact = tmp_path / "source_raw.glb"
    source_artifact.write_bytes(b"fixture source artifact")
    animated_glb = tmp_path / "animated.glb"
    animated_glb.write_bytes(b"fixture animated GLB")
    review_audit = tmp_path / "review_audit.json"
    review_audit.write_text('{"fixture":"review-audit"}\n', encoding="utf-8")
    idle_side = tmp_path / "idle_side.mp4"
    idle_side.write_bytes(b"current Pixel3D Idle side readback")
    walking_side = tmp_path / "walking_side.mp4"
    walking_side.write_bytes(b"current Pixel3D Walking side readback")
    presentation_root = tmp_path / "presentation"
    presentation_root.mkdir()
    presentation_video = presentation_root / freezer.presentation.OUTPUT_VIDEO_NAME
    presentation_video.write_bytes(b"fixture authenticated six-view video")
    review_payload = {
        "schema": freezer.bridge.BRANCHED_GENERATED_REVIEW_SCHEMA,
        "status": "research_candidate_pending_human_review",
        "formal_dataset_registration_authorized": False,
        "automatic_admission_gates": {
            "all_automatic_gates_passed": True,
        },
        "outputs": {
            "media": {
                "idle_side": freezer.bridge._absolute_record(idle_side),
                "walking_side": freezer.bridge._absolute_record(walking_side),
            }
        },
    }
    source_asset = {
        "asset_id": "dog_fixture_generated_animal_0123456789ab",
        "asset_class": "animal",
    }
    presentation_payload = {
        "schema": freezer.presentation.PRESENTATION_SCHEMA,
        "expected_source_review_sha256": _sha256(review_path),
        "source_review": {
            "path": str(review_path.resolve()),
            "sha256": _sha256(review_path),
            "size_bytes": review_path.stat().st_size,
        },
        "output": {
            "path": str(presentation_video.resolve()),
            "sha256": _sha256(presentation_video),
            "size_bytes": presentation_video.stat().st_size,
            "codec": "h264",
            "width": 1536,
            "height": 768,
            "frame_count": 8,
            "frame_rate": "8/1",
            "duration_seconds": 1.0,
            "readback": {"fixture": "already validated by compositor"},
            "full_decode_passed": True,
        },
        "receipt_sha256": None,
    }
    presentation_payload["receipt_sha256"] = freezer.presentation.hash_without(
        presentation_payload,
        "receipt_sha256",
    )
    presentation_receipt = presentation_root / freezer.presentation.RECEIPT_NAME
    presentation_receipt.write_text(
        json.dumps(presentation_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    presentation_video.chmod(0o444)
    presentation_receipt.chmod(0o444)
    presentation_root.chmod(0o555)

    def load_registry(path, selected_source, *, expected_file_sha256):
        assert Path(path) == registry_path
        assert Path(selected_source) == source_path
        if expected_file_sha256 != _sha256(registry_path):
            raise contracts.ContractError("source registry changed")
        return (
            registry_path,
            {"fixture": "registry"},
            {"fixture": "request"},
            {"fixture": "profile"},
            "current_exact_rebuild",
        )

    def load_source(
        path,
        roots,
        *,
        request,
        profile,
        require_derived_authority,
        expected_raw_static_decision_batch,
    ):
        assert Path(path) == source_path
        assert roots
        assert request == {"fixture": "request"}
        assert profile == {"fixture": "profile"}
        assert require_derived_authority is False
        assert expected_raw_static_decision_batch is None
        return (
            source_path,
            copy.deepcopy(source_asset),
            {"artifact:raw": source_artifact},
        )

    def load_review(path, *, source_asset, source_artifacts):
        assert Path(path) == review_path
        assert source_asset["asset_id"] == "dog_fixture_generated_animal_0123456789ab"
        assert source_artifacts == {"artifact:raw": source_artifact}
        return (
            review_path,
            copy.deepcopy(review_payload),
            animated_glb,
            {
                "output:animated_glb": animated_glb,
                "output:audit": review_audit,
            },
        )

    def load_presentation(
        path,
        expected_receipt_file_sha256,
        *,
        expected_source_review_sha256=None,
    ):
        receipt_path = freezer.presentation.resolve_regular_file(
            Path(path),
            "presentation receipt",
        )
        encoded, _guard = freezer.presentation.read_stable_bytes(
            receipt_path,
            "presentation receipt",
        )
        observed_sha256 = hashlib.sha256(encoded).hexdigest()
        if observed_sha256 != expected_receipt_file_sha256:
            raise freezer.presentation.PresentationContractError(
                "presentation receipt failed external SHA-256 authentication"
            )
        try:
            payload = freezer.presentation.strict_json_loads(encoded)
        except freezer.presentation.StrictJSONError as error:
            raise freezer.presentation.PresentationContractError(
                "presentation receipt is not strict JSON"
            ) from error
        if payload["expected_source_review_sha256"] != expected_source_review_sha256:
            raise freezer.presentation.PresentationContractError(
                "presentation receipt source-review authority changed"
            )
        return payload, {
            "path": str(receipt_path),
            "sha256": observed_sha256,
            "size_bytes": len(encoded),
        }

    monkeypatch.setattr(freezer.bridge, "load_source_registry_anchor", load_registry)
    monkeypatch.setattr(freezer.bridge, "load_source_asset", load_source)
    monkeypatch.setattr(freezer.bridge, "load_animation_review", load_review)
    monkeypatch.setattr(
        freezer.presentation,
        "load_presentation_receipt",
        load_presentation,
    )
    return {
        "registry_path": registry_path,
        "registry_sha256": _sha256(registry_path),
        "source_path": source_path,
        "review_path": review_path,
        "review_sha256": _sha256(review_path),
        "review_payload": review_payload,
        "source_asset": source_asset,
        "source_artifact": source_artifact,
        "animated_glb": animated_glb,
        "review_audit": review_audit,
        "idle_side": idle_side,
        "walking_side": walking_side,
        "presentation_root": presentation_root,
        "presentation_video": presentation_video,
        "presentation_payload": presentation_payload,
        "presentation_receipt": presentation_receipt,
        "presentation_receipt_sha256": _sha256(presentation_receipt),
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
        "presentation_receipt_path": authority["presentation_receipt"],
        "expected_presentation_receipt_sha256": authority[
            "presentation_receipt_sha256"
        ],
        "decision": freezer.APPROVED,
        "checks": _checks(),
        "caveats": [],
        "notes": "The user explicitly approved all six authenticated views.",
        "user_explicit_decision": freezer.APPROVED,
        "user_explicit_review_sha256": authority["review_sha256"],
        "user_explicit_presentation_receipt_sha256": authority[
            "presentation_receipt_sha256"
        ],
        "output_root": output_root,
    }
    values.update(overrides)
    return freezer.freeze_decision(**values)


def _compact_evidence_arguments(
    authority,
    tmp_path,
    *,
    rebind_current_readback_to_style_video=False,
):
    style_approval = {
        "schema": freezer.bridge.MOTION_STYLE_APPROVAL_SCHEMA,
        "status": "approved_for_idle_walking_motion_style",
        "actions": copy.deepcopy(freezer.bridge.ANIMATION_ACTIONS),
        "evidence_video": freezer.bridge._absolute_record(
            authority["presentation_video"]
        ),
    }
    style_approval["approval_sha256"] = freezer.bridge._hash_without(
        style_approval,
        "approval_sha256",
    )
    style_path = tmp_path / "motion_style_approval.json"
    style_path.write_text(
        json.dumps(style_approval, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    action_readbacks = {
        "Idle": freezer.bridge._absolute_record(authority["idle_side"]),
        "Walking": freezer.bridge._absolute_record(authority["walking_side"]),
    }
    if rebind_current_readback_to_style_video:
        action_readbacks["Walking"] = freezer.bridge._absolute_record(
            authority["presentation_video"]
        )
    short_readback = {
        "schema": freezer.bridge.CURRENT_ASSET_SHORT_READBACK_SCHEMA,
        "status": "passed_current_asset_geometry_and_actions",
        "asset_id": authority["source_asset"]["asset_id"],
        "animation_review": freezer.bridge._absolute_record(
            authority["review_path"]
        ),
        "reviewed_animated_glb": freezer.bridge._absolute_record(
            authority["animated_glb"]
        ),
        "actions": copy.deepcopy(freezer.bridge.ANIMATION_ACTIONS),
        "action_readbacks": action_readbacks,
        "checks": copy.deepcopy(
            freezer.bridge.CURRENT_ASSET_SHORT_READBACK_CHECKS
        ),
    }
    short_readback["receipt_sha256"] = freezer.bridge._hash_without(
        short_readback,
        "receipt_sha256",
    )
    readback_path = tmp_path / "current_asset_short_readback.json"
    readback_path.write_text(
        json.dumps(short_readback, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "presentation_receipt_path": None,
        "expected_presentation_receipt_sha256": None,
        "user_explicit_presentation_receipt_sha256": None,
        "checks": None,
        "notes": (
            "Reused the approved Idle/Walking motion style; the current review "
            "machine readback passed."
        ),
        "user_explicit_decision": None,
        "user_explicit_review_sha256": None,
        "motion_style_approval_path": style_path,
        "expected_motion_style_approval_sha256": _sha256(style_path),
        "current_asset_short_readback_path": readback_path,
        "expected_current_asset_short_readback_sha256": _sha256(
            readback_path
        ),
        "user_explicit_motion_style_approval_sha256": _sha256(style_path),
    }


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
    assert receipt["schema"].endswith("_v2")
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
    assert set(receipt["presentation_evidence"]) == (
        freezer.PRESENTATION_EVIDENCE_FIELDS
    )
    assert receipt["presentation_evidence"] == {
        "presentation_receipt": {
            "path": str(authenticated_review["presentation_receipt"].resolve()),
            "sha256": authenticated_review["presentation_receipt_sha256"],
            "size_bytes": authenticated_review["presentation_receipt"].stat().st_size,
        },
        "expected_presentation_receipt_file_sha256": authenticated_review[
            "presentation_receipt_sha256"
        ],
        "presentation_receipt_sha256": authenticated_review["presentation_payload"][
            "receipt_sha256"
        ],
        "output_video": authenticated_review["presentation_payload"]["output"],
    }
    assert receipt["user_instruction_binding"] == {
        "decision": freezer.APPROVED,
        "review_sha256": authenticated_review["review_sha256"],
        "presentation_receipt_file_sha256": authenticated_review[
            "presentation_receipt_sha256"
        ],
        "all_six_checks_explicit": True,
    }
    assert receipt["authenticated_review_artifact_count"] == 2
    assert receipt["receipt_sha256"] == freezer.bridge._hash_without(
        receipt, "receipt_sha256"
    )
    assert decision_path.stat().st_mode & 0o222 == 0
    assert receipt_path.stat().st_mode & 0o222 == 0
    assert stat.S_IMODE(decision_path.parent.stat().st_mode) == 0o555
    assert {path.name for path in decision_path.parent.iterdir()} == (
        freezer.PUBLISHED_FILE_NAMES
    )


def test_freezes_motion_style_baseline_with_current_asset_short_readback(
    authenticated_review,
    tmp_path,
):
    arguments = _compact_evidence_arguments(
        authenticated_review,
        tmp_path,
    )
    decision_path = _freeze(
        authenticated_review,
        tmp_path / "frozen_compact_approval",
        **arguments,
    )
    receipt = contracts.load_json(
        decision_path.parent / "decision_freeze_receipt.json"
    )
    decision = contracts.load_json(decision_path)

    assert receipt["presentation_evidence"]["mode"] == (
        freezer.MOTION_STYLE_AND_CURRENT_READBACK_MODE
    )
    assert set(receipt["presentation_evidence"]) == (
        freezer.MOTION_STYLE_AND_CURRENT_READBACK_EVIDENCE_FIELDS
    )
    assert receipt["user_instruction_binding"] == {
        "decision": freezer.APPROVED,
        "motion_style_approval_file_sha256": (
            arguments["expected_motion_style_approval_sha256"]
        ),
        "current_asset_short_readback_file_sha256": (
            arguments["expected_current_asset_short_readback_sha256"]
        ),
        "current_asset_readback_is_machine_gate": True,
    }
    assert all(decision["checks"].values())
    assert decision["notes"] == (
        "Reused the approved Idle/Walking motion style; the current review "
        "machine readback passed."
    )
    assert receipt["receipt_sha256"] == freezer.bridge._hash_without(
        receipt,
        "receipt_sha256",
    )


def test_old_style_video_cannot_masquerade_as_current_review_action_readback(
    authenticated_review,
    tmp_path,
):
    arguments = _compact_evidence_arguments(
        authenticated_review,
        tmp_path,
        rebind_current_readback_to_style_video=True,
    )
    output = tmp_path / "rejected_style_video_as_current_readback"

    with pytest.raises(
        contracts.ContractError,
        match="current geometry/action binding is invalid",
    ):
        _freeze(
            authenticated_review,
            output,
            **arguments,
        )
    assert not output.exists()


def test_freeze_reauthenticates_direct_derived_source_authority(
    authenticated_review,
    tmp_path,
    monkeypatch,
):
    decision_batch = {
        "path": str((tmp_path / "direct_static_decisions.json").resolve()),
        "sha256": "1" * 64,
        "decision_batch_sha256": "2" * 64,
    }

    def load_registry(path, selected_source, *, expected_file_sha256):
        assert Path(path) == authenticated_review["registry_path"]
        assert Path(selected_source) == authenticated_review["source_path"]
        assert expected_file_sha256 == authenticated_review["registry_sha256"]
        return (
            authenticated_review["registry_path"],
            {
                "schema": freezer.bridge.source_registry.DERIVED_REGISTRY_SCHEMA,
                "direct_source_authority": {"fixture": "direct"},
                "static_decision_batch": decision_batch,
            },
            {"fixture": "direct authority"},
            {"fixture": "direct context"},
            freezer.bridge.DIRECT_SOURCE_AUTHORITY_VALIDATION_MODE,
        )

    def load_source(
        path,
        roots,
        *,
        request,
        profile,
        require_derived_authority,
        expected_raw_static_decision_batch,
    ):
        assert Path(path) == authenticated_review["source_path"]
        assert roots
        assert request == {"fixture": "direct authority"}
        assert profile == {"fixture": "direct context"}
        assert require_derived_authority is True
        assert expected_raw_static_decision_batch == decision_batch
        return (
            authenticated_review["source_path"],
            copy.deepcopy(authenticated_review["source_asset"]),
            {
                "artifact:raw": authenticated_review["source_artifact"]
            },
        )

    monkeypatch.setattr(
        freezer.bridge,
        "load_source_registry_anchor",
        load_registry,
    )
    monkeypatch.setattr(freezer.bridge, "load_source_asset", load_source)

    decision_path = _freeze(
        authenticated_review,
        tmp_path / "frozen_direct_derived_decision",
    )
    receipt = json.loads(
        (decision_path.parent / "decision_freeze_receipt.json").read_text(
            encoding="utf-8"
        )
    )

    assert receipt["source_asset_registry_validation_mode"] == (
        freezer.bridge.DIRECT_SOURCE_AUTHORITY_VALIDATION_MODE
    )


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


def test_rejects_wrong_external_presentation_receipt_raw_sha(
    authenticated_review,
    tmp_path,
):
    with pytest.raises(contracts.ContractError, match="external SHA-256"):
        _freeze(
            authenticated_review,
            tmp_path / "wrong_presentation_raw_sha",
            expected_presentation_receipt_sha256="f" * 64,
            user_explicit_presentation_receipt_sha256="f" * 64,
        )


def test_rejects_user_instruction_bound_to_another_presentation(
    authenticated_review,
    tmp_path,
):
    with pytest.raises(contracts.ContractError, match="expected presentation"):
        _freeze(
            authenticated_review,
            tmp_path / "wrong_user_presentation_sha",
            user_explicit_presentation_receipt_sha256="f" * 64,
        )


def test_rejects_presentation_internal_self_hash_mismatch(
    authenticated_review,
    tmp_path,
):
    payload = copy.deepcopy(authenticated_review["presentation_payload"])
    payload["receipt_sha256"] = "f" * 64
    raw_sha256 = _rewrite_presentation_receipt(authenticated_review, payload)

    with pytest.raises(contracts.ContractError, match="canonical self-hash"):
        _freeze(
            authenticated_review,
            tmp_path / "wrong_internal_presentation_sha",
            expected_presentation_receipt_sha256=raw_sha256,
            user_explicit_presentation_receipt_sha256=raw_sha256,
        )


def test_rejects_presentation_bound_to_another_review(
    authenticated_review,
    tmp_path,
):
    payload = copy.deepcopy(authenticated_review["presentation_payload"])
    payload["expected_source_review_sha256"] = "f" * 64
    payload["source_review"]["sha256"] = "f" * 64
    payload["receipt_sha256"] = freezer.presentation.hash_without(
        payload,
        "receipt_sha256",
    )
    raw_sha256 = _rewrite_presentation_receipt(authenticated_review, payload)

    with pytest.raises(contracts.ContractError, match="source-review authority"):
        _freeze(
            authenticated_review,
            tmp_path / "wrong_presentation_review",
            expected_presentation_receipt_sha256=raw_sha256,
            user_explicit_presentation_receipt_sha256=raw_sha256,
        )


def test_rejects_presentation_video_real_byte_mismatch(
    authenticated_review,
    tmp_path,
):
    _unseal_presentation(authenticated_review)
    authenticated_review["presentation_video"].write_bytes(
        b"replaced six-view video bytes"
    )
    _seal_presentation(authenticated_review)

    with pytest.raises(contracts.ContractError, match="real-byte authentication"):
        _freeze(
            authenticated_review,
            tmp_path / "wrong_presentation_video",
        )


def test_rejects_presentation_receipt_symlink(
    authenticated_review,
    tmp_path,
):
    alias = tmp_path / "presentation_receipt_alias.json"
    alias.symlink_to(authenticated_review["presentation_receipt"])

    with pytest.raises(contracts.ContractError, match="unsafe symlink"):
        _freeze(
            authenticated_review,
            tmp_path / "symlinked_presentation_receipt",
            presentation_receipt_path=alias,
        )


def test_rejects_non_strict_presentation_receipt_json(
    authenticated_review,
    tmp_path,
):
    _unseal_presentation(authenticated_review)
    authenticated_review["presentation_receipt"].write_text(
        '{"schema":"first","schema":"duplicate"}\n',
        encoding="utf-8",
    )
    _seal_presentation(authenticated_review)
    raw_sha256 = _sha256(authenticated_review["presentation_receipt"])

    with pytest.raises(contracts.ContractError, match="not strict JSON"):
        _freeze(
            authenticated_review,
            tmp_path / "non_strict_presentation",
            expected_presentation_receipt_sha256=raw_sha256,
            user_explicit_presentation_receipt_sha256=raw_sha256,
        )


def test_final_presentation_failure_occurs_before_ready_boundary(
    authenticated_review,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "pre_publish_presentation_race"
    original_authenticate = freezer.authenticate_presentation
    calls = 0

    def race_before_ready(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert not output.exists()
            raise contracts.ContractError(
                "presentation evidence authentication failed: raced"
            )
        return original_authenticate(**kwargs)

    monkeypatch.setattr(
        freezer,
        "authenticate_presentation",
        race_before_ready,
    )

    with pytest.raises(contracts.ContractError, match="raced"):
        _freeze(authenticated_review, output)

    assert calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".pre_publish_presentation_race.*.staging"))


def test_atomic_no_replace_rejects_concurrent_output_without_touching_it(
    authenticated_review,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "atomic_collision"
    original_publish = freezer._atomic_publish_no_replace

    def create_collision_then_publish(
        parent_fd,
        staging_name,
        output_name,
        **kwargs,
    ):
        output.mkdir()
        (output / "sentinel").write_text("preserve", encoding="utf-8")
        original_publish(
            parent_fd,
            staging_name,
            output_name,
            **kwargs,
        )

    monkeypatch.setattr(
        freezer,
        "_atomic_publish_no_replace",
        create_collision_then_publish,
    )

    with pytest.raises(contracts.ContractError, match="refusing to replace"):
        _freeze(authenticated_review, output)

    assert (output / "sentinel").read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob(".atomic_collision.*.staging"))


def test_parent_fsync_failure_after_ready_never_deletes_final(
    authenticated_review,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "published_before_fsync_failure"
    original_publish = freezer._atomic_publish_no_replace
    original_fsync = freezer.os.fsync
    ready_parent_fd = None

    def publish_then_mark_ready(
        parent_fd,
        staging_name,
        output_name,
        **kwargs,
    ):
        nonlocal ready_parent_fd
        original_publish(
            parent_fd,
            staging_name,
            output_name,
            **kwargs,
        )
        ready_parent_fd = parent_fd

    def fail_ready_parent_fsync(descriptor):
        if ready_parent_fd is not None and descriptor == ready_parent_fd:
            raise OSError("simulated parent fsync failure after ready")
        return original_fsync(descriptor)

    monkeypatch.setattr(
        freezer,
        "_atomic_publish_no_replace",
        publish_then_mark_ready,
    )
    monkeypatch.setattr(freezer.os, "fsync", fail_ready_parent_fsync)

    with pytest.raises(OSError, match="after ready"):
        _freeze(authenticated_review, output)

    assert {path.name for path in output.iterdir()} == freezer.PUBLISHED_FILE_NAMES
    assert not list(tmp_path.glob(f".{output.name}.*.staging"))


@pytest.mark.parametrize("staging_file_name", sorted(freezer.PUBLISHED_FILE_NAMES))
def test_staging_byte_rewrite_in_publication_hook_never_becomes_visible(
    authenticated_review,
    tmp_path,
    monkeypatch,
    staging_file_name,
):
    output = tmp_path / f"staging_tamper_{staging_file_name}"
    original_publish = freezer._atomic_publish_no_replace

    def rewrite_staging_then_publish(
        parent_fd,
        staging_name,
        output_name,
        **kwargs,
    ):
        staging_fd = kwargs["staging_fd"]
        chmod_descriptor = os.open(
            staging_file_name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=staging_fd,
        )
        try:
            os.fchmod(chmod_descriptor, 0o600)
        finally:
            os.close(chmod_descriptor)
        descriptor = os.open(
            staging_file_name,
            os.O_WRONLY | os.O_NOFOLLOW,
            dir_fd=staging_fd,
        )
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"X")
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        original_publish(
            parent_fd,
            staging_name,
            output_name,
            **kwargs,
        )

    monkeypatch.setattr(
        freezer,
        "_atomic_publish_no_replace",
        rewrite_staging_then_publish,
    )

    with pytest.raises(contracts.ContractError, match="raw bytes changed"):
        _freeze(authenticated_review, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.staging"))


def test_parent_swap_after_identity_check_cannot_redirect_publish(
    authenticated_review,
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "publish_parent"
    parent.mkdir()
    output = parent / "frozen"
    moved_parent = tmp_path / "moved_parent"
    attacker_parent = tmp_path / "attacker_parent"
    attacker_parent.mkdir()
    sentinel = attacker_parent / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    original_publish = freezer._atomic_publish_no_replace

    def swap_parent_then_publish(
        parent_fd,
        staging_name,
        output_name,
        **kwargs,
    ):
        parent.rename(moved_parent)
        parent.symlink_to(attacker_parent, target_is_directory=True)
        original_publish(
            parent_fd,
            staging_name,
            output_name,
            **kwargs,
        )

    monkeypatch.setattr(
        freezer,
        "_atomic_publish_no_replace",
        swap_parent_then_publish,
    )

    with pytest.raises(contracts.ContractError, match="symlink path component"):
        _freeze(authenticated_review, output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (attacker_parent / output.name).exists()
    assert not (moved_parent / output.name).exists()
    assert not list(moved_parent.glob(".*.staging"))


def test_review_inode_swap_between_read_and_path_recheck_is_rejected(
    authenticated_review,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "review_inode_race"
    review_path = authenticated_review["review_path"]
    review_identity = (review_path.stat().st_dev, review_path.stat().st_ino)
    replacement = tmp_path / "review_replacement.json"
    replacement.write_bytes(b"X" * review_path.stat().st_size)
    original_read = freezer.os.read
    swapped = False

    def read_then_swap(descriptor, size):
        nonlocal swapped
        block = original_read(descriptor, size)
        current = os.fstat(descriptor)
        if (
            block
            and not swapped
            and (current.st_dev, current.st_ino) == review_identity
        ):
            swapped = True
            replacement.replace(review_path)
        return block

    monkeypatch.setattr(freezer.os, "read", read_then_swap)

    with pytest.raises(
        contracts.ContractError,
        match="changed during stable read|path identity changed",
    ):
        _freeze(authenticated_review, output)

    assert swapped is True
    assert not output.exists()


@pytest.mark.parametrize(
    "authority_path_key",
    (
        "registry_path",
        "source_path",
        "source_artifact",
        "review_audit",
        "animated_glb",
    ),
)
def test_complete_authority_graph_race_before_commit_fails_closed(
    authenticated_review,
    tmp_path,
    monkeypatch,
    authority_path_key,
):
    output = tmp_path / f"authority_race_{authority_path_key}"
    original_seal = freezer._seal_staging_at

    def seal_then_mutate(staging_fd):
        original_seal(staging_fd)
        authenticated_review[authority_path_key].write_bytes(
            f"raced-{authority_path_key}".encode("utf-8")
        )

    monkeypatch.setattr(freezer, "_seal_staging_at", seal_then_mutate)

    with pytest.raises(contracts.ContractError):
        _freeze(authenticated_review, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.staging"))


def test_cleanup_root_swap_never_touches_external_tree(tmp_path):
    parent = tmp_path / "cleanup_parent"
    parent.mkdir()
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_name, staging_fd, identity = freezer._create_staging_at(
        parent_fd,
        "cleanup",
    )
    freezer._write_json_at(staging_fd, "animation_decision.json", {"fixture": True})
    staging_path = parent / staging_name
    moved_staging = tmp_path / "moved_staging"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("must survive", encoding="utf-8")
    initial_mode = stat.S_IMODE(external.stat().st_mode)
    staging_path.rename(moved_staging)
    staging_path.symlink_to(external, target_is_directory=True)
    try:
        with pytest.raises(contracts.ContractError, match="identity changed"):
            freezer._remove_owned_staging_at(
                parent_fd,
                staging_fd,
                staging_name,
                identity,
            )
        assert sentinel.read_text(encoding="utf-8") == "must survive"
        assert stat.S_IMODE(external.stat().st_mode) == initial_mode
    finally:
        os.close(staging_fd)
        os.close(parent_fd)


def test_cleanup_unknown_artifact_is_quarantined(tmp_path):
    parent = tmp_path / "quarantine_parent"
    parent.mkdir()
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_name, staging_fd, identity = freezer._create_staging_at(
        parent_fd,
        "quarantine",
    )
    unexpected_fd = os.open(
        "unexpected",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=staging_fd,
    )
    os.close(unexpected_fd)
    try:
        with pytest.raises(contracts.ContractError, match="unknown artifacts"):
            freezer._remove_owned_staging_at(
                parent_fd,
                staging_fd,
                staging_name,
                identity,
            )
        assert "unexpected" in os.listdir(staging_fd)
        assert (parent / staging_name).is_dir()
    finally:
        os.close(staging_fd)
        os.close(parent_fd)


def test_freezer_exposes_both_complete_approval_evidence_modes():
    parameters = inspect.signature(freezer.freeze_decision).parameters
    for name in (
        "presentation_receipt_path",
        "expected_presentation_receipt_sha256",
        "user_explicit_presentation_receipt_sha256",
        "motion_style_approval_path",
        "expected_motion_style_approval_sha256",
        "current_asset_short_readback_path",
        "expected_current_asset_short_readback_sha256",
        "user_explicit_motion_style_approval_sha256",
        "checks",
        "user_explicit_decision",
    ):
        assert parameters[name].default is None


def test_publisher_uses_renameat2_no_replace_not_plain_rename():
    publisher = inspect.getsource(freezer._atomic_publish_no_replace)
    source = Path(freezer.__file__).read_text(encoding="utf-8")
    assert "renameat2" in publisher
    assert publisher.count("parent_fd") >= 3
    assert "-100" not in publisher
    assert "os.rename(" not in source
    assert "os.walk(" not in source


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


@pytest.mark.skipif(
    not PRODUCTION_SHIBA_PRESENTATION_RECEIPT.is_file(),
    reason="sealed production Shiba presentation receipt is unavailable",
)
def test_real_production_presentation_loader_and_raw_internal_sha_separation():
    receipt_path = PRODUCTION_SHIBA_PRESENTATION_RECEIPT
    encoded = receipt_path.read_bytes()
    raw_sha256 = hashlib.sha256(encoded).hexdigest()
    payload = freezer.presentation.strict_json_loads(encoded)
    internal_sha256 = payload["receipt_sha256"]
    review_path = Path(payload["source_review"]["path"])
    review_sha256 = payload["expected_source_review_sha256"]

    evidence, guards = freezer.authenticate_presentation(
        presentation_receipt_path=receipt_path,
        expected_presentation_receipt_sha256=raw_sha256,
        animation_review_path=review_path,
        expected_animation_review_sha256=review_sha256,
    )

    assert raw_sha256 != internal_sha256
    assert evidence["presentation_receipt"]["sha256"] == raw_sha256
    assert evidence["presentation_receipt_sha256"] == internal_sha256
    assert evidence["output_video"]["sha256"] == (
        "5d776dd594cf5e63f1633baf9c5efac2387b9ed07948cad5f84ff81e7524f0c6"
    )
    assert set(guards) == {
        "animation_review",
        "publication_directory",
        "presentation_receipt",
        "output_video",
    }

    with pytest.raises(contracts.ContractError, match="external SHA-256"):
        freezer.authenticate_presentation(
            presentation_receipt_path=receipt_path,
            expected_presentation_receipt_sha256=internal_sha256,
            animation_review_path=review_path,
            expected_animation_review_sha256=review_sha256,
        )
