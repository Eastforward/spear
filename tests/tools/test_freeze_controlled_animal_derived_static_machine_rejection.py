from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import (
    controlled_animal_derived_static_machine_rejection_contract as receipt_contract,
)
from tools import controlled_animal_derived_static_review_contract as review_contract
from tools import controlled_source_asset_schema as contracts
from tools import freeze_controlled_animal_derived_static_machine_rejection as freezer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contracts.canonical_json(payload) + "\n", encoding="utf-8")
    return path


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


@pytest.fixture
def pending_review(tmp_path, monkeypatch):
    root = tmp_path / "derived_review"
    front = _write_bytes(root / "pbr_five_view/views/front.png", b"front png")
    back = _write_bytes(root / "pbr_five_view/views/back.png", b"back png")
    contact = _write_bytes(
        root / "pbr_five_view/contact_sheet.png", b"contact sheet png"
    )
    raw_decision = {
        "schema": "avengine_controlled_animal_static_decision_v1",
        "instance_id": "cat_fixture",
        "decision": "rejected",
        "state_classification": "rejected",
        "formal_dataset_registration_authorized": False,
        "next_gate": "stop",
        "decision_sha256": "",
    }
    raw_decision["decision_sha256"] = review_contract.hash_without(
        raw_decision, "decision_sha256"
    )
    raw_path = _write_json(tmp_path / "raw/static_decision.json", raw_decision)
    review = {
        "instance_identity": {"instance_id": "cat_fixture"},
        "status": review_contract.REVIEW_STATUS,
        "formal_dataset_registration_authorized": False,
        "human_review": {"status": "pending", "decision": None},
        "source_authorities": {
            "raw_static_decision": {
                "decision": "rejected",
                "decision_sha256": raw_decision["decision_sha256"],
                "file": _record(raw_path),
                "formal_dataset_registration_authorized": False,
                "state_classification": "rejected",
            }
        },
        "evidence": {
            "pbr_five_view": {
                "contact_sheet": {
                    "path": "pbr_five_view/contact_sheet.png",
                    "sha256": _sha256(contact),
                    "size_bytes": contact.stat().st_size,
                },
                "views": {
                    "front": {
                        "path": "pbr_five_view/views/front.png",
                        "sha256": _sha256(front),
                        "size_bytes": front.stat().st_size,
                    },
                    "back": {
                        "path": "pbr_five_view/views/back.png",
                        "sha256": _sha256(back),
                        "size_bytes": back.stat().st_size,
                    },
                },
            }
        },
        "review_sha256": "",
    }
    review_path = root / "derived_static_review.json"

    def write_review() -> None:
        review["review_sha256"] = review_contract.hash_without(review, "review_sha256")
        _write_json(review_path, review)

    write_review()
    monkeypatch.setattr(
        freezer.review_contract,
        "validate_review",
        lambda value: copy.deepcopy(value),
    )
    return {
        "root": root,
        "review": review,
        "review_path": review_path,
        "write_review": write_review,
        "front": front,
        "back": back,
        "contact": contact,
        "raw_path": raw_path,
    }


def _freeze(pending_review, output: Path, **overrides) -> Path:
    values = {
        "instance_id": "cat_fixture",
        "review_path": pending_review["review_path"],
        "expected_review_file_sha256": _sha256(pending_review["review_path"]),
        "expected_internal_review_sha256": pending_review["review"]["review_sha256"],
        "observations": [
            (
                "visible_stray_bar_or_spike_geometry",
                "pbr_front",
                _sha256(pending_review["front"]),
            )
        ],
        "output_path": output,
    }
    values.update(overrides)
    return freezer.freeze_rejection(**values)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(receipt: dict) -> None:
    receipt["receipt_sha256"] = receipt_contract.hash_without(receipt, "receipt_sha256")


def test_cli_requires_both_review_hashes_and_exact_artifact_sha():
    parser = freezer.build_argument_parser()
    args = parser.parse_args(
        [
            "--instance-id",
            "cat_fixture",
            "--review",
            "review.json",
            "--expected-review-file-sha256",
            "a" * 64,
            "--expected-internal-review-sha256",
            "b" * 64,
            "--observation",
            "visible_hole_like_opening",
            "pbr_back",
            "c" * 64,
            "--output",
            "receipt.json",
        ]
    )
    assert args.observation == [["visible_hole_like_opening", "pbr_back", "c" * 64]]

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--instance-id",
                "cat_fixture",
                "--review",
                "review.json",
                "--expected-review-file-sha256",
                "a" * 64,
                "--expected-internal-review-sha256",
                "b" * 64,
                "--observation",
                "visible_hole_like_opening",
                "pbr_back",
                "--output",
                "receipt.json",
            ]
        )


def test_freezes_exact_fail_closed_receipt_without_any_human_decision(
    pending_review, tmp_path
):
    output = _freeze(pending_review, tmp_path / "machine_rejection.json")
    receipt = receipt_contract.validate_receipt(_load(output))

    assert output.stat().st_mode & 0o777 == 0o444
    assert receipt["state_classification"] == "rejected"
    assert receipt["formal_dataset_registration_authorized"] is False
    assert receipt["review_binding"]["review_file"]["sha256"] == _sha256(
        pending_review["review_path"]
    )
    assert (
        receipt["review_binding"]["internal_review_sha256"]
        == pending_review["review"]["review_sha256"]
    )
    assert receipt["raw_static_rejection"]["decision"] == "rejected"
    assert receipt["raw_static_rejection"]["preserved"] is True
    assert receipt["raw_static_rejection"]["overwritten"] is False
    assert receipt["authority"] == {
        "observer_kind": receipt_contract.OBSERVER_KIND,
        "user_decision_claimed": False,
        "human_decision_claimed": False,
        "approval_claimed": False,
        "product_definition_changed": False,
    }
    assert receipt["effect"]["derived_static_admission"] == "rejected_fail_closed"
    assert not any(
        receipt["effect"][field]
        for field in (
            "animation_authorized",
            "native_change_authorized",
            "repaired_asset_registration_authorized",
            "rigging_authorized",
            "source_asset_v2_authorized",
            "ue_execution_authorized",
        )
    )
    assert receipt["observations"][0]["artifact"]["sha256"] == _sha256(
        pending_review["front"]
    )
    assert "decision" not in receipt
    assert not list(tmp_path.glob(".*machine_rejection.json.*.staging"))


def test_output_is_atomic_no_replace_and_original_bytes_survive(
    pending_review, tmp_path
):
    output = _freeze(pending_review, tmp_path / "machine_rejection.json")
    original = output.read_bytes()

    with pytest.raises(contracts.ContractError, match="refusing to replace"):
        _freeze(pending_review, output)

    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".*machine_rejection.json.*.staging"))


def test_concurrent_destination_wins_without_being_overwritten(
    pending_review, tmp_path, monkeypatch
):
    output = tmp_path / "machine_rejection.json"
    original_link = freezer.os.link

    def race_link(source, destination, **kwargs):
        output.write_bytes(b"concurrent publisher")
        raise FileExistsError(destination)

    monkeypatch.setattr(freezer.os, "link", race_link)
    with pytest.raises(contracts.ContractError, match="appeared concurrently"):
        _freeze(pending_review, output)
    monkeypatch.setattr(freezer.os, "link", original_link)

    assert output.read_bytes() == b"concurrent publisher"
    assert not list(tmp_path.glob(".*machine_rejection.json.*.staging"))


def test_rejects_symlink_review_artifact_and_output_parent(pending_review, tmp_path):
    review_link = tmp_path / "review_link.json"
    review_link.symlink_to(pending_review["review_path"])
    with pytest.raises(contracts.ContractError, match="unsafe symlink"):
        _freeze(
            pending_review,
            tmp_path / "rejected_review_link.json",
            review_path=review_link,
        )

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(contracts.ContractError, match="output parent.*symlink"):
        _freeze(pending_review, linked_parent / "receipt.json")

    target = _write_bytes(tmp_path / "artifact_target.png", b"front png")
    artifact_link = pending_review["root"] / "pbr_five_view/views/front_link.png"
    artifact_link.symlink_to(target)
    descriptor = pending_review["review"]["evidence"]["pbr_five_view"]["views"]["front"]
    descriptor.update(
        {
            "path": "pbr_five_view/views/front_link.png",
            "sha256": _sha256(target),
            "size_bytes": target.stat().st_size,
        }
    )
    pending_review["write_review"]()
    with pytest.raises(contracts.ContractError, match="unsafe symlink"):
        _freeze(
            pending_review,
            tmp_path / "rejected_artifact_link.json",
            observations=[
                (
                    "visible_stray_bar_or_spike_geometry",
                    "pbr_front",
                    _sha256(target),
                )
            ],
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_review_file_sha256": "0" * 64}, "expected SHA-256"),
        ({"expected_internal_review_sha256": "0" * 64}, "internal SHA-256"),
        (
            {
                "observations": [
                    (
                        "visible_stray_bar_or_spike_geometry",
                        "pbr_front",
                        "0" * 64,
                    )
                ]
            },
            "external SHA-256",
        ),
        (
            {"observations": [("unbounded_subjective_dislike", "pbr_front", "0" * 64)]},
            "unknown objective defect code",
        ),
        (
            {
                "observations": [
                    (
                        "visible_stray_bar_or_spike_geometry",
                        "arbitrary_file",
                        "0" * 64,
                    )
                ]
            },
            "unknown evidence role",
        ),
    ],
)
def test_rejects_unbound_review_artifact_or_unenumerated_observation(
    pending_review, tmp_path, overrides, message
):
    with pytest.raises(contracts.ContractError, match=message):
        _freeze(
            pending_review,
            tmp_path / "rejected_unbound.json",
            **overrides,
        )


def test_rejects_duplicate_observation(pending_review, tmp_path):
    observation = (
        "visible_stray_bar_or_spike_geometry",
        "pbr_front",
        _sha256(pending_review["front"]),
    )
    with pytest.raises(contracts.ContractError, match="duplicate"):
        _freeze(
            pending_review,
            tmp_path / "rejected_duplicate.json",
            observations=[observation, observation],
        )


def test_rejects_raw_rejection_changed_after_review_was_frozen(
    pending_review, tmp_path
):
    pending_review["raw_path"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(contracts.ContractError, match="expected SHA-256"):
        _freeze(pending_review, tmp_path / "rejected_raw_tamper.json")


@pytest.mark.parametrize(
    "ambiguous_json",
    [
        b'{"instance_identity":{"instance_id":"cat_fixture"},'
        b'"instance_identity":{"instance_id":"cat_fixture"}}\n',
        b'{"non_finite":NaN}\n',
    ],
)
def test_rejects_ambiguous_or_nonstandard_review_json(
    pending_review, tmp_path, ambiguous_json
):
    pending_review["review_path"].write_bytes(ambiguous_json)

    with pytest.raises(contracts.ContractError, match="strict UTF-8 JSON"):
        _freeze(
            pending_review,
            tmp_path / "rejected_ambiguous_review.json",
            expected_review_file_sha256=_sha256(pending_review["review_path"]),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda receipt: receipt["authority"].update(
                {"user_decision_claimed": True}
            ),
            "authority.user_decision_claimed",
        ),
        (
            lambda receipt: receipt.update(
                {"formal_dataset_registration_authorized": True}
            ),
            "formal_dataset_registration_authorized",
        ),
        (
            lambda receipt: receipt["effect"].update(
                {"source_asset_v2_authorized": True}
            ),
            "effect.source_asset_v2_authorized",
        ),
        (
            lambda receipt: receipt.update({"decision": "approved"}),
            "receipt fields are invalid",
        ),
    ],
)
def test_contract_rejects_human_authority_approval_or_registration_claims(
    pending_review, tmp_path, mutate, message
):
    output = _freeze(pending_review, tmp_path / "valid_receipt.json")
    receipt = _load(output)
    mutate(receipt)
    _rehash(receipt)

    with pytest.raises(receipt_contract.MachineRejectionContractError, match=message):
        receipt_contract.validate_receipt(receipt)


def test_contract_rejects_unsorted_or_duplicate_observations(pending_review, tmp_path):
    output = _freeze(
        pending_review,
        tmp_path / "valid_receipt.json",
        observations=[
            (
                "visible_stray_bar_or_spike_geometry",
                "pbr_front",
                _sha256(pending_review["front"]),
            ),
            (
                "visible_hole_like_opening",
                "pbr_back",
                _sha256(pending_review["back"]),
            ),
        ],
    )
    receipt = _load(output)
    receipt["observations"].reverse()
    _rehash(receipt)
    with pytest.raises(
        receipt_contract.MachineRejectionContractError,
        match="canonically sorted",
    ):
        receipt_contract.validate_receipt(receipt)

    receipt = _load(output)
    receipt["observations"].append(copy.deepcopy(receipt["observations"][0]))
    _rehash(receipt)
    with pytest.raises(
        receipt_contract.MachineRejectionContractError,
        match="canonically sorted",
    ):
        receipt_contract.validate_receipt(receipt)


def test_source_uses_descriptor_relative_atomic_no_replace_publication():
    source = Path(freezer.__file__).read_text(encoding="utf-8")
    assert "os.O_NOFOLLOW" in source
    assert "os.O_EXCL" in source
    assert "src_dir_fd=directory" in source
    assert "dst_dir_fd=directory" in source
    assert "follow_symlinks=False" in source
    assert "os.fsync(directory)" in source
    assert "os.link(" in source
    assert "os.rename(" not in source
