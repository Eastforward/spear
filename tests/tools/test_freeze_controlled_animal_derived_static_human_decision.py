from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path

import pytest

from tools import controlled_animal_derived_static_review_contract as review_contract
from tools import controlled_source_asset_schema as contracts
from tools import freeze_controlled_animal_derived_static_human_decision as freezer


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


def _relative_record(path: Path, root: Path) -> dict:
    record = _record(path)
    record["path"] = path.resolve().relative_to(root.resolve()).as_posix()
    return record


@pytest.fixture
def pending_review(tmp_path, monkeypatch):
    review_root = tmp_path / "derived_review"
    source_root = tmp_path / "source"
    source_files = {
        name: _write_bytes(source_root / f"{name}.bin", name.encode("utf-8"))
        for name in (
            "preflight",
            "pixal_batch",
            "raw_decision_batch",
            "raw_pixal",
            "reference",
        )
    }
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
    raw_path = _write_json(source_root / "raw_decision.json", raw_decision)

    geometry_files = {
        name: _write_bytes(source_root / f"{name}.bin", name.encode("utf-8"))
        for name in (
            "geometry_closure",
            "repaired_glb",
            "repair_manifest",
            "geometry_audit",
        )
    }
    clay_root = source_root / "clay"
    clay_render = _write_json(clay_root / "render_manifest.json", {"mode": "clay"})
    clay_views = {
        name: _write_bytes(clay_root / f"{name}.png", name.encode("utf-8"))
        for name in review_contract.VIEWS
    }
    clay_contact = _write_bytes(clay_root / "contact.png", b"clay contact")

    pbr_root = review_root / "pbr_five_view"
    pbr_render = _write_json(pbr_root / "render_manifest.json", {"mode": "pbr"})
    pbr_views = {
        name: _write_bytes(pbr_root / "views" / f"{name}.png", name.encode("utf-8"))
        for name in review_contract.VIEWS
    }
    pbr_contact = _write_bytes(pbr_root / "contact.png", b"pbr contact")
    pbr_log = _write_bytes(pbr_root / "blender.log", b"blender log")
    producer_tool = _write_bytes(source_root / "review_tool.py", b"review tool")
    renderer = _write_bytes(source_root / "renderer.py", b"renderer")

    review = {
        "instance_identity": {
            "instance_id": "cat_fixture",
            "sampled_attributes": {
                "body_build": "stocky",
                "coat_color": "blue",
                "size": "medium",
            },
            "target_physical_profile": {"control_attribute": "size"},
        },
        "status": review_contract.REVIEW_STATUS,
        "formal_dataset_registration_authorized": False,
        "human_review": {"status": "pending", "decision": None},
        "source_authorities": {
            "frozen_preflight": {"file": _record(source_files["preflight"])},
            "pixal_batch": {"file": _record(source_files["pixal_batch"])},
            "raw_static_decision_batch": {
                "file": _record(source_files["raw_decision_batch"])
            },
            "raw_static_decision": {
                "decision": "rejected",
                "decision_sha256": raw_decision["decision_sha256"],
                "file": _record(raw_path),
                "formal_dataset_registration_authorized": False,
                "state_classification": "rejected",
            },
            "raw_pixal_glb": _record(source_files["raw_pixal"]),
            "reference_2d": _record(source_files["reference"]),
        },
        "derived_geometry": {
            "geometry_closure": _record(geometry_files["geometry_closure"]),
            "repaired_glb": _record(geometry_files["repaired_glb"]),
            "repair_manifest": _record(geometry_files["repair_manifest"]),
            "independent_geometry_audit": _record(geometry_files["geometry_audit"]),
        },
        "evidence": {
            "clay_five_view": {
                "render_manifest": _record(clay_render),
                "views": {name: _record(path) for name, path in clay_views.items()},
                "contact_sheet": _record(clay_contact),
            },
            "pbr_five_view": {
                "render_manifest": _relative_record(pbr_render, review_root),
                "views": {
                    name: _relative_record(path, review_root)
                    for name, path in pbr_views.items()
                },
                "contact_sheet": _relative_record(pbr_contact, review_root),
                "execution_log": _relative_record(pbr_log, review_root),
            },
        },
        "producer": {
            "tool": _record(producer_tool),
            "renderer": _record(renderer),
        },
        "review_sha256": "",
    }
    review_path = review_root / "derived_static_review.json"

    def write_review() -> None:
        review["review_sha256"] = review_contract.hash_without(
            review, "review_sha256"
        )
        _write_json(review_path, review)

    write_review()
    monkeypatch.setattr(
        freezer.review_contract,
        "validate_review",
        lambda value: copy.deepcopy(value),
    )
    return {
        "review": review,
        "review_path": review_path,
        "write_review": write_review,
        "raw_decision": raw_decision,
        "raw_path": raw_path,
        "repaired": geometry_files["repaired_glb"],
    }


def _checks(**overrides) -> dict[str, bool]:
    result = {name: True for name in freezer.CHECK_FIELDS}
    result.update(overrides)
    return result


def _freeze(pending_review, output: Path, **overrides) -> Path:
    review_path = pending_review["review_path"]
    values = {
        "instance_id": "cat_fixture",
        "review_path": review_path,
        "expected_review_file_sha256": _sha256(review_path),
        "expected_internal_review_sha256": pending_review["review"]["review_sha256"],
        "decision": freezer.APPROVED,
        "checks": _checks(),
        "caveats": [],
        "notes": "The owner approved this exact repaired static review.",
        "user_explicit_decision": freezer.APPROVED,
        "user_explicit_review_file_sha256": _sha256(review_path),
        "output_path": output,
    }
    values.update(overrides)
    return freezer.freeze_decision(**values)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(payload: dict) -> None:
    payload["decision_sha256"] = review_contract.hash_without(
        payload, "decision_sha256"
    )


def test_freezes_exact_research_only_approval_and_preserves_raw_rejection(
    pending_review, tmp_path
):
    output = _freeze(pending_review, tmp_path / "human_decision.json")
    decision = freezer.validate_decision(_load(output))

    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert decision["schema"] == freezer.DECISION_SCHEMA
    assert decision["decision"] == freezer.APPROVED
    assert set(decision["checks"]) == freezer.CHECK_FIELDS
    assert all(decision["checks"].values())
    assert decision["attribute_evidence"] == {
        "body_build": "passed_static_visual",
        "coat_color": "passed_static_visual",
        "size": "deferred_to_metric_3d",
    }
    assert decision["review_binding"]["review_file"]["sha256"] == _sha256(
        pending_review["review_path"]
    )
    assert (
        decision["review_binding"]["internal_review_sha256"]
        == pending_review["review"]["review_sha256"]
    )
    assert decision["raw_static_rejection"]["decision"] == "rejected"
    assert decision["raw_static_rejection"]["preserved"] is True
    assert decision["raw_static_rejection"]["overwritten"] is False
    assert decision["authenticated_review_artifact_count"] == 27
    assert decision["effect"]["source_asset_v2_authorized"] is True
    assert decision["effect"]["rigging_authorized"] is True
    for name in (
        "animation_authorized",
        "ue_execution_authorized",
        "native_change_authorized",
        "formal_dataset_registration_authorized",
    ):
        assert decision["effect"][name] is False
    assert decision["formal_dataset_registration_authorized"] is False
    assert decision["state_classification"] == "research_candidate"


def test_validate_decision_rejects_rebound_raw_rejection_descriptor(
    pending_review,
    tmp_path,
):
    output = _freeze(pending_review, tmp_path / "valid.json")
    payload = _load(output)
    replacement = tmp_path / "replacement_rejection.json"
    replacement.write_bytes(Path(payload["raw_static_rejection"]["file"]["path"]).read_bytes())
    payload["raw_static_rejection"]["file"] = _record(replacement)
    _rehash(payload)

    with pytest.raises(
        contracts.ContractError,
        match="exactly bound to the frozen review",
    ):
        freezer.validate_decision(payload)


def test_requires_explicit_approval_bound_to_exact_external_review(
    pending_review, tmp_path
):
    with pytest.raises(contracts.ContractError, match="explicit approval"):
        _freeze(
            pending_review,
            tmp_path / "wrong_user_decision.json",
            user_explicit_decision="rejected",
        )
    with pytest.raises(contracts.ContractError, match="not bound"):
        _freeze(
            pending_review,
            tmp_path / "wrong_user_review.json",
            user_explicit_review_file_sha256="f" * 64,
        )


def test_rejects_approval_with_any_failed_or_missing_human_check(
    pending_review, tmp_path
):
    failed = _checks()
    failed["one_tail_without_duplicate_tail"] = False
    with pytest.raises(contracts.ContractError, match="all six human checks"):
        _freeze(
            pending_review,
            tmp_path / "failed_check.json",
            checks=failed,
        )

    missing = _checks()
    missing.pop("coat_and_pbr_appearance_coherent")
    with pytest.raises(contracts.ContractError, match="all six"):
        _freeze(
            pending_review,
            tmp_path / "missing_check.json",
            checks=missing,
        )


def test_reauthenticates_repaired_geometry_before_approval(
    pending_review, tmp_path
):
    pending_review["repaired"].write_bytes(b"rebound repaired geometry")

    with pytest.raises(contracts.ContractError, match="derived repaired GLB changed"):
        _freeze(pending_review, tmp_path / "rebound_geometry.json")


def test_semantically_preserves_original_raw_static_rejection(
    pending_review, tmp_path
):
    raw = pending_review["raw_decision"]
    raw.update(
        {
            "decision": freezer.APPROVED,
            "state_classification": "research_candidate",
            "next_gate": freezer.NEXT_GATE,
        }
    )
    raw["decision_sha256"] = review_contract.hash_without(raw, "decision_sha256")
    _write_json(pending_review["raw_path"], raw)
    authority = pending_review["review"]["source_authorities"]["raw_static_decision"]
    authority.update(
        {
            "decision": freezer.APPROVED,
            "decision_sha256": raw["decision_sha256"],
            "file": _record(pending_review["raw_path"]),
            "state_classification": "research_candidate",
        }
    )
    pending_review["write_review"]()

    with pytest.raises(contracts.ContractError, match="raw static rejection"):
        _freeze(pending_review, tmp_path / "upgraded_raw.json")


def test_atomic_no_replace_preserves_existing_output(pending_review, tmp_path):
    output = tmp_path / "existing.json"
    sentinel = b'{"keep":true}\n'
    output.write_bytes(sentinel)

    with pytest.raises(contracts.ContractError, match="refusing to replace"):
        _freeze(pending_review, output)

    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(".*existing.json.*.staging"))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["effect"].update({"native_change_authorized": True}),
        lambda value: value.update({"formal_dataset_registration_authorized": True}),
        lambda value: value["effect"].update({"animation_authorized": True}),
        lambda value: value["attribute_evidence"].update(
            {"size": "passed_static_visual"}
        ),
    ),
)
def test_validate_decision_rejects_authority_upgrade(
    pending_review, tmp_path, mutate
):
    output = _freeze(pending_review, tmp_path / "valid.json")
    payload = _load(output)
    mutate(payload)
    _rehash(payload)

    with pytest.raises(contracts.ContractError):
        freezer.validate_decision(payload)


def test_cli_requires_exact_user_binding_and_all_six_checks():
    parser = freezer.build_argument_parser()
    argv = [
        "--instance-id",
        "cat_fixture",
        "--review",
        "review.json",
        "--expected-review-file-sha256",
        "a" * 64,
        "--expected-internal-review-sha256",
        "b" * 64,
        "--decision",
        freezer.APPROVED,
        "--user-explicit-decision",
        freezer.APPROVED,
        "--user-explicit-review-file-sha256",
        "a" * 64,
        "--notes",
        "Explicit approval.",
        "--output",
        "decision.json",
    ]
    for name in sorted(freezer.CHECK_FIELDS):
        argv.extend((f"--{name.replace('_', '-')}", "True"))

    parsed = parser.parse_args(argv)
    assert all(getattr(parsed, name) is True for name in freezer.CHECK_FIELDS)

    missing_one = argv[:-2]
    with pytest.raises(SystemExit):
        parser.parse_args(missing_one)
