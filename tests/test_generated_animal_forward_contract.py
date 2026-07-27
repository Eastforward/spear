"""Unit tests for the single-point forward declaration contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generated_animal_forward_contract import (
    CANONICAL_TARGET_FRONT_AXIS,
    ForwardContractError,
    MOTION_DONOR_ARTIFACTS,
    assert_declared_motion_basis,
    assert_declared_motion_donor_artifact,
    build_forward_declaration,
    expected_motion_basis,
    load_forward_declaration,
    sha256_file,
)


DONOR = "quaternius_universal_quadruped_v1"


@pytest.fixture
def workspace(tmp_path):
    glb = tmp_path / "rig.glb"
    glb.write_bytes(b"glTF-fake-bytes")
    evidence = tmp_path / "head_end_review.json"
    evidence.write_text(json.dumps({"reviewed_by": "owner"}), encoding="utf-8")
    return tmp_path, glb, evidence


def write_declaration(tmp_path, declaration):
    path = tmp_path / "forward_declaration.json"
    path.write_text(json.dumps(declaration, indent=2), encoding="utf-8")
    return path


def test_build_and_load_roundtrip(workspace):
    tmp_path, glb, evidence = workspace
    declaration = build_forward_declaration(
        asset_workspace="test_asset_v1",
        input_glb=glb,
        reviewed_source_front_yaw_deg=49.325,
        head_end_decision_source="human_confirming_estimator",
        head_end_evidence=evidence,
        motion_donor_tag=DONOR,
        estimate={"estimated_front_yaw_deg": 47.1},
    )
    path = write_declaration(tmp_path, declaration)
    loaded = load_forward_declaration(path)
    assert loaded["reviewed_source_front_yaw_deg"] == pytest.approx(49.325)
    assert loaded["target_front_axis"] == CANONICAL_TARGET_FRONT_AXIS
    assert loaded["expected_motion_basis"] == expected_motion_basis(DONOR)


def test_tampered_declaration_fails_authentication(workspace):
    tmp_path, glb, evidence = workspace
    declaration = build_forward_declaration(
        asset_workspace="test_asset_v1",
        input_glb=glb,
        reviewed_source_front_yaw_deg=0.0,
        head_end_decision_source="human_review",
        head_end_evidence=evidence,
        motion_donor_tag=DONOR,
    )
    declaration["reviewed_source_front_yaw_deg"] = 180.0
    path = write_declaration(tmp_path, declaration)
    with pytest.raises(ForwardContractError, match="self-hash"):
        load_forward_declaration(path)


def test_missing_original_head_end_evidence_fails_closed(workspace):
    tmp_path, glb, evidence = workspace
    declaration = build_forward_declaration(
        asset_workspace="test_asset_v1",
        input_glb=glb,
        reviewed_source_front_yaw_deg=0.0,
        head_end_decision_source="human_review",
        head_end_evidence=evidence,
        motion_donor_tag=DONOR,
    )
    path = write_declaration(tmp_path, declaration)
    evidence.unlink()
    with pytest.raises(ForwardContractError, match="evidence is missing"):
        load_forward_declaration(path)


def test_non_canonical_target_axis_is_rejected(workspace):
    tmp_path, glb, evidence = workspace
    declaration = build_forward_declaration(
        asset_workspace="test_asset_v1",
        input_glb=glb,
        reviewed_source_front_yaw_deg=0.0,
        head_end_decision_source="human_review",
        head_end_evidence=evidence,
        motion_donor_tag=DONOR,
    )
    declaration["target_front_axis"] = "negative-x"
    declaration["declaration_sha256"] = None
    path = write_declaration(tmp_path, declaration)
    with pytest.raises(ForwardContractError):
        load_forward_declaration(path)


def test_unknown_donor_fails_at_build_time(workspace):
    _tmp_path, glb, evidence = workspace
    with pytest.raises(ForwardContractError, match="unknown motion donor"):
        build_forward_declaration(
            asset_workspace="test_asset_v1",
            input_glb=glb,
            reviewed_source_front_yaw_deg=0.0,
            head_end_decision_source="human_review",
            head_end_evidence=evidence,
            motion_donor_tag="unregistered_donor_v1",
        )


def test_motion_basis_deviation_is_a_contract_error():
    assert_declared_motion_basis(DONOR, 0, "matched")
    with pytest.raises(ForwardContractError, match="fix the"):
        assert_declared_motion_basis(DONOR, 180, "matched")
    with pytest.raises(ForwardContractError):
        assert_declared_motion_basis(DONOR, 0, "swapped")


def test_motion_donor_tag_authenticates_the_supplied_bytes(tmp_path, monkeypatch):
    donor = tmp_path / "Dog.glb"
    donor.write_bytes(b"approved-motion-donor")
    monkeypatch.setitem(
        MOTION_DONOR_ARTIFACTS,
        DONOR,
        {
            "sha256": sha256_file(donor),
            "size_bytes": donor.stat().st_size,
        },
    )
    assert_declared_motion_donor_artifact(DONOR, donor)
    donor.write_bytes(b"different-motion-donor")
    with pytest.raises(ForwardContractError, match="does not match"):
        assert_declared_motion_donor_artifact(DONOR, donor)
