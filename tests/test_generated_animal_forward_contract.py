"""Unit tests for the single-point forward declaration contract."""

from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generated_animal_forward_contract import (
    CANONICAL_TARGET_FRONT_AXIS,
    CAT_MOTION_DONOR_ID,
    DOG_MOTION_DONOR_ID,
    ForwardContractError,
    MOTION_DONOR_ARTIFACTS,
    assert_declared_motion_basis,
    assert_declared_motion_donor_artifact,
    build_forward_declaration,
    expected_motion_basis,
    load_forward_declaration,
    motion_donor_for_species,
    sha256_file,
)


DONOR = DOG_MOTION_DONOR_ID
SPEAR_ROOT = Path(__file__).resolve().parents[1]
CAT_GLB = (
    SPEAR_ROOT.parents[1]
    / "assets/mesh_library/quaternius_animalpack/Cat.glb"
)


def write_motion_glb(path: Path, actions=("Idle", "Walking")) -> Path:
    document = {
        "asset": {"version": "2.0"},
        "animations": [{"name": name} for name in actions],
    }
    return write_raw_motion_glb(
        path, json.dumps(document, separators=(",", ":")).encode("utf-8")
    )


def write_raw_motion_glb(path: Path, encoded: bytes) -> Path:
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    total_size = 12 + 8 + len(encoded)
    payload = (
        struct.pack("<4sII", b"glTF", 2, total_size)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
    )
    path.write_bytes(payload)
    return path


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


def test_forward_declaration_rejects_duplicate_keys_before_self_hash(workspace):
    tmp_path, glb, evidence = workspace
    declaration = build_forward_declaration(
        asset_workspace="test_asset_v1",
        input_glb=glb,
        reviewed_source_front_yaw_deg=0.0,
        head_end_decision_source="human_review",
        head_end_evidence=evidence,
        motion_donor_tag=DONOR,
    )
    encoded = json.dumps(declaration, separators=(",", ":"))
    path = tmp_path / "duplicate_forward_declaration.json"
    path.write_text('{"schema":"shadow",' + encoded[1:], encoding="utf-8")

    with pytest.raises(ForwardContractError, match="duplicate JSON object key"):
        load_forward_declaration(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_forward_declaration_rejects_non_finite_json_numbers(
    workspace, constant
):
    tmp_path, glb, evidence = workspace
    declaration = build_forward_declaration(
        asset_workspace="test_asset_v1",
        input_glb=glb,
        reviewed_source_front_yaw_deg=0.0,
        head_end_decision_source="human_review",
        head_end_evidence=evidence,
        motion_donor_tag=DONOR,
    )
    encoded = json.dumps(declaration, separators=(",", ":"))
    encoded = encoded.replace(
        '"reviewed_source_front_yaw_deg":0.0',
        f'"reviewed_source_front_yaw_deg":{constant}',
        1,
    )
    path = tmp_path / f"non_finite_{constant}.json"
    path.write_text(encoded, encoding="utf-8")

    with pytest.raises(ForwardContractError, match="non-finite JSON number"):
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
    write_motion_glb(donor)
    monkeypatch.setitem(
        MOTION_DONOR_ARTIFACTS,
        DONOR,
        {
            "sha256": sha256_file(donor),
            "size_bytes": donor.stat().st_size,
        },
    )
    assert_declared_motion_donor_artifact(DONOR, donor)
    changed = bytearray(donor.read_bytes())
    changed[-1] ^= 1
    donor.write_bytes(changed)
    with pytest.raises(ForwardContractError, match="does not match"):
        assert_declared_motion_donor_artifact(DONOR, donor)


@pytest.mark.parametrize(
    ("document", "expected_error"),
    [
        (
            b'{"asset":{"version":"2.0"},"animations":[],"animations":[]}',
            "duplicate JSON object key",
        ),
        (
            b'{"asset":{"version":"2.0","score":NaN},"animations":[]}',
            "non-finite JSON number",
        ),
    ],
)
def test_motion_donor_glb_json_chunk_uses_strict_parser(
    tmp_path, monkeypatch, document, expected_error
):
    donor = write_raw_motion_glb(tmp_path / "Dog.glb", document)
    monkeypatch.setitem(
        MOTION_DONOR_ARTIFACTS,
        DONOR,
        {
            "sha256": sha256_file(donor),
            "size_bytes": donor.stat().st_size,
        },
    )

    with pytest.raises(
        ForwardContractError, match="invalid GLB JSON chunk"
    ) as caught:
        assert_declared_motion_donor_artifact(DONOR, donor)
    assert expected_error in str(caught.value.__cause__)


def test_motion_donor_external_hash_is_checked_before_glb_json_parse(tmp_path):
    donor = write_raw_motion_glb(
        tmp_path / "Dog.glb",
        b'{"asset":{"version":"2.0"},"animations":[],"animations":[]}',
    )

    with pytest.raises(ForwardContractError, match="does not match") as caught:
        assert_declared_motion_donor_artifact(DONOR, donor)
    assert caught.value.__cause__ is None


def test_exact_cat_donor_identity_and_idle_walking_semantics():
    expected = MOTION_DONOR_ARTIFACTS[CAT_MOTION_DONOR_ID]
    assert expected == {
        "sha256": (
            "af2afb5e92c6d9daae98a918f8bd2bcb13ea4d7cfb880020d0d263e4d2f1277e"
        ),
        "size_bytes": 163516,
    }
    assert CAT_GLB.stat().st_size == expected["size_bytes"]
    assert sha256_file(CAT_GLB) == expected["sha256"]

    readback = assert_declared_motion_donor_artifact(
        CAT_MOTION_DONOR_ID, CAT_GLB
    )
    assert readback["animation_names"] == ["Idle", "Walking"]
    assert readback["authority_contract"] == {
        "target_species": "cat",
        "donor_role": "skeleton_semantics_and_animation_only",
        "required_actions": ["Walking", "Idle"],
        "animation_channels_used": ["translation", "rotation", "scale"],
        "geometry_used": False,
        "weights_used": False,
        "target_mesh_authority": "generated_target_glb",
    }


def test_cat_donor_rejects_hash_and_size_tampering(tmp_path):
    donor = tmp_path / "Cat.glb"
    shutil.copyfile(CAT_GLB, donor)
    assert_declared_motion_donor_artifact(CAT_MOTION_DONOR_ID, donor)

    changed = bytearray(donor.read_bytes())
    changed[-1] ^= 1
    donor.write_bytes(changed)
    with pytest.raises(ForwardContractError, match="does not match"):
        assert_declared_motion_donor_artifact(CAT_MOTION_DONOR_ID, donor)

    shutil.copyfile(CAT_GLB, donor)
    with donor.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ForwardContractError, match="does not match"):
        assert_declared_motion_donor_artifact(CAT_MOTION_DONOR_ID, donor)


def test_cat_donor_requires_both_idle_and_walking_actions(tmp_path, monkeypatch):
    donor = write_motion_glb(tmp_path / "Cat.glb", actions=("Idle",))
    monkeypatch.setitem(
        MOTION_DONOR_ARTIFACTS,
        CAT_MOTION_DONOR_ID,
        {
            "sha256": sha256_file(donor),
            "size_bytes": donor.stat().st_size,
        },
    )
    with pytest.raises(ForwardContractError, match="action semantics"):
        assert_declared_motion_donor_artifact(CAT_MOTION_DONOR_ID, donor)


def test_species_selects_a_distinct_dog_or_cat_motion_donor():
    assert motion_donor_for_species("dog") == DOG_MOTION_DONOR_ID
    assert motion_donor_for_species("cat") == CAT_MOTION_DONOR_ID
    with pytest.raises(ForwardContractError, match="unsupported"):
        motion_donor_for_species("ferret")


def test_cat_selection_preserves_generated_target_mesh_provenance(workspace):
    _tmp_path, glb, evidence = workspace
    declarations = {
        species: build_forward_declaration(
            asset_workspace=f"test_{species}_v1",
            input_glb=glb,
            reviewed_source_front_yaw_deg=0.0,
            head_end_decision_source="human_review",
            head_end_evidence=evidence,
            target_species=species,
        )
        for species in ("dog", "cat")
    }

    assert declarations["dog"]["motion_donor_tag"] == DOG_MOTION_DONOR_ID
    assert declarations["cat"]["motion_donor_tag"] == CAT_MOTION_DONOR_ID
    assert declarations["dog"]["input_glb"] == declarations["cat"]["input_glb"]
    for declaration in declarations.values():
        contract = declaration["expected_motion_donor_contract"]
        assert contract["target_mesh_authority"] == "generated_target_glb"
        assert contract["geometry_used"] is False
        assert contract["weights_used"] is False

    with pytest.raises(ForwardContractError, match="does not match"):
        build_forward_declaration(
            asset_workspace="mismatched_cat_v1",
            input_glb=glb,
            reviewed_source_front_yaw_deg=0.0,
            head_end_decision_source="human_review",
            head_end_evidence=evidence,
            target_species="cat",
            motion_donor_tag=DOG_MOTION_DONOR_ID,
        )
