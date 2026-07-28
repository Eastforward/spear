import json
from pathlib import Path

import pytest

from tools import build_generated_animal_forward_declaration as subject
from tools.generated_animal_forward_contract import (
    CAT_MOTION_DONOR_ID,
    DOG_MOTION_DONOR_ID,
)


def _base_args(tmp_path: Path) -> list[str]:
    source = tmp_path / "generated.glb"
    source.write_bytes(b"generated target fixture")
    evidence = tmp_path / "head_end_review.json"
    evidence.write_text('{"status":"reviewed"}\n', encoding="utf-8")
    return [
        "--asset-workspace",
        "cat_fixture_v1",
        "--input-glb",
        str(source),
        "--confirmed-front-yaw-deg",
        "0",
        "--head-end-evidence",
        str(evidence),
        "--head-end-decision-source",
        "human_review",
        "--output",
        str(tmp_path / "forward_declaration.json"),
    ]


def test_cli_selects_the_registered_cat_motion_donor(tmp_path: Path) -> None:
    args = [*_base_args(tmp_path), "--target-species", "cat"]

    assert subject.main(args) == 0

    declaration = json.loads(
        (tmp_path / "forward_declaration.json").read_text(encoding="utf-8")
    )
    assert declaration["target_species"] == "cat"
    assert declaration["motion_donor_tag"] == CAT_MOTION_DONOR_ID
    assert declaration["expected_motion_donor_contract"][
        "target_mesh_authority"
    ] == "generated_target_glb"


def test_cli_rejects_a_dog_donor_for_a_cat_target(tmp_path: Path) -> None:
    args = [
        *_base_args(tmp_path),
        "--target-species",
        "cat",
        "--motion-donor-tag",
        DOG_MOTION_DONOR_ID,
    ]

    with pytest.raises(SystemExit, match="does not match"):
        subject.main(args)


def test_cli_default_remains_the_registered_dog_donor(tmp_path: Path) -> None:
    args = _base_args(tmp_path)

    assert subject.main(args) == 0

    declaration = json.loads(
        (tmp_path / "forward_declaration.json").read_text(encoding="utf-8")
    )
    assert declaration["target_species"] == "dog"
    assert declaration["motion_donor_tag"] == DOG_MOTION_DONOR_ID


def test_cli_accepts_matching_physical_generated_animal_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spear_root = tmp_path / "SPEAR"
    workspace = (
        spear_root
        / "tmp/new_animal_assets"
        / "physical_generated_animal_workspace"
    )
    workspace.mkdir(parents=True)
    input_glb = workspace / "tokenrig" / "tokenrig_native.glb"
    input_glb.parent.mkdir()
    input_glb.write_bytes(b"production generated target fixture")
    monkeypatch.setattr(subject, "SPEAR_ROOT", spear_root)
    args = _base_args(tmp_path)
    args[args.index("--asset-workspace") + 1] = workspace.name
    args[args.index("--input-glb") + 1] = str(input_glb)

    assert subject.main(args) == 0

    declaration = json.loads(
        (tmp_path / "forward_declaration.json").read_text(encoding="utf-8")
    )
    assert declaration["asset_workspace"] == workspace.name


def test_cli_rejects_source_asset_id_in_place_of_physical_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spear_root = tmp_path / "SPEAR"
    workspace = (
        spear_root
        / "tmp/new_animal_assets"
        / "physical_generated_animal_workspace"
    )
    workspace.mkdir(parents=True)
    input_glb = workspace / "tokenrig" / "tokenrig_native.glb"
    input_glb.parent.mkdir()
    input_glb.write_bytes(b"production generated target fixture")
    monkeypatch.setattr(subject, "SPEAR_ROOT", spear_root)
    args = _base_args(tmp_path)
    args[args.index("--asset-workspace") + 1] = "dog_registry_identity_v1"
    args[args.index("--input-glb") + 1] = str(input_glb)
    output = tmp_path / "forward_declaration.json"

    with pytest.raises(SystemExit, match="physical generated-animal workspace"):
        subject.main(args)

    assert not output.exists()
