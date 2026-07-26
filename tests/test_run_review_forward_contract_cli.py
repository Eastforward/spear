"""CLI-contract tests for the post-TokenRig review runner forward modes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generated_animal_forward_contract import build_forward_declaration
from tools.run_target_native_generated_quadruped_review import main


@pytest.fixture
def workspace(tmp_path):
    rig = tmp_path / "rig.glb"
    rig.write_bytes(b"glTF-rig")
    motion = tmp_path / "motion.glb"
    motion.write_bytes(b"glTF-motion")
    evidence = tmp_path / "head_end_review.json"
    evidence.write_text("{}", encoding="utf-8")
    blender = tmp_path / "blender"
    blender.write_text("#!/bin/sh\n", encoding="utf-8")
    declaration = build_forward_declaration(
        asset_workspace="cli_test_asset_v1",
        input_glb=rig,
        reviewed_source_front_yaw_deg=49.325,
        head_end_decision_source="human_review",
        head_end_evidence=evidence,
        motion_donor_tag="quaternius_universal_quadruped_v1",
    )
    declaration_path = tmp_path / "forward_declaration.json"
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    return {
        "rig": rig,
        "motion": motion,
        "evidence": evidence,
        "blender": blender,
        "declaration": declaration_path,
        "output_root": tmp_path / "out",
    }


def base_argv(workspace):
    return [
        "--target-rig-glb", str(workspace["rig"]),
        "--source-motion-glb", str(workspace["motion"]),
        "--output-root", str(workspace["output_root"]),
        "--blender", str(workspace["blender"]),
        "--validate-only",
    ]


def test_declaration_mode_validates_and_plans_gait_stage(workspace, capsys):
    argv = base_argv(workspace) + [
        "--forward-declaration", str(workspace["declaration"]),
    ]
    assert main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    labels = [label for label, _command in plan["commands"]]
    assert "gait_direction" in labels
    assert labels.index("retarget") < labels.index("gait_direction")
    retarget = dict(plan["commands"])["retarget"]
    assert "--motion-basis-yaw-deg" in retarget
    assert retarget[retarget.index("--motion-basis-yaw-deg") + 1] == "0"
    assert retarget[retarget.index("--side-chain-mode") + 1] == "matched"


def test_declaration_mode_forbids_legacy_flags(workspace):
    argv = base_argv(workspace) + [
        "--forward-declaration", str(workspace["declaration"]),
        "--motion-basis-yaw-deg", "180",
        "--side-chain-mode", "matched",
    ]
    with pytest.raises(ValueError, match="declared exactly once"):
        main(argv)


def test_legacy_mode_requires_all_free_parameters(workspace):
    argv = base_argv(workspace) + [
        "--heading-review-evidence", str(workspace["evidence"]),
        "--reviewed-source-front-yaw-deg", "49.325",
    ]
    with pytest.raises(ValueError, match="legacy mode requires"):
        main(argv)


def test_preview_only_truncates_after_cheap_walking_renders(workspace, capsys):
    argv = base_argv(workspace) + [
        "--forward-declaration", str(workspace["declaration"]),
        "--preview-only",
    ]
    assert main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    labels = [label for label, _command in plan["commands"]]
    assert labels == [
        "heading", "rig_audit", "support_plane", "retarget", "gait_direction",
        "render_walking_side", "encode_walking_side",
        "render_walking_front", "encode_walking_front",
    ]
    assert "deformation" not in labels
