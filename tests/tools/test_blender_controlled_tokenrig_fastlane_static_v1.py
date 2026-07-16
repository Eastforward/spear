from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_controlled_tokenrig_fastlane_static_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "blender_controlled_tokenrig_fastlane_static_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


CHAINS = {
    "left_arm": ["la"],
    "left_leg": ["ll"],
    "right_arm": ["ra"],
    "right_leg": ["rl"],
}


def test_obvious_bilateral_gate_ignores_microscopic_but_blocks_large_cross_weights():
    positions = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)] * 100
    acceptable = [{"ra": 0.99995, "la": 0.00005}, {"la": 0.99995, "ra": 0.00005}] * 100
    metrics = runner.obvious_bilateral_metrics(positions, acceptable, CHAINS)
    assert metrics["passed"] is True
    assert metrics["advisory_over_1e4_count"] == 0

    rejected = [{"ra": 0.8, "la": 0.2}, {"la": 0.8, "ra": 0.2}] * 100
    with pytest.raises(runner.FastlaneError, match="obvious bilateral"):
        runner.obvious_bilateral_metrics(positions, rejected, CHAINS)


def test_fastlane_preserves_strict_rejections_and_keeps_animation_blocked():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"strict_topology_status": "advisory_not_erased"' in source
    assert '"state_classification": "research_candidate_fastlane"' in source
    assert '"animation_authorized": False' in source
    assert '"formal_dataset_registration_authorized": False' in source
    assert "EXPECTED_BONE_COUNT = 52" in source
    assert "compare_pbr_payloads" in source


def test_fastlane_requires_independent_output_directory_and_bind_media():
    assert runner.OUTPUT_DIRNAME == "fastlane_static_v1"
    assert set(runner.ARTIFACT_NAMES) >= {
        "bind_pose.glb",
        "bind_front.png",
        "bind_back.png",
        "bind_side.png",
        "bind_top.png",
        "skeleton_overlay.png",
        "weights_contact.png",
        "texture_compare.png",
        "joint_hierarchy.txt",
    }
