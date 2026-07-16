from __future__ import annotations

import copy

import pytest

from tools import blender_render_tokenrig_fastlane_review_v1 as fast_review


def _manifest():
    action = {
        "status": "fastlane_hard_gates_passed_pending_visual_qa",
        "action_name": "Walking",
    }
    advisory = {
        "policy": fast_review.EDGE_POLICY,
        "actual_maximum_ratio": 56.7,
        "strict_formal_limit_ratio": 1.35,
        "strict_formal_limit_exceeded": True,
        "formal_registration_authorized": False,
    }
    return {
        "schema": fast_review.renderer.RETARGET_SCHEMA,
        "execution_track": fast_review.TRACK,
        "formal_dataset_asset": False,
        "automatic_checks": "passed",
        "automatic_check_scope": "fastlane_hard_gates_only",
        "strict_formal_registration_status": "blocked_by_recorded_edge_stretch",
        "strict_failure_evidence": {
            "path": "/strict/retarget_failure.json",
            "sha256": "a" * 64,
            "size_bytes": 123,
        },
        "edge_stretch_advisories": {
            "Walking": advisory,
            "Standing_Idle": {**advisory, "actual_maximum_ratio": 9.5},
        },
        "actions": {
            "Walking": action,
            "Standing_Idle": {**action, "action_name": "Standing_Idle"},
        },
    }


def test_projection_changes_only_in_memory_action_status_for_strict_renderer():
    source = _manifest()
    original = copy.deepcopy(source)

    projected = fast_review.project_fastlane_manifest_for_strict_renderer(source)

    assert source == original
    assert projected["actions"]["Walking"]["status"] == "passed"
    assert projected["actions"]["Standing_Idle"]["status"] == "passed"
    assert projected["execution_track"] == fast_review.TRACK
    assert projected["formal_dataset_asset"] is False
    assert projected["edge_stretch_advisories"] == source["edge_stretch_advisories"]


def test_projection_rejects_missing_fastlane_or_formal_claim():
    payload = _manifest()
    payload["execution_track"] = "formal"
    with pytest.raises(fast_review.FastlaneReviewError, match="track"):
        fast_review.project_fastlane_manifest_for_strict_renderer(payload)

    payload = _manifest()
    payload["formal_dataset_asset"] = True
    with pytest.raises(fast_review.FastlaneReviewError, match="formal"):
        fast_review.project_fastlane_manifest_for_strict_renderer(payload)


def test_review_manifest_decoration_keeps_fastlane_nonformal():
    decorated = fast_review.decorate_review_manifest(
        {"schema": fast_review.renderer.REVIEW_MANIFEST_SCHEMA},
        source_manifest={
            "sha256": "b" * 64,
            "size_bytes": 456,
            "path": "/retarget/retarget_manifest.json",
        },
    )
    assert decorated["execution_track"] == fast_review.TRACK
    assert decorated["formal_dataset_asset"] is False
    assert decorated["fastlane_visual_gate_status"] == "rendered_pending_agent_visual_qa"
    assert decorated["source_fastlane_retarget_manifest"]["sha256"] == "b" * 64
