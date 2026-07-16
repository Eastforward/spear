from __future__ import annotations

from pathlib import Path

import pytest

from tools import blender_retarget_fastlane_v1 as fastlane


def _metrics(actual: float = 56.7, allowed: float = 1.35):
    return {
        "action_name": "Walking",
        "deformation": {
            "maximum_skinned_edge_stretch_ratio": actual,
            "allowed_maximum_skinned_edge_stretch_ratio": allowed,
        },
    }


def test_edge_stretch_is_recorded_advisory_while_strict_delegate_sees_only_its_cap():
    seen = []

    def strict_delegate(metrics, *, semantic_mapping):
        seen.append((metrics, semantic_mapping))
        assert metrics["deformation"]["maximum_skinned_edge_stretch_ratio"] == 1.35
        return {"status": "passed", "action_name": "Walking"}

    source = _metrics()
    result = fastlane.validate_fastlane_action_metrics(
        source,
        semantic_mapping={"fixture": True},
        strict_validator=strict_delegate,
    )

    assert source["deformation"]["maximum_skinned_edge_stretch_ratio"] == 56.7
    assert result["status"] == "fastlane_hard_gates_passed_pending_visual_qa"
    assert result["edge_stretch_advisory"] == {
        "policy": "recorded_numeric_advisory_requires_visual_tear_review_v1",
        "actual_maximum_ratio": 56.7,
        "strict_formal_limit_ratio": 1.35,
        "strict_formal_limit_exceeded": True,
        "formal_registration_authorized": False,
    }
    assert len(seen) == 1


def test_every_non_edge_strict_failure_still_fails_closed():
    def reject_grounding(metrics, *, semantic_mapping):
        raise fastlane.runner.RetargetError("penetration exceeds 0.010 m")

    with pytest.raises(fastlane.runner.RetargetError, match="penetration"):
        fastlane.validate_fastlane_action_metrics(
            _metrics(),
            semantic_mapping={},
            strict_validator=reject_grounding,
        )


def test_manifest_is_explicit_research_fastlane_and_preserves_numeric_evidence():
    payload = {
        "schema": fastlane.runner.MANIFEST_SCHEMA,
        "state_classification": "research_candidate",
        "automatic_checks": "passed",
        "actions": {},
    }
    metrics = {
        "actions": {
            "Walking": _metrics(56.7),
            "Standing_Idle": {
                "action_name": "Standing_Idle",
                "deformation": {
                    "maximum_skinned_edge_stretch_ratio": 9.5,
                    "allowed_maximum_skinned_edge_stretch_ratio": 1.35,
                },
            },
        }
    }

    strict_failure = {
        "path": "/strict/retarget_failure.json",
        "sha256": "a" * 64,
        "size_bytes": 123,
    }
    decorated = fastlane.decorate_fastlane_manifest(
        payload, metrics, strict_failure=strict_failure
    )

    assert decorated is not payload
    assert decorated["execution_track"] == "research_candidate_fastlane"
    assert decorated["formal_dataset_asset"] is False
    assert decorated["automatic_check_scope"] == "fastlane_hard_gates_only"
    assert decorated["strict_formal_registration_status"] == (
        "blocked_by_recorded_edge_stretch"
    )
    assert decorated["edge_stretch_advisories"]["Walking"]["actual_maximum_ratio"] == 56.7
    assert decorated["edge_stretch_advisories"]["Standing_Idle"]["actual_maximum_ratio"] == 9.5
    assert Path(decorated["fastlane_wrapper"]["path"]).name == (
        "blender_retarget_fastlane_v1.py"
    )
    assert decorated["strict_failure_evidence"] == strict_failure


def test_cli_reuses_runner_arguments_and_has_no_formal_output_alias():
    args = fastlane.parse_args(
        [
            "--asset-id",
            "person_01",
            "--base-avatar-id",
            "rocketbox_male_adult_01",
            "--bind-pose-glb",
            "/static/bind_pose.glb",
            "--static-qa-json",
            "/static/static_qa.json",
            "--baseline-retarget-blend",
            "/baseline/retarget.blend",
            "--baseline-retarget-manifest",
            "/baseline/retarget_manifest.json",
            "--idle-motion-fbx",
            "/idle.fbx",
            "--motion-basis-selection",
            "/selection.json",
            "--motion-basis-review-manifest",
            "/review.json",
            "--output-dir",
            "/output/retarget_fastlane_v1",
            "--strict-failure-evidence",
            "/strict/retarget_failure.json",
        ]
    )
    assert args.output_dir == Path("/output/retarget_fastlane_v1")
    assert "formal" not in args.output_dir.name
    assert args.strict_failure_evidence == Path("/strict/retarget_failure.json")
