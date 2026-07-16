from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.spike_rlr import human_attribute_review as review


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def candidate_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "short_sleeve_color"
    root.mkdir()
    image_names = {
        "source.png": "RGB",
        "raw_candidate.png": "RGB",
        "candidate.png": "RGB",
        "source_alpha.png": "L",
        "candidate_alpha.png": "L",
        "candidate_rgba.png": "RGBA",
        "edit_core.png": "L",
        "transition_band.png": "L",
        "protected_guard.png": "L",
        "overlay.png": "RGB",
        "diff.png": "RGB",
    }
    for filename, mode in image_names.items():
        if mode == "RGBA":
            image = Image.new(mode, (16, 20), (20, 30, 40, 0))
            for x in range(4, 12):
                for y in range(2, 19):
                    image.putpixel((x, y), (90, 100, 110, 255))
        elif mode == "L":
            image = Image.new(mode, (16, 20), 0)
            for x in range(4, 12):
                for y in range(2, 19):
                    image.putpixel((x, y), 255)
        else:
            image = Image.new(mode, (16, 20), (40, 50, 60))
        image.save(root / filename)
    pending = root / "agent_2d_decision.json"
    pending.write_text(
        json.dumps(
            {
                "schema": "human_attribute_agent_2d_decision_v1",
                "case_id": "short_sleeve_color",
                "status": "pending_agent_2d_visual_qa",
                "reviewer_kind": "agent",
                "user_acceptance": "pending_user_review",
            }
        ),
        encoding="utf-8",
    )
    (root / "generation_attempt.json").write_text(
        json.dumps(
            {
                "schema": "flux2_human_attribute_generation_attempt_v1",
                "attempt_id": "fixture_attempt_001",
                "status": "succeeded",
            }
        ),
        encoding="utf-8",
    )
    artifacts = {
        path.name: {
            "path": str(path),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.iterdir())
    }
    base_id = "rocketbox_male_adult_01"
    qualified_pointer = tmp_path / base_id / "qualified_candidate_v1.json"
    qualified_pointer.parent.mkdir()
    qualified_pointer.write_text("{}", encoding="utf-8")
    qualified_pointer.chmod(0o444)
    final_branch = {
        "branch_id": "sanitized_weights",
        "path": str(qualified_pointer.parent / "fitted_skeleton_v1/sanitized_weights_v1"),
        "relative_root": "fitted_skeleton_v1/sanitized_weights_v1",
    }
    review_dir = f"{final_branch['path']}/dynamic_review_v1"
    monkeypatch.setattr(
        review.qualified_candidate,
        "validate_qualified_candidate",
        lambda path: {
            "asset_id": base_id,
            "base_avatar_id": base_id,
            "status": review.PASS_STATUS,
            "final_branch": final_branch,
            "dynamic": {"review_dir": review_dir},
        },
    )
    manifest = {
        "schema": "flux2_human_attribute_candidate_v2",
        "case_id": "short_sleeve_color",
        "base_asset_id": "rocketbox_male_adult_01",
        "downstream_asset_id": "route2_short_sleeve_color_v1",
        "base_route2_qualification": {
            "asset_id": base_id,
            "status": "agent_qa_passed_pending_user_acceptance",
            "qualified_candidate": {
                "path": str(qualified_pointer),
                "sha256": _sha(qualified_pointer),
                "size_bytes": qualified_pointer.stat().st_size,
            },
            "final_branch": final_branch,
            "review_dir": review_dir,
        },
        "state_classification": "research_candidate",
        "bundle_status": "generated_pending_agent_2d_visual_qa",
        "agent_qa_status": "pending_agent_2d_visual_qa",
        "user_acceptance": "pending_user_review",
        "quantitative_snapshot": {
            "automatic_checks": "passed",
            "pixel_proof": {
                "outside_changed_pixels": 0,
                "outside_max_abs_channel_delta": 0,
                "transition_is_feathered": True,
            },
            "alpha_proof": {"outside_changed_pixels": 0},
            "case_metrics": {"passed": True, "checks": {"synthetic": True}},
        },
        "artifacts": artifacts,
    }
    (root / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    for path in root.iterdir():
        path.chmod(0o444)
    return root


def _passing_checks() -> dict[str, bool]:
    return {name: True for name in review.AGENT_2D_VISUAL_CHECKS}


def test_agent_2d_pass_is_exclusive_full_snapshot_and_never_user_approved(candidate_bundle):
    decision = review.record_agent_2d_visual_qa(
        candidate_bundle,
        status="agent_qa_passed_pending_user_acceptance",
        reviewer="codex-route2-attribute-qa",
        notes="Inspected source, candidate, masks, diff, and RGBA at full resolution.",
        checks=_passing_checks(),
    )

    payload = json.loads(decision.read_text())
    assert decision == candidate_bundle.with_name(
        "short_sleeve_color.agent_2d_visual_qa.json"
    )
    assert decision.stat().st_mode & 0o777 == 0o444
    assert payload["status"] == "agent_qa_passed_pending_user_acceptance"
    assert payload["snapshot"]["candidate_manifest_sha256"] == _sha(
        candidate_bundle / "candidate_manifest.json"
    )
    assert set(payload["snapshot"]["artifact_sha256"]) == review.CANDIDATE_ARTIFACTS
    assert "user_approved" not in decision.read_text()
    assert review.assert_agent_2d_qa_passed(candidate_bundle)["status"] == payload["status"]
    with pytest.raises(review.AttributeReviewError, match="already exists"):
        review.record_agent_2d_visual_qa(
            candidate_bundle,
            status="rejected",
            reviewer="codex",
            notes="cannot replace",
            checks=_passing_checks(),
        )


def test_agent_2d_decision_is_invalidated_by_any_artifact_change(candidate_bundle):
    review.record_agent_2d_visual_qa(
        candidate_bundle,
        status="agent_qa_passed_pending_user_acceptance",
        reviewer="codex",
        notes="all checks pass",
        checks=_passing_checks(),
    )
    (candidate_bundle / "diff.png").chmod(0o644)
    (candidate_bundle / "diff.png").write_bytes(b"tampered")

    with pytest.raises(review.AttributeReviewError, match="snapshot changed"):
        review.assert_agent_2d_qa_passed(candidate_bundle)


def test_candidate_snapshot_rejects_unqualified_route2_base(candidate_bundle):
    manifest = candidate_bundle / "candidate_manifest.json"
    manifest.chmod(0o644)
    payload = json.loads(manifest.read_text())
    payload["base_route2_qualification"]["status"] = "not_checked_configuration_only"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.chmod(0o444)

    with pytest.raises(review.AttributeReviewError, match="base qualification"):
        review.validated_candidate_snapshot(candidate_bundle)


def test_candidate_snapshot_requires_an_exact_qualified_pointer_descriptor(
    candidate_bundle,
):
    manifest = candidate_bundle / "candidate_manifest.json"
    manifest.chmod(0o644)
    payload = json.loads(manifest.read_text())
    del payload["base_route2_qualification"]["qualified_candidate"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.chmod(0o444)
    with pytest.raises(review.AttributeReviewError, match="qualified candidate"):
        review.validated_candidate_snapshot(candidate_bundle)


@pytest.mark.parametrize("status", ["pending_agent_2d_visual_qa", "approved", "user_approved"])
def test_agent_2d_recorder_rejects_invalid_status(candidate_bundle, status):
    with pytest.raises(review.AttributeReviewError, match="status"):
        review.record_agent_2d_visual_qa(
            candidate_bundle,
            status=status,
            reviewer="codex",
            notes="invalid",
            checks=_passing_checks(),
        )
