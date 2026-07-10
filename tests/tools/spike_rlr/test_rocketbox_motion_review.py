import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from rocketbox_motion_review import (  # noqa: E402
    EXPECTED_ASSET_IDS,
    REQUIRED_MEDIA,
    MotionReviewNotApproved,
    assert_motion_approved,
    assert_pair_approved,
    ensure_pending_review,
    record_decision,
    sha256_file,
)


def write_ready_fixture(tmp_path, asset_id, *, automatic_checks=None):
    review_dir = tmp_path / asset_id
    review_dir.mkdir(parents=True)
    media = {}
    for name in REQUIRED_MEDIA:
        filename = f"{name}.png" if name == "contact_sheet" else f"{name}.mp4"
        path = review_dir / filename
        path.write_bytes(f"{asset_id}:{name}".encode("ascii"))
        media[name] = filename
    manifest = {
        "schema_version": "rocketbox_retarget_manifest_v1",
        "asset_id": asset_id,
        "immutable_input_hashes": {
            "avatar_fbx": "a" * 64,
            "motion_fbx": "b" * 64,
            "source_review": "c" * 64,
        },
        "media": media,
        "automatic_checks": automatic_checks or {"overall": "passed"},
    }
    (review_dir / "retarget_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return review_dir


def test_record_decision_pins_current_manifest_and_media(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")

    result = record_decision(review_dir, "approved", "jzy", "motion looks stable")

    assert result["schema_version"] == "rocketbox_motion_review_v1"
    assert result["decision"] == "approved"
    assert result["retarget_manifest_sha256"] == sha256_file(
        review_dir / "retarget_manifest.json"
    )
    assert result["media_sha256"]["front"] == sha256_file(review_dir / "front.mp4")
    assert not (review_dir / "motion_review.json.tmp").exists()


def test_changed_media_invalidates_approval(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    record_decision(review_dir, "approved", "jzy", "approved")
    (review_dir / "front.mp4").write_bytes(b"rerendered")

    with pytest.raises(MotionReviewNotApproved, match="front.*hash"):
        assert_motion_approved(review_dir)


def test_manifest_media_path_must_stay_inside_review_directory(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    manifest_path = review_dir / "retarget_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["media"]["front"] = "../outside.mp4"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="front.*review directory"):
        record_decision(review_dir, "approved", "jzy", "approved")


def test_failed_automatic_checks_make_manifest_unreviewable(tmp_path):
    review_dir = write_ready_fixture(
        tmp_path,
        "rocketbox_male_adult_01",
        automatic_checks={"overall": "failed", "glb_reimport": "failed"},
    )

    with pytest.raises(ValueError, match="automatic checks"):
        ensure_pending_review(review_dir)


def test_nested_failed_automatic_check_is_reported_as_validation_error(tmp_path):
    review_dir = write_ready_fixture(
        tmp_path,
        "rocketbox_male_adult_01",
        automatic_checks={
            "overall": "passed",
            "glb_reimport": {"status": "failed"},
        },
    )

    with pytest.raises(ValueError, match="automatic checks"):
        ensure_pending_review(review_dir)


def test_pair_gate_requires_both_current_approvals(tmp_path):
    root = tmp_path / "reviews"
    male = write_ready_fixture(root, EXPECTED_ASSET_IDS[0])
    female = write_ready_fixture(root, EXPECTED_ASSET_IDS[1])
    record_decision(male, "approved", "jzy", "male approved")

    with pytest.raises(MotionReviewNotApproved, match="female"):
        assert_pair_approved(root)

    record_decision(female, "rejected", "jzy", "female needs work")
    with pytest.raises(MotionReviewNotApproved, match="rejected"):
        assert_pair_approved(root)

    record_decision(female, "approved", "jzy", "female approved")
    (male / "side.mp4").write_bytes(b"stale")
    with pytest.raises(MotionReviewNotApproved, match="side.*hash"):
        assert_pair_approved(root)

    record_decision(male, "approved", "jzy", "male re-approved")
    result = assert_pair_approved(root)
    assert set(result) == set(EXPECTED_ASSET_IDS)
    assert all(record["decision"] == "approved" for record in result.values())


def test_record_decision_rejects_invalid_decision_and_reviewer(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")

    with pytest.raises(ValueError, match="approved or rejected"):
        record_decision(review_dir, "pending", "jzy", "notes")
    with pytest.raises(ValueError, match="reviewer must be non-empty"):
        record_decision(review_dir, "approved", "  ", "notes")


def test_ensure_pending_review_writes_current_pending_record(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")

    result = ensure_pending_review(review_dir)

    assert result["schema_version"] == "rocketbox_motion_review_v1"
    assert result["decision"] == "pending"
    assert result["retarget_manifest_sha256"] == sha256_file(
        review_dir / "retarget_manifest.json"
    )
    assert (review_dir / "motion_review.json").exists()
