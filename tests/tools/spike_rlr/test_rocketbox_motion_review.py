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
    validate_ready_manifest,
)


REQUIRED_INPUT_HASHES = (
    "avatar_fbx",
    "motion_fbx",
    "source_review",
    "body_color_texture",
    "head_color_texture",
    "opacity_color_texture",
    "retarget_glb",
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
            name: value * 64
            for name, value in zip(
                REQUIRED_INPUT_HASHES, ("a", "b", "c", "d", "e", "f", "0")
            )
        },
        "binding": {
            "target_asset_id": asset_id,
            "target_mesh_bound": True,
            "official_textures_attached": True,
        },
        "media": media,
        "automatic_checks": automatic_checks or {"overall": "passed"},
    }
    (review_dir / "retarget_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return review_dir


def write_approval_record(review_dir, **updates):
    record_decision(review_dir, "approved", "jzy", "approved")
    path = review_dir / "motion_review.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(updates)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_schema_less_approval_is_not_approved(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    write_approval_record(review_dir)
    path = review_dir / "motion_review.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("schema_version")
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(MotionReviewNotApproved, match="schema"):
        assert_motion_approved(review_dir)


def test_cross_asset_approval_is_not_approved(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    write_approval_record(review_dir, asset_id="rocketbox_female_adult_01")

    with pytest.raises(MotionReviewNotApproved, match="asset_id"):
        assert_motion_approved(review_dir)


def test_blank_reviewer_approval_is_not_approved(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    write_approval_record(review_dir, reviewer=" ")

    with pytest.raises(MotionReviewNotApproved, match="reviewer"):
        assert_motion_approved(review_dir)


def test_naive_reviewed_at_approval_is_not_approved(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    write_approval_record(review_dir, reviewed_at="2026-07-10T12:00:00")

    with pytest.raises(MotionReviewNotApproved, match="reviewed_at"):
        assert_motion_approved(review_dir)


def test_non_timezone_reviewed_at_approval_is_not_approved(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    write_approval_record(review_dir, reviewed_at="not-an-iso-timestamp")

    with pytest.raises(MotionReviewNotApproved, match="reviewed_at"):
        assert_motion_approved(review_dir)


def test_record_decision_rejects_symlinked_asset_directory(tmp_path):
    actual_dir = write_ready_fixture(tmp_path / "actual", "rocketbox_male_adult_01")
    symlink_dir = tmp_path / "asset-link"
    symlink_dir.symlink_to(actual_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        record_decision(symlink_dir, "approved", "jzy", "approved")


def test_pair_gate_rejects_external_symlinked_asset_directory(tmp_path):
    review_root = tmp_path / "reviews"
    male_dir = write_ready_fixture(review_root, "rocketbox_male_adult_01")
    outside_female_dir = write_ready_fixture(
        tmp_path / "outside", "rocketbox_female_adult_01"
    )
    (review_root / "rocketbox_female_adult_01").symlink_to(
        outside_female_dir, target_is_directory=True
    )
    record_decision(male_dir, "approved", "jzy", "approved")

    with pytest.raises(MotionReviewNotApproved, match="containment|symlink"):
        assert_pair_approved(review_root)


def test_wrong_manifest_schema_is_not_ready(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    path = review_dir / "retarget_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "wrong_schema"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        record_decision(review_dir, "approved", "jzy", "approved")


@pytest.mark.parametrize(
    "binding_update",
    (
        {"target_asset_id": None},
        {"target_mesh_bound": None},
        {"official_textures_attached": None},
    ),
)
def test_missing_required_binding_provenance_is_not_ready(tmp_path, binding_update):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    path = review_dir / "retarget_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["binding"].pop(next(iter(binding_update)))
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="binding"):
        record_decision(review_dir, "approved", "jzy", "approved")


def test_missing_required_hash_provenance_is_not_ready(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    path = review_dir / "retarget_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["immutable_input_hashes"].pop("retarget_glb")
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable_input_hashes"):
        record_decision(review_dir, "approved", "jzy", "approved")


@pytest.mark.parametrize("bad_hash", ("a" * 63, "a" * 65, "A" * 64, "g" * 64))
def test_malformed_hash_provenance_is_not_ready(tmp_path, bad_hash):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    path = review_dir / "retarget_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["immutable_input_hashes"]["retarget_glb"] = bad_hash
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="64-character lowercase hex"):
        record_decision(review_dir, "approved", "jzy", "approved")


def test_external_manifest_symlink_rejected_by_validate_and_record(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    manifest_path = review_dir / "retarget_manifest.json"
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(external_manifest)

    with pytest.raises(ValueError, match="manifest.*regular|manifest.*root|symlink"):
        validate_ready_manifest(review_dir)
    with pytest.raises(ValueError, match="manifest.*regular|manifest.*root|symlink"):
        record_decision(review_dir, "approved", "jzy", "approved")


def test_pair_gate_rejects_external_manifest_symlink(tmp_path):
    review_root = tmp_path / "reviews"
    male_dir = write_ready_fixture(review_root, "rocketbox_male_adult_01")
    female_dir = write_ready_fixture(review_root, "rocketbox_female_adult_01")
    record_decision(male_dir, "approved", "jzy", "approved")
    record_decision(female_dir, "approved", "jzy", "approved")

    manifest_path = male_dir / "retarget_manifest.json"
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(external_manifest)

    with pytest.raises(ValueError, match="manifest.*regular|manifest.*root|symlink"):
        assert_pair_approved(review_root)


def test_record_decision_rejects_external_review_symlink(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    record_decision(review_dir, "approved", "jzy", "approved")
    review_path = review_dir / "motion_review.json"
    external_review = tmp_path / "external-review.json"
    external_review.write_bytes(review_path.read_bytes())
    review_path.unlink()
    review_path.symlink_to(external_review)

    with pytest.raises(ValueError, match="review.*regular|review.*root|symlink"):
        record_decision(review_dir, "approved", "jzy", "replaced")


def test_approval_rejects_external_review_symlink(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    record_decision(review_dir, "approved", "jzy", "approved")
    review_path = review_dir / "motion_review.json"
    external_review = tmp_path / "external-review.json"
    external_review.write_bytes(review_path.read_bytes())
    review_path.unlink()
    review_path.symlink_to(external_review)

    with pytest.raises(MotionReviewNotApproved, match="review.*regular|review.*root|symlink"):
        assert_motion_approved(review_dir)


def test_pair_gate_rejects_external_review_symlink(tmp_path):
    review_root = tmp_path / "reviews"
    male_dir = write_ready_fixture(review_root, "rocketbox_male_adult_01")
    female_dir = write_ready_fixture(review_root, "rocketbox_female_adult_01")
    record_decision(male_dir, "approved", "jzy", "approved")
    record_decision(female_dir, "approved", "jzy", "approved")

    review_path = female_dir / "motion_review.json"
    external_review = tmp_path / "external-review.json"
    external_review.write_bytes(review_path.read_bytes())
    review_path.unlink()
    review_path.symlink_to(external_review)

    with pytest.raises(MotionReviewNotApproved, match="review.*regular|review.*root|symlink"):
        assert_pair_approved(review_root)


def test_atomic_write_does_not_follow_precreated_tmp_symlink(tmp_path):
    review_dir = write_ready_fixture(tmp_path, "rocketbox_male_adult_01")
    external_file = tmp_path / "external-write-target.json"
    external_file.write_text("sentinel", encoding="utf-8")
    (review_dir / "motion_review.json.tmp").symlink_to(external_file)

    record_decision(review_dir, "approved", "jzy", "approved")

    assert external_file.read_text(encoding="utf-8") == "sentinel"


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
