"""Tests for the hash-locked human reference review contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

import human_reference_review as review_contract  # noqa: E402

from human_reference_review import (  # noqa: E402
    EXPECTED_ASSET_IDS,
    HumanReferenceNotApproved,
    assert_pair_approved,
    assert_reference_approved,
    read_review_state,
    record_review,
    sha256_file,
    validate_candidate_manifest,
    write_candidate_manifest,
)


MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
SOURCE_APPROVAL_SHA256 = "a" * 64


def write_candidate_fixture(root: Path, asset_id: str) -> Path:
    candidate_dir = root / asset_id
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "source.png").write_bytes(f"{asset_id}:source".encode("ascii"))
    (candidate_dir / "candidate.png").write_bytes(
        f"{asset_id}:candidate".encode("ascii")
    )
    write_candidate_manifest(
        candidate_dir,
        asset_id=asset_id,
        model_revision=MODEL_REVISION,
        prompt=f"Exact review prompt for {asset_id}.",
        seed=4242,
        width=1024,
        height=1536,
        steps=28,
        guidance_scale=4.0,
        source_approval_sha256=SOURCE_APPROVAL_SHA256,
    )
    return candidate_dir


def current_snapshot(candidate_dir: Path) -> dict[str, str]:
    return {
        "candidate_manifest_sha256": sha256_file(
            candidate_dir / "candidate_manifest.json"
        ),
        "source_sha256": sha256_file(candidate_dir / "source.png"),
        "candidate_sha256": sha256_file(candidate_dir / "candidate.png"),
    }


def write_approval(candidate_dir: Path, **updates) -> None:
    record_review(
        candidate_dir,
        "approved",
        "reviewer",
        "looks good",
        expected_snapshot=current_snapshot(candidate_dir),
    )
    review_path = candidate_dir / "reference_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.update(updates)
    review_path.write_text(json.dumps(review), encoding="utf-8")


def test_write_candidate_manifest_records_complete_hash_locked_provenance(tmp_path):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])

    manifest, images = validate_candidate_manifest(candidate_dir)

    assert manifest == {
        "schema_version": "human_reference_candidate_v1",
        "asset_id": EXPECTED_ASSET_IDS[0],
        "model_revision": MODEL_REVISION,
        "prompt": f"Exact review prompt for {EXPECTED_ASSET_IDS[0]}.",
        "seed": 4242,
        "width": 1024,
        "height": 1536,
        "steps": 28,
        "guidance_scale": 4.0,
        "source_approval_sha256": SOURCE_APPROVAL_SHA256,
        "input_sha256": sha256_file(candidate_dir / "source.png"),
        "output_sha256": sha256_file(candidate_dir / "candidate.png"),
        "output_size_bytes": (candidate_dir / "candidate.png").stat().st_size,
    }
    assert images == {
        "source": candidate_dir / "source.png",
        "candidate": candidate_dir / "candidate.png",
    }
    assert not (candidate_dir / "candidate_manifest.json.tmp").exists()


@pytest.mark.parametrize("asset_id", ("unexpected", "../rocketbox_male_adult_01"))
def test_writer_rejects_asset_ids_outside_the_exact_allowlist(tmp_path, asset_id):
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "source.png").write_bytes(b"source")
    (candidate_dir / "candidate.png").write_bytes(b"candidate")

    with pytest.raises(ValueError, match="asset_id"):
        write_candidate_manifest(
            candidate_dir,
            asset_id=asset_id,
            model_revision=MODEL_REVISION,
            prompt="Prompt.",
            seed=1,
            width=1024,
            height=1536,
            steps=28,
            guidance_scale=4.0,
            source_approval_sha256=SOURCE_APPROVAL_SHA256,
        )


@pytest.mark.parametrize("field", ("input_sha256", "output_sha256", "source_approval_sha256"))
@pytest.mark.parametrize("bad_hash", ("a" * 63, "A" * 64, "g" * 64))
def test_validation_rejects_malformed_hashes(tmp_path, field, bad_hash):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    manifest_path = candidate_dir / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = bad_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="64-character lowercase hex"):
        validate_candidate_manifest(candidate_dir)


@pytest.mark.parametrize("filename", ("source.png", "candidate.png", "candidate_manifest.json"))
def test_validation_rejects_symlinked_contract_files(tmp_path, filename):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    target = tmp_path / f"external-{filename}"
    target.write_bytes((candidate_dir / filename).read_bytes())
    (candidate_dir / filename).unlink()
    (candidate_dir / filename).symlink_to(target)

    with pytest.raises(ValueError, match="regular file|symlink|asset root"):
        validate_candidate_manifest(candidate_dir)


def test_changed_source_image_makes_the_manifest_unreviewable(tmp_path):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    (candidate_dir / "source.png").write_bytes(b"replaced source")

    with pytest.raises(ValueError, match="input.*hash"):
        validate_candidate_manifest(candidate_dir)


def test_changed_candidate_image_makes_the_manifest_unreviewable(tmp_path):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    (candidate_dir / "candidate.png").write_bytes(b"regenerated candidate")

    with pytest.raises(ValueError, match="output.*hash|size"):
        validate_candidate_manifest(candidate_dir)


def test_validated_candidate_snapshot_binds_manifest_and_image_hashes(tmp_path):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])

    manifest, images, snapshot = review_contract.validated_candidate_snapshot(
        candidate_dir
    )

    assert manifest["prompt"] == f"Exact review prompt for {EXPECTED_ASSET_IDS[0]}."
    assert images["source"] == candidate_dir / "source.png"
    assert snapshot == current_snapshot(candidate_dir)


def test_validated_candidate_snapshot_retries_a_generation_change(
    tmp_path, monkeypatch
):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    original_read_once = review_contract._read_candidate_snapshot_once
    read_count = 0

    def regenerate_after_first_read(path):
        nonlocal read_count
        result = original_read_once(path)
        read_count += 1
        if read_count == 1:
            (candidate_dir / "source.png").write_bytes(b"source B")
            (candidate_dir / "candidate.png").write_bytes(b"candidate B")
            write_candidate_manifest(
                candidate_dir,
                asset_id=EXPECTED_ASSET_IDS[0],
                model_revision=MODEL_REVISION,
                prompt="Candidate B prompt.",
                seed=4243,
                width=1024,
                height=1536,
                steps=28,
                guidance_scale=4.0,
                source_approval_sha256=SOURCE_APPROVAL_SHA256,
            )
        return result

    monkeypatch.setattr(
        review_contract, "_read_candidate_snapshot_once", regenerate_after_first_read
    )

    manifest, _, snapshot = review_contract.validated_candidate_snapshot(candidate_dir)

    assert manifest["prompt"] == "Candidate B prompt."
    assert snapshot == current_snapshot(candidate_dir)


def test_review_binds_to_current_manifest_and_both_images(tmp_path):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    expected_snapshot = current_snapshot(candidate_dir)

    review = record_review(
        candidate_dir,
        "approved",
        "  reviewer-a ",
        "  ready ",
        expected_snapshot=expected_snapshot,
    )

    assert review["schema_version"] == "human_reference_review_v1"
    assert review["reviewer"] == "reviewer-a"
    assert review["notes"] == "ready"
    assert review["candidate_manifest_sha256"] == sha256_file(
        candidate_dir / "candidate_manifest.json"
    )
    assert review["source_sha256"] == sha256_file(candidate_dir / "source.png")
    assert review["candidate_sha256"] == sha256_file(candidate_dir / "candidate.png")
    assert_reference_approved(candidate_dir)


def test_record_review_requires_an_expected_snapshot(tmp_path):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])

    with pytest.raises(TypeError, match="expected_snapshot"):
        record_review(candidate_dir, "approved", "reviewer", "ready")


def test_record_review_rejects_a_snapshot_that_changed_before_internal_validation(
    tmp_path,
):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    expected_snapshot = current_snapshot(candidate_dir)
    (candidate_dir / "source.png").write_bytes(b"source B")
    (candidate_dir / "candidate.png").write_bytes(b"candidate B")
    write_candidate_manifest(
        candidate_dir,
        asset_id=EXPECTED_ASSET_IDS[0],
        model_revision=MODEL_REVISION,
        prompt="Candidate B prompt.",
        seed=4243,
        width=1024,
        height=1536,
        steps=28,
        guidance_scale=4.0,
        source_approval_sha256=SOURCE_APPROVAL_SHA256,
    )

    with pytest.raises(ValueError, match="snapshot changed"):
        record_review(
            candidate_dir,
            "approved",
            "reviewer",
            "ready",
            expected_snapshot=expected_snapshot,
        )

    assert not (candidate_dir / "reference_review.json").exists()


def test_record_review_writes_expected_hashes_if_files_change_after_validation(
    tmp_path, monkeypatch
):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    expected_snapshot = current_snapshot(candidate_dir)
    original_atomic_write = review_contract._atomic_write_json

    def regenerate_before_review_write(path, payload):
        if Path(path).name == "reference_review.json":
            (candidate_dir / "source.png").write_bytes(b"source B")
            (candidate_dir / "candidate.png").write_bytes(b"candidate B")
            write_candidate_manifest(
                candidate_dir,
                asset_id=EXPECTED_ASSET_IDS[0],
                model_revision=MODEL_REVISION,
                prompt="Candidate B prompt.",
                seed=4243,
                width=1024,
                height=1536,
                steps=28,
                guidance_scale=4.0,
                source_approval_sha256=SOURCE_APPROVAL_SHA256,
            )
        original_atomic_write(path, payload)

    monkeypatch.setattr(
        review_contract, "_atomic_write_json", regenerate_before_review_write
    )

    review = record_review(
        candidate_dir,
        "approved",
        "reviewer",
        "ready",
        expected_snapshot=expected_snapshot,
    )

    assert {
        field: review[field] for field in expected_snapshot
    } == expected_snapshot
    with pytest.raises(HumanReferenceNotApproved, match="hash is stale"):
        assert_reference_approved(candidate_dir)


def test_changed_valid_manifest_stales_an_existing_approval(tmp_path):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    record_review(
        candidate_dir,
        "approved",
        "reviewer",
        "ready",
        expected_snapshot=current_snapshot(candidate_dir),
    )
    manifest_path = candidate_dir / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt"] = "The same candidate with a corrected exact prompt."
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HumanReferenceNotApproved, match="manifest hash is stale"):
        assert_reference_approved(candidate_dir)
    assert read_review_state(candidate_dir)["decision"] == "pending"


def test_changed_candidate_hash_stales_an_existing_approval(tmp_path):
    candidate_dir = write_candidate_fixture(tmp_path, EXPECTED_ASSET_IDS[0])
    record_review(
        candidate_dir,
        "approved",
        "reviewer",
        "ready",
        expected_snapshot=current_snapshot(candidate_dir),
    )
    (candidate_dir / "candidate.png").write_bytes(b"new candidate")
    write_candidate_manifest(
        candidate_dir,
        asset_id=EXPECTED_ASSET_IDS[0],
        model_revision=MODEL_REVISION,
        prompt="Exact review prompt for rocketbox_male_adult_01.",
        seed=4243,
        width=1024,
        height=1536,
        steps=28,
        guidance_scale=4.0,
        source_approval_sha256=SOURCE_APPROVAL_SHA256,
    )

    with pytest.raises(HumanReferenceNotApproved, match="candidate image hash is stale"):
        assert_reference_approved(candidate_dir)


def test_pair_gate_rejects_a_partial_pair_then_accepts_current_approvals(tmp_path):
    review_root = tmp_path / "reviews"
    male_dir = write_candidate_fixture(review_root, EXPECTED_ASSET_IDS[0])
    female_dir = write_candidate_fixture(review_root, EXPECTED_ASSET_IDS[1])
    record_review(
        male_dir,
        "approved",
        "reviewer",
        "male ready",
        expected_snapshot=current_snapshot(male_dir),
    )

    with pytest.raises(HumanReferenceNotApproved, match="female"):
        assert_pair_approved(review_root)

    record_review(
        female_dir,
        "approved",
        "reviewer",
        "female ready",
        expected_snapshot=current_snapshot(female_dir),
    )
    approvals = assert_pair_approved(review_root)

    assert set(approvals) == set(EXPECTED_ASSET_IDS)


def test_pair_gate_rejects_a_symlinked_asset_directory(tmp_path):
    review_root = tmp_path / "reviews"
    male_dir = write_candidate_fixture(review_root, EXPECTED_ASSET_IDS[0])
    female_dir = write_candidate_fixture(tmp_path / "outside", EXPECTED_ASSET_IDS[1])
    (review_root / EXPECTED_ASSET_IDS[1]).symlink_to(female_dir, target_is_directory=True)
    record_review(
        male_dir,
        "approved",
        "reviewer",
        "male ready",
        expected_snapshot=current_snapshot(male_dir),
    )

    with pytest.raises(HumanReferenceNotApproved, match="symlink|containment"):
        assert_pair_approved(review_root)
