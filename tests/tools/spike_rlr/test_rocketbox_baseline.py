from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

import rocketbox_baseline as baseline_seal  # noqa: E402

from rocketbox_baseline import (  # noqa: E402
    BASELINE_FILES,
    BASELINE_SCHEMA,
    BaselineSealError,
    seal_baseline,
)
from rocketbox_motion_review import (  # noqa: E402
    EXPECTED_ASSET_IDS,
    MotionReviewNotApproved,
    record_decision,
)


def _write_approved_review(review_root: Path, asset_id: str) -> Path:
    review_dir = review_root / asset_id
    review_dir.mkdir(parents=True)
    media = {}
    for name in ("front", "side", "top", "joints", "feet", "source_target"):
        filename = f"{name}.mp4"
        (review_dir / filename).write_bytes(f"{asset_id}:{name}".encode("ascii"))
        media[name] = filename
    (review_dir / "contact_sheet.png").write_bytes(f"{asset_id}:contact".encode("ascii"))
    media["contact_sheet"] = "contact_sheet.png"

    retarget_glb = review_dir / "retarget.glb"
    retarget_glb.write_bytes(f"{asset_id}:glb".encode("ascii"))
    (review_dir / "retarget.blend").write_bytes(f"{asset_id}:blend".encode("ascii"))
    (review_dir / "retarget_metrics.json").write_bytes(
        f"{asset_id}:metrics".encode("ascii")
    )
    manifest = {
        "schema_version": "rocketbox_retarget_manifest_v1",
        "asset_id": asset_id,
        "immutable_input_hashes": {
            "avatar_fbx": "a" * 64,
            "motion_fbx": "b" * 64,
            "source_review": "c" * 64,
            "body_color_texture": "d" * 64,
            "head_color_texture": "e" * 64,
            "opacity_color_texture": "f" * 64,
            "retarget_glb": hashlib.sha256(retarget_glb.read_bytes()).hexdigest(),
        },
        "artifacts": {"blend": "retarget.blend", "glb": "retarget.glb", "metrics": "retarget_metrics.json"},
        "binding": {
            "target_asset_id": asset_id,
            "target_mesh_bound": True,
            "official_textures_attached": True,
        },
        "media": media,
        "automatic_checks": {"overall": "passed"},
    }
    (review_dir / "retarget_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    record_decision(review_dir, "approved", "reviewer", "approved")
    return review_dir


@pytest.fixture
def approved_reviews(tmp_path: Path) -> Path:
    review_root = tmp_path / "reviews"
    for asset_id in EXPECTED_ASSET_IDS:
        _write_approved_review(review_root, asset_id)
    return review_root


def test_seal_requires_the_exact_approved_pair(tmp_path: Path):
    review_root = tmp_path / "reviews"
    _write_approved_review(review_root, EXPECTED_ASSET_IDS[0])

    with pytest.raises(MotionReviewNotApproved, match="female|pair|missing"):
        seal_baseline(review_root, tmp_path / "baseline")


def test_seal_copies_only_allowlisted_artifacts_and_records_hashes_and_sizes(
    approved_reviews: Path, tmp_path: Path
):
    output_root = tmp_path / "baseline"

    manifest = seal_baseline(approved_reviews, output_root)

    assert manifest["schema_version"] == BASELINE_SCHEMA
    assert manifest["baseline_id"] == "rocketbox_neutral_walk_v1"
    assert set(manifest["assets"]) == set(EXPECTED_ASSET_IDS)
    assert (output_root / "baseline_manifest.json").is_file()
    for asset_id in EXPECTED_ASSET_IDS:
        copied_dir = output_root / asset_id
        assert {path.name for path in copied_dir.iterdir()} == set(BASELINE_FILES)
        records = manifest["assets"][asset_id]["files"]
        assert set(records) == set(BASELINE_FILES)
        for filename in BASELINE_FILES:
            source = approved_reviews / asset_id / filename
            copied = copied_dir / filename
            assert copied.read_bytes() == source.read_bytes()
            assert records[filename] == {
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "size": source.stat().st_size,
            }


def test_seal_creates_manifest_and_directory_atomically(
    approved_reviews: Path, tmp_path: Path
):
    output_root = tmp_path / "baseline"

    seal_baseline(approved_reviews, output_root)

    assert not list(tmp_path.glob(".baseline.*"))
    assert not list(tmp_path.glob("baseline.*.tmp"))
    assert json.loads((output_root / "baseline_manifest.json").read_text())


def test_identical_rerun_succeeds_without_changing_bytes(
    approved_reviews: Path, tmp_path: Path
):
    output_root = tmp_path / "baseline"
    first = seal_baseline(approved_reviews, output_root)
    before = {
        path.relative_to(output_root): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file()
    }

    second = seal_baseline(approved_reviews, output_root)

    after = {
        path.relative_to(output_root): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    assert second == first
    assert after == before


def test_nonidentical_existing_version_is_rejected(
    approved_reviews: Path, tmp_path: Path
):
    output_root = tmp_path / "baseline"
    seal_baseline(approved_reviews, output_root)
    (output_root / EXPECTED_ASSET_IDS[0] / "retarget.glb").write_bytes(b"tampered")

    with pytest.raises(BaselineSealError, match="differ|immutable|tampered"):
        seal_baseline(approved_reviews, output_root)


def test_symlinked_allowlisted_artifact_is_rejected(
    approved_reviews: Path, tmp_path: Path
):
    source = approved_reviews / EXPECTED_ASSET_IDS[0] / "retarget.blend"
    external = tmp_path / "external.blend"
    external.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(external)

    with pytest.raises((BaselineSealError, ValueError), match="symlink|regular"):
        seal_baseline(approved_reviews, tmp_path / "baseline")


def test_missing_allowlisted_artifact_is_rejected(
    approved_reviews: Path, tmp_path: Path
):
    (approved_reviews / EXPECTED_ASSET_IDS[1] / "feet.mp4").unlink()

    with pytest.raises((BaselineSealError, ValueError), match="missing|feet"):
        seal_baseline(approved_reviews, tmp_path / "baseline")


def test_approval_revoked_after_staging_is_not_published(
    approved_reviews: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_root = tmp_path / "baseline"
    original_copy = baseline_seal._copy_to_temporary

    def copy_then_revoke(review_root, temporary_root, manifest):
        original_copy(review_root, temporary_root, manifest)
        record_decision(
            review_root / EXPECTED_ASSET_IDS[0],
            "rejected",
            "reviewer",
            "revoked during staging",
        )

    monkeypatch.setattr(baseline_seal, "_copy_to_temporary", copy_then_revoke)

    with pytest.raises(MotionReviewNotApproved, match="not approved"):
        seal_baseline(approved_reviews, output_root)

    assert not output_root.exists()
    assert not list(tmp_path.glob(".baseline.*"))


def test_source_regenerated_after_staging_is_not_published(
    approved_reviews: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_root = tmp_path / "baseline"
    original_copy = baseline_seal._copy_to_temporary

    def copy_then_regenerate(review_root, temporary_root, manifest):
        original_copy(review_root, temporary_root, manifest)
        (review_root / EXPECTED_ASSET_IDS[0] / "retarget.blend").write_bytes(
            b"regenerated during staging"
        )

    monkeypatch.setattr(
        baseline_seal, "_copy_to_temporary", copy_then_regenerate
    )

    with pytest.raises(BaselineSealError, match="changed during staging"):
        seal_baseline(approved_reviews, output_root)

    assert not output_root.exists()
    assert not list(tmp_path.glob(".baseline.*"))


@pytest.mark.parametrize("populate_destination", (False, True))
def test_concurrent_destination_appearance_is_never_overwritten(
    approved_reviews: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    populate_destination: bool,
):
    output_root = tmp_path / "baseline"
    original_copy = baseline_seal._copy_to_temporary

    def copy_then_create_destination(review_root, temporary_root, manifest):
        original_copy(review_root, temporary_root, manifest)
        output_root.mkdir()
        if populate_destination:
            (output_root / "owner.txt").write_text("concurrent owner", encoding="utf-8")

    monkeypatch.setattr(
        baseline_seal, "_copy_to_temporary", copy_then_create_destination
    )

    with pytest.raises(BaselineSealError, match="already exists|concurrent"):
        seal_baseline(approved_reviews, output_root)

    assert output_root.is_dir()
    if populate_destination:
        assert (output_root / "owner.txt").read_text(encoding="utf-8") == "concurrent owner"
    else:
        assert list(output_root.iterdir()) == []
    assert not list(tmp_path.glob(".baseline.*"))


def test_mid_copy_failure_leaves_no_published_destination(
    approved_reviews: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_root = tmp_path / "baseline"
    original_copy = baseline_seal._copy_and_verify
    calls = 0

    def fail_during_copy(source, destination, expected):
        nonlocal calls
        calls += 1
        original_copy(source, destination, expected)
        if calls == 2:
            raise OSError("injected mid-copy failure")

    monkeypatch.setattr(baseline_seal, "_copy_and_verify", fail_during_copy)

    with pytest.raises(OSError, match="injected mid-copy failure"):
        seal_baseline(approved_reviews, output_root)

    assert calls == 2
    assert not output_root.exists()
    assert not list(tmp_path.glob(".baseline.*"))
