"""Unit tests for the VLM review calibration set collector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_vlm_review_calibration_set import collect_records, main


def make_animation_decision(root, asset_id, decision, walking_direction, media=True):
    asset_dir = root / asset_id
    asset_dir.mkdir(parents=True)
    review_path = asset_dir / "review.json"
    if media:
        review_path.write_text("{}", encoding="utf-8")
        (asset_dir / "walking_side.mp4").write_bytes(b"fake-mp4")
    payload = {
        "schema": "avengine_controlled_animal_animation_decision_v1",
        "asset_id": asset_id,
        "decision": decision,
        "checks": {
            "walking_direction": walking_direction,
            "walking_limb_deformation": True,
            "walking_ground_contact": True,
            "idle_ground_contact": True,
            "body_stability": True,
            "detached_geometry_absent": True,
        },
        "review": {"path": str(review_path)},
    }
    (asset_dir / "animation_decision.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def make_canonical_decision(root, name, decision):
    image_dir = root / name
    image_dir.mkdir(parents=True)
    (image_dir / "canonical.png").write_bytes(b"fake-png")
    payload = {
        "decision": decision,
        "breed_identity": "pass",
        "single_tail": "fail" if decision.startswith("reject") else "pass",
        "reason": "test fixture",
        "pixel3d_authorized": not decision.startswith("reject"),
    }
    (image_dir / "review_decision.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_collects_both_decision_families(tmp_path):
    make_animation_decision(
        tmp_path, "dog_a", "approved_for_ue_apartment", walking_direction=True
    )
    make_animation_decision(
        tmp_path, "cat_b", "rejected", walking_direction=False
    )
    make_canonical_decision(tmp_path, "canonical_two_tails", "reject_before_pixel3d")
    records = collect_records([tmp_path])
    assert len(records) == 3
    categories = sorted(record["category"] for record in records)
    assert categories == [
        "animation_six_checks", "animation_six_checks", "canonical_2d",
    ]
    rejected = next(
        record for record in records if record.get("asset_id") == "cat_b"
    )
    assert rejected["human_verdict"]["checks"]["walking_direction"] is False
    assert any(item["media_present"] for item in rejected["media"])


def test_missing_media_is_recorded_not_dropped(tmp_path):
    make_animation_decision(
        tmp_path, "dog_gone", "rejected", walking_direction=False, media=False
    )
    records = collect_records([tmp_path])
    assert len(records) == 1
    assert not any(item.get("media_present") for item in records[0]["media"])


def test_main_writes_manifest_and_refuses_replacement(tmp_path, capsys):
    make_canonical_decision(tmp_path, "canonical_ok", "accept")
    output = tmp_path / "calibration.json"
    assert main(["--search-root", str(tmp_path), "--output", str(output)]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 1
    assert manifest["summary_by_category"]["canonical_2d"]["count"] == 1
    with pytest.raises(SystemExit, match="refusing to replace"):
        main(["--search-root", str(tmp_path), "--output", str(output)])


def test_empty_roots_fail_closed(tmp_path):
    with pytest.raises(SystemExit, match="no human review decisions"):
        main(
            [
                "--search-root", str(tmp_path),
                "--output", str(tmp_path / "calibration.json"),
            ]
        )
