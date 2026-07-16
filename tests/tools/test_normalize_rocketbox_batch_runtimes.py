from __future__ import annotations

import json
from pathlib import Path

from tools.normalize_rocketbox_batch_runtimes import (
    expected_ue_qa,
    source_manifest_is_verified,
)


def test_expected_ue_qa_preserves_adult_and_child_authored_height():
    adult = expected_ue_qa(
        {
            "category": "Adults",
            "authored_height_cm": 183.1,
            "mouth_audio_height_cm": 183.1 * 0.90,
            "apartment_ceiling_cm": 280.0,
            "ceiling_headroom_cm": 96.9,
        },
        demographic="adult",
    )
    child = expected_ue_qa(
        {
            "category": "Children",
            "authored_height_cm": 143.3,
            "mouth_audio_height_cm": 143.3 * 0.88,
            "apartment_ceiling_cm": 280.0,
            "ceiling_headroom_cm": 136.7,
        },
        demographic="child",
    )

    assert adult["actor_scale"] == child["actor_scale"] == 1.0
    assert adult["authored_height_cm"] == 183.1
    assert child["authored_height_cm"] == 143.3
    assert adult["height_range_cm"] == [140.0, 215.0]
    assert child["height_range_cm"] == [80.0, 170.0]
    assert child["mouth_audio_height_cm"] < adult["mouth_audio_height_cm"]


def test_source_manifest_verifier_checks_runtime_hash_and_overall(tmp_path):
    runtime = tmp_path / "runtime.glb"
    runtime.write_bytes(b"runtime")
    import hashlib

    manifest = {
        "schema": "rocketbox_batch_native_runtime_v1",
        "base_avatar_id": "rocketbox_a",
        "usage_scope": "research_candidate",
        "automatic_checks": {"overall": "passed"},
        "runtime_glb": {
            "filename": "runtime.glb",
            "size_bytes": 7,
            "sha256": hashlib.sha256(b"runtime").hexdigest(),
        },
    }
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert source_manifest_is_verified(manifest_path, "rocketbox_a") is True
    runtime.write_bytes(b"changed")
    assert source_manifest_is_verified(manifest_path, "rocketbox_a") is False
