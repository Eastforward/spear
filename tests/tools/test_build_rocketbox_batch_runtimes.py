from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_rocketbox_batch_runtimes import (
    output_is_verified,
    plan_jobs,
)


def _inventory(ids):
    return {
        "schema_version": "rocketbox_human_inventory_v1",
        "automatic_checks": {"overall": "passed"},
        "avatars": [
            {"base_avatar_id": avatar_id, "inventory_status": "passed"}
            for avatar_id in ids
        ],
    }


def test_plan_jobs_is_deterministic_and_skips_verified_no_replace_outputs(tmp_path):
    inventory = _inventory(["rocketbox_b", "rocketbox_a"])
    existing = tmp_path / "rocketbox_a_original_v1"
    existing.mkdir()
    runtime = existing / "runtime.glb"
    runtime.write_bytes(b"runtime")
    import hashlib

    (existing / "build_manifest.json").write_text(
        json.dumps(
            {
                "schema": "rocketbox_batch_native_runtime_v1",
                "base_avatar_id": "rocketbox_a",
                "automatic_checks": {"overall": "passed"},
                "runtime_glb": {
                    "filename": "runtime.glb",
                    "size_bytes": len(b"runtime"),
                    "sha256": hashlib.sha256(b"runtime").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    jobs, skipped = plan_jobs(inventory, tmp_path)

    assert [job["base_avatar_id"] for job in jobs] == ["rocketbox_b"]
    assert skipped == ["rocketbox_a"]


def test_output_verification_fails_closed_on_wrong_runtime_hash(tmp_path):
    output = tmp_path / "avatar_original_v1"
    output.mkdir()
    (output / "runtime.glb").write_bytes(b"changed")
    (output / "build_manifest.json").write_text(
        json.dumps(
            {
                "schema": "rocketbox_batch_native_runtime_v1",
                "base_avatar_id": "avatar",
                "automatic_checks": {"overall": "passed"},
                "runtime_glb": {
                    "filename": "runtime.glb",
                    "size_bytes": 7,
                    "sha256": "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    assert output_is_verified(output, "avatar") is False


def test_plan_jobs_rejects_unexpected_existing_directory(tmp_path):
    inventory = _inventory(["rocketbox_a"])
    (tmp_path / "rocketbox_a_original_v1").mkdir()

    with pytest.raises(RuntimeError, match="unverified existing"):
        plan_jobs(inventory, tmp_path)
