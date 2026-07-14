from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import run_generated_quadruped_deformation_stabilization as stage


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _fixture(tmp_path: Path, *, yaw: float = 0.0, human_approved: bool = True):
    raw = tmp_path / "raw.glb"
    rig = tmp_path / "rig.glb"
    raw.write_bytes(b"raw-pbr")
    rig.write_bytes(b"animated-rig")
    asset_id = "dog_test_v1"
    decision = {
        "schema": stage.DECISION_SCHEMA,
        "asset_id": asset_id,
        "status": stage.DECISION_STATUS,
        "human_approved": human_approved,
        "human_approved_by": "reviewer",
        "target_animation_generation_authorized": True,
        "formal_dataset_registration_authorized": False,
        "manual_cardinal_motion_basis_yaw_deg": yaw,
        "target": {"reviewed_front_axis": "positive-x"},
    }
    decision["decision_sha256"] = stage.canonical_hash_without(
        decision, "decision_sha256"
    )
    decision_path = tmp_path / "decision.json"
    _write_json(decision_path, decision)
    manifest = {
        "schema": stage.SCHEMA,
        "jobs": [
            {
                "asset_id": asset_id,
                "raw_pbr_glb": {"path": str(raw), "sha256": _sha256(raw)},
                "approved_animated_rig_glb": {
                    "path": str(rig),
                    "sha256": _sha256(rig),
                },
                "motion_basis_decision": {
                    "path": str(decision_path),
                    "sha256": _sha256(decision_path),
                },
                "target_preprocess": {
                    "flip_x": True,
                    "rotate_z_deg": 0,
                    "automatic_orientation_inference_used": False,
                    "provenance": "reviewed fixed-skeleton lineage",
                },
            }
        ],
    }
    manifest_path = tmp_path / "jobs.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_validate_only_authenticates_jobs_without_creating_output(tmp_path):
    manifest = _fixture(tmp_path)
    output = tmp_path / "never-created"

    result = stage.run_batch(manifest, output, validate_only=True)

    assert result == {
        "schema": stage.RESULT_SCHEMA,
        "status": "validated_only",
        "job_count": 1,
        "asset_ids": ["dog_test_v1"],
        "formal_dataset_registration_authorized": False,
    }
    assert not output.exists()


def test_validate_job_rejects_fine_yaw_even_with_valid_decision_hash(tmp_path):
    manifest = _fixture(tmp_path, yaw=15.0)

    with pytest.raises(stage.StabilizationError, match="fine-yaw"):
        stage.load_jobs_manifest(manifest)


def test_validate_job_requires_explicit_human_approval(tmp_path):
    manifest = _fixture(tmp_path, human_approved=False)

    with pytest.raises(stage.StabilizationError, match="human approval"):
        stage.load_jobs_manifest(manifest)


def test_validate_job_rejects_mutated_pinned_input(tmp_path):
    manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    Path(payload["jobs"][0]["raw_pbr_glb"]["path"]).write_bytes(b"changed")

    with pytest.raises(stage.StabilizationError, match="hash changed"):
        stage.load_jobs_manifest(manifest)


def test_validate_job_rejects_relative_input_path(tmp_path):
    manifest = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["jobs"][0]["raw_pbr_glb"]["path"] = "raw.glb"
    _write_json(manifest, payload)

    with pytest.raises(stage.StabilizationError, match="absolute"):
        stage.load_jobs_manifest(manifest)


def test_build_commands_freezes_200k_bvh_and_non_destructive_surface_policy(tmp_path):
    manifest = _fixture(tmp_path)
    _, jobs = stage.load_jobs_manifest(manifest)

    commands = stage.build_commands(jobs[0], tmp_path / "out")
    runtime = commands["runtime"]
    binding = commands["binding"]
    repair = commands["repair"]

    assert runtime[runtime.index("--target-faces") + 1] == "200000"
    assert "--double-sided" in runtime
    assert binding[binding.index("--nearest-backend") + 1] == "bvh"
    assert binding[binding.index("--remove-ground-artifacts") + 1] == "no"
    assert binding[binding.index("--remove-limb-bridges") + 1] == "no"
    assert "--flip-x" in binding
    assert binding[binding.index("--export-action-policy") + 1] == "walk-idle"
    assert repair[repair.index("--repair-mode") + 1] == "edge-average"
    assert repair[repair.index("--maximum-passes") + 1] == "12"
    assert commands["diagnostic_after"][
        commands["diagnostic_after"].index("--samples") + 1
    ] == "41"
    assert commands["render_walking"][
        commands["render_walking"].index("--n-frames") + 1
    ] == "24"


def test_validate_only_rejects_unsafe_extension_limit(tmp_path):
    manifest = _fixture(tmp_path)

    with pytest.raises(stage.StabilizationError, match="max_extension"):
        stage.run_batch(
            manifest,
            tmp_path / "out",
            max_extension=0.081,
            validate_only=True,
        )


def test_validate_only_rejects_unknown_asset_filter(tmp_path):
    manifest = _fixture(tmp_path)

    with pytest.raises(stage.StabilizationError, match="absent"):
        stage.run_batch(
            manifest,
            tmp_path / "out",
            asset_ids=["missing_asset"],
            validate_only=True,
        )


def test_canonical_hash_matches_controlled_asset_contract():
    value = {"b": 2, "a": 1}
    expected = hashlib.sha256(
        contracts.canonical_json(value).encode("utf-8")
    ).hexdigest()

    assert stage.canonical_hash_without(
        {**value, "decision_sha256": "ignored"}, "decision_sha256"
    ) == expected
