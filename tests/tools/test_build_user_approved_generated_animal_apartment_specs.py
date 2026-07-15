from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import build_user_approved_generated_animal_apartment_specs as subject
from tools import controlled_source_asset_schema as contracts


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_bytes(value)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_hash(value: dict) -> str:
    payload = {key: copy.deepcopy(item) for key, item in value.items() if key != "decision_sha256"}
    return hashlib.sha256(contracts.canonical_json(payload).encode("utf-8")).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    asset_id = "horse_candidate_001"
    tag = "pixal_horse_candidate_001"
    profile = "horse_profile_v1"
    sampled = {
        "body_build": "standard",
        "coat_tone": "bay",
        "life_stage": "adult",
        "size": "medium",
    }
    runtime = _write(tmp_path / "runtime.glb", b"fake-glb")
    review = _write(tmp_path / "review.json", {"visual_qa": "approved"})
    review_descriptor = {
        "path": str(review.resolve()),
        "sha256": _sha(review),
        "size_bytes": review.stat().st_size,
    }
    decision = {
        "schema": "avengine_controlled_animal_animation_decision_v1",
        "asset_id": asset_id,
        "review": review_descriptor,
        "review_sha256": review_descriptor["sha256"],
        "reviewer": "fixture",
        "decision": "approved_for_ue_apartment",
        "checks": {"walking_direction": True},
        "caveats": [],
        "notes": "fixture",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": "ue_apartment",
    }
    decision["decision_sha256"] = _decision_hash(decision)
    decision_path = _write(tmp_path / "decision.json", decision)
    config_path = _write(
        tmp_path / "config.json",
        {
            "schema": subject.CONFIG_SCHEMA,
            "asset_id": asset_id,
            "tag": tag,
            "profile_schema_id": profile,
            "species": "horse",
            "breed": "bay_horse",
            "sampled_attributes": sampled,
            "actor_scale": 0.332,
            "walking_forward_yaw_offset_deg": 90.0,
            "ground_snap_max_abs_correction_cm": 66.4,
            "audio_lookup": "horse_neigh",
            "audio_source_height_offset_m": 1.3,
            "scale_rationale": "fixture scale",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
        },
    )
    jobs_path = _write(
        tmp_path / "jobs.json",
        {
            "schema": "pixal_animal_ue_import_batch_v1",
            "jobs": [
                {
                    "asset_id": asset_id,
                    "legacy_tag": asset_id,
                    "tag": tag,
                    "profile_schema_id": profile,
                    "sampled_attributes": sampled,
                    "expected_actions": ["Walking", "Idle"],
                    "rigged_glb": str(runtime.resolve()),
                    "rigged_glb_sha256": _sha(runtime),
                }
            ],
            "non_destructive_policy": "fixture",
        },
    )
    result_path = _write(
        tmp_path / "result.json",
        {
            "schema": "pixal_animal_ue_import_result_v1",
            "input_manifest": str(jobs_path.resolve()),
            "passed_count": 1,
            "results": [
                {
                    "legacy_tag": asset_id,
                    "tag": tag,
                    "source_sha256": _sha(runtime),
                    "actions": ["Walking", "Idle"],
                    "status": "passed",
                }
            ],
        },
    )
    trajectory = [[float(index), 0.0, 0.0] for index in range(5)]
    template_path = _write(
        tmp_path / "template.json",
        {
            "render_config": {"duration_s": 1.0, "fps": 5, "n_frames": 5},
            "audio_config": {"duration_s": 1.0, "sample_rate_hz": 16000},
            "trajectory_profile": "fixture",
            "rig_direction_check_windows": [{"frame_a": 0, "frame_b": 1, "label": "start"}],
            "camera_pass_table_loop_contract": {"left_front_nearest_frame": 2},
            "sources": [
                {
                    "tag": "template",
                    "start_pos_m": trajectory[0],
                    "end_pos_m": trajectory[-1],
                    "trajectory_m": trajectory,
                }
            ],
        },
    )
    return {
        "config_path": config_path,
        "ue_jobs": jobs_path,
        "ue_result": result_path,
        "animation_decision": decision_path,
        "template": template_path,
    }


def test_builds_authenticated_walk_idle_pair(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    output_root = tmp_path / "output"
    manifest_path = subject.build_specs(**inputs, output_root=output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == subject.OUTPUT_SCHEMA
    assert manifest["avatar_count"] == 1
    assert manifest["clip_count"] == 2
    assert manifest["manifest_sha256"] == contracts.manifest_sha256(manifest)
    assert manifest["formal_registration_authorized"] is False
    actions = manifest["records"][0]["actions"]
    assert set(actions) == {"Walking", "Idle"}
    walking = json.loads(Path(actions["Walking"]["spec"]).read_text(encoding="utf-8"))
    idle = json.loads(Path(actions["Idle"]["spec"]).read_text(encoding="utf-8"))
    source = walking["sources"][0]
    assert source["walking_forward_yaw_offset_deg"] == 90.0
    assert source["controlled_animal_gate"]["status"] == "approved_for_research_candidate_apartment"
    assert source["controlled_animal_gate"]["formal_dataset_registration_authorized"] is False
    assert "rig_direction_check_windows" in walking
    assert "rig_direction_check_windows" not in idle
    assert idle["sources"][0]["trajectory_m"] == [[2.0, 0.0, 0.0]] * 5


def test_refuses_changed_decision_and_existing_output(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    output_root = tmp_path / "output"
    subject.build_specs(**inputs, output_root=output_root)
    with pytest.raises(contracts.ContractError, match="refusing to replace output"):
        subject.build_specs(**inputs, output_root=output_root)

    changed = json.loads(inputs["animation_decision"].read_text(encoding="utf-8"))
    changed["decision"] = "rejected"
    inputs["animation_decision"].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(contracts.ContractError, match="decision"):
        subject.build_specs(**inputs, output_root=tmp_path / "second")
