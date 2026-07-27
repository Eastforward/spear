from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

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
    payload = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "decision_sha256"
    }
    return hashlib.sha256(contracts.canonical_json(payload).encode("utf-8")).hexdigest()


def _hash_without(value: dict, key: str) -> str:
    payload = {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    return hashlib.sha256(contracts.canonical_json(payload).encode("utf-8")).hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_descriptor(path: Path, root: Path) -> dict:
    descriptor = _descriptor(path)
    descriptor["path"] = path.resolve().relative_to(root.resolve()).as_posix()
    return descriptor


def _repin(inputs: dict[str, Any], *path_keys: str) -> None:
    pin_keys = {
        "ue_preparation": "expected_ue_preparation_sha256",
        "ue_jobs": "expected_ue_jobs_sha256",
        "ue_result": "expected_ue_result_sha256",
        "animation_decision": "expected_animation_decision_sha256",
        "animation_decision_freeze_receipt": (
            "expected_animation_decision_freeze_receipt_sha256"
        ),
    }
    for path_key in path_keys:
        inputs[pin_keys[path_key]] = _sha(inputs[path_key])


def _inject_duplicate_schema(path: Path) -> None:
    payload = contracts.load_json(path)
    encoded = json.dumps(payload)
    duplicate = json.dumps(payload["schema"])
    path.write_text(
        f'{{"schema":{duplicate},' + encoded[1:] + "\n",
        encoding="utf-8",
    )


def _write_preparation_and_rebind_result(
    inputs: dict[str, Any],
    preparation: dict[str, Any],
) -> None:
    preparation["manifest_sha256"] = _hash_without(
        preparation,
        "manifest_sha256",
    )
    _write(inputs["ue_preparation"], preparation)
    result = contracts.load_json(inputs["ue_result"])
    descriptor = _descriptor(inputs["ue_preparation"])
    descriptor["manifest_sha256"] = preparation["manifest_sha256"]
    result["preparation_manifest"] = descriptor
    _write(inputs["ue_result"], result)
    _repin(inputs, "ue_preparation", "ue_result")


def _target_physical_profile() -> dict:
    return {
        "profile_id": "horse_fixture_physical_v1",
        "control_attribute": "size",
        "selected_value": "medium",
        "measurement": "shoulder_height_cm",
        "mode": "relative_to_profile_reference",
        "reference_value_cm": 50.0,
        "reference_provenance": {
            "status": "provisional",
            "source_id": "fixture_reference_v1",
            "artifact": None,
            "notes": "Fixture-only provisional physical target.",
        },
        "scale_ratio": 1.0,
        "tolerance_cm": 3.0,
        "target_value_cm": 50.0,
    }


def _weight_repair_evidence(runtime: Path) -> dict:
    return {
        "schema": subject.WEIGHT_REPAIR_SCHEMA,
        "status": "research_candidate_pending_readback_and_visual_qa",
        "output": _descriptor(runtime),
        "authority_contract": {
            "native_mesh_geometry_preserved": True,
            "native_mesh_topology_preserved": True,
            "pbr_material_preserved": True,
            "fitted_skeleton_rest_matrices_preserved": True,
            "approved_animation_curves_preserved": True,
            "only_vertex_weights_modified_in_memory": True,
        },
        "semantic_rig": {
            "chains": {
                "front_side_negative": [
                    "generated_front_negative_upper",
                    "generated_front_negative_lower",
                ],
                "front_side_positive": [
                    "generated_front_positive_upper",
                    "generated_front_positive_lower",
                ],
            }
        },
    }


def _retarget_evidence(runtime: Path) -> dict:
    return {
        "schema": subject.RETARGET_SCHEMA,
        "export": _descriptor(runtime),
        "semantic_inference": {
            "bone_name_independent_target": True,
            "complete_target_bone_coverage": True,
            "chains": {
                "front_side_negative": [
                    "generated_front_negative_upper",
                    "generated_front_negative_lower",
                ],
                "front_side_positive": [
                    "generated_front_positive_upper",
                    "generated_front_positive_lower",
                ],
            },
        },
    }


def _fixture(
    tmp_path: Path,
    *,
    review_schema: str = subject.FORMAL_GENERATED_REVIEW_SCHEMA,
    weight_repair_branch: str = "primary",
) -> dict[str, Any]:
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
    semantic_payload = (
        _retarget_evidence(runtime)
        if weight_repair_branch == "not_needed"
        else _weight_repair_evidence(runtime)
    )
    semantic_evidence = _write(
        tmp_path
        / (
            "retarget_manifest.json"
            if weight_repair_branch == "not_needed"
            else "weight_repair_manifest.json"
        ),
        semantic_payload,
    )
    final_artifact = subject.V4_FINAL_WEIGHT_REPAIR_ARTIFACTS[weight_repair_branch]
    final_glb_role = final_artifact["glb_output_descriptor"]
    final_manifest_role = final_artifact["manifest_output_descriptor"]
    outputs = {
        "animated_glb": _descriptor(runtime),
        "retargeted_animated_glb": _descriptor(runtime),
        final_glb_role: _descriptor(runtime),
    }
    if final_manifest_role is not None:
        outputs[final_manifest_role] = _descriptor(semantic_evidence)
        outputs["weight_repair_manifest"] = _descriptor(semantic_evidence)
    else:
        outputs["retarget_manifest"] = _descriptor(semantic_evidence)
    review = _write(
        tmp_path / "review.json",
        {
            "schema": review_schema,
            "status": "research_candidate_pending_human_review",
            "formal_dataset_registration_authorized": False,
            "automatic_admission_gates": {
                "weight_repair_branch": weight_repair_branch,
                "weight_repair_final_artifact": final_artifact,
                "all_automatic_gates_passed": True,
            },
            "outputs": outputs,
        },
    )
    review_descriptor = _descriptor(review)
    decision = {
        "schema": "avengine_controlled_animal_animation_decision_v1",
        "asset_id": asset_id,
        "review": review_descriptor,
        "review_sha256": review_descriptor["sha256"],
        "decision": "approved_for_ue_apartment",
        "checks": {name: True for name in sorted(subject.DECISION_CHECK_FIELDS)},
        "caveats": [],
        "notes": "fixture",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": "ue_import_metric_trajectory_audio_and_apartment_media",
    }
    decision["decision_sha256"] = _decision_hash(decision)
    decision_path = _write(tmp_path / "decision.json", decision)
    target_physical_profile = _target_physical_profile()
    source_asset = _write(
        tmp_path / "source_asset.json",
        {
            "schema": contracts.SOURCE_ASSET_SCHEMA,
            "asset_id": asset_id,
            "profile_schema_id": profile,
            "profile_sha256": "2" * 64,
            "request_sha256": "1" * 64,
            "taxonomy": {"species": "horse", "breed": "bay_horse"},
            "fixed_attributes": {},
            "sampled_attributes": sampled,
            "target_physical_profile": target_physical_profile,
        },
    )
    registry_entry = {
        "asset_id": asset_id,
        "profile_schema_id": profile,
        "request_sha256": "1" * 64,
        "sampled_attributes": sampled,
        "attribute_evidence": {},
        "source_asset": _relative_descriptor(source_asset, tmp_path),
        "state_classification": "research_candidate",
        "next_gate": "lod_then_species_rig_binding",
    }
    source_registry_payload = {
        "schema": subject.source_registry.REGISTRY_SCHEMA,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "preflight": {
            "path": str((tmp_path / "fixture_preflight.json").resolve()),
            "sha256": "4" * 64,
            "preflight_sha256": "5" * 64,
            "validation_mode": "current_exact_rebuild",
        },
        "pixal_batch": {
            "path": str((tmp_path / "fixture_pixal_batch.json").resolve()),
            "sha256": "6" * 64,
            "batch_sha256": "7" * 64,
        },
        "static_decision_batch": {
            "path": str((tmp_path / "fixture_static_decisions.json").resolve()),
            "sha256": "8" * 64,
            "decision_batch_sha256": "9" * 64,
        },
        "source_asset_count": 1,
        "source_assets": [registry_entry],
        "automatic_checks": copy.deepcopy(
            subject.preparation_bridge.REGISTRY_AUTOMATIC_CHECKS
        ),
    }
    source_registry_payload["registry_sha256"] = _hash_without(
        source_registry_payload,
        "registry_sha256",
    )
    source_registry = _write(
        tmp_path / "source_registry.json",
        source_registry_payload,
    )
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
            "target_physical_profile": target_physical_profile,
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
    job = {
        "job_type": subject.JOB_TYPE,
        "asset_id": asset_id,
        "legacy_tag": asset_id,
        "tag": tag,
        "profile_schema_id": profile,
        "sampled_attributes": sampled,
        "expected_actions": subject.EXPECTED_ACTIONS,
        "rigged_glb": str(runtime.resolve()),
        "rigged_glb_sha256": _sha(runtime),
        "source_registry_sha256": _sha(source_registry),
        "source_asset_sha256": _sha(source_asset),
        "request_sha256": "1" * 64,
        "animation_decision_file_sha256": _sha(decision_path),
        "animation_decision_sha256": decision["decision_sha256"],
    }
    policy = subject._expected_non_destructive_policy(tag)
    jobs = {
        "schema": subject.FORMAL_BATCH_SCHEMA,
        "status": "ready_for_new_ue_import",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "job_type": subject.JOB_TYPE,
        "job_count": 1,
        "jobs": [job],
        "non_destructive_policy": policy,
    }
    jobs["batch_sha256"] = _hash_without(jobs, "batch_sha256")
    jobs_path = _write(
        tmp_path / "jobs.json",
        jobs,
    )
    freeze_receipt = {
        "schema": subject.DECISION_FREEZE_RECEIPT_SCHEMA,
        "status": "frozen",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "source_asset_registry": _descriptor(source_registry),
        "expected_source_asset_registry_file_sha256": _sha(source_registry),
        "source_asset_registry_validation_mode": "current_exact_rebuild",
        "source_asset": _descriptor(source_asset),
        "animation_review": _descriptor(review),
        "expected_animation_review_file_sha256": _sha(review),
        "user_instruction_binding": {
            "decision": "approved_for_ue_apartment",
            "review_sha256": _sha(review),
            "all_six_checks_explicit": True,
        },
        "user_instruction_authority": copy.deepcopy(subject.USER_INSTRUCTION_AUTHORITY),
        "authenticated_review_artifact_count": 1,
        "animation_decision": _relative_descriptor(decision_path, tmp_path),
        "decision_sha256": decision["decision_sha256"],
    }
    freeze_receipt["receipt_sha256"] = _hash_without(
        freeze_receipt,
        "receipt_sha256",
    )
    freeze_receipt_path = _write(
        tmp_path / "decision_freeze_receipt.json",
        freeze_receipt,
    )
    automatic_checks = {
        field: True
        for field in subject.PREPARATION_AUTOMATIC_CHECK_FIELDS
        if field != "overall"
    }
    automatic_checks["overall"] = "passed"
    preparation = {
        "schema": subject.PREPARATION_SCHEMA,
        "status": "ready_for_new_ue_import",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "created_at": "2026-07-28T00:00:00+00:00",
        "canonical_identity": {
            "asset_id": asset_id,
            "legacy_tag": asset_id,
            "tag": tag,
            "profile_schema_id": profile,
            "profile_sha256": "2" * 64,
            "request_sha256": job["request_sha256"],
            "taxonomy": {"species": "horse", "breed": "bay_horse"},
            "fixed_attributes": {},
            "sampled_attributes": sampled,
            "target_physical_profile": target_physical_profile,
        },
        "source_asset": _descriptor(source_asset),
        "source_asset_registry": _descriptor(source_registry),
        "source_asset_registry_sha256": source_registry_payload["registry_sha256"],
        "expected_source_asset_registry_file_sha256": _sha(source_registry),
        "source_asset_registry_validation_mode": "current_exact_rebuild",
        "source_artifact_roots": {"new_animal_assets": str(tmp_path.resolve())},
        "authenticated_source_artifact_count": 1,
        "animation_review": _descriptor(review),
        "animation_review_schema": review_schema,
        "animation_decision": _descriptor(decision_path),
        "expected_animation_decision_file_sha256": _sha(decision_path),
        "animation_decision_sha256": decision["decision_sha256"],
        "animation_decision_freeze_receipt": _descriptor(freeze_receipt_path),
        "expected_animation_decision_freeze_receipt_file_sha256": _sha(
            freeze_receipt_path
        ),
        "animation_decision_freeze_receipt_sha256": freeze_receipt["receipt_sha256"],
        "user_instruction_authority": copy.deepcopy(subject.USER_INSTRUCTION_AUTHORITY),
        "reviewed_animated_glb": _descriptor(runtime),
        "authenticated_review_artifact_count": 1,
        "ue_import_jobs": _relative_descriptor(jobs_path, tmp_path),
        "automatic_checks": automatic_checks,
    }
    preparation["manifest_sha256"] = _hash_without(preparation, "manifest_sha256")
    preparation_path = _write(
        tmp_path / "preparation.json",
        preparation,
    )
    mesh_dir = f"/Game/MyAssets/Audioset/Meshes/gate_{tag}"
    skeletal_mesh = f"{mesh_dir}/SK_gate_{tag}.SK_gate_{tag}"
    idle_animation = f"{mesh_dir}/Idle.Idle"
    walking_animation = f"{mesh_dir}/Walking.Walking"
    assets = sorted([skeletal_mesh, idle_animation, walking_animation])
    preparation_descriptor = _descriptor(preparation_path)
    preparation_descriptor["manifest_sha256"] = preparation["manifest_sha256"]
    input_descriptor = _descriptor(jobs_path)
    input_descriptor["batch_sha256"] = jobs["batch_sha256"]
    result_path = _write(
        tmp_path / "result.json",
        {
            "schema": subject.FORMAL_RESULT_SCHEMA,
            "generated_at": "2026-07-28T00:01:00+00:00",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "preparation_manifest": preparation_descriptor,
            "input_manifest": input_descriptor,
            "batch_identity": {
                "schema": subject.FORMAL_BATCH_SCHEMA,
                "job_type": subject.JOB_TYPE,
                "job_count": 1,
                "batch_sha256": jobs["batch_sha256"],
                "asset_ids": [asset_id],
                "tags": [tag],
                "job_identity_sha256s": [_json_hash(job)],
            },
            "non_destructive_policy": policy,
            "status": "passed",
            "passed_count": 1,
            "results": [
                {
                    "job_type": subject.JOB_TYPE,
                    "asset_id": asset_id,
                    "legacy_tag": asset_id,
                    "tag": tag,
                    "job_identity_sha256": _json_hash(job),
                    "source": str(runtime.resolve()),
                    "source_sha256": _sha(runtime),
                    "mesh_content_dir": mesh_dir,
                    "skeletal_mesh": skeletal_mesh,
                    "walking_animation": walking_animation,
                    "blueprint": (
                        f"/Game/MyAssets/Audioset/Blueprints/gate_{tag}/BP_gate_{tag}"
                    ),
                    "asset_count": len(assets),
                    "assets": assets,
                    "actions": subject.EXPECTED_ACTIONS,
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
            "rig_direction_check_windows": [
                {"frame_a": 0, "frame_b": 1, "label": "start"}
            ],
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
        "ue_preparation": preparation_path,
        "expected_ue_preparation_sha256": _sha(preparation_path),
        "expected_ue_jobs_sha256": _sha(jobs_path),
        "expected_ue_result_sha256": _sha(result_path),
        "expected_animation_decision_sha256": _sha(decision_path),
        "animation_decision_freeze_receipt": freeze_receipt_path,
        "expected_animation_decision_freeze_receipt_sha256": _sha(freeze_receipt_path),
        "template": template_path,
        "semantic_evidence": semantic_evidence,
    }


def _legacy_fixture(
    tmp_path: Path,
    *,
    review_schema: str = "avengine_target_native_generated_quadruped_review_run_v3",
) -> dict[str, Any]:
    inputs = _fixture(
        tmp_path,
        review_schema=review_schema,
    )
    if review_schema == subject.LEGACY_UNGATED_GENERATED_REVIEW_SCHEMA:
        decision = json.loads(inputs["animation_decision"].read_text(encoding="utf-8"))
        review_path = Path(decision["review"]["path"])
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review.pop("automatic_admission_gates")
        _write(review_path, review)
        decision["review"] = _descriptor(review_path)
        decision["review_sha256"] = decision["review"]["sha256"]
        decision["decision_sha256"] = _decision_hash(decision)
        _write(inputs["animation_decision"], decision)
    formal_jobs = json.loads(inputs["ue_jobs"].read_text(encoding="utf-8"))
    formal_job = formal_jobs["jobs"][0]
    legacy_job = {
        field: copy.deepcopy(formal_job[field])
        for field in (
            "asset_id",
            "legacy_tag",
            "tag",
            "profile_schema_id",
            "sampled_attributes",
            "expected_actions",
            "rigged_glb",
            "rigged_glb_sha256",
        )
    }
    _write(
        inputs["ue_jobs"],
        {
            "schema": subject.LEGACY_BATCH_SCHEMA,
            "jobs": [legacy_job],
            "non_destructive_policy": "historical fixture",
        },
    )
    _write(
        inputs["ue_result"],
        {
            "schema": subject.LEGACY_RESULT_SCHEMA,
            "input_manifest": str(inputs["ue_jobs"].resolve()),
            "passed_count": 1,
            "results": [
                {
                    "legacy_tag": legacy_job["asset_id"],
                    "tag": legacy_job["tag"],
                    "source_sha256": legacy_job["rigged_glb_sha256"],
                    "actions": legacy_job["expected_actions"],
                    "status": "passed",
                }
            ],
        },
    )
    inputs["expected_ue_jobs_sha256"] = _sha(inputs["ue_jobs"])
    inputs["expected_ue_result_sha256"] = _sha(inputs["ue_result"])
    inputs["expected_animation_decision_sha256"] = _sha(inputs["animation_decision"])
    return inputs


def test_builds_authenticated_walk_idle_pair(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    output_root = tmp_path / "output"
    semantic_evidence = inputs.pop("semantic_evidence")
    manifest_path = subject.build_specs(**inputs, output_root=output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == subject.OUTPUT_SCHEMA
    assert manifest["avatar_count"] == 1
    assert manifest["clip_count"] == 2
    assert manifest["manifest_sha256"] == contracts.manifest_sha256(manifest)
    assert manifest["formal_registration_authorized"] is False
    record = manifest["records"][0]
    actions = record["actions"]
    assert set(actions) == {"Walking", "Idle"}
    walking = json.loads(Path(actions["Walking"]["spec"]).read_text(encoding="utf-8"))
    idle = json.loads(Path(actions["Idle"]["spec"]).read_text(encoding="utf-8"))
    source = walking["sources"][0]
    assert source["target_physical_profile"] == _target_physical_profile()
    assert idle["sources"][0]["target_physical_profile"] == _target_physical_profile()
    assert record["target_physical_profile"] == _target_physical_profile()
    assert record["rig_semantic_evidence"] == {
        "schema": subject.RIG_SEMANTIC_EVIDENCE_SCHEMA,
        "kind": "motion_aware_weight_repair",
        "artifact": _descriptor(semantic_evidence),
        "semantic_schema": subject.WEIGHT_REPAIR_SCHEMA,
        "source_glb_sha256": _sha(Path(record["source_glb"]["path"])),
    }
    assert source["walking_forward_yaw_offset_deg"] == 90.0
    assert (
        source["controlled_animal_gate"]["status"]
        == "approved_for_research_candidate_apartment"
    )
    assert (
        source["controlled_animal_gate"]["formal_dataset_registration_authorized"]
        is False
    )
    assert "rig_direction_check_windows" in walking
    assert "rig_direction_check_windows" not in idle
    assert idle["sources"][0]["trajectory_m"] == [[2.0, 0.0, 0.0]] * 5


def test_builds_from_v4_fallback_b_final_semantic_artifact(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path, weight_repair_branch="fallback_a_b")
    semantic_evidence = inputs.pop("semantic_evidence")

    manifest_path = subject.build_specs(
        **inputs,
        output_root=tmp_path / "output",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["records"][0]["rig_semantic_evidence"]["artifact"] == (
        _descriptor(semantic_evidence)
    )


def test_builds_from_v4_not_needed_retarget_semantic_artifact(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path, weight_repair_branch="not_needed")
    semantic_evidence = inputs.pop("semantic_evidence")

    manifest_path = subject.build_specs(
        **inputs,
        output_root=tmp_path / "output",
    )
    evidence = json.loads(manifest_path.read_text(encoding="utf-8"))["records"][0][
        "rig_semantic_evidence"
    ]

    assert evidence["kind"] == "bone_name_independent_retarget"
    assert evidence["artifact"] == _descriptor(semantic_evidence)
    assert evidence["semantic_schema"] == subject.RETARGET_SCHEMA


@pytest.mark.parametrize(
    ("review_schema", "semantic_status"),
    [
        (
            "avengine_target_native_generated_quadruped_review_run_v1",
            "legacy_v1_unavailable",
        ),
        (
            "avengine_target_native_generated_quadruped_review_run_v2",
            "authenticated",
        ),
        (
            "avengine_target_native_generated_quadruped_review_run_v3",
            "authenticated",
        ),
    ],
)
def test_legacy_v1_to_v3_inputs_are_audit_only_and_cannot_publish(
    tmp_path: Path,
    review_schema: str,
    semantic_status: str,
) -> None:
    inputs = _legacy_fixture(tmp_path, review_schema=review_schema)
    inputs.pop("semantic_evidence")
    audit = subject.audit_inputs(
        config_path=inputs["config_path"],
        ue_jobs=inputs["ue_jobs"],
        ue_result=inputs["ue_result"],
        animation_decision=inputs["animation_decision"],
    )
    assert audit["mode"] == "legacy_audit_only"
    assert audit["review_schema"] == review_schema
    assert audit["rig_semantic_evidence_status"] == semantic_status
    assert audit["formal_apartment_publication_authorized"] is False

    with pytest.raises(contracts.ContractError, match="audit-only"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_v2_result_cannot_reuse_a_v3_review_for_formal_publication(
    tmp_path: Path,
) -> None:
    inputs = _fixture(
        tmp_path,
        review_schema="avengine_target_native_generated_quadruped_review_run_v3",
    )
    inputs.pop("semantic_evidence")

    with pytest.raises(contracts.ContractError, match="exact approved v4 review"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_rejects_mixed_v2_batch_and_v1_result_schema(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    result = json.loads(inputs["ue_result"].read_text(encoding="utf-8"))
    result["schema"] = subject.LEGACY_RESULT_SCHEMA
    _write(inputs["ue_result"], result)
    _repin(inputs, "ue_result")

    with pytest.raises(contracts.ContractError, match="mixed|downgraded"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_rejects_v2_result_that_claims_idle_without_idle_object(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    result = json.loads(inputs["ue_result"].read_text(encoding="utf-8"))
    imported = result["results"][0]
    imported["assets"] = sorted(
        value.replace("/Idle.Idle", "/Other.Other") for value in imported["assets"]
    )
    _write(inputs["ue_result"], result)
    _repin(inputs, "ue_result")

    with pytest.raises(contracts.ContractError, match="object readback"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("preparation_manifest_self_hash", "preparation contract"),
        ("result_input_batch_sha", "batch self-hash"),
        ("result_batch_identity", "batch identity"),
        ("decision_file_bytes", "decision"),
        ("runtime_glb_bytes", "runtime GLB hash"),
    ],
)
def test_rejects_tamper_across_every_formal_v2_identity_segment(
    tmp_path: Path,
    tamper: str,
    error: str,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    result = json.loads(inputs["ue_result"].read_text(encoding="utf-8"))
    write_result = False
    if tamper == "preparation_manifest_self_hash":
        preparation_path = Path(result["preparation_manifest"]["path"])
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        preparation["manifest_sha256"] = "f" * 64
        _write(preparation_path, preparation)
        descriptor = _descriptor(preparation_path)
        descriptor["manifest_sha256"] = "f" * 64
        result["preparation_manifest"] = descriptor
        write_result = True
    elif tamper == "result_input_batch_sha":
        result["input_manifest"]["batch_sha256"] = "f" * 64
        write_result = True
    elif tamper == "result_batch_identity":
        result["batch_identity"]["job_identity_sha256s"] = ["f" * 64]
        write_result = True
    elif tamper == "decision_file_bytes":
        decision_path = inputs["animation_decision"]
        decision_path.write_text(
            decision_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    elif tamper == "runtime_glb_bytes":
        jobs = json.loads(inputs["ue_jobs"].read_text(encoding="utf-8"))
        Path(jobs["jobs"][0]["rigged_glb"]).write_bytes(b"tampered GLB")
    else:
        raise AssertionError(f"unhandled tamper case: {tamper}")
    if write_result:
        _write(inputs["ue_result"], result)
        _repin(inputs, "ue_result")
    if tamper == "preparation_manifest_self_hash":
        _repin(inputs, "ue_preparation")

    with pytest.raises(contracts.ContractError, match=error):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_generated_dog_pair_embeds_pinned_audio_contract() -> None:
    trajectory = [[float(index), 0.0, 0.0] for index in range(5)]
    template = {
        "sources": [
            {
                "tag": "template",
                "trajectory_m": trajectory,
                "start_pos_m": trajectory[0],
                "end_pos_m": trajectory[-1],
            }
        ],
        "camera_pass_table_loop_contract": {"left_front_nearest_frame": 2},
        "rig_direction_check_windows": [{"frame_a": 0, "frame_b": 1}],
    }
    config = {
        "asset_id": "corgi_generated_001",
        "tag": "pixal_generated_corgi_001",
        "species": "dog",
        "breed": "pembroke_welsh_corgi",
        "profile_schema_id": "dog_corgi_v1",
        "sampled_attributes": {"size": "medium"},
        "target_physical_profile": _target_physical_profile(),
        "walking_forward_yaw_offset_deg": 0.0,
        "actor_scale": 0.1,
        "ground_snap_max_abs_correction_cm": 25.0,
        "audio_lookup": "dog_bark",
        "audio_source_height_offset_m": 0.28,
        "scale_rationale": "fixture",
    }

    source = subject._build_pair(
        template,
        config=config,
        gate={"status": "approved_for_research_candidate_apartment"},
    )["Walking"]["sources"][0]

    assert source["audio_contract"]["audio_lookup"] == "dog_bark"
    assert source["audio_sha256"] == source["audio_contract"]["sha256"]
    assert source["audio_source_channels"] == 1


def test_refuses_changed_decision_and_existing_output(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    output_root = tmp_path / "output"
    subject.build_specs(**inputs, output_root=output_root)
    with pytest.raises(contracts.ContractError, match="refusing to replace output"):
        subject.build_specs(**inputs, output_root=output_root)

    changed = json.loads(inputs["animation_decision"].read_text(encoding="utf-8"))
    changed["decision"] = "rejected"
    inputs["animation_decision"].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(contracts.ContractError, match="decision"):
        subject.build_specs(**inputs, output_root=tmp_path / "second")


def test_refuses_missing_or_inconsistent_target_physical_profile(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    config = json.loads(inputs["config_path"].read_text(encoding="utf-8"))
    config.pop("target_physical_profile")
    _write(inputs["config_path"], config)
    with pytest.raises(contracts.ContractError, match="config fields"):
        subject.build_specs(**inputs, output_root=tmp_path / "missing")

    inputs = _fixture(tmp_path / "second")
    inputs.pop("semantic_evidence")
    config = json.loads(inputs["config_path"].read_text(encoding="utf-8"))
    config["target_physical_profile"]["selected_value"] = "large"
    _write(inputs["config_path"], config)
    with pytest.raises(contracts.ContractError, match="target_physical_profile"):
        subject.build_specs(**inputs, output_root=tmp_path / "mismatched")


def test_refuses_config_taxonomy_not_bound_by_preparation(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    config = json.loads(inputs["config_path"].read_text(encoding="utf-8"))
    config["species"] = "zebra"
    config["breed"] = "british_shorthair"
    _write(inputs["config_path"], config)

    with pytest.raises(contracts.ContractError, match="taxonomy"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_refuses_changed_hash_bound_rig_semantic_evidence(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    semantic_evidence = inputs.pop("semantic_evidence")
    semantic_evidence.write_text("{}\n", encoding="utf-8")

    with pytest.raises(contracts.ContractError, match="missing or changed"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("actor_scale", float("nan")),
        ("actor_scale", True),
        ("walking_forward_yaw_offset_deg", float("inf")),
        ("ground_snap_max_abs_correction_cm", float("-inf")),
        ("audio_source_height_offset_m", False),
    ],
)
def test_refuses_nonfinite_or_boolean_runtime_numbers(
    tmp_path: Path, field: str, bad_value
) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    config = json.loads(inputs["config_path"].read_text(encoding="utf-8"))
    config[field] = bad_value
    _write(inputs["config_path"], config)

    with pytest.raises(contracts.ContractError, match="non-finite|finite number"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_refuses_nonfinite_numbers_anywhere_in_json_inputs(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    jobs = json.loads(inputs["ue_jobs"].read_text(encoding="utf-8"))
    jobs["unused_diagnostic"] = {"score": float("nan")}
    _write(inputs["ue_jobs"], jobs)
    _repin(inputs, "ue_jobs")

    with pytest.raises(contracts.ContractError, match="non-finite.*number"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_refuses_duplicate_json_contract_keys(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    jobs_text = inputs["ue_jobs"].read_text(encoding="utf-8")
    jobs_text = jobs_text.replace(
        f'"schema": "{subject.FORMAL_BATCH_SCHEMA}",',
        (
            f'"schema": "{subject.FORMAL_BATCH_SCHEMA}",\n'
            f'  "schema": "{subject.FORMAL_BATCH_SCHEMA}",'
        ),
        1,
    )
    inputs["ue_jobs"].write_text(jobs_text, encoding="utf-8")
    _repin(inputs, "ue_jobs")

    with pytest.raises(contracts.ContractError, match="duplicate JSON object key"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


@pytest.mark.parametrize(
    ("path_key", "error"),
    [
        ("ue_preparation", "external UE preparation SHA-256"),
        ("ue_jobs", "external UE jobs SHA-256"),
        ("ue_result", "external UE result SHA-256"),
        ("animation_decision", "external animation decision SHA-256"),
        (
            "animation_decision_freeze_receipt",
            "external animation decision freeze receipt SHA-256",
        ),
    ],
)
def test_formal_publication_requires_all_external_full_file_sha_anchors(
    tmp_path: Path,
    path_key: str,
    error: str,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    path = inputs[path_key]
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(contracts.ContractError, match=error):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_legacy_preparation_v1_is_never_formal(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    preparation = contracts.load_json(inputs["ue_preparation"])
    preparation["schema"] = subject.LEGACY_PREPARATION_SCHEMA
    _write_preparation_and_rebind_result(inputs, preparation)

    with pytest.raises(contracts.ContractError, match="v1 is audit-only"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_apartment_cannot_upgrade_caller_assertion_to_crypto_identity(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    receipt = contracts.load_json(inputs["animation_decision_freeze_receipt"])
    receipt["user_instruction_authority"]["cryptographic_user_identity_verified"] = True
    receipt["receipt_sha256"] = _hash_without(receipt, "receipt_sha256")
    _write(inputs["animation_decision_freeze_receipt"], receipt)
    _repin(inputs, "animation_decision_freeze_receipt")

    preparation = contracts.load_json(inputs["ue_preparation"])
    preparation["animation_decision_freeze_receipt"] = _descriptor(
        inputs["animation_decision_freeze_receipt"]
    )
    preparation["expected_animation_decision_freeze_receipt_file_sha256"] = _sha(
        inputs["animation_decision_freeze_receipt"]
    )
    preparation["animation_decision_freeze_receipt_sha256"] = receipt["receipt_sha256"]
    preparation["user_instruction_authority"] = copy.deepcopy(
        receipt["user_instruction_authority"]
    )
    _write_preparation_and_rebind_result(inputs, preparation)

    with pytest.raises(contracts.ContractError, match="authority contract"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


@pytest.mark.parametrize(
    "path_key",
    (
        "ue_jobs",
        "ue_result",
        "animation_decision",
        "ue_preparation",
        "animation_decision_freeze_receipt",
    ),
)
def test_formal_consumer_rejects_duplicate_keys_at_every_top_level_anchor(
    tmp_path: Path,
    path_key: str,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    target = inputs[path_key]
    _inject_duplicate_schema(target)
    _repin(inputs, path_key)
    if path_key == "ue_preparation":
        result = contracts.load_json(inputs["ue_result"])
        descriptor = _descriptor(target)
        descriptor["manifest_sha256"] = json.loads(target.read_text(encoding="utf-8"))[
            "manifest_sha256"
        ]
        result["preparation_manifest"] = descriptor
        _write(inputs["ue_result"], result)
        _repin(inputs, "ue_result")
    elif path_key == "animation_decision_freeze_receipt":
        preparation = contracts.load_json(inputs["ue_preparation"])
        preparation["animation_decision_freeze_receipt"] = _descriptor(target)
        preparation["expected_animation_decision_freeze_receipt_file_sha256"] = _sha(
            target
        )
        _write_preparation_and_rebind_result(inputs, preparation)

    with pytest.raises(contracts.ContractError, match="duplicate JSON object key"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_synchronized_internal_rehash_and_fake_glb_cannot_bypass_external_pins(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    semantic_path = inputs.pop("semantic_evidence")
    jobs = contracts.load_json(inputs["ue_jobs"])
    job = jobs["jobs"][0]
    runtime = Path(job["rigged_glb"])
    runtime.write_bytes(b"attacker-controlled fake GLB")

    semantic = contracts.load_json(semantic_path)
    semantic["output"] = _descriptor(runtime)
    _write(semantic_path, semantic)

    decision = contracts.load_json(inputs["animation_decision"])
    review_path = Path(decision["review"]["path"])
    review = contracts.load_json(review_path)
    final_artifact = review["automatic_admission_gates"]["weight_repair_final_artifact"]
    for name in {
        "animated_glb",
        "retargeted_animated_glb",
        final_artifact["glb_output_descriptor"],
    }:
        review["outputs"][name] = _descriptor(runtime)
    for name in {
        "weight_repair_manifest",
        final_artifact["manifest_output_descriptor"],
    }:
        if name is not None:
            review["outputs"][name] = _descriptor(semantic_path)
    _write(review_path, review)

    decision["review"] = _descriptor(review_path)
    decision["review_sha256"] = _sha(review_path)
    decision["decision_sha256"] = _decision_hash(decision)
    _write(inputs["animation_decision"], decision)

    receipt = contracts.load_json(inputs["animation_decision_freeze_receipt"])
    receipt["animation_review"] = _descriptor(review_path)
    receipt["expected_animation_review_file_sha256"] = _sha(review_path)
    receipt["user_instruction_binding"]["review_sha256"] = _sha(review_path)
    receipt["animation_decision"] = _relative_descriptor(
        inputs["animation_decision"],
        inputs["animation_decision_freeze_receipt"].parent,
    )
    receipt["decision_sha256"] = decision["decision_sha256"]
    receipt["receipt_sha256"] = _hash_without(receipt, "receipt_sha256")
    _write(inputs["animation_decision_freeze_receipt"], receipt)

    job["rigged_glb_sha256"] = _sha(runtime)
    job["animation_decision_file_sha256"] = _sha(inputs["animation_decision"])
    job["animation_decision_sha256"] = decision["decision_sha256"]
    jobs["batch_sha256"] = _hash_without(jobs, "batch_sha256")
    _write(inputs["ue_jobs"], jobs)

    preparation = contracts.load_json(inputs["ue_preparation"])
    preparation["animation_review"] = _descriptor(review_path)
    preparation["animation_decision"] = _descriptor(inputs["animation_decision"])
    preparation["expected_animation_decision_file_sha256"] = _sha(
        inputs["animation_decision"]
    )
    preparation["animation_decision_sha256"] = decision["decision_sha256"]
    preparation["animation_decision_freeze_receipt"] = _descriptor(
        inputs["animation_decision_freeze_receipt"]
    )
    preparation["expected_animation_decision_freeze_receipt_file_sha256"] = _sha(
        inputs["animation_decision_freeze_receipt"]
    )
    preparation["animation_decision_freeze_receipt_sha256"] = receipt["receipt_sha256"]
    preparation["reviewed_animated_glb"] = _descriptor(runtime)
    preparation["ue_import_jobs"] = _relative_descriptor(
        inputs["ue_jobs"],
        inputs["ue_preparation"].parent,
    )
    preparation["manifest_sha256"] = _hash_without(
        preparation,
        "manifest_sha256",
    )
    _write(inputs["ue_preparation"], preparation)

    result = contracts.load_json(inputs["ue_result"])
    preparation_descriptor = _descriptor(inputs["ue_preparation"])
    preparation_descriptor["manifest_sha256"] = preparation["manifest_sha256"]
    result["preparation_manifest"] = preparation_descriptor
    input_descriptor = _descriptor(inputs["ue_jobs"])
    input_descriptor["batch_sha256"] = jobs["batch_sha256"]
    result["input_manifest"] = input_descriptor
    result["batch_identity"]["batch_sha256"] = jobs["batch_sha256"]
    result["batch_identity"]["job_identity_sha256s"] = [_json_hash(job)]
    result["results"][0]["job_identity_sha256"] = _json_hash(job)
    result["results"][0]["source_sha256"] = _sha(runtime)
    _write(inputs["ue_result"], result)

    assert decision["decision_sha256"] == _decision_hash(decision)
    assert receipt["receipt_sha256"] == _hash_without(receipt, "receipt_sha256")
    assert jobs["batch_sha256"] == _hash_without(jobs, "batch_sha256")
    assert preparation["manifest_sha256"] == _hash_without(
        preparation,
        "manifest_sha256",
    )
    assert result["preparation_manifest"]["sha256"] == _sha(inputs["ue_preparation"])
    assert result["input_manifest"]["sha256"] == _sha(inputs["ue_jobs"])
    assert result["results"][0]["source_sha256"] == _sha(runtime)

    with pytest.raises(
        contracts.ContractError,
        match="external UE preparation SHA-256",
    ):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


@pytest.mark.parametrize("bad_value", [True, 1.0])
def test_refuses_noninteger_ue_pass_count(tmp_path: Path, bad_value) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    result = json.loads(inputs["ue_result"].read_text(encoding="utf-8"))
    result["passed_count"] = bad_value
    _write(inputs["ue_result"], result)
    _repin(inputs, "ue_result")

    with pytest.raises(contracts.ContractError, match="coverage is incomplete"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


@pytest.mark.parametrize("location", ["trajectory_component", "render_fps"])
def test_refuses_boolean_template_numeric_fields(tmp_path: Path, location: str) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    template = json.loads(inputs["template"].read_text(encoding="utf-8"))
    if location == "trajectory_component":
        template["sources"][0]["trajectory_m"][0][0] = True
        error = "finite number"
    else:
        template["render_config"]["fps"] = True
        error = "integer"
    _write(inputs["template"], template)

    with pytest.raises(contracts.ContractError, match=error):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_refuses_boolean_left_front_frame(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    template = json.loads(inputs["template"].read_text(encoding="utf-8"))
    template["camera_pass_table_loop_contract"]["left_front_nearest_frame"] = True
    _write(inputs["template"], template)

    with pytest.raises(contracts.ContractError, match="integer"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_refuses_noncanonical_physical_measurement_name(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    config = json.loads(inputs["config_path"].read_text(encoding="utf-8"))
    config["target_physical_profile"]["measurement"] = "withers_height_cm"
    _write(inputs["config_path"], config)

    with pytest.raises(contracts.ContractError, match="target_physical_profile"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_horse_apartment_config_uses_pipeline_canonical_measurement() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "controlled_source_attributes_v1"
        / "candidate_profiles"
        / "animal"
        / "horse_compact_tail_pixal_v4_r8_apartment_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert (
        config["target_physical_profile"]["measurement"]
        == subject.PHYSICAL_MEASUREMENT
        == "shoulder_height_cm"
    )
    assert (
        config["target_physical_profile"]["profile_id"]
        == "horse_bay_shoulder_height_physical_candidate_v1"
    )
    authoritative_profile_path = (
        config_path.parents[2]
        / "profiles"
        / "animal"
        / "horse_bay_native_action_composite_side_clay_v4.json"
    )
    authoritative_profile = json.loads(
        authoritative_profile_path.read_text(encoding="utf-8")
    )["target_physical_profiles"]
    assert authoritative_profile["measurement"] == "withers_height_cm"
    assert (
        authoritative_profile["profile_id"]
        != config["target_physical_profile"]["profile_id"]
    )


def test_artifacts_and_descriptors_reject_symlink_files(tmp_path: Path) -> None:
    target = _write(tmp_path / "target.json", {"status": "passed"})
    link = tmp_path / "link.json"
    link.symlink_to(target)
    descriptor = _descriptor(target)
    descriptor["path"] = str(link)

    with pytest.raises(contracts.ContractError, match="symlink"):
        subject._artifact(link)
    assert subject._descriptor_matches(descriptor) is False


def test_artifacts_and_descriptors_reject_symlink_parent_directories(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    target = _write(real / "target.json", {"status": "passed"})
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    aliased_target = alias / "target.json"
    descriptor = _descriptor(target)
    descriptor["path"] = str(aliased_target)

    with pytest.raises(contracts.ContractError, match="symlink"):
        subject._artifact(aliased_target)
    assert subject._descriptor_matches(descriptor) is False


def test_builder_rejects_symlink_config_before_resolve(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    config_link = tmp_path / "config-link.json"
    config_link.symlink_to(inputs["config_path"])
    inputs["config_path"] = config_link

    with pytest.raises(contracts.ContractError, match="symlink"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_builder_rejects_symlink_runtime_glb_before_resolve(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    jobs = json.loads(inputs["ue_jobs"].read_text(encoding="utf-8"))
    runtime = Path(jobs["jobs"][0]["rigged_glb"])
    runtime_link = tmp_path / "runtime-link.glb"
    runtime_link.symlink_to(runtime)
    jobs["jobs"][0]["rigged_glb"] = str(runtime_link)
    _write(inputs["ue_jobs"], jobs)
    _repin(inputs, "ue_jobs")

    with pytest.raises(contracts.ContractError, match="batch|identity|path"):
        subject.build_specs(**inputs, output_root=tmp_path / "output")


def test_formal_external_input_and_output_reject_arbitrary_symlink_parents(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.pop("semantic_evidence")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    inputs["ue_jobs"] = alias / inputs["ue_jobs"].name

    with pytest.raises(contracts.ContractError, match="symlink path component"):
        subject.build_specs(**inputs, output_root=tmp_path / "input_rejected")

    inputs = _fixture(tmp_path / "output_case")
    inputs.pop("semantic_evidence")
    output_real = tmp_path / "output_real"
    output_real.mkdir()
    output_alias = tmp_path / "output_alias"
    output_alias.symlink_to(output_real, target_is_directory=True)
    with pytest.raises(contracts.ContractError, match="symlink path component"):
        subject.build_specs(
            **inputs,
            output_root=output_alias / "publication",
        )


def test_exact_spear_tmp_bridge_is_allowed_but_nested_symlinks_are_not(
    tmp_path: Path,
    monkeypatch,
) -> None:
    physical = tmp_path / "physical"
    inputs = _fixture(physical)
    inputs.pop("semantic_evidence")
    bridge = tmp_path / "tmp"
    bridge.symlink_to(physical, target_is_directory=True)
    monkeypatch.setattr(subject, "SPEAR_TMP_BRIDGE", bridge.absolute())
    inputs["ue_jobs"] = bridge / inputs["ue_jobs"].name

    manifest = subject.build_specs(
        **inputs,
        output_root=bridge / "bridged_output",
    )
    assert manifest.is_file()

    inputs = _fixture(tmp_path / "nested_case")
    inputs.pop("semantic_evidence")
    nested_bridge = tmp_path / "nested_tmp"
    nested_bridge.symlink_to(
        tmp_path / "nested_case",
        target_is_directory=True,
    )
    monkeypatch.setattr(subject, "SPEAR_TMP_BRIDGE", nested_bridge.absolute())
    inside_alias = tmp_path / "nested_case" / "inside_alias"
    inside_alias.symlink_to(
        tmp_path / "nested_case",
        target_is_directory=True,
    )
    inputs["ue_jobs"] = nested_bridge / "inside_alias" / "jobs.json"
    with pytest.raises(contracts.ContractError, match="symlink path component"):
        subject.build_specs(
            **inputs,
            output_root=nested_bridge / "nested_rejected",
        )
