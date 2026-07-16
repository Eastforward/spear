"""Static and pure-math contracts for the generic Rocketbox-to-TokenRig retargeter."""

from __future__ import annotations

import math
import hashlib
import json
import shutil
import subprocess
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import blender_retarget_rocketbox_to_tokenrig as runner


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "blender_retarget_rocketbox_to_tokenrig.py"
)


def test_runner_exists_before_any_blender_contract_is_exercised():
    assert SCRIPT.is_file(), "Task 6 retarget runner has not been implemented"


def test_cli_consumes_only_the_task6_authenticated_inputs():
    values = {
        "--asset-id": "property_0042_person_a",
        "--base-avatar-id": "rocketbox_male_adult_01",
        "--bind-pose-glb": "/approved/static/bind_pose.glb",
        "--static-qa-json": "/approved/static/static_qa.json",
        "--baseline-retarget-blend": "/sealed/male/retarget.blend",
        "--baseline-retarget-manifest": "/sealed/male/retarget_manifest.json",
        "--idle-motion-fbx": "/rocketbox/m_idle_neutral_01.max.fbx",
        "--motion-basis-selection": "/approved/review/selection/retarget_motion_basis_correction_v1.json",
        "--motion-basis-review-manifest": "/approved/review/motion_basis_review_manifest.json",
        "--output-dir": "/new/animation",
    }
    argv = [item for pair in values.items() for item in pair]

    args = runner.parse_args(argv)

    assert args.asset_id == "property_0042_person_a"
    assert args.base_avatar_id == "rocketbox_male_adult_01"
    assert args.bind_pose_glb == Path(values["--bind-pose-glb"])
    assert args.static_qa_json == Path(values["--static-qa-json"])
    assert args.baseline_retarget_blend == Path(values["--baseline-retarget-blend"])
    assert args.baseline_retarget_manifest == Path(values["--baseline-retarget-manifest"])
    assert args.idle_motion_fbx == Path(values["--idle-motion-fbx"])
    assert args.motion_basis_selection == Path(values["--motion-basis-selection"])
    assert args.motion_basis_review_manifest == Path(
        values["--motion-basis-review-manifest"]
    )
    assert args.output_dir == Path(values["--output-dir"])


def test_contract_names_only_two_actions_and_keeps_axis_identity():
    assert runner.ACTION_NAMES == {
        "walk": "Walking",
        "idle": "Standing_Idle",
    }
    assert runner.CANONICAL_FRONT == "negative-y"
    assert runner.CANONICAL_UP == "positive-z"
    assert runner.AXIS_MAP_3X3 == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert runner.MAXIMUM_PENETRATION_M == 0.010


def _semantic_mapping(spines=("bone_spine_a", "bone_spine_b", "bone_spine_c")):
    semantic_bones = {
        "pelvis": "bone_pelvis",
        "spine": list(spines),
        "neck": "bone_neck",
        "head": "bone_head",
        "left_clavicle": "bone_l_clavicle",
        "left_upper_arm": "bone_l_upper_arm",
        "left_forearm": "bone_l_forearm",
        "left_hand": "bone_l_hand",
        "right_clavicle": "bone_r_clavicle",
        "right_upper_arm": "bone_r_upper_arm",
        "right_forearm": "bone_r_forearm",
        "right_hand": "bone_r_hand",
        "left_thigh": "bone_l_thigh",
        "left_calf": "bone_l_calf",
        "left_foot": "bone_l_foot",
        "left_toe": "bone_l_toe",
        "right_thigh": "bone_r_thigh",
        "right_calf": "bone_r_calf",
        "right_foot": "bone_r_foot",
        "right_toe": "bone_r_toe",
    }
    return {
        "method": "unique_topology_and_canonical_rest_position_v1",
        "side_basis": {"left": "positive-x", "right": "negative-x"},
        "semantic_bones": semantic_bones,
        "chains": {
            "axial": [
                semantic_bones["pelvis"],
                *semantic_bones["spine"],
                semantic_bones["neck"],
                semantic_bones["head"],
            ],
            "left_arm": [
                semantic_bones["left_clavicle"],
                semantic_bones["left_upper_arm"],
                semantic_bones["left_forearm"],
                semantic_bones["left_hand"],
            ],
            "right_arm": [
                semantic_bones["right_clavicle"],
                semantic_bones["right_upper_arm"],
                semantic_bones["right_forearm"],
                semantic_bones["right_hand"],
            ],
            "left_leg": [
                semantic_bones["left_thigh"],
                semantic_bones["left_calf"],
                semantic_bones["left_foot"],
                semantic_bones["left_toe"],
            ],
            "right_leg": [
                semantic_bones["right_thigh"],
                semantic_bones["right_calf"],
                semantic_bones["right_foot"],
                semantic_bones["right_toe"],
            ],
        },
        "ignored_proven_distal_descendants": ["bone_l_finger", "bone_r_finger"],
        "ignored_proven_head_descendants": ["bone_hat"],
    }


@pytest.mark.parametrize("spine_count", [2, 3, 4])
def test_semantic_contract_accepts_generic_two_three_or_four_bone_spines(spine_count):
    mapping = _semantic_mapping(
        tuple(f"generic_spine_{index}" for index in range(spine_count))
    )

    result = runner.validate_semantic_mapping(mapping)

    assert result["semantic_bones"]["spine"] == mapping["semantic_bones"]["spine"]
    assert result["target_bone_names"] == sorted(
        {
            value
            for role, value in mapping["semantic_bones"].items()
            for value in (value if role == "spine" else [value])
        }
    )
    assert result["rest_descendants"] == [
        "bone_hat",
        "bone_l_finger",
        "bone_r_finger",
    ]
    assert result["head_bound_descendants"] == ["bone_hat"]
    assert result["hand_bound_descendants"] == ["bone_l_finger", "bone_r_finger"]


def test_semantic_contract_rejects_ambiguous_or_inconsistent_target_names():
    mapping = _semantic_mapping()
    mapping["semantic_bones"]["right_hand"] = mapping["semantic_bones"]["left_hand"]
    with pytest.raises(runner.RetargetError, match="duplicate|ambiguous"):
        runner.validate_semantic_mapping(mapping)

    mapping = _semantic_mapping()
    mapping["chains"]["axial"][1] = "different_spine"
    with pytest.raises(runner.RetargetError, match="axial|semantic"):
        runner.validate_semantic_mapping(mapping)


def _z_points(*values):
    return [(0.0, 0.0, float(value)) for value in values]


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)
    return path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path):
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path, payload):
    return _write(path, json.dumps(payload, sort_keys=True) + "\n")


def _motion_basis_approval(tmp_path, *, asset_id="rocketbox_male_adult_01"):
    review_root = tmp_path / "review"
    artifact = _write(review_root / "yaw_000" / "walking.glb", b"reviewed walk")
    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    review = {
        "schema": "shared_limb_motion_basis_review_v1",
        "asset_id": asset_id,
        "automatic_checks": "all_candidates_generated_and_hash_locked",
        "candidates": {
            "yaw_000": {
                "yaw_degrees": 0,
                "matrix_3x3": matrix,
                "artifacts": {
                    "walking.glb": {
                        "filename": "yaw_000/walking.glb",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": _sha256(artifact),
                    }
                },
                "metrics": {
                    "asset_id": asset_id,
                    "candidate_id": "yaw_000",
                    "canonical_front": "negative-y",
                    "canonical_up": "positive-z",
                    "overall_classification": "four_limb_sagittal_motion",
                    "anatomical_axial_pose_gate": {
                        "schema": "anatomical_axial_pose_gate_v1",
                        "automatic_checks": "passed",
                        "overall_classification": (
                            "axial_pose_within_source_motion_envelope"
                        ),
                    },
                    "anatomical_axial_transfer": {
                        "schema": "tokenrig_anatomical_axial_body_transfer_v1",
                        "automatic_checks": "passed",
                    },
                    "shared_limb_motion_basis": {
                        "schema": "tokenrig_shared_canonical_limb_motion_basis_v1",
                        "automatic_checks": "passed",
                        "motion_basis_3x3": matrix,
                    },
                },
            }
        },
    }
    review_path = _write_json(review_root / "motion_basis_review_manifest.json", review)
    selection_root = tmp_path / "selection"
    selection = {
        "schema": "retarget_motion_basis_correction_v1",
        "asset_id": asset_id,
        "candidate_bundle_manifest_sha256": _sha256(review_path),
        "candidate_id": "yaw_000",
        "canonical_front": "negative-y",
        "canonical_up": "positive-z",
        "decision": "selected_for_next_retarget",
        "formal_dataset_asset": False,
        "matrix_3x3": matrix,
        "reviewed_at": "2026-07-12T05:23:36+00:00",
        "reviewer": "human-reviewer",
        "scope": "bilateral_arm_and_leg_chains_only",
        "yaw_degrees": 0,
    }
    selection_path = _write_json(
        selection_root / "retarget_motion_basis_correction_v1.json", selection
    )
    artifact.chmod(0o444)
    review_path.chmod(0o444)
    selection_path.chmod(0o444)
    return SimpleNamespace(
        artifact=artifact,
        review=review,
        review_path=review_path,
        selection=selection,
        selection_path=selection_path,
    )


def test_motion_basis_approval_authenticates_reviewed_four_limb_and_axial_result(
    tmp_path,
):
    approval = _motion_basis_approval(tmp_path)

    result = runner.authenticate_motion_basis_selection(
        base_avatar_id="rocketbox_male_adult_01",
        motion_basis_selection=approval.selection_path,
        motion_basis_review_manifest=approval.review_path,
    )

    assert result["candidate_id"] == "yaw_000"
    assert result["matrix_3x3"] == pytest.approx(np.eye(3), abs=1.0e-12)
    assert result["four_limb_classification"] == "four_limb_sagittal_motion"
    assert result["axial_classification"] == (
        "axial_pose_within_source_motion_envelope"
    )
    assert result["selected_artifacts"]["walking.glb"]["sha256"] == _sha256(
        approval.artifact
    )


def test_motion_basis_approval_rejects_stale_review_manifest(tmp_path):
    approval = _motion_basis_approval(tmp_path)
    approval.review_path.chmod(0o644)
    payload = json.loads(approval.review_path.read_text(encoding="utf-8"))
    payload["automatic_checks"] = "changed_after_approval"
    _write_json(approval.review_path, payload)
    approval.review_path.chmod(0o444)

    with pytest.raises(runner.RetargetError, match="stale review manifest"):
        runner.authenticate_motion_basis_selection(
            base_avatar_id="rocketbox_male_adult_01",
            motion_basis_selection=approval.selection_path,
            motion_basis_review_manifest=approval.review_path,
        )


def test_motion_basis_approval_rejects_non_sagittal_candidate(tmp_path):
    approval = _motion_basis_approval(tmp_path)
    approval.review_path.chmod(0o644)
    payload = json.loads(approval.review_path.read_text(encoding="utf-8"))
    payload["candidates"]["yaw_000"]["metrics"]["overall_classification"] = (
        "not_sagittal"
    )
    _write_json(approval.review_path, payload)
    approval.review_path.chmod(0o444)
    approval.selection_path.chmod(0o644)
    selection = json.loads(approval.selection_path.read_text(encoding="utf-8"))
    selection["candidate_bundle_manifest_sha256"] = _sha256(approval.review_path)
    _write_json(approval.selection_path, selection)
    approval.selection_path.chmod(0o444)

    with pytest.raises(runner.RetargetError, match="four-limb sagittal"):
        runner.authenticate_motion_basis_selection(
            base_avatar_id="rocketbox_male_adult_01",
            motion_basis_selection=approval.selection_path,
            motion_basis_review_manifest=approval.review_path,
        )


def _static_bundle(tmp_path, *, asset_id="route2_male"):
    root = tmp_path / "route2" / "static_audit"
    bind = _write(root / "bind_pose.glb", b"authenticated canonical bind pose")
    mapping = _semantic_mapping()
    qa = {
        "schema": runner.STATIC_QA_SCHEMA,
        "asset_id": asset_id,
        "decision": runner.STATIC_QA_DECISION,
        "authenticated": {"task3_manifest_sha256": "a" * 64},
        "checks": {
            "semantic_mapping": mapping,
            "axis_canonicalization": {
                "source_front": "positive-y",
                "canonical_front": "negative-y",
                "yaw_radians": math.pi,
                "transform_count": 1,
                "matrix": [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "canonical_front_vector": [0.0, -1.0, 0.0],
                "canonical_up_vector": [0.0, 0.0, 1.0],
                "determinant": 1.0,
            },
            "grounding": {
                "canonical_floor_z": 0.0,
                "post_floor_z": 0.0,
                "transform_count": 1,
            },
            "canonical_mesh_contract": {
                "vertex_count": 24,
                "polygon_count": 12,
                "uv_layer_count": 1,
                "material_slot_count": 1,
                "bind_mesh_sha256": "b" * 64,
            },
            "exported_pbr": {
                "passed": True,
                "source": {"slot:0/base_color": {"sha256": "c" * 64}},
            },
        },
        "artifacts": {"bind_pose.glb": _record(bind)},
    }
    qa_path = _write_json(root / "static_qa.json", qa)
    return SimpleNamespace(root=root, bind=bind, qa=qa, qa_path=qa_path)


def _input_contract(tmp_path):
    baseline_root = tmp_path / "sealed" / "rocketbox_neutral_walk_v1"
    rocketbox_root = tmp_path / "rocketbox"
    male_idle = _write(
        rocketbox_root
        / "Assets/Animations/all_animations_max_motextr_static/m_idle_neutral_01.max.fbx",
        b"exact male idle",
    )
    female_idle = _write(
        rocketbox_root
        / "Assets/Animations/all_animations_max_motextr_static/f_idle_neutral_01.max.fbx",
        b"exact female idle",
    )
    return runner.RetargetInputContract(
        baseline_root=baseline_root,
        baseline_manifest_sha256="",
        baseline_manifest_size=0,
        rocketbox_root=rocketbox_root,
        rocketbox_commit="d" * 40,
        idle_by_baseline_asset={
            "rocketbox_male_adult_01": runner.IdleMotionContract(
                relative_path=male_idle.relative_to(rocketbox_root),
                sha256=_sha256(male_idle),
                size_bytes=male_idle.stat().st_size,
                git_blob_sha1="e" * 40,
            ),
            "rocketbox_female_adult_01": runner.IdleMotionContract(
                relative_path=female_idle.relative_to(rocketbox_root),
                sha256=_sha256(female_idle),
                size_bytes=female_idle.stat().st_size,
                git_blob_sha1="f" * 40,
            ),
        },
    )


def _sealed_baseline(contract, baseline_asset_id="rocketbox_male_adult_01"):
    asset_root = contract.baseline_root / baseline_asset_id
    blend = _write(asset_root / "retarget.blend", b"sealed walk blend")
    retarget_manifest = {
        "schema_version": "rocketbox_retarget_manifest_v1",
        "asset_id": baseline_asset_id,
        "stage": "retargeted",
        "source_animation": {
            "fps": 30,
            "frame_start": 1,
            "frame_end": 33,
            "frame_count": 33,
        },
        "automatic_checks": {"overall": "passed"},
        "artifacts": {"blend": "retarget.blend", "glb": "retarget.glb"},
    }
    manifest_path = _write_json(asset_root / "retarget_manifest.json", retarget_manifest)
    root_manifest = {
        "schema_version": "rocketbox_baseline_manifest_v1",
        "baseline_id": "rocketbox_neutral_walk_v1",
        "motion": "walk_neutral",
        "artifact_allowlist": ["retarget.blend", "retarget_manifest.json"],
        "assets": {
            baseline_asset_id: {
                "files": {
                    "retarget.blend": {
                        "sha256": _sha256(blend),
                        "size": blend.stat().st_size,
                    },
                    "retarget_manifest.json": {
                        "sha256": _sha256(manifest_path),
                        "size": manifest_path.stat().st_size,
                    },
                }
            }
        },
    }
    root_manifest_path = _write_json(contract.baseline_root / "baseline_manifest.json", root_manifest)
    return SimpleNamespace(
        blend=blend,
        manifest=manifest_path,
        root_manifest=root_manifest_path,
        contract=runner.RetargetInputContract(
            baseline_root=contract.baseline_root,
            baseline_manifest_sha256=_sha256(root_manifest_path),
            baseline_manifest_size=root_manifest_path.stat().st_size,
            rocketbox_root=contract.rocketbox_root,
            rocketbox_commit=contract.rocketbox_commit,
            idle_by_baseline_asset=contract.idle_by_baseline_asset,
        ),
    )


def test_static_gate_authenticates_decision_hash_axis_floor_and_generic_map(tmp_path):
    static = _static_bundle(tmp_path)

    result = runner.authenticate_static_gate(
        asset_id="route2_male",
        bind_pose_glb=static.bind,
        static_qa_json=static.qa_path,
    )

    assert result["bind_pose"]["sha256"] == _sha256(static.bind)
    assert result["static_qa"]["sha256"] == _sha256(static.qa_path)
    assert result["floor_z_m"] == 0.0
    assert result["axis_map_3x3"] == [list(row) for row in runner.AXIS_MAP_3X3]
    assert result["semantic_mapping"]["semantic_bones"]["head"] == "bone_head"


def test_static_gate_rejects_stale_bind_failed_decision_or_symlink(tmp_path):
    static = _static_bundle(tmp_path)
    static.bind.write_bytes(b"tampered after review")
    with pytest.raises(runner.RetargetError, match="bind_pose.*SHA-256"):
        runner.authenticate_static_gate(
            asset_id="route2_male",
            bind_pose_glb=static.bind,
            static_qa_json=static.qa_path,
        )

    static = _static_bundle(tmp_path / "second")
    static.qa["decision"] = "rejected"
    _write_json(static.qa_path, static.qa)
    with pytest.raises(runner.RetargetError, match="static.*decision"):
        runner.authenticate_static_gate(
            asset_id="route2_male",
            bind_pose_glb=static.bind,
            static_qa_json=static.qa_path,
        )

    static = _static_bundle(tmp_path / "third")
    alias = static.root / "alias.glb"
    alias.symlink_to(static.bind)
    static.qa["artifacts"]["bind_pose.glb"] = {
        **_record(static.bind),
        "filename": "alias.glb",
    }
    _write_json(static.qa_path, static.qa)
    with pytest.raises(runner.RetargetError, match="symlink|exact.*bind_pose"):
        runner.authenticate_static_gate(
            asset_id="route2_male",
            bind_pose_glb=alias,
            static_qa_json=static.qa_path,
        )


def test_static_gate_rejects_a_noncanonical_up_vector_or_axis_matrix(tmp_path):
    static = _static_bundle(tmp_path)
    static.qa["checks"]["axis_canonicalization"]["canonical_up_vector"] = [0.0, 1.0, 0.0]
    _write_json(static.qa_path, static.qa)
    with pytest.raises(runner.RetargetError, match="axis|UP"):
        runner.authenticate_static_gate(
            asset_id="route2_male",
            bind_pose_glb=static.bind,
            static_qa_json=static.qa_path,
        )

    static = _static_bundle(tmp_path / "matrix")
    static.qa["checks"]["axis_canonicalization"]["matrix"][0][0] = 1.0
    _write_json(static.qa_path, static.qa)
    with pytest.raises(runner.RetargetError, match="axis|matrix"):
        runner.authenticate_static_gate(
            asset_id="route2_male",
            bind_pose_glb=static.bind,
            static_qa_json=static.qa_path,
        )


def test_sealed_walk_authenticates_root_manifest_and_exact_gender_files(tmp_path):
    baseline = _sealed_baseline(_input_contract(tmp_path))

    result = runner.authenticate_sealed_walk(
        base_avatar_id="rocketbox_male_adult_01",
        baseline_retarget_blend=baseline.blend,
        baseline_retarget_manifest=baseline.manifest,
        contract=baseline.contract,
    )

    assert result["base_avatar_id"] == "rocketbox_male_adult_01"
    assert result["retarget_blend"]["sha256"] == _sha256(baseline.blend)
    assert result["retarget_manifest"]["sha256"] == _sha256(baseline.manifest)
    assert result["source_animation"] == {
        "fps": 30,
        "frame_start": 1,
        "frame_end": 33,
        "frame_count": 33,
    }


def test_sealed_walk_rejects_manifest_hash_or_noncanonical_path(tmp_path):
    baseline = _sealed_baseline(_input_contract(tmp_path))
    baseline.manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(runner.RetargetError, match="retarget manifest.*SHA-256"):
        runner.authenticate_sealed_walk(
            base_avatar_id="rocketbox_male_adult_01",
            baseline_retarget_blend=baseline.blend,
            baseline_retarget_manifest=baseline.manifest,
            contract=baseline.contract,
        )

    baseline = _sealed_baseline(_input_contract(tmp_path / "other"))
    copied = _write(tmp_path / "copied.blend", baseline.blend.read_bytes())
    with pytest.raises(runner.RetargetError, match="canonical sealed.*retarget.blend"):
        runner.authenticate_sealed_walk(
            base_avatar_id="rocketbox_male_adult_01",
            baseline_retarget_blend=copied,
            baseline_retarget_manifest=baseline.manifest,
            contract=baseline.contract,
        )


class _GitRunner:
    def __init__(self, contract, blob):
        self.contract = contract
        self.blob = blob
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(([str(item) for item in command], kwargs))
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout=self.contract.rocketbox_commit + "\n", stderr="")
        if "hash-object" in command:
            return SimpleNamespace(returncode=0, stdout=self.blob + "\n", stderr="")
        raise AssertionError(command)


def test_idle_authentication_pins_gender_path_hash_size_commit_and_blob(tmp_path):
    baseline = _sealed_baseline(_input_contract(tmp_path))
    idle_contract = baseline.contract.idle_by_baseline_asset["rocketbox_male_adult_01"]
    idle_path = baseline.contract.rocketbox_root / idle_contract.relative_path
    git = _GitRunner(baseline.contract, idle_contract.git_blob_sha1)

    result = runner.authenticate_idle_motion(
        base_avatar_id="rocketbox_male_adult_01",
        idle_motion_fbx=idle_path,
        contract=baseline.contract,
        subprocess_runner=git,
    )

    assert result["sha256"] == idle_contract.sha256
    assert result["git_commit"] == baseline.contract.rocketbox_commit
    assert result["git_blob_sha1"] == idle_contract.git_blob_sha1
    assert len(git.calls) == 2


def test_idle_authentication_rejects_cross_gender_or_wrong_checkout_commit(tmp_path):
    baseline = _sealed_baseline(_input_contract(tmp_path))
    female = baseline.contract.idle_by_baseline_asset["rocketbox_female_adult_01"]
    female_path = baseline.contract.rocketbox_root / female.relative_path
    git = _GitRunner(baseline.contract, female.git_blob_sha1)
    with pytest.raises(runner.RetargetError, match="exact gender idle"):
        runner.authenticate_idle_motion(
            base_avatar_id="rocketbox_male_adult_01",
            idle_motion_fbx=female_path,
            contract=baseline.contract,
            subprocess_runner=git,
        )

    male = baseline.contract.idle_by_baseline_asset["rocketbox_male_adult_01"]
    male_path = baseline.contract.rocketbox_root / male.relative_path
    wrong = _GitRunner(
        runner.RetargetInputContract(
            **{
                **baseline.contract.__dict__,
                "rocketbox_commit": "0" * 40,
            }
        ),
        male.git_blob_sha1,
    )
    with pytest.raises(runner.RetargetError, match="Rocketbox checkout commit"):
        runner.authenticate_idle_motion(
            base_avatar_id="rocketbox_male_adult_01",
            idle_motion_fbx=male_path,
            contract=baseline.contract,
            subprocess_runner=wrong,
        )


def test_unique_instance_asset_id_is_independent_from_the_base_avatar_contract(tmp_path):
    contract = _input_contract(tmp_path)

    assert runner.validate_base_avatar_id(
        asset_id="property_0042_person_a",
        base_avatar_id="rocketbox_male_adult_01",
        contract=contract,
    ) == "rocketbox_male_adult_01"
    with pytest.raises(runner.RetargetError, match="base_avatar_id"):
        runner.validate_base_avatar_id(
            asset_id="property_0042_person_a",
            base_avatar_id="property_0042_person_a",
            contract=contract,
        )
    source = SCRIPT.read_text(encoding="utf-8")
    run_source = source[source.index("def run_retarget(") :]
    assert "base_avatar_id: str" in run_source
    assert "validate_base_avatar_id(" in run_source


def test_sealed_walk_must_match_the_explicit_base_avatar_id(tmp_path):
    baseline = _sealed_baseline(_input_contract(tmp_path))
    with pytest.raises(runner.RetargetError, match="base_avatar_id"):
        runner.authenticate_sealed_walk(
            base_avatar_id="rocketbox_female_adult_01",
            baseline_retarget_blend=baseline.blend,
            baseline_retarget_manifest=baseline.manifest,
            contract=baseline.contract,
        )


@pytest.mark.parametrize(
    "target_names,target_points,expected",
    [
        (
            ("t0", "t1"),
            _z_points(0, 4),
            [((0, 1), (1.0, 0.0)), ((1, 2), (0.0, 1.0))],
        ),
        (
            ("t0", "t1", "t2"),
            _z_points(0, 2, 4),
            [((0, 1), (1.0, 0.0)), ((1, 2), (0.75, 0.25)), ((1, 2), (0.0, 1.0))],
        ),
        (
            ("t0", "t1", "t2", "t3"),
            _z_points(0, 1, 3, 6),
            [
                ((0, 1), (1.0, 0.0)),
                ((0, 1), (0.5, 0.5)),
                ((1, 2), (0.75, 0.25)),
                ((1, 2), (0.0, 1.0)),
            ],
        ),
    ],
)
def test_spine_plan_resamples_two_three_and_four_target_bones_by_rest_arc(
    target_names, target_points, expected
):
    plan = runner.build_spine_resample_plan(
        source_bones=runner.ROCKETBOX_SPINE_BONES,
        source_rest_heads=_z_points(0, 1, 3),
        target_bones=target_names,
        target_rest_heads=target_points,
    )

    assert [record["target_bone"] for record in plan] == list(target_names)
    assert [tuple(record["source_indices"]) for record in plan] == [
        item[0] for item in expected
    ]
    assert np.allclose(
        [record["weights"] for record in plan],
        [item[1] for item in expected],
        atol=1.0e-12,
    )
    assert plan[0]["target_normalized_arc"] == 0.0
    assert plan[-1]["target_normalized_arc"] == 1.0
    assert all(
        record["interpolation_domain"] == "cumulative_parent_to_child_rotation"
        for record in plan
    )


def test_spine_plan_rejects_degenerate_or_ambiguous_rest_arcs():
    with pytest.raises(runner.RetargetError, match="ambiguous.*source.*arc"):
        runner.build_spine_resample_plan(
            source_bones=runner.ROCKETBOX_SPINE_BONES,
            source_rest_heads=_z_points(0, 0, 1),
            target_bones=("t0", "t1"),
            target_rest_heads=_z_points(0, 1),
        )
    with pytest.raises(runner.RetargetError, match="degenerate.*target.*arc"):
        runner.build_spine_resample_plan(
            source_bones=runner.ROCKETBOX_SPINE_BONES,
            source_rest_heads=_z_points(0, 1, 2),
            target_bones=("t0", "t1", "t2"),
            target_rest_heads=_z_points(4, 4, 4),
        )


def _rz(degrees):
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def _rx(degrees):
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.array(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    )


def test_rest_corrected_pose_delta_never_copies_absolute_source_pose():
    source_rest = _rz(70)
    source_pose = _rz(100)
    target_rest = _rz(-25)

    target_pose = runner.rest_corrected_local_rotation(
        source_rest, source_pose, target_rest
    )

    assert np.allclose(target_pose, _rz(5), atol=1.0e-12)
    assert not np.allclose(target_pose, source_pose, atol=1.0e-6)


def test_global_rest_alignment_conjugates_delta_into_target_rest_frame():
    source_rest = _rz(70)
    source_pose = _rz(100)
    target_rest = _rz(-25)

    target_pose, evidence = runner.rest_aligned_global_rotation(
        source_rest, source_pose, target_rest
    )

    expected_alignment = target_rest @ source_rest.T
    expected_source_delta = source_pose @ source_rest.T
    expected = (
        expected_alignment
        @ expected_source_delta
        @ expected_alignment.T
        @ target_rest
    )
    assert np.allclose(target_pose, expected, atol=1.0e-12)
    assert np.allclose(evidence["source_to_target_rest_alignment"], expected_alignment)
    assert np.allclose(evidence["source_global_pose_delta"], expected_source_delta)


def test_distal_canonical_world_delta_preserves_surface_pitch_across_unrelated_bone_axes():
    source_base = _rz(90)
    target_base = _rz(180)
    source_rest = _rx(18) @ _rz(37)
    target_rest = _rx(-71) @ _rz(-22)
    canonical_delta = _rx(19)
    source_pose = source_base.T @ canonical_delta @ source_base @ source_rest

    target_pose, evidence = runner.canonical_world_delta_rotation(
        source_rest,
        source_pose,
        target_rest,
        source_base_rotation_3x3=source_base,
        target_base_rotation_3x3=target_base,
    )

    reconstructed = (
        target_base @ target_pose @ (target_base @ target_rest).T
    )
    assert np.allclose(reconstructed, canonical_delta, atol=1.0e-12)
    target_surface_in_bone = np.asarray((0.0, 0.0, -1.0))
    rest_surface = target_base @ target_rest @ target_surface_in_bone
    posed_surface = target_base @ target_pose @ target_surface_in_bone
    assert np.allclose(posed_surface, canonical_delta @ rest_surface, atol=1.0e-12)
    assert evidence["method"] == "canonical_world_distal_delta_v1"
    assert evidence["target_rest_axis_conjugation_used"] is False
    assert evidence["canonical_delta_reconstruction_error"] < 1.0e-12


def test_distal_canonical_world_delta_rejects_reflected_base_and_reconstructs_loop():
    identity = np.eye(3)
    first, _ = runner.canonical_world_delta_rotation(
        identity,
        identity,
        _rz(33),
        source_base_rotation_3x3=identity,
        target_base_rotation_3x3=_rz(180),
    )
    last, _ = runner.canonical_world_delta_rotation(
        identity,
        identity,
        _rz(33),
        source_base_rotation_3x3=identity,
        target_base_rotation_3x3=_rz(180),
    )
    assert np.allclose(first, last, atol=1.0e-12)
    with pytest.raises(runner.RetargetError, match="handedness|reflection"):
        runner.canonical_world_delta_rotation(
            identity,
            identity,
            identity,
            source_base_rotation_3x3=np.diag((-1.0, 1.0, 1.0)),
            target_base_rotation_3x3=identity,
        )


def test_shared_canonical_limb_delta_keeps_one_world_axis_across_unrelated_rest_axes():
    source_base = _rz(90)
    target_base = _rz(180)
    source_rest = _rx(18) @ _rz(37)
    target_rest = _rx(-71) @ _rz(-22)
    canonical_delta = _rx(19)
    source_pose = source_base.T @ canonical_delta @ source_base @ source_rest

    target_pose, evidence = runner.shared_canonical_limb_rotation(
        source_rest,
        source_pose,
        target_rest,
        source_base_rotation_3x3=source_base,
        target_base_rotation_3x3=target_base,
        motion_basis_3x3=np.eye(3),
    )

    reconstructed = target_base @ target_pose @ (target_base @ target_rest).T
    assert reconstructed == pytest.approx(canonical_delta, abs=1.0e-12)
    assert evidence["method"] == "shared_canonical_limb_delta_v1"
    assert evidence["per_bone_rest_axis_conjugation_used"] is False
    assert evidence["motion_basis_3x3"] == pytest.approx(np.eye(3), abs=1.0e-12)
    assert evidence["canonical_delta_reconstruction_error"] < 1.0e-12


def test_shared_canonical_limb_delta_conjugates_one_reviewed_basis_for_every_limb():
    canonical_delta = _rx(23)
    correction = _rz(90)
    source_pose = canonical_delta

    target_pose, evidence = runner.shared_canonical_limb_rotation(
        np.eye(3),
        source_pose,
        _rz(-31),
        source_base_rotation_3x3=np.eye(3),
        target_base_rotation_3x3=np.eye(3),
        motion_basis_3x3=correction,
    )

    reconstructed = target_pose @ _rz(-31).T
    expected = correction @ canonical_delta @ correction.T
    assert reconstructed == pytest.approx(expected, abs=1.0e-12)
    assert evidence["corrected_canonical_delta_3x3"] == pytest.approx(
        expected, abs=1.0e-12
    )


def test_shared_canonical_limb_delta_rejects_a_reflected_review_basis():
    with pytest.raises(runner.RetargetError, match="handedness|reflection"):
        runner.shared_canonical_limb_rotation(
            np.eye(3),
            np.eye(3),
            np.eye(3),
            source_base_rotation_3x3=np.eye(3),
            target_base_rotation_3x3=np.eye(3),
            motion_basis_3x3=np.diag((-1.0, 1.0, 1.0)),
        )


def test_production_bake_uses_one_shared_canonical_basis_for_all_limb_chains():
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index("def bake_rest_corrected_action(")
    bake = source[start : source.index("\ndef _remove_everything_except", start)]
    expected_roles = {
        "left_upper_arm",
        "left_forearm",
        "left_hand",
        "right_upper_arm",
        "right_forearm",
        "right_hand",
        "left_thigh",
        "left_calf",
        "left_foot",
        "left_toe",
        "right_thigh",
        "right_calf",
        "right_foot",
        "right_toe",
    }

    assert set(runner.SHARED_CANONICAL_LIMB_ROLES) == expected_roles
    assert "limb_motion_basis_3x3=AXIS_MAP_3X3" in bake
    assert "shared_canonical_limb_rotation(" in bake
    assert "if semantic_role in SHARED_CANONICAL_LIMB_ROLES" in bake
    assert "axis_map_3x3=limb_motion_basis" in bake
    assert "build_anatomical_axial_transfer(" in bake
    assert "semantic_role in ANATOMICAL_AXIAL_EXACT_ROLES" in bake
    assert "semantic_role in ANATOMICAL_CLAVICLE_ROLES" in bake
    assert "rest_aligned_global_rotation(" not in bake
    root_mapping = bake[bake.index("root_delta =") : bake.index("axial_frame =")]
    assert "limb_motion_basis" not in root_mapping
    assert '"shared_limb_motion_basis"' in bake
    assert '"anatomical_axial_transfer"' in bake


def test_anatomical_frame_lateral_primary_preserves_hip_chord_and_is_proper():
    frame = runner.anatomical_frame_from_points(
        left=(1.0, 0.0, 0.0),
        right=(-1.0, 0.0, 0.0),
        lower=(0.0, 0.0, 0.0),
        upper=(0.25, 0.0, 1.0),
        primary_axis="lateral",
        description="synthetic pelvis",
    )

    assert frame[:, 0] == pytest.approx((-1.0, 0.0, 0.0), abs=1.0e-12)
    assert frame[:, 1] == pytest.approx((0.0, -1.0, 0.0), abs=1.0e-12)
    assert frame[:, 2] == pytest.approx((0.0, 0.0, 1.0), abs=1.0e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1.0e-12)
    assert frame.T @ frame == pytest.approx(np.eye(3), abs=1.0e-12)


def test_anatomical_frame_up_primary_preserves_neck_head_chord_and_is_proper():
    up = np.asarray((0.2, -0.1, 1.0), dtype=np.float64)
    up /= np.linalg.norm(up)
    frame = runner.anatomical_frame_from_points(
        left=(1.0, 0.0, 0.0),
        right=(-1.0, 0.1, 0.2),
        lower=(0.0, 0.0, 0.0),
        upper=(0.2, -0.1, 1.0),
        primary_axis="up",
        description="synthetic neck",
    )

    assert frame[:, 2] == pytest.approx(up, abs=1.0e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1.0e-12)
    assert frame.T @ frame == pytest.approx(np.eye(3), abs=1.0e-12)


def test_anatomical_frame_rejects_degenerate_or_unknown_primary_axis():
    arguments = {
        "left": (0.0, 0.0, 0.0),
        "right": (0.0, 0.0, 0.0),
        "lower": (0.0, 0.0, 0.0),
        "upper": (0.0, 0.0, 1.0),
        "description": "bad frame",
    }
    with pytest.raises(runner.RetargetError, match="lateral chord"):
        runner.anatomical_frame_from_points(
            **arguments, primary_axis="lateral"
        )
    with pytest.raises(runner.RetargetError, match="primary axis"):
        runner.anatomical_frame_from_points(
            **{**arguments, "right": (1.0, 0.0, 0.0)}, primary_axis="other"
        )


def test_real_rocketbox_rest_drift_projects_to_nearest_so3_with_audited_context():
    # Captured verbatim from the sealed male retarget.blend on Blender 4.2.1:
    # Bip01 R Clavicle bone.matrix_local.to_3x3().  It is right-handed but its
    # accumulated float serialization drift exceeds the former exact 1e-8 gate.
    real_matrix = np.asarray(
        [
            [-0.012585473246872425, -0.9988877177238464, 0.0454423651099205],
            [-0.9749764204025269, 0.02234623022377491, 0.22118902206420898],
            [-0.22195862233638763, -0.041521597653627396, -0.9741716980934143],
        ],
        dtype=np.float64,
    )
    context = {
        "action": "Walking",
        "frame": 1,
        "semantic_role": "right_clavicle",
        "source_bone": "Bip01 R Clavicle",
        "target_bone": "bone_25",
        "matrix_stage": "source_global_rest",
    }

    projected, evidence = runner.project_near_rotation(
        real_matrix,
        "source global rest",
        context=context,
    )

    assert np.allclose(projected.T @ projected, np.eye(3), atol=1.0e-12)
    assert math.isclose(float(np.linalg.det(projected)), 1.0, abs_tol=1.0e-12)
    assert evidence["method"] == "svd_polar_nearest_so3_v1"
    assert evidence["context"] == context
    assert evidence["input_handedness"] == "right"
    assert math.isclose(
        evidence["input_determinant"], 1.0000016017980289, abs_tol=1.0e-12
    )
    assert math.isclose(
        evidence["input_orthogonality_max_error"],
        3.0445072396467765e-06,
        abs_tol=1.0e-15,
    )
    assert evidence["maximum_singular_value_deviation"] < 2.0e-6
    assert evidence["polar_residual_max_abs"] < 2.0e-6
    assert evidence["projection_applied"] is True


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.diag((-1.0, 1.0, 1.0)), "handedness|reflection"),
        (np.diag((1.0, 1.0, 0.0)), "degenerate|singular"),
        (np.diag((1.0001, 1.0, 1.0)), "near-rotation bounds"),
        (np.asarray(((1.0, 0.01, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))), "near-rotation bounds"),
    ],
)
def test_rotation_projection_rejects_reflection_degeneracy_and_real_scale_shear(
    matrix, message
):
    context = {
        "action": "Walking",
        "frame": 17,
        "semantic_role": "right_clavicle",
        "source_bone": "Bip01 R Clavicle",
        "target_bone": "bone_25",
        "matrix_stage": "source_global_pose",
    }
    with pytest.raises(runner.RetargetError, match=message) as captured:
        runner.project_near_rotation(matrix, "source global pose", context=context)
    text = str(captured.value)
    assert "Walking" in text
    assert "frame=17" in text
    assert "right_clavicle" in text
    assert "Bip01 R Clavicle" in text
    assert "determinant=" in text
    assert "orthogonality_max_error=" in text
    assert "singular_values=" in text


def test_uniform_source_object_scale_is_separated_before_so3_projection():
    rotation = _rz(23.0)
    value = rotation @ np.diag((0.010000001, 0.010000002, 0.010000000))
    projected, evidence = runner.project_uniform_scaled_rotation(
        value,
        "source armature object transform",
        context={"action": "Walking", "frame": 1, "matrix_stage": "source_object"},
    )
    assert np.allclose(projected, rotation, atol=2.0e-7)
    assert math.isclose(evidence["uniform_scale"], 0.010000001, rel_tol=2.0e-7)
    assert evidence["normalized_rotation"]["input_handedness"] == "right"


def test_global_alignment_exposes_all_projection_evidence_and_context():
    source_rest = _rz(70) @ np.diag((1.000001, 1.0, 1.0))
    source_pose = _rz(100)
    target_rest = _rz(-25)
    context = {
        "action": "Walking",
        "frame": 4,
        "semantic_role": "pelvis",
        "source_bone": "Bip01 Pelvis",
        "target_bone": "bone_0",
    }
    _, evidence = runner.rest_aligned_global_rotation(
        source_rest,
        source_pose,
        target_rest,
        context=context,
    )
    assert set(evidence["rotation_projections"]) == {
        "source_global_rest",
        "source_global_pose",
        "target_global_rest",
    }
    assert all(
        record["context"]["action"] == "Walking"
        for record in evidence["rotation_projections"].values()
    )


def test_rotation_projection_summary_retains_per_bone_stage_and_worst_frame():
    records = []
    for frame, scale in ((1, 1.000001), (9, 1.000002)):
        _, record = runner.project_near_rotation(
            _rz(15) @ np.diag((scale, 1.0, 1.0)),
            "source global pose",
            context={
                "action": "Walking",
                "frame": frame,
                "semantic_role": "right_clavicle",
                "source_bone": "Bip01 R Clavicle",
                "target_bone": "bone_25",
                "matrix_stage": "source_global_pose",
            },
        )
        records.append(record)

    summary = runner.summarize_rotation_projections(records)

    assert summary["schema"] == "strict_rotation_projection_evidence_v1"
    assert summary["record_count"] == 2
    assert summary["context_group_count"] == 1
    assert summary["projection_applied_count"] == 2
    assert summary["maximum_input_orthogonality_error"] > 3.0e-6
    group = summary["per_bone_stage"][0]
    assert group["source_bone"] == "Bip01 R Clavicle"
    assert group["matrix_stage"] == "source_global_pose"
    assert group["sample_count"] == 2
    assert group["worst_frames"]["input_orthogonality_max_error"] == 9
    assert runner.validate_rotation_projection_summary(summary)["status"] == "passed"


def test_source_motion_cache_projects_raw_matrices_instead_of_blind_quaternion_casts():
    source = SCRIPT.read_text(encoding="utf-8")
    cache = source[source.index("def cache_source_motion(") : source.index("def _semantic_height(")]
    assert "project_near_rotation(" in cache
    assert "project_uniform_scaled_rotation(" in cache
    assert ".matrix.to_quaternion()" not in cache
    assert ".matrix_world.to_quaternion()" not in cache


def test_translation_preserves_target_rest_and_scales_only_source_delta():
    value = runner.scaled_target_translation(
        source_rest=(1.0, 2.0, 3.0),
        source_pose=(2.0, 0.0, 7.0),
        target_rest=(10.0, 20.0, 30.0),
        height_scale=0.5,
    )
    assert np.allclose(value, (10.5, 19.0, 32.0))


def test_root_and_pelvis_use_height_scales_in_their_respective_spaces():
    scales = runner.compute_height_scales(
        source_armature_height=180.0,
        target_armature_height=1.70,
        source_world_height=1.80,
        target_world_height=1.70,
    )
    assert math.isclose(scales["pelvis_local_scale"], 1.70 / 180.0)
    assert math.isclose(scales["root_world_scale"], 1.70 / 1.80)
    assert scales["pelvis_local_scale"] != scales["root_world_scale"]


def test_deterministic_stratified_mesh_plan_covers_support_and_lower_body_edges():
    positions = np.asarray(
        [(x / 9.0, y / 9.0, z / 3.0) for z in range(4) for y in range(10) for x in range(10)],
        dtype=np.float64,
    )
    edges = np.asarray(
        [(index, index + 1) for index in range(len(positions) - 1)],
        dtype=np.int64,
    )
    weights = []
    for index, position in enumerate(positions):
        if position[2] == 0.0 and position[0] < 0.5:
            weights.append({"left_foot": 1.0})
        elif position[2] == 0.0:
            weights.append({"right_foot": 1.0})
        elif position[2] < 0.67:
            weights.append({"pelvis": 0.5, "left_thigh": 0.5})
        else:
            weights.append({"head": 1.0})
    semantic = {
        "pelvis": "pelvis",
        "left_thigh": "left_thigh",
        "left_calf": "left_calf",
        "left_foot": "left_foot",
        "left_toe": "left_toe",
        "right_thigh": "right_thigh",
        "right_calf": "right_calf",
        "right_foot": "right_foot",
        "right_toe": "right_toe",
    }

    first = runner.build_deterministic_mesh_sample_plan(
        rest_positions=positions,
        edges=edges,
        vertex_weights=weights,
        semantic_bones=semantic,
        maximum_global_vertices=48,
        maximum_support_vertices=24,
        maximum_edges=64,
    )
    second = runner.build_deterministic_mesh_sample_plan(
        rest_positions=positions,
        edges=edges,
        vertex_weights=weights,
        semantic_bones=semantic,
        maximum_global_vertices=48,
        maximum_support_vertices=24,
        maximum_edges=64,
    )

    assert first["evidence"] == second["evidence"]
    assert first["evidence"]["method"] == "deterministic_spatial_skin_support_core_v2"
    assert first["evidence"]["seed"] == runner.QUALITY_SAMPLE_SEED
    assert first["evidence"]["evaluated_vertex_count"] < len(positions)
    assert first["evidence"]["sampled_edge_count"] < len(edges)
    assert first["evidence"]["left_support_candidate_count"] > 0
    assert first["evidence"]["right_support_candidate_count"] > 0
    assert first["evidence"]["lower_body_edge_candidate_count"] > 0
    assert len(first["evidence"]["index_sha256"]) == 64
    assert set(first["support_vertex_indices"]) <= set(first["evaluation_vertex_indices"])


def test_support_core_excludes_opposite_trace_weights_and_nonsole_rest_regions():
    positions = np.asarray(
        [
            (0.10, 0.00, 0.000),
            (-0.10, 0.00, 0.001),
            (-0.11, 0.01, 0.010),
            (-0.10, 0.00, 0.200),
            (0.00, 0.00, 1.000),
            (0.00, 0.10, 0.500),
        ],
        dtype=np.float64,
    )
    edges = np.asarray(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)), dtype=np.int64)
    weights = [
        {"left_foot": 0.9999, "right_toe": 0.0001},
        {"right_toe": 0.99, "right_foot": 0.01},
        {"right_foot": 0.60, "left_foot": 0.40},
        {"right_toe": 0.99, "right_calf": 0.01},
        {"head": 1.0},
        {"pelvis": 1.0},
    ]
    semantic = {
        "pelvis": "pelvis",
        "left_thigh": "left_thigh",
        "left_calf": "left_calf",
        "left_foot": "left_foot",
        "left_toe": "left_toe",
        "right_thigh": "right_thigh",
        "right_calf": "right_calf",
        "right_foot": "right_foot",
        "right_toe": "right_toe",
    }

    plan = runner.build_deterministic_mesh_sample_plan(
        rest_positions=positions,
        edges=edges,
        vertex_weights=weights,
        semantic_bones=semantic,
        maximum_global_vertices=6,
        maximum_support_vertices=6,
        maximum_edges=5,
    )

    assert plan["evidence"]["method"] == "deterministic_spatial_skin_support_core_v2"
    assert plan["evidence"]["support_core_definition"] == {
        "minimum_combined_side_foot_toe_weight": 0.5,
        "dominant_bone_must_be_side_foot_or_toe": True,
        "combined_side_weight_must_exceed_opposite_side": True,
        "maximum_opposite_side_foot_toe_weight": 1.0e-4,
        "rest_lower_band_max_z_m": pytest.approx(0.05),
    }
    assert plan["evidence"]["right_any_nonzero_support_candidate_count"] == 4
    assert plan["evidence"]["right_combined_mass_candidate_count"] == 3
    assert plan["evidence"]["right_dominant_side_candidate_count"] == 3
    assert plan["evidence"]["right_support_candidate_count"] == 1
    assert set(plan["right_support_vertex_indices"]) == {1}
    assert 0 not in plan["right_support_vertex_indices"]
    assert 2 not in plan["right_support_vertex_indices"]
    assert 3 not in plan["right_support_vertex_indices"]
    assert set(plan["left_support_vertex_indices"]) == {0}
    assert len(plan["evidence"]["right_support_candidate_index_sha256"]) == 64
    assert plan["evidence"]["right_support_core_rest_bounds_m"] == {
        "minimum": [-0.1, 0.0, 0.001],
        "maximum": [-0.1, 0.0, 0.001],
    }

    forged = copy.deepcopy(plan)
    forged["right_support_vertex_indices"] = [0, 1, 2]
    with pytest.raises(runner.RetargetError, match="support core|forged|inconsistent"):
        runner.validate_deterministic_mesh_sample_plan_support_core(forged)


def test_source_support_core_is_topology_independent_reproducible_and_schema_distinct():
    positions = np.asarray(
        [
            (0.10, 0.00, 0.000),
            (0.10, 0.00, 0.000),
            (-0.10, 0.00, 0.001),
            (-0.11, 0.01, 0.010),
            (0.00, 0.00, 1.000),
            (0.00, 0.10, 0.500),
        ],
        dtype=np.float64,
    )
    weights = [
        {"left_foot": 1.0},
        {"left_toe": 1.0},
        {"right_toe": 0.99, "right_foot": 0.01},
        {"right_foot": 0.60, "left_foot": 0.40},
        {"head": 1.0},
        {"pelvis": 1.0},
    ]
    semantic = {
        "pelvis": "pelvis",
        "left_thigh": "left_thigh",
        "left_calf": "left_calf",
        "left_foot": "left_foot",
        "left_toe": "left_toe",
        "right_thigh": "right_thigh",
        "right_calf": "right_calf",
        "right_foot": "right_foot",
        "right_toe": "right_toe",
    }

    first = runner.build_deterministic_support_core_plan(
        rest_positions=positions,
        vertex_weights=weights,
        semantic_bones=semantic,
        maximum_support_vertices=6,
    )
    second = runner.build_deterministic_support_core_plan(
        rest_positions=positions,
        vertex_weights=weights,
        semantic_bones=semantic,
        maximum_support_vertices=6,
    )

    assert first["evidence"] == second["evidence"]
    assert first["evidence"]["method"] == "deterministic_skin_support_core_only_v2"
    assert first["evidence"]["scope"] == "source_contact_support_vertices_only"
    assert first["evidence"]["topology_edges_required"] is False
    assert len(first["evidence"]["index_sha256"]) == 64
    assert first["left_support_vertex_indices"] == [0, 1]
    assert first["right_support_vertex_indices"] == [2]
    assert first["support_vertex_indices"] == [0, 1, 2]
    target_plan = runner.build_deterministic_mesh_sample_plan(
        rest_positions=positions,
        edges=np.asarray(((0, 2),), dtype=np.int64),
        vertex_weights=weights,
        semantic_bones=semantic,
        maximum_global_vertices=6,
        maximum_support_vertices=6,
        maximum_edges=1,
    )
    assert first["left_support_vertex_indices"] == target_plan["left_support_vertex_indices"]
    assert first["right_support_vertex_indices"] == target_plan["right_support_vertex_indices"]
    assert (
        first["evidence"]["support_core_definition"]
        == target_plan["evidence"]["support_core_definition"]
    )
    runner.validate_deterministic_support_core_plan(first)

    forged = copy.deepcopy(first)
    forged["left_support_vertex_indices"] = [0]
    with pytest.raises(runner.RetargetError, match="support core|forged|inconsistent"):
        runner.validate_deterministic_support_core_plan(forged)


def test_target_deformation_sampler_still_rejects_sampled_zero_length_edges():
    positions = np.asarray(
        [
            (0.10, 0.00, 0.000),
            (0.10, 0.00, 0.000),
            (-0.10, 0.00, 0.001),
            (-0.11, 0.01, 0.010),
            (0.00, 0.00, 1.000),
            (0.00, 0.10, 0.500),
        ],
        dtype=np.float64,
    )
    edges = np.asarray(((0, 1),), dtype=np.int64)
    weights = [
        {"left_foot": 1.0},
        {"left_toe": 1.0},
        {"right_toe": 0.99, "right_foot": 0.01},
        {"right_foot": 0.60, "left_foot": 0.40},
        {"head": 1.0},
        {"pelvis": 1.0},
    ]
    semantic = {
        "pelvis": "pelvis",
        "left_thigh": "left_thigh",
        "left_calf": "left_calf",
        "left_foot": "left_foot",
        "left_toe": "left_toe",
        "right_thigh": "right_thigh",
        "right_calf": "right_calf",
        "right_foot": "right_foot",
        "right_toe": "right_toe",
    }

    with pytest.raises(runner.RetargetError, match="degenerate edge"):
        runner.build_deterministic_mesh_sample_plan(
            rest_positions=positions,
            edges=edges,
            vertex_weights=weights,
            semantic_bones=semantic,
            maximum_global_vertices=6,
            maximum_support_vertices=6,
            maximum_edges=5,
        )


def test_constant_grounding_preserves_pre_ground_evidence_and_is_bounded_to_one_cm():
    evidence = runner.plan_constant_grounding(
        frame_minimum_z={1: -0.004, 2: 0.001, 3: -0.002},
        floor_z_m=0.0,
    )
    assert evidence["correction_m"] == pytest.approx(0.004)
    assert evidence["pre_ground_maximum_penetration_m"] == pytest.approx(0.004)
    assert evidence["pre_ground_penetration_by_frame_m"] == {
        "1": 0.004,
        "2": 0.0,
        "3": 0.002,
    }
    with pytest.raises(runner.RetargetError, match="ground.*0.010|correction"):
        runner.plan_constant_grounding(
            frame_minimum_z={1: -0.01001, 2: 0.0},
            floor_z_m=0.0,
        )


def test_surface_contact_plan_turns_pixal_sole_penetration_into_bounded_leg_ik():
    plan = runner.plan_surface_contact_leg_ik(
        side_minimum_z_m={"left": -0.02316976897418499, "right": 0.018},
        floor_z_m=0.0,
    )

    assert plan["upward_correction_m"]["left"] == pytest.approx(
        0.02316976897418499 + runner.IK_CONTACT_READBACK_SAFETY_MARGIN_M
    )
    assert plan["upward_correction_m"]["right"] == 0.0
    assert plan["target_surface_z_m"] == pytest.approx(
        runner.IK_CONTACT_READBACK_SAFETY_MARGIN_M
    )

    with pytest.raises(runner.GroundingError) as captured:
        runner.plan_surface_contact_leg_ik(
            side_minimum_z_m={"left": -0.031, "right": 0.0},
            floor_z_m=0.0,
        )
    assert captured.value.evidence["schema"] == (
        "tokenrig_surface_contact_ik_rejection_v1"
    )


def test_grounding_rejection_carries_action_worst_vertex_weights_and_distribution():
    semantic = runner.validate_semantic_mapping(_semantic_mapping())
    sample = runner.describe_grounding_sample(
        frame=25,
        vertex_index=77852,
        evaluated_position=(-0.09, -0.56, -0.0223123989),
        rest_position=(-0.0968922, 0.0298725, 0.00581267),
        weights={
            "bone_r_toe": 0.92572975,
            "bone_r_foot": 0.07403684,
            "bone_r_calf": 0.00023341,
        },
        semantic_bones=semantic["semantic_bones"],
    )

    with pytest.raises(runner.GroundingError) as captured:
        runner.plan_constant_grounding(
            frame_minimum_z={24: -0.0211436953, 25: -0.0223123989, 26: -0.0192958992},
            floor_z_m=0.0,
            action_name="Walking",
            frame_minimum_samples={25: sample},
        )

    evidence = captured.value.evidence
    assert evidence["schema"] == "tokenrig_grounding_rejection_v1"
    assert evidence["action_name"] == "Walking"
    assert evidence["worst_frame"] == 25
    assert evidence["worst_minimum_z_m"] == pytest.approx(-0.0223123989)
    assert evidence["required_constant_correction_m"] == pytest.approx(0.0223123989)
    assert evidence["worst_sample"]["vertex_index"] == 77852
    assert evidence["worst_sample"]["evaluated_position_m"][2] == pytest.approx(
        -0.0223123989
    )
    assert evidence["worst_sample"]["rest_position_m"][2] == pytest.approx(
        0.00581267
    )
    assert evidence["worst_sample"]["dominant_semantic_region"] == "right_toe"
    assert evidence["worst_sample"]["side"] == "right"
    assert evidence["worst_sample"]["dominant_weights"][0] == {
        "bone": "bone_r_toe",
        "semantic_role": "right_toe",
        "weight": pytest.approx(0.92572975),
    }
    assert evidence["frame_minimum_z_m"] == {
        "24": pytest.approx(-0.0211436953),
        "25": pytest.approx(-0.0223123989),
        "26": pytest.approx(-0.0192958992),
    }
    assert evidence["frame_minimum_distribution_m"]["frame_count"] == 3
    assert evidence["frame_minimum_distribution_m"]["minimum"] == pytest.approx(
        -0.0223123989
    )


def test_runtime_grounding_plan_retains_authenticated_sample_metadata():
    positions = np.asarray(
        [(-0.1, 0.0, 0.0), (0.1, 0.0, 0.0), (-0.1, 0.0, 0.1), (0.1, 0.0, 0.1)]
    )
    edges = np.asarray(((0, 2), (1, 3), (2, 3)), dtype=np.int64)
    weights = [
        {"left_foot": 1.0},
        {"right_toe": 1.0},
        {"left_thigh": 1.0},
        {"right_thigh": 1.0},
    ]
    semantic = {
        "pelvis": "pelvis",
        "left_thigh": "left_thigh",
        "left_calf": "left_calf",
        "left_foot": "left_foot",
        "left_toe": "left_toe",
        "right_thigh": "right_thigh",
        "right_calf": "right_calf",
        "right_foot": "right_foot",
        "right_toe": "right_toe",
    }

    plan = runner.build_deterministic_mesh_sample_plan(
        rest_positions=positions,
        edges=edges,
        vertex_weights=weights,
        semantic_bones=semantic,
        maximum_global_vertices=4,
        maximum_support_vertices=2,
        maximum_edges=3,
    )

    assert np.array_equal(plan["_runtime_rest_positions"], positions)
    assert plan["_runtime_vertex_weights"] is weights
    assert plan["_runtime_semantic_bones"] == semantic
    source = SCRIPT.read_text(encoding="utf-8")
    apply_grounding = source[
        source.index("def _apply_constant_grounding(") : source.index("\ndef _joint_world")
    ]
    assert "describe_grounding_sample(" in apply_grounding
    assert "frame_minimum_samples=" in apply_grounding


def test_two_bone_leg_ik_lifts_only_ankle_z_and_preserves_lengths_and_knee_side():
    hip = np.asarray((0.0, 0.0, 0.90))
    knee = np.asarray((0.0, -0.08, 0.48))
    ankle = np.asarray((0.0, 0.0, 0.08))
    target = ankle + np.asarray((0.0, 0.0, 0.0223123989))

    solved = runner.solve_two_bone_leg_ik(
        hip=hip,
        knee=knee,
        ankle=ankle,
        target_ankle=target,
    )

    assert np.allclose(solved["hip"], hip, atol=1.0e-12)
    assert np.allclose(solved["ankle"], target, atol=1.0e-12)
    assert np.allclose(solved["ankle"][:2], ankle[:2], atol=1.0e-12)
    assert np.linalg.norm(solved["knee"] - hip) == pytest.approx(
        np.linalg.norm(knee - hip), abs=1.0e-12
    )
    assert np.linalg.norm(target - solved["knee"]) == pytest.approx(
        np.linalg.norm(ankle - knee), abs=1.0e-12
    )
    assert solved["evidence"]["ankle_correction_m"] == pytest.approx(0.0223123989)
    assert solved["evidence"]["knee_plane_dot"] >= runner.MINIMUM_IK_KNEE_PLANE_DOT
    assert solved["evidence"]["knee_side_dot"] > 0.0

    # Exercise the signed correction path with a deliberately bent, reachable
    # chain; lowering the nearly straight fixture above would correctly exceed
    # the chain's reach before it exercised signed correction handling.
    lowered_knee = np.asarray((0.0, -0.16, 0.48))
    lowered = runner.solve_two_bone_leg_ik(
        hip=hip,
        knee=lowered_knee,
        ankle=ankle,
        target_ankle=ankle - np.asarray((0.0, 0.0, 0.018879)),
    )
    assert lowered["evidence"]["ankle_correction_m"] == pytest.approx(-0.018879)
    assert np.allclose(lowered["ankle"][:2], ankle[:2], atol=1.0e-12)
    assert lowered["evidence"]["knee_plane_dot"] >= runner.MINIMUM_IK_KNEE_PLANE_DOT


def test_two_bone_leg_ik_rejects_horizontal_root_like_or_oversized_and_unreachable_edits():
    hip = np.asarray((0.0, 0.0, 0.90))
    knee = np.asarray((0.0, -0.08, 0.48))
    ankle = np.asarray((0.0, 0.0, 0.08))

    with pytest.raises(runner.RetargetError, match="X-Y|horizontal"):
        runner.solve_two_bone_leg_ik(
            hip=hip,
            knee=knee,
            ankle=ankle,
            target_ankle=ankle + np.asarray((0.001, 0.0, 0.01)),
        )
    with pytest.raises(runner.RetargetError, match="correction|0.030"):
        runner.solve_two_bone_leg_ik(
            hip=hip,
            knee=knee,
            ankle=ankle,
            target_ankle=ankle + np.asarray((0.0, 0.0, 0.030001)),
        )
    with pytest.raises(runner.RetargetError, match="correction|0.030"):
        runner.solve_two_bone_leg_ik(
            hip=hip,
            knee=knee,
            ankle=ankle,
            target_ankle=ankle - np.asarray((0.0, 0.0, 0.030001)),
        )
    with pytest.raises(runner.RetargetError, match="reach|unreachable"):
        runner.solve_two_bone_leg_ik(
            hip=hip,
            knee=knee,
            ankle=ankle,
            target_ankle=np.asarray((0.0, 0.0, 2.0)),
            maximum_correction_m=2.0,
        )


def test_rest_frame_endpoint_map_is_proper_reconstructs_rest_and_maps_current_vector():
    source_rest = {
        "hip": (0.0, 0.0, 1.0),
        "knee": (0.0, -0.20, 0.50),
        "ankle": (0.0, 0.0, 0.0),
    }
    rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    target_hip = np.asarray((1.0, 2.0, 1.5))
    target_rest = {
        name: target_hip + 0.5 * rotation @ (np.asarray(value) - source_rest["hip"])
        for name, value in source_rest.items()
    }
    source_current_hip = np.asarray((0.1, -0.2, 1.1))
    source_current_ankle = source_current_hip + np.asarray((0.1, -0.1, -0.9))
    target_current_hip = np.asarray((1.2, 2.3, 1.7))

    mapped = runner.map_source_leg_endpoint_rest_frame(
        source_rest_hip=source_rest["hip"],
        source_rest_knee=source_rest["knee"],
        source_rest_ankle=source_rest["ankle"],
        target_rest_hip=target_rest["hip"],
        target_rest_knee=target_rest["knee"],
        target_rest_ankle=target_rest["ankle"],
        source_current_hip=source_current_hip,
        source_current_ankle=source_current_ankle,
        target_current_hip=target_current_hip,
        scale_basis="rest_chord",
    )

    expected = target_current_hip + 0.5 * rotation @ (
        source_current_ankle - source_current_hip
    )
    assert np.allclose(mapped["endpoint"], expected, atol=1.0e-12)
    assert mapped["evidence"]["source_rest_frame_determinant"] == pytest.approx(1.0)
    assert mapped["evidence"]["target_rest_frame_determinant"] == pytest.approx(1.0)
    assert mapped["evidence"]["alignment_determinant"] == pytest.approx(1.0)
    assert mapped["evidence"]["rest_reconstruction_error_m"] < 1.0e-12
    assert mapped["evidence"]["scale_basis"] == "rest_chord"
    assert mapped["evidence"]["scale"] == pytest.approx(0.5)

    segment = runner.map_source_leg_endpoint_rest_frame(
        source_rest_hip=source_rest["hip"],
        source_rest_knee=source_rest["knee"],
        source_rest_ankle=source_rest["ankle"],
        target_rest_hip=target_rest["hip"],
        target_rest_knee=target_rest["knee"],
        target_rest_ankle=target_rest["ankle"],
        source_current_hip=source_current_hip,
        source_current_ankle=source_current_ankle,
        target_current_hip=target_current_hip,
        scale_basis="segment_sum",
    )
    assert segment["evidence"]["scale_basis"] == "segment_sum"
    assert segment["evidence"]["rest_reconstruction_error_m"] < 1.0e-12


def test_rest_frame_endpoint_map_records_chord_segment_difference_and_rejects_invalid_frames():
    arguments = {
        "source_rest_hip": (0.0, 0.0, 1.0),
        "source_rest_knee": (0.0, -0.20, 0.50),
        "source_rest_ankle": (0.0, 0.0, 0.0),
        "target_rest_hip": (0.0, 0.0, 0.50),
        "target_rest_knee": (0.0, -0.15, 0.25),
        "target_rest_ankle": (0.0, 0.0, 0.0),
        "source_current_hip": (0.0, 0.0, 1.0),
        "source_current_ankle": (0.1, 0.0, 0.1),
        "target_current_hip": (0.0, 0.0, 0.50),
    }
    chord = runner.map_source_leg_endpoint_rest_frame(
        **arguments, scale_basis="rest_chord"
    )
    segment = runner.map_source_leg_endpoint_rest_frame(
        **arguments, scale_basis="segment_sum"
    )
    anisotropic = runner.map_source_leg_endpoint_rest_frame(
        **arguments, scale_basis="axial_chord_perpendicular_segment"
    )
    assert chord["evidence"]["rest_reconstruction_error_m"] < 1.0e-12
    assert segment["evidence"]["scale"] != pytest.approx(chord["evidence"]["scale"])
    assert segment["evidence"]["rest_reconstruction_error_m"] > 0.0
    assert anisotropic["evidence"]["rest_reconstruction_error_m"] < 1.0e-12
    assert anisotropic["evidence"]["scale_basis"] == (
        "axial_chord_perpendicular_segment"
    )
    assert anisotropic["evidence"]["mapping_scale_components"] == {
        "rest_axis": pytest.approx(chord["evidence"]["rest_chord_scale"]),
        "bend_axis": pytest.approx(segment["evidence"]["segment_sum_scale"]),
        "normal_axis": pytest.approx(segment["evidence"]["segment_sum_scale"]),
    }

    with pytest.raises(runner.RetargetError, match="degenerate|bend|plane"):
        runner.map_source_leg_endpoint_rest_frame(
            **{**arguments, "source_rest_knee": (0.0, 0.0, 0.5)},
            scale_basis="rest_chord",
        )
    with pytest.raises(runner.RetargetError, match="finite"):
        runner.map_source_leg_endpoint_rest_frame(
            **{**arguments, "target_current_hip": (0.0, math.nan, 0.5)},
            scale_basis="rest_chord",
        )
    with pytest.raises(runner.RetargetError, match="reflection|handed|axis"):
        runner.map_source_leg_endpoint_rest_frame(
            **arguments,
            scale_basis="rest_chord",
            axis_map_3x3=((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )


def test_rest_frame_piecewise_reach_mapping_preserves_rest_and_normalized_flexion_extension():
    arguments = {
        "source_rest_hip": (0.0, 0.0, 1.0),
        "source_rest_knee": (0.0, -0.20, 0.50),
        "source_rest_ankle": (0.0, 0.0, 0.0),
        "target_rest_hip": (0.0, 0.0, 0.50),
        "target_rest_knee": (0.0, -0.15, 0.25),
        "target_rest_ankle": (0.0, 0.0, 0.0),
        "source_current_hip": (0.0, 0.0, 1.0),
        "target_current_hip": (0.0, 0.0, 0.50),
    }
    rest = runner.map_source_leg_endpoint_rest_frame(
        **arguments,
        source_current_ankle=(0.0, 0.0, 0.0),
        scale_basis="reach_normalized_piecewise",
    )
    assert rest["evidence"]["piecewise_branch"] == "rest"
    assert rest["evidence"]["normalized_reach_fraction"] == 0.0
    assert rest["evidence"]["rest_reconstruction_error_m"] < 1.0e-12
    assert np.linalg.norm(
        np.asarray(rest["endpoint"]) - np.asarray(arguments["target_current_hip"])
    ) == pytest.approx(0.5)

    flexed = runner.map_source_leg_endpoint_rest_frame(
        **arguments,
        source_current_ankle=(0.0, 0.0, 0.10),
        scale_basis="reach_normalized_piecewise",
    )
    assert flexed["evidence"]["piecewise_branch"] == "flexion"
    assert flexed["evidence"]["normalized_reach_fraction"] == pytest.approx(0.1)
    assert flexed["evidence"]["mapped_target_distance_m"] == pytest.approx(0.45)

    source_maximum = flexed["evidence"]["source_maximum_reach_m"]
    extension_distance = 1.05
    extended = runner.map_source_leg_endpoint_rest_frame(
        **arguments,
        source_current_ankle=(0.0, 0.0, 1.0 - extension_distance),
        scale_basis="reach_normalized_piecewise",
    )
    expected_fraction = (extension_distance - 1.0) / (source_maximum - 1.0)
    expected_distance = 0.5 + expected_fraction * (
        extended["evidence"]["target_maximum_reach_m"] - 0.5
    )
    assert extended["evidence"]["piecewise_branch"] == "extension"
    assert extended["evidence"]["normalized_reach_fraction"] == pytest.approx(
        expected_fraction
    )
    assert extended["evidence"]["mapped_target_distance_m"] == pytest.approx(
        expected_distance
    )
    assert extended["evidence"]["target_reach_margin_m"] > 0.0
    assert extended["evidence"]["distance_was_clamped"] is False

    with pytest.raises(runner.RetargetError, match="source.*reach|outside"):
        runner.map_source_leg_endpoint_rest_frame(
            **arguments,
            source_current_ankle=(0.0, 0.0, 1.0 - source_maximum - 0.001),
            scale_basis="reach_normalized_piecewise",
        )


def test_piecewise_canonical_chord_alignment_does_not_turn_sagittal_swing_lateral():
    arguments = {
        "source_rest_hip": (0.0, 0.0, 1.0),
        "source_rest_knee": (0.0, -0.10, 0.50),
        "source_rest_ankle": (0.0, 0.0, 0.0),
        "target_rest_hip": (0.0, 0.0, 0.50),
        # Same nearly vertical rest chord, but a 90-degree knee-plane twist.
        "target_rest_knee": (0.05, 0.0, 0.25),
        "target_rest_ankle": (0.0, 0.0, 0.0),
        "source_current_hip": (0.0, 0.0, 1.0),
        "source_current_knee": (0.0, -0.20, 0.55),
        "source_current_ankle": (0.0, -0.30, 0.10),
        "target_current_hip": (1.0, 2.0, 1.0),
        "scale_basis": "reach_normalized_piecewise",
    }
    full_frame = runner.map_source_leg_endpoint_rest_frame(
        **arguments,
        orientation_basis="full_leg_rest_frame",
    )
    canonical = runner.map_source_leg_endpoint_rest_frame(
        **arguments,
        orientation_basis="canonical_axis_minimal_rest_chord",
    )
    full_relative = np.asarray(full_frame["endpoint"]) - arguments["target_current_hip"]
    canonical_relative = (
        np.asarray(canonical["endpoint"]) - arguments["target_current_hip"]
    )

    assert abs(full_relative[0]) > 0.1
    assert abs(full_relative[1]) < 1.0e-12
    assert abs(canonical_relative[0]) < 1.0e-12
    assert canonical_relative[1] < -0.1
    assert canonical["evidence"]["orientation_basis"] == (
        "canonical_axis_minimal_rest_chord"
    )
    assert canonical["evidence"]["formula"] == (
        "proper_canonical_axis_minimal_rest_chord_plus_piecewise_normalized_reach_v1"
    )
    assert canonical["evidence"]["rest_reconstruction_error_m"] < 1.0e-12
    assert canonical["evidence"]["alignment_determinant"] == pytest.approx(1.0)
    runner.validate_source_leg_endpoint_rest_frame_mapping(canonical)


def test_piecewise_canonical_front_frame_preserves_front_semantics_and_bilateral_symmetry():
    def mapped(side_sign):
        return runner.map_source_leg_endpoint_rest_frame(
            source_rest_hip=(0.10 * side_sign, 0.0, 1.0),
            source_rest_knee=(0.10 * side_sign, -0.10, 0.50),
            source_rest_ankle=(0.10 * side_sign, 0.0, 0.0),
            target_rest_hip=(0.05 * side_sign, 0.0, 0.50),
            target_rest_knee=(0.05 * side_sign, 0.05, 0.25),
            target_rest_ankle=(0.05 * side_sign, 0.0, 0.0),
            source_current_hip=(0.10 * side_sign, 0.0, 1.0),
            source_current_knee=(0.16 * side_sign, -0.20, 0.55),
            source_current_ankle=(0.22 * side_sign, -0.30, 0.10),
            target_current_hip=(0.05 * side_sign, 0.0, 0.50),
            scale_basis="reach_normalized_piecewise",
            orientation_basis="canonical_front_constrained_chord_frame",
        )

    left = mapped(1.0)
    right = mapped(-1.0)
    left_relative = np.asarray(left["endpoint"]) - np.asarray((0.05, 0.0, 0.50))
    right_relative = np.asarray(right["endpoint"]) - np.asarray((-0.05, 0.0, 0.50))

    assert left_relative[1] < 0.0
    assert right_relative[1] < 0.0
    assert left_relative[0] == pytest.approx(-right_relative[0], abs=1.0e-12)
    assert left_relative[1] == pytest.approx(right_relative[1], abs=1.0e-12)
    assert left_relative[2] == pytest.approx(right_relative[2], abs=1.0e-12)
    assert left_relative[0] > 0.0 and right_relative[0] < 0.0
    for evidence in (left["evidence"], right["evidence"]):
        assert evidence["orientation_basis"] == (
            "canonical_front_constrained_chord_frame"
        )
        assert evidence["canonical_front"] == "negative-y"
        assert evidence["canonical_up"] == "positive-z"
        assert evidence["source_rest_frame_determinant"] == pytest.approx(1.0)
        assert evidence["target_rest_frame_determinant"] == pytest.approx(1.0)
        assert evidence["alignment_determinant"] == pytest.approx(1.0)
        assert evidence["rest_reconstruction_error_m"] < 1.0e-12

    with pytest.raises(runner.RetargetError, match="FRONT projection|degenerate"):
        runner.map_source_leg_endpoint_rest_frame(
            source_rest_hip=(0.0, 0.0, 0.0),
            source_rest_knee=(0.0, -0.5, 0.1),
            source_rest_ankle=(0.0, -1.0, 0.0),
            target_rest_hip=(0.0, 0.0, 0.0),
            target_rest_knee=(0.0, -0.5, 0.1),
            target_rest_ankle=(0.0, -1.0, 0.0),
            source_current_hip=(0.0, 0.0, 0.0),
            source_current_knee=(0.0, -0.5, 0.1),
            source_current_ankle=(0.0, -1.0, 0.0),
            target_current_hip=(0.0, 0.0, 0.0),
            scale_basis="reach_normalized_piecewise",
            orientation_basis="canonical_front_constrained_chord_frame",
        )


def test_piecewise_canonical_absolute_direction_ignores_bent_target_rest_chord():
    source = {
        "source_rest_hip": (0.0, 0.0, 1.0),
        "source_rest_knee": (0.331662479, 0.0, 0.5),
        "source_rest_ankle": (0.0, 0.0, 0.0),
        "source_current_hip": (0.0, 0.0, 1.0),
        "source_current_knee": (0.08, -0.25, 0.52),
        "source_current_ankle": (0.12, -0.18, 0.05),
        "target_current_hip": (1.0, 2.0, 1.0),
        "scale_basis": "reach_normalized_piecewise",
        "orientation_basis": "canonical_absolute_source_direction",
        "coordinate_space": "canonical_static_object_frame_m",
    }
    straight = runner.map_source_leg_endpoint_rest_frame(
        **source,
        target_rest_hip=(0.0, 0.0, 0.5),
        target_rest_knee=(0.1658312395, 0.0, 0.25),
        target_rest_ankle=(0.0, 0.0, 0.0),
    )
    bent = runner.map_source_leg_endpoint_rest_frame(
        **source,
        target_rest_hip=(0.0, 0.0, 0.5),
        target_rest_knee=(0.1658312395, 0.15, 0.30),
        target_rest_ankle=(0.0, 0.30, 0.10),
    )
    expected_direction = np.asarray((0.12, -0.18, -0.95), dtype=np.float64)
    expected_direction /= np.linalg.norm(expected_direction)
    for mapping in (straight, bent):
        relative = np.asarray(mapping["endpoint"]) - np.asarray(
            source["target_current_hip"]
        )
        assert relative / np.linalg.norm(relative) == pytest.approx(
            expected_direction, abs=1.0e-12
        )
        assert mapping["mapped_source_knee_vector"] == pytest.approx(
            (0.08, -0.25, -0.48), abs=1.0e-12
        )
        evidence = mapping["evidence"]
        assert evidence["orientation_basis"] == (
            "canonical_absolute_source_direction"
        )
        assert evidence["formula"] == (
            "canonical_absolute_source_direction_plus_piecewise_normalized_reach_v1"
        )
        assert evidence["target_rest_chord_used_for_orientation"] is False
        assert evidence["source_to_mapped_direction_alignment_dot"] == pytest.approx(1.0)
        runner.validate_source_leg_endpoint_rest_frame_mapping(mapping)
    assert straight["endpoint"] == pytest.approx(bent["endpoint"], abs=1.0e-12)

    legacy = runner.map_source_leg_endpoint_rest_frame(
        **{key: value for key, value in source.items() if key != "orientation_basis"},
        target_rest_hip=(0.0, 0.0, 0.5),
        target_rest_knee=(0.1658312395, 0.15, 0.30),
        target_rest_ankle=(0.0, 0.30, 0.10),
        orientation_basis="canonical_axis_minimal_rest_chord",
    )
    assert legacy["evidence"]["target_rest_chord_used_for_orientation"] is True
    runner.validate_source_leg_endpoint_rest_frame_mapping(legacy)


def test_canonical_absolute_direction_preserves_f17_f23_vertical_mirror_and_loop():
    common = {
        "source_rest_hip": (0.0, 0.0, 0.40),
        "source_rest_knee": (0.10, 0.0, 0.20),
        "source_rest_ankle": (0.0, 0.0, 0.0),
        "target_rest_hip": (0.0, 0.0, 0.40),
        "target_rest_knee": (0.10, 0.12, 0.22),
        "target_rest_ankle": (0.0, 0.18, 0.0436),
        "source_current_hip": (0.0, 0.0, 0.40),
        "target_current_hip": (0.0, 0.0, 0.40),
        "scale_basis": "reach_normalized_piecewise",
        "orientation_basis": "canonical_absolute_source_direction",
        "coordinate_space": "canonical_static_object_frame_m",
    }

    def mapped(horizontal, vertical, side_sign=1.0):
        return runner.map_source_leg_endpoint_rest_frame(
            **common,
            source_current_knee=(0.04 * side_sign, -0.08, 0.20),
            source_current_ankle=(0.02 * side_sign, horizontal, 0.40 - vertical),
        )

    for horizontal, vertical in ((0.021955553, 0.391442709), (0.148544793, 0.362194858)):
        result = mapped(horizontal, vertical)
        relative = np.asarray(result["endpoint"]) - np.asarray(
            common["target_current_hip"]
        )
        expected = np.asarray((0.02, horizontal, -vertical), dtype=np.float64)
        assert relative / np.linalg.norm(relative) == pytest.approx(
            expected / np.linalg.norm(expected), abs=1.0e-12
        )
        assert relative[2] < 0.0

    left = mapped(0.10, 0.36, 1.0)
    right = mapped(0.10, 0.36, -1.0)
    left_relative = np.asarray(left["endpoint"]) - np.asarray(
        common["target_current_hip"]
    )
    right_relative = np.asarray(right["endpoint"]) - np.asarray(
        common["target_current_hip"]
    )
    assert left_relative[0] == pytest.approx(-right_relative[0], abs=1.0e-12)
    assert left_relative[1:] == pytest.approx(right_relative[1:], abs=1.0e-12)
    assert np.asarray(left["mapped_source_knee_vector"])[0] == pytest.approx(
        -np.asarray(right["mapped_source_knee_vector"])[0], abs=1.0e-12
    )
    assert mapped(0.10, 0.36, 1.0)["endpoint"] == pytest.approx(
        left["endpoint"], abs=1.0e-12
    )


def test_canonical_absolute_direction_rejects_degenerate_source_current_endpoint():
    with pytest.raises(runner.RetargetError, match="source current chord.*degenerate"):
        runner.map_source_leg_endpoint_rest_frame(
            source_rest_hip=(0.0, 0.0, 1.0),
            source_rest_knee=(0.1, 0.0, 0.5),
            source_rest_ankle=(0.0, 0.0, 0.0),
            target_rest_hip=(0.0, 0.0, 0.5),
            target_rest_knee=(0.1, 0.1, 0.25),
            target_rest_ankle=(0.0, 0.2, 0.05),
            source_current_hip=(0.0, 0.0, 1.0),
            source_current_knee=(0.1, 0.0, 0.5),
            source_current_ankle=(0.0, 0.0, 1.0),
            target_current_hip=(0.0, 0.0, 0.5),
            scale_basis="reach_normalized_piecewise",
            orientation_basis="canonical_absolute_source_direction",
            coordinate_space="canonical_static_object_frame_m",
        )


def _source_driven_leg_mapping_fixture():
    source_rest = {
        "hip": np.asarray((0.0, 0.0, 1.0)),
        "knee": np.asarray((0.0, -0.30, 0.52)),
        "ankle": np.asarray((0.0, 0.0, 0.0)),
    }
    target_rest = {
        "hip": np.asarray((0.0, 0.0, 1.0)),
        "knee": np.asarray((0.0, -0.24, 0.52)),
        "ankle": np.asarray((0.0, 0.0, 0.0)),
    }
    source_current_hip = np.asarray((0.20, -0.10, 1.10))
    source_current_knee = source_current_hip + np.asarray((0.12, -0.35, -0.46))
    source_current_ankle = source_current_hip + np.asarray((0.25, 0.15, -0.92))
    target_current_hip = np.asarray((2.0, -1.0, 1.70))
    mapping = runner.map_source_leg_endpoint_rest_frame(
        source_rest_hip=source_rest["hip"],
        source_rest_knee=source_rest["knee"],
        source_rest_ankle=source_rest["ankle"],
        target_rest_hip=target_rest["hip"],
        target_rest_knee=target_rest["knee"],
        target_rest_ankle=target_rest["ankle"],
        source_current_hip=source_current_hip,
        source_current_knee=source_current_knee,
        source_current_ankle=source_current_ankle,
        target_current_hip=target_current_hip,
        scale_basis="reach_normalized_piecewise",
    )
    return source_rest, target_rest, target_current_hip, mapping


def test_piecewise_endpoint_evidence_recomputes_every_input_and_rejects_forgery():
    _, _, _, mapping = _source_driven_leg_mapping_fixture()

    runner.validate_source_leg_endpoint_rest_frame_mapping(mapping)
    assert mapping["evidence"]["schema"] == "proper_leg_rest_frame_endpoint_v1"
    assert mapping["evidence"]["formula"] == (
        "proper_rest_frame_direction_plus_piecewise_normalized_reach_v1"
    )
    assert 0.0 <= mapping["evidence"]["source_normalized_reach_0_1"] <= 1.0
    assert 0.0 <= mapping["evidence"]["target_normalized_reach_0_1"] <= 1.0
    assert mapping["mapped_source_knee_vector"] is not None

    forged_endpoint = copy.deepcopy(mapping)
    forged_endpoint["endpoint"] = [99.0, 0.0, 0.0]
    forged_fraction = copy.deepcopy(mapping)
    forged_fraction["evidence"]["normalized_reach_fraction"] += 0.01
    forged_global_reach = copy.deepcopy(mapping)
    forged_global_reach["evidence"]["source_normalized_reach_0_1"] += 0.01
    forged_input = copy.deepcopy(mapping)
    forged_input["evidence"]["authenticated_inputs"]["source_current_knee"][0] += 0.01
    for forged in (
        forged_endpoint,
        forged_fraction,
        forged_global_reach,
        forged_input,
    ):
        with pytest.raises(runner.RetargetError, match="inconsistent|forged"):
            runner.validate_source_leg_endpoint_rest_frame_mapping(forged)


def test_piecewise_endpoint_authenticates_canonical_static_object_coordinate_space():
    _, _, _, mapping = _source_driven_leg_mapping_fixture()
    inputs = copy.deepcopy(mapping["evidence"]["authenticated_inputs"])
    inputs["coordinate_space"] = "canonical_static_object_frame_m"

    canonical = runner.map_source_leg_endpoint_rest_frame(**inputs)

    assert canonical["evidence"]["coordinate_space"] == (
        "canonical_static_object_frame_m"
    )
    runner.validate_source_leg_endpoint_rest_frame_mapping(canonical)
    forged = copy.deepcopy(canonical)
    forged["evidence"]["coordinate_space"] = "authenticated_world_m"
    with pytest.raises(runner.RetargetError, match="forged|inconsistent"):
        runner.validate_source_leg_endpoint_rest_frame_mapping(forged)


def test_canonical_static_armature_point_applies_scale_and_base_rotation_once_without_translation():
    rotation = np.asarray(
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    point = np.asarray((20.0, 3.0, -70.0))

    canonical = runner.canonical_static_armature_point(
        point,
        uniform_scale_m_per_unit=0.01,
        base_rotation_3x3=rotation,
    )
    reconstructed = runner.canonical_static_to_armature_point(
        canonical,
        uniform_scale_m_per_unit=0.01,
        base_rotation_3x3=rotation,
    )

    assert np.allclose(canonical, (0.03, -0.20, -0.70), atol=1.0e-12)
    assert np.allclose(reconstructed, point, atol=1.0e-12)


def test_piecewise_endpoint_is_source_root_translation_invariant_and_target_hip_relative():
    _, _, _, mapping = _source_driven_leg_mapping_fixture()
    inputs = mapping["evidence"]["authenticated_inputs"]
    source_shift = np.asarray((93.0, -41.0, 7.5))
    target_shift = np.asarray((-12.0, 5.0, 2.25))
    shifted_inputs = copy.deepcopy(inputs)
    for name in (
        "source_rest_hip",
        "source_rest_knee",
        "source_rest_ankle",
        "source_current_hip",
        "source_current_knee",
        "source_current_ankle",
    ):
        shifted_inputs[name] = (
            np.asarray(shifted_inputs[name]) + source_shift
        ).tolist()
    for name in (
        "target_rest_hip",
        "target_rest_knee",
        "target_rest_ankle",
        "target_current_hip",
    ):
        shifted_inputs[name] = (
            np.asarray(shifted_inputs[name]) + target_shift
        ).tolist()

    shifted = runner.map_source_leg_endpoint_rest_frame(**shifted_inputs)

    assert np.allclose(
        np.asarray(shifted["endpoint"]),
        np.asarray(mapping["endpoint"]) + target_shift,
        atol=1.0e-12,
    )
    assert np.allclose(
        shifted["mapped_source_knee_vector"],
        mapping["mapped_source_knee_vector"],
        atol=1.0e-12,
    )
    assert shifted["evidence"]["mapped_target_distance_m"] == pytest.approx(
        mapping["evidence"]["mapped_target_distance_m"], abs=1.0e-12
    )


def test_source_driven_two_bone_ik_handles_large_xyz_delta_and_preserves_exact_lengths():
    _, target_rest, hip, mapping = _source_driven_leg_mapping_fixture()
    upper_length = float(np.linalg.norm(target_rest["knee"] - target_rest["hip"]))
    lower_length = float(np.linalg.norm(target_rest["ankle"] - target_rest["knee"]))
    current_upper_direction = np.asarray((0.20, -0.90, -0.30))
    current_upper_direction /= np.linalg.norm(current_upper_direction)
    current_lower_direction = np.asarray((0.65, 0.40, -0.65))
    current_lower_direction /= np.linalg.norm(current_lower_direction)
    knee = hip + upper_length * current_upper_direction
    ankle = knee + lower_length * current_lower_direction

    solved = runner.solve_source_driven_two_bone_leg_ik(
        hip=hip,
        knee=knee,
        ankle=ankle,
        endpoint_mapping=mapping,
        body_height_m=1.82,
    )

    solved_knee = np.asarray(solved["knee"])
    solved_ankle = np.asarray(solved["ankle"])
    endpoint_delta = solved_ankle - ankle
    assert np.all(np.abs(endpoint_delta) > 0.01)
    assert np.allclose(solved["hip"], hip, atol=1.0e-12)
    assert np.allclose(solved_ankle, mapping["endpoint"], atol=1.0e-12)
    assert np.linalg.norm(solved_knee - hip) == pytest.approx(upper_length, abs=1.0e-12)
    assert np.linalg.norm(solved_ankle - solved_knee) == pytest.approx(
        lower_length, abs=1.0e-12
    )
    evidence = solved["evidence"]
    assert evidence["method"] == "source_driven_piecewise_two_bone_ik_v1"
    assert evidence["mapped_source_pole_alignment_dot"] > 1.0 - 1.0e-12
    assert evidence["solved_bend_frame_determinant"] == pytest.approx(1.0)
    assert evidence["endpoint_delta_body_height_ratio"] == pytest.approx(
        np.linalg.norm(endpoint_delta) / 1.82
    )
    assert evidence["correction_cap_m"] is None
    assert evidence["root_pelvis_unchanged"] is True

    upper_direction = (knee - hip) / np.linalg.norm(knee - hip)
    blender_float_knee = knee + 2.0e-8 * upper_direction
    blender_float_ankle = blender_float_knee + lower_length * current_lower_direction
    tolerated = runner.solve_source_driven_two_bone_leg_ik(
        hip=hip,
        knee=blender_float_knee,
        ankle=blender_float_ankle,
        endpoint_mapping=mapping,
        body_height_m=1.82,
    )
    assert tolerated["evidence"]["input_upper_length_drift_m"] == pytest.approx(
        2.0e-8, abs=1.0e-12
    )
    assert tolerated["evidence"]["maximum_input_segment_length_drift_m"] == (
        runner.MAXIMUM_IK_SEGMENT_LENGTH_INPUT_DRIFT_M
    )

    excessive_knee = knee + 2.0e-6 * upper_direction
    excessive_ankle = excessive_knee + lower_length * current_lower_direction
    with pytest.raises(runner.RetargetError, match="upper length|piecewise evidence"):
        runner.solve_source_driven_two_bone_leg_ik(
            hip=hip,
            knee=excessive_knee,
            ankle=excessive_ankle,
            endpoint_mapping=mapping,
            body_height_m=1.82,
        )


def test_source_driven_two_bone_ik_obeys_reversed_source_pole_and_rejects_degenerate_pole():
    _, target_rest, hip, mapping = _source_driven_leg_mapping_fixture()
    upper_length = float(np.linalg.norm(target_rest["knee"] - target_rest["hip"]))
    lower_length = float(np.linalg.norm(target_rest["ankle"] - target_rest["knee"]))
    target_axis = np.asarray(mapping["endpoint"]) - hip
    target_axis /= np.linalg.norm(target_axis)
    mapped_pole = np.asarray(mapping["mapped_source_knee_vector"])
    mapped_pole -= float(np.dot(mapped_pole, target_axis)) * target_axis
    mapped_pole /= np.linalg.norm(mapped_pole)
    knee = hip - upper_length * mapped_pole
    perpendicular = np.cross(target_axis, mapped_pole)
    ankle = knee + lower_length * perpendicular

    solved = runner.solve_source_driven_two_bone_leg_ik(
        hip=hip,
        knee=knee,
        ankle=ankle,
        endpoint_mapping=mapping,
        body_height_m=1.82,
    )
    solved_bend = np.asarray(solved["knee"]) - hip
    solved_bend -= float(np.dot(solved_bend, target_axis)) * target_axis
    assert float(np.dot(solved_bend, mapped_pole)) > 0.0
    assert solved["evidence"]["current_target_to_mapped_source_pole_dot"] < 0.0

    degenerate = copy.deepcopy(mapping)
    degenerate["evidence"]["authenticated_inputs"]["source_current_knee"] = (
        degenerate["evidence"]["authenticated_inputs"]["source_current_ankle"]
    )
    degenerate = runner.map_source_leg_endpoint_rest_frame(
        **degenerate["evidence"]["authenticated_inputs"]
    )
    with pytest.raises(runner.RetargetError, match="pole|degenerate|ambiguous"):
        runner.solve_source_driven_two_bone_leg_ik(
            hip=hip,
            knee=knee,
            ankle=ankle,
            endpoint_mapping=degenerate,
            body_height_m=1.82,
        )


def test_source_driven_two_bone_ik_records_near_extension_bend_height_and_knee_angle():
    rest = {
        "hip": np.asarray((0.0, 0.0, 0.0)),
        "knee": np.asarray((0.0, 0.10, -0.50)),
        "ankle": np.asarray((0.0, 0.0, -1.0)),
    }
    upper_length = float(np.linalg.norm(rest["knee"] - rest["hip"]))
    lower_length = float(np.linalg.norm(rest["ankle"] - rest["knee"]))
    maximum_reach = upper_length + lower_length
    source_direction = np.asarray((0.20, 0.05, -1.0))
    source_direction /= np.linalg.norm(source_direction)
    current_distance = maximum_reach - 0.001
    hip = np.asarray((1.0, 2.0, 3.0))
    mapping = runner.map_source_leg_endpoint_rest_frame(
        source_rest_hip=rest["hip"],
        source_rest_knee=rest["knee"],
        source_rest_ankle=rest["ankle"],
        target_rest_hip=rest["hip"],
        target_rest_knee=rest["knee"],
        target_rest_ankle=rest["ankle"],
        source_current_hip=(0.0, 0.0, 0.0),
        source_current_knee=(0.0, 0.10, -0.50),
        source_current_ankle=current_distance * source_direction,
        target_current_hip=hip,
        scale_basis="reach_normalized_piecewise",
    )
    knee = hip + np.asarray((0.0, upper_length, 0.0))
    ankle = knee + np.asarray((lower_length, 0.0, 0.0))

    solved = runner.solve_source_driven_two_bone_leg_ik(
        hip=hip,
        knee=knee,
        ankle=ankle,
        endpoint_mapping=mapping,
        body_height_m=1.75,
    )

    evidence = solved["evidence"]
    assert evidence["reach_margin_m"] == pytest.approx(0.001, abs=1.0e-12)
    assert evidence["bend_height_m"] > 0.0
    assert 0.0 < evidence["knee_flexion_angle_rad"] < 0.2
    assert math.pi - 0.2 < evidence["knee_internal_angle_rad"] < math.pi
    assert evidence["upper_length_residual_m"] < 1.0e-12
    assert evidence["lower_length_residual_m"] < 1.0e-12


def test_source_driven_two_bone_ik_requires_validated_piecewise_nonreflected_mapping_and_same_hip():
    _, target_rest, hip, mapping = _source_driven_leg_mapping_fixture()
    upper_length = float(np.linalg.norm(target_rest["knee"] - target_rest["hip"]))
    lower_length = float(np.linalg.norm(target_rest["ankle"] - target_rest["knee"]))
    knee = hip + np.asarray((0.0, -upper_length, 0.0))
    ankle = knee + np.asarray((lower_length, 0.0, 0.0))

    forged_reflection = copy.deepcopy(mapping)
    forged_reflection["alignment_3x3"][0][0] *= -1.0
    with pytest.raises(runner.RetargetError, match="inconsistent|forged|reflection"):
        runner.solve_source_driven_two_bone_leg_ik(
            hip=hip,
            knee=knee,
            ankle=ankle,
            endpoint_mapping=forged_reflection,
            body_height_m=1.82,
        )

    non_piecewise = runner.map_source_leg_endpoint_rest_frame(
        **{
            **mapping["evidence"]["authenticated_inputs"],
            "scale_basis": "rest_chord",
        },
    )
    with pytest.raises(runner.RetargetError, match="piecewise"):
        runner.solve_source_driven_two_bone_leg_ik(
            hip=hip,
            knee=knee,
            ankle=ankle,
            endpoint_mapping=non_piecewise,
            body_height_m=1.82,
        )

    with pytest.raises(runner.RetargetError, match="hip|pelvis|root"):
        runner.solve_source_driven_two_bone_leg_ik(
            hip=hip + np.asarray((0.001, 0.0, 0.0)),
            knee=knee,
            ankle=ankle,
            endpoint_mapping=mapping,
            body_height_m=1.82,
        )


def test_real_bake_uses_world_joint_cache_canonical_piecewise_primary_ik_and_restores_distal_orientation():
    source = SCRIPT.read_text(encoding="utf-8")
    cache = source[
        source.index("def cache_source_motion(") : source.index("\ndef _semantic_height")
    ]
    bake = source[
        source.index("def bake_rest_corrected_action(") : source.index("\ndef remove_source_objects")
    ]
    apply_pose = source[
        source.index("def apply_source_driven_leg_ik_pose(") : source.index(
            "\ndef bake_rest_corrected_action"
        )
    ]

    assert "world_joint_positions" in runner.CachedMotionFrame.__annotations__
    assert "source.armature.matrix_world @ source.armature.pose.bones[name].head" in cache
    assert '"rest_world_heads"' in cache
    assert '"first_object_frame"' in cache
    assert 'frame_record.joint_positions[' in bake
    assert 'frame_record.world_joint_positions[' not in bake
    assert 'cached["rest_heads"][' in bake
    assert '"canonical_static_object_frame_m"' in bake
    assert (
        '"source and target armature-local hip-knee-ankle points scaled to "'
        in bake
    )
    assert "sealed source world hip-knee-ankle" not in bake
    assert 'scale_basis="reach_normalized_piecewise"' in bake
    assert 'orientation_basis="canonical_absolute_source_direction"' in bake
    assert 'orientation_basis="canonical_axis_minimal_rest_chord"' not in bake
    assert "solve_source_driven_two_bone_leg_ik(" in apply_pose
    assert "original_foot_world" in apply_pose
    assert "original_toe_world" in apply_pose
    assert "foot_orientation_error_rad" in apply_pose
    assert "toe_orientation_error_rad" in apply_pose
    assert '"primary_leg_ik"' in bake
    assert '"root_object_mapping"' in bake
    assert '"target_base_rotation @ (source_first_rotation.inverted() @ "' in bake
    assert '"source_current_rotation)"' in bake
    assert '"source_world_translation_delta_scaled_once"' in bake


def test_source_contact_ik_policy_uses_raw_source_contact_and_symmetric_bounded_edits():
    plan = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=25,
        source_clearance_m={"right": 0.040, "left": 0.004},
        target_clearance_m={"right": -0.002, "left": -0.005},
        height_scale=0.5,
    )

    assert plan["schema"] == "tokenrig_source_contact_ik_plan_v2"
    assert plan["source_support_basis"] == runner.WALK_SOURCE_SUPPORT_BASIS
    assert plan["contact_clearance_m"] == 0.015
    assert plan["physical_valid_clearance_interval_m"] == {
        "minimum": 0.0,
        "maximum": 0.015,
    }
    assert plan["readback_safety_margin_m"] == 0.0001
    assert plan["interior_target_clearance_interval_m"] == {
        "minimum": 0.0001,
        "maximum": 0.0149,
    }
    assert plan["primary_support_side"] == "left"
    assert plan["primary_tie_break_order"] == [
        "authenticated_single_candidate_side"
    ]
    assert plan["walking_selection_reason"] == "single_source_candidate_same_side"
    assert plan["sides"]["left"] == {
        "source_clearance_m": 0.004,
        "scaled_source_clearance_m": 0.002,
        "source_contact_candidate": True,
        "target_before_clearance_m": -0.005,
        "target_desired_clearance_m": 0.002,
        "ankle_correction_m": 0.007,
        "absolute_ankle_correction_m": 0.007,
        "accumulated_absolute_correction_before_m": 0.0,
        "accumulated_absolute_correction_after_m": 0.007,
        "apply_correction": True,
        "reason": "source_primary_contact_penetration",
    }
    assert plan["sides"]["right"]["source_contact_candidate"] is False
    assert plan["sides"]["right"]["target_desired_clearance_m"] == pytest.approx(
        0.020
    )
    assert plan["sides"]["right"]["ankle_correction_m"] == pytest.approx(0.022)
    assert (
        plan["sides"]["right"]["reason"]
        == "target_penetration_during_source_swing"
    )
    runner.validate_source_contact_ik_plan(plan)

    # Contact phase must be selected from the raw, authenticated source
    # clearance. Scaling both values below 15 mm must not create a second
    # candidate, and equal candidates must resolve deterministically to left.
    raw_threshold = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=7,
        source_clearance_m={"left": 0.014, "right": 0.016},
        target_clearance_m={"left": 0.028, "right": 0.040},
        height_scale=0.5,
    )
    assert raw_threshold["primary_support_side"] == "left"
    assert raw_threshold["sides"]["right"]["source_contact_candidate"] is False
    assert raw_threshold["sides"]["left"]["target_desired_clearance_m"] == pytest.approx(
        runner.CONTACT_CLEARANCE_M
        - runner.IK_CONTACT_READBACK_SAFETY_MARGIN_M
    )
    assert raw_threshold["sides"]["left"]["ankle_correction_m"] == pytest.approx(
        -0.0131
    )
    assert raw_threshold["sides"]["left"]["reason"] == "source_primary_contact_hover"

    idle = runner.plan_source_contact_ik_correction(
        action_name="Standing_Idle",
        source_support_basis=runner.IDLE_SOURCE_SUPPORT_BASIS,
        frame=1,
        source_clearance_m={"right": 0.003, "left": 0.002},
        target_clearance_m={"right": -0.004, "left": 0.020},
        height_scale=0.5,
    )
    assert idle["primary_support_side"] is None
    assert idle["support_sides"] == ["left", "right"]
    assert idle["sides"]["left"]["reason"] == "idle_bilateral_contact_hover"
    assert idle["sides"]["left"]["ankle_correction_m"] == pytest.approx(-0.0051)
    assert idle["sides"]["right"]["reason"] == "idle_bilateral_contact_penetration"
    assert idle["sides"]["right"]["target_desired_clearance_m"] == pytest.approx(
        0.0015
    )
    assert idle["sides"]["right"]["ankle_correction_m"] == pytest.approx(0.0055)
    runner.validate_source_contact_ik_plan(idle)

    edge = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=7,
        source_clearance_m={"left": -0.001, "right": 0.050},
        target_clearance_m={"left": 0.01500005, "right": 0.030},
        height_scale=0.5,
    )
    assert edge["sides"]["left"]["target_desired_clearance_m"] == pytest.approx(
        0.0149
    )
    assert edge["sides"]["left"]["apply_correction"] is True
    settled = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=7,
        source_clearance_m={"left": -0.001, "right": 0.050},
        target_clearance_m={"left": 0.01490008, "right": 0.030},
        height_scale=0.5,
        accumulated_absolute_correction_m={
            "left": edge["sides"]["left"][
                "accumulated_absolute_correction_after_m"
            ],
            "right": 0.0,
        },
    )
    assert settled["sides"]["left"]["apply_correction"] is False
    assert settled["sides"]["left"]["reason"] == "source_primary_contact_already_valid"


def test_walk_double_candidate_uses_already_valid_target_foot_without_forcing_peer():
    plan = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=5,
        source_clearance_m={"left": 0.011921, "right": 0.000700},
        target_clearance_m={"left": 0.004521, "right": 0.017285},
        height_scale=0.493607,
        candidate_contact_correction_reachable={"left": True, "right": False},
    )

    assert plan["source_contact_candidate_sides"] == ["left", "right"]
    assert plan["primary_support_side"] == "left"
    assert plan["support_sides"] == ["left"]
    assert plan["walking_selection_reason"] == (
        "double_source_candidate_existing_target_contact"
    )
    assert plan["primary_tie_break_order"] == [
        "target_contact_already_valid",
        "source_clearance_m",
        "side",
    ]
    assert plan["sides"]["left"]["apply_correction"] is False
    assert plan["sides"]["right"]["apply_correction"] is False
    assert plan["sides"]["right"]["reason"] == "nonprimary_clearance_unchanged"
    runner.validate_source_contact_ik_plan(plan)


def test_walk_single_candidate_must_correct_same_side_and_cannot_switch_feet():
    plan = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=8,
        source_clearance_m={"left": 0.002, "right": 0.050},
        target_clearance_m={"left": 0.020, "right": 0.005},
        height_scale=0.5,
        candidate_contact_correction_reachable={"left": True, "right": True},
    )

    assert plan["source_contact_candidate_sides"] == ["left"]
    assert plan["primary_support_side"] == "left"
    assert plan["walking_selection_reason"] == "single_source_candidate_same_side"
    assert plan["sides"]["left"]["ankle_correction_m"] == pytest.approx(-0.0051)
    assert plan["sides"]["right"]["apply_correction"] is False

    with pytest.raises(runner.RetargetError, match="single.*unreachable|unreachable.*single"):
        runner.plan_source_contact_ik_correction(
            action_name="Walking",
            source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
            frame=8,
            source_clearance_m={"left": 0.002, "right": 0.050},
            target_clearance_m={"left": 0.020, "right": 0.005},
            height_scale=0.5,
            candidate_contact_correction_reachable={"left": False, "right": True},
        )


def test_walk_double_hover_selects_minimum_reachable_edit_then_source_then_side():
    minimum_edit = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=12,
        source_clearance_m={"left": 0.010, "right": 0.002},
        target_clearance_m={"left": 0.018, "right": 0.025},
        height_scale=0.5,
        candidate_contact_correction_reachable={"left": True, "right": True},
    )
    assert minimum_edit["primary_support_side"] == "left"
    assert minimum_edit["walking_selection_reason"] == (
        "double_source_candidate_minimum_reachable_correction"
    )
    assert minimum_edit["primary_tie_break_order"] == [
        "requested_correction_reachable",
        "absolute_correction_m",
        "source_clearance_m",
        "side",
    ]

    reachable = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=12,
        source_clearance_m={"left": 0.010, "right": 0.002},
        target_clearance_m={"left": 0.018, "right": 0.025},
        height_scale=0.5,
        candidate_contact_correction_reachable={"left": False, "right": True},
    )
    assert reachable["primary_support_side"] == "right"

    source_tie = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=12,
        source_clearance_m={"left": 0.010, "right": 0.002},
        target_clearance_m={"left": 0.020, "right": 0.020},
        height_scale=0.5,
        candidate_contact_correction_reachable={"left": True, "right": True},
    )
    assert source_tie["primary_support_side"] == "right"

    side_tie = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=12,
        source_clearance_m={"left": 0.002, "right": 0.002},
        target_clearance_m={"left": 0.020, "right": 0.020},
        height_scale=0.5,
        candidate_contact_correction_reachable={"left": True, "right": True},
    )
    assert side_tie["primary_support_side"] == "left"


def test_idle_contact_policy_remains_bilateral_despite_walk_reachability_input():
    idle = runner.plan_source_contact_ik_correction(
        action_name="Standing_Idle",
        source_support_basis=runner.IDLE_SOURCE_SUPPORT_BASIS,
        frame=1,
        source_clearance_m={"left": 0.002, "right": 0.003},
        target_clearance_m={"left": 0.020, "right": -0.004},
        height_scale=0.5,
        candidate_contact_correction_reachable={"left": True, "right": True},
    )
    assert idle["primary_support_side"] is None
    assert idle["support_sides"] == ["left", "right"]
    assert idle["walking_selection_reason"] is None
    assert idle["sides"]["left"]["apply_correction"] is True
    assert idle["sides"]["right"]["apply_correction"] is True


def test_source_contact_ik_policy_fails_closed_and_validator_recomputes_evidence():
    with pytest.raises(runner.RetargetError, match="source contact candidate"):
        runner.plan_source_contact_ik_correction(
            action_name="Walking",
            source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
            frame=1,
            source_clearance_m={"left": 0.016, "right": 0.020},
            target_clearance_m={"left": 0.0, "right": 0.0},
            height_scale=0.5,
        )

    with pytest.raises(runner.RetargetError, match="0.030|correction"):
        runner.plan_source_contact_ik_correction(
            action_name="Walking",
            source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
            frame=1,
            source_clearance_m={"left": 0.001, "right": 0.080},
            target_clearance_m={"left": 0.060, "right": -0.001},
            height_scale=0.5,
        )

    with pytest.raises(runner.RetargetError, match="accumulated.*0.030|0.030.*accumulated"):
        runner.plan_source_contact_ik_correction(
            action_name="Walking",
            source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
            frame=1,
            source_clearance_m={"left": 0.001, "right": 0.080},
            target_clearance_m={"left": 0.017, "right": 0.020},
            height_scale=0.5,
            accumulated_absolute_correction_m={"left": 0.029, "right": 0.0},
        )

    valid = runner.plan_source_contact_ik_correction(
        action_name="Walking",
        source_support_basis=runner.WALK_SOURCE_SUPPORT_BASIS,
        frame=9,
        source_clearance_m={"left": 0.002, "right": 0.050},
        target_clearance_m={"left": 0.025, "right": 0.020},
        height_scale=0.5,
    )
    for field, value in (
        ("primary_support_side", "right"),
        ("contact_clearance_m", 0.014),
        ("readback_safety_margin_m", 0.0002),
    ):
        forged = json.loads(json.dumps(valid))
        forged[field] = value
        with pytest.raises(runner.RetargetError, match="forged|inconsistent"):
            runner.validate_source_contact_ik_plan(forged)

    with pytest.raises(runner.RetargetError, match="bilateral|Idle support"):
        runner.plan_source_contact_ik_correction(
            action_name="Standing_Idle",
            source_support_basis=runner.IDLE_SOURCE_SUPPORT_BASIS,
            frame=1,
            source_clearance_m={"left": 0.001, "right": 0.016},
            target_clearance_m={"left": 0.0, "right": 0.0},
            height_scale=0.5,
        )

    forged_basis = json.loads(json.dumps(valid))
    forged_basis["source_support_basis"] = runner.IDLE_SOURCE_SUPPORT_BASIS
    with pytest.raises(runner.RetargetError, match="forged|inconsistent"):
        runner.validate_source_contact_ik_plan(forged_basis)

    for field, value in (
        ("reason", "source_primary_contact_already_valid"),
        ("target_desired_clearance_m", 0.012),
        ("ankle_correction_m", -0.012),
    ):
        forged = json.loads(json.dumps(valid))
        forged["sides"]["left"][field] = value
        with pytest.raises(runner.RetargetError, match="forged|inconsistent"):
            runner.validate_source_contact_ik_plan(forged)


def test_source_contact_ik_final_readback_keeps_physical_gate_and_cross_foot_cap():
    evidence = runner.validate_source_contact_ik_frame_readback(
        action_name="Walking",
        frame=7,
        target_before_clearance_m={
            "left": 0.03395575284957886,
            "right": 0.03437689691781998,
        },
        target_after_clearance_m={
            "left": 0.03395312651991844,
            "right": 0.014900085346400738,
        },
        corrected_sides=("right",),
        required_contact_sides=("right",),
        desired_clearance_m={"left": None, "right": 0.0149},
        accumulated_absolute_correction_m={"left": 0.0, "right": 0.0236893},
    )

    assert evidence["contact_clearance_m"] == 0.015
    assert evidence["readback_safety_margin_m"] == 0.0001
    assert evidence["sides"]["right"]["actual_interval_result"] == "inside_physical_contact_interval"
    assert evidence["sides"]["right"]["desired_clearance_m"] == 0.0149
    assert evidence["sides"]["left"]["cross_foot_clearance_change_m"] == pytest.approx(
        -2.6263296604156494e-06
    )
    assert evidence["automatic_checks"] == "passed"
    runner.validate_source_contact_ik_frame_readback_evidence(evidence)

    for field, value in (
        ("actual_interval_result", "above_physical_contact_interval"),
        ("desired_clearance_m", 0.015),
    ):
        forged = json.loads(json.dumps(evidence))
        forged["sides"]["right"][field] = value
        with pytest.raises(runner.RetargetError, match="forged|inconsistent"):
            runner.validate_source_contact_ik_frame_readback_evidence(forged)


def test_source_contact_ik_final_readback_rejects_interval_cross_foot_and_cumulative_failures():
    common = {
        "action_name": "Walking",
        "frame": 7,
        "target_before_clearance_m": {"left": 0.020, "right": 0.034},
        "corrected_sides": ("right",),
        "required_contact_sides": ("right",),
        "desired_clearance_m": {"left": None, "right": 0.0149},
        "accumulated_absolute_correction_m": {"left": 0.0, "right": 0.020},
    }
    with pytest.raises(runner.RetargetError, match="physical.*interval|contact interval"):
        runner.validate_source_contact_ik_frame_readback(
            **common,
            target_after_clearance_m={"left": 0.020, "right": 0.015001},
        )
    with pytest.raises(runner.RetargetError, match="cross-foot|uncorrected"):
        runner.validate_source_contact_ik_frame_readback(
            **common,
            target_after_clearance_m={"left": 0.020101, "right": 0.0149},
        )
    excessive = dict(common)
    excessive["accumulated_absolute_correction_m"] = {
        "left": 0.0,
        "right": 0.030001,
    }
    with pytest.raises(runner.RetargetError, match="accumulated.*0.030|0.030.*accumulated"):
        runner.validate_source_contact_ik_frame_readback(
            **excessive,
            target_after_clearance_m={"left": 0.020, "right": 0.0149},
        )

    swing = runner.validate_source_contact_ik_frame_readback(
        action_name="Walking",
        frame=25,
        target_before_clearance_m={"left": 0.005, "right": -0.002},
        target_after_clearance_m={"left": 0.005002, "right": 0.020001},
        corrected_sides=("right",),
        required_contact_sides=("left",),
        desired_clearance_m={"left": None, "right": 0.020},
        accumulated_absolute_correction_m={"left": 0.0, "right": 0.022},
    )
    assert swing["sides"]["right"]["actual_interval_result"] == (
        "above_physical_contact_interval"
    )
    assert swing["sides"]["right"]["support_phase_result"] == (
        "nonpenetrating_swing"
    )


def _synthetic_exact_idle_world_inputs():
    phase = np.linspace(0.0, 2.0 * math.pi, 351)

    def trajectory(x, y, z, amplitude):
        return np.column_stack(
            (
                x + amplitude * np.sin(phase),
                y + 0.5 * amplitude * np.sin(phase),
                z + amplitude * (1.0 - np.cos(phase)),
            )
        ).tolist()

    joints = {
        "left": {
            "foot": trajectory(0.1, 0.0, 0.08, 0.0003),
            "toe": trajectory(0.1, -0.1, 0.0, 0.0005),
        },
        "right": {
            "foot": trajectory(-0.1, 0.0, 0.08, 0.00025),
            "toe": trajectory(-0.1, -0.1, 0.0, 0.00045),
        },
    }
    matrices = []
    for value in np.sin(phase):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] *= 0.01
        matrix[0, 3] = 0.001 * value
        matrices.append(matrix.tolist())
    return joints, matrices


def test_exact_idle_world_support_proves_four_joint_trajectories_and_recomputes():
    joints, matrices = _synthetic_exact_idle_world_inputs()
    evidence = runner.summarize_idle_source_support(
        frame_start=1,
        frame_end=351,
        fps=30,
        head_world_position_frame_start_m=(0.0, 0.0, 1.6),
        joint_world_positions_m=joints,
        object_world_matrices=matrices,
    )

    assert evidence["schema"] == "rocketbox_exact_idle_world_support_v1"
    assert evidence["action_name"] == "Standing_Idle"
    assert evidence["frame_count"] == 351
    assert evidence["coordinate_space"] == "blender_world_m"
    assert evidence["semantic_world_height_m"] == pytest.approx(1.6)
    assert evidence["object_transform"]["minimum_uniform_scale"] == pytest.approx(
        0.01
    )
    assert (
        evidence["object_transform"]["maximum_relative_scale_time_variation"]
        < 1.0e-12
    )
    assert (
        evidence["thresholds"][
            "maximum_object_uniform_scale_relative_variation"
        ]
        == runner.MAXIMUM_OBJECT_UNIFORM_SCALE_RELATIVE_VARIATION
    )
    assert evidence["sides"]["left"]["planted"] is True
    assert evidence["sides"]["right"]["planted"] is True
    assert evidence["sides"]["left"]["maximum_support_relative_clearance_m"] == pytest.approx(
        0.001
    )
    assert len(evidence["sides"]["left"]["support_relative_clearance_by_frame_m"]) == 351
    assert set(evidence["sides"]["right"]["joints"]) == {"foot", "toe"}
    assert (
        evidence["sides"]["right"]["joints"]["toe"]["maximum_speed_m_per_s"]
        < runner.MAXIMUM_IDLE_SPEED_MPS
    )
    runner.validate_idle_source_support(evidence)

    forged = json.loads(json.dumps(evidence))
    forged["sides"]["left"]["maximum_support_relative_clearance_m"] = 0.0
    with pytest.raises(runner.RetargetError, match="forged|inconsistent"):
        runner.validate_idle_source_support(forged)


def test_exact_idle_world_support_fails_closed_on_coverage_units_motion_and_loop():
    joints, matrices = _synthetic_exact_idle_world_inputs()
    missing_toe = copy.deepcopy(joints)
    del missing_toe["right"]["toe"]
    with pytest.raises(runner.RetargetError, match="foot.*toe|four.*trajectories"):
        runner.summarize_idle_source_support(
            frame_start=1,
            frame_end=351,
            fps=30,
            head_world_position_frame_start_m=(0.0, 0.0, 1.6),
            joint_world_positions_m=missing_toe,
            object_world_matrices=matrices,
        )

    with pytest.raises(runner.RetargetError, match="351|coverage"):
        runner.summarize_idle_source_support(
            frame_start=1,
            frame_end=350,
            fps=30,
            head_world_position_frame_start_m=(0.0, 0.0, 1.6),
            joint_world_positions_m={
                side: {part: values[:-1] for part, values in parts.items()}
                for side, parts in joints.items()
            },
            object_world_matrices=matrices[:-1],
        )

    fast = copy.deepcopy(joints)
    fast["left"]["foot"][1][0] += 0.01
    with pytest.raises(runner.RetargetError, match="speed|planted"):
        runner.summarize_idle_source_support(
            frame_start=1,
            frame_end=351,
            fps=30,
            head_world_position_frame_start_m=(0.0, 0.0, 1.6),
            joint_world_positions_m=fast,
            object_world_matrices=matrices,
        )

    open_loop = copy.deepcopy(joints)
    open_loop["right"]["toe"][-1][0] += 0.001
    with pytest.raises(runner.RetargetError, match="endpoint|loop|planted"):
        runner.summarize_idle_source_support(
            frame_start=1,
            frame_end=351,
            fps=30,
            head_world_position_frame_start_m=(0.0, 0.0, 1.6),
            joint_world_positions_m=open_loop,
            object_world_matrices=matrices,
        )

    nonuniform = copy.deepcopy(matrices)
    nonuniform[100][0][0] = 0.02
    with pytest.raises(runner.RetargetError, match="rotation|scale|condition"):
        runner.summarize_idle_source_support(
            frame_start=1,
            frame_end=351,
            fps=30,
            head_world_position_frame_start_m=(0.0, 0.0, 1.6),
            joint_world_positions_m=joints,
            object_world_matrices=nonuniform,
        )

    scale_drift = copy.deepcopy(matrices)
    for row in scale_drift[100:]:
        for axis in range(3):
            row[axis][axis] *= 1.00001
    with pytest.raises(runner.RetargetError, match="scale.*relative|dimensionless"):
        runner.summarize_idle_source_support(
            frame_start=1,
            frame_end=351,
            fps=30,
            head_world_position_frame_start_m=(0.0, 0.0, 1.6),
            joint_world_positions_m=joints,
            object_world_matrices=scale_drift,
        )


def test_per_foot_contact_summary_tracks_ratio_hover_and_stance_slide():
    clearances = [0.002, 0.003, 0.020, 0.08, 0.09, 0.07, 0.02, 0.004, 0.003, 0.002]
    anchors = [
        (0.0, 0.0),
        (0.001, 0.0),
        (0.002, 0.0),
        *([(0.1, 0.0)] * 3),
        (0.003, 0.0),
        (0.004, 0.0),
        (0.005, 0.0),
        (0.006, 0.0),
    ]

    summary = runner.summarize_foot_contact(
        clearances_m=clearances,
        anchor_xy_m=anchors,
        fps=30,
    )

    assert summary["contact_frame_count"] == 5
    assert summary["stance_frame_count"] == 7
    assert summary["contact_ratio"] == pytest.approx(0.5)
    assert summary["stance_contact_ratio"] == pytest.approx(5 / 7)
    assert summary["maximum_consecutive_hover_frames"] == 5
    assert summary["maximum_stance_slide_m"] <= 0.0031


def test_walking_rejects_long_flight_even_when_each_foot_has_nine_contacts():
    left_contact = [True] * 9 + [False] * 24
    right_contact = [False] * 24 + [True] * 9

    support = runner.summarize_support_union(
        left_contact=left_contact,
        right_contact=right_contact,
    )

    assert sum(left_contact) == 9
    assert sum(right_contact) == 9
    assert support["support_coverage_ratio"] == pytest.approx(18 / 33)
    assert support["maximum_consecutive_both_feet_airborne_frames"] == 15
    assert support["support_union"] == [True] * 9 + [False] * 15 + [True] * 9
    with pytest.raises(runner.RetargetError, match="Walking.*flight|both-feet-airborne"):
        runner.validate_walking_support(support)


def test_body_forward_and_loop_helpers_cover_root_velocity_and_foot_phase():
    forward = runner.body_forward_vector(
        left_shoulder=(1.0, 0.0, 1.5),
        right_shoulder=(-1.0, 0.0, 1.5),
        pelvis=(0.0, 0.0, 0.8),
        neck=(0.0, 0.0, 1.6),
    )
    assert np.allclose(forward, (0.0, -1.0, 0.0))
    trajectories = {
        "root": [(0.0, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, -0.2, 0.0)],
        "pelvis": [(0.0, 0.0, 0.8), (0.0, -0.1, 0.8), (0.0, -0.2, 0.8)],
    }
    assert runner.maximum_boundary_velocity_residual(trajectories, fps=30) <= 1.0e-12
    assert runner.foot_phase_is_continuous(
        {"left": [True, False, True], "right": [False, True, False]}
    )
    assert not runner.foot_phase_is_continuous(
        {"left": [True, False, False], "right": [False, True, False]}
    )


def test_loop_boundary_gate_is_per_semantic_source_calibrated_not_globally_relaxed():
    source = {
        "source_foot": {"residual_m_per_s": 1.0},
        "source_head": {"residual_m_per_s": 0.04},
    }
    target = {
        "target_foot": {"residual_m_per_s": 0.40},
        "target_head": {"residual_m_per_s": 0.14},
    }
    gate = runner.source_calibrated_boundary_velocity_gate(
        target_records=target,
        source_records=source,
        target_to_source={
            "target_foot": ["source_foot"],
            "target_head": ["source_head"],
        },
        height_scale=0.9,
    )

    assert gate["checks"]["target_foot"]["maximum_allowed_residual_m_per_s"] == (
        pytest.approx(0.9)
    )
    assert gate["checks"]["target_head"]["maximum_allowed_residual_m_per_s"] == (
        pytest.approx(runner.LOOP_BOUNDARY_VELOCITY_TOLERANCE_MPS)
    )
    assert gate["automatic_checks"] == "passed"

    target["target_head"]["residual_m_per_s"] = 0.16
    with pytest.raises(runner.RetargetError, match="corresponding sealed source"):
        runner.source_calibrated_boundary_velocity_gate(
            target_records=target,
            source_records=source,
            target_to_source={
                "target_foot": ["source_foot"],
                "target_head": ["source_head"],
            },
            height_scale=0.9,
        )


def test_deformation_calibration_tracks_approved_source_without_relaxing_hard_floors():
    calibrated = runner.calibrate_deformation_thresholds(
        source_minimum_shoulder_ratio=0.96,
        source_minimum_hip_ratio=0.93,
    )
    assert calibrated["required_minimum_shoulder_span_ratio"] == pytest.approx(0.91)
    assert calibrated["required_minimum_hip_span_ratio"] == pytest.approx(0.88)
    assert calibrated["allowed_maximum_skinned_edge_stretch_ratio"] == 1.35

    floor_limited = runner.calibrate_deformation_thresholds(
        source_minimum_shoulder_ratio=0.70,
        source_minimum_hip_ratio=0.70,
    )
    assert floor_limited["required_minimum_shoulder_span_ratio"] == 0.80
    assert floor_limited["required_minimum_hip_span_ratio"] == 0.85


def test_spine_slerp_preserves_endpoints_and_records_interpolation():
    source_deltas = {
        "Bip01 Spine": (1.0, 0.0, 0.0, 0.0),
        "Bip01 Spine1": (
            math.cos(math.pi / 8.0),
            0.0,
            0.0,
            math.sin(math.pi / 8.0),
        ),
        "Bip01 Spine2": (
            math.cos(math.pi / 4.0),
            0.0,
            0.0,
            math.sin(math.pi / 4.0),
        ),
    }
    plan = runner.build_spine_resample_plan(
        source_bones=runner.ROCKETBOX_SPINE_BONES,
        source_rest_heads=_z_points(0, 1, 2),
        target_bones=("t0", "t1", "t2", "t3"),
        target_rest_heads=_z_points(0, 1, 2, 3),
    )

    resampled = runner.resample_spine_quaternions(plan, source_deltas)

    assert np.allclose(resampled["t0"], source_deltas["Bip01 Spine"])
    assert all(abs(np.linalg.norm(value) - 1.0) <= 1.0e-12 for value in resampled.values())


def _axis_quaternion(axis, degrees):
    radians = math.radians(degrees) * 0.5
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    return np.asarray(
        [math.cos(radians), *(math.sin(radians) * vector)], dtype=np.float64
    )


def _quaternion_product(values):
    result = np.asarray((1.0, 0.0, 0.0, 0.0))
    for value in values:
        result = runner.multiply_quaternions(result, value)
    return result


@pytest.mark.parametrize("target_count", [2, 3, 4])
def test_spine_cumulative_field_preserves_noncommuting_total_rotation(target_count):
    source_deltas = {
        "Bip01 Spine": _axis_quaternion((1, 0, 0), 35),
        "Bip01 Spine1": _axis_quaternion((0, 1, 0), -50),
        "Bip01 Spine2": _axis_quaternion((0, 0, 1), 65),
    }
    target_names = tuple(f"target_{index}" for index in range(target_count))
    plan = runner.build_spine_resample_plan(
        source_bones=runner.ROCKETBOX_SPINE_BONES,
        source_rest_heads=_z_points(0, 1, 3),
        target_bones=target_names,
        target_rest_heads=_z_points(*range(target_count)),
    )

    target_deltas = runner.resample_spine_quaternions(plan, source_deltas)
    source_total = _quaternion_product(
        source_deltas[name] for name in runner.ROCKETBOX_SPINE_BONES
    )
    target_total = _quaternion_product(target_deltas[name] for name in target_names)

    assert abs(float(np.dot(source_total, target_total))) >= 1.0 - 1.0e-12
    assert np.allclose(target_deltas[target_names[0]], source_deltas["Bip01 Spine"])


def test_spine_slerp_does_not_mutate_the_hash_locked_source_deltas():
    first = np.array([2.0, 0.0, 0.0, 0.0])
    second = np.array([0.0, 0.0, 0.0, 3.0])
    first_before = first.copy()
    second_before = second.copy()

    runner.slerp_quaternion(first, second, 0.5)

    assert np.array_equal(first, first_before)
    assert np.array_equal(second, second_before)


def test_exact_limb_map_uses_static_generic_names_without_target_bone_inventory():
    validated = runner.validate_semantic_mapping(_semantic_mapping(("s0", "s1")))

    exact = runner.build_exact_semantic_correspondence(validated)

    assert exact["Bip01 Pelvis"] == "bone_pelvis"
    assert exact["Bip01 Head"] == "bone_head"
    assert exact["Bip01 L Foot"] == "bone_l_foot"
    assert exact["Bip01 R Toe0"] == "bone_r_toe"
    assert not set(runner.ROCKETBOX_SPINE_BONES) & set(exact)
    assert set(exact.values()).isdisjoint({"Bip01 Pelvis", "Bip01 Head"})


def _passing_action_metrics(action_name):
    if action_name == "Walking":
        contacts = {
            "left": [True, False, True],
            "right": [False, True, False],
        }
    else:
        contacts = {
            "left": [True, True, True],
            "right": [True, True, True],
        }
    support_union = [
        left or right
        for left, right in zip(contacts["left"], contacts["right"])
    ]
    performance_record = {
        "full_evaluated_vertex_count": 707000,
        "full_evaluated_edge_count": 1670000,
        "sampled_vertex_count": 80000,
        "wall_time_seconds": 0.12,
        "process_peak_rss_bytes": 4_000_000_000,
    }
    target_boundary_records = {
        "armature_root": {
            "start_velocity_m_per_s": [0.0, -1.0, 0.0],
            "end_velocity_m_per_s": [0.0, -0.98, 0.0],
            "residual_m_per_s": 0.02,
        }
    }
    source_boundary_records = {
        "armature_root": {
            "start_velocity_m_per_s": [0.0, -1.1, 0.0],
            "end_velocity_m_per_s": [0.0, -1.05, 0.0],
            "residual_m_per_s": 0.05,
        }
    }
    boundary_gate = runner.source_calibrated_boundary_velocity_gate(
        target_records=target_boundary_records,
        source_records=source_boundary_records,
        target_to_source={"armature_root": ["armature_root"]},
        height_scale=0.95,
    )
    return {
        "action_name": action_name,
        "frame_start": 1,
        "frame_end": 3,
        "rest_delta": {
            "target_rest_translations_preserved": True,
            "finite_rest_and_pose_matrices": True,
            "parent_first": True,
            "maximum_global_rest_alignment_error": 1.0e-8,
            "maximum_local_rest_delta_error": 1.0e-8,
            "maximum_target_translation_error_m": 1.0e-8,
        },
        "rotation_projection": _passing_rotation_projection_summary(action_name),
        "root_motion": {
            "axis_map_3x3": [list(row) for row in runner.AXIS_MAP_3X3],
            "height_scale": 0.95,
            "reconstruction_error_m": 1.0e-8,
            "endpoint_direction_dot_negative_y": 1.0 if action_name == "Walking" else None,
            "minimum_body_forward_dot_negative_y": 0.92,
            "minimum_body_forward_dot_travel": 0.91 if action_name == "Walking" else None,
        },
        "speed": {
            "source_scaled_speed_m_per_s": 1.2 if action_name == "Walking" else 0.0,
            "target_speed_m_per_s": 1.2 if action_name == "Walking" else 0.0,
            "absolute_reconstruction_error_m_per_s": 1.0e-8,
        },
        "loop": {
            "maximum_rotation_residual_rad": 1.0e-5,
            "root_cycle_reconstruction_error_m": 1.0e-8,
            "armature_root_rotation_residual_rad": 1.0e-5,
            "pelvis_local_translation_residual_m": 1.0e-5,
            "maximum_boundary_velocity_residual_m_per_s": 0.02,
            "source_calibrated_boundary_velocity": boundary_gate,
            "foot_phase_continuous": True,
        },
        "surface_contact_ik": {
            "schema": runner.SURFACE_CONTACT_IK_SCHEMA,
            "method": "evaluated_surface_minimum_to_vertical_two_bone_leg_ik_v1",
            "action_name": action_name,
            "frame_start": 1,
            "frame_end": 3,
            "frame_count": 3,
            "fixed_floor_z_m": 0.0,
            "safety_margin_m": runner.IK_CONTACT_READBACK_SAFETY_MARGIN_M,
            "maximum_allowed_ankle_correction_m": (
                runner.MAXIMUM_IK_ANKLE_CORRECTION_M
            ),
            "maximum_cumulative_upward_correction_m": 0.0,
            "root_pelvis_hip_translation_preserved": True,
            "ankle_xy_preserved": True,
            "foot_toe_global_orientation_preserved": True,
            "records_by_frame": {
                str(frame): {
                    "pre_minimum_z_m": {"left": 0.001, "right": 0.001},
                    "post_minimum_z_m": {"left": 0.001, "right": 0.001},
                    "cumulative_upward_correction_m": {
                        "left": 0.0,
                        "right": 0.0,
                    },
                    "iteration_count": 0,
                    "iterations": [],
                }
                for frame in range(1, 4)
            },
            "automatic_checks": "passed",
        },
        "floor": {
            "fixed_floor_z_m": 0.0,
            "grounding_correction_m": 0.004,
            "pre_ground_maximum_penetration_m": 0.004,
            "pre_ground_penetration_by_frame_m": {
                "1": 0.004,
                "2": 0.0,
                "3": 0.002,
            },
            "penetration_by_frame_m": {"1": 0.005, "2": 0.0, "3": 0.004},
            "maximum_penetration_m": 0.005,
            "maximum_per_foot_cycle_minimum_clearance_m": 0.015,
            "left_contact": True,
            "right_contact": True,
        },
        "contact": {
            "bilateral_contact_ratio": 0.0 if action_name == "Walking" else 1.0,
            "support_union": support_union,
            "support_coverage_ratio": 1.0,
            "maximum_consecutive_both_feet_airborne_frames": 0,
            "feet": {
                side: {
                    "frame_count": 3,
                    "contact_frame_count": sum(contacts[side]),
                    "stance_frame_count": sum(contacts[side]),
                    "contact_ratio": sum(contacts[side]) / 3,
                    "stance_contact_ratio": 1.0,
                    "maximum_consecutive_hover_frames": (
                        1 if side == "left" and action_name == "Walking"
                        else 2 if action_name == "Walking"
                        else 0
                    ),
                    "maximum_stance_slide_m": 0.01,
                    "maximum_stance_speed_m_per_s": 0.08,
                    "contact_by_frame": contacts[side],
                    "stance_by_frame": contacts[side],
                }
                for side in ("left", "right")
            },
        },
        "performance": {
            "schema": "indexed_evaluated_mesh_performance_v1",
            "passes": {
                phase: {
                    str(frame): dict(performance_record)
                    for frame in range(1, 4)
                }
                for phase in ("grounding", "quality")
            },
        },
        "sampling": {
            "method": "deterministic_spatial_skin_support_core_v2",
            "seed": runner.QUALITY_SAMPLE_SEED,
            "index_sha256": "d" * 64,
            "vertex_coverage_ratio": 0.04,
            "edge_coverage_ratio": 0.02,
            "lower_body_edge_coverage_ratio": 0.2,
        },
        "feet": {
            "minimum_foot_to_toe_rest_dot": 0.75,
            "inverted": False,
        },
        "deformation": {
            "calibration_basis": "approved_source_motion_and_static_bind_v1",
            "required_minimum_shoulder_span_ratio": 0.82,
            "required_minimum_hip_span_ratio": 0.87,
            "allowed_maximum_skinned_edge_stretch_ratio": 1.30,
            "minimum_shoulder_span_ratio": 0.9,
            "minimum_hip_span_ratio": 0.92,
            "maximum_skinned_edge_stretch_ratio": 1.20,
        },
        "roundtrip": {
            "one_armature": True,
            "one_skinned_mesh": True,
            "one_action": True,
            "action_name": action_name,
            "uv_present": True,
            "skin_present": True,
            "pbr_payloads_unchanged": True,
            "skeleton_exact": True,
            "loop_endpoints_exact": True,
            "finite_matrices": True,
        },
    }


def _passing_rotation_projection_summary(action_name):
    context = {
        "action": action_name,
        "frame": 1,
        "semantic_role": "pelvis",
        "source_bone": "Bip01 Pelvis",
        "target_bone": "bone_pelvis",
        "matrix_stage": "source_global_rest",
    }
    _, rest = runner.project_near_rotation(
        _rz(5) @ np.diag((1.000001, 1.0, 1.0)),
        "source global rest",
        context=context,
    )
    _, root = runner.project_uniform_scaled_rotation(
        0.01 * _rz(5),
        "source object world",
        context={
            **context,
            "semantic_role": "armature_root",
            "source_bone": "Bip01",
            "target_bone": None,
            "matrix_stage": "source_object_world",
        },
    )
    return runner.summarize_rotation_projections((rest, root))


@pytest.mark.parametrize("action_name", ["Walking", "Standing_Idle"])
def test_action_metrics_require_direction_speed_loop_floor_feet_deformation_and_readback(
    action_name,
):
    metrics = _passing_action_metrics(action_name)
    assert runner.validate_action_metrics(metrics)["status"] == "passed"


def test_action_metrics_reject_missing_or_out_of_bound_rotation_projection_evidence():
    missing = _passing_action_metrics("Walking")
    del missing["rotation_projection"]
    with pytest.raises(runner.RetargetError, match="rotation projection"):
        runner.validate_action_metrics(missing)

    invalid = _passing_action_metrics("Walking")
    invalid["rotation_projection"]["maximum_input_orthogonality_error"] = 6.0e-6
    with pytest.raises(runner.RetargetError, match="rotation projection"):
        runner.validate_action_metrics(invalid)


def test_rotation_projection_validation_recomputes_every_group_instead_of_trusting_aggregates():
    valid = _passing_rotation_projection_summary("Walking")
    assert runner.validate_rotation_projection_summary(valid)["status"] == "passed"

    corruptions = []
    value = copy.deepcopy(valid)
    value["per_bone_stage"][0]["sample_count"] = 0
    corruptions.append(value)
    value = copy.deepcopy(valid)
    value["per_bone_stage"][0]["action"] = None
    corruptions.append(value)
    value = copy.deepcopy(valid)
    value["per_bone_stage"][0]["matrix_stage"] = ""
    corruptions.append(value)
    value = copy.deepcopy(valid)
    value["per_bone_stage"][0]["maximum_input_orthogonality_error"] = math.inf
    corruptions.append(value)
    value = copy.deepcopy(valid)
    value["per_bone_stage"][0]["maximum_singular_value_deviation"] = 6.0e-6
    corruptions.append(value)
    value = copy.deepcopy(valid)
    value["per_bone_stage"][0]["worst_frames"][
        "input_orthogonality_max_error"
    ] = "1"
    corruptions.append(value)
    value = copy.deepcopy(valid)
    object_group = next(
        group
        for group in value["per_bone_stage"]
        if group["matrix_stage"] == "source_object_world"
    )
    object_group["minimum_uniform_scale"] = 0.0
    corruptions.append(value)

    for corrupted in corruptions:
        with pytest.raises(runner.RetargetError, match="rotation projection"):
            runner.validate_rotation_projection_summary(corrupted)


def _rotation_record(
    *,
    action="Walking",
    frame=1,
    semantic_role,
    source_bone,
    target_bone,
    matrix_stage,
    uniform_scale=None,
):
    context = {
        "action": action,
        "frame": frame,
        "semantic_role": semantic_role,
        "source_bone": source_bone,
        "target_bone": target_bone,
        "matrix_stage": matrix_stage,
    }
    if uniform_scale is None:
        _, record = runner.project_near_rotation(
            _rz(3), matrix_stage, context=context
        )
    else:
        _, record = runner.project_uniform_scaled_rotation(
            float(uniform_scale) * _rz(3), matrix_stage, context=context
        )
    return record


def test_rotation_projection_stage_context_contract_accepts_only_real_pipeline_shapes():
    semantic = runner.validate_semantic_mapping(_semantic_mapping())
    records = (
        _rotation_record(
            semantic_role="pelvis",
            source_bone="Bip01 Pelvis",
            target_bone=None,
            matrix_stage="source_global_pose",
        ),
        _rotation_record(
            semantic_role="pelvis",
            source_bone="Bip01 Pelvis",
            target_bone="bone_pelvis",
            matrix_stage="source_global_pose",
        ),
        _rotation_record(
            semantic_role="spine",
            source_bone="Bip01 Spine",
            target_bone=None,
            matrix_stage="source_local_pose",
        ),
        _rotation_record(
            semantic_role="armature_root",
            source_bone="Bip01",
            target_bone=None,
            matrix_stage="source_object_world",
            uniform_scale=0.01,
        ),
        _rotation_record(
            frame=None,
            semantic_role="armature_root",
            source_bone=None,
            target_bone="Armature",
            matrix_stage="target_object_world",
            uniform_scale=1.0,
        ),
        _rotation_record(
            frame=None,
            semantic_role="canonical_axis",
            source_bone=None,
            target_bone=None,
            matrix_stage="canonical_axis_identity",
        ),
        _rotation_record(
            frame=None,
            semantic_role="spine",
            source_bone="Bip01 Spine1",
            target_bone=None,
            matrix_stage="source_local_rest",
        ),
        _rotation_record(
            frame=None,
            semantic_role="spine",
            source_bone=None,
            target_bone="bone_spine_a",
            matrix_stage="target_local_rest",
        ),
        _rotation_record(
            frame=None,
            semantic_role="pelvis",
            source_bone="Bip01 Pelvis",
            target_bone="bone_pelvis",
            matrix_stage="source_global_rest",
        ),
        _rotation_record(
            frame=None,
            semantic_role="pelvis",
            source_bone="Bip01 Pelvis",
            target_bone="bone_pelvis",
            matrix_stage="target_global_rest",
        ),
    )

    summary = runner.summarize_rotation_projections(records)

    assert runner.validate_rotation_projection_summary(
        summary,
        expected_action="Walking",
        frame_start=1,
        frame_end=3,
        semantic_mapping=semantic,
    ) == {"status": "passed"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda group: group.__setitem__("semantic_role", "not_a_role"),
            "context|semantic",
        ),
        (
            lambda group: group.__setitem__("source_bone", None),
            "context|source bone",
        ),
        (
            lambda group: group.__setitem__("target_bone", "wrong_target"),
            "context|target bone",
        ),
        (
            lambda group: group["worst_frames"].__setitem__(
                "input_orthogonality_max_error", 999999
            ),
            "frame|context",
        ),
        (
            lambda group: group.__setitem__("matrix_stage", "invented_stage"),
            "stage|context",
        ),
    ],
)
def test_rotation_projection_context_tampering_is_rejected(mutation, message):
    metrics = _passing_action_metrics("Walking")
    group = next(
        item
        for item in metrics["rotation_projection"]["per_bone_stage"]
        if item["matrix_stage"] == "source_global_rest"
    )
    mutation(group)

    with pytest.raises(runner.RetargetError, match=message):
        runner.validate_action_metrics(
            metrics,
            semantic_mapping=runner.validate_semantic_mapping(_semantic_mapping()),
        )


def test_rotation_projection_dynamic_frames_and_action_must_match_action_metrics():
    semantic = runner.validate_semantic_mapping(_semantic_mapping())
    wrong_action = _passing_action_metrics("Walking")
    for group in wrong_action["rotation_projection"]["per_bone_stage"]:
        group["action"] = "Standing_Idle"
    with pytest.raises(runner.RetargetError, match="action|context"):
        runner.validate_action_metrics(wrong_action, semantic_mapping=semantic)

    dynamic_none = _passing_action_metrics("Walking")
    object_group = next(
        group
        for group in dynamic_none["rotation_projection"]["per_bone_stage"]
        if group["matrix_stage"] == "source_object_world"
    )
    object_group["worst_frames"] = {
        field: None for field in object_group["worst_frames"]
    }
    with pytest.raises(runner.RetargetError, match="frame|context"):
        runner.validate_action_metrics(dynamic_none, semantic_mapping=semantic)


def test_rotation_projection_object_scale_and_group_uniqueness_are_mandatory():
    semantic = runner.validate_semantic_mapping(_semantic_mapping())
    missing_scale = _passing_action_metrics("Walking")
    object_group = next(
        group
        for group in missing_scale["rotation_projection"]["per_bone_stage"]
        if group["matrix_stage"] == "source_object_world"
    )
    del object_group["minimum_uniform_scale"]
    del object_group["maximum_uniform_scale"]
    with pytest.raises(runner.RetargetError, match="uniform.scale|object"):
        runner.validate_action_metrics(missing_scale, semantic_mapping=semantic)

    duplicate = _passing_action_metrics("Walking")
    duplicate_group = copy.deepcopy(duplicate["rotation_projection"]["per_bone_stage"][0])
    duplicate["rotation_projection"]["per_bone_stage"].append(duplicate_group)
    duplicate["rotation_projection"]["context_group_count"] += 1
    duplicate["rotation_projection"]["record_count"] += duplicate_group["sample_count"]
    duplicate["rotation_projection"]["projection_applied_count"] += duplicate_group[
        "projection_applied_count"
    ]
    duplicate["rotation_projection"]["right_handed_record_count"] += duplicate_group[
        "right_handed_record_count"
    ]
    with pytest.raises(runner.RetargetError, match="duplicate|context"):
        runner.validate_action_metrics(duplicate, semantic_mapping=semantic)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("maximum_input_orthogonality_error", -1.0),
        ("maximum_singular_value_deviation", -1.0),
        ("maximum_condition_number", 0.5),
        ("maximum_polar_residual", -1.0),
        ("maximum_output_orthogonality_error", -1.0),
    ],
)
def test_rotation_projection_group_and_top_aggregates_reject_impossible_domains(
    field, invalid
):
    summary = _passing_rotation_projection_summary("Walking")
    for group in summary["per_bone_stage"]:
        group[field] = invalid
    summary[field] = invalid

    with pytest.raises(runner.RetargetError, match="rotation projection"):
        runner.validate_rotation_projection_summary(summary)


@pytest.mark.parametrize(
    "path,value,message",
    [
        (("root_motion", "axis_map_3x3"), [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], "axis"),
        (("rest_delta", "maximum_global_rest_alignment_error"), 0.01, "global rest"),
        (("rest_delta", "maximum_local_rest_delta_error"), 0.01, "spine local"),
        (("rest_delta", "maximum_target_translation_error_m"), 0.01, "translation"),
        (("root_motion", "reconstruction_error_m"), 0.02, "root motion"),
        (("speed", "absolute_reconstruction_error_m_per_s"), 0.1, "speed"),
        (("loop", "maximum_rotation_residual_rad"), 0.1, "loop"),
        (("loop", "armature_root_rotation_residual_rad"), 0.1, "root rotation"),
        (("loop", "pelvis_local_translation_residual_m"), 0.1, "pelvis translation"),
        (("loop", "maximum_boundary_velocity_residual_m_per_s"), 0.3, "boundary velocity"),
        (("loop", "foot_phase_continuous"), False, "foot phase"),
        (("floor", "maximum_penetration_m"), 0.01001, "penetration"),
        (("floor", "grounding_correction_m"), 0.01001, "grounding correction"),
        (("floor", "pre_ground_maximum_penetration_m"), 0.01001, "pre-ground"),
        (("floor", "maximum_per_foot_cycle_minimum_clearance_m"), 0.031, "hover"),
        (("floor", "left_contact"), False, "bilateral"),
        (("feet", "inverted"), True, "inversion"),
        (("deformation", "minimum_shoulder_span_ratio"), 0.79, "shoulder"),
        (("deformation", "minimum_hip_span_ratio"), 0.84, "hip"),
        (("deformation", "maximum_skinned_edge_stretch_ratio"), 1.36, "tearing"),
        (("roundtrip", "pbr_payloads_unchanged"), False, "PBR"),
        (("roundtrip", "one_action"), False, "one-action"),
    ],
)
def test_action_metrics_fail_closed_for_every_required_gate(path, value, message):
    metrics = _passing_action_metrics("Walking")
    metrics[path[0]][path[1]] = value
    with pytest.raises(runner.RetargetError, match=message):
        runner.validate_action_metrics(metrics)


def test_walking_direction_must_be_front_negative_y_but_idle_has_no_fake_travel():
    walking = _passing_action_metrics("Walking")
    walking["root_motion"]["endpoint_direction_dot_negative_y"] = -1.0
    with pytest.raises(runner.RetargetError, match="FRONT -Y|reverse"):
        runner.validate_action_metrics(walking)

    idle = _passing_action_metrics("Standing_Idle")
    idle["speed"]["target_speed_m_per_s"] = 0.05
    with pytest.raises(runner.RetargetError, match="idle.*speed"):
        runner.validate_action_metrics(idle)


def test_walking_body_forward_must_agree_with_travel_and_front_negative_y():
    walking = _passing_action_metrics("Walking")
    walking["root_motion"]["minimum_body_forward_dot_negative_y"] = -0.8
    with pytest.raises(runner.RetargetError, match="body forward|backward"):
        runner.validate_action_metrics(walking)

    walking = _passing_action_metrics("Walking")
    walking["root_motion"]["minimum_body_forward_dot_travel"] = -0.8
    with pytest.raises(runner.RetargetError, match="body forward|travel"):
        runner.validate_action_metrics(walking)


def test_deformation_thresholds_cannot_relax_below_calibrated_visual_floors():
    metrics = _passing_action_metrics("Walking")
    metrics["deformation"]["required_minimum_shoulder_span_ratio"] = 0.5
    with pytest.raises(runner.RetargetError, match="calibrated shoulder"):
        runner.validate_action_metrics(metrics)

    metrics = _passing_action_metrics("Walking")
    metrics["deformation"]["allowed_maximum_skinned_edge_stretch_ratio"] = 4.0
    with pytest.raises(runner.RetargetError, match="calibrated edge"):
        runner.validate_action_metrics(metrics)


def test_temporal_contact_gates_reject_hidden_hover_and_idle_one_foot_support():
    walking = _passing_action_metrics("Walking")
    walking["contact"]["feet"]["left"]["stance_contact_ratio"] = 0.4
    with pytest.raises(runner.RetargetError, match="stance.*contact"):
        runner.validate_action_metrics(walking)

    walking = _passing_action_metrics("Walking")
    walking["contact"]["feet"]["right"]["maximum_consecutive_hover_frames"] = 30
    with pytest.raises(runner.RetargetError, match="consecutive.*hover"):
        runner.validate_action_metrics(walking)

    idle = _passing_action_metrics("Standing_Idle")
    idle["contact"]["feet"]["right"]["contact_ratio"] = 0.5
    with pytest.raises(runner.RetargetError, match="Idle.*contact"):
        runner.validate_action_metrics(idle)

    walking = _passing_action_metrics("Walking")
    walking["contact"]["feet"]["left"]["maximum_stance_slide_m"] = 0.50
    walking["contact"]["feet"]["left"]["maximum_stance_speed_m_per_s"] = 4.0
    walking_result = runner.validate_action_metrics(walking)
    assert walking_result["stance_slide_policy"] == (
        "recorded_advisory_clearance_phase_v1"
    )
    assert walking_result["recorded_stance_slide"]["left"] == {
        "maximum_stance_slide_m": 0.50,
        "maximum_stance_speed_m_per_s": 4.0,
    }

    idle = _passing_action_metrics("Standing_Idle")
    idle["contact"]["feet"]["left"]["maximum_stance_slide_m"] = 0.04
    with pytest.raises(runner.RetargetError, match="Idle left stance slide"):
        runner.validate_action_metrics(idle)

    walking = _passing_action_metrics("Walking")
    walking["contact"]["feet"]["left"]["maximum_stance_slide_m"] = -0.01
    with pytest.raises(runner.RetargetError, match="cannot be negative"):
        runner.validate_action_metrics(walking)

    walking = _passing_action_metrics("Walking")
    walking["contact"]["feet"]["left"]["contact_by_frame"] = [True, True, False]
    walking["contact"]["feet"]["left"]["stance_by_frame"] = [True, True, False]
    with pytest.raises(runner.RetargetError, match="foot phase"):
        runner.validate_action_metrics(walking)


def test_every_recorded_frame_must_meet_the_penetration_limit():
    metrics = _passing_action_metrics("Walking")
    assert "maximum_unsupported_clearance_m" not in metrics["floor"]
    assert metrics["floor"]["maximum_per_foot_cycle_minimum_clearance_m"] == 0.015
    metrics["floor"]["penetration_by_frame_m"]["2"] = 0.01001
    metrics["floor"]["maximum_penetration_m"] = 0.005
    with pytest.raises(runner.RetargetError, match="per-frame penetration"):
        runner.validate_action_metrics(metrics)

    metrics = _passing_action_metrics("Walking")
    del metrics["floor"]["penetration_by_frame_m"]["2"]
    with pytest.raises(runner.RetargetError, match="every action frame"):
        runner.validate_action_metrics(metrics)

    metrics = _passing_action_metrics("Walking")
    del metrics["floor"]["pre_ground_penetration_by_frame_m"]["2"]
    with pytest.raises(runner.RetargetError, match="pre-ground.*every action frame"):
        runner.validate_action_metrics(metrics)


def test_performance_telemetry_covers_full_evaluated_mesh_for_every_pass_and_frame():
    metrics = _passing_action_metrics("Walking")
    assert runner.validate_action_metrics(metrics)["status"] == "passed"

    missing = _passing_action_metrics("Walking")
    del missing["performance"]["passes"]["quality"]["2"]
    with pytest.raises(runner.RetargetError, match="performance.*every action frame"):
        runner.validate_action_metrics(missing)

    invalid = _passing_action_metrics("Walking")
    invalid["performance"]["passes"]["quality"]["2"]["sampled_vertex_count"] = 800000
    with pytest.raises(runner.RetargetError, match="sampled.*full evaluated mesh"):
        runner.validate_action_metrics(invalid)


def test_atomic_directory_publication_is_no_replace(tmp_path):
    staging = tmp_path / ".animation.test.staging"
    destination = tmp_path / "animation"
    staging.mkdir()
    _write(staging / "retarget_manifest.json", "{}\n")

    runner.rename_directory_noreplace(staging, destination)

    assert destination.is_dir()
    assert not staging.exists()
    replacement = tmp_path / ".animation.other.staging"
    replacement.mkdir()
    with pytest.raises(runner.RetargetError, match="already exists|no-replace"):
        runner.rename_directory_noreplace(replacement, destination)
    assert replacement.is_dir()


def test_failed_staging_is_preserved_without_publishing_readiness(tmp_path):
    destination = tmp_path / "animation"
    staging = tmp_path / ".animation.run.staging"
    staging.mkdir()
    _write(staging / "partial.glb", b"failed but preserved")

    evidence = runner.preserve_failed_staging(
        staging=staging,
        output_dir=destination,
        asset_id="route2_male",
        error=RuntimeError("roundtrip rejected"),
        authenticated={"static": {"sha256": "a" * 64}},
    )

    assert not destination.exists()
    assert not staging.exists()
    assert evidence.parent.name.startswith("animation.failed.")
    assert (evidence.parent / "partial.glb").read_bytes() == b"failed but preserved"
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["decision"] == "rejected"
    assert payload["readiness_bundle_published"] is False
    inventory_path = evidence.parent.parent / payload["external_inventory_descriptor"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["inventory_scope"] == (
        "rejected_bundle_files_excluding_this_external_descriptor"
    )
    assert set(inventory["artifacts"]) == {"partial.glb", "retarget_failure.json"}
    for name, descriptor in inventory["artifacts"].items():
        artifact = evidence.parent / name
        assert descriptor["sha256"] == _sha256(artifact)
        assert descriptor["size_bytes"] == artifact.stat().st_size
        assert artifact.stat().st_mode & 0o777 == 0o444
    assert evidence.parent.stat().st_mode & 0o777 == 0o555
    assert inventory_path.stat().st_mode & 0o777 == 0o444
    evidence.parent.chmod(0o700)
    for artifact in evidence.parent.iterdir():
        artifact.chmod(0o600)
    inventory_path.chmod(0o600)


def test_pre_staging_failure_has_an_external_hashed_inventory_descriptor(tmp_path):
    destination = tmp_path / "animation"
    evidence = runner._write_failure_evidence(
        output_dir=destination,
        asset_id="property_0042_person_a",
        error=RuntimeError("authentication rejected"),
        authenticated={"static": {"sha256": "a" * 64}},
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    inventory_path = evidence.parent / payload["external_inventory_descriptor"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["inventory_scope"] == (
        "standalone_failure_evidence_excluding_this_external_descriptor"
    )
    assert inventory["artifacts"] == {
        evidence.name: runner.file_descriptor(evidence, public_path=evidence.name)
    }
    assert evidence.stat().st_mode & 0o777 == 0o444
    assert inventory_path.stat().st_mode & 0o777 == 0o444
    evidence.chmod(0o600)
    inventory_path.chmod(0o600)


def test_failure_publisher_serializes_structured_grounding_evidence(tmp_path):
    error = runner.GroundingError(
        "constant ground correction exceeds 0.010 m",
        evidence={
            "schema": "tokenrig_grounding_rejection_v1",
            "action_name": "Walking",
            "worst_frame": 25,
            "worst_minimum_z_m": -0.0223123989,
            "worst_sample": {
                "vertex_index": 77852,
                "dominant_semantic_region": "right_toe",
            },
            "frame_minimum_z_m": {"25": -0.0223123989},
        },
    )

    evidence_path = runner._write_failure_evidence(
        output_dir=tmp_path / "retarget_v1",
        asset_id="route2_male",
        error=error,
        authenticated={"code": {"sha256": "a" * 64}},
    )

    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert payload["error_evidence"] == error.evidence
    assert payload["error_evidence"]["worst_sample"]["vertex_index"] == 77852
    inventory_path = evidence_path.parent / payload["external_inventory_descriptor"]
    evidence_path.chmod(0o600)
    inventory_path.chmod(0o600)


def test_success_bundle_is_sealed_readonly_before_atomic_publication(tmp_path):
    staging = tmp_path / ".animation.staging"
    staging.mkdir()
    artifact = _write(staging / "walking.glb", b"walking")
    manifest = _write(staging / "retarget_manifest.json", b"{}\n")

    runner.seal_staged_bundle_readonly(staging)

    assert artifact.stat().st_mode & 0o777 == 0o444
    assert manifest.stat().st_mode & 0o777 == 0o444
    assert staging.stat().st_mode & 0o777 == 0o555
    source = SCRIPT.read_text(encoding="utf-8")
    publication = source[source.index("def run_retarget(") :]
    assert publication.index("seal_staged_bundle_readonly(staging)") < publication.index(
        "rename_directory_noreplace(staging, destination)"
    )
    staging.chmod(0o700)
    artifact.chmod(0o600)
    manifest.chmod(0o600)


@pytest.mark.parametrize(
    "sealer",
    [runner._seal_failure_bundle_readonly, runner.seal_staged_bundle_readonly],
)
def test_bundle_sealers_fsync_files_and_directory_after_readonly_chmod(
    tmp_path, monkeypatch, sealer
):
    staging = tmp_path / sealer.__name__
    staging.mkdir()
    artifact = _write(staging / "artifact.bin", b"immutable evidence")
    descriptor_paths = {}
    fsync_modes = []
    real_open = runner.os.open
    real_close = runner.os.close
    real_fsync = runner.os.fsync

    def tracked_open(path, flags):
        descriptor = real_open(path, flags)
        descriptor_paths[descriptor] = Path(path)
        return descriptor

    def tracked_fsync(descriptor):
        path = descriptor_paths.get(descriptor)
        if path is not None:
            fsync_modes.append((path, path.stat().st_mode & 0o777))
        return real_fsync(descriptor)

    def tracked_close(descriptor):
        descriptor_paths.pop(descriptor, None)
        return real_close(descriptor)

    monkeypatch.setattr(runner.os, "open", tracked_open)
    monkeypatch.setattr(runner.os, "fsync", tracked_fsync)
    monkeypatch.setattr(runner.os, "close", tracked_close)

    sealer(staging)

    assert (artifact, 0o444) in fsync_modes
    assert (staging, 0o555) in fsync_modes
    staging.chmod(0o700)
    artifact.chmod(0o600)


def test_manifest_records_every_input_output_map_rest_and_quality_gate(tmp_path):
    artifact = _write(tmp_path / "walking.glb", b"walking")
    authenticated = {
        "static": {"static_qa": {"sha256": "a" * 64}},
        "walk": {"retarget_blend": {"sha256": "b" * 64}},
        "idle": {"sha256": "c" * 64},
    }
    metrics = {
        "schema": runner.METRICS_SCHEMA,
        "actions": {
            name: _passing_action_metrics(name) for name in runner.ACTION_NAMES.values()
        },
        "rest_matrices": {"source": {}, "target": {}},
        "semantic_mapping": runner.validate_semantic_mapping(_semantic_mapping()),
        "spine_resample_plan": [{"target_bone": "bone_spine_a", "weights": [1.0, 0.0]}],
        "export_parameters": {
            name: runner.gltf_export_parameters(name)
            for name in runner.ACTION_NAMES.values()
        },
    }

    manifest = runner.build_retarget_manifest(
        asset_id="property_0042_person_a",
        base_avatar_id="rocketbox_male_adult_01",
        authenticated=authenticated,
        metrics=metrics,
        artifacts={"walking.glb": runner.file_descriptor(artifact)},
        command=["blender", "--background"],
        blender_version="4.2.0",
    )

    assert manifest["schema"] == runner.MANIFEST_SCHEMA
    assert manifest["asset_id"] == "property_0042_person_a"
    assert manifest["base_avatar_id"] == "rocketbox_male_adult_01"
    assert manifest["state_classification"] == "research_candidate"
    assert manifest["canonical_front"] == "negative-y"
    assert manifest["axis_transform_at_retarget"] == "identity"
    assert manifest["authenticated_inputs"] == authenticated
    assert manifest["semantic_mapping_sha256"] == runner.sha256_json(
        metrics["semantic_mapping"]
    )
    assert manifest["rest_matrices_sha256"] == runner.sha256_json(
        metrics["rest_matrices"]
    )
    assert manifest["spine_resample_plan"] == metrics["spine_resample_plan"]
    assert manifest["export_parameters_sha256"] == runner.sha256_json(
        metrics["export_parameters"]
    )
    assert set(manifest["actions"]) == {"Walking", "Standing_Idle"}


def test_manifest_accepts_the_runtime_per_action_spine_weight_plans(tmp_path):
    artifact = _write(tmp_path / "walking.glb", b"walking")
    spine_plans = {
        "Walking": [{"target_bone": "s0", "weights": [1.0, 0.0]}],
        "Standing_Idle": [{"target_bone": "s0", "weights": [1.0, 0.0]}],
    }
    metrics = {
        "schema": runner.METRICS_SCHEMA,
        "actions": {
            name: _passing_action_metrics(name) for name in runner.ACTION_NAMES.values()
        },
        "rest_matrices": {"source": {}, "target": {}},
        "semantic_mapping": runner.validate_semantic_mapping(_semantic_mapping()),
        "spine_resample_plan": spine_plans,
        "export_parameters": {
            name: runner.gltf_export_parameters(name)
            for name in runner.ACTION_NAMES.values()
        },
    }

    manifest = runner.build_retarget_manifest(
        asset_id="route2_male",
        base_avatar_id="rocketbox_male_adult_01",
        authenticated={},
        metrics=metrics,
        artifacts={"walking.glb": runner.file_descriptor(artifact)},
        command=["blender"],
        blender_version="4.2.0",
    )

    assert manifest["spine_resample_plan"] == spine_plans


def test_blender_runner_is_generic_target_only_and_performs_two_glb_roundtrips():
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "TARGET_BONES",
        "EXPECTED_SOURCE_ONLY_NUB_BONES",
        "len(armature.data.bones) == 80",
        "len(armature.data.bones) != 80",
    )
    assert all(item not in source for item in forbidden)
    required = (
        "bpy.ops.wm.open_mainfile",
        "bpy.ops.import_scene.gltf",
        "bpy.ops.import_scene.fbx",
        "bpy.ops.wm.save_as_mainfile",
        "bpy.ops.export_scene.gltf",
        "gltf_export_parameters",
        "Walking",
        "Standing_Idle",
        "walking.glb",
        "standing_idle.glb",
        "animated.blend",
        "retarget_manifest.json",
        "retarget_metrics.json",
        "capture_rest_matrices",
        "cache_source_motion",
        "bake_rest_corrected_action",
        "roundtrip_validate_action",
        "compare_mesh_contracts",
        "compare_surface_references",
        "compare_skin_by_position",
        "remove_source_objects",
        "validate_target_only_scene",
        "bpy.ops.outliner.orphans_purge",
    )
    assert all(item in source for item in required)
    assert "Matrix(target_global.tolist()).to_quaternion()" in source
    assert "root_world_scale" in source
    assert "pelvis_local_scale" in source


def test_walk_and_idle_reuse_one_canonical_target_object_base():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "target_base_transform = capture_target_base_transform(target_armature)" in source
    assert source.count("target_base_transform=target_base_transform") >= 2
    assert "restore_target_base_transform(target_armature, target_base_transform)" in source


def test_roundtrip_compares_bind_geometry_at_the_canonical_unanimated_base():
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index("def roundtrip_validate_action(")
    end = source.index("\ndef _stable_sample_ranks", start)
    function = source[start:end]
    assert "armature.animation_data.action = None" in function
    assert "restore_target_base_transform(armature, target_base_transform)" in function
    assert function.index("animation_data.action = None") < function.index(
        "capture_blender_mesh_contract(mesh)"
    )


def test_roundtrip_compares_serialized_animation_endpoint_matrices():
    expected = {
        "frame_start": 1,
        "frame_end": 33,
        "frames": {
            "1": {"armature_world": np.eye(4).tolist(), "bones": {"pelvis": np.eye(4).tolist()}},
            "33": {"armature_world": np.eye(4).tolist(), "bones": {"pelvis": np.eye(4).tolist()}},
        },
    }
    actual = json.loads(json.dumps(expected))
    assert runner.compare_animation_endpoint_matrices(expected, actual) == 0.0
    actual["frames"]["33"]["bones"]["pelvis"][0][0] += 0.01
    with pytest.raises(runner.RetargetError, match="endpoint"):
        runner.compare_animation_endpoint_matrices(expected, actual)

    source = SCRIPT.read_text(encoding="utf-8")
    roundtrip = source[
        source.index("def roundtrip_validate_action(") : source.index("\ndef _stable_sample_ranks")
    ]
    assert "capture_animation_endpoint_matrices" in roundtrip
    assert "compare_animation_endpoint_matrices" in roundtrip


def test_blender_42_imported_action_uses_exact_glb_nla_name_before_normalization():
    action = SimpleNamespace(name="Walking_Armature")
    strip = SimpleNamespace(name="Walking_Armature", action=action)
    track = SimpleNamespace(name="Walking", strips=[strip])
    armature = SimpleNamespace(
        animation_data=SimpleNamespace(action=action, nla_tracks=[track])
    )

    normalized, evidence = runner.authenticate_and_normalize_imported_action(
        armature,
        action_name="Walking",
        actions=[action],
    )

    assert normalized is action
    assert action.name == "Walking"
    assert evidence["glb_animation_name"] == "Walking"
    assert evidence["nla_track_name"] == "Walking"
    assert evidence["imported_action_datablock_name"] == "Walking_Armature"
    assert evidence["normalized_action_datablock_name"] == "Walking"

    action = SimpleNamespace(name="Walking_Armature")
    bad_track = SimpleNamespace(
        name="WrongAnimation", strips=[SimpleNamespace(action=action)]
    )
    bad_armature = SimpleNamespace(
        animation_data=SimpleNamespace(action=action, nla_tracks=[bad_track])
    )
    with pytest.raises(runner.RetargetError, match="NLA name"):
        runner.authenticate_and_normalize_imported_action(
            bad_armature,
            action_name="Walking",
            actions=[action],
        )


def test_runtime_quality_uses_indexed_samples_not_full_mesh_or_edge_scans_per_frame():
    source = SCRIPT.read_text(encoding="utf-8")
    measure = source[
        source.index("def _measure_action_quality(") : source.index("\ndef _remove_everything_except")
    ]
    grounding = source[
        source.index("def _apply_constant_grounding(") : source.index("\ndef _joint_world")
    ]
    assert "_evaluated_indexed_positions" in measure
    assert "_evaluated_indexed_positions" in grounding
    assert "_evaluated_mesh_positions" not in measure
    assert 'mesh_rest["edges"]' not in measure
    assert "full_evaluated_vertex_count" in source
    assert "process_peak_rss_bytes" in source
    assert "wall_time_seconds" in source


def test_runner_is_import_safe_and_main_requires_blender_only_at_execution():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import bpy\n" not in source
    assert "def _require_bpy()" in source
    assert "TOKENRIG_RETARGET_PUBLISHED" in source


def test_execution_requires_the_pinned_blender_4_2_runtime():
    runner.validate_blender_runtime(SimpleNamespace(app=SimpleNamespace(version=(4, 2, 5))))
    with pytest.raises(runner.RetargetError, match="Blender 4.2"):
        runner.validate_blender_runtime(SimpleNamespace(app=SimpleNamespace(version=(5, 0, 1))))


def test_blender_4_2_export_names_each_merged_active_action_explicitly():
    walking = runner.gltf_export_parameters("Walking")
    idle = runner.gltf_export_parameters("Standing_Idle")
    assert walking["export_animation_mode"] == "ACTIVE_ACTIONS"
    assert walking["export_nla_strips_merged_animation_name"] == "Walking"
    assert idle["export_nla_strips_merged_animation_name"] == "Standing_Idle"
    assert {
        "export_frame_range": True,
        "export_frame_step": 1,
        "export_def_bones": False,
        "export_rest_position_armature": True,
        "export_anim_slide_to_zero": False,
        "export_optimize_animation_size": False,
        "export_anim_single_armature": True,
        "export_reset_pose_bones": True,
        "export_negative_frame": "SLIDE",
    }.items() <= walking.items()
    source = SCRIPT.read_text(encoding="utf-8")
    export = source[source.index("def _export_one_action(") : source.index("\ndef _finite_glb_nodes")]
    assert "**gltf_export_parameters(action_name)" in export


def test_blender_4_2_merged_animation_name_smoke(tmp_path):
    blender = shutil.which("blender")
    if blender is None:
        pytest.skip("Blender 4.2 executable is unavailable")
    output = tmp_path / "walking.glb"
    code = f'''
import bpy, json, struct
bpy.ops.object.armature_add()
armature = bpy.context.object
armature.name = "SmokeArmature"
action = bpy.data.actions.new(name="Walking")
armature.animation_data_create()
armature.animation_data.action = action
armature.location = (0.0, 0.0, 0.0)
armature.keyframe_insert(data_path="location", frame=1)
armature.location = (0.0, -0.1, 0.0)
armature.keyframe_insert(data_path="location", frame=2)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 2
bpy.ops.object.select_all(action="DESELECT")
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
result = bpy.ops.export_scene.gltf(
    filepath={str(output)!r},
    export_format="GLB",
    use_selection=True,
    export_animations=True,
    export_animation_mode="ACTIVE_ACTIONS",
    export_nla_strips_merged_animation_name="Walking",
    export_force_sampling=True,
    export_frame_range=True,
    export_frame_step=1,
    export_def_bones=False,
    export_rest_position_armature=True,
    export_anim_slide_to_zero=False,
    export_optimize_animation_size=False,
    export_anim_single_armature=True,
    export_reset_pose_bones=True,
    export_negative_frame="SLIDE",
    export_skins=True,
    export_texcoords=True,
    export_normals=True,
    export_materials="EXPORT",
)
assert "FINISHED" in result
payload = open({str(output)!r}, "rb").read()
json_length, json_type = struct.unpack_from("<II", payload, 12)
assert json_type == 0x4E4F534A
document = json.loads(payload[20:20 + json_length].decode("utf-8"))
assert [item.get("name") for item in document.get("animations", [])] == ["Walking"]
print("TASK6_EXPORT_NAME_SMOKE_OK")
'''
    result = subprocess.run(
        [blender, "--background", "--factory-startup", "--python-expr", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TASK6_EXPORT_NAME_SMOKE_OK" in result.stdout


def test_runtime_provenance_hashes_helper_blender_build_and_numpy(tmp_path):
    blender = _write(tmp_path / "blender", b"pinned blender binary")
    helper = _write(tmp_path / "static_audit.py", b"pinned static helper")
    numpy_module = _write(tmp_path / "numpy_init.py", b"pinned numpy module")
    bpy = SimpleNamespace(
        app=SimpleNamespace(
            version=(4, 2, 1),
            version_string="4.2.1 LTS",
            binary_path=str(blender),
            build_hash=b"abc123",
            build_date=b"2024-08-19",
            build_time=b"23:32:23",
            build_branch=b"blender-v4.2-release",
            build_platform=b"Linux",
        )
    )

    provenance = runner.runtime_provenance(
        bpy,
        static_audit_helper_path=helper,
        numpy_module_path=numpy_module,
    )

    assert provenance["static_audit_helper"]["sha256"] == _sha256(helper)
    assert provenance["blender"]["binary"]["sha256"] == _sha256(blender)
    assert provenance["blender"]["build_hash"] == "abc123"
    assert provenance["numpy"]["module"]["sha256"] == _sha256(numpy_module)
    assert provenance["numpy"]["version"] == np.__version__


def test_task6_report_is_published_at_the_master_avengine_sdd_root():
    report = SCRIPT.parents[3] / ".superpowers" / "sdd" / "task-6-report.md"
    assert report.is_file()
    contents = report.read_text(encoding="utf-8")
    assert "Task 6" in contents
    assert "One authorized real Blender retarget was started" in contents
    assert "No `retarget_v1` readiness bundle was published" in contents
    assert "No production success is claimed yet" in contents
    assert "No real Blender retarget" not in contents
    assert "Final focused verification: `94 passed" not in contents
    assert _sha256(SCRIPT) in contents
    assert _sha256(Path(__file__).resolve()) in contents
