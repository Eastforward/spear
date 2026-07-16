import json
from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as contracts
from tools import run_controlled_animal_lod_binding as runner
from tools.spike_rlr import runtime_proxy_mesh


def _job(tmp_path, species="dog"):
    spec = runner.RIG_SPECS[species]
    return {
        "raw_path": tmp_path / "raw.glb",
        "rig": {**spec, "path": Path(spec["path"]), "species": species},
        "direction_decision": {
            "manual_cardinal_yaw_about_gltf_positive_y_deg": 90,
        },
    }


def test_build_commands_pins_approved_lod_and_binding_contract(tmp_path):
    lod, bind = runner.build_commands(_job(tmp_path), tmp_path / "job", target_faces=100_000)

    assert lod[-3:] == ["--target-faces", "100000", "--double-sided"]
    assert "--double-sided" in lod
    assert bind[bind.index("--rig-glb") + 1].endswith("Dog.glb")
    assert "--flip-x" not in bind
    assert bind[bind.index("--target-rotate-z-deg") + 1] == "90"
    assert bind[bind.index("--align-mode") + 1] == "uniform"
    assert bind[bind.index("--weight-mode") + 1] == "region"
    assert bind[bind.index("--segmentation-mode") + 1] == "proximity"
    assert bind[bind.index("--semantic-forward-axis") + 1] == "positive-x"
    assert bind[bind.index("--delete-limb-bridge-faces") + 1] == "no"
    assert bind[bind.index("--export-action-policy") + 1] == "walk-idle"


def test_build_commands_can_keep_prelock_binding_as_evidence(tmp_path):
    prelock = tmp_path / "prelock.glb"

    _lod, bind = runner.build_commands(
        _job(tmp_path),
        tmp_path / "job",
        target_faces=100_000,
        bind_output=prelock,
    )

    assert bind[bind.index("--output") + 1] == str(prelock)


def test_locked_paw_profile_pins_approved_motion_carrier_and_commands(tmp_path):
    spec = runner._locked_paw_motion_spec("quadruped_dog_locked_paws_v2")
    target = tmp_path / "prelock.glb"
    output = tmp_path / "locked.glb"
    manifest = tmp_path / "transplant.json"
    audit = tmp_path / "lateral.json"

    transplant, lateral = runner.build_locked_paw_commands(
        target, output, manifest, audit, spec
    )

    assert spec["sha256"] == (
        "083cafc7d99ae1e9e752b512adedef71bf3a124f1d648493874fddc8abc62117"
    )
    assert transplant[transplant.index("--source-glb") + 1] == str(spec["path"])
    assert [
        transplant[index + 1]
        for index, value in enumerate(transplant)
        if value == "--action"
    ] == ["Idle", "Walking"]
    assert lateral[lateral.index("--front-axis") + 1] == "positive-x"
    assert lateral[lateral.index("--samples") + 1] == "41"


def test_locked_paw_audit_requires_all_four_limbs_below_threshold(tmp_path):
    output = tmp_path / "locked.glb"
    output.write_bytes(b"locked motion fixture")
    spec = runner._locked_paw_motion_spec("quadruped_dog_locked_paws_v2")
    limb = {
        "paw_relative_to_hip_lateral_excursion_ratio_of_mesh_diagonal": 0.001,
        "paw_yaw_excursion_degrees": 0.01,
    }
    payload = {
        "schema": "avengine_quadruped_lateral_gait_audit_v1",
        "input": {"sha256": runner._sha256_file(output)},
        "coordinate_contract": {"front_axis": "positive-x"},
        "action": "Walking",
        "summary": {
            name: dict(limb)
            for name in (
                "front_side_negative",
                "front_side_positive",
                "hind_side_negative",
                "hind_side_positive",
            )
        },
    }

    result = runner.validate_locked_paw_audit(payload, output, spec)

    assert result["overall"] == "passed"
    payload["summary"]["hind_side_positive"][
        "paw_relative_to_hip_lateral_excursion_ratio_of_mesh_diagonal"
    ] = 0.01
    with pytest.raises(contracts.ContractError, match="exceeds pinned"):
        runner.validate_locked_paw_audit(payload, output, spec)

    payload["summary"]["hind_side_positive"] = dict(limb)
    payload["action"] = "Walking_Armature"
    assert runner.validate_locked_paw_audit(payload, output, spec)["overall"] == (
        "passed"
    )


def test_rig_spec_rejects_profile_and_species_mismatch():
    source = {
        "asset_id": "animal_x",
        "taxonomy": {"species": "cat"},
        "rig": {
            "profile_id": "quadruped_dog_v1",
            "skeleton_family": "quaternius_cat",
            "front_axis": "positive_x",
            "actions": ["Walking", "Idle"],
        },
    }

    with pytest.raises(contracts.ContractError, match="rig contract"):
        runner._rig_spec(source)


def test_rewrite_runtime_metadata_records_published_path(tmp_path):
    runtime = tmp_path / "runtime.glb"
    runtime.write_bytes(b"glb fixture")
    source = tmp_path / "source.glb"
    source.write_bytes(b"source fixture")
    metadata = tmp_path / "runtime.json"
    metadata.write_text(
        json.dumps(
            {
                "algorithm": "blender_decimate_v1",
                "source_mesh_sha256": runner._sha256_file(source),
                "runtime_mesh": str(runtime),
                "runtime_mesh_sha256": runner._sha256_file(runtime),
            }
        )
    )
    published = tmp_path / "published/runtime.glb"

    runner._rewrite_runtime_metadata(
        metadata, published, source_sha256=runner._sha256_file(source)
    )

    assert json.loads(metadata.read_text())["runtime_mesh"] == str(published.resolve())


def test_rewrite_runtime_metadata_accepts_welded_topology_without_new_cracks(tmp_path):
    runtime = tmp_path / "runtime.glb"
    runtime.write_bytes(b"glb fixture")
    source = tmp_path / "source.glb"
    source.write_bytes(b"source fixture")
    metadata = tmp_path / "runtime.json"
    metadata.write_text(
        json.dumps(
            {
                "algorithm": runtime_proxy_mesh.RUNTIME_PROXY_ALGORITHM,
                "source_mesh_sha256": runner._sha256_file(source),
                "runtime_mesh": str(runtime),
                "runtime_mesh_sha256": runner._sha256_file(runtime),
                "topology": {
                    "position_weld": {"vertices_welded": 12},
                    "source_after_position_weld": {"boundary_edges": 2},
                    "runtime_after_decimate": {"boundary_edges": 0},
                    "boundary_cracks_introduced": -2,
                },
            }
        )
    )

    runner._rewrite_runtime_metadata(
        metadata,
        tmp_path / "published/runtime.glb",
        source_sha256=runner._sha256_file(source),
    )

    assert json.loads(metadata.read_text())["algorithm"] == (
        "blender_welded_decimate_v2"
    )


def test_rewrite_runtime_metadata_rejects_welded_topology_with_new_cracks(tmp_path):
    runtime = tmp_path / "runtime.glb"
    runtime.write_bytes(b"glb fixture")
    source = tmp_path / "source.glb"
    source.write_bytes(b"source fixture")
    metadata = tmp_path / "runtime.json"
    metadata.write_text(
        json.dumps(
            {
                "algorithm": runtime_proxy_mesh.RUNTIME_PROXY_ALGORITHM,
                "source_mesh_sha256": runner._sha256_file(source),
                "runtime_mesh": str(runtime),
                "runtime_mesh_sha256": runner._sha256_file(runtime),
                "topology": {
                    "position_weld": {"vertices_welded": 12},
                    "source_after_position_weld": {"boundary_edges": 0},
                    "runtime_after_decimate": {"boundary_edges": 4},
                    "boundary_cracks_introduced": 4,
                },
            }
        )
    )

    with pytest.raises(contracts.ContractError, match="boundary cracks"):
        runner._rewrite_runtime_metadata(
            metadata,
            tmp_path / "published/runtime.glb",
            source_sha256=runner._sha256_file(source),
        )


def test_prebind_geometry_gate_rejects_large_source_defects():
    with pytest.raises(contracts.ContractError, match="rejected before LOD/binding"):
        runner._enforce_prebind_geometry_audit(
            {
                "decision": {
                    "status": "reject_before_lod_and_binding",
                    "rejection_reasons": [
                        "torso_centerline_bend_exceeds_10_degrees",
                    ],
                }
            }
        )


@pytest.mark.parametrize(
    "status",
    [
        "passed_automatic_geometry_measurements",
        "manual_source_geometry_review_required",
    ],
)
def test_prebind_geometry_gate_allows_pass_or_manual_review(status):
    runner._enforce_prebind_geometry_audit({"decision": {"status": status}})


@pytest.mark.parametrize("workers", [0, 17])
def test_run_batch_rejects_unsafe_worker_counts(tmp_path, workers):
    with pytest.raises(contracts.ContractError, match="workers"):
        runner.run_batch([], tmp_path / "output", workers=workers)


def test_load_jobs_requires_a_registry():
    with pytest.raises(contracts.ContractError, match="registry"):
        runner.load_jobs([])


def _write_approved_direction_decision(
    tmp_path, *, asset_id="animal_x", yaw=90, automatic=False
):
    reviewed_lod = tmp_path / "reviewed_lod.glb"
    reviewed_lod.write_bytes(b"reviewed deterministic lod")
    decision = {
        "schema": "controlled_animal_pose_direction_manual_decision_v2",
        "asset_id": asset_id,
        "status": "source_pose_and_cardinal_orientation_approved",
        "automatic_orientation_inference_used": automatic,
        "manual_cardinal_yaw_about_gltf_positive_y_deg": yaw,
        "manual_rotation_matrix_3x3": [
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ],
        "determinant": 1,
        "source_prebind_lod": {
            "absolute_path": str(reviewed_lod),
            "sha256": runner._sha256_file(reviewed_lod),
            "size_bytes": reviewed_lod.stat().st_size,
        },
        "manual_pose_checks_are_advisory": True,
    }
    decision["decision_sha256"] = runner._hash_without(
        decision, "decision_sha256"
    )
    path = tmp_path / f"{asset_id}.json"
    path.write_text(json.dumps(decision))
    return path, decision


def test_direction_decision_gate_authenticates_human_cardinal_choice(tmp_path):
    path, expected = _write_approved_direction_decision(tmp_path)

    decision = runner.load_direction_decision(path, expected_asset_id="animal_x")

    assert decision == expected
    assert decision["manual_cardinal_yaw_about_gltf_positive_y_deg"] == 90


@pytest.mark.parametrize(
    ("yaw", "automatic", "message"),
    [
        (15, False, "cardinal"),
        (90, True, "automatic"),
    ],
)
def test_direction_decision_gate_rejects_guess_or_fine_yaw(
    tmp_path, yaw, automatic, message
):
    path, _ = _write_approved_direction_decision(
        tmp_path, yaw=yaw, automatic=automatic
    )

    with pytest.raises(contracts.ContractError, match=message):
        runner.load_direction_decision(path, expected_asset_id="animal_x")


def _write_approved_two_stage_direction_decision(
    tmp_path, *, asset_id="animal_x", axis_yaw=25, cardinal_yaw=180
):
    reviewed_lod = tmp_path / "reviewed_lod_v3.glb"
    reviewed_lod.write_bytes(b"reviewed deterministic lod v3")
    total_yaw = runner._normalize_yaw(axis_yaw + cardinal_yaw)
    decision = {
        "schema": runner.DIRECTION_DECISION_SCHEMA_V3,
        "asset_id": asset_id,
        "status": runner.DIRECTION_APPROVED_STATUS_V3,
        "automatic_orientation_inference_used": False,
        "manual_axis_alignment_yaw_about_gltf_positive_y_deg": axis_yaw,
        "manual_cardinal_head_tail_yaw_about_gltf_positive_y_deg": cardinal_yaw,
        "manual_total_yaw_about_gltf_positive_y_deg": total_yaw,
        "axis_alignment_authority": "human_visual_torso_spine_axis",
        "head_tail_authority": "human_visual_head_tail_direction",
        "manual_rotation_matrix_3x3": runner._yaw_matrix_y_up(total_yaw),
        "determinant": 1,
        "source_prebind_lod": {
            "absolute_path": str(reviewed_lod),
            "sha256": runner._sha256_file(reviewed_lod),
            "size_bytes": reviewed_lod.stat().st_size,
        },
        "manual_pose_checks_are_advisory": True,
    }
    decision["decision_sha256"] = runner._hash_without(
        decision, "decision_sha256"
    )
    path = tmp_path / f"{asset_id}.json"
    path.write_text(json.dumps(decision))
    return path, decision


def test_direction_decision_gate_authenticates_two_stage_human_yaw(tmp_path):
    path, expected = _write_approved_two_stage_direction_decision(tmp_path)

    decision = runner.load_direction_decision(path, expected_asset_id="animal_x")

    assert decision == expected
    assert runner._decision_binding_yaw(decision) == -155


def test_two_stage_direction_build_command_uses_composed_yaw(tmp_path):
    _, decision = _write_approved_two_stage_direction_decision(tmp_path)
    job = _job(tmp_path)
    job["direction_decision"] = decision

    _, bind = runner.build_commands(job, tmp_path / "job", target_faces=100_000)

    assert bind[bind.index("--target-rotate-z-deg") + 1] == "-155"


def test_two_stage_direction_rejects_tampered_composition(tmp_path):
    path, decision = _write_approved_two_stage_direction_decision(tmp_path)
    decision["manual_total_yaw_about_gltf_positive_y_deg"] = -150
    decision["manual_rotation_matrix_3x3"] = runner._yaw_matrix_y_up(-150)
    decision["decision_sha256"] = runner._hash_without(
        decision, "decision_sha256"
    )
    path.write_text(json.dumps(decision))

    with pytest.raises(contracts.ContractError, match="composition"):
        runner.load_direction_decision(path, expected_asset_id="animal_x")


def test_reviewed_lod_gate_rejects_regenerated_geometry_change(tmp_path):
    _, decision = _write_approved_direction_decision(tmp_path)
    regenerated = tmp_path / "regenerated.glb"
    regenerated.write_bytes(b"different lod")

    with pytest.raises(contracts.ContractError, match="reviewed LOD"):
        runner.assert_lod_matches_direction_review(regenerated, decision)
