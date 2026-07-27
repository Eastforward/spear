"""CLI-contract tests for the post-TokenRig review runner forward modes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generated_animal_forward_contract import (
    MOTION_DONOR_ARTIFACTS,
    build_forward_declaration,
)
from tools.run_target_native_generated_quadruped_review import (
    DEFORMATION_AUDIT_SCHEMA,
    DEFORMATION_THRESHOLDS,
    GAIT_AUDIT_SCHEMA,
    GAIT_RESULT_SCHEMA,
    REPAIR_AUTHORITY_FIELDS,
    RETARGET_SCHEMA,
    RIG_AUDIT_SCHEMA,
    SUPPORT_PLANE_SCHEMA,
    WEIGHT_REPAIR_SCHEMA,
    WEIGHT_REPAIR_PARAMETERS,
    WEIGHT_REPAIR_READY_STATUS,
    file_record,
    main,
    require_deformation_audit,
    require_gait_audit,
    require_rig_audit,
    require_stage_inputs_unchanged,
    require_retarget_manifest,
    require_support_plane,
    require_weight_repair,
    weight_repair_required,
)


DONOR = "quaternius_universal_quadruped_v1"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    rig = tmp_path / "rig.glb"
    rig.write_bytes(b"glTF-rig")
    motion = tmp_path / "motion.glb"
    motion.write_bytes(b"glTF-motion")
    motion_record = file_record(motion)
    monkeypatch.setitem(
        MOTION_DONOR_ARTIFACTS,
        DONOR,
        {
            "sha256": motion_record["sha256"],
            "size_bytes": motion_record["size_bytes"],
        },
    )
    evidence = tmp_path / "head_end_review.json"
    evidence.write_text("{}", encoding="utf-8")
    blender = tmp_path / "blender"
    blender.write_text("#!/bin/sh\n", encoding="utf-8")
    blender.chmod(0o755)
    declaration = build_forward_declaration(
        asset_workspace="cli_test_asset_v1",
        input_glb=rig,
        reviewed_source_front_yaw_deg=49.325,
        head_end_decision_source="human_review",
        head_end_evidence=evidence,
        motion_donor_tag=DONOR,
    )
    declaration_path = tmp_path / "forward_declaration.json"
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    return {
        "rig": rig,
        "motion": motion,
        "evidence": evidence,
        "blender": blender,
        "declaration": declaration_path,
        "output_root": tmp_path / "out",
    }


def base_argv(workspace):
    return [
        "--target-rig-glb", str(workspace["rig"]),
        "--source-motion-glb", str(workspace["motion"]),
        "--output-root", str(workspace["output_root"]),
        "--blender", str(workspace["blender"]),
        "--validate-only",
    ]


def deformation_action(name, *, samples, extension, edge, area, decision):
    return {
        "requested_action": name,
        "resolved_action": f"{name}_Armature",
        "frame_range": [0.0, 40.0],
        "sampled_frames": [
            {
                "source_frame": index,
                "evaluated_frame": index,
                "metrics": {
                    "edge_extension_ratio_of_rest_rotation_invariant_scale": {
                        "maximum": extension
                    },
                    "edge_stretch_ratio": {"maximum": edge},
                    "triangle_area_stretch_ratio": {"maximum": area},
                },
            }
            for index in range(samples)
        ],
        "worst_case": {
            "maximum_edge_extension_ratio_of_rest_rotation_invariant_scale": (
                extension
            ),
            "maximum_edge_stretch_ratio": edge,
            "maximum_triangle_area_stretch_ratio": area,
        },
        "decision": decision,
    }


def test_declaration_mode_validates_and_plans_gait_stage(workspace, capsys):
    argv = base_argv(workspace) + [
        "--forward-declaration", str(workspace["declaration"]),
    ]
    assert main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    labels = [label for label, _command in plan["commands"]]
    assert "gait_direction" in labels
    assert labels.index("retarget") < labels.index("gait_direction")
    retarget = dict(plan["commands"])["retarget"]
    assert "--motion-basis-yaw-deg" in retarget
    assert retarget[retarget.index("--motion-basis-yaw-deg") + 1] == "0"
    assert retarget[retarget.index("--side-chain-mode") + 1] == "matched"


def test_full_plan_defaults_to_mesh_feet_and_gentle_conditional_repair(
    workspace, capsys
):
    argv = base_argv(workspace) + [
        "--forward-declaration", str(workspace["declaration"]),
    ]
    assert main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    commands = dict(plan["commands"])
    labels = [label for label, _command in plan["commands"]]

    assert plan["weight_repair_policy"] == "auto"
    assert labels.index("deformation") < labels.index("weight_repair")
    assert labels.index("weight_repair") < labels.index("deformation_repaired")
    assert labels.index("deformation_repaired") < labels.index("render_walking_side")
    support = commands["support_plane"]
    assert support[support.index("--plane-source") + 1] == "mesh-foot-bottoms"
    repair = commands["weight_repair"]
    assert repair[repair.index("--repair-mode") + 1] == "component-parent-lock"
    assert repair[repair.index("--component-rings") + 1] == "4"
    assert repair[repair.index("--extension-threshold") + 1] == "0.02"
    assert repair[repair.index("--maximum-passes") + 1] == "6"
    assert repair[repair.index("--inner-iterations") + 1] == "4"


def test_declaration_mode_forbids_legacy_flags(workspace):
    argv = base_argv(workspace) + [
        "--forward-declaration", str(workspace["declaration"]),
        "--motion-basis-yaw-deg", "180",
        "--side-chain-mode", "matched",
    ]
    with pytest.raises(ValueError, match="declared exactly once"):
        main(argv)


def test_declaration_mode_rejects_a_different_target_rig(workspace):
    different_rig = workspace["rig"].with_name("different.glb")
    different_rig.write_bytes(b"glTF-different-rig")
    argv = base_argv(workspace) + [
        "--target-rig-glb", str(different_rig),
        "--forward-declaration", str(workspace["declaration"]),
    ]
    with pytest.raises(RuntimeError, match="forward declaration target rig"):
        main(argv)


def test_declaration_mode_rejects_bytes_outside_the_donor_registry(workspace):
    workspace["motion"].write_bytes(b"glTF-unapproved-motion")
    argv = base_argv(workspace) + [
        "--forward-declaration", str(workspace["declaration"]),
    ]
    with pytest.raises(ValueError, match="motion donor artifact does not match"):
        main(argv)


def test_heading_stage_guard_rejects_changed_forward_declaration(workspace):
    target_record = file_record(workspace["rig"])
    heading_record = file_record(workspace["declaration"])
    motion_record = file_record(workspace["motion"])
    workspace["declaration"].write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="heading review evidence"):
        require_stage_inputs_unchanged(
            "heading",
            target_record,
            workspace["rig"],
            heading_record,
            workspace["declaration"],
            motion_record,
            workspace["motion"],
        )


def test_legacy_mode_requires_all_free_parameters(workspace):
    argv = base_argv(workspace) + [
        "--heading-review-evidence", str(workspace["evidence"]),
        "--reviewed-source-front-yaw-deg", "49.325",
    ]
    with pytest.raises(ValueError, match="legacy mode requires"):
        main(argv)


def test_legacy_input_axis_is_not_reused_after_canonical_retarget(workspace, capsys):
    argv = base_argv(workspace) + [
        "--heading-review-evidence", str(workspace["evidence"]),
        "--reviewed-source-front-yaw-deg", "90",
        "--target-front-axis", "negative-y",
        "--motion-basis-yaw-deg", "0",
        "--side-chain-mode", "matched",
    ]
    assert main(argv) == 0
    commands = dict(json.loads(capsys.readouterr().out)["commands"])
    retarget = commands["retarget"]
    repair = commands["weight_repair"]
    assert retarget[retarget.index("--target-front-axis") + 1] == "negative-y"
    assert repair[repair.index("--front-axis") + 1] == "positive-x"


def test_legacy_mode_cannot_execute_or_publish_a_result(workspace):
    argv = [
        argument
        for argument in base_argv(workspace)
        if argument != "--validate-only"
    ] + [
        "--heading-review-evidence", str(workspace["evidence"]),
        "--reviewed-source-front-yaw-deg", "90",
        "--target-front-axis", "positive-x",
        "--motion-basis-yaw-deg", "0",
        "--side-chain-mode", "matched",
    ]
    with pytest.raises(ValueError, match="plan-only"):
        main(argv)


def test_zero_motion_amplitude_is_rejected_during_preflight(workspace):
    argv = base_argv(workspace) + [
        "--forward-declaration", str(workspace["declaration"]),
        "--motion-amplitude", "0",
    ]
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        main(argv)


def test_preview_only_truncates_after_cheap_walking_renders(workspace, capsys):
    argv = base_argv(workspace) + [
        "--forward-declaration", str(workspace["declaration"]),
        "--preview-only",
    ]
    assert main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    labels = [label for label, _command in plan["commands"]]
    assert labels == [
        "heading", "rig_audit", "support_plane", "retarget", "gait_direction",
        "render_walking_side", "encode_walking_side",
        "render_walking_front", "encode_walking_front",
    ]
    assert "deformation" not in labels
    assert plan["conditional_stages"]["weight_repair"] == "not_run_preview_only"


@pytest.mark.parametrize("overall", ["manual_review_required", "rejected"])
def test_auto_policy_repairs_every_nonpassing_deformation(overall):
    assert weight_repair_required("auto", overall) is True


def test_never_policy_fails_closed_for_rejected_deformation():
    with pytest.raises(RuntimeError, match="forbids repair"):
        weight_repair_required("never", "rejected")


def test_gait_gate_derives_pass_from_walking_direction_evidence(tmp_path):
    animated = tmp_path / "animated.glb"
    animated.write_bytes(b"animated")
    manifest = tmp_path / "gait.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": GAIT_AUDIT_SCHEMA,
                "input": file_record(animated),
                "action": "Idle_Armature",
                "samples": 12,
                "result": {
                    "schema": GAIT_RESULT_SCHEMA,
                    "coordinate_frame": (
                        "blender_world_z_up_front_positive_x"
                    ),
                    "stance_drift_yaw_deg": 0.0,
                    "classification": "forward",
                    "walks_head_first": False,
                },
                "status": "pass",
                "formal_dataset_registration_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not pass"):
        require_gait_audit(manifest, animated, "gait audit")


def test_rig_and_final_deformation_gate_rejections_are_not_admitted(tmp_path):
    animated = tmp_path / "animated.glb"
    animated.write_bytes(b"animated")

    rig_path = tmp_path / "rig.json"
    rig_path.write_text(
        json.dumps(
            {
                "schema": RIG_AUDIT_SCHEMA,
                "input": file_record(animated),
                "automatic_checks": {"overall": "rejected"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="rig audit rejected"):
        require_rig_audit(rig_path, animated, front_axis="positive-x")

    deformation_path = tmp_path / "deformation.json"
    deformation_path.write_text(
        json.dumps(
            {
                "schema": DEFORMATION_AUDIT_SCHEMA,
                "input": str(animated.resolve()),
                "input_sha256": file_record(animated)["sha256"],
                "input_size_bytes": animated.stat().st_size,
                "rest_geometry": {
                    "vertices": 10,
                    "edges": 20,
                    "triangles": 12,
                    "decision_scale": "centroid_bounding_sphere_diameter",
                },
                "thresholds": DEFORMATION_THRESHOLDS,
                "actions": [
                    deformation_action(
                        "Walking",
                        samples=4,
                        extension=0.09,
                        edge=5.0,
                        area=9.0,
                        decision="reject_visible_skinning_fan_or_membrane",
                    ),
                    deformation_action(
                        "Idle",
                        samples=4,
                        extension=0.01,
                        edge=1.0,
                        area=1.0,
                        decision="passed_automatic_deformation_measurements",
                    ),
                ],
                "overall": "rejected",
                "formal_dataset_registration_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not pass"):
        require_deformation_audit(
            deformation_path,
            animated,
            "final deformation audit",
            expected_samples=4,
            require_pass=True,
        )


def test_deformation_gate_rejects_empty_actions_and_changed_thresholds(tmp_path):
    animated = tmp_path / "animated.glb"
    animated.write_bytes(b"animated")
    payload = {
        "schema": DEFORMATION_AUDIT_SCHEMA,
        "input": str(animated.resolve()),
        "input_sha256": file_record(animated)["sha256"],
        "input_size_bytes": animated.stat().st_size,
        "rest_geometry": {
            "vertices": 10,
            "edges": 20,
            "triangles": 12,
            "decision_scale": "centroid_bounding_sphere_diameter",
        },
        "thresholds": {**DEFORMATION_THRESHOLDS, "reject_edge_extension_ratio": 999},
        "actions": [],
        "overall": "passed",
        "formal_dataset_registration_authorized": False,
    }
    manifest = tmp_path / "deformation.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="threshold contract"):
        require_deformation_audit(
            manifest,
            animated,
            "deformation audit",
            expected_samples=4,
            require_pass=True,
        )


def test_deformation_gate_recomputes_worst_case_from_every_sample(tmp_path):
    animated = tmp_path / "animated.glb"
    animated.write_bytes(b"animated")
    actions = [
        deformation_action(
            name,
            samples=4,
            extension=0.0,
            edge=0.0,
            area=0.0,
            decision="passed_automatic_deformation_measurements",
        )
        for name in ("Walking", "Idle")
    ]
    for action in actions:
        for frame in action["sampled_frames"]:
            frame["metrics"][
                "edge_extension_ratio_of_rest_rotation_invariant_scale"
            ]["maximum"] = 999.0
            frame["metrics"]["edge_stretch_ratio"]["maximum"] = 999.0
            frame["metrics"]["triangle_area_stretch_ratio"]["maximum"] = 999.0
    manifest = tmp_path / "deformation.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": DEFORMATION_AUDIT_SCHEMA,
                "input": str(animated.resolve()),
                "input_sha256": file_record(animated)["sha256"],
                "input_size_bytes": animated.stat().st_size,
                "rest_geometry": {
                    "vertices": 10,
                    "edges": 20,
                    "triangles": 12,
                    "decision_scale": "centroid_bounding_sphere_diameter",
                },
                "thresholds": DEFORMATION_THRESHOLDS,
                "actions": actions,
                "overall": "passed",
                "formal_dataset_registration_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="contradicts sampled frames"):
        require_deformation_audit(
            manifest,
            animated,
            "deformation audit",
            expected_samples=4,
            require_pass=True,
        )


def test_incomplete_weight_repair_is_not_admitted(tmp_path):
    source = tmp_path / "source.glb"
    source.write_bytes(b"source")
    repaired = tmp_path / "repaired.glb"
    repaired.write_bytes(b"repaired")
    manifest = tmp_path / "repair.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": WEIGHT_REPAIR_SCHEMA,
                "input": file_record(source),
                "output": file_record(repaired),
                "authority_contract": {
                    field: True for field in REPAIR_AUTHORITY_FIELDS
                },
                "status": "research_candidate_repair_incomplete",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        require_weight_repair(manifest, source, repaired)


def test_weight_repair_gate_rejects_a_changed_recipe(tmp_path):
    source = tmp_path / "source.glb"
    source.write_bytes(b"source")
    repaired = tmp_path / "repaired.glb"
    repaired.write_bytes(b"repaired")
    authority = {field: True for field in REPAIR_AUTHORITY_FIELDS}
    authority.update(
        {
            "source_animation_fingerprints": {"Walking": "w", "Idle": "i"},
            "post_repair_animation_fingerprints": {
                "Walking": "w",
                "Idle": "i",
            },
            "source_animation_curve_stats": {"Walking": {}, "Idle": {}},
            "post_repair_animation_curve_stats": {"Walking": {}, "Idle": {}},
            "rest_geometry_topology_fingerprint_before": "topology",
            "rest_geometry_topology_fingerprint_after": "topology",
            "maximum_rest_geometry_delta": 0.0,
        }
    )
    manifest = tmp_path / "repair.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": WEIGHT_REPAIR_SCHEMA,
                "input": file_record(source),
                "output": file_record(repaired),
                "front_axis": "negative-y",
                "parameters": WEIGHT_REPAIR_PARAMETERS,
                "authority_contract": authority,
                "final_measurements": {
                    "maximum_extension_ratio_of_rest_diagonal": 0.019,
                    "remaining_seed_edge_count": 0,
                },
                "status": WEIGHT_REPAIR_READY_STATUS,
                "formal_dataset_registration_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="recipe"):
        require_weight_repair(manifest, source, repaired)


def test_mesh_foot_support_plane_rejects_silent_bone_anchor_fallback(tmp_path):
    source = tmp_path / "heading.glb"
    source.write_bytes(b"heading")
    leveled = tmp_path / "leveled.glb"
    leveled.write_bytes(b"leveled")
    evidence = tmp_path / "rig.json"
    evidence.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "level.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": SUPPORT_PLANE_SCHEMA,
                "input": file_record(source),
                "output": file_record(leveled),
                "review_evidence": file_record(evidence),
                "support_plane": {
                    "front_axis": "positive-x",
                    "plane_source": "mesh-foot-bottoms",
                    "foot_leaves": ["a", "b", "c", "d"],
                    "mesh_foot_contact_band_sizes": [12, 0, 14, 11],
                    "maximum_residual_ratio_of_mesh_diagonal": 0.001,
                    "maximum_reviewed_residual_ratio_of_mesh_diagonal": 0.02,
                    "tilt_deg": 1.0,
                    "maximum_tilt_deg": 30.0,
                    "policy": (
                        "four_semantic_feet_least_squares_plane_rigid_leveling"
                    ),
                },
                "preservation_contract": {
                    "mesh_topology_changed": False,
                    "material_changed": False,
                    "skeleton_hierarchy_changed": False,
                    "skin_weights_changed": False,
                    "animation_present_or_changed": False,
                },
                "scene_before": {},
                "scene_after": {},
                "status": "technical_spike_only_pending_retarget_and_visual_qa",
                "formal_dataset_registration_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="non-empty contact band"):
        require_support_plane(
            manifest,
            source,
            leveled,
            plane_source="mesh-foot-bottoms",
            review_evidence=evidence,
            front_axis="positive-x",
        )


def test_retarget_manifest_binds_both_inputs_and_canonical_export(tmp_path):
    target = tmp_path / "target.glb"
    target.write_bytes(b"target")
    source = tmp_path / "source.glb"
    source.write_bytes(b"source")
    output = tmp_path / "output.glb"
    output.write_bytes(b"output")
    manifest = tmp_path / "retarget.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": RETARGET_SCHEMA,
                "target": {
                    **file_record(target),
                    "mesh_pbr_skeleton_and_weights_authority": True,
                    "reviewed_front_axis": "negative-y",
                },
                "source_motion": {
                    **file_record(source),
                    "geometry_used": False,
                    "weights_used": False,
                    "animation_channels_used": [
                        "translation", "rotation", "scale"
                    ],
                },
                "motion_basis_gate": {
                    "mode": "technical_spike_only_unreviewed",
                    "human_approved": False,
                    "target_animation_generation_authorized": False,
                    "formal_dataset_registration_authorized": False,
                    "selected_motion_basis_yaw_deg": 0,
                    "selected_side_chain_mode": "matched",
                    "selected_rotation_solver": "world-left-delta-v2",
                },
                "runtime_pose_transfer": {
                    "mode": "world-rotation-retarget-v2",
                    "world_rotation_resampling_used": True,
                },
                "rotation_transfer": {
                    "method": "world-left-delta-v2",
                    "approved_preview_method": "world-left-delta-v2",
                    "used_for_runtime": True,
                    "motion_amplitude": 1.0,
                    "motion_basis_yaw_deg": 0,
                    "side_chain_mode": "matched",
                },
                "semantic_inference": {
                    "bone_name_independent_target": True,
                    "complete_target_bone_coverage": True,
                },
                "export": {
                    **file_record(output),
                    "canonical_front_axis": "positive-x",
                    "action_names": ["Walking", "Idle"],
                },
                "status": (
                    "technical_spike_only_pending_deformation_and_visual_qa"
                ),
                "formal_dataset_registration_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    require_retarget_manifest(
        manifest,
        target,
        source,
        output,
        target_front_axis="negative-y",
        motion_amplitude=1.0,
        motion_basis_yaw_deg=0,
        side_chain_mode="matched",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["export"]["canonical_front_axis"] = "negative-y"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"canonical \+X"):
        require_retarget_manifest(
            manifest,
            target,
            source,
            output,
            target_front_axis="negative-y",
            motion_amplitude=1.0,
            motion_basis_yaw_deg=0,
            side_chain_mode="matched",
        )
