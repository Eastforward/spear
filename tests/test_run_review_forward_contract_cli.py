"""CLI-contract tests for the post-TokenRig review runner forward modes."""

from __future__ import annotations

from copy import deepcopy
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generated_animal_forward_contract import (
    CAT_MOTION_DONOR_ID,
    DOG_MOTION_DONOR_ID,
    MOTION_DONOR_ARTIFACTS,
    build_forward_declaration,
    expected_motion_donor_contract,
)
from tools.generated_animal_support_plane_contract import (
    CROSSCHECK_METHOD,
    EVIDENCE_SCHEMA as SUPPORT_PLANE_EVIDENCE_SCHEMA,
    MAXIMUM_PRIMARY_POST_LEVEL_TILT_DEG,
    MAXIMUM_POST_LEVEL_BBOX_DIAGONAL_RATIO_DELTA,
    MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO,
    MAXIMUM_RIGID_VERTEX_DELTA_RATIO,
    MAXIMUM_SERIALIZATION_VERTEX_EXPANSION_RATIO,
    MAXIMUM_SKIN_WEIGHT_DELTA,
    OUTPUT_READBACK_SCHEMA,
    POLICY as SUPPORT_PLANE_POLICY,
    PRIMARY_METHOD,
)
import tools.run_target_native_generated_quadruped_review as review_runner
from tools.run_target_native_generated_quadruped_review import (
    DEFORMATION_AUDIT_SCHEMA,
    DEFORMATION_THRESHOLDS,
    ENCODE_MANIFEST_SCHEMA,
    GAIT_AUDIT_SCHEMA,
    GAIT_RESULT_SCHEMA,
    REPAIR_AUTHORITY_FIELDS,
    RENDER_MANIFEST_SCHEMA,
    REVIEW_RENDER_CONFIG,
    RETARGET_SCHEMA,
    RIG_AUDIT_SCHEMA,
    SCHEMA as REVIEW_SCHEMA,
    SUPPORT_PLANE_SCHEMA,
    WEIGHT_REPAIR_FALLBACK_PARAMETERS,
    WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS,
    WEIGHT_REPAIR_INCOMPLETE_STATUS,
    WEIGHT_REPAIR_SCHEMA,
    WEIGHT_REPAIR_PARAMETERS,
    WEIGHT_REPAIR_READY_STATUS,
    WEIGHT_REPAIR_STAGE_CONTRACTS,
    WeightRepairIncomplete,
    file_record,
    expected_review_ffmpeg_config,
    load_json_artifact,
    main,
    require_deformation_audit,
    require_gait_audit,
    require_encode_manifest,
    require_render_manifest,
    require_rig_audit,
    require_stage_inputs_unchanged,
    require_retarget_manifest,
    require_support_plane,
    require_weight_repair,
    require_weight_repair_branch_consistency,
    require_weight_repair_pipeline_order,
    weight_repair_attempt_record,
    weight_repair_final_artifact,
    weight_repair_gate_attempts,
    weight_repair_output_descriptors,
    weight_repair_required,
    render_frame_set,
    verify_video,
)


DONOR = DOG_MOTION_DONOR_ID


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ('{"schema":"a","schema":"b"}', "duplicate JSON object key"),
        ('{"value":Infinity}', "non-finite JSON number"),
    ],
)
def test_review_artifact_loader_rejects_ambiguous_json(
    tmp_path, payload, expected_error
):
    path = tmp_path / "manifest.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(RuntimeError, match="not valid JSON") as caught:
        load_json_artifact(path, "test manifest")
    assert expected_error in str(caught.value.__cause__)


@pytest.mark.parametrize(
    "payload",
    [
        '{"streams":[],"streams":[]}',
        '{"streams":[],"format":{"duration":NaN}}',
    ],
)
def test_ffprobe_readback_rejects_ambiguous_json(tmp_path, monkeypatch, payload):
    video = tmp_path / "review.mp4"
    video.write_bytes(b"video")

    class Result:
        stdout = payload

    monkeypatch.setattr(review_runner.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError, match="ffprobe returned invalid JSON"):
        verify_video(video, 8)


def dual_support_plane_evidence():
    points = [
        [-1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    plane = {
        "z_equals_ax_plus_by_plus_c": [0.0, 0.0, 0.0],
        "residual_z": [0.0, 0.0, 0.0, 0.0],
        "maximum_residual": 0.0,
        "maximum_residual_ratio_of_mesh_diagonal": 0.0,
        "normal": [0.0, 0.0, 1.0],
        "tilt_deg": 0.0,
        "rank": 3,
        "singular_values": [2.0, 2.0, 2.0],
    }
    return {
        "schema": SUPPORT_PLANE_EVIDENCE_SCHEMA,
        "mesh_diagonal": 4.0,
        "thresholds": {
            "capture_radius_ratio_of_mesh_diagonal": 0.05,
            "capture_radius": 0.2,
            "contact_band_absolute_floor": 0.004,
            "contact_band_ratio_of_mesh_diagonal": 0.003,
            "contact_band_thickness": 0.012,
            "minimum_capture_vertices": 10,
            "minimum_contact_band_vertices": 10,
            "minimum_weight_owner_score": 0.1,
            "maximum_floor_delta_between_authorities": 0.012,
            "maximum_contact_centroid_xy_delta_between_authorities": 0.2,
            "maximum_plane_residual_ratio_of_mesh_diagonal": 0.02,
            "maximum_tilt_deg": 30.0,
        },
        "primary": {
            "method": PRIMARY_METHOD,
            "exclusive_vertex_assignment": True,
            "capture_counts": [12, 12, 12, 12],
            "contact_band_sizes": [12, 12, 12, 12],
            "foot_points": points,
            "plane": dict(plane),
            "maximum_captured_segment_distances": [0.1, 0.1, 0.1, 0.1],
        },
        "crosscheck": {
            "method": CROSSCHECK_METHOD,
            "exclusive_vertex_assignment": True,
            "capture_counts": [12, 12, 12, 12],
            "contact_band_sizes": [12, 12, 12, 12],
            "foot_points": points,
            "plane": dict(plane),
            "minimum_captured_owner_scores": [0.5, 0.5, 0.5, 0.5],
        },
        "agreement": {
            "per_foot_floor_z_absolute_delta": [0.0, 0.0, 0.0, 0.0],
            "maximum_floor_z_absolute_delta": 0.0,
            "maximum_allowed_floor_z_absolute_delta": 0.012,
            "per_foot_contact_centroid_xy_distance": [0.0, 0.0, 0.0, 0.0],
            "maximum_contact_centroid_xy_distance": 0.0,
            "maximum_allowed_contact_centroid_xy_distance": 0.2,
            "passed": True,
        },
        "fallback_used": False,
    }


def support_rig_records():
    def bone(name, parent, x, y, z, children=()):
        return {
            "name": name,
            "parent": parent,
            "children": list(children),
            "head_world": [x, y, z],
            "tail_world": [x, y, z - 0.02],
        }

    # +X-facing version of the generic synthetic quadruped semantics fixture.
    return [
        bone("root", None, -0.2, 0.0, 0.5, ("spine", "hl", "hr", "tail")),
        bone("spine", "root", 0.0, 0.0, 0.5, ("neck", "fl", "fr")),
        bone("neck", "spine", 0.2, 0.0, 0.6, ("head",)),
        bone("head", "neck", 0.4, 0.0, 0.65),
        bone("tail", "root", -0.45, 0.0, 0.55, ("tail_tip",)),
        bone("tail_tip", "tail", -0.65, 0.0, 0.6),
        bone("fl", "spine", 0.15, -0.2, 0.35, ("fl_foot",)),
        bone("fl_foot", "fl", 0.2, -0.2, 0.02),
        bone("fr", "spine", 0.15, 0.2, 0.35, ("fr_foot",)),
        bone("fr_foot", "fr", 0.2, 0.2, 0.02),
        bone("hl", "root", -0.2, -0.2, 0.35, ("hl_foot",)),
        bone("hl_foot", "hl", -0.25, -0.2, 0.02),
        bone("hr", "root", -0.2, 0.2, 0.35, ("hr_foot",)),
        bone("hr_foot", "hr", -0.25, 0.2, 0.02),
    ]


def support_semantic_rig():
    return {
        "foot_leaves": ["fl_foot", "fr_foot", "hl_foot", "hr_foot"],
        "limb_chains": {
            "front_side_negative": ["fl", "fl_foot"],
            "front_side_positive": ["fr", "fr_foot"],
            "hind_side_negative": ["hl", "hl_foot"],
            "hind_side_positive": ["hr", "hr_foot"],
        },
        "distal_owner_bones": [
            ["fl", "fl_foot"],
            ["fr", "fr_foot"],
            ["hl", "hl_foot"],
            ["hr", "hr_foot"],
        ],
    }


def support_output_readback(
    *,
    source,
    leveled,
    manifest,
    rig_audit,
    dual,
    scene,
):
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    semantic = support_semantic_rig()
    bone_names = sorted(record["name"] for record in support_rig_records())
    return {
        "schema": OUTPUT_READBACK_SCHEMA,
        "status": "passed_independent_glb_reimport",
        "formal_dataset_registration_authorized": False,
        "pre_level_input": file_record(source),
        "leveled_output": file_record(leveled),
        "support_manifest": file_record(manifest),
        "rig_audit": file_record(rig_audit),
        "front_axis": "positive-x",
        "thresholds": {
            "maximum_rigid_vertex_delta_ratio_of_mesh_diagonal": (
                MAXIMUM_RIGID_VERTEX_DELTA_RATIO
            ),
            "maximum_rigid_vertex_delta": (
                4.0 * MAXIMUM_RIGID_VERTEX_DELTA_RATIO
            ),
            "maximum_rigid_bone_endpoint_delta_ratio_of_mesh_diagonal": (
                MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO
            ),
            "maximum_rigid_bone_endpoint_delta": (
                4.0 * MAXIMUM_RIGID_BONE_ENDPOINT_DELTA_RATIO
            ),
            "maximum_skin_weight_delta": MAXIMUM_SKIN_WEIGHT_DELTA,
            "maximum_primary_post_level_tilt_deg": (
                MAXIMUM_PRIMARY_POST_LEVEL_TILT_DEG
            ),
            "maximum_serialization_vertex_expansion_ratio": (
                MAXIMUM_SERIALIZATION_VERTEX_EXPANSION_RATIO
            ),
            "maximum_foot_readback_delta": 2.0e-6,
            "maximum_post_level_semantic_reacquisition_delta": 0.2,
            "maximum_post_level_floor_reacquisition_delta": 0.012,
            "maximum_post_level_bbox_diagonal_ratio_delta": (
                MAXIMUM_POST_LEVEL_BBOX_DIAGONAL_RATIO_DELTA
            ),
        },
        "pre_level_scene": dict(scene),
        "post_level_scene": dict(scene),
        "pre_level_semantic_rig": deepcopy(semantic),
        "post_level_semantic_rig": deepcopy(semantic),
        "pre_level_dual_authority": deepcopy(dual),
        "post_level_dual_authority": deepcopy(dual),
        "object_transform_readback": {
            "pre_mesh_world_matrix": identity,
            "post_mesh_world_matrix": identity,
            "pre_armature_world_matrix": identity,
            "post_armature_world_matrix": identity,
            "pre_root_objects": [
                {"name": "root", "type": "ARMATURE", "world_matrix": identity}
            ],
            "post_root_objects": [
                {"name": "root", "type": "ARMATURE", "world_matrix": identity}
            ],
        },
        "mesh_and_weight_binding": {
            "vertex_count": 48,
            "serialized_vertex_count": 48,
            "serialization_vertex_expansion_ratio": 0.0,
            "polygon_count": 80,
            "topology_sha256_before": "0" * 64,
            "topology_sha256_after": "0" * 64,
            "bone_group_names": bone_names,
            "maximum_skin_weight_delta": 0.0,
            "weights_changed_above_tolerance": 0,
            "materials": ["material"],
            "uv_layers": ["UVMap"],
        },
        "comparison": {
            "maximum_world_vertex_delta_from_declared_transform": 0.0,
            "maximum_bone_endpoint_delta_from_declared_transform": 0.0,
            "pre_level_primary_foot_delta": 0.0,
            "pre_level_crosscheck_foot_delta": 0.0,
            "post_level_primary_foot_delta": 0.0,
            "post_level_crosscheck_foot_delta": 0.0,
            "actual_minimum_primary_foot_z": 0.0,
            "primary_post_level_tilt_deg": 0.0,
            "post_level_bbox_diagonal_ratio_delta": 0.0,
            "passed": True,
        },
    }


def write_motion_glb(path: Path, actions=("Idle", "Walking")) -> Path:
    document = {
        "asset": {"version": "2.0"},
        "animations": [{"name": name} for name in actions],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    total_size = 12 + 8 + len(encoded)
    payload = (
        struct.pack("<4sII", b"glTF", 2, total_size)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
    )
    path.write_bytes(payload)
    return path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    rig = tmp_path / "rig.glb"
    rig.write_bytes(b"glTF-rig")
    motion = tmp_path / "motion.glb"
    write_motion_glb(motion)
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
    closure_manifest = tmp_path / "tokenrig_closure_manifest.json"
    closure_manifest.write_text('{"fixture":true}\n', encoding="utf-8")
    closure_manifest_record = file_record(closure_manifest)

    def fixture_tokenrig_closure(
        manifest_path,
        *,
        expected_manifest_sha256,
        expected_target_rig_glb,
    ):
        assert Path(manifest_path) == closure_manifest
        assert (
            expected_manifest_sha256
            == closure_manifest_record["sha256"]
        )
        target_record = file_record(Path(expected_target_rig_glb))
        return {
            "schema": (
                "avengine_generated_animal_tokenrig_closure_descriptor_v1"
            ),
            "status": "passed",
            "asset_id": "cli_test_asset_v1",
            "evidence_mode": "complete_load_audit_v1",
            "closure_manifest": dict(closure_manifest_record),
            "lineage": {
                "raw_pixal_glb": {
                    "path": str(tmp_path / "raw.glb"),
                    "sha256": "1" * 64,
                    "size_bytes": 1,
                },
                "tokenrig_input": {
                    "path": str(tmp_path / "input.glb"),
                    "sha256": "2" * 64,
                    "size_bytes": 1,
                },
                "tokenrig_output": target_record,
                "upstream_kind": "bounded_geometry_closure",
            },
            "geometry_readback": {
                "path": str(tmp_path / "readback.json"),
                "sha256": "3" * 64,
                "size_bytes": 1,
                "manifest_sha256": "4" * 64,
            },
            "execution": {
                "seed": 42,
                "exact_argv_sha256": "5" * 64,
                "model_checkpoint_sha256": "6" * 64,
                "model_snapshot_revision": "7" * 40,
                "runtime_patch_sha256": "8" * 64,
                "skintokens_revision": "9" * 40,
            },
            "formal_dataset_registration_authorized": False,
        }

    monkeypatch.setattr(
        review_runner,
        "validate_tokenrig_closure_manifest",
        fixture_tokenrig_closure,
    )
    return {
        "rig": rig,
        "motion": motion,
        "evidence": evidence,
        "blender": blender,
        "declaration": declaration_path,
        "tokenrig_closure_manifest": closure_manifest,
        "tokenrig_closure_sha256": closure_manifest_record["sha256"],
        "output_root": tmp_path / "out",
    }


def base_argv(workspace):
    return [
        "--target-rig-glb", str(workspace["rig"]),
        "--tokenrig-closure-manifest",
        str(workspace["tokenrig_closure_manifest"]),
        "--expected-tokenrig-closure-sha256",
        workspace["tokenrig_closure_sha256"],
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
    assert plan["forward_contract"]["target_species"] == "dog"
    assert plan["forward_contract"]["motion_donor_tag"] == DOG_MOTION_DONOR_ID
    assert plan["forward_contract"]["target_rig_lineage"]["schema"] == (
        "avengine_generated_animal_tokenrig_closure_descriptor_v1"
    )
    assert plan["forward_contract"]["target_rig_lineage"]["lineage"][
        "tokenrig_output"
    ] == file_record(workspace["rig"])


def test_cat_species_selects_cat_actions_without_changing_target_mesh_provenance(
    workspace, monkeypatch, capsys
):
    cat_motion = write_motion_glb(workspace["motion"].with_name("Cat.glb"))
    cat_motion_record = file_record(cat_motion)
    monkeypatch.setitem(
        MOTION_DONOR_ARTIFACTS,
        CAT_MOTION_DONOR_ID,
        {
            "sha256": cat_motion_record["sha256"],
            "size_bytes": cat_motion_record["size_bytes"],
        },
    )
    declaration = build_forward_declaration(
        asset_workspace="cli_test_cat_v1",
        input_glb=workspace["rig"],
        reviewed_source_front_yaw_deg=49.325,
        head_end_decision_source="human_review",
        head_end_evidence=workspace["evidence"],
        target_species="cat",
    )
    declaration_path = workspace["declaration"].with_name(
        "cat_forward_declaration.json"
    )
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")

    argv = base_argv(workspace) + [
        "--source-motion-glb", str(cat_motion),
        "--forward-declaration", str(declaration_path),
    ]
    assert main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    forward = plan["forward_contract"]
    assert forward["target_species"] == "cat"
    assert forward["motion_donor_tag"] == CAT_MOTION_DONOR_ID
    assert forward["motion_donor"]["animation_names"] == ["Idle", "Walking"]
    assert forward["motion_donor_contract"]["geometry_used"] is False
    assert forward["motion_donor_contract"]["weights_used"] is False
    assert forward["motion_donor_contract"]["required_actions"] == [
        "Walking", "Idle"
    ]
    assert forward["target_asset_authority"] == {
        "input_glb": file_record(workspace["rig"]),
        "mesh_geometry_source": "target_rig_glb",
        "motion_donor_geometry_used": False,
    }
    retarget = dict(plan["commands"])["retarget"]
    assert retarget[retarget.index("--source-rig-glb") + 1] == str(
        cat_motion.resolve()
    )


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
    assert plan["schema"] == REVIEW_SCHEMA
    assert REVIEW_SCHEMA == "avengine_target_native_generated_quadruped_review_run_v4"
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
    fallback_label = "weight_repair_low_slice_edge_average"
    fallback_gait_label = "gait_direction_repaired_low_slice_edge_average"
    fallback_deformation_label = (
        "deformation_repaired_low_slice_edge_average"
    )
    residual_label = "weight_repair_low_slice_edge_average_residual"
    assert labels.index("weight_repair") < labels.index(fallback_label)
    assert labels.index(fallback_label) < labels.index(fallback_gait_label)
    assert labels.index(fallback_gait_label) < labels.index(
        fallback_deformation_label
    )
    assert labels.index(fallback_deformation_label) < labels.index(
        residual_label
    )
    assert labels.index(residual_label) < labels.index(
        "deformation_repaired_low_slice_edge_average_residual"
    )
    assert labels.index(
        "deformation_repaired_low_slice_edge_average_residual"
    ) < labels.index("render_walking_side")
    fallback = commands[fallback_label]
    residual = commands[residual_label]
    assert fallback[fallback.index("--input") + 1].endswith(
        "/04_motion/target_animated.glb"
    )
    assert fallback[fallback.index("--repair-mode") + 1] == "edge-average"
    assert (
        fallback[fallback.index("--cross-limb-authority") + 1]
        == "low-slice-components"
    )
    assert fallback[fallback.index("--limb-slice-height-fraction") + 1] == "0.2"
    assert fallback[fallback.index("--extension-threshold") + 1] == "0.02"
    assert fallback[fallback.index("--maximum-passes") + 1] == "12"
    assert fallback[fallback.index("--inner-iterations") + 1] == "8"
    fallback_output = fallback[fallback.index("--output") + 1]
    fallback_gait = commands[fallback_gait_label]
    fallback_deformation = commands[fallback_deformation_label]
    assert fallback_output != repair[repair.index("--output") + 1]
    assert fallback_gait[fallback_gait.index("--input") + 1] == fallback_output
    assert (
        fallback_deformation[
            fallback_deformation.index("--input") + 1
        ]
        == fallback_output
    )
    assert residual[residual.index("--input") + 1] == fallback_output
    assert residual[residual.index("--output") + 1] != fallback_output
    assert "--skip-cross-limb-preclean" not in fallback
    assert "--skip-cross-limb-preclean" in residual
    render = commands["render_walking_side"]
    encode = commands["encode_walking_side"]
    assert render[render.index("--manifest") + 1].endswith(
        "/05_review/walking_side_render_manifest.json"
    )
    assert render[render.index("--fps") + 1] == "8"
    assert encode[1].endswith("/tools/encode_quadruped_review_media.py")
    assert encode[encode.index("--render-manifest") + 1] == render[
        render.index("--manifest") + 1
    ]
    assert encode[encode.index("--label") + 1] == "walking_side"
    assert encode[encode.index("--action") + 1] == "Walking"
    assert encode[encode.index("--view") + 1] == "side"


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
        "heading", "rig_audit", "support_plane", "support_plane_readback",
        "retarget", "gait_direction",
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


def review_render_manifest_payload(
    input_glb,
    frame_dir,
    *,
    action="Walking",
    resolved_action="Walking",
    view="side",
    asset_yaw_deg=0.0,
    n_frames=4,
):
    frame_dir.mkdir()
    frames = []
    start, end = 1.0, 9.0
    for index in range(n_frames):
        path = frame_dir / f"frame_{index:04d}.png"
        path.write_bytes(f"png-{index}".encode("utf-8"))
        fraction = 0.0 if n_frames == 1 else index / (n_frames - 1)
        frames.append(
            {
                "index": index,
                "sample_fraction": fraction,
                "source_action_frame": float(
                    int(round(start + (end - start) * fraction))
                ),
                "artifact": file_record(path),
            }
        )
    return {
        "schema": RENDER_MANIFEST_SCHEMA,
        "status": "frames_rendered",
        "formal_dataset_registration_authorized": False,
        "input_glb": file_record(input_glb),
        "request": {
            "action": action,
            "resolved_action": resolved_action,
            "rest_pose": False,
            "view": view,
            "asset_yaw_deg": asset_yaw_deg,
            "n_frames": n_frames,
            "resolution": {"width": 512, "height": 384},
            "fps": 8,
            "output_dir": str(frame_dir.resolve()),
        },
        "render_config": deepcopy(REVIEW_RENDER_CONFIG),
        "action_frame_range": [start, end],
        "frames": frames,
    }


def test_render_manifest_binds_action_view_input_and_every_frame(tmp_path):
    input_glb = tmp_path / "final.glb"
    input_glb.write_bytes(b"final-animation")
    frame_dir = tmp_path / "walking_side_frames"
    payload = review_render_manifest_payload(input_glb, frame_dir)
    manifest = tmp_path / "walking_side_render_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    accepted = require_render_manifest(
        manifest,
        input_glb=input_glb,
        frame_dir=frame_dir,
        action="Walking",
        view="side",
        asset_yaw_deg=0.0,
        n_frames=4,
    )
    assert accepted["input_glb"] == file_record(input_glb)
    assert len(accepted["frames"]) == 4

    swapped = deepcopy(payload)
    swapped["request"]["action"] = "Idle"
    swapped["request"]["resolved_action"] = "Idle"
    swapped["request"]["view"] = "front"
    manifest.write_text(json.dumps(swapped), encoding="utf-8")
    with pytest.raises(RuntimeError, match="request identity"):
        require_render_manifest(
            manifest,
            input_glb=input_glb,
            frame_dir=frame_dir,
            action="Walking",
            view="side",
            asset_yaw_deg=0.0,
            n_frames=4,
        )

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    (frame_dir / "frame_0002.png").write_bytes(b"copied-frame")
    with pytest.raises(RuntimeError, match="file binding"):
        require_render_manifest(
            manifest,
            input_glb=input_glb,
            frame_dir=frame_dir,
            action="Walking",
            view="side",
            asset_yaw_deg=0.0,
            n_frames=4,
        )


def test_encode_manifest_rejects_copied_black_video_and_swapped_identity(
    tmp_path,
):
    input_glb = tmp_path / "final.glb"
    input_glb.write_bytes(b"final-animation")
    frame_dir = tmp_path / "walking_side_frames"
    render_payload = review_render_manifest_payload(input_glb, frame_dir)
    render_manifest = tmp_path / "walking_side_render_manifest.json"
    render_manifest.write_text(json.dumps(render_payload), encoding="utf-8")
    video = tmp_path / "walking_side.mp4"
    video.write_bytes(b"authenticated-h264-video")
    video_record = {
        **file_record(video),
        "codec": "h264",
        "width": 512,
        "height": 384,
        "frame_count": 4,
        "frame_rate": "8/1",
        "duration_seconds": 0.5,
    }
    encode_payload = {
        "schema": ENCODE_MANIFEST_SCHEMA,
        "status": "video_encoded_and_probed",
        "formal_dataset_registration_authorized": False,
        "media_identity": {
            "label": "walking_side",
            "action": "Walking",
            "view": "side",
            "asset_yaw_deg": 0.0,
        },
        "render_manifest": file_record(render_manifest),
        "frame_set": render_frame_set(render_payload),
        "ffmpeg": expected_review_ffmpeg_config(frame_dir, 4),
        "video": video_record,
    }
    encode_manifest = tmp_path / "walking_side_encode_manifest.json"
    encode_manifest.write_text(json.dumps(encode_payload), encoding="utf-8")

    accepted = require_encode_manifest(
        encode_manifest,
        label="walking_side",
        render_manifest_path=render_manifest,
        render_payload=render_payload,
        frame_dir=frame_dir,
        video_path=video,
        video_record=video_record,
        action="Walking",
        view="side",
        asset_yaw_deg=0.0,
        n_frames=4,
    )
    assert accepted["video"] == video_record

    swapped = deepcopy(encode_payload)
    swapped["media_identity"]["action"] = "Idle"
    swapped["media_identity"]["view"] = "front"
    encode_manifest.write_text(json.dumps(swapped), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity/status"):
        require_encode_manifest(
            encode_manifest,
            label="walking_side",
            render_manifest_path=render_manifest,
            render_payload=render_payload,
            frame_dir=frame_dir,
            video_path=video,
            video_record=video_record,
            action="Walking",
            view="side",
            asset_yaw_deg=0.0,
            n_frames=4,
        )

    encode_manifest.write_text(json.dumps(encode_payload), encoding="utf-8")
    video.write_bytes(b"copied-black-h264-video")
    copied_black_record = {
        **file_record(video),
        "codec": "h264",
        "width": 512,
        "height": 384,
        "frame_count": 4,
        "frame_rate": "8/1",
        "duration_seconds": 0.5,
    }
    with pytest.raises(RuntimeError, match="file binding"):
        require_encode_manifest(
            encode_manifest,
            label="walking_side",
            render_manifest_path=render_manifest,
            render_payload=render_payload,
            frame_dir=frame_dir,
            video_path=video,
            video_record=copied_black_record,
            action="Walking",
            view="side",
            asset_yaw_deg=0.0,
            n_frames=4,
        )


def repair_manifest_payload(
    source,
    repaired,
    *,
    parameters,
    status,
    extension,
    remaining_seed_edges,
):
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
            "maximum_allowed_rest_geometry_delta": 1.0e-6,
        }
    )
    payload = {
        "schema": WEIGHT_REPAIR_SCHEMA,
        "input": file_record(source),
        "output": file_record(repaired),
        "front_axis": "positive-x",
        "parameters": parameters,
        "authority_contract": authority,
        "final_measurements": {
            "maximum_extension_ratio_of_rest_diagonal": extension,
            "remaining_seed_edge_count": remaining_seed_edges,
        },
        "status": status,
        "formal_dataset_registration_authorized": False,
    }
    if parameters["cross_limb_authority"] == "low-slice-components":
        chains = (
            "front_side_negative",
            "front_side_positive",
            "hind_side_negative",
            "hind_side_positive",
        )
        payload["cross_limb_preclean"] = {
            "entries_after": 0,
            "weight_mass_after": 0,
            "final_forbidden_entries": 0,
            "final_forbidden_weight_mass": 0,
            "skipped_for_residual_pass": parameters[
                "skip_cross_limb_preclean"
            ],
            "spatial_authority": {
                "method": "four_largest_disconnected_low_slice_components",
                "immutable_through_motion_repair": True,
                "blender_up_axis": "positive-z",
                "height_fraction": parameters["limb_slice_height_fraction"],
                "substantial_component_sizes_descending": [40, 30, 20, 10],
                "assignment_counts": {
                    chain: index + 10 for index, chain in enumerate(chains)
                },
                "components": [
                    {"semantic_chain": chain} for chain in chains
                ],
            },
        }
    return payload


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


def test_fallback_incomplete_evidence_is_inspectable_but_not_admitted(tmp_path):
    source = tmp_path / "source.glb"
    source.write_bytes(b"source")
    repaired = tmp_path / "fallback.glb"
    repaired.write_bytes(b"fallback")
    manifest = tmp_path / "fallback.json"
    payload = repair_manifest_payload(
        source,
        repaired,
        parameters=WEIGHT_REPAIR_FALLBACK_PARAMETERS,
        status=WEIGHT_REPAIR_INCOMPLETE_STATUS,
        extension=0.0304,
        remaining_seed_edges=27,
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    inspected = require_weight_repair(
        manifest,
        source,
        repaired,
        expected_parameters=WEIGHT_REPAIR_FALLBACK_PARAMETERS,
        allow_incomplete=True,
    )
    assert inspected["status"] == WEIGHT_REPAIR_INCOMPLETE_STATUS
    with pytest.raises(WeightRepairIncomplete, match="did not converge"):
        require_weight_repair(
            manifest,
            source,
            repaired,
            expected_parameters=WEIGHT_REPAIR_FALLBACK_PARAMETERS,
        )


def test_residual_fallback_still_requires_zero_remaining_seeds(tmp_path):
    source = tmp_path / "fallback.glb"
    source.write_bytes(b"fallback")
    repaired = tmp_path / "residual.glb"
    repaired.write_bytes(b"residual")
    manifest = tmp_path / "residual.json"
    payload = repair_manifest_payload(
        source,
        repaired,
        parameters=WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS,
        status=WEIGHT_REPAIR_READY_STATUS,
        extension=0.019,
        remaining_seed_edges=1,
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="contradicts convergence"):
        require_weight_repair(
            manifest,
            source,
            repaired,
            expected_parameters=WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS,
        )


def test_residual_fallback_ready_contract_accepts_only_strict_convergence(
    tmp_path,
):
    source = tmp_path / "fallback.glb"
    source.write_bytes(b"fallback")
    repaired = tmp_path / "residual.glb"
    repaired.write_bytes(b"residual")
    manifest = tmp_path / "residual.json"
    payload = repair_manifest_payload(
        source,
        repaired,
        parameters=WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS,
        status=WEIGHT_REPAIR_READY_STATUS,
        extension=0.019998,
        remaining_seed_edges=0,
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    accepted = require_weight_repair(
        manifest,
        source,
        repaired,
        expected_parameters=WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS,
    )
    assert accepted["status"] == WEIGHT_REPAIR_READY_STATUS


def review_repair_attempt(
    tmp_path,
    *,
    stage,
    status,
    extension,
    remaining_seed_edges,
):
    manifest = tmp_path / f"{stage}_manifest.json"
    manifest.write_text(json.dumps({"stage": stage}), encoding="utf-8")
    output = tmp_path / f"{stage}.glb"
    output.write_bytes(stage.encode("utf-8"))
    return weight_repair_attempt_record(
        stage,
        {
            "status": status,
            "final_measurements": {
                "maximum_extension_ratio_of_rest_diagonal": extension,
                "remaining_seed_edge_count": remaining_seed_edges,
            },
        },
        manifest,
        output,
    )


def test_fallback_review_schema_names_every_stage_and_final_artifact(tmp_path):
    attempts = [
        review_repair_attempt(
            tmp_path,
            stage="primary",
            status=WEIGHT_REPAIR_INCOMPLETE_STATUS,
            extension=0.14,
            remaining_seed_edges=1477,
        ),
        review_repair_attempt(
            tmp_path,
            stage="fallback_a",
            status=WEIGHT_REPAIR_INCOMPLETE_STATUS,
            extension=0.0304,
            remaining_seed_edges=27,
        ),
        review_repair_attempt(
            tmp_path,
            stage="fallback_b",
            status=WEIGHT_REPAIR_READY_STATUS,
            extension=0.019998,
            remaining_seed_edges=0,
        ),
    ]

    require_weight_repair_branch_consistency("fallback_a_b", attempts)
    gates = weight_repair_gate_attempts(attempts)
    outputs = weight_repair_output_descriptors(attempts)
    final_artifact = weight_repair_final_artifact("fallback_a_b")

    assert [attempt["stage"] for attempt in gates] == [
        "primary",
        "fallback_a",
        "fallback_b",
    ]
    assert gates[1]["pipeline_stage"] == (
        "weight_repair_low_slice_edge_average"
    )
    assert gates[1]["input_glb_output_descriptor"] == (
        "retargeted_animated_glb"
    )
    assert gates[2]["input_glb_output_descriptor"] == (
        "weight_repair_fallback_a_glb"
    )
    assert set(outputs) == {
        "weight_repair_primary_glb",
        "weight_repair_primary_manifest",
        "weight_repair_fallback_a_glb",
        "weight_repair_fallback_a_manifest",
        "weight_repair_fallback_b_glb",
        "weight_repair_fallback_b_manifest",
    }
    assert final_artifact == {
        "stage": "fallback_b",
        "glb_output_descriptor": "weight_repair_fallback_b_glb",
        "manifest_output_descriptor": "weight_repair_fallback_b_manifest",
    }
    assert outputs[final_artifact["glb_output_descriptor"]] == attempts[-1][
        "output_glb"
    ]
    assert outputs[final_artifact["manifest_output_descriptor"]] == attempts[-1][
        "manifest"
    ]
    require_weight_repair_pipeline_order(
        [
            "heading",
            "deformation",
            "weight_repair",
            "weight_repair_low_slice_edge_average",
            "weight_repair_low_slice_edge_average_residual",
            "gait_direction_repaired_low_slice_edge_average_residual",
            "deformation_repaired_low_slice_edge_average_residual",
            "render_walking_side",
        ],
        "fallback_a_b",
    )


def test_review_repair_branch_rejects_missing_or_relabelled_fallback_b(tmp_path):
    attempts = [
        review_repair_attempt(
            tmp_path,
            stage="primary",
            status=WEIGHT_REPAIR_INCOMPLETE_STATUS,
            extension=0.14,
            remaining_seed_edges=1477,
        ),
        review_repair_attempt(
            tmp_path,
            stage="fallback_a",
            status=WEIGHT_REPAIR_INCOMPLETE_STATUS,
            extension=0.0304,
            remaining_seed_edges=27,
        ),
    ]
    with pytest.raises(RuntimeError, match="attempt sequence"):
        require_weight_repair_branch_consistency("fallback_a_b", attempts)

    attempts.append(
        review_repair_attempt(
            tmp_path,
            stage="fallback_b",
            status=WEIGHT_REPAIR_READY_STATUS,
            extension=0.019998,
            remaining_seed_edges=0,
        )
    )
    unnecessary_residual = deepcopy(attempts)
    unnecessary_residual[1]["status"] = WEIGHT_REPAIR_READY_STATUS
    with pytest.raises(RuntimeError, match="contradictory"):
        require_weight_repair_branch_consistency(
            "fallback_a_b", unnecessary_residual
        )
    attempts[-1]["input_glb_output_descriptor"] = (
        "retargeted_animated_glb"
    )
    with pytest.raises(RuntimeError, match="descriptor contract"):
        require_weight_repair_branch_consistency("fallback_a_b", attempts)
    with pytest.raises(RuntimeError, match="pipeline order"):
        require_weight_repair_pipeline_order(
            [
                "deformation",
                "weight_repair",
                "weight_repair_low_slice_edge_average",
                "render_walking_side",
            ],
            "fallback_a_b",
        )


def test_primary_and_unrepaired_review_final_artifact_contracts(tmp_path):
    primary = review_repair_attempt(
        tmp_path,
        stage="primary",
        status=WEIGHT_REPAIR_READY_STATUS,
        extension=0.019,
        remaining_seed_edges=0,
    )
    require_weight_repair_branch_consistency("primary", [primary])
    assert weight_repair_final_artifact("primary") == {
        "stage": "primary",
        "glb_output_descriptor": WEIGHT_REPAIR_STAGE_CONTRACTS["primary"][
            "output_glb_output_descriptor"
        ],
        "manifest_output_descriptor": WEIGHT_REPAIR_STAGE_CONTRACTS["primary"][
            "manifest_output_descriptor"
        ],
    }
    fallback_a = review_repair_attempt(
        tmp_path,
        stage="fallback_a",
        status=WEIGHT_REPAIR_READY_STATUS,
        extension=0.0195,
        remaining_seed_edges=0,
    )
    primary_incomplete = review_repair_attempt(
        tmp_path,
        stage="primary",
        status=WEIGHT_REPAIR_INCOMPLETE_STATUS,
        extension=0.14,
        remaining_seed_edges=1477,
    )
    require_weight_repair_branch_consistency(
        "fallback_a", [primary_incomplete, fallback_a]
    )
    assert weight_repair_final_artifact("fallback_a") == {
        "stage": "fallback_a",
        "glb_output_descriptor": "weight_repair_fallback_a_glb",
        "manifest_output_descriptor": "weight_repair_fallback_a_manifest",
    }
    require_weight_repair_pipeline_order(
        [
            "deformation",
            "weight_repair",
            "weight_repair_low_slice_edge_average",
            "gait_direction_repaired_low_slice_edge_average",
            "deformation_repaired_low_slice_edge_average",
            "render_walking_side",
        ],
        "fallback_a",
    )
    require_weight_repair_branch_consistency("not_needed", [])
    assert weight_repair_final_artifact("not_needed") == {
        "stage": "not_needed",
        "glb_output_descriptor": "retargeted_animated_glb",
        "manifest_output_descriptor": None,
    }


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
    evidence.write_text(
        json.dumps(
            {
                "mesh": {
                    "world_bbox_diagonal": 4.0,
                    "world_bbox_min": [-2.0, -1.0, 0.0],
                    "world_bbox_extent": [4.0, 2.0, 1.0],
                    "vertices": 48,
                    "polygons": 80,
                },
                "armature": {
                    "bone_count": 14,
                    "records": support_rig_records(),
                },
            }
        ),
        encoding="utf-8",
    )
    dual = dual_support_plane_evidence()
    primary = dual["primary"]
    plane = primary["plane"]
    manifest = tmp_path / "level.json"
    scene_summary = {
        "mesh_count": 2,
        "skinned_mesh_count": 1,
        "armature_count": 1,
        "bone_count": 14,
        "material_count": 1,
        "image_count": 2,
        "action_count": 0,
    }
    valid_payload = {
                "schema": SUPPORT_PLANE_SCHEMA,
                "input": file_record(source),
                "output": file_record(leveled),
                "review_evidence": file_record(evidence),
                "support_plane": {
                    "front_axis": "positive-x",
                    "plane_source": "mesh-foot-bottoms",
                    "foot_leaves": support_semantic_rig()["foot_leaves"],
                    "dual_authority": dual,
                    "mesh_foot_capture_counts": primary["capture_counts"],
                    "mesh_foot_contact_band_sizes": (
                        primary["contact_band_sizes"]
                    ),
                    "foot_points_before": deepcopy(primary["foot_points"]),
                    "z_equals_ax_plus_by_plus_c": plane[
                        "z_equals_ax_plus_by_plus_c"
                    ],
                    "residual_z": plane["residual_z"],
                    "maximum_residual": plane["maximum_residual"],
                    "maximum_residual_ratio_of_mesh_diagonal": plane[
                        "maximum_residual_ratio_of_mesh_diagonal"
                    ],
                    "maximum_reviewed_residual_ratio_of_mesh_diagonal": 0.02,
                    "normal_before": plane["normal"],
                    "tilt_deg": plane["tilt_deg"],
                    "maximum_tilt_deg": 30.0,
                    "applied_vertical_translation": 0.0,
                    "foot_points_after": deepcopy(primary["foot_points"]),
                    "crosscheck_foot_points_after": deepcopy(
                        dual["crosscheck"]["foot_points"]
                    ),
                    "minimum_foot_z_after": 0.0,
                    "policy": SUPPORT_PLANE_POLICY,
                },
                "preservation_contract": {
                    "mesh_topology_changed": False,
                    "material_changed": False,
                    "skeleton_hierarchy_changed": False,
                    "skin_weights_changed": False,
                    "animation_present_or_changed": False,
                },
                "scene_before": dict(scene_summary),
                "scene_after": dict(scene_summary),
                "status": "technical_spike_only_pending_retarget_and_visual_qa",
                "formal_dataset_registration_authorized": False,
            }
    manifest.write_text(
        json.dumps(valid_payload),
        encoding="utf-8",
    )
    readback = tmp_path / "level_readback.json"
    readback.write_text(
        json.dumps(
            support_output_readback(
                source=source,
                leveled=leveled,
                manifest=manifest,
                rig_audit=evidence,
                dual=dual,
                scene=scene_summary,
            )
        ),
        encoding="utf-8",
    )

    require_support_plane(
        manifest,
        source,
        leveled,
        readback_path=readback,
        plane_source="mesh-foot-bottoms",
        review_evidence=evidence,
        front_axis="positive-x",
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["support_plane"]["dual_authority"]["primary"][
        "contact_band_sizes"
    ][1] = 0
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="primary contact band is sparse"):
        require_support_plane(
            manifest,
            source,
            leveled,
            readback_path=readback,
            plane_source="mesh-foot-bottoms",
            review_evidence=evidence,
            front_axis="positive-x",
        )

    payload = deepcopy(valid_payload)
    payload["support_plane"]["foot_points_after"][0][0] += 0.001
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="rigid leveling transform"):
        require_support_plane(
            manifest,
            source,
            leveled,
            readback_path=readback,
            plane_source="mesh-foot-bottoms",
            review_evidence=evidence,
            front_axis="positive-x",
        )

    payload = deepcopy(valid_payload)
    payload["scene_before"] = {}
    payload["scene_after"] = {}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scene preservation summary"):
        require_support_plane(
            manifest,
            source,
            leveled,
            readback_path=readback,
            plane_source="mesh-foot-bottoms",
            review_evidence=evidence,
            front_axis="positive-x",
        )

    payload = deepcopy(valid_payload)
    payload["scene_before"]["bone_count"] = 15
    payload["scene_after"]["bone_count"] = 15
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match the rig audit"):
        require_support_plane(
            manifest,
            source,
            leveled,
            readback_path=readback,
            plane_source="mesh-foot-bottoms",
            review_evidence=evidence,
            front_axis="positive-x",
        )

    payload = deepcopy(valid_payload)
    payload["output"] = file_record(source)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="support-plane output file binding does not match"
    ):
        require_support_plane(
            manifest,
            source,
            leveled,
            readback_path=readback,
            plane_source="mesh-foot-bottoms",
            review_evidence=evidence,
            front_axis="positive-x",
        )

    payload = deepcopy(valid_payload)
    payload["support_plane"]["foot_leaves"] = [
        "fake_a",
        "fake_b",
        "fake_c",
        "fake_d",
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="independent rig semantics"):
        require_support_plane(
            manifest,
            source,
            leveled,
            readback_path=readback,
            plane_source="mesh-foot-bottoms",
            review_evidence=evidence,
            front_axis="positive-x",
        )

    manifest.write_text(json.dumps(valid_payload), encoding="utf-8")
    readback_payload = support_output_readback(
        source=source,
        leveled=leveled,
        manifest=manifest,
        rig_audit=evidence,
        dual=dual,
        scene=scene_summary,
    )
    readback_payload["mesh_and_weight_binding"][
        "maximum_skin_weight_delta"
    ] = 0.1
    readback.write_text(json.dumps(readback_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="post-smoothing weights changed"):
        require_support_plane(
            manifest,
            source,
            leveled,
            readback_path=readback,
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
        motion_donor_contract=expected_motion_donor_contract(
            CAT_MOTION_DONOR_ID
        ),
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
            motion_donor_contract=expected_motion_donor_contract(
                CAT_MOTION_DONOR_ID
            ),
        )

    payload["export"]["canonical_front_axis"] = "positive-x"
    payload["source_motion"]["geometry_used"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="motion-source authority"):
        require_retarget_manifest(
            manifest,
            target,
            source,
            output,
            target_front_axis="negative-y",
            motion_amplitude=1.0,
            motion_basis_yaw_deg=0,
            side_chain_mode="matched",
            motion_donor_contract=expected_motion_donor_contract(
                CAT_MOTION_DONOR_ID
            ),
        )

    payload["source_motion"]["geometry_used"] = False
    payload["target"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="retarget target"):
        require_retarget_manifest(
            manifest,
            target,
            source,
            output,
            target_front_axis="negative-y",
            motion_amplitude=1.0,
            motion_basis_yaw_deg=0,
            side_chain_mode="matched",
            motion_donor_contract=expected_motion_donor_contract(
                CAT_MOTION_DONOR_ID
            ),
        )
