from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_retarget_quaternius_to_generated_quadruped.py"
)


def test_generated_quadruped_retarget_uses_semantics_and_world_space_transfer():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'SCHEMA = "avengine_generated_quadruped_retarget_v5"' in text
    assert "infer_quadruped_semantics" in text
    assert "source_sampling_plan" in text
    assert "target_chain_fraction" in text
    assert "source_first.slerp(source_second" in text
    assert '"--motion-amplitude"' in text
    assert "source_delta_world" in text
    assert "scaled_delta_world" in text
    assert '"--rotation-transfer-mode"' in text
    assert '"world-left-delta-v2"' in text
    assert '"legacy-rest-local-right-delta-v1"' in text
    assert '"--motion-basis-yaw-deg"' in text
    assert '"--side-chain-mode"' in text
    assert '"--motion-basis-decision"' in text
    assert "target_animation_generation_authorized" in text
    assert "human_approved" in text
    assert "decision_sha256" in text
    assert 'choices=("matched", "swapped")' in text
    assert "CARDINAL_MOTION_BASIS_YAWS" in text
    assert "basis_rotation" in text
    assert "@ source_delta_world" in text
    assert "@ basis_rotation.inverted()" in text
    assert (
        'source_pose_world @ entry["source_rest_world"].inverted()' in text
    )
    assert 'scaled_delta_world @ entry["target_rest_world"]' in text
    assert "motion_amplitude" in text
    assert '"--disable-foot-grounding"' in text
    assert "minimum_semantic_foot_or_evaluated_mesh_floor_delta_from_rest_v2" in text
    assert "evaluated_mesh_floor" in text
    assert "correction_world_z" in text
    assert "source_rest_world" in text
    assert "target_rest_world" in text
    assert '"local_bone_roll_copied": (' in text
    assert '== "legacy-rest-local-right-delta-v1"' in text
    assert "auxiliary_branches" in text
    assert "rigid_follow_nearest_semantic_parent" in text
    assert '"formal_dataset_registration_authorized": False' in text


def test_generated_quadruped_retarget_can_preserve_proven_template_full_pose():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--pose-transfer-mode"' in text
    assert '"template-local-full-pose-v1"' in text
    assert "template_local_pose_plan" in text
    assert "parent_topology_uniform_armature_local_scale_rest_matrix_v2" in text
    assert "bone_armature_endpoints" in text
    assert "target_armature_object_world_scale" in text
    assert '"matrix_basis"' in text
    assert "source.pose.bones[name].matrix_basis.copy()" in text
    assert "bake_template_local_action" in text
    assert '"channels_copied": ["translation", "rotation", "scale"]' in text
    assert "scaled_location" in text
    assert "scaled_rotation" in text
    assert "scaled_scale" in text
    assert "minimum_semantic_foot_delta_from_rest_v3" in text
    assert '"evaluated_mesh_floor_is_diagnostic_only": True' in text
    assert '"template_compatibility_proof": template_compatibility' in text
    assert '"full_local_translation_rotation_scale_copied": (' in text


def test_generated_quadruped_retarget_can_pin_fitted_skeleton_ankle_trajectories():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"world-rotation-foot-ik-v3"' in text
    assert "limb_ik_plan" in text
    assert '"heads_world"' in text
    assert "solve_two_bone_ankle_ik" in text
    assert "rotate_pose_joint_toward" in text
    assert "root_motion_plan" in text
    assert "apply_source_root_motion" in text
    assert "source_root_head_delta_scaled_to_target_v1" in text
    assert "source_rest_relative_ankle_trajectory_ccd_v2" in text
    assert '"source_rest_foot_world"' in text
    assert '"target_rest_foot_world"' in text
    assert '"source_rest_hip_world"' in text
    assert '"target_rest_hip_world"' in text
    assert "componentwise_median_source_hip_trajectory_v1" in text
    assert "upper_segment_length = (lower.head - upper.head).length" in text
    assert "lower_segment_length = (foot.head - lower.head).length" in text
    assert '"source_foot_rest_world_rotation"' in text
    assert '"target_foot_rest_world_rotation"' in text
    assert "desired_foot_armature_rotation" in text
    assert "lock_target_rest_world_v1" in text
    assert '"minimum_semantic_foot_delta_from_rest_v3"' in text
    assert '"limb_ik_plan": serializable_limb_ik_plan(limb_ik_specs)' in text
    assert '"--target-derivation-manifest"' in text
    assert "load_target_derivation_manifest" in text
    assert "avengine_generated_quadruped_joint_weight_smoothing_v1" in text
    assert "avengine_generated_quadruped_joint_weight_smoothing_v3" in text
    assert 'payload.get("fitted_skeleton_rest_matrices_preserved") is not True' in text
    assert 'payload.get("only_vertex_weights_modified") is not True' in text
    assert '"authenticated_target_derivation": motion_basis_decision.get(' in text


def test_generated_quadruped_retarget_preserves_target_authority_and_no_replace():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"mesh_pbr_skeleton_and_weights_authority": True' in text
    assert "if path.exists() or path.is_symlink()" in text
    assert "refusing to replace" in text
    assert '"geometry_used": False' in text
    assert '"weights_used": False' in text


def test_generated_quadruped_retarget_supports_only_cardinal_front_axes():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"positive-y"' in text
    assert '"negative-y"' in text
    assert '"negative-y": 90.0' in text
    assert '"positive-y": -90.0' in text
