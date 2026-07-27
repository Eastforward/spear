"""Static contract for the post-TokenRig target-native review runner."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/run_target_native_generated_quadruped_review.py"
)


def test_runner_preserves_the_required_stage_order_and_review_views():
    text = SCRIPT.read_text(encoding="utf-8")
    command_builder = text[text.index("def build_initial_commands(") :]

    assert (
        'SCHEMA = "avengine_target_native_generated_quadruped_review_run_v4"'
        in text
    )
    assert "avengine_target_native_generated_quadruped_review_run_v3" not in text
    assert '"--tokenrig-closure-manifest"' in text
    assert '"--expected-tokenrig-closure-sha256"' in text
    assert "validate_tokenrig_closure_manifest(" in text
    assert "require_target_rig_lineage_unchanged(" in text
    assert '"target_rig_lineage": dict(target_rig_lineage)' in text
    expected = [
        '"heading"',
        '"rig_audit"',
        '"support_plane"',
        '"support_plane_readback"',
        '"retarget"',
        '"gait_direction"',
        '"deformation"',
        '"weight_repair"',
        '"gait_direction_repaired"',
        '"deformation_repaired"',
        '"walking_side"',
    ]
    positions = [command_builder.index(item) for item in expected]
    assert positions == sorted(positions)
    assert '"--python-exit-code"' in text
    assert '"--target-glb", str(paths["leveled_glb"])' in command_builder
    assert '"support_plane_manifest": file_record(paths["level_manifest"])' in text
    assert (
        '"support_plane_output_readback": file_record('
        in text
    )
    for label in (
        "walking_side",
        "walking_front",
        "walking_rear",
        "idle_side",
        "idle_front",
        "idle_rear",
    ):
        assert label in text


def test_runner_uses_gentle_repair_and_fails_closed_before_rendering():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'default="mesh-foot-bottoms"' in text
    assert 'choices=("mesh-foot-bottoms",)' in text
    assert "generated_animal_support_plane_contract" in text
    assert "validate_rigid_leveling_transform" in text
    assert "validate_output_glb_readback" in text
    assert "infer_quadruped_semantics" in text
    assert "support-plane endpoint authority is forbidden" in text
    assert 'default="auto"' in text
    assert '"--repair-mode", "component-parent-lock"' in text
    assert '"--component-rings", "4"' in text
    assert '"--extension-threshold", "0.02"' in text
    assert '"weight_repair_low_slice_edge_average"' in text
    assert '"weight_repair_low_slice_edge_average_residual"' in text
    assert '"gait_direction_repaired_low_slice_edge_average"' in text
    assert '"deformation_repaired_low_slice_edge_average"' in text
    assert (
        'if fallback_manifest["status"] == WEIGHT_REPAIR_READY_STATUS:'
        in text
    )
    assert "residual_label, residual_command = fallback_commands[3]" in text
    assert '"weight_repair_branch": repair_branch' in text
    assert '"weight_repair_final_artifact": repair_final_artifact' in text
    assert '"weight_repair_primary_glb"' in text
    assert '"weight_repair_primary_manifest"' in text
    assert '"weight_repair_fallback_a_glb"' in text
    assert '"weight_repair_fallback_a_manifest"' in text
    assert '"weight_repair_fallback_b_glb"' in text
    assert '"weight_repair_fallback_b_manifest"' in text
    assert "require_weight_repair_branch_consistency" in text
    assert '"cross_limb_authority": "low-slice-components"' in text
    assert '"limb_slice_height_fraction": 0.2' in text
    assert '"maximum_passes": 12' in text
    assert '"inner_iterations": 8' in text
    assert '"skip_cross_limb_preclean": True' in text
    assert "final_extension <= 0.02" in text
    assert "remaining_seed_edges == 0" in text
    assert "require_rig_audit" in text
    assert "require_weight_repair" in text
    assert "require_pass=True" in text
    assert '"all_automatic_gates_passed": True' in text


def test_runner_uses_reviewed_semantics_and_keeps_research_status_explicit():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--reviewed-source-front-yaw-deg" in text
    assert "--maximum-foot-plane-residual-ratio" in text
    assert '"--action", "Walking"' in text
    assert '"--action", "Idle"' in text
    assert '"--technical-spike-only"' in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert "refusing to replace output root" in text
    assert "ffprobe" in text


def test_runner_requires_render_and_encode_lineage_for_every_v4_video():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_glb_animation_frame_render_v1" in text
    assert "avengine_glb_animation_video_encode_v1" in text
    assert '"--manifest", str(media_paths["render_manifest"])' in text
    assert '"encode_quadruped_review_media.py"' in text
    assert '"--render-manifest", str(media_paths["render_manifest"])' in text
    assert "require_review_render_stage(" in text
    assert "require_review_encode_stage(" in text
    assert "require_review_media_set(" in text
    assert '"media_lineage": media_lineage' in text
    assert '"render_manifest": file_record(media_paths["render_manifest"])' in text
    assert '"encode_manifest": file_record(media_paths["encode_manifest"])' in text
