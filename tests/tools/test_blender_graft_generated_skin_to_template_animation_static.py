from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_graft_generated_skin_to_template_animation.py"
)


def test_graft_preserves_generated_mesh_pbr_and_skintokens_weights():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_generated_skin_template_animation_graft_v2" in text
    assert '"mesh_preserved": True' in text
    assert '"pbr_preserved": True' in text
    assert '"skintokens_vertex_weights_preserved": True' in text
    assert "rename_vertex_groups" in text
    assert "semantic_vertex_group_mapping" in text
    assert "hierarchy_vertex_group_mapping" in text
    assert "conditioning_hierarchy_child_assignment_v1" in text
    assert "complete_generated_bone_and_group_coverage" in text


def test_graft_uses_approved_unchanged_template_animation_without_retarget():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--motion-basis-decision"' in text
    assert "load_motion_basis_decision" in text
    assert "load_fixed_lineage_contract" in text
    assert "agent_delegated_research" in text
    assert '"human_approved": False' in text
    assert '"second_retarget_performed": False' in text
    assert '"skeleton_used": True' in text
    assert '"walk_idle_actions_used": True' in text
    assert "approved_motion_basis_yaw_deg" in text
    assert "approved_side_chain_mode" in text


def test_graft_fails_closed_on_alignment_or_rest_bind_changes():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--maximum-carrier-bbox-error-ratio"' in text
    assert '"--maximum-rest-bind-error-ratio"' in text
    assert '"--minimum-action-deformation-ratio"' in text
    assert "animated carrier and generated mesh are not aligned" in text
    assert "template animation graft changed the generated rest mesh" in text
    assert '"rest_mesh_preserved": True' in text
    assert 'carrier.data.pose_position = "POSE"' in text
    assert '"export_armature_pose_position": "POSE"' in text
    assert "probe_action_deformation" in text
    assert "is effectively static after skeleton graft" in text
    assert '"pre_export_action_deformation_probe": action_deformation_probe' in text
    assert "refusing to replace" in (
        Path(__file__).resolve().parents[2]
        / "tools/blender_retarget_quaternius_to_generated_quadruped.py"
    ).read_text(encoding="utf-8")
