from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_smooth_generated_quadruped_joint_weights.py"
)


def test_joint_weight_smoothing_preserves_native_asset_and_exports_no_animation():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--input"' in text
    assert '"--output"' in text
    assert '"--manifest"' in text
    assert "export_animations=False" in text
    assert '"native_mesh_topology_preserved": True' in text
    assert '"pbr_material_preserved": True' in text


def test_joint_weight_smoothing_is_limb_semantic_and_local():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "infer_quadruped_semantics" in text
    assert "remove_cross_limb_leakage" in text
    assert "hard_joint_seams" in text
    assert "affected_neighbourhood" in text
    assert "dominant_chains[index] == chain" in text
    assert "adjacent_bone_pairs" in text
    assert "cross_limb_dominant_edges" in text
    assert "stabilize_cross_limb_bridges" in text
    assert "axial_limb_attachment_edges" in text
    assert "stabilize_axial_limb_attachments" in text
    assert "lowest_common_ancestor_name" in text
    assert "bridge_vertices_bound_toward_common_torso_ancestor" in text
    assert "attachment_vertices_bound_toward_common_torso_ancestor" in text
    assert "semantic_chain_bounded_graph_expansion" in text
    assert '"post_same_chain_smoothing_pass"' in text
    assert '"topology_deleted": False' in text


def test_joint_weight_smoothing_restores_bounded_normalized_influences():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "top_k_normalize" in text
    assert "--top-k" in text
    assert "--minimum-weight" in text
    assert "maximum_influences" in text
    assert "minimum_weight_sum" in text
    assert "maximum_weight_sum" in text
    assert "bmesh" in text


def test_joint_weight_smoothing_defaults_to_three_canary_full_gait_profile():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'default=0.4' in text
    assert 'default=5' in text
    assert 'default=8' in text
    assert 'default=0.75' in text
    assert 'default=3' in text
    assert '"profile": "native_quadruped_full_gait_v1"' in text
