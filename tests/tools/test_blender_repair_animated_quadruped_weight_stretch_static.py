"""Static safety contracts for motion-aware quadruped weight repair."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_repair_animated_quadruped_weight_stretch.py"
)


def test_repair_samples_real_walk_and_idle_before_changing_weights():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_motion_aware_quadruped_weight_repair_v1" in text
    assert '(("Walking", "walk"), ("Idle", "idle"))' in text
    assert "evaluated_geometry" in text
    assert "maximum_extension_ratio_of_rest_diagonal" in text
    assert "select_seed_edges" in text


def test_repair_preserves_geometry_skeleton_materials_and_action_curves():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "action_fingerprint" in text
    assert "weight repair changed approved action curves" in text
    assert "weight repair changed rest topology" in text
    assert "weight repair changed rest geometry" in text
    assert '"pbr_material_preserved": True' in text
    assert '"fitted_skeleton_rest_matrices_preserved": True' in text
    assert '"only_vertex_weights_modified_in_memory": True' in text


def test_repair_exports_in_pose_mode_and_requires_new_output_paths():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'armature.data.pose_position = "POSE"' in text
    assert 'export_animation_mode="NLA_TRACKS"' in text
    assert "refusing to replace" in text
    assert 'with manifest.open("x"' in text


def test_repair_supports_conservative_and_residual_component_modes():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'choices=("edge-average", "component-lock", "component-parent-lock")' in text
    assert 'if mode == "component-lock"' in text
    assert 'if mode == "component-parent-lock"' in text
    assert "lowest_common_ancestor_name" in text
    assert 'mode_details["hierarchical_parent_lock"] = True' in text
    assert '"--component-rings"' in text
    assert 'mode_details["component_rings"] = int(component_rings)' in text
    assert 'if mode != "edge-average"' in text
    assert "top_k_normalize" in text


def test_repair_uses_manual_front_axis_not_direction_inference():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--front-axis"' in text
    assert "infer_quadruped_semantics" in text
    assert "automatic_direction" not in text
