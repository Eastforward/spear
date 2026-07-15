"""Static contracts for the Blender Walk/Idle review renderer."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2] / "tools/blender_render_glb_animation.py"
)


def test_animation_renderer_exposes_bounded_camera_distance():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--camera-distance-multiplier" in text
    assert "0.75 <= args.camera_distance_multiplier <= 4.0" in text
    assert "radius = framing_diag * args.camera_distance_multiplier" in text


def test_animation_renderer_keeps_canonical_actions_and_review_views():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'p.add_argument("--action", default="Walking")' in text
    assert 'choices=["side", "front", "quarter"]' in text
    assert "armature.animation_data.action = action" in text


def test_animation_renderer_can_show_a_rest_pose_contact_floor():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--ground-plane" in text
    assert "def add_review_ground" in text
    assert 'ground.name = "FootContactReviewGround"' in text
    assert "add_review_ground(center, mn[2], diag)" in text


def test_animation_renderer_can_render_the_authored_armature_rest_pose():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--rest-pose" in text
    assert 'armature.data.pose_position = "REST"' in text
    assert 'else "authored_rest_pose"' in text


def test_animation_renderer_can_make_an_exact_orthographic_pose_template():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--orthographic" in text
    assert 'cam_data.type = "ORTHO"' in text
    assert "cam_data.ortho_scale = framing_diag * 1.05" in text


def test_animation_renderer_can_hold_instance_size_on_one_visual_scale():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--camera-reference-diagonal"' in text
    assert "args.camera_reference_diagonal or diag" in text
    assert "--camera-reference-diagonal requires --orthographic" in text
    assert "camera_reference_diagonal={framing_diag:.3f}" in text


def test_animation_renderer_can_make_a_bounded_depth_disambiguating_pose_template():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--pose-template-yaw-deg"' in text
    assert "-30.0 <= args.pose_template_yaw_deg <= 30.0" in text
    assert "--pose-template-yaw-deg requires --rest-pose" in text
    assert "--pose-template-yaw-deg requires --orthographic" in text
    assert "def apply_pose_template_yaw" in text
    assert "pose_template_yaw_deg=" in text
    assert "automatic_direction_inference=false" in text


def test_animation_renderer_can_expose_far_limbs_without_camera_yaw():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--quadruped-far-limb-offset-ratio" in text
    assert "def apply_quadruped_far_limb_offset" in text
    assert "infer_quadruped_semantics" in text
    assert "def weighted_skin_bone_names" in text
    assert "include_names = weighted_skin_bone_names(body, armature)" in text
    assert "ignored_detached_control_roots" in text
    assert "children_recursive" in text
    assert "terminal_markers" in text
    assert "if not child.children" in text
    assert "semantics.front_side_positive[0]" in text
    assert "semantics.hind_side_positive[0]" in text
    assert "world_to_armature" in text
    assert "semantic_front_axis=world_positive_x" in text
    assert "torso_transform=identity camera_yaw_deg=0 ground_delta=0" in text
    assert "MAX_QUADRUPED_FAR_LIMB_OFFSET_RATIO = 0.35" in text


def test_animation_renderer_can_compose_grounded_native_action_limb_poses():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--quadruped-far-limb-action-pose-ratio" in text
    assert "--quadruped-pose-action" in text
    assert "def apply_quadruped_far_limb_action_pose" in text
    assert "native_action=" in text
    assert "chain_root_translation=false" in text
    assert "maximum_ground_delta_ratio=" in text
    assert "choose only one quadruped far-limb pose mode" in text


def test_animation_renderer_can_replace_faceted_source_material_with_smooth_clay():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--pose-template-clay-color" in text
    assert "def apply_pose_template_clay_material" in text
    assert "body.data.materials.clear()" in text
    assert "polygon.use_smooth = True" in text
    assert "template_material=uniform_clay" in text


def test_animation_renderer_has_non_mutating_clay_override_for_animation_qa():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--review-clay-color" in text
    assert "if args.review_clay_color" in text
    assert "apply_pose_template_clay_material(body, args.review_clay_color)" in text
    assert "bpy.ops.export_scene" not in text


def test_animation_renderer_can_make_manual_cardinal_walk_candidates():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--asset-yaw-deg"' in text
    assert '"--trajectory-distance-ratio"' in text
    assert "apply_asset_cardinal_yaw" in text
    assert "trajectory_base_location" in text
    assert "world_positive_x" in text
    assert "asset_yaw_deg must be one of" in text


def test_animation_renderer_uses_configurable_low_latency_review_samples():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'p.add_argument("--samples", type=int, default=16)' in text
    assert "1 <= args.samples <= 256" in text
    assert "scene.eevee.taa_render_samples = args.samples" in text
    assert "scene.cycles.samples = args.samples" in text


def test_animation_renderer_has_non_mutating_preserve_volume_diagnostic():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--preserve-volume"' in text
    assert "modifier.use_deform_preserve_volume = True" in text
    assert "purpose=render_only_diagnostic" in text
    assert "input_asset_unchanged=true" in text
