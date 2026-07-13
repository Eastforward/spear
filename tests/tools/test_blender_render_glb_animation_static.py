"""Static contracts for the Blender Walk/Idle review renderer."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2] / "tools/blender_render_glb_animation.py"
)


def test_animation_renderer_exposes_bounded_camera_distance():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--camera-distance-multiplier" in text
    assert "0.75 <= args.camera_distance_multiplier <= 4.0" in text
    assert "radius = diag * args.camera_distance_multiplier" in text


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
    assert "cam_data.ortho_scale = diag * 1.05" in text


def test_animation_renderer_can_expose_far_limbs_without_camera_yaw():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--quadruped-far-limb-offset-ratio" in text
    assert "def apply_quadruped_far_limb_offset" in text
    assert '"Bone.017": -offset' in text
    assert '"Bone.008": offset' in text
    assert "torso_transform=identity camera_yaw_deg=0 ground_delta=0" in text


def test_animation_renderer_can_replace_faceted_source_material_with_smooth_clay():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--pose-template-clay-color" in text
    assert "def apply_pose_template_clay_material" in text
    assert "body.data.materials.clear()" in text
    assert "polygon.use_smooth = True" in text
    assert "template_material=uniform_clay" in text
