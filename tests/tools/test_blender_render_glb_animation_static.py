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
