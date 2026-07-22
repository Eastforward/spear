"""Static contract for the post-TokenRig target-native review runner."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/run_target_native_generated_quadruped_review.py"
)


def test_runner_preserves_the_required_stage_order_and_review_views():
    text = SCRIPT.read_text(encoding="utf-8")
    command_builder = text[text.index("def build_commands(") :]

    expected = [
        '"heading"',
        '"rig_audit"',
        '"support_plane"',
        '"retarget"',
        '"deformation"',
    ]
    positions = [command_builder.index(item) for item in expected]
    assert positions == sorted(positions)
    for label in (
        "walking_side",
        "walking_front",
        "walking_rear",
        "idle_side",
        "idle_front",
        "idle_rear",
    ):
        assert label in text


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
