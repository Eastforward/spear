"""Static contract for the post-TokenRig target-native review runner."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/run_target_native_generated_quadruped_review.py"
)


def test_runner_preserves_the_required_stage_order_and_review_views():
    text = SCRIPT.read_text(encoding="utf-8")
    command_builder = text[text.index("def build_initial_commands(") :]

    expected = [
        '"heading"',
        '"rig_audit"',
        '"support_plane"',
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
    assert 'default="auto"' in text
    assert '"--repair-mode", "component-parent-lock"' in text
    assert '"--component-rings", "4"' in text
    assert '"--extension-threshold", "0.02"' in text
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
