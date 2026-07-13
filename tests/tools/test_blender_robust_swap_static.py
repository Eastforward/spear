from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "blender_robust_swap_mesh_keep_rig.py"


def test_blender_robust_swap_exposes_nearest_weight_mode():
    text = SCRIPT.read_text()

    assert 'choices=["region", "auto", "nearest"]' in text
    assert "transfer_weights_by_nearest_surface" in text
    assert 'args.weight_mode == "nearest"' in text


def test_blender_robust_swap_exposes_target_yaw_rotation():
    text = SCRIPT.read_text()

    assert "--target-rotate-z-deg" in text
    assert "rotate_target_z_degrees" in text
    assert "args.target_rotate_z_deg" in text
    assert "Matrix.Rotation" in text
    assert ".data.transform(" in text


def test_blender_robust_swap_supports_explicit_animal_forward_axes():
    text = SCRIPT.read_text()

    assert "--semantic-forward-axis" in text
    assert 'choices=["positive-x", "negative-y"]' in text
    assert 'forward_axis="positive-x"' in text
    assert "-vertices[:, 1]" in text
    assert "args.semantic_forward_axis" in text


def test_blender_robust_swap_can_export_only_canonical_walk_and_idle():
    text = SCRIPT.read_text()

    assert "--export-action-policy" in text
    assert 'choices=["all", "walk-idle"]' in text
    assert "keep_canonical_walk_idle_actions" in text
    assert 'idle.name = "Idle"' in text
    assert 'walking.name = "Walking"' in text
    assert 'gltf["animations"] = [idle, walking]' in text


def test_blender_robust_swap_keeps_approved_animal_dampening_defaults():
    text = SCRIPT.read_text()

    assert 'p.add_argument("--dampen-foot-rotations", type=float, default=1.0' in text
    assert 'p.add_argument("--dampen-head-rotations", type=float, default=0.0' in text
    assert 'p.add_argument("--dampen-tail-rotations", type=float, default=0.0' in text
