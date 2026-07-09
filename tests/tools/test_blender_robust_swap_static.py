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
