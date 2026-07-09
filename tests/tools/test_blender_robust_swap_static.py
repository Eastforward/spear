from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "blender_robust_swap_mesh_keep_rig.py"


def test_blender_robust_swap_exposes_nearest_weight_mode():
    text = SCRIPT.read_text()

    assert 'choices=["region", "auto", "nearest"]' in text
    assert "transfer_weights_by_nearest_surface" in text
    assert 'args.weight_mode == "nearest"' in text
