from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_blender_combine_glb_actions_script_exposes_human_action_args():
    text = (REPO / "tools" / "blender_combine_glb_actions.py").read_text()

    assert "--base-glb" in text
    assert "--append-glb" in text
    assert "--base-action-name" in text
    assert "--append-action-name" in text
    assert "export_extra_animations=True" in text
