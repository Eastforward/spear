from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/spike_rlr/generated_animal_motion_basis_review_server.py"
)


def test_server_exposes_pre_animation_gate_and_immutable_decision():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "generated_animal_motion_basis_manual_decision_v1" in text
    assert '"target_animation_generation_authorized"' in text
    assert "open(\"x\"" in text
    assert "motion_basis_approved" in text
    assert "motion_basis_rejected" in text
    assert "preview_sha256" in text
    assert "candidate_id" in text
    assert "manual_cardinal" in text


def test_ui_is_live_skeleton_overlay_before_mesh_animation():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "绑定前动作方向人工审核" in text
    assert "尚未生成目标蒙皮动画" in text
    assert "requestAnimationFrame" in text
    assert "目标模型 +X" in text
    assert "源 Walk forward" in text
    assert "screenArrow" in text
    assert "FORWARD +X" in text
    assert "UP +Z" in text
    assert "BACK / -X" in text
    assert "Hunyuan style direction reference" in text
    assert "目标正方向 +X" in text
    assert "世界向上 +Z" in text
    assert "黄色箭头必须与绿色箭头同向" in text
    assert "sourceArrow" in text
    assert "左右腿链" in text
    assert "整90°" in text
