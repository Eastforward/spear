from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "blender_build_prebind_gait_review.py"


def _module():
    spec = importlib.util.spec_from_file_location("prebind_gait_builder", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics():
    leg = {
        "foot_forward_excursion_m": 0.7,
        "foot_lateral_excursion_m": 0.1,
        "lateral_to_forward_excursion_ratio": 0.14,
        "mean_knee_normal_dot_lateral_abs": 0.99,
        "mean_knee_normal_dot_forward_abs": 0.08,
    }
    target_leg = {
        **leg,
        "foot_forward_excursion_m": 0.28,
        "foot_lateral_excursion_m": 0.24,
        "lateral_to_forward_excursion_ratio": 0.86,
        "mean_knee_normal_dot_lateral_abs": 0.34,
        "mean_knee_normal_dot_forward_abs": 0.81,
    }
    return {
        "schema": "prebind_vs_second_retarget_gait_plane_v1",
        "source_prebind": {
            "frame_count": 33,
            "overall_classification": "sagittal_forward_gait",
            "legs": {"left": leg, "right": leg},
        },
        "target_second_retarget": {
            "frame_count": 33,
            "overall_classification": "sideways_leg_swing",
            "legs": {"left": target_leg, "right": target_leg},
        },
    }


def test_module_imports_without_blender_and_samples_prebind_before_target():
    module = _module()
    assert module.FPS == 30
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import bpy" not in "\n".join(source.splitlines()[:40])
    assert source.index("bpy.ops.wm.open_mainfile") < source.index("bpy.ops.import_scene.gltf")
    for token in (
        "authenticate_sealed_walk",
        "authenticate_second_attempt",
        "validate_facing_bundle",
        "compute_gait_plane_samples",
        "ROCKETBOX_ROLE_TO_BONE",
        'target_semantic["left_calf"]',
        'target_semantic["right_calf"]',
    ):
        assert token in source
    assert "second_retarget_facing_review_server" not in source
    assert "from second_retarget_facing_review import" in source


def test_page_shows_prebind_and_postbind_media_and_leg_plane_metrics():
    html = _module().build_prebind_html(_metrics()).decode("utf-8")
    for route in (
        "/source/front",
        "/source/side",
        "/source/top",
        "/source/source_target",
        "/target/front",
        "/target/side",
        "/target/feet",
        "/target/top",
    ):
        assert route in html
    for text in (
        "绑定前 Rocketbox 源动作",
        "第二次 TokenRig 绑定后",
        "脚横向/前后摆幅比",
        "膝平面法向·身体横轴",
        "膝平面法向·身体前向",
        "retarget_introduced_sideways_plane",
        "source_animation_wrong",
        "target_bind_basis_wrong",
        "人工判断优先",
        "currentTime",
        "0.5 / FPS",
    ):
        assert text in html
    assert "Approve" not in html
    assert "<form" not in html


def test_page_rejects_wrong_stage_classifications():
    module = _module()
    metrics = _metrics()
    metrics["source_prebind"]["overall_classification"] = "sideways_leg_swing"
    try:
        module.build_prebind_html(metrics)
    except Exception as error:
        assert "source prebind gait" in str(error)
    else:
        raise AssertionError("wrong source gait classification was accepted")
