from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "blender_render_second_retarget_facing.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("second_facing_renderer", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_renderer_module_imports_without_blender_and_pins_environment():
    module = _module()
    assert module.FPS == 30
    assert module.FRAME_COUNT == 33
    assert module.VIDEO_SIZE == (640, 360)
    assert module.validate_blender_version((4, 2, 1)) == "4.2.1"
    with pytest.raises(Exception, match="exactly 4.2.1"):
        module.validate_blender_version((4, 3, 0))
    assert "import bpy" not in "\n".join(SCRIPT.read_text().splitlines()[:40])


def test_renderer_samples_body_basis_and_root_independently():
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        'semantic_bones["left_clavicle"]',
        'semantic_bones["right_clavicle"]',
        'semantic_bones["left_thigh"]',
        'semantic_bones["right_thigh"]',
        'semantic_bones["pelvis"]',
        "compute_facing_samples",
        "armature.matrix_world @ pose_bone.head",
    ):
        assert token in source
    assert source.index("scene.render.fps = FPS") < source.index(
        "bpy.ops.import_scene.gltf"
    )


def test_arrow_keyframes_use_three_independent_directions_and_undefined_hide():
    source = inspect.getsource(_module()._keyframe_direction_arrows)
    assert 'sample["body_forward"]' in source
    assert 'sample["travel_direction"]' in source
    assert "CANONICAL_FRONT" in source
    assert 'travel.hide_render = sample["travel_direction"] is None' in source
    assert 'keyframe_insert(data_path="hide_render"' in source


def test_aggregate_bounds_hold_the_whole_path_in_one_fixed_top_camera():
    module = _module()
    bounds = [
        {"minimum": (-0.2, -0.3, 0.0), "maximum": (0.2, 0.3, 1.8)},
        {"minimum": (-0.1, -1.3, -0.02), "maximum": (0.3, -0.7, 1.82)},
    ]
    aggregate = module.aggregate_top_bounds(bounds, padding=0.2)
    assert aggregate["minimum"] == pytest.approx((-0.4, -1.5, -0.22))
    assert aggregate["maximum"] == pytest.approx((0.5, 0.5, 2.02))
    assert aggregate["center"] == pytest.approx((0.05, -0.5, 0.9))


def test_cli_requires_immutable_source_and_no_replace_output():
    module = _module()
    args = module.parse_args(
        [
            "--diagnostic-dir",
            "/tmp/source",
            "--output-dir",
            "/tmp/output",
        ]
    )
    assert args.diagnostic_dir == Path("/tmp/source")
    assert args.output_dir == Path("/tmp/output")


def test_renderer_publishes_hash_locked_review_html_in_same_atomic_bundle():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "build_review_html" in source
    assert '_write_bytes_exclusive(staging / "review.html"' in source
    assert '"review.html": _record(' in source
    assert '"review.html",' in source[source.index("expected = {") :]
