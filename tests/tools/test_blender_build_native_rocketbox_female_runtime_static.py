from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[2]
    / "tools"
    / "blender_build_native_rocketbox_female_runtime.py"
)


def test_female_native_wrapper_is_syntax_valid_and_hash_locked():
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    assert "rocketbox_female_adult_01" in source
    assert "dd2b174301b31468511c4c49c1ea53daf3bb53148220a9e026dc5511fac4d3be" in source
    assert "b678cc011c27a42d3a7833c0529af401533a0c68cc2c63448ab0367d3359048e" in source
    assert 'EXPECTED_WALK_RANGE = (1, 38)' in source
    assert 'EXPECTED_IDLE_RANGE = (1, 467)' in source


def test_female_native_wrapper_uses_female_idle_and_all_seven_pbr_images():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "f_idle_neutral_01.max.fbx" in source
    for suffix in (
        "body_color",
        "body_normal",
        "body_specular",
        "head_color",
        "head_normal",
        "head_specular",
        "opacity_color",
    ):
        assert f"f001_{suffix}" in source
    assert "head_normal_wrinkle" not in source


def test_female_native_wrapper_delegates_immutable_build_not_source_editing():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "TOOLS_DIR" in source and "sys.path.insert" in source
    assert "native.main" in source
    assert "native.ASSET_ID" in source
    assert "bpy.ops.wm.save" not in source
    assert "transform_apply" not in source
