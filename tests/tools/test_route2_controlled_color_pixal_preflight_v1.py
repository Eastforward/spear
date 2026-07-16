from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/route2_controlled_color_pixal_preflight_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "route2_controlled_color_pixal_preflight_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_only_six_visually_passed_color_jobs_are_executable():
    assert runner.asset_ids() == (
        "route2_color_v3_male_top_cobalt",
        "route2_color_v3_male_trousers_navy",
        "route2_color_v3_female_trousers_khaki",
        "route2_color_v3_male_shoes_black",
        "route2_color_v3_female_shoes_brown",
        "route2_color_v3_male_hair_auburn",
    )
    assert all("female_top_teal" not in value for value in runner.asset_ids())
    assert all("female_hair_chestnut" not in value for value in runner.asset_ids())


def test_delegated_command_keeps_single_reviewed_pixal_parameter_set():
    _, jobs = runner.source_jobs()
    command = runner.executor._command(jobs[0], Path("/tmp/output.glb"), "2")
    assert command[command.index("--manual-fov") + 1] == "0.2"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--resolution") + 1] == "1024"
    assert command.count("--low-vram") == 1
    assert "0.35" not in command and "0.5" not in command


def test_wrapper_uses_new_root_and_does_not_modify_geometry_executor():
    assert runner.OUTPUT_ROOT == runner.color.PIXAL_OUTPUT_ROOT
    assert runner.OUTPUT_ROOT != runner.executor.SPEAR_ROOT / "tmp/i23d_controlled_geometry_v3/pixal3d"
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"geometry_attribute": f"color_only:{job[\'attribute\']}"' in source
    assert "Hunyuan" not in source
