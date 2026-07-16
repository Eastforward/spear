from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/route2_controlled_geometry_pixal_preflight_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "route2_controlled_geometry_pixal_preflight_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_fixed_model_runtime_and_single_reviewed_parameter_set():
    assert runner.PIXAL_PYTHON == Path(
        "/data/jzy/miniconda3/envs/avengine-3dgen/bin/python3.10"
    )
    assert runner.PIXAL_REVISION == "0b31f9160aa400719af409098bff7936a932f726"
    assert runner.PARAMETERS == {
        "seed": 42,
        "manual_fov": 0.2,
        "resolution": 1024,
        "low_vram": True,
    }
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"--manual-fov",\n        "0.2"' in source
    assert '"0.35"' not in source
    assert '"0.5"' not in source
    assert "Hunyuan" not in source


def test_live_source_jobs_are_eight_agent_accepted_controlled_references():
    payload, jobs = runner._source_jobs()
    assert payload["formal_registration_authorized"] is False
    assert len(jobs) == 8
    assert [job["asset_id"] for job in jobs] == [
        f"route2_v3_{case_id}" for case_id in runner.geometry.CASE_BY_ID
    ]
    assert all(job["parameters"] == runner.PARAMETERS for job in jobs)
    assert all(job["state_classification"] == "research_candidate" for job in jobs)


def test_command_is_exact_and_gpu_is_explicit(tmp_path):
    _, jobs = runner._source_jobs()
    command = runner._command(jobs[0], tmp_path / "candidate.glb", "2")
    assert command[0] == str(runner.PIXAL_PYTHON)
    assert command[1] == str(runner.PIXAL_WRAPPER)
    assert command.count("pixal3d") == 1
    assert command[command.index("--gpu") + 1] == "2"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--resolution") + 1] == "1024"
    assert command[command.index("--manual-fov") + 1] == "0.2"
    assert command.count("--low-vram") == 1


def test_cli_rejects_implicit_gpu_and_unknown_asset():
    with pytest.raises(SystemExit):
        runner.parse_args(["run", "--asset-id", "route2_v3_male_long_sleeve"])
    with pytest.raises(SystemExit):
        runner.parse_args(["run", "--asset-id", "unknown", "--gpu", "0"])
    args = runner.parse_args(
        ["run", "--asset-id", "route2_v3_female_shorts", "--gpu", "3"]
    )
    assert args.gpu == "3"


def test_wrapper_hash_is_exact_current_generator():
    assert runner.sha256_file(runner.PIXAL_WRAPPER) == runner.PIXAL_WRAPPER_SHA256
