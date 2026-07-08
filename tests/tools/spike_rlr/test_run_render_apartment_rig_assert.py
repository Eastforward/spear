"""Ensure the rig-direction assertion is invocable and env-gated correctly.

Full end-to-end test (spawn actor in UE and observe walking) is manual —
run `SPEAR_RIG_ASSERT=1 python tools/spike_rlr/run_render_pass_apartment.py`
and check the log for '[apt_render] rig direction check ...'.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def test_run_render_apartment_help_shows_rig_assert_flag():
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python",
         str(REPO / "tools" / "spike_rlr" / "run_render_pass_apartment.py"),
         "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stderr={r.stderr}"
    # We add a --rig-assert flag (CLI convenience; env var also works)
    assert "--rig-assert" in r.stdout or "rig direction" in r.stdout.lower()


def test_rig_assert_env_var_recognized():
    """The script must at least import when SPEAR_RIG_ASSERT=1."""
    env = dict(os.environ)
    env["SPEAR_RIG_ASSERT"] = "1"
    # Just check --help still works with env var set
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python",
         str(REPO / "tools" / "spike_rlr" / "run_render_pass_apartment.py"),
         "--help"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, f"stderr={r.stderr}"


def test_review_gate_refuses_unapproved_tag(tmp_path):
    """When SPEAR_SKIP_REVIEW_GATE not set, an unapproved tag should raise
    MeshNotApprovedError before UE spawn."""
    import json
    # Minimal spec pointing to a made-up tag that definitely isn't approved
    spec = {
        "spec_version": "apartment_v1",
        "room_backend": "apartment_shell",
        "apartment_shell_map": "data/apartment_shell_map.json",
        "apartment_furniture_map": "data/apartment_furniture_map.json",
        "furniture_mode": "subset",
        "furniture_include_categories": ["core"],
        "furniture_include_actors_extra": [],
        "furniture_exclude_actors": [],
        "mic": {"pos_m": [0, 0, 1.2], "yaw_deg": 0, "forward": [1, 0, 0],
                "type_rlr": "binaural_native"},
        "camera_configs": [{"name": "v0", "pos_m": [0, 0, 1.2],
                             "yaw_deg": 0, "fov_deg": 90.0}],
        "render_config": {"width": 320, "height": 240, "fps": 15,
                           "n_frames": 15, "duration_s": 1.0},
        "audio_config": {"sample_rate_hz": 16000, "duration_s": 1.0,
                          "n_samples": 16000, "output_channels": 2},
        "source_height_m": 0.45,
        "sources": [{"tag": "nonexistent_tag_zzz", "audio_lookup": "dog_bark",
                     "kind": "moving", "start_pos_m": [1, 1, 0.45],
                     "end_pos_m": [2, 1, 0.45], "motion": "linear_uniform",
                     "wanted_anim": "Walking"}],
    }
    spec_path = tmp_path / "bad_spec.json"
    spec_path.write_text(json.dumps(spec))

    env = dict(os.environ)
    env.pop("SPEAR_SKIP_REVIEW_GATE", None)
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python",
         str(REPO / "tools/spike_rlr/run_render_pass_apartment.py"),
         "--spec", str(spec_path),
         "--out-dir", str(tmp_path / "out"),
         "--clip-id", "test_gate"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert r.returncode != 0
    # Expect the review_gate REFUSED marker to appear in stdout or stderr
    combined = r.stdout + r.stderr
    assert "MeshNotApprovedError" in combined or "review_gate REFUSED" in combined
