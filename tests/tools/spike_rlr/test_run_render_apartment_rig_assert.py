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
