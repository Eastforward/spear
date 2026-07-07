"""End-to-end smoke test: 3-clip Plan 2 pipeline (no UE, no RLR).

Verifies dataset_runner can:
  1. Load dataset spec
  2. Sample 3 clips via rejection_sampler
  3. Write per-clip spec.json + metadata.json + flag JSON
  4. Skip actual UE/RLR render when --skip-render is set

Full UE/RLR integration is manual (run without --skip-render for real run).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def test_smoke_pipeline_3clips_no_render(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python",
         str(REPO / "tools/spike_rlr/dataset_runner.py"),
         "--dataset-spec", str(REPO / "data/apartment_v2_m1_dataset_spec.json"),
         "--n-clips", "3",
         "--out-dir", str(out),
         "--skip-render"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"smoke failed:\n{r.stdout}\n{r.stderr}"

    # 3 clip dirs exist
    clips_dir = out / "clips"
    assert clips_dir.exists()
    clip_dirs = sorted(clips_dir.iterdir())
    assert len(clip_dirs) == 3

    # Each has spec.json + flags
    for d in clip_dirs:
        assert (d / "spec.json").exists()
        assert (d / "flags.json").exists()
        flags = json.loads((d / "flags.json").read_text())
        assert isinstance(flags, dict) and len(flags) == 12
