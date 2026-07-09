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

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from dataset_runner import _load_obstacle_context  # noqa: E402
from flag_verifier import verify_all_flags  # noqa: E402


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

    dataset_spec = json.loads((REPO / "data/apartment_v2_m1_dataset_spec.json").read_text())
    obstacle_ctx = _load_obstacle_context(dataset_spec)
    valid_regions = obstacle_ctx["valid_regions"]
    for d in clip_dirs:
        spec = json.loads((d / "spec.json").read_text())
        npz = np.load(d / "trajectories.npz")
        mic = spec["mic"]["pos_m"]
        assert any(x0 <= mic[0] <= x1 and y0 <= mic[1] <= y1
                   for x0, y0, x1, y1 in valid_regions)
        trajectories = []
        for i, src in enumerate(spec["sources"]):
            traj = np.asarray(src["trajectory_m"])
            assert np.allclose(traj, npz[f"arr_{i}"])
            for x, y in traj[:, :2]:
                assert any(x0 <= x <= x1 and y0 <= y <= y1
                           for x0, y0, x1, y1 in valid_regions)
            trajectories.append(traj)
        recomputed = verify_all_flags(
            spec, trajectories,
            obstacle_ctx["furniture_bboxes"], obstacle_ctx["wall_bboxes"],
        )
        assert recomputed == json.loads((d / "flags.json").read_text())


def test_review_mic_height_override_no_render(tmp_path):
    out = tmp_path / "height_review"
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python",
         str(REPO / "tools/spike_rlr/dataset_runner.py"),
         "--dataset-spec", str(REPO / "data/apartment_v2_m1_dataset_spec.json"),
         "--n-clips", "3",
         "--out-dir", str(out),
         "--skip-render",
         "--review-mic-height-range", "1.2", "1.4"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"height override failed:\n{r.stdout}\n{r.stderr}"

    for spec_path in sorted((out / "clips").glob("clip_*/spec.json")):
        spec = json.loads(spec_path.read_text())
        assert 1.2 <= spec["mic"]["pos_m"][2] <= 1.4
