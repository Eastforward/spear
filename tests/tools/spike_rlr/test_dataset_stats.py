import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
STATS_TOOL = REPO / "tools/spike_rlr/dataset_stats.py"


def _write_synth_clip(out_dir, clip_index, flags_true):
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from flag_definitions import ALL_FLAGS
    clip_dir = out_dir / "clips" / f"clip_{clip_index:04d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    flags = {f: (f in flags_true) for f in ALL_FLAGS}
    (clip_dir / "flags.json").write_text(json.dumps(flags))


def test_dataset_stats_aggregates_flag_counts(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "clips").mkdir()
    (out / "analysis").mkdir()
    _write_synth_clip(out, 0, {"occluded_by_furniture", "stays_in_camera_fov"})
    _write_synth_clip(out, 1, {"occluded_by_furniture", "stationary"})
    _write_synth_clip(out, 2, {"steady_walk"})

    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python", str(STATS_TOOL),
         "--out-dir", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    stats = json.loads((out / "analysis" / "dataset_stats.json").read_text())
    assert stats["n_clips"] == 3
    assert stats["flag_coverage"]["occluded_by_furniture"] == 2
    assert stats["flag_coverage"]["stationary"] == 1
    assert stats["flag_coverage"]["steady_walk"] == 1


def test_dataset_stats_chart_generation(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "clips").mkdir()
    (out / "analysis").mkdir()
    _write_synth_clip(out, 0, {"steady_walk"})

    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python", str(STATS_TOOL),
         "--out-dir", str(out), "--generate-charts"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "analysis" / "coverage_bar.png").exists()
