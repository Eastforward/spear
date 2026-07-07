"""Tests for tools/spike_rlr/profiling.py."""
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from profiling import StageTimer, print_stage_summary, reset_aggregate  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    reset_aggregate()
    yield
    reset_aggregate()


def test_stage_timer_records_seconds(tmp_path):
    csv = tmp_path / "prof.csv"
    with StageTimer("scene_gen", clip_id="clip_000", csv_path=csv):
        time.sleep(0.02)
    contents = csv.read_text().strip().splitlines()
    # Header + 1 row
    assert len(contents) == 2
    hdr = contents[0].split(",")
    assert hdr == ["clip_id", "stage", "seconds", "retry_count", "flags_json"]
    row = contents[1].split(",")
    assert row[0] == "clip_000"
    assert row[1] == "scene_gen"
    assert 0.015 < float(row[2]) < 0.5
    assert row[3] == "0"


def test_aggregate_sums_across_clips(tmp_path):
    csv = tmp_path / "prof.csv"
    for cid in ["clip_000", "clip_001", "clip_002"]:
        with StageTimer("ue_render", clip_id=cid, csv_path=csv):
            time.sleep(0.01)
    assert 0.025 < StageTimer.aggregate["ue_render"] < 0.5


def test_print_summary_formats_output(tmp_path):
    csv = tmp_path / "prof.csv"
    with StageTimer("scene_gen", clip_id="c0", csv_path=csv):
        time.sleep(0.01)
    with StageTimer("ue_render", clip_id="c0", csv_path=csv):
        time.sleep(0.02)
    out_path = tmp_path / "summary.txt"
    txt = print_stage_summary(total_clips=1, out_path=out_path)
    assert "scene_gen" in txt
    assert "ue_render" in txt
    assert "TOTAL" in txt
    assert out_path.read_text() == txt


def test_flags_json_field_persisted(tmp_path):
    import csv as csvmod
    csv = tmp_path / "prof.csv"
    with StageTimer("scene_gen", clip_id="c0", csv_path=csv,
                    flags=["occluded_by_furniture", "steady_walk"]):
        time.sleep(0.005)
    with csv.open() as f:
        rows = list(csvmod.DictReader(f))
    assert len(rows) == 1
    assert json.loads(rows[0]["flags_json"]) == ["occluded_by_furniture", "steady_walk"]


def test_reset_aggregate_clears_state(tmp_path):
    csv = tmp_path / "prof.csv"
    with StageTimer("foo", clip_id="c0", csv_path=csv):
        time.sleep(0.001)
    assert "foo" in StageTimer.aggregate
    reset_aggregate()
    assert StageTimer.aggregate == {}


def test_stage_timer_exception_still_records(tmp_path):
    csv = tmp_path / "prof.csv"
    with pytest.raises(RuntimeError):
        with StageTimer("failing_stage", clip_id="c0", csv_path=csv):
            time.sleep(0.005)
            raise RuntimeError("simulated failure")
    # Aggregate and CSV should still be updated even after exception
    assert "failing_stage" in StageTimer.aggregate
    assert csv.exists()
    row = csv.read_text().strip().splitlines()[1]
    assert row.split(",")[1] == "failing_stage"
