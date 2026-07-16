import pytest

from tools.audit_pipeline_timings import summarize


def test_timing_summary_interpolates_p95_and_preserves_samples():
    summary = summarize([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["median_seconds"] == pytest.approx(2.5)
    assert summary["p95_seconds"] == pytest.approx(3.85)
    assert summary["values_seconds"] == [1.0, 2.0, 3.0, 4.0]
