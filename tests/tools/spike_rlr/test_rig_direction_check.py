"""Unit tests for rig_direction_check.py (offline path only).

Full integration tests that spawn a real SPEAR actor are in Task 8's
run_render_pass integration.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_write_and_read_calibration_roundtrip(tmp_path, monkeypatch):
    from rig_direction_check import (
        write_rig_calibration_json, read_rig_calibration_json,
    )
    calib_path = tmp_path / "rig_calibration.json"
    monkeypatch.setattr("rig_direction_check.CALIBRATION_FILE", calib_path)

    write_rig_calibration_json("dog_golden", offset_deg=180.0,
                                algorithm_version="rig_calib_v1")
    got = read_rig_calibration_json("dog_golden")
    assert got is not None
    assert got["walking_forward_yaw_offset_deg"] == 180.0
    assert got["algorithm_version"] == "rig_calib_v1"

    # Second write for a different tag preserves the first
    write_rig_calibration_json("dog_husky", offset_deg=170.0,
                                algorithm_version="rig_calib_v1")
    assert read_rig_calibration_json("dog_golden")["walking_forward_yaw_offset_deg"] == 180.0
    assert read_rig_calibration_json("dog_husky")["walking_forward_yaw_offset_deg"] == 170.0


def test_yaw_difference_within_tolerance():
    from rig_direction_check import _yaw_difference_deg
    assert abs(_yaw_difference_deg(10.0, 15.0)) == pytest.approx(5.0, abs=0.01)
    assert abs(_yaw_difference_deg(-170.0, 170.0)) == pytest.approx(20.0, abs=0.01)  # wrap
    assert abs(_yaw_difference_deg(0.0, 359.0)) == pytest.approx(1.0, abs=0.01)  # wrap
    assert abs(_yaw_difference_deg(45.0, 45.0)) == pytest.approx(0.0, abs=0.01)


def test_assert_yaw_ok_within_tolerance():
    from rig_direction_check import _assert_yaw_ok
    _assert_yaw_ok(observed=10.0, expected=15.0, tolerance_deg=15.0, context="test")
    # Should not raise


def test_assert_yaw_ok_raises_outside_tolerance():
    from rig_direction_check import _assert_yaw_ok
    with pytest.raises(AssertionError, match="test"):
        _assert_yaw_ok(observed=10.0, expected=90.0, tolerance_deg=15.0,
                        context="test")


def test_read_nonexistent_calibration_returns_none(tmp_path, monkeypatch):
    from rig_direction_check import read_rig_calibration_json
    calib_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr("rig_direction_check.CALIBRATION_FILE", calib_path)
    assert read_rig_calibration_json("anytag") is None
