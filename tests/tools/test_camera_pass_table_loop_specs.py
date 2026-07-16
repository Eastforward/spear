import sys
from pathlib import Path

import numpy as np


SPEAR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPEAR_ROOT / "tools"))

from build_camera_pass_table_loop_apartment_specs import (
    build_camera_pass_table_loop_trajectory,
)


def test_camera_relative_start_pass_and_loop_contract():
    trajectory, contract = build_camera_pass_table_loop_trajectory(
        n_frames=270,
        camera_pos_m=(0.5, 0.15, 1.2),
        camera_yaw_deg=145.0,
    )

    assert trajectory.shape == (270, 3)
    assert contract["requested_start_components_m"]["right"] == pytest.approx(0.8)
    assert contract["requested_start_components_m"]["rear"] == pytest.approx(3.2)
    assert contract["requested_left_front_components_m"]["left"] == pytest.approx(0.8)
    assert contract["requested_left_front_components_m"]["front"] == pytest.approx(2.0)
    assert contract["table_loop_turns"] == 1.0
    assert contract["left_front_nearest_frame"] < contract["table_loop_entry_nearest_frame"]
    assert 90 <= contract["table_loop_entry_nearest_frame"] <= 130
    assert np.allclose(trajectory[-1, :2], contract["table_loop_entry_xy_m"], atol=1e-6)


import pytest
