import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from flag_verifier import verify_all_flags, verify_flag_details, set_flags  # noqa: E402
from flag_definitions import ALL_FLAGS  # noqa: E402


def _stub_spec(mic_pos=(0, 0, 1.2), mic_yaw=0, fov_h=90, fov_v=60, fps=15):
    return {
        "mic": {"pos_m": list(mic_pos), "yaw_deg": mic_yaw},
        "camera_configs": [{"fov_deg": fov_h, "fov_v_deg": fov_v}],
        "render_config": {"fps": fps},
    }


def test_verify_returns_all_12_flags_for_1_source():
    spec = _stub_spec()
    traj = np.linspace([3, 0, 1.2], [3, 3, 1.2], num=30)
    result = verify_all_flags(
        spec_dict=spec,
        trajectories=[traj],
        furniture_bboxes=[], wall_bboxes=[],
    )
    assert set(result.keys()) == set(ALL_FLAGS)
    assert result["sources_pass_each_other"] is False


def test_verify_returns_all_12_flags_for_0_source():
    spec = _stub_spec()
    result = verify_all_flags(
        spec_dict=spec, trajectories=[],
        furniture_bboxes=[], wall_bboxes=[],
    )
    assert set(result.keys()) == set(ALL_FLAGS)
    for name in ALL_FLAGS:
        assert result[name] is False


def test_verify_for_2_sources_evaluates_multi_source_flag():
    spec = _stub_spec()
    t1 = np.linspace([-3, 0, 1.2], [3, 0, 1.2], num=30)
    t2 = np.linspace([3, 0, 1.2], [-3, 0, 1.2], num=30)
    result = verify_all_flags(
        spec_dict=spec, trajectories=[t1, t2],
        furniture_bboxes=[], wall_bboxes=[],
    )
    assert result["sources_pass_each_other"] is True


def test_set_flags_returns_only_true():
    d = {"a": True, "b": False, "c": True}
    assert set_flags(d) == {"a", "c"}


def test_per_source_flag_is_or_over_sources():
    """Occluded_by_furniture should be True if ANY source is occluded."""
    spec = _stub_spec()
    t1 = np.array([[3, 3, 1.2]] * 20)
    t2 = np.array([[4, 0, 0.5]] * 20)
    obs = [((1.0, -1.0, 0.0), (2.0, 1.0, 1.5))]
    result = verify_all_flags(
        spec_dict=spec, trajectories=[t1, t2],
        furniture_bboxes=obs, wall_bboxes=[],
    )
    assert result["occluded_by_furniture"] is True


def test_flag_details_reports_per_source_and_pairwise_flags():
    spec = _stub_spec()
    stationary = np.array([[0, 0, 1.2]] * 30)
    moving = np.linspace([-1, 1.0, 1.2], [1, -1.0, 1.2], num=30)

    details = verify_flag_details(
        spec_dict=spec,
        trajectories=[stationary, moving],
        furniture_bboxes=[],
        wall_bboxes=[],
        source_tags=["front_idle", "moving"],
    )

    assert set(details["aggregate"].keys()) == set(ALL_FLAGS)
    assert details["per_source"]["front_idle"]["stationary"] is True
    assert details["per_source"]["front_idle"]["steady_walk"] is False
    assert details["per_source"]["moving"]["stationary"] is False
    assert details["per_source"]["moving"]["crosses_azimuth_zero"] is True
    assert details["pairwise"]["sources_pass_each_other"] is True
    assert details["pairwise"]["pairs"] == [
        {
            "tags": ["front_idle", "moving"],
            "sources_pass_each_other": True,
        }
    ]
