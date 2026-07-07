"""Tests for each of the 12 flag functions."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from flag_definitions import (  # noqa: E402
    ALL_FLAGS,
    is_occluded_by_furniture, is_occluded_by_wall, is_never_occluded,
    is_leaves_camera_fov, is_stays_in_camera_fov,
    is_crosses_azimuth_zero, is_passes_close_to_mic, is_far_from_mic_whole_clip,
    is_stationary, is_steady_walk, is_stop_and_go,
    is_sources_pass_each_other,
)


def test_all_flags_list_has_twelve_entries():
    assert len(ALL_FLAGS) == 12
    assert len(set(ALL_FLAGS)) == 12  # no duplicates


# ---- Group A: occlusion ----

def test_occluded_by_furniture_true_when_ray_hits_bbox():
    traj = np.array([[4, 0, 0.5]] * 30)
    obs_furn = [((1.0, -1.0, 0.0), (2.0, 1.0, 1.5))]
    r = is_occluded_by_furniture(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), mic_yaw_deg=0,
        fov_h_deg=90, fov_v_deg=60,
        furniture_bboxes=obs_furn, wall_bboxes=[],
    )
    assert r is True


def test_occluded_by_furniture_false_when_never_occluded():
    traj = np.array([[3, 3, 1.2]] * 30)
    r = is_occluded_by_furniture(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), mic_yaw_deg=0,
        fov_h_deg=90, fov_v_deg=60,
        furniture_bboxes=[((-5, -5, 0), (-4, -4, 1))],  # far away
        wall_bboxes=[],
    )
    assert r is False


def test_never_occluded_true_when_zero_occlusion_frames():
    traj = np.array([[3, 3, 1.2]] * 30)
    r = is_never_occluded(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), mic_yaw_deg=0,
        fov_h_deg=90, fov_v_deg=60,
        furniture_bboxes=[], wall_bboxes=[],
    )
    assert r is True


# ---- Group B: FOV ----

def test_leaves_camera_fov_true_when_any_frame_out():
    traj = np.array([[3, 0, 1.2]] * 20 + [[-3, 0, 1.2]] * 20)
    r = is_leaves_camera_fov(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), mic_yaw_deg=0,
        fov_h_deg=90, fov_v_deg=60,
        furniture_bboxes=[], wall_bboxes=[],
    )
    assert r is True


def test_stays_in_camera_fov_true_when_all_frames_in():
    traj = np.array([[3, 0, 1.2]] * 40)
    r = is_stays_in_camera_fov(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), mic_yaw_deg=0,
        fov_h_deg=90, fov_v_deg=60,
        furniture_bboxes=[], wall_bboxes=[],
    )
    assert r is True


# ---- Group C: spatial ----

def test_crosses_azimuth_zero_true_when_azi_flips_sign():
    traj = np.linspace([3, -2, 1.2], [3, 2, 1.2], num=40)
    r = is_crosses_azimuth_zero(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), mic_yaw_deg=0,
    )
    assert r is True


def test_crosses_azimuth_zero_false_when_stays_on_one_side():
    traj = np.linspace([3, 1, 1.2], [3, 3, 1.2], num=40)
    r = is_crosses_azimuth_zero(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), mic_yaw_deg=0,
    )
    assert r is False


def test_passes_close_to_mic_true_when_min_dist_below_threshold():
    traj = np.linspace([-3, 0, 1.2], [3, 0, 1.2], num=20)
    r = is_passes_close_to_mic(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), threshold_m=1.0,
    )
    assert r is True


def test_passes_close_to_mic_false_when_all_far():
    traj = np.array([[5, 0, 1.2]] * 20)
    r = is_passes_close_to_mic(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), threshold_m=1.0,
    )
    assert r is False


def test_far_from_mic_whole_clip():
    traj = np.array([[5, 0, 1.2]] * 20)
    r = is_far_from_mic_whole_clip(
        traj_xyz=traj, mic_pos=(0, 0, 1.2), threshold_m=4.0,
    )
    assert r is True


# ---- Group D: motion ----

def test_stationary_true_when_all_speed_zero():
    traj = np.array([[3, 0, 1.2]] * 30)
    assert is_stationary(traj_xyz=traj, fps=15) is True


def test_stationary_false_when_moving():
    traj = np.linspace([-3, 0, 1.2], [3, 0, 1.2], num=30)
    assert is_stationary(traj_xyz=traj, fps=15) is False


def test_steady_walk_true_when_speed_variance_low():
    traj = np.linspace([-3, 0, 1.2], [3, 0, 1.2], num=30)
    assert is_steady_walk(traj_xyz=traj, fps=15) is True


def test_stop_and_go_true_when_speed_varies():
    a = np.linspace([-3, 0, 1.2], [0, 0, 1.2], num=15)
    b = np.array([[0, 0, 1.2]] * 15)
    traj = np.concatenate([a, b], axis=0)
    assert is_stop_and_go(traj_xyz=traj, fps=15) is True


# ---- Group E: multi-source ----

def test_sources_pass_each_other_true_when_dist_below_threshold_at_any_frame():
    t1 = np.linspace([-3, 0, 1.2], [3, 0, 1.2], num=30)
    t2 = np.linspace([3, 0, 1.2], [-3, 0, 1.2], num=30)
    r = is_sources_pass_each_other(
        traj_xyz_a=t1, traj_xyz_b=t2, threshold_m=0.5,
    )
    assert r is True


def test_sources_pass_each_other_false_when_parallel_far():
    t1 = np.array([[2, 0, 1.2]] * 30)
    t2 = np.array([[-2, 0, 1.2]] * 30)
    r = is_sources_pass_each_other(
        traj_xyz_a=t1, traj_xyz_b=t2, threshold_m=1.0,
    )
    assert r is False
