"""Tests for tools/spike_rlr/visibility.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from visibility import frame_visibility, batch_frame_visibility  # noqa: E402


def test_source_directly_ahead_is_in_fov():
    """Mic at origin looking +X (yaw=0), source at (3, 0, 1.2). In FOV."""
    r = frame_visibility(
        src_xyz=(3.0, 0.0, 1.2), mic_pos=(0.0, 0.0, 1.2), mic_yaw_deg=0.0,
        fov_h_deg=90.0, fov_v_deg=60.0, obstacles_xyz=None,
    )
    assert r["in_fov"] is True
    assert r["occluded_by_furniture"] is False
    assert r["visible"] is True


def test_source_behind_is_out_of_fov():
    r = frame_visibility(
        src_xyz=(-3.0, 0.0, 1.2), mic_pos=(0.0, 0.0, 1.2), mic_yaw_deg=0.0,
        fov_h_deg=90.0, fov_v_deg=60.0,
    )
    assert r["in_fov"] is False
    assert r["visible"] is False


def test_source_at_edge_of_h_fov():
    """FOV 90° means half-angle 45°; source at (3, 3) is at yaw 45° from mic."""
    r = frame_visibility(
        src_xyz=(3.0, 3.0, 1.2), mic_pos=(0.0, 0.0, 1.2), mic_yaw_deg=0.0,
        fov_h_deg=90.0, fov_v_deg=60.0,
    )
    # Exactly at edge — accept in_fov=True or False (tolerance)
    # Just check it doesn't crash
    assert "in_fov" in r


def test_source_low_ground_at_far_distance_out_of_vertical_fov():
    """Mic at Z=1.2 looking horizontally; source at Z=0 at X=10.
    Elevation to source ~= atan(-1.2 / 10) = -6.8 deg; FOV_V half = 30 deg.
    So it IS in vertical FOV. Now put source close: X=1, then elev = -50 deg,
    outside FOV_V/2=30. Should be out of FOV."""
    r_near = frame_visibility(
        src_xyz=(1.0, 0.0, 0.0), mic_pos=(0.0, 0.0, 1.2), mic_yaw_deg=0.0,
        fov_h_deg=90.0, fov_v_deg=60.0,
    )
    assert r_near["in_fov"] is False, "very-close low-Z source should be below FOV_V"

    r_far = frame_visibility(
        src_xyz=(10.0, 0.0, 0.0), mic_pos=(0.0, 0.0, 1.2), mic_yaw_deg=0.0,
        fov_h_deg=90.0, fov_v_deg=60.0,
    )
    assert r_far["in_fov"] is True


def test_source_occluded_by_furniture_between_mic_and_source():
    """Ray from mic (0, 0, 1.2) to source (4, 0, 0.5) passes through a
    furniture bbox at X=[1, 2], Y=[-1, 1], Z=[0, 1.5] -> occluded."""
    obstacles = [((1.0, -1.0, 0.0), (2.0, 1.0, 1.5))]  # list of (bmin, bmax)
    r = frame_visibility(
        src_xyz=(4.0, 0.0, 0.5), mic_pos=(0.0, 0.0, 1.2), mic_yaw_deg=0.0,
        fov_h_deg=90.0, fov_v_deg=60.0, obstacles_xyz=obstacles,
    )
    assert r["in_fov"] is True
    assert r["occluded_by_furniture"] is True
    assert r["visible"] is False


def test_source_not_occluded_when_furniture_off_ray():
    """Furniture bbox exists but not on the ray -> not occluded."""
    obstacles = [((1.0, 2.0, 0.0), (2.0, 3.0, 1.5))]  # in +Y half
    r = frame_visibility(
        src_xyz=(4.0, 0.0, 1.2), mic_pos=(0.0, 0.0, 1.2), mic_yaw_deg=0.0,
        fov_h_deg=90.0, fov_v_deg=60.0, obstacles_xyz=obstacles,
    )
    assert r["in_fov"] is True
    assert r["occluded_by_furniture"] is False
    assert r["visible"] is True


def test_batch_returns_arrays():
    src_xyz_array = np.array([[3, 0, 1.2], [-3, 0, 1.2], [3, 3, 1.2]])
    r = batch_frame_visibility(
        src_xyz_array=src_xyz_array,
        mic_pos=(0.0, 0.0, 1.2), mic_yaw_deg=0.0,
        fov_h_deg=90.0, fov_v_deg=60.0, obstacles_xyz=None,
    )
    assert r["in_fov"].shape == (3,)
    assert r["visible"].shape == (3,)
    assert r["in_fov"][0] == True    # front
    assert r["in_fov"][1] == False   # behind
