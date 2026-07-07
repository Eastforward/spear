"""Tests for compute_acoustic_metadata.py output."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
META = REPO / "tmp" / "spike_output_apartment" / "apartment_v1_metadata.json"


def test_metadata_json_written():
    if not META.exists():
        pytest.skip("metadata not yet computed — run compute_acoustic_metadata.py")
    d = json.loads(META.read_text())
    assert d["clip_id"] == "apartment_v1_000"
    assert d["n_frames"] == 75


def test_two_sources_in_metadata():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    assert len(d["sources"]) == 2
    tags = {s["tag"] for s in d["sources"]}
    assert tags == {"dog_golden", "dog_husky"}


def test_per_frame_arrays_correct_length():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    for s in d["sources"]:
        assert len(s["drr_db_per_frame"]) == 75
        assert len(s["source_world_xyz_per_frame"]) == 75
        assert len(s["source_azi_ele_dist_mic_local_per_frame"]) == 75
        assert len(s["source_amp_gain_per_frame"]) == 75


def test_azi_ele_within_ranges():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    for s in d["sources"]:
        for azi, ele, dist in s["source_azi_ele_dist_mic_local_per_frame"]:
            assert -180 <= azi <= 180
            assert -90 <= ele <= 90
            assert 0 < dist < 20  # apartment biggest dim ~13m + margin


def test_metadata_mic_pose_matches_spec():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    spec = json.loads((REPO / "data" / "apartment_v1_spec.json").read_text())
    assert d["mic_pose_6DoF"]["pos_m"] == spec["mic"]["pos_m"]
    assert d["mic_pose_6DoF"]["yaw_deg"] == spec["mic"]["yaw_deg"]


def test_source_category_and_is_synthetic():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    golden = [s for s in d["sources"] if s["tag"] == "dog_golden"][0]
    husky = [s for s in d["sources"] if s["tag"] == "dog_husky"][0]
    assert golden["category"] == "dog_bark"
    assert golden["is_synthetic"] is False
    assert husky["category"] == "music_piano"
    assert husky["is_synthetic"] is True


def test_azi_ele_dist_local_offset_pure_x():
    """Direct unit test of the spherical-coord helper."""
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from compute_acoustic_metadata import azi_ele_dist_local

    # Mic at origin looking +X (yaw=0). Source at +X = directly ahead.
    azi, ele, dist = azi_ele_dist_local(
        src_xyz=[3.0, 0.0, 0.0], mic_xyz=[0.0, 0.0, 0.0], mic_yaw_deg=0.0)
    assert abs(azi) < 0.1     # directly ahead
    assert abs(ele) < 0.1
    assert abs(dist - 3.0) < 0.01

    # Source at +Y (mic-left when facing +X): azi = +90.
    azi, ele, dist = azi_ele_dist_local(
        src_xyz=[0.0, 2.0, 0.0], mic_xyz=[0.0, 0.0, 0.0], mic_yaw_deg=0.0)
    assert abs(azi - 90.0) < 0.1
    assert abs(dist - 2.0) < 0.01

    # Source at +Z overhead: ele = +90.
    azi, ele, dist = azi_ele_dist_local(
        src_xyz=[0.0, 0.0, 1.5], mic_xyz=[0.0, 0.0, 0.0], mic_yaw_deg=0.0)
    assert abs(ele - 90.0) < 0.1
    assert abs(dist - 1.5) < 0.01


def test_visibility_fields_present():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    for s in d["sources"]:
        assert "source_in_fov_per_frame" in s
        assert "source_occluded_by_furniture_per_frame" in s
        assert "source_visible_from_camera_per_frame" in s
        assert len(s["source_in_fov_per_frame"]) == 75
        # All are bool
        for v in s["source_in_fov_per_frame"]:
            assert isinstance(v, bool)


def test_visible_implies_in_fov_and_not_occluded():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    for s in d["sources"]:
        for k in range(len(s["source_visible_from_camera_per_frame"])):
            vis = s["source_visible_from_camera_per_frame"][k]
            in_fov = s["source_in_fov_per_frame"][k]
            occ = s["source_occluded_by_furniture_per_frame"][k]
            assert vis == (in_fov and not occ), \
                f"frame {k}: visible={vis} but in_fov={in_fov}, occluded={occ}"
