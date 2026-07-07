"""Tests for scene_two_dogs_apartment composer."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from scene_two_dogs_apartment import compose_two_dog_scene_apartment  # noqa: E402

SPEC = REPO / "data" / "apartment_v1_spec.json"


def test_compose_returns_two_animals():
    sc = compose_two_dog_scene_apartment(SPEC)
    tags = {a.tag for a in sc.animals}
    assert tags == {"dog_golden", "dog_husky"}


def test_trajectories_have_75_frames():
    sc = compose_two_dog_scene_apartment(SPEC)
    for a in sc.animals:
        assert len(a.trajectory_m) == 75, f"{a.tag} has {len(a.trajectory_m)} frames"
        assert len(a.yaw_deg) == 75


def test_husky_traj_matches_spec_endpoints():
    spec = json.loads(SPEC.read_text())
    husky_spec = [s for s in spec["sources"] if s["tag"] == "dog_husky"][0]
    sc = compose_two_dog_scene_apartment(SPEC)
    husky = [a for a in sc.animals if a.tag == "dog_husky"][0]
    assert np.allclose(husky.trajectory_m[0], husky_spec["start_pos_m"], atol=1e-3)
    assert np.allclose(husky.trajectory_m[-1], husky_spec["end_pos_m"], atol=1e-3)


def test_golden_traj_matches_spec_endpoints():
    spec = json.loads(SPEC.read_text())
    golden_spec = [s for s in spec["sources"] if s["tag"] == "dog_golden"][0]
    sc = compose_two_dog_scene_apartment(SPEC)
    golden = [a for a in sc.animals if a.tag == "dog_golden"][0]
    assert np.allclose(golden.trajectory_m[0], golden_spec["start_pos_m"], atol=1e-3)
    assert np.allclose(golden.trajectory_m[-1], golden_spec["end_pos_m"], atol=1e-3)


def test_yaw_is_finite_everywhere():
    sc = compose_two_dog_scene_apartment(SPEC)
    for a in sc.animals:
        yaws = np.asarray(a.yaw_deg)
        assert np.all(np.isfinite(yaws)), f"{a.tag}: non-finite yaw"


def test_yaw_diffs_are_small_for_linear_motion():
    """Uniform linear motion should have essentially constant yaw."""
    sc = compose_two_dog_scene_apartment(SPEC)
    for a in sc.animals:
        yaws = np.asarray(a.yaw_deg)
        # Wrap-safe diff via unwrap
        unwrapped = np.unwrap(np.deg2rad(yaws))
        diffs_deg = np.rad2deg(np.diff(unwrapped))
        assert np.max(np.abs(diffs_deg)) < 5.0, \
            f"{a.tag}: yaw jumps too big ({np.max(np.abs(diffs_deg)):.2f} deg)"


def test_scene_seed_is_deterministic():
    sc1 = compose_two_dog_scene_apartment(SPEC)
    sc2 = compose_two_dog_scene_apartment(SPEC)
    assert sc1.seed == sc2.seed
    for a1, a2 in zip(sc1.animals, sc2.animals):
        assert np.allclose(a1.trajectory_m, a2.trajectory_m)
        assert np.allclose(a1.yaw_deg, a2.yaw_deg)
