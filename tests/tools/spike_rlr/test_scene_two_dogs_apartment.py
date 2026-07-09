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
    assert tags == {"dog_golden", "dog_beagle_v2"}


def test_trajectories_have_75_frames():
    sc = compose_two_dog_scene_apartment(SPEC)
    for a in sc.animals:
        assert len(a.trajectory_m) == 75, f"{a.tag} has {len(a.trajectory_m)} frames"
        assert len(a.yaw_deg) == 75


def test_beagle_traj_matches_spec_endpoints():
    spec = json.loads(SPEC.read_text())
    beagle_spec = [s for s in spec["sources"] if s["tag"] == "dog_beagle_v2"][0]
    sc = compose_two_dog_scene_apartment(SPEC)
    beagle = [a for a in sc.animals if a.tag == "dog_beagle_v2"][0]
    assert np.allclose(beagle.trajectory_m[0], beagle_spec["start_pos_m"], atol=1e-3)
    assert np.allclose(beagle.trajectory_m[-1], beagle_spec["end_pos_m"], atol=1e-3)


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


def test_yaw_diffs_are_smooth():
    """Planned + Chaikin-smoothed motion should have small step-to-step yaw
    changes (no >20 deg jumps). Pure linear would be <1 deg; planner curves
    add a bit."""
    sc = compose_two_dog_scene_apartment(SPEC)
    for a in sc.animals:
        yaws = np.asarray(a.yaw_deg)
        unwrapped = np.unwrap(np.deg2rad(yaws))
        diffs_deg = np.rad2deg(np.diff(unwrapped))
        assert np.max(np.abs(diffs_deg)) < 20.0, \
            f"{a.tag}: yaw jumps too big ({np.max(np.abs(diffs_deg)):.2f} deg)"


def test_scene_seed_is_deterministic():
    sc1 = compose_two_dog_scene_apartment(SPEC)
    sc2 = compose_two_dog_scene_apartment(SPEC)
    assert sc1.seed == sc2.seed
    for a1, a2 in zip(sc1.animals, sc2.animals):
        assert np.allclose(a1.trajectory_m, a2.trajectory_m)
        assert np.allclose(a1.yaw_deg, a2.yaw_deg)


def test_explicit_trajectory_m_is_used_without_replanning(tmp_path):
    spec = json.loads(SPEC.read_text())
    spec["sources"] = [spec["sources"][0]]
    n_frames = spec["render_config"]["n_frames"]
    explicit = np.column_stack([
        np.linspace(-3.0, -3.0, n_frames),
        np.linspace(-4.0, -3.0, n_frames),
        np.full(n_frames, 0.45),
    ])
    spec["sources"][0]["start_pos_m"] = explicit[0].tolist()
    spec["sources"][0]["end_pos_m"] = explicit[-1].tolist()
    spec["sources"][0]["trajectory_m"] = explicit.tolist()
    out = tmp_path / "spec.json"
    out.write_text(json.dumps(spec))

    scene = compose_two_dog_scene_apartment(out)

    assert len(scene.animals) == 1
    assert np.allclose(scene.animals[0].trajectory_m, explicit)


def test_stationary_source_defaults_to_idle_anim(tmp_path):
    spec = json.loads(SPEC.read_text())
    spec["sources"] = [spec["sources"][0]]
    n_frames = spec["render_config"]["n_frames"]
    stationary = np.tile(np.asarray([-1.3, -3.2, 0.45], dtype=np.float64),
                         (n_frames, 1))
    spec["sources"][0]["start_pos_m"] = stationary[0].tolist()
    spec["sources"][0]["end_pos_m"] = stationary[-1].tolist()
    spec["sources"][0]["trajectory_m"] = stationary.tolist()
    spec["sources"][0]["motion_style"] = "stationary"
    spec["sources"][0].pop("wanted_anim", None)
    out = tmp_path / "spec.json"
    out.write_text(json.dumps(spec))

    scene = compose_two_dog_scene_apartment(out)

    assert len(scene.animals) == 1
    assert scene.animals[0].wanted_anim == "Idle"
