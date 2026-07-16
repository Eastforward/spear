"""Verifies scene_two_dogs_v2 compose_two_dog_scene_v2() output matches SSOT."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "data" / "shoebox_v2_spec.json"
SCENE_PATH = REPO_ROOT / "tools" / "spike_rlr" / "scene_two_dogs_v2.py"

# scene_two_dogs_v2 imports from tools/gpurir_scenes/scene_spec, which
# needs tools/ on sys.path.
sys.path.insert(0, str(REPO_ROOT / "tools"))


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("scene_two_dogs_v2", SCENE_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def scene(module):
    return module.compose_two_dog_scene_v2(SPEC_PATH)


@pytest.fixture(scope="module")
def spec():
    with open(SPEC_PATH) as f:
        return json.load(f)


def test_scene_has_two_animals(scene):
    assert len(scene.animals) == 2
    tags = {a.tag for a in scene.animals}
    assert tags == {"dog_golden", "dog_husky"}


def test_trajectory_length(scene, spec):
    n = spec["render_config"]["n_frames"]
    for a in scene.animals:
        assert a.trajectory_m.shape == (n, 3), (
            f"{a.tag} trajectory shape {a.trajectory_m.shape}, expected ({n}, 3)"
        )
        assert a.yaw_deg.shape == (n,)


def test_husky_start_and_end_match_spec(scene, spec):
    husky = next(a for a in scene.animals if a.tag == "dog_husky")
    husky_spec = next(s for s in spec["sources"] if s["tag"] == "dog_husky")
    first_seg = husky_spec["trajectory_m"][0]
    last_seg = husky_spec["trajectory_m"][-1]
    np.testing.assert_allclose(husky.trajectory_m[0], first_seg["start_m"], atol=1e-6)
    np.testing.assert_allclose(husky.trajectory_m[-1], last_seg["end_m"], atol=1e-6)


def test_husky_segment_transitions_are_piecewise_linear(scene, spec):
    """Verify each segment starts and ends where spec says, at the frame boundary."""
    husky = next(a for a in scene.animals if a.tag == "dog_husky")
    for seg in spec["sources"][1]["trajectory_m"]:
        # spec source[1] is husky (source[0]=golden), but let's be robust
        pass
    husky_spec = next(s for s in spec["sources"] if s["tag"] == "dog_husky")
    for seg in husky_spec["trajectory_m"]:
        f0 = seg["frame_start"]
        f1 = seg["frame_end"]
        np.testing.assert_allclose(husky.trajectory_m[f0], seg["start_m"], atol=1e-6,
                                   err_msg=f"segment {seg['phase']} frame_start mismatch")
        np.testing.assert_allclose(husky.trajectory_m[f1], seg["end_m"], atol=1e-6,
                                   err_msg=f"segment {seg['phase']} frame_end mismatch")


def test_husky_hold_segment_is_stationary(scene, spec):
    """D segment (frame 60-74) must be constant position (hold)."""
    husky = next(a for a in scene.animals if a.tag == "dog_husky")
    hold = husky.trajectory_m[60:75]
    ref = hold[0]
    for i in range(1, 15):
        np.testing.assert_allclose(hold[i], ref, atol=1e-6,
                                   err_msg=f"hold segment moved at offset {i}")


def test_golden_uniform_walk(scene, spec):
    """Golden is uniform linear between spec's start and end."""
    golden = next(a for a in scene.animals if a.tag == "dog_golden")
    golden_spec = next(s for s in spec["sources"] if s["tag"] == "dog_golden")
    np.testing.assert_allclose(golden.trajectory_m[0], golden_spec["start_pos_m"], atol=1e-6)
    np.testing.assert_allclose(golden.trajectory_m[-1], golden_spec["end_pos_m"], atol=1e-6)
    # Uniformly spaced: consecutive step distance should be identical (within rounding)
    step = np.linalg.norm(np.diff(golden.trajectory_m[:, :2], axis=0), axis=1)
    assert step.max() - step.min() < 1e-6, (
        f"golden step not uniform: range=[{step.min()}, {step.max()}]"
    )


def test_no_wall_clipping_v2(scene, module):
    """Reuse the shared check_no_clipping from scene_spec.py, ensuring the
    v2 layout satisfies footprint+wall+mutual-clearance invariants."""
    from gpurir_scenes.scene_spec import check_no_clipping
    check_no_clipping(scene)  # will raise on any violation


def test_husky_end_visually_occluded_by_sofa(scene, spec):
    """Sanity: mic->husky_end line must pass through sofa bounding volume."""
    mic = np.array(spec["mic"]["pos_m"])
    husky = next(a for a in scene.animals if a.tag == "dog_husky")
    end = husky.trajectory_m[-1]
    sofa = next(f for f in spec["furniture"] if f["name"] == "sofa")
    c = np.array(sofa["center_m"])
    s = np.array(sofa["size_m"])

    # Find intersection of line mic->end with sofa AABB
    # Line: mic + t*(end - mic), t in [0, 1]
    d = end - mic
    aabb_min = c - s / 2
    aabb_max = c + s / 2

    tmin, tmax = -np.inf, np.inf
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if mic[i] < aabb_min[i] or mic[i] > aabb_max[i]:
                pytest.fail(f"line-of-sight axis {i} misses sofa on parallel axis")
            continue
        t1 = (aabb_min[i] - mic[i]) / d[i]
        t2 = (aabb_max[i] - mic[i]) / d[i]
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
    assert tmax >= tmin and tmin <= 1.0 and tmax >= 0.0, (
        f"mic->husky_end does NOT intersect sofa AABB "
        f"(tmin={tmin}, tmax={tmax}); occlusion invalid."
    )
