import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from scene_generator import (  # noqa: E402
    SceneSample, sample_mic_pose, sample_n_sources, sample_source_position,
    sample_scene,
)
from audio_library import load_library  # noqa: E402


BOUNDS = (-4.0, -5.0, 6.0, 6.0)  # x_min, y_min, x_max, y_max
OBSTACLES = [((1.0, -1.0, 0.0), (2.0, 1.0, 1.5))]


def test_sample_mic_pose_avoids_obstacles():
    rng = np.random.default_rng(0)
    for _ in range(100):
        pos, yaw = sample_mic_pose(BOUNDS, OBSTACLES, rng,
                                     height_range=(0.5, 1.8))
        assert BOUNDS[0] < pos[0] < BOUNDS[2]
        assert BOUNDS[1] < pos[1] < BOUNDS[3]
        assert 0.5 <= pos[2] <= 1.8
        assert 0.0 <= yaw < 360.0
        (x0, y0, z0), (x1, y1, z1) = OBSTACLES[0]
        assert not (x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1)


def test_sample_n_sources_distribution():
    rng = np.random.default_rng(0)
    counts = [0, 0, 0]
    for _ in range(3000):
        n = sample_n_sources(rng)
        assert n in (0, 1, 2)
        counts[n] += 1
    fractions = [c / 3000 for c in counts]
    assert abs(fractions[0] - 0.20) < 0.05
    assert abs(fractions[1] - 0.40) < 0.05
    assert abs(fractions[2] - 0.40) < 0.05


def test_sample_source_position_respects_distance():
    rng = np.random.default_rng(0)
    mic = (0.0, 0.0, 1.2)
    for _ in range(50):
        pos = sample_source_position(BOUNDS, OBSTACLES, mic, rng,
                                       distance_range=(0.5, 6.0), z_m=0.45)
        d = np.linalg.norm(np.array(pos[:2]) - np.array(mic[:2]))
        assert 0.5 <= d <= 6.0 + 0.1
        assert pos[2] == 0.45


def test_sample_scene_returns_scenesample(tmp_path):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"samples": [
        {"category": "dog_bark", "path": "a.wav", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
        {"category": "music_piano", "path": "b.wav", "is_synthetic": True,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
    ]}))
    lib = load_library(p)
    template = {
        "bounds_xy": list(BOUNDS),
        "obstacles": OBSTACLES,
        "distance_range_m": [0.5, 6.0],
        "mic_height_range_m": [0.5, 1.8],
        "source_height_m": 0.45,
    }
    rng = np.random.default_rng(42)
    scene = sample_scene(template, lib, rng)
    assert isinstance(scene, SceneSample)
    assert 0 <= len(scene.source_specs) <= 2
