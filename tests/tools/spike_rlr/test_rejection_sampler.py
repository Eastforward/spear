import sys
from pathlib import Path

import json
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from rejection_sampler import SamplerConfig, generate_batch  # noqa: E402
from audio_library import load_library  # noqa: E402
from flag_definitions import ALL_FLAGS  # noqa: E402


def _stub_lib(tmp_path):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"samples": [
        {"category": "dog_bark", "path": "a.wav", "is_synthetic": False,
         "duration_s": 5.0, "sample_rate": 16000, "source": "T"},
        {"category": "music_piano", "path": "b.wav", "is_synthetic": True,
         "duration_s": 5.0, "sample_rate": 16000, "source": "T"},
    ]}))
    return load_library(p)


def _stub_template():
    return {
        "bounds_xy": [-5.0, -5.0, 5.0, 5.0],
        "obstacles": [],
        "distance_range_m": [0.5, 6.0],
        "mic_height_range_m": [0.5, 1.8],
        "source_height_m": 0.45,
        "n_frames": 30, "fps": 15,
        "camera_fov_h_deg": 90, "camera_fov_v_deg": 60,
    }


def test_generate_batch_returns_n_clips(tmp_path):
    lib = _stub_lib(tmp_path)
    cfg = SamplerConfig(n_clips_target=5, per_flag_min_coverage=1)
    batch = generate_batch(cfg, _stub_template(), lib,
                            np.random.default_rng(0),
                            obstacle_context={"furniture_bboxes": [], "wall_bboxes": []})
    assert len(batch) == 5


def test_each_clip_has_expected_fields(tmp_path):
    lib = _stub_lib(tmp_path)
    cfg = SamplerConfig(n_clips_target=3, per_flag_min_coverage=1)
    batch = generate_batch(cfg, _stub_template(), lib,
                            np.random.default_rng(0),
                            obstacle_context={"furniture_bboxes": [], "wall_bboxes": []})
    for clip in batch:
        assert "scene_sample" in clip
        assert "trajectories" in clip
        assert "flags" in clip
        assert set(clip["flags"].keys()) == set(ALL_FLAGS)


def test_deterministic_with_seed(tmp_path):
    lib = _stub_lib(tmp_path)
    cfg = SamplerConfig(n_clips_target=3, per_flag_min_coverage=1)
    a = generate_batch(cfg, _stub_template(), lib, np.random.default_rng(42),
                        obstacle_context={"furniture_bboxes": [], "wall_bboxes": []})
    b = generate_batch(cfg, _stub_template(), lib, np.random.default_rng(42),
                        obstacle_context={"furniture_bboxes": [], "wall_bboxes": []})
    for x, y in zip(a, b):
        assert x["scene_sample"].mic_pos_m == y["scene_sample"].mic_pos_m
