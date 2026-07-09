import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from scene_generator import (  # noqa: E402
    SceneSample, sample_mic_pose, sample_n_sources, sample_source_position,
    sample_scene, sample_source_position_in_camera_sector,
)
from audio_library import load_library  # noqa: E402
from source_asset_registry import resolve_source_pool  # noqa: E402


BOUNDS = (-4.0, -5.0, 6.0, 6.0)  # x_min, y_min, x_max, y_max
OBSTACLES = [((1.0, -1.0, 0.0), (2.0, 1.0, 1.5))]


def _azimuth_relative_to_yaw(src_pos, mic_pos, mic_yaw_deg):
    v = np.asarray(src_pos[:2], dtype=np.float64) - np.asarray(mic_pos[:2])
    yr = np.deg2rad(mic_yaw_deg)
    c, s = np.cos(yr), np.sin(yr)
    x_local = c * v[0] + s * v[1]
    y_local = -s * v[0] + c * v[1]
    return float(np.degrees(np.arctan2(y_local, x_local)))


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


def test_sample_source_position_in_camera_sector_respects_fov_margin():
    rng = np.random.default_rng(2)
    mic = (0.0, 0.0, 1.2)
    yaw = 0.0
    pos = sample_source_position_in_camera_sector(
        (-4.0, -4.0, 4.0, 4.0),
        [],
        mic,
        yaw,
        rng,
        distance_range=(1.0, 3.0),
        z_m=0.45,
        fov_h_deg=90.0,
        fov_margin_deg=10.0,
        valid_regions=[(-4.0, -4.0, 4.0, 4.0)],
    )

    assert abs(_azimuth_relative_to_yaw(pos, mic, yaw)) <= 35.0


def test_sample_mic_and_source_stay_inside_valid_regions():
    """Regression for apartment_v2 smoke clip_0002: bbox bounds included a
    top-right outdoor void. Valid regions must reject that void."""
    rng = np.random.default_rng(1)
    bounds = (0.0, 0.0, 5.0, 5.0)
    valid_regions = [(0.0, 0.0, 2.0, 5.0), (0.0, 0.0, 5.0, 2.0)]
    obstacles = []
    for _ in range(100):
        mic, _ = sample_mic_pose(
            bounds, obstacles, rng, valid_regions=valid_regions,
            height_range=(0.5, 1.8),
        )
        src = sample_source_position(
            bounds, obstacles, mic, rng, valid_regions=valid_regions,
            distance_range=(0.5, 5.0), z_m=0.45,
        )
        for pos in (mic, src):
            x, y = pos[:2]
            assert any(x0 <= x <= x1 and y0 <= y <= y1
                       for x0, y0, x1, y1 in valid_regions), pos


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
        "valid_regions": [BOUNDS],
        "distance_range_m": [0.5, 6.0],
        "mic_height_range_m": [0.5, 1.8],
        "source_height_m": 0.45,
    }
    rng = np.random.default_rng(42)
    scene = sample_scene(template, lib, rng)
    assert isinstance(scene, SceneSample)
    assert 0 <= len(scene.source_specs) <= 2


def test_sample_scene_default_pool_uses_registered_assets(tmp_path):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"samples": [
        {"category": "dog_bark", "path": "dog.wav", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
    ]}))
    lib = load_library(p)
    template = {
        "bounds_xy": list(BOUNDS),
        "obstacles": [],
        "valid_regions": [BOUNDS],
        "distance_range_m": [0.5, 6.0],
        "mic_height_range_m": [0.5, 1.8],
        "source_height_m": 0.45,
        "n_sources_override": 2,
    }

    scene = sample_scene(template, lib, np.random.default_rng(8))

    assert {s["asset_id"] for s in scene.source_specs} == {
        "dog_golden_0001",
        "dog_beagle_0002",
    }
    assert {s["tag"] for s in scene.source_specs} == {
        "dog_golden",
        "dog_beagle_v2",
    }


def test_sample_scene_uses_configured_source_pool(tmp_path, monkeypatch):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"samples": [
        {"category": "dog_bark", "path": "dog.wav", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
        {"category": "cat_purring", "path": "cat.mp3", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
    ]}))
    lib = load_library(p)
    template = {
        "bounds_xy": list(BOUNDS),
        "obstacles": [],
        "valid_regions": [BOUNDS],
        "distance_range_m": [0.5, 6.0],
        "mic_height_range_m": [0.5, 1.8],
        "source_height_m": 0.45,
        "source_pool": [
            {"tag": "dog_beagle_v2", "audio_lookup": "dog_bark"},
            {"tag": "cat_british_shorthair_v2", "audio_lookup": "cat_purring"},
        ],
    }
    import scene_generator as sg
    monkeypatch.setattr(sg, "sample_n_sources", lambda rng: 2)

    scene = sample_scene(template, lib, np.random.default_rng(9))

    assert {s["tag"] for s in scene.source_specs} == {
        "dog_beagle_v2",
        "cat_british_shorthair_v2",
    }
    assert {s["audio_lookup"] for s in scene.source_specs} == {
        "dog_bark",
        "cat_purring",
    }


def test_sample_scene_resolves_asset_id_source_pool(tmp_path):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"samples": [
        {"category": "dog_bark", "path": "dog.wav", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
        {"category": "cat_purring", "path": "cat.mp3", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
    ]}))
    lib = load_library(p)
    template = {
        "bounds_xy": list(BOUNDS),
        "obstacles": [],
        "valid_regions": [BOUNDS],
        "distance_range_m": [0.5, 6.0],
        "mic_height_range_m": [0.5, 1.8],
        "source_height_m": 0.45,
        "n_sources_override": 2,
        "source_pool": [
            {"asset_id": "dog_beagle_0002"},
            {"asset_id": "cat_british_shorthair_0002"},
        ],
    }

    scene = sample_scene(template, lib, np.random.default_rng(10))

    assert {s["asset_id"] for s in scene.source_specs} == {
        "dog_beagle_0002",
        "cat_british_shorthair_0002",
    }
    assert {s["tag"] for s in scene.source_specs} == {
        "dog_beagle_v2",
        "cat_british_shorthair_v2",
    }
    assert {s["audio_lookup"] for s in scene.source_specs} == {
        "dog_bark",
        "cat_purring",
    }


def test_sample_scene_supports_n_sources_override(tmp_path, monkeypatch):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"samples": [
        {"category": "dog_bark", "path": "dog.wav", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
        {"category": "cat_purring", "path": "cat.mp3", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
    ]}))
    lib = load_library(p)
    template = {
        "bounds_xy": list(BOUNDS),
        "obstacles": [],
        "valid_regions": [BOUNDS],
        "distance_range_m": [0.5, 6.0],
        "mic_height_range_m": [0.5, 1.8],
        "source_height_m": 0.45,
        "n_sources_override": 2,
        "source_pool": [
            {"tag": "dog_beagle_v2", "audio_lookup": "dog_bark"},
            {"tag": "cat_british_shorthair_v2", "audio_lookup": "cat_purring"},
        ],
    }
    import scene_generator as sg
    monkeypatch.setattr(sg, "sample_n_sources", lambda rng: 0)

    scene = sample_scene(template, lib, np.random.default_rng(11))

    assert len(scene.source_specs) == 2


def test_sample_scene_can_target_camera_sector_sources(tmp_path):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({"samples": [
        {"category": "dog_bark", "path": "dog.wav", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
        {"category": "cat_purring", "path": "cat.mp3", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
    ]}))
    lib = load_library(p)
    bounds = (-4.0, -4.0, 4.0, 4.0)
    template = {
        "bounds_xy": list(bounds),
        "obstacles": [],
        "valid_regions": [bounds],
        "distance_range_m": [1.0, 3.0],
        "mic_height_range_m": [1.2, 1.2],
        "source_height_m": 0.45,
        "n_sources_override": 2,
        "source_position_mode": "camera_sector",
        "camera_fov_h_deg": 90.0,
        "source_visible_fov_margin_deg": 10.0,
        "source_pool": [
            {"tag": "dog_beagle_v2", "audio_lookup": "dog_bark"},
            {"tag": "cat_british_shorthair_v2", "audio_lookup": "cat_purring"},
        ],
    }

    scene = sample_scene(template, lib, np.random.default_rng(12))

    assert len(scene.source_specs) == 2
    for src in scene.source_specs:
        for key in ("start_pos_m", "end_pos_m"):
            assert abs(_azimuth_relative_to_yaw(
                src[key], scene.mic_pos_m, scene.mic_yaw_deg,
            )) <= 35.0


def test_apartment_m1_spec_declares_review_source_pool():
    spec = json.loads((REPO / "data" / "apartment_v2_m1_dataset_spec.json").read_text())
    pool = spec["source_pool"]
    assert {s["asset_id"] for s in pool} == {
        "dog_golden_0001",
        "dog_beagle_0002",
        "cat_british_shorthair_0002",
    }
    resolved_pool = resolve_source_pool(pool)
    tags = {s["tag"] for s in resolved_pool}
    assert {"dog_golden", "dog_beagle_v2", "cat_british_shorthair_v2"} <= tags
    audio_by_tag = {s["tag"]: s["audio_lookup"] for s in resolved_pool}
    assert audio_by_tag["dog_beagle_v2"] == "dog_bark"
    assert audio_by_tag["cat_british_shorthair_v2"] == "cat_purring"


def test_audio_library_has_cat_purring_sample():
    lib = load_library(REPO / "data" / "audio_library_v1.json")
    assert "cat_purring" in lib.categories
    sample = lib.sample("cat_purring", np.random.default_rng(0))
    assert sample.path.exists()
    assert sample.is_synthetic is False
