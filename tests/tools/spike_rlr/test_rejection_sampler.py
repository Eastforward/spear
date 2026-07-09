import sys
from pathlib import Path

import json
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from rejection_sampler import (  # noqa: E402
    SamplerConfig, generate_batch, joint_visible_frame_count,
    meets_min_visible_frames, optimize_camera_yaw_for_visible_sources,
    visible_frame_counts,
)
from scene_generator import SceneSample  # noqa: E402
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


def test_review_visible_gate_counts_visible_frames_per_source():
    template = _stub_template()
    sample = SceneSample(
        mic_pos_m=(0.0, 0.0, 1.0),
        mic_yaw_deg=0.0,
        source_specs=[{"tag": "dog_beagle_v2"}],
    )
    visible_traj = np.array([[2.0, 0.0, 1.0]] * 4)
    behind_traj = np.array([[-2.0, 0.0, 1.0]] * 4)
    occluding_ctx = {
        "furniture_bboxes": [((1.0, -0.5, 0.0), (1.5, 0.5, 2.0))],
        "wall_bboxes": [],
    }
    empty_ctx = {"furniture_bboxes": [], "wall_bboxes": []}

    assert visible_frame_counts(sample, [visible_traj], template, empty_ctx) == [4]
    assert meets_min_visible_frames(
        sample, [visible_traj], template, empty_ctx, min_visible_frames=4
    )
    assert not meets_min_visible_frames(
        sample, [visible_traj], template, empty_ctx, min_visible_frames=5
    )
    assert visible_frame_counts(sample, [behind_traj], template, empty_ctx) == [0]
    assert not meets_min_visible_frames(
        sample, [behind_traj], template, empty_ctx, min_visible_frames=1
    )
    assert visible_frame_counts(sample, [visible_traj], template, occluding_ctx) == [0]
    assert not meets_min_visible_frames(sample, [], template, empty_ctx, 1)


def test_review_yaw_optimizer_targets_two_visible_sources():
    template = _stub_template()
    sample = SceneSample(
        mic_pos_m=(0.0, 0.0, 1.0),
        mic_yaw_deg=180.0,
        source_specs=[{"tag": "dog_beagle_v2"}, {"tag": "cat_british_shorthair_v2"}],
    )
    left = np.array([[2.0, 0.5, 1.0]] * 8)
    right = np.array([[2.0, -0.5, 1.0]] * 8)
    empty_ctx = {"furniture_bboxes": [], "wall_bboxes": []}

    assert visible_frame_counts(sample, [left, right], template, empty_ctx) == [0, 0]
    counts = optimize_camera_yaw_for_visible_sources(
        sample, [left, right], template, empty_ctx, yaw_step_deg=2.0
    )

    assert counts == [8, 8]
    assert joint_visible_frame_count(sample, [left, right], template, empty_ctx) == 8
    assert meets_min_visible_frames(
        sample, [left, right], template, empty_ctx, min_visible_frames=8
    )


def test_review_visible_gate_can_require_fov_margin():
    template = _stub_template()
    sample = SceneSample(
        mic_pos_m=(0.0, 0.0, 1.0),
        mic_yaw_deg=0.0,
        source_specs=[{"tag": "dog_beagle_v2"}],
    )
    edge_traj = np.array([[1.0, 0.93, 1.0]] * 4)  # ~43 deg azimuth
    empty_ctx = {"furniture_bboxes": [], "wall_bboxes": []}

    assert visible_frame_counts(sample, [edge_traj], template, empty_ctx) == [4]
    assert visible_frame_counts(
        sample, [edge_traj], template, empty_ctx, fov_margin_deg=5.0
    ) == [0]
    assert not meets_min_visible_frames(
        sample, [edge_traj], template, empty_ctx,
        min_visible_frames=1, fov_margin_deg=5.0,
    )


def test_review_joint_visible_gate_requires_same_frames():
    template = _stub_template()
    sample = SceneSample(
        mic_pos_m=(0.0, 0.0, 1.0),
        mic_yaw_deg=0.0,
        source_specs=[{"tag": "dog_beagle_v2"}, {"tag": "cat_british_shorthair_v2"}],
    )
    front = [2.0, 0.0, 1.0]
    behind = [-2.0, 0.0, 1.0]
    first_visible = np.array([front, front, front, behind, behind, behind])
    second_visible = np.array([behind, behind, behind, front, front, front])
    empty_ctx = {"furniture_bboxes": [], "wall_bboxes": []}

    assert visible_frame_counts(
        sample, [first_visible, second_visible], template, empty_ctx,
    ) == [3, 3]
    assert joint_visible_frame_count(
        sample, [first_visible, second_visible], template, empty_ctx,
    ) == 0
    assert meets_min_visible_frames(
        sample, [first_visible, second_visible], template, empty_ctx,
        min_visible_frames=3,
    )
    assert not meets_min_visible_frames(
        sample, [first_visible, second_visible], template, empty_ctx,
        min_visible_frames=3, min_joint_visible_frames=1,
    )
