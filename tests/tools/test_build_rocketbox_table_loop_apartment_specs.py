import math

import numpy as np

from tools.build_rocketbox_table_loop_apartment_specs import (
    build_rounded_table_loop,
    build_table_loop_spec,
)


def _template():
    return {
        "spec_version": "apartment_v1",
        "description": "old linear review",
        "usage_scope": "research_candidate",
        "camera_configs": [
            {
                "name": "view0",
                "pos_m": [0.5, 0.15, 1.2],
                "yaw_deg": 145.0,
                "fov_deg": 75.0,
            }
        ],
        "render_config": {
            "width": 960,
            "height": 720,
            "fps": 15,
            "n_frames": 75,
            "duration_s": 5.0,
            "streaming_warmup_frames": 120,
            "camera_warmup_frames": 40,
        },
        "sources": [
            {
                "tag": "rocketbox_adults_male_adult_01_original_ue_v1",
                "asset_id": "rocketbox_male_adult_01",
                "actor_scale": 1.0,
                "start_pos_m": [-1.68, 1.55, 0.0],
                "end_pos_m": [-2.6, 2.45, 0.0],
                "motion": "linear_uniform",
                "wanted_anim": "Walking",
                "walking_forward_yaw_offset_deg": 90.0,
            }
        ],
    }


def _point_to_bbox_clearance(point, bbox):
    x, y = point
    x0, y0, x1, y1 = bbox
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def test_rounded_table_loop_is_closed_constant_speed_and_clears_furniture():
    trajectory = build_rounded_table_loop(n_frames=181)

    assert trajectory.shape == (181, 3)
    assert np.allclose(trajectory[0], trajectory[-1])
    assert np.allclose(trajectory[:, 2], 0.0)
    assert trajectory[:, 0].min() <= -3.27
    assert trajectory[:, 0].max() >= 0.01
    assert trajectory[:, 1].min() <= 2.56
    assert trajectory[:, 1].max() >= 6.14

    step_lengths = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
    assert step_lengths.min() > 0.0
    assert step_lengths.max() / step_lengths.min() < 1.04

    obstacles = {
        "round_table": (-2.832, 3.413, -0.633, 5.611),
        "round_table_chair": (-2.700, 3.127, -1.753, 4.061),
        "shelf": (-4.285, 4.125, -3.730, 5.478),
    }
    for bbox in obstacles.values():
        clearance = min(
            _point_to_bbox_clearance(point, bbox)
            for point in trajectory[:, :2]
        )
        assert clearance >= 0.44


def test_table_loop_spec_changes_only_scene_path_timing_and_review_fov():
    template = _template()

    spec = build_table_loop_spec(
        template,
        role_label="adult_male",
        fps=15,
        duration_s=12.0,
    )

    source = spec["sources"][0]
    assert template["sources"][0]["motion"] == "linear_uniform"
    assert spec["render_config"]["n_frames"] == 180
    assert spec["render_config"]["duration_s"] == 12.0
    assert spec["camera_configs"][0]["fov_deg"] == 105.0
    assert source["tag"] == template["sources"][0]["tag"]
    assert source["asset_id"] == template["sources"][0]["asset_id"]
    assert source["actor_scale"] == 1.0
    assert source["wanted_anim"] == "Walking"
    assert source["motion"] == "explicit_rounded_table_loop"
    assert len(source["trajectory_m"]) == 180
    assert source["start_pos_m"] == source["end_pos_m"]
    assert spec["table_loop_contract"]["target_actor"].endswith(
        "Round_Table:SM_table_circular_polySurface65_47"
    )
    assert 1.0 <= spec["table_loop_contract"]["average_speed_mps"] <= 1.2
    assert len(spec["rig_direction_check_windows"]) == 4
