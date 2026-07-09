import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _source(spec, tag):
    return next(src for src in spec["sources"] if src["tag"] == tag)


def test_front_idle_rear_left_to_right_demo_spec_constraints_pass():
    from demo_scenarios import compose_front_idle_rear_left_to_right_demo
    from event_constraints import (
        constraint_behind_camera,
        constraint_left_to_right,
        constraint_not_visible,
        constraint_stationary,
        verify_constraints,
    )

    spec = compose_front_idle_rear_left_to_right_demo(
        REPO / "data" / "apartment_v1_spec.json"
    )
    front = _source(spec, "dog_golden")
    rear = _source(spec, "dog_beagle_v2")
    mic = spec["mic"]["pos_m"]
    yaw = spec["mic"]["yaw_deg"]

    front_traj = np.asarray(front["trajectory_m"], dtype=float)
    rear_traj = np.asarray(rear["trajectory_m"], dtype=float)
    report = verify_constraints([
        constraint_stationary("dog_golden", front_traj),
        constraint_behind_camera("dog_beagle_v2", rear_traj, mic, yaw),
        constraint_not_visible("dog_beagle_v2", rear_traj, mic, yaw),
        constraint_left_to_right("dog_beagle_v2", rear_traj, mic, yaw),
    ])

    assert report["passed"], report
    assert spec["event_constraint_report"]["passed"]


def test_front_idle_rear_left_to_right_demo_mutes_front_and_uses_rear_dog_sound():
    from demo_scenarios import compose_front_idle_rear_left_to_right_demo

    spec = compose_front_idle_rear_left_to_right_demo(
        REPO / "data" / "apartment_v1_spec.json"
    )
    front = _source(spec, "dog_golden")
    rear = _source(spec, "dog_beagle_v2")

    assert front["wanted_anim"] == "Idle"
    assert front["motion_style"] == "stationary"
    assert rear["wanted_anim"] == "Walking"
    assert front["audio_lookup"] == "silent"
    assert front["mute_audio"] is True
    assert rear["audio_lookup"] == "dog_sharp_bark"
    assert Path(rear["audio_path"]).exists()
    assert rear["audio_clip_start_s"] >= 0.0
    assert rear["audio_clip_duration_s"] > 0.0
    assert rear["audio_repeat_interval_s"] >= rear["audio_clip_duration_s"]


def test_front_idle_rear_left_to_right_demo_has_strong_listener_local_sweep():
    from demo_scenarios import compose_front_idle_rear_left_to_right_demo
    from event_constraints import listener_local_xy

    spec = compose_front_idle_rear_left_to_right_demo(
        REPO / "data" / "apartment_v1_spec.json"
    )
    rear = _source(spec, "dog_beagle_v2")
    local = listener_local_xy(
        np.asarray(rear["trajectory_m"], dtype=float),
        spec["mic"]["pos_m"],
        spec["mic"]["yaw_deg"],
    )

    assert local[0, 1] >= 1.8
    assert local[-1, 1] <= -3.5
    assert local[:, 0].max() < -0.05


def test_front_idle_left_rear_to_right_front_demo_enters_camera_right_side():
    from demo_scenarios import compose_front_idle_left_rear_to_right_front_demo
    from event_constraints import listener_local_xy
    from visibility import batch_frame_visibility

    spec = compose_front_idle_left_rear_to_right_front_demo(
        REPO / "data" / "apartment_v1_spec.json"
    )
    rear = _source(spec, "dog_beagle_v2")
    rear_traj = np.asarray(rear["trajectory_m"], dtype=float)
    local = listener_local_xy(
        rear_traj,
        spec["mic"]["pos_m"],
        spec["mic"]["yaw_deg"],
    )
    vis = batch_frame_visibility(
        rear_traj,
        spec["mic"]["pos_m"],
        spec["mic"]["yaw_deg"],
        fov_h_deg=spec["camera_configs"][0]["fov_deg"],
        fov_v_deg=60.0,
        obstacles_xyz=None,
    )
    path_m = float(np.linalg.norm(np.diff(rear_traj[:, :2], axis=0), axis=1).sum())
    duration_s = spec["render_config"]["n_frames"] / spec["render_config"]["fps"]

    assert spec["source_collision_policy"] == "walls_only_center"
    assert local[0, 0] < -0.5
    assert local[0, 1] > 1.0
    assert not bool(vis["in_fov"][0])
    assert local[-1, 0] > 1.0
    assert local[-1, 1] < -0.8
    assert bool(vis["in_fov"][-1])
    assert int(np.count_nonzero(vis["in_fov"])) >= 18
    assert path_m / duration_s <= 1.6


def test_front_idle_left_rear_to_right_front_demo_mutes_front_and_uses_rear_dog_sound():
    from demo_scenarios import compose_front_idle_left_rear_to_right_front_demo

    spec = compose_front_idle_left_rear_to_right_front_demo(
        REPO / "data" / "apartment_v1_spec.json"
    )
    front = _source(spec, "dog_golden")
    rear = _source(spec, "dog_beagle_v2")

    assert front["audio_lookup"] == "silent"
    assert front["mute_audio"] is True
    assert front["wanted_anim"] == "Idle"
    assert rear["audio_lookup"] == "dog_sharp_bark"
    assert rear["wanted_anim"] == "Walking"
    assert Path(rear["audio_path"]).exists()


def test_demo_flags_are_computed_from_generated_spec():
    from demo_scenarios import (
        compose_front_idle_left_rear_to_right_front_demo,
        compute_demo_flags,
    )
    from flag_definitions import ALL_FLAGS

    spec = compose_front_idle_left_rear_to_right_front_demo(
        REPO / "data" / "apartment_v1_spec.json"
    )

    flags = compute_demo_flags(spec)

    assert set(flags) == set(ALL_FLAGS)
    assert flags["leaves_camera_fov"] is True
    assert flags["crosses_azimuth_zero"] is True
    assert flags["sources_pass_each_other"] is False
