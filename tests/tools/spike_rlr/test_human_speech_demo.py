import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _write_wav(path: Path, duration_s: float = 2.0, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(duration_s * sample_rate), dtype=np.float32) / sample_rate
    y = 0.15 * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(path, y, sample_rate)


def _source(spec, tag):
    return next(src for src in spec["sources"] if src["tag"] == tag)


def test_visible_human_speech_demo_uses_registered_human_and_real_speech(tmp_path):
    from demo_scenarios import compose_visible_human_speech_demo
    from event_constraints import constraint_front_of_camera, constraint_stationary
    from event_constraints import verify_constraints
    from scene_two_dogs_apartment import compose_two_dog_scene_apartment
    from visibility import batch_frame_visibility

    speech_root = tmp_path / "LibriTTS"
    wav = (
        speech_root
        / "train-clean-100"
        / "1234"
        / "5678"
        / "1234_5678_000001_000000.wav"
    )
    _write_wav(wav, duration_s=2.0)
    wav.with_suffix(".normalized.txt").write_text(
        "This is a visible human speech demo.\n",
        encoding="utf-8",
    )

    out_spec = tmp_path / "clip" / "spec.json"
    spec = compose_visible_human_speech_demo(
        REPO / "data" / "apartment_v1_spec.json",
        out_spec_path=out_spec,
        speech_root=speech_root,
    )

    src = _source(spec, "human_male_blue_hoodie_v1")
    traj = np.asarray(src["trajectory_m"], dtype=float)
    report = verify_constraints([
        constraint_front_of_camera(src["tag"], traj, spec["mic"]["pos_m"], spec["mic"]["yaw_deg"]),
        constraint_stationary(src["tag"], traj),
    ])
    vis = batch_frame_visibility(
        traj,
        spec["mic"]["pos_m"],
        spec["mic"]["yaw_deg"],
        fov_h_deg=spec["camera_configs"][0]["fov_deg"],
        fov_v_deg=60.0,
        obstacles_xyz=None,
    )

    assert report["passed"], report
    assert bool(vis["in_fov"].all())
    assert src["asset_id"] == "human_male_blue_hoodie_0001"
    assert src["asset_class"] == "human"
    assert src["category"] == "human"
    assert src["audio_lookup"] == "speech"
    assert src["audio_path"] == str(wav)
    assert src["wanted_anim"] == "Standing_Idle"
    assert src["motion_style"] == "stationary"
    assert "facing_yaw_deg" in src
    assert src["actor_scale"] == 1.0
    assert src["actor_z_lift_cm"] == 14.0
    assert src["walking_forward_yaw_offset_deg"] == 90.0
    assert src["source_role"] == "visible_human_speaker"
    assert spec["event_constraint_report"]["passed"]

    scene = compose_two_dog_scene_apartment(out_spec)
    placement = scene.animals[0]
    expected_yaw = (
        float(src["facing_yaw_deg"]) + float(src["walking_forward_yaw_offset_deg"])
    ) % 360.0
    assert np.allclose(placement.yaw_deg, expected_yaw)
    # Keep the human away from the old yaw=180 kitchen-island view where the
    # head source was in FOV but the body was hidden in the render.
    assert spec["mic"]["yaw_deg"] == 140.0
    assert src["start_pos_m"] == [-1.5, 1.5, 1.55]


def test_visible_moving_human_speech_demo_walks_left_to_right_with_real_speech(tmp_path):
    from demo_scenarios import compose_visible_moving_human_speech_demo
    from event_constraints import (
        constraint_front_of_camera,
        constraint_in_fov_min_frames,
        constraint_left_to_right,
        verify_constraints,
    )
    from scene_two_dogs_apartment import compose_two_dog_scene_apartment

    speech_root = tmp_path / "LibriTTS"
    wav = (
        speech_root
        / "train-clean-100"
        / "1234"
        / "5678"
        / "1234_5678_000001_000001.wav"
    )
    _write_wav(wav, duration_s=2.0)
    wav.with_suffix(".normalized.txt").write_text(
        "This is a moving human speech demo.\n",
        encoding="utf-8",
    )

    out_spec = tmp_path / "clip" / "spec.json"
    spec = compose_visible_moving_human_speech_demo(
        REPO / "data" / "apartment_v1_spec.json",
        out_spec_path=out_spec,
        speech_root=speech_root,
    )

    src = _source(spec, "human_male_blue_hoodie_v1")
    traj = np.asarray(src["trajectory_m"], dtype=float)
    report = verify_constraints([
        constraint_front_of_camera(src["tag"], traj, spec["mic"]["pos_m"], spec["mic"]["yaw_deg"]),
        constraint_in_fov_min_frames(src["tag"], traj, spec["mic"]["pos_m"], spec["mic"]["yaw_deg"], 75),
        constraint_left_to_right(src["tag"], traj, spec["mic"]["pos_m"], spec["mic"]["yaw_deg"], margin_m=0.8),
    ])

    assert report["passed"], report
    assert src["asset_id"] == "human_male_blue_hoodie_0001"
    assert src["audio_lookup"] == "speech"
    assert src["audio_path"] == str(wav)
    assert src["wanted_anim"] == "Walking"
    assert src["motion_style"] == "walking"
    assert src["source_role"] == "visible_moving_human_speaker"
    assert src["actor_scale"] == 1.0
    assert src["actor_z_lift_cm"] == 14.0
    assert "facing_yaw_deg" not in src
    assert spec["event_constraint_report"]["passed"]

    scene = compose_two_dog_scene_apartment(out_spec)
    placement = scene.animals[0]
    assert placement.wanted_anim == "Walking"
    assert float(np.ptp(placement.yaw_deg)) < 0.01
    assert not np.allclose(traj[0, :2], traj[-1, :2])


def test_human_visual_marker_falls_back_when_ue_bounds_are_implausible():
    from gpurir_scenes.scene_spec import AnimalPlacement
    from run_render_pass_apartment import _sanitize_actor_visual_center_ssot_m

    placement = AnimalPlacement(
        tag="human_male_blue_hoodie_v1",
        is_animated=True,
        trajectory_m=np.asarray([[-1.8, 0.6, 1.55]], dtype=float),
        yaw_deg=np.asarray([0.0], dtype=float),
    )

    bad_center = (7.5, -5.2, 57.8)
    assert _sanitize_actor_visual_center_ssot_m(bad_center, placement, 0) == [
        -1.8,
        0.6,
        1.55,
    ]

    good_center = (-1.76, 0.62, 0.95)
    assert _sanitize_actor_visual_center_ssot_m(good_center, placement, 0) == [
        -1.76,
        0.62,
        0.95,
    ]
