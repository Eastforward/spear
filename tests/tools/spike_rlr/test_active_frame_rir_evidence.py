import numpy as np
import pytest

from tools.spike_rlr.active_frame_rir_evidence import (
    active_frame_indices,
    load_active_frame_rir_evidence,
    replay_active_frame_rirs,
    serialize_active_frame_rir_evidence,
)


def _fixture():
    dry = np.zeros(16, dtype=np.float32)
    dry[4:8] = np.asarray([0.25, -0.5, 0.75, -0.25], dtype=np.float32)
    dry[12:16] = np.asarray([0.1, 0.2, -0.2, -0.1], dtype=np.float32)
    trajectory = np.asarray(
        [[float(index), 1.0, 0.5] for index in range(4)],
        dtype=np.float64,
    )
    frame_indices = active_frame_indices(
        dry,
        n_frames=4,
        samples_per_frame=4,
    )
    rirs = [
        np.asarray([[1.0, 0.25], [0.5, 0.1]], dtype=np.float32),
        np.asarray([[0.8], [0.4]], dtype=np.float32),
    ]
    payload = serialize_active_frame_rir_evidence(
        source_tag="stable_test_animal",
        frame_indices=frame_indices,
        rirs=rirs,
        source_positions_scene_m=trajectory[frame_indices],
        mic_position_scene_m=np.asarray([0.0, 0.0, 1.2]),
        mic_yaw_deg=90.0,
        sample_rate_hz=16,
        n_samples_total=16,
        n_frames=4,
        fps=4.0,
        samples_per_frame=4,
    )
    return dry, trajectory, rirs, payload


def test_active_frame_rir_evidence_replays_exact_convolve_and_ola():
    dry, trajectory, rirs, payload = _fixture()
    evidence = load_active_frame_rir_evidence(
        payload,
        source_tag="stable_test_animal",
        dry=dry,
        expected_source_trajectory_scene_m=trajectory,
        mic_position_scene_m=np.asarray([0.0, 0.0, 1.2]),
        mic_yaw_deg=90.0,
        sample_rate_hz=16,
        n_samples_total=16,
        n_frames=4,
        fps=4.0,
        samples_per_frame=4,
    )

    observed = replay_active_frame_rirs(
        dry,
        evidence,
        n_samples_total=16,
        samples_per_frame=4,
    )
    expected = np.zeros((2, 16), dtype=np.float32)
    for frame_index, rir in zip((1, 3), rirs):
        start = frame_index * 4
        chunk = dry[start : start + 4]
        for channel in range(2):
            convolved = np.convolve(chunk, rir[channel], mode="full")
            end = min(16, start + len(convolved))
            expected[channel, start:end] += convolved[: end - start]
    np.testing.assert_array_equal(observed, expected)


def test_active_frame_rir_evidence_rejects_missing_frame_and_channel_swap():
    dry, trajectory, rirs, _payload = _fixture()
    missing_frame = serialize_active_frame_rir_evidence(
        source_tag="stable_test_animal",
        frame_indices=[1],
        rirs=[rirs[0]],
        source_positions_scene_m=trajectory[[1]],
        mic_position_scene_m=np.asarray([0.0, 0.0, 1.2]),
        mic_yaw_deg=90.0,
        sample_rate_hz=16,
        n_samples_total=16,
        n_frames=4,
        fps=4.0,
        samples_per_frame=4,
    )
    with pytest.raises(ValueError, match="active-frame payload"):
        load_active_frame_rir_evidence(
            missing_frame,
            source_tag="stable_test_animal",
            dry=dry,
            expected_source_trajectory_scene_m=trajectory,
            mic_position_scene_m=np.asarray([0.0, 0.0, 1.2]),
            mic_yaw_deg=90.0,
            sample_rate_hz=16,
            n_samples_total=16,
            n_frames=4,
            fps=4.0,
            samples_per_frame=4,
        )

    with pytest.raises(ValueError, match="render configuration"):
        serialize_active_frame_rir_evidence(
            source_tag="stable_test_animal",
            frame_indices=[1, 3],
            rirs=rirs,
            source_positions_scene_m=trajectory[[1, 3]],
            mic_position_scene_m=np.asarray([0.0, 0.0, 1.2]),
            mic_yaw_deg=90.0,
            sample_rate_hz=16,
            n_samples_total=16,
            n_frames=4,
            fps=4.0,
            samples_per_frame=4,
            channel_order=(1, 0),
        )
