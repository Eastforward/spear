import sys
from pathlib import Path

import numpy as np


SPEAR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPEAR_ROOT / "tools"))

from audio_event_schedule import prepare_animal_call


def test_short_bark_is_repeated_with_real_silence_gaps():
    sample_rate = 1000
    source = np.zeros(2000, dtype=np.float32)
    source[400:650] = np.hanning(250).astype(np.float32)

    output, metadata = prepare_animal_call(
        source,
        sample_rate=sample_rate,
        duration_s=6.0,
        rng=np.random.default_rng(7),
    )

    assert len(output) == 6000
    assert metadata["mode"] == "repeated_events_with_silence_gaps"
    assert metadata["event_count"] >= 3
    assert metadata["short_call_detected"] is True
    starts = [event["start_sample"] for event in metadata["events"]]
    ends = [event["end_sample"] for event in metadata["events"]]
    assert all(b - a >= 850 for a, b in zip(ends[:-1], starts[1:]))


def test_long_call_is_played_once_and_silence_padded_not_tiled():
    sample_rate = 1000
    source = np.sin(2 * np.pi * 30 * np.arange(4000) / sample_rate).astype(np.float32)

    output, metadata = prepare_animal_call(
        source,
        sample_rate=sample_rate,
        duration_s=6.0,
        rng=np.random.default_rng(9),
    )

    assert len(output) == 6000
    assert metadata["mode"] == "single_event_silence_padded"
    assert metadata["event_count"] == 1
    assert np.count_nonzero(output[-1000:]) == 0


def test_silent_input_stays_silent():
    output, metadata = prepare_animal_call(
        np.zeros(1000, dtype=np.float32),
        sample_rate=1000,
        duration_s=3.0,
        rng=np.random.default_rng(0),
    )

    assert not np.any(output)
    assert metadata["mode"] == "silence"
    assert metadata["event_count"] == 0
