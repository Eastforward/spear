"""Energy-aware scheduling for short animal vocalizations.

Animal calls are events, not stationary ambience.  A one-bark recording is
trimmed by an RMS threshold and placed several times across the requested
clip with deterministic silence gaps.  Long recordings are played once and
silence-padded; nothing is tiled sample-contiguously.
"""
from __future__ import annotations

import math

import numpy as np


def _frame_rms(signal: np.ndarray, frame_samples: int) -> np.ndarray:
    if len(signal) == 0:
        return np.zeros(0, dtype=np.float64)
    frame_count = int(math.ceil(len(signal) / frame_samples))
    padded = np.pad(signal, (0, frame_count * frame_samples - len(signal)))
    frames = padded.reshape(frame_count, frame_samples).astype(np.float64)
    return np.sqrt(np.mean(frames * frames, axis=1))


def prepare_animal_call(
    signal,
    *,
    sample_rate: int,
    duration_s: float,
    rng,
    frame_ms: float = 20.0,
    threshold_below_peak_db: float = 32.0,
    minimum_silence_gap_s: float = 0.85,
    target_silence_gap_s: float = 1.55,
    edge_margin_s: float = 0.20,
) -> tuple[np.ndarray, dict]:
    """Return a fixed-duration animal track and an auditable event schedule."""
    sample_rate = int(sample_rate)
    duration_s = float(duration_s)
    if sample_rate <= 0 or duration_s <= 0.0:
        raise ValueError("sample_rate and duration_s must be positive")
    source = np.asarray(signal, dtype=np.float32).reshape(-1)
    target_samples = int(round(sample_rate * duration_s))
    output = np.zeros(target_samples, dtype=np.float32)
    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    rms = _frame_rms(source, frame_samples)
    peak_rms = float(rms.max()) if len(rms) else 0.0
    noise_rms = float(np.percentile(rms, 20.0)) if len(rms) else 0.0
    relative_floor = peak_rms * 10.0 ** (-float(threshold_below_peak_db) / 20.0)
    # On a continuous vocalization the 20th percentile can be close to the
    # peak; cap the noise-derived threshold so it cannot classify the entire
    # non-silent recording as silence.
    noise_threshold = min(noise_rms * 2.5, peak_rms * 0.8)
    threshold_rms = max(1e-5, relative_floor, noise_threshold)
    active = rms >= threshold_rms

    metadata = {
        "schema": "animal_audio_event_schedule_v1",
        "source_samples": int(len(source)),
        "source_duration_s": len(source) / sample_rate,
        "target_duration_s": duration_s,
        "sample_rate_hz": sample_rate,
        "analysis_frame_ms": float(frame_ms),
        "peak_rms": peak_rms,
        "noise_rms_p20": noise_rms,
        "threshold_rms": threshold_rms,
        "threshold_below_peak_db": float(threshold_below_peak_db),
        "events": [],
    }
    if not np.any(active):
        metadata.update(
            {
                "mode": "silence",
                "active_duration_s": 0.0,
                "event_duration_s": 0.0,
                "event_count": 0,
            }
        )
        return output, metadata

    active_indices = np.flatnonzero(active)
    # Active regions separated by at least 300 ms are distinct calls.  This
    # turns a ten-second source containing three isolated barks into three
    # reusable events instead of preserving eight seconds of embedded silence.
    merge_gap_frames = max(1, int(round(0.30 * sample_rate / frame_samples)))
    split_points = np.flatnonzero(np.diff(active_indices) > merge_gap_frames) + 1
    active_groups = np.split(active_indices, split_points)
    pad_frames = max(1, int(round(0.04 * sample_rate / frame_samples)))
    source_events = []
    source_event_meta = []
    for group_index, group in enumerate(active_groups):
        first_frame = int(group[0])
        last_frame = int(group[-1])
        crop_start = max(0, (first_frame - pad_frames) * frame_samples)
        crop_end = min(len(source), (last_frame + 1 + pad_frames) * frame_samples)
        event = source[crop_start:crop_end].copy()
        fade_samples = min(
            len(event) // 2,
            max(1, int(round(0.008 * sample_rate))),
        )
        if fade_samples:
            fade = np.linspace(
                0.0,
                1.0,
                fade_samples,
                endpoint=False,
                dtype=np.float32,
            )
            event[:fade_samples] *= fade
            event[-fade_samples:] *= fade[::-1]
        source_events.append(event)
        source_event_meta.append(
            {
                "source_event_index": group_index,
                "crop_start_s": crop_start / sample_rate,
                "crop_end_s": crop_end / sample_rate,
                "duration_s": len(event) / sample_rate,
            }
        )

    active_duration_s = float(active.sum() * frame_samples / sample_rate)
    event_durations_s = [len(event) / sample_rate for event in source_events]
    median_event_duration_s = float(np.median(event_durations_s))
    short_call = median_event_duration_s <= min(2.25, duration_s * 0.42)
    margin_samples = max(0, int(round(edge_margin_s * sample_rate)))
    usable_samples = max(1, target_samples - 2 * margin_samples)
    min_gap_samples = int(round(minimum_silence_gap_s * sample_rate))
    target_gap_samples = int(round(target_silence_gap_s * sample_rate))

    if not short_call:
        # A continuous long call is played once, then silence-padded.
        event_sequence = [source_events[0][:usable_samples]]
        mode = "single_event_silence_padded"
    else:
        mean_event_samples = max(
            1,
            int(round(np.mean([len(event) for event in source_events]))),
        )
        desired_count = max(
            2,
            int(
                math.floor(
                    (usable_samples + target_gap_samples)
                    / (mean_event_samples + target_gap_samples)
                )
            ),
        )
        desired_count = min(desired_count, 12)
        event_sequence = [
            source_events[index % len(source_events)]
            for index in range(desired_count)
        ]
        while len(event_sequence) > 1 and (
            sum(len(event) for event in event_sequence)
            + min_gap_samples * (len(event_sequence) - 1)
            > usable_samples
        ):
            event_sequence.pop()
        mode = (
            "repeated_events_with_silence_gaps"
            if len(event_sequence) > 1
            else "single_event_silence_padded"
        )

    total_event_samples = sum(len(event) for event in event_sequence)
    if len(event_sequence) > 1:
        minimum_total_gap = min_gap_samples * (len(event_sequence) - 1)
        extra_gap = max(0, usable_samples - total_event_samples - minimum_total_gap)
        weights = rng.uniform(0.8, 1.2, len(event_sequence) - 1)
        weights = weights / weights.sum()
        gaps = [
            min_gap_samples + int(round(extra_gap * weight))
            for weight in weights
        ]
        # Rounding can move a few samples past the right margin.
        overflow = total_event_samples + sum(gaps) - usable_samples
        if overflow > 0:
            gaps[-1] = max(min_gap_samples, gaps[-1] - overflow)
    else:
        gaps = []

    starts = []
    cursor = min(margin_samples, max(0, target_samples - len(event_sequence[0])))
    for index, event in enumerate(event_sequence):
        starts.append(cursor)
        cursor += len(event)
        if index < len(gaps):
            cursor += gaps[index]

    for event_index, (start, event) in enumerate(zip(starts, event_sequence)):
        end = min(start + len(event), target_samples)
        output[start:end] += event[: end - start]
        metadata["events"].append(
            {
                "index": event_index,
                "source_event_index": event_index % len(source_events),
                "start_sample": int(start),
                "end_sample": int(end),
                "start_s": start / sample_rate,
                "end_s": end / sample_rate,
            }
        )

    metadata.update(
        {
            "mode": mode,
            "source_events": source_event_meta,
            "source_event_count": len(source_events),
            "active_duration_s": active_duration_s,
            "event_duration_s": median_event_duration_s,
            "median_event_duration_s": median_event_duration_s,
            "short_call_detected": bool(short_call),
            "minimum_silence_gap_s": float(minimum_silence_gap_s),
            "event_count": len(starts),
        }
    )
    return output, metadata
