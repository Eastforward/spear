"""Reconstruct and authenticate scheduled dry animal audio evidence.

This module deliberately does not import the RLR runner.  It closes the gap
between an authenticated pinned mono PCM16 source, one
``animal_audio_event_schedule_v1`` record, and a materialized mono PCM16
scheduled-dry WAV.  The reconstruction mirrors the current scheduling path:

* PCM16 -> float32 decoding;
* ``np.interp`` resampling;
* source-event crops;
* the scheduler's 8 ms float32 edge fades;
* placement by ``events[*].source_event_index``; and
* final peak normalization to 0.8.

The validator compares every PCM sample against the deterministic
reconstruction.  A waveform that merely has plausible duration, energy, or
event timing is therefore not accepted.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import re
from typing import Any, Mapping
import wave

import numpy as np

if __package__:
    from .animal_audio import (
        PINNED_AUDIO_CONTRACT_SCHEMA,
        load_authenticated_file_bytes,
    )
else:  # pragma: no cover - supports direct imports used by legacy tools
    from animal_audio import (  # type: ignore
        PINNED_AUDIO_CONTRACT_SCHEMA,
        load_authenticated_file_bytes,
    )


SCHEDULE_SCHEMA = "animal_audio_event_schedule_v1"
VALIDATION_SCHEMA = "avengine_scheduled_dry_pcm16_validation_v1"
PCM16_QUANTIZATION_TOLERANCE_LSB = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEDULE_CONTRACT_FIELDS = {
    "audio_lookup": "audio_lookup",
    "source_species": "species",
    "source_sha256": "sha256",
    "source_codec": "codec",
    "source_channels": "channels",
    "source_sample_width_bytes": "sample_width_bytes",
    "source_original_sample_rate_hz": "sample_rate_hz",
    "source_original_frame_count": "frame_count",
    "source_original_channels": "channels",
    "source_catalog_frame_count": "frame_count",
    "source_catalog_duration_s": "duration_s",
    "dry_source_policy": "dry_source_policy",
    "spatialization_status": "spatialization_status",
    "known_spatialized_derivative_sha256": (
        "known_spatialized_derivative_sha256"
    ),
    "item_origin": "item_origin",
    "objective_audio_content_qa_status": "objective_audio_content_qa_status",
    "item_level_license_status": "item_level_license_status",
    "item_level_license_snapshot": "item_level_license_snapshot",
    "formal_registration_authorized": "formal_registration_authorized",
}


class ScheduledDryEvidenceError(ValueError):
    """Raised when scheduled-dry evidence is malformed or does not match."""


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScheduledDryEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
    ):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise ScheduledDryEvidenceError(f"{label} must be finite{suffix}")
    return float(value)


def _authenticated_descriptor_bytes(
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, bytes, str]:
    if not isinstance(descriptor, Mapping):
        raise ScheduledDryEvidenceError(f"{label} descriptor must be an object")
    configured_path = descriptor.get("path")
    expected_sha256 = descriptor.get("sha256")
    expected_size_bytes = descriptor.get("size_bytes")
    if not isinstance(configured_path, str) or not Path(configured_path).is_absolute():
        raise ScheduledDryEvidenceError(f"{label} descriptor path must be absolute")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise ScheduledDryEvidenceError(f"{label} descriptor SHA-256 is invalid")
    size_bytes = _require_int(
        expected_size_bytes,
        f"{label} descriptor size_bytes",
        minimum=1,
    )
    try:
        payload = load_authenticated_file_bytes(Path(configured_path))
    except (OSError, ValueError) as error:
        raise ScheduledDryEvidenceError(
            f"{label} descriptor path is not an authenticated regular file"
        ) from error
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != size_bytes or observed_sha256 != expected_sha256:
        raise ScheduledDryEvidenceError(f"{label} descriptor bytes changed")
    return Path(configured_path), payload, observed_sha256


def _validate_source_contract(
    source_contract: Mapping[str, Any],
) -> tuple[bytes, np.ndarray, int]:
    if not isinstance(source_contract, Mapping):
        raise ScheduledDryEvidenceError("source_contract must be an object")
    if source_contract.get("schema") != PINNED_AUDIO_CONTRACT_SCHEMA:
        raise ScheduledDryEvidenceError("source contract schema is not pinned")
    if source_contract.get("codec") != "pcm_s16le":
        raise ScheduledDryEvidenceError("pinned source must use pcm_s16le")
    if (
        _require_int(
            source_contract.get("channels"),
            "source channels",
            minimum=1,
        )
        != 1
        or _require_int(
            source_contract.get("sample_width_bytes"),
            "source sample_width_bytes",
            minimum=1,
        )
        != 2
        or source_contract.get("dry_source_policy")
        != "mono_no_hrtf_no_pre_spatialization"
        or source_contract.get("spatialization_status")
        != "mono_unspatialized_dry_source"
    ):
        raise ScheduledDryEvidenceError(
            "pinned source is not mono, PCM16, and unspatialized"
        )
    if (
        source_contract.get("item_level_license_status") != "missing"
        or source_contract.get("item_level_license_snapshot") is not None
        or source_contract.get("formal_registration_authorized") is not False
    ):
        raise ScheduledDryEvidenceError(
            "pinned v1 source changed its nonformal rights classification"
        )
    if (
        not isinstance(source_contract.get("audio_lookup"), str)
        or not source_contract["audio_lookup"]
        or not isinstance(source_contract.get("species"), str)
        or not source_contract["species"]
        or not isinstance(source_contract.get("item_origin"), Mapping)
        or not source_contract["item_origin"]
        or not isinstance(
            source_contract.get("objective_audio_content_qa_status"),
            str,
        )
        or not source_contract["objective_audio_content_qa_status"]
    ):
        raise ScheduledDryEvidenceError("pinned source identity is incomplete")
    known_derivative = source_contract.get(
        "known_spatialized_derivative_sha256"
    )
    if not isinstance(known_derivative, str) or not _SHA256_RE.fullmatch(
        known_derivative
    ):
        raise ScheduledDryEvidenceError(
            "pinned source known spatialized derivative SHA-256 is invalid"
        )

    source_rate = _require_int(
        source_contract.get("sample_rate_hz"),
        "source sample_rate_hz",
        minimum=1,
    )
    source_frame_count = _require_int(
        source_contract.get("frame_count"),
        "source frame_count",
        minimum=1,
    )
    source_duration_s = _require_finite_number(
        source_contract.get("duration_s"),
        "source duration_s",
        minimum=0.0,
    )
    if not math.isclose(
        source_duration_s,
        source_frame_count / source_rate,
        rel_tol=0.0,
        abs_tol=1.0 / source_rate,
    ):
        raise ScheduledDryEvidenceError("source duration and frame count differ")

    _source_path, payload, _digest = _authenticated_descriptor_bytes(
        source_contract,
        label="pinned source",
    )
    try:
        with wave.open(io.BytesIO(payload), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frame_count = stream.getnframes()
            compression = stream.getcomptype()
            frames = stream.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ScheduledDryEvidenceError(
            "pinned source is not a decodable PCM WAV"
        ) from error
    if (
        channels != 1
        or sample_width != 2
        or sample_rate != source_rate
        or frame_count != source_frame_count
        or compression != "NONE"
        or len(frames) != frame_count * sample_width
    ):
        raise ScheduledDryEvidenceError("pinned source WAV metadata changed")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    samples /= np.float32(32768.0)
    if (
        samples.shape != (source_frame_count,)
        or not np.all(np.isfinite(samples))
        or float(np.max(np.abs(samples))) < 1.0e-3
    ):
        raise ScheduledDryEvidenceError("pinned source samples are invalid")
    return payload, samples, source_rate


def _validate_schedule_contract_binding(
    source_contract: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> tuple[int, float]:
    if not isinstance(schedule, Mapping):
        raise ScheduledDryEvidenceError("schedule must be an object")
    if schedule.get("schema") != SCHEDULE_SCHEMA:
        raise ScheduledDryEvidenceError("schedule schema changed")
    if not isinstance(schedule.get("tag"), str) or not schedule["tag"]:
        raise ScheduledDryEvidenceError("schedule tag is missing")
    embedded = schedule.get("source_contract")
    if not isinstance(embedded, Mapping) or dict(embedded) != dict(source_contract):
        raise ScheduledDryEvidenceError(
            "schedule source_contract does not match the pinned source"
        )
    for schedule_field, contract_field in _SCHEDULE_CONTRACT_FIELDS.items():
        if schedule.get(schedule_field) != source_contract.get(contract_field):
            raise ScheduledDryEvidenceError(
                f"schedule {schedule_field} changed from the pinned source"
            )
    audio_path = schedule.get("audio_path")
    if (
        not isinstance(audio_path, str)
        or Path(audio_path).resolve() != Path(str(source_contract["path"])).resolve()
    ):
        raise ScheduledDryEvidenceError("schedule audio_path changed")
    if schedule.get("adaptive_repeat_short_calls") is not True:
        raise ScheduledDryEvidenceError(
            "schedule is not an adaptive animal-call schedule"
        )
    if schedule.get("mode") not in {
        "repeated_events_with_silence_gaps",
        "single_event_silence_padded",
    }:
        raise ScheduledDryEvidenceError("schedule mode is not reconstructable")

    render_rate = _require_int(
        schedule.get("render_sample_rate_hz"),
        "schedule render_sample_rate_hz",
        minimum=1,
    )
    if (
        _require_int(
            schedule.get("sample_rate_hz"),
            "schedule sample_rate_hz",
            minimum=1,
        )
        != render_rate
    ):
        raise ScheduledDryEvidenceError("schedule sample rates differ")
    target_duration_s = _require_finite_number(
        schedule.get("target_duration_s"),
        "schedule target_duration_s",
        minimum=0.0,
    )
    if target_duration_s <= 0.0:
        raise ScheduledDryEvidenceError("schedule target_duration_s must be positive")
    return render_rate, target_duration_s


def _resample_like_runner(
    source: np.ndarray,
    *,
    source_rate: int,
    render_rate: int,
) -> np.ndarray:
    if source_rate == render_rate:
        return source.copy()
    new_len = int(round(len(source) * render_rate / source_rate))
    if new_len <= 0:
        raise ScheduledDryEvidenceError("resampling produced no samples")
    return np.interp(
        np.linspace(0, len(source), new_len, endpoint=False),
        np.arange(len(source)),
        source,
    ).astype(np.float32)


def _source_event_crops(
    source: np.ndarray,
    *,
    render_rate: int,
    schedule: Mapping[str, Any],
) -> list[np.ndarray]:
    records = schedule.get("source_events")
    source_event_count = schedule.get("source_event_count")
    if (
        not isinstance(records, list)
        or not records
        or _require_int(
            source_event_count,
            "schedule source_event_count",
            minimum=1,
        )
        != len(records)
    ):
        raise ScheduledDryEvidenceError("schedule source event count changed")

    result: list[np.ndarray] = []
    previous_end = 0
    for expected_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ScheduledDryEvidenceError("source event must be an object")
        if (
            isinstance(record.get("source_event_index"), bool)
            or not isinstance(record.get("source_event_index"), int)
            or record.get("source_event_index") != expected_index
        ):
            raise ScheduledDryEvidenceError("source event index changed")
        crop_start_s = _require_finite_number(
            record.get("crop_start_s"),
            "source event crop_start_s",
            minimum=0.0,
        )
        crop_end_s = _require_finite_number(
            record.get("crop_end_s"),
            "source event crop_end_s",
            minimum=0.0,
        )
        duration_s = _require_finite_number(
            record.get("duration_s"),
            "source event duration_s",
            minimum=0.0,
        )
        crop_start = int(round(crop_start_s * render_rate))
        crop_end = int(round(crop_end_s * render_rate))
        if (
            crop_start < previous_end
            or crop_end <= crop_start
            or crop_end > len(source)
            or not math.isclose(
                crop_start_s,
                crop_start / render_rate,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                crop_end_s,
                crop_end / render_rate,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                duration_s,
                (crop_end - crop_start) / render_rate,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ScheduledDryEvidenceError("source event crop bounds changed")

        event = source[crop_start:crop_end].copy()
        fade_samples = min(
            len(event) // 2,
            max(1, int(round(0.008 * render_rate))),
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
        result.append(event)
        previous_end = crop_end
    return result


def _place_scheduled_events(
    source_events: list[np.ndarray],
    *,
    render_rate: int,
    target_duration_s: float,
    schedule: Mapping[str, Any],
) -> np.ndarray:
    records = schedule.get("events")
    event_count = schedule.get("event_count")
    if (
        not isinstance(records, list)
        or not records
        or _require_int(event_count, "schedule event_count", minimum=1)
        != len(records)
    ):
        raise ScheduledDryEvidenceError("schedule event count changed")
    target_samples = int(round(render_rate * target_duration_s))
    if target_samples <= 0:
        raise ScheduledDryEvidenceError("scheduled output has no samples")
    output = np.zeros(target_samples, dtype=np.float32)
    previous_end = 0
    mode = schedule["mode"]
    for expected_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ScheduledDryEvidenceError("scheduled event must be an object")
        source_event_index = record.get("source_event_index")
        if (
            isinstance(record.get("index"), bool)
            or not isinstance(record.get("index"), int)
            or record.get("index") != expected_index
            or isinstance(source_event_index, bool)
            or not isinstance(source_event_index, int)
            or not 0 <= source_event_index < len(source_events)
            or source_event_index != expected_index % len(source_events)
        ):
            raise ScheduledDryEvidenceError("scheduled event identity changed")
        start = _require_int(
            record.get("start_sample"),
            "scheduled event start_sample",
        )
        end = _require_int(
            record.get("end_sample"),
            "scheduled event end_sample",
            minimum=1,
        )
        start_s = _require_finite_number(
            record.get("start_s"),
            "scheduled event start_s",
            minimum=0.0,
        )
        end_s = _require_finite_number(
            record.get("end_s"),
            "scheduled event end_s",
            minimum=0.0,
        )
        event = source_events[source_event_index]
        scheduled_length = end - start
        if (
            start < previous_end
            or end <= start
            or end > target_samples
            or scheduled_length > len(event)
            or (
                mode == "repeated_events_with_silence_gaps"
                and scheduled_length != len(event)
            )
            or not math.isclose(
                start_s,
                start / render_rate,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                end_s,
                end / render_rate,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ScheduledDryEvidenceError("scheduled event bounds changed")
        output[start:end] += event[:scheduled_length]
        previous_end = end

    peak = float(np.max(np.abs(output)))
    if not math.isfinite(peak) or peak <= 1.0e-9:
        raise ScheduledDryEvidenceError("scheduled dry reconstruction is silent")
    output = output * (0.8 / peak)
    if not np.all(np.isfinite(output)):
        raise ScheduledDryEvidenceError("scheduled dry reconstruction is non-finite")
    return output.astype(np.float32, copy=False)


def reconstruct_scheduled_dry(
    *,
    source_contract: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> np.ndarray:
    """Return the runner-equivalent normalized scheduled dry float32 waveform."""
    _payload, source, source_rate = _validate_source_contract(source_contract)
    render_rate, target_duration_s = _validate_schedule_contract_binding(
        source_contract,
        schedule,
    )
    resampled = _resample_like_runner(
        source,
        source_rate=source_rate,
        render_rate=render_rate,
    )
    expected_source_samples = int(
        round(len(source) * render_rate / source_rate)
    )
    if (
        _require_int(
            schedule.get("source_samples"),
            "schedule source_samples",
            minimum=1,
        )
        != expected_source_samples
        or not math.isclose(
            _require_finite_number(
                schedule.get("source_duration_s"),
                "schedule source_duration_s",
                minimum=0.0,
            ),
            expected_source_samples / render_rate,
            rel_tol=0.0,
            abs_tol=1.0 / render_rate,
        )
    ):
        raise ScheduledDryEvidenceError("schedule resampled source length changed")
    source_events = _source_event_crops(
        resampled,
        render_rate=render_rate,
        schedule=schedule,
    )
    return _place_scheduled_events(
        source_events,
        render_rate=render_rate,
        target_duration_s=target_duration_s,
        schedule=schedule,
    )


def _decode_scheduled_pcm16(
    payload: bytes,
    *,
    sample_rate_hz: int,
    frame_count: int,
) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(payload), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            observed_rate = stream.getframerate()
            observed_frames = stream.getnframes()
            compression = stream.getcomptype()
            frames = stream.readframes(observed_frames)
    except (EOFError, wave.Error) as error:
        raise ScheduledDryEvidenceError(
            "scheduled dry evidence is not a decodable PCM WAV"
        ) from error
    if (
        channels != 1
        or sample_width != 2
        or compression != "NONE"
        or observed_rate != sample_rate_hz
        or observed_frames != frame_count
        or len(frames) != observed_frames * sample_width
    ):
        raise ScheduledDryEvidenceError(
            "scheduled dry WAV format or duration changed"
        )
    return np.frombuffer(frames, dtype="<i2").copy()


def validate_scheduled_dry_evidence(
    *,
    scheduled_wav_descriptor: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a mono PCM16 scheduled WAV and compare every sample.

    The comparison allows one signed-PCM16 least-significant bit, covering the
    two common float-to-PCM16 quantizers (floor and round-to-nearest).  The
    tolerance is intentionally fixed and cannot be weakened by callers.
    """
    reconstructed = reconstruct_scheduled_dry(
        source_contract=source_contract,
        schedule=schedule,
    )
    render_rate = int(schedule["render_sample_rate_hz"])
    target_frame_count = len(reconstructed)
    path, payload, digest = _authenticated_descriptor_bytes(
        scheduled_wav_descriptor,
        label="scheduled dry WAV",
    )
    observed = _decode_scheduled_pcm16(
        payload,
        sample_rate_hz=render_rate,
        frame_count=target_frame_count,
    )
    expected = np.floor(
        np.clip(reconstructed.astype(np.float64), -1.0, 1.0)
        * 32768.0
    )
    expected = np.clip(expected, -32768, 32767).astype(np.int32)
    error_lsb = np.abs(observed.astype(np.int32) - expected)
    max_abs_error_lsb = int(np.max(error_lsb)) if len(error_lsb) else 0
    if max_abs_error_lsb > PCM16_QUANTIZATION_TOLERANCE_LSB:
        raise ScheduledDryEvidenceError(
            "scheduled dry WAV does not match the authenticated source and "
            f"schedule: max PCM16 error {max_abs_error_lsb} LSB"
        )
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "passed",
        "path": str(path),
        "sha256": digest,
        "size_bytes": len(payload),
        "source_sha256": source_contract["sha256"],
        "tag": schedule["tag"],
        "sample_rate_hz": render_rate,
        "frame_count": target_frame_count,
        "channels": 1,
        "sample_width_bytes": 2,
        "quantization_tolerance_lsb": PCM16_QUANTIZATION_TOLERANCE_LSB,
        "max_abs_error_lsb": max_abs_error_lsb,
        "nonzero_error_sample_count": int(np.count_nonzero(error_lsb)),
    }


__all__ = [
    "PCM16_QUANTIZATION_TOLERANCE_LSB",
    "ScheduledDryEvidenceError",
    "reconstruct_scheduled_dry",
    "validate_scheduled_dry_evidence",
]
