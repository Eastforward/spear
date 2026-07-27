from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest
import soundfile as sf

from tools.audio_event_schedule import prepare_animal_call
from tools.spike_rlr.scheduled_dry_evidence import (
    ScheduledDryEvidenceError,
    reconstruct_scheduled_dry,
    validate_scheduled_dry_evidence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        sample_rate = stream.getframerate()
        samples = np.frombuffer(
            stream.readframes(stream.getnframes()),
            dtype="<i2",
        ).copy()
    return samples, sample_rate


def _fixture(tmp_path: Path) -> tuple[dict, dict, np.ndarray, Path]:
    source_rate = 8000
    render_rate = 1000
    source_frames = 16000
    source_pcm = np.zeros(source_frames, dtype="<i2")
    first = np.round(22000 * np.hanning(1200)).astype("<i2")
    second = np.round(
        17000
        * np.sin(2 * np.pi * 173 * np.arange(1600) / source_rate)
        * np.hanning(1600)
    ).astype("<i2")
    source_pcm[1600:2800] = first
    source_pcm[9200:10800] = second
    source_path = tmp_path / "pinned_source.wav"
    with wave.open(str(source_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(source_rate)
        stream.writeframes(source_pcm.tobytes())

    contract = {
        "schema": "avengine_pinned_animal_dry_source_v1",
        "audio_lookup": "dog_bark",
        "species": "dog",
        "path": str(source_path.resolve()),
        "sha256": _sha256(source_path),
        "size_bytes": source_path.stat().st_size,
        "codec": "pcm_s16le",
        "channels": 1,
        "sample_width_bytes": 2,
        "sample_rate_hz": source_rate,
        "frame_count": source_frames,
        "duration_s": source_frames / source_rate,
        "dry_source_policy": "mono_no_hrtf_no_pre_spatialization",
        "spatialization_status": "mono_unspatialized_dry_source",
        "known_spatialized_derivative_sha256": "d" * 64,
        "item_origin": {"fixture": "unit_test"},
        "objective_audio_content_qa_status": "fixture_passed",
        "item_level_license_status": "missing",
        "item_level_license_snapshot": None,
        "formal_registration_authorized": False,
    }

    source_float = source_pcm.astype(np.float32) / np.float32(32768.0)
    resampled_length = int(round(len(source_float) * render_rate / source_rate))
    resampled = np.interp(
        np.linspace(0, len(source_float), resampled_length, endpoint=False),
        np.arange(len(source_float)),
        source_float,
    ).astype(np.float32)
    scheduled, schedule = prepare_animal_call(
        resampled,
        sample_rate=render_rate,
        duration_s=6.0,
        rng=np.random.default_rng(42),
    )
    schedule.update(
        {
            "tag": "dog_fixture",
            "audio_path": contract["path"],
            "adaptive_repeat_short_calls": True,
            "source_sha256": contract["sha256"],
            "source_original_sample_rate_hz": source_rate,
            "source_original_frame_count": source_frames,
            "source_original_channels": 1,
            "render_sample_rate_hz": render_rate,
            "audio_lookup": contract["audio_lookup"],
            "source_species": contract["species"],
            "source_codec": contract["codec"],
            "source_channels": contract["channels"],
            "source_sample_width_bytes": contract["sample_width_bytes"],
            "source_catalog_frame_count": contract["frame_count"],
            "source_catalog_duration_s": contract["duration_s"],
            "dry_source_policy": contract["dry_source_policy"],
            "spatialization_status": contract["spatialization_status"],
            "known_spatialized_derivative_sha256": contract[
                "known_spatialized_derivative_sha256"
            ],
            "item_origin": contract["item_origin"],
            "objective_audio_content_qa_status": contract[
                "objective_audio_content_qa_status"
            ],
            "item_level_license_status": contract[
                "item_level_license_status"
            ],
            "item_level_license_snapshot": contract[
                "item_level_license_snapshot"
            ],
            "formal_registration_authorized": False,
            "source_contract": copy.deepcopy(contract),
        }
    )
    peak = float(np.max(np.abs(scheduled)))
    scheduled = scheduled * (0.8 / peak)
    scheduled_path = tmp_path / "scheduled_dry.wav"
    sf.write(
        str(scheduled_path),
        scheduled,
        render_rate,
        subtype="PCM_16",
    )
    return contract, schedule, scheduled, scheduled_path


def test_reconstruction_matches_scheduler_resample_fade_events_and_normalization(
    tmp_path,
):
    contract, schedule, scheduled, _path = _fixture(tmp_path)

    reconstructed = reconstruct_scheduled_dry(
        source_contract=contract,
        schedule=schedule,
    )

    assert reconstructed.dtype == np.float32
    assert reconstructed.shape == scheduled.shape
    np.testing.assert_array_equal(reconstructed, scheduled)
    assert float(np.max(np.abs(reconstructed))) == pytest.approx(0.8)


def test_validator_authenticates_descriptor_and_matches_pcm16_quantization(
    tmp_path,
):
    contract, schedule, _scheduled, path = _fixture(tmp_path)

    result = validate_scheduled_dry_evidence(
        scheduled_wav_descriptor=_descriptor(path),
        source_contract=contract,
        schedule=schedule,
    )

    observed, sample_rate = _read_pcm16(path)
    assert result == {
        "schema": "avengine_scheduled_dry_pcm16_validation_v1",
        "status": "passed",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "source_sha256": contract["sha256"],
        "tag": "dog_fixture",
        "sample_rate_hz": sample_rate,
        "frame_count": len(observed),
        "channels": 1,
        "sample_width_bytes": 2,
        "quantization_tolerance_lsb": 1,
        "max_abs_error_lsb": 0,
        "nonzero_error_sample_count": 0,
    }


def test_arbitrary_sine_spoof_fails_even_with_valid_descriptor_and_format(
    tmp_path,
):
    contract, schedule, _scheduled, path = _fixture(tmp_path)
    frame_count = int(
        round(schedule["target_duration_s"] * schedule["render_sample_rate_hz"])
    )
    phase = np.arange(frame_count, dtype=np.float64)
    spoof = (0.8 * np.sin(2 * np.pi * 37 * phase / 1000.0)).astype(np.float32)
    sf.write(str(path), spoof, 1000, subtype="PCM_16")

    with pytest.raises(ScheduledDryEvidenceError, match="does not match"):
        validate_scheduled_dry_evidence(
            scheduled_wav_descriptor=_descriptor(path),
            source_contract=contract,
            schedule=schedule,
        )


def test_one_lsb_quantizer_difference_is_allowed_but_two_are_rejected(tmp_path):
    contract, schedule, _scheduled, path = _fixture(tmp_path)
    samples, sample_rate = _read_pcm16(path)
    one_lsb = samples.copy()
    one_lsb[100] = np.int16(int(one_lsb[100]) + 1)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(one_lsb.astype("<i2").tobytes())
    validate_scheduled_dry_evidence(
        scheduled_wav_descriptor=_descriptor(path),
        source_contract=contract,
        schedule=schedule,
    )

    two_lsb = samples.copy()
    two_lsb[100] = np.int16(int(two_lsb[100]) + 2)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(two_lsb.astype("<i2").tobytes())
    with pytest.raises(ScheduledDryEvidenceError, match="2 LSB"):
        validate_scheduled_dry_evidence(
            scheduled_wav_descriptor=_descriptor(path),
            source_contract=contract,
            schedule=schedule,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda schedule: schedule["source_events"][0].__setitem__(
            "crop_start_s",
            schedule["source_events"][0]["crop_start_s"] + 0.001,
        ),
        lambda schedule: schedule["events"][0].__setitem__(
            "source_event_index",
            1,
        ),
        lambda schedule: schedule["events"][0].__setitem__(
            "start_sample",
            schedule["events"][0]["start_sample"] + 1,
        ),
        lambda schedule: schedule.__setitem__(
            "source_sha256",
            "0" * 64,
        ),
        lambda schedule: schedule["source_contract"].__setitem__(
            "sha256",
            "0" * 64,
        ),
    ],
)
def test_schedule_or_contract_tampering_fails_closed(tmp_path, mutate):
    contract, schedule, _scheduled, path = _fixture(tmp_path)
    mutate(schedule)

    with pytest.raises(ScheduledDryEvidenceError):
        validate_scheduled_dry_evidence(
            scheduled_wav_descriptor=_descriptor(path),
            source_contract=contract,
            schedule=schedule,
        )


def test_descriptor_hash_and_mono_pcm16_format_are_required(tmp_path):
    contract, schedule, _scheduled, path = _fixture(tmp_path)
    stale = _descriptor(path)
    stale["sha256"] = "0" * 64
    with pytest.raises(ScheduledDryEvidenceError, match="descriptor bytes changed"):
        validate_scheduled_dry_evidence(
            scheduled_wav_descriptor=stale,
            source_contract=contract,
            schedule=schedule,
        )

    stereo = np.zeros((6000, 2), dtype=np.float32)
    stereo[:, 0] = 0.2
    stereo[:, 1] = -0.2
    sf.write(str(path), stereo, 1000, subtype="PCM_16")
    with pytest.raises(ScheduledDryEvidenceError, match="format or duration"):
        validate_scheduled_dry_evidence(
            scheduled_wav_descriptor=_descriptor(path),
            source_contract=contract,
            schedule=schedule,
        )


def test_source_wav_must_still_match_the_pinned_descriptor(tmp_path):
    contract, schedule, _scheduled, path = _fixture(tmp_path)
    source_path = Path(contract["path"])
    source_path.write_bytes(source_path.read_bytes() + b"tamper")

    with pytest.raises(ScheduledDryEvidenceError, match="descriptor bytes changed"):
        validate_scheduled_dry_evidence(
            scheduled_wav_descriptor=_descriptor(path),
            source_contract=contract,
            schedule=schedule,
        )


def test_schedule_is_json_round_trip_stable(tmp_path):
    contract, schedule, _scheduled, path = _fixture(tmp_path)
    schedule = json.loads(json.dumps(schedule, sort_keys=True))

    result = validate_scheduled_dry_evidence(
        scheduled_wav_descriptor=_descriptor(path),
        source_contract=contract,
        schedule=schedule,
    )

    assert result["status"] == "passed"
