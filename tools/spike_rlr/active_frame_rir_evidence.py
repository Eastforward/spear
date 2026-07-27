"""Authenticated active-frame native-binaural RIR evidence and replay."""

from __future__ import annotations

import io
import math
import re
import zipfile
from typing import Any, Iterable

import numpy as np


ACTIVE_FRAME_RIR_SCHEMA = "avengine_active_frame_binaural_rir_v1"
RLR_NATIVE_BINAURAL_CHANNEL_ORDER = (0, 1)
MAX_RIR_EVIDENCE_BYTES = 256 * 1024 * 1024
_MAX_CONCATENATED_RIR_BYTES = MAX_RIR_EVIDENCE_BYTES - 1024 * 1024
_TAG_RE = re.compile(r"^[A-Za-z0-9_]+$")
_ARRAY_KEYS = {
    "schema",
    "source_tag",
    "frame_indices",
    "ir_offsets",
    "rir_left",
    "rir_right",
    "source_positions_scene_m",
    "mic_position_scene_m",
    "mic_yaw_deg",
    "sample_rate_hz",
    "n_samples_total",
    "n_frames",
    "fps",
    "samples_per_frame",
    "channel_order",
}


def active_frame_indices(
    dry: np.ndarray,
    *,
    n_frames: int,
    samples_per_frame: int,
) -> np.ndarray:
    """Return frames whose exact renderer dry chunk contains any nonzero sample."""
    dry = np.asarray(dry)
    if dry.ndim != 1:
        raise ValueError("scheduled dry waveform must be mono")
    result = []
    for frame_index in range(n_frames):
        start = frame_index * samples_per_frame
        end = min(start + samples_per_frame, len(dry))
        if start >= len(dry):
            break
        if np.any(dry[start:end] != 0.0):
            result.append(frame_index)
    return np.asarray(result, dtype=np.int64)


def serialize_active_frame_rir_evidence(
    *,
    source_tag: str,
    frame_indices: Iterable[int],
    rirs: Iterable[np.ndarray],
    source_positions_scene_m: np.ndarray,
    mic_position_scene_m: np.ndarray,
    mic_yaw_deg: float,
    sample_rate_hz: int,
    n_samples_total: int,
    n_frames: int,
    fps: float,
    samples_per_frame: int,
    channel_order: tuple[int, int] = RLR_NATIVE_BINAURAL_CHANNEL_ORDER,
) -> bytes:
    """Serialize exact float32 RIRs in an uncompressed, pickle-free NPZ."""
    if not isinstance(source_tag, str) or not _TAG_RE.fullmatch(source_tag):
        raise ValueError("RIR evidence source tag is malformed")
    indices = np.asarray(list(frame_indices), dtype=np.int64)
    positions = np.asarray(source_positions_scene_m, dtype=np.float64)
    if (
        indices.ndim != 1
        or len(indices) == 0
        or np.any(indices < 0)
        or np.any(indices >= n_frames)
        or np.any(np.diff(indices) <= 0)
        or positions.shape != (len(indices), 3)
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("RIR evidence active-frame identity is malformed")

    normalized_rirs = []
    lengths = []
    total_rir_samples = 0
    for raw in rirs:
        rir = np.asarray(raw, dtype=np.float32)
        if (
            rir.ndim != 2
            or rir.shape[0] != 2
            or rir.shape[1] <= 0
            or rir.shape[1] > n_samples_total
            or not np.all(np.isfinite(rir))
            or float(np.max(np.abs(rir))) <= 0.0
        ):
            raise ValueError("native binaural RIR is malformed")
        total_rir_samples += int(rir.shape[1])
        if total_rir_samples * 2 * np.dtype(np.float32).itemsize > (
            _MAX_CONCATENATED_RIR_BYTES
        ):
            raise ValueError("RIR evidence exceeds the authenticated size limit")
        normalized_rirs.append(rir.copy())
        lengths.append(rir.shape[1])
    if len(normalized_rirs) != len(indices):
        raise ValueError("RIR evidence frame/RIR counts differ")

    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(lengths, dtype=np.int64))
    left = np.concatenate([rir[0] for rir in normalized_rirs]).astype(
        np.float32,
        copy=False,
    )
    right = np.concatenate([rir[1] for rir in normalized_rirs]).astype(
        np.float32,
        copy=False,
    )
    mic_position = np.asarray(mic_position_scene_m, dtype=np.float64)
    if (
        mic_position.shape != (3,)
        or not np.all(np.isfinite(mic_position))
        or not math.isfinite(float(mic_yaw_deg))
        or isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 0
        or isinstance(n_samples_total, bool)
        or not isinstance(n_samples_total, int)
        or n_samples_total <= 0
        or isinstance(n_frames, bool)
        or not isinstance(n_frames, int)
        or n_frames <= 0
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
        or isinstance(samples_per_frame, bool)
        or not isinstance(samples_per_frame, int)
        or samples_per_frame <= 0
        or tuple(channel_order) != RLR_NATIVE_BINAURAL_CHANNEL_ORDER
    ):
        raise ValueError("RIR evidence render configuration is malformed")

    stream = io.BytesIO()
    np.savez(
        stream,
        schema=np.asarray(ACTIVE_FRAME_RIR_SCHEMA),
        source_tag=np.asarray(source_tag),
        frame_indices=indices,
        ir_offsets=offsets,
        rir_left=left,
        rir_right=right,
        source_positions_scene_m=positions,
        mic_position_scene_m=mic_position,
        mic_yaw_deg=np.asarray(float(mic_yaw_deg), dtype=np.float64),
        sample_rate_hz=np.asarray(sample_rate_hz, dtype=np.int64),
        n_samples_total=np.asarray(n_samples_total, dtype=np.int64),
        n_frames=np.asarray(n_frames, dtype=np.int64),
        fps=np.asarray(float(fps), dtype=np.float64),
        samples_per_frame=np.asarray(samples_per_frame, dtype=np.int64),
        channel_order=np.asarray(channel_order, dtype=np.int64),
    )
    payload = stream.getvalue()
    if len(payload) > MAX_RIR_EVIDENCE_BYTES:
        raise ValueError("RIR evidence exceeds the authenticated size limit")
    return payload


def _scalar(array: np.ndarray, *, label: str) -> Any:
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"RIR evidence {label} scalar is malformed")
    return array.item()


def _inspect_npz_container(payload: bytes) -> None:
    if not payload or len(payload) > MAX_RIR_EVIDENCE_BYTES:
        raise ValueError("RIR evidence byte size is invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = {info.filename for info in infos}
            expected_names = {f"{key}.npy" for key in _ARRAY_KEYS}
            if (
                names != expected_names
                or len(infos) != len(expected_names)
                or any(
                    info.compress_type != zipfile.ZIP_STORED
                    or info.flag_bits & 0x1
                    or info.file_size <= 0
                    for info in infos
                )
                or sum(info.file_size for info in infos)
                > MAX_RIR_EVIDENCE_BYTES
            ):
                raise ValueError("RIR evidence NPZ container is unsafe")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("RIR evidence is not a valid NPZ container") from error


def load_active_frame_rir_evidence(
    payload: bytes,
    *,
    source_tag: str,
    dry: np.ndarray,
    expected_source_trajectory_scene_m: np.ndarray,
    mic_position_scene_m: np.ndarray,
    mic_yaw_deg: float,
    sample_rate_hz: int,
    n_samples_total: int,
    n_frames: int,
    fps: float,
    samples_per_frame: int,
) -> dict[str, Any]:
    """Parse evidence and bind every frame/config field to trusted inputs."""
    _inspect_npz_container(payload)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != _ARRAY_KEYS:
                raise ValueError("RIR evidence array set changed")
            arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("cannot decode active-frame RIR evidence") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("RIR evidence cannot contain object arrays")
    if (
        _scalar(arrays["schema"], label="schema") != ACTIVE_FRAME_RIR_SCHEMA
        or _scalar(arrays["source_tag"], label="source_tag") != source_tag
    ):
        raise ValueError("RIR evidence identity changed")

    expected_active = active_frame_indices(
        np.asarray(dry),
        n_frames=n_frames,
        samples_per_frame=samples_per_frame,
    )
    indices = arrays["frame_indices"]
    offsets = arrays["ir_offsets"]
    left = arrays["rir_left"]
    right = arrays["rir_right"]
    positions = arrays["source_positions_scene_m"]
    expected_trajectory = np.asarray(
        expected_source_trajectory_scene_m,
        dtype=np.float64,
    )
    if (
        indices.dtype != np.dtype(np.int64)
        or indices.ndim != 1
        or len(indices) == 0
        or not np.array_equal(indices, expected_active)
        or offsets.dtype != np.dtype(np.int64)
        or offsets.shape != (len(indices) + 1,)
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
        or left.dtype != np.dtype(np.float32)
        or right.dtype != np.dtype(np.float32)
        or left.ndim != 1
        or right.ndim != 1
        or len(left) != offsets[-1]
        or len(right) != offsets[-1]
        or offsets[-1] > len(indices) * n_samples_total
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or positions.dtype != np.dtype(np.float64)
        or positions.shape != (len(indices), 3)
        or expected_trajectory.shape != (n_frames, 3)
        or not np.array_equal(positions, expected_trajectory[indices])
    ):
        raise ValueError("RIR evidence active-frame payload changed")
    for start, end in zip(offsets[:-1], offsets[1:]):
        if (
            end - start > n_samples_total
            or float(
                max(
                    np.max(np.abs(left[start:end])),
                    np.max(np.abs(right[start:end])),
                )
            )
            <= 0.0
        ):
            raise ValueError("RIR evidence contains an invalid impulse response")

    expected_mic = np.asarray(mic_position_scene_m, dtype=np.float64)
    if (
        arrays["mic_position_scene_m"].dtype != np.dtype(np.float64)
        or not np.array_equal(arrays["mic_position_scene_m"], expected_mic)
        or arrays["mic_yaw_deg"].dtype != np.dtype(np.float64)
        or not math.isclose(
            float(_scalar(arrays["mic_yaw_deg"], label="mic_yaw_deg")),
            float(mic_yaw_deg),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or arrays["sample_rate_hz"].dtype != np.dtype(np.int64)
        or int(_scalar(arrays["sample_rate_hz"], label="sample_rate_hz"))
        != sample_rate_hz
        or arrays["n_samples_total"].dtype != np.dtype(np.int64)
        or int(_scalar(arrays["n_samples_total"], label="n_samples_total"))
        != n_samples_total
        or arrays["n_frames"].dtype != np.dtype(np.int64)
        or int(_scalar(arrays["n_frames"], label="n_frames")) != n_frames
        or arrays["fps"].dtype != np.dtype(np.float64)
        or not math.isclose(
            float(_scalar(arrays["fps"], label="fps")),
            float(fps),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or arrays["samples_per_frame"].dtype != np.dtype(np.int64)
        or int(
            _scalar(arrays["samples_per_frame"], label="samples_per_frame")
        )
        != samples_per_frame
        or arrays["channel_order"].dtype != np.dtype(np.int64)
        or not np.array_equal(
            arrays["channel_order"],
            np.asarray(RLR_NATIVE_BINAURAL_CHANNEL_ORDER, dtype=np.int64),
        )
    ):
        raise ValueError("RIR evidence render configuration changed")

    rirs = [
        np.stack((left[start:end], right[start:end]), axis=0)
        for start, end in zip(offsets[:-1], offsets[1:])
    ]
    return {
        "frame_indices": indices,
        "rirs": rirs,
        "source_positions_scene_m": positions,
    }


def replay_active_frame_rirs(
    dry: np.ndarray,
    evidence: dict[str, Any],
    *,
    n_samples_total: int,
    samples_per_frame: int,
    channel_order: tuple[int, int] = RLR_NATIVE_BINAURAL_CHANNEL_ORDER,
) -> np.ndarray:
    """Reproduce the renderer's float32 per-frame convolve + overlap-add."""
    dry = np.asarray(dry, dtype=np.float32)
    if dry.shape != (n_samples_total,):
        raise ValueError("scheduled dry length changed for RIR replay")
    if tuple(channel_order) != RLR_NATIVE_BINAURAL_CHANNEL_ORDER:
        raise ValueError("native binaural channel order changed")
    wet = np.zeros((2, n_samples_total), dtype=np.float32)
    for frame_index, rir in zip(
        evidence["frame_indices"],
        evidence["rirs"],
    ):
        frame_start = int(frame_index) * samples_per_frame
        frame_end = min(frame_start + samples_per_frame, n_samples_total)
        dry_chunk = dry[frame_start:frame_end]
        for channel in range(2):
            wet_chunk = np.convolve(
                dry_chunk,
                np.asarray(rir[channel], dtype=np.float32),
                mode="full",
            )
            wet_end = min(frame_start + len(wet_chunk), n_samples_total)
            wet[channel, frame_start:wet_end] += wet_chunk[
                : wet_end - frame_start
            ]
    return wet[list(channel_order), :]


__all__ = [
    "ACTIVE_FRAME_RIR_SCHEMA",
    "MAX_RIR_EVIDENCE_BYTES",
    "RLR_NATIVE_BINAURAL_CHANNEL_ORDER",
    "active_frame_indices",
    "load_active_frame_rir_evidence",
    "replay_active_frame_rirs",
    "serialize_active_frame_rir_evidence",
]
