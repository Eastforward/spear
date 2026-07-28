#!/usr/bin/env python3
"""Publish post-Apartment revisions of controlled animal source_asset_v2 records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import (
    build_user_approved_generated_animal_apartment_specs as apartment_builder,
)
from tools import (
    measure_controlled_animal_physical_attributes as physical_measurement,
)
from tools import register_controlled_animal_source_assets as static_registry
from tools import rocketbox_native_material_canary as immutable
from tools.spike_rlr.animal_audio import (
    load_authenticated_file_bytes,
    load_authenticated_json,
    validate_animal_audio_evidence,
)
from tools.spike_rlr.acoustic_scene_contract import (
    approved_independent_rlr_subprocess_contract,
)
from tools.spike_rlr.active_frame_rir_evidence import (
    ACTIVE_FRAME_RIR_SCHEMA,
    MAX_RIR_EVIDENCE_BYTES,
)


REGISTRY_SCHEMA = "avengine_controlled_animal_apartment_source_asset_registry_v1"
APARTMENT_SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
APARTMENT_SCHEMA_V2 = "controlled_animal_walk_idle_apartment_specs_v2"
APARTMENT_SCHEMAS = frozenset({APARTMENT_SCHEMA, APARTMENT_SCHEMA_V2})
APARTMENT_V2_FIELDS = frozenset(
    {
        "schema",
        "generated_at",
        "usage_scope",
        "formal_registration_authorized",
        "trajectory_policy",
        "audio_policy",
        "avatar_count",
        "clip_count",
        "presentation_evidence",
        "presentation_automatic_checks",
        "inputs",
        "records",
        "manifest_sha256",
    }
)
APARTMENT_V2_PRESENTATION_CHECKS = {
    "presentation_receipt_raw_file_sha256_reauthenticated": True,
    "presentation_receipt_internal_sha256_reauthenticated": True,
    "presentation_exact_v4_review_sha256_reauthenticated": True,
    "presentation_output_video_bytes_and_directory_reauthenticated": True,
}
MEASUREMENT_BATCH_SCHEMA = "controlled_animal_physical_measurement_batch_v1"
MEASUREMENT_SCHEMA = "controlled_animal_physical_measurement_v1"
MEASUREMENT_METHOD = physical_measurement.METHOD
APARTMENT_REGISTRY_SCHEMA = (
    "controlled_animal_apartment_research_candidate_registry_v1"
)
PINNED_AUDIO_SOURCE_ROOT = Path(
    "/data/datasets/omniaudio/source_data/processed/clothov2"
)
PINNED_AUDIO_SOURCE_ROOT_ID = "animal_audio_pinned_source_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT_RLR_REQUEST_SCHEMA = "avengine_independent_rlr_sample_request_v2"
INDEPENDENT_RLR_RESULT_SCHEMA = "avengine_independent_rlr_sample_result_v2"
INDEPENDENT_RLR_REGISTRATION_EVIDENCE_SCHEMA = (
    "avengine_independent_rlr_registration_evidence_v2"
)
MAX_INDEPENDENT_RLR_RESULT_BYTES = 256 * 1024 * 1024
_INDEPENDENT_RESULT_KEYS = {
    "schema",
    "request_sha256",
    "subprocess_contract_sha256",
    "renderer_contract_sha256",
    "environment_contract_sha256",
    "runtime_packages_sha256",
    "python_sha256",
    "worker_sha256",
    "source_tag",
    "frame_index",
    "selection_seed_sha256",
    "source_position_coordinate_frame",
    "trajectory_prefix_sha256",
    "trajectory_prefix_shape",
    "channel_order",
    "rir",
}
_ACTIVE_RIR_KEYS = {
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


def _run_isolated_process_group(
    command: Sequence[str],
    *,
    executable: str,
    cwd: str,
    env: Mapping[str, str],
    stdin: Any,
    stdout: Any,
    stderr: Any,
    close_fds: bool,
    pass_fds: Sequence[int],
    timeout: float,
) -> subprocess.CompletedProcess:
    """Run a pinned worker and kill its whole session on timeout."""

    process = subprocess.Popen(
        command,
        executable=executable,
        cwd=cwd,
        env=dict(env),
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        close_fds=close_fds,
        pass_fds=tuple(pass_fds),
        start_new_session=True,
    )
    try:
        stdout_payload, stderr_payload = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.poll() is None:
                process.kill()
        try:
            stdout_payload, stderr_payload = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            # A trusted worker should not be able to escape its new session,
            # but keep the registration caller bounded if a library does.
            if process.poll() is None:
                process.kill()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            process.wait(timeout=5.0)
            stdout_payload = b""
            stderr_payload = b""
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout_payload,
            stderr=stderr_payload,
        ) from error
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=stdout_payload,
        stderr=stderr_payload,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _descriptor_file(value: Mapping[str, Any], label: str) -> Path:
    try:
        path = Path(os.path.abspath(os.fspath(value["path"])))
        expected_sha = value["sha256"]
        expected_size = value["size_bytes"]
    except (KeyError, TypeError) as error:
        raise contracts.ContractError(f"invalid {label} descriptor") from error
    try:
        payload = load_authenticated_file_bytes(path)
    except (OSError, TypeError, ValueError) as error:
        raise contracts.ContractError(f"{label} artifact changed: {path}") from error
    if (
        len(payload) != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_sha
    ):
        raise contracts.ContractError(f"{label} artifact changed: {path}")
    return path


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _pread_regular_file(
    descriptor: int,
    *,
    max_size_bytes: int,
    label: str,
) -> bytes:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > max_size_bytes
    ):
        raise contracts.ContractError(f"{label} is not a bounded regular file")
    chunks = []
    offset = 0
    while offset < metadata.st_size:
        chunk = os.pread(
            descriptor,
            min(1024 * 1024, metadata.st_size - offset),
            offset,
        )
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    if (
        offset != metadata.st_size
        or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        raise contracts.ContractError(f"{label} changed while being read")
    return b"".join(chunks)


def _open_approved_regular_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    max_size_bytes: int,
    label: str,
) -> int:
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        initial_payload = load_authenticated_file_bytes(
            path,
            max_size_bytes=max_size_bytes,
        )
    except (OSError, TypeError, ValueError) as error:
        raise contracts.ContractError(f"{label} is missing or unsafe: {path}") from error
    if (
        len(initial_payload) != expected_size_bytes
        or hashlib.sha256(initial_payload).hexdigest() != expected_sha256
    ):
        raise contracts.ContractError(f"{label} hash changed: {path}")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise contracts.ContractError(f"{label} cannot be pinned: {path}") from error
    try:
        pinned_payload = _pread_regular_file(
            descriptor,
            max_size_bytes=max_size_bytes,
            label=label,
        )
        if (
            len(pinned_payload) != expected_size_bytes
            or hashlib.sha256(pinned_payload).hexdigest() != expected_sha256
        ):
            raise contracts.ContractError(f"{label} changed before pinning: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _descriptor_payload(
    value: Mapping[str, Any],
    *,
    label: str,
    max_size_bytes: int = MAX_RIR_EVIDENCE_BYTES,
) -> bytes:
    try:
        path = Path(os.path.abspath(os.fspath(value["path"])))
        expected_sha256 = value["sha256"]
        expected_size_bytes = value["size_bytes"]
    except (KeyError, TypeError) as error:
        raise contracts.ContractError(f"invalid {label} descriptor") from error
    try:
        payload = load_authenticated_file_bytes(
            path,
            max_size_bytes=max_size_bytes,
        )
    except (OSError, TypeError, ValueError) as error:
        raise contracts.ContractError(f"{label} changed: {path}") from error
    if (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes <= 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or len(payload) != expected_size_bytes
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise contracts.ContractError(f"{label} changed: {path}")
    return payload


def _inspect_npz(
    payload: bytes,
    *,
    expected_keys: set[str],
    max_size_bytes: int,
    label: str,
) -> None:
    if not payload or len(payload) > max_size_bytes:
        raise contracts.ContractError(f"{label} byte size is invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = {item.filename for item in infos}
            expected_names = {f"{name}.npy" for name in expected_keys}
            if (
                names != expected_names
                or len(infos) != len(expected_names)
                or any(
                    item.compress_type != zipfile.ZIP_STORED
                    or item.flag_bits & 0x1
                    or item.file_size <= 0
                    for item in infos
                )
                or sum(item.file_size for item in infos) > max_size_bytes
            ):
                raise contracts.ContractError(f"{label} NPZ container changed")
    except (OSError, zipfile.BadZipFile) as error:
        raise contracts.ContractError(f"{label} is not a safe NPZ") from error


def _np_scalar(array: Any, *, label: str) -> Any:
    if array.shape != () or array.dtype.hasobject:
        raise contracts.ContractError(f"{label} scalar is malformed")
    return array.item()


def _select_expected_active_frame_rir(
    *,
    descriptor: Mapping[str, Any],
    tag: str,
    action: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    payload = _descriptor_payload(
        descriptor,
        label=f"{tag}/{action} active-frame RIR",
    )
    _inspect_npz(
        payload,
        expected_keys=_ACTIVE_RIR_KEYS,
        max_size_bytes=MAX_RIR_EVIDENCE_BYTES,
        label=f"{tag}/{action} active-frame RIR",
    )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != _ACTIVE_RIR_KEYS:
                raise contracts.ContractError("active-frame RIR array set changed")
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in archive.files
            }
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise contracts.ContractError("cannot decode active-frame RIR evidence") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise contracts.ContractError("active-frame RIR contains object arrays")
    if (
        _np_scalar(arrays["schema"], label="active-frame RIR schema")
        != ACTIVE_FRAME_RIR_SCHEMA
        or _np_scalar(arrays["source_tag"], label="active-frame RIR source")
        != tag
    ):
        raise contracts.ContractError("active-frame RIR identity changed")
    frame_indices = arrays["frame_indices"]
    offsets = arrays["ir_offsets"]
    left = arrays["rir_left"]
    right = arrays["rir_right"]
    source_positions = arrays["source_positions_scene_m"]
    mic_position = arrays["mic_position_scene_m"]
    mic_yaw_array = arrays["mic_yaw_deg"]
    sample_rate_array = arrays["sample_rate_hz"]
    channel_order = arrays["channel_order"]
    n_samples_total_array = arrays["n_samples_total"]
    n_frames_array = arrays["n_frames"]
    fps_array = arrays["fps"]
    samples_per_frame_array = arrays["samples_per_frame"]
    if (
        frame_indices.dtype != np.dtype(np.int64)
        or frame_indices.ndim != 1
        or len(frame_indices) == 0
        or np.any(frame_indices < 0)
        or np.any(np.diff(frame_indices) <= 0)
        or offsets.dtype != np.dtype(np.int64)
        or offsets.shape != (len(frame_indices) + 1,)
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
        or left.dtype != np.dtype(np.float32)
        or right.dtype != np.dtype(np.float32)
        or left.ndim != 1
        or right.ndim != 1
        or len(left) != int(offsets[-1])
        or len(right) != int(offsets[-1])
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or source_positions.dtype != np.dtype(np.float64)
        or source_positions.shape != (len(frame_indices), 3)
        or not np.all(np.isfinite(source_positions))
        or mic_position.dtype != np.dtype(np.float64)
        or mic_position.shape != (3,)
        or not np.all(np.isfinite(mic_position))
        or mic_yaw_array.dtype != np.dtype(np.float64)
        or mic_yaw_array.shape != ()
        or not math.isfinite(float(mic_yaw_array.item()))
        or sample_rate_array.dtype != np.dtype(np.int64)
        or sample_rate_array.shape != ()
        or channel_order.dtype != np.dtype(np.int64)
        or not np.array_equal(channel_order, np.asarray([0, 1], dtype=np.int64))
        or n_samples_total_array.dtype != np.dtype(np.int64)
        or n_samples_total_array.shape != ()
        or n_frames_array.dtype != np.dtype(np.int64)
        or n_frames_array.shape != ()
        or fps_array.dtype != np.dtype(np.float64)
        or fps_array.shape != ()
        or not math.isfinite(float(fps_array.item()))
        or samples_per_frame_array.dtype != np.dtype(np.int64)
        or samples_per_frame_array.shape != ()
    ):
        raise contracts.ContractError("active-frame RIR payload changed")
    n_samples_total = int(n_samples_total_array.item())
    n_frames = int(n_frames_array.item())
    sample_rate_hz = int(sample_rate_array.item())
    fps = float(fps_array.item())
    samples_per_frame = int(samples_per_frame_array.item())
    if (
        n_samples_total <= 0
        or n_frames <= 0
        or sample_rate_hz <= 0
        or fps <= 0.0
        or samples_per_frame <= 0
        or np.any(frame_indices >= n_frames)
        or manifest.get("sample_rate_hz") != sample_rate_hz
        or manifest.get("n_frames") != n_frames
        or not math.isclose(
            float(manifest.get("fps", float("nan"))),
            fps,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or manifest.get("renderer_contract", {})
        .get("acoustics", {})
        .get("sample_rate_hz")
        != sample_rate_hz
    ):
        raise contracts.ContractError(
            "active-frame RIR render dimensions changed"
        )
    selection_contract = {
        "schema": "avengine_independent_rlr_frame_selection_v1",
        "action": action,
        "source_tag": tag,
        "active_frame_rir_sha256": descriptor["sha256"],
        "spec_sha256": manifest["spec"]["sha256"],
        "acoustic_mesh_sha256": manifest["acoustic_mesh"]["sha256"],
        "derived_rlr_materials_sha256": manifest["derived_rlr_materials"][
            "sha256"
        ],
        "renderer_contract_sha256": _canonical_sha256(
            manifest["renderer_contract"]
        ),
        "active_frame_indices": frame_indices.tolist(),
    }
    seed_sha256 = _canonical_sha256(selection_contract)
    selected_slot = int(seed_sha256, 16) % len(frame_indices)
    start = int(offsets[selected_slot])
    end = int(offsets[selected_slot + 1])
    expected = np.stack((left[start:end], right[start:end]), axis=0)
    if (
        expected.shape[1] <= 0
        or not np.all(np.isfinite(expected))
        or float(np.max(np.abs(expected))) <= 0.0
    ):
        raise contracts.ContractError("selected active-frame RIR is malformed")
    return {
        "payload": payload,
        "frame_index": int(frame_indices[selected_slot]),
        "expected_rir": expected,
        "selection_seed_sha256": seed_sha256,
        "selected_slot": selected_slot,
        "active_frame_count": len(frame_indices),
        "frame_indices": frame_indices,
        "source_positions_scene_m": source_positions,
        "mic_position_scene_m": mic_position,
        "mic_yaw_deg": float(mic_yaw_array.item()),
        "sample_rate_hz": sample_rate_hz,
        "n_samples_total": n_samples_total,
        "n_frames": n_frames,
        "fps": fps,
        "samples_per_frame": samples_per_frame,
    }


def _write_exclusive_readonly(path: Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o400)
    except BaseException:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    return path


def _snapshot_descriptor(name: str, payload: bytes) -> dict[str, Any]:
    return {
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _load_independent_result(
    payload: bytes,
    *,
    request_sha256: str,
    subprocess_contract: Mapping[str, Any],
    renderer_contract: Mapping[str, Any],
    tag: str,
    frame_index: int,
    selection_seed_sha256: str,
    trajectory_prefix_sha256: str,
    trajectory_prefix_shape: Sequence[int],
    n_samples_total: int,
) -> Any:
    import numpy as np

    _inspect_npz(
        payload,
        expected_keys=_INDEPENDENT_RESULT_KEYS,
        max_size_bytes=MAX_INDEPENDENT_RLR_RESULT_BYTES,
        label="independent RLR result",
    )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != _INDEPENDENT_RESULT_KEYS:
                raise contracts.ContractError(
                    "independent RLR result array set changed"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in archive.files
            }
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise contracts.ContractError("cannot decode independent RLR result") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise contracts.ContractError("independent RLR result contains object arrays")
    scalar_expectations = {
        "schema": INDEPENDENT_RLR_RESULT_SCHEMA,
        "request_sha256": request_sha256,
        "subprocess_contract_sha256": _canonical_sha256(
            subprocess_contract
        ),
        "renderer_contract_sha256": _canonical_sha256(renderer_contract),
        "environment_contract_sha256": _canonical_sha256(
            subprocess_contract["environment"]
        ),
        "runtime_packages_sha256": _canonical_sha256(
            subprocess_contract["runtime_packages"]
        ),
        "python_sha256": subprocess_contract["python"]["sha256"],
        "worker_sha256": subprocess_contract["worker"]["sha256"],
        "source_tag": tag,
        "selection_seed_sha256": selection_seed_sha256,
        "source_position_coordinate_frame": "right-handed Z-up meters",
        "trajectory_prefix_sha256": trajectory_prefix_sha256,
    }
    for name, expected in scalar_expectations.items():
        if _np_scalar(arrays[name], label=f"independent RLR {name}") != expected:
            raise contracts.ContractError(
                f"independent RLR result {name} changed"
            )
    if (
        arrays["frame_index"].dtype != np.dtype(np.int64)
        or _np_scalar(
            arrays["frame_index"],
            label="independent RLR frame_index",
        )
        != frame_index
        or arrays["channel_order"].dtype != np.dtype(np.int64)
        or not np.array_equal(
            arrays["channel_order"],
            np.asarray([0, 1], dtype=np.int64),
        )
        or arrays["trajectory_prefix_shape"].dtype != np.dtype(np.int64)
        or not np.array_equal(
            arrays["trajectory_prefix_shape"],
            np.asarray(trajectory_prefix_shape, dtype=np.int64),
        )
    ):
        raise contracts.ContractError(
            "independent RLR result frame/channel identity changed"
        )
    rir = arrays["rir"]
    if (
        rir.dtype != np.dtype(np.float32)
        or rir.ndim != 2
        or rir.shape[0] != 2
        or rir.shape[1] <= 0
        or rir.shape[1] > n_samples_total
        or not np.all(np.isfinite(rir))
        or float(np.max(np.abs(rir))) <= 0.0
    ):
        raise contracts.ContractError("independent RLR result channels are malformed")
    return rir


def _pinned_audio_source_artifact(path: Path) -> dict[str, Any]:
    path = Path(os.path.abspath(os.fspath(path)))
    root = PINNED_AUDIO_SOURCE_ROOT.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise contracts.ContractError(
            f"pinned audio source escaped its approved root: {path}"
        ) from error
    payload = load_authenticated_file_bytes(path)
    return {
        "root_id": PINNED_AUDIO_SOURCE_ROOT_ID,
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _load_source_assets(roots: Sequence[Path]) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in sorted(Path(root).resolve().rglob("*.json")):
            try:
                payload = contracts.load_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("schema") != contracts.SOURCE_ASSET_SCHEMA:
                continue
            asset = contracts.validate_source_asset_v2(payload)
            asset_id = asset["asset_id"]
            if asset_id in assets:
                raise contracts.ContractError(f"duplicate source_asset_v2: {asset_id}")
            assets[asset_id] = asset
    if not assets:
        raise contracts.ContractError("no source_asset_v2 records found")
    return assets


def _load_apartment_records(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for supplied_path in paths:
        raw_path = Path(supplied_path)
        path = raw_path.resolve()
        payload = contracts.load_json(path)
        items = payload.get("records", []) if isinstance(payload, dict) else []
        manifest_schema = payload.get("schema")
        authority: Mapping[str, Any] | None = None
        if manifest_schema == APARTMENT_SCHEMA_V2:
            try:
                authority = apartment_builder.authenticate_apartment_v2_manifest(
                    raw_path
                )
            except (contracts.ContractError, OSError, KeyError, TypeError, ValueError) as error:
                raise contracts.ContractError(
                    f"invalid Apartment v2 authority: {path}: {error}"
                ) from error
            payload = authority["manifest"]
            items = [authority["record"]]
        if (
            manifest_schema not in APARTMENT_SCHEMAS
            or payload.get("avatar_count") != len(items)
            or payload.get("clip_count") != len(items) * 2
            or payload.get("manifest_sha256") != _hash_without(payload, "manifest_sha256")
        ):
            raise contracts.ContractError(f"invalid Apartment manifest: {path}")
        for record in items:
            asset_id = record.get("base_avatar_id")
            if not asset_id or asset_id in records:
                raise contracts.ContractError(f"duplicate Apartment asset: {asset_id}")
            if (
                manifest_schema == APARTMENT_SCHEMA_V2
                and (
                    record.get("asset_id") != asset_id
                    or not isinstance(record.get("tag"), str)
                    or not record["tag"]
                    or not isinstance(record.get("profile_schema_id"), str)
                    or not record["profile_schema_id"]
                    or not isinstance(record.get("sampled_attributes"), Mapping)
                    or not record["sampled_attributes"]
                    or not isinstance(record.get("source_glb"), Mapping)
                    or set(record["source_glb"]) != {"path", "sha256"}
                    or not isinstance(record.get("actions"), Mapping)
                    or set(record["actions"]) != {"Walking", "Idle"}
                )
            ):
                raise contracts.ContractError(
                    f"invalid Apartment v2 record identity: {asset_id}"
                )
            authenticated_record = copy.deepcopy(dict(record))
            if authority is not None:
                authenticated_record["_authenticated_apartment_v2"] = {
                    "manifest": {
                        "path": str(path),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                    },
                    "inputs": copy.deepcopy(authority["inputs"]),
                    "presentation_evidence": copy.deepcopy(
                        authority["presentation_evidence"]
                    ),
                    "runtime_lineage": copy.deepcopy(
                        record["runtime_lineage"]
                    ),
                    "emitter_measurement": copy.deepcopy(
                        record["emitter_measurement"]
                    ),
                    "audio_source_height_offset_m": record[
                        "audio_source_height_offset_m"
                    ],
                }
            records[asset_id] = authenticated_record
    return records


def _load_measurements(
    path: Path,
    apartment_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    path = Path(path).resolve()
    payload = contracts.load_json(path)
    items = payload.get("measurements", []) if isinstance(payload, dict) else []
    if (
        set(payload)
        != {
            "schema",
            "generated_at",
            "method",
            "asset_count",
            "inputs",
            "measurements",
            "size_ordering",
            "admission_policy",
            "automatic_checks",
            "batch_sha256",
        }
        or payload.get("schema") != MEASUREMENT_BATCH_SCHEMA
        or payload.get("method") != MEASUREMENT_METHOD
        or not isinstance(payload.get("generated_at"), str)
        or not payload["generated_at"]
        or payload.get("asset_count") != len(items)
        or payload.get("asset_count") != len(apartment_records)
        or not isinstance(items, list)
        or not items
        or payload.get("batch_sha256") != _hash_without(payload, "batch_sha256")
        or payload.get("admission_policy")
        != {
            "target_tolerance": (
                "required_for_downstream_source_asset_registration"
            ),
            "single_size_ordering": (
                "not_applicable_not_vacuously_passed"
            ),
            "rejected_batch_usage": (
                "measurement_and_scale_recalibration_evidence_only"
            ),
        }
    ):
        raise contracts.ContractError("invalid physical measurement batch")
    expected_manifest_descriptors = [
        record["_authenticated_apartment_v2"]["manifest"]
        for record in apartment_records.values()
        if "_authenticated_apartment_v2" in record
    ]
    if expected_manifest_descriptors:
        supplied_inputs = payload.get("inputs")
        if (
            not isinstance(supplied_inputs, list)
            or sorted(
                supplied_inputs,
                key=lambda item: (
                    str(item.get("path", "")),
                    str(item.get("sha256", "")),
                ),
            )
            != sorted(
                expected_manifest_descriptors,
                key=lambda item: (
                    str(item.get("path", "")),
                    str(item.get("sha256", "")),
                ),
            )
        ):
            raise contracts.ContractError(
                "physical measurement Apartment inputs changed"
            )
    for descriptor in payload.get("inputs", []):
        _descriptor_file(descriptor, "physical measurement Apartment input")

    records: dict[str, dict[str, Any]] = {}
    measured_payloads: list[dict[str, Any]] = []
    for index in items:
        if (
            not isinstance(index, Mapping)
            or set(index)
            != {
                "asset_id",
                "profile_schema_id",
                "sampled_size",
                "physical_measurements",
                "target_comparison",
                "record",
            }
        ):
            raise contracts.ContractError("physical measurement index changed")
        record_path = _descriptor_file(index["record"], "physical measurement")
        record = contracts.load_json(record_path)
        asset_id = record.get("asset_id")
        apartment = apartment_records.get(str(asset_id))
        if (
            record.get("schema") != MEASUREMENT_SCHEMA
            or record.get("method") != MEASUREMENT_METHOD
            or record.get("physical_measurements", {}).get("method")
            != MEASUREMENT_METHOD
            or asset_id != index.get("asset_id")
            or apartment is None
            or record.get("tag") != apartment.get("tag")
            or record.get("profile_schema_id")
            != apartment.get("profile_schema_id")
            or record.get("sampled_size")
            != apartment.get("target_physical_profile", {}).get(
                "selected_value"
            )
            or record.get("profile_schema_id")
            != index.get("profile_schema_id")
            or record.get("sampled_size") != index.get("sampled_size")
            or record.get("physical_measurements") != index.get("physical_measurements")
            or record.get("target_comparison") != index.get("target_comparison")
            or asset_id in records
        ):
            raise contracts.ContractError("physical measurement identity changed")
        runtime = record.get("physical_measurements", {}).get("runtime")
        target = apartment.get("target_physical_profile")
        comparison = record.get("target_comparison")
        if (
            not isinstance(runtime, Mapping)
            or not isinstance(target, Mapping)
            or not isinstance(comparison, Mapping)
        ):
            raise contracts.ContractError(
                "physical measurement comparison is malformed"
            )
        measured_height = runtime.get("shoulder_height_cm")
        target_height = target.get("target_value_cm")
        tolerance = target.get("tolerance_cm")
        if (
            isinstance(measured_height, bool)
            or not isinstance(measured_height, (int, float))
            or not math.isfinite(float(measured_height))
            or isinstance(target_height, bool)
            or not isinstance(target_height, (int, float))
            or not math.isfinite(float(target_height))
            or isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) <= 0.0
        ):
            raise contracts.ContractError(
                "physical measurement comparison is non-finite"
            )
        residual = float(measured_height) - float(target_height)
        status = (
            "within_tolerance"
            if abs(residual) <= float(tolerance)
            else "outside_tolerance"
        )
        expected_comparison = {
            "target_value_cm": target_height,
            "tolerance_cm": tolerance,
            "measured_minus_target_cm": round(residual, 6),
            "status": status,
            "target_reference_status": target.get(
                "reference_provenance", {}
            ).get("status"),
        }
        expected_audio_height = apartment.get(
            "audio_source_height_offset_m"
        )
        if (
            comparison != expected_comparison
            or status != "within_tolerance"
            or not isinstance(expected_audio_height, (int, float))
            or isinstance(expected_audio_height, bool)
            or not math.isclose(
                float(runtime.get("audio_source_height_offset_m", float("nan"))),
                round(float(expected_audio_height), 9),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise contracts.ContractError(
                "physical measurement tolerance/emitter admission failed"
            )

        evidence = record.get("evidence")
        if not isinstance(evidence, Mapping):
            raise contracts.ContractError(
                "physical measurement evidence is missing"
            )
        expected_evidence_fields = {
            "walking_spec",
            "ue_visual_metadata",
            "geometry_measurement",
            "emitter_measurement",
        }
        if apartment.get("rig_semantic_evidence") is not None:
            expected_evidence_fields.add("rig_semantic_evidence")
        if set(evidence) != expected_evidence_fields:
            raise contracts.ContractError(
                "physical measurement evidence fields changed"
            )
        evidence_paths = {
            name: _descriptor_file(
                descriptor,
                f"physical measurement {name}",
            )
            for name, descriptor in evidence.items()
        }
        if (
            evidence["walking_spec"]
            != apartment["actions"]["Walking"]["spec_evidence"]
            or evidence["emitter_measurement"]
            != apartment["emitter_measurement"]
            or evidence_paths["ue_visual_metadata"]
            != (
                Path(apartment["actions"]["Walking"]["output_dir"])
                / "videos"
                / "actor_visual_metadata.json"
            ).resolve()
            or (
                apartment.get("rig_semantic_evidence") is not None
                and evidence["rig_semantic_evidence"]
                != apartment["rig_semantic_evidence"]["artifact"]
            )
        ):
            raise contracts.ContractError(
                "physical measurement evidence lineage changed"
            )
        geometry = contracts.load_json(evidence_paths["geometry_measurement"])
        geometry_input = geometry.get("input_glb")
        source_glb = apartment.get("source_glb")
        if (
            not isinstance(geometry_input, Mapping)
            or not isinstance(source_glb, Mapping)
            or geometry_input.get("path") != source_glb.get("path")
            or geometry_input.get("sha256") != source_glb.get("sha256")
            or record.get("calibration", {}).get("geometry") != geometry
        ):
            raise contracts.ContractError(
                "physical measurement measured a different GLB"
            )
        records[asset_id] = {"payload": record, "path": record_path}
        measured_payloads.append(record)

    try:
        recomputed_ordering = physical_measurement.summarize_size_ordering(
            measured_payloads
        )
        recomputed_checks = physical_measurement.build_admission_checks(
            measured_payloads,
            recomputed_ordering,
        )
    except physical_measurement.MeasurementError as error:
        raise contracts.ContractError(
            f"physical measurement admission cannot be recomputed: {error}"
        ) from error
    if (
        payload.get("size_ordering") != recomputed_ordering
        or payload.get("automatic_checks") != recomputed_checks
        or recomputed_checks.get("downstream_source_asset_registration_ready")
        is not True
        or recomputed_checks.get("overall") != "passed"
        or set(records) != set(apartment_records)
    ):
        raise contracts.ContractError(
            "physical measurement downstream admission changed"
        )
    return records


def _validate_audio(action_record: Mapping[str, Any], tag: str) -> dict[str, Any]:
    output = Path(action_record["output_dir"]).resolve()
    audio = output / "binaural.wav"
    schedule_path = output / "binaural_source_schedule.json"
    render_manifest_path = output / "binaural_audio_render_manifest.json"
    spec_path = Path(action_record["spec"])
    if (
        audio.is_symlink()
        or schedule_path.is_symlink()
        or render_manifest_path.is_symlink()
        or spec_path.is_symlink()
    ):
        raise contracts.ContractError(f"audio evidence cannot be symlinked: {tag}")
    try:
        schedule = load_authenticated_json(schedule_path)
        spec = load_authenticated_json(spec_path)
        render_manifest = load_authenticated_json(render_manifest_path)
        validated_evidence = validate_animal_audio_evidence(
            spec=spec,
            schedule=schedule,
            audio_path=audio,
            expected_tags={tag},
            render_manifest=render_manifest,
            spec_path=spec_path,
            schedule_path=schedule_path,
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise contracts.ContractError(
            f"animal audio source contract failed: {tag}"
        ) from error
    source = schedule.get("sources", {}).get(tag, {})
    event_count = source.get("event_count")
    minimum_gap_s = source.get("minimum_silence_gap_s")
    if (
        source.get("adaptive_repeat_short_calls") is not True
        or source.get("short_call_detected") is not True
        or source.get("mode") != "repeated_events_with_silence_gaps"
        or isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count <= 1
        or isinstance(minimum_gap_s, bool)
        or not isinstance(minimum_gap_s, (int, float))
        or not math.isfinite(float(minimum_gap_s))
        or float(minimum_gap_s) < 0.85
    ):
        raise contracts.ContractError(f"animal audio schedule failed: {tag}")
    return {
        "audio": audio,
        "schedule": schedule_path,
        "render_manifest": render_manifest_path,
        "render_manifest_value": render_manifest,
        "validated_source_trajectory_scene_m": validated_evidence[
            "render_manifest"
        ]["solo_wavs"][tag]["active_frame_rir_replay"][
            "validated_source_trajectory_scene_m"
        ],
        "render_spec_snapshot": _descriptor_file(
            render_manifest["spec"],
            f"RLR render spec snapshot for {tag}",
        ),
        "binaural_solo": _descriptor_file(
            render_manifest["per_source_outputs"][tag]["binaural"],
            f"RLR binaural solo for {tag}",
        ),
        "scheduled_dry": _descriptor_file(
            render_manifest["per_source_outputs"][tag]["scheduled_dry"],
            f"RLR scheduled dry source for {tag}",
        ),
        "active_frame_rir": _descriptor_file(
            render_manifest["per_source_outputs"][tag]["active_frame_rir"],
            f"RLR active-frame RIR for {tag}",
        ),
        "acoustic_mesh": _descriptor_file(
            render_manifest["acoustic_mesh"],
            f"RLR acoustic mesh for {tag}",
        ),
        "acoustic_materials": _descriptor_file(
            render_manifest["acoustic_materials"],
            f"RLR acoustic materials for {tag}",
        ),
        "derived_rlr_materials": _descriptor_file(
            render_manifest["derived_rlr_materials"],
            f"RLR derived materials for {tag}",
        ),
        "pinned_dry_source": _descriptor_file(
            source["source_contract"],
            f"pinned dry source for {tag}",
        ),
    }


def _validate_independent_active_frame_rir_sample(
    *,
    render_manifest: Mapping[str, Any],
    validated_source_trajectory_scene_m: Sequence[Sequence[float]],
    tag: str,
    action: str,
    evidence_output_base: Path,
) -> dict[str, Any]:
    import numpy as np

    try:
        per_source = render_manifest["per_source_outputs"][tag]
        rir_descriptor = per_source["active_frame_rir"]
        renderer_contract = render_manifest["renderer_contract"]
        quality_mode = render_manifest["quality_mode"]
    except (KeyError, TypeError) as error:
        raise contracts.ContractError(
            f"independent RLR manifest inputs are missing: {tag}/{action}"
        ) from error
    selected = _select_expected_active_frame_rir(
        descriptor=rir_descriptor,
        tag=tag,
        action=action,
        manifest=render_manifest,
    )
    try:
        validated_trajectory = np.asarray(
            validated_source_trajectory_scene_m,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise contracts.ContractError(
            f"validated RLR trajectory is malformed: {tag}/{action}"
        ) from error
    if (
        validated_trajectory.dtype != np.dtype(np.float64)
        or validated_trajectory.shape != (selected["n_frames"], 3)
        or not np.all(np.isfinite(validated_trajectory))
        or not np.array_equal(
            validated_trajectory[selected["frame_indices"]],
            selected["source_positions_scene_m"],
        )
    ):
        raise contracts.ContractError(
            f"validated RLR trajectory changed: {tag}/{action}"
        )
    trajectory_prefix = np.ascontiguousarray(
        validated_trajectory[: selected["frame_index"] + 1],
        dtype=np.dtype("<f8"),
    )
    trajectory_prefix_sha256 = hashlib.sha256(
        trajectory_prefix.tobytes(order="C")
    ).hexdigest()
    subprocess_contract = approved_independent_rlr_subprocess_contract()
    if (
        subprocess_contract.get("schema")
        != "avengine_independent_rlr_subprocess_contract_v2"
        or subprocess_contract.get("worker", {}).get("root_id") != "spear_repo"
        or subprocess_contract.get("worker", {}).get("path")
        != "tools/spike_rlr/sample_active_frame_rir_rlr.py"
        or subprocess_contract.get("python", {}).get("path")
        != "/data/jzy/miniconda3/envs/ss2/bin/python"
        or subprocess_contract.get("python", {}).get("resolved_path")
        != "/data/jzy/miniconda3/envs/ss2/bin/python3.9"
        or subprocess_contract.get("python", {}).get("isolated_flags")
        != ["-I", "-B", "-S"]
        or not isinstance(subprocess_contract.get("environment"), Mapping)
        or set(subprocess_contract["environment"])
        != {
            "CUDA_VISIBLE_DEVICES",
            "DISPLAY",
            "LANG",
            "LC_ALL",
            "LD_PRELOAD",
            "PATH",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONHASHSEED",
            "PYTHONIOENCODING",
            "PYTHONNOUSERSITE",
            "PYTHONUNBUFFERED",
            "VK_ICD_FILENAMES",
            "XDG_RUNTIME_DIR",
        }
        or subprocess_contract.get("runtime_packages")
        != {
            "site_packages_path": (
                "/data/jzy/miniconda3/envs/ss2/lib/python3.9/site-packages"
            ),
            "numpy_version": "1.26.4",
            "habitat_sim_version": "0.2.2",
            "numpy_quaternion_version": "2023.0.4",
        }
        or not isinstance(subprocess_contract.get("timeout_s"), (int, float))
        or isinstance(subprocess_contract.get("timeout_s"), bool)
        or not math.isfinite(float(subprocess_contract["timeout_s"]))
        or float(subprocess_contract["timeout_s"]) <= 0.0
        or float(subprocess_contract["timeout_s"]) > 600.0
    ):
        raise contracts.ContractError(
            "independent RLR approved subprocess contract changed"
        )

    python_contract = subprocess_contract["python"]
    try:
        resolved_python = Path(python_contract["path"]).resolve(strict=True)
    except OSError as error:
        raise contracts.ContractError(
            "independent RLR approved Python is missing"
        ) from error
    if str(resolved_python) != python_contract["resolved_path"]:
        raise contracts.ContractError(
            "independent RLR approved Python target changed"
        )
    python_fd = _open_approved_regular_file(
        resolved_python,
        expected_sha256=python_contract["sha256"],
        expected_size_bytes=python_contract["size_bytes"],
        max_size_bytes=64 * 1024 * 1024,
        label="independent RLR Python",
    )
    worker_contract = subprocess_contract["worker"]
    worker_path = SPEAR_ROOT / worker_contract["path"]
    try:
        worker_fd = _open_approved_regular_file(
            worker_path,
            expected_sha256=worker_contract["sha256"],
            expected_size_bytes=worker_contract["size_bytes"],
            max_size_bytes=4 * 1024 * 1024,
            label="independent RLR worker",
        )
    except BaseException:
        os.close(python_fd)
        raise

    snapshot_payloads = {}
    for name, manifest_key in (
        ("spec.json", "spec"),
        ("acoustic_mesh.glb", "acoustic_mesh"),
        ("acoustic_materials.json", "acoustic_materials"),
        ("rlr_materials.json", "derived_rlr_materials"),
    ):
        snapshot_payloads[name] = _descriptor_payload(
            render_manifest[manifest_key],
            label=f"{tag}/{action} independent RLR {manifest_key}",
        )
    request = {
        "schema": INDEPENDENT_RLR_REQUEST_SCHEMA,
        "source_tag": tag,
        "frame_index": selected["frame_index"],
        "selection_seed_sha256": selected["selection_seed_sha256"],
        "source_position_coordinate_frame": "right-handed Z-up meters",
        "trajectory_prefix_shape": list(trajectory_prefix.shape),
        "trajectory_prefix_sha256": trajectory_prefix_sha256,
        "warmup_source_positions_scene_m": trajectory_prefix.tolist(),
        "mic_position_scene_m": selected["mic_position_scene_m"].tolist(),
        "mic_yaw_deg": selected["mic_yaw_deg"],
        "quality_mode": quality_mode,
        "renderer_contract": copy.deepcopy(renderer_contract),
        "subprocess_contract": copy.deepcopy(subprocess_contract),
        "spec": _snapshot_descriptor("spec.json", snapshot_payloads["spec.json"]),
        "acoustic_mesh": _snapshot_descriptor(
            "acoustic_mesh.glb",
            snapshot_payloads["acoustic_mesh.glb"],
        ),
        "acoustic_materials": _snapshot_descriptor(
            "acoustic_materials.json",
            snapshot_payloads["acoustic_materials.json"],
        ),
        "derived_rlr_materials": _snapshot_descriptor(
            "rlr_materials.json",
            snapshot_payloads["rlr_materials.json"],
        ),
    }
    request_payload = (
        json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    request_sha256 = hashlib.sha256(request_payload).hexdigest()

    completed = None
    result_payload = b""
    try:
        with tempfile.TemporaryDirectory(
            prefix=".independent_rlr_registration_snapshot."
        ) as snapshot_name:
            snapshot_root = Path(snapshot_name)
            os.chmod(snapshot_root, 0o700)
            for name, payload in snapshot_payloads.items():
                _write_exclusive_readonly(snapshot_root / name, payload)
            _write_exclusive_readonly(
                snapshot_root / "request.json",
                request_payload,
            )
            os.chmod(snapshot_root, 0o500)
            snapshot_fd = os.open(
                snapshot_root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                with tempfile.TemporaryFile(mode="w+b") as result_stream:
                    output_fd = result_stream.fileno()
                    command = [
                        str(resolved_python),
                        *python_contract["isolated_flags"],
                        f"/proc/self/fd/{worker_fd}",
                        "--snapshot-fd",
                        str(snapshot_fd),
                        "--request-name",
                        "request.json",
                        "--output-fd",
                        str(output_fd),
                        "--worker-fd",
                        str(worker_fd),
                    ]
                    try:
                        completed = _run_isolated_process_group(
                            command,
                            executable=f"/proc/self/fd/{python_fd}",
                            cwd=str(SPEAR_ROOT),
                            env=dict(subprocess_contract["environment"]),
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            close_fds=True,
                            pass_fds=(
                                snapshot_fd,
                                output_fd,
                                worker_fd,
                                python_fd,
                            ),
                            timeout=float(subprocess_contract["timeout_s"]),
                        )
                    except subprocess.TimeoutExpired as error:
                        raise contracts.ContractError(
                            f"independent RLR sample timed out: {tag}/{action}"
                        ) from error
                    except OSError as error:
                        raise contracts.ContractError(
                            f"independent RLR sample could not start: {tag}/{action}"
                        ) from error
                    if completed.returncode != 0:
                        stderr = (completed.stderr or b"")[-2048:].decode(
                            "utf-8",
                            errors="replace",
                        )
                        raise contracts.ContractError(
                            f"independent RLR sample crashed: {tag}/{action}; "
                            f"exit={completed.returncode}; stderr={stderr!r}"
                        )
                    result_size = os.fstat(output_fd).st_size
                    if (
                        result_size <= 0
                        or result_size > MAX_INDEPENDENT_RLR_RESULT_BYTES
                    ):
                        raise contracts.ContractError(
                            f"independent RLR result is missing/oversized: "
                            f"{tag}/{action}"
                        )
                    result_stream.seek(0)
                    result_payload = result_stream.read(
                        MAX_INDEPENDENT_RLR_RESULT_BYTES + 1
                    )
                    if len(result_payload) != result_size:
                        raise contracts.ContractError(
                            f"independent RLR result changed while reading: "
                            f"{tag}/{action}"
                        )
            finally:
                os.close(snapshot_fd)
    finally:
        os.close(worker_fd)
        os.close(python_fd)

    observed = _load_independent_result(
        result_payload,
        request_sha256=request_sha256,
        subprocess_contract=subprocess_contract,
        renderer_contract=renderer_contract,
        tag=tag,
        frame_index=selected["frame_index"],
        selection_seed_sha256=selected["selection_seed_sha256"],
        trajectory_prefix_sha256=trajectory_prefix_sha256,
        trajectory_prefix_shape=trajectory_prefix.shape,
        n_samples_total=selected["n_samples_total"],
    )
    expected = selected["expected_rir"]
    remaining = selected["n_samples_total"] - (
        selected["frame_index"] * selected["samples_per_frame"]
    )
    if (
        observed.shape[1] < expected.shape[1]
        or (
            expected.shape[1] < remaining
            and observed.shape[1] != expected.shape[1]
        )
    ):
        raise contracts.ContractError(
            f"independent RLR result length changed: {tag}/{action}"
        )
    compared = observed[:, : expected.shape[1]]
    difference = compared.astype(np.float64) - expected.astype(np.float64)
    max_absolute_error = float(np.max(np.abs(difference)))
    rms_error = float(np.sqrt(np.mean(np.square(difference))))
    comparison = subprocess_contract["comparison"]
    if not np.allclose(
        compared,
        expected,
        rtol=float(comparison["relative_tolerance"]),
        atol=float(comparison["absolute_tolerance"]),
        equal_nan=False,
    ):
        raise contracts.ContractError(
            f"independent RLR sample disagrees with manifest RIR: "
            f"{tag}/{action}; max_abs={max_absolute_error:.9g}; "
            f"rms={rms_error:.9g}"
        )

    result_path = Path(f"{evidence_output_base}.npz")
    report_path = Path(f"{evidence_output_base}.json")
    result_sha256 = hashlib.sha256(result_payload).hexdigest()
    report = {
        "schema": INDEPENDENT_RLR_REGISTRATION_EVIDENCE_SCHEMA,
        "status": "passed",
        "source_tag": tag,
        "action": action,
        "frame_selection": {
            "algorithm": "sha256_seed_modulo_active_frame_count_v1",
            "seed_sha256": selected["selection_seed_sha256"],
            "selected_slot": selected["selected_slot"],
            "frame_index": selected["frame_index"],
            "active_frame_count": selected["active_frame_count"],
        },
        "temporal_coherence_input": {
            "coordinate_frame": "right-handed Z-up meters",
            "frame_range_inclusive": [0, selected["frame_index"]],
            "shape": list(trajectory_prefix.shape),
            "little_endian_float64_sha256": trajectory_prefix_sha256,
        },
        "expected_active_frame_rir": {
            "sha256": rir_descriptor["sha256"],
            "size_bytes": rir_descriptor["size_bytes"],
            "sample_length": int(expected.shape[1]),
        },
        "independent_worker_result": {
            "path": result_path.name,
            "sha256": result_sha256,
            "size_bytes": len(result_payload),
        },
        "subprocess_contract_sha256": _canonical_sha256(subprocess_contract),
        "runtime_packages_sha256": _canonical_sha256(
            subprocess_contract["runtime_packages"]
        ),
        "runtime_assurance_scope": {
            "classification": (
                "process_isolated_artifact_pinned_not_hermetic_v1"
            ),
            "byte_pinned": [
                "python_executable",
                "worker_source",
                "libRLRAudioPropagation.so",
            ],
            "version_and_path_pinned": [
                "numpy",
                "habitat_sim",
                "numpy_quaternion",
            ],
            "not_byte_pinned": [
                "numpy_native_dependency_closure",
                "habitat_sim_bindings",
                "egl_vulkan_gpu_driver_closure",
                "kernel_and_gpu_firmware",
            ],
        },
        "renderer_contract_sha256": _canonical_sha256(renderer_contract),
        "request_sha256": request_sha256,
        "comparison": {
            "channel_order": [0, 1],
            "absolute_tolerance": comparison["absolute_tolerance"],
            "relative_tolerance": comparison["relative_tolerance"],
            "max_absolute_error": max_absolute_error,
            "rms_error": rms_error,
        },
    }
    report_payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        _write_exclusive_readonly(result_path, result_payload)
        _write_exclusive_readonly(report_path, report_payload)
    except BaseException:
        for path in (result_path, report_path):
            try:
                path.chmod(0o600)
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return {
        "result_path": result_path,
        "report_path": report_path,
        "report": report,
        "stdout": (completed.stdout or b"").decode(
            "utf-8",
            errors="replace",
        )[-4096:],
    }


def _validate_registration_audio(
    action_record: Mapping[str, Any],
    *,
    tag: str,
    action: str,
    evidence_output_base: Path,
) -> dict[str, Any]:
    audio = _validate_audio(action_record, tag)
    audio["independent_rlr_sample"] = (
        _validate_independent_active_frame_rir_sample(
            render_manifest=audio["render_manifest_value"],
            validated_source_trajectory_scene_m=audio[
                "validated_source_trajectory_scene_m"
            ],
            tag=tag,
            action=action,
            evidence_output_base=evidence_output_base,
        )
    )
    return audio


def _future_spear_artifact(staged_path: Path, final_path: Path) -> dict[str, Any]:
    payload = load_authenticated_file_bytes(Path(staged_path))
    final_path = Path(final_path).resolve(strict=False)
    try:
        relative = final_path.relative_to(SPEAR_ROOT.resolve())
    except ValueError:
        try:
            tmp_relative = final_path.relative_to((SPEAR_ROOT / "tmp").resolve())
        except ValueError as error:
            raise contracts.ContractError(
                f"future registration evidence is outside SPEAR root: {final_path}"
            ) from error
        relative = Path("tmp") / tmp_relative
    return {
        "root_id": "spear_repo",
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _validate_apartment_registry(
    record: Mapping[str, Any],
    *,
    independent_evidence_staging_root: Path,
    independent_evidence_final_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    tag = record["tag"]
    authority = record.get("_authenticated_apartment_v2")
    if authority is not None and not isinstance(authority, Mapping):
        raise contracts.ContractError(
            f"Apartment v2 authority is malformed: {tag}"
        )
    walking_output = Path(record["actions"]["Walking"]["output_dir"]).resolve()
    registry_path = walking_output.parent / "registry" / f"{tag}.json"
    registry = contracts.load_json(registry_path)
    if (
        registry.get("schema_version") != APARTMENT_REGISTRY_SCHEMA
        or registry.get("usage_scope") != "research_candidate"
        or registry.get("formal_registry_promotion") is not False
        or registry.get("tag") != tag
        or registry.get("asset_id") != record["base_avatar_id"]
        or registry.get("sampled_attributes") != record["sampled_attributes"]
        or set(registry.get("clips", {})) != {"Walking", "Idle"}
    ):
        raise contracts.ContractError(f"Apartment registry identity changed: {tag}")
    for descriptor_name in ("animation_decision", "ue_import_result"):
        _descriptor_file(registry[descriptor_name], descriptor_name)
    if authority is not None and (
        any(
            registry.get("animation_decision", {}).get(field)
            != authority["inputs"]["animation_decision"].get(field)
            for field in ("path", "sha256", "size_bytes")
        )
        or any(
            registry.get("ue_import_result", {}).get(field)
            != authority["inputs"]["ue_import_result"].get(field)
            for field in ("path", "sha256", "size_bytes")
        )
    ):
        raise contracts.ContractError(
            f"Apartment registry detached from builder authority: {tag}"
        )
    decision = contracts.load_json(Path(registry["animation_decision"]["path"]))
    if (
        decision.get("asset_id") != record["base_avatar_id"]
        or decision.get("decision") != "approved_for_ue_apartment"
        or not decision.get("checks")
        or not all(decision["checks"].values())
    ):
        raise contracts.ContractError(f"animation decision not approved: {tag}")
    imported = contracts.load_json(Path(registry["ue_import_result"]["path"]))
    imports = [item for item in imported.get("results", []) if item.get("tag") == tag]
    expected_ue_source_sha256 = registry.get("ue_source_sha256")
    if authority is not None:
        lineage = authority["runtime_lineage"]
        reviewed = lineage["reviewed_animated_glb"]
        ue_import = lineage["ue_import_glb"]
        transcode = lineage["texture_transcode_manifest"]
        if (
            reviewed.get("path") != record["source_glb"]["path"]
            or reviewed.get("sha256") != record["source_glb"]["sha256"]
            or expected_ue_source_sha256 != ue_import.get("sha256")
            or (
                transcode is None
                and reviewed.get("sha256") != ue_import.get("sha256")
            )
            or (
                transcode is not None
                and reviewed.get("sha256") == ue_import.get("sha256")
            )
        ):
            raise contracts.ContractError(
                f"Apartment reviewed/UE runtime lineage changed: {tag}"
            )
        expected_ue_source_sha256 = ue_import["sha256"]
    if (
        len(imports) != 1
        or imports[0].get("status") != "passed"
        or set(imports[0].get("actions", [])) != {"Walking", "Idle"}
        or imports[0].get("source_sha256") != expected_ue_source_sha256
    ):
        raise contracts.ContractError(f"UE import readback failed: {tag}")

    extra: dict[str, Any] = {
        "apartment_registry": registry_path,
        "animation_decision": Path(registry["animation_decision"]["path"]).resolve(),
        "ue_import_result": Path(registry["ue_import_result"]["path"]).resolve(),
    }
    if authority is not None:
        extra.update(
            {
                "ue_import_jobs": Path(
                    authority["inputs"]["ue_import_jobs"]["path"]
                ).resolve(),
                "ue_import_preparation": Path(
                    authority["inputs"]["ue_import_preparation"]["path"]
                ).resolve(),
                "animation_decision_freeze_receipt": Path(
                    authority["inputs"][
                        "animation_decision_freeze_receipt"
                    ]["path"]
                ).resolve(),
                "generated_animal_emitter_measurement": Path(
                    authority["emitter_measurement"]["path"]
                ).resolve(),
            }
        )
        presentation_evidence = authority["presentation_evidence"]
        if apartment_builder._uses_compact_approval_evidence(
            presentation_evidence
        ):
            extra.update(
                {
                    "motion_style_approval": _descriptor_file(
                        presentation_evidence.get("motion_style_approval"),
                        "motion-style approval",
                    ),
                    "current_asset_short_readback": _descriptor_file(
                        presentation_evidence.get(
                            "current_asset_short_readback"
                        ),
                        "current-asset short readback",
                    ),
                }
            )
        else:
            extra.update(
                {
                    "owner_review_presentation_receipt": Path(
                        presentation_evidence["presentation_receipt"]["path"]
                    ).resolve(),
                    "owner_review_presentation_video": Path(
                        presentation_evidence["output_video"]["path"]
                    ).resolve(),
                }
            )
    for action in ("Walking", "Idle"):
        clip = registry["clips"][action]
        action_record = record["actions"][action]
        if clip.get("clip_id") != action_record["clip_id"]:
            raise contracts.ContractError(f"Apartment clip identity changed: {tag}/{action}")
        if authority is not None:
            clip_root = Path(action_record["output_dir"]).resolve()
            expected_clip_paths = {
                "spec": clip_root / "spec.json",
                "runtime_gate": clip_root / "runtime_gate.json",
                "actor_visual_metadata": (
                    clip_root / "videos" / "actor_visual_metadata.json"
                ),
                "apartment_video": (
                    clip_root / "videos" / "apartment_v1_view0.mp4"
                ),
                "topdown_review_video": (
                    clip_root / "videos" / "topdown_review.mp4"
                ),
                "annotated_review_video": (
                    clip_root / "videos" / "side_by_side_review_annotated.mp4"
                ),
            }
            rendered_spec = clip.get("spec")
            source_spec = action_record.get("spec_evidence")
            if (
                set(clip) != {"clip_id", *expected_clip_paths}
                or not isinstance(rendered_spec, Mapping)
                or not isinstance(source_spec, Mapping)
                or rendered_spec.get("sha256") != source_spec.get("sha256")
                or rendered_spec.get("size_bytes")
                != source_spec.get("size_bytes")
                or any(
                    _descriptor_file(
                        clip.get(name, {}),
                        f"{tag}/{action}/{name}",
                    )
                    != expected_path
                    for name, expected_path in expected_clip_paths.items()
                )
            ):
                raise contracts.ContractError(
                    f"Apartment rendered a different spec: {tag}/{action}"
                )
        for name, descriptor in clip.items():
            if name == "clip_id":
                continue
            extra[f"apartment_{action.lower()}_{name}"] = _descriptor_file(
                descriptor, f"{tag}/{action}/{name}"
            )
        evidence_staging_base = (
            Path(independent_evidence_staging_root)
            / record["base_avatar_id"]
            / action.lower()
        )
        evidence_final_base = (
            Path(independent_evidence_final_root)
            / record["base_avatar_id"]
            / action.lower()
        )
        audio = _validate_registration_audio(
            action_record,
            tag=tag,
            action=action,
            evidence_output_base=evidence_staging_base,
        )
        extra[f"apartment_{action.lower()}_binaural_audio"] = audio["audio"]
        extra[f"apartment_{action.lower()}_audio_schedule"] = audio["schedule"]
        extra[f"apartment_{action.lower()}_audio_render_manifest"] = audio[
            "render_manifest"
        ]
        for role in (
            "render_spec_snapshot",
            "binaural_solo",
            "scheduled_dry",
            "active_frame_rir",
            "acoustic_mesh",
            "acoustic_materials",
            "derived_rlr_materials",
        ):
            extra[f"apartment_{action.lower()}_{role}"] = audio[role]
        independent = audio["independent_rlr_sample"]
        extra[
            f"apartment_{action.lower()}_independent_rlr_sample"
        ] = _future_spear_artifact(
            independent["result_path"],
            Path(f"{evidence_final_base}.npz"),
        )
        extra[
            f"apartment_{action.lower()}_independent_rlr_sample_report"
        ] = _future_spear_artifact(
            independent["report_path"],
            Path(f"{evidence_final_base}.json"),
        )
        pinned_artifact = _pinned_audio_source_artifact(
            audio["pinned_dry_source"]
        )
        if (
            "animal_audio_pinned_dry_source" in extra
            and extra["animal_audio_pinned_dry_source"] != pinned_artifact
        ):
            raise contracts.ContractError(
                f"Walk/Idle pinned dry sources differ for {tag}"
            )
        extra["animal_audio_pinned_dry_source"] = pinned_artifact
    return registry_path, registry, extra


def upgrade_source_asset(
    source_asset: Mapping[str, Any],
    *,
    physical_measurements: Mapping[str, Any],
    added_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    asset = contracts.validate_source_asset_v2(source_asset)
    if asset["state_classification"] != "research_candidate":
        raise contracts.ContractError("Apartment revision requires a research candidate")
    upgraded = copy.deepcopy(asset)
    overlap = set(upgraded["artifacts"]) & set(added_artifacts)
    if overlap:
        raise contracts.ContractError(f"Apartment artifact roles already exist: {sorted(overlap)}")
    upgraded["artifacts"].update(copy.deepcopy(dict(added_artifacts)))
    upgraded["physical_measurements"] = copy.deepcopy(dict(physical_measurements))
    upgraded["qa"].update(
        {
            "binding": "passed",
            "walking": "passed",
            "idle": "passed",
            "ue_import_readback": "passed",
            "apartment_media": "passed",
            # The RLR render/evidence chain has passed, but the currently
            # pinned sources still have pending content and license review.
            "audio": "pending",
        }
    )
    upgraded["provenance"]["attempt_id"] = (
        f"apartment_{upgraded['request_sha256'][:16]}"
    )
    return contracts.validate_source_asset_v2(upgraded)


def register(
    *,
    source_asset_roots: Sequence[Path],
    apartment_manifests: Sequence[Path],
    measurement_batch: Path,
    output_root: Path,
) -> Path:
    output_root = Path(output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"output already exists: {output_root}")
    source_assets = _load_source_assets(source_asset_roots)
    apartment = _load_apartment_records(apartment_manifests)
    measurements = _load_measurements(measurement_batch, apartment)
    if set(apartment) != set(measurements):
        raise contracts.ContractError("Apartment and measurement asset sets differ")
    missing = set(apartment) - set(source_assets)
    if missing:
        raise contracts.ContractError(f"Apartment assets lack static source_asset_v2: {sorted(missing)}")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent)
    )
    try:
        entries = []
        for asset_id, record in sorted(apartment.items()):
            source = source_assets[asset_id]
            if (
                source["profile_schema_id"] != record["profile_schema_id"]
                or source["sampled_attributes"] != record["sampled_attributes"]
                or record["source_glb"]["sha256"]
                != _sha256(Path(record["source_glb"]["path"]))
            ):
                raise contracts.ContractError(f"Apartment/source identity changed: {asset_id}")
            independent_relative_root = Path("independent_rlr_samples")
            _registry_path, _registry, evidence_paths = _validate_apartment_registry(
                record,
                independent_evidence_staging_root=(
                    staging / independent_relative_root
                ),
                independent_evidence_final_root=(
                    output_root / independent_relative_root
                ),
            )
            measurement = measurements[asset_id]
            added = {
                "rigged_walk_idle_glb": static_registry.spear_artifact(
                    Path(record["source_glb"]["path"])
                ),
                "physical_measurement": static_registry.spear_artifact(
                    measurement["path"]
                ),
                **{
                    role: (
                        copy.deepcopy(path)
                        if isinstance(path, Mapping)
                        else static_registry.spear_artifact(path)
                    )
                    for role, path in sorted(evidence_paths.items())
                },
            }
            authority = record.get("_authenticated_apartment_v2")
            if isinstance(authority, Mapping):
                lineage = authority["runtime_lineage"]
                added["ue_import_compatible_glb"] = (
                    static_registry.spear_artifact(
                        Path(lineage["ue_import_glb"]["path"])
                    )
                )
                if lineage["texture_transcode_manifest"] is not None:
                    added["texture_transcode_manifest"] = (
                        static_registry.spear_artifact(
                            Path(
                                lineage["texture_transcode_manifest"]["path"]
                            )
                        )
                    )
            upgraded = upgrade_source_asset(
                source,
                physical_measurements=measurement["payload"]["physical_measurements"],
                added_artifacts=added,
            )
            destination = staging / "source_assets" / f"{asset_id}.json"
            contracts.write_json_no_replace(destination, upgraded)
            entries.append(
                {
                    "asset_id": asset_id,
                    "profile_schema_id": upgraded["profile_schema_id"],
                    "sampled_attributes": upgraded["sampled_attributes"],
                    "physical_measurements": upgraded["physical_measurements"],
                    "qa": upgraded["qa"],
                    "rights": upgraded["rights"],
                    "source_asset": {
                        "path": f"source_assets/{asset_id}.json",
                        "sha256": _sha256(destination),
                        "size_bytes": destination.stat().st_size,
                    },
                }
            )
        manifest: dict[str, Any] = {
            "schema": REGISTRY_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "source_asset_count": len(entries),
            "inputs": {
                "source_asset_roots": [str(Path(path).resolve()) for path in source_asset_roots],
                "apartment_manifests": [
                    static_registry.spear_artifact(Path(path)) for path in apartment_manifests
                ],
                "physical_measurement_batch": static_registry.spear_artifact(
                    measurement_batch
                ),
            },
            "source_assets": entries,
            "automatic_checks": {
                "all_static_source_assets_revalidated": True,
                "all_physical_measurements_observed": True,
                "all_animation_decisions_approved": True,
                "all_ue_imports_read_back": True,
                "all_walk_idle_apartment_media_passed": True,
                "all_species_audio_schedules_passed": True,
                "all_independent_active_frame_rir_samples_passed": True,
                "all_rights_blockers_preserved": all(
                    item["rights"]["status"] == "review_required"
                    and bool(item["rights"]["blockers"])
                    for item in entries
                ),
                "overall": "passed",
            },
        }
        manifest["registry_sha256"] = _hash_without(manifest, "registry_sha256")
        contracts.write_json_no_replace(staging / "registry_manifest.json", manifest)
        immutable._seal_readonly_tree(staging)
        os.rename(staging, output_root)
        return output_root / "registry_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-asset-root", action="append", required=True, type=Path)
    parser.add_argument("--apartment-manifest", action="append", required=True, type=Path)
    parser.add_argument("--measurement-batch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = register(
            source_asset_roots=args.source_asset_root,
            apartment_manifests=args.apartment_manifest,
            measurement_batch=args.measurement_batch,
            output_root=args.output_root,
        )
        payload = contracts.load_json(manifest)
    except (contracts.ContractError, OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"CONTROLLED_ANIMAL_APARTMENT_REGISTRATION_FAILED {error}", flush=True)
        return 2
    print(
        f"CONTROLLED_ANIMAL_APARTMENT_REGISTRATION_OK assets={payload['source_asset_count']} "
        f"output={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
