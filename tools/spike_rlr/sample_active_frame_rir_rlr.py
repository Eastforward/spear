#!/usr/bin/env python3
"""Rerender one native-binaural RIR frame in a pinned isolated process.

The worker is intentionally self-contained.  Before importing NumPy, Habitat,
or any repository module it authenticates its environment, Python executable,
and its own already-open source inode.  It never imports live SPEAR code.

The parent supplies:

* a read-only directory containing authenticated immutable input snapshots,
* an unlinked empty result file,
* the authenticated worker source inode used for this invocation, and
* the exact source-position prefix needed to reproduce RLR temporal coherence.

The request never contains an expected RIR.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import types
from typing import Any, Mapping


REQUEST_SCHEMA = "avengine_independent_rlr_sample_request_v2"
RESULT_SCHEMA = "avengine_independent_rlr_sample_result_v2"
SUBPROCESS_SCHEMA = "avengine_independent_rlr_subprocess_contract_v2"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
_REQUEST_KEYS = {
    "schema",
    "source_tag",
    "frame_index",
    "selection_seed_sha256",
    "source_position_coordinate_frame",
    "trajectory_prefix_shape",
    "trajectory_prefix_sha256",
    "warmup_source_positions_scene_m",
    "mic_position_scene_m",
    "mic_yaw_deg",
    "quality_mode",
    "renderer_contract",
    "subprocess_contract",
    "spec",
    "acoustic_mesh",
    "acoustic_materials",
    "derived_rlr_materials",
}
_DESCRIPTOR_KEYS = {"name", "sha256", "size_bytes"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_TAG = re.compile(r"^[A-Za-z0-9_]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COORDINATE_FRAME = "right-handed Z-up meters"

_PYTHON_CONTRACT = {
    "path": "/data/jzy/miniconda3/envs/ss2/bin/python",
    "resolved_path": "/data/jzy/miniconda3/envs/ss2/bin/python3.9",
    "sha256": (
        "363d6db32b097aa00bcbaab4fa02c76d66640ae5626ccc14dc73a3eba206ca03"
    ),
    "size_bytes": 16267992,
    "version": "3.9",
    "isolated_flags": ["-I", "-B", "-S"],
}
_WORKER_PATH = "tools/spike_rlr/sample_active_frame_rir_rlr.py"
_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "2",
    "DISPLAY": ":99",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LD_PRELOAD": (
        "/usr/lib/x86_64-linux-gnu/libEGL.so.1:"
        "/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0"
    ),
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUNBUFFERED": "1",
    "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
    "XDG_RUNTIME_DIR": "/run/user/1003",
}
_RUNTIME_PACKAGES = {
    "site_packages_path": (
        "/data/jzy/miniconda3/envs/ss2/lib/python3.9/site-packages"
    ),
    "numpy_version": "1.26.4",
    "habitat_sim_version": "0.2.2",
    "numpy_quaternion_version": "2023.0.4",
}
_TIMEOUT_S = 600.0
_COMPARISON = {
    "absolute_tolerance": 1.0e-6,
    "relative_tolerance": 1.0e-4,
    "channel_order": [0, 1],
}

ACOUSTIC_SCENE_CONTRACT_SCHEMA = "avengine_acoustic_scene_contract_v1"
RLR_RENDERER_CONTRACT_SCHEMA = "avengine_rlr_renderer_contract_v1"
RLR_RENDERER_BUILD_ID = (
    "habitat_sim_0.2.2_rlr_native_binaural_active_frame_replay_v3"
)
APPROVED_RLR_RUNTIME_ARTIFACTS = {
    "rlr_audio_propagation_shared_library": {
        "basename": "libRLRAudioPropagation.so",
        "sha256": (
            "31e948eef4908d8cbb403b5f445d9d0eab59fc81b05a658538f8795984f9bfb4"
        ),
        "size_bytes": 7327344,
    },
}
RLR_QUALITY_INDIRECT_RAY_COUNTS = {
    "low": 100,
    "high": 500,
    "max": 5000,
}
RLR_FIXED_ACOUSTICS_CONFIG = {
    "thread_count": 4,
    "direct": True,
    "indirect": True,
    "diffraction": True,
    "transmission": True,
    "temporal_coherence": True,
    "source_ray_count": 200,
    "indirect_ray_depth": 50,
    "frequency_bands": 4,
    "unit_scale_m": 1.0,
}
APPROVED_APARTMENT_ACOUSTIC_SCENE_CONTRACT = {
    "schema": ACOUSTIC_SCENE_CONTRACT_SCHEMA,
    "scene_id": "apartment_v1_ssot_acoustic_shell",
    "spec_version": "apartment_v1",
    "acoustic_mesh": {
        "root_id": "spear_repo",
        "path": "tmp/spike_rlr/apartment_v1_mesh.glb",
        "sha256": (
            "c91484db254e224d1f46504521b9bbaab5cb3c01d406ac6d548486a6059654d9"
        ),
        "size_bytes": 5816,
    },
    "acoustic_materials": {
        "root_id": "spear_repo",
        "path": "tmp/spike_rlr/apartment_v1_materials.json",
        "sha256": (
            "9ca50861ba3c85117d60fc3ee257d4186b20c7e7ef234cddf32d54c83d491826"
        ),
        "size_bytes": 8403,
    },
    "formal_registration_authorized": False,
}
RLR_MATERIAL_BANDS_HZ = (125.0, 500.0, 2000.0, 8000.0)

np = None


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _event(stage: str, started: float) -> None:
    print(
        json.dumps(
            {"stage": stage, "elapsed_s": time.monotonic() - started},
            sort_keys=True,
        ),
        flush=True,
    )


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
        raise ValueError(f"independent RLR {label} is not a bounded regular file")
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
        raise ValueError(f"independent RLR {label} changed while being read")
    return b"".join(chunks)


def _snapshot_payload(
    snapshot_fd: int,
    record: Mapping[str, Any],
    label: str,
) -> bytes:
    if not isinstance(record, Mapping) or set(record) != _DESCRIPTOR_KEYS:
        raise ValueError(f"independent RLR {label} descriptor is malformed")
    name = record.get("name")
    sha256 = record.get("sha256")
    size_bytes = record.get("size_bytes")
    if (
        not isinstance(name, str)
        or not _SAFE_NAME.fullmatch(name)
        or name in {".", ".."}
        or not isinstance(sha256, str)
        or not _SHA256.fullmatch(sha256)
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or size_bytes > MAX_SNAPSHOT_BYTES
    ):
        raise ValueError(f"independent RLR {label} descriptor is malformed")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=snapshot_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            raise ValueError(f"independent RLR {label} snapshot is writable")
        payload = _pread_regular_file(
            descriptor,
            max_size_bytes=MAX_SNAPSHOT_BYTES,
            label=f"{label} snapshot",
        )
    finally:
        os.close(descriptor)
    if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"independent RLR {label} descriptor changed")
    return payload


def _approved_subprocess_contract(worker_payload: bytes) -> dict[str, Any]:
    return {
        "schema": SUBPROCESS_SCHEMA,
        "python": copy.deepcopy(_PYTHON_CONTRACT),
        "worker": {
            "root_id": "spear_repo",
            "path": _WORKER_PATH,
            "sha256": hashlib.sha256(worker_payload).hexdigest(),
            "size_bytes": len(worker_payload),
        },
        "environment": copy.deepcopy(_ENVIRONMENT),
        "runtime_packages": copy.deepcopy(_RUNTIME_PACKAGES),
        "timeout_s": _TIMEOUT_S,
        "comparison": copy.deepcopy(_COMPARISON),
    }


def _validate_runtime_identity(worker_fd: int) -> tuple[dict[str, Any], str, str]:
    if dict(os.environ) != _ENVIRONMENT:
        raise ValueError("independent RLR subprocess environment changed")
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or any(
            "site-packages" in entry or "dist-packages" in entry
            for entry in sys.path
        )
    ):
        raise ValueError("independent RLR Python isolation flags changed")

    executable = Path(sys.executable).resolve(strict=True)
    if str(executable) != _PYTHON_CONTRACT["resolved_path"]:
        raise ValueError("independent RLR Python executable changed")
    python_descriptor = os.open(
        executable,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        python_payload = _pread_regular_file(
            python_descriptor,
            max_size_bytes=64 * 1024 * 1024,
            label="Python executable",
        )
    finally:
        os.close(python_descriptor)
    if (
        len(python_payload) != _PYTHON_CONTRACT["size_bytes"]
        or hashlib.sha256(python_payload).hexdigest()
        != _PYTHON_CONTRACT["sha256"]
        or sys.version_info[:2]
        != tuple(
            int(part) for part in _PYTHON_CONTRACT["version"].split(".")
        )
    ):
        raise ValueError("independent RLR Python runtime is not approved")

    worker_payload = _pread_regular_file(
        worker_fd,
        max_size_bytes=4 * 1024 * 1024,
        label="worker source",
    )
    worker_sha256 = hashlib.sha256(worker_payload).hexdigest()
    return (
        _approved_subprocess_contract(worker_payload),
        hashlib.sha256(python_payload).hexdigest(),
        worker_sha256,
    )


def _load_request(snapshot_fd: int, request_name: str) -> tuple[dict[str, Any], str]:
    if (
        not isinstance(request_name, str)
        or not _SAFE_NAME.fullmatch(request_name)
        or request_name in {".", ".."}
    ):
        raise ValueError("independent RLR request name is malformed")
    descriptor = os.open(
        request_name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=snapshot_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            raise ValueError("independent RLR request snapshot is writable")
        request_payload = _pread_regular_file(
            descriptor,
            max_size_bytes=MAX_REQUEST_BYTES,
            label="request snapshot",
        )
    finally:
        os.close(descriptor)
    try:
        request = json.loads(request_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("independent RLR request is not valid UTF-8 JSON") from error
    if (
        not isinstance(request, dict)
        or set(request) != _REQUEST_KEYS
        or request.get("schema") != REQUEST_SCHEMA
    ):
        raise ValueError("independent RLR request schema changed")
    return request, hashlib.sha256(request_payload).hexdigest()


def _activate_runtime(subprocess_contract: Mapping[str, Any]) -> None:
    global np

    runtime = subprocess_contract.get("runtime_packages")
    if runtime != _RUNTIME_PACKAGES:
        raise ValueError("independent RLR runtime package contract changed")
    site_packages = Path(runtime["site_packages_path"]).resolve(strict=True)
    if str(site_packages) != runtime["site_packages_path"]:
        raise ValueError("independent RLR site-packages path changed")
    sys.path.append(str(site_packages))
    import numpy as numpy_module

    if (
        numpy_module.__version__ != runtime["numpy_version"]
        or site_packages not in Path(numpy_module.__file__).resolve().parents
    ):
        raise ValueError("independent RLR NumPy runtime changed")
    np = numpy_module


def _finite_vector(value: Any, *, label: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    result = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{label} contains a non-finite value")
        result.append(float(item))
    return result


def build_rlr_materials_payload(materials_sidecar: Mapping[str, Any]) -> bytes:
    if not isinstance(materials_sidecar, Mapping):
        raise ValueError("acoustic materials sidecar must be an object")
    materials = materials_sidecar.get("materials")
    if not isinstance(materials, list) or not materials:
        raise ValueError("acoustic materials sidecar has no materials")

    def interleave(values: list[float]) -> list[float]:
        return [
            value
            for frequency, coefficient in zip(RLR_MATERIAL_BANDS_HZ, values)
            for value in (frequency, coefficient)
        ]

    converted = []
    seen_names = set()
    for index, material in enumerate(materials):
        if not isinstance(material, Mapping):
            raise ValueError(f"acoustic material {index} must be an object")
        name = material.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError("acoustic material names are missing or duplicated")
        seen_names.add(name)
        absorption = _finite_vector(
            material.get("alpha"),
            label=f"acoustic material {name!r} alpha",
            length=4,
        )
        transmission = _finite_vector(
            material.get("transmission"),
            label=f"acoustic material {name!r} transmission",
            length=4,
        )
        scattering = material.get("scattering")
        if (
            isinstance(scattering, bool)
            or not isinstance(scattering, (int, float))
            or not math.isfinite(float(scattering))
        ):
            raise ValueError(
                f"acoustic material {name!r} scattering is non-finite"
            )
        converted.append(
            {
                "name": name,
                "absorption": interleave(absorption),
                "scattering": interleave([float(scattering)] * 4),
                "transmission": interleave(transmission),
                "labels": [name],
            }
        )
    return (
        json.dumps({"materials": converted}, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def validate_approved_acoustic_scene_inputs(
    spec: Mapping[str, Any],
    *,
    mesh_payload: bytes,
    materials_payload: bytes,
) -> dict[str, Any]:
    if (
        not isinstance(spec, Mapping)
        or spec.get("spec_version") != "apartment_v1"
    ):
        raise ValueError("controlled RLR sample requires apartment_v1")
    for label, payload in (
        ("acoustic_mesh", mesh_payload),
        ("acoustic_materials", materials_payload),
    ):
        expected = APPROVED_APARTMENT_ACOUSTIC_SCENE_CONTRACT[label]
        if (
            len(payload) != expected["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"{label} does not match the approved scene contract")
    return copy.deepcopy(APPROVED_APARTMENT_ACOUSTIC_SCENE_CONTRACT)


def approved_rlr_renderer_contract(
    *,
    sample_rate_hz: int,
    quality_mode: str,
) -> dict[str, Any]:
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 0
        or quality_mode not in RLR_QUALITY_INDIRECT_RAY_COUNTS
    ):
        raise ValueError("independent RLR renderer inputs changed")
    return {
        "schema": RLR_RENDERER_CONTRACT_SCHEMA,
        "build_id": RLR_RENDERER_BUILD_ID,
        "backend": "habitat_sim_rlr_audio_sensor",
        "habitat_sim_version": "0.2.2",
        "python_bootstrap": "audio_only_optional_dependency_fail_closed_stub_v3",
        "runtime_artifacts": copy.deepcopy(APPROVED_RLR_RUNTIME_ARTIFACTS),
        "simulator": {
            "enable_physics": False,
            "create_renderer": False,
            "load_semantic_mesh": False,
            "requires_textures": False,
        },
        "sensor": {
            "uuid": "audio_sensor",
            "position_m": [0.0, 0.0, 0.0],
            "enable_materials_from_glb": False,
            "channel_layout": "binaural_native",
            "channel_count": 2,
            "native_channel_order": [0, 1],
        },
        "acoustics": {
            **copy.deepcopy(RLR_FIXED_ACOUSTICS_CONFIG),
            "sample_rate_hz": sample_rate_hz,
            "indirect_ray_count": RLR_QUALITY_INDIRECT_RAY_COUNTS[quality_mode],
        },
        "quality_mode": quality_mode,
        "quality_indirect_ray_counts": copy.deepcopy(
            RLR_QUALITY_INDIRECT_RAY_COUNTS
        ),
        "formal_registration_authorized": False,
    }


def _read_regular_path(path: Path, *, max_size_bytes: int, label: str) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        return _pread_regular_file(
            descriptor,
            max_size_bytes=max_size_bytes,
            label=label,
        )
    finally:
        os.close(descriptor)


def _validate_runtime_artifact(
    *,
    artifact_name: str,
    path: Path,
) -> None:
    expected = APPROVED_RLR_RUNTIME_ARTIFACTS.get(artifact_name)
    payload = _read_regular_path(
        path,
        max_size_bytes=64 * 1024 * 1024,
        label=artifact_name,
    )
    if (
        expected is None
        or path.name != expected["basename"]
        or len(payload) != expected["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != expected["sha256"]
    ):
        raise ValueError(f"RLR runtime artifact is not approved: {artifact_name}")


def _install_habitat_audio_only_scipy_stub() -> None:
    if "scipy" in sys.modules or "scipy.stats" in sys.modules:
        return

    class _UnavailableTruncatedNormal:
        @staticmethod
        def rvs(*_args, **_kwargs):
            raise RuntimeError(
                "SciPy noisy locomotion is disabled in the audio-only RLR worker"
            )

    class _UnavailableCubicSpline:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                "SciPy random-scene splines are disabled in the controlled "
                "audio-only RLR worker"
            )

    scipy_module = types.ModuleType("scipy")
    scipy_module.__path__ = []
    scipy_module.__version__ = "1.10.0"
    stats_module = types.ModuleType("scipy.stats")
    stats_module.truncnorm = _UnavailableTruncatedNormal()
    interpolate_module = types.ModuleType("scipy.interpolate")
    interpolate_module.CubicSpline = _UnavailableCubicSpline
    scipy_module.stats = stats_module
    scipy_module.interpolate = interpolate_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.stats"] = stats_module
    sys.modules["scipy.interpolate"] = interpolate_module
    if "torch" not in sys.modules:
        sys.modules["torch"] = None


def _habitat_from_scene(pos_scene):
    x, y, z = pos_scene
    return np.array([x, z, -y], dtype=np.float32)


def _habitat_agent_yaw_deg_for_scene_yaw_deg(scene_yaw_deg: float) -> float:
    return float((float(scene_yaw_deg) - 90.0) % 360.0)


def _set_agent_pose(sim, mic_pos_scene, mic_yaw_deg: float = 90.0) -> None:
    import quaternion

    if getattr(quaternion, "__version__", None) != _RUNTIME_PACKAGES[
        "numpy_quaternion_version"
    ]:
        raise ValueError("independent RLR quaternion runtime changed")
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = _habitat_from_scene(mic_pos_scene)
    agent_yaw = np.deg2rad(
        _habitat_agent_yaw_deg_for_scene_yaw_deg(mic_yaw_deg)
    )
    half = agent_yaw / 2.0
    state.rotation = np.quaternion(
        float(np.cos(half)),
        0.0,
        float(np.sin(half)),
        0.0,
    )
    agent.set_state(state)


def build_rlr_sim(
    glb_path,
    materials_json_path,
    sample_rate=16000,
    channel_layout="ambisonics",
    indirect_ray_count=500,
):
    _install_habitat_audio_only_scipy_stub()
    import habitat_sim
    from habitat_sim.sensor import RLRAudioPropagationChannelLayoutType

    if getattr(habitat_sim, "__version__", None) != _RUNTIME_PACKAGES[
        "habitat_sim_version"
    ]:
        raise ValueError("independent RLR Habitat runtime changed")
    rlr_library_path = (
        Path(habitat_sim.__file__).resolve().parent
        / "_ext"
        / "libRLRAudioPropagation.so"
    )
    _validate_runtime_artifact(
        artifact_name="rlr_audio_propagation_shared_library",
        path=rlr_library_path,
    )

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(glb_path)
    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = False
    sim_cfg.load_semantic_mesh = False
    sim_cfg.requires_textures = False
    agent_cfg = habitat_sim.AgentConfiguration()
    audio_spec = habitat_sim.AudioSensorSpec()
    audio_spec.uuid = "audio_sensor"
    audio_spec.position = [0.0, 0.0, 0.0]
    audio_spec.enableMaterials = False
    if channel_layout == "binaural":
        audio_spec.channelLayout.type = (
            RLRAudioPropagationChannelLayoutType.Binaural
        )
        audio_spec.channelLayout.channelCount = 2
    else:
        raise ValueError("independent RLR worker only permits native binaural")
    audio_spec.acousticsConfig.sampleRate = sample_rate
    audio_spec.acousticsConfig.threadCount = RLR_FIXED_ACOUSTICS_CONFIG[
        "thread_count"
    ]
    audio_spec.acousticsConfig.direct = RLR_FIXED_ACOUSTICS_CONFIG["direct"]
    audio_spec.acousticsConfig.indirect = RLR_FIXED_ACOUSTICS_CONFIG["indirect"]
    audio_spec.acousticsConfig.diffraction = RLR_FIXED_ACOUSTICS_CONFIG[
        "diffraction"
    ]
    audio_spec.acousticsConfig.transmission = RLR_FIXED_ACOUSTICS_CONFIG[
        "transmission"
    ]
    audio_spec.acousticsConfig.temporalCoherence = RLR_FIXED_ACOUSTICS_CONFIG[
        "temporal_coherence"
    ]
    audio_spec.acousticsConfig.indirectRayCount = indirect_ray_count
    audio_spec.acousticsConfig.sourceRayCount = RLR_FIXED_ACOUSTICS_CONFIG[
        "source_ray_count"
    ]
    audio_spec.acousticsConfig.indirectRayDepth = RLR_FIXED_ACOUSTICS_CONFIG[
        "indirect_ray_depth"
    ]
    audio_spec.acousticsConfig.frequencyBands = RLR_FIXED_ACOUSTICS_CONFIG[
        "frequency_bands"
    ]
    audio_spec.acousticsConfig.unitScale = RLR_FIXED_ACOUSTICS_CONFIG[
        "unit_scale_m"
    ]
    agent_cfg.sensor_specifications = [audio_spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioMaterialsJSON(str(materials_json_path))
    return sim, audio_sensor


def _write_result(output_fd: int, arrays: Mapping[str, Any]) -> None:
    metadata = os.fstat(output_fd)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 0:
        raise ValueError("independent RLR result descriptor is not empty")
    os.lseek(output_fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(output_fd), "wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _validated_float_vector(value: Any, *, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            not isinstance(item, float) or not math.isfinite(item)
            for item in value
        )
    ):
        raise ValueError(f"independent RLR {label} is malformed")
    return value


def _validated_trajectory_prefix(
    request: Mapping[str, Any],
    *,
    frame_index: int,
):
    raw_prefix = request.get("warmup_source_positions_scene_m")
    expected_shape = [frame_index + 1, 3]
    if (
        request.get("trajectory_prefix_shape") != expected_shape
        or not isinstance(raw_prefix, list)
        or len(raw_prefix) != frame_index + 1
    ):
        raise ValueError("independent RLR trajectory prefix shape changed")
    for index, position in enumerate(raw_prefix):
        _validated_float_vector(
            position,
            label=f"trajectory prefix frame {index}",
        )
    trajectory_prefix = np.asarray(raw_prefix, dtype=np.dtype("<f8"))
    trajectory_prefix_sha256 = hashlib.sha256(
        np.ascontiguousarray(trajectory_prefix).tobytes(order="C")
    ).hexdigest()
    if (
        trajectory_prefix.shape != tuple(expected_shape)
        or not np.all(np.isfinite(trajectory_prefix))
        or request.get("trajectory_prefix_sha256") != trajectory_prefix_sha256
    ):
        raise ValueError("independent RLR trajectory prefix changed")
    return trajectory_prefix, trajectory_prefix_sha256


def run(
    *,
    snapshot_fd: int,
    request_name: str,
    output_fd: int,
    worker_fd: int,
) -> None:
    started = time.monotonic()
    snapshot_metadata = os.fstat(snapshot_fd)
    if (
        not stat.S_ISDIR(snapshot_metadata.st_mode)
        or stat.S_IMODE(snapshot_metadata.st_mode) & 0o222
    ):
        raise ValueError("independent RLR input snapshot root is writable")
    subprocess_contract, python_sha256, worker_sha256 = (
        _validate_runtime_identity(worker_fd)
    )
    request, request_sha256 = _load_request(snapshot_fd, request_name)
    if request.get("subprocess_contract") != subprocess_contract:
        raise ValueError("independent RLR subprocess contract changed")
    _activate_runtime(subprocess_contract)

    spec_payload = _snapshot_payload(snapshot_fd, request["spec"], "spec")
    mesh_payload = _snapshot_payload(
        snapshot_fd,
        request["acoustic_mesh"],
        "mesh",
    )
    materials_payload = _snapshot_payload(
        snapshot_fd,
        request["acoustic_materials"],
        "materials",
    )
    derived_payload = _snapshot_payload(
        snapshot_fd,
        request["derived_rlr_materials"],
        "derived materials",
    )
    try:
        spec = json.loads(spec_payload.decode("utf-8"))
        materials = json.loads(materials_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("independent RLR JSON snapshot is malformed") from error
    if not isinstance(spec, dict) or not isinstance(materials, dict):
        raise ValueError("independent RLR JSON snapshot is not an object")
    validate_approved_acoustic_scene_inputs(
        spec,
        mesh_payload=mesh_payload,
        materials_payload=materials_payload,
    )
    if derived_payload != build_rlr_materials_payload(materials):
        raise ValueError("independent RLR derived materials changed")

    sample_rate = int(spec["audio_config"]["sample_rate_hz"])
    quality_mode = request["quality_mode"]
    renderer = approved_rlr_renderer_contract(
        sample_rate_hz=sample_rate,
        quality_mode=quality_mode,
    )
    if request.get("renderer_contract") != renderer:
        raise ValueError("independent RLR renderer contract changed")
    tag = request.get("source_tag")
    frame_index = request.get("frame_index")
    seed_sha256 = request.get("selection_seed_sha256")
    raw_sources = spec.get("sources")
    source_tags = (
        [source.get("tag") for source in raw_sources]
        if isinstance(raw_sources, list)
        and all(isinstance(source, Mapping) for source in raw_sources)
        else []
    )
    if (
        not isinstance(tag, str)
        or not _SAFE_TAG.fullmatch(tag)
        or source_tags.count(tag) != 1
        or isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
        or not isinstance(seed_sha256, str)
        or not _SHA256.fullmatch(seed_sha256)
        or request.get("source_position_coordinate_frame") != _COORDINATE_FRAME
        or spec.get("coordinate_frame", {}).get("system") != _COORDINATE_FRAME
    ):
        raise ValueError("independent RLR sample identity is malformed")

    expected_shape = [frame_index + 1, 3]
    trajectory_prefix, trajectory_prefix_sha256 = (
        _validated_trajectory_prefix(
            request,
            frame_index=frame_index,
        )
    )
    mic_position = np.asarray(
        _validated_float_vector(
            request.get("mic_position_scene_m"),
            label="microphone position",
        ),
        dtype=np.float64,
    )
    mic_yaw = request.get("mic_yaw_deg")
    if (
        not isinstance(mic_yaw, float)
        or not math.isfinite(mic_yaw)
        or not np.array_equal(
            mic_position,
            np.asarray(spec.get("mic", {}).get("pos_m"), dtype=np.float64),
        )
        or not math.isclose(
            mic_yaw,
            float(spec.get("mic", {}).get("yaw_deg")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("independent RLR microphone transform changed")
    _event("inputs_authenticated", started)

    pinned_root = Path("/proc/self/fd") / str(snapshot_fd)
    if not pinned_root.is_dir():
        raise ValueError("independent RLR pinned snapshot path is unavailable")
    sim = None
    try:
        sim, sensor = build_rlr_sim(
            pinned_root / request["acoustic_mesh"]["name"],
            pinned_root / request["derived_rlr_materials"]["name"],
            sample_rate=sample_rate,
            channel_layout="binaural",
            indirect_ray_count=renderer["acoustics"]["indirect_ray_count"],
        )
        _event("simulator_ready", started)
        _set_agent_pose(sim, mic_position, mic_yaw)
        observed = None
        for position in trajectory_prefix:
            sensor.setAudioSourceTransform(_habitat_from_scene(position))
            observed = np.asarray(
                sim.get_sensor_observations()["audio_sensor"],
                dtype=np.float32,
            )
        if observed is None:
            raise ValueError("independent RLR produced no observation")
        if observed.ndim == 2 and observed.shape[0] > observed.shape[1]:
            observed = observed.T
        elif observed.ndim == 1:
            observed = observed[None, :]
        if (
            observed.dtype != np.dtype(np.float32)
            or observed.ndim != 2
            or observed.shape[0] != 2
            or observed.shape[1] <= 0
            or not np.all(np.isfinite(observed))
            or float(np.max(np.abs(observed))) <= 0.0
        ):
            raise ValueError("independent RLR observation is malformed")
        _event("observation_ready", started)
        _write_result(
            output_fd,
            {
                "schema": np.asarray(RESULT_SCHEMA),
                "request_sha256": np.asarray(request_sha256),
                "subprocess_contract_sha256": np.asarray(
                    _canonical_sha256(subprocess_contract)
                ),
                "renderer_contract_sha256": np.asarray(
                    _canonical_sha256(renderer)
                ),
                "environment_contract_sha256": np.asarray(
                    _canonical_sha256(subprocess_contract["environment"])
                ),
                "runtime_packages_sha256": np.asarray(
                    _canonical_sha256(subprocess_contract["runtime_packages"])
                ),
                "python_sha256": np.asarray(python_sha256),
                "worker_sha256": np.asarray(worker_sha256),
                "source_tag": np.asarray(tag),
                "frame_index": np.asarray(frame_index, dtype=np.int64),
                "selection_seed_sha256": np.asarray(seed_sha256),
                "source_position_coordinate_frame": np.asarray(
                    _COORDINATE_FRAME
                ),
                "trajectory_prefix_sha256": np.asarray(
                    trajectory_prefix_sha256
                ),
                "trajectory_prefix_shape": np.asarray(
                    expected_shape,
                    dtype=np.int64,
                ),
                "channel_order": np.asarray([0, 1], dtype=np.int64),
                "rir": observed.astype(np.float32, copy=False),
            },
        )
    finally:
        if sim is not None:
            sim.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-fd", type=int, required=True)
    parser.add_argument("--request-name", required=True)
    parser.add_argument("--output-fd", type=int, required=True)
    parser.add_argument("--worker-fd", type=int, required=True)
    args = parser.parse_args()
    run(
        snapshot_fd=args.snapshot_fd,
        request_name=args.request_name,
        output_fd=args.output_fd,
        worker_fd=args.worker_fd,
    )


if __name__ == "__main__":
    main()
