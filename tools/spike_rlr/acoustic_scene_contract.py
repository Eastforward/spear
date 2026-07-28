"""Approved acoustic environment contract for controlled Apartment reviews."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping


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
INDEPENDENT_RLR_SAMPLE_TIMEOUT_S = 600.0
APPROVED_INDEPENDENT_RLR_SUBPROCESS_CONTRACT = {
    "schema": "avengine_independent_rlr_subprocess_contract_v2",
    "python": {
        "path": "/data/jzy/miniconda3/envs/ss2/bin/python",
        "resolved_path": "/data/jzy/miniconda3/envs/ss2/bin/python3.9",
        "sha256": (
            "363d6db32b097aa00bcbaab4fa02c76d66640ae5626ccc14dc73a3eba206ca03"
        ),
        "size_bytes": 16267992,
        "version": "3.9",
        "isolated_flags": ["-I", "-B", "-S"],
    },
    "worker": {
        "root_id": "spear_repo",
        "path": "tools/spike_rlr/sample_active_frame_rir_rlr.py",
        # The registration launcher authenticates and executes this same open
        # inode, so a pathname replacement cannot switch code after hashing.
        "sha256": (
            "7e2d0991f3b8dbb1b2adbcf3e14a3795b698b6f06baba6e42c98af966be0e1d7"
        ),
        "size_bytes": 34079,
    },
    "environment": {
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
    },
    "runtime_packages": {
        "site_packages_path": (
            "/data/jzy/miniconda3/envs/ss2/lib/python3.9/site-packages"
        ),
        "numpy_version": "1.26.4",
        "habitat_sim_version": "0.2.2",
        "numpy_quaternion_version": "2023.0.4",
    },
    "timeout_s": INDEPENDENT_RLR_SAMPLE_TIMEOUT_S,
    "comparison": {
        "absolute_tolerance": 1.0e-6,
        "relative_tolerance": 1.0e-4,
        "channel_order": [0, 1],
    },
}

# These are the reviewed SSOT apartment shell artifacts used by the current
# dynamic-animal admission route.  Changing either artifact is an explicit
# contract migration and intentionally invalidates resume/registration.
APPROVED_APARTMENT_ACOUSTIC_SCENE_CONTRACT = {
    "schema": ACOUSTIC_SCENE_CONTRACT_SCHEMA,
    "scene_id": "apartment_v1_ssot_acoustic_shell",
    "spec_version": "apartment_v1",
    "acoustic_mesh": {
        "root_id": "spear_repo",
        "path": "tmp/spike_rlr/apartment_v1_mesh.glb",
        "sha256": "c91484db254e224d1f46504521b9bbaab5cb3c01d406ac6d548486a6059654d9",
        "size_bytes": 5816,
    },
    "acoustic_materials": {
        "root_id": "spear_repo",
        "path": "tmp/spike_rlr/apartment_v1_materials.json",
        "sha256": "9ca50861ba3c85117d60fc3ee257d4186b20c7e7ef234cddf32d54c83d491826",
        "size_bytes": 8403,
    },
    "formal_registration_authorized": False,
}


def approved_acoustic_scene_contract(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the only approved contract for the supplied controlled spec."""
    if not isinstance(spec, Mapping):
        raise ValueError("RLR render spec must be an object")
    if spec.get("spec_version") != "apartment_v1":
        raise ValueError(
            "controlled binaural admission requires the approved apartment_v1 "
            "acoustic scene"
        )
    return copy.deepcopy(APPROVED_APARTMENT_ACOUSTIC_SCENE_CONTRACT)


def validate_approved_acoustic_scene_inputs(
    spec: Mapping[str, Any],
    *,
    mesh_payload: bytes,
    materials_payload: bytes,
) -> dict[str, Any]:
    """Authenticate CLI-selected mesh/material bytes against the approval."""
    contract = approved_acoustic_scene_contract(spec)
    for label, payload in (
        ("acoustic_mesh", mesh_payload),
        ("acoustic_materials", materials_payload),
    ):
        expected = contract[label]
        if (
            len(payload) != expected["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"{label} does not match the approved scene contract")
    return contract


def approved_rlr_renderer_contract(
    *,
    sample_rate_hz: int,
    quality_mode: str,
) -> dict[str, Any]:
    """Bind every RLR option that affects controlled native-binaural output."""
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int):
        raise ValueError("RLR renderer sample rate must be an integer")
    if sample_rate_hz <= 0:
        raise ValueError("RLR renderer sample rate must be positive")
    if quality_mode not in RLR_QUALITY_INDIRECT_RAY_COUNTS:
        raise ValueError(f"unsupported RLR quality mode: {quality_mode!r}")
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
            "indirect_ray_count": RLR_QUALITY_INDIRECT_RAY_COUNTS[
                quality_mode
            ],
        },
        "quality_mode": quality_mode,
        "quality_indirect_ray_counts": copy.deepcopy(
            RLR_QUALITY_INDIRECT_RAY_COUNTS
        ),
        "formal_registration_authorized": False,
    }


def validate_approved_rlr_runtime_artifact(
    *,
    artifact_name: str,
    path: str,
    payload: bytes,
) -> dict[str, Any]:
    """Authenticate the reviewed native RLR binary before simulator startup."""
    expected = APPROVED_RLR_RUNTIME_ARTIFACTS.get(artifact_name)
    if (
        expected is None
        or path.rsplit("/", 1)[-1] != expected["basename"]
        or len(payload) != expected["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != expected["sha256"]
    ):
        raise ValueError(f"RLR runtime artifact is not approved: {artifact_name}")
    return copy.deepcopy(expected)


def approved_independent_rlr_subprocess_contract() -> dict[str, Any]:
    """Return the fixed process/runtime boundary for registration rerenders."""
    return copy.deepcopy(APPROVED_INDEPENDENT_RLR_SUBPROCESS_CONTRACT)


__all__ = [
    "ACOUSTIC_SCENE_CONTRACT_SCHEMA",
    "APPROVED_APARTMENT_ACOUSTIC_SCENE_CONTRACT",
    "APPROVED_INDEPENDENT_RLR_SUBPROCESS_CONTRACT",
    "APPROVED_RLR_RUNTIME_ARTIFACTS",
    "INDEPENDENT_RLR_SAMPLE_TIMEOUT_S",
    "RLR_FIXED_ACOUSTICS_CONFIG",
    "RLR_QUALITY_INDIRECT_RAY_COUNTS",
    "RLR_RENDERER_BUILD_ID",
    "RLR_RENDERER_CONTRACT_SCHEMA",
    "approved_acoustic_scene_contract",
    "approved_independent_rlr_subprocess_contract",
    "approved_rlr_renderer_contract",
    "validate_approved_acoustic_scene_inputs",
    "validate_approved_rlr_runtime_artifact",
]
