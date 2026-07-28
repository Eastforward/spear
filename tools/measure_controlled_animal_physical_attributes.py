#!/usr/bin/env python3
"""Measure controlled quadruped dimensions from rigged mesh and UE evidence.

The normal entry point is standard-library only.  It authenticates any
bone-name-independent retarget/weight-repair semantics before launching
Blender workers to measure the weighted foreleg/shoulder surface in rig space,
then calibrates those ratios with the bounds that UE actually observed at the
scene actor scale.  Prompt targets are retained only for comparison and are
never copied into the measured result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
SCHEMA_V2 = "controlled_animal_walk_idle_apartment_specs_v2"
SCHEMAS = frozenset({SCHEMA, SCHEMA_V2})
MEASUREMENT_SCHEMA = "controlled_animal_physical_measurement_v1"
BATCH_SCHEMA = "controlled_animal_physical_measurement_batch_v1"
METHOD = "ue_bounds_calibrated_weighted_foreleg_surface_v2"
PHYSICAL_MEASUREMENT = "shoulder_height_cm"
RIG_SEMANTIC_EVIDENCE_SCHEMA = "controlled_animal_rig_semantic_evidence_v1"
RETARGET_SCHEMA = "avengine_generated_quadruped_retarget_v5"
WEIGHT_REPAIR_SCHEMA = "avengine_motion_aware_quadruped_weight_repair_v2"
LEGACY_FRONT_UPPER_GROUPS = frozenset({"Bone.014", "Bone.017"})
REPAIR_AUTHORITY_FIELDS = (
    "native_mesh_geometry_preserved",
    "native_mesh_topology_preserved",
    "pbr_material_preserved",
    "fitted_skeleton_rest_matrices_preserved",
    "approved_animation_curves_preserved",
    "only_vertex_weights_modified_in_memory",
)
TARGET_PHYSICAL_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "control_attribute",
        "selected_value",
        "measurement",
        "mode",
        "reference_value_cm",
        "reference_provenance",
        "scale_ratio",
        "tolerance_cm",
        "target_value_cm",
    }
)


class MeasurementError(RuntimeError):
    """Raised when physical evidence is missing or internally inconsistent."""


def _has_symlink_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(os.fspath(path)))
    trusted_workspace_tmp = (SPEAR_ROOT / "tmp").absolute()
    return any(
        component.is_symlink() and component != trusted_workspace_tmp
        for component in (absolute, *absolute.parents)
    )


def _direct_file(path: Path, label: str) -> Path:
    raw = Path(path)
    if _has_symlink_component(raw):
        raise MeasurementError(
            f"{label} cannot be a symlink or contain a symlink path component: {raw}"
        )
    resolved = raw.resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise MeasurementError(f"{label} is not a direct non-empty file: {raw}")
    return resolved


def resolve_front_upper_groups(
    group_names: Iterable[str],
    *,
    authenticated_semantic_groups: Iterable[str] | None = None,
) -> frozenset[str]:
    """Resolve the two front upper-limb groups without asset-specific numbers.

    TokenRig-style ``bone_#`` names have no semantics of their own.  They are
    accepted only when the standard-library parent process has authenticated a
    retarget or weight-repair artifact and passes the two resolved groups
    explicitly.  The Quaternius pair and descriptive Rocketbox names remain as
    legacy conventions for already published controlled-animal manifests.
    """

    names = frozenset(str(name) for name in group_names)
    if authenticated_semantic_groups is not None:
        selected = frozenset(str(name) for name in authenticated_semantic_groups)
        if (
            len(selected) != 2
            or any(not name for name in selected)
            or not selected.issubset(names)
        ):
            raise MeasurementError(
                "authenticated front upper-limb groups are missing or ambiguous: "
                f"selected={sorted(selected)} available={sorted(names)}"
            )
        return selected
    if LEGACY_FRONT_UPPER_GROUPS.issubset(names):
        return LEGACY_FRONT_UPPER_GROUPS
    normalized = {
        re.sub(r"[^a-z0-9]+", " ", name.lower()).strip(): name for name in names
    }
    left = [
        original
        for label, original in normalized.items()
        if label.endswith(" l upperarm") or label.endswith(" left upperarm")
    ]
    right = [
        original
        for label, original in normalized.items()
        if label.endswith(" r upperarm") or label.endswith(" right upperarm")
    ]
    if len(left) == 1 and len(right) == 1:
        return frozenset({left[0], right[0]})
    raise MeasurementError(
        "missing or ambiguous foreleg upper-arm vertex groups: "
        f"left={sorted(left)} right={sorted(right)}"
    )


def _reject_nonfinite_numbers(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise MeasurementError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for name, item in value.items():
            _reject_nonfinite_numbers(item, f"{label}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, f"{label}[{index}]")


def _read_json(path: Path) -> dict[str, Any]:
    path = _direct_file(path, "JSON input")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MeasurementError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise MeasurementError(f"JSON root must be an object: {path}")
    _reject_nonfinite_numbers(value, f"JSON input {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    path = _direct_file(path, "artifact")
    published = Path(published_path) if published_path is not None else path
    if _has_symlink_component(published):
        raise MeasurementError(
            f"artifact publication path cannot contain a symlink: {published}"
        )
    return {
        "path": str(published.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _descriptor_binds_file(descriptor: Any, path: Path, sha256: str) -> bool:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        return False
    try:
        described = Path(str(descriptor["path"]))
        size = descriptor["size_bytes"]
        described_sha256 = descriptor["sha256"]
        expected_raw = Path(path)
        if (
            _has_symlink_component(described)
            or _has_symlink_component(expected_raw)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(described_sha256, str)
            or len(described_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in described_sha256
            )
            or described_sha256 != sha256
        ):
            return False
        resolved = described.resolve()
        path = expected_raw.resolve()
        return bool(
            resolved == path
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == size
            and _sha256(path) == sha256
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _semantic_front_chains(
    payload: Mapping[str, Any], *, expected_schema: str
) -> Mapping[str, Any]:
    if payload.get("schema") != expected_schema:
        raise MeasurementError("rig semantic evidence schema changed")
    if expected_schema == WEIGHT_REPAIR_SCHEMA:
        authority = payload.get("authority_contract")
        if not isinstance(authority, Mapping) or any(
            authority.get(field) is not True for field in REPAIR_AUTHORITY_FIELDS
        ):
            raise MeasurementError("weight-repair semantic authority did not pass")
        chains = payload.get("semantic_rig", {}).get("chains")
    elif expected_schema == RETARGET_SCHEMA:
        inference = payload.get("semantic_inference")
        if (
            not isinstance(inference, Mapping)
            or inference.get("bone_name_independent_target") is not True
            or inference.get("complete_target_bone_coverage") is not True
        ):
            raise MeasurementError(
                "retarget semantic inference is not bone-name independent"
            )
        chains = inference.get("chains")
    else:
        raise MeasurementError(
            f"unsupported rig semantic evidence schema: {expected_schema}"
        )
    if not isinstance(chains, Mapping):
        raise MeasurementError("rig semantic evidence has no chains")
    return chains


def authenticate_record_front_upper_groups(
    record: Mapping[str, Any],
) -> tuple[frozenset[str], dict[str, Any]] | None:
    """Resolve front upper bones from evidence bound to ``record.source_glb``."""

    wrapper = record.get("rig_semantic_evidence")
    if wrapper is None:
        return None
    if not isinstance(wrapper, Mapping) or set(wrapper) != {
        "schema",
        "kind",
        "artifact",
        "semantic_schema",
        "source_glb_sha256",
    }:
        raise MeasurementError("rig_semantic_evidence fields changed")
    source = record.get("source_glb")
    if not isinstance(source, Mapping):
        raise MeasurementError("record source_glb is missing")
    try:
        source_raw = Path(str(source["path"]))
        if _has_symlink_component(source_raw):
            raise MeasurementError("record source_glb cannot be a symlink")
        source_path = source_raw.resolve()
        source_sha256 = str(source["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise MeasurementError("record source_glb is invalid") from error
    if (
        wrapper.get("schema") != RIG_SEMANTIC_EVIDENCE_SCHEMA
        or wrapper.get("source_glb_sha256") != source_sha256
    ):
        raise MeasurementError("rig semantic evidence/source GLB binding changed")

    expected_schema = wrapper.get("semantic_schema")
    expected_kind = {
        RETARGET_SCHEMA: "bone_name_independent_retarget",
        WEIGHT_REPAIR_SCHEMA: "motion_aware_weight_repair",
    }.get(expected_schema)
    if expected_kind is None or wrapper.get("kind") != expected_kind:
        raise MeasurementError("rig semantic evidence kind changed")
    artifact = wrapper.get("artifact")
    if not isinstance(artifact, Mapping):
        raise MeasurementError("rig semantic evidence artifact is missing")
    try:
        evidence_raw = Path(str(artifact["path"]))
        if _has_symlink_component(evidence_raw):
            raise MeasurementError("rig semantic evidence artifact cannot be a symlink")
        evidence_path = evidence_raw.resolve()
        evidence_sha256 = str(artifact["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise MeasurementError("rig semantic evidence artifact is invalid") from error
    if not _descriptor_binds_file(artifact, evidence_path, evidence_sha256):
        raise MeasurementError("rig semantic evidence artifact changed")
    payload = _read_json(evidence_raw)
    binding_field = "output" if expected_schema == WEIGHT_REPAIR_SCHEMA else "export"
    if not _descriptor_binds_file(
        payload.get(binding_field), source_path, source_sha256
    ):
        raise MeasurementError(
            "rig semantic evidence does not bind the measured source GLB"
        )
    chains = _semantic_front_chains(payload, expected_schema=expected_schema)
    groups = []
    for label in ("front_side_negative", "front_side_positive"):
        chain = chains.get(label)
        if (
            not isinstance(chain, list)
            or len(chain) < 2
            or any(not isinstance(name, str) or not name for name in chain)
            or len(set(chain)) != len(chain)
        ):
            raise MeasurementError(f"invalid rig semantic chain: {label}")
        groups.append(chain[0])
    if len(set(groups)) != 2:
        raise MeasurementError("rig semantic front upper-limb groups are not distinct")
    return frozenset(groups), copy_mapping(wrapper)


def copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-compatible defensive copy without importing repo code."""

    return json.loads(json.dumps(dict(value)))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise MeasurementError(
            f"output is not finite canonical JSON: {error}"
        ) from error
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _finite_number(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MeasurementError(f"{label} must be a finite number")
    number = float(value)
    if positive and number <= 0.0:
        raise MeasurementError(f"{label} must be positive")
    if nonnegative and number < 0.0:
        raise MeasurementError(f"{label} must be nonnegative")
    return number


def _finite_vector(
    value: Any, label: str, *, positive: bool = False
) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise MeasurementError(f"{label} must contain exactly three numbers")
    return tuple(
        _finite_number(component, f"{label}[{index}]", positive=positive)
        for index, component in enumerate(value)
    )


def _integer(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeasurementError(f"{label} must be an integer")
    if positive and value <= 0:
        raise MeasurementError(f"{label} must be positive")
    if nonnegative and value < 0:
        raise MeasurementError(f"{label} must be nonnegative")
    return value


def _validate_target_physical_profile(
    value: Any,
    *,
    sampled_attributes: Mapping[str, str],
    asset_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != TARGET_PHYSICAL_PROFILE_FIELDS:
        raise MeasurementError(
            f"controlled animal physical identity changed: {asset_id}"
        )
    target = dict(value)
    control = target.get("control_attribute")
    provenance = target.get("reference_provenance")
    if (
        not isinstance(target.get("profile_id"), str)
        or not target["profile_id"]
        or not isinstance(control, str)
        or not control
        or control not in sampled_attributes
        or not isinstance(target.get("selected_value"), str)
        or target["selected_value"] != sampled_attributes[control]
        or target.get("measurement") != PHYSICAL_MEASUREMENT
        or target.get("mode")
        not in {"relative_to_profile_reference", "absolute_measurement"}
        or not isinstance(provenance, Mapping)
        or set(provenance) != {"status", "source_id", "artifact", "notes"}
        or provenance.get("status") not in {"provisional", "verified"}
        or not isinstance(provenance.get("source_id"), str)
        or not provenance["source_id"]
        or not isinstance(provenance.get("notes"), str)
        or not provenance["notes"]
    ):
        raise MeasurementError(
            f"controlled animal physical identity changed: {asset_id}"
        )

    reference_value_cm = _finite_number(
        target.get("reference_value_cm"),
        f"target reference value for {asset_id}",
        positive=True,
    )
    scale_ratio = _finite_number(
        target.get("scale_ratio"),
        f"target scale ratio for {asset_id}",
        positive=True,
    )
    _finite_number(
        target.get("tolerance_cm"),
        f"target tolerance for {asset_id}",
        positive=True,
    )
    target_value_cm = _finite_number(
        target.get("target_value_cm"),
        f"target value for {asset_id}",
        positive=True,
    )
    if not math.isclose(
        target_value_cm,
        reference_value_cm * scale_ratio,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise MeasurementError(
            f"controlled animal physical target formula changed: {asset_id}"
        )

    artifact = provenance.get("artifact")
    if artifact is not None:
        if not isinstance(artifact, Mapping):
            raise MeasurementError(
                f"controlled animal physical reference artifact changed: {asset_id}"
            )
        artifact_raw = Path(str(artifact.get("path", "")))
        artifact_path = _direct_file(
            artifact_raw, f"physical reference artifact for {asset_id}"
        )
        artifact_sha256 = artifact.get("sha256")
        if not isinstance(artifact_sha256, str) or not _descriptor_binds_file(
            artifact, artifact_path, artifact_sha256
        ):
            raise MeasurementError(
                f"controlled animal physical reference artifact changed: {asset_id}"
            )
    if provenance["status"] == "verified" and artifact is None:
        raise MeasurementError(
            f"verified physical target requires a reference artifact: {asset_id}"
        )
    return copy_mapping(target)


def _validate_spec_numeric_contract(
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    asset_id: str,
    action_name: str,
) -> None:
    label = f"{asset_id}/{action_name}"
    actor_scale = _finite_number(
        source.get("actor_scale"), f"{label} actor_scale", positive=True
    )
    if not 0.01 <= actor_scale <= 2.0:
        raise MeasurementError(f"{label} actor_scale is outside the safe range")

    numeric_fields = {
        "actor_z_lift_cm": {},
        "walking_forward_yaw_offset_deg": {},
        "animation_play_rate": {"positive": True},
        "ground_snap_max_abs_correction_cm": {"positive": True},
        "audio_source_height_offset_m": {"nonnegative": True},
        "audio_source_duration_s": {"positive": True},
    }
    for name, options in numeric_fields.items():
        if name in source:
            _finite_number(source[name], f"{label} {name}", **options)
    integer_fields = {
        "audio_source_size_bytes": {"positive": True},
        "audio_source_channels": {"positive": True},
        "audio_source_sample_width_bytes": {"positive": True},
        "audio_source_sample_rate_hz": {"positive": True},
        "audio_source_frame_count": {"positive": True},
    }
    for name, options in integer_fields.items():
        if name in source:
            _integer(source[name], f"{label} {name}", **options)
    if "ground_snap_to_floor" in source and not isinstance(
        source["ground_snap_to_floor"], bool
    ):
        raise MeasurementError(f"{label} ground_snap_to_floor must be boolean")

    render = spec.get("render_config")
    audio = spec.get("audio_config")
    camera_contract = spec.get("camera_pass_table_loop_contract")
    trajectory = source.get("trajectory_m")
    if not isinstance(render, Mapping):
        raise MeasurementError(f"{label} render_config must be an object")
    if not isinstance(audio, Mapping):
        raise MeasurementError(f"{label} audio_config must be an object")
    if not isinstance(camera_contract, Mapping):
        raise MeasurementError(
            f"{label} camera_pass_table_loop_contract must be an object"
        )
    if not isinstance(trajectory, list) or not trajectory:
        raise MeasurementError(f"{label} trajectory_m must be non-empty")

    _finite_number(
        render.get("duration_s"), f"{label} render duration_s", positive=True
    )
    _integer(render.get("fps"), f"{label} render fps", positive=True)
    frame_count = _integer(
        render.get("n_frames"), f"{label} render n_frames", positive=True
    )
    for name in ("width", "height", "resolution_x", "resolution_y"):
        if name in render:
            _integer(render[name], f"{label} render {name}", positive=True)
    for name in ("streaming_warmup_frames", "camera_warmup_frames"):
        if name in render:
            _integer(render[name], f"{label} render {name}", nonnegative=True)
    _finite_number(audio.get("duration_s"), f"{label} audio duration_s", positive=True)
    _integer(
        audio.get("sample_rate_hz"),
        f"{label} audio sample_rate_hz",
        positive=True,
    )

    trajectory_vectors = [
        _finite_vector(point, f"{label} trajectory_m[{index}]")
        for index, point in enumerate(trajectory)
    ]
    if len(trajectory_vectors) != frame_count:
        raise MeasurementError(f"{label} trajectory length/n_frames changed")
    start = _finite_vector(source.get("start_pos_m"), f"{label} start_pos_m")
    end = _finite_vector(source.get("end_pos_m"), f"{label} end_pos_m")
    if start != trajectory_vectors[0] or end != trajectory_vectors[-1]:
        raise MeasurementError(f"{label} trajectory endpoint binding changed")

    left_front_frame = _integer(
        camera_contract.get("left_front_nearest_frame"),
        f"{label} left-front frame",
        nonnegative=True,
    )
    if left_front_frame >= frame_count:
        raise MeasurementError(f"{label} left-front frame is out of range")

    windows = spec.get("rig_direction_check_windows")
    if action_name == "Walking":
        if not isinstance(windows, list) or not windows:
            raise MeasurementError(f"{label} rig direction windows are missing")
        for index, window in enumerate(windows):
            if not isinstance(window, Mapping):
                raise MeasurementError(
                    f"{label} rig direction window {index} must be an object"
                )
            if "frame_a" in window or "frame_b" in window:
                pair = ("frame_a", "frame_b")
            elif "start_frame" in window or "end_frame" in window:
                pair = ("start_frame", "end_frame")
            else:
                raise MeasurementError(
                    f"{label} rig direction window {index} has no frame bounds"
                )
            bounds = [
                _integer(
                    window.get(name),
                    f"{label} rig direction window {index}.{name}",
                    nonnegative=True,
                )
                for name in pair
            ]
            if bounds[0] > bounds[1] or bounds[1] >= frame_count:
                raise MeasurementError(
                    f"{label} rig direction window {index} is out of range"
                )
    elif windows is not None:
        raise MeasurementError(f"{label} stationary spec retained direction windows")

    stationary = spec.get("stationary_idle_contract")
    if action_name == "Idle":
        if not isinstance(stationary, Mapping):
            raise MeasurementError(f"{label} stationary contract is missing")
        if (
            _integer(
                stationary.get("frame_count"),
                f"{label} stationary frame_count",
                positive=True,
            )
            != frame_count
        ):
            raise MeasurementError(f"{label} stationary frame count changed")
        _finite_vector(stationary.get("position_m"), f"{label} stationary position")
        _finite_number(
            stationary.get("maximum_position_delta_m"),
            f"{label} stationary maximum_position_delta_m",
            nonnegative=True,
        )
        waypoint = _integer(
            stationary.get("source_waypoint_frame"),
            f"{label} stationary source_waypoint_frame",
            nonnegative=True,
        )
        if waypoint >= frame_count:
            raise MeasurementError(f"{label} stationary waypoint is out of range")


def _canonical_hash_without(value: Mapping[str, Any], key: str, *, label: str) -> str:
    payload = {name: item for name, item in value.items() if name != key}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MeasurementError(f"{label} is not canonical JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def quantile(values: Iterable[float], fraction: float) -> float:
    numbers = sorted(
        _finite_number(value, f"quantile value[{index}]")
        for index, value in enumerate(values)
    )
    fraction = _finite_number(fraction, "quantile fraction", nonnegative=True)
    if not numbers or fraction > 1.0:
        raise MeasurementError("quantile requires values and a fraction in [0, 1]")
    position = (len(numbers) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return numbers[lower]
    weight = position - lower
    return numbers[lower] * (1.0 - weight) + numbers[upper] * weight


def build_runtime_measurement(
    *,
    record: Mapping[str, Any],
    walking_spec: Mapping[str, Any],
    visual: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine independent mesh ratios with observed UE centimeter bounds."""

    tag = str(record["tag"])
    sources = walking_spec.get("sources")
    visual_sources = visual.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise MeasurementError(f"Walking spec must contain one source: {tag}")
    source = sources[0]
    if not isinstance(source, Mapping):
        raise MeasurementError(f"Walking source must be an object: {tag}")
    if not isinstance(visual_sources, list) or any(
        not isinstance(item, Mapping) for item in visual_sources
    ):
        raise MeasurementError(f"visual sources must be an object list: {tag}")
    matching = [item for item in visual_sources if item.get("tag") == tag]
    if len(matching) != 1:
        raise MeasurementError(f"visual metadata must contain source {tag}")
    frames = matching[0].get("runtime_frames")
    if not isinstance(frames, list) or not frames:
        raise MeasurementError(f"visual metadata has no runtime frames: {tag}")
    automatic_checks = visual.get("automatic_checks")
    if (
        not isinstance(automatic_checks, Mapping)
        or automatic_checks.get("overall") != "passed"
    ):
        raise MeasurementError(f"visual automatic checks did not pass: {tag}")

    actor_scale = _finite_number(
        source.get("actor_scale"), f"actor scale: {tag}", positive=True
    )
    bounds_heights: list[float] = []
    observed_scales: list[float] = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise MeasurementError(
                f"UE frame evidence must be an object: {tag}/{frame_index}"
            )
        bounds = frame.get("bounds_ue")
        root_transform = frame.get("root_transform_ue")
        floor = frame.get("floor_contact")
        if (
            not isinstance(bounds, Mapping)
            or not isinstance(root_transform, Mapping)
            or not isinstance(floor, Mapping)
            or floor.get("within_penetration_tolerance") is not True
        ):
            raise MeasurementError(f"incomplete UE frame evidence: {tag}")
        minimum = bounds.get("minimum_cm")
        maximum = bounds.get("maximum_cm")
        root_scale = root_transform.get("scale")
        minimum_values = _finite_vector(
            minimum, f"UE bounds minimum: {tag}/{frame_index}"
        )
        maximum_values = _finite_vector(
            maximum, f"UE bounds maximum: {tag}/{frame_index}"
        )
        scale_values = _finite_vector(
            root_scale,
            f"UE root scale: {tag}/{frame_index}",
            positive=True,
        )
        height = maximum_values[2] - minimum_values[2]
        if height <= 0.0:
            raise MeasurementError(f"invalid UE bounds height: {tag}")
        bounds_heights.append(height)
        observed_scales.extend(scale_values)
    if max(abs(value - actor_scale) for value in observed_scales) > 1.0e-5:
        raise MeasurementError(f"UE actor scale changed across frames: {tag}")

    bounds_units = _finite_number(
        geometry.get("bounds_height_units"),
        f"Blender bounds height: {tag}",
        positive=True,
    )
    shoulder_units = _finite_number(
        geometry.get("shoulder_height_units"),
        f"Blender shoulder height: {tag}",
        positive=True,
    )
    length_units = _finite_number(
        geometry.get("nose_to_tail_length_units"),
        f"Blender nose-to-tail length: {tag}",
        positive=True,
    )
    if shoulder_units >= bounds_units:
        raise MeasurementError(f"invalid Blender geometry measurement: {tag}")

    bounds_height_cm = statistics.median(bounds_heights)
    centimeters_per_unit = bounds_height_cm / bounds_units
    shoulder_height_cm = shoulder_units * centimeters_per_unit
    nose_to_tail_length_cm = length_units * centimeters_per_unit
    target = record.get("target_physical_profile", {})
    if (
        not isinstance(target, Mapping)
        or target.get("measurement") != PHYSICAL_MEASUREMENT
    ):
        raise MeasurementError(f"unsupported physical measurement target: {tag}")
    control = target.get("control_attribute")
    sampled = record.get("sampled_attributes")
    if (
        not isinstance(control, str)
        or not control
        or not isinstance(sampled, Mapping)
        or control not in sampled
        or (
            target.get("selected_value") is not None
            and target.get("selected_value") != sampled[control]
        )
    ):
        raise MeasurementError(f"physical target identity changed: {tag}")
    target_cm = _finite_number(
        target.get("target_value_cm"), f"physical target: {tag}", positive=True
    )
    tolerance_cm = _finite_number(
        target.get("tolerance_cm"),
        f"physical target tolerance: {tag}",
        positive=True,
    )
    residual_cm = shoulder_height_cm - target_cm
    comparison = (
        "within_tolerance" if abs(residual_cm) <= tolerance_cm else "outside_tolerance"
    )
    runtime_measurements = {
        "actor_scale": round(actor_scale, 6),
        PHYSICAL_MEASUREMENT: round(shoulder_height_cm, 6),
        "bounds_height_cm": round(bounds_height_cm, 6),
        "nose_to_tail_length_cm": round(nose_to_tail_length_cm, 6),
    }
    if "audio_source_height_offset_m" in record:
        runtime_measurements["audio_source_height_offset_m"] = round(
            _finite_number(
                record.get("audio_source_height_offset_m"),
                f"measured emitter-derived audio height: {tag}",
                positive=True,
            ),
            9,
        )
    return {
        "schema": MEASUREMENT_SCHEMA,
        "asset_id": record["base_avatar_id"],
        "tag": tag,
        "profile_schema_id": record["profile_schema_id"],
        "sampled_size": sampled[control],
        "method": METHOD,
        "physical_measurements": {
            "status": "measured",
            "method": METHOD,
            "runtime": runtime_measurements,
        },
        "target_comparison": {
            "target_value_cm": target_cm,
            "tolerance_cm": tolerance_cm,
            "measured_minus_target_cm": round(residual_cm, 6),
            "status": comparison,
            "target_reference_status": target.get("reference_provenance", {}).get(
                "status"
            ),
        },
        "calibration": {
            "ue_frame_count": len(frames),
            "ue_bounds_height_min_cm": round(min(bounds_heights), 6),
            "ue_bounds_height_median_cm": round(bounds_height_cm, 6),
            "ue_bounds_height_max_cm": round(max(bounds_heights), 6),
            "centimeters_per_blender_unit": round(centimeters_per_unit, 9),
            "geometry": dict(geometry),
        },
    }


def summarize_size_ordering(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        raise MeasurementError("size ordering requires at least one measurement")
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        height = _finite_number(
            record.get("physical_measurements", {})
            .get("runtime", {})
            .get(PHYSICAL_MEASUREMENT),
            f"measured {PHYSICAL_MEASUREMENT}",
            positive=True,
        )
        grouped.setdefault(str(record["profile_schema_id"]), {}).setdefault(
            str(record["sampled_size"]), []
        ).append(height)
    summaries = []
    for profile_id, values_by_size in sorted(grouped.items()):
        unknown_sizes = set(values_by_size) - {"small", "medium", "large"}
        if not profile_id or unknown_sizes:
            raise MeasurementError(
                f"unsupported size ordering identity: {profile_id}/{sorted(unknown_sizes)}"
            )
        ordered = [
            size for size in ("small", "medium", "large") if size in values_by_size
        ]
        if not ordered:
            raise MeasurementError(f"profile has no ordered size values: {profile_id}")
        medians = {
            size: round(statistics.median(values_by_size[size]), 6) for size in ordered
        }
        ordering_applicable = len(ordered) >= 2
        strictly_increasing = (
            all(
                medians[left] < medians[right]
                for left, right in zip(ordered, ordered[1:])
            )
            if ordering_applicable
            else None
        )
        summaries.append(
            {
                "profile_schema_id": profile_id,
                "ordered_sizes": ordered,
                "median_shoulder_height_cm": medians,
                "ordering_applicable": ordering_applicable,
                "strictly_increasing": strictly_increasing,
                "status": (
                    "passed"
                    if ordering_applicable and strictly_increasing is True
                    else (
                        "rejected"
                        if ordering_applicable
                        else "not_applicable_single_size"
                    )
                ),
            }
        )
    return summaries


def build_admission_checks(
    measured: Sequence[Mapping[str, Any]],
    ordering: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not measured or not ordering:
        raise MeasurementError(
            "measurement admission checks require non-empty evidence"
        )
    comparison_statuses = [
        item.get("target_comparison", {}).get("status") for item in measured
    ]
    if any(
        status not in {"within_tolerance", "outside_tolerance"}
        for status in comparison_statuses
    ):
        raise MeasurementError("measurement target comparison status is invalid")
    within_count = comparison_statuses.count("within_tolerance")
    outside_count = comparison_statuses.count("outside_tolerance")
    applicable_ordering = []
    for item in ordering:
        applicable = item.get("ordering_applicable")
        increasing = item.get("strictly_increasing")
        status = item.get("status")
        if (
            applicable is True
            and isinstance(increasing, bool)
            and status == ("passed" if increasing else "rejected")
        ):
            applicable_ordering.append(item)
        elif (
            applicable is False
            and increasing is None
            and status == "not_applicable_single_size"
        ):
            continue
        else:
            raise MeasurementError("measurement size ordering status is invalid")
    applicable_ordering_passed = (
        not any(item["status"] == "rejected" for item in applicable_ordering)
        if applicable_ordering
        else None
    )
    ordering_admission_satisfied = not any(
        item.get("status") == "rejected" for item in ordering
    )
    ordering_status = (
        "not_applicable_single_size_only"
        if not applicable_ordering
        else ("passed" if applicable_ordering_passed else "rejected")
    )
    targets_passed = within_count == len(measured)
    registration_ready = ordering_admission_satisfied and targets_passed
    return {
        "all_measurements_derived_from_mesh_and_ue_evidence": True,
        "all_applicable_profile_size_medians_strictly_increasing": (
            applicable_ordering_passed
        ),
        "size_ordering_applicable_profile_count": len(applicable_ordering),
        "size_ordering_not_applicable_profile_count": (
            len(ordering) - len(applicable_ordering)
        ),
        "size_ordering_status": ordering_status,
        "size_ordering_admission_satisfied": ordering_admission_satisfied,
        "target_tolerance_is_admission_constraint": True,
        "all_targets_within_tolerance": targets_passed,
        "within_provisional_target_tolerance_count": within_count,
        "outside_provisional_target_tolerance_count": outside_count,
        "downstream_source_asset_registration_ready": registration_ready,
        "overall": "passed" if registration_ready else "rejected",
    }


def _blender_geometry_measurement(
    input_glb: Path,
    output: Path,
    *,
    authenticated_front_upper_groups: Iterable[str] | None = None,
) -> None:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    if output.exists() or output.is_symlink():
        raise MeasurementError(f"worker output already exists: {output}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not meshes:
        raise MeasurementError("rigged GLB contains no mesh")
    mesh = max(meshes, key=lambda obj: len(obj.data.vertices))
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    front_upper_groups = resolve_front_upper_groups(
        group_names.values(),
        authenticated_semantic_groups=authenticated_front_upper_groups,
    )

    all_x: list[float] = []
    all_z: list[float] = []
    shoulder_z: list[float] = []
    for vertex_index, vertex in enumerate(mesh.data.vertices):
        point = mesh.matrix_world @ Vector(vertex.co)
        point_x = _finite_number(point.x, f"mesh vertex {vertex_index} x")
        point_z = _finite_number(point.z, f"mesh vertex {vertex_index} z")
        all_x.append(point_x)
        all_z.append(point_z)
        weight = sum(
            _finite_number(
                membership.weight,
                f"mesh vertex {vertex_index} front-upper weight",
                nonnegative=True,
            )
            for membership in vertex.groups
            if group_names.get(membership.group) in front_upper_groups
        )
        if weight >= 0.05:
            shoulder_z.append(point_z)
    if len(shoulder_z) < 100:
        raise MeasurementError(f"too few weighted shoulder vertices: {len(shoulder_z)}")
    floor_z = quantile(all_z, 0.001)
    top_z = quantile(all_z, 0.999)
    shoulder_surface_z = quantile(shoulder_z, 0.95)
    min_x = quantile(all_x, 0.005)
    max_x = quantile(all_x, 0.995)
    bounds_height_units = top_z - floor_z
    shoulder_height_units = shoulder_surface_z - floor_z
    nose_to_tail_length_units = max_x - min_x
    for value, label in (
        (bounds_height_units, "mesh bounds height"),
        (shoulder_height_units, "mesh shoulder height"),
        (nose_to_tail_length_units, "mesh nose-to-tail length"),
    ):
        _finite_number(value, label, positive=True)
    shoulder_fraction = shoulder_height_units / bounds_height_units
    geometry = {
        "schema": "weighted_quadruped_geometry_measurement_v1",
        "input_glb": _artifact(input_glb),
        "mesh_name": mesh.name,
        "vertex_count": len(mesh.data.vertices),
        "front_upper_groups": sorted(front_upper_groups),
        "front_upper_group_authority": (
            "hash_bound_rig_semantic_evidence_v1"
            if authenticated_front_upper_groups is not None
            else "legacy_vertex_group_semantics_v1"
        ),
        "selected_shoulder_vertex_count": len(shoulder_z),
        "quantiles": {
            "floor": 0.001,
            "top": 0.999,
            "shoulder_surface": 0.95,
            "length_min": 0.005,
            "length_max": 0.995,
        },
        "bounds_height_units": bounds_height_units,
        "shoulder_height_units": shoulder_height_units,
        "nose_to_tail_length_units": nose_to_tail_length_units,
        "shoulder_fraction_of_bounds_height": shoulder_fraction,
    }
    if not (
        0.2 < geometry["shoulder_fraction_of_bounds_height"] < 0.98
        and geometry["nose_to_tail_length_units"] > 0.0
    ):
        raise MeasurementError("implausible weighted shoulder geometry")
    _atomic_json(output, geometry)


def _validate_record_identity(record: Any, *, manifest_path: Path) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise MeasurementError(f"manifest record must be an object: {manifest_path}")
    asset_id = record.get("base_avatar_id")
    sampled = record.get("sampled_attributes")
    if (
        not isinstance(asset_id, str)
        or not asset_id
        or record.get("asset_id") != asset_id
        or not isinstance(record.get("tag"), str)
        or not record["tag"]
        or not isinstance(record.get("profile_schema_id"), str)
        or not record["profile_schema_id"]
        or not isinstance(sampled, Mapping)
        or not sampled
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            for name, value in sampled.items()
        )
    ):
        raise MeasurementError(f"controlled animal record identity changed: {asset_id}")

    target = _validate_target_physical_profile(
        record.get("target_physical_profile"),
        sampled_attributes=sampled,
        asset_id=asset_id,
    )

    source = record.get("source_glb")
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise MeasurementError(f"controlled animal source identity changed: {asset_id}")
    source_raw = Path(str(source.get("path", "")))
    source_path = _direct_file(source_raw, f"source GLB for {asset_id}")
    source_sha256 = source.get("sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
        or _sha256(source_path) != source_sha256
    ):
        raise MeasurementError(f"controlled animal source hash changed: {asset_id}")

    actions = record.get("actions")
    if not isinstance(actions, Mapping) or set(actions) != {"Walking", "Idle"}:
        raise MeasurementError(f"controlled animal action identity changed: {asset_id}")
    for action_name, expected_motion, expected_kind in (
        ("Walking", "walking", "moving"),
        ("Idle", "idle", "stationary"),
    ):
        action = actions[action_name]
        if (
            not isinstance(action, Mapping)
            or action.get("motion") != expected_motion
            or not isinstance(action.get("clip_id"), str)
            or not action["clip_id"]
            or not isinstance(action.get("output_dir"), str)
            or not action["output_dir"]
            or not isinstance(action.get("spec"), str)
            or not action["spec"]
        ):
            raise MeasurementError(
                f"controlled animal action record changed: {asset_id}/{action_name}"
            )
        spec_raw = Path(action["spec"])
        spec_path = _direct_file(spec_raw, f"{action_name} spec for {asset_id}")
        spec_evidence = action.get("spec_evidence")
        if not isinstance(spec_evidence, Mapping) or not _descriptor_binds_file(
            spec_evidence, spec_path, _sha256(spec_path)
        ):
            raise MeasurementError(
                f"controlled animal spec evidence changed: {asset_id}/{action_name}"
            )
        spec = _read_json(spec_raw)
        sources = spec.get("sources")
        if (
            not isinstance(sources, list)
            or len(sources) != 1
            or not isinstance(sources[0], Mapping)
        ):
            raise MeasurementError(
                f"controlled animal spec source changed: {asset_id}/{action_name}"
            )
        spec_source = sources[0]
        if (
            spec_source.get("asset_id") != asset_id
            or spec_source.get("tag") != record["tag"]
            or spec_source.get("profile_schema_id") != record["profile_schema_id"]
            or spec_source.get("sampled_attributes") != record["sampled_attributes"]
            or spec_source.get("wanted_anim") != action_name
            or spec_source.get("kind") != expected_kind
            or spec_source.get("target_physical_profile") != target
        ):
            raise MeasurementError(
                f"controlled animal spec identity changed: {asset_id}/{action_name}"
            )
        _validate_spec_numeric_contract(
            spec,
            spec_source,
            asset_id=asset_id,
            action_name=action_name,
        )
    return copy_mapping(record)


def _load_records(manifest_paths: Sequence[Path]) -> list[dict[str, Any]]:
    if not manifest_paths:
        raise MeasurementError("at least one controlled animal manifest is required")
    records: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    seen_tags: set[str] = set()
    seen_clips: set[str] = set()
    for raw_path in manifest_paths:
        path = _direct_file(raw_path, "controlled animal manifest")
        payload = _read_json(raw_path)
        if payload.get("schema") == SCHEMA_V2:
            try:
                from tools import (
                    build_user_approved_generated_animal_apartment_specs
                    as apartment_builder,
                )

                authority = apartment_builder.authenticate_apartment_v2_manifest(path)
            except Exception as error:
                raise MeasurementError(
                    f"invalid authenticated Apartment v2 authority: {path}: {error}"
                ) from error
            items = [authority["record"]]
        else:
            authority = None
            items = payload.get("records")
        if (
            payload.get("schema") not in SCHEMAS
            or not isinstance(items, list)
            or not items
            or isinstance(payload.get("avatar_count"), bool)
            or not isinstance(payload.get("avatar_count"), int)
            or payload.get("avatar_count") != len(items)
            or isinstance(payload.get("clip_count"), bool)
            or not isinstance(payload.get("clip_count"), int)
            or payload.get("clip_count") != len(items) * 2
            or payload.get("manifest_sha256")
            != _canonical_hash_without(
                payload, "manifest_sha256", label="controlled animal manifest"
            )
        ):
            raise MeasurementError(f"invalid controlled animal manifest: {path}")
        for raw_record in items:
            record = _validate_record_identity(raw_record, manifest_path=path)
            if authority is not None:
                record["_authenticated_apartment_v2"] = {
                    "manifest": _artifact(path),
                    "inputs": copy_mapping(authority["inputs"]),
                    "runtime_lineage": {
                        "reviewed_animated_glb": copy_mapping(
                            raw_record["runtime_lineage"][
                                "reviewed_animated_glb"
                            ]
                        ),
                        "ue_import_glb": copy_mapping(
                            raw_record["runtime_lineage"]["ue_import_glb"]
                        ),
                        "texture_transcode_manifest": (
                            copy_mapping(
                                raw_record["runtime_lineage"][
                                    "texture_transcode_manifest"
                                ]
                            )
                            if raw_record["runtime_lineage"][
                                "texture_transcode_manifest"
                            ]
                            is not None
                            else None
                        ),
                    },
                }
            asset_id = record.get("base_avatar_id")
            tag = record.get("tag")
            clip_ids = {action["clip_id"] for action in record["actions"].values()}
            if (
                not asset_id
                or asset_id in seen_assets
                or tag in seen_tags
                or len(clip_ids) != 2
                or not seen_clips.isdisjoint(clip_ids)
            ):
                raise MeasurementError(f"duplicate/empty controlled asset: {asset_id}")
            seen_assets.add(asset_id)
            seen_tags.add(str(tag))
            seen_clips.update(clip_ids)
            records.append(record)
    return sorted(records, key=lambda item: item["base_avatar_id"])


def build_measurements(
    *,
    manifest_paths: Sequence[Path],
    output_root: Path,
    blender: Path,
    workers: int,
) -> Path:
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise MeasurementError(f"output root already exists: {output_root}")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 8
    ):
        raise MeasurementError("workers must be in [1, 8]")
    records = _load_records(manifest_paths)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        geometry_paths = {
            record["base_avatar_id"]: staging
            / "geometry"
            / f"{record['base_avatar_id']}.json"
            for record in records
        }
        semantic_selections: dict[
            str, tuple[frozenset[str], dict[str, Any]] | None
        ] = {}
        for record in records:
            source_raw = Path(record["source_glb"]["path"])
            source = source_raw.resolve()
            if (
                _has_symlink_component(source_raw)
                or not source.is_file()
                or source.is_symlink()
                or _sha256(source) != record["source_glb"]["sha256"]
            ):
                raise MeasurementError(
                    f"source GLB changed: {record['base_avatar_id']}"
                )
            semantic_selections[str(record["base_avatar_id"])] = (
                authenticate_record_front_upper_groups(record)
            )

        def run_worker(
            record: Mapping[str, Any],
        ) -> tuple[str, subprocess.CompletedProcess[str]]:
            asset_id = str(record["base_avatar_id"])
            output = geometry_paths[asset_id]
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(blender),
                "-b",
                "--python",
                str(Path(__file__).resolve()),
                "--",
                "--blender-worker",
                "--input-glb",
                str(Path(record["source_glb"]["path"]).resolve()),
                "--output",
                str(output),
            ]
            semantic_selection = semantic_selections[asset_id]
            if semantic_selection is not None:
                for group in sorted(semantic_selection[0]):
                    command.extend(["--front-upper-group", group])
            return asset_id, subprocess.run(
                command,
                cwd=SPEAR_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

        failures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_worker, record): record for record in records
            }
            for future in as_completed(futures):
                asset_id, process = future.result()
                log = staging / "logs" / f"{asset_id}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(process.stdout, encoding="utf-8")
                if process.returncode != 0 or not geometry_paths[asset_id].is_file():
                    failures.append(
                        {
                            "asset_id": asset_id,
                            "returncode": process.returncode,
                            "log": str(log),
                        }
                    )
        if failures:
            raise MeasurementError(f"Blender physical measurement failed: {failures}")

        measured = []
        for record in records:
            asset_id = record["base_avatar_id"]
            action = record["actions"]["Walking"]
            walking_spec_path = Path(action["spec"])
            walking_spec = _read_json(walking_spec_path)
            visual_path = (
                Path(action["output_dir"]) / "videos" / "actor_visual_metadata.json"
            )
            visual = _read_json(visual_path)
            geometry = _read_json(geometry_paths[asset_id])
            result = build_runtime_measurement(
                record=record,
                walking_spec=walking_spec,
                visual=visual,
                geometry=geometry,
            )
            result["evidence"] = {
                "walking_spec": _artifact(walking_spec_path),
                "ue_visual_metadata": _artifact(visual_path),
                "geometry_measurement": _artifact(
                    geometry_paths[asset_id],
                    published_path=output_root / "geometry" / f"{asset_id}.json",
                ),
            }
            semantic_selection = semantic_selections[str(asset_id)]
            if semantic_selection is not None:
                result["evidence"]["rig_semantic_evidence"] = copy_mapping(
                    semantic_selection[1]["artifact"]
                )
            if "emitter_measurement" in record:
                result["evidence"]["emitter_measurement"] = copy_mapping(
                    record["emitter_measurement"]
                )
            destination = staging / "measurements" / f"{asset_id}.json"
            _atomic_json(destination, result)
            measured.append(result)
        ordering = summarize_size_ordering(measured)
        automatic_checks = build_admission_checks(measured, ordering)
        manifest: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": METHOD,
            "asset_count": len(measured),
            "inputs": [_artifact(Path(path)) for path in manifest_paths],
            "measurements": [
                {
                    "asset_id": item["asset_id"],
                    "profile_schema_id": item["profile_schema_id"],
                    "sampled_size": item["sampled_size"],
                    "physical_measurements": item["physical_measurements"],
                    "target_comparison": item["target_comparison"],
                    "record": _artifact(
                        staging / "measurements" / f"{item['asset_id']}.json",
                        published_path=output_root
                        / "measurements"
                        / f"{item['asset_id']}.json",
                    ),
                }
                for item in measured
            ],
            "size_ordering": ordering,
            "admission_policy": {
                "target_tolerance": "required_for_downstream_source_asset_registration",
                "single_size_ordering": "not_applicable_not_vacuously_passed",
                "rejected_batch_usage": "measurement_and_scale_recalibration_evidence_only",
            },
            "automatic_checks": automatic_checks,
        }
        manifest["batch_sha256"] = _canonical_hash_without(
            manifest, "batch_sha256", label="physical measurement batch"
        )
        _atomic_json(staging / "measurement_batch_manifest.json", manifest)
        os.rename(staging, output_root)
        return output_root / "measurement_batch_manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--blender", type=Path, default=Path("/data/jzy/.local/bin/blender")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blender-worker", action="store_true")
    parser.add_argument("--input-glb", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--front-upper-group", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    args = _parse_args(argv)
    try:
        if args.blender_worker:
            if args.input_glb is None or args.output is None:
                raise MeasurementError(
                    "Blender worker requires --input-glb and --output"
                )
            if len(args.front_upper_group) not in {0, 2}:
                raise MeasurementError(
                    "Blender worker requires zero or two --front-upper-group values"
                )
            _blender_geometry_measurement(
                _direct_file(args.input_glb, "Blender worker input GLB"),
                args.output.resolve(),
                authenticated_front_upper_groups=(args.front_upper_group or None),
            )
            print(
                f"CONTROLLED_ANIMAL_GEOMETRY_MEASUREMENT_OK output={args.output.resolve()}"
            )
            return 0
        if not args.manifest or args.output_root is None:
            raise MeasurementError("batch mode requires --manifest and --output-root")
        manifest = build_measurements(
            manifest_paths=args.manifest,
            output_root=args.output_root,
            blender=args.blender,
            workers=args.workers,
        )
        payload = _read_json(manifest)
    except (
        MeasurementError,
        OSError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        print(f"CONTROLLED_ANIMAL_PHYSICAL_MEASUREMENT_FAILED {error}", flush=True)
        return 2
    overall = payload["automatic_checks"]["overall"]
    prefix = (
        "CONTROLLED_ANIMAL_PHYSICAL_MEASUREMENT_OK"
        if overall == "passed"
        else "CONTROLLED_ANIMAL_PHYSICAL_MEASUREMENT_REJECTED"
    )
    print(
        f"{prefix} assets={payload['asset_count']} overall={overall} output={manifest}",
        flush=True,
    )
    return 0 if overall == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
