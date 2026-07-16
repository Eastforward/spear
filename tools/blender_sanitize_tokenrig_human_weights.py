#!/usr/bin/env python3
"""Deterministically sanitize fitted TokenRig weights without new inference.

The pure functions in this module are importable without Blender.  The Blender
publication path is intentionally added behind the same contracts so the
weight algorithm can be tested independently of scene I/O.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import blender_tokenrig_human_static_audit as static_audit


ALGORITHM_VERSION = "tokenrig_side_transfer_seam_hybrid_export_floor_v3"
BLENDER_EXPORT_MIN_INFLUENCE = 0.0001
BLENDER_EXPORT_SAFE_FLOOR = 0.00010000000474974513
EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX = 1.0e-8
SCHEMA = "pixal_tokenrig_sanitized_weights_v1"
ASSET_ID = "rocketbox_male_adult_01"
RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[1]
FITTED_DIR = (
    REPO_ROOT
    / "tmp/pixal_tokenrig_route2_v1"
    / ASSET_ID
    / "fitted_skeleton_v1"
)
FITTED_GLB = FITTED_DIR / "tokenrig_transfer.glb"
FITTED_MANIFEST = FITTED_DIR / "tokenrig_manifest.json"
OUTPUT_DIR = FITTED_DIR / "sanitized_weights_v1"
ORIGINAL_SOURCE_GLB = (
    REPO_ROOT
    / "tmp/i23d_human_bakeoff_v1/pixal3d"
    / ASSET_ID
    / "canary_1024_seed42.glb"
)
ORIGINAL_SOURCE_MANIFEST = ORIGINAL_SOURCE_GLB.with_suffix(".manifest.json")
DIRECT_DIR = REPO_ROOT / "tmp/pixal_tokenrig_route2_v1" / ASSET_ID
DIRECT_FAILED_DIR = DIRECT_DIR.with_name(f"{ASSET_ID}.tokenrig_failed_attempt")


@dataclass(frozen=True)
class PinnedFile:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SanitationContract:
    asset_id: str
    original_source_glb: PinnedFile
    original_source_manifest: PinnedFile
    direct_glb: PinnedFile
    recovery_manifest: PinnedFile
    direct_failures: tuple[PinnedFile, PinnedFile]
    fitted_glb: PinnedFile
    fitted_manifest: PinnedFile
    fitted_failures: tuple[PinnedFile, PinnedFile]
    sanitation_failures: tuple[PinnedFile, ...]
    output_dir: Path


PINNED_CONTRACT = SanitationContract(
    asset_id=ASSET_ID,
    original_source_glb=PinnedFile(
        ORIGINAL_SOURCE_GLB,
        "1df2490d6b83e52fa3b7c4e9d6b69207fa59cad0deae80e3dc3f894dfc443c42",
        36_430_844,
    ),
    original_source_manifest=PinnedFile(
        ORIGINAL_SOURCE_MANIFEST,
        "f0658fbcf84d3505d5ea08fcf3011c9070de46933da50c7304d44875a3b038e3",
        1_135,
    ),
    direct_glb=PinnedFile(
        DIRECT_FAILED_DIR / "tokenrig_transfer.glb",
        "8606c013fba02f722e1d5c65accddc4398eab1fa925467a9233aaf458d93f01c",
        50_843_552,
    ),
    recovery_manifest=PinnedFile(
        DIRECT_DIR / "tokenrig_manifest.json",
        "cae4aac8f6472b893ce695173ad9a1766ef8f2ecf86cbbc7a80440b2ad949e96",
        5_071,
    ),
    direct_failures=(
        PinnedFile(
            DIRECT_DIR
            / "static_audit_v1.failed.554c2e6ca92a409497c80b39673b60b7.json",
            "0eab61c2dfcb5a7fe0a05ee8f5109c60a9c756a6ae048a0545df6d9e64c590ce",
            1_127,
        ),
        PinnedFile(
            DIRECT_DIR
            / "static_audit_v1.failed.f868faf6c0ea4870aef5e114c403eb9c.json",
            "c78d5f1e4d7c127781f8b24b5a9bed906f96b5034d3b5bda9509c23c8f492e26",
            1_154,
        ),
    ),
    fitted_glb=PinnedFile(
        FITTED_GLB,
        "eb9566f091b6de5357375dee750e66a48bcf4b12ba97a87615c26bed4cf77017",
        50_972_456,
    ),
    fitted_manifest=PinnedFile(
        FITTED_MANIFEST,
        "f2be8c719ea5049b76efc77220af5ae686e72c50913acbe85b7555276a506e56",
        25_211,
    ),
    fitted_failures=(
        PinnedFile(
            FITTED_DIR
            / "static_audit_v1.failed.24312333e985427699108a7e4d42b7b1.json",
            "39a5b61542c9355cb8f584637a692d801fc4ba6e2e5c33b757b39f1351b7d9ee",
            1_520,
        ),
        PinnedFile(
            FITTED_DIR
            / "static_audit_v1.failed.8641b5fa58af4abeaf016b4276148ecc.json",
            "1b3a11c0708ffe2b70f2c363d7617ca4437deab0503cbe165cac9e7c1d0366e4",
            1_541,
        ),
    ),
    sanitation_failures=(
        PinnedFile(
            FITTED_DIR
            / "sanitized_weights_v1.failed.dc0c6c25a7fa49aab49fc28dda0c66f4.json",
            "dc4aafc914d8ebbd521dcf0c14320d1f2fdf93666be1787e0b73c2a24c6ae4e4",
            395,
        ),
        PinnedFile(
            FITTED_DIR
            / "sanitized_weights_v1.failed.373ea709d8bb477c9335e1229975c183.json",
            "841a350f1b9178f76b09f72a65f9873e46b3745326d9a2a15eb84d6e4549021c",
            422,
        ),
        PinnedFile(
            FITTED_DIR
            / "sanitized_weights_v1.failed.457d0c9a47c54effbcacf15fa5c86308.json",
            "a1796a7e9b159bbe66bc3218a02327322253b0e2a9bf8f02e8d720103210228e",
            422,
        ),
    ),
    output_dir=OUTPUT_DIR,
)


class SanitationError(RuntimeError):
    """Raised when deterministic sanitation cannot satisfy its closed contract."""


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": static_audit.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _require_pin(pin: PinnedFile, description: str) -> Path:
    path = Path(os.path.abspath(os.fspath(pin.path)))
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise SanitationError(f"{description} must be a direct regular file")
    if (
        path.stat().st_size != pin.size_bytes
        or static_audit.sha256_file(path) != pin.sha256
    ):
        raise SanitationError(f"{description} hash/size changed")
    return path


def capture_code_snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, value in sorted(paths.items()):
        path = Path(os.path.abspath(os.fspath(value)))
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise SanitationError(f"code snapshot path is invalid: {name}")
        snapshot[name] = _file_record(path)
    if not snapshot:
        raise SanitationError("code snapshot is empty")
    return snapshot


def verify_code_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    if not snapshot:
        raise SanitationError("code snapshot is empty")
    for name, record in snapshot.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise SanitationError(f"code snapshot descriptor is invalid: {name}")
        path = Path(record["path"])
        if path.is_symlink() or not path.is_file():
            raise SanitationError(f"authenticated code changed during sanitation: {name}")
        current = _file_record(path)
        if any(
            current.get(field) != record.get(field)
            for field in ("path", "sha256", "size_bytes")
        ):
            raise SanitationError(f"authenticated code changed during sanitation: {name}")


def _read_json(path: Path, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SanitationError(f"invalid {description}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SanitationError(f"{description} root must be an object")
    return payload


def validate_fitted_failure_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(payloads) != 2:
        raise SanitationError("exactly two fitted static failure records are required")
    messages = []
    for payload in payloads:
        failure = payload.get("failure")
        if (
            payload.get("decision") != "rejected"
            or payload.get("readiness_bundle_published") is not False
            or not isinstance(failure, Mapping)
        ):
            raise SanitationError("fitted static evidence is not a preserved rejection")
        messages.append(str(failure.get("message", "")))
    if "surface unique position count changed" not in messages[0]:
        raise SanitationError("historical exact-tuple import rejection is missing")
    if "UV seam duplicate vertex" not in messages[1]:
        raise SanitationError("ordered fitted seam rejection is missing")
    return {
        "obsolete_exact_tuple_import_gate": "rejected",
        "ordered_fitted_skin_gate": "rejected_at_seam",
        "animation_authorized": False,
    }


def _descendants(root: str, parents: Mapping[str, str | None]) -> set[str]:
    descendants: set[str] = set()
    for name in parents:
        current = parents.get(name)
        visited = {name}
        while current is not None:
            if current in visited:
                raise SanitationError("bone parent map contains a cycle")
            if current == root:
                descendants.add(name)
                break
            visited.add(current)
            if current not in parents:
                raise SanitationError(f"bone parent map is missing {current!r}")
            current = parents[current]
    return descendants


def build_bilateral_transfer_maps(
    chains: Mapping[str, Sequence[str]],
    parents: Mapping[str, str | None],
) -> dict[str, Any]:
    """Pair the final four arm/leg bones and fold opposite fingers to the hand."""

    required = ("left_arm", "right_arm", "left_leg", "right_leg")
    if any(name not in chains for name in required):
        raise SanitationError("semantic chains are incomplete")
    cores: dict[str, tuple[str, ...]] = {}
    for name in required:
        values = tuple(str(value) for value in chains[name])
        if len(values) < 4:
            raise SanitationError(f"{name} must contain four core bones")
        cores[name] = values[-4:]
    left_core = cores["left_arm"] + cores["left_leg"]
    right_core = cores["right_arm"] + cores["right_leg"]
    if len(set(left_core)) != 8 or len(set(right_core)) != 8 or set(left_core) & set(right_core):
        raise SanitationError("left/right core chains must be disjoint")
    missing = sorted((set(left_core) | set(right_core)) - set(parents))
    if missing:
        raise SanitationError(f"core bones are missing from parent map: {missing}")

    core_pairs = list(zip(cores["left_arm"], cores["right_arm"])) + list(
        zip(cores["left_leg"], cores["right_leg"])
    )
    to_left = {right: left for left, right in core_pairs}
    to_right = {left: right for left, right in core_pairs}
    left_hand = cores["left_arm"][-1]
    right_hand = cores["right_arm"][-1]
    left_fingers = _descendants(left_hand, parents) - set(left_core)
    right_fingers = _descendants(right_hand, parents) - set(right_core)
    if left_fingers & right_fingers:
        raise SanitationError("left/right hand descendants must be disjoint")
    for name in sorted(right_fingers):
        to_left[name] = left_hand
    for name in sorted(left_fingers):
        to_right[name] = right_hand
    return {
        "to_left": to_left,
        "to_right": to_right,
        "core_pairs": [[left, right] for left, right in core_pairs],
        "left_finger_descendants": sorted(left_fingers),
        "right_finger_descendants": sorted(right_fingers),
    }


def _normalized_top_four(
    values: Mapping[str, float],
) -> tuple[dict[str, float], float]:
    finite = {
        str(name): float(value)
        for name, value in values.items()
        if math.isfinite(float(value)) and float(value) > 0.0
    }
    if not finite:
        raise SanitationError("sanitation produced a zero-weight vertex")
    ranked = sorted(finite.items(), key=lambda item: (-item[1], item[0]))
    kept = ranked[:4]
    truncated = sum(value for _, value in ranked[4:])
    total = sum(value for _, value in kept)
    if not math.isfinite(total) or total <= 0.0:
        raise SanitationError("sanitation produced a non-finite weight sum")
    normalized = {name: value / total for name, value in kept}
    return dict(sorted(normalized.items())), truncated


def project_export_safe_weights(
    weights: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    values = dict(sorted((str(name), float(value)) for name, value in weights.items()))
    if (
        not 1 <= len(values) <= 4
        or any(not math.isfinite(value) or value <= 0.0 for value in values.values())
        or abs(sum(values.values()) - 1.0) > static_audit.WEIGHT_SUM_TOLERANCE
    ):
        raise SanitationError("export-floor input weights are invalid")
    affected = [
        name
        for name, value in values.items()
        if value <= BLENDER_EXPORT_MIN_INFLUENCE
    ]
    added_mass = sum(
        BLENDER_EXPORT_SAFE_FLOOR - values[name] for name in affected
    )
    if added_mass > EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX:
        raise SanitationError(
            "export-floor support projection exceeds the micro-mass budget: "
            f"added={added_mass} budget={EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX}"
        )
    donor = None
    if affected:
        donor = min(values, key=lambda name: (-values[name], name))
        if donor in affected or values[donor] - added_mass <= 0.0:
            raise SanitationError("export-floor projection has no safe donor influence")
        for name in affected:
            values[name] = BLENDER_EXPORT_SAFE_FLOOR
        values[donor] -= added_mass
    if any(value <= BLENDER_EXPORT_MIN_INFLUENCE for value in values.values()):
        raise SanitationError("export-floor projection left a droppable influence")
    return values, {
        "component_count": len(affected),
        "added_mass": added_mass,
        "l1": 2.0 * added_mass,
        "donor_bone": donor,
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower]) * (1.0 - fraction) + float(
        sorted_values[upper]
    ) * fraction


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "maximum": ordered[-1] if ordered else 0.0,
    }


def sanitize_weight_maps(
    *,
    positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    chains: Mapping[str, Sequence[str]],
    parents: Mapping[str, str | None],
) -> tuple[tuple[dict[str, float], ...], dict[str, Any], tuple[dict[str, Any], ...]]:
    """Transfer opposite-side weights, then deterministically reconcile seams."""

    if not positions or len(positions) != len(vertex_weights):
        raise SanitationError("positions and weights are empty or mismatched")
    bone_names = set(parents)
    try:
        static_audit.validate_vertex_weights(vertex_weights, bone_names=bone_names)
    except static_audit.StaticAuditError as exc:
        raise SanitationError(f"input skin is invalid: {exc}") from exc
    transfer = build_bilateral_transfer_maps(chains, parents)
    xs = [float(position[0]) for position in positions]
    if any(not math.isfinite(value) for value in xs):
        raise SanitationError("positions contain a non-finite X coordinate")
    center = (min(xs) + max(xs)) * 0.5
    half_width = (max(xs) - min(xs)) * 0.5
    if half_width <= 1.0e-9:
        raise SanitationError("positions have degenerate bilateral width")
    distal_cutoff = 0.25 * half_width

    transferred: list[dict[str, float]] = []
    vertex_transferred_mass = [0.0] * len(vertex_weights)
    pair_mass: dict[str, float] = {}
    for index, (position, source_weights) in enumerate(zip(positions, vertex_weights)):
        side_x = float(position[0]) - center
        mapping: Mapping[str, str]
        if side_x >= distal_cutoff:
            mapping = transfer["to_left"]
        elif side_x <= -distal_cutoff:
            mapping = transfer["to_right"]
        else:
            transferred.append(dict(sorted((str(k), float(v)) for k, v in source_weights.items())))
            continue
        result: dict[str, float] = {}
        moved = 0.0
        for source_name, value in sorted(source_weights.items()):
            amount = float(value)
            target_name = mapping.get(source_name, source_name)
            result[target_name] = result.get(target_name, 0.0) + amount
            if target_name != source_name:
                moved += amount
                key = f"{source_name}->{target_name}"
                pair_mass[key] = pair_mass.get(key, 0.0) + amount
        if moved > 0.0:
            result, unexpected_truncation = _normalized_top_four(result)
            if unexpected_truncation > 1.0e-15:
                raise SanitationError("side transfer unexpectedly increased influence count")
        else:
            result = dict(sorted(result.items()))
        transferred.append(result)
        vertex_transferred_mass[index] = moved

    try:
        flattened_positions = tuple(
            float(value) for position in positions for value in position
        )
        position_mapping, _, _ = static_audit._match_unique_positions(
            flattened_positions, flattened_positions
        )
    except (ValueError, static_audit.StaticAuditError) as exc:
        raise SanitationError(f"could not cluster seam positions: {exc}") from exc
    groups: dict[int, list[int]] = {}
    for index, representative in enumerate(position_mapping):
        groups.setdefault(int(representative), []).append(index)
    sanitized = [dict(values) for values in transferred]
    total_proposed_truncated_mass = 0.0
    maximum_proposed_truncated_mass = 0.0
    total_applied_truncated_mass = 0.0
    maximum_applied_truncated_mass = 0.0
    seam_group_count = 0
    method_counts = {"weighted_average": 0, "l1_medoid": 0}
    seam_group_records = []
    for representative, indices in groups.items():
        if len(indices) <= 1:
            continue
        group_index = seam_group_count
        seam_group_count += 1
        average: dict[str, float] = {}
        for index in indices:
            for name, value in transferred[index].items():
                average[name] = average.get(name, 0.0) + float(value) / len(indices)
        averaged, proposed_truncated = _normalized_top_four(average)
        total_proposed_truncated_mass += proposed_truncated
        maximum_proposed_truncated_mass = max(
            maximum_proposed_truncated_mass, proposed_truncated
        )
        medoid_vertex_index: int | None = None
        average_export_floor_mass = sum(
            BLENDER_EXPORT_SAFE_FLOOR - value
            for value in average.values()
            if value <= BLENDER_EXPORT_MIN_INFLUENCE
        )
        if (
            len(average) <= 4
            and average_export_floor_mass
            <= EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX
        ):
            method = "weighted_average"
            method_reason = "influence_union_within_four"
            reconciled = averaged
            applied_truncated = proposed_truncated
        else:
            method = "l1_medoid"
            method_reason = (
                "influence_union_exceeds_four"
                if len(average) > 4
                else "export_floor_guard"
            )
            candidates = []
            for index in indices:
                candidate, candidate_truncated = _normalized_top_four(
                    transferred[index]
                )
                if candidate_truncated > 1.0e-15:
                    raise SanitationError(
                        "seam medoid candidate unexpectedly exceeds four influences"
                    )
                canonical = json.dumps(
                    candidate, sort_keys=True, separators=(",", ":")
                )
                score = sum(
                    static_audit._weight_l1(candidate, transferred[other])
                    for other in indices
                )
                candidates.append((score, canonical, index, candidate))
            _, _, medoid_vertex_index, reconciled = min(
                candidates, key=lambda item: (item[0], item[1], item[2])
            )
            applied_truncated = 0.0
        method_counts[method] += 1
        total_applied_truncated_mass += applied_truncated
        maximum_applied_truncated_mass = max(
            maximum_applied_truncated_mass, applied_truncated
        )
        member_errors = [
            static_audit._weight_l1(reconciled, transferred[index])
            for index in indices
        ]
        seam_group_records.append(
            {
                "group_index": group_index,
                "representative_vertex_index": representative,
                "vertex_count": len(indices),
                "method": method,
                "method_reason": method_reason,
                "union_influence_count": len(average),
                "medoid_vertex_index": medoid_vertex_index,
                "proposed_average_truncated_mass": proposed_truncated,
                "applied_truncated_mass": applied_truncated,
                "maximum_member_l1_to_reconciled": max(member_errors),
                "total_member_l1_to_reconciled": sum(member_errors),
            }
        )
        for index in indices:
            sanitized[index] = dict(reconciled)

    vertex_export_floor_mass = [0.0] * len(sanitized)
    vertex_export_floor_components = [0] * len(sanitized)
    export_floor_component_count = 0
    export_floor_vertex_count = 0
    export_floor_l1 = []
    for index, weights in enumerate(sanitized):
        projected, projection = project_export_safe_weights(weights)
        sanitized[index] = projected
        added = float(projection["added_mass"])
        vertex_export_floor_mass[index] = added
        vertex_export_floor_components[index] = int(projection["component_count"])
        export_floor_component_count += int(projection["component_count"])
        export_floor_vertex_count += int(projection["component_count"] > 0)
        export_floor_l1.append(float(projection["l1"]))

    sanitized_values = tuple(sanitized)
    try:
        weight_validation = static_audit.validate_vertex_weights(
            sanitized_values, bone_names=bone_names
        )
        seam_validation = static_audit.validate_seam_weights(
            positions, sanitized_values
        )
        bilateral_validation = static_audit.validate_bilateral_contamination(
            positions, sanitized_values, chains
        )
    except static_audit.StaticAuditError as exc:
        raise SanitationError(f"sanitized skin failed its validator: {exc}") from exc

    changes = []
    l1_values = []
    for index, (before, after) in enumerate(zip(vertex_weights, sanitized_values)):
        l1 = static_audit._weight_l1(before, after)
        l1_values.append(l1)
        if l1 > 1.0e-15:
            changes.append(
                {
                    "vertex_index": index,
                    "before": dict(sorted((str(k), float(v)) for k, v in before.items())),
                    "after": dict(sorted(after.items())),
                    "l1_before_after": l1,
                    "transferred_mass": vertex_transferred_mass[index],
                    "export_floor_added_mass": vertex_export_floor_mass[index],
                    "export_floor_component_count": vertex_export_floor_components[
                        index
                    ],
                }
            )
    changed_l1 = [record["l1_before_after"] for record in changes]
    report = {
        "algorithm_version": ALGORITHM_VERSION,
        "inference_used": False,
        "vertex_count": len(sanitized_values),
        "changed_vertex_count": len(changes),
        "changed_vertex_ratio": len(changes) / len(sanitized_values),
        "per_vertex_l1_accounting": {
            "vertex_count": len(sanitized_values),
            "explicit_changed_record_count": len(changes),
            "implicit_unchanged_vertex_count": len(sanitized_values) - len(changes),
            "implicit_unchanged_l1_before_after": 0.0,
        },
        "considered_distal_vertex_count": bilateral_validation[
            "considered_distal_vertex_count"
        ],
        "seam_duplicate_group_count": seam_group_count,
        "seam_reconciliation_method_counts": method_counts,
        "seam_group_records": seam_group_records,
        "total_transferred_mass": sum(vertex_transferred_mass),
        "export_floor_projection": {
            "policy": "raise_droppable_support_to_next_float32_and_debit_largest_v1",
            "blender_min_influence": BLENDER_EXPORT_MIN_INFLUENCE,
            "safe_floor": BLENDER_EXPORT_SAFE_FLOOR,
            "maximum_added_mass_per_vertex_budget": (
                EXPORT_FLOOR_MAX_ADDED_MASS_PER_VERTEX
            ),
            "projected_vertex_count": export_floor_vertex_count,
            "projected_component_count": export_floor_component_count,
            "total_added_mass": sum(vertex_export_floor_mass),
            "maximum_added_mass": max(vertex_export_floor_mass, default=0.0),
            "l1_all_vertices": _distribution(export_floor_l1),
            "minimum_output_weight": min(
                value for weights in sanitized_values for value in weights.values()
            ),
        },
        "total_proposed_average_truncated_mass": total_proposed_truncated_mass,
        "maximum_proposed_average_truncated_mass": maximum_proposed_truncated_mass,
        "total_truncated_mass": total_applied_truncated_mass,
        "maximum_truncated_mass": maximum_applied_truncated_mass,
        "transferred_mass_by_bone_pair": dict(sorted(pair_mass.items())),
        "transfer_map": transfer,
        "l1_all_vertices": _distribution(l1_values),
        "l1_changed_vertices": _distribution(changed_l1),
        "weight_validation": weight_validation,
        "seam_validation": seam_validation,
        "bilateral_validation": bilateral_validation,
    }
    return sanitized_values, report, tuple(changes)


def serialize_change_records(records: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted(records, key=lambda record: int(record["vertex_index"]))
    return b"".join(
        (json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for record in ordered
    )


def serialize_seam_group_records(records: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted(records, key=lambda record: int(record["group_index"]))
    return b"".join(
        (json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for record in ordered
    )


def authenticate_inputs() -> dict[str, Any]:
    contract = PINNED_CONTRACT
    execution_code_snapshot = capture_code_snapshot(
        {
            "sanitizer": RUNNER_PATH,
            "static_audit": Path(static_audit.__file__).resolve(),
        }
    )
    original = _require_pin(contract.original_source_glb, "original Pixal GLB")
    original_manifest = _require_pin(
        contract.original_source_manifest, "original Pixal manifest"
    )
    direct = _require_pin(contract.direct_glb, "preserved direct TokenRig GLB")
    recovery = _require_pin(contract.recovery_manifest, "direct recovery manifest")
    fitted = _require_pin(contract.fitted_glb, "fitted TokenRig GLB")
    fitted_manifest = _require_pin(
        contract.fitted_manifest, "fitted TokenRig manifest"
    )
    direct_failures = []
    for index, pin in enumerate(contract.direct_failures):
        path = _require_pin(pin, f"direct static failure {index}")
        if path.stat().st_mode & 0o222:
            raise SanitationError("direct static failure evidence is mutable")
        direct_failures.append(_read_json(path, f"direct static failure {index}"))
    direct_messages = [
        str(payload.get("failure", {}).get("message", ""))
        for payload in direct_failures
    ]
    if (
        "raw GLB triangle count changed" not in direct_messages[0]
        or "opposite-limb contamination" not in direct_messages[1]
    ):
        raise SanitationError("direct rejection sequence changed")
    fitted_failure_paths = []
    fitted_failure_payloads = []
    for index, pin in enumerate(contract.fitted_failures):
        path = _require_pin(pin, f"fitted static failure {index}")
        if path.stat().st_mode & 0o222:
            raise SanitationError("fitted static failure evidence is mutable")
        fitted_failure_paths.append(path)
        fitted_failure_payloads.append(
            _read_json(path, f"fitted static failure {index}")
        )
    fitted_failure_summary = validate_fitted_failure_payloads(
        fitted_failure_payloads
    )
    sanitation_failure_paths = []
    expected_sanitation_failure_messages = (
        "surface unique position coverage changed",
        "roundtrip skin position coverage changed",
        "roundtrip skin weights changed",
    )
    for index, pin in enumerate(contract.sanitation_failures):
        path = _require_pin(pin, f"prior sanitation failure {index}")
        if path.stat().st_mode & 0o222:
            raise SanitationError("prior sanitation failure evidence is mutable")
        payload = _read_json(path, f"prior sanitation failure {index}")
        if (
            payload.get("decision") != "rejected"
            or payload.get("readiness_bundle_published") is not False
            or expected_sanitation_failure_messages[index]
            not in str(payload.get("failure", {}).get("message", ""))
        ):
            raise SanitationError("prior sanitation failure sequence changed")
        sanitation_failure_paths.append(path)
    try:
        fitted_authentication = static_audit.authenticate_task3_inputs(
            asset_id=contract.asset_id,
            source_glb=original,
            tokenrig_glb=fitted,
            tokenrig_manifest=fitted_manifest,
        )
    except static_audit.StaticAuditError as exc:
        raise SanitationError(f"fitted provenance authentication failed: {exc}") from exc
    if (
        fitted_authentication.get("attempt_ledger_sha256")
        != "6125db2e71674c714b2e00360b49685be6c94f017bf16e909dc79b2ba1eda9d1"
        or fitted_authentication.get("orchestrator_runner_sha256")
        != "69c265a58c923e81bfc105e42b01a8aeb191f9f5278b8eeb24eb1fd40c0076ae"
        or fitted_authentication.get("delegated_runner_sha256")
        != static_audit.FITTED_BASE_RUNNER_SHA256
        or fitted_authentication.get("server_hygiene_load_event_count") != 6
    ):
        raise SanitationError("fitted execution/code provenance pins changed")
    fitted_payload = _read_json(fitted_manifest, "fitted TokenRig manifest")
    orchestrator = fitted_payload.get("orchestrator")
    runner_record = orchestrator.get("runner") if isinstance(orchestrator, Mapping) else None
    if not isinstance(runner_record, Mapping) or not isinstance(
        runner_record.get("path"), str
    ):
        raise SanitationError("fitted wrapper descriptor is missing")
    wrapper = Path(runner_record["path"]).resolve()
    delegated = wrapper.with_name("tokenrig_human_canary.py")

    original_parsed = static_audit.read_glb(original)
    fitted_parsed = static_audit.read_glb(fitted)
    pbr = static_audit.compare_pbr_payloads(
        static_audit.pbr_payload_contract(original_parsed),
        static_audit.pbr_payload_contract(fitted_parsed),
    )
    raw_surface = static_audit.analyze_raw_serialization_equivalence(
        original_parsed, fitted_parsed
    )
    return {
        "original_source_glb": _file_record(original),
        "original_source_manifest": _file_record(original_manifest),
        "direct_glb": _file_record(direct),
        "recovery_manifest": _file_record(recovery),
        "direct_failures": [
            _file_record(pin.path) for pin in contract.direct_failures
        ],
        "fitted_glb": _file_record(fitted),
        "fitted_manifest": _file_record(fitted_manifest),
        "fitted_failures": [_file_record(path) for path in fitted_failure_paths],
        "prior_sanitation_failures": [
            _file_record(path) for path in sanitation_failure_paths
        ],
        "fitted_failure_summary": fitted_failure_summary,
        "fitted_authentication": fitted_authentication,
        "fitted_wrapper": _file_record(wrapper),
        "delegated_base_runner": _file_record(delegated),
        "execution_code_snapshot": execution_code_snapshot,
        "input_pbr": pbr,
        "input_raw_surface": raw_surface,
    }


def _measure_seam_contamination(
    positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, position in enumerate(positions):
        groups.setdefault(static_audit._position_key(position), []).append(index)
    duplicate_groups = [values for values in groups.values() if len(values) > 1]
    maximum = 0.0
    violations = 0
    for indices in duplicate_groups:
        first = vertex_weights[indices[0]]
        group_failed = False
        for index in indices[1:]:
            error = static_audit._weight_l1(first, vertex_weights[index])
            maximum = max(maximum, error)
            group_failed = group_failed or error > static_audit.SEAM_WEIGHT_L1_TOLERANCE
        violations += int(group_failed)
    return {
        "duplicate_position_group_count": len(duplicate_groups),
        "violating_group_count": violations,
        "maximum_weight_l1_error": maximum,
        "tolerance": static_audit.SEAM_WEIGHT_L1_TOLERANCE,
    }


def _measure_bilateral_contamination(
    positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    chains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    xs = [float(position[0]) for position in positions]
    center = (min(xs) + max(xs)) * 0.5
    half_width = (max(xs) - min(xs)) * 0.5
    cutoff = 0.25 * half_width
    left = set(chains["left_arm"]) | set(chains["left_leg"])
    right = set(chains["right_arm"]) | set(chains["right_leg"])
    considered = 0
    contaminated = 0
    maximum = 0.0
    for position, weights in zip(positions, vertex_weights):
        side_x = float(position[0]) - center
        if abs(side_x) < cutoff:
            continue
        considered += 1
        opposite = right if side_x > 0.0 else left
        value = sum(float(weights.get(name, 0.0)) for name in opposite)
        maximum = max(maximum, value)
        contaminated += int(value > static_audit.OPPOSITE_LIMB_WEIGHT_TOLERANCE)
    return {
        "considered_distal_vertex_count": considered,
        "contaminated_vertex_count": contaminated,
        "maximum_opposite_limb_weight": maximum,
        "tolerance": static_audit.OPPOSITE_LIMB_WEIGHT_TOLERANCE,
    }


def _expected_rejection(function: Any, required_message: str) -> str:
    try:
        function()
    except static_audit.StaticAuditError as exc:
        message = str(exc)
        if required_message not in message:
            raise SanitationError(
                f"upstream rejection changed: expected {required_message!r}, got {message!r}"
            ) from exc
        return message
    raise SanitationError(f"upstream fitted skin no longer fails {required_message!r}")


def _apply_changed_weights(
    mesh: Any,
    weights: Sequence[Mapping[str, float]],
    changed_indices: Sequence[int],
) -> None:
    groups = {group.name: group for group in mesh.vertex_groups}
    group_names = {group.index: group.name for group in mesh.vertex_groups}
    missing = sorted({name for index in changed_indices for name in weights[index]} - set(groups))
    if missing:
        raise SanitationError(f"sanitized weights reference missing groups: {missing}")
    for index in changed_indices:
        vertex = mesh.data.vertices[index]
        current = {group_names[item.group] for item in vertex.groups}
        target = weights[index]
        for name in sorted(current - set(target)):
            groups[name].remove([index])
        for name, value in sorted(target.items()):
            groups[name].add([index], float(value), "REPLACE")
    mesh.data.update()


def _export_sanitized_glb(bpy: Any, armature: Any, mesh: Any, path: Path) -> None:
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action, do_unlink=True)
    if armature.animation_data is not None:
        armature.animation_data_clear()
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    static_audit._select_runtime_only(bpy, armature, mesh)
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size <= 0:
        raise SanitationError("sanitized bind-pose GLB export failed")


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if path.exists():
            path.unlink()
        raise


def _rename_file_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SanitationError("atomic no-replace file publication is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    value = ctypes.get_errno()
    if value == errno.EEXIST:
        raise SanitationError(f"immutable evidence already exists: {destination}")
    raise SanitationError(
        f"atomic no-replace evidence rename failed: {os.strerror(value)}"
    )


def atomic_write_immutable_noreplace(path: Path, payload: bytes) -> Path:
    path = Path(path)
    parent = path.parent.resolve()
    if parent.is_symlink() or not parent.is_dir():
        raise SanitationError("immutable evidence parent is invalid")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        _rename_file_noreplace(temporary, path)
        static_audit._fsync_directory(parent)
        return path
    finally:
        if temporary.exists():
            temporary.unlink()


def _final_record(staged: Path, final: Path) -> dict[str, Any]:
    record = _file_record(staged)
    record["path"] = str(final.resolve())
    return record


def _write_failure_evidence(error: BaseException) -> Path:
    parent = PINNED_CONTRACT.output_dir.parent
    path = parent / f"{PINNED_CONTRACT.output_dir.name}.failed.{uuid.uuid4().hex}.json"
    payload = {
        "schema": "pixal_tokenrig_weight_sanitation_failure_v1",
        "asset_id": ASSET_ID,
        "decision": "rejected",
        "readiness_bundle_published": False,
        "animation_authorized": False,
        "inference_used": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure": {"type": type(error).__name__, "message": str(error)},
    }
    atomic_write_immutable_noreplace(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return path


def run_sanitation() -> Path:
    import bpy
    from mathutils import Matrix

    contract = PINNED_CONTRACT
    output_dir = Path(os.path.abspath(os.fspath(contract.output_dir)))
    parent = output_dir.parent
    staging: Path | None = None
    authenticated: Mapping[str, Any] | None = None
    try:
        if output_dir != OUTPUT_DIR.resolve() or output_dir.name != "sanitized_weights_v1":
            raise SanitationError("sanitized output path changed")
        if output_dir.exists() or output_dir.is_symlink():
            raise SanitationError("immutable sanitized output already exists")
        authenticated = authenticate_inputs()
        fitted_parsed = static_audit.read_glb(contract.fitted_glb.path)
        input_inverse_bind = static_audit.extract_inverse_bind_contract(
            fitted_parsed
        )
        if len(input_inverse_bind["joint_names"]) != 52:
            raise SanitationError("fitted inverse-bind contract is not 52 joints")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.", suffix=".staging", dir=str(parent)
            )
        )
        bpy.ops.wm.read_factory_settings(use_empty=True)
        imported = bpy.ops.import_scene.gltf(filepath=str(contract.fitted_glb.path))
        if "FINISHED" not in imported:
            raise SanitationError("could not import the authenticated fitted GLB")
        import_helpers = static_audit.remove_gltf_import_helpers(bpy)
        armature, mesh, orphans = static_audit.identify_exact_runtime(bpy)
        removed_orphans = static_audit.remove_proven_runtime_orphans(bpy, orphans)
        input_mesh = static_audit.capture_blender_mesh_contract(mesh)
        input_surface = static_audit.capture_blender_surface_reference(mesh)
        input_bones = static_audit.bone_records_from_armature(armature)
        static_audit.validate_hierarchy(input_bones)
        input_full_rest = static_audit.capture_blender_full_rest_contract(
            armature
        )
        if len(input_full_rest["bones"]) != 52:
            raise SanitationError("fitted full rest contract is not 52 bones")
        input_weights, input_positions = static_audit.extract_vertex_weights(
            mesh, armature
        )
        roots = static_audit.runtime_roots({armature, mesh})
        saved_matrices = {root: root.matrix_world.copy() for root in roots}
        floor = static_audit.mesh_floor_z(mesh)
        closure = Matrix.Translation((0.0, 0.0, -floor)) @ Matrix.Rotation(
            math.pi, 4, "Z"
        )
        for root in roots:
            root.matrix_world = closure @ root.matrix_world
        bpy.context.view_layer.update()
        canonical_bones = static_audit.bone_records_from_armature(armature)
        semantics = static_audit.resolve_five_semantic_chains(canonical_bones)
        before_weights, canonical_positions = static_audit.extract_vertex_weights(
            mesh, armature
        )
        if before_weights != input_weights:
            raise SanitationError("canonical transform changed fitted weights")
        pre_seam = _measure_seam_contamination(canonical_positions, before_weights)
        pre_bilateral = _measure_bilateral_contamination(
            canonical_positions, before_weights, semantics["chains"]
        )
        seam_rejection = _expected_rejection(
            lambda: static_audit.validate_seam_weights(
                canonical_positions, before_weights
            ),
            "UV seam duplicate vertex",
        )
        bilateral_rejection = _expected_rejection(
            lambda: static_audit.validate_bilateral_contamination(
                canonical_positions, before_weights, semantics["chains"]
            ),
            "opposite-limb contamination",
        )
        parents = {bone.name: bone.parent for bone in canonical_bones}
        planned_weights, report, planned_changes = sanitize_weight_maps(
            positions=canonical_positions,
            vertex_weights=before_weights,
            chains=semantics["chains"],
            parents=parents,
        )
        seam_group_records = report.pop("seam_group_records")
        changed_indices = [record["vertex_index"] for record in planned_changes]
        _apply_changed_weights(mesh, planned_weights, changed_indices)
        applied_weights, applied_positions = static_audit.extract_vertex_weights(
            mesh, armature
        )
        if applied_positions != canonical_positions:
            raise SanitationError("weight application changed canonical positions")
        storage_error = max(
            static_audit._weight_l1(planned, applied)
            for planned, applied in zip(planned_weights, applied_weights)
        )
        if storage_error > static_audit.SEAM_WEIGHT_L1_TOLERANCE:
            raise SanitationError(
                f"Blender weight storage changed planned weights by L1={storage_error}"
            )
        post_weight_validation = static_audit.validate_vertex_weights(
            applied_weights, bone_names=set(parents)
        )
        post_seam_validation = static_audit.validate_seam_weights(
            canonical_positions, applied_weights
        )
        post_bilateral_validation = static_audit.validate_bilateral_contamination(
            canonical_positions, applied_weights, semantics["chains"]
        )
        minimum_applied_weight = min(
            value for weights in applied_weights for value in weights.values()
        )
        if minimum_applied_weight <= BLENDER_EXPORT_MIN_INFLUENCE:
            raise SanitationError(
                "Blender storage left an influence at or below the exporter cutoff"
            )

        planned_by_index = {
            int(record["vertex_index"]): record for record in planned_changes
        }
        applied_changes = []
        all_l1 = []
        for index, (before, after) in enumerate(zip(before_weights, applied_weights)):
            l1 = static_audit._weight_l1(before, after)
            all_l1.append(l1)
            if l1 > 1.0e-15:
                applied_changes.append(
                    {
                        "vertex_index": index,
                        "before": dict(sorted(before.items())),
                        "after": dict(sorted(after.items())),
                        "l1_before_after": l1,
                        "transferred_mass": float(
                            planned_by_index.get(index, {}).get(
                                "transferred_mass", 0.0
                            )
                        ),
                        "export_floor_added_mass": float(
                            planned_by_index.get(index, {}).get(
                                "export_floor_added_mass", 0.0
                            )
                        ),
                        "export_floor_component_count": int(
                            planned_by_index.get(index, {}).get(
                                "export_floor_component_count", 0
                            )
                        ),
                    }
                )
        report = dict(report)
        report["export_floor_projection"] = dict(
            report["export_floor_projection"]
        )
        report["export_floor_projection"][
            "minimum_applied_blender_weight"
        ] = minimum_applied_weight
        report.update(
            {
                "changed_vertex_count": len(applied_changes),
                "changed_vertex_ratio": len(applied_changes) / len(applied_weights),
                "per_vertex_l1_accounting": {
                    "vertex_count": len(applied_weights),
                    "explicit_changed_record_count": len(applied_changes),
                    "implicit_unchanged_vertex_count": len(applied_weights)
                    - len(applied_changes),
                    "implicit_unchanged_l1_before_after": 0.0,
                },
                "l1_all_vertices": _distribution(all_l1),
                "l1_changed_vertices": _distribution(
                    [record["l1_before_after"] for record in applied_changes]
                ),
                "maximum_storage_l1_from_plan": storage_error,
                "weight_validation": post_weight_validation,
                "seam_validation": post_seam_validation,
                "bilateral_validation": post_bilateral_validation,
            }
        )
        change_path = staging / "weight_changes.jsonl"
        _write_exclusive(change_path, serialize_change_records(applied_changes))
        seam_group_path = staging / "seam_groups.jsonl"
        _write_exclusive(
            seam_group_path,
            serialize_seam_group_records(seam_group_records),
        )

        for root, matrix in saved_matrices.items():
            root.matrix_world = matrix
        bpy.context.view_layer.update()
        restored_error = max(
            abs(float(root.matrix_world[row][column]) - float(matrix[row][column]))
            for root, matrix in saved_matrices.items()
            for row in range(4)
            for column in range(4)
        )
        if restored_error > 1.0e-12:
            raise SanitationError("runtime root transforms were not restored exactly")
        output_bones = static_audit.bone_records_from_armature(armature)
        rest_validation = static_audit._compare_rest_bones(
            input_bones, output_bones
        )
        output_full_rest = static_audit.capture_blender_full_rest_contract(
            armature
        )
        in_scene_full_rest_validation = static_audit.compare_full_rest_contracts(
            input_full_rest, output_full_rest
        )
        output_mesh = static_audit.capture_blender_mesh_contract(mesh)
        mesh_validation = static_audit.compare_mesh_contracts(
            input_mesh, output_mesh, allow_serialization_splits=False
        )
        output_surface = static_audit.capture_blender_surface_reference(mesh)
        surface_validation = static_audit.compare_surface_references(
            input_surface, output_surface
        )
        export_weights, export_positions = static_audit.extract_vertex_weights(
            mesh, armature
        )
        export_semantics = static_audit.resolve_five_semantic_chains(output_bones)
        glb_path = staging / "tokenrig_transfer.glb"
        _export_sanitized_glb(bpy, armature, mesh, glb_path)
        original_parsed = static_audit.read_glb(contract.original_source_glb.path)
        output_parsed = static_audit.read_glb(glb_path)
        output_pbr = static_audit.compare_pbr_payloads(
            static_audit.pbr_payload_contract(original_parsed),
            static_audit.pbr_payload_contract(output_parsed),
        )
        raw_surface = static_audit.analyze_raw_serialization_equivalence(
            original_parsed, output_parsed
        )
        output_inverse_bind = static_audit.extract_inverse_bind_contract(
            output_parsed
        )
        inverse_bind_validation = static_audit.compare_inverse_bind_contracts(
            input_inverse_bind, output_inverse_bind
        )
        if inverse_bind_validation["joint_count"] != 52:
            raise SanitationError("output inverse-bind contract is not 52 joints")
        roundtrip_armature, _, roundtrip = static_audit.roundtrip_validate_bind(
            bpy=bpy,
            glb_path=glb_path,
            source_pbr=static_audit.pbr_payload_contract(original_parsed),
            expected_mesh=output_mesh,
            expected_surface=output_surface,
            expected_bones=output_bones,
            expected_positions=export_positions,
            expected_weights=export_weights,
            expected_semantics=export_semantics,
        )
        roundtrip_full_rest = static_audit.capture_blender_full_rest_contract(
            roundtrip_armature
        )
        roundtrip_full_rest_validation = static_audit.compare_full_rest_contracts(
            input_full_rest, roundtrip_full_rest
        )

        verify_code_snapshot(authenticated["execution_code_snapshot"])
        code = {
            "sanitizer": authenticated["execution_code_snapshot"]["sanitizer"],
            "static_audit": authenticated["execution_code_snapshot"]["static_audit"],
            "fitted_wrapper": authenticated["fitted_wrapper"],
            "delegated_base_runner": authenticated["delegated_base_runner"],
        }
        manifest = {
            "schema": SCHEMA,
            "asset_id": ASSET_ID,
            "attempt": "deterministic_learned_weight_sanitation",
            "algorithm_version": ALGORITHM_VERSION,
            "source_front": static_audit.SOURCE_FRONT,
            "canonical_front": static_audit.CANONICAL_FRONT,
            "inference_used": False,
            "rocketbox_mesh_used": False,
            "rocketbox_weights_used": False,
            "animation_authorized": False,
            "static_audit_status": "pending_sanitized_static_audit",
            "publication": {
                "directory_mode": "0755",
                "artifact_mode": "0444",
                "no_replace": True,
                "directory_mode_reason": (
                    "owner write permission is required only to create the nested "
                    "static_audit_v1 readiness bundle"
                ),
            },
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                key: authenticated[key]
                for key in (
                    "original_source_glb",
                    "original_source_manifest",
                    "direct_glb",
                    "recovery_manifest",
                    "direct_failures",
                    "fitted_glb",
                    "fitted_manifest",
                    "fitted_failures",
                    "prior_sanitation_failures",
                    "fitted_failure_summary",
                    "fitted_authentication",
                )
            },
            "code": code,
            "pre_sanitation": {
                "seam_measurement": pre_seam,
                "seam_rejection": seam_rejection,
                "bilateral_measurement": pre_bilateral,
                "bilateral_rejection": bilateral_rejection,
            },
            "sanitation": report,
            "validation": {
                "input_pbr": authenticated["input_pbr"],
                "input_raw_surface": authenticated["input_raw_surface"],
                "in_scene_mesh": mesh_validation,
                "in_scene_surface": surface_validation,
                "in_scene_rest": rest_validation,
                "in_scene_full_rest": in_scene_full_rest_validation,
                "roundtrip_full_rest": roundtrip_full_rest_validation,
                "inverse_bind": inverse_bind_validation,
                "restored_root_matrix_maximum_error": restored_error,
                "output_pbr": output_pbr,
                "output_raw_surface": raw_surface,
                "roundtrip": roundtrip,
                "removed_gltf_import_helpers": import_helpers,
                "removed_proven_orphans": list(removed_orphans),
            },
            "artifacts": {
                "weight_changes": _final_record(
                    change_path, output_dir / change_path.name
                ),
                "seam_groups": _final_record(
                    seam_group_path, output_dir / seam_group_path.name
                ),
            },
            "output": _final_record(glb_path, output_dir / glb_path.name),
        }
        manifest_path = staging / "tokenrig_manifest.json"
        _write_exclusive(
            manifest_path,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        for path in (glb_path, change_path, seam_group_path, manifest_path):
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise SanitationError(f"staged artifact is missing: {path.name}")
            static_audit._fsync_file(path)
            path.chmod(0o444)
        static_audit._fsync_directory(staging)
        verify_code_snapshot(authenticated["execution_code_snapshot"])
        static_audit.rename_directory_noreplace(staging, output_dir)
        staging = None
        output_dir.chmod(0o755)
        static_audit._fsync_directory(parent)
        return output_dir / "tokenrig_manifest.json"
    except BaseException as error:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        try:
            evidence = _write_failure_evidence(error)
        except BaseException as evidence_error:
            raise SanitationError(
                f"sanitation failed ({error}); failure evidence also failed ({evidence_error})"
            ) from error
        raise SanitationError(
            f"sanitation rejected: {error}; failure_evidence={evidence}"
        ) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(
        asset_id=ASSET_ID,
        fitted_glb=FITTED_GLB,
        fitted_manifest=FITTED_MANIFEST,
        output_dir=OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.asset_id != ASSET_ID
        or args.fitted_glb != FITTED_GLB
        or args.fitted_manifest != FITTED_MANIFEST
        or args.output_dir != OUTPUT_DIR
    ):
        raise SanitationError("CLI defaults changed from the pinned contract")
    manifest = run_sanitation()
    print(f"TOKENRIG_SANITIZED_WEIGHTS_PUBLISHED {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []))
