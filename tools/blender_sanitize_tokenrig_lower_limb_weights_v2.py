#!/usr/bin/env python3
"""Generic lower-limb-only TokenRig weight sanitation and immutable export.

``passed_static_owner`` consumes an owner-audited bind GLB and its exact static
semantic evidence. ``pre_static_repair`` consumes an unaudited TokenRig candidate,
its producer manifest, source Pixal GLB, and immutable rejection evidence; semantic
chains are then resolved deterministically from the runtime skeleton. Both modes
publish only a research preflight that must pass a fresh full static audit.

Side classification is performed in canonical world space (left = +X), never in
mesh-local coordinates and never from whole-body/arm width.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import blender_sanitize_tokenrig_human_weights as legacy
from tools import blender_tokenrig_human_static_audit as static_audit
from tools import route2_human_contract_common as common


SCHEMA = "pixal_tokenrig_lower_limb_sanitized_weights_v2"
FAILURE_SCHEMA = "pixal_tokenrig_lower_limb_weight_sanitation_failure_v2"
ALGORITHM_VERSION = "canonical_world_lower_body_leg_mass_side_transfer_v2"
OUTPUT_GLB_NAME = "tokenrig_transfer.glb"
OUTPUT_MANIFEST_NAME = "tokenrig_manifest.json"
CHANGE_LOG_NAME = "lower_limb_weight_changes.jsonl"
ROUNDTRIP_DIAGNOSTIC_NAME = "roundtrip_surface_skin_diagnostic.json"
LEG_DOMINANT_MINIMUM_MASS = 0.5
LOWER_BODY_EXTENT_PERCENTILE = 0.99
LOWER_BODY_TRANSITION_RATIO = 0.08
MINIMUM_TRANSITION_HALF_WIDTH_M = 8.0e-6
OPPOSITE_LEG_TOLERANCE = 1.0e-6
MAXIMUM_INFLUENCES = 4
MODES = ("passed_static_owner", "pre_static_repair")
_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9_]{0,126}[a-z0-9])?")


class LowerLimbSanitationError(RuntimeError):
    """Input evidence, sanitation, roundtrip, or publication is invalid."""


def _require_asset_id(value: Any) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise LowerLimbSanitationError("asset_id must be a canonical lower-case identifier")
    return value


def _absolute_direct_file(
    value: Path,
    description: str,
    *,
    require_mode: int | None = 0o444,
    allow_empty: bool = False,
) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise LowerLimbSanitationError(f"{description} is unavailable: {error}") from error
    if (
        not path.is_absolute()
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or (metadata.st_size <= 0 and not allow_empty)
        or path.resolve() != path
    ):
        raise LowerLimbSanitationError(f"{description} must be a non-empty direct regular file")
    if require_mode is not None and stat.S_IMODE(metadata.st_mode) != require_mode:
        raise LowerLimbSanitationError(
            f"{description} must have mode {require_mode:04o}"
        )
    return path


def _file_record(
    path: Path,
    description: str,
    *,
    require_mode: int | None = 0o444,
    allow_empty: bool = False,
) -> dict[str, Any]:
    path = _absolute_direct_file(
        path, description, require_mode=require_mode, allow_empty=allow_empty
    )
    if path.stat().st_size == 0:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(b"").hexdigest(),
            "size_bytes": 0,
            "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
        }
    record = common.hash_file_snapshot(
        path,
        root=path.parent,
        description=description,
        error_type=LowerLimbSanitationError,
        require_mode=require_mode,
    )
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
        "mode": record["mode"],
    }


def _json_snapshot(path: Path, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _absolute_direct_file(path, description)
    data, snapshot = common.read_file_snapshot(
        path,
        root=path.parent,
        description=description,
        error_type=LowerLimbSanitationError,
        require_mode=0o444,
    )
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LowerLimbSanitationError(f"{description} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise LowerLimbSanitationError(f"{description} root must be an object")
    return value, {
        "path": snapshot["path"],
        "sha256": snapshot["sha256"],
        "size_bytes": snapshot["size_bytes"],
        "mode": snapshot["mode"],
    }


def _record_matches(record: Any, current: Mapping[str, Any]) -> bool:
    return isinstance(record, Mapping) and (
        record.get("sha256") == current["sha256"]
        and record.get("size_bytes") == current["size_bytes"]
        and (
            record.get("path") in (None, current["path"])
            or record.get("filename") == Path(str(current["path"])).name
        )
    )


def _contains_record(value: Any, current: Mapping[str, Any]) -> bool:
    if _record_matches(value, current):
        return True
    if isinstance(value, Mapping):
        return any(_contains_record(child, current) for child in value.values())
    if isinstance(value, list):
        return any(_contains_record(child, current) for child in value)
    return False


def _validated_semantic_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LowerLimbSanitationError("static semantic mapping must be an object")
    chains = value.get("chains")
    semantic_bones = value.get("semantic_bones")
    side_basis = value.get("side_basis")
    if not isinstance(chains, Mapping) or not isinstance(semantic_bones, Mapping):
        raise LowerLimbSanitationError("static semantic chains/bones are missing")
    if side_basis != {"left": "positive-x", "right": "negative-x"}:
        raise LowerLimbSanitationError("static semantic side basis must be left +X/right -X")
    required_chains = ("axial", "left_arm", "right_arm", "left_leg", "right_leg")
    if set(chains) != set(required_chains):
        raise LowerLimbSanitationError("static semantic chain set is not exact")
    normalized_chains: dict[str, list[str]] = {}
    for role in required_chains:
        chain = chains.get(role)
        if (
            not isinstance(chain, list)
            or len(chain) < 4
            or any(not isinstance(name, str) or not name for name in chain)
            or len(set(chain)) != len(chain)
        ):
            raise LowerLimbSanitationError(f"static semantic chain is invalid: {role}")
        normalized_chains[role] = list(chain)
    left = set(normalized_chains["left_leg"][-4:])
    right = set(normalized_chains["right_leg"][-4:])
    if left & right:
        raise LowerLimbSanitationError("left/right lower-limb chains overlap")
    expected_roles = {
        "pelvis",
        "head",
        "neck",
        "spine",
        "left_thigh",
        "left_calf",
        "left_foot",
        "left_toe",
        "right_thigh",
        "right_calf",
        "right_foot",
        "right_toe",
    }
    if not expected_roles <= set(semantic_bones):
        raise LowerLimbSanitationError("static semantic bone roles are incomplete")
    for side in ("left", "right"):
        expected = normalized_chains[f"{side}_leg"][-4:]
        actual = [
            semantic_bones[f"{side}_thigh"],
            semantic_bones[f"{side}_calf"],
            semantic_bones[f"{side}_foot"],
            semantic_bones[f"{side}_toe"],
        ]
        if actual != expected:
            raise LowerLimbSanitationError(
                f"{side} semantic roles do not equal the final four leg-chain bones"
            )
    return {
        "method": value.get("method"),
        "chains": normalized_chains,
        "semantic_bones": json.loads(json.dumps(semantic_bones, sort_keys=True)),
        "side_basis": dict(side_basis),
    }


def authenticate_inputs(
    *,
    mode: str,
    asset_id: str,
    source_glb: Path,
    input_glb: Path,
    input_manifest: Path,
    static_qa_json: Path | None,
    prior_failures: Sequence[Path],
) -> dict[str, Any]:
    asset_id = _require_asset_id(asset_id)
    if mode not in MODES:
        raise LowerLimbSanitationError(f"mode must be one of {MODES}")
    source_record = _file_record(
        source_glb, "source Pixal GLB", require_mode=None
    )
    glb_record = _file_record(input_glb, "input TokenRig GLB")
    with Path(glb_record["path"]).open("rb") as stream:
        header = stream.read(4)
    if header != b"glTF":
        raise LowerLimbSanitationError("input TokenRig GLB has no glTF binary header")
    producer, producer_record = _json_snapshot(input_manifest, "input TokenRig manifest")
    if producer.get("asset_id") != asset_id:
        raise LowerLimbSanitationError("input manifest asset_id mismatch")
    if not _contains_record(producer, source_record):
        raise LowerLimbSanitationError(
            "input producer manifest does not bind the exact source Pixal GLB"
        )
    source_parsed = static_audit.read_glb(Path(source_record["path"]))
    input_parsed = static_audit.read_glb(Path(glb_record["path"]))
    input_pbr = static_audit.compare_pbr_payloads(
        static_audit.pbr_payload_contract(source_parsed),
        static_audit.pbr_payload_contract(input_parsed),
    )
    static_record: dict[str, Any] | None = None
    semantic: dict[str, Any] | None = None
    failure_records: list[dict[str, Any]] = []
    if mode == "passed_static_owner":
        if static_qa_json is None or prior_failures:
            raise LowerLimbSanitationError(
                "passed_static_owner requires static QA and forbids repair failures"
            )
        static_qa, static_record = _json_snapshot(static_qa_json, "input static QA")
        if (
            static_qa.get("schema") != "tokenrig_human_static_qa_v1"
            or static_qa.get("asset_id") != asset_id
            or static_qa.get("readiness_bundle_published") is not True
            or static_qa.get("decision") != "automatic_static_checks_passed"
        ):
            raise LowerLimbSanitationError("static QA is not the passed owner evidence")
        checks = static_qa.get("checks")
        if not isinstance(checks, Mapping) or checks.get("automatic_static_checks") != "passed":
            raise LowerLimbSanitationError("static QA automatic checks did not pass")
        artifacts = static_qa.get("artifacts")
        if not isinstance(artifacts, Mapping) or not any(
            isinstance(descriptor, Mapping)
            and descriptor.get("filename") == "bind_pose.glb"
            for descriptor in artifacts.values()
        ):
            raise LowerLimbSanitationError("static QA bind-pose artifact is missing")
        authenticated = static_qa.get("authenticated")
        output = producer.get("output")
        if (
            not isinstance(authenticated, Mapping)
            or authenticated.get("tokenrig_manifest_sha256") != producer_record["sha256"]
            or authenticated.get("source_glb_sha256") != source_record["sha256"]
            or not isinstance(output, Mapping)
            or not _record_matches(output, glb_record)
            or authenticated.get("tokenrig_glb_sha256") != glb_record["sha256"]
        ):
            raise LowerLimbSanitationError(
                "static QA is not chained to the exact producer/source output"
            )
        semantic = _validated_semantic_evidence(checks.get("semantic_mapping"))
    else:
        if static_qa_json is not None or not prior_failures:
            raise LowerLimbSanitationError(
                "pre_static_repair forbids passed static QA and requires failures"
            )
        if not _record_matches(producer.get("output"), glb_record):
            raise LowerLimbSanitationError(
                "pre-static producer output does not bind the exact input GLB"
            )
        for index, failure_path in enumerate(prior_failures):
            failure, failure_record = _json_snapshot(
                failure_path, f"prior static failure {index}"
            )
            if (
                failure.get("asset_id") != asset_id
                or failure.get("decision") != "rejected"
                or failure.get("readiness_bundle_published") is not False
                or not isinstance(failure.get("failure"), Mapping)
                or not str(failure["failure"].get("message", ""))
            ):
                raise LowerLimbSanitationError(
                    "pre-static failure evidence is not an exact rejection"
                )
            if not _contains_record(producer, failure_record):
                raise LowerLimbSanitationError(
                    "producer manifest does not bind prior static failure evidence"
                )
            failure_records.append(failure_record)
    if _file_record(Path(glb_record["path"]), "input TokenRig GLB") != glb_record:
        raise LowerLimbSanitationError("input TokenRig GLB changed while parsing")
    if _file_record(
        Path(source_record["path"]), "source Pixal GLB", require_mode=None
    ) != source_record:
        raise LowerLimbSanitationError("source Pixal GLB changed while parsing")
    inverse_bind = static_audit.extract_inverse_bind_contract(input_parsed)
    if len(inverse_bind["joint_names"]) < 19:
        raise LowerLimbSanitationError("input TokenRig skeleton is incomplete")
    return {
        "mode": mode,
        "asset_id": asset_id,
        "source_glb": source_record,
        "input_glb": glb_record,
        "input_manifest": producer_record,
        "static_qa": static_record,
        "prior_failures": failure_records,
        "semantic_evidence": semantic,
        "inverse_bind": inverse_bind,
        "input_pbr": input_pbr,
    }


def build_leg_transfer_maps(chains: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    if not isinstance(chains, Mapping):
        raise LowerLimbSanitationError("semantic chains must be a mapping")
    try:
        left = tuple(str(name) for name in chains["left_leg"][-4:])
        right = tuple(str(name) for name in chains["right_leg"][-4:])
    except (KeyError, TypeError) as error:
        raise LowerLimbSanitationError("left/right leg semantic chains are missing") from error
    if (
        len(left) != 4
        or len(right) != 4
        or len(set(left)) != 4
        or len(set(right)) != 4
        or set(left) & set(right)
    ):
        raise LowerLimbSanitationError("lower-limb core chains must be four disjoint bones")
    pairs = list(zip(left, right, strict=True))
    return {
        "left_chain": list(left),
        "right_chain": list(right),
        "pairs": [[left_name, right_name] for left_name, right_name in pairs],
        "to_left": {right_name: left_name for left_name, right_name in pairs},
        "to_right": {left_name: right_name for left_name, right_name in pairs},
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise LowerLimbSanitationError("cannot compute an empty percentile")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def derive_lower_body_gate(
    *,
    canonical_world_positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    chains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if not canonical_world_positions or len(canonical_world_positions) != len(vertex_weights):
        raise LowerLimbSanitationError("positions and weights are empty or mismatched")
    transfer = build_leg_transfer_maps(chains)
    left_bones = set(transfer["left_chain"])
    right_bones = set(transfer["right_chain"])
    leg_bones = left_bones | right_bones
    positive: list[float] = []
    negative: list[float] = []
    candidate_count = 0
    for position, weights in zip(canonical_world_positions, vertex_weights, strict=True):
        if len(position) != 3 or any(not math.isfinite(float(value)) for value in position):
            raise LowerLimbSanitationError("canonical world positions must be finite XYZ")
        leg_mass = sum(float(weights.get(name, 0.0)) for name in leg_bones)
        if leg_mass + 1.0e-15 < LEG_DOMINANT_MINIMUM_MASS:
            continue
        candidate_count += 1
        x = float(position[0])
        left_mass = sum(float(weights.get(name, 0.0)) for name in left_bones)
        right_mass = sum(float(weights.get(name, 0.0)) for name in right_bones)
        if x > 0.0 and left_mass >= right_mass:
            positive.append(x)
        elif x < 0.0 and right_mass >= left_mass:
            negative.append(-x)
    if candidate_count == 0 or not positive or not negative:
        raise LowerLimbSanitationError(
            "leg-dominant vertices do not cover both canonical lateral sides"
        )
    positive_extent = _percentile(positive, LOWER_BODY_EXTENT_PERCENTILE)
    negative_extent = _percentile(negative, LOWER_BODY_EXTENT_PERCENTILE)
    half_width = min(positive_extent, negative_extent)
    if half_width <= MINIMUM_TRANSITION_HALF_WIDTH_M:
        raise LowerLimbSanitationError("lower-body lateral extent is degenerate")
    cutoff = max(
        MINIMUM_TRANSITION_HALF_WIDTH_M,
        half_width * LOWER_BODY_TRANSITION_RATIO,
    )
    return {
        "coordinate_space": "canonical_world",
        "canonical_center_x_m": 0.0,
        "side_basis": {"left": "positive-x", "right": "negative-x"},
        "leg_dominant_minimum_mass": LEG_DOMINANT_MINIMUM_MASS,
        "lower_body_extent_percentile": LOWER_BODY_EXTENT_PERCENTILE,
        "positive_extent_m": positive_extent,
        "negative_extent_m": negative_extent,
        "symmetric_half_width_m": half_width,
        "transition_ratio": LOWER_BODY_TRANSITION_RATIO,
        "transition_half_width_m": cutoff,
        "leg_dominant_vertex_count": candidate_count,
        "positive_leg_dominant_vertex_count": len(positive),
        "negative_leg_dominant_vertex_count": len(negative),
    }


def _validated_weights(values: Mapping[str, float]) -> dict[str, float]:
    finite = {
        str(name): float(value)
        for name, value in values.items()
        if math.isfinite(float(value)) and float(value) > 0.0
    }
    if not finite or len(finite) > MAXIMUM_INFLUENCES:
        raise LowerLimbSanitationError("vertex weights must have one to four finite influences")
    total = sum(finite.values())
    if (
        not math.isfinite(total)
        or total <= 0.0
        or abs(total - 1.0) > static_audit.WEIGHT_SUM_TOLERANCE
    ):
        raise LowerLimbSanitationError("vertex weight total is invalid")
    return dict(sorted(finite.items()))


def _normalize_by_leg_donor(
    values: Mapping[str, float], leg_bones: set[str]
) -> dict[str, float]:
    normalized = dict(sorted((str(name), float(value)) for name, value in values.items()))
    total = sum(normalized.values())
    delta = 1.0 - total
    if abs(delta) > 0.0:
        donors = [name for name in normalized if name in leg_bones]
        if not donors:
            raise LowerLimbSanitationError("changed lower-limb vertex has no leg donor")
        donor = min(donors, key=lambda name: (-normalized[name], name))
        normalized[donor] += delta
        if normalized[donor] <= 0.0 or not math.isfinite(normalized[donor]):
            raise LowerLimbSanitationError("leg donor cannot absorb normalization residual")
    return dict(sorted(normalized.items()))


def sanitize_lower_limb_weight_maps(
    *,
    canonical_world_positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    chains: Mapping[str, Sequence[str]],
) -> tuple[tuple[dict[str, float], ...], dict[str, Any], tuple[dict[str, Any], ...]]:
    """Move opposite leg-chain mass only on leg-dominant lateral vertices."""

    if not canonical_world_positions or len(canonical_world_positions) != len(vertex_weights):
        raise LowerLimbSanitationError("positions and weights are empty or mismatched")
    transfer = build_leg_transfer_maps(chains)
    bone_names = {
        str(name) for weights in vertex_weights for name in weights
    } | set(transfer["left_chain"]) | set(transfer["right_chain"])
    try:
        static_audit.validate_vertex_weights(vertex_weights, bone_names=bone_names)
        static_audit.validate_seam_weights(canonical_world_positions, vertex_weights)
    except static_audit.StaticAuditError as error:
        raise LowerLimbSanitationError(f"input skin is invalid: {error}") from error
    gate = derive_lower_body_gate(
        canonical_world_positions=canonical_world_positions,
        vertex_weights=vertex_weights,
        chains=chains,
    )
    left = set(transfer["left_chain"])
    right = set(transfer["right_chain"])
    all_legs = left | right
    cutoff = float(gate["transition_half_width_m"])
    results: list[dict[str, float]] = []
    changes: list[dict[str, Any]] = []
    pair_mass: dict[str, float] = {}
    eligible = 0
    center_preserved = 0
    non_leg_maximum_error = 0.0
    maximum_before_opposite = 0.0
    maximum_after_opposite = 0.0
    for index, (position, source) in enumerate(
        zip(canonical_world_positions, vertex_weights, strict=True)
    ):
        before = _validated_weights(source)
        x = float(position[0])
        leg_mass = sum(before.get(name, 0.0) for name in all_legs)
        left_mass = sum(before.get(name, 0.0) for name in left)
        right_mass = sum(before.get(name, 0.0) for name in right)
        mapping: Mapping[str, str] = {}
        side = "center"
        opposite_before = 0.0
        if leg_mass + 1.0e-15 >= LEG_DOMINANT_MINIMUM_MASS:
            eligible += 1
            if x >= cutoff:
                side = "left"
                mapping = transfer["to_left"]
                opposite_before = right_mass
            elif x <= -cutoff:
                side = "right"
                mapping = transfer["to_right"]
                opposite_before = left_mass
            else:
                center_preserved += 1
        maximum_before_opposite = max(maximum_before_opposite, opposite_before)
        merged: dict[str, float] = {}
        moved = 0.0
        moved_pairs: dict[str, float] = {}
        for name, value in before.items():
            target = mapping.get(name, name)
            merged[target] = merged.get(target, 0.0) + value
            if target != name:
                moved += value
                key = f"{name}->{target}"
                moved_pairs[key] = moved_pairs.get(key, 0.0) + value
                pair_mass[key] = pair_mass.get(key, 0.0) + value
        after = (
            _normalize_by_leg_donor(merged, all_legs)
            if moved > 0.0
            else dict(before)
        )
        opposite_after = (
            sum(after.get(name, 0.0) for name in right)
            if side == "left"
            else sum(after.get(name, 0.0) for name in left)
            if side == "right"
            else 0.0
        )
        maximum_after_opposite = max(maximum_after_opposite, opposite_after)
        non_leg_names = (set(before) | set(after)) - all_legs
        non_leg_error = max(
            (abs(before.get(name, 0.0) - after.get(name, 0.0)) for name in non_leg_names),
            default=0.0,
        )
        non_leg_maximum_error = max(non_leg_maximum_error, non_leg_error)
        results.append(after)
        l1 = static_audit._weight_l1(before, after)
        if l1 > 1.0e-15:
            changes.append(
                {
                    "vertex_index": index,
                    "canonical_world_position_m": [float(value) for value in position],
                    "classified_side": side,
                    "leg_mass": leg_mass,
                    "opposite_leg_mass_before": opposite_before,
                    "opposite_leg_mass_after": opposite_after,
                    "transferred_mass": moved,
                    "transferred_mass_by_bone_pair": dict(sorted(moved_pairs.items())),
                    "before": before,
                    "after": after,
                    "l1_before_after": l1,
                }
            )
    sanitized = tuple(results)
    try:
        weight_validation = static_audit.validate_vertex_weights(
            sanitized, bone_names=bone_names
        )
        seam_validation = static_audit.validate_seam_weights(
            canonical_world_positions, sanitized
        )
    except static_audit.StaticAuditError as error:
        raise LowerLimbSanitationError(f"sanitized skin failed validation: {error}") from error
    if maximum_after_opposite > OPPOSITE_LEG_TOLERANCE:
        raise LowerLimbSanitationError(
            "sanitized eligible lateral vertices retain opposite-leg mass"
        )
    if non_leg_maximum_error > 1.0e-12:
        raise LowerLimbSanitationError("sanitation changed a non-leg influence")
    second, _, second_changes = _sanitize_idempotence_pass(
        canonical_world_positions=canonical_world_positions,
        vertex_weights=sanitized,
        transfer=transfer,
        gate=gate,
    )
    if second != sanitized or second_changes:
        raise LowerLimbSanitationError("lower-limb sanitation is not idempotent")
    before_contamination = lower_limb_contamination_statistics(
        canonical_world_positions=canonical_world_positions,
        vertex_weights=vertex_weights,
        chains=chains,
        gate=gate,
    )
    after_contamination = lower_limb_contamination_statistics(
        canonical_world_positions=canonical_world_positions,
        vertex_weights=sanitized,
        chains=chains,
        gate=gate,
    )
    report = {
        "algorithm_version": ALGORITHM_VERSION,
        "inference_used": False,
        "vertex_count": len(sanitized),
        "gate": gate,
        "eligible_leg_dominant_vertex_count": eligible,
        "preserved_center_transition_vertex_count": center_preserved,
        "changed_vertex_count": len(changes),
        "changed_vertex_ratio": len(changes) / len(sanitized),
        "total_transferred_mass": sum(change["transferred_mass"] for change in changes),
        "transferred_mass_by_bone_pair": dict(sorted(pair_mass.items())),
        "maximum_opposite_leg_mass_before": maximum_before_opposite,
        "maximum_opposite_leg_mass_after": maximum_after_opposite,
        "opposite_leg_mass_tolerance": OPPOSITE_LEG_TOLERANCE,
        "maximum_non_leg_weight_error": non_leg_maximum_error,
        "maximum_influences": max(len(weights) for weights in sanitized),
        "idempotence": {"passed": True, "second_pass_changed_vertex_count": 0},
        "weight_validation": weight_validation,
        "seam_validation": seam_validation,
        "transfer_map": transfer,
        "contamination_statistics": {
            "before": before_contamination,
            "after": after_contamination,
        },
    }
    return sanitized, report, tuple(changes)


def _sanitize_idempotence_pass(
    *,
    canonical_world_positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    transfer: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> tuple[tuple[dict[str, float], ...], float, tuple[int, ...]]:
    left = set(transfer["left_chain"])
    right = set(transfer["right_chain"])
    all_legs = left | right
    cutoff = float(gate["transition_half_width_m"])
    result = []
    changed = []
    maximum_l1 = 0.0
    for index, (position, source) in enumerate(
        zip(canonical_world_positions, vertex_weights, strict=True)
    ):
        before = _validated_weights(source)
        leg_mass = sum(before.get(name, 0.0) for name in all_legs)
        mapping: Mapping[str, str] = {}
        x = float(position[0])
        if leg_mass + 1.0e-15 >= LEG_DOMINANT_MINIMUM_MASS:
            if x >= cutoff:
                mapping = transfer["to_left"]
            elif x <= -cutoff:
                mapping = transfer["to_right"]
        merged: dict[str, float] = {}
        for name, value in before.items():
            target = mapping.get(name, name)
            merged[target] = merged.get(target, 0.0) + value
        after = (
            _normalize_by_leg_donor(merged, all_legs)
            if mapping and any(name in mapping for name in before)
            else dict(before)
        )
        l1 = static_audit._weight_l1(before, after)
        maximum_l1 = max(maximum_l1, l1)
        if l1 > 1.0e-15:
            changed.append(index)
        result.append(after)
    return tuple(result), maximum_l1, tuple(changed)


def serialize_change_records(records: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted(records, key=lambda record: int(record["vertex_index"]))
    return b"".join(
        (json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for record in ordered
    )


def lower_limb_contamination_statistics(
    *,
    canonical_world_positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    chains: Mapping[str, Sequence[str]],
    gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if gate is None:
        gate = derive_lower_body_gate(
            canonical_world_positions=canonical_world_positions,
            vertex_weights=vertex_weights,
            chains=chains,
        )
    transfer = build_leg_transfer_maps(chains)
    left = set(transfer["left_chain"])
    right = set(transfer["right_chain"])
    all_legs = left | right
    cutoff = float(gate["transition_half_width_m"])
    region_names = ("thigh", "calf", "foot", "toe")
    semantic_bones = {
        bone: {"semantic_side": side, "semantic_region": region}
        for side, bones in (
            ("left", transfer["left_chain"]),
            ("right", transfer["right_chain"]),
        )
        for region, bone in zip(region_names, bones, strict=True)
    }

    def empty_record() -> dict[str, Any]:
        return {
            "considered_vertex_count": 0,
            "contaminated_vertex_count": 0,
            "total_opposite_leg_mass": 0.0,
            "maximum_opposite_leg_mass": 0.0,
        }

    by_side = {"left": empty_record(), "right": empty_record()}
    by_region = {name: empty_record() for name in region_names}
    by_bone = {
        name: {**empty_record(), **semantic_bones[name]}
        for name in sorted(semantic_bones)
    }
    considered = 0
    contaminated = 0
    maximum = 0.0
    total = 0.0
    contaminated_positions: list[tuple[float, float, float]] = []
    for position, weights in zip(
        canonical_world_positions, vertex_weights, strict=True
    ):
        leg_mass = sum(float(weights.get(name, 0.0)) for name in all_legs)
        if leg_mass + 1.0e-15 < LEG_DOMINANT_MINIMUM_MASS:
            continue
        x = float(position[0])
        if x >= cutoff:
            classified_side = "left"
            opposite_bones = transfer["right_chain"]
        elif x <= -cutoff:
            classified_side = "right"
            opposite_bones = transfer["left_chain"]
        else:
            continue
        opposite_by_bone = {
            name: float(weights.get(name, 0.0)) for name in opposite_bones
        }
        opposite = sum(opposite_by_bone.values())
        considered += 1
        side_record = by_side[classified_side]
        side_record["considered_vertex_count"] += 1
        side_record["total_opposite_leg_mass"] += opposite
        side_record["maximum_opposite_leg_mass"] = max(
            side_record["maximum_opposite_leg_mass"], opposite
        )
        total += opposite
        maximum = max(maximum, opposite)
        is_contaminated = opposite > OPPOSITE_LEG_TOLERANCE
        if is_contaminated:
            contaminated += 1
            side_record["contaminated_vertex_count"] += 1
            contaminated_positions.append(tuple(float(value) for value in position))
        for region, bone in zip(region_names, opposite_bones, strict=True):
            value = opposite_by_bone[bone]
            region_record = by_region[region]
            bone_record = by_bone[bone]
            region_record["considered_vertex_count"] += 1
            bone_record["considered_vertex_count"] += 1
            region_record["total_opposite_leg_mass"] += value
            bone_record["total_opposite_leg_mass"] += value
            region_record["maximum_opposite_leg_mass"] = max(
                region_record["maximum_opposite_leg_mass"], value
            )
            bone_record["maximum_opposite_leg_mass"] = max(
                bone_record["maximum_opposite_leg_mass"], value
            )
            if value > OPPOSITE_LEG_TOLERANCE:
                region_record["contaminated_vertex_count"] += 1
                bone_record["contaminated_vertex_count"] += 1
    if contaminated_positions:
        position_bounds: dict[str, list[float]] | None = {
            "minimum": [
                min(position[axis] for position in contaminated_positions)
                for axis in range(3)
            ],
            "maximum": [
                max(position[axis] for position in contaminated_positions)
                for axis in range(3)
            ],
        }
        position_quantiles: dict[str, dict[str, float]] | None = {
            axis_name: {
                "p05": _percentile(
                    [position[axis] for position in contaminated_positions], 0.05
                ),
                "p50": _percentile(
                    [position[axis] for position in contaminated_positions], 0.50
                ),
                "p95": _percentile(
                    [position[axis] for position in contaminated_positions], 0.95
                ),
            }
            for axis, axis_name in enumerate(("x", "y", "z"))
        }
    else:
        position_bounds = None
        position_quantiles = None
    return {
        "basis": "canonical_world_lower_body_leg_mass_v2",
        "considered_distal_vertex_count": considered,
        "contaminated_vertex_count": contaminated,
        "total_opposite_leg_mass": total,
        "maximum_opposite_leg_mass": maximum,
        "tolerance": OPPOSITE_LEG_TOLERANCE,
        "gate": dict(gate),
        "by_side": by_side,
        "by_region": by_region,
        "by_bone": by_bone,
        "contaminated_position_bounds_m": position_bounds,
        "contaminated_position_quantiles_m": position_quantiles,
    }


def validate_lower_limb_bilateral_contamination(
    *,
    canonical_world_positions: Sequence[Sequence[float]],
    vertex_weights: Sequence[Mapping[str, float]],
    chains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    statistics = lower_limb_contamination_statistics(
        canonical_world_positions=canonical_world_positions,
        vertex_weights=vertex_weights,
        chains=chains,
    )
    contaminated = int(statistics["contaminated_vertex_count"])
    maximum = float(statistics["maximum_opposite_leg_mass"])
    if contaminated:
        raise LowerLimbSanitationError(
            "canonical-world opposite-leg contamination remains: "
            f"count={contaminated} maximum={maximum}"
        )
    return {**statistics, "maximum_opposite_limb_weight": maximum}


def compare_surface_corner_skin_weights(
    *,
    expected_surface: Any,
    expected_corner_vertex_indices: Sequence[int],
    expected_weights: Sequence[Mapping[str, float]],
    actual_surface: Any,
    actual_corner_vertex_indices: Sequence[int],
    actual_weights: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    expected_mapping, actual_mapping, maximum_position_error = (
        static_audit._match_unique_positions(
            expected_surface.unique_positions,
            actual_surface.unique_positions,
        )
    )

    def corner_weights(
        surface: Any,
        corner_vertices: Sequence[int],
        weights: Sequence[Mapping[str, float]],
        description: str,
    ) -> tuple[dict[int, Mapping[str, float]], int]:
        if len(corner_vertices) != len(surface.corner_unique_indices):
            raise LowerLimbSanitationError(
                f"{description} polygon-corner skin coverage changed"
            )
        values: dict[int, Mapping[str, float]] = {}
        for unique_index, vertex_index in zip(
            surface.corner_unique_indices, corner_vertices, strict=True
        ):
            key = int(unique_index)
            value = weights[int(vertex_index)]
            if (
                key in values
                and static_audit._weight_l1(values[key], value)
                > static_audit.SEAM_WEIGHT_L1_TOLERANCE
            ):
                raise LowerLimbSanitationError(
                    f"{description} polygon-corner seam weights disagree"
                )
            values[key] = value
        return values, len(set(int(index) for index in corner_vertices))

    expected_unique_weights, expected_vertex_count = corner_weights(
        expected_surface,
        expected_corner_vertex_indices,
        expected_weights,
        "expected",
    )
    actual_unique_weights, actual_vertex_count = corner_weights(
        actual_surface,
        actual_corner_vertex_indices,
        actual_weights,
        "actual",
    )
    expected_by_representative: dict[int, Mapping[str, float]] = {}
    for unique_index, weights in expected_unique_weights.items():
        representative = int(expected_mapping[unique_index])
        prior = expected_by_representative.get(representative)
        if (
            prior is not None
            and static_audit._weight_l1(prior, weights)
            > static_audit.SEAM_WEIGHT_L1_TOLERANCE
        ):
            raise LowerLimbSanitationError(
                "expected tolerance-cluster surface weights disagree"
            )
        expected_by_representative[representative] = weights
    actual_targets = {
        int(actual_mapping[unique_index]) for unique_index in actual_unique_weights
    }
    expected_targets = set(expected_by_representative)
    missing_targets = sorted(expected_targets - actual_targets)
    extra_targets = sorted(actual_targets - expected_targets)
    if missing_targets or extra_targets:
        raise LowerLimbSanitationError(
            "polygon-corner skin representative coverage changed: "
            f"missing={len(missing_targets)} extra={len(extra_targets)}"
        )
    weight_errors: list[float] = []
    position_errors: list[float] = []
    expected_positions = expected_surface.unique_positions
    actual_positions = actual_surface.unique_positions
    violations = 0
    for actual_unique_index, actual_weight in actual_unique_weights.items():
        target = int(actual_mapping[actual_unique_index])
        error = static_audit._weight_l1(
            expected_by_representative[target], actual_weight
        )
        weight_errors.append(error)
        violations += int(error > static_audit.SEAM_WEIGHT_L1_TOLERANCE)
        position_errors.append(
            math.sqrt(
                sum(
                    (
                        float(actual_positions[actual_unique_index * 3 + axis])
                        - float(expected_positions[target * 3 + axis])
                    )
                    ** 2
                    for axis in range(3)
                )
            )
        )
    if violations:
        raise LowerLimbSanitationError(
            "polygon-corner skin weights changed: "
            f"violations={violations} maximum_l1={max(weight_errors)}"
        )
    position_errors.sort()
    weight_errors.sort()
    return {
        "passed": True,
        "scope": "every_polygon_corner_via_authenticated_surface_correspondence_v2",
        "polygon_corner_count": len(actual_corner_vertex_indices),
        "matched_polygon_corner_count": len(actual_corner_vertex_indices),
        "expected_polygon_referenced_vertex_count": expected_vertex_count,
        "actual_polygon_referenced_vertex_count": actual_vertex_count,
        "expected_exact_unique_position_count": len(expected_surface.unique_positions)
        // 3,
        "actual_exact_unique_position_count": len(actual_surface.unique_positions) // 3,
        "expected_polygon_referenced_unique_count": len(expected_unique_weights),
        "actual_polygon_referenced_unique_count": len(actual_unique_weights),
        "expected_tolerance_cluster_count": len(set(expected_mapping)),
        "actual_tolerance_cluster_count": len(set(actual_mapping)),
        "missing_representative_count": 0,
        "extra_representative_count": 0,
        "nearest_position_error_m": {
            "p95": _percentile(position_errors, 0.95),
            "maximum": max(position_errors, default=0.0),
            "surface_contract_maximum": maximum_position_error,
            "tolerance": static_audit.SURFACE_POSITION_TOLERANCE_M,
        },
        "weight_l1_error": {
            "p95": _percentile(weight_errors, 0.95),
            "maximum": max(weight_errors, default=0.0),
            "violation_count": 0,
            "tolerance": static_audit.SEAM_WEIGHT_L1_TOLERANCE,
        },
    }


def validate_extracted_world_positions(
    mesh: Any, extracted_positions: Sequence[Sequence[float]]
) -> tuple[tuple[float, float, float], ...]:
    """Authenticate the world-space coordinate contract of weight extraction.

    ``static_audit.extract_vertex_weights`` already calls
    ``mesh_world_positions``.  Applying ``mesh.matrix_world`` to those values a
    second time reverses canonical X under a 180-degree root closure.
    """

    values = tuple(
        tuple(float(component) for component in position)
        for position in extracted_positions
    )
    direct = static_audit.mesh_world_positions(mesh)
    if len(values) != len(direct) or any(len(position) != 3 for position in values):
        raise LowerLimbSanitationError(
            "extracted vertex positions are not a complete world-space sequence"
        )
    maximum_error = max(
        (
            abs(values[index][axis] - float(direct[index][axis]))
            for index in range(len(values))
            for axis in range(3)
        ),
        default=0.0,
    )
    if maximum_error > 1.0e-12:
        raise LowerLimbSanitationError(
            "extracted vertex positions are not world-space: "
            f"maximum_error={maximum_error}"
        )
    return values


def roundtrip_validate_bind_v2(
    *,
    bpy: Any,
    glb_path: Path,
    source_pbr: Mapping[str, Mapping[str, Any]],
    expected_mesh: Mapping[str, Any],
    expected_surface: Any,
    expected_bones: Sequence[Any],
    expected_weights: Sequence[Mapping[str, float]],
    expected_corner_vertex_indices: Sequence[int],
    expected_semantics: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    from mathutils import Matrix

    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.gltf(filepath=str(glb_path))
    if "FINISHED" not in result:
        raise LowerLimbSanitationError("could not re-import sanitized bind GLB")
    import_helpers = static_audit.remove_gltf_import_helpers(bpy)
    armature, mesh, orphans = static_audit.identify_exact_runtime(bpy)
    removed_orphans = static_audit.remove_proven_runtime_orphans(bpy, orphans)
    mesh_validation = static_audit.compare_mesh_contracts(
        expected_mesh,
        static_audit.capture_blender_mesh_contract(mesh),
        allow_serialization_splits=True,
    )
    actual_surface = static_audit.capture_blender_surface_reference(mesh)
    surface_validation = static_audit.compare_surface_references(
        expected_surface, actual_surface
    )
    pbr_validation = static_audit.compare_pbr_payloads(
        source_pbr, static_audit.pbr_payload_contract(static_audit.read_glb(glb_path))
    )
    actual_bones = static_audit.bone_records_from_armature(armature)
    hierarchy = static_audit.validate_hierarchy(actual_bones)
    rest_validation = {
        **static_audit._compare_rest_bones(expected_bones, actual_bones),
        "passed": True,
    }
    actual_weights, actual_positions = static_audit.extract_vertex_weights(mesh, armature)
    weight_validation = static_audit.validate_vertex_weights(
        actual_weights, bone_names={bone.name for bone in actual_bones}
    )
    validate_extracted_world_positions(mesh, actual_positions)
    actual_corner_vertex_indices = [
        int(loop.vertex_index) for loop in mesh.data.loops
    ]
    try:
        skin_validation = compare_surface_corner_skin_weights(
            expected_surface=expected_surface,
            expected_corner_vertex_indices=expected_corner_vertex_indices,
            expected_weights=expected_weights,
            actual_surface=actual_surface,
            actual_corner_vertex_indices=actual_corner_vertex_indices,
            actual_weights=actual_weights,
        )
    except BaseException as error:
        checkpoint = {
            "schema": "tokenrig_surface_skin_roundtrip_diagnostic_v2",
            "status": "failed",
            "surface_validation": surface_validation,
            "expected_polygon_corner_count": len(expected_corner_vertex_indices),
            "actual_polygon_corner_count": len(actual_corner_vertex_indices),
            "expected_exact_unique_position_count": len(
                expected_surface.unique_positions
            )
            // 3,
            "actual_exact_unique_position_count": len(actual_surface.unique_positions)
            // 3,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        legacy._write_exclusive(
            glb_path.parent / ROUNDTRIP_DIAGNOSTIC_NAME,
            (json.dumps(checkpoint, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        raise
    legacy._write_exclusive(
        glb_path.parent / ROUNDTRIP_DIAGNOSTIC_NAME,
        (
            json.dumps(
                {
                    "schema": "tokenrig_surface_skin_roundtrip_diagnostic_v2",
                    "status": "passed",
                    "surface_validation": surface_validation,
                    "skin_validation": skin_validation,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    roots = static_audit.runtime_roots({armature, mesh})
    saved_matrices = {root: root.matrix_world.copy() for root in roots}
    canonical_yaw = Matrix.Rotation(math.pi, 4, "Z")
    for root in roots:
        root.matrix_world = canonical_yaw @ root.matrix_world
    bpy.context.view_layer.update()
    semantics = static_audit.resolve_five_semantic_chains(
        static_audit.bone_records_from_armature(armature)
    )
    if semantics["chains"] != expected_semantics["chains"]:
        raise LowerLimbSanitationError("GLB roundtrip changed semantic chains")
    canonical_weights, extracted_canonical_world_positions = (
        static_audit.extract_vertex_weights(mesh, armature)
    )
    if canonical_weights != actual_weights:
        raise LowerLimbSanitationError(
            "canonical root closure unexpectedly changed vertex weights"
        )
    canonical_world_positions = validate_extracted_world_positions(
        mesh, extracted_canonical_world_positions
    )
    seam_validation = static_audit.validate_seam_weights(
        canonical_world_positions, actual_weights
    )
    contamination = validate_lower_limb_bilateral_contamination(
        canonical_world_positions=canonical_world_positions,
        vertex_weights=actual_weights,
        chains=semantics["chains"],
    )
    for root, matrix in saved_matrices.items():
        root.matrix_world = matrix
    bpy.context.view_layer.update()
    return armature, mesh, {
        "passed": True,
        "mesh": mesh_validation,
        "surface": surface_validation,
        "pbr": pbr_validation,
        "hierarchy": hierarchy,
        "rest": rest_validation,
        "weights": weight_validation,
        "seams": seam_validation,
        "skin": skin_validation,
        "bilateral_contamination": contamination,
        "lower_limb_bilateral_contamination": contamination,
        "removed_proven_orphans": list(removed_orphans),
        "removed_gltf_import_helpers": import_helpers,
    }


def _code_snapshot() -> dict[str, Any]:
    return {
        "sanitizer_v2": _file_record(Path(__file__).resolve(), "sanitizer v2 code", require_mode=None),
        "static_audit": _file_record(Path(static_audit.__file__).resolve(), "static audit code", require_mode=None),
        "legacy_export_helpers": _file_record(Path(legacy.__file__).resolve(), "legacy export helper code", require_mode=None),
    }


def _verify_code_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected in snapshot.items():
        current = _file_record(Path(str(expected["path"])), f"{name} code", require_mode=None)
        if current != expected:
            raise LowerLimbSanitationError(f"authenticated execution code changed: {name}")


def _staged_record(
    staged: Path, final: Path, *, allow_empty: bool = False
) -> dict[str, Any]:
    record = _file_record(
        staged,
        f"staged {staged.name}",
        require_mode=None,
        allow_empty=allow_empty,
    )
    record["path"] = str(final.resolve())
    record["mode"] = "0444"
    return record


def _manifest_payload(
    *,
    asset_id: str,
    authenticated: Mapping[str, Any],
    code: Mapping[str, Any],
    report: Mapping[str, Any],
    validation: Mapping[str, Any],
    output_record: Mapping[str, Any],
    change_record: Mapping[str, Any],
    diagnostic_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "asset_id": asset_id,
        "input_mode": authenticated["mode"],
        "algorithm_version": ALGORITHM_VERSION,
        "state_classification": "research_candidate",
        "static_owner_status": "fresh_full_static_audit_required",
        "inference_used": False,
        "animation_authorized": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "source_glb": dict(authenticated["source_glb"]),
            "glb": dict(authenticated["input_glb"]),
            "manifest": dict(authenticated["input_manifest"]),
            "static_qa": (
                dict(authenticated["static_qa"])
                if authenticated["static_qa"] is not None
                else None
            ),
            "prior_failures": [
                dict(record) for record in authenticated["prior_failures"]
            ],
            "semantic_evidence": json.loads(
                json.dumps(authenticated["semantic_evidence"], sort_keys=True)
            ),
        },
        "code": json.loads(json.dumps(code, sort_keys=True)),
        "sanitation": json.loads(json.dumps(report, sort_keys=True)),
        "validation": json.loads(json.dumps(validation, sort_keys=True)),
        "artifacts": {
            "weight_changes": dict(change_record),
            "roundtrip_surface_skin_diagnostic": dict(diagnostic_record),
        },
        "output": dict(output_record),
        "publication": {
            "no_replace": True,
            "artifact_mode": "0444",
            "directory_mode": "0755",
        },
    }


def _validate_record(
    value: Any,
    *,
    root: Path,
    filename: str,
    description: str,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
        "mode",
    }:
        raise LowerLimbSanitationError(f"{description} record fields are invalid")
    path = Path(str(value.get("path")))
    if path.name != filename or path.parent != root:
        raise LowerLimbSanitationError(f"{description} record path is not canonical")
    current = _file_record(path, description, allow_empty=allow_empty)
    if current != value:
        raise LowerLimbSanitationError(f"{description} record is stale")
    return current


def _validate_external_record(
    value: Any, description: str, *, require_mode: int | None
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
        "mode",
    }:
        raise LowerLimbSanitationError(f"{description} record fields are invalid")
    current = _file_record(
        Path(str(value.get("path"))), description, require_mode=require_mode
    )
    if current != value:
        raise LowerLimbSanitationError(f"{description} record is stale")
    return current


def validate_published_manifest(path: Path) -> dict[str, Any]:
    manifest_path = _absolute_direct_file(path, "lower-limb sanitation manifest")
    if manifest_path.name != OUTPUT_MANIFEST_NAME:
        raise LowerLimbSanitationError("sanitation manifest filename is not canonical")
    payload, record = _json_snapshot(manifest_path, "lower-limb sanitation manifest")
    if set(payload) != {
        "schema",
        "asset_id",
        "input_mode",
        "algorithm_version",
        "state_classification",
        "static_owner_status",
        "inference_used",
        "animation_authorized",
        "recorded_at_utc",
        "input",
        "code",
        "sanitation",
        "validation",
        "artifacts",
        "output",
        "publication",
    }:
        raise LowerLimbSanitationError("sanitation manifest fields are incomplete or unexpected")
    common.reject_user_approval(payload, LowerLimbSanitationError, "sanitation manifest")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("input_mode") not in MODES
        or payload.get("algorithm_version") != ALGORITHM_VERSION
        or payload.get("state_classification") != "research_candidate"
        or payload.get("static_owner_status")
        != "fresh_full_static_audit_required"
        or payload.get("inference_used") is not False
        or payload.get("animation_authorized") is not False
        or payload.get("publication")
        != {"no_replace": True, "artifact_mode": "0444", "directory_mode": "0755"}
    ):
        raise LowerLimbSanitationError("sanitation manifest policy changed")
    _require_asset_id(payload.get("asset_id"))
    try:
        datetime.fromisoformat(str(payload.get("recorded_at_utc")))
    except ValueError as error:
        raise LowerLimbSanitationError("sanitation timestamp is invalid") from error
    root = manifest_path.parent
    input_evidence = payload.get("input")
    if not isinstance(input_evidence, Mapping) or set(input_evidence) != {
        "source_glb",
        "glb",
        "manifest",
        "static_qa",
        "prior_failures",
        "semantic_evidence",
    }:
        raise LowerLimbSanitationError("manifest input evidence is incomplete")
    _validate_external_record(
        input_evidence["source_glb"],
        "manifest source Pixal GLB",
        require_mode=None,
    )
    _validate_external_record(
        input_evidence["glb"], "manifest input GLB", require_mode=0o444
    )
    _validate_external_record(
        input_evidence["manifest"],
        "manifest input producer manifest",
        require_mode=0o444,
    )
    mode = payload["input_mode"]
    static_descriptor = input_evidence["static_qa"]
    failures = input_evidence["prior_failures"]
    if mode == "passed_static_owner":
        if static_descriptor is None or failures != []:
            raise LowerLimbSanitationError("passed-owner manifest evidence is contradictory")
        _validate_external_record(
            static_descriptor, "manifest input static QA", require_mode=0o444
        )
    else:
        if static_descriptor is not None or not isinstance(failures, list) or not failures:
            raise LowerLimbSanitationError("pre-static manifest evidence is contradictory")
        for index, descriptor in enumerate(failures):
            _validate_external_record(
                descriptor,
                f"manifest prior static failure {index}",
                require_mode=0o444,
            )
    _validated_semantic_evidence(input_evidence["semantic_evidence"])
    code = payload.get("code")
    if not isinstance(code, Mapping) or set(code) != {
        "sanitizer_v2",
        "static_audit",
        "legacy_export_helpers",
    }:
        raise LowerLimbSanitationError("manifest code snapshot is incomplete")
    for name, descriptor in code.items():
        _validate_external_record(
            descriptor, f"manifest {name} code", require_mode=None
        )
    _validate_record(payload.get("output"), root=root, filename=OUTPUT_GLB_NAME, description="sanitized GLB")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "weight_changes",
        "roundtrip_surface_skin_diagnostic",
    }:
        raise LowerLimbSanitationError("sanitation artifact inventory is invalid")
    change_record = _validate_record(
        artifacts["weight_changes"],
        root=root,
        filename=CHANGE_LOG_NAME,
        description="lower-limb weight changes",
        allow_empty=True,
    )
    _validate_record(
        artifacts["roundtrip_surface_skin_diagnostic"],
        root=root,
        filename=ROUNDTRIP_DIAGNOSTIC_NAME,
        description="surface-skin roundtrip diagnostic",
    )
    sanitation = payload.get("sanitation")
    if (
        not isinstance(sanitation, Mapping)
        or sanitation.get("algorithm_version") != ALGORITHM_VERSION
        or sanitation.get("idempotence")
        != {"passed": True, "second_pass_changed_vertex_count": 0}
        or sanitation.get("maximum_opposite_leg_mass_after", math.inf)
        > OPPOSITE_LEG_TOLERANCE
        or sanitation.get("maximum_influences", math.inf) > MAXIMUM_INFLUENCES
    ):
        raise LowerLimbSanitationError("sanitation report does not pass the v2 gates")
    change_path = Path(change_record["path"])
    try:
        change_lines = change_path.read_text(encoding="utf-8").splitlines()
        change_values = [json.loads(line) for line in change_lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LowerLimbSanitationError(f"weight change log is invalid: {error}") from error
    if (
        len(change_values) != sanitation.get("changed_vertex_count")
        or any(not isinstance(value, Mapping) for value in change_values)
        or [value.get("vertex_index") for value in change_values]
        != sorted({value.get("vertex_index") for value in change_values})
    ):
        raise LowerLimbSanitationError("weight change log is incomplete or noncanonical")
    validation = payload.get("validation")
    required_passed = (
        "in_scene_mesh",
        "in_scene_surface",
        "in_scene_rest",
        "in_scene_full_rest",
        "inverse_bind",
        "output_pbr",
        "roundtrip_full_rest",
    )
    if not isinstance(validation, Mapping) or any(
        not isinstance(validation.get(name), Mapping)
        or validation[name].get("passed") is not True
        for name in required_passed
    ):
        raise LowerLimbSanitationError("sanitation preservation/roundtrip gates did not pass")
    roundtrip = validation.get("roundtrip")
    if not isinstance(roundtrip, Mapping) or roundtrip.get("passed") is not True:
        raise LowerLimbSanitationError("sanitation GLB roundtrip did not pass")
    if record["mode"] != "0444":  # pragma: no cover - _json_snapshot enforces it.
        raise LowerLimbSanitationError("sanitation manifest is mutable")
    return payload


def _write_failure(output_dir: Path, asset_id: str, error: BaseException) -> Path:
    destination = output_dir.parent / f"{output_dir.name}.failed.{uuid.uuid4().hex}.json"
    payload = {
        "schema": FAILURE_SCHEMA,
        "asset_id": asset_id,
        "decision": "rejected",
        "readiness_bundle_published": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure": {"type": type(error).__name__, "message": str(error)},
    }
    return common.write_json_immutable_noreplace(
        destination,
        payload,
        LowerLimbSanitationError,
        "lower-limb sanitation failure evidence",
    )


def _preserve_failed_staging(
    *, staging: Path, output_dir: Path, asset_id: str, error: BaseException
) -> Path:
    failure_dir = output_dir.parent / f"{output_dir.name}.failed.{uuid.uuid4().hex}"
    if failure_dir.exists() or failure_dir.is_symlink():
        raise LowerLimbSanitationError("failure bundle destination already exists")
    preserved: dict[str, Any] = {}
    for path in sorted(staging.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise LowerLimbSanitationError(
                f"failure staging contains a non-file artifact: {path.name}"
            )
        record = _file_record(
            path,
            f"failed staged {path.name}",
            require_mode=None,
            allow_empty=True,
        )
        record["path"] = str(failure_dir / path.name)
        record["mode"] = "0444"
        preserved[path.name] = record
    payload = {
        "schema": FAILURE_SCHEMA,
        "asset_id": asset_id,
        "decision": "rejected",
        "readiness_bundle_published": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure": {"type": type(error).__name__, "message": str(error)},
        "preserved_artifacts": preserved,
        "external_inventory_descriptor": {
            "directory": str(failure_dir),
            "artifact_names": sorted(preserved),
            "artifact_count": len(preserved),
            "all_artifacts_sha256_and_size_recorded": True,
        },
    }
    failure_manifest = staging / "failure_manifest.json"
    legacy._write_exclusive(
        failure_manifest,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    for path in sorted(staging.iterdir()):
        static_audit._fsync_file(path)
        path.chmod(0o444)
    static_audit._fsync_directory(staging)
    staging.chmod(0o555)
    static_audit.rename_directory_noreplace(staging, failure_dir)
    static_audit._fsync_directory(failure_dir.parent)
    return failure_dir / "failure_manifest.json"


def run_sanitation(
    *,
    mode: str,
    asset_id: str,
    source_glb: Path,
    input_glb: Path,
    input_manifest: Path,
    static_qa_json: Path | None,
    prior_failures: Sequence[Path],
    output_dir: Path,
) -> Path:
    import bpy
    from mathutils import Matrix

    asset_id = _require_asset_id(asset_id)
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    parent = output_dir.parent
    if output_dir.exists() or output_dir.is_symlink():
        raise LowerLimbSanitationError("sanitized output directory already exists")
    if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
        raise LowerLimbSanitationError("sanitized output parent must be a direct real directory")
    staging: Path | None = None
    try:
        authenticated = authenticate_inputs(
            mode=mode,
            asset_id=asset_id,
            source_glb=source_glb,
            input_glb=input_glb,
            input_manifest=input_manifest,
            static_qa_json=static_qa_json,
            prior_failures=prior_failures,
        )
        code = _code_snapshot()
        input_path = Path(authenticated["input_glb"]["path"])
        input_parsed = static_audit.read_glb(input_path)
        input_inverse_bind = static_audit.extract_inverse_bind_contract(input_parsed)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.", suffix=".staging", dir=str(parent)
            )
        )
        bpy.ops.wm.read_factory_settings(use_empty=True)
        imported = bpy.ops.import_scene.gltf(filepath=str(input_path))
        if "FINISHED" not in imported:
            raise LowerLimbSanitationError("could not import the authenticated TokenRig GLB")
        import_helpers = static_audit.remove_gltf_import_helpers(bpy)
        armature, mesh, orphans = static_audit.identify_exact_runtime(bpy)
        removed_orphans = static_audit.remove_proven_runtime_orphans(bpy, orphans)
        input_mesh = static_audit.capture_blender_mesh_contract(mesh)
        input_surface = static_audit.capture_blender_surface_reference(mesh)
        input_bones = static_audit.bone_records_from_armature(armature)
        input_full_rest = static_audit.capture_blender_full_rest_contract(armature)
        roots = static_audit.runtime_roots({armature, mesh})
        saved_matrices = {root: root.matrix_world.copy() for root in roots}
        canonical_yaw = Matrix.Rotation(math.pi, 4, "Z")
        for root in roots:
            root.matrix_world = canonical_yaw @ root.matrix_world
        bpy.context.view_layer.update()
        canonical_bones = static_audit.bone_records_from_armature(armature)
        runtime_semantics = static_audit.resolve_five_semantic_chains(canonical_bones)
        if authenticated["semantic_evidence"] is None:
            evidence_semantics = _validated_semantic_evidence(runtime_semantics)
            authenticated = dict(authenticated)
            authenticated["semantic_evidence"] = evidence_semantics
        else:
            evidence_semantics = authenticated["semantic_evidence"]
            if (
                runtime_semantics.get("chains") != evidence_semantics["chains"]
                or runtime_semantics.get("semantic_bones")
                != evidence_semantics["semantic_bones"]
                or runtime_semantics.get("side_basis") != evidence_semantics["side_basis"]
            ):
                raise LowerLimbSanitationError(
                    "runtime skeleton semantics differ from authenticated static evidence"
                )
        before_weights, extracted_world_positions = static_audit.extract_vertex_weights(
            mesh, armature
        )
        world_positions = validate_extracted_world_positions(
            mesh, extracted_world_positions
        )
        planned, report, planned_changes = sanitize_lower_limb_weight_maps(
            canonical_world_positions=world_positions,
            vertex_weights=before_weights,
            chains=evidence_semantics["chains"],
        )
        changed_indices = [int(record["vertex_index"]) for record in planned_changes]
        legacy._apply_changed_weights(mesh, planned, changed_indices)
        applied_weights, applied_world_positions = static_audit.extract_vertex_weights(
            mesh, armature
        )
        if applied_world_positions != extracted_world_positions:
            raise LowerLimbSanitationError("weight application changed mesh positions")
        storage_error = max(
            static_audit._weight_l1(expected, actual)
            for expected, actual in zip(planned, applied_weights, strict=True)
        )
        if storage_error > static_audit.SEAM_WEIGHT_L1_TOLERANCE:
            raise LowerLimbSanitationError(
                f"Blender weight storage differs from plan by L1={storage_error}"
            )
        static_audit.validate_vertex_weights(
            applied_weights, bone_names={bone.name for bone in input_bones}
        )
        static_audit.validate_seam_weights(world_positions, applied_weights)
        applied_changes = []
        planned_by_index = {
            int(record["vertex_index"]): record for record in planned_changes
        }
        for index, (before, after) in enumerate(
            zip(before_weights, applied_weights, strict=True)
        ):
            l1 = static_audit._weight_l1(before, after)
            if l1 <= 1.0e-15:
                continue
            record = dict(planned_by_index[index])
            record["after"] = dict(sorted(after.items()))
            record["l1_before_after"] = l1
            applied_changes.append(record)
        report = dict(report)
        report["changed_vertex_count"] = len(applied_changes)
        report["changed_vertex_ratio"] = len(applied_changes) / len(applied_weights)
        report["maximum_storage_l1_from_plan"] = storage_error
        change_path = staging / CHANGE_LOG_NAME
        legacy._write_exclusive(change_path, serialize_change_records(applied_changes))
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
            raise LowerLimbSanitationError("runtime root matrices were not restored")
        output_bones = static_audit.bone_records_from_armature(armature)
        output_mesh = static_audit.capture_blender_mesh_contract(mesh)
        output_surface = static_audit.capture_blender_surface_reference(mesh)
        output_full_rest = static_audit.capture_blender_full_rest_contract(armature)
        corner_vertex_indices = [int(loop.vertex_index) for loop in mesh.data.loops]
        in_scene_mesh = static_audit.compare_mesh_contracts(
            input_mesh, output_mesh, allow_serialization_splits=False
        )
        in_scene_surface = static_audit.compare_surface_references(
            input_surface, output_surface
        )
        in_scene_rest = {
            **static_audit._compare_rest_bones(input_bones, output_bones),
            "passed": True,
        }
        in_scene_full_rest = static_audit.compare_full_rest_contracts(
            input_full_rest, output_full_rest
        )
        output_glb = staging / OUTPUT_GLB_NAME
        legacy._export_sanitized_glb(bpy, armature, mesh, output_glb)
        output_parsed = static_audit.read_glb(output_glb)
        output_pbr = static_audit.compare_pbr_payloads(
            static_audit.pbr_payload_contract(input_parsed),
            static_audit.pbr_payload_contract(output_parsed),
        )
        output_inverse_bind = static_audit.extract_inverse_bind_contract(output_parsed)
        inverse_bind = static_audit.compare_inverse_bind_contracts(
            input_inverse_bind, output_inverse_bind
        )
        roundtrip_armature, _, roundtrip = roundtrip_validate_bind_v2(
            bpy=bpy,
            glb_path=output_glb,
            source_pbr=static_audit.pbr_payload_contract(input_parsed),
            expected_mesh=output_mesh,
            expected_surface=output_surface,
            expected_bones=output_bones,
            expected_weights=applied_weights,
            expected_corner_vertex_indices=corner_vertex_indices,
            expected_semantics=runtime_semantics,
        )
        roundtrip_full_rest = static_audit.compare_full_rest_contracts(
            input_full_rest,
            static_audit.capture_blender_full_rest_contract(roundtrip_armature),
        )
        validation = {
            "in_scene_mesh": in_scene_mesh,
            "in_scene_surface": in_scene_surface,
            "in_scene_rest": in_scene_rest,
            "in_scene_full_rest": in_scene_full_rest,
            "inverse_bind": inverse_bind,
            "output_pbr": output_pbr,
            "roundtrip": roundtrip,
            "roundtrip_full_rest": roundtrip_full_rest,
            "removed_gltf_import_helpers": import_helpers,
            "removed_proven_orphans": list(removed_orphans),
            "restored_root_matrix_maximum_error": restored_error,
        }
        diagnostic_path = staging / ROUNDTRIP_DIAGNOSTIC_NAME
        for artifact in (output_glb, change_path, diagnostic_path):
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise LowerLimbSanitationError(f"staged artifact is empty: {artifact.name}")
            static_audit._fsync_file(artifact)
            artifact.chmod(0o444)
        output_record = _staged_record(output_glb, output_dir / OUTPUT_GLB_NAME)
        change_record = _staged_record(
            change_path, output_dir / CHANGE_LOG_NAME, allow_empty=True
        )
        diagnostic_record = _staged_record(
            diagnostic_path, output_dir / ROUNDTRIP_DIAGNOSTIC_NAME
        )
        manifest = _manifest_payload(
            asset_id=asset_id,
            authenticated=authenticated,
            code=code,
            report=report,
            validation=validation,
            output_record=output_record,
            change_record=change_record,
            diagnostic_record=diagnostic_record,
        )
        manifest_path = staging / OUTPUT_MANIFEST_NAME
        legacy._write_exclusive(
            manifest_path,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        static_audit._fsync_file(manifest_path)
        manifest_path.chmod(0o444)
        static_audit._fsync_directory(staging)
        _verify_code_snapshot(code)
        static_audit.rename_directory_noreplace(staging, output_dir)
        staging = None
        output_dir.chmod(0o755)
        static_audit._fsync_directory(parent)
        published = output_dir / OUTPUT_MANIFEST_NAME
        validate_published_manifest(published)
        return published
    except BaseException as error:
        try:
            if staging is not None and staging.exists():
                evidence = _preserve_failed_staging(
                    staging=staging,
                    output_dir=output_dir,
                    asset_id=asset_id,
                    error=error,
                )
                staging = None
            else:
                evidence = _write_failure(output_dir, asset_id, error)
        except BaseException as evidence_error:
            if staging is not None and staging.exists():
                # Keep the staging tree intact for manual recovery if bundling itself fails.
                pass
            raise LowerLimbSanitationError(
                f"sanitation failed ({error}); failure evidence also failed ({evidence_error})"
            ) from error
        raise LowerLimbSanitationError(
            f"lower-limb sanitation rejected: {error}; failure_evidence={evidence}"
        ) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--static-qa-json", type=Path)
    parser.add_argument("--prior-failure", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_sanitation(
        mode=args.mode,
        asset_id=args.asset_id,
        source_glb=args.source_glb,
        input_glb=args.input_glb,
        input_manifest=args.input_manifest,
        static_qa_json=args.static_qa_json,
        prior_failures=args.prior_failure,
        output_dir=args.output_dir,
    )
    print(f"TOKENRIG_LOWER_LIMB_SANITIZED_V2_PUBLISHED {manifest}")
    return 0


__all__ = [
    "ALGORITHM_VERSION",
    "CHANGE_LOG_NAME",
    "LEG_DOMINANT_MINIMUM_MASS",
    "LOWER_BODY_TRANSITION_RATIO",
    "LowerLimbSanitationError",
    "OPPOSITE_LEG_TOLERANCE",
    "OUTPUT_GLB_NAME",
    "OUTPUT_MANIFEST_NAME",
    "ROUNDTRIP_DIAGNOSTIC_NAME",
    "SCHEMA",
    "authenticate_inputs",
    "build_leg_transfer_maps",
    "compare_surface_corner_skin_weights",
    "derive_lower_body_gate",
    "parse_args",
    "run_sanitation",
    "sanitize_lower_limb_weight_maps",
    "serialize_change_records",
    "validate_lower_limb_bilateral_contamination",
    "validate_published_manifest",
]


if __name__ == "__main__":
    blender_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    raise SystemExit(main(blender_args))
