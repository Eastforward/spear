"""Strict, bone-name-independent planning for render-only quadruped guides.

This module deliberately has no Blender dependency.  It validates a small JSON
contract and turns inferred quadruped semantics plus rest-bone records into
target bone segments.  The Blender renderer is responsible only for applying
those targets to the imported in-memory pose; it never exports or rewrites the
source GLB.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


MORPHOTYPE_GUIDE_SCHEMA = "avengine_quadruped_morphotype_guide_v1"
MIN_LEG_LENGTH_RATIO = 0.55
# Keep every inferred tail segment non-degenerate while admitting a bounded
# render-only stump guide.  Zero is deliberately forbidden because Blender
# cannot construct a pose basis for a zero-length target segment.
MIN_TAIL_LENGTH_RATIO = 0.05
MAX_LENGTH_RATIO = 1.0
MAX_GROUND_RESIDUAL_HEIGHT_RATIO = 0.01
MAX_FOOT_GROUND_SPREAD_HEIGHT_RATIO = 0.01
MAX_BODY_DROP_HEIGHT_RATIO = 0.25
MAX_EFFECTIVE_LEG_RATIO_ERROR = 0.05
MIN_LIMB_REACH_HEIGHT_RATIO = 0.10
MIN_BONE_LENGTH_HEIGHT_RATIO = 1.0e-7


class MorphotypeGuideError(ValueError):
    """Raised when a guide profile or inferred rig cannot be used safely."""


@dataclass(frozen=True)
class MorphotypeGuideProfile:
    leg_length_ratio: float
    tail_length_ratio: float
    maximum_ground_residual_height_ratio: float


@dataclass(frozen=True)
class SegmentTarget:
    head_world: tuple[float, float, float]
    tail_world: tuple[float, float, float]


@dataclass(frozen=True)
class MorphotypeGuidePlan:
    targets: Mapping[str, SegmentTarget]
    limb_chains: Mapping[str, tuple[str, ...]]
    body_bones: tuple[str, ...]
    tail_chain: tuple[str, ...]
    foot_leaves: tuple[str, ...]
    body_drop: float
    body_drop_height_ratio: float
    effective_leg_length_ratios: Mapping[str, float]
    source_tail_length: float
    target_tail_length: float
    maximum_torso_segment_scale_error: float
    maximum_target_foot_displacement_height_ratio: float
    source_foot_ground_spread_height_ratio: float


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MorphotypeGuideError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_keys(value, expected, label):
    if not isinstance(value, Mapping):
        raise MorphotypeGuideError(f"{label} must be an object")
    expected = set(expected)
    actual = set(value)
    if actual != expected:
        raise MorphotypeGuideError(
            f"{label} keys must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MorphotypeGuideError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MorphotypeGuideError(f"{label} must be a finite number")
    return result


def parse_morphotype_guide_profile(document) -> MorphotypeGuideProfile:
    """Validate the complete v1 profile without accepting implicit defaults."""

    _exact_keys(document, {"schema", "transforms", "validation"}, "profile")
    if document["schema"] != MORPHOTYPE_GUIDE_SCHEMA:
        raise MorphotypeGuideError(
            f"profile.schema must be {MORPHOTYPE_GUIDE_SCHEMA!r}"
        )
    transforms = document["transforms"]
    validation = document["validation"]
    _exact_keys(
        transforms,
        {"leg_length_ratio", "tail_length_ratio"},
        "profile.transforms",
    )
    _exact_keys(
        validation,
        {"maximum_ground_residual_height_ratio"},
        "profile.validation",
    )
    leg_ratio = _finite_number(
        transforms["leg_length_ratio"],
        "profile.transforms.leg_length_ratio",
    )
    tail_ratio = _finite_number(
        transforms["tail_length_ratio"],
        "profile.transforms.tail_length_ratio",
    )
    ground_ratio = _finite_number(
        validation["maximum_ground_residual_height_ratio"],
        "profile.validation.maximum_ground_residual_height_ratio",
    )
    if not MIN_LEG_LENGTH_RATIO <= leg_ratio <= MAX_LENGTH_RATIO:
        raise MorphotypeGuideError(
            "profile.transforms.leg_length_ratio must be in "
            f"[{MIN_LEG_LENGTH_RATIO}, {MAX_LENGTH_RATIO}]"
        )
    if not MIN_TAIL_LENGTH_RATIO <= tail_ratio <= MAX_LENGTH_RATIO:
        raise MorphotypeGuideError(
            "profile.transforms.tail_length_ratio must be in "
            f"[{MIN_TAIL_LENGTH_RATIO}, {MAX_LENGTH_RATIO}]"
        )
    if leg_ratio == 1.0 and tail_ratio == 1.0:
        raise MorphotypeGuideError("morphotype guide profile cannot be a no-op")
    if not 0.0 < ground_ratio <= MAX_GROUND_RESIDUAL_HEIGHT_RATIO:
        raise MorphotypeGuideError(
            "profile.validation.maximum_ground_residual_height_ratio must be in "
            f"(0, {MAX_GROUND_RESIDUAL_HEIGHT_RATIO}]"
        )
    return MorphotypeGuideProfile(
        leg_length_ratio=leg_ratio,
        tail_length_ratio=tail_ratio,
        maximum_ground_residual_height_ratio=ground_ratio,
    )


def load_morphotype_guide_profile(path) -> MorphotypeGuideProfile:
    """Load a strict JSON profile and reject duplicate keys and malformed data."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise MorphotypeGuideError(f"morphotype guide profile is not a file: {path}")
    try:
        document = json.loads(
            profile_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MorphotypeGuideError(
            f"cannot read morphotype guide profile {path}: {error}"
        ) from error
    return parse_morphotype_guide_profile(document)


def _point(record, key):
    value = record.get(key)
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise MorphotypeGuideError(
            f"bone {record.get('name')!r} has invalid {key}"
        )
    point = tuple(
        _finite_number(component, f"bone {record.get('name')!r} {key}")
        for component in value
    )
    return point


def _subtract(left, right):
    return tuple(left[index] - right[index] for index in range(3))


def _add(left, right):
    return tuple(left[index] + right[index] for index in range(3))


def _scale(vector, ratio):
    return tuple(component * ratio for component in vector)


def _distance(first, second):
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def _translate_z(point, delta):
    return (point[0], point[1], point[2] + delta)


def _validate_chain(chain, label, by_name, *, minimum_bones=1):
    if isinstance(chain, (str, bytes)) or not isinstance(chain, Sequence):
        raise MorphotypeGuideError(f"{label} must be a bone chain")
    chain = tuple(chain)
    if len(chain) < minimum_bones:
        raise MorphotypeGuideError(
            f"{label} must contain at least {minimum_bones} bone"
            f"{'s' if minimum_bones != 1 else ''}"
        )
    if len(set(chain)) != len(chain):
        raise MorphotypeGuideError(f"{label} repeats a bone")
    missing = [name for name in chain if name not in by_name]
    if missing:
        raise MorphotypeGuideError(f"{label} references missing bones: {missing}")
    for parent, child in zip(chain, chain[1:]):
        if by_name[child].get("parent") != parent:
            raise MorphotypeGuideError(
                f"{label} is not parent-contiguous at {parent!r} -> {child!r}"
            )
    return chain


def _segment_length(record):
    return _distance(_point(record, "head_world"), _point(record, "tail_world"))


def _target_segment_length(target):
    return _distance(target.head_world, target.tail_world)


def build_morphotype_guide_plan(
    profile: MorphotypeGuideProfile,
    semantics,
    records,
    *,
    bbox_height,
) -> MorphotypeGuidePlan:
    """Build bounded pose targets from geometry-inferred quadruped semantics."""

    height = _finite_number(bbox_height, "bbox_height")
    if height <= 0.0:
        raise MorphotypeGuideError("bbox_height must be positive")
    records = [dict(record) for record in records]
    by_name = {record.get("name"): record for record in records}
    if None in by_name or len(by_name) != len(records):
        raise MorphotypeGuideError("bone names must be present and unique")

    axial = _validate_chain(semantics.axial, "axial chain", by_name)
    head = _validate_chain(semantics.head_chain, "head chain", by_name)
    if by_name[head[0]].get("parent") != axial[-1]:
        raise MorphotypeGuideError("head chain does not attach to the axial chain")
    tail = _validate_chain(semantics.tail_chain, "tail chain", by_name)
    if by_name[tail[0]].get("parent") not in set(axial):
        raise MorphotypeGuideError("tail chain does not attach to the axial chain")

    limb_chains = {
        "front_side_negative": _validate_chain(
            semantics.front_side_negative,
            "front-side-negative limb chain",
            by_name,
            minimum_bones=2,
        ),
        "front_side_positive": _validate_chain(
            semantics.front_side_positive,
            "front-side-positive limb chain",
            by_name,
            minimum_bones=2,
        ),
        "hind_side_negative": _validate_chain(
            semantics.hind_side_negative,
            "hind-side-negative limb chain",
            by_name,
            minimum_bones=2,
        ),
        "hind_side_positive": _validate_chain(
            semantics.hind_side_positive,
            "hind-side-positive limb chain",
            by_name,
            minimum_bones=2,
        ),
    }
    limb_bones = [name for chain in limb_chains.values() for name in chain]
    if len(limb_bones) != len(set(limb_bones)):
        raise MorphotypeGuideError("the four limb chains must be bone-disjoint")
    axial_set = set(axial)
    for label, chain in limb_chains.items():
        if by_name[chain[0]].get("parent") not in axial_set:
            raise MorphotypeGuideError(f"{label} does not attach to the axial chain")
    foot_leaves = tuple(chain[-1] for chain in limb_chains.values())
    inferred_feet = tuple(semantics.foot_leaves)
    if len(set(inferred_feet)) != 4 or set(foot_leaves) != set(inferred_feet):
        raise MorphotypeGuideError(
            "four limb endpoints must exactly match the four inferred foot leaves"
        )
    core_groups = (set(axial) | set(head), set(tail), set(limb_bones))
    if any(
        core_groups[left] & core_groups[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise MorphotypeGuideError("body, tail, and limb chains must be disjoint")

    minimum_bone_length = MIN_BONE_LENGTH_HEIGHT_RATIO * height
    transformed_bones = set().union(*core_groups)
    for name in transformed_bones:
        if _segment_length(by_name[name]) <= minimum_bone_length:
            raise MorphotypeGuideError(
                f"bone {name!r} has degenerate render-guide segment length"
            )

    foot_points = {
        label: _point(by_name[chain[-1]], "head_world")
        for label, chain in limb_chains.items()
    }
    foot_z_values = [point[2] for point in foot_points.values()]
    foot_spread_ratio = (max(foot_z_values) - min(foot_z_values)) / height
    if foot_spread_ratio > MAX_FOOT_GROUND_SPREAD_HEIGHT_RATIO:
        raise MorphotypeGuideError(
            "inferred feet are not on one ground plane: "
            f"spread_height_ratio={foot_spread_ratio:.9f}"
        )

    reaches = {}
    for label, chain in limb_chains.items():
        attachment_z = _point(by_name[chain[0]], "head_world")[2]
        reach = attachment_z - foot_points[label][2]
        if reach < MIN_LIMB_REACH_HEIGHT_RATIO * height:
            raise MorphotypeGuideError(
                f"{label} vertical reach is too small for bounded shortening"
            )
        reaches[label] = reach
    mean_reach = sum(reaches.values()) / len(reaches)
    body_drop = (1.0 - profile.leg_length_ratio) * mean_reach
    body_drop_ratio = body_drop / height
    if not 0.0 <= body_drop_ratio <= MAX_BODY_DROP_HEIGHT_RATIO:
        raise MorphotypeGuideError(
            "requested leg shortening exceeds the body-drop bound: "
            f"body_drop_height_ratio={body_drop_ratio:.9f}"
        )
    effective_ratios = {
        label: (reach - body_drop) / reach
        for label, reach in reaches.items()
    }
    for label, ratio in effective_ratios.items():
        ratio_error = abs(ratio - profile.leg_length_ratio)
        if ratio <= 0.0 or ratio_error > MAX_EFFECTIVE_LEG_RATIO_ERROR:
            raise MorphotypeGuideError(
                f"{label} cannot satisfy one rigid torso drop at the requested "
                f"leg ratio: effective={ratio:.9f} "
                f"requested={profile.leg_length_ratio:.9f}"
            )

    targets = {}
    body_bones = tuple(dict.fromkeys(axial + head))
    for name in body_bones:
        record = by_name[name]
        targets[name] = SegmentTarget(
            head_world=_translate_z(_point(record, "head_world"), -body_drop),
            tail_world=_translate_z(_point(record, "tail_world"), -body_drop),
        )

    for label, chain in limb_chains.items():
        foot_z = foot_points[label][2]
        reach = reaches[label]

        def shorten_leg_point(point):
            fraction = min(1.0, max(0.0, (point[2] - foot_z) / reach))
            return (point[0], point[1], point[2] - body_drop * fraction)

        for name in chain:
            record = by_name[name]
            targets[name] = SegmentTarget(
                head_world=shorten_leg_point(_point(record, "head_world")),
                tail_world=shorten_leg_point(_point(record, "tail_world")),
            )

    tail_root = _translate_z(_point(by_name[tail[0]], "head_world"), -body_drop)
    for name in tail:
        record = by_name[name]
        shifted_head = _translate_z(_point(record, "head_world"), -body_drop)
        shifted_tail = _translate_z(_point(record, "tail_world"), -body_drop)
        targets[name] = SegmentTarget(
            head_world=_add(
                tail_root,
                _scale(_subtract(shifted_head, tail_root), profile.tail_length_ratio),
            ),
            tail_world=_add(
                tail_root,
                _scale(_subtract(shifted_tail, tail_root), profile.tail_length_ratio),
            ),
        )

    maximum_torso_scale_error = max(
        abs(
            _target_segment_length(targets[name]) / _segment_length(by_name[name])
            - 1.0
        )
        for name in body_bones
    )
    source_tail_length = sum(_segment_length(by_name[name]) for name in tail)
    target_tail_length = sum(_target_segment_length(targets[name]) for name in tail)
    target_foot_displacements = [
        _distance(
            targets[chain[-1]].head_world,
            foot_points[label],
        )
        for label, chain in limb_chains.items()
    ]
    maximum_target_foot_displacement_ratio = (
        max(target_foot_displacements) / height
    )
    if maximum_torso_scale_error > 1.0e-9:
        raise MorphotypeGuideError(
            "torso/head target is not a rigid translation: "
            f"maximum_segment_scale_error={maximum_torso_scale_error:.12g}"
        )
    if maximum_target_foot_displacement_ratio > 1.0e-9:
        raise MorphotypeGuideError(
            "leg target moved an inferred foot: "
            f"maximum_displacement_height_ratio="
            f"{maximum_target_foot_displacement_ratio:.12g}"
        )
    realized_tail_ratio = target_tail_length / source_tail_length
    if abs(realized_tail_ratio - profile.tail_length_ratio) > 1.0e-9:
        raise MorphotypeGuideError(
            "tail target ratio drifted from the profile: "
            f"realized={realized_tail_ratio:.12g}"
        )

    return MorphotypeGuidePlan(
        targets=targets,
        limb_chains=limb_chains,
        body_bones=body_bones,
        tail_chain=tail,
        foot_leaves=foot_leaves,
        body_drop=body_drop,
        body_drop_height_ratio=body_drop_ratio,
        effective_leg_length_ratios=effective_ratios,
        source_tail_length=source_tail_length,
        target_tail_length=target_tail_length,
        maximum_torso_segment_scale_error=maximum_torso_scale_error,
        maximum_target_foot_displacement_height_ratio=(
            maximum_target_foot_displacement_ratio
        ),
        source_foot_ground_spread_height_ratio=foot_spread_ratio,
    )
