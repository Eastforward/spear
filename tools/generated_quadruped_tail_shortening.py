#!/usr/bin/env python3
"""Pure geometry contract for local shortening of a generated quadruped tail.

The profile supplies a reviewed world-space tail centerline and a narrow
selection corridor.  Points outside that corridor, and points in the retained
tail-root band, are unchanged.  The free shaft is resampled onto an earlier
part of the same centerline, so the operation never introduces template
geometry or changes the rest of the generated body.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


SCHEMA = "avengine_generated_quadruped_local_tail_shortening_v1"
MIN_FREE_TAIL_LENGTH_RATIO = 0.01
MAX_FREE_TAIL_LENGTH_RATIO = 0.5
MIN_RADIAL_THICKNESS_RATIO = 0.25
MAX_RADIAL_THICKNESS_RATIO = 1.0


class TailShorteningError(ValueError):
    """Raised when a tail-shortening profile or point mapping is unsafe."""


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class TailShorteningProfile:
    centerline_world: tuple[Point3, ...]
    cumulative_lengths: tuple[float, ...]
    selection_radius_world: float
    base_keep_arc_length_world: float
    free_tail_length_ratio: float
    radial_thickness_ratio: float
    minimum_moved_vertices: int
    maximum_moved_vertex_fraction: float
    minimum_initial_free_length_world: float
    maximum_final_free_length_world: float

    @property
    def total_arc_length_world(self) -> float:
        return self.cumulative_lengths[-1]

    @property
    def initial_free_length_world(self) -> float:
        return self.total_arc_length_world - self.base_keep_arc_length_world

    @property
    def final_free_length_world(self) -> float:
        return self.initial_free_length_world * self.free_tail_length_ratio


@dataclass(frozen=True)
class PointMapping:
    source_world: Point3
    output_world: Point3
    selected: bool
    moved: bool
    nearest_arc_length_world: float
    nearest_centerline_distance_world: float


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise TailShorteningError(f"{label} must be a finite number")
    return float(value)


def _positive_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise TailShorteningError(f"{label} must be positive")
    return result


def _point(value: Any, label: str) -> Point3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TailShorteningError(f"{label} must be a three-number point")
    if len(value) != 3:
        raise TailShorteningError(f"{label} must be a three-number point")
    return tuple(
        _finite_number(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _subtract(first: Point3, second: Point3) -> Point3:
    return tuple(first[index] - second[index] for index in range(3))  # type: ignore[return-value]


def _add(first: Point3, second: Point3) -> Point3:
    return tuple(first[index] + second[index] for index in range(3))  # type: ignore[return-value]


def _scale(point: Point3, factor: float) -> Point3:
    return tuple(component * factor for component in point)  # type: ignore[return-value]


def _dot(first: Point3, second: Point3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _length(vector: Point3) -> float:
    return math.sqrt(_dot(vector, vector))


def _distance(first: Point3, second: Point3) -> float:
    return _length(_subtract(first, second))


def _profile_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise TailShorteningError(
            f"{label} keys changed: missing={sorted(missing)} extra={sorted(extra)}"
        )


def load_profile(value: Mapping[str, Any]) -> TailShorteningProfile:
    """Validate one fail-closed local shortening profile."""

    if not isinstance(value, Mapping):
        raise TailShorteningError("profile must be an object")
    _profile_keys(
        value,
        {
            "schema",
            "state_classification",
            "formal_dataset_registration_authorized",
            "source",
            "coordinate_space",
            "tail_corridor",
            "shortening",
            "gates",
            "authority",
        },
        "profile",
    )
    if value.get("schema") != SCHEMA:
        raise TailShorteningError("profile schema changed")
    if value.get("state_classification") != "technical_spike_only":
        raise TailShorteningError("tail shortening is technical_spike_only")
    if value.get("formal_dataset_registration_authorized") is not False:
        raise TailShorteningError("tail shortening cannot authorize registration")
    if value.get("coordinate_space") != "blender_world":
        raise TailShorteningError("profile coordinate_space must be blender_world")

    source = value.get("source")
    if not isinstance(source, Mapping):
        raise TailShorteningError("profile source must be an object")
    _profile_keys(source, {"path", "sha256", "size_bytes"}, "profile.source")
    if (
        not isinstance(source.get("path"), str)
        or not source["path"]
        or not isinstance(source.get("sha256"), str)
        or len(source["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in source["sha256"])
        or not isinstance(source.get("size_bytes"), int)
        or isinstance(source["size_bytes"], bool)
        or source["size_bytes"] <= 0
    ):
        raise TailShorteningError("profile source identity is invalid")

    authority = value.get("authority")
    if not isinstance(authority, Mapping):
        raise TailShorteningError("profile authority must be an object")
    _profile_keys(
        authority,
        {
            "generated_mesh_is_geometry_authority",
            "template_geometry_substitution_forbidden",
            "non_tail_vertex_positions_must_remain_exact",
            "mesh_topology_must_remain_exact",
            "uv_and_pbr_must_remain_exact_in_memory",
        },
        "profile.authority",
    )
    if any(authority.get(key) is not True for key in authority):
        raise TailShorteningError("all tail-shortening authority guards must be true")

    corridor = value.get("tail_corridor")
    if not isinstance(corridor, Mapping):
        raise TailShorteningError("profile tail_corridor must be an object")
    _profile_keys(
        corridor,
        {
            "centerline_world",
            "selection_radius_world",
            "base_keep_arc_length_world",
        },
        "profile.tail_corridor",
    )
    raw_centerline = corridor.get("centerline_world")
    if (
        not isinstance(raw_centerline, Sequence)
        or isinstance(raw_centerline, (str, bytes))
        or not 2 <= len(raw_centerline) <= 64
    ):
        raise TailShorteningError("tail centerline needs 2-64 reviewed points")
    centerline = tuple(
        _point(point, f"tail_corridor.centerline_world[{index}]")
        for index, point in enumerate(raw_centerline)
    )
    cumulative = [0.0]
    for index in range(1, len(centerline)):
        segment = _distance(centerline[index - 1], centerline[index])
        if segment <= 1.0e-9:
            raise TailShorteningError("tail centerline contains a degenerate segment")
        cumulative.append(cumulative[-1] + segment)
    total = cumulative[-1]
    radius = _positive_number(
        corridor.get("selection_radius_world"),
        "tail_corridor.selection_radius_world",
    )
    base_keep = _positive_number(
        corridor.get("base_keep_arc_length_world"),
        "tail_corridor.base_keep_arc_length_world",
    )
    if base_keep >= total:
        raise TailShorteningError("tail-root keep length must be below total length")

    shortening = value.get("shortening")
    if not isinstance(shortening, Mapping):
        raise TailShorteningError("profile shortening must be an object")
    _profile_keys(
        shortening,
        {"free_tail_length_ratio", "radial_thickness_ratio"},
        "profile.shortening",
    )
    length_ratio = _finite_number(
        shortening.get("free_tail_length_ratio"),
        "shortening.free_tail_length_ratio",
    )
    thickness_ratio = _finite_number(
        shortening.get("radial_thickness_ratio"),
        "shortening.radial_thickness_ratio",
    )
    if not MIN_FREE_TAIL_LENGTH_RATIO <= length_ratio <= MAX_FREE_TAIL_LENGTH_RATIO:
        raise TailShorteningError(
            "free_tail_length_ratio must be in "
            f"[{MIN_FREE_TAIL_LENGTH_RATIO}, {MAX_FREE_TAIL_LENGTH_RATIO}]"
        )
    if not MIN_RADIAL_THICKNESS_RATIO <= thickness_ratio <= MAX_RADIAL_THICKNESS_RATIO:
        raise TailShorteningError(
            "radial_thickness_ratio must be in "
            f"[{MIN_RADIAL_THICKNESS_RATIO}, {MAX_RADIAL_THICKNESS_RATIO}]"
        )

    gates = value.get("gates")
    if not isinstance(gates, Mapping):
        raise TailShorteningError("profile gates must be an object")
    _profile_keys(
        gates,
        {
            "minimum_moved_vertices",
            "maximum_moved_vertex_fraction",
            "minimum_initial_free_length_world",
            "maximum_final_free_length_world",
        },
        "profile.gates",
    )
    minimum_moved = gates.get("minimum_moved_vertices")
    if (
        not isinstance(minimum_moved, int)
        or isinstance(minimum_moved, bool)
        or minimum_moved < 3
    ):
        raise TailShorteningError("minimum_moved_vertices must be an integer >= 3")
    maximum_fraction = _finite_number(
        gates.get("maximum_moved_vertex_fraction"),
        "gates.maximum_moved_vertex_fraction",
    )
    if not 0.0 < maximum_fraction <= 0.25:
        raise TailShorteningError(
            "maximum_moved_vertex_fraction must be in (0, 0.25]"
        )
    minimum_initial = _positive_number(
        gates.get("minimum_initial_free_length_world"),
        "gates.minimum_initial_free_length_world",
    )
    maximum_final = _positive_number(
        gates.get("maximum_final_free_length_world"),
        "gates.maximum_final_free_length_world",
    )
    initial_free = total - base_keep
    final_free = initial_free * length_ratio
    if initial_free < minimum_initial:
        raise TailShorteningError(
            "reviewed centerline free shaft is below the minimum initial length"
        )
    if final_free > maximum_final:
        raise TailShorteningError(
            "requested free shaft remains above the maximum final length"
        )

    return TailShorteningProfile(
        centerline_world=centerline,
        cumulative_lengths=tuple(cumulative),
        selection_radius_world=radius,
        base_keep_arc_length_world=base_keep,
        free_tail_length_ratio=length_ratio,
        radial_thickness_ratio=thickness_ratio,
        minimum_moved_vertices=minimum_moved,
        maximum_moved_vertex_fraction=maximum_fraction,
        minimum_initial_free_length_world=minimum_initial,
        maximum_final_free_length_world=maximum_final,
    )


def point_at_arc_length(profile: TailShorteningProfile, distance: float) -> Point3:
    distance = min(max(float(distance), 0.0), profile.total_arc_length_world)
    for index in range(len(profile.centerline_world) - 1):
        start_length = profile.cumulative_lengths[index]
        end_length = profile.cumulative_lengths[index + 1]
        if distance <= end_length or index == len(profile.centerline_world) - 2:
            fraction = (distance - start_length) / (end_length - start_length)
            start = profile.centerline_world[index]
            end = profile.centerline_world[index + 1]
            return _add(start, _scale(_subtract(end, start), fraction))
    raise AssertionError("validated centerline has no segment")


def nearest_centerline_projection(
    profile: TailShorteningProfile, point: Sequence[float]
) -> tuple[Point3, float, float]:
    source = _point(point, "point")
    best_point = profile.centerline_world[0]
    best_arc = 0.0
    best_distance_squared = math.inf
    for index in range(len(profile.centerline_world) - 1):
        start = profile.centerline_world[index]
        end = profile.centerline_world[index + 1]
        segment = _subtract(end, start)
        segment_squared = _dot(segment, segment)
        fraction = min(
            max(_dot(_subtract(source, start), segment) / segment_squared, 0.0),
            1.0,
        )
        projected = _add(start, _scale(segment, fraction))
        delta = _subtract(source, projected)
        distance_squared = _dot(delta, delta)
        if distance_squared < best_distance_squared:
            best_point = projected
            best_arc = (
                profile.cumulative_lengths[index]
                + fraction * math.sqrt(segment_squared)
            )
            best_distance_squared = distance_squared
    return best_point, best_arc, math.sqrt(best_distance_squared)


def map_point(
    profile: TailShorteningProfile, point: Sequence[float]
) -> PointMapping:
    source = _point(point, "point")
    nearest, arc_length, distance = nearest_centerline_projection(profile, source)
    selected = distance <= profile.selection_radius_world
    if not selected or arc_length <= profile.base_keep_arc_length_world:
        return PointMapping(
            source_world=source,
            output_world=source,
            selected=selected,
            moved=False,
            nearest_arc_length_world=arc_length,
            nearest_centerline_distance_world=distance,
        )

    free_position = arc_length - profile.base_keep_arc_length_world
    normalized = free_position / profile.initial_free_length_world
    output_arc = (
        profile.base_keep_arc_length_world
        + free_position * profile.free_tail_length_ratio
    )
    output_center = point_at_arc_length(profile, output_arc)
    radial_scale = 1.0 - normalized * (
        1.0 - profile.radial_thickness_ratio
    )
    output = _add(output_center, _scale(_subtract(source, nearest), radial_scale))
    moved = _distance(source, output) > 1.0e-12
    return PointMapping(
        source_world=source,
        output_world=output,
        selected=True,
        moved=moved,
        nearest_arc_length_world=arc_length,
        nearest_centerline_distance_world=distance,
    )
