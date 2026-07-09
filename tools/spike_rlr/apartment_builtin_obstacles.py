"""Built-in apartment obstacles missing from the static furniture dump.

The apartment_0000 map contains kitchen counter / island geometry that is
visible in UE but is not present in either apartment_furniture_map.json or
apartment_shell_map.json.  These conservative bboxes keep path planning,
visibility metadata, and review top-down videos aligned with what the camera
actually sees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BuiltinObstacle:
    name: str
    bbox_xy: tuple[float, float, float, float]
    z_min_m: float = 0.0
    z_max_m: float = 1.15
    note: str = ""


_BUILTIN_VISUAL_OBSTACLES: tuple[BuiltinObstacle, ...] = (
    BuiltinObstacle(
        name="kitchen_island_counter",
        bbox_xy=(-3.35, -4.25, -0.35, -1.10),
        note="Long kitchen island / counter line visible in UE view.",
    ),
    BuiltinObstacle(
        name="kitchen_peninsula_counter",
        bbox_xy=(-3.95, -1.25, -2.55, 0.65),
        note="Short peninsula / counter return near the kitchen opening.",
    ),
    BuiltinObstacle(
        name="kitchen_sink_counter",
        bbox_xy=(-1.05, -0.55, 1.55, 0.85),
        note="Black lower cabinets / sink counter visible beside the island.",
    ),
)


def _enabled(spec: dict | None) -> bool:
    if spec is None:
        return True
    return bool(spec.get("apartment_builtin_visual_obstacles", True))


def apartment_builtin_visual_obstacles(
    spec: dict | None = None,
) -> tuple[BuiltinObstacle, ...]:
    """Return manual apartment_0000 built-ins that block walking/visibility."""
    return _BUILTIN_VISUAL_OBSTACLES if _enabled(spec) else ()


def apartment_builtin_visual_obstacle_bboxes_xy(
    spec: dict | None = None,
) -> list[tuple[float, float, float, float]]:
    return [obs.bbox_xy for obs in apartment_builtin_visual_obstacles(spec)]


def apartment_builtin_visual_obstacle_bboxes_xyz(
    spec: dict | None = None,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    out = []
    for obs in apartment_builtin_visual_obstacles(spec):
        x0, y0, x1, y1 = obs.bbox_xy
        out.append(((x0, y0, obs.z_min_m), (x1, y1, obs.z_max_m)))
    return out


def xy_bboxes_to_xyz(
    bboxes_xy: Iterable[tuple[float, float, float, float]],
    z_min_m: float,
    z_max_m: float,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    return [
        ((x0, y0, z_min_m), (x1, y1, z_max_m))
        for x0, y0, x1, y1 in bboxes_xy
    ]
