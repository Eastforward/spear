"""Actor-name based classifier: shell (structural) vs furniture (independent).

Shell = anything attached to walls/floor/ceiling (walls, floor, ceiling,
doors, windows, curtains, pictures, mirrors). These don't occupy interior
free space and stay across furniture-mode toggles.

Furniture = independent items occupying floor area (chairs, sofa, tables,
lamps, pillows, etc.). These are the items subset/full modes toggle.
"""
from __future__ import annotations

Z_CEILING_CM = 300.0
Z_FLOOR_CM = 5.0
BBOX_AREA_MAX_CM2 = 200000.0

_SHELL_NAME_KEYWORDS = {
    "wall": "shell_wall",
    "floor": "shell_floor",
    "ceiling": "shell_ceiling",
    "ground": "shell_floor",
    "door": "shell_door",
    "window": "shell_window",
    "curtain": "shell_curtain",
    "picture": "shell_picture",
    "mirror": "shell_mirror",
}

SHELL_LABELS = frozenset({
    "shell_ceiling", "shell_floor", "shell_wall",
    "shell_door", "shell_window", "shell_curtain",
    "shell_picture", "shell_mirror",
    "structural",
})


def classify_actor(actor_name: str, bbox_min_z: float, bbox_max_z: float,
                   x_extent_cm: float, y_extent_cm: float) -> str:
    """Return one of SHELL_LABELS or 'furniture'."""
    # Ceiling by z-min
    if bbox_min_z > Z_CEILING_CM:
        return "shell_ceiling"
    # Floor by z-max
    if bbox_max_z < Z_FLOOR_CM:
        return "shell_floor"
    # Name-based shell classification (walls, doors, windows, curtains, pictures, mirrors)
    nl = str(actor_name).lower()
    for kw, label in _SHELL_NAME_KEYWORDS.items():
        if kw in nl:
            return label
    # Very large bbox -> structural mesh not caught by name (e.g. big built-in cabinet
    # merged with a wall). Threshold from the original furniture-only dumper.
    if x_extent_cm * y_extent_cm > BBOX_AREA_MAX_CM2:
        return "structural"
    return "furniture"
