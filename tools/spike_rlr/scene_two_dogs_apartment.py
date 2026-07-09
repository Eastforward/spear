"""Hand-authored two-dog scene composer for apartment_v1_spec.json.

Sibling of scene_two_dogs_v2.py, but for the apartment_v1 spec:
  - Both sources use motion == "linear_uniform" (single start->end line).
  - Body yaw is derived from motion direction + per-rig forward offset,
    identical to the shoebox v2 pipeline.
  - Returns a SceneSpec (same dataclass as v1/v2) with 2 AnimalPlacements.

All numbers come from apartment_v1_spec.json (SSOT). Nothing hardcoded here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SPEAR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SPEAR_ROOT / "tools"))
sys.path.insert(0, str(_SPEAR_ROOT / "tools" / "spike_rlr"))

from gpurir_scenes.scene_spec import (  # noqa: E402
    AnimalPlacement,
    SceneSpec,
)
from species_rig_map import ANIMATED_RIG_MAP  # noqa: E402
from scene_two_dogs_v2 import (  # noqa: E402
    _linear_between, _motion_yaw_from_trajectory,
)
from path_planner import plan_path_2d  # noqa: E402
from apartment_builtin_obstacles import (  # noqa: E402
    apartment_builtin_visual_obstacle_bboxes_xy,
)

DEFAULT_SPEC_PATH = _SPEAR_ROOT / "data" / "apartment_v1_spec.json"

# Apartment UE-to-SSOT conversion (matches gen_mesh_apartment.py).
APARTMENT_MIC_ORIGIN_UE_CM = (-120.0, 80.0, 120.0)
APARTMENT_FLOOR_Z_UE_CM = 27.1


def _ue_to_ssot_xy(ue_min, ue_max):
    """Convert UE-cm bbox min/max to SSOT-m (x0, y0, x1, y1)."""
    x0 = (ue_min[0] - APARTMENT_MIC_ORIGIN_UE_CM[0]) / 100.0
    x1 = (ue_max[0] - APARTMENT_MIC_ORIGIN_UE_CM[0]) / 100.0
    y0 = -(ue_min[1] - APARTMENT_MIC_ORIGIN_UE_CM[1]) / 100.0
    y1 = -(ue_max[1] - APARTMENT_MIC_ORIGIN_UE_CM[1]) / 100.0
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _forward_yaw_offset_for_tag(tag):
    return ANIMATED_RIG_MAP[tag]["walking_forward_yaw_offset_deg"]


def _forward_yaw_offset_for_source(src):
    if src.get("walking_forward_yaw_offset_deg") is not None:
        return float(src["walking_forward_yaw_offset_deg"])
    return _forward_yaw_offset_for_tag(src["tag"])


def _kept_furniture_bboxes(spec, cats):
    """Return list of (x0, y0, x1, y1) XY-rectangles for furniture kept in
    this clip's furniture_mode."""
    mode = spec["furniture_mode"]
    include_cats = set(spec.get("furniture_include_categories", []))
    keep = set()
    if mode == "full":
        for c in ("core", "decoration", "misc"):
            keep.update(cats.get(c, []))
    elif mode == "subset":
        for c in include_cats:
            keep.update(cats.get(c, []))
    # mode == "shell" -> keep stays empty
    keep.update(spec.get("furniture_include_actors_extra", []))
    keep.difference_update(spec.get("furniture_exclude_actors", []))

    furn_map = json.loads((_SPEAR_ROOT / spec["apartment_furniture_map"]).read_text())
    out = []
    for f in furn_map["furniture"]:
        if f["actor_name"] not in keep:
            continue
        out.append(_ue_to_ssot_xy(f["bbox_min_ue_cm"], f["bbox_max_ue_cm"]))
    return out


def _shell_wall_bboxes(spec):
    """Return XY-rectangles for shell walls/doors/curtains (things a dog
    physically cannot pass through). Windows are treated as walls too."""
    shell_map = json.loads((_SPEAR_ROOT / spec["apartment_shell_map"]).read_text())
    out = []
    for a in shell_map["shell_actors"]:
        if a["shell_label"] in ("shell_floor", "shell_ceiling"):
            continue
        out.append(_ue_to_ssot_xy(a["bbox_min_ue_cm"], a["bbox_max_ue_cm"]))
    return out


def _static_obstacle_bboxes(spec, cats):
    """Return all 2D obstacles that should block apartment source motion."""
    return (
        _kept_furniture_bboxes(spec, cats)
        + _shell_wall_bboxes(spec)
        + apartment_builtin_visual_obstacle_bboxes_xy(spec)
    )


def _valid_region_bboxes(spec):
    """Return coarse indoor walkable rectangles for apartment_v1.

    The apartment shell is L-shaped, but the raw shell extent is one large
    rectangle that includes the outdoor/top-right void. These two rectangles
    approximate the actual indoor union:
      - left/tall room
      - lower/right room

    Walls remain hard obstacles elsewhere; furniture remains a softer planning
    obstacle, not a dog-radius hard clearance rule.
    """
    shell_map = json.loads((_SPEAR_ROOT / spec["apartment_shell_map"]).read_text())
    rects = []
    for a in shell_map["shell_actors"]:
        if a["shell_label"] in ("shell_floor", "shell_ceiling"):
            continue
        rects.append((*_ue_to_ssot_xy(a["bbox_min_ue_cm"], a["bbox_max_ue_cm"]),
                      a["shell_label"]))
    geom = [(x0, y0, x1, y1) for x0, y0, x1, y1, _ in rects]
    min_x = min(r[0] for r in geom)
    max_x = max(r[2] for r in geom)
    min_y = min(r[1] for r in geom)
    max_y = max(r[3] for r in geom)

    vertical = [r for r in geom if (r[3] - r[1]) > 3.0 and (r[2] - r[0]) < 0.8]
    horizontal = [r for r in geom if (r[2] - r[0]) > 3.0 and (r[3] - r[1]) < 0.8]

    left_wall = min(vertical, key=lambda r: r[0])
    bottom_wall = min(horizontal, key=lambda r: r[1])
    top_wall = max(horizontal, key=lambda r: r[3])
    right_boundary = max(vertical, key=lambda r: r[2])

    inner_verticals = [
        r for r in vertical
        if r is not left_wall and r[0] > left_wall[2] + 1.0
        and r[2] < right_boundary[0] - 1.0
    ]
    inner_horizontals = [
        r for r in horizontal
        if r is not bottom_wall and r is not top_wall
        and r[1] > bottom_wall[3] + 1.0
        and r[3] < top_wall[1] - 1.0
    ]
    if not inner_verticals or not inner_horizontals:
        return [(min_x, min_y, max_x, max_y)]

    inner_v = max(inner_verticals, key=lambda r: r[3] - r[1])
    inner_h = max(inner_horizontals, key=lambda r: r[2] - r[0])

    left_inner_x = left_wall[2]
    bottom_inner_y = bottom_wall[3]
    top_inner_y = top_wall[1]
    right_inner_x = right_boundary[0]
    corner_x = inner_v[0]
    corner_y = inner_h[1]

    return [
        (left_inner_x, bottom_inner_y, corner_x, top_inner_y),
        (left_inner_x, bottom_inner_y, right_inner_x, corner_y),
    ]


def _planning_bounds(spec):
    """Return (x_min, y_min, x_max, y_max) for the planner from shell extent."""
    shell_map = json.loads((_SPEAR_ROOT / spec["apartment_shell_map"]).read_text())
    xs, ys = [], []
    for a in shell_map["shell_actors"]:
        x0, y0, x1, y1 = _ue_to_ssot_xy(a["bbox_min_ue_cm"], a["bbox_max_ue_cm"])
        xs += [x0, x1]; ys += [y0, y1]
    # A small margin inside the shell so planner can hug corners
    return (min(xs) + 0.1, min(ys) + 0.1, max(xs) - 0.1, max(ys) - 0.1)


def _build_planned_trajectory(src, spec, cats, n_frames):
    """Plan a smooth start->end trajectory that avoids kept furniture + walls."""
    start = np.asarray(src["start_pos_m"], dtype=np.float64)
    end = np.asarray(src["end_pos_m"], dtype=np.float64)
    obstacles = _static_obstacle_bboxes(spec, cats)
    bounds = _planning_bounds(spec)
    z = float(src.get("start_pos_m", [0, 0, 0.45])[2])
    return plan_path_2d(
        start_xy=(start[0], start[1]),
        end_xy=(end[0], end[1]),
        obstacles_xy=obstacles,
        bounds_xy=bounds,
        cell_m=0.15,
        inflate_m=0.15,   # 15 cm safety margin around obstacles
        n_frames=n_frames,
        chaikin_iters=2,
        z_m=z,
        valid_xy_rects=_valid_region_bboxes(spec),
    )


def _wanted_anim_for_source(src):
    if src.get("wanted_anim"):
        return src["wanted_anim"]
    motion = src.get("motion_style", src.get("motion", ""))
    return "Idle" if motion == "stationary" else "Walking"


_N_FRAMES = None  # set inside compose


def compose_two_dog_scene_apartment(spec_path=DEFAULT_SPEC_PATH):
    """Build the apartment_v1 SceneSpec (2 animated dogs: golden + husky)."""
    global _N_FRAMES
    with open(spec_path) as f:
        spec = json.load(f)

    assert spec["spec_version"] == "apartment_v1", (
        f"scene_two_dogs_apartment requires spec_version apartment_v1, "
        f"got {spec['spec_version']}"
    )
    n_frames = int(spec["render_config"]["n_frames"])
    _N_FRAMES = n_frames

    # Load per-actor category classification (Plan-1 furniture: core / decoration / misc)
    cats = json.loads((_SPEAR_ROOT / "tools" / "spike_rlr"
                        / "apartment_furniture_categories.json").read_text())

    animals = []
    for src in spec["sources"]:
        tag = src["tag"]
        # Try planned (obstacle-avoiding) path first; fall back to straight line
        # ONLY if the spec explicitly opts out (motion == "linear_uniform_raw").
        motion = src.get("motion", "linear_uniform")
        if "trajectory_m" in src:
            traj = np.asarray(src["trajectory_m"], dtype=np.float64)
            assert traj.shape == (n_frames, 3), (
                f"{tag} trajectory_m must be ({n_frames}, 3), got {traj.shape}"
            )
        elif motion == "linear_uniform_raw":
            traj = _linear_between(
                np.asarray(src["start_pos_m"], dtype=np.float64),
                np.asarray(src["end_pos_m"], dtype=np.float64),
                n_frames,
            )
        else:
            traj = _build_planned_trajectory(src, spec, cats, n_frames)
        offset = _forward_yaw_offset_for_source(src)
        if "facing_yaw_deg" in src:
            yaw = np.full(
                n_frames,
                (float(src["facing_yaw_deg"]) + offset) % 360.0,
            )
        else:
            motion_yaw = _motion_yaw_from_trajectory(traj)
            yaw = (motion_yaw + offset) % 360.0
        animals.append(AnimalPlacement(
            tag=tag,
            is_animated=True,
            trajectory_m=traj,
            yaw_deg=yaw,
            wanted_anim=_wanted_anim_for_source(src),
            actor_scale=src.get("actor_scale"),
            actor_z_lift_cm=src.get("actor_z_lift_cm"),
        ))

    # For the apartment spec there's no "room_size_m" (non-rectangular shell);
    # we pass a large bounding box for downstream code that only uses it as
    # a sanity check. Actual geometry lives in apartment_shell_map.json.
    return SceneSpec(
        seed=20260707,
        room_size_m=(12.0, 14.0, 3.0),  # loose bounding of apartment_0000 extent
        t60_s=0.7,   # rough apartment estimate; real value computed via RLR IR
        mic_pos_m=tuple(spec["mic"]["pos_m"]),
        animals=animals,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(DEFAULT_SPEC_PATH))
    args = ap.parse_args()
    scene = compose_two_dog_scene_apartment(args.spec)
    print(f"animals: {len(scene.animals)}")
    for a in scene.animals:
        t = a.trajectory_m
        print(f"  {a.tag}: {len(t)} frames, "
              f"pos[0]={t[0].tolist()}, pos[end]={t[-1].tolist()}, "
              f"yaw range [{a.yaw_deg.min():.1f}, {a.yaw_deg.max():.1f}]")


if __name__ == "__main__":
    main()
