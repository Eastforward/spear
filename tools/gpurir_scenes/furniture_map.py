"""Load apartment furniture bbox dump, convert UE cm -> SceneSpec m coords,
and expose fast AABB point/series hit queries.

Data file: data/apartment_furniture_map.json (produced by
dump_apartment_furniture.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

# Must match tools/gpurir_scenes/run_render_pass.py::APARTMENT_MIC_ORIGIN_CM.
# If run_render_pass changes it, the loader here needs the same value --
# but only at load-time (not stored in the JSON), so no re-dump required.
DEFAULT_APARTMENT_MIC_ORIGIN_CM = (-120.0, 80.0, 120.0)
DEFAULT_APARTMENT_MIC_POS_SCENE_M = (2.6, 2.2)
DEFAULT_JSON_PATH = "/data/jzy/code/SPEAR/data/apartment_furniture_map.json"


@dataclass(frozen=True)
class FurnitureBBox:
    """AABB in SceneSpec m coordinates, already inflated by margin."""
    actor_name: str
    xy_min_m: tuple  # (x_min, y_min)
    xy_max_m: tuple  # (x_max, y_max)
    z_min_m: float
    z_max_m: float

    def contains_xy(self, x: float, y: float) -> bool:
        return (self.xy_min_m[0] <= x <= self.xy_max_m[0]
                and self.xy_min_m[1] <= y <= self.xy_max_m[1])


def _ue_cm_to_scene_m(x_ue_cm, y_ue_cm, z_ue_cm,
                      mic_origin_cm, mic_pos_scene_m, y_flip):
    """Mirror of run_render_pass._world_from_scene() but inverse direction.

    Forward (run_render_pass): scene_m -> UE cm.
    Here we invert: UE cm -> scene_m.
    """
    x_scene = (x_ue_cm - mic_origin_cm[0]) / 100.0 + mic_pos_scene_m[0]
    if y_flip:
        y_scene = -(y_ue_cm - mic_origin_cm[1]) / 100.0 + mic_pos_scene_m[1]
    else:
        y_scene = (y_ue_cm - mic_origin_cm[1]) / 100.0 + mic_pos_scene_m[1]
    z_scene = z_ue_cm / 100.0
    return x_scene, y_scene, z_scene


def load_apartment_furniture(
    json_path=DEFAULT_JSON_PATH,
    apartment_mic_origin_cm=DEFAULT_APARTMENT_MIC_ORIGIN_CM,
    apartment_mic_pos_scene_m=DEFAULT_APARTMENT_MIC_POS_SCENE_M,
    y_flip=True,
    margin_m=0.1,
):
    """Load JSON, transform UE cm -> SceneSpec m, inflate by margin_m.

    Returns list of FurnitureBBox in scene m coordinates, ready for
    scene_spec.check_no_clipping / compose_scene consumption.
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    bboxes = []
    for rec in data.get("furniture", []):
        bmin_ue = rec["bbox_min_ue_cm"]
        bmax_ue = rec["bbox_max_ue_cm"]
        # Transform each corner independently; y_flip may swap min/max order.
        x1, y1, z1 = _ue_cm_to_scene_m(
            bmin_ue[0], bmin_ue[1], bmin_ue[2],
            apartment_mic_origin_cm, apartment_mic_pos_scene_m, y_flip)
        x2, y2, z2 = _ue_cm_to_scene_m(
            bmax_ue[0], bmax_ue[1], bmax_ue[2],
            apartment_mic_origin_cm, apartment_mic_pos_scene_m, y_flip)
        # Re-order so min < max after possible y-flip.
        xy_min = (min(x1, x2) - margin_m, min(y1, y2) - margin_m)
        xy_max = (max(x1, x2) + margin_m, max(y1, y2) + margin_m)
        z_min = min(z1, z2)
        z_max = max(z1, z2)
        bboxes.append(FurnitureBBox(
            actor_name=rec["actor_name"],
            xy_min_m=xy_min,
            xy_max_m=xy_max,
            z_min_m=z_min,
            z_max_m=z_max,
        ))
    return bboxes


def any_bbox_hits_point(bboxes, x, y):
    for b in bboxes:
        if b.contains_xy(x, y):
            return True
    return False


def any_bbox_hits_series(bboxes, xy_series):
    """xy_series shape (K, 2). Return True if ANY frame in ANY bbox."""
    if len(bboxes) == 0 or xy_series.size == 0:
        return False
    xs = xy_series[:, 0]
    ys = xy_series[:, 1]
    for b in bboxes:
        inside = (
            (xs >= b.xy_min_m[0]) & (xs <= b.xy_max_m[0])
            & (ys >= b.xy_min_m[1]) & (ys <= b.xy_max_m[1])
        )
        if inside.any():
            return True
    return False
