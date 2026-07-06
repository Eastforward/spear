"""Unit tests for furniture_map.py: coord conversion + AABB queries."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

import sys
sys.path.insert(0, "/data/jzy/code/SPEAR/tools")

from gpurir_scenes.furniture_map import (
    FurnitureBBox,
    load_apartment_furniture,
    any_bbox_hits_point,
    any_bbox_hits_series,
)


def _write_fake_json(path):
    """Create a tiny fixture JSON with one 100x100 cm sofa at UE origin (0,0)."""
    payload = {
        "meta": {
            "apartment_map_path": "/Game/fake/Maps/fake",
            "dump_date_utc": "2026-07-06T00:00:00Z",
            "spear_commit": "test",
            "ue_version": "5.5",
            "apartment_mic_origin_cm_at_dump": [-120.0, 80.0, 120.0],
            "num_actors_seen": 1,
            "num_actors_after_filter": 1,
            "filter_reasons": {"kept": 1},
        },
        "furniture": [
            {
                "actor_name": "SM_FakeSofa",
                "uclass": "AStaticMeshActor",
                # 100x100x90 cm bbox at UE origin
                "bbox_min_ue_cm": [-50.0, -50.0, 0.0],
                "bbox_max_ue_cm": [50.0, 50.0, 90.0],
                "actor_location_ue_cm": [0.0, 0.0, 45.0],
                "actor_rotation_deg": [0.0, 0.0, 0.0],
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def test_load_produces_one_bbox_with_margin():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _write_fake_json(tf.name)
        bboxes = load_apartment_furniture(json_path=tf.name)
    os.unlink(tf.name)
    assert len(bboxes) == 1
    b = bboxes[0]
    assert b.actor_name == "SM_FakeSofa"
    # Fake bbox spans UE x in [-50, 50] cm. With apartment_mic_origin_cm[0] = -120,
    # scene_m origin at (2.6, 2.2): x_scene = (x_ue - (-120))/100 + 2.6.
    # ⇒ x=-50 UE → (70/100)+2.6 = 3.3 m ; x=50 UE → (170/100)+2.6 = 4.3 m
    # With 0.1 m margin: xy_min_m[0] = 3.2, xy_max_m[0] = 4.4
    assert b.xy_min_m[0] == pytest.approx(3.2, abs=1e-6)
    assert b.xy_max_m[0] == pytest.approx(4.4, abs=1e-6)
    # y with flip: y=-50 UE → -((-50-80)/100)+2.2 = 1.3+2.2 = 3.5; y=50 UE → -((50-80)/100)+2.2 = 0.3+2.2 = 2.5
    # y_flip swaps min<->max ordering; loader must produce xy_min_m[1] < xy_max_m[1].
    assert b.xy_min_m[1] < b.xy_max_m[1]
    assert b.xy_min_m[1] == pytest.approx(2.4, abs=1e-6)   # 2.5 - 0.1 margin
    assert b.xy_max_m[1] == pytest.approx(3.6, abs=1e-6)   # 3.5 + 0.1 margin


def test_any_bbox_hits_point_inside_and_outside():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _write_fake_json(tf.name)
        bboxes = load_apartment_furniture(json_path=tf.name)
    os.unlink(tf.name)
    # Center of fake bbox (in scene m): x ≈ 3.8, y ≈ 3.0
    assert any_bbox_hits_point(bboxes, 3.8, 3.0) is True
    # A point clearly outside: x=0, y=0
    assert any_bbox_hits_point(bboxes, 0.0, 0.0) is False


def test_any_bbox_hits_series_one_frame_inside():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _write_fake_json(tf.name)
        bboxes = load_apartment_furniture(json_path=tf.name)
    os.unlink(tf.name)
    # 3-frame trajectory that passes through the bbox at frame 1
    xy = np.array([[0.0, 0.0], [3.8, 3.0], [10.0, 10.0]])
    assert any_bbox_hits_series(bboxes, xy) is True
    # trajectory that never enters
    xy_clean = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert any_bbox_hits_series(bboxes, xy_clean) is False


def test_empty_bboxes_never_hit():
    xy = np.array([[3.8, 3.0]])
    assert any_bbox_hits_point([], 3.8, 3.0) is False
    assert any_bbox_hits_series([], xy) is False
