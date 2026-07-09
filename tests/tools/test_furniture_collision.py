"""V1/V2/V3 acceptance tests for apartment furniture collision (spec 5)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from gpurir_scenes.furniture_map import (
    load_apartment_furniture,
    any_bbox_hits_series,
)
from gpurir_scenes import scene_spec
from gpurir_scenes.scene_spec import (
    SceneSpec, AnimalPlacement, ROOM_SIZE_M, MIC_POS_M, T60_S, SOURCE_HEIGHT_M,
)

FURNITURE_JSON = REPO / "data" / "apartment_furniture_map.json"


def _apartment_json_exists() -> bool:
    return os.path.exists(FURNITURE_JSON)


pytestmark = pytest.mark.skipif(
    not _apartment_json_exists(),
    reason=f"{FURNITURE_JSON} not present -- run Task 1 first",
)


# ---- V1: Dump 数量 sanity ---------------------------------------------------
def test_V1_dump_yields_reasonable_count():
    with open(FURNITURE_JSON) as f:
        data = json.load(f)
    n = len(data["furniture"])
    reasons = data["meta"]["filter_reasons"]
    assert 30 <= n <= 500, (
        f"apartment kept-furniture count = {n} not in [30, 500].\n"
        f"Too few → filter too aggressive; too many → structural mesh leaked.\n"
        f"filter_reasons: {reasons}"
    )


# ---- V2: 回归 - 已知穿模场景必须被拒绝 --------------------------------------
def _static_beagle_spec_at(x: float, y: float) -> SceneSpec:
    traj = np.tile(np.array([x, y, SOURCE_HEIGHT_M]), (75, 1))
    yaw = np.zeros(75)
    return SceneSpec(
        seed=0, room_size_m=ROOM_SIZE_M, t60_s=T60_S, mic_pos_m=MIC_POS_M,
        animals=[AnimalPlacement(
            tag="dog_beagle_v2", is_animated=True,
            trajectory_m=traj, yaw_deg=yaw,
        )],
    )


def test_V2_known_clipping_trajectory_rejected():
    furniture = load_apartment_furniture()
    assert furniture, "V2 requires V1's non-empty dump"
    # Pick the largest-area furniture whose CENTER is fully inside room bounds
    # with some wall margin. Otherwise a huge window/curtain mesh (whose center
    # sits outside the room) would trigger wall-clip before furniture-clip and
    # never exercise the furniture assert we want to test.
    rx, ry, _ = ROOM_SIZE_M
    margin = 0.8  # generous, so wall check is quiet
    candidates = []
    for b in furniture:
        cx = (b.xy_min_m[0] + b.xy_max_m[0]) / 2
        cy = (b.xy_min_m[1] + b.xy_max_m[1]) / 2
        if margin <= cx <= rx - margin and margin <= cy <= ry - margin:
            area = (b.xy_max_m[0] - b.xy_min_m[0]) * (b.xy_max_m[1] - b.xy_min_m[1])
            candidates.append((area, cx, cy, b.actor_name))
    assert candidates, (
        f"no furniture has its center inside room bounds with {margin}m wall margin"
    )
    _, cx, cy, name = max(candidates, key=lambda t: t[0])
    print(f"V2 fixture: using {name} at ({cx:.2f}, {cy:.2f}) as clipping target")
    spec = _static_beagle_spec_at(cx, cy)
    with pytest.raises(AssertionError, match="clips furniture"):
        scene_spec.check_no_clipping(spec, furniture_bboxes=furniture)


# ---- V3: 随机 compose 成功率 >= 80% ------------------------------------------
def test_V3_random_compose_success_rate():
    furniture = load_apartment_furniture()
    successes = 0
    total = 50
    failures = []
    for seed in range(total):
        try:
            spec = scene_spec.compose_scene(seed, furniture_bboxes=furniture)
            # double-check: no animal hits furniture
            for a in spec.animals:
                xy = scene_spec._placement_xy_series(a)
                if any_bbox_hits_series(furniture, xy):
                    raise AssertionError(f"seed {seed}: {a.tag} hits furniture")
            successes += 1
        except (AssertionError, RuntimeError) as e:
            failures.append((seed, str(e)))
    ratio = successes / total
    msg = (f"apartment compose success rate = {successes}/{total} = {ratio:.0%}. "
           f"First 3 failures: {failures[:3]}")
    assert ratio >= 0.80, msg
