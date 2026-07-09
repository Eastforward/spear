import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from apartment_builtin_obstacles import (  # noqa: E402
    apartment_builtin_visual_obstacle_bboxes_xy,
    apartment_builtin_visual_obstacle_bboxes_xyz,
)
from dataset_runner import _load_obstacle_context  # noqa: E402
from visibility import frame_visibility  # noqa: E402


def test_builtin_kitchen_counter_blocks_problematic_review_sightline():
    obstacles = apartment_builtin_visual_obstacle_bboxes_xyz()
    visible = frame_visibility(
        src_xyz=(-0.81, -1.60, 0.45),
        mic_pos=(-2.937, -2.721, 1.229),
        mic_yaw_deg=4.74,
        fov_h_deg=90.0,
        fov_v_deg=60.0,
        obstacles_xyz=obstacles,
    )

    assert visible["in_fov"]
    assert visible["occluded_by_furniture"]
    assert not visible["visible"]


def test_builtin_sink_counter_covers_black_cabinet_review_path():
    bboxes = apartment_builtin_visual_obstacle_bboxes_xy()

    assert any(
        x0 <= 0.94 <= x1 and y0 <= 0.24 <= y1
        for x0, y0, x1, y1 in bboxes
    )


def test_dataset_obstacle_context_includes_builtin_visual_blockers():
    spec = json.loads((REPO / "data/apartment_v2_m1_dataset_spec.json").read_text())
    ctx = _load_obstacle_context(spec)
    builtin_xy = apartment_builtin_visual_obstacle_bboxes_xy()
    ctx_xy = [
        (bmin[0], bmin[1], bmax[0], bmax[1])
        for bmin, bmax in ctx["furniture_bboxes"]
    ]

    for bbox in builtin_xy:
        assert bbox in ctx_xy
