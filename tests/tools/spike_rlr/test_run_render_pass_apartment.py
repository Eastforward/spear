import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _scene_with_point(tag, xyz):
    traj = np.asarray([xyz], dtype=np.float64)
    return SimpleNamespace(animals=[SimpleNamespace(tag=tag, trajectory_m=traj)])


def test_walls_only_center_collision_policy_allows_furniture_center_points():
    from run_render_pass_apartment import _check_no_clipping_apartment
    from scene_two_dogs_apartment import _kept_furniture_bboxes

    spec = json.loads((REPO / "data" / "apartment_v1_spec.json").read_text())
    cats = json.loads(
        (REPO / "tools/spike_rlr/apartment_furniture_categories.json").read_text()
    )
    x0, y0, x1, y1 = _kept_furniture_bboxes(spec, cats)[0]
    scene = _scene_with_point("dog_beagle_v2", [(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.45])

    with pytest.raises(AssertionError):
        _check_no_clipping_apartment(spec, scene, cats)

    spec["source_collision_policy"] = "walls_only_center"
    _check_no_clipping_apartment(spec, scene, cats)


def test_walls_only_center_collision_policy_still_rejects_wall_points():
    from run_render_pass_apartment import _check_no_clipping_apartment
    from scene_two_dogs_apartment import _shell_wall_bboxes

    spec = json.loads((REPO / "data" / "apartment_v1_spec.json").read_text())
    spec["source_collision_policy"] = "walls_only_center"
    cats = json.loads(
        (REPO / "tools/spike_rlr/apartment_furniture_categories.json").read_text()
    )
    x0, y0, x1, y1 = _shell_wall_bboxes(spec)[0]
    scene = _scene_with_point("dog_beagle_v2", [(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.45])

    with pytest.raises(AssertionError):
        _check_no_clipping_apartment(spec, scene, cats)
