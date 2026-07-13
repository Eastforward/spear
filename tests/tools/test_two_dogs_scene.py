"""Tests for the hand-authored two-dog demo scene."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from gpurir_scenes.scene_two_dogs import compose_two_dog_scene


def _animal(spec, tag):
    return next(a for a in spec.animals if a.tag == tag)


def _view0_angle_deg(spec, xy):
    mx, my, _ = spec.mic_pos_m
    dx = xy[0] - mx
    dy = xy[1] - my
    return math.degrees(math.atan2(dx, dy))


def test_static_golden_faces_view0_camera():
    spec = compose_two_dog_scene()
    golden = _animal(spec, "dog_golden")

    assert np.allclose(golden.yaw_deg, 270.0)


def test_two_dog_centers_stay_inside_view0_with_margin():
    spec = compose_two_dog_scene()

    for animal in spec.animals:
        pts = animal.trajectory_m[:, :2]
        angles = [_view0_angle_deg(spec, xy) for xy in pts]
        assert max(abs(a) for a in angles) <= 45.0


def test_beagle_path_keeps_extra_back_wall_clearance():
    spec = compose_two_dog_scene()
    beagle = _animal(spec, "dog_beagle_v2")
    _rx, ry, _rz = spec.room_size_m

    back_wall_slack = ry - float(beagle.trajectory_m[:, 1].max())

    assert back_wall_slack >= 0.70


def test_beagle_path_avoids_apartment_right_wall_corridor():
    spec = compose_two_dog_scene()
    beagle = _animal(spec, "dog_beagle_v2")

    assert float(beagle.trajectory_m[:, 0].max()) <= 2.85
