"""Regression tests for room world<->UE conventions.

The two rooms we support so far (shoebox, apartment) have different
position/rotation transforms. This test asserts they stay internally
consistent. When Kujiale rooms are added in Plan 3, extend with more
room parameters.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "gpurir_scenes"))
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
sys.path.insert(0, str(REPO / "tools"))


def test_apartment_yaw_world_to_ue_is_negation():
    """apartment convention: UE yaw = -world yaw (due to Y-flip)."""
    from run_render_pass import _yaw_world_to_ue
    assert _yaw_world_to_ue(0.0, "apartment") == -0.0
    assert _yaw_world_to_ue(90.0, "apartment") == -90.0
    assert _yaw_world_to_ue(180.0, "apartment") == -180.0


def test_shoebox_yaw_world_to_ue_is_identity():
    """shoebox convention: UE yaw = world yaw (no flip)."""
    from run_render_pass import _yaw_world_to_ue
    assert _yaw_world_to_ue(0.0, "shoebox") == 0.0
    assert _yaw_world_to_ue(90.0, "shoebox") == 90.0
    assert _yaw_world_to_ue(180.0, "shoebox") == 180.0


def test_apartment_position_and_rotation_are_consistent():
    """Consistency: if position uses Y-flip in apartment, then a source at
    world (0, +1, 0) should map to UE (0, -1, 0) in cm, AND yaw pointing
    at that source in world (+90 = +Y) should map to UE yaw -90 (=+Y_UE
    after Y-flip)."""
    from run_render_pass import _world_from_scene, _yaw_world_to_ue, APARTMENT_MIC_ORIGIN_CM

    # Fake spec object with mic_pos_m attribute (SceneSpec dataclass)
    class FakeSpec:
        mic_pos_m = (0.0, 0.0, 1.2)

    # Source at world (0, +1, 0). SSOT convention: world +Y is one direction.
    world_pos = (0.0, 1.0, 0.0)
    ue_pos = _world_from_scene(world_pos, room="apartment", spec=FakeSpec(),
                                actor_z_lift_cm=0.0)
    # Apartment origin is APARTMENT_MIC_ORIGIN_CM; dy_cm = -(1.0 - 0.0)*100 = -100
    # So UE Y should be APARTMENT_MIC_ORIGIN_CM[1] - 100
    expected_ue_y = APARTMENT_MIC_ORIGIN_CM[1] - 100.0
    assert ue_pos[1] == pytest.approx(expected_ue_y, abs=0.1), \
        f"apartment world +Y expected UE Y={expected_ue_y} (after mic anchor + flip), got {ue_pos}"

    # And yaw pointing at world +Y (yaw=90 world) should be UE yaw=-90
    yaw_ue = _yaw_world_to_ue(90.0, "apartment")
    assert yaw_ue == pytest.approx(-90.0, abs=0.1), \
        f"apartment world yaw +90 expected UE yaw -90, got {yaw_ue}"

    # Consistency check: after position flip AND rotation flip, a source that
    # is 'in front of' the mic in world (world +Y at yaw 90) should still be
    # 'in front of' the mic in UE (UE -Y at yaw -90 → UE-forward is -Y_UE).
    # Both flipped, so directionality preserved.


def test_apartment_per_clip_renderer_uses_absolute_ssot_for_random_mic():
    """Plan-2 apartment clips store mic and source positions in the same
    absolute SSOT apartment frame. The UE renderer must therefore place both
    camera and actors through the same absolute SSOT->UE transform; otherwise
    a source that metadata/topdown says is in front of the camera can be
    rendered behind it.
    """
    import math

    from run_render_pass import _world_from_scene, _yaw_world_to_ue
    from run_render_pass_apartment import (
        _apartment_camera_ue_cm,
        _absolute_apartment_render_scene,
    )
    from gpurir_scenes.scene_spec import SceneSpec

    mic_pos_m = (0.6035292184577328, -3.6151139684246543, 0.8056184070759604)
    mic_yaw_deg = 260.2961920851162
    # clip_0003 frame 35: topdown/metadata place this near the center of view.
    source_pos_m = (0.4649450621971391, -4.672269280947141, 0.45)

    camera_ue = np.asarray(_apartment_camera_ue_cm(mic_pos_m), dtype=float)
    render_scene = _absolute_apartment_render_scene(
        SceneSpec(seed=0, mic_pos_m=mic_pos_m, animals=[])
    )
    actor_ue = np.asarray(
        _world_from_scene(source_pos_m, room="apartment", spec=render_scene),
        dtype=float,
    )

    yaw_ue = _yaw_world_to_ue(mic_yaw_deg, "apartment")
    forward_ue = np.asarray([
        math.cos(math.radians(yaw_ue)),
        math.sin(math.radians(yaw_ue)),
    ])
    forward_distance_cm = float((actor_ue[:2] - camera_ue[:2]) @ forward_ue)

    assert forward_distance_cm > 0.0


def test_apartment_ue_cm_to_ssot_m_inverse_uses_absolute_origin():
    from run_render_pass import APARTMENT_FLOOR_Z_CM, APARTMENT_MIC_ORIGIN_CM
    from run_render_pass_apartment import _apartment_ue_cm_to_ssot_m

    ue = (
        APARTMENT_MIC_ORIGIN_CM[0] + 250.0,
        APARTMENT_MIC_ORIGIN_CM[1] - 175.0,
        APARTMENT_FLOOR_Z_CM + 90.0,
    )

    assert _apartment_ue_cm_to_ssot_m(ue) == pytest.approx((2.5, 1.75, 0.9))
