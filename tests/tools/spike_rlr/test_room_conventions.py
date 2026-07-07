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
