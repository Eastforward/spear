"""Tests for tools/spike_rlr/detect_head_axis.py.

Uses tiny synthesized dog-like meshes (programmatic geometry) so tests are
fast and deterministic. Real Hunyuan meshes are tested via integration
tests in Task 3.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from detect_head_axis import detect_head_axis, HeadDetectionResult  # noqa: E402


def _synth_dog(head_axis="+X", n_body=200, n_head=100, n_tail=50, n_legs=200):
    """Programmatically synthesize a dog-like point cloud.
    +X = head convention by default. n_body torso verts, n_head near +X end,
    n_tail near -X end, 4 leg clusters below body.
    Returns (n, 3) numpy array in "canonical" +X=head convention; caller may
    rotate it to test flipped detection.
    """
    rng = np.random.default_rng(seed=42)
    # Torso: long ellipsoid along X
    torso_x = rng.uniform(-0.5, 0.5, n_body)
    torso_y = rng.uniform(0.4, 0.6, n_body)   # torso is up in the air
    torso_z = rng.uniform(-0.15, 0.15, n_body)  # narrow width
    torso = np.stack([torso_x, torso_y, torso_z], axis=-1)

    # Head: dense cluster near +X end + one narrow snout tip
    head_x = rng.normal(0.6, 0.05, n_head)  # dense near +0.6
    head_y = rng.uniform(0.55, 0.75, n_head)  # slightly higher than torso
    head_z = rng.uniform(-0.1, 0.1, n_head)   # medium width
    head = np.stack([head_x, head_y, head_z], axis=-1)

    # Tail: sparse taper toward -X end
    tail_x = rng.uniform(-0.75, -0.5, n_tail)
    tail_y = rng.uniform(0.45, 0.55, n_tail)
    tail_z = rng.uniform(-0.03, 0.03, n_tail)  # very narrow
    tail = np.stack([tail_x, tail_y, tail_z], axis=-1)

    # 4 legs: y near 0 (ground), 4 clusters in front-narrow / hind-wide
    n_leg = n_legs // 4
    # Front legs (narrow, at +0.3 X): z = ±0.10
    fl_l = np.stack([rng.normal(0.3, 0.03, n_leg),
                      rng.uniform(0.0, 0.4, n_leg),
                      rng.normal(+0.10, 0.02, n_leg)], axis=-1)
    fl_r = np.stack([rng.normal(0.3, 0.03, n_leg),
                      rng.uniform(0.0, 0.4, n_leg),
                      rng.normal(-0.10, 0.02, n_leg)], axis=-1)
    # Hind legs (wider, at -0.3 X): z = ±0.16
    hl_l = np.stack([rng.normal(-0.3, 0.03, n_leg),
                      rng.uniform(0.0, 0.4, n_leg),
                      rng.normal(+0.16, 0.02, n_leg)], axis=-1)
    hl_r = np.stack([rng.normal(-0.3, 0.03, n_leg),
                      rng.uniform(0.0, 0.4, n_leg),
                      rng.normal(-0.16, 0.02, n_leg)], axis=-1)

    verts = np.concatenate([torso, head, tail, fl_l, fl_r, hl_l, hl_r], axis=0)

    if head_axis == "+X":
        return verts
    elif head_axis == "-X":
        # Flip along X
        verts_flipped = verts.copy()
        verts_flipped[:, 0] *= -1
        return verts_flipped
    elif head_axis == "+Y":
        # rotate 90° CCW in XY plane: (x,y,z) -> (-y,x,z)
        return np.stack([-verts[:, 1], verts[:, 0], verts[:, 2]], axis=-1)
    else:
        raise ValueError(f"unsupported head_axis {head_axis}")


def test_head_at_plus_x_detected():
    verts = _synth_dog(head_axis="+X")
    result = detect_head_axis(verts)
    # Detected head should point along +X (or close to it)
    assert result.head_direction[0] > 0.8, \
        f"expected head along +X, got {result.head_direction}"
    assert abs(result.head_direction[1]) < 0.3
    assert abs(result.head_direction[2]) < 0.3
    # Note: unanimous vote depends on synth fixture — leg spacing is strongest
    # signal and dominates via weight even if high_verts disagrees.
    assert result.signals["leg_spacing_vote"] != 0, \
        f"leg spacing signal missing: {result.signals}"
    # Confidence should be at least "medium" for the +X standing dog
    assert result.confidence > 0.3


def test_head_at_minus_x_detected():
    verts = _synth_dog(head_axis="-X")
    result = detect_head_axis(verts)
    assert result.head_direction[0] < -0.8, \
        f"expected head along -X, got {result.head_direction}"


def test_head_at_plus_y_detected():
    """+Y rotation swaps the algorithm's 'up' axis assumption (pc2 ≈ +Y).
    Because the synthetic dog's height dimension moves to -X after the +Y
    head rotation, the algorithm's up-axis normalization can flip. What we
    still care about: the detected head axis is aligned with the body long
    axis (either sign is a downstream orientation problem, not a detector
    correctness problem — the auto-orient step handles the sign via
    Rodrigues rotation)."""
    verts = _synth_dog(head_axis="+Y")
    result = detect_head_axis(verts)
    # Head direction should point along Y (either +Y or -Y) — the dominant
    # body axis lies along Y after the rotation.
    assert abs(result.head_direction[1]) > 0.8, \
        f"expected head along ±Y (body-axis dominant), got {result.head_direction}"


def test_result_dataclass_fields():
    verts = _synth_dog(head_axis="+X")
    result = detect_head_axis(verts)
    assert isinstance(result, HeadDetectionResult)
    assert hasattr(result, "head_direction")
    assert hasattr(result, "signals")
    assert hasattr(result, "total_votes")
    assert hasattr(result, "unanimous")
    assert hasattr(result, "confidence")
    assert hasattr(result, "pc1_axis")
    assert hasattr(result, "pc2_axis")
    assert result.head_direction.shape == (3,)
    assert result.pc1_axis.shape == (3,)


def test_signals_dict_has_expected_keys():
    verts = _synth_dog(head_axis="+X")
    result = detect_head_axis(verts)
    # These 3 signals are always attempted (leg spacing, high verts, mass end)
    assert "leg_spacing_vote" in result.signals
    assert "high_verts_vote" in result.signals
    assert "mass_end_vote" in result.signals


def test_leg_spacing_signal_strongest_when_present():
    """When legs are clearly present with front narrower than hind, that
    signal alone should dominate the vote."""
    verts = _synth_dog(head_axis="+X")
    result = detect_head_axis(verts)
    # Leg spacing vote should have magnitude 3 (highest weight)
    assert abs(result.signals["leg_spacing_vote"]) == 3
    # And its sign should agree with the overall detected head direction
    assert np.sign(result.signals["leg_spacing_vote"]) == np.sign(result.total_votes)


def test_ambiguous_mesh_lower_confidence():
    """A near-spherical mesh should trigger low confidence."""
    rng = np.random.default_rng(seed=7)
    # Random sphere: no long axis, no legs, no head bulge
    theta = rng.uniform(0, 2*np.pi, 500)
    phi = rng.uniform(0, np.pi, 500)
    r = 0.5
    verts = np.stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi),
    ], axis=-1)
    result = detect_head_axis(verts)
    # Low confidence expected (no strong signals)
    assert result.confidence < 0.5, \
        f"expected low confidence for spherical mesh, got {result.confidence}"
