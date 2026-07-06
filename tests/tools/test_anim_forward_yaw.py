"""Regression test: rig-based animated animals must face their motion direction.

Root cause of the "backward-walk" bug: Quaternius Dog/Cat "Walking" animation
has its local forward = -X_local. If we set the actor's body_yaw = motion_yaw
directly (which is what scene_spec._generate_trajectory did before this fix),
the actor's +X_local aligns with motion direction, so the anim's -X_local
(the walking direction) points 180 degrees AWAY from motion → the dog walks
backward relative to its head.

Correct formula: body_yaw_world = motion_yaw_world + 180 (mod 360).

This regression test hard-codes the expected relationship so if anyone ever
tries to remove the +180 again, it fails immediately.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, "/data/jzy/code/SPEAR/tools")

from gpurir_scenes import scene_spec


def test_ANIM_FORWARD_YAW_OFFSET_is_180():
    """The constant is 180 for Quaternius rigs (Dog/Cat). Prevents accidental
    removal or "fix" that would flip all animated animals backward."""
    assert scene_spec.ANIM_FORWARD_YAW_OFFSET_DEG == 180.0, (
        "Quaternius Walking anim local-forward is -X_local, so body_yaw must "
        "be motion_yaw + 180 for the animal to face its motion direction. "
        "Removing this offset makes ALL animated animals moonwalk."
    )


def test_generated_trajectory_yaw_matches_motion_direction():
    """For any animated tag, the generated yaw must satisfy
    (yaw - motion_direction) mod 360 == 180 (within numerical tolerance)."""
    rng = np.random.default_rng(42)
    tag = "dog_husky"
    traj, yaw = scene_spec._generate_trajectory(rng, scene_spec.ROOM_SIZE_M, tag)
    # motion direction per frame (from gradient)
    xs = traj[:, 0]
    ys = traj[:, 1]
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    motion_deg = np.degrees(np.arctan2(dy, dx))
    diff = (yaw - motion_deg + 360) % 360
    # Every frame should have diff ≈ 180
    for i, d in enumerate(diff):
        # Skip frames where motion is nearly zero (degenerate direction)
        speed = math.hypot(dx[i], dy[i])
        if speed < 1e-3:
            continue
        assert abs(d - 180.0) < 1e-3, (
            f"frame {i}: yaw={yaw[i]:.2f} motion={motion_deg[i]:.2f} "
            f"diff={d:.2f} (expected 180)"
        )


def test_scene_two_dogs_uses_scene_spec_constant():
    """scene_two_dogs.py must reuse scene_spec.ANIM_FORWARD_YAW_OFFSET_DEG,
    not hard-code its own copy. This prevents drift between the two paths."""
    from gpurir_scenes import scene_two_dogs
    # scene_two_dogs previously had its own _ANIM_FORWARD_YAW_OFFSET = 180.
    # After the fix it should import from scene_spec so the two values can
    # never diverge.
    assert (
        getattr(scene_two_dogs, "_ANIM_FORWARD_YAW_OFFSET", None)
        == scene_spec.ANIM_FORWARD_YAW_OFFSET_DEG
    ), (
        "scene_two_dogs must use scene_spec.ANIM_FORWARD_YAW_OFFSET_DEG "
        "(directly or via a local alias with the same value) so both paths "
        "always agree."
    )
