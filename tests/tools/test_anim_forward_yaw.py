"""Regression test: rig-based animated animals must face their motion direction.

2026-07-08 correction (visual verification):
  The Quaternius Dog/Cat "Walking" animation local-forward is +X_local
  (NOT -X_local as previously assumed). This means body_yaw = motion_yaw
  directly with NO offset. See git commit history for the visual proof:
  shoebox_v2 view2 rendered with offset=180 showed golden walking head-first
  BACKWARDS (head pointed at -X while motion was +X); after switching to
  offset=0 the same scene rendered head-first FORWARD. Same result
  reproduced in apartment_v1 mic-yaw-180 view (data/apartment_v1_spec.json).

Formula: body_yaw_world = (motion_yaw_world + offset) mod 360, where
         offset = Quaternius rig's walking_forward_yaw_offset_deg = 0.

DO NOT change offset back to 180 without visual re-verification against
BOTH shoebox AND apartment rooms — the earlier "180" assumption looked
plausible from BP inspection but was visually wrong.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools")

from gpurir_scenes import scene_spec


def test_QUATERNIUS_forward_offset_is_zero():
    """Quaternius Dog/Cat walking anim local-forward = +X_local; no offset.
    Verified visually on shoebox_v2 view2 and apartment_v1 mic-yaw-180 view.
    If this ever needs to change, re-render both rooms with the new value
    and inspect the output — do NOT trust rig-blueprint inspection alone."""
    from species_rig_map import QUATERNIUS_FORWARD_YAW_OFFSET_DEG
    assert QUATERNIUS_FORWARD_YAW_OFFSET_DEG == 0.0, (
        f"Quaternius Dog/Cat Walk anim local-forward is +X_local, so no "
        f"offset needed. Got {QUATERNIUS_FORWARD_YAW_OFFSET_DEG}."
    )


def test_generated_trajectory_yaw_matches_motion_direction():
    """For any animated Quaternius tag, generated yaw ≈ motion direction."""
    rng = np.random.default_rng(42)
    tag = "dog_beagle_v2"
    traj, yaw = scene_spec._generate_trajectory(rng, scene_spec.ROOM_SIZE_M, tag)
    xs = traj[:, 0]
    ys = traj[:, 1]
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    motion_deg = np.degrees(np.arctan2(dy, dx))
    diff = (yaw - motion_deg + 360) % 360
    for i, d in enumerate(diff):
        speed = math.hypot(dx[i], dy[i])
        if speed < 1e-3:
            continue
        # After 2026-07-08 fix: diff should be ~0 (body faces motion direction).
        assert abs(d) < 1e-3 or abs(d - 360) < 1e-3, (
            f"frame {i}: yaw={yaw[i]:.2f} motion={motion_deg[i]:.2f} "
            f"diff={d:.2f} (expected 0 mod 360)"
        )


def test_all_animated_tags_have_walking_yaw_offset_field():
    """Every ANIMATED_RIG_MAP entry must declare walking_forward_yaw_offset_deg
    so a new rig with a different forward convention has to make an explicit
    choice (can't silently inherit the wrong default)."""
    from species_rig_map import ANIMATED_RIG_MAP
    for tag, meta in ANIMATED_RIG_MAP.items():
        assert "walking_forward_yaw_offset_deg" in meta, (
            f"animated tag {tag} missing walking_forward_yaw_offset_deg. "
            f"For Quaternius Dog/Cat use QUATERNIUS_FORWARD_YAW_OFFSET_DEG (0.0)."
        )
        assert isinstance(meta["walking_forward_yaw_offset_deg"], (int, float))


def test_all_current_tags_use_quaternius_offset_0():
    """The 5 current animated tags (Cat, Dog, chipmunk) are all Quaternius,
    so all must declare 0.0. If someone swaps in a non-Quaternius rig with
    a different local-forward convention, they must set a NEW offset value
    and this test will remind them."""
    from species_rig_map import ANIMATED_RIG_MAP
    for tag, meta in ANIMATED_RIG_MAP.items():
        assert meta["walking_forward_yaw_offset_deg"] == 0.0, (
            f"tag {tag} declares walking_forward_yaw_offset_deg="
            f"{meta['walking_forward_yaw_offset_deg']} but all current rigs "
            f"are Quaternius (needs 0.0). Did you swap in a non-Quaternius "
            f"rig? If so, add a new constant + document its local-forward."
        )


def test_generate_trajectory_uses_per_tag_offset():
    """_generate_trajectory looks up the offset from ANIMATED_RIG_MAP for
    the tag, not the global constant. Verified by monkey-patching a fake
    offset=180 and confirming yaw now differs from motion by 180."""
    from species_rig_map import ANIMATED_RIG_MAP
    saved_meta = ANIMATED_RIG_MAP.get("dog_beagle_v2")
    fake_meta = dict(saved_meta)
    fake_meta["walking_forward_yaw_offset_deg"] = 180.0
    ANIMATED_RIG_MAP["dog_beagle_v2"] = fake_meta
    try:
        rng = np.random.default_rng(42)
        traj, yaw = scene_spec._generate_trajectory(
            rng, scene_spec.ROOM_SIZE_M, "dog_beagle_v2"
        )
        dx = np.gradient(traj[:, 0])
        dy = np.gradient(traj[:, 1])
        motion = np.degrees(np.arctan2(dy, dx))
        diff = (yaw - motion + 360) % 360
        for i, d in enumerate(diff):
            speed = (dx[i] ** 2 + dy[i] ** 2) ** 0.5
            if speed < 1e-3:
                continue
            assert abs(d - 180.0) < 1e-3, (
                f"frame {i}: with offset=180, yaw should be motion+180; "
                f"got yaw={yaw[i]:.2f} motion={motion[i]:.2f} diff={d:.2f}"
            )
    finally:
        ANIMATED_RIG_MAP["dog_beagle_v2"] = saved_meta
