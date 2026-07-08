"""Rig direction runtime assertion via bone query.

Two capabilities:
  1. calibrate_rig_forward_from_velocity(actor, instance) -> observed rig
     forward direction in world frame (used to build rig_calibration.json).
  2. assert_body_forward(actor, expected_yaw_deg, tolerance_deg) -> raises
     AssertionError if observed body forward diverges from expected motion
     direction. Called per-clip or per-frame in run_render_pass_apartment
     to catch coordinate-system bugs.

Query strategy:
  Read Root (or Pelvis, or Spine1) bone WORLD position at frame T and T+N.
  velocity = pos_T+N - pos_T. Direction of velocity = rig's actual forward.

Note: works even when Head/Tail bones are dampened (they follow the root
rigidly during walking). See Plan 1.5.A analysis for the reasoning.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_FILE = REPO_ROOT / "tools" / "spike_rlr" / "rig_calibration.json"
RIG_CALIBRATION_ALGORITHM_VERSION = "rig_calib_v1"

# Preferred body-center bones in order of fallback
_BODY_BONE_CANDIDATES = ("Root", "Pelvis", "Hips", "Spine1", "Spine", "Bone")


def _yaw_difference_deg(a: float, b: float) -> float:
    """Signed shortest angular difference b-a in (-180, 180] degrees."""
    d = ((b - a + 180.0) % 360.0) - 180.0
    return d


def _assert_yaw_ok(observed: float, expected: float, tolerance_deg: float,
                    context: str) -> None:
    diff = abs(_yaw_difference_deg(observed, expected))
    if diff > tolerance_deg:
        raise AssertionError(
            f"[{context}] rig direction check FAILED: observed body-forward "
            f"yaw = {observed:.1f} deg, expected = {expected:.1f} deg, "
            f"diff = {diff:.1f} deg > tolerance {tolerance_deg:.1f} deg. "
            f"This usually means the rig walked in the WRONG direction. "
            f"Root causes: (a) rig walking_forward_yaw_offset_deg is wrong; "
            f"(b) mesh not yet approved-and-oriented in tmp/hy3d_batch/approved/; "
            f"(c) room world<->UE convention (position/rotation) got desynced."
        )


def sample_body_bone_position_in_frame(actor, bone_name: str):
    """Query a bone's world-space location via SPEAR RPC.

    IMPORTANT: caller must be inside an active `instance.begin_frame()`
    context — this function does NOT open its own frame. That was the bug
    in v1: opening a nested begin_frame after the render loop tore down
    frame state triggered engine_service.begin_frame:157 assert False.

    Returns np.ndarray shape (3,) in UE cm world frame, or None if the bone
    doesn't exist.
    """
    try:
        # SPEAR uses SkeletalMeshComponent.GetBoneTransform(InBoneName, RTS_World)
        comp = actor.GetComponentByClass(
            ComponentClass="/Script/Engine.SkeletalMeshComponent")
        if comp is None:
            return None
        tf = comp.GetBoneTransform(InBoneName=bone_name, TransformSpace="RTS_World")
        loc = tf["Location"] if isinstance(tf, dict) else tf.Location
        return np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64)
    except Exception:
        return None


def find_body_bone_in_frame(actor) -> Optional[str]:
    """Return the first available candidate bone name on this actor.

    Must be called inside an active begin_frame context."""
    for name in _BODY_BONE_CANDIDATES:
        pos = sample_body_bone_position_in_frame(actor, name)
        if pos is not None:
            return name
    return None


def assert_body_yaw_from_positions(pos_start, pos_end, expected_yaw_world_deg: float,
                                     tolerance_deg: float = 15.0,
                                     context: str = "clip") -> None:
    """Compare a velocity yaw (from two already-sampled positions) to expected.

    No SPEAR access — pure math. Caller samples pos_start and pos_end inside
    two different begin_frame windows (e.g. frames T and T+N of the render
    loop) and then passes them here.

    Raises AssertionError if outside tolerance; silently skips if actor
    isn't moving (pos_end - pos_start too small).
    """
    if pos_start is None or pos_end is None:
        raise RuntimeError(f"[{context}] missing bone position sample")
    v = np.asarray(pos_end) - np.asarray(pos_start)
    if np.linalg.norm(v[:2]) < 1e-3:
        return  # not moving — skip (paused / hold segment)
    observed_yaw = float(np.degrees(np.arctan2(v[1], v[0])))
    _assert_yaw_ok(observed=observed_yaw, expected=expected_yaw_world_deg,
                    tolerance_deg=tolerance_deg, context=context)


# ---- Legacy convenience wrappers (open their own begin_frame windows) ----
# Kept for tests + calibration CLI. Do NOT call these from inside a render
# loop that has its own frame management.

def _sample_body_bone_position(actor, instance, bone_name: str):
    with instance.begin_frame():
        pos = sample_body_bone_position_in_frame(actor, bone_name)
    with instance.end_frame():
        pass
    return pos


def _find_body_bone(actor, instance) -> Optional[str]:
    for name in _BODY_BONE_CANDIDATES:
        pos = _sample_body_bone_position(actor, instance, name)
        if pos is not None:
            return name
    return None


def calibrate_rig_forward_from_velocity(actor, instance, n_step_frames: int = 30) -> float:
    """Spawn actor at (0,0,0) with body_yaw=0, play walking, return the observed
    forward yaw (world-frame degrees). Caller uses this as offset baseline.

    This is a static-scene calibration: caller must have already spawned the
    actor + set body yaw to 0 before calling. Opens its own begin_frame
    windows — safe to call between clips, NOT inside another begin_frame.
    """
    body_bone = _find_body_bone(actor, instance)
    if body_bone is None:
        raise RuntimeError("no body-center bone found on actor")
    pos_start = _sample_body_bone_position(actor, instance, body_bone)
    instance.step(num_frames=n_step_frames)
    pos_end = _sample_body_bone_position(actor, instance, body_bone)
    if pos_start is None or pos_end is None:
        raise RuntimeError("failed to sample bone positions")
    v = pos_end - pos_start
    if np.linalg.norm(v[:2]) < 1e-3:
        raise RuntimeError(
            f"observed velocity too small ({np.linalg.norm(v):.4f} cm) — "
            f"is the animation actually playing?"
        )
    return float(np.degrees(np.arctan2(v[1], v[0])))


def assert_body_forward(actor, instance, expected_yaw_world_deg: float,
                         tolerance_deg: float = 15.0, n_step_frames: int = 5,
                         context: str = "clip") -> None:
    """DEPRECATED for in-render-loop use.

    Uses its own begin_frame windows — will crash if called after a render
    loop has torn down frame state. For in-render-loop use, call
    sample_body_bone_position_in_frame() + assert_body_yaw_from_positions().
    """
    body_bone = _find_body_bone(actor, instance)
    if body_bone is None:
        raise RuntimeError(f"[{context}] no body-center bone found")
    pos_start = _sample_body_bone_position(actor, instance, body_bone)
    instance.step(num_frames=n_step_frames)
    pos_end = _sample_body_bone_position(actor, instance, body_bone)
    assert_body_yaw_from_positions(
        pos_start=pos_start, pos_end=pos_end,
        expected_yaw_world_deg=expected_yaw_world_deg,
        tolerance_deg=tolerance_deg, context=context,
    )


def write_rig_calibration_json(tag: str, offset_deg: float,
                                algorithm_version: str) -> None:
    """Write/update rig_calibration.json for one tag."""
    p = CALIBRATION_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        db = json.loads(p.read_text())
    else:
        db = {}
    db[tag] = {
        "walking_forward_yaw_offset_deg": float(offset_deg),
        "algorithm_version": algorithm_version,
        "calibrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(db, indent=2, sort_keys=True))


def read_rig_calibration_json(tag: str) -> Optional[dict]:
    p = CALIBRATION_FILE
    if not p.exists():
        return None
    db = json.loads(p.read_text())
    if not isinstance(db, dict):
        return None
    entry = db.get(tag)
    # Skip doc-only fields (keys starting with "_")
    if not isinstance(entry, dict):
        return None
    return entry
