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


def _sample_body_bone_position(actor, instance, bone_name: str):
    """Query a bone's world-space location via SPEAR RPC.

    Returns np.ndarray shape (3,) in UE cm world frame, or None if the bone
    doesn't exist.
    """
    with instance.begin_frame():
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


def _find_body_bone(actor, instance) -> Optional[str]:
    """Return the first available candidate bone name on this actor."""
    for name in _BODY_BONE_CANDIDATES:
        pos = _sample_body_bone_position(actor, instance, name)
        if pos is not None:
            return name
    return None


def calibrate_rig_forward_from_velocity(actor, instance, n_step_frames: int = 30) -> float:
    """Spawn actor at (0,0,0) with body_yaw=0, play walking, return the observed
    forward yaw (world-frame degrees). Caller uses this as offset baseline.

    This is a static-scene calibration: caller must have already spawned the
    actor + set body yaw to 0 before calling.
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
    # UE world convention: +X = right, +Y = forward (varies per room).
    # We return the raw world-frame yaw = atan2(vy, vx).
    return float(np.degrees(np.arctan2(v[1], v[0])))


def assert_body_forward(actor, instance, expected_yaw_world_deg: float,
                         tolerance_deg: float = 15.0, n_step_frames: int = 5,
                         context: str = "clip") -> None:
    """Assert that actor's body is moving in the expected world-frame direction.

    Samples body-center bone position at frame T and T+N, computes velocity
    yaw, compares to expected. Raises AssertionError if outside tolerance.
    """
    body_bone = _find_body_bone(actor, instance)
    if body_bone is None:
        raise RuntimeError(f"[{context}] no body-center bone found")
    pos_start = _sample_body_bone_position(actor, instance, body_bone)
    instance.step(num_frames=n_step_frames)
    pos_end = _sample_body_bone_position(actor, instance, body_bone)
    if pos_start is None or pos_end is None:
        raise RuntimeError(f"[{context}] failed to sample bone positions")
    v = pos_end - pos_start
    if np.linalg.norm(v[:2]) < 1e-3:
        # Actor isn't moving; skip assertion (probably paused / hold segment)
        return
    observed_yaw = float(np.degrees(np.arctan2(v[1], v[0])))
    _assert_yaw_ok(observed=observed_yaw, expected=expected_yaw_world_deg,
                    tolerance_deg=tolerance_deg, context=context)


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
