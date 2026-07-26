"""Deterministic gait-direction classification for in-place quadruped walks.

Pure-numpy core shared by the Blender gait audit wrapper and unit tests.

An in-place Walking action keeps the root fixed while planted (stance) feet
sweep backward relative to the body.  For a rig whose anatomical front is the
canonical +X axis, correct forward locomotion therefore shows the stance-band
vertices drifting toward -X.  Stance drift toward +X means the animal walks
backward; dominant lateral drift means it walks sideways.  This turns the
historical "cats running sideways, dogs running backwards" human-only finding
into a fail-closed automatic check.

Frames must share one vertex indexing (the same evaluated mesh sampled at
successive action frames).  Vertices are expected in a Z-up world frame with
the asset already heading-normalized to +X-forward.
"""

from __future__ import annotations

import math

import numpy as np


SCHEMA = "avengine_quadruped_gait_direction_audit_v1"

STANCE_BAND_PERCENTILE = 4.0
# Dominant-axis margin before calling a direction instead of "ambiguous".
AXIS_DOMINANCE_RATIO = 1.5
# Minimum mean per-frame stance drift, as a fraction of the mesh bounding
# diagonal, below which the motion carries no usable direction signal.
MINIMUM_DRIFT_RATIO = 1e-4


class GaitDirectionError(ValueError):
    pass


def _validate_frames(frames) -> list[np.ndarray]:
    if len(frames) < 3:
        raise GaitDirectionError("gait classification needs at least 3 sampled frames")
    validated = []
    expected = None
    for index, frame in enumerate(frames):
        verts = np.asarray(frame, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 50:
            raise GaitDirectionError(
                f"frame {index} must be an (N, 3) array with N >= 50, got {verts.shape}"
            )
        if not np.all(np.isfinite(verts)):
            raise GaitDirectionError(f"frame {index} contains non-finite values")
        if expected is None:
            expected = len(verts)
        elif len(verts) != expected:
            raise GaitDirectionError(
                "frames must share one vertex indexing; "
                f"frame {index} has {len(verts)} vertices, expected {expected}"
            )
        validated.append(verts)
    return validated


def stance_drift(frames) -> dict:
    """Mean world-space drift of vertices planted in consecutive frame pairs."""
    validated = _validate_frames(frames)
    stacked = np.concatenate(validated, axis=0)
    diagonal = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    if diagonal <= 0.0:
        raise GaitDirectionError("degenerate mesh bounds")
    drifts = []
    for previous, current in zip(validated[:-1], validated[1:]):
        threshold_previous = np.percentile(previous[:, 2], STANCE_BAND_PERCENTILE)
        threshold_current = np.percentile(current[:, 2], STANCE_BAND_PERCENTILE)
        planted = (previous[:, 2] <= threshold_previous) & (
            current[:, 2] <= threshold_current
        )
        if int(planted.sum()) < 10:
            continue
        drifts.append((current[planted] - previous[planted]).mean(axis=0))
    if len(drifts) < 2:
        raise GaitDirectionError("insufficient planted-foot overlap between frames")
    mean_drift = np.asarray(drifts, dtype=np.float64).mean(axis=0)
    return {
        "mean_drift_per_frame": [float(value) for value in mean_drift],
        "mean_drift_ratio_of_diagonal": float(
            np.linalg.norm(mean_drift[:2])/diagonal
        ),
        "frame_pair_count": len(drifts),
        "bounding_diagonal": diagonal,
    }


def classify_gait_direction(frames) -> dict:
    """Classify an in-place walk against the canonical +X anatomical front."""
    drift = stance_drift(frames)
    drift_x, drift_y = drift["mean_drift_per_frame"][:2]
    magnitude = drift["mean_drift_ratio_of_diagonal"]
    drift_yaw_deg = math.degrees(math.atan2(drift_y, drift_x))
    if magnitude < MINIMUM_DRIFT_RATIO:
        classification = "ambiguous"
    elif abs(drift_x) >= abs(drift_y)*AXIS_DOMINANCE_RATIO:
        # Stance feet sweep opposite to travel: -X drift is forward walking.
        classification = "forward" if drift_x < 0.0 else "backward"
    elif abs(drift_y) >= abs(drift_x)*AXIS_DOMINANCE_RATIO:
        classification = "sideways"
    else:
        classification = "ambiguous"
    return {
        "schema": SCHEMA,
        "coordinate_frame": "blender_world_z_up_front_positive_x",
        "stance_band_percentile": STANCE_BAND_PERCENTILE,
        "stance_drift": drift,
        "stance_drift_yaw_deg": float(drift_yaw_deg),
        "classification": classification,
        "walks_head_first": classification == "forward",
    }
