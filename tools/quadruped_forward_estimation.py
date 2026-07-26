"""Deterministic forward-axis estimation for generated quadruped meshes.

Pure-numpy helpers shared by the Blender estimation wrapper and unit tests.
All functions work on world-space vertex arrays in a Z-up frame (Blender
world after glTF import): yaw is measured in the ground X/Y plane as degrees
about +Z from +X.

The estimate is review support, never review authority.  It produces the
continuous unsigned torso axis plus a head-end vote with an explicit
confidence, and always reports both cardinal-opposite candidates so a human
(or a calibrated visual pre-screener) confirms which end is the head before
the value enters the forward declaration contract.
"""

from __future__ import annotations

import math

import numpy as np


SCHEMA = "avengine_quadruped_forward_estimate_v1"

# Head-end voting weights mirror the retired spike_rlr/detect_head_axis.py
# detector: leg-pair spacing is the strongest anatomical cue, raised/dense
# geometry follows, end mass is the weakest.
LEG_SPACING_WEIGHT = 3
HIGH_VERTS_WEIGHT = 2
MASS_END_WEIGHT = 1
MAX_TOTAL_VOTES = LEG_SPACING_WEIGHT + HIGH_VERTS_WEIGHT + MASS_END_WEIGHT


class ForwardEstimationError(ValueError):
    pass


def _validate_vertices(vertices) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 200:
        raise ForwardEstimationError(
            f"expected at least 200 (N, 3) vertices, got shape {verts.shape}"
        )
    if not np.all(np.isfinite(verts)):
        raise ForwardEstimationError("vertex array contains non-finite values")
    return verts


def estimate_unsigned_axis_yaw_deg(vertices) -> float:
    """Principal ground-plane axis of the torso band, in [-90, 90) degrees.

    The torso band excludes the lowest region (legs) so leg posture cannot
    bias the axis.  The sign of the axis is deliberately ambiguous here;
    resolving head versus tail is `vote_head_end`'s job.
    """
    verts = _validate_vertices(vertices)
    z_low, z_high = np.percentile(verts[:, 2], (2.0, 98.0))
    torso_floor = z_low + 0.35*(z_high - z_low)
    torso = verts[verts[:, 2] >= torso_floor]
    if len(torso) < 100:
        raise ForwardEstimationError("insufficient torso vertices above leg band")
    ground = torso[:, :2] - torso[:, :2].mean(axis=0)
    cov = (ground.T @ ground) / max(len(ground) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    yaw = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    if yaw >= 90.0:
        yaw -= 180.0
    elif yaw < -90.0:
        yaw += 180.0
    return float(yaw)


def _rotated_to_axis_frame(verts: np.ndarray, unsigned_yaw_deg: float) -> np.ndarray:
    angle = math.radians(-unsigned_yaw_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotated = verts.copy()
    rotated[:, 0] = cos_a*verts[:, 0] - sin_a*verts[:, 1]
    rotated[:, 1] = sin_a*verts[:, 0] + cos_a*verts[:, 1]
    return rotated


def _leg_spacing_vote(rotated: np.ndarray) -> int:
    """Narrower leg pair marks the front.  Deterministic quadrant clustering:
    the leg band is split by longitudinal and lateral sign around its median
    instead of KMeans so repeated runs cannot disagree."""
    leg_threshold = np.percentile(rotated[:, 2], 20.0)
    legs = rotated[rotated[:, 2] < leg_threshold]
    if len(legs) < 40:
        return 0
    longitudinal = legs[:, 0] - np.median(legs[:, 0])
    lateral = legs[:, 1] - np.median(legs[:, 1])
    widths = {}
    for end, mask_end in (("neg", longitudinal < 0), ("pos", longitudinal >= 0)):
        side_centers = []
        for mask_side in (lateral < 0, lateral >= 0):
            quadrant = legs[mask_end & mask_side]
            if len(quadrant) < 10:
                return 0
            side_centers.append(float(np.median(quadrant[:, 1])))
        widths[end] = abs(side_centers[1] - side_centers[0])
    if widths["pos"] < widths["neg"]*0.9:
        return +LEG_SPACING_WEIGHT
    elif widths["neg"] < widths["pos"]*0.9:
        return -LEG_SPACING_WEIGHT
    else:
        return 0


def _high_verts_vote(rotated: np.ndarray) -> int:
    """Raised geometry (head and ears of a standing quadruped) marks the front."""
    top_threshold = np.percentile(rotated[:, 2], 90.0)
    top = rotated[rotated[:, 2] > top_threshold]
    if len(top) < 10:
        return 0
    tolerance = 0.05*float(np.abs(rotated[:, 0]).max())
    high_mean = float(top[:, 0].mean() - rotated[:, 0].mean())
    if high_mean > tolerance:
        return +HIGH_VERTS_WEIGHT
    elif high_mean < -tolerance:
        return -HIGH_VERTS_WEIGHT
    else:
        return 0


def _mass_end_vote(rotated: np.ndarray) -> int:
    """Dense face/head reconstruction usually outweighs the tail end."""
    longitudinal = rotated[:, 0] - rotated[:, 0].mean()
    positive_end = int(np.count_nonzero(longitudinal > longitudinal.max()*0.7))
    negative_end = int(np.count_nonzero(longitudinal < longitudinal.min()*0.7))
    if positive_end > negative_end*1.2:
        return +MASS_END_WEIGHT
    elif negative_end > positive_end*1.2:
        return -MASS_END_WEIGHT
    else:
        return 0


def vote_head_end(vertices, unsigned_yaw_deg: float) -> dict:
    verts = _validate_vertices(vertices)
    rotated = _rotated_to_axis_frame(verts, unsigned_yaw_deg)
    signals = {
        "leg_spacing_vote": _leg_spacing_vote(rotated),
        "high_verts_vote": _high_verts_vote(rotated),
        "mass_end_vote": _mass_end_vote(rotated),
    }
    total = sum(signals.values())
    nonzero = [ value for value in signals.values() if value != 0 ]
    unanimous = len(nonzero) >= 2 and all(
        math.copysign(1, value) == math.copysign(1, nonzero[0]) for value in nonzero
    )
    confidence = min(1.0, abs(total)/MAX_TOTAL_VOTES + (0.15 if unanimous else 0.0))
    return {
        "signals": signals,
        "total_votes": int(total),
        "unanimous": bool(unanimous),
        "confidence": float(confidence),
        "head_along_positive_axis": total >= 0,
    }


def estimate_forward(vertices) -> dict:
    """Full estimate: unsigned axis, head-end vote and both candidates."""
    unsigned_yaw = estimate_unsigned_axis_yaw_deg(vertices)
    vote = vote_head_end(vertices, unsigned_yaw)
    if vote["head_along_positive_axis"]:
        estimated = unsigned_yaw
    else:
        estimated = unsigned_yaw + 180.0
    if estimated > 180.0:
        estimated -= 360.0
    opposite = estimated - 180.0 if estimated > 0.0 else estimated + 180.0
    return {
        "schema": SCHEMA,
        "coordinate_frame": "blender_world_z_up_yaw_about_positive_z_from_positive_x",
        "unsigned_axis_yaw_deg": float(unsigned_yaw),
        "head_end_vote": vote,
        "estimated_front_yaw_deg": float(estimated),
        "candidate_front_yaw_degrees": [float(estimated), float(opposite)],
        "estimate_is_review_support_not_review_authority": True,
    }
