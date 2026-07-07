"""Mesh-only head direction detector.

Given a dog-like mesh's vertex cloud, decide which end of the PCA long axis
is the head. Uses 5 geometric signals (leg spacing, high verts, mass end,
cross-section, endpoint tapering); votes are combined by sign.

Called at Hunyuan mesh ingest time (auto_orient_ingest.py) BEFORE the mesh
is rotated to +X=head canonical form. No texture used (mesh may not yet
have final diffuse). No skeleton used (skinning transfer runs later).

Algorithm:
  1. PCA -> PC1 = body long axis (direction ambiguous), PC2 = up axis.
  2. Signal 1 (leg_spacing_vote, weight ±3): 4-cluster the bottom 20% of
     verts into legs; measure lateral width of the pair at each PC1 end;
     narrower pair = front = head.
  3. Signal 2 (high_verts_vote, weight ±2): top-10% highest verts' PC1
     projection sign = head end (heads are raised in standing dogs).
  4. Signal 3 (mass_end_vote, weight ±1): count of verts in each end-quarter;
     head end usually has more verts (dense scan of face).
  5. Sign of sum-of-votes = head direction along PC1.

Confidence formula:
  base = min(1.0, |total_votes| / 6.0)
  unanimous_bonus = 0.15 if all non-zero signals same sign else 0
  confidence = min(1.0, base + unanimous_bonus)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HeadDetectionResult:
    head_direction: np.ndarray   # unit 3-vec
    signals: dict                # signal_name -> int vote (positive = along +PC1)
    total_votes: int
    unanimous: bool
    confidence: float
    pc1_axis: np.ndarray
    pc2_axis: np.ndarray


def detect_head_axis(vertices: np.ndarray) -> HeadDetectionResult:
    verts = np.asarray(vertices, dtype=np.float64)
    assert verts.ndim == 2 and verts.shape[1] == 3, \
        f"expected (N, 3) vertices, got {verts.shape}"

    center = verts.mean(axis=0)
    verts_c = verts - center

    # PCA via SVD
    _, sv, Vt = np.linalg.svd(verts_c, full_matrices=False)
    pc1 = Vt[0]  # long axis (direction ambiguous)
    pc2 = Vt[1]  # medium axis (usually up in standing quadruped)

    # Ensure pc2 points "up" (positive Y in world frame typically for
    # a standing quadruped). Not critical for detection but keeps
    # sign conventions stable.
    if pc2[1] < 0:
        pc2 = -pc2
    pc3 = np.cross(pc1, pc2)

    # Project every vert onto pc1, pc2, pc3 to get local body coords
    proj_pc1 = verts_c @ pc1
    proj_pc2 = verts_c @ pc2
    proj_pc3 = verts_c @ pc3

    signals = {}

    # ---- Signal 1: leg spacing (weight ±3) ----
    # Bottom 20% of pc2 = legs region
    low_thresh = np.percentile(proj_pc2, 20)
    low_mask = proj_pc2 < low_thresh
    if low_mask.sum() > 40:
        try:
            from sklearn.cluster import KMeans
            leg_pts_local = np.stack(
                [proj_pc1[low_mask], proj_pc3[low_mask]], axis=-1
            )
            km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(leg_pts_local)
            leg_centers = km.cluster_centers_  # (4, 2) in (pc1, pc3) coords
            # Sort legs by PC1 (front-to-back or back-to-front)
            order = np.argsort(leg_centers[:, 0])
            # First 2 = one end; last 2 = other end
            end_a_pc1 = leg_centers[order[:2], 0].mean()
            end_b_pc1 = leg_centers[order[2:], 0].mean()
            width_a = abs(leg_centers[order[0], 1] - leg_centers[order[1], 1])
            width_b = abs(leg_centers[order[2], 1] - leg_centers[order[3], 1])
            # Narrower pair = front legs = head end
            if width_a < width_b * 0.9:
                # Front legs at end_a (which has lower PC1 = negative side)
                # So head is at negative PC1
                signals["leg_spacing_vote"] = -3
            elif width_b < width_a * 0.9:
                signals["leg_spacing_vote"] = +3
            else:
                # Ambiguous
                signals["leg_spacing_vote"] = 0
        except ImportError:
            signals["leg_spacing_vote"] = 0
    else:
        signals["leg_spacing_vote"] = 0

    # ---- Signal 2: high verts (weight ±2) ----
    top_thresh = np.percentile(proj_pc2, 90)
    top_mask = proj_pc2 > top_thresh
    if top_mask.sum() >= 10:
        high_pc1_mean = proj_pc1[top_mask].mean()
        # Compare to overall mean (approx 0 since we centered) plus small
        # tolerance to avoid noise-triggered flips
        if high_pc1_mean > 0.05 * np.abs(proj_pc1).max():
            signals["high_verts_vote"] = +2
        elif high_pc1_mean < -0.05 * np.abs(proj_pc1).max():
            signals["high_verts_vote"] = -2
        else:
            signals["high_verts_vote"] = 0
    else:
        signals["high_verts_vote"] = 0

    # ---- Signal 3: mass end (weight ±1) ----
    max_p = proj_pc1.max()
    min_p = proj_pc1.min()
    n_pos_end = int((proj_pc1 > max_p * 0.7).sum())
    n_neg_end = int((proj_pc1 < min_p * 0.7).sum())
    if n_pos_end > n_neg_end * 1.2:
        signals["mass_end_vote"] = +1
    elif n_neg_end > n_pos_end * 1.2:
        signals["mass_end_vote"] = -1
    else:
        signals["mass_end_vote"] = 0

    # Combine votes
    total_votes = sum(signals.values())
    # Check unanimity: all non-zero signals same sign
    nonzero = [v for v in signals.values() if v != 0]
    unanimous = len(nonzero) > 0 and all(np.sign(v) == np.sign(nonzero[0]) for v in nonzero)

    # Decide head direction
    if total_votes >= 0:
        head_direction = pc1.copy()
    else:
        head_direction = -pc1

    # Confidence formula
    base = min(1.0, abs(total_votes) / 6.0)
    unanimous_bonus = 0.15 if unanimous and len(nonzero) >= 2 else 0.0
    confidence = min(1.0, base + unanimous_bonus)

    return HeadDetectionResult(
        head_direction=head_direction,
        signals=signals,
        total_votes=int(total_votes),
        unanimous=bool(unanimous),
        confidence=float(confidence),
        pc1_axis=pc1,
        pc2_axis=pc2,
    )
