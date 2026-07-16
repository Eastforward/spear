"""Tests for the shared arm/leg motion-basis correction contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SPEAR_ROOT = Path(__file__).resolve().parents[3]
SPIKE_TOOLS = SPEAR_ROOT / "tools" / "spike_rlr"
if str(SPIKE_TOOLS) not in sys.path:
    sys.path.insert(0, str(SPIKE_TOOLS))

import retarget_motion_basis_review as review


def _motion_frame(horizontal: float, *, sideways: bool) -> dict[str, tuple[float, float, float]]:
    frame: dict[str, tuple[float, float, float]] = {
        "pelvis": (0.0, 0.0, 1.0),
        "left_clavicle": (0.30, 0.0, 1.60),
        "right_clavicle": (-0.30, 0.0, 1.60),
        "left_thigh": (0.12, 0.0, 1.00),
        "right_thigh": (-0.12, 0.0, 1.00),
        "left_upper_arm": (0.30, 0.0, 1.50),
        "right_upper_arm": (-0.30, 0.0, 1.50),
        "neck": (0.0, 0.0, 1.65),
        "head": (0.0, 0.0, 1.75),
        "head_tail": (0.0, 0.0, 1.85),
    }
    for side, side_sign in (("left", 1.0), ("right", -1.0)):
        hip_x = 0.12 * side_sign
        shoulder_x = 0.30 * side_sign
        phase = horizontal * side_sign
        if sideways:
            frame[f"{side}_calf"] = (
                hip_x + 0.5 * phase + 0.06 * side_sign,
                0.0,
                0.55,
            )
            frame[f"{side}_foot"] = (hip_x + phase, 0.0, 0.08)
            frame[f"{side}_forearm"] = (
                shoulder_x + 0.5 * phase + 0.05 * side_sign,
                0.0,
                1.22,
            )
            frame[f"{side}_hand"] = (shoulder_x + phase, 0.0, 0.96)
        else:
            frame[f"{side}_calf"] = (
                hip_x,
                0.5 * phase - 0.06,
                0.55,
            )
            frame[f"{side}_foot"] = (hip_x, phase, 0.08)
            frame[f"{side}_forearm"] = (
                shoulder_x,
                0.5 * phase - 0.05,
                1.22,
            )
            frame[f"{side}_hand"] = (shoulder_x, phase, 0.96)
    return frame


def _motion_frames(*, sideways: bool) -> list[dict[str, tuple[float, float, float]]]:
    return [
        _motion_frame(horizontal, sideways=sideways)
        for horizontal in (-0.24, -0.12, 0.0, 0.12, 0.24)
    ]


@pytest.mark.parametrize("angle", [0, -90, 90, 180])
def test_yaw_is_proper_and_keeps_canonical_up(angle):
    value = review.yaw_matrix(angle)

    assert value.T @ value == pytest.approx(np.eye(3), abs=1.0e-12)
    assert np.linalg.det(value) == pytest.approx(1.0, abs=1.0e-12)
    assert value @ np.asarray((0.0, 0.0, 1.0)) == pytest.approx(
        (0.0, 0.0, 1.0), abs=1.0e-12
    )


def test_positive_and_negative_yaw_have_auditable_canonical_signs():
    canonical_front = np.asarray((0.0, -1.0, 0.0))

    assert review.yaw_matrix(90) @ canonical_front == pytest.approx(
        (1.0, 0.0, 0.0), abs=1.0e-12
    )
    assert review.yaw_matrix(-90) @ canonical_front == pytest.approx(
        (-1.0, 0.0, 0.0), abs=1.0e-12
    )


@pytest.mark.parametrize("value", [15, -180, 360, True, 0.0, "90"])
def test_yaw_rejects_every_unreviewed_value(value):
    with pytest.raises(review.MotionBasisReviewError, match="allowed"):
        review.yaw_matrix(value)


def test_four_limb_metrics_distinguish_sagittal_from_sideways_motion():
    sagittal = review.compute_four_limb_motion_metrics(
        _motion_frames(sideways=False), fps=30
    )
    sideways = review.compute_four_limb_motion_metrics(
        _motion_frames(sideways=True), fps=30
    )

    assert sagittal["overall_classification"] == "four_limb_sagittal_motion"
    assert sideways["overall_classification"] == "sideways_limb_motion"
    assert set(sagittal["limbs"]) == {
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
    }
    for limb in sagittal["limbs"].values():
        assert limb["lateral_to_forward_excursion_ratio"] < 0.5
        assert limb["mean_plane_normal_dot_lateral_abs"] > 0.8
        assert limb["mean_plane_normal_dot_forward_abs"] < 0.5
    for limb in sideways["limbs"].values():
        assert limb["lateral_to_forward_excursion_ratio"] > 0.65
        assert limb["mean_plane_normal_dot_forward_abs"] > 0.8


def test_four_limb_metrics_require_all_arm_and_leg_semantics():
    frames = _motion_frames(sideways=False)
    del frames[2]["left_hand"]

    with pytest.raises(review.MotionBasisReviewError, match="left_hand"):
        review.compute_four_limb_motion_metrics(frames, fps=30)


def test_axial_pose_metrics_accept_upright_chain_and_reject_head_roll():
    upright = review.compute_axial_pose_metrics(
        _motion_frames(sideways=False), fps=30
    )
    assert upright["automatic_checks"] == "passed"
    assert upright["overall_classification"] == (
        "axial_pose_within_source_motion_envelope"
    )
    assert all(
        value["status"] == "passed" for value in upright["metrics"].values()
    )

    rolled = _motion_frames(sideways=False)
    rolled[2]["head_tail"] = (0.08, 0.0, 1.85)
    failed = review.compute_axial_pose_metrics(rolled, fps=30)
    assert failed["automatic_checks"] == "failed"
    assert failed["metrics"]["head_bone_lateral_tilt_deg"]["status"] == "failed"


def test_axial_pose_metrics_require_head_tail_semantic():
    frames = _motion_frames(sideways=False)
    del frames[0]["head_tail"]
    with pytest.raises(review.MotionBasisReviewError, match="head_tail"):
        review.compute_axial_pose_metrics(frames, fps=30)


@pytest.mark.parametrize("fps", [0, -1, True, 30.0])
def test_four_limb_metrics_reject_invalid_fps(fps):
    with pytest.raises(review.MotionBasisReviewError, match="FPS"):
        review.compute_four_limb_motion_metrics(
            _motion_frames(sideways=False), fps=fps
        )
