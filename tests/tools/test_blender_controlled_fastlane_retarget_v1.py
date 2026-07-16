from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_controlled_fastlane_retarget_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "blender_controlled_fastlane_retarget_v1", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_main_retarget_core_is_unchanged_and_hash_pinned():
    path = Path(runner.retarget.__file__).resolve()
    assert runner.retarget.sha256_file(path) == runner.MAIN_RUNNER_SHA256


def test_female_inheritance_is_narrow_and_never_claims_approval_or_formal_status():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "identity_shared_limb_basis_and_semantic_anatomical_transfer_algorithm_only" in source
    assert '"female_user_approval_claimed": False' in source
    assert '"formal_dataset_asset": False' in source
    assert '"state_classification": "research_candidate_fastlane"' in source
    assert '"user_acceptance": "not_claimed"' in source
    assert '"formal_dataset_registration_authorized": False' in source


def test_wrapper_requires_exact_male_v2_yaw_zero_identity_basis():
    inherited = runner.inherited_female_motion_basis()
    assert inherited["candidate_id"] == "yaw_000"
    assert inherited["matrix_3x3"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert inherited["female_user_approval_claimed"] is False


def test_female_fastlane_only_relaxes_tiny_source_rotation_drift_not_grounding():
    assert runner.FASTLANE_SOURCE_ROTATION_ORTHOGONALITY_MAX == 6.0e-6
    assert runner.retarget.MAXIMUM_GROUNDING_CORRECTION_M == 0.01
    assert runner.retarget.MAXIMUM_PENETRATION_M == 0.01
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "retarget.MAXIMUM_ROTATION_ORTHOGONALITY_ERROR = (" in source
    assert "retarget.MAXIMUM_GROUNDING_CORRECTION_M =" not in source
    assert "retarget.MAXIMUM_PENETRATION_M =" not in source


def test_source_reach_adjustment_is_bounded_below_pose_translation_tolerance():
    assert runner.FASTLANE_SOURCE_REACH_OVERSHOOT_MAX_M == 1.0e-4
    assert (
        runner.FASTLANE_SOURCE_REACH_OVERSHOOT_MAX_M
        <= runner.retarget.POSE_TRANSLATION_TOLERANCE_M
    )
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "source_endpoint_float_overshoot_clamped_inside_exact_reach" in source
    assert '"bounded_source_reach_adjustments": reach_adjustments' in source
