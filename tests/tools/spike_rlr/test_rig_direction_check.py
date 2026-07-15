"""Unit tests for rig_direction_check.py (offline path only).

Full integration tests that spawn a real SPEAR actor are in Task 8's
run_render_pass integration.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_write_and_read_calibration_roundtrip(tmp_path, monkeypatch):
    from rig_direction_check import (
        write_rig_calibration_json, read_rig_calibration_json,
    )
    calib_path = tmp_path / "rig_calibration.json"
    monkeypatch.setattr("rig_direction_check.CALIBRATION_FILE", calib_path)

    write_rig_calibration_json("dog_golden", offset_deg=180.0,
                                algorithm_version="rig_calib_v1")
    got = read_rig_calibration_json("dog_golden")
    assert got is not None
    assert got["walking_forward_yaw_offset_deg"] == 180.0
    assert got["algorithm_version"] == "rig_calib_v1"

    # Second write for a different tag preserves the first
    write_rig_calibration_json("dog_beagle_v2", offset_deg=170.0,
                                algorithm_version="rig_calib_v1")
    assert read_rig_calibration_json("dog_golden")["walking_forward_yaw_offset_deg"] == 180.0
    assert read_rig_calibration_json("dog_beagle_v2")["walking_forward_yaw_offset_deg"] == 170.0


def test_yaw_difference_within_tolerance():
    from rig_direction_check import _yaw_difference_deg
    assert abs(_yaw_difference_deg(10.0, 15.0)) == pytest.approx(5.0, abs=0.01)
    assert abs(_yaw_difference_deg(-170.0, 170.0)) == pytest.approx(20.0, abs=0.01)  # wrap
    assert abs(_yaw_difference_deg(0.0, 359.0)) == pytest.approx(1.0, abs=0.01)  # wrap
    assert abs(_yaw_difference_deg(45.0, 45.0)) == pytest.approx(0.0, abs=0.01)


def test_assert_yaw_ok_within_tolerance():
    from rig_direction_check import _assert_yaw_ok
    _assert_yaw_ok(observed=10.0, expected=15.0, tolerance_deg=15.0, context="test")
    # Should not raise


def test_assert_yaw_ok_raises_outside_tolerance():
    from rig_direction_check import _assert_yaw_ok
    with pytest.raises(AssertionError, match="test"):
        _assert_yaw_ok(observed=10.0, expected=90.0, tolerance_deg=15.0,
                        context="test")


def test_in_frame_api_exists():
    """Plan 1.5.B fix: sample_body_bone_position_in_frame + assert_body_yaw_from_positions
    must exist so the render loop can call them without opening its own begin_frame."""
    from rig_direction_check import (
        sample_body_bone_position_in_frame, find_body_bone_in_frame,
        assert_body_yaw_from_positions,
    )
    # Pure-math yaw assertion works without SPEAR
    import numpy as np
    # walking +Y direction (world) -> UE yaw ~ 90
    assert_body_yaw_from_positions(
        pos_start=np.array([0, 0, 0]), pos_end=np.array([0, 10, 0]),
        expected_yaw_world_deg=90.0, tolerance_deg=15.0, context="t",
    )
    # velocity too small -> silent skip
    assert_body_yaw_from_positions(
        pos_start=np.array([0, 0, 0]), pos_end=np.array([1e-5, 1e-5, 0]),
        expected_yaw_world_deg=90.0, tolerance_deg=15.0, context="t",
    )
    # opposite direction -> raises
    with pytest.raises(AssertionError):
        assert_body_yaw_from_positions(
            pos_start=np.array([0, 0, 0]), pos_end=np.array([0, -10, 0]),
            expected_yaw_world_deg=90.0, tolerance_deg=15.0, context="t",
        )


def test_in_frame_bone_query_uses_unreal_service_class_marshalling():
    from rig_direction_check import (
        find_body_bone_in_frame,
        sample_body_bone_position_in_frame,
    )

    class Actor:
        def GetComponentByClass(self, **_kwargs):
            raise AssertionError("unsafe dynamic UClass string marshalling was used")

    class Component:
        def GetNumBones(self):
            return 80

        def GetBoneIndex(self, BoneName):
            assert BoneName == "Root"
            return 0

        def GetBoneTransform(self, InBoneName, TransformSpace, as_dict):
            assert InBoneName == "Root"
            assert TransformSpace == "RTS_World"
            assert as_dict is True
            return {
                "ReturnValue": {
                    "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "scale3D": {"x": 1.0, "y": 1.0, "z": 1.0},
                }
            }

    actor = Actor()
    component = Component()

    class UnrealService:
        def get_components_by_class(self, *, actor, uclass):
            assert actor is actor_instance
            assert uclass == "USkeletalMeshComponent"
            return [component]

    actor_instance = actor
    service = UnrealService()

    np.testing.assert_allclose(
        sample_body_bone_position_in_frame(
            actor, "Root", unreal_service=service
        ),
        [1.0, 2.0, 3.0],
    )
    assert find_body_bone_in_frame(actor, unreal_service=service) == "Root"


def test_missing_bone_is_rejected_by_index_before_transform_query():
    from rig_direction_check import sample_body_bone_position_in_frame

    class Component:
        def GetNumBones(self):
            return 80

        def GetBoneIndex(self, BoneName):
            assert BoneName == "NotARealBone"
            return -1

        def GetBoneTransform(self, **_kwargs):
            raise AssertionError("missing bones must not query fallback transform")

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    diagnostics = []
    assert sample_body_bone_position_in_frame(
        object(),
        "NotARealBone",
        unreal_service=UnrealService(),
        diagnostics=diagnostics,
    ) is None
    assert diagnostics == [{
        "bone_name": "NotARealBone",
        "stage": "bone_lookup",
        "error_type": "MissingBone",
        "error": "GetBoneIndex returned -1",
    }]


def test_in_frame_bone_query_never_falls_back_to_unsafe_actor_uclass_call():
    from rig_direction_check import sample_body_bone_position_in_frame

    class Actor:
        called = False

        def GetComponentByClass(self, **_kwargs):
            self.called = True
            raise AssertionError("must not be called")

    actor = Actor()
    with pytest.raises(RuntimeError, match="unreal_service"):
        sample_body_bone_position_in_frame(actor, "Root")
    assert actor.called is False


def test_body_bone_candidates_include_rocketbox_bip01_root():
    from rig_direction_check import _BODY_BONE_CANDIDATES

    assert "Bip01" in _BODY_BONE_CANDIDATES
    assert "Bip01 Pelvis" in _BODY_BONE_CANDIDATES
    assert "Bip02" in _BODY_BONE_CANDIDATES
    assert "Bip02 Pelvis" in _BODY_BONE_CANDIDATES


def test_missing_body_bone_diagnostic_lists_actual_component_bones():
    from rig_direction_check import find_body_bone_in_frame

    class Component:
        bone_names = ["Armature", "Bip01-Pelvis", "Bip01-Spine"]

        def GetNumBones(self):
            return len(self.bone_names)

        def GetBoneIndex(self, BoneName):
            return -1

        def GetBoneName(self, BoneIndex):
            return self.bone_names[BoneIndex]

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    diagnostics = []
    assert find_body_bone_in_frame(
        object(),
        unreal_service=UnrealService(),
        diagnostics=diagnostics,
    ) is None
    assert diagnostics[-1] == {
        "stage": "bone_inventory",
        "available_bone_names": ["Armature", "Bip01-Pelvis", "Bip01-Spine"],
    }


def test_find_body_bone_matches_interchange_sanitized_rocketbox_name():
    from rig_direction_check import find_body_bone_in_frame

    class Component:
        bone_names = ["Bip01-Pelvis", "Bip01-Spine", "Bip01-Spine1"]

        def GetNumBones(self):
            return len(self.bone_names)

        def GetBoneName(self, BoneIndex):
            return self.bone_names[BoneIndex]

        def GetBoneIndex(self, BoneName):
            try:
                return self.bone_names.index(BoneName)
            except ValueError:
                return -1

        def GetBoneTransform(self, InBoneName, TransformSpace, as_dict):
            assert InBoneName == "Bip01-Pelvis"
            assert TransformSpace == "RTS_World"
            assert as_dict is True
            return {"translation": {"x": 1.0, "y": 2.0, "z": 3.0}}

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    assert find_body_bone_in_frame(
        object(),
        unreal_service=UnrealService(),
    ) == "Bip01-Pelvis"


def test_body_basis_from_bones_recovers_forward_and_upright_axes():
    from rig_direction_check import body_basis_from_positions

    # UE body convention is +X forward, +Y anatomical right, +Z up.
    basis = body_basis_from_positions(
        pelvis=[0.0, 0.0, 0.0],
        spine=[0.0, 0.0, 1.0],
        left_clavicle=[0.0, -1.0, 1.0],
        right_clavicle=[0.0, 1.0, 1.0],
    )

    np.testing.assert_allclose(basis["up_vector_ue"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(basis["right_vector_ue"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(basis["forward_vector_ue"], [1.0, 0.0, 0.0])
    assert basis["forward_yaw_ue_deg"] == pytest.approx(0.0)
    assert basis["up_alignment_z"] == pytest.approx(1.0)


def test_sample_body_basis_matches_sanitized_rocketbox_bones():
    from rig_direction_check import sample_body_basis_in_frame

    positions = {
        "Bip01-Pelvis": [0.0, 0.0, 100.0],
        "Bip01-Spine2": [0.0, 0.0, 150.0],
        "Bip01-L-Clavicle": [0.0, -20.0, 150.0],
        "Bip01-R-Clavicle": [0.0, 20.0, 150.0],
    }

    class Component:
        bone_names = list(positions)

        def GetNumBones(self):
            return len(self.bone_names)

        def GetBoneName(self, BoneIndex):
            return self.bone_names[BoneIndex]

        def GetBoneIndex(self, BoneName):
            return self.bone_names.index(BoneName)

        def GetBoneTransform(self, InBoneName, TransformSpace, as_dict):
            assert TransformSpace == "RTS_World"
            assert as_dict is True
            x, y, z = positions[InBoneName]
            return {"translation": {"x": x, "y": y, "z": z}}

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    basis = sample_body_basis_in_frame(
        object(),
        unreal_service=UnrealService(),
    )

    assert basis["bone_names"] == {
        "pelvis": "Bip01-Pelvis",
        "spine": "Bip01-Spine2",
        "left_clavicle": "Bip01-L-Clavicle",
        "right_clavicle": "Bip01-R-Clavicle",
    }
    assert basis["forward_yaw_ue_deg"] == pytest.approx(0.0)


def test_sample_body_basis_matches_sanitized_rocketbox_bip02_child_bones():
    from rig_direction_check import sample_body_basis_in_frame

    positions = {
        "Bip02-Pelvis": [0.0, 0.0, 80.0],
        "Bip02-Spine2": [0.0, 0.0, 120.0],
        "Bip02-L-Clavicle": [0.0, -15.0, 120.0],
        "Bip02-R-Clavicle": [0.0, 15.0, 120.0],
    }

    class Component:
        bone_names = list(positions)

        def GetNumBones(self):
            return len(self.bone_names)

        def GetBoneName(self, BoneIndex):
            return self.bone_names[BoneIndex]

        def GetBoneIndex(self, BoneName):
            return self.bone_names.index(BoneName)

        def GetBoneTransform(self, InBoneName, TransformSpace, as_dict):
            x, y, z = positions[InBoneName]
            return {"translation": {"x": x, "y": y, "z": z}}

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    basis = sample_body_basis_in_frame(
        object(), unreal_service=UnrealService()
    )

    assert basis["bone_names"]["pelvis"] == "Bip02-Pelvis"
    assert basis["forward_yaw_ue_deg"] == pytest.approx(0.0)


def test_sample_body_basis_matches_quaternius_numeric_dog_bones():
    from rig_direction_check import sample_body_basis_in_frame

    positions = {
        "bone": [0.0, 0.0, 100.0],
        "Bone_002": [100.0, 0.0, 105.0],
        "Bone_010": [-10.0, -30.0, 0.0],
        "Bone_013": [-10.0, 30.0, 0.0],
    }

    class Component:
        bone_names = list(positions)

        def GetNumBones(self):
            return len(self.bone_names)

        def GetBoneName(self, BoneIndex):
            return self.bone_names[BoneIndex]

        def GetBoneIndex(self, BoneName):
            return self.bone_names.index(BoneName)

        def GetBoneTransform(self, InBoneName, TransformSpace, as_dict):
            x, y, z = positions[InBoneName]
            return {"translation": {"x": x, "y": y, "z": z}}

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    basis = sample_body_basis_in_frame(
        object(), unreal_service=UnrealService()
    )

    assert basis["basis_kind"] == "quadruped_longitudinal_v1"
    assert basis["bone_names"] == {
        "rear": "bone",
        "front": "Bone_002",
        "body": "bone",
        "left_foot": "Bone_010",
        "right_foot": "Bone_013",
    }
    assert basis["forward_yaw_ue_deg"] == pytest.approx(0.0, abs=3.0)
    assert basis["up_alignment_z"] > 0.99


def test_sample_body_basis_matches_quaternius_native_named_husky_bones():
    from rig_direction_check import sample_body_basis_in_frame

    # The stable Ultimate Animal Pack Husky does not use the older numeric
    # Bone/Bone.002 convention.  Its longitudinal chain is
    # Back -> Torso* -> neck/head and the rear feet use explicit side names.
    positions = {
        "Body": [-20.0, 0.0, 25.0],
        "Back": [0.0, 0.0, 100.0],
        "Torso": [25.0, 0.0, 105.0],
        "Torso2": [60.0, 0.0, 108.0],
        "Torso3": [100.0, 0.0, 110.0],
        "BackLowerLeg_L_end": [-10.0, -30.0, 0.0],
        "BackLowerLeg_R_end": [-10.0, 30.0, 0.0],
    }

    class Component:
        bone_names = list(positions)

        def GetNumBones(self):
            return len(self.bone_names)

        def GetBoneName(self, BoneIndex):
            return self.bone_names[BoneIndex]

        def GetBoneIndex(self, BoneName):
            return self.bone_names.index(BoneName)

        def GetBoneTransform(self, InBoneName, TransformSpace, as_dict):
            x, y, z = positions[InBoneName]
            return {"translation": {"x": x, "y": y, "z": z}}

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    basis = sample_body_basis_in_frame(
        object(), unreal_service=UnrealService()
    )

    assert basis["basis_kind"] == "quaternius_native_named_longitudinal_v1"
    assert basis["bone_names"] == {
        "rear": "Back",
        "front": "Torso3",
        "body": "Back",
        "left_foot": "BackLowerLeg_L_end",
        "right_foot": "BackLowerLeg_R_end",
    }
    assert basis["forward_yaw_ue_deg"] == pytest.approx(0.0, abs=3.0)
    assert basis["up_alignment_z"] > 0.99


def test_sample_body_basis_matches_namespaced_rocketbox_beagle_bones():
    from rig_direction_check import sample_body_basis_in_frame

    # UE preserves the Rocketbox asset namespace with hyphens.  The torso
    # chain runs rear-to-front along +X and the paired hind feet establish the
    # ground/up plane.
    positions = {
        "beagle-Pelvis": [0.0, 0.0, 33.0],
        "beagle-Spine": [5.0, 0.0, 33.0],
        "beagle-Spine1": [13.0, 0.0, 33.0],
        "beagle-Spine2": [21.0, 0.0, 34.0],
        "beagle-L-Foot": [-12.0, -6.0, 10.0],
        "beagle-R-Foot": [-12.0, 6.0, 10.0],
        "beagle-Tail": [-4.0, 0.0, 34.0],
    }

    class Component:
        bone_names = list(positions)

        def GetNumBones(self):
            return len(self.bone_names)

        def GetBoneName(self, BoneIndex):
            return self.bone_names[BoneIndex]

        def GetBoneIndex(self, BoneName):
            return self.bone_names.index(BoneName)

        def GetBoneTransform(self, InBoneName, TransformSpace, as_dict):
            x, y, z = positions[InBoneName]
            return {"translation": {"x": x, "y": y, "z": z}}

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    basis = sample_body_basis_in_frame(
        object(), unreal_service=UnrealService()
    )

    assert basis["basis_kind"] == "prefixed_bip_quadruped_longitudinal_v1"
    assert basis["bone_names"] == {
        "rear": "beagle-Pelvis",
        "front": "beagle-Spine2",
        "body": "beagle-Pelvis",
        "left_foot": "beagle-L-Foot",
        "right_foot": "beagle-R-Foot",
    }
    assert basis["forward_yaw_ue_deg"] == pytest.approx(0.0, abs=3.0)
    assert basis["up_alignment_z"] > 0.85


def test_namespaced_bip_without_quadruped_tail_marker_is_not_auto_classified():
    from rig_direction_check import sample_body_basis_in_frame

    positions = {
        "avatar-Pelvis": [0.0, 0.0, 100.0],
        "avatar-Spine2": [0.0, 0.0, 130.0],
        "avatar-L-Foot": [0.0, -10.0, 0.0],
        "avatar-R-Foot": [0.0, 10.0, 0.0],
    }

    class Component:
        bone_names = list(positions)

        def GetNumBones(self):
            return len(self.bone_names)

        def GetBoneName(self, BoneIndex):
            return self.bone_names[BoneIndex]

        def GetBoneIndex(self, BoneName):
            return self.bone_names.index(BoneName)

        def GetBoneTransform(self, InBoneName, TransformSpace, as_dict):
            x, y, z = positions[InBoneName]
            return {"translation": {"x": x, "y": y, "z": z}}

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    diagnostics = []
    basis = sample_body_basis_in_frame(
        object(), unreal_service=UnrealService(), diagnostics=diagnostics
    )

    assert basis is None
    assert diagnostics[-1]["stage"] == "body_basis_bone_lookup"
    assert diagnostics[-1]["matched_prefixed_bip_roles"] == []


def test_sample_body_basis_matches_quaternius_farm_horse_bones():
    from rig_direction_check import sample_body_basis_in_frame

    positions = {
        "Hips": [0.0, 0.0, 105.0],
        "Shoulders": [100.0, 0.0, 110.0],
        "Body": [45.0, 0.0, 100.0],
        "BackFoot_L": [-10.0, -30.0, 0.0],
        "BackFoot_R": [-10.0, 30.0, 0.0],
    }

    class Component:
        bone_names = list(positions)

        def GetNumBones(self):
            return len(self.bone_names)

        def GetBoneName(self, BoneIndex):
            return self.bone_names[BoneIndex]

        def GetBoneIndex(self, BoneName):
            return self.bone_names.index(BoneName)

        def GetBoneTransform(self, InBoneName, TransformSpace, as_dict):
            x, y, z = positions[InBoneName]
            return {"translation": {"x": x, "y": y, "z": z}}

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            return [Component()]

    basis = sample_body_basis_in_frame(
        object(), unreal_service=UnrealService()
    )

    assert basis["basis_kind"] == "quadruped_longitudinal_v1"
    assert basis["bone_names"] == {
        "rear": "Hips",
        "front": "Shoulders",
        "body": "Hips",
        "left_foot": "BackFoot_L",
        "right_foot": "BackFoot_R",
    }
    assert basis["forward_yaw_ue_deg"] == pytest.approx(0.0, abs=3.0)
    assert basis["up_alignment_z"] > 0.95


def test_bone_query_records_rpc_or_parse_diagnostics_instead_of_hiding_them():
    from rig_direction_check import sample_body_bone_position_in_frame

    class UnrealService:
        def get_components_by_class(self, **_kwargs):
            raise ValueError("diagnostic failure")

    diagnostics = []
    assert sample_body_bone_position_in_frame(
        object(),
        "Bip01",
        unreal_service=UnrealService(),
        diagnostics=diagnostics,
    ) is None
    assert diagnostics == [{
        "bone_name": "Bip01",
        "stage": "query",
        "error_type": "ValueError",
        "error": "diagnostic failure",
    }]


def test_read_nonexistent_calibration_returns_none(tmp_path, monkeypatch):
    from rig_direction_check import read_rig_calibration_json
    calib_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr("rig_direction_check.CALIBRATION_FILE", calib_path)
    assert read_rig_calibration_json("anytag") is None
