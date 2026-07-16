"""Tests for gpurir_scenes.run_render_pass helpers."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from gpurir_scenes.run_render_pass import (
    CAMERA_FOV_DEG,
    _play_anim_on_actor,
    _set_capture_fov,
)


class FakeCaptureComponent:
    def __init__(self):
        self.calls = []
        self.value = None

    def set_property_value(self, property_name, property_value):
        self.calls.append((property_name, property_value))
        self.value = property_value

    def get_property_value(self, property_name):
        assert property_name == "FOVAngle"
        return self.value


def test_set_capture_fov_uses_spear_property_signature():
    comp = FakeCaptureComponent()

    _set_capture_fov(comp)

    assert comp.calls == [("FOVAngle", CAMERA_FOV_DEG)]


class FakeSkeletalMeshComponent:
    def __init__(self, bone_count=80, reported_play_rate=None):
        self.bone_count = bone_count
        self.reported_play_rate = reported_play_rate
        self.tick_calls = []
        self.play_calls = []
        self.property_calls = []

    def GetNumBones(self):
        return self.bone_count

    def SetComponentTickEnabled(self, bEnabled):
        self.tick_calls.append(bEnabled)

    def PlayAnimation(self, NewAnimToPlay, bLooping):
        self.play_calls.append((NewAnimToPlay, bLooping))

    def set_property_value(self, property_name, property_value):
        self.property_calls.append((property_name, property_value))

    def get_property_value(self, property_name):
        assert property_name == "GlobalAnimRateScale"
        if self.reported_play_rate is not None:
            return self.reported_play_rate
        return self.property_calls[-1][1]


class FakeUnrealService:
    def __init__(self, skeletal_components):
        self.skeletal_components = skeletal_components
        self.loaded_objects = []

    def get_components_by_class(self, actor, uclass):
        assert uclass == "USkeletalMeshComponent"
        return self.skeletal_components

    def load_object(self, uclass, name):
        self.loaded_objects.append((uclass, name))
        return f"anim:{name}"


def test_play_anim_on_actor_explicitly_starts_walking_animation():
    empty_smc = FakeSkeletalMeshComponent(bone_count=0)
    smc = FakeSkeletalMeshComponent()
    unreal = FakeUnrealService([empty_smc, smc])
    game = SimpleNamespace(unreal_service=unreal)
    placement = SimpleNamespace(
        tag="dog_beagle_v2",
        wanted_anim="Walking",
        animation_play_rate=0.65,
    )

    _play_anim_on_actor(game, actor=object(), placement=placement)

    anim_path = "/Game/MyAssets/Audioset/Meshes/gate_dog_beagle_v2/Walking"
    assert smc.tick_calls == [True]
    assert empty_smc.tick_calls == []
    assert unreal.loaded_objects == [("UAnimationAsset", anim_path)]
    assert smc.play_calls == [(f"anim:{anim_path}", True)]
    assert smc.property_calls == [("GlobalAnimRateScale", 0.65)]


def test_play_anim_on_actor_rejects_play_rate_readback_mismatch():
    smc = FakeSkeletalMeshComponent(reported_play_rate=1.0)
    game = SimpleNamespace(unreal_service=FakeUnrealService([smc]))
    placement = SimpleNamespace(
        tag="human_test",
        wanted_anim="Walking",
        animation_play_rate=0.65,
    )

    import pytest

    with pytest.raises(RuntimeError, match="GlobalAnimRateScale"):
        _play_anim_on_actor(game, actor=object(), placement=placement)
