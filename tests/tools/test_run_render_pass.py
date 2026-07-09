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
    def __init__(self):
        self.tick_calls = []
        self.play_calls = []

    def SetComponentTickEnabled(self, bEnabled):
        self.tick_calls.append(bEnabled)

    def PlayAnimation(self, NewAnimToPlay, bLooping):
        self.play_calls.append((NewAnimToPlay, bLooping))


class FakeUnrealService:
    def __init__(self, smc):
        self.smc = smc
        self.loaded_objects = []

    def get_component_by_class(self, actor, uclass):
        assert uclass == "USkeletalMeshComponent"
        return self.smc

    def load_object(self, uclass, name):
        self.loaded_objects.append((uclass, name))
        return f"anim:{name}"


def test_play_anim_on_actor_explicitly_starts_walking_animation():
    smc = FakeSkeletalMeshComponent()
    unreal = FakeUnrealService(smc)
    game = SimpleNamespace(unreal_service=unreal)
    placement = SimpleNamespace(tag="dog_beagle_v2", wanted_anim="Walking")

    _play_anim_on_actor(game, actor=object(), placement=placement)

    anim_path = "/Game/MyAssets/Audioset/Meshes/gate_dog_beagle_v2/Walking"
    assert smc.tick_calls == [True]
    assert unreal.loaded_objects == [("UAnimationAsset", anim_path)]
    assert smc.play_calls == [(f"anim:{anim_path}", True)]
