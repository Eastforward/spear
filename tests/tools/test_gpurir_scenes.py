"""Tests for gpurir_scenes.scene_spec."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from gpurir_scenes.scene_spec import (
    compose_scene, SceneSpec, AnimalPlacement,
    ANIMATED_TAGS, STATIC_TAGS, SOURCE_HEIGHT_M, N_FRAMES, check_no_clipping,
)


def test_seed_reproducible():
    a = compose_scene(seed=0)
    b = compose_scene(seed=0)
    assert a.animals[0].tag == b.animals[0].tag
    assert len(a.animals) == len(b.animals)
    if a.animals[0].is_animated:
        np.testing.assert_allclose(a.animals[0].trajectory_m, b.animals[0].trajectory_m)


def test_animal_count_is_one_or_two():
    for seed in range(20):
        spec = compose_scene(seed=seed)
        assert 1 <= len(spec.animals) <= 2


def test_animated_static_tag_partition():
    assert set(ANIMATED_TAGS).isdisjoint(STATIC_TAGS)
    assert set(ANIMATED_TAGS) >= {"cat_siamese_v1", "dog_pug_v1"}
    assert len(ANIMATED_TAGS) == 8
    assert len(STATIC_TAGS) == 7


def test_animated_has_trajectory_static_has_pos():
    for seed in range(10):
        spec = compose_scene(seed=seed)
        for a in spec.animals:
            if a.is_animated:
                assert a.trajectory_m is not None and a.trajectory_m.shape == (75, 3)
                assert a.yaw_deg is not None and a.yaw_deg.shape == (75,)
                assert a.static_pos_m is None
            else:
                assert a.static_pos_m is not None and len(a.static_pos_m) == 3
                assert a.static_yaw_deg is not None
                assert a.trajectory_m is None


def test_trajectory_within_wall_margin():
    for seed in range(20):
        spec = compose_scene(seed=seed)
        rx, ry, rz = spec.room_size_m
        for a in spec.animals:
            if a.trajectory_m is not None:
                xs = a.trajectory_m[:, 0]
                ys = a.trajectory_m[:, 1]
                assert xs.min() >= 0.5 and xs.max() <= rx - 0.5
                assert ys.min() >= 0.5 and ys.max() <= ry - 0.5
            elif a.static_pos_m is not None:
                x, y, _z = a.static_pos_m
                assert 0.5 <= x <= rx - 0.5
                assert 0.5 <= y <= ry - 0.5
                mx, my, _mz = spec.mic_pos_m
                assert ((x - mx) ** 2 + (y - my) ** 2) ** 0.5 >= 1.0


def test_uses_only_known_tags():
    known = set(ANIMATED_TAGS) | set(STATIC_TAGS)
    for seed in range(50):
        spec = compose_scene(seed=seed)
        for a in spec.animals:
            assert a.tag in known


def test_check_no_clipping_rejects_dog_footprint_near_wall():
    dog = AnimalPlacement(
        tag="dog_golden",
        is_animated=False,
        static_pos_m=(0.60, 2.2, SOURCE_HEIGHT_M),
        static_yaw_deg=0.0,
    )
    spec = SceneSpec(seed=123, animals=[dog])

    with pytest.raises(AssertionError, match="wall"):
        check_no_clipping(spec)


def test_check_no_clipping_allows_animated_paths_reusing_space_at_different_times():
    t = np.linspace(0.0, 1.0, N_FRAMES)
    path_a = np.column_stack([
        0.9 + 3.4 * t,
        np.full(N_FRAMES, 2.2),
        np.full(N_FRAMES, SOURCE_HEIGHT_M),
    ])
    path_b = np.column_stack([
        0.9 + 3.4 * ((t + 0.5) % 1.0),
        np.full(N_FRAMES, 2.2),
        np.full(N_FRAMES, SOURCE_HEIGHT_M),
    ])
    spec = SceneSpec(seed=456, animals=[
        AnimalPlacement(
            tag="dog_golden",
            is_animated=True,
            trajectory_m=path_a,
            yaw_deg=np.zeros(N_FRAMES),
        ),
        AnimalPlacement(
            tag="dog_beagle_v2",
            is_animated=True,
            trajectory_m=path_b,
            yaw_deg=np.zeros(N_FRAMES),
        ),
    ])

    check_no_clipping(spec)


def test_animal_placement_can_carry_runtime_actor_scale():
    human = AnimalPlacement(
        tag="human_male_blue_hoodie_v1",
        is_animated=True,
        trajectory_m=np.zeros((N_FRAMES, 3)),
        yaw_deg=np.zeros(N_FRAMES),
        wanted_anim="Standing_Idle",
        actor_scale=75.0,
        actor_z_lift_cm=0.0,
    )

    assert human.actor_scale == 75.0
    assert human.actor_z_lift_cm == 0.0


def test_render_helpers_prefer_placement_runtime_scale():
    from gpurir_scenes.run_render_pass import (
        _actor_scale_for_placement,
        _actor_z_lift_cm_for_placement,
    )

    human = AnimalPlacement(
        tag="human_male_blue_hoodie_v1",
        is_animated=True,
        trajectory_m=np.zeros((N_FRAMES, 3)),
        yaw_deg=np.zeros(N_FRAMES),
        actor_scale=75.0,
        actor_z_lift_cm=3.0,
    )
    dog = AnimalPlacement(
        tag="dog_golden",
        is_animated=True,
        trajectory_m=np.zeros((N_FRAMES, 3)),
        yaw_deg=np.zeros(N_FRAMES),
    )

    assert _actor_scale_for_placement(human) == 75.0
    assert _actor_z_lift_cm_for_placement(human) == 3.0
    assert _actor_scale_for_placement(dog) == 0.15
    assert _actor_z_lift_cm_for_placement(dog) == 0.0
