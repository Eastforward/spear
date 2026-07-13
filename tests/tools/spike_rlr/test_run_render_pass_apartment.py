import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _scene_with_point(tag, xyz):
    traj = np.asarray([xyz], dtype=np.float64)
    return SimpleNamespace(animals=[SimpleNamespace(tag=tag, trajectory_m=traj)])


class _RuntimeActor:
    def __init__(self, *, bottom_z_cm=29.0):
        self._bottom_z_cm = float(bottom_z_cm)

    def GetActorBounds(self, bOnlyCollidingComponents, as_dict):
        assert bOnlyCollidingComponents is False
        assert as_dict is True
        return {
            "Origin": {"x": 110.0, "y": -220.0, "z": self._bottom_z_cm + 90.0},
            "BoxExtent": {"x": 30.0, "y": 25.0, "z": 90.0},
        }

    def K2_GetActorLocation(self, as_dict):
        assert as_dict is True
        return {"ReturnValue": {"x": 100.0, "y": -200.0, "z": 29.0}}

    def K2_GetActorRotation(self, as_dict):
        assert as_dict is True
        return {"ReturnValue": {"roll": 0.0, "pitch": 0.0, "yaw": -136.0}}

    def GetActorScale3D(self, as_dict):
        assert as_dict is True
        return {"ReturnValue": {"x": 1.0, "y": 1.0, "z": 1.0}}


class _GroundSnapActor(_RuntimeActor):
    def __init__(self, *, bottom_z_cm, location_z_cm=29.0):
        super().__init__(bottom_z_cm=bottom_z_cm)
        self._location = [100.0, -200.0, float(location_z_cm)]
        self._rotation = [0.0, 0.0, -136.0]
        self.calls = []

    def K2_GetActorLocation(self, as_dict):
        assert as_dict is True
        return {
            "ReturnValue": {
                "x": self._location[0],
                "y": self._location[1],
                "z": self._location[2],
            }
        }

    def K2_GetActorRotation(self, as_dict):
        assert as_dict is True
        return {
            "ReturnValue": {
                "roll": self._rotation[0],
                "pitch": self._rotation[1],
                "yaw": self._rotation[2],
            }
        }

    def K2_SetActorLocationAndRotation(
        self, *, NewLocation, NewRotation, bSweep, bTeleport
    ):
        assert bSweep is False
        assert bTeleport is True
        old_z = self._location[2]
        new_z = float(NewLocation["Z"])
        self._bottom_z_cm += new_z - old_z
        self._location = [
            float(NewLocation["X"]),
            float(NewLocation["Y"]),
            new_z,
        ]
        self._rotation = [
            float(NewRotation["Roll"]),
            float(NewRotation["Pitch"]),
            float(NewRotation["Yaw"]),
        ]
        self.calls.append((dict(NewLocation), dict(NewRotation)))
        return True


def test_actor_runtime_frame_state_records_bounds_root_animation_and_floor_contact():
    from run_render_pass_apartment import _actor_runtime_frame_state

    placement = SimpleNamespace(
        tag="hy3d_rocketbox_male_adult_01_spike",
        wanted_anim="Walking",
    )
    state = _actor_runtime_frame_state(
        _RuntimeActor(bottom_z_cm=29.0),
        placement,
        frame_i=7,
        floor_z_cm=29.0,
        penetration_tolerance_cm=1.0,
    )

    assert state["frame_index"] == 7
    assert state["animation_name"] == "Walking"
    assert state["root_transform_ue"]["location_cm"] == [100.0, -200.0, 29.0]
    assert state["root_transform_ue"]["rotation_deg"] == [0.0, 0.0, -136.0]
    assert state["root_transform_ue"]["scale"] == [1.0, 1.0, 1.0]
    assert state["bounds_ue"]["minimum_cm"] == [80.0, -245.0, 29.0]
    assert state["bounds_ue"]["maximum_cm"] == [140.0, -195.0, 209.0]
    assert state["floor_contact"] == {
        "floor_z_ue_cm": 29.0,
        "bounds_bottom_z_ue_cm": 29.0,
        "clearance_cm": 0.0,
        "penetration_cm": 0.0,
        "penetration_tolerance_cm": 1.0,
        "within_penetration_tolerance": True,
    }


def test_runtime_summary_fails_human_floor_gate_over_one_centimeter():
    from run_render_pass_apartment import (
        _actor_runtime_frame_state,
        _summarize_actor_runtime_frames,
    )

    placement = SimpleNamespace(
        tag="hy3d_rocketbox_female_adult_01_spike",
        wanted_anim="Standing_Idle",
    )
    states = [
        _actor_runtime_frame_state(
            _RuntimeActor(bottom_z_cm=29.0 - penetration_cm),
            placement,
            frame_i=i,
            floor_z_cm=29.0,
            penetration_tolerance_cm=1.0,
        )
        for i, penetration_cm in enumerate((0.0, 0.4, 1.2))
    ]

    summary = _summarize_actor_runtime_frames(
        states,
        expected_frame_count=3,
        penetration_tolerance_cm=1.0,
    )

    assert summary["frame_count"] == 3
    assert summary["maximum_floor_penetration_cm"] == pytest.approx(1.2)
    assert summary["floor_contact_status"] == "failed"


@pytest.mark.parametrize(
    ("bottom_before_cm", "expected_correction_cm"),
    [(20.0, 9.0), (31.5, -2.5)],
)
def test_dynamic_ground_snap_moves_only_actor_z_and_verifies_bounds(
    bottom_before_cm, expected_correction_cm
):
    from run_render_pass_apartment import _snap_actor_bounds_bottom_to_floor

    actor = _GroundSnapActor(bottom_z_cm=bottom_before_cm)
    evidence = _snap_actor_bounds_bottom_to_floor(
        actor,
        floor_z_cm=29.0,
        maximum_abs_correction_cm=12.0,
    )

    assert evidence["status"] == "passed"
    assert evidence["bounds_bottom_before_cm"] == pytest.approx(bottom_before_cm)
    assert evidence["applied_z_correction_cm"] == pytest.approx(
        expected_correction_cm
    )
    assert evidence["bounds_bottom_after_cm"] == pytest.approx(29.0)
    assert evidence["residual_clearance_cm"] == pytest.approx(0.0)
    assert actor._location == pytest.approx(
        [100.0, -200.0, 29.0 + expected_correction_cm]
    )
    assert actor._rotation == pytest.approx([0.0, 0.0, -136.0])


def test_dynamic_ground_snap_rejects_implausible_bounds_correction():
    from run_render_pass_apartment import _snap_actor_bounds_bottom_to_floor

    actor = _GroundSnapActor(bottom_z_cm=-50.0)
    with pytest.raises(AssertionError, match="ground snap correction"):
        _snap_actor_bounds_bottom_to_floor(
            actor,
            floor_z_cm=29.0,
            maximum_abs_correction_cm=12.0,
        )
    assert actor.calls == []


def test_visual_metadata_embeds_runtime_frames_and_enforces_human_floor_gate():
    from run_render_pass_apartment import (
        _actor_runtime_frame_state,
        _build_actor_visual_metadata,
    )

    placement = SimpleNamespace(
        tag="hy3d_rocketbox_male_adult_01_spike",
        wanted_anim="Walking",
    )
    scene = SimpleNamespace(animals=[placement])
    states = [
        _actor_runtime_frame_state(
            _RuntimeActor(bottom_z_cm=29.0),
            placement,
            frame_i=i,
            floor_z_cm=29.0,
        )
        for i in range(2)
    ]
    payload = _build_actor_visual_metadata(
        scene=scene,
        visual_centers={placement.tag: [[0.0, 0.0, 0.9]] * 2},
        runtime_frames={placement.tag: states},
        human_gate_evidence=[{"tag": placement.tag}],
        n_frames=2,
    )

    source = payload["sources"][0]
    assert source["tag"] == placement.tag
    assert source["runtime_frames"] == states
    assert source["runtime_summary"]["floor_contact_status"] == "passed"
    assert payload["automatic_checks"]["overall"] == "passed"

    bad_states = list(states)
    bad_states[-1] = _actor_runtime_frame_state(
        _RuntimeActor(bottom_z_cm=27.8),
        placement,
        frame_i=1,
        floor_z_cm=29.0,
    )
    with pytest.raises(AssertionError, match="floor penetration"):
        _build_actor_visual_metadata(
            scene=scene,
            visual_centers={placement.tag: [[0.0, 0.0, 0.9]] * 2},
            runtime_frames={placement.tag: bad_states},
            human_gate_evidence=[{"tag": placement.tag}],
            n_frames=2,
        )


def test_walls_only_center_collision_policy_allows_furniture_center_points():
    from run_render_pass_apartment import _check_no_clipping_apartment
    from scene_two_dogs_apartment import _kept_furniture_bboxes

    spec = json.loads((REPO / "data" / "apartment_v1_spec.json").read_text())
    cats = json.loads(
        (REPO / "tools/spike_rlr/apartment_furniture_categories.json").read_text()
    )
    x0, y0, x1, y1 = _kept_furniture_bboxes(spec, cats)[0]
    scene = _scene_with_point("dog_beagle_v2", [(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.45])

    with pytest.raises(AssertionError):
        _check_no_clipping_apartment(spec, scene, cats)

    spec["source_collision_policy"] = "walls_only_center"
    _check_no_clipping_apartment(spec, scene, cats)


def test_walls_only_center_collision_policy_still_rejects_wall_points():
    from run_render_pass_apartment import _check_no_clipping_apartment
    from scene_two_dogs_apartment import _shell_wall_bboxes

    spec = json.loads((REPO / "data" / "apartment_v1_spec.json").read_text())
    spec["source_collision_policy"] = "walls_only_center"
    cats = json.loads(
        (REPO / "tools/spike_rlr/apartment_furniture_categories.json").read_text()
    )
    x0, y0, x1, y1 = _shell_wall_bboxes(spec)[0]
    scene = _scene_with_point("dog_beagle_v2", [(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.45])

    with pytest.raises(AssertionError):
        _check_no_clipping_apartment(spec, scene, cats)


def test_source_separation_gate_reports_pairwise_minimum_distance():
    from run_render_pass_apartment import _check_source_separation

    scene = SimpleNamespace(animals=[
        SimpleNamespace(
            tag="human_walk",
            trajectory_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ),
        SimpleNamespace(
            tag="human_idle",
            trajectory_m=np.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        ),
    ])

    report = _check_source_separation(
        {"minimum_source_separation_m": 0.75},
        scene,
    )

    assert report["status"] == "passed"
    assert report["minimum_required_m"] == pytest.approx(0.75)
    assert report["pairs"] == [{
        "tags": ["human_walk", "human_idle"],
        "minimum_distance_m": pytest.approx(1.0),
        "frame_index": 0,
    }]


def test_source_separation_gate_rejects_intersecting_human_trajectories():
    from run_render_pass_apartment import _check_source_separation

    scene = SimpleNamespace(animals=[
        SimpleNamespace(
            tag="male",
            trajectory_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        ),
        SimpleNamespace(
            tag="female",
            trajectory_m=np.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        ),
    ])

    with pytest.raises(AssertionError, match="male/female.*frame 1"):
        _check_source_separation(
            {"minimum_source_separation_m": 0.75},
            scene,
        )


def test_rig_direction_evidence_records_moving_source_and_stationary_n_a():
    from run_render_pass_apartment import _build_rig_direction_evidence

    moving = SimpleNamespace(
        tag="male",
        wanted_anim="Walking",
        yaw_deg=np.asarray([90.0, 90.0, 90.0]),
        trajectory_m=np.asarray([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ]),
    )
    idle = SimpleNamespace(
        tag="female",
        wanted_anim="Standing_Idle",
        yaw_deg=np.asarray([0.0, 0.0, 0.0]),
        trajectory_m=np.asarray([[1.0, 0.0, 0.0]] * 3),
    )

    evidence = _build_rig_direction_evidence(
        SimpleNamespace(animals=[moving, idle]),
        {
            "male": {
            "bone": "Bip01",
            "a": np.asarray([0.0, 0.0, 100.0]),
            "b": np.asarray([0.0, -100.0, 100.0]),
            "body_basis": {
                "bone_names": {"pelvis": "Bip01-Pelvis"},
                "forward_yaw_ue_deg": -90.0,
                "up_alignment_z": 1.0,
                "forward_vector_ue": [0.0, -1.0, 0.0],
            },
        },
        "female": {
            "body_basis": {
                "bone_names": {"pelvis": "Bip01-Pelvis"},
                "forward_yaw_ue_deg": 0.0,
                "up_alignment_z": 1.0,
                "forward_vector_ue": [1.0, 0.0, 0.0],
            },
        },
        },
        frame_a=0,
        frame_b=2,
        enabled=True,
        tolerance_deg=25.0,
    )

    assert evidence["male"]["status"] == "passed"
    assert evidence["male"]["bone_name"] == "Bip01"
    assert evidence["male"]["expected_yaw_ue_deg"] == pytest.approx(-90.0)
    assert evidence["male"]["observed_yaw_ue_deg"] == pytest.approx(-90.0)
    assert evidence["male"]["body_forward_yaw_ue_deg"] == pytest.approx(-90.0)
    assert evidence["male"]["body_forward_error_deg"] == pytest.approx(0.0)
    assert evidence["male"]["root_motion_status"] == "passed"
    assert evidence["female"]["status"] == "passed"
    assert evidence["female"]["body_forward_yaw_ue_deg"] == pytest.approx(0.0)
    assert evidence["female"]["root_motion_status"] == "not_applicable"


def test_rig_direction_uses_actor_root_instead_of_quadruped_gait_sway():
    from run_render_pass_apartment import _build_rig_direction_evidence

    moving = SimpleNamespace(
        tag="dog",
        wanted_anim="Walking",
        yaw_deg=np.asarray([0.0, 0.0]),
        trajectory_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    evidence = _build_rig_direction_evidence(
        SimpleNamespace(animals=[moving]),
        {
            "dog": {
                "bone": "bone",
                # Gait sway points the body bone along +Y and must not govern
                # the authored actor-root trajectory.
                "a": np.asarray([0.0, 0.0, 50.0]),
                "b": np.asarray([0.0, 30.0, 50.0]),
                "root_a": [0.0, 0.0, 0.0],
                "root_b": [100.0, 0.0, 0.0],
                "body_basis": {
                    "forward_yaw_ue_deg": 0.0,
                    "up_alignment_z": 1.0,
                    "forward_vector_ue": [1.0, 0.0, 0.0],
                },
            }
        },
        frame_a=0,
        frame_b=1,
        enabled=True,
    )

    assert evidence["dog"]["status"] == "passed"
    assert evidence["dog"]["translation_sample_kind"] == "actor_root_transform"
    assert evidence["dog"]["observed_yaw_ue_deg"] == pytest.approx(0.0)


def test_rig_direction_evidence_refuses_missing_moving_bone_samples():
    from run_render_pass_apartment import _build_rig_direction_evidence

    moving = SimpleNamespace(
        tag="male",
        wanted_anim="Walking",
        yaw_deg=np.asarray([0.0, 0.0]),
        trajectory_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    with pytest.raises(AssertionError, match="male.*missing rig direction"):
        _build_rig_direction_evidence(
            SimpleNamespace(animals=[moving]),
            {"male": {
                "body_basis": {
                    "bone_names": {"pelvis": "Bip01-Pelvis"},
                    "forward_yaw_ue_deg": 0.0,
                    "up_alignment_z": 1.0,
                    "forward_vector_ue": [1.0, 0.0, 0.0],
                },
            }},
            frame_a=0,
            frame_b=1,
            enabled=True,
        )


def test_rig_direction_evidence_includes_body_basis_when_facing_mismatches():
    from run_render_pass_apartment import _build_rig_direction_evidence

    idle = SimpleNamespace(
        tag="female",
        wanted_anim="Standing_Idle",
        yaw_deg=np.asarray([0.0, 0.0]),
        trajectory_m=np.asarray([[0.0, 0.0, 0.0]] * 2),
    )
    body_basis = {
        "bone_names": {"pelvis": "Bip01-Pelvis"},
        "positions_ue_cm": {"pelvis": [0.0, 0.0, 100.0]},
        "forward_yaw_ue_deg": 90.0,
        "up_alignment_z": 1.0,
        "forward_vector_ue": [0.0, 1.0, 0.0],
    }

    with pytest.raises(AssertionError, match="body_basis=.*forward_vector_ue"):
        _build_rig_direction_evidence(
            SimpleNamespace(animals=[idle]),
            {"female": {"body_basis": body_basis}},
            frame_a=0,
            frame_b=1,
            enabled=True,
            tolerance_deg=25.0,
        )


def test_rig_direction_evidence_removes_asset_yaw_offset_for_semantic_facing():
    from run_render_pass_apartment import _build_rig_direction_evidence

    idle = SimpleNamespace(
        tag="female",
        wanted_anim="Standing_Idle",
        yaw_deg=np.asarray([-60.0, -60.0]),
        walking_forward_yaw_offset_deg=-90.0,
        trajectory_m=np.asarray([[0.0, 0.0, 0.0]] * 2),
    )
    body_basis = {
        "bone_names": {"pelvis": "Bip01-Pelvis"},
        "forward_yaw_ue_deg": -30.0,
        "up_alignment_z": 1.0,
        "forward_vector_ue": [0.8660254, -0.5, 0.0],
    }

    evidence = _build_rig_direction_evidence(
        SimpleNamespace(animals=[idle]),
        {"female": {"body_basis": body_basis}},
        frame_a=0,
        frame_b=1,
        enabled=True,
        tolerance_deg=25.0,
    )

    assert evidence["female"]["expected_yaw_ue_deg"] == pytest.approx(-30.0)
    assert evidence["female"]["body_forward_error_deg"] == pytest.approx(0.0)


def test_rig_direction_window_evidence_checks_both_sides_of_turnaround():
    from run_render_pass_apartment import _build_rig_direction_window_evidence

    moving = SimpleNamespace(
        tag="male",
        wanted_anim="Walking",
        walking_forward_yaw_offset_deg=90.0,
        yaw_deg=np.asarray([90.0, 90.0, 90.0, 270.0, 270.0]),
        trajectory_m=np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]),
    )
    windows = [
        {"label": "outbound", "frame_a": 0, "frame_b": 1},
        {"label": "return", "frame_a": 3, "frame_b": 4},
    ]
    samples = {
        "outbound": {
            "male": {
                "bone": "Bip01",
                "a": np.asarray([0.0, 0.0, 100.0]),
                "b": np.asarray([100.0, 0.0, 100.0]),
                "body_basis": {
                    "forward_yaw_ue_deg": 0.0,
                    "up_alignment_z": 1.0,
                    "forward_vector_ue": [1.0, 0.0, 0.0],
                },
            },
        },
        "return": {
            "male": {
                "bone": "Bip01",
                "a": np.asarray([50.0, 0.0, 100.0]),
                "b": np.asarray([0.0, 0.0, 100.0]),
                "body_basis": {
                    "forward_yaw_ue_deg": 180.0,
                    "up_alignment_z": 1.0,
                    "forward_vector_ue": [-1.0, 0.0, 0.0],
                },
            },
        },
    }

    evidence = _build_rig_direction_window_evidence(
        SimpleNamespace(animals=[moving]),
        samples,
        windows=windows,
        enabled=True,
    )

    assert evidence["male"]["status"] == "passed"
    assert [window["label"] for window in evidence["male"]["windows"]] == [
        "outbound",
        "return",
    ]
    assert [window["sample_frames"] for window in evidence["male"]["windows"]] == [
        [0, 1],
        [3, 4],
    ]
    assert all(window["root_motion_status"] == "passed" for window in evidence["male"]["windows"])
    assert evidence["male"]["windows"][1]["body_forward_yaw_ue_deg"] == pytest.approx(180.0)
