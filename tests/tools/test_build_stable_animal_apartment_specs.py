from tools.build_stable_animal_apartment_specs import build_pair


def _template():
    trajectory = [[float(index), 0.0, 0.0] for index in range(6)]
    return {
        "sources": [
            {
                "tag": "template",
                "trajectory_m": trajectory,
                "start_pos_m": trajectory[0],
                "end_pos_m": trajectory[-1],
            }
        ],
        "camera_pass_table_loop_contract": {"left_front_nearest_frame": 2},
        "rig_direction_check_windows": [{"start_frame": 0, "end_frame": 5}],
        "trajectory_profile": "camera_pass_table_loop_v1",
    }


def test_stable_pair_keeps_cardinal_offset_and_builds_stationary_idle():
    job = {
        "asset_id": "quaternius_ultimate_husky_v1",
        "template_id": "quaternius_ultimate_husky_v1",
        "tag": "stable_dog_husky_quaternius_ultimate_husky_v1",
        "species": "dog",
        "breed": "husky",
        "actor_scale": 0.15,
        "audio_lookup": "dog_bark",
        "audio_source_height_offset_m": 0.45,
        "walking_forward_yaw_offset_deg": 90.0,
        "sampled_attributes": {"size": "small"},
        "fixed_attributes": {"coat_pattern": "tricolor"},
        "target_physical_profile": {"target_value_cm": 32.4},
    }
    gate = {
        "schema": "stable_animal_apartment_gate_v1",
        "human_visual_review": "pending",
        "formal_dataset_registration_authorized": False,
    }

    pair = build_pair(_template(), job=job, gate=gate)

    walking = pair["Walking"]["sources"][0]
    idle = pair["Idle"]["sources"][0]
    assert walking["walking_forward_yaw_offset_deg"] == 90.0
    assert walking["actor_scale"] == 0.15
    assert walking["audio_lookup"] == "dog_bark"
    assert walking["sampled_attributes"] == {"size": "small"}
    assert walking["fixed_attributes"] == {"coat_pattern": "tricolor"}
    assert walking["target_physical_profile"] == {"target_value_cm": 32.4}
    assert walking["ground_snap_max_abs_correction_cm"] == 30.0
    assert walking["stable_animal_gate"] == gate
    assert idle["wanted_anim"] == "Idle"
    assert idle["trajectory_m"] == [[2.0, 0.0, 0.0]] * 6
    assert "rig_direction_check_windows" not in pair["Idle"]

    meter_scale_job = dict(job, actor_scale=1.0)
    meter_scale = build_pair(_template(), job=meter_scale_job, gate=gate)
    assert (
        meter_scale["Walking"]["sources"][0][
            "ground_snap_max_abs_correction_cm"
        ]
        == 50.0
    )
