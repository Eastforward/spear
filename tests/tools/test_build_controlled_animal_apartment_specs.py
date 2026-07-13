from tools.build_controlled_animal_apartment_specs import PROFILE, build_pair


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


def _attempt(profile_schema_id, *, size, ratio, target_cm):
    return {
        "asset_id": "animal_example",
        "profile_schema_id": profile_schema_id,
        "sampled_attributes": {"body_build": "standard", "size": size},
        "target_physical_profile": {
            "control_attribute": "size",
            "measurement": "shoulder_height_cm",
            "scale_ratio": ratio,
            "target_value_cm": target_cm,
        },
    }


def _job(profile_schema_id):
    return {
        "legacy_tag": "animal_example",
        "tag": "pixal_animal_example",
        "profile_schema_id": profile_schema_id,
    }


def test_cat_scale_uses_sampled_physical_ratio_and_idle_is_stationary():
    attempt = _attempt(
        "cat_siamese_bindpose_v2", size="small", ratio=0.9, target_cm=22.5
    )

    pair = build_pair(
        _template(), job=_job("cat_siamese_bindpose_v2"), attempt=attempt
    )

    walking = pair["Walking"]["sources"][0]
    idle = pair["Idle"]["sources"][0]
    assert walking["actor_scale"] == 0.081
    assert walking["species"] == "cat"
    assert walking["audio_lookup"] == "cat_meow"
    assert walking["sampled_attributes"]["size"] == "small"
    assert walking["ground_snap_max_abs_correction_cm"] == 25.0
    assert idle["wanted_anim"] == "Idle"
    assert idle["ground_snap_max_abs_correction_cm"] == 25.0
    assert idle["trajectory_m"] == [[2.0, 0.0, 0.0]] * 6
    assert "rig_direction_check_windows" not in pair["Idle"]


def test_large_golden_scale_is_breed_base_times_ratio():
    attempt = _attempt(
        "dog_golden_retriever_v1", size="large", ratio=1.1, target_cm=60.5
    )

    pair = build_pair(
        _template(), job=_job("dog_golden_retriever_v1"), attempt=attempt
    )

    source = pair["Walking"]["sources"][0]
    rationale = pair["Walking"]["camera_pass_table_loop_contract"][
        "animal_scale_rationale"
    ]
    assert source["actor_scale"] == 0.143
    assert source["ground_snap_max_abs_correction_cm"] == 28.6
    assert source["breed"] == "golden_retriever"
    assert source["audio_lookup"] == "dog_bark"
    assert rationale["base_actor_scale"] == PROFILE["dog_golden_retriever_v1"][
        "base_actor_scale"
    ]
    assert rationale["physical_scale_ratio"] == 1.1
