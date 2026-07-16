from tools.build_pixal_animal_walk_idle_apartment_specs import PROFILE, build_pair


def test_walk_moves_and_idle_stays_at_left_front():
    trajectory = [[float(i), 0.0, 0.0] for i in range(6)]
    template = {
        "description": "template",
        "sources": [{"trajectory_m": trajectory}],
        "camera_pass_table_loop_contract": {"left_front_nearest_frame": 2},
        "rig_direction_check_windows": [{"label": "moving"}],
    }
    pair = build_pair(
        template,
        ue_tag="pixal_cat_tabby",
        legacy_tag="cat_tabby",
    )

    assert set(PROFILE) == {
        "cat_siamese_v1",
        "cat_tabby",
        "dog_beagle_v2",
        "dog_golden",
    }
    assert pair["walking"]["sources"][0]["wanted_anim"] == "Walking"
    assert pair["idle"]["sources"][0]["wanted_anim"] == "Idle"
    assert pair["idle"]["sources"][0]["trajectory_m"] == [[2.0, 0.0, 0.0]] * 6
    assert pair["idle"]["rig_direction_check_windows"] == []
