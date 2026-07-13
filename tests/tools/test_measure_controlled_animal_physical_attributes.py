from tools.measure_controlled_animal_physical_attributes import (
    build_runtime_measurement,
    quantile,
    summarize_size_ordering,
)


def _record(size="medium", profile="dog_golden_retriever_v1"):
    return {
        "base_avatar_id": f"dog_{size}",
        "tag": f"pixal_dog_{size}",
        "profile_schema_id": profile,
        "sampled_attributes": {"size": size},
        "target_physical_profile": {
            "control_attribute": "size",
            "measurement": "shoulder_height_cm",
            "target_value_cm": 50.0,
            "tolerance_cm": 3.0,
            "reference_provenance": {"status": "provisional"},
        },
    }


def _visual(tag, height=80.0, scale=0.13):
    frame = {
        "bounds_ue": {"minimum_cm": [0, 0, 27.1], "maximum_cm": [90, 30, 27.1 + height]},
        "root_transform_ue": {"scale": [scale, scale, scale]},
        "floor_contact": {"within_penetration_tolerance": True},
    }
    return {
        "automatic_checks": {"overall": "passed"},
        "sources": [{"tag": tag, "runtime_frames": [frame, frame]}],
    }


def test_runtime_measurement_uses_geometry_ratio_not_prompt_target():
    record = _record()
    result = build_runtime_measurement(
        record=record,
        walking_spec={"sources": [{"tag": record["tag"], "actor_scale": 0.13}]},
        visual=_visual(record["tag"]),
        geometry={
            "bounds_height_units": 4.0,
            "shoulder_height_units": 2.5,
            "nose_to_tail_length_units": 5.0,
        },
    )

    runtime = result["physical_measurements"]["runtime"]
    assert runtime == {
        "actor_scale": 0.13,
        "shoulder_height_cm": 50.0,
        "bounds_height_cm": 80.0,
        "nose_to_tail_length_cm": 100.0,
    }
    assert result["target_comparison"]["status"] == "within_tolerance"


def test_size_ordering_uses_observed_shoulder_heights():
    rows = []
    for size, height in (("large", 60.0), ("small", 40.0), ("medium", 50.0)):
        rows.append(
            {
                "profile_schema_id": "dog_example_v1",
                "sampled_size": size,
                "physical_measurements": {
                    "runtime": {"shoulder_height_cm": height}
                },
            }
        )

    summary = summarize_size_ordering(rows)

    assert summary[0]["ordered_sizes"] == ["small", "medium", "large"]
    assert summary[0]["strictly_increasing"] is True
    assert quantile([0.0, 10.0], 0.25) == 2.5
