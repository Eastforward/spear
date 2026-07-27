import hashlib
from pathlib import Path

from tools.build_stable_animal_apartment_specs import build_pair
import pytest


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
        "status": "approved_for_automated_research_candidate_apartment",
        "human_visual_review": "pending",
        "formal_dataset_registration_authorized": False,
    }

    pair = build_pair(_template(), job=job, gate=gate)

    walking = pair["Walking"]["sources"][0]
    idle = pair["Idle"]["sources"][0]
    assert walking["walking_forward_yaw_offset_deg"] == 90.0
    assert walking["actor_scale"] == 0.15
    assert walking["audio_lookup"] == "dog_bark"
    assert walking["audio_contract"]["audio_lookup"] == "dog_bark"
    assert walking["audio_sha256"] == walking["audio_contract"]["sha256"]
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


@pytest.mark.parametrize(
    ("species", "tag"),
    [
        ("alpaca", "stable_alpaca_quaternius"),
        ("donkey_ass", "stable_donkey_ass_quaternius"),
    ],
)
def test_stable_explicit_silence_is_authenticated(
    species,
    tag,
    tmp_path,
):
    asset_id = f"quaternius_{species}"
    registry_path = tmp_path / "registry.json"
    import_path = tmp_path / "import.json"
    registry_path.write_text("{}")
    import_path.write_text("{}")

    def evidence(path):
        payload = Path(path).read_bytes()
        return {
            "path": str(Path(path).resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    gate = {
        "schema": "stable_animal_apartment_gate_v1",
        "status": "approved_for_automated_research_candidate_apartment",
        "asset_id": asset_id,
        "template_id": asset_id,
        "tag": tag,
        "species": species,
        "source_sha256": "a" * 64,
        "template_registry": evidence(registry_path),
        "ue_import_result": evidence(import_path),
        "formal_dataset_registration_authorized": False,
    }
    job = {
        "asset_id": asset_id,
        "template_id": asset_id,
        "tag": tag,
        "species": species,
        "breed": species,
        "actor_scale": 0.3,
        "audio_lookup": "silent",
        "audio_source_height_offset_m": 1.0,
        "walking_forward_yaw_offset_deg": 90.0,
    }

    pair = build_pair(_template(), job=job, gate=gate)
    for action in ("Walking", "Idle"):
        source = pair[action]["sources"][0]
        assert source["audio_lookup"] == "silent"
        assert source["mute_audio"] is True
        assert source["strict_audio"] is True
        assert source["audio_silence_contract"]["species"] == species
        assert source["audio_silence_contract"]["tag"] == tag
        assert "audio_contract" not in source


def test_stable_dog_cannot_select_explicit_silence():
    tag = "stable_dog_husky_silent"
    asset_id = "dog_silent"
    job = {
        "asset_id": asset_id,
        "template_id": asset_id,
        "tag": tag,
        "species": "dog",
        "breed": "husky",
        "actor_scale": 0.15,
        "audio_lookup": "silent",
        "audio_source_height_offset_m": 0.45,
        "walking_forward_yaw_offset_deg": 90.0,
    }
    gate = {
        "schema": "stable_animal_apartment_gate_v1",
        "status": "approved_for_automated_research_candidate_apartment",
        "asset_id": asset_id,
        "template_id": asset_id,
        "tag": tag,
        "species": "dog",
        "source_sha256": "a" * 64,
        "template_registry": {
            "path": "/fixture/registry.json",
            "sha256": "b" * 64,
            "size_bytes": 1,
        },
        "ue_import_result": {
            "path": "/fixture/import.json",
            "sha256": "c" * 64,
            "size_bytes": 1,
        },
        "formal_dataset_registration_authorized": False,
    }

    with pytest.raises(ValueError, match="cannot bypass"):
        build_pair(_template(), job=job, gate=gate)
