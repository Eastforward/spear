import json
from pathlib import Path

from tools.build_rocketbox_batch_apartment_specs import build_avatar_specs


def test_specs_preserve_authored_height_and_pair_walk_idle():
    avatar = {
        "base_avatar_id": "rocketbox_children_female_child_01",
        "legacy_asset_id": "rocketbox_female_child_01",
        "category": "Children",
        "demographic": "child",
        "gender": "female",
        "inventory_status": "passed",
        "height_contract": {
            "status": "passed",
            "actor_scale": 1.0,
            "authored_height_cm": 143.3028,
            "allowed_height_cm": [80.0, 170.0],
            "apartment_ceiling_cm": 280.0,
            "minimum_ceiling_headroom_cm": 25.0,
            "ceiling_headroom_cm": 136.6972,
            "mouth_audio_height_cm": 126.1065,
        },
    }
    tag = f"{avatar['base_avatar_id']}_original_ue_v1"
    ue_manifest = {
        "schema": "rocketbox_batch_native_ue_import_v1",
        "base_avatar_id": avatar["base_avatar_id"],
        "asset_id": avatar["legacy_asset_id"],
        "tag": tag,
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "reload_verification": {"status": "passed"},
        "runtime_contract": {
            "actor_scale": 1.0,
            "bone_count": 80,
            "bounds": {
                "height_cm": 142.9338,
                "authored_height_cm": 143.3028,
                "authored_height_preserved": True,
                "height_passed": True,
                "ground_passed": True,
            },
        },
        "content": {
            "animations": {"Walking": "/Game/Walking", "Standing_Idle": "/Game/Idle"},
            "blueprint": f"/Game/BP_gate_{tag}",
        },
    }

    specs = build_avatar_specs(avatar, ue_manifest, fps=15, duration_s=5.0)

    assert set(specs) == {"Walking", "Standing_Idle"}
    walking = specs["Walking"]
    idle = specs["Standing_Idle"]
    assert walking["render_config"]["n_frames"] == 75
    assert idle["render_config"]["n_frames"] == 75
    for spec in specs.values():
        source = spec["sources"][0]
        assert source["tag"] == tag
        assert source["asset_id"] == "rocketbox_female_child_01"
        assert source["actor_scale"] == 1.0
        assert source["authored_height_cm"] == 143.3028
        assert source["expected_ue_height_cm"] == 142.9338
        assert source["expected_height_range_cm"] == [80.0, 170.0]
        assert source["audio_source_height_offset_m"] == 1.261065
        assert source["ground_snap_to_floor"] is True
        assert spec["apartment_height_contract"]["ceiling_cm"] == 280.0
        assert spec["apartment_height_contract"]["minimum_headroom_cm"] == 25.0
    assert walking["sources"][0]["kind"] == "moving"
    assert walking["sources"][0]["wanted_anim"] == "Walking"
    assert walking["sources"][0]["start_pos_m"] != walking["sources"][0]["end_pos_m"]
    assert idle["sources"][0]["kind"] == "stationary"
    assert idle["sources"][0]["wanted_anim"] == "Standing_Idle"
    assert idle["sources"][0]["start_pos_m"] == idle["sources"][0]["end_pos_m"]
