import copy
import json
from pathlib import Path

from tools.build_pixal_animal_camera_pass_table_loop_spec import TAG, build_spec


def _template():
    path = Path(
        "external/SPEAR/tmp/rocketbox_camera_pass_table_loop_apartment_review_v2/specs/"
        "rocketbox_adults_male_adult_01_original_ue_v1/"
        "camera_pass_table_loop_walking.json"
    )
    return json.loads(path.read_text())


def test_pixal_dog_reuses_exact_review_path_without_human_identity_fields():
    template = _template()
    original = copy.deepcopy(template)

    spec = build_spec(template)

    assert template == original
    assert spec["sources"][0]["tag"] == TAG
    assert spec["sources"][0]["trajectory_m"] == original["sources"][0]["trajectory_m"]
    assert spec["sources"][0]["actor_scale"] == 0.15
    assert spec["sources"][0]["walking_forward_yaw_offset_deg"] == 0.0
    assert spec["sources"][0]["audio_lookup"] == "dog_bark"
    assert spec["sources"][0]["adaptive_repeat_short_calls"] is True
    assert spec["sources"][0]["ground_snap_to_floor"] is True
    assert "apartment_height_contract" not in spec
    assert spec["usage_scope"] == "research_candidate"
    assert spec["formal_registration_authorized"] is False
