from __future__ import annotations

import json
from pathlib import Path

from tools import controlled_source_asset_schema as contracts


ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = (
    ROOT / "data/controlled_source_attributes_v1/candidate_profiles/static_object"
)


def _load_profile(filename: str) -> dict:
    return contracts.validate_attribute_profile(
        json.loads((PROFILE_ROOT / filename).read_text(encoding="utf-8"))
    )


def test_doorbell_v3_requires_a_front_lit_momentary_button_not_a_side_control():
    profile = _load_profile(
        "door_hardware_doorbell_chime_unit_product_view_v1.json"
    )
    generation = profile["generation_contract"]
    positive = generation["positive_template"].lower()
    negative = generation["negative_prompt"].lower()

    assert profile["profile_revision"] == (
        "2026_07_27_v3_front_button_method_revision"
    )
    assert generation["prompt_template_id"] == (
        "static_product_view_t2i_v2_doorbell_front_grille_lit_button"
    )
    assert "upper front" in positive
    assert "translucent momentary push button" in positive
    assert "glowing amber or blue" in positive
    assert "side panels are plain with no controls" in positive
    for guard in (
        "missing front button",
        "unlit button",
        "side button",
        "rotary knob",
        "rotary dial",
    ):
        assert guard in negative


def test_kettle_v3_requires_a_closed_linked_whistle_and_one_piece_stove_base():
    profile = _load_profile(
        "kitchen_appliance_stovetop_kettle_product_view_v1.json"
    )
    generation = profile["generation_contract"]
    positive = generation["positive_template"].lower()
    negative = generation["negative_prompt"].lower()

    assert profile["profile_revision"] == (
        "2026_07_27_v3_closed_whistle_method_revision"
    )
    assert generation["prompt_template_id"] == (
        "static_product_view_t2i_v2_closed_whistle_one_piece_base"
    )
    assert "closed hinged whistle cap" in positive
    assert "seals the entire spout tip" in positive
    assert "hinge and lifting lever are visibly linked" in positive
    assert "flat metal stovetop bottom form one continuous object" in positive
    for guard in (
        "electric kettle",
        "separate base",
        "open spout hole",
        "visible spout opening",
        "missing whistle cap",
        "detached whistle cap",
    ):
        assert guard in negative
