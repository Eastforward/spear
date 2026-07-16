import copy
import json
from pathlib import Path

import pytest

from tools import controlled_source_asset_schema as schema


ROOT = Path(__file__).resolve().parents[2]
BASE_PROFILE = (
    ROOT
    / "data"
    / "controlled_source_attributes_v1"
    / "profiles"
    / "animal"
    / "dog_beagle_v1.json"
)


def stable_beagle_profile():
    profile = json.loads(BASE_PROFILE.read_text(encoding="utf-8"))
    profile["profile_schema_id"] = "dog_beagle_rocketbox_stable_v1"
    profile["profile_revision"] = "2026_07_14_v1"
    profile["lineage_group_id"] = "rocketbox_dog_beagle_01"
    profile["base_template"].update(
        {
            "template_id": "rocketbox_dog_beagle_01_walk_idle_v1",
            "kind": "stable_animal_template",
            "provenance_status": "verified",
        }
    )
    profile["fixed_attributes"].pop("life_stage")
    profile["sampled_attribute_domains"]["life_stage"] = [
        "young",
        "adult",
        "senior",
    ]
    profile["locked_attributes"].remove("life_stage")
    labels = profile["generation_contract"]["value_labels"]
    labels["life_stage"] = {
        "young": "young",
        "adult": "adult",
        "senior": "senior",
    }
    profile["generation_contract"] = {
        "route": "stable_animal_template_v1",
        "plan_schema": "stable_animal_instance_plan_v1",
        "prompt_template_id": "stable_beagle_instance_reference_v1",
        "positive_template": (
            "A {size} {life_stage} {breed} {species} with a {body_build} "
            "anatomically normal body, a {coat_tone} {coat_length} "
            "{coat_pattern} coat, {ear_shape} ears, and an {tail_shape} tail."
        ),
        "pose_guard_prompt": (
            "Preserve the audited side pose, separated limbs, level paws, "
            "breed identity, and plain background."
        ),
        "negative_prompt": (
            "tilted torso, merged legs, hidden paws, holes, extra limbs, text"
        ),
        "value_labels": labels,
        "model_revisions": {"flux2": "flux_revision"},
        "flux_reference_policy": (
            "qa_and_optional_semantic_texture_detail_only"
        ),
        "attribute_bindings": {
            "size": {
                "operation": "uniform_actor_scale_from_physical_profile_v1",
                "values": {
                    value: {"scale_source": "target_physical_profile"}
                    for value in ("small", "medium", "large")
                },
            },
            "body_build": {
                "operation": (
                    "semantic_torso_girth_preserve_topology_and_skin_v1"
                ),
                "values": {
                    "slim": {"torso_girth_scale": 0.92},
                    "standard": {"torso_girth_scale": 1.0},
                    "stocky": {"torso_girth_scale": 1.08},
                },
            },
            "coat_tone": {
                "operation": (
                    "semantic_coat_luminance_preserve_pattern_pbr_v1"
                ),
                "values": {
                    "light_tricolor": {"coat_luminance_gain": 1.12},
                    "standard_tricolor": {"coat_luminance_gain": 1.0},
                    "dark_tricolor": {"coat_luminance_gain": 0.86},
                },
            },
            "life_stage": {
                "operation": (
                    "semantic_age_morph_and_muzzle_tone_preserve_skin_v1"
                ),
                "values": {
                    "young": {"head_scale": 1.04, "muzzle_gray_mix": 0.0},
                    "adult": {"head_scale": 1.0, "muzzle_gray_mix": 0.0},
                    "senior": {"head_scale": 1.0, "muzzle_gray_mix": 0.28},
                },
            },
        },
    }
    profile["rig_profile"].update(
        {
            "profile_id": "rocketbox_beagle_bip01_v1",
            "skeleton_family": "rocketbox_beagle_bip01",
        }
    )
    profile["qa_contract"]["attributes"]["life_stage"] = {
        "kind": "ordered",
        "label": "life stage",
        "value_order": ["young", "adult", "senior"],
        "value_labels": {
            "young": "young",
            "adult": "adult",
            "senior": "senior",
        },
        "comparison_question": "Which beagle appears older?",
    }
    return profile


def test_stable_route_compiles_all_instance_controls_without_rerigging():
    profile = stable_beagle_profile()
    validated = schema.validate_attribute_profile(profile)
    request = schema.build_instance_request(
        validated,
        {
            "size": "large",
            "body_build": "stocky",
            "coat_tone": "dark_tricolor",
            "life_stage": "senior",
        },
        batch_seed=20260714,
        sample_ordinal=0,
    )

    plan = request["generation_plan"]
    assert plan["route"] == "stable_animal_template_v1"
    assert plan["geometry_topology_changes_allowed"] is False
    assert plan["skin_weight_changes_allowed"] is False
    assert plan["skeleton_changes_allowed"] is False
    assert {item["attribute"] for item in plan["attribute_operations"]} == {
        "size",
        "body_build",
        "coat_tone",
        "life_stage",
    }
    prompt = plan["appearance_reference"]["prompt"]
    assert "large" in prompt
    assert "stocky" in prompt
    assert "dark-toned tricolor" in prompt
    assert "senior" in prompt
    assert plan["appearance_reference"]["flux_invocations"] == 1
    schema.validate_instance_request(request, validated)


def test_stable_route_rejects_unrealized_or_unbounded_attributes():
    profile = stable_beagle_profile()
    broken = copy.deepcopy(profile)
    broken["generation_contract"]["attribute_bindings"].pop("life_stage")
    with pytest.raises(schema.ContractError, match="cover every sampled attribute"):
        schema.validate_attribute_profile(broken)

    broken = copy.deepcopy(profile)
    broken["base_template"]["kind"] = "reference_image"
    with pytest.raises(schema.ContractError, match="stable_animal_template"):
        schema.validate_attribute_profile(broken)
