from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import controlled_source_asset_schema as schema
from tools import build_controlled_source_dataset as dataset_builder
from tools import build_controlled_source_asset_inputs as input_builder


REPO = Path(__file__).resolve().parents[2]
BUILD_INPUTS = REPO / "tools" / "build_controlled_source_asset_inputs.py"
BUILD_DATASET = REPO / "tools" / "build_controlled_source_dataset.py"
BUILD_DATASET_INPUT_MANIFEST = (
    REPO / "tools" / "build_controlled_source_dataset_input_manifest.py"
)
PROFILE_ROOT = REPO / "data" / "controlled_source_attributes_v1" / "profiles"


HEX_A = "a" * 64
HEX_B = "b" * 64


def artifact(
    path: str = "data/reference.png",
    digest: str = HEX_A,
    root_id: str = "fixture_root",
) -> dict:
    return {
        "root_id": root_id,
        "path": path,
        "sha256": digest,
        "size_bytes": 1234,
    }


def animal_profile(
    *,
    profile_id: str = "dog_golden_retriever_v1",
    lineage_group_id: str = "golden_reference_01",
) -> dict:
    combined_values = {
        "species": {"dog": "dog"},
        "breed": {"golden_retriever": "golden retriever"},
        "life_stage": {"adult": "adult"},
        "coat_length": {"long": "long"},
        "coat_pattern": {"solid": "solid"},
        "ear_shape": {"drop": "drop"},
        "tail_shape": {"feathered": "feathered"},
        "size": {
            "small": "small",
            "medium": "medium-sized",
            "large": "large",
        },
        "coat_color": {
            "light_golden": "light golden",
            "golden": "golden",
            "dark_golden": "dark golden",
        },
        "body_build": {
            "slim": "slim",
            "standard": "standard",
            "stocky": "stocky",
        },
    }
    return {
        "schema": schema.PROFILE_SCHEMA,
        "profile_schema_id": profile_id,
        "profile_revision": "2026_07_13_v1",
        "asset_class": "animal",
        "lineage_group_id": lineage_group_id,
        "state_classification": "research_candidate",
        "taxonomy": {
            "species": "dog",
            "breed": "golden_retriever",
        },
        "base_template": {
            "template_id": lineage_group_id,
            "kind": "reference_image",
            "artifact": artifact(),
            "provenance_status": "verified",
            "usage_scope": "research_candidate",
        },
        "fixed_attributes": {
            "life_stage": "adult",
            "coat_length": "long",
            "coat_pattern": "solid",
            "ear_shape": "drop",
            "tail_shape": "feathered",
        },
        "sampled_attribute_domains": {
            "size": ["small", "medium", "large"],
            "coat_color": ["light_golden", "golden", "dark_golden"],
            "body_build": ["slim", "standard", "stocky"],
        },
        "forbidden_combinations": [],
        "generation_contract": {
            "route": "flux2_pixal3d_animal_v1",
            "prompt_template_id": "quadruped_i2i_v1",
            "positive_template": (
                "A {size} {life_stage} {breed} {species} with a "
                "{body_build} anatomically normal body, {coat_color} "
                "{coat_length} fur, a {coat_pattern} coat pattern, "
                "{ear_shape} ears, and a {tail_shape} tail."
            ),
            "pose_guard_prompt": (
                "Preserve the canonical quadruped standing pose, side "
                "viewpoint, body orientation, visible separated legs and "
                "paws, tail separation, and plain background."
            ),
            "negative_prompt": (
                "cropped body, sitting, lying down, merged legs, hidden paws, "
                "fused tail, extra limbs, background geometry, text"
            ),
            "value_labels": combined_values,
            "model_revisions": {
                "flux2": "flux_revision",
                "pixal3d": "pixal_revision",
                "dino": "dino_revision",
            },
        },
        "target_physical_profiles": {
            "profile_id": "dog_golden_retriever_physical_v1",
            "control_attribute": "size",
            "measurement": "shoulder_height_cm",
            "mode": "relative_to_profile_reference",
            "reference_value_cm": 55.0,
            "reference_provenance": {
                "status": "verified",
                "source_id": "fixture_breed_measurement_v1",
                "artifact": artifact("references/fixture_measurement.json", HEX_B),
                "notes": "Test fixture only.",
            },
            "values": {
                "small": {"scale_ratio": 0.9, "tolerance_cm": 4.0},
                "medium": {"scale_ratio": 1.0, "tolerance_cm": 4.0},
                "large": {"scale_ratio": 1.1, "tolerance_cm": 4.0},
            },
        },
        "rig_profile": {
            "profile_id": "quadruped_dog_v1",
            "skeleton_family": "quaternius_dog",
            "actions": ["Walking", "Idle"],
            "front_axis": "positive_x",
        },
        "acoustic_profile": {
            "profile_id": "dog_vocalization_v1",
            "default_event_class": "dog_bark",
            "allowed_event_classes": ["dog_bark", "dog_growl", "silent"],
            "selection_attributes": ["species", "breed", "life_stage"],
        },
        "locked_attributes": ["species", "breed", "life_stage"],
        "qa_contract": {
            "subject_label": "golden retriever",
            "attributes": {
                "size": {
                    "kind": "ordered",
                    "label": "size",
                    "value_order": ["small", "medium", "large"],
                    "value_labels": {
                        "small": "small",
                        "medium": "medium-sized",
                        "large": "large",
                    },
                    "comparison_question": "Which golden retriever is larger?",
                },
                "coat_color": {
                    "kind": "ordered",
                    "label": "coat color",
                    "value_order": [
                        "light_golden",
                        "golden",
                        "dark_golden",
                    ],
                    "value_labels": {
                        "light_golden": "light golden",
                        "golden": "golden",
                        "dark_golden": "dark golden",
                    },
                    "comparison_question": (
                        "Which golden retriever has the darker coat?"
                    ),
                },
                "body_build": {
                    "kind": "ordered",
                    "label": "body build",
                    "value_order": ["slim", "standard", "stocky"],
                    "value_labels": {
                        "slim": "slim",
                        "standard": "standard",
                        "stocky": "stocky",
                    },
                    "comparison_question": (
                        "Which golden retriever has the stockier build?"
                    ),
                },
            },
        },
    }


def human_profile() -> dict:
    return {
        "schema": schema.PROFILE_SCHEMA,
        "profile_schema_id": "rocketbox_adults_male_adult_01_material_v1",
        "profile_revision": "2026_07_13_v1",
        "asset_class": "human",
        "lineage_group_id": "rocketbox_adults_male_adult_01",
        "state_classification": "research_candidate",
        "taxonomy": {"species": "human"},
        "base_template": {
            "template_id": "rocketbox_adults_male_adult_01",
            "kind": "rocketbox_avatar",
            "artifact": artifact(
                "Assets/Avatars/Adults/Male_Adult_01/Export/Male_Adult_01.fbx"
            ),
            "provenance_status": "verified",
            "usage_scope": "research_candidate",
        },
        "fixed_attributes": {
            "gender": "male",
            "life_stage": "adult",
            "top_geometry": "plaid_short_sleeve_shirt",
            "bottom_geometry": "shorts",
            "footwear_geometry": "shoes",
            "headwear": "absent",
            "eyewear": "absent",
            "accessories": "base_locked",
        },
        "sampled_attribute_domains": {
            "top_color": ["blue", "green", "burgundy"]
        },
        "forbidden_combinations": [],
        "generation_contract": {
            "route": "rocketbox_material_v1",
            "material_plan_schema": "rocketbox_material_edit_plan_v1",
            "material_bindings": {
                "top_color": {
                    "semantic_mask": "shirt_main_color",
                    "mask_registry": artifact(
                        "tmp/rocketbox_native_material_canary_v1/"
                        "rocketbox_male_adult_01/shirt_blue_v1/"
                        "mask_registry.json",
                        HEX_B,
                    ),
                    "source_texture_role": "body_base_color",
                    "operation": "replace_base_color_preserve_pbr_detail_v1",
                    "palette": {
                        "blue": {"srgb_u8": [36, 88, 207]},
                        "green": {"srgb_u8": [47, 107, 69]},
                        "burgundy": {"srgb_u8": [122, 48, 69]},
                    },
                }
            },
            "flux_texture_detail": {
                "enabled": False,
                "policy": "approved_mask_optional_only",
            },
        },
        "target_physical_profiles": {
            "profile_id": "rocketbox_authored_height_v1",
            "control_attribute": None,
            "measurement": "authored_height_cm",
            "mode": "authored",
            "reference_value_cm": 182.36783742904663,
            "reference_provenance": {
                "status": "verified",
                "source_id": "rocketbox_inventory_v1",
                "artifact": artifact("inventory/rocketbox_inventory.json", HEX_B),
                "notes": "Authored Rocketbox bind-pose height.",
            },
            "values": {
                "fixed": {
                    "actor_scale": 1.0,
                    "target_value_cm": 182.36783742904663,
                    "tolerance_cm": 3.0,
                }
            },
        },
        "rig_profile": {
            "profile_id": "rocketbox_bip01_v1",
            "skeleton_family": "Bip01",
            "actions": ["Walking", "Standing_Idle"],
            "front_axis": "negative_y",
        },
        "acoustic_profile": {
            "profile_id": "licensed_human_speech_v1",
            "default_event_class": "speech",
            "allowed_event_classes": ["speech", "silent"],
            "selection_attributes": ["gender", "life_stage"],
        },
        "locked_attributes": [
            "identity",
            "body_geometry",
            "skin",
            "eyes",
            "garment_geometry",
            "headwear",
            "eyewear",
            "accessories",
        ],
        "qa_contract": {
            "subject_label": "male adult",
            "attributes": {
                "top_color": {
                    "kind": "categorical",
                    "label": "shirt color",
                    "value_labels": {
                        "blue": "blue",
                        "green": "green",
                        "burgundy": "burgundy",
                    },
                    "identification_question": (
                        "What is the shirt color of {instance_label}?"
                    ),
                }
            },
        },
    }


def test_profile_validation_is_strict_and_hash_stable():
    profile = animal_profile()

    validated = schema.validate_attribute_profile(profile)

    assert validated == profile
    assert schema.profile_sha256(profile) == schema.profile_sha256(copy.deepcopy(profile))


def test_checked_in_profile_catalog_is_valid_breed_specific_and_balanced():
    paths = sorted(PROFILE_ROOT.rglob("*.json"))
    profiles = [
        schema.validate_attribute_profile(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in paths
    ]

    assert len(profiles) >= 16
    assert len({profile["profile_schema_id"] for profile in profiles}) == len(profiles)
    by_id = {profile["profile_schema_id"]: profile for profile in profiles}
    assert by_id["dog_golden_retriever_v1"]["sampled_attribute_domains"][
        "coat_color"
    ] == ["light_golden", "golden", "dark_golden"]
    assert by_id["dog_pug_v1"]["sampled_attribute_domains"]["coat_color"] == [
        "light_fawn",
        "fawn",
        "apricot",
    ]
    assert by_id["cat_tabby_bindpose_v2"]["base_template"][
        "provenance_status"
    ] == "verified"
    assert by_id["cat_siamese_bindpose_v2"]["generation_contract"][
        "prompt_template_id"
    ] == "quadruped_bindpose_i2i_v2"
    horse = by_id["horse_bay_native_action_composite_side_clay_v2"]
    assert horse["base_template"]["provenance_status"] == "verified"
    assert horse["base_template"]["artifact"]["sha256"] == (
        "565c18aa24c3a0fdc8e56938b7b72468784cb804a5dd667b422faf830cec6a4e"
    )
    assert horse["rig_profile"]["front_axis"] == "positive_x"
    assert horse["rig_profile"]["actions"] == ["Walking", "Idle"]
    assert horse["sampled_attribute_domains"]["coat_tone"] == [
        "light_bay",
        "standard_bay",
        "dark_bay",
    ]
    strict_side_ids = {
        "cat_tabby_four_limb_rest_side_v5",
        "cat_siamese_four_limb_rest_side_v5",
        "dog_beagle_four_limb_rest_side_v5",
        "dog_golden_retriever_four_limb_rest_side_v5",
        "dog_pug_four_limb_rest_side_v5",
    }
    for profile_id in strict_side_ids:
        contract = by_id[profile_id]["generation_contract"]
        assert contract["prompt_template_id"] == "quadruped_four_limb_authored_rest_side_i2i_v5"
        assert by_id[profile_id]["base_template"]["provenance_status"] == "verified"
        assert "authored_rest_pose_side_four_limb_v2" in by_id[profile_id]["base_template"][
            "template_id"
        ]
        assert "orthographic exact side view" in contract["pose_guard_prompt"]
        assert "head and neck must align" in contract["pose_guard_prompt"]
        assert "all four individually visible legs" in contract["pose_guard_prompt"]
        assert "same level ground plane" in contract["pose_guard_prompt"]
        assert "three-quarter view" in contract["negative_prompt"]
    clay_profile_ids = {
        "dog_beagle_four_limb_rest_side_clay_v6",
        "dog_golden_retriever_four_limb_rest_side_clay_v6",
        "dog_pug_four_limb_rest_side_clay_v6",
    }
    for profile_id in clay_profile_ids:
        dog_clay = by_id[profile_id]
        assert dog_clay["generation_contract"]["prompt_template_id"] == (
            "quadruped_four_limb_uniform_clay_side_i2i_v6"
        )
        assert dog_clay["base_template"]["template_id"] == (
            "quaternius_dog_authored_rest_pose_side_four_limb_clay_v3"
        )
        assert "replace every visible clay surface" in dog_clay[
            "generation_contract"
        ]["pose_guard_prompt"].lower()
        assert "untextured patch" in dog_clay["generation_contract"][
            "negative_prompt"
        ]
    for profile in profiles:
        requests = schema.sample_instance_requests(
            profile, count=9, batch_seed=20260713
        )
        for attribute, values in profile["sampled_attribute_domains"].items():
            counts = [
                sum(
                    request["sampled_attributes"][attribute] == value
                    for request in requests
                )
                for value in values
            ]
            assert max(counts) - min(counts) <= 1


def test_profile_rejects_more_than_three_values_and_relative_fields():
    too_many = animal_profile()
    too_many["sampled_attribute_domains"]["size"].append("very_large")
    with pytest.raises(schema.ContractError, match="one to three"):
        schema.validate_attribute_profile(too_many)

    relative = animal_profile()
    relative["fixed_attributes"]["from"] = "medium"
    with pytest.raises(schema.ContractError, match="relative"):
        schema.validate_attribute_profile(relative)


def test_animal_sampling_is_deterministic_balanced_and_compiles_one_prompt():
    profile = animal_profile()

    first = schema.sample_instance_requests(profile, count=27, batch_seed=42)
    second = schema.sample_instance_requests(profile, count=27, batch_seed=42)

    assert first == second
    assert len({item["instance_id"] for item in first}) == 27
    for attribute, values in profile["sampled_attribute_domains"].items():
        counts = {
            value: sum(item["sampled_attributes"][attribute] == value for item in first)
            for value in values
        }
        assert max(counts.values()) - min(counts.values()) <= 1

    request = first[0]
    assert request["schema"] == schema.REQUEST_SCHEMA
    assert request["generation_plan"]["route"] == "flux2_pixal3d_animal_v1"
    assert request["generation_plan"]["flux_invocations"] == 1
    assert "free tail visibly separated from both hind legs" in request[
        "generation_plan"
    ]["prompt"]
    assert "does not need to point upward" in request["generation_plan"]["prompt"]
    assert "fused tail and leg" in request["generation_plan"]["negative_prompt"]
    assert "from" not in schema.canonical_json(request["sampled_attributes"])
    for value in request["sampled_attributes"].values():
        label = profile["generation_contract"]["value_labels"]
        assert any(
            label_for_value.get(value, "") in request["generation_plan"]["prompt"]
            for label_for_value in label.values()
        )
    schema.validate_instance_request(request, profile)


def test_human_request_selects_fixed_rocketbox_and_compiles_material_plan():
    profile = human_profile()

    request = schema.sample_instance_requests(profile, count=3, batch_seed=7)[0]
    plan = request["generation_plan"]

    assert plan["route"] == "rocketbox_material_v1"
    assert plan["base_avatar_id"] == "rocketbox_adults_male_adult_01"
    assert plan["geometry_changes_allowed"] is False
    assert plan["flux_texture_detail"]["enabled"] is False
    assert len(plan["edits"]) == 1
    assert plan["edits"][0]["semantic_mask"] == "shirt_main_color"
    assert plan["edits"][0]["value"] == request["sampled_attributes"]["top_color"]
    assert "eyes" in request["locked_attributes"]
    assert "accessories" in request["locked_attributes"]
    assert request["target_physical_profile"]["actor_scale"] == 1.0
    schema.validate_instance_request(request, profile)


def test_request_tampering_is_detected():
    profile = animal_profile()
    request = schema.sample_instance_requests(profile, count=1, batch_seed=4)[0]
    request["generation_plan"]["prompt"] += " Add a hat."

    with pytest.raises(schema.ContractError, match="canonical request"):
        schema.validate_instance_request(request, profile)


def test_request_batch_is_rebuilt_from_exact_profile_revision():
    profile = animal_profile()
    batch = schema.build_request_batch(
        [profile], count_per_profile=3, batch_seed=20260713
    )

    assert schema.validate_request_batch(batch, [profile]) == batch

    changed_profile = copy.deepcopy(profile)
    changed_profile["profile_revision"] = "2026_07_13_v2"
    with pytest.raises(schema.ContractError, match="deterministic profiles"):
        schema.validate_request_batch(batch, [changed_profile])

    changed_batch = copy.deepcopy(batch)
    first_profile = profile["profile_schema_id"]
    first_attribute = next(iter(changed_batch["distribution"][first_profile]))
    first_value = next(iter(changed_batch["distribution"][first_profile][first_attribute]))
    changed_batch["distribution"][first_profile][first_attribute][first_value] += 1
    with pytest.raises(schema.ContractError, match="deterministic profiles"):
        schema.validate_request_batch(changed_batch, [profile])


def test_qa_pairs_compare_absolute_profiles_without_edit_history():
    profile = animal_profile()
    requests = schema.sample_instance_requests(profile, count=27, batch_seed=12)

    pairs = schema.build_instance_qa_pairs(
        requests,
        {profile["profile_schema_id"]: profile},
        single_attribute_only=True,
    )

    assert pairs
    pair = pairs[0]
    assert pair["schema"] == schema.PAIR_SCHEMA
    assert len(pair["different_attributes"]) == 1
    assert "from" not in schema.canonical_json(pair)
    assert "to" not in pair["different_attributes"]
    assert pair["questions"]
    assert pair["questions"][0]["answer"]["instance_id"] in {
        pair["instance_a"],
        pair["instance_b"],
    }


def test_generation_plan_marks_attribute_answers_as_pending_visual_evidence():
    profile = animal_profile()
    requests = schema.sample_instance_requests(profile, count=27, batch_seed=19)

    plan = schema.build_generation_plan_manifest(
        requests,
        {profile["profile_schema_id"]: profile},
        plan_id="controlled_generation_canary_v1",
        split_salt="controlled-generation-split-v1",
    )

    assert plan["schema"] == schema.GENERATION_PLAN_SCHEMA
    assert plan["evidence_state"] == "planned"
    assert plan["qa_pairs"]
    assert all(
        pair["answer_status"] == "planned_from_attributes_pending_visual_asset"
        for pair in plan["qa_pairs"]
    )
    assert plan["automatic_checks"]["lineage_leakage"] is False
    assert plan["manifest_sha256"] == schema.manifest_sha256(plan)


def realized_asset(request: dict, *, state: str = "research_candidate") -> dict:
    qa_value = "passed" if state == "formal_dataset_asset" else "pending"
    rights_status = "cleared" if state == "formal_dataset_asset" else "review_required"
    physical = {
        "status": "measured",
        "method": "fixture_measurement_v1",
        "runtime": {
            "actor_scale": 1.0,
            "shoulder_height_cm": 55.0,
            "head_height_cm": 70.0,
            "body_length_cm": 90.0,
        },
    }
    qa = {
        key: qa_value
        for key in schema.QA_FIELDS
    }
    return schema.build_source_asset_v2(
        request,
        artifacts={"runtime_glb": artifact("registry/asset.glb")},
        physical_measurements=physical,
        provenance={
            "attempt_id": "fixture_attempt",
            "request_sha256": request["request_sha256"],
            "models": copy.deepcopy(
                request["generation_plan"].get("model_revisions", {})
            ),
        },
        rights={
            "status": rights_status,
            "licenses": [artifact("licenses/license.txt")],
            "blockers": [] if state == "formal_dataset_asset" else ["review_pending"],
        },
        qa=qa,
        state_classification=state,
    )


def test_source_asset_v2_preserves_request_and_formal_gate():
    profile = animal_profile()
    request = schema.sample_instance_requests(profile, count=1, batch_seed=1)[0]

    asset = realized_asset(request, state="formal_dataset_asset")

    validated = schema.validate_source_asset_v2(asset, request=request)
    assert validated["semantic_attributes"]["size"] == request["sampled_attributes"]["size"]
    assert validated["request_sha256"] == request["request_sha256"]

    broken = copy.deepcopy(asset)
    broken["qa"]["walking"] = "pending"
    with pytest.raises(schema.ContractError, match="formal"):
        schema.validate_source_asset_v2(broken, request=request)


def test_dataset_split_is_lineage_grouped_and_qa_pairs_never_cross_splits():
    profiles = []
    assets = []
    for index in range(12):
        profile = animal_profile(
            profile_id=f"dog_golden_retriever_reference_{index:02d}_v1",
            lineage_group_id=f"golden_reference_{index:02d}",
        )
        profiles.append(profile)
        requests = schema.sample_instance_requests(
            profile, count=3, batch_seed=100 + index
        )
        assets.extend(realized_asset(request) for request in requests)

    profile_map = {item["profile_schema_id"]: item for item in profiles}
    first = schema.build_dataset_manifest(
        assets,
        profile_map,
        dataset_id="controlled_canary_v1",
        split_salt="controlled-split-v1",
        allowed_states={"research_candidate"},
    )
    second = schema.build_dataset_manifest(
        copy.deepcopy(assets),
        copy.deepcopy(profile_map),
        dataset_id="controlled_canary_v1",
        split_salt="controlled-split-v1",
        allowed_states={"research_candidate"},
    )

    assert first == second
    group_splits: dict[str, set[str]] = {}
    asset_splits = {}
    for item in first["assets"]:
        group_splits.setdefault(item["lineage_group_id"], set()).add(item["split"])
        asset_splits[item["asset_id"]] = item["split"]
    assert all(len(splits) == 1 for splits in group_splits.values())
    for pair in first["qa_pairs"]:
        assert asset_splits[pair["instance_a"]] == pair["split"]
        assert asset_splits[pair["instance_b"]] == pair["split"]
    assert first["automatic_checks"]["lineage_leakage"] is False
    assert first["manifest_sha256"] == schema.manifest_sha256(first)

    qa_dataset = dataset_builder.build_realized_qa_dataset(first)
    source_pool = dataset_builder.build_scene_source_pool(first)
    assert qa_dataset["pair_count"] == len(first["qa_pairs"])
    assert qa_dataset["question_count"] == sum(
        len(pair["questions"]) for pair in first["qa_pairs"]
    )
    assert "pending_formal_acceptance" in qa_dataset["answer_policy"]
    assert source_pool["source_count"] == len(first["assets"])
    assert source_pool["eligible_source_count"] == 0
    assert source_pool["ineligible_source_count"] == len(first["assets"])
    assert sum(len(items) for items in source_pool["splits"].values()) == len(
        first["assets"]
    )
    assert all(
        not source["scene_readiness"]["eligible_for_apartment_generation"]
        for sources in source_pool["splits"].values()
        for source in sources
    )


def test_scene_source_pool_only_marks_fully_qaed_assets_eligible():
    profile = animal_profile()
    request = schema.sample_instance_requests(profile, count=1, batch_seed=9)[0]
    asset = realized_asset(request, state="formal_dataset_asset")
    manifest = schema.build_dataset_manifest(
        [asset],
        {profile["profile_schema_id"]: profile},
        dataset_id="formal_scene_pool_canary_v1",
        split_salt="formal-scene-pool-v1",
    )

    source_pool = dataset_builder.build_scene_source_pool(manifest)

    assert source_pool["source_count"] == 1
    assert source_pool["eligible_source_count"] == 1
    source = next(
        source
        for sources in source_pool["splits"].values()
        for source in sources
    )
    assert source["scene_readiness"] == {
        "eligible_for_apartment_generation": True,
        "blocking_qa": [],
        "rights_ready": True,
        "policy": "all_scene_qa_passed_and_rights_cleared_v1",
    }


def test_animal_size_qa_waits_for_metric_measurement_evidence():
    profile = animal_profile()
    requests = schema.sample_instance_requests(profile, count=27, batch_seed=29)
    assets = []
    for request in requests:
        asset = realized_asset(request)
        asset["qa"]["reference_2d"] = "passed"
        asset["qa"]["static_mesh"] = "passed"
        asset["physical_measurements"] = {"status": "pending"}
        assets.append(asset)

    pending = schema.build_dataset_manifest(
        assets,
        {profile["profile_schema_id"]: profile},
        dataset_id="animal_size_pending_v1",
        split_salt="animal-size-evidence-v1",
        allowed_states={"research_candidate"},
    )

    assert pending["qa_pairs"]
    assert all(
        "size" not in pair["different_attributes"] for pair in pending["qa_pairs"]
    )
    assert all(
        "size" not in row["qa_evidence_attributes"] for row in pending["assets"]
    )

    size_height = {"small": 45.0, "medium": 55.0, "large": 65.0}
    for asset in assets:
        asset["physical_measurements"] = {
            "status": "measured",
            "method": "fixture_measurement_v1",
            "runtime": {
                "actor_scale": 1.0,
                "shoulder_height_cm": size_height[
                    asset["sampled_attributes"]["size"]
                ],
            },
        }
    measured = schema.build_dataset_manifest(
        assets,
        {profile["profile_schema_id"]: profile},
        dataset_id="animal_size_measured_v1",
        split_salt="animal-size-evidence-v1",
        allowed_states={"research_candidate"},
    )

    assert any(
        "size" in pair["different_attributes"] for pair in measured["qa_pairs"]
    )
    assert all("size" in row["qa_evidence_attributes"] for row in measured["assets"])


def test_realized_size_pair_is_removed_when_metric_order_contradicts_label():
    profile = animal_profile()
    requests = schema.sample_instance_requests(profile, count=27, batch_seed=29)
    small_request = next(
        request for request in requests if request["sampled_attributes"]["size"] == "small"
    )
    large_request = next(
        request
        for request in requests
        if request["sampled_attributes"]["size"] == "large"
        and all(
            request["sampled_attributes"][key]
            == small_request["sampled_attributes"][key]
            for key in request["sampled_attributes"]
            if key != "size"
        )
    )
    assets = [realized_asset(small_request), realized_asset(large_request)]
    for asset, height in zip(assets, (60.0, 50.0)):
        asset["qa"]["reference_2d"] = "passed"
        asset["qa"]["static_mesh"] = "passed"
        asset["physical_measurements"] = {
            "status": "measured",
            "method": "fixture_measurement_v1",
            "runtime": {
                "actor_scale": 1.0,
                "shoulder_height_cm": height,
            },
        }

    pairs = schema.build_instance_qa_pairs(assets, {profile["profile_schema_id"]: profile})

    assert pairs == []


def test_profile_loader_accepts_frozen_snapshot_and_rejects_changed_payload(tmp_path):
    profile = animal_profile()
    snapshot = input_builder.build_profile_snapshot(
        [profile],
        {
            profile["profile_schema_id"]: [
                {
                    "role": "base_template",
                    "root_id": "fixture_root",
                    "path": "reference.png",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                    "status": "passed",
                }
            ]
        },
    )
    path = tmp_path / "profile_snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    loaded = input_builder.load_profiles([path])

    assert loaded == [schema.validate_attribute_profile(profile)]
    snapshot["profiles"][0]["profile"]["profile_revision"] = "changed"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(schema.ContractError, match="snapshot"):
        input_builder.load_profiles([path])


def test_cli_publishes_normalized_requests_jobs_and_qa_without_overwrite(tmp_path):
    profile = animal_profile()
    for artifact_record, relative_path, payload in (
        (
            profile["base_template"]["artifact"],
            "data/reference.png",
            b"fixture reference image",
        ),
        (
            profile["target_physical_profiles"]["reference_provenance"]["artifact"],
            "references/fixture_measurement.json",
            b'{"fixture": true}\n',
        ),
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifact_record["path"] = relative_path
        artifact_record["size_bytes"] = len(payload)
        artifact_record["sha256"] = hashlib.sha256(payload).hexdigest()
    profile_path = tmp_path / "golden.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    output = tmp_path / "compiled"
    command = [
        "/data/jzy/miniconda3/envs/spear-env/bin/python",
        str(BUILD_INPUTS),
        "--profile",
        str(profile_path),
        "--count-per-profile",
        "27",
        "--seed",
        "42",
        "--plan-id",
        "controlled_cli_canary_v1",
        "--split-salt",
        "controlled-cli-split-v1",
        "--artifact-root",
        f"fixture_root={tmp_path}",
        "--output-dir",
        str(output),
    ]

    completed = subprocess.run(command, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert "CONTROLLED_SOURCE_INPUT_OK" in completed.stdout
    expected = {
        "profile_snapshot.json",
        "instance_requests.json",
        "execution_jobs.json",
        "qa_pair_plan.json",
        "generation_plan.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    jobs = json.loads((output / "execution_jobs.json").read_text(encoding="utf-8"))
    assert jobs["job_count"] == 27
    assert len(jobs["routes"]["flux2_pixal3d_animal_v1"]) == 27
    pair_plan = json.loads((output / "qa_pair_plan.json").read_text(encoding="utf-8"))
    assert pair_plan["pair_count"] > 0
    assert pair_plan["answer_policy"] == "pending_until_source_asset_v2_visual_qa_passes"

    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode == 2
    assert "refusing to replace" in repeated.stderr


def test_dataset_cli_authenticates_artifacts_and_publishes_realized_qa(tmp_path):
    profile = animal_profile()

    def write_artifact(record: dict, relative_path: str, payload: bytes) -> None:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        record.update(
            {
                "root_id": "fixture_root",
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )

    write_artifact(
        profile["base_template"]["artifact"],
        "profile/reference.png",
        b"reference",
    )
    write_artifact(
        profile["target_physical_profiles"]["reference_provenance"]["artifact"],
        "profile/measurement.json",
        b'{"height_cm": 55}\n',
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    runtime_record = artifact("assets/runtime.glb")
    license_record = artifact("licenses/license.txt")
    write_artifact(runtime_record, "assets/runtime.glb", b"runtime glb fixture")
    write_artifact(license_record, "licenses/license.txt", b"fixture license\n")

    asset_dir = tmp_path / "source_assets"
    asset_dir.mkdir()
    requests = schema.sample_instance_requests(profile, count=3, batch_seed=88)
    for index, request in enumerate(requests):
        asset = realized_asset(request)
        asset["artifacts"]["runtime_glb"] = copy.deepcopy(runtime_record)
        asset["rights"]["licenses"] = [copy.deepcopy(license_record)]
        (asset_dir / f"asset_{index}.json").write_text(
            json.dumps(asset), encoding="utf-8"
        )

    output = tmp_path / "dataset"
    command = [
        "/data/jzy/miniconda3/envs/spear-env/bin/python",
        str(BUILD_DATASET),
        "--profile",
        str(profile_path),
        "--asset",
        str(asset_dir),
        "--dataset-id",
        "controlled_dataset_cli_canary_v1",
        "--split-salt",
        "controlled-dataset-cli-split-v1",
        "--allow-state",
        "research_candidate",
        "--artifact-root",
        f"fixture_root={tmp_path}",
        "--output-dir",
        str(output),
    ]

    completed = subprocess.run(command, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert "CONTROLLED_SOURCE_DATASET_OK" in completed.stdout
    assert {path.name for path in output.iterdir()} == {
        "dataset_manifest.json",
        "qa_dataset.json",
        "scene_source_pool.json",
        "artifact_audit.json",
    }
    audit = json.loads((output / "artifact_audit.json").read_text(encoding="utf-8"))
    assert audit["automatic_checks"] == {
        "profile_count": 1,
        "asset_count": 3,
        "all_artifacts_authenticated": True,
    }
    qa = json.loads((output / "qa_dataset.json").read_text(encoding="utf-8"))
    assert qa["evidence_state"] == "realized"
    assert "pending_formal_acceptance" in qa["answer_policy"]


def test_dataset_cli_consumes_frozen_request_lineage_manifest(tmp_path):
    profile = animal_profile()

    def write_artifact(record: dict, relative_path: str, payload: bytes) -> None:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        record.update(
            {
                "root_id": "fixture_root",
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )

    write_artifact(
        profile["base_template"]["artifact"],
        "profile/reference.png",
        b"reference",
    )
    write_artifact(
        profile["target_physical_profiles"]["reference_provenance"]["artifact"],
        "profile/measurement.json",
        b'{"height_cm": 55}\n',
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    runtime_record = artifact("assets/runtime.glb")
    license_record = artifact("licenses/license.txt")
    write_artifact(runtime_record, "assets/runtime.glb", b"runtime glb fixture")
    write_artifact(license_record, "licenses/license.txt", b"fixture license\n")

    request_batch = schema.build_request_batch(
        [profile], count_per_profile=3, batch_seed=88
    )
    request_path = tmp_path / "instance_requests.json"
    request_path.write_text(json.dumps(request_batch), encoding="utf-8")
    asset_dir = tmp_path / "source_assets"
    asset_dir.mkdir()
    for index, request in enumerate(request_batch["requests"]):
        asset = realized_asset(request)
        asset["artifacts"]["runtime_glb"] = copy.deepcopy(runtime_record)
        asset["rights"]["licenses"] = [copy.deepcopy(license_record)]
        (asset_dir / f"asset_{index}.json").write_text(
            json.dumps(asset), encoding="utf-8"
        )

    input_manifest_path = tmp_path / "dataset_input_manifest.json"
    compile_command = [
        "/data/jzy/miniconda3/envs/spear-env/bin/python",
        str(BUILD_DATASET_INPUT_MANIFEST),
        "--profile",
        str(profile_path),
        "--request-batch",
        str(request_path),
        "--asset",
        str(asset_dir),
        "--dataset-id",
        "controlled_dataset_lineage_canary_v1",
        "--split-salt",
        "controlled-dataset-lineage-split-v1",
        "--allow-state",
        "research_candidate",
        "--artifact-root",
        f"fixture_root={tmp_path}",
        "--output",
        str(input_manifest_path),
    ]

    compiled = subprocess.run(compile_command, capture_output=True, text=True)

    assert compiled.returncode == 0, compiled.stderr
    assert "CONTROLLED_DATASET_INPUT_MANIFEST_OK" in compiled.stdout
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    assert input_manifest["request_lineage"] == {
        "bindings": input_manifest["request_lineage"]["bindings"],
        "profile_count": 1,
        "request_batch_count": 1,
        "request_count": 3,
        "realized_asset_count": 3,
        "matched_asset_count": 3,
        "unused_request_count": 0,
        "all_realized_assets_match_exactly_one_request": True,
    }
    assert all(
        binding["status"] == "passed"
        for binding in input_manifest["request_lineage"]["bindings"]
    )

    output = tmp_path / "dataset_from_manifest"
    build_command = [
        "/data/jzy/miniconda3/envs/spear-env/bin/python",
        str(BUILD_DATASET),
        "--input-manifest",
        str(input_manifest_path),
        "--artifact-root",
        f"fixture_root={tmp_path}",
        "--output-dir",
        str(output),
    ]
    built = subprocess.run(build_command, capture_output=True, text=True)

    assert built.returncode == 0, built.stderr
    assert "request_lineage=verified" in built.stdout
    assert {path.name for path in output.iterdir()} == {
        "dataset_manifest.json",
        "qa_dataset.json",
        "scene_source_pool.json",
        "artifact_audit.json",
        "dataset_input_manifest.json",
        "build_receipt.json",
    }
    receipt = json.loads((output / "build_receipt.json").read_text(encoding="utf-8"))
    assert receipt["request_lineage_status"] == "passed"
    assert receipt["dataset_input_manifest_sha256"] == input_manifest["manifest_sha256"]

    first_asset_path = sorted(asset_dir.glob("*.json"))[0]
    first_asset_path.write_text(
        first_asset_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    rejected = subprocess.run(
        build_command[:-1] + [str(tmp_path / "tampered_output")],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "mismatch" in rejected.stderr


def test_dataset_input_manifest_rejects_asset_without_normalized_request(tmp_path):
    profile = animal_profile()

    def write_artifact(record: dict, relative_path: str, payload: bytes) -> None:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        record.update(
            {
                "root_id": "fixture_root",
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )

    write_artifact(profile["base_template"]["artifact"], "reference.png", b"reference")
    write_artifact(
        profile["target_physical_profiles"]["reference_provenance"]["artifact"],
        "measurement.json",
        b"measurement",
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    request_batch = schema.build_request_batch(
        [profile], count_per_profile=1, batch_seed=1
    )
    request_path = tmp_path / "instance_requests.json"
    request_path.write_text(json.dumps(request_batch), encoding="utf-8")

    unmatched_request = schema.sample_instance_requests(
        profile, count=1, batch_seed=999
    )[0]
    runtime = artifact("runtime.glb")
    license_record = artifact("license.txt")
    write_artifact(runtime, "runtime.glb", b"runtime")
    write_artifact(license_record, "license.txt", b"license")
    asset = realized_asset(unmatched_request)
    asset["artifacts"]["runtime_glb"] = runtime
    asset["rights"]["licenses"] = [license_record]
    asset_path = tmp_path / "asset.json"
    asset_path.write_text(json.dumps(asset), encoding="utf-8")

    command = [
        "/data/jzy/miniconda3/envs/spear-env/bin/python",
        str(BUILD_DATASET_INPUT_MANIFEST),
        "--profile",
        str(profile_path),
        "--request-batch",
        str(request_path),
        "--asset",
        str(asset_path),
        "--dataset-id",
        "unmatched_request_canary_v1",
        "--split-salt",
        "unmatched-request-split-v1",
        "--allow-state",
        "research_candidate",
        "--artifact-root",
        f"fixture_root={tmp_path}",
        "--output",
        str(tmp_path / "input_manifest.json"),
    ]

    completed = subprocess.run(command, capture_output=True, text=True)

    assert completed.returncode == 2
    assert "has no matching normalized request" in completed.stderr
