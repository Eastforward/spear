from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts


ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = (
    ROOT / "data/controlled_source_attributes_v1/candidate_profiles/animal"
)
STABLE_TEMPLATE_CATALOG = (
    ROOT / "data/controlled_source_attributes_v1/stable_template_attribute_catalog_v1.json"
)
LEGACY_SIAMESE_PROFILE = (
    ROOT
    / "data/controlled_source_attributes_v1/profiles/animal/"
    "cat_siamese_four_limb_rest_side_v5.json"
)

W2_CANONICAL_PROFILES = {
    "dog_siberian_husky_four_limb_rest_side_clay_v1.json": {
        "taxonomy": {"species": "dog", "breed": "siberian_husky"},
        "sampled": {
            "size": ["medium"],
            "coat_color": ["silver_gray"],
            "body_build": ["standard"],
        },
        "reference_value_cm": 55,
        "reference_source_id": "stable_template_attribute_catalog_v1_husky_candidate",
        "reference_sha256": (
            "cc92fe7adbee7b75b4d5b1ec3c6f79002e8ebd3194d68c2dbec0ec5ea5f4a0be"
        ),
    },
    "cat_siamese_four_limb_rest_side_v1.json": {
        "taxonomy": {"species": "cat", "breed": "siamese"},
        "sampled": {
            "size": ["medium"],
            "point_color": ["seal_point"],
            "body_build": ["standard"],
        },
        "reference_value_cm": 25,
        "reference_source_id": "siamese_pipeline_calibration_pending_v1",
        "reference_sha256": (
            "98760810e629f2a5829ae0e81469e74df99551ccbc6d0660b83f4fded9768412"
        ),
    },
}

BEAGLE_ACQUISITION_PROFILE = (
    ROOT
    / "data/controlled_source_attributes_v1/profiles/animal/"
    "dog_beagle_open_tricolor_photorealistic_recolor_canary_v3.json"
)
BEAGLE_STABLE_PROFILE = (
    PROFILE_ROOT / "dog_beagle_mesh_first_pbr_stable_v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_w2_new_canonical_profiles_validate_compile_and_authenticate_reference():
    for filename, expected in W2_CANONICAL_PROFILES.items():
        profile = contracts.validate_attribute_profile(_load(PROFILE_ROOT / filename))

        assert profile["taxonomy"] == expected["taxonomy"]
        assert profile["sampled_attribute_domains"] == expected["sampled"]
        assert (
            profile["generation_contract"]["base_acquisition_policy"]
            == one_shot.base_acquisition_record()
        )
        assert profile["generation_contract"]["route"] == "flux2_pixal3d_animal_v1"
        assert profile["state_classification"] == "research_candidate"
        assert profile["target_physical_profiles"]["reference_value_cm"] == (
            expected["reference_value_cm"]
        )
        assert (
            profile["target_physical_profiles"]["reference_provenance"]["source_id"]
            == expected["reference_source_id"]
        )
        assert (
            profile["target_physical_profiles"]["reference_provenance"]["status"]
            == "provisional"
        )

        reference = profile["base_template"]["artifact"]
        reference_path = ROOT / reference["path"]
        assert reference_path.stat().st_size == reference["size_bytes"]
        assert _sha256(reference_path) == reference["sha256"]
        assert reference["sha256"] == expected["reference_sha256"]

        combinations = contracts.legal_attribute_combinations(profile)
        assert combinations == [
            {
                attribute: values[0]
                for attribute, values in profile["sampled_attribute_domains"].items()
            }
        ]
        request = contracts.build_instance_request(
            profile,
            combinations[0],
            batch_seed=20260727,
            sample_ordinal=0,
        )
        assert contracts.validate_instance_request(request, profile) == request
        assert request["target_physical_profile"]["target_value_cm"] == (
            expected["reference_value_cm"]
        )


def test_w2_new_canonical_profiles_keep_objective_tail_and_limb_fail_closed_guards():
    for filename in W2_CANONICAL_PROFILES:
        profile = contracts.validate_attribute_profile(_load(PROFILE_ROOT / filename))
        generation = profile["generation_contract"]
        guard = generation["pose_guard_prompt"].lower()
        negative = generation["negative_prompt"].lower()

        assert "all four individually visible legs" in guard
        assert "all four paws planted on the same level ground plane" in guard
        assert "exactly one tail" in guard
        assert "extra tail" in negative
        assert "double tail" in negative
        assert "merged legs" in negative
        assert "hidden paws" in negative


def test_w2_new_profiles_only_project_locally_recorded_breed_contracts():
    catalog = _load(STABLE_TEMPLATE_CATALOG)
    husky_source = next(
        entry
        for entry in catalog["entries"]
        if entry["template_id"] == "quaternius_ultimate_husky_v1"
    )
    husky = contracts.validate_attribute_profile(
        _load(
            PROFILE_ROOT
            / "dog_siberian_husky_four_limb_rest_side_clay_v1.json"
        )
    )
    assert husky["taxonomy"] == {
        "species": husky_source["species"],
        "breed": husky_source["breed"],
    }
    assert husky["fixed_attributes"] == {
        "life_stage": "adult",
        **husky_source["fixed_attributes"],
    }
    assert husky["sampled_attribute_domains"]["coat_color"] == [
        husky_source["coat_tone_values"][1]
    ]
    assert husky["target_physical_profiles"]["reference_value_cm"] == (
        husky_source["physical_profile"]["reference_value_cm"]
    )
    assert husky["target_physical_profiles"]["values"]["medium"]["tolerance_cm"] == (
        husky_source["physical_profile"]["tolerance_cm"]
    )
    assert husky["acoustic_profile"] == {
        **husky_source["acoustic_profile"],
        "selection_attributes": ["species", "breed", "life_stage"],
    }

    siamese_source = contracts.validate_attribute_profile(
        _load(LEGACY_SIAMESE_PROFILE)
    )
    siamese = contracts.validate_attribute_profile(
        _load(PROFILE_ROOT / "cat_siamese_four_limb_rest_side_v1.json")
    )
    assert siamese["taxonomy"] == siamese_source["taxonomy"]
    assert siamese["base_template"] == siamese_source["base_template"]
    assert siamese["fixed_attributes"] == siamese_source["fixed_attributes"]
    assert siamese["sampled_attribute_domains"] == {
        "size": [siamese_source["sampled_attribute_domains"]["size"][1]],
        "point_color": [
            siamese_source["sampled_attribute_domains"]["point_color"][0]
        ],
        "body_build": [
            siamese_source["sampled_attribute_domains"]["body_build"][1]
        ],
    }
    assert siamese["target_physical_profiles"]["reference_value_cm"] == (
        siamese_source["target_physical_profiles"]["reference_value_cm"]
    )
    assert siamese["acoustic_profile"] == siamese_source["acoustic_profile"]


def test_w2_generated_beagle_already_has_acquisition_and_stable_profile_coverage():
    acquisition = contracts.validate_attribute_profile(
        _load(BEAGLE_ACQUISITION_PROFILE)
    )
    stable = contracts.validate_attribute_profile(_load(BEAGLE_STABLE_PROFILE))

    assert acquisition["taxonomy"] == {"species": "dog", "breed": "beagle"}
    assert acquisition["generation_contract"]["route"] == (
        "flux2_pixal3d_animal_v1"
    )
    assert acquisition["generation_contract"]["base_acquisition_policy"] == (
        one_shot.base_acquisition_record()
    )
    assert all(
        len(values) == 1
        for values in acquisition["sampled_attribute_domains"].values()
    )
    assert acquisition["target_physical_profiles"]["reference_value_cm"] == 36
    assert acquisition["base_template"]["provenance_status"] == "verified"

    assert stable["taxonomy"] == acquisition["taxonomy"]
    assert stable["generation_contract"]["route"] == "stable_animal_template_v1"
    assert stable["base_template"]["kind"] == "stable_animal_template"
    assert stable["rig_profile"]["profile_id"] == (
        "mesh_first_beagle_bone20_locked_paws_v2"
    )
    assert stable["target_physical_profiles"]["reference_value_cm"] == 36
