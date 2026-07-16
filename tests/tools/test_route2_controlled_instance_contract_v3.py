from __future__ import annotations

import copy
import hashlib
import json
import math
import stat
from pathlib import Path

import pytest

from tools import route2_controlled_instance_contract_v3 as contract


SPEAR_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_JOBS_V2 = SPEAR_ROOT / "tmp/human_attribute_instances_v1/jobs_v2.json"


def _default_attributes(sex: str) -> dict[str, object]:
    return {
        "sex": sex,
        "base_avatar_id": contract.BASE_AVATAR_IDS[sex],
        **copy.deepcopy(contract.BASE_DEFAULTS[sex]),
    }


def _instance_for(attributes: dict[str, object]) -> dict[str, object]:
    return contract.build_instance_from_ordinal(
        contract.ordinal_from_attributes(attributes)
    )


def test_v3_space_is_finite_complete_and_contains_both_qualified_bases():
    assert contract.COMBINATION_COUNT == 288_000
    assert contract.BASE_AVATAR_IDS == {
        "male": "rocketbox_male_adult_01",
        "female": "rocketbox_female_adult_01",
    }
    assert contract.TOP_STYLES == ("short_sleeve", "long_sleeve")
    assert contract.BOTTOM_STYLES == ("trousers", "shorts")
    assert contract.HEIGHT_CLASSES == ("short", "standard", "tall")

    manifest = contract.build_space_manifest(42)
    exhaustive = manifest["exhaustive_space"]
    assert exhaustive == {
        **exhaustive,
        "representation": "compact_compatible_dimensions_plus_full_id_digest",
        "combination_count": 288_000,
        "canonical_instance_id_sha256": (
            "7c5f4db6e40ad77c8a6e4992a5ef80609e68d056a687d40567036fdc0b57c8dc"
        ),
        "complete": True,
    }
    assert math.prod(exhaustive["radices"]) == exhaustive["combination_count"]
    assert exhaustive["dimension_order"] == [
        "sex",
        "top",
        "bottom",
        "shoe_color",
        "hat",
        "glasses",
        "hair_color",
        "height_class",
    ]
    assert len(contract.DIMENSION_ORDER) == len(contract.DIMENSION_OPTIONS) == 8
    assert [len(options) for options in contract.DIMENSION_OPTIONS] == [
        2,
        12,
        10,
        4,
        5,
        4,
        5,
        3,
    ]


def test_every_ordinal_roundtrips_and_every_dimension_value_is_reachable():
    covered = {
        "sex": set(),
        "top": set(),
        "bottom": set(),
        "shoe_color": set(),
        "hat": set(),
        "glasses": set(),
        "hair_color": set(),
        "height_class": set(),
    }
    for ordinal in range(contract.COMBINATION_COUNT):
        attributes = contract.attributes_from_ordinal(ordinal)
        assert contract.ordinal_from_attributes(attributes) == ordinal
        covered["sex"].add(attributes["sex"])
        covered["top"].add((attributes["top_style"], attributes["top_color"]))
        covered["bottom"].add(
            (attributes["bottom_style"], attributes["bottom_color"])
        )
        covered["shoe_color"].add(attributes["shoe_color"])
        covered["hat"].add(
            (
                attributes["hat_enabled"],
                attributes["hat_color"],
                attributes["hair_state"],
            )
        )
        covered["glasses"].add(
            (attributes["glasses_enabled"], attributes["glasses_color"])
        )
        covered["hair_color"].add(attributes["hair_color"])
        covered["height_class"].add(attributes["height_class"])

    assert covered == {
        name: set(options)
        for name, options in zip(
            contract.DIMENSION_ORDER, contract.DIMENSION_OPTIONS, strict=True
        )
    }
    last = contract.attributes_from_ordinal(contract.COMBINATION_COUNT - 1)
    assert last["hair_color"] == contract.HAIR_COLORS[-1]
    assert last["height_class"] == contract.HEIGHT_CLASSES[-1]


@pytest.mark.parametrize(
    "ordinal", [0, 1, 2, 17, 143_999, 144_000, 287_998, 287_999]
)
def test_mixed_radix_roundtrip_is_exact_at_boundaries(ordinal):
    attributes = contract.attributes_from_ordinal(ordinal)

    assert contract.ordinal_from_attributes(attributes) == ordinal
    assert contract.validate_instance(contract.build_instance_from_ordinal(ordinal))[
        "ordinal"
    ] == ordinal
    assert contract.canonical_instance_id(ordinal).endswith(f"_{ordinal:06d}")


def test_seeded_enumeration_is_reproducible_and_a_full_permutation():
    seed = 20260712
    first = [contract.seeded_ordinal(seed, index) for index in range(64)]
    repeated = [contract.seeded_ordinal(seed, index) for index in range(64)]
    other = [contract.seeded_ordinal(seed + 1, index) for index in range(64)]

    assert first == repeated
    assert first != other
    assert len(first) == len(set(first))
    assert len(
        {contract.seeded_ordinal(seed, index) for index in range(contract.COMBINATION_COUNT)}
    ) == contract.COMBINATION_COUNT

    sampled = list(contract.iter_seeded_instances(seed, start_index=13, limit=7))
    assert [row["ordinal"] for row in sampled] == [
        contract.seeded_ordinal(seed, index) for index in range(13, 20)
    ]


@pytest.mark.parametrize("seed", [True, -1, 1 << 63, 1.5, "42"])
def test_seed_and_enumeration_ranges_fail_closed(seed):
    with pytest.raises(contract.ControlledInstanceContractError, match="seed"):
        contract.build_space_manifest(seed)

    with pytest.raises(contract.ControlledInstanceContractError):
        list(contract.iter_seeded_instances(42, start_index=-1, limit=1))
    with pytest.raises(contract.ControlledInstanceContractError):
        list(
            contract.iter_seeded_instances(
                42, start_index=contract.COMBINATION_COUNT, limit=1
            )
        )


def test_hat_and_glasses_compatibility_is_encoded_not_postfiltered():
    attributes = _default_attributes("female")
    attributes.update(
        {
            "hat_enabled": True,
            "hat_color": "navy",
            "hair_state": "hat_compatible",
            "glasses_enabled": True,
            "glasses_color": "black",
        }
    )
    canonical = contract.validate_attributes(attributes)
    assert canonical["hair_state"] == "hat_compatible"

    bad_hat_hair = copy.deepcopy(attributes)
    bad_hat_hair["hair_state"] = "standard_exposed"
    with pytest.raises(contract.ControlledInstanceContractError, match="hair_state"):
        contract.validate_attributes(bad_hat_hair)

    bad_hat_color = _default_attributes("male")
    bad_hat_color["hat_color"] = "navy"
    with pytest.raises(contract.ControlledInstanceContractError, match="hat color"):
        contract.validate_attributes(bad_hat_color)

    bad_glasses_color = _default_attributes("male")
    bad_glasses_color["glasses_color"] = "black"
    with pytest.raises(contract.ControlledInstanceContractError, match="glasses color"):
        contract.validate_attributes(bad_glasses_color)


def test_identity_base_and_finite_values_are_strictly_pinned():
    wrong_base = _default_attributes("female")
    wrong_base["base_avatar_id"] = contract.BASE_AVATAR_IDS["male"]
    with pytest.raises(contract.ControlledInstanceContractError, match="pinned base"):
        contract.validate_attributes(wrong_base)

    extra = _default_attributes("male")
    extra["skin_tone"] = "new"
    with pytest.raises(contract.ControlledInstanceContractError, match="unexpected"):
        contract.validate_attributes(extra)

    unknown_color = _default_attributes("male")
    unknown_color["shoe_color"] = "neon_orange"
    with pytest.raises(contract.ControlledInstanceContractError, match="finite palette"):
        contract.validate_attributes(unknown_color)

    boolean_alias = _default_attributes("male")
    boolean_alias["hat_enabled"] = 0
    with pytest.raises(contract.ControlledInstanceContractError, match="boolean"):
        contract.validate_attributes(boolean_alias)


def test_request_classifier_separates_locked_attributes_and_new_geometry():
    valid = _default_attributes("male")
    assert contract.classify_request(valid) == {
        "classification": "qualified_base_reuse",
        "execution_authorized": True,
        "reason": "eligible_after_qualified_template_preflight",
    }

    locked = {**valid, "body_type": "muscular"}
    assert contract.classify_request(locked) == {
        "classification": "rejected_locked_attribute",
        "execution_authorized": False,
        "reason": "skin_body_face_or_disallowed_accessory_change_requested",
    }

    new_shoe = {**valid, "shoe_style": "platform_boot"}
    assert contract.classify_request(new_shoe) == {
        "classification": "full_geometry_template_qualification_required",
        "execution_authorized": False,
        "reason": "new_or_unpinned_geometry_requested",
    }

    unknown_top = {**valid, "top_style": "hoodie"}
    assert contract.classify_request(unknown_top)["classification"] == (
        "full_geometry_template_qualification_required"
    )


def test_instance_uses_only_deterministic_materials_and_qualified_template_slots():
    attributes = _default_attributes("female")
    attributes.update(
        {
            "top_style": "long_sleeve",
            "top_color": "navy",
            "bottom_style": "shorts",
            "bottom_color": "beige",
            "shoe_color": "brown",
            "hat_enabled": True,
            "hat_color": "burgundy",
            "hair_state": "hat_compatible",
            "glasses_enabled": True,
            "glasses_color": "brown",
            "hair_color": "gray",
            "height_class": "short",
        }
    )
    instance = _instance_for(attributes)

    assert instance["execution"] == {
        "instance_backend": "deterministic_template_and_semantic_material_v1",
        "template_precondition": "all_resolved_templates_must_be_qualified",
        "flux2": "forbidden_for_instance_expansion",
        "pixal3d": "forbidden_for_instance_expansion",
        "new_geometry": "full_template_qualification_required",
    }
    assert instance["resolved_templates"]["shoes"] == (
        contract.TEMPLATE_REGISTRY["female"]["shoes"]
    )
    assert instance["resolved_templates"]["hat"] == (
        contract.TEMPLATE_REGISTRY["shared_accessories"]["hat"]
    )
    assert instance["resolved_templates"]["glasses"] == (
        contract.TEMPLATE_REGISTRY["shared_accessories"]["glasses"]
    )
    assert instance["material_parameters"]["top"]["color_id"] == "navy"
    assert instance["material_parameters"]["bottom"]["semantic_mask"] == (
        "garment_bottom"
    )
    assert instance["material_parameters"]["shoes"]["operation"] == (
        "replace_base_color_preserve_pbr_detail_v1"
    )
    assert "skin" not in instance["material_parameters"]


def test_automatic_qa_classifies_base_material_and_template_paths():
    base = _instance_for(_default_attributes("male"))
    assert base["automatic_qa"]["qa_class"] == "qualified_base_reuse"

    recolored_attributes = _default_attributes("male")
    recolored_attributes["top_color"] = "navy"
    recolored_attributes["hair_color"] = "gray"
    recolored = _instance_for(recolored_attributes)
    assert recolored["automatic_qa"]["qa_class"] == (
        "deterministic_material_variant"
    )

    composed_attributes = copy.deepcopy(recolored_attributes)
    composed_attributes.update(
        {
            "top_style": "long_sleeve",
            "bottom_style": "shorts",
            "hat_enabled": True,
            "hat_color": "black",
            "hair_state": "hat_compatible",
            "glasses_enabled": True,
            "glasses_color": "navy",
            "height_class": "tall",
        }
    )
    composed = _instance_for(composed_attributes)
    checks = set(composed["automatic_qa"]["required_checks"])
    assert composed["automatic_qa"]["qa_class"] == "approved_template_composition"
    assert {
        "top_template_topology_skinning_and_cloth_clearance_passed",
        "bottom_template_topology_skinning_and_cloth_clearance_passed",
        "height_profile_ratio_and_grounding_passed",
        "hat_rigid_head_binding_passed",
        "hat_hair_clearance_and_hat_compatible_state_passed",
        "glasses_rigid_head_binding_passed",
        "glasses_face_clearance_passed",
    } <= checks
    assert composed["automatic_qa"]["locomotion_policy"] == (
        "grounding_required_minor_foot_sliding_accepted"
    )


def test_qa_result_classifier_requires_all_checks_and_rejects_any_failure():
    attributes = _default_attributes("female")
    attributes.update(
        {
            "hat_enabled": True,
            "hat_color": "gray",
            "hair_state": "hat_compatible",
        }
    )
    instance = _instance_for(attributes)
    checks = instance["automatic_qa"]["required_checks"]
    passed_results = {name: True for name in checks}

    passed = contract.classify_qa_results(instance, passed_results)
    assert passed["status"] == "passed"
    assert passed["state_classification"] == "research_candidate"
    assert passed["eligible_for_formal_registration"] is True

    failed_results = dict(passed_results)
    failed_results["feet_grounded_without_obvious_floating"] = False
    failed = contract.classify_qa_results(instance, failed_results)
    assert failed["status"] == "failed"
    assert failed["state_classification"] == "rejected"
    assert failed["failed_checks"] == ["feet_grounded_without_obvious_floating"]

    missing = dict(passed_results)
    missing.pop(checks[0])
    with pytest.raises(contract.ControlledInstanceContractError, match="exactly"):
        contract.classify_qa_results(instance, missing)

    non_boolean = dict(passed_results)
    non_boolean[checks[0]] = 1
    with pytest.raises(contract.ControlledInstanceContractError, match="boolean"):
        contract.classify_qa_results(instance, non_boolean)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update({"extra": True}),
        lambda row: row.update({"instance_id": "route2_controlled_v3_male_999999"}),
        lambda row: row["attributes"].update({"top_color": "burgundy"}),
        lambda row: row["execution"].update({"flux2": "allowed"}),
        lambda row: row["automatic_qa"].update({"user_approved": True}),
    ],
)
def test_instance_validator_rejects_every_noncanonical_or_approval_claim(mutation):
    instance = _instance_for(_default_attributes("male"))
    mutation(instance)

    with pytest.raises(contract.ControlledInstanceContractError):
        contract.validate_instance(instance)


def test_manifest_validator_is_recursive_strict_and_seed_only_changes_order():
    manifest = contract.build_space_manifest(42)
    other_seed = contract.build_space_manifest(43)

    assert contract.validate_space_manifest(manifest) == manifest
    assert manifest["policy_sha256"] == other_seed["policy_sha256"]
    assert manifest["exhaustive_space"] == other_seed["exhaustive_space"]
    assert manifest["seeded_enumeration"] != other_seed["seeded_enumeration"]
    assert manifest["historical_v2"] == {
        "jobs_v2_role": "immutable_historical_flux_pixal_canary_only",
        "superseded_for_bulk_instance_expansion": True,
        "overwrite_forbidden": True,
    }

    extra = copy.deepcopy(manifest)
    extra["policy"]["execution_policy"]["fallback"] = "arbitrary_generation"
    with pytest.raises(contract.ControlledInstanceContractError, match="canonical"):
        contract.validate_space_manifest(extra)

    bad_count = copy.deepcopy(manifest)
    bad_count["exhaustive_space"]["combination_count"] -= 1
    with pytest.raises(contract.ControlledInstanceContractError, match="canonical"):
        contract.validate_space_manifest(bad_count)

    approval_claim = copy.deepcopy(manifest)
    approval_claim["user_acceptance"] = "user_approved"
    with pytest.raises(contract.ControlledInstanceContractError, match="user_approved"):
        contract.validate_space_manifest(approval_claim)


def test_manifest_publication_is_readonly_noreplace_and_never_touches_jobs_v2(tmp_path):
    historical_before = hashlib.sha256(HISTORICAL_JOBS_V2.read_bytes()).hexdigest()
    historical_mode_before = stat.S_IMODE(HISTORICAL_JOBS_V2.stat().st_mode)
    manifest = contract.build_space_manifest(987654321)

    published = contract.publish_space_manifest(manifest, tmp_path)

    assert published == tmp_path / contract.FILENAME
    assert stat.S_IMODE(published.stat().st_mode) == 0o444
    assert contract.validate_published_space_manifest(published) == manifest
    with pytest.raises(contract.ControlledInstanceContractError, match="already exists"):
        contract.publish_space_manifest(manifest, tmp_path)

    assert hashlib.sha256(HISTORICAL_JOBS_V2.read_bytes()).hexdigest() == historical_before
    assert stat.S_IMODE(HISTORICAL_JOBS_V2.stat().st_mode) == historical_mode_before

    published.chmod(0o644)
    with pytest.raises(contract.ControlledInstanceContractError, match="mode 0444"):
        contract.validate_published_space_manifest(published)


def test_manifest_json_contains_no_free_form_geometry_or_generation_escape_hatch():
    encoded = json.dumps(contract.build_space_manifest(42), sort_keys=True)

    assert "forbidden_for_instance_expansion" not in encoded  # space uses shorter policy form
    assert '"flux2_for_instance_expansion": "forbidden"' in encoded
    assert '"pixal3d_for_instance_expansion": "forbidden"' in encoded
    assert '"skin_tone": "base_locked_v1"' in encoded
    assert '"shoe_geometry": "fixed_per_base_v1"' in encoded
    assert "free_form" not in encoded
