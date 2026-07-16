#!/usr/bin/env python3
"""Fail-closed finite instance space for qualified Route-2 human templates.

Version 3 deliberately separates *instance expansion* from *template creation*.
Every emitted instance is assembled from a pinned, qualified geometry slot and
uses deterministic semantic-mask material edits.  FLUX.2 and Pixal3D are not
instance-expansion backends; they may only be used by the separate full
qualification workflow that introduces a new geometry template revision.

The exhaustive space is stored compactly as fixed compatible dimensions plus a
digest of every canonical instance ID.  Seeded enumeration is an affine
permutation of canonical ordinals, so it is reproducible, exhaustive, and does
not require a multi-gigabyte list of repeated instance objects.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import route2_human_contract_common as common


SCHEMA = "route2_controlled_human_instance_space_v3"
INSTANCE_SCHEMA = "route2_controlled_human_instance_v3"
QA_RESULT_SCHEMA = "route2_controlled_human_instance_qa_result_v3"
FILENAME = "route2_controlled_human_instance_space_v3.json"
CONTRACT_REVISION = "route2_controlled_instance_policy_2026_07_12_v3"
ENUMERATION_ALGORITHM = "mixed_radix_affine_permutation_v1"
STATE_CLASSIFICATION = "research_candidate"
MAX_SEED = (1 << 63) - 1

SEXES = ("male", "female")
BASE_AVATAR_IDS = {
    "male": "rocketbox_male_adult_01",
    "female": "rocketbox_female_adult_01",
}
TOP_STYLES = ("short_sleeve", "long_sleeve")
BOTTOM_STYLES = ("trousers", "shorts")
TOP_COLORS = ("black", "white", "gray", "navy", "forest_green", "burgundy")
BOTTOM_COLORS = ("black", "gray", "navy", "brown", "beige")
SHOE_COLORS = ("black", "white", "gray", "brown")
HAT_COLORS = ("black", "gray", "navy", "burgundy")
GLASSES_COLORS = ("black", "brown", "navy")
HAIR_COLORS = ("black", "dark_brown", "brown", "blonde", "gray")
HEIGHT_CLASSES = ("short", "standard", "tall")
NOT_APPLICABLE = "not_applicable"

COLOR_PARAMETERS: dict[str, dict[str, Any]] = {
    "black": {"srgb_hex": "#202124", "srgb_u8": [32, 33, 36]},
    "white": {"srgb_hex": "#E8EAED", "srgb_u8": [232, 234, 237]},
    "gray": {"srgb_hex": "#777B80", "srgb_u8": [119, 123, 128]},
    "navy": {"srgb_hex": "#243B63", "srgb_u8": [36, 59, 99]},
    "forest_green": {"srgb_hex": "#2F6B45", "srgb_u8": [47, 107, 69]},
    "burgundy": {"srgb_hex": "#7A3045", "srgb_u8": [122, 48, 69]},
    "brown": {"srgb_hex": "#65483A", "srgb_u8": [101, 72, 58]},
    "beige": {"srgb_hex": "#B9A98C", "srgb_u8": [185, 169, 140]},
    "dark_brown": {"srgb_hex": "#352821", "srgb_u8": [53, 40, 33]},
    "blonde": {"srgb_hex": "#C6A86B", "srgb_u8": [198, 168, 107]},
}

HEIGHT_PROFILES: dict[str, dict[str, Any]] = {
    "short": {"scale_ratio": 0.94, "ratio_tolerance": 0.005},
    "standard": {"scale_ratio": 1.0, "ratio_tolerance": 0.005},
    "tall": {"scale_ratio": 1.06, "ratio_tolerance": 0.005},
}

BASE_DEFAULTS: dict[str, dict[str, Any]] = {
    "male": {
        "top_style": "short_sleeve",
        "top_color": "forest_green",
        "bottom_style": "trousers",
        "bottom_color": "gray",
        "shoe_color": "gray",
        "hat_enabled": False,
        "hat_color": NOT_APPLICABLE,
        "glasses_enabled": False,
        "glasses_color": NOT_APPLICABLE,
        "hair_color": "dark_brown",
        "hair_state": "standard_exposed",
        "height_class": "standard",
    },
    "female": {
        "top_style": "short_sleeve",
        "top_color": "burgundy",
        "bottom_style": "trousers",
        "bottom_color": "black",
        "shoe_color": "gray",
        "hat_enabled": False,
        "hat_color": NOT_APPLICABLE,
        "glasses_enabled": False,
        "glasses_color": NOT_APPLICABLE,
        "hair_color": "blonde",
        "hair_state": "standard_exposed",
        "height_class": "standard",
    },
}

# These are pinned slots, not self-authenticating claims that files already
# exist.  Execution must authenticate a qualified-template record for every
# resolved non-null slot before assembling an instance.
TEMPLATE_REGISTRY: dict[str, Any] = {
    "male": {
        "body": "route2_male_qualified_body_v1",
        "top": {
            "short_sleeve": "route2_male_top_short_sleeve_v1",
            "long_sleeve": "route2_male_top_long_sleeve_v1",
        },
        "bottom": {
            "trousers": "route2_male_bottom_trousers_v1",
            "shorts": "route2_male_bottom_shorts_v1",
        },
        "shoes": "route2_male_shoes_fixed_v1",
        "hair": {
            "standard_exposed": "route2_male_hair_fixed_exposed_v1",
            "hat_compatible": "route2_male_hair_fixed_hat_compatible_v1",
        },
    },
    "female": {
        "body": "route2_female_qualified_body_v1",
        "top": {
            "short_sleeve": "route2_female_top_short_sleeve_v1",
            "long_sleeve": "route2_female_top_long_sleeve_v1",
        },
        "bottom": {
            "trousers": "route2_female_bottom_trousers_v1",
            "shorts": "route2_female_bottom_shorts_v1",
        },
        "shoes": "route2_female_shoes_fixed_v1",
        "hair": {
            "standard_exposed": "route2_female_hair_fixed_exposed_v1",
            "hat_compatible": "route2_female_hair_fixed_hat_compatible_v1",
        },
    },
    "shared_accessories": {
        "hat": "route2_hat_plain_baseball_cap_v1",
        "glasses": "route2_glasses_thin_rectangular_v1",
    },
}

ATTRIBUTE_FIELDS = frozenset(
    {
        "sex",
        "base_avatar_id",
        "top_style",
        "top_color",
        "bottom_style",
        "bottom_color",
        "shoe_color",
        "hat_enabled",
        "hat_color",
        "glasses_enabled",
        "glasses_color",
        "hair_color",
        "hair_state",
        "height_class",
    }
)

LOCKED_ATTRIBUTE_FIELDS = frozenset(
    {
        "skin_color",
        "skin_tone",
        "body_shape",
        "body_type",
        "face_identity",
        "facial_hair",
        "beard",
        "jewelry",
        "backpack",
        "heel_height",
        "high_heels",
    }
)
GEOMETRY_PROPOSAL_FIELDS = frozenset(
    {
        "shoe_style",
        "shoe_geometry",
        "hat_style",
        "glasses_style",
        "hair_style",
        "hair_geometry",
        "top_geometry",
        "bottom_geometry",
    }
)

BASE_QA_CHECKS = (
    "qualified_base_record_authenticated",
    "resolved_template_records_authenticated",
    "identity_and_body_shape_locked",
    "semantic_masks_nonoverlapping_and_complete",
    "non_target_regions_byte_stable",
    "deterministic_material_parameters_reproduced",
    "pbr_material_glb_roundtrip_passed",
    "bind_pose_skin_glb_roundtrip_passed",
    "walking_and_idle_glb_roundtrip_passed",
    "feet_grounded_without_obvious_floating",
    "front_negative_y_and_motion_direction_aligned",
    "front_side_top_feet_skeleton_media_rendered",
)


class ControlledInstanceContractError(RuntimeError):
    """The controlled v3 space or one of its instances is not canonical."""


def _require_seed(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= MAX_SEED:
        raise ControlledInstanceContractError(
            f"seed must be an integer in [0, {MAX_SEED}]"
        )
    return value


def _require_index(value: Any, description: str, *, upper: int) -> int:
    if type(value) is not int or not 0 <= value < upper:
        raise ControlledInstanceContractError(
            f"{description} must be an integer in [0, {upper})"
        )
    return value


TOP_OPTIONS = tuple((style, color) for style in TOP_STYLES for color in TOP_COLORS)
BOTTOM_OPTIONS = tuple(
    (style, color) for style in BOTTOM_STYLES for color in BOTTOM_COLORS
)
HAT_OPTIONS = ((False, NOT_APPLICABLE, "standard_exposed"),) + tuple(
    (True, color, "hat_compatible") for color in HAT_COLORS
)
GLASSES_OPTIONS = ((False, NOT_APPLICABLE),) + tuple(
    (True, color) for color in GLASSES_COLORS
)
DIMENSION_ORDER = (
    "sex",
    "top",
    "bottom",
    "shoe_color",
    "hat",
    "glasses",
    "hair_color",
    "height_class",
)
DIMENSION_OPTIONS: tuple[tuple[Any, ...], ...] = (
    SEXES,
    TOP_OPTIONS,
    BOTTOM_OPTIONS,
    SHOE_COLORS,
    HAT_OPTIONS,
    GLASSES_OPTIONS,
    HAIR_COLORS,
    HEIGHT_CLASSES,
)
RADICES = tuple(len(options) for options in DIMENSION_OPTIONS)
COMBINATION_COUNT = math.prod(RADICES)


def validate_attributes(value: Any) -> dict[str, Any]:
    """Validate one request without accepting aliases or unconstrained fields."""

    if not isinstance(value, Mapping) or set(value) != ATTRIBUTE_FIELDS:
        fields = set(value) if isinstance(value, Mapping) else set()
        missing = sorted(ATTRIBUTE_FIELDS - fields)
        extra = sorted(fields - ATTRIBUTE_FIELDS)
        raise ControlledInstanceContractError(
            f"controlled attributes are incomplete or unexpected: "
            f"missing={missing} extra={extra}"
        )
    sex = value.get("sex")
    if sex not in SEXES:
        raise ControlledInstanceContractError("sex must be exactly male or female")
    if value.get("base_avatar_id") != BASE_AVATAR_IDS[str(sex)]:
        raise ControlledInstanceContractError(
            "base_avatar_id must match the pinned base for sex"
        )
    if value.get("top_style") not in TOP_STYLES:
        raise ControlledInstanceContractError("top_style is outside the pinned templates")
    if value.get("top_color") not in TOP_COLORS:
        raise ControlledInstanceContractError("top_color is outside the finite palette")
    if value.get("bottom_style") not in BOTTOM_STYLES:
        raise ControlledInstanceContractError(
            "bottom_style is outside the pinned templates"
        )
    if value.get("bottom_color") not in BOTTOM_COLORS:
        raise ControlledInstanceContractError("bottom_color is outside the finite palette")
    if value.get("shoe_color") not in SHOE_COLORS:
        raise ControlledInstanceContractError("shoe_color is outside the finite palette")
    if type(value.get("hat_enabled")) is not bool:
        raise ControlledInstanceContractError("hat_enabled must be a boolean")
    expected_hair_state = (
        "hat_compatible" if value["hat_enabled"] else "standard_exposed"
    )
    allowed_hat_colors = HAT_COLORS if value["hat_enabled"] else (NOT_APPLICABLE,)
    if value.get("hat_color") not in allowed_hat_colors:
        raise ControlledInstanceContractError(
            "hat color must be finite when enabled and not_applicable when disabled"
        )
    if value.get("hair_state") != expected_hair_state:
        raise ControlledInstanceContractError(
            "hair_state must switch deterministically with the hat"
        )
    if type(value.get("glasses_enabled")) is not bool:
        raise ControlledInstanceContractError("glasses_enabled must be a boolean")
    allowed_glasses_colors = (
        GLASSES_COLORS if value["glasses_enabled"] else (NOT_APPLICABLE,)
    )
    if value.get("glasses_color") not in allowed_glasses_colors:
        raise ControlledInstanceContractError(
            "glasses color must be finite when enabled and not_applicable when disabled"
        )
    if value.get("hair_color") not in HAIR_COLORS:
        raise ControlledInstanceContractError("hair_color is outside the finite palette")
    if value.get("height_class") not in HEIGHT_CLASSES:
        raise ControlledInstanceContractError(
            "height_class must be exactly short, standard, or tall"
        )
    return {field: copy.deepcopy(value[field]) for field in sorted(ATTRIBUTE_FIELDS)}


def _indices_from_ordinal(ordinal: int) -> tuple[int, ...]:
    ordinal = _require_index(ordinal, "ordinal", upper=COMBINATION_COUNT)
    remainder = ordinal
    reversed_indices: list[int] = []
    for radix in reversed(RADICES):
        reversed_indices.append(remainder % radix)
        remainder //= radix
    if remainder != 0:  # pragma: no cover - the range check makes this impossible.
        raise ControlledInstanceContractError("ordinal did not fit the mixed-radix space")
    return tuple(reversed(reversed_indices))


def attributes_from_ordinal(ordinal: int) -> dict[str, Any]:
    indices = _indices_from_ordinal(ordinal)
    sex = SEXES[indices[0]]
    top_style, top_color = TOP_OPTIONS[indices[1]]
    bottom_style, bottom_color = BOTTOM_OPTIONS[indices[2]]
    hat_enabled, hat_color, hair_state = HAT_OPTIONS[indices[4]]
    glasses_enabled, glasses_color = GLASSES_OPTIONS[indices[5]]
    return validate_attributes(
        {
            "sex": sex,
            "base_avatar_id": BASE_AVATAR_IDS[sex],
            "top_style": top_style,
            "top_color": top_color,
            "bottom_style": bottom_style,
            "bottom_color": bottom_color,
            "shoe_color": SHOE_COLORS[indices[3]],
            "hat_enabled": hat_enabled,
            "hat_color": hat_color,
            "glasses_enabled": glasses_enabled,
            "glasses_color": glasses_color,
            "hair_color": HAIR_COLORS[indices[6]],
            "hair_state": hair_state,
            "height_class": HEIGHT_CLASSES[indices[7]],
        }
    )


def ordinal_from_attributes(value: Any) -> int:
    attributes = validate_attributes(value)
    options = (
        attributes["sex"],
        (attributes["top_style"], attributes["top_color"]),
        (attributes["bottom_style"], attributes["bottom_color"]),
        attributes["shoe_color"],
        (
            attributes["hat_enabled"],
            attributes["hat_color"],
            attributes["hair_state"],
        ),
        (attributes["glasses_enabled"], attributes["glasses_color"]),
        attributes["hair_color"],
        attributes["height_class"],
    )
    ordinal = 0
    for choices, selected in zip(DIMENSION_OPTIONS, options, strict=True):
        try:
            index = choices.index(selected)
        except ValueError as error:  # pragma: no cover - validation has checked this.
            raise ControlledInstanceContractError(
                "validated attributes were absent from the canonical dimensions"
            ) from error
        ordinal = ordinal * len(choices) + index
    return ordinal


def canonical_instance_id(ordinal: int) -> str:
    ordinal = _require_index(ordinal, "ordinal", upper=COMBINATION_COUNT)
    combinations_per_sex = COMBINATION_COUNT // len(SEXES)
    sex = SEXES[ordinal // combinations_per_sex]
    return f"route2_controlled_v3_{sex}_{ordinal:06d}"


def _material_descriptor(region: str, color_id: str) -> dict[str, Any]:
    if color_id not in COLOR_PARAMETERS:
        raise ControlledInstanceContractError(f"unknown color parameter: {color_id}")
    color = COLOR_PARAMETERS[color_id]
    return {
        "semantic_mask": region,
        "color_id": color_id,
        "srgb_hex": color["srgb_hex"],
        "srgb_u8": list(color["srgb_u8"]),
        "alpha_u8": 255,
        "color_space": "iec_61966_2_1_srgb",
        "operation": "replace_base_color_preserve_pbr_detail_v1",
    }


def _resolved_templates(attributes: Mapping[str, Any]) -> dict[str, Any]:
    sex = str(attributes["sex"])
    registry = TEMPLATE_REGISTRY[sex]
    shared = TEMPLATE_REGISTRY["shared_accessories"]
    return {
        "body": registry["body"],
        "top": registry["top"][attributes["top_style"]],
        "bottom": registry["bottom"][attributes["bottom_style"]],
        "shoes": registry["shoes"],
        "hat": shared["hat"] if attributes["hat_enabled"] else None,
        "glasses": shared["glasses"] if attributes["glasses_enabled"] else None,
        "hair": registry["hair"][attributes["hair_state"]],
    }


def _material_parameters(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "top": _material_descriptor("garment_top", str(attributes["top_color"])),
        "bottom": _material_descriptor(
            "garment_bottom", str(attributes["bottom_color"])
        ),
        "shoes": _material_descriptor("footwear", str(attributes["shoe_color"])),
        "hat": (
            _material_descriptor("headwear", str(attributes["hat_color"]))
            if attributes["hat_enabled"]
            else None
        ),
        "glasses": (
            _material_descriptor("eyewear", str(attributes["glasses_color"]))
            if attributes["glasses_enabled"]
            else None
        ),
        "hair": _material_descriptor("hair", str(attributes["hair_color"])),
    }


def _qa_class(attributes: Mapping[str, Any]) -> str:
    defaults = BASE_DEFAULTS[str(attributes["sex"])]
    if all(attributes[key] == defaults[key] for key in defaults):
        return "qualified_base_reuse"
    geometry_keys = (
        "top_style",
        "bottom_style",
        "hat_enabled",
        "glasses_enabled",
        "hair_state",
        "height_class",
    )
    if all(attributes[key] == defaults[key] for key in geometry_keys):
        return "deterministic_material_variant"
    return "approved_template_composition"


def _required_qa_checks(attributes: Mapping[str, Any]) -> list[str]:
    checks = list(BASE_QA_CHECKS)
    defaults = BASE_DEFAULTS[str(attributes["sex"])]
    if attributes["top_style"] != defaults["top_style"]:
        checks.append("top_template_topology_skinning_and_cloth_clearance_passed")
    if attributes["bottom_style"] != defaults["bottom_style"]:
        checks.append("bottom_template_topology_skinning_and_cloth_clearance_passed")
    if attributes["height_class"] != "standard":
        checks.append("height_profile_ratio_and_grounding_passed")
    if attributes["hat_enabled"]:
        checks.extend(
            [
                "hat_rigid_head_binding_passed",
                "hat_hair_clearance_and_hat_compatible_state_passed",
            ]
        )
    if attributes["glasses_enabled"]:
        checks.extend(
            [
                "glasses_rigid_head_binding_passed",
                "glasses_face_clearance_passed",
            ]
        )
    return checks


def build_instance_from_ordinal(ordinal: int) -> dict[str, Any]:
    ordinal = _require_index(ordinal, "ordinal", upper=COMBINATION_COUNT)
    attributes = attributes_from_ordinal(ordinal)
    height = HEIGHT_PROFILES[str(attributes["height_class"])]
    return {
        "schema": INSTANCE_SCHEMA,
        "instance_id": canonical_instance_id(ordinal),
        "ordinal": ordinal,
        "base_avatar_id": attributes["base_avatar_id"],
        "attributes": attributes,
        "resolved_templates": _resolved_templates(attributes),
        "material_parameters": _material_parameters(attributes),
        "height_profile": {
            "height_class": attributes["height_class"],
            "scale_ratio": height["scale_ratio"],
            "ratio_tolerance": height["ratio_tolerance"],
            "operation": "uniform_mesh_skeleton_scale_from_qualified_base_v1",
        },
        "execution": {
            "instance_backend": "deterministic_template_and_semantic_material_v1",
            "template_precondition": "all_resolved_templates_must_be_qualified",
            "flux2": "forbidden_for_instance_expansion",
            "pixal3d": "forbidden_for_instance_expansion",
            "new_geometry": "full_template_qualification_required",
        },
        "automatic_qa": {
            "qa_class": _qa_class(attributes),
            "required_checks": _required_qa_checks(attributes),
            "locomotion_policy": "grounding_required_minor_foot_sliding_accepted",
            "initial_classification": STATE_CLASSIFICATION,
            "all_checks_pass": "eligible_for_formal_registration",
            "any_check_fail": "rejected",
        },
        "state_classification": STATE_CLASSIFICATION,
    }


def validate_instance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlledInstanceContractError("controlled instance must be a mapping")
    if set(value) != {
        "schema",
        "instance_id",
        "ordinal",
        "base_avatar_id",
        "attributes",
        "resolved_templates",
        "material_parameters",
        "height_profile",
        "execution",
        "automatic_qa",
        "state_classification",
    }:
        raise ControlledInstanceContractError(
            "controlled instance fields are incomplete or unexpected"
        )
    common.reject_user_approval(
        value, ControlledInstanceContractError, "controlled instance"
    )
    if value.get("schema") != INSTANCE_SCHEMA:
        raise ControlledInstanceContractError(
            f"controlled instance schema must be {INSTANCE_SCHEMA}"
        )
    ordinal = _require_index(
        value.get("ordinal"), "ordinal", upper=COMBINATION_COUNT
    )
    expected = build_instance_from_ordinal(ordinal)
    if value != expected:
        raise ControlledInstanceContractError(
            "controlled instance differs from its canonical ordinal expansion"
        )
    return expected


def _seed_parameters(seed: int) -> tuple[int, int]:
    seed = _require_seed(seed)
    digest = hashlib.sha256(
        f"{ENUMERATION_ALGORITHM}:{CONTRACT_REVISION}:{seed}".encode("utf-8")
    ).digest()
    multiplier = int.from_bytes(digest[:8], "big") % COMBINATION_COUNT
    while math.gcd(multiplier, COMBINATION_COUNT) != 1:
        multiplier = (multiplier + 1) % COMBINATION_COUNT
    offset = int.from_bytes(digest[8:16], "big") % COMBINATION_COUNT
    return multiplier, offset


def seeded_ordinal(seed: int, index: int) -> int:
    seed = _require_seed(seed)
    index = _require_index(index, "seeded index", upper=COMBINATION_COUNT)
    multiplier, offset = _seed_parameters(seed)
    return (offset + multiplier * index) % COMBINATION_COUNT


def iter_seeded_instances(
    seed: int, *, start_index: int = 0, limit: int | None = None
) -> Iterator[dict[str, Any]]:
    seed = _require_seed(seed)
    if type(start_index) is not int or not 0 <= start_index <= COMBINATION_COUNT:
        raise ControlledInstanceContractError(
            f"start_index must be in [0, {COMBINATION_COUNT}]"
        )
    remaining = COMBINATION_COUNT - start_index
    if limit is None:
        limit = remaining
    if type(limit) is not int or not 0 <= limit <= remaining:
        raise ControlledInstanceContractError(
            "limit must be a non-negative integer within the remaining permutation"
        )
    multiplier, offset = _seed_parameters(seed)
    for index in range(start_index, start_index + limit):
        ordinal = (offset + multiplier * index) % COMBINATION_COUNT
        yield build_instance_from_ordinal(ordinal)


def classify_request(value: Any) -> dict[str, Any]:
    """Explain invalid proposals without weakening the strict instance validator."""

    if not isinstance(value, Mapping):
        return {
            "classification": "rejected_out_of_contract",
            "execution_authorized": False,
            "reason": "request_must_be_a_mapping",
        }
    fields = set(value)
    if fields & LOCKED_ATTRIBUTE_FIELDS:
        return {
            "classification": "rejected_locked_attribute",
            "execution_authorized": False,
            "reason": "skin_body_face_or_disallowed_accessory_change_requested",
        }
    if fields & GEOMETRY_PROPOSAL_FIELDS:
        return {
            "classification": "full_geometry_template_qualification_required",
            "execution_authorized": False,
            "reason": "new_or_unpinned_geometry_requested",
        }
    try:
        attributes = validate_attributes(value)
    except ControlledInstanceContractError:
        geometry_values = (
            value.get("top_style") not in (None, *TOP_STYLES)
            or value.get("bottom_style") not in (None, *BOTTOM_STYLES)
        )
        if geometry_values:
            return {
                "classification": "full_geometry_template_qualification_required",
                "execution_authorized": False,
                "reason": "new_or_unpinned_geometry_requested",
            }
        return {
            "classification": "rejected_out_of_contract",
            "execution_authorized": False,
            "reason": "request_failed_finite_space_validation",
        }
    return {
        "classification": _qa_class(attributes),
        "execution_authorized": True,
        "reason": "eligible_after_qualified_template_preflight",
    }


def classify_qa_results(instance: Any, results: Any) -> dict[str, Any]:
    canonical = validate_instance(instance)
    required = canonical["automatic_qa"]["required_checks"]
    if not isinstance(results, Mapping) or set(results) != set(required):
        raise ControlledInstanceContractError(
            "QA results must contain exactly every required automatic check"
        )
    if any(type(results[name]) is not bool for name in required):
        raise ControlledInstanceContractError("every QA result must be a boolean")
    failed = [name for name in required if not results[name]]
    passed = not failed
    return {
        "schema": QA_RESULT_SCHEMA,
        "instance_id": canonical["instance_id"],
        "qa_class": canonical["automatic_qa"]["qa_class"],
        "status": "passed" if passed else "failed",
        "failed_checks": failed,
        "state_classification": STATE_CLASSIFICATION if passed else "rejected",
        "eligible_for_formal_registration": passed,
    }


def _policy_payload() -> dict[str, Any]:
    return {
        "bases": {
            sex: {
                "base_avatar_id": BASE_AVATAR_IDS[sex],
                "default_attributes": copy.deepcopy(BASE_DEFAULTS[sex]),
            }
            for sex in SEXES
        },
        "palettes": {
            "top": list(TOP_COLORS),
            "bottom": list(BOTTOM_COLORS),
            "shoes": list(SHOE_COLORS),
            "hat": list(HAT_COLORS),
            "glasses": list(GLASSES_COLORS),
            "hair": list(HAIR_COLORS),
            "parameters": copy.deepcopy(COLOR_PARAMETERS),
        },
        "template_registry": copy.deepcopy(TEMPLATE_REGISTRY),
        "height_profiles": copy.deepcopy(HEIGHT_PROFILES),
        "identity_and_shape_locks": {
            "skin_tone": "base_locked_v1",
            "body_shape": "base_locked_v1",
            "face_identity": "base_locked_v1",
            "hairstyle_geometry": "fixed_per_base_with_hat_compatible_state_v1",
            "facial_hair": "forbidden_v1",
            "jewelry": "forbidden_v1",
            "backpack": "forbidden_v1",
            "shoe_geometry": "fixed_per_base_v1",
            "high_heels_and_platforms": "forbidden_v1",
            "skirts_and_dresses": "forbidden_v1",
        },
        "execution_policy": {
            "ordinary_color_instances": "deterministic_semantic_mask_material_transform",
            "approved_shape_instances": "qualified_template_composition",
            "flux2_for_instance_expansion": "forbidden",
            "pixal3d_for_instance_expansion": "forbidden",
            "new_geometry": "flux2_pixal_tokenrig_walk_idle_full_qualification_required",
            "template_authentication": "required_before_every_batch",
        },
        "compatibility_rules": [
            "sex_selects_exactly_one_pinned_base_avatar",
            "hat_off_requires_not_applicable_color_and_standard_exposed_hair",
            "hat_on_requires_finite_color_and_hat_compatible_hair",
            "glasses_off_requires_not_applicable_color",
            "glasses_on_requires_finite_color",
            "hat_and_glasses_are_rigid_head_bound",
            "shoe_geometry_never_changes_between_instances",
            "height_uses_only_pinned_uniform_mesh_skeleton_profiles",
        ],
        "qa_policy": {
            "base_checks": list(BASE_QA_CHECKS),
            "locomotion": "grounding_required_minor_foot_sliding_accepted",
            "all_checks_pass": "eligible_for_formal_registration",
            "any_check_fail": "rejected",
        },
    }


def _canonical_instance_id_digest() -> str:
    digest = hashlib.sha256()
    for ordinal in range(COMBINATION_COUNT):
        digest.update(canonical_instance_id(ordinal).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


_CANONICAL_INSTANCE_ID_SHA256 = _canonical_instance_id_digest()


def build_space_manifest(seed: int) -> dict[str, Any]:
    seed = _require_seed(seed)
    policy = _policy_payload()
    policy_sha256 = hashlib.sha256(common.canonical_json(policy).encode("utf-8")).hexdigest()
    multiplier, offset = _seed_parameters(seed)
    dimension_manifest = {
        "sex": list(SEXES),
        "top": [list(option) for option in TOP_OPTIONS],
        "bottom": [list(option) for option in BOTTOM_OPTIONS],
        "shoe_color": list(SHOE_COLORS),
        "hat": [list(option) for option in HAT_OPTIONS],
        "glasses": [list(option) for option in GLASSES_OPTIONS],
        "hair_color": list(HAIR_COLORS),
        "height_class": list(HEIGHT_CLASSES),
    }
    return {
        "schema": SCHEMA,
        "contract_revision": CONTRACT_REVISION,
        "state_classification": STATE_CLASSIFICATION,
        "policy": policy,
        "policy_sha256": policy_sha256,
        "exhaustive_space": {
            "representation": "compact_compatible_dimensions_plus_full_id_digest",
            "dimension_order": list(DIMENSION_ORDER),
            "radices": list(RADICES),
            "dimensions": dimension_manifest,
            "combination_count": COMBINATION_COUNT,
            "canonical_instance_id_sha256": _CANONICAL_INSTANCE_ID_SHA256,
            "complete": True,
        },
        "seeded_enumeration": {
            "algorithm": ENUMERATION_ALGORITHM,
            "seed": seed,
            "multiplier": multiplier,
            "offset": offset,
            "count": COMBINATION_COUNT,
            "full_permutation": True,
            "first_instance_ids": [
                canonical_instance_id(seeded_ordinal(seed, index))
                for index in range(16)
            ],
        },
        "historical_v2": {
            "jobs_v2_role": "immutable_historical_flux_pixal_canary_only",
            "superseded_for_bulk_instance_expansion": True,
            "overwrite_forbidden": True,
        },
        "publication": {"no_replace": True, "artifact_mode": "0444"},
        "user_acceptance": "not_claimed_by_contract",
    }


def validate_space_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlledInstanceContractError("controlled space manifest must be a mapping")
    common.reject_user_approval(
        value, ControlledInstanceContractError, "controlled space manifest"
    )
    enumeration = value.get("seeded_enumeration")
    if not isinstance(enumeration, Mapping):
        raise ControlledInstanceContractError("seeded_enumeration must be a mapping")
    seed = _require_seed(enumeration.get("seed"))
    expected = build_space_manifest(seed)
    if value != expected:
        raise ControlledInstanceContractError(
            "controlled space manifest differs from the canonical v3 policy"
        )
    return expected


def publish_space_manifest(payload: Any, output_dir: Path) -> Path:
    validated = common.stable_mapping_snapshot(
        lambda: validate_space_manifest(payload),
        ControlledInstanceContractError,
        "controlled space manifest source",
    )
    root = common.require_real_directory(
        output_dir, "controlled space output directory", ControlledInstanceContractError
    )
    destination = root / FILENAME
    return common.write_json_immutable_noreplace(
        destination,
        validated,
        ControlledInstanceContractError,
        "Route-2 controlled instance space v3",
        prelink_validator=lambda: validate_space_manifest(payload),
    )


def _validate_published_space_manifest_once(path: Path) -> dict[str, Any]:
    supplied = common.absolute(path)
    if supplied.name != FILENAME:
        raise ControlledInstanceContractError(
            f"controlled space manifest must be named {FILENAME}"
        )
    root = common.require_real_directory(
        supplied.parent,
        "controlled space manifest directory",
        ControlledInstanceContractError,
    )
    payload, _ = common.load_json_mapping_record(
        supplied,
        root=root,
        description="controlled space manifest",
        error_type=ControlledInstanceContractError,
        require_mode=0o444,
    )
    return validate_space_manifest(payload)


def validate_published_space_manifest(path: Path) -> dict[str, Any]:
    return common.stable_mapping_snapshot(
        lambda: _validate_published_space_manifest_once(path),
        ControlledInstanceContractError,
        "controlled space manifest",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = publish_space_manifest(build_space_manifest(args.seed), args.output_dir)
    result = validate_published_space_manifest(destination)
    print(
        json.dumps(
            {
                "path": str(destination),
                "combination_count": result["exhaustive_space"]["combination_count"],
                "seed": result["seeded_enumeration"]["seed"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ATTRIBUTE_FIELDS",
    "BASE_AVATAR_IDS",
    "BASE_DEFAULTS",
    "BOTTOM_COLORS",
    "BOTTOM_STYLES",
    "COLOR_PARAMETERS",
    "COMBINATION_COUNT",
    "CONTRACT_REVISION",
    "ControlledInstanceContractError",
    "ENUMERATION_ALGORITHM",
    "FILENAME",
    "GLASSES_COLORS",
    "HAIR_COLORS",
    "HAT_COLORS",
    "HEIGHT_CLASSES",
    "HEIGHT_PROFILES",
    "INSTANCE_SCHEMA",
    "NOT_APPLICABLE",
    "QA_RESULT_SCHEMA",
    "SCHEMA",
    "SEXES",
    "SHOE_COLORS",
    "STATE_CLASSIFICATION",
    "TEMPLATE_REGISTRY",
    "TOP_COLORS",
    "TOP_STYLES",
    "attributes_from_ordinal",
    "build_instance_from_ordinal",
    "build_space_manifest",
    "canonical_instance_id",
    "classify_qa_results",
    "classify_request",
    "iter_seeded_instances",
    "ordinal_from_attributes",
    "publish_space_manifest",
    "seeded_ordinal",
    "validate_attributes",
    "validate_instance",
    "validate_published_space_manifest",
    "validate_space_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
