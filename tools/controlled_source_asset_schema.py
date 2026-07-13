#!/usr/bin/env python3
"""Strict controlled-attribute contracts for AVEngine source assets.

The module is intentionally standard-library only.  It turns a versioned,
audited attribute profile into immutable generation requests, compiles the
request into either one FLUX.2/Pixal3D job or one Rocketbox material edit plan,
builds QA pairs by comparing absolute attributes, and creates lineage-grouped
dataset manifests.

Legacy ``source_asset_v1`` manifests are not modified or accepted as v2 input.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import re
import string
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


PROFILE_SCHEMA = "avengine_attribute_profile_v1"
REQUEST_SCHEMA = "avengine_instance_request_v1"
REQUEST_BATCH_SCHEMA = "avengine_instance_request_batch_v1"
PAIR_SCHEMA = "avengine_instance_pair_v1"
PAIR_PLAN_SCHEMA = "avengine_instance_pair_plan_v1"
SOURCE_ASSET_SCHEMA = "source_asset_v2"
GENERATION_PLAN_SCHEMA = "avengine_controlled_generation_plan_v1"
DATASET_SCHEMA = "avengine_controlled_source_dataset_v1"

SAMPLER_ALGORITHM = "balanced_quota_sampler_v1"

STATE_CLASSIFICATIONS = frozenset(
    {
        "formal_dataset_asset",
        "research_candidate",
        "technical_spike_only",
        "rejected",
    }
)
ASSET_CLASSES = frozenset({"animal", "human"})
ROUTES = frozenset({"flux2_pixal3d_animal_v1", "rocketbox_material_v1"})
QA_STATUSES = frozenset({"passed", "pending", "rejected", "not_applicable"})
QA_FIELDS = (
    "reference_2d",
    "static_mesh",
    "binding",
    "walking",
    "idle",
    "ue_import_readback",
    "apartment_media",
    "audio",
)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE_ATTRIBUTE_NAMES = frozenset(
    {
        "from",
        "to",
        "operation",
        "one_step_lighter",
        "one_step_darker",
        "lighter_than_original",
        "darker_than_original",
        "based_on_original",
    }
)

_PROFILE_FIELDS = frozenset(
    {
        "schema",
        "profile_schema_id",
        "profile_revision",
        "asset_class",
        "lineage_group_id",
        "state_classification",
        "taxonomy",
        "base_template",
        "fixed_attributes",
        "sampled_attribute_domains",
        "forbidden_combinations",
        "generation_contract",
        "target_physical_profiles",
        "rig_profile",
        "acoustic_profile",
        "locked_attributes",
        "qa_contract",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_schema_id",
        "profile_sha256",
        "asset_class",
        "lineage_group_id",
        "state_classification",
        "sampler",
        "taxonomy",
        "fixed_attributes",
        "sampled_attributes",
        "target_physical_profile",
        "generation_plan",
        "rig_profile",
        "acoustic_profile",
        "locked_attributes",
    }
)
_SOURCE_ASSET_FIELDS = frozenset(
    {
        "schema",
        "asset_id",
        "profile_schema_id",
        "profile_sha256",
        "request_sha256",
        "asset_class",
        "lineage_group_id",
        "taxonomy",
        "fixed_attributes",
        "sampled_attributes",
        "semantic_attributes",
        "target_physical_profile",
        "artifacts",
        "physical_measurements",
        "rig",
        "acoustic_profile",
        "provenance",
        "rights",
        "qa",
        "state_classification",
    }
)


class ContractError(ValueError):
    """Raised when a controlled-asset contract is incomplete or contradictory."""


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON representation used for all identities."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"value is not canonical JSON: {error}") from error


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _require_exact_fields(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    mapping = _require_mapping(value, label)
    actual = set(mapping)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        raise ContractError(f"{label} fields are invalid: missing={missing} extra={extra}")
    return mapping


def _require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ContractError(f"{label} must match {_ID_RE.pattern}")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be non-empty text")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be a finite number")
    if positive and result <= 0.0:
        raise ContractError(f"{label} must be positive")
    return result


def _require_scalar(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractError(f"{label} must be finite")
        return value
    raise ContractError(f"{label} must be a JSON scalar")


def _validate_artifact(value: Any, label: str) -> dict[str, Any]:
    artifact = _require_exact_fields(
        value, frozenset({"root_id", "path", "sha256", "size_bytes"}), label
    )
    _require_id(artifact["root_id"], f"{label}.root_id")
    path = _require_text(artifact["path"], f"{label}.path")
    if "\x00" in path:
        raise ContractError(f"{label}.path contains NUL")
    _require_sha256(artifact["sha256"], f"{label}.sha256")
    size = artifact["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ContractError(f"{label}.size_bytes must be a positive integer")
    return _deepcopy(dict(artifact))


def _validate_attribute_key(value: Any, label: str) -> str:
    key = _require_id(value, label)
    if key in _RELATIVE_ATTRIBUTE_NAMES:
        raise ContractError(f"{label} uses a forbidden relative attribute name: {key}")
    return key


def _validate_attribute_values(value: Any, label: str) -> dict[str, Any]:
    mapping = _require_mapping(value, label)
    result: dict[str, Any] = {}
    for raw_key, raw_value in mapping.items():
        key = _validate_attribute_key(raw_key, f"{label} key")
        result[key] = _require_scalar(raw_value, f"{label}.{key}")
    return result


def _combined_attribute_values(profile: Mapping[str, Any], sampled: Mapping[str, str]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for section_name, section in (
        ("taxonomy", profile["taxonomy"]),
        ("fixed_attributes", profile["fixed_attributes"]),
        ("sampled_attributes", sampled),
    ):
        overlap = set(combined) & set(section)
        if overlap:
            raise ContractError(f"attribute names overlap in {section_name}: {sorted(overlap)}")
        combined.update(section)
    return combined


def _format_placeholders(template: str, label: str) -> list[str]:
    placeholders: list[str] = []
    try:
        parsed = string.Formatter().parse(template)
        for _literal, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if not field_name or "." in field_name or "[" in field_name:
                raise ContractError(f"{label} contains an invalid placeholder: {field_name!r}")
            if format_spec or conversion:
                raise ContractError(f"{label} placeholders cannot use conversion or format specs")
            placeholders.append(field_name)
    except ValueError as error:
        raise ContractError(f"{label} is not a valid format template: {error}") from error
    if len(placeholders) != len(set(placeholders)):
        raise ContractError(f"{label} must mention every attribute exactly once")
    return placeholders


def _validate_value_label_map(
    value: Any,
    *,
    expected_values: Mapping[str, Sequence[Any]],
    label: str,
) -> dict[str, dict[str, str]]:
    mapping = _require_mapping(value, label)
    if set(mapping) != set(expected_values):
        raise ContractError(
            f"{label} attributes do not match the complete absolute attribute set"
        )
    result: dict[str, dict[str, str]] = {}
    for attribute, values in expected_values.items():
        labels = _require_mapping(mapping[attribute], f"{label}.{attribute}")
        expected_keys = {str(item) for item in values}
        if set(labels) != expected_keys:
            raise ContractError(f"{label}.{attribute} values do not match its domain")
        result[attribute] = {
            key: _require_text(text, f"{label}.{attribute}.{key}")
            for key, text in labels.items()
        }
    return result


def _validate_generation_contract(
    value: Any,
    *,
    asset_class: str,
    base_template: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
    fixed: Mapping[str, Any],
    domains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    contract = _require_mapping(value, "generation_contract")
    route = contract.get("route")
    if route not in ROUTES:
        raise ContractError(f"unsupported generation route: {route!r}")

    if asset_class == "animal":
        fields = frozenset(
            {
                "route",
                "prompt_template_id",
                "positive_template",
                "pose_guard_prompt",
                "negative_prompt",
                "value_labels",
                "model_revisions",
            }
        )
        _require_exact_fields(contract, fields, "animal generation_contract")
        if route != "flux2_pixal3d_animal_v1":
            raise ContractError("animal profiles must use flux2_pixal3d_animal_v1")
        if base_template.get("kind") != "reference_image":
            raise ContractError("animal base_template.kind must be reference_image")
        _require_id(contract["prompt_template_id"], "prompt_template_id")
        template = _require_text(contract["positive_template"], "positive_template")
        _require_text(contract["pose_guard_prompt"], "pose_guard_prompt")
        _require_text(contract["negative_prompt"], "negative_prompt")

        expected_values: dict[str, Sequence[Any]] = {
            key: [item] for key, item in taxonomy.items()
        }
        expected_values.update({key: [item] for key, item in fixed.items()})
        expected_values.update({key: list(items) for key, items in domains.items()})
        placeholders = _format_placeholders(template, "positive_template")
        if set(placeholders) != set(expected_values):
            raise ContractError(
                "positive_template must mention every taxonomy, fixed, and sampled "
                "attribute exactly once"
            )
        _validate_value_label_map(
            contract["value_labels"],
            expected_values=expected_values,
            label="generation_contract.value_labels",
        )
        revisions = _require_exact_fields(
            contract["model_revisions"],
            frozenset({"flux2", "pixal3d", "dino"}),
            "model_revisions",
        )
        for name, revision in revisions.items():
            _require_text(revision, f"model_revisions.{name}")
        return _deepcopy(dict(contract))

    fields = frozenset(
        {
            "route",
            "material_plan_schema",
            "material_bindings",
            "flux_texture_detail",
        }
    )
    _require_exact_fields(contract, fields, "human generation_contract")
    if route != "rocketbox_material_v1":
        raise ContractError("human profiles must use rocketbox_material_v1")
    if base_template.get("kind") != "rocketbox_avatar":
        raise ContractError("human base_template.kind must be rocketbox_avatar")
    if contract["material_plan_schema"] != "rocketbox_material_edit_plan_v1":
        raise ContractError("unsupported Rocketbox material plan schema")
    bindings = _require_mapping(contract["material_bindings"], "material_bindings")
    if set(bindings) != set(domains):
        raise ContractError("material_bindings must cover every sampled human attribute")
    for attribute, domain in domains.items():
        binding = _require_exact_fields(
            bindings[attribute],
            frozenset(
                {
                    "semantic_mask",
                    "mask_registry",
                    "source_texture_role",
                    "operation",
                    "palette",
                }
            ),
            f"material_bindings.{attribute}",
        )
        _require_id(binding["semantic_mask"], f"{attribute}.semantic_mask")
        _validate_artifact(binding["mask_registry"], f"{attribute}.mask_registry")
        _require_id(binding["source_texture_role"], f"{attribute}.source_texture_role")
        if binding["operation"] != "replace_base_color_preserve_pbr_detail_v1":
            raise ContractError(f"{attribute} uses an unsupported material operation")
        palette = _require_mapping(binding["palette"], f"{attribute}.palette")
        if set(palette) != set(domain):
            raise ContractError(f"{attribute}.palette must match the sampled domain")
        for value_name, color in palette.items():
            color_record = _require_exact_fields(
                color, frozenset({"srgb_u8"}), f"{attribute}.palette.{value_name}"
            )
            srgb = color_record["srgb_u8"]
            if (
                not isinstance(srgb, list)
                or len(srgb) != 3
                or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 or c > 255 for c in srgb)
            ):
                raise ContractError(f"{attribute}.palette.{value_name}.srgb_u8 is invalid")
    detail = _require_exact_fields(
        contract["flux_texture_detail"],
        frozenset({"enabled", "policy"}),
        "flux_texture_detail",
    )
    if not isinstance(detail["enabled"], bool):
        raise ContractError("flux_texture_detail.enabled must be boolean")
    if detail["policy"] != "approved_mask_optional_only":
        raise ContractError("flux_texture_detail policy must remain mask constrained")
    return _deepcopy(dict(contract))


def _validate_target_physical_profiles(
    value: Any,
    *,
    asset_class: str,
    domains: Mapping[str, Sequence[str]],
    state_classification: str,
) -> dict[str, Any]:
    fields = frozenset(
        {
            "profile_id",
            "control_attribute",
            "measurement",
            "mode",
            "reference_value_cm",
            "reference_provenance",
            "values",
        }
    )
    profile = _require_exact_fields(value, fields, "target_physical_profiles")
    _require_id(profile["profile_id"], "target physical profile_id")
    _require_id(profile["measurement"], "target physical measurement")
    reference = profile["reference_value_cm"]
    if reference is not None:
        _require_finite_number(reference, "reference_value_cm", positive=True)
    elif state_classification == "formal_dataset_asset":
        raise ContractError("formal physical profiles require a reference_value_cm")

    provenance = _require_exact_fields(
        profile["reference_provenance"],
        frozenset({"status", "source_id", "artifact", "notes"}),
        "target physical reference_provenance",
    )
    if provenance["status"] not in {"verified", "provisional"}:
        raise ContractError("physical reference provenance must be verified or provisional")
    _require_id(provenance["source_id"], "physical reference source_id")
    _require_text(provenance["notes"], "physical reference notes")
    if provenance["artifact"] is not None:
        _validate_artifact(provenance["artifact"], "physical reference artifact")
    if provenance["status"] == "verified" and provenance["artifact"] is None:
        raise ContractError("verified physical reference provenance requires an artifact")
    if state_classification == "formal_dataset_asset" and provenance["status"] != "verified":
        raise ContractError("formal physical profiles require verified reference provenance")

    values = _require_mapping(profile["values"], "target physical values")
    if asset_class == "animal":
        control = _require_id(profile["control_attribute"], "physical control_attribute")
        if control not in domains:
            raise ContractError("physical control_attribute must be sampled")
        if profile["mode"] not in {
            "relative_to_profile_reference",
            "absolute_measurement",
        }:
            raise ContractError("unsupported animal physical profile mode")
        if set(values) != set(domains[control]):
            raise ContractError("physical profile values must match its control domain")
        for name, record in values.items():
            item = _require_exact_fields(
                record,
                frozenset({"scale_ratio", "tolerance_cm"}),
                f"target physical value {name}",
            )
            _require_finite_number(item["scale_ratio"], f"{name}.scale_ratio", positive=True)
            _require_finite_number(item["tolerance_cm"], f"{name}.tolerance_cm", positive=True)
        return _deepcopy(dict(profile))

    if profile["control_attribute"] is not None:
        raise ContractError("Rocketbox authored height cannot have a sampled control attribute")
    if profile["mode"] != "authored" or set(values) != {"fixed"}:
        raise ContractError("human physical profile must be one authored fixed value")
    item = _require_exact_fields(
        values["fixed"],
        frozenset({"actor_scale", "target_value_cm", "tolerance_cm"}),
        "target physical fixed value",
    )
    actor_scale = _require_finite_number(item["actor_scale"], "actor_scale", positive=True)
    if actor_scale != 1.0:
        raise ContractError("Rocketbox controlled profiles must preserve actor_scale=1.0")
    target = _require_finite_number(item["target_value_cm"], "target_value_cm", positive=True)
    _require_finite_number(item["tolerance_cm"], "tolerance_cm", positive=True)
    if reference is None or not math.isclose(float(reference), target, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError("authored reference and target height must match")
    return _deepcopy(dict(profile))


def _validate_rig_profile(value: Any, *, asset_class: str) -> dict[str, Any]:
    rig = _require_exact_fields(
        value,
        frozenset({"profile_id", "skeleton_family", "actions", "front_axis"}),
        "rig_profile",
    )
    _require_id(rig["profile_id"], "rig profile_id")
    # Skeleton families are names from the source asset and may be case
    # sensitive (Rocketbox uses ``Bip01``/``Bip02``).  Do not normalize them.
    _require_text(rig["skeleton_family"], "rig skeleton_family")
    _require_id(rig["front_axis"], "rig front_axis")
    actions = rig["actions"]
    if not isinstance(actions, list) or not actions or any(not isinstance(x, str) or not x for x in actions):
        raise ContractError("rig actions must be a non-empty string list")
    if len(actions) != len(set(actions)):
        raise ContractError("rig actions must be unique")
    expected = {"Walking", "Idle"} if asset_class == "animal" else {"Walking", "Standing_Idle"}
    if set(actions) != expected:
        raise ContractError(f"{asset_class} rig actions must be exactly {sorted(expected)}")
    return _deepcopy(dict(rig))


def _validate_acoustic_profile(
    value: Any, *, available_attributes: set[str]
) -> dict[str, Any]:
    acoustic = _require_exact_fields(
        value,
        frozenset(
            {
                "profile_id",
                "default_event_class",
                "allowed_event_classes",
                "selection_attributes",
            }
        ),
        "acoustic_profile",
    )
    _require_id(acoustic["profile_id"], "acoustic profile_id")
    default = _require_id(acoustic["default_event_class"], "default_event_class")
    allowed = acoustic["allowed_event_classes"]
    if (
        not isinstance(allowed, list)
        or not allowed
        or len(allowed) != len(set(allowed))
        or any(not isinstance(item, str) or not _ID_RE.fullmatch(item) for item in allowed)
    ):
        raise ContractError("allowed_event_classes must be a unique ID list")
    if default not in allowed:
        raise ContractError("default_event_class must be allowed")
    selectors = acoustic["selection_attributes"]
    if (
        not isinstance(selectors, list)
        or not selectors
        or len(selectors) != len(set(selectors))
        or any(item not in available_attributes for item in selectors)
    ):
        raise ContractError("selection_attributes must name available absolute attributes")
    return _deepcopy(dict(acoustic))


def _validate_qa_contract(
    value: Any, *, domains: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    qa = _require_exact_fields(
        value, frozenset({"subject_label", "attributes"}), "qa_contract"
    )
    _require_text(qa["subject_label"], "qa subject_label")
    attributes = _require_mapping(qa["attributes"], "qa attributes")
    if set(attributes) != set(domains):
        raise ContractError("qa_contract must cover every sampled attribute")
    for attribute, domain in domains.items():
        record = _require_mapping(attributes[attribute], f"qa attribute {attribute}")
        kind = record.get("kind")
        if kind == "ordered":
            _require_exact_fields(
                record,
                frozenset(
                    {
                        "kind",
                        "label",
                        "value_order",
                        "value_labels",
                        "comparison_question",
                    }
                ),
                f"qa attribute {attribute}",
            )
            if record["value_order"] != list(domain):
                raise ContractError(f"qa {attribute}.value_order must equal its domain")
            _require_text(record["comparison_question"], f"qa {attribute}.comparison_question")
        elif kind == "categorical":
            _require_exact_fields(
                record,
                frozenset(
                    {
                        "kind",
                        "label",
                        "value_labels",
                        "identification_question",
                    }
                ),
                f"qa attribute {attribute}",
            )
            question = _require_text(
                record["identification_question"],
                f"qa {attribute}.identification_question",
            )
            if _format_placeholders(question, f"qa {attribute}.identification_question") != [
                "instance_label"
            ]:
                raise ContractError(
                    f"qa {attribute}.identification_question must contain "
                    "{{instance_label}} exactly once"
                )
        else:
            raise ContractError(f"qa {attribute}.kind must be ordered or categorical")
        _require_text(record["label"], f"qa {attribute}.label")
        labels = _require_mapping(record["value_labels"], f"qa {attribute}.value_labels")
        if set(labels) != set(domain):
            raise ContractError(f"qa {attribute}.value_labels must equal its domain")
        for name, text in labels.items():
            _require_text(text, f"qa {attribute}.value_labels.{name}")
    return _deepcopy(dict(qa))


def validate_attribute_profile(value: Any) -> dict[str, Any]:
    """Validate and return a detached copy of one attribute profile."""

    profile = _require_exact_fields(value, _PROFILE_FIELDS, "attribute profile")
    if profile["schema"] != PROFILE_SCHEMA:
        raise ContractError(f"attribute profile schema must be {PROFILE_SCHEMA}")
    _require_id(profile["profile_schema_id"], "profile_schema_id")
    _require_id(profile["profile_revision"], "profile_revision")
    asset_class = profile["asset_class"]
    if asset_class not in ASSET_CLASSES:
        raise ContractError(f"asset_class must be one of {sorted(ASSET_CLASSES)}")
    _require_id(profile["lineage_group_id"], "lineage_group_id")
    state = profile["state_classification"]
    if state not in STATE_CLASSIFICATIONS:
        raise ContractError("invalid state_classification")
    if state == "rejected":
        raise ContractError("rejected profiles cannot generate new requests")

    taxonomy = _validate_attribute_values(profile["taxonomy"], "taxonomy")
    if not taxonomy:
        raise ContractError("taxonomy cannot be empty")
    fixed = _validate_attribute_values(profile["fixed_attributes"], "fixed_attributes")
    if not fixed:
        raise ContractError("fixed_attributes cannot be empty")

    raw_domains = _require_mapping(
        profile["sampled_attribute_domains"], "sampled_attribute_domains"
    )
    if not raw_domains:
        raise ContractError("sampled_attribute_domains cannot be empty")
    domains: dict[str, list[str]] = {}
    for raw_attribute, raw_values in raw_domains.items():
        attribute = _validate_attribute_key(raw_attribute, "sampled attribute")
        if (
            not isinstance(raw_values, list)
            or not (1 <= len(raw_values) <= 3)
            or len(raw_values) != len(set(raw_values))
            or any(not isinstance(item, str) or not _ID_RE.fullmatch(item) for item in raw_values)
        ):
            raise ContractError(
                f"sampled attribute {attribute} must have one to three unique ID values"
            )
        domains[attribute] = list(raw_values)

    sections = {"taxonomy": taxonomy, "fixed": fixed, "sampled": domains}
    section_names = list(sections)
    for index, left_name in enumerate(section_names):
        for right_name in section_names[index + 1 :]:
            overlap = set(sections[left_name]) & set(sections[right_name])
            if overlap:
                raise ContractError(
                    f"absolute attribute names overlap between {left_name} and "
                    f"{right_name}: {sorted(overlap)}"
                )

    base = _require_exact_fields(
        profile["base_template"],
        frozenset(
            {
                "template_id",
                "kind",
                "artifact",
                "provenance_status",
                "usage_scope",
            }
        ),
        "base_template",
    )
    _require_id(base["template_id"], "base_template.template_id")
    if base["kind"] not in {"reference_image", "rocketbox_avatar"}:
        raise ContractError("unsupported base_template.kind")
    _validate_artifact(base["artifact"], "base_template.artifact")
    if base["provenance_status"] not in {"verified", "review_required", "legacy_unknown"}:
        raise ContractError("unsupported base_template.provenance_status")
    if base["usage_scope"] not in STATE_CLASSIFICATIONS - {"rejected"}:
        raise ContractError("unsupported base_template.usage_scope")
    if state == "formal_dataset_asset" and (
        base["provenance_status"] != "verified"
        or base["usage_scope"] != "formal_dataset_asset"
    ):
        raise ContractError("formal profiles require a verified formal base template")

    forbidden = profile["forbidden_combinations"]
    if not isinstance(forbidden, list):
        raise ContractError("forbidden_combinations must be a list")
    seen_forbidden: set[str] = set()
    for index, combination in enumerate(forbidden):
        item = _require_mapping(combination, f"forbidden_combinations[{index}]")
        if not item:
            raise ContractError("forbidden combinations cannot be empty")
        for attribute, selected in item.items():
            if attribute not in domains or selected not in domains[attribute]:
                raise ContractError("forbidden combination uses an unknown domain value")
        digest = canonical_json(item)
        if digest in seen_forbidden:
            raise ContractError("forbidden combinations must be unique")
        seen_forbidden.add(digest)

    _validate_generation_contract(
        profile["generation_contract"],
        asset_class=asset_class,
        base_template=base,
        taxonomy=taxonomy,
        fixed=fixed,
        domains=domains,
    )
    _validate_target_physical_profiles(
        profile["target_physical_profiles"],
        asset_class=asset_class,
        domains=domains,
        state_classification=state,
    )
    _validate_rig_profile(profile["rig_profile"], asset_class=asset_class)
    available = set(taxonomy) | set(fixed) | set(domains)
    _validate_acoustic_profile(
        profile["acoustic_profile"], available_attributes=available
    )

    locked = profile["locked_attributes"]
    if (
        not isinstance(locked, list)
        or not locked
        or len(locked) != len(set(locked))
        or any(not isinstance(item, str) or not _ID_RE.fullmatch(item) for item in locked)
    ):
        raise ContractError("locked_attributes must be a unique non-empty ID list")
    if asset_class == "human":
        required_locks = {"eyes", "headwear", "eyewear", "accessories", "garment_geometry"}
        if not required_locks.issubset(locked):
            raise ContractError(
                "human profiles must lock eyes, headwear, eyewear, accessories, "
                "and garment_geometry"
            )
    _validate_qa_contract(profile["qa_contract"], domains=domains)
    return _deepcopy(dict(profile))


def profile_sha256(profile: Any) -> str:
    return _json_sha256(validate_attribute_profile(profile))


def _is_forbidden(sampled: Mapping[str, str], forbidden: Sequence[Mapping[str, str]]) -> bool:
    return any(all(sampled.get(key) == value for key, value in rule.items()) for rule in forbidden)


def legal_attribute_combinations(profile: Any) -> list[dict[str, str]]:
    validated = validate_attribute_profile(profile)
    domains = validated["sampled_attribute_domains"]
    names = sorted(domains)
    combinations = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(domains[name] for name in names))
    ]
    legal = [
        item
        for item in combinations
        if not _is_forbidden(item, validated["forbidden_combinations"])
    ]
    if not legal:
        raise ContractError("attribute profile has no legal combinations")
    return legal


def _seeded_rank(seed: int, profile_id: str, value: Any) -> int:
    payload = f"{seed}\0{profile_id}\0{canonical_json(value)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _derive_generation_seed(batch_seed: int, profile_id: str, ordinal: int) -> int:
    payload = f"generation\0{batch_seed}\0{profile_id}\0{ordinal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _quota_variants(
    domains: Mapping[str, Sequence[str]],
    count: int,
    *,
    batch_seed: int,
    profile_id: str,
) -> list[dict[str, dict[str, int]]]:
    """Enumerate exact floor/ceil quota assignments in deterministic order."""

    per_attribute: list[tuple[str, list[dict[str, int]]]] = []
    for attribute in sorted(domains):
        values = list(domains[attribute])
        base, remainder = divmod(count, len(values))
        high_sets = list(itertools.combinations(values, remainder))
        if not high_sets:
            high_sets = [tuple()]
        variants = [
            {
                value: base + (1 if value in high_values else 0)
                for value in values
            }
            for high_values in high_sets
        ]
        variants.sort(
            key=lambda item: _seeded_rank(
                batch_seed,
                profile_id,
                {"quota_attribute": attribute, "quota": item},
            )
        )
        per_attribute.append((attribute, variants))

    results: list[dict[str, dict[str, int]]] = []
    for selected in itertools.product(*(variants for _name, variants in per_attribute)):
        results.append(
            {
                attribute: dict(quota)
                for (attribute, _variants), quota in zip(
                    per_attribute, selected, strict=True
                )
            }
        )
    results.sort(
        key=lambda item: _seeded_rank(
            batch_seed, profile_id, {"quota_variant": item}
        )
    )
    return results


def _select_unique_balanced_subset(
    legal: Sequence[Mapping[str, str]],
    domains: Mapping[str, Sequence[str]],
    count: int,
    *,
    batch_seed: int,
    profile_id: str,
) -> list[dict[str, str]] | None:
    """Find a unique legal subset that satisfies exact per-value quotas."""

    if count == 0:
        return []
    candidate_indices = list(range(len(legal)))
    candidate_indices.sort(
        key=lambda index: _seeded_rank(
            batch_seed,
            profile_id,
            {"balanced_candidate": legal[index]},
        )
    )
    attributes = sorted(domains)

    for quota in _quota_variants(
        domains,
        count,
        batch_seed=batch_seed,
        profile_id=profile_id,
    ):
        remaining = {
            attribute: dict(value_counts)
            for attribute, value_counts in quota.items()
        }
        selected: list[int] = []
        used: set[int] = set()
        failed_states: set[tuple[Any, ...]] = set()

        def state_key() -> tuple[Any, ...]:
            quota_state = tuple(
                remaining[attribute][value]
                for attribute in attributes
                for value in domains[attribute]
            )
            return quota_state + (tuple(sorted(used)),)

        def enough_candidates_left(rows_left: int) -> bool:
            if len(legal) - len(used) < rows_left:
                return False
            for attribute in attributes:
                for value in domains[attribute]:
                    need = remaining[attribute][value]
                    if need <= 0:
                        continue
                    available = sum(
                        index not in used and legal[index][attribute] == value
                        for index in candidate_indices
                    )
                    if available < need:
                        return False
            return True

        def search(rows_left: int) -> bool:
            if rows_left == 0:
                return all(
                    count_left == 0
                    for attribute in attributes
                    for count_left in remaining[attribute].values()
                )
            key = state_key()
            if key in failed_states:
                return False
            if not enough_candidates_left(rows_left):
                failed_states.add(key)
                return False

            eligible = [
                index
                for index in candidate_indices
                if index not in used
                and all(
                    remaining[attribute][legal[index][attribute]] > 0
                    for attribute in attributes
                )
            ]
            eligible.sort(
                key=lambda index: _seeded_rank(
                    batch_seed,
                    profile_id,
                    {
                        "row": count - rows_left,
                        "candidate": legal[index],
                        "remaining": remaining,
                    },
                )
            )
            for index in eligible:
                candidate = legal[index]
                used.add(index)
                selected.append(index)
                for attribute in attributes:
                    remaining[attribute][candidate[attribute]] -= 1
                if search(rows_left - 1):
                    return True
                for attribute in attributes:
                    remaining[attribute][candidate[attribute]] += 1
                selected.pop()
                used.remove(index)
            failed_states.add(key)
            return False

        if search(count):
            return [dict(legal[index]) for index in selected]
    return None


def _balanced_combinations(profile: Mapping[str, Any], count: int, batch_seed: int) -> list[dict[str, str]]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ContractError("count must be a positive integer")
    if isinstance(batch_seed, bool) or not isinstance(batch_seed, int) or batch_seed < 0:
        raise ContractError("batch_seed must be a non-negative integer")
    legal = legal_attribute_combinations(profile)
    domains = profile["sampled_attribute_domains"]
    profile_id = str(profile["profile_schema_id"])
    complete_cycles, remainder = divmod(count, len(legal))
    ordered_legal = sorted(
        (dict(item) for item in legal),
        key=lambda item: _seeded_rank(
            batch_seed, profile_id, {"complete_cycle": item}
        ),
    )
    selected = [dict(item) for _cycle in range(complete_cycles) for item in ordered_legal]
    if remainder:
        subset = _select_unique_balanced_subset(
            legal,
            domains,
            remainder,
            batch_seed=batch_seed + complete_cycles,
            profile_id=profile_id,
        )
        if subset is None:
            # A forbidden-combination profile can make exact floor/ceil quotas
            # mathematically impossible.  Preserve determinism and uniqueness,
            # but let the batch distribution expose the residual imbalance.
            subset = ordered_legal[:remainder]
        selected.extend(subset)
    return selected


def _compile_animal_generation_plan(
    profile: Mapping[str, Any], sampled: Mapping[str, str], generation_seed: int
) -> dict[str, Any]:
    contract = profile["generation_contract"]
    combined = _combined_attribute_values(profile, sampled)
    labels: dict[str, str] = {}
    for attribute, value in combined.items():
        labels[attribute] = contract["value_labels"][attribute][str(value)]
    subject = contract["positive_template"].format(**labels).strip()
    prompt = f"{subject} {contract['pose_guard_prompt'].strip()}"
    return {
        "schema": "flux2_pixal3d_generation_plan_v1",
        "route": contract["route"],
        "prompt_template_id": contract["prompt_template_id"],
        "base_template": _deepcopy(profile["base_template"]),
        "prompt": prompt,
        "negative_prompt": contract["negative_prompt"],
        "generation_seed": generation_seed,
        "flux_invocations": 1,
        "model_revisions": _deepcopy(contract["model_revisions"]),
    }


def _srgb_hex(values: Sequence[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*values)


def _compile_human_material_plan(
    profile: Mapping[str, Any], sampled: Mapping[str, str], generation_seed: int
) -> dict[str, Any]:
    contract = profile["generation_contract"]
    edits: list[dict[str, Any]] = []
    for attribute in sorted(sampled):
        value = sampled[attribute]
        binding = contract["material_bindings"][attribute]
        srgb = list(binding["palette"][value]["srgb_u8"])
        edits.append(
            {
                "attribute": attribute,
                "value": value,
                "semantic_mask": binding["semantic_mask"],
                "mask_registry": _deepcopy(binding["mask_registry"]),
                "source_texture_role": binding["source_texture_role"],
                "operation": binding["operation"],
                "target_srgb_u8": srgb,
                "target_srgb_hex": _srgb_hex(srgb),
            }
        )
    return {
        "schema": contract["material_plan_schema"],
        "route": contract["route"],
        "base_avatar_id": profile["base_template"]["template_id"],
        "base_template": _deepcopy(profile["base_template"]),
        "geometry_changes_allowed": False,
        "edits": edits,
        "flux_texture_detail": _deepcopy(contract["flux_texture_detail"]),
        "generation_seed": generation_seed,
    }


def _select_target_physical_profile(
    profile: Mapping[str, Any], sampled: Mapping[str, str]
) -> dict[str, Any]:
    contract = profile["target_physical_profiles"]
    if contract["control_attribute"] is None:
        selected_value = "fixed"
    else:
        selected_value = sampled[contract["control_attribute"]]
    selected = contract["values"][selected_value]
    result = {
        "profile_id": contract["profile_id"],
        "control_attribute": contract["control_attribute"],
        "selected_value": selected_value,
        "measurement": contract["measurement"],
        "mode": contract["mode"],
        "reference_value_cm": contract["reference_value_cm"],
        "reference_provenance": _deepcopy(contract["reference_provenance"]),
    }
    result.update(_deepcopy(selected))
    if "scale_ratio" in selected and contract["reference_value_cm"] is not None:
        result["target_value_cm"] = round(
            float(contract["reference_value_cm"]) * float(selected["scale_ratio"]), 6
        )
    return result


def _request_core(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _deepcopy(value)
        for key, value in request.items()
        if key not in {"instance_id", "request_sha256"}
    }


def _instance_id_prefix(profile_id: str) -> str:
    prefix = re.sub(r"_v[0-9]+$", "", profile_id)
    return prefix[:80].rstrip("_.-")


def build_instance_request(
    profile: Any,
    sampled_attributes: Mapping[str, str],
    *,
    batch_seed: int,
    sample_ordinal: int,
) -> dict[str, Any]:
    validated = validate_attribute_profile(profile)
    domains = validated["sampled_attribute_domains"]
    if set(sampled_attributes) != set(domains):
        raise ContractError("sampled_attributes must contain every sampled domain exactly")
    sampled = {
        attribute: _require_id(value, f"sampled_attributes.{attribute}")
        for attribute, value in sampled_attributes.items()
    }
    for attribute, value in sampled.items():
        if value not in domains[attribute]:
            raise ContractError(f"sampled value {attribute}={value!r} is outside its domain")
    if _is_forbidden(sampled, validated["forbidden_combinations"]):
        raise ContractError("sampled_attributes match a forbidden combination")
    if isinstance(batch_seed, bool) or not isinstance(batch_seed, int) or batch_seed < 0:
        raise ContractError("batch_seed must be a non-negative integer")
    if isinstance(sample_ordinal, bool) or not isinstance(sample_ordinal, int) or sample_ordinal < 0:
        raise ContractError("sample_ordinal must be a non-negative integer")
    generation_seed = _derive_generation_seed(
        batch_seed, validated["profile_schema_id"], sample_ordinal
    )
    if validated["asset_class"] == "animal":
        plan = _compile_animal_generation_plan(validated, sampled, generation_seed)
    else:
        plan = _compile_human_material_plan(validated, sampled, generation_seed)

    request: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "profile_schema_id": validated["profile_schema_id"],
        "profile_sha256": profile_sha256(validated),
        "asset_class": validated["asset_class"],
        "lineage_group_id": validated["lineage_group_id"],
        "state_classification": validated["state_classification"],
        "sampler": {
            "algorithm": SAMPLER_ALGORITHM,
            "batch_seed": batch_seed,
            "sample_ordinal": sample_ordinal,
            "generation_seed": generation_seed,
        },
        "taxonomy": _deepcopy(validated["taxonomy"]),
        "fixed_attributes": _deepcopy(validated["fixed_attributes"]),
        "sampled_attributes": dict(sorted(sampled.items())),
        "target_physical_profile": _select_target_physical_profile(validated, sampled),
        "generation_plan": plan,
        "rig_profile": _deepcopy(validated["rig_profile"]),
        "acoustic_profile": _deepcopy(validated["acoustic_profile"]),
        "locked_attributes": _deepcopy(validated["locked_attributes"]),
    }
    request_digest = _json_sha256(request)
    request["instance_id"] = (
        f"{_instance_id_prefix(validated['profile_schema_id'])}_{request_digest[:12]}"
    )
    request["request_sha256"] = request_digest
    ordered = {key: request[key] for key in sorted(_REQUEST_FIELDS)}
    return ordered


def validate_instance_request(value: Any, profile: Any) -> dict[str, Any]:
    request = _require_exact_fields(value, _REQUEST_FIELDS, "instance request")
    if request["schema"] != REQUEST_SCHEMA:
        raise ContractError(f"instance request schema must be {REQUEST_SCHEMA}")
    validated_profile = validate_attribute_profile(profile)
    sampler = _require_exact_fields(
        request["sampler"],
        frozenset({"algorithm", "batch_seed", "sample_ordinal", "generation_seed"}),
        "request sampler",
    )
    if sampler["algorithm"] != SAMPLER_ALGORITHM:
        raise ContractError("unsupported request sampler algorithm")
    rebuilt = build_instance_request(
        validated_profile,
        _require_mapping(request["sampled_attributes"], "sampled_attributes"),
        batch_seed=sampler["batch_seed"],
        sample_ordinal=sampler["sample_ordinal"],
    )
    if canonical_json(request) != canonical_json(rebuilt):
        raise ContractError("instance request differs from the canonical request")
    return _deepcopy(dict(request))


def validate_request_integrity(value: Any) -> dict[str, Any]:
    """Verify request self-integrity when the profile snapshot is unavailable."""

    request = _require_exact_fields(value, _REQUEST_FIELDS, "instance request")
    if request["schema"] != REQUEST_SCHEMA:
        raise ContractError(f"instance request schema must be {REQUEST_SCHEMA}")
    _require_id(request["instance_id"], "instance_id")
    digest = _require_sha256(request["request_sha256"], "request_sha256")
    if _json_sha256(_request_core(request)) != digest:
        raise ContractError("request_sha256 does not authenticate request content")
    if not request["instance_id"].endswith(digest[:12]):
        raise ContractError("instance_id does not match request_sha256")
    return _deepcopy(dict(request))


def sample_instance_requests(
    profile: Any,
    *,
    count: int,
    batch_seed: int,
    start_ordinal: int = 0,
) -> list[dict[str, Any]]:
    validated = validate_attribute_profile(profile)
    if isinstance(start_ordinal, bool) or not isinstance(start_ordinal, int) or start_ordinal < 0:
        raise ContractError("start_ordinal must be a non-negative integer")
    combinations = _balanced_combinations(validated, count, batch_seed)
    return [
        build_instance_request(
            validated,
            combination,
            batch_seed=batch_seed,
            sample_ordinal=start_ordinal + index,
        )
        for index, combination in enumerate(combinations)
    ]


def build_request_batch(
    profiles: Sequence[Any], *, count_per_profile: int, batch_seed: int
) -> dict[str, Any]:
    validated_profiles = [validate_attribute_profile(profile) for profile in profiles]
    ids = [profile["profile_schema_id"] for profile in validated_profiles]
    if len(ids) != len(set(ids)):
        raise ContractError("profile_schema_id must be unique within a batch")
    requests: list[dict[str, Any]] = []
    for profile_index, profile in enumerate(sorted(validated_profiles, key=lambda item: item["profile_schema_id"])):
        profile_seed = _seeded_rank(batch_seed, profile["profile_schema_id"], profile_index) & ((1 << 63) - 1)
        requests.extend(
            sample_instance_requests(
                profile,
                count=count_per_profile,
                batch_seed=profile_seed,
            )
        )
    distribution: dict[str, Any] = {}
    for profile in validated_profiles:
        profile_requests = [
            item for item in requests if item["profile_schema_id"] == profile["profile_schema_id"]
        ]
        distribution[profile["profile_schema_id"]] = {
            attribute: {
                value: sum(req["sampled_attributes"][attribute] == value for req in profile_requests)
                for value in values
            }
            for attribute, values in profile["sampled_attribute_domains"].items()
        }
    core = {
        "schema": REQUEST_BATCH_SCHEMA,
        "sampler": {
            "algorithm": SAMPLER_ALGORITHM,
            "batch_seed": batch_seed,
            "count_per_profile": count_per_profile,
        },
        "profiles": [
            {
                "profile_schema_id": profile["profile_schema_id"],
                "profile_sha256": profile_sha256(profile),
            }
            for profile in sorted(validated_profiles, key=lambda item: item["profile_schema_id"])
        ],
        "requests": sorted(requests, key=lambda item: item["instance_id"]),
        "distribution": distribution,
    }
    core["batch_sha256"] = _json_sha256(core)
    core["batch_id"] = f"controlled_request_batch_{core['batch_sha256'][:12]}"
    return core


def validate_request_batch(value: Any, profiles: Sequence[Any]) -> dict[str, Any]:
    """Validate a frozen request batch against its exact profile revisions.

    Rebuilding the batch is intentional: it authenticates the sampler inputs,
    every request, the recorded distribution, and both batch identifiers in a
    single canonical comparison.
    """

    batch = _require_exact_fields(
        value,
        frozenset(
            {
                "schema",
                "sampler",
                "profiles",
                "requests",
                "distribution",
                "batch_sha256",
                "batch_id",
            }
        ),
        "instance request batch",
    )
    if batch["schema"] != REQUEST_BATCH_SCHEMA:
        raise ContractError(
            f"instance request batch schema must be {REQUEST_BATCH_SCHEMA}"
        )
    sampler = _require_exact_fields(
        batch["sampler"],
        frozenset({"algorithm", "batch_seed", "count_per_profile"}),
        "instance request batch sampler",
    )
    if sampler["algorithm"] != SAMPLER_ALGORITHM:
        raise ContractError("unsupported request batch sampler algorithm")
    if (
        isinstance(sampler["count_per_profile"], bool)
        or not isinstance(sampler["count_per_profile"], int)
        or sampler["count_per_profile"] <= 0
    ):
        raise ContractError("request batch count_per_profile must be positive")
    if (
        isinstance(sampler["batch_seed"], bool)
        or not isinstance(sampler["batch_seed"], int)
        or sampler["batch_seed"] < 0
    ):
        raise ContractError("request batch batch_seed must be non-negative")
    rebuilt = build_request_batch(
        profiles,
        count_per_profile=sampler["count_per_profile"],
        batch_seed=sampler["batch_seed"],
    )
    if canonical_json(batch) != canonical_json(rebuilt):
        raise ContractError(
            "instance request batch does not match deterministic profiles and sampling"
        )
    return _deepcopy(dict(batch))


def _validate_physical_measurements(value: Any, *, formal: bool) -> dict[str, Any]:
    measurements = _require_mapping(value, "physical_measurements")
    status = measurements.get("status")
    if status == "pending":
        if set(measurements) != {"status"}:
            raise ContractError("pending physical_measurements can only contain status")
        if formal:
            raise ContractError("formal assets require measured physical measurements")
        return _deepcopy(dict(measurements))
    if status != "measured" or set(measurements) != {"status", "method", "runtime"}:
        raise ContractError("physical_measurements must be pending or measured")
    _require_id(measurements["method"], "physical measurement method")
    runtime = _require_mapping(measurements["runtime"], "physical runtime measurements")
    if "actor_scale" not in runtime:
        raise ContractError("physical runtime measurements require actor_scale")
    _require_finite_number(runtime["actor_scale"], "actor_scale", positive=True)
    centimeter_fields = [key for key in runtime if key.endswith("_cm")]
    if not centimeter_fields:
        raise ContractError("physical runtime measurements require at least one *_cm value")
    for key, number in runtime.items():
        if key == "actor_scale":
            continue
        if not key.endswith("_cm"):
            raise ContractError(f"unexpected physical runtime field: {key}")
        _require_finite_number(number, f"physical runtime {key}", positive=True)
    return _deepcopy(dict(measurements))


def _validate_rights(value: Any, *, formal: bool) -> dict[str, Any]:
    rights = _require_exact_fields(
        value, frozenset({"status", "licenses", "blockers"}), "rights"
    )
    if rights["status"] not in {"cleared", "review_required", "blocked"}:
        raise ContractError("invalid rights status")
    licenses = rights["licenses"]
    if not isinstance(licenses, list) or not licenses:
        raise ContractError("rights.licenses must contain at least one snapshot")
    for index, license_artifact in enumerate(licenses):
        _validate_artifact(license_artifact, f"rights.licenses[{index}]")
    blockers = rights["blockers"]
    if (
        not isinstance(blockers, list)
        or len(blockers) != len(set(blockers))
        or any(not isinstance(item, str) or not item for item in blockers)
    ):
        raise ContractError("rights.blockers must be a unique string list")
    if formal and (rights["status"] != "cleared" or blockers):
        raise ContractError("formal assets require cleared rights with no blockers")
    return _deepcopy(dict(rights))


def _validate_qa(value: Any, *, formal: bool, rejected: bool) -> dict[str, str]:
    qa = _require_mapping(value, "qa")
    if set(qa) != set(QA_FIELDS):
        raise ContractError(f"qa fields must be exactly {list(QA_FIELDS)}")
    for key, status in qa.items():
        if status not in QA_STATUSES:
            raise ContractError(f"invalid QA status for {key}: {status!r}")
    if formal and any(status not in {"passed", "not_applicable"} for status in qa.values()):
        raise ContractError("formal assets require every QA gate to pass or be not_applicable")
    if rejected and "rejected" not in qa.values():
        raise ContractError("rejected assets require at least one rejected QA gate")
    return dict(qa)


def _formal_artifact_path(path: str, label: str) -> None:
    pure = PurePosixPath(path)
    if pure.is_absolute() or "tmp" in pure.parts or ".." in pure.parts:
        raise ContractError(f"formal {label} must be an immutable relative non-tmp path")


def build_source_asset_v2(
    request: Any,
    *,
    artifacts: Mapping[str, Any],
    physical_measurements: Mapping[str, Any],
    provenance: Mapping[str, Any],
    rights: Mapping[str, Any],
    qa: Mapping[str, str],
    state_classification: str,
) -> dict[str, Any]:
    request_value = validate_request_integrity(request)
    semantic = _combined_attribute_values(
        {
            "taxonomy": request_value["taxonomy"],
            "fixed_attributes": request_value["fixed_attributes"],
        },
        request_value["sampled_attributes"],
    )
    asset = {
        "schema": SOURCE_ASSET_SCHEMA,
        "asset_id": request_value["instance_id"],
        "profile_schema_id": request_value["profile_schema_id"],
        "profile_sha256": request_value["profile_sha256"],
        "request_sha256": request_value["request_sha256"],
        "asset_class": request_value["asset_class"],
        "lineage_group_id": request_value["lineage_group_id"],
        "taxonomy": _deepcopy(request_value["taxonomy"]),
        "fixed_attributes": _deepcopy(request_value["fixed_attributes"]),
        "sampled_attributes": _deepcopy(request_value["sampled_attributes"]),
        "semantic_attributes": semantic,
        "target_physical_profile": _deepcopy(request_value["target_physical_profile"]),
        "artifacts": _deepcopy(dict(artifacts)),
        "physical_measurements": _deepcopy(dict(physical_measurements)),
        "rig": _deepcopy(request_value["rig_profile"]),
        "acoustic_profile": _deepcopy(request_value["acoustic_profile"]),
        "provenance": _deepcopy(dict(provenance)),
        "rights": _deepcopy(dict(rights)),
        "qa": _deepcopy(dict(qa)),
        "state_classification": state_classification,
    }
    return validate_source_asset_v2(asset, request=request_value)


def validate_source_asset_v2(
    value: Any,
    *,
    request: Any | None = None,
    profile: Any | None = None,
) -> dict[str, Any]:
    asset = _require_exact_fields(value, _SOURCE_ASSET_FIELDS, "source_asset_v2")
    if asset["schema"] != SOURCE_ASSET_SCHEMA:
        raise ContractError(f"source asset schema must be {SOURCE_ASSET_SCHEMA}")
    _require_id(asset["asset_id"], "asset_id")
    _require_id(asset["profile_schema_id"], "profile_schema_id")
    _require_sha256(asset["profile_sha256"], "profile_sha256")
    _require_sha256(asset["request_sha256"], "request_sha256")
    if asset["asset_class"] not in ASSET_CLASSES:
        raise ContractError("invalid asset_class")
    _require_id(asset["lineage_group_id"], "lineage_group_id")
    state = asset["state_classification"]
    if state not in STATE_CLASSIFICATIONS:
        raise ContractError("invalid source asset state_classification")
    formal = state == "formal_dataset_asset"
    rejected = state == "rejected"

    taxonomy = _validate_attribute_values(asset["taxonomy"], "taxonomy")
    fixed = _validate_attribute_values(asset["fixed_attributes"], "fixed_attributes")
    sampled = _validate_attribute_values(asset["sampled_attributes"], "sampled_attributes")
    expected_semantic = _combined_attribute_values(
        {"taxonomy": taxonomy, "fixed_attributes": fixed}, sampled
    )
    if canonical_json(asset["semantic_attributes"]) != canonical_json(expected_semantic):
        raise ContractError("semantic_attributes must be the exact absolute attribute union")

    artifacts = _require_mapping(asset["artifacts"], "artifacts")
    if not artifacts:
        raise ContractError("source asset artifacts cannot be empty")
    for role, artifact in artifacts.items():
        _require_id(role, "artifact role")
        validated_artifact = _validate_artifact(artifact, f"artifacts.{role}")
        if formal:
            _formal_artifact_path(validated_artifact["path"], f"artifact {role}")
    _validate_physical_measurements(asset["physical_measurements"], formal=formal)
    _validate_rig_profile(asset["rig"], asset_class=asset["asset_class"])
    available = set(taxonomy) | set(fixed) | set(sampled)
    _validate_acoustic_profile(
        asset["acoustic_profile"], available_attributes=available
    )

    provenance = _require_exact_fields(
        asset["provenance"],
        frozenset({"attempt_id", "request_sha256", "models"}),
        "provenance",
    )
    _require_id(provenance["attempt_id"], "provenance.attempt_id")
    if provenance["request_sha256"] != asset["request_sha256"]:
        raise ContractError("provenance request hash does not match source asset")
    models = _require_mapping(provenance["models"], "provenance.models")
    for model, revision in models.items():
        _require_id(model, "provenance model name")
        _require_text(revision, f"provenance model {model}")
    _validate_rights(asset["rights"], formal=formal)
    _validate_qa(asset["qa"], formal=formal, rejected=rejected)

    if request is not None:
        request_value = validate_request_integrity(request)
        expected = {
            "asset_id": request_value["instance_id"],
            "profile_schema_id": request_value["profile_schema_id"],
            "profile_sha256": request_value["profile_sha256"],
            "request_sha256": request_value["request_sha256"],
            "asset_class": request_value["asset_class"],
            "lineage_group_id": request_value["lineage_group_id"],
            "taxonomy": request_value["taxonomy"],
            "fixed_attributes": request_value["fixed_attributes"],
            "sampled_attributes": request_value["sampled_attributes"],
            "target_physical_profile": request_value["target_physical_profile"],
            "rig": request_value["rig_profile"],
            "acoustic_profile": request_value["acoustic_profile"],
        }
        for key, expected_value in expected.items():
            if canonical_json(asset[key]) != canonical_json(expected_value):
                raise ContractError(f"source asset {key} does not match its request")
        expected_models = request_value["generation_plan"].get("model_revisions", {})
        if canonical_json(models) != canonical_json(expected_models):
            raise ContractError("source asset model revisions do not match its request")

    if profile is not None:
        profile_value = validate_attribute_profile(profile)
        if asset["profile_schema_id"] != profile_value["profile_schema_id"]:
            raise ContractError("source asset profile_schema_id mismatch")
        if asset["profile_sha256"] != profile_sha256(profile_value):
            raise ContractError("source asset profile_sha256 mismatch")
        if asset["lineage_group_id"] != profile_value["lineage_group_id"]:
            raise ContractError("source asset lineage_group_id mismatch")
        domains = profile_value["sampled_attribute_domains"]
        if set(sampled) != set(domains) or any(sampled[key] not in domains[key] for key in domains):
            raise ContractError("source asset sampled attributes violate its profile")
    return _deepcopy(dict(asset))


def _instance_view(value: Mapping[str, Any]) -> dict[str, Any]:
    schema_name = value.get("schema")
    if schema_name == REQUEST_SCHEMA:
        request = validate_request_integrity(value)
        return {
            "instance_id": request["instance_id"],
            "profile_schema_id": request["profile_schema_id"],
            "taxonomy": request["taxonomy"],
            "fixed_attributes": request["fixed_attributes"],
            "sampled_attributes": request["sampled_attributes"],
            "target_physical_profile": request["target_physical_profile"],
            "physical_measurements": None,
            "evidence_state": "planned",
        }
    if schema_name == SOURCE_ASSET_SCHEMA:
        asset = validate_source_asset_v2(value)
        return {
            "instance_id": asset["asset_id"],
            "profile_schema_id": asset["profile_schema_id"],
            "taxonomy": asset["taxonomy"],
            "fixed_attributes": asset["fixed_attributes"],
            "sampled_attributes": asset["sampled_attributes"],
            "target_physical_profile": asset["target_physical_profile"],
            "physical_measurements": asset["physical_measurements"],
            "evidence_state": "realized",
        }
    raise ContractError("QA input must be an instance request or source_asset_v2")


def _question_id(pair_key: str, index: int, question: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{pair_key}\0{index}\0{canonical_json(question)}".encode("utf-8")
    ).hexdigest()
    return f"qa_{digest[:16]}"


def _realized_ordered_measurement_supports_difference(
    *,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    attribute: str,
    qa_record: Mapping[str, Any],
) -> bool:
    """Require observed metric order to agree with a realized size label.

    Planned requests intentionally have no metric evidence yet.  For realized
    assets, however, a categorical ordered answer must not claim that A is
    larger when the generated/UE-measured artifact is actually smaller.
    """

    if left["evidence_state"] != "realized":
        return True
    physical = left.get("target_physical_profile")
    if not isinstance(physical, Mapping) or physical.get("control_attribute") != attribute:
        return True
    if qa_record.get("kind") != "ordered":
        return True
    if right.get("target_physical_profile", {}).get("control_attribute") != attribute:
        return False
    measurement_name = physical.get("measurement")
    if right["target_physical_profile"].get("measurement") != measurement_name:
        return False
    try:
        left_runtime = left["physical_measurements"]["runtime"]
        right_runtime = right["physical_measurements"]["runtime"]
        left_number = float(left_runtime[measurement_name])
        right_number = float(right_runtime[measurement_name])
        order = qa_record["value_order"]
        left_index = order.index(left["sampled_attributes"][attribute])
        right_index = order.index(right["sampled_attributes"][attribute])
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    if not math.isfinite(left_number) or not math.isfinite(right_number):
        return False
    # Half a centimetre keeps near-equal/noisy measurements out of a visual QA
    # comparison while remaining far below the configured category spacing.
    margin_cm = 0.5
    return (
        left_number + margin_cm < right_number
        if left_index < right_index
        else right_number + margin_cm < left_number
    )


def _questions_for_difference(
    *,
    pair_key: str,
    attribute: str,
    value_a: str,
    value_b: str,
    instance_a: str,
    instance_b: str,
    qa_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    if qa_record["kind"] == "ordered":
        order = qa_record["value_order"]
        if order.index(value_a) > order.index(value_b):
            answer_id, answer_value = instance_a, value_a
        else:
            answer_id, answer_value = instance_b, value_b
        questions.append(
            {
                "attribute": attribute,
                "kind": "comparison",
                "text": qa_record["comparison_question"],
                "answer": {
                    "instance_id": answer_id,
                    "value": answer_value,
                    "label": qa_record["value_labels"][answer_value],
                },
            }
        )
    else:
        for label, instance_id, selected in (
            ("instance A", instance_a, value_a),
            ("instance B", instance_b, value_b),
        ):
            questions.append(
                {
                    "attribute": attribute,
                    "kind": "identification",
                    "text": qa_record["identification_question"].format(
                        instance_label=label
                    ),
                    "answer": {
                        "instance_id": instance_id,
                        "value": selected,
                        "label": qa_record["value_labels"][selected],
                    },
                }
            )
    for index, question in enumerate(questions):
        question["question_id"] = _question_id(pair_key, index, question)
    return questions


def build_instance_qa_pairs(
    instances: Sequence[Mapping[str, Any]],
    profiles_by_id: Mapping[str, Any],
    *,
    single_attribute_only: bool = False,
    max_pairs: int | None = None,
) -> list[dict[str, Any]]:
    if max_pairs is not None and (
        isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs <= 0
    ):
        raise ContractError("max_pairs must be a positive integer")
    profiles = {
        profile_id: validate_attribute_profile(profile)
        for profile_id, profile in profiles_by_id.items()
    }
    if any(profile_id != profile["profile_schema_id"] for profile_id, profile in profiles.items()):
        raise ContractError("profiles_by_id keys must equal profile_schema_id")
    views = [_instance_view(instance) for instance in instances]
    ids = [view["instance_id"] for view in views]
    if len(ids) != len(set(ids)):
        raise ContractError("QA inputs contain duplicate instance IDs")

    candidates: list[tuple[int, str, str, dict[str, Any], dict[str, Any]]] = []
    for left, right in itertools.combinations(sorted(views, key=lambda item: item["instance_id"]), 2):
        if left["profile_schema_id"] != right["profile_schema_id"]:
            continue
        if left["evidence_state"] != right["evidence_state"]:
            raise ContractError("QA pairs cannot mix planned requests and realized assets")
        profile_id = left["profile_schema_id"]
        if profile_id not in profiles:
            raise ContractError(f"missing QA profile: {profile_id}")
        profile = profiles[profile_id]
        differences = {
            attribute: {
                "a": left["sampled_attributes"][attribute],
                "b": right["sampled_attributes"][attribute],
            }
            for attribute in sorted(left["sampled_attributes"])
            if left["sampled_attributes"][attribute]
            != right["sampled_attributes"][attribute]
            and _realized_ordered_measurement_supports_difference(
                left=left,
                right=right,
                attribute=attribute,
                qa_record=profile["qa_contract"]["attributes"][attribute],
            )
        }
        if not differences or (single_attribute_only and len(differences) != 1):
            continue
        candidates.append(
            (
                len(differences),
                left["instance_id"],
                right["instance_id"],
                left,
                right,
            )
        )

    results: list[dict[str, Any]] = []
    for _difference_count, _left_id, _right_id, left, right in sorted(candidates):
        profile_id = left["profile_schema_id"]
        if profile_id not in profiles:
            raise ContractError(f"missing QA profile: {profile_id}")
        profile = profiles[profile_id]
        if left["sampled_attributes"].keys() != right["sampled_attributes"].keys():
            raise ContractError("QA pair sampled attribute sets differ")
        differences = {
            attribute: {
                "a": left["sampled_attributes"][attribute],
                "b": right["sampled_attributes"][attribute],
            }
            for attribute in sorted(left["sampled_attributes"])
            if left["sampled_attributes"][attribute]
            != right["sampled_attributes"][attribute]
            and _realized_ordered_measurement_supports_difference(
                left=left,
                right=right,
                attribute=attribute,
                qa_record=profile["qa_contract"]["attributes"][attribute],
            )
        }
        combined_a = {
            **left["taxonomy"],
            **left["fixed_attributes"],
            **left["sampled_attributes"],
        }
        combined_b = {
            **right["taxonomy"],
            **right["fixed_attributes"],
            **right["sampled_attributes"],
        }
        same = {
            attribute: combined_a[attribute]
            for attribute in sorted(combined_a)
            if attribute in combined_b and combined_a[attribute] == combined_b[attribute]
        }
        pair_key = canonical_json(
            {
                "profile_schema_id": profile_id,
                "instance_a": left["instance_id"],
                "instance_b": right["instance_id"],
                "differences": differences,
            }
        )
        pair_digest = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()
        questions: list[dict[str, Any]] = []
        for attribute, values in differences.items():
            questions.extend(
                _questions_for_difference(
                    pair_key=pair_key,
                    attribute=attribute,
                    value_a=values["a"],
                    value_b=values["b"],
                    instance_a=left["instance_id"],
                    instance_b=right["instance_id"],
                    qa_record=profile["qa_contract"]["attributes"][attribute],
                )
            )
        result = {
            "schema": PAIR_SCHEMA,
            "pair_id": f"pair_{pair_digest[:16]}",
            "profile_schema_id": profile_id,
            "evidence_state": left["evidence_state"],
            "instance_a": left["instance_id"],
            "instance_b": right["instance_id"],
            "same_attributes": same,
            "different_attributes": differences,
            "questions": questions,
        }
        results.append(result)
        if max_pairs is not None and len(results) >= max_pairs:
            break
    return results


def _validate_split_ratios(value: Mapping[str, float]) -> dict[str, float]:
    if set(value) != {"train", "validation", "test"}:
        raise ContractError("split ratios must contain train, validation, and test")
    ratios = {
        key: _require_finite_number(number, f"split ratio {key}")
        for key, number in value.items()
    }
    if any(number < 0.0 for number in ratios.values()) or not math.isclose(
        sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ContractError("split ratios must be non-negative and sum to 1")
    if ratios["train"] <= 0.0:
        raise ContractError("train split ratio must be positive")
    return ratios


def split_for_lineage(
    lineage_group_id: str,
    *,
    split_salt: str,
    split_ratios: Mapping[str, float],
) -> str:
    _require_id(lineage_group_id, "lineage_group_id")
    _require_text(split_salt, "split_salt")
    ratios = _validate_split_ratios(split_ratios)
    digest = hashlib.sha256(f"{split_salt}\0{lineage_group_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if unit < ratios["train"]:
        return "train"
    if unit < ratios["train"] + ratios["validation"]:
        return "validation"
    return "test"


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {key: _deepcopy(value) for key, value in manifest.items() if key != "manifest_sha256"}
    return _json_sha256(payload)


def _distribution_stats(asset_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split = {name: 0 for name in ("train", "validation", "test")}
    by_class: dict[str, int] = {}
    by_profile: dict[str, int] = {}
    attributes: dict[str, dict[str, int]] = {}
    for row in asset_rows:
        by_split[row["split"]] += 1
        by_class[row["asset_class"]] = by_class.get(row["asset_class"], 0) + 1
        profile_id = row["profile_schema_id"]
        by_profile[profile_id] = by_profile.get(profile_id, 0) + 1
        for attribute, selected in row["sampled_attributes"].items():
            key = f"{profile_id}:{attribute}"
            values = attributes.setdefault(key, {})
            values[selected] = values.get(selected, 0) + 1
    return {
        "asset_count": len(asset_rows),
        "by_split": by_split,
        "by_asset_class": dict(sorted(by_class.items())),
        "by_profile": dict(sorted(by_profile.items())),
        "sampled_attribute_counts": dict(sorted(attributes.items())),
    }


def _qa_evidence_attributes(asset: Mapping[str, Any]) -> list[str]:
    """Return sampled attributes supported by the asset's current evidence."""

    reference_ready = asset["qa"]["reference_2d"] in {
        "passed",
        "not_applicable",
    }
    if not reference_ready or asset["qa"]["static_mesh"] != "passed":
        return []
    eligible = set(asset["sampled_attributes"])
    physical = asset["target_physical_profile"]
    control_attribute = physical.get("control_attribute")
    if asset["asset_class"] == "animal" and control_attribute in eligible:
        measurements = asset["physical_measurements"]
        measurement_name = physical.get("measurement")
        if (
            measurements.get("status") != "measured"
            or not isinstance(measurements.get("runtime"), Mapping)
            or measurement_name not in measurements["runtime"]
        ):
            eligible.remove(control_attribute)
    return sorted(eligible)


def build_generation_plan_manifest(
    requests: Sequence[Mapping[str, Any]],
    profiles_by_id: Mapping[str, Any],
    *,
    plan_id: str,
    split_salt: str,
    split_ratios: Mapping[str, float] | None = None,
    single_attribute_qa_only: bool = True,
    max_qa_pairs_per_split: int | None = None,
) -> dict[str, Any]:
    """Build the pre-generation dataset/QA plan from immutable requests.

    This manifest is deliberately marked ``planned``.  It can schedule image,
    material, 3D, and review work, but it cannot be used as evidence that an
    asset exists or that a QA answer is visually true.  Final dataset manifests
    are built separately from validated ``source_asset_v2`` records.
    """

    _require_id(plan_id, "plan_id")
    _require_text(split_salt, "split_salt")
    ratios = _validate_split_ratios(
        split_ratios or {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    profiles = {
        profile_id: validate_attribute_profile(profile)
        for profile_id, profile in profiles_by_id.items()
    }
    if any(profile_id != profile["profile_schema_id"] for profile_id, profile in profiles.items()):
        raise ContractError("profiles_by_id keys must equal profile_schema_id")

    validated_requests: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_request in requests:
        profile_id = raw_request.get("profile_schema_id")
        if profile_id not in profiles:
            raise ContractError(f"missing profile for request: {profile_id!r}")
        request = validate_instance_request(raw_request, profiles[profile_id])
        if request["instance_id"] in seen_ids:
            raise ContractError(f"duplicate generation request: {request['instance_id']}")
        seen_ids.add(request["instance_id"])
        validated_requests.append(request)

    request_rows: list[dict[str, Any]] = []
    for request in sorted(validated_requests, key=lambda item: item["instance_id"]):
        split = split_for_lineage(
            request["lineage_group_id"],
            split_salt=split_salt,
            split_ratios=ratios,
        )
        request_rows.append(
            {
                "instance_id": request["instance_id"],
                "request_sha256": request["request_sha256"],
                "profile_schema_id": request["profile_schema_id"],
                "lineage_group_id": request["lineage_group_id"],
                "asset_class": request["asset_class"],
                "route": request["generation_plan"]["route"],
                "taxonomy": _deepcopy(request["taxonomy"]),
                "sampled_attributes": _deepcopy(request["sampled_attributes"]),
                "split": split,
            }
        )

    split_by_instance = {row["instance_id"]: row["split"] for row in request_rows}
    qa_pairs: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        split_requests = [
            request
            for request in validated_requests
            if split_by_instance[request["instance_id"]] == split
        ]
        pairs = build_instance_qa_pairs(
            split_requests,
            profiles,
            single_attribute_only=single_attribute_qa_only,
            max_pairs=max_qa_pairs_per_split,
        )
        for pair in pairs:
            pair["split"] = split
            pair["answer_status"] = "planned_from_attributes_pending_visual_asset"
        qa_pairs.extend(pairs)

    group_splits: dict[str, set[str]] = {}
    for row in request_rows:
        group_splits.setdefault(row["lineage_group_id"], set()).add(row["split"])
    leakage = any(len(splits) > 1 for splits in group_splits.values())
    pair_cross_split = any(
        split_by_instance[pair["instance_a"]]
        != split_by_instance[pair["instance_b"]]
        or split_by_instance[pair["instance_a"]] != pair["split"]
        for pair in qa_pairs
    )
    manifest: dict[str, Any] = {
        "schema": GENERATION_PLAN_SCHEMA,
        "plan_id": plan_id,
        "evidence_state": "planned",
        "split_policy": {
            "algorithm": "sha256_lineage_group_v1",
            "salt": split_salt,
            "ratios": ratios,
            "group_key": "lineage_group_id",
        },
        "profiles": [
            {
                "profile_schema_id": profile["profile_schema_id"],
                "profile_sha256": profile_sha256(profile),
                "lineage_group_id": profile["lineage_group_id"],
            }
            for profile in sorted(profiles.values(), key=lambda item: item["profile_schema_id"])
        ],
        "requests": request_rows,
        "qa_pairs": sorted(qa_pairs, key=lambda item: item["pair_id"]),
        "statistics": {
            **_distribution_stats(
                [
                    {
                        **row,
                        "split": row["split"],
                    }
                    for row in request_rows
                ]
            ),
            "qa_pair_count": len(qa_pairs),
        },
        "automatic_checks": {
            "lineage_leakage": leakage,
            "qa_pair_cross_split": pair_cross_split,
            "duplicate_instance_ids": False,
            "all_answers_pending_visual_asset": all(
                pair["answer_status"]
                == "planned_from_attributes_pending_visual_asset"
                for pair in qa_pairs
            ),
            "overall": "passed" if not leakage and not pair_cross_split else "failed",
        },
    }
    if manifest["automatic_checks"]["overall"] != "passed":
        raise ContractError("generation plan automatic leakage checks failed")
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def build_dataset_manifest(
    assets: Sequence[Mapping[str, Any]],
    profiles_by_id: Mapping[str, Any],
    *,
    dataset_id: str,
    split_salt: str,
    split_ratios: Mapping[str, float] | None = None,
    allowed_states: set[str] | frozenset[str] = frozenset({"formal_dataset_asset"}),
    max_qa_pairs_per_split: int | None = None,
) -> dict[str, Any]:
    _require_id(dataset_id, "dataset_id")
    _require_text(split_salt, "split_salt")
    ratios = _validate_split_ratios(
        split_ratios or {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    if not allowed_states or not set(allowed_states).issubset(STATE_CLASSIFICATIONS):
        raise ContractError("allowed_states must be a non-empty subset of state classifications")
    profiles = {
        profile_id: validate_attribute_profile(profile)
        for profile_id, profile in profiles_by_id.items()
    }
    if any(profile_id != profile["profile_schema_id"] for profile_id, profile in profiles.items()):
        raise ContractError("profiles_by_id keys must equal profile_schema_id")

    validated_assets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_asset in assets:
        profile_id = raw_asset.get("profile_schema_id")
        if profile_id not in profiles:
            raise ContractError(f"missing profile for source asset: {profile_id!r}")
        asset = validate_source_asset_v2(raw_asset, profile=profiles[profile_id])
        if asset["state_classification"] not in allowed_states:
            raise ContractError(
                f"asset {asset['asset_id']} state {asset['state_classification']} is not allowed"
            )
        if asset["asset_id"] in seen_ids:
            raise ContractError(f"duplicate dataset asset_id: {asset['asset_id']}")
        seen_ids.add(asset["asset_id"])
        validated_assets.append(asset)

    asset_rows: list[dict[str, Any]] = []
    for asset in sorted(validated_assets, key=lambda item: item["asset_id"]):
        split = split_for_lineage(
            asset["lineage_group_id"],
            split_salt=split_salt,
            split_ratios=ratios,
        )
        asset_rows.append(
            {
                "asset_id": asset["asset_id"],
                "asset_manifest_sha256": _json_sha256(asset),
                "profile_schema_id": asset["profile_schema_id"],
                "lineage_group_id": asset["lineage_group_id"],
                "asset_class": asset["asset_class"],
                "state_classification": asset["state_classification"],
                "taxonomy": _deepcopy(asset["taxonomy"]),
                "fixed_attributes": _deepcopy(asset["fixed_attributes"]),
                "sampled_attributes": _deepcopy(asset["sampled_attributes"]),
                "semantic_attributes": _deepcopy(asset["semantic_attributes"]),
                "target_physical_profile": _deepcopy(
                    asset["target_physical_profile"]
                ),
                "physical_measurements": _deepcopy(
                    asset["physical_measurements"]
                ),
                "artifacts": _deepcopy(asset["artifacts"]),
                "rig": _deepcopy(asset["rig"]),
                "acoustic_profile": _deepcopy(asset["acoustic_profile"]),
                "acoustic_profile_id": asset["acoustic_profile"]["profile_id"],
                "rights": _deepcopy(asset["rights"]),
                "qa": _deepcopy(asset["qa"]),
                "qa_evidence_attributes": _qa_evidence_attributes(asset),
                "split": split,
            }
        )

    split_by_asset = {row["asset_id"]: row["split"] for row in asset_rows}
    qa_pairs: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        split_assets = [
            asset for asset in validated_assets if split_by_asset[asset["asset_id"]] == split
        ]
        split_pairs = build_instance_qa_pairs(
            split_assets,
            profiles,
            single_attribute_only=False,
            max_pairs=max_qa_pairs_per_split,
        )
        evidence_by_asset = {
            asset["asset_id"]: set(_qa_evidence_attributes(asset))
            for asset in split_assets
        }
        split_pairs = [
            pair
            for pair in split_pairs
            if set(pair["different_attributes"]).issubset(
                evidence_by_asset[pair["instance_a"]]
                & evidence_by_asset[pair["instance_b"]]
            )
        ]
        for pair in split_pairs:
            pair["split"] = split
        qa_pairs.extend(split_pairs)

    group_splits: dict[str, set[str]] = {}
    for row in asset_rows:
        group_splits.setdefault(row["lineage_group_id"], set()).add(row["split"])
    leakage = any(len(splits) > 1 for splits in group_splits.values())
    pair_cross_split = any(
        split_by_asset[pair["instance_a"]] != split_by_asset[pair["instance_b"]]
        or split_by_asset[pair["instance_a"]] != pair["split"]
        for pair in qa_pairs
    )
    manifest: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "dataset_id": dataset_id,
        "evidence_state": "realized",
        "split_policy": {
            "algorithm": "sha256_lineage_group_v1",
            "salt": split_salt,
            "ratios": ratios,
            "group_key": "lineage_group_id",
        },
        "allowed_states": sorted(allowed_states),
        "assets": asset_rows,
        "qa_pairs": sorted(qa_pairs, key=lambda item: item["pair_id"]),
        "statistics": _distribution_stats(asset_rows),
        "automatic_checks": {
            "lineage_leakage": leakage,
            "qa_pair_cross_split": pair_cross_split,
            "duplicate_asset_ids": False,
            "qa_pairs_use_only_evidenced_attributes": all(
                set(pair["different_attributes"]).issubset(
                    set(
                        next(
                            row["qa_evidence_attributes"]
                            for row in asset_rows
                            if row["asset_id"] == pair["instance_a"]
                        )
                    )
                    & set(
                        next(
                            row["qa_evidence_attributes"]
                            for row in asset_rows
                            if row["asset_id"] == pair["instance_b"]
                        )
                    )
                )
                for pair in qa_pairs
            ),
            "overall": "passed" if not leakage and not pair_cross_split else "failed",
        },
    }
    if manifest["automatic_checks"]["overall"] != "passed":
        raise ContractError("dataset automatic leakage checks failed")
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def load_json(path: Path | str) -> Any:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load JSON {path}: {error}") from error


def write_json_no_replace(path: Path | str, value: Any) -> Path:
    """Publish canonical pretty JSON without replacing an existing artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ContractError(f"refusing to replace existing artifact: {path}") from error
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path
