"""Strict contract for a machine-observed derived-static rejection receipt.

The receipt is intentionally narrower than a human decision.  It can only
freeze fail-closed objective defect observations against one exact pending
derived-static review and its exact visual artifacts.  It cannot approve an
asset, alter the preserved raw rejection, or authorize registration, rigging,
animation, UE, or Native work.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import controlled_source_asset_schema as contracts

RECEIPT_SCHEMA = (
    "avengine_controlled_animal_derived_static_machine_rejection_receipt_v1"
)
RECEIPT_STATUS = "frozen_fail_closed_objective_rejection"
RECEIPT_SCOPE = "exact_derived_static_review_only"
REVIEW_STATUS = "rendered_pending_human_derived_static_review"
NEXT_GATE = "remediate_and_publish_new_derived_static_review"
OBSERVATION_MODE = "machine_visual_inspection_of_exact_artifact_v1"
OBSERVER_KIND = "machine_objective_defect_observer_v1"

DEFECT_CODES = frozenset(
    {
        "visible_cross_limb_membrane",
        "visible_detached_component",
        "visible_duplicate_tail",
        "visible_fused_limbs",
        "visible_hole_like_opening",
        "visible_missing_limb",
        "visible_stray_bar_or_spike_geometry",
        "visible_texture_projection_discontinuity",
    }
)
EVIDENCE_ROLES = frozenset(
    {
        "clay_back",
        "clay_contact_sheet",
        "clay_front",
        "clay_quarter",
        "clay_side",
        "clay_top",
        "pbr_back",
        "pbr_contact_sheet",
        "pbr_front",
        "pbr_quarter",
        "pbr_side",
        "pbr_top",
    }
)
AUTOMATIC_CHECK_FIELDS = frozenset(
    {
        "all_defect_codes_enumerated",
        "all_evidence_artifacts_reauthenticated",
        "external_review_file_sha256_matched",
        "internal_review_sha256_matched",
        "no_approval_or_registration_authorized",
        "no_ue_or_native_execution_performed",
        "no_user_or_human_decision_claimed",
        "overall",
        "raw_static_rejection_preserved",
        "review_contract_revalidated",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class MachineRejectionContractError(ValueError):
    """Fail-closed machine rejection receipt error."""


def _canonical(value: Any) -> str:
    try:
        return contracts.canonical_json(value)
    except contracts.ContractError as error:
        raise MachineRejectionContractError(str(error)) from error


def hash_without(value: Mapping[str, Any], field: str) -> str:
    payload = {
        name: copy.deepcopy(item) for name, item in value.items() if name != field
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MachineRejectionContractError(f"{label} must be an object")
    return value


def _exact(
    value: Any, fields: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    if set(result) != set(fields):
        missing = sorted(set(fields) - set(result))
        extra = sorted(set(result) - set(fields))
        raise MachineRejectionContractError(
            f"{label} fields are invalid: missing={missing} extra={extra}"
        )
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MachineRejectionContractError(f"{label} must be non-empty text")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _text(value, label)
    if not _IDENTIFIER_RE.fullmatch(value):
        raise MachineRejectionContractError(f"{label} is not a canonical identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise MachineRejectionContractError(f"{label} must be a lowercase SHA-256")
    return value


def _file_record(value: Any, label: str) -> dict[str, Any]:
    record = _exact(value, {"path", "sha256", "size_bytes"}, label)
    path = _text(record["path"], f"{label}.path")
    if not Path(path).is_absolute():
        raise MachineRejectionContractError(f"{label}.path must be absolute")
    _sha256(record["sha256"], f"{label}.sha256")
    size = record["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise MachineRejectionContractError(
            f"{label}.size_bytes must be a positive integer"
        )
    return copy.deepcopy(dict(record))


def _must_be_false(value: Any, label: str) -> None:
    if value is not False:
        raise MachineRejectionContractError(f"{label} must remain false")


def _finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise MachineRejectionContractError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise MachineRejectionContractError(
                    f"{label} contains a non-string key"
                )
            _finite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_json(child, f"{label}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise MachineRejectionContractError(f"{label} is not JSON-compatible")


def _validate_review_binding(value: Any) -> None:
    binding = _exact(
        value,
        {
            "review_file",
            "expected_external_review_file_sha256",
            "internal_review_sha256",
            "expected_internal_review_sha256",
            "review_status",
            "human_review_status_at_freeze",
            "human_review_decision_at_freeze",
        },
        "review_binding",
    )
    review_file = _file_record(binding["review_file"], "review_binding.review_file")
    external = _sha256(
        binding["expected_external_review_file_sha256"],
        "review_binding.expected_external_review_file_sha256",
    )
    internal = _sha256(
        binding["internal_review_sha256"],
        "review_binding.internal_review_sha256",
    )
    expected_internal = _sha256(
        binding["expected_internal_review_sha256"],
        "review_binding.expected_internal_review_sha256",
    )
    if external != review_file["sha256"]:
        raise MachineRejectionContractError(
            "external review SHA-256 is not bound to the review file record"
        )
    if internal != expected_internal:
        raise MachineRejectionContractError(
            "internal review SHA-256 is not bound to caller authority"
        )
    if (
        binding["review_status"] != REVIEW_STATUS
        or binding["human_review_status_at_freeze"] != "pending"
        or binding["human_review_decision_at_freeze"] is not None
    ):
        raise MachineRejectionContractError(
            "machine rejection must bind a still-pending derived-static review"
        )


def _validate_raw_rejection(value: Any) -> None:
    raw = _exact(
        value,
        {
            "decision",
            "decision_sha256",
            "file",
            "formal_dataset_registration_authorized",
            "overwritten",
            "preserved",
            "state_classification",
        },
        "raw_static_rejection",
    )
    _file_record(raw["file"], "raw_static_rejection.file")
    _sha256(raw["decision_sha256"], "raw_static_rejection.decision_sha256")
    if (
        raw["decision"] != "rejected"
        or raw["state_classification"] != "rejected"
        or raw["preserved"] is not True
        or raw["overwritten"] is not False
    ):
        raise MachineRejectionContractError(
            "the original raw static rejection must remain preserved"
        )
    _must_be_false(
        raw["formal_dataset_registration_authorized"],
        "raw_static_rejection.formal_dataset_registration_authorized",
    )


def _validate_observations(value: Any) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise MachineRejectionContractError(
            "observations must contain at least one objective defect"
        )
    canonical_order: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        observation = _exact(
            item,
            {
                "artifact",
                "defect_code",
                "evidence_role",
                "expected_artifact_sha256",
                "gate_result",
                "observation_mode",
            },
            f"observations[{index}]",
        )
        code = observation["defect_code"]
        role = observation["evidence_role"]
        if code not in DEFECT_CODES:
            raise MachineRejectionContractError(
                f"observations[{index}].defect_code is not enumerated"
            )
        if role not in EVIDENCE_ROLES:
            raise MachineRejectionContractError(
                f"observations[{index}].evidence_role is not enumerated"
            )
        artifact = _file_record(
            observation["artifact"], f"observations[{index}].artifact"
        )
        expected = _sha256(
            observation["expected_artifact_sha256"],
            f"observations[{index}].expected_artifact_sha256",
        )
        if artifact["sha256"] != expected:
            raise MachineRejectionContractError(
                f"observations[{index}] artifact SHA-256 binding changed"
            )
        if (
            observation["observation_mode"] != OBSERVATION_MODE
            or observation["gate_result"] != "rejected"
        ):
            raise MachineRejectionContractError(
                f"observations[{index}] is not a fail-closed machine observation"
            )
        canonical_order.append((code, role))
    if canonical_order != sorted(canonical_order) or len(canonical_order) != len(
        set(canonical_order)
    ):
        raise MachineRejectionContractError(
            "observations must be unique and canonically sorted"
        )


def validate_receipt(value: Any) -> dict[str, Any]:
    """Validate and return a defensive copy of a machine rejection receipt."""

    receipt = _exact(
        value,
        {
            "authority",
            "automatic_checks",
            "effect",
            "formal_dataset_registration_authorized",
            "instance_id",
            "observations",
            "producer",
            "raw_static_rejection",
            "receipt_sha256",
            "review_binding",
            "schema",
            "scope",
            "state_classification",
            "status",
        },
        "receipt",
    )
    _finite_json(receipt, "receipt")
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["status"] != RECEIPT_STATUS
        or receipt["scope"] != RECEIPT_SCOPE
        or receipt["state_classification"] != "rejected"
    ):
        raise MachineRejectionContractError(
            "machine rejection receipt identity or state changed"
        )
    _identifier(receipt["instance_id"], "instance_id")
    _must_be_false(
        receipt["formal_dataset_registration_authorized"],
        "formal_dataset_registration_authorized",
    )
    _validate_review_binding(receipt["review_binding"])
    _validate_raw_rejection(receipt["raw_static_rejection"])
    _validate_observations(receipt["observations"])

    authority = _exact(
        receipt["authority"],
        {
            "approval_claimed",
            "human_decision_claimed",
            "observer_kind",
            "product_definition_changed",
            "user_decision_claimed",
        },
        "authority",
    )
    if authority["observer_kind"] != OBSERVER_KIND:
        raise MachineRejectionContractError("machine observer kind changed")
    for field in (
        "approval_claimed",
        "human_decision_claimed",
        "product_definition_changed",
        "user_decision_claimed",
    ):
        _must_be_false(authority[field], f"authority.{field}")

    effect = _exact(
        receipt["effect"],
        {
            "animation_authorized",
            "derived_static_admission",
            "native_change_authorized",
            "next_gate",
            "repaired_asset_registration_authorized",
            "rigging_authorized",
            "source_asset_v2_authorized",
            "ue_execution_authorized",
        },
        "effect",
    )
    if (
        effect["derived_static_admission"] != "rejected_fail_closed"
        or effect["next_gate"] != NEXT_GATE
    ):
        raise MachineRejectionContractError("machine rejection effect changed")
    for field in (
        "animation_authorized",
        "native_change_authorized",
        "repaired_asset_registration_authorized",
        "rigging_authorized",
        "source_asset_v2_authorized",
        "ue_execution_authorized",
    ):
        _must_be_false(effect[field], f"effect.{field}")

    checks = _exact(
        receipt["automatic_checks"],
        AUTOMATIC_CHECK_FIELDS,
        "automatic_checks",
    )
    if any(value is not True for value in checks.values()):
        raise MachineRejectionContractError(
            "all machine rejection automatic checks must pass"
        )

    producer = _exact(receipt["producer"], {"contract", "tool"}, "producer")
    _file_record(producer["contract"], "producer.contract")
    _file_record(producer["tool"], "producer.tool")

    _sha256(receipt["receipt_sha256"], "receipt_sha256")
    if receipt["receipt_sha256"] != hash_without(receipt, "receipt_sha256"):
        raise MachineRejectionContractError("machine rejection self-hash changed")
    return copy.deepcopy(dict(receipt))
