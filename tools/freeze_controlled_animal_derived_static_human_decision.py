#!/usr/bin/env python3
"""Freeze one explicit human approval for a repaired controlled-animal review.

The pending derived-static review remains immutable.  This tool publishes a
separate, hash-bound research-only decision that preserves the original raw
Pixel3D static decision without rewriting it and may authorize only
``source_asset_v2`` registration and rigging.  It never authorizes animation,
UE execution, Native changes, or formal dataset registration.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_animal_derived_static_review_contract as review_contract
from tools import controlled_source_asset_schema as contracts
from tools import freeze_controlled_animal_derived_static_machine_rejection as stable


LEGACY_DECISION_SCHEMA = (
    "avengine_controlled_animal_derived_static_human_decision_v1"
)
DECISION_SCHEMA = "avengine_controlled_animal_derived_static_human_decision_v2"
APPROVED = "approved_for_lod_and_binding"
CHECK_FIELDS = frozenset(review_contract.HUMAN_CHECK_FIELDS)
NEXT_GATE = "lod_then_species_rig_binding"
USER_INSTRUCTION_AUTHORITY = {
    "mode": "caller_assertion_v1",
    "cryptographic_user_identity_verified": False,
    "policy": (
        "the caller may invoke this tool only after an explicit project-owner "
        "approval of the exact externally hashed derived-static review"
    ),
}
EFFECT = {
    "derived_static_admission": "approved_research_only",
    "repaired_asset_registration_authorized": True,
    "source_asset_v2_authorized": True,
    "rigging_authorized": True,
    "animation_authorized": False,
    "ue_execution_authorized": False,
    "native_change_authorized": False,
    "formal_dataset_registration_authorized": False,
}
AUTOMATIC_CHECKS = {
    "external_review_file_sha256_matched": True,
    "internal_review_sha256_matched": True,
    "review_contract_revalidated": True,
    "all_review_artifacts_reauthenticated": True,
    "raw_static_decision_preserved": True,
    "all_six_human_checks_explicit": True,
    "explicit_user_approval_bound_to_exact_review": True,
    "research_only_source_asset_and_rigging_authorized": True,
    "no_animation_ue_native_or_formal_authorization": True,
    "overall": True,
}
AUTHORITY = {
    "review_authority": "explicit_project_owner_caller_assertion_v1",
    "user_decision_claimed": True,
    "human_decision_claimed": True,
    "approval_claimed": True,
    "cryptographic_user_identity_verified": False,
    "product_definition_changed": False,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_DECISION_FIELDS = frozenset(
    {
        "schema",
        "instance_id",
        "decision",
        "checks",
        "attribute_evidence",
        "caveats",
        "notes",
        "review_binding",
        "raw_static_decision",
        "user_instruction_binding",
        "user_instruction_authority",
        "authority",
        "authenticated_review_artifact_count",
        "effect",
        "state_classification",
        "formal_dataset_registration_authorized",
        "next_gate",
        "automatic_checks",
        "producer",
        "decision_sha256",
    }
)
_LEGACY_DECISION_FIELDS = frozenset(
    (set(_DECISION_FIELDS) - {"raw_static_decision"})
    | {"raw_static_rejection"}
)
LEGACY_AUTOMATIC_CHECKS = {
    **{
        name: value
        for name, value in AUTOMATIC_CHECKS.items()
        if name != "raw_static_decision_preserved"
    },
    "raw_static_rejection_preserved": True,
}


def _exact(value: Any, fields: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise contracts.ContractError(f"{label} must be an object")
    if set(value) != set(fields):
        missing = sorted(set(fields) - set(value))
        extra = sorted(set(value) - set(fields))
        raise contracts.ContractError(
            f"{label} fields are invalid: missing={missing} extra={extra}"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise contracts.ContractError(f"{label} must be non-empty text")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise contracts.ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _file_record(value: Any, label: str) -> Mapping[str, Any]:
    record = _exact(value, {"path", "sha256", "size_bytes"}, label)
    path = _text(record["path"], f"{label}.path")
    if not Path(path).is_absolute():
        raise contracts.ContractError(f"{label}.path must be absolute")
    _sha256(record["sha256"], f"{label}.sha256")
    size = record["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise contracts.ContractError(f"{label}.size_bytes must be positive")
    return record


def _validate_raw_static_decision(
    value: Any,
    *,
    require_rejection: bool = False,
) -> None:
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
        "raw_static_decision",
    )
    _file_record(raw["file"], "raw_static_decision.file")
    _sha256(raw["decision_sha256"], "raw_static_decision.decision_sha256")
    expected_state = {
        "rejected": "rejected",
        "approved_for_lod_and_binding": "research_candidate",
    }.get(raw["decision"])
    if (
        expected_state is None
        or raw["state_classification"] != expected_state
        or raw["formal_dataset_registration_authorized"] is not False
        or raw["overwritten"] is not False
        or raw["preserved"] is not True
        or (require_rejection and raw["decision"] != "rejected")
    ):
        raise contracts.ContractError(
            "the original raw static decision must remain preserved"
        )


def _expected_attribute_evidence(review: Mapping[str, Any]) -> dict[str, str]:
    identity = review.get("instance_identity")
    if not isinstance(identity, Mapping):
        raise contracts.ContractError("derived-static review identity is missing")
    sampled = identity.get("sampled_attributes")
    physical = identity.get("target_physical_profile")
    if (
        not isinstance(sampled, Mapping)
        or not sampled
        or not isinstance(physical, Mapping)
        or not isinstance(physical.get("control_attribute"), str)
        or physical["control_attribute"] not in sampled
    ):
        raise contracts.ContractError(
            "derived-static review sampled/control attributes are invalid"
        )
    control = physical["control_attribute"]
    return {
        name: (
            "deferred_to_metric_3d"
            if name == control
            else "passed_static_visual"
        )
        for name in sorted(sampled)
    }


def validate_decision(value: Any) -> dict[str, Any]:
    """Validate a decision and reauthenticate its exact derived-static review."""

    schema = value.get("schema") if isinstance(value, Mapping) else None
    if schema == LEGACY_DECISION_SCHEMA:
        decision = _exact(
            value,
            _LEGACY_DECISION_FIELDS,
            "legacy derived-static human decision",
        )
        raw_field = "raw_static_rejection"
        expected_automatic_checks = LEGACY_AUTOMATIC_CHECKS
    elif schema == DECISION_SCHEMA:
        decision = _exact(value, _DECISION_FIELDS, "derived-static human decision")
        raw_field = "raw_static_decision"
        expected_automatic_checks = AUTOMATIC_CHECKS
    else:
        raise contracts.ContractError("derived-static human decision schema changed")
    instance_id = _text(decision["instance_id"], "instance_id")
    if not _IDENTIFIER_RE.fullmatch(instance_id):
        raise contracts.ContractError("instance_id is not canonical")
    if decision["decision"] != APPROVED:
        raise contracts.ContractError("human decision is not the supported approval")

    checks = _exact(decision["checks"], CHECK_FIELDS, "checks")
    if any(value is not True for value in checks.values()):
        raise contracts.ContractError("approval requires all six human checks to pass")
    caveats = decision["caveats"]
    if (
        not isinstance(caveats, list)
        or any(not isinstance(item, str) or not item for item in caveats)
        or len(caveats) != len(set(caveats))
    ):
        raise contracts.ContractError("caveats must be a unique string list")
    _text(decision["notes"], "notes")

    binding = _exact(
        decision["review_binding"],
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
    if (
        review_file["sha256"] != external
        or internal != expected_internal
        or binding["review_status"] != review_contract.REVIEW_STATUS
        or binding["human_review_status_at_freeze"] != "pending"
        or binding["human_review_decision_at_freeze"] is not None
    ):
        raise contracts.ContractError(
            "decision must bind one exact still-pending derived-static review"
        )
    (
        authenticated_review_path,
        review_bytes,
        authenticated_review_record,
    ) = stable._authenticated_snapshot(
        Path(review_file["path"]),
        external,
        "decision-bound derived-static review",
        expected_size=review_file["size_bytes"],
    )
    if authenticated_review_record != review_file:
        raise contracts.ContractError(
            "decision-bound derived-static review record changed"
        )
    authenticated_review = review_contract.validate_review(
        stable._json_object(
            review_bytes,
            "decision-bound derived-static review",
        )
    )
    if (
        authenticated_review_path != Path(review_file["path"])
        or authenticated_review["instance_identity"]["instance_id"] != instance_id
        or authenticated_review["review_sha256"] != internal
    ):
        raise contracts.ContractError(
            "decision identity/attribute authority changed from its exact review"
        )
    expected_evidence = _expected_attribute_evidence(authenticated_review)
    evidence = decision["attribute_evidence"]
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != set(
            authenticated_review["instance_identity"]["sampled_attributes"]
        )
        or dict(evidence) != expected_evidence
    ):
        raise contracts.ContractError(
            "attribute_evidence does not match the exact review attributes"
        )

    _validate_raw_static_decision(
        decision[raw_field],
        require_rejection=schema == LEGACY_DECISION_SCHEMA,
    )
    review_raw_decision = authenticated_review["source_authorities"][
        "raw_static_decision"
    ]
    decision_raw_authority = decision[raw_field]
    if (
        contracts.canonical_json(decision_raw_authority["file"])
        != contracts.canonical_json(review_raw_decision["file"])
        or decision_raw_authority["decision_sha256"]
        != review_raw_decision["decision_sha256"]
        or decision_raw_authority["decision"]
        != review_raw_decision["decision"]
        or decision_raw_authority["state_classification"]
        != review_raw_decision["state_classification"]
        or decision_raw_authority["formal_dataset_registration_authorized"]
        != review_raw_decision["formal_dataset_registration_authorized"]
    ):
        raise contracts.ContractError(
            "raw static decision is not exactly bound to the frozen review"
        )
    instruction = _exact(
        decision["user_instruction_binding"],
        {
            "decision",
            "review_file_sha256",
            "internal_review_sha256",
            "all_six_checks_explicit",
        },
        "user_instruction_binding",
    )
    if (
        instruction["decision"] != APPROVED
        or instruction["review_file_sha256"] != external
        or instruction["internal_review_sha256"] != internal
        or instruction["all_six_checks_explicit"] is not True
    ):
        raise contracts.ContractError(
            "explicit user approval is not bound to the exact review"
        )
    if decision["user_instruction_authority"] != USER_INSTRUCTION_AUTHORITY:
        raise contracts.ContractError("user instruction authority was upgraded")
    if decision["authority"] != AUTHORITY:
        raise contracts.ContractError("human approval authority changed")

    artifact_count = decision["authenticated_review_artifact_count"]
    if (
        isinstance(artifact_count, bool)
        or not isinstance(artifact_count, int)
        or artifact_count <= 0
        or artifact_count != len(_artifact_descriptors(authenticated_review))
    ):
        raise contracts.ContractError(
            "authenticated_review_artifact_count does not match the frozen review"
        )
    if decision["effect"] != EFFECT:
        raise contracts.ContractError(
            "derived-static approval escaped its research-only authority"
        )
    if (
        decision["state_classification"] != "research_candidate"
        or decision["formal_dataset_registration_authorized"] is not False
        or decision["next_gate"] != NEXT_GATE
        or decision["automatic_checks"] != expected_automatic_checks
    ):
        raise contracts.ContractError(
            "derived-static approval state or gate boundary changed"
        )
    producer = _exact(decision["producer"], {"tool"}, "producer")
    _file_record(producer["tool"], "producer.tool")
    _sha256(decision["decision_sha256"], "decision_sha256")
    if decision["decision_sha256"] != review_contract.hash_without(
        decision, "decision_sha256"
    ):
        raise contracts.ContractError("derived-static decision self-hash changed")
    return copy.deepcopy(dict(decision))


def _artifact_descriptors(review: Mapping[str, Any]) -> list[tuple[str, Any]]:
    authorities = review["source_authorities"]
    geometry = review["derived_geometry"]
    evidence = review["evidence"]
    producer = review["producer"]
    if review.get("schema") == review_contract.DIRECT_REVIEW_SCHEMA:
        direct_authority = authorities["direct_source_authority"]
        primary_authority = (
            "direct source authority",
            {
                name: direct_authority[name]
                for name in ("path", "sha256", "size_bytes")
            },
        )
    else:
        primary_authority = (
            "source frozen preflight",
            authorities["frozen_preflight"]["file"],
        )
    result: list[tuple[str, Any]] = [
        primary_authority,
        ("source Pixal batch", authorities["pixal_batch"]["file"]),
        (
            "source raw static decision batch",
            authorities["raw_static_decision_batch"]["file"],
        ),
        ("source raw static decision", authorities["raw_static_decision"]["file"]),
        ("source raw Pixal GLB", authorities["raw_pixal_glb"]),
        ("source 2D reference", authorities["reference_2d"]),
        ("derived geometry closure", geometry["geometry_closure"]),
        ("derived repaired GLB", geometry["repaired_glb"]),
        ("derived repair manifest", geometry["repair_manifest"]),
        ("derived independent geometry audit", geometry["independent_geometry_audit"]),
        ("derived review producer", producer["tool"]),
        ("derived review renderer", producer["renderer"]),
    ]
    for evidence_name in ("clay_five_view", "pbr_five_view"):
        view = evidence[evidence_name]
        result.append((f"{evidence_name} render manifest", view["render_manifest"]))
        for view_name in review_contract.VIEWS:
            result.append(
                (f"{evidence_name} {view_name} view", view["views"][view_name])
            )
        result.append((f"{evidence_name} contact sheet", view["contact_sheet"]))
        if evidence_name == "pbr_five_view":
            result.append((f"{evidence_name} execution log", view["execution_log"]))
    return result


def _authenticate_review_artifacts(
    review: Mapping[str, Any], review_root: Path
) -> int:
    descriptors = _artifact_descriptors(review)
    for label, descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise contracts.ContractError(f"{label} descriptor is missing")
        stable._descriptor(
            descriptor,
            review_root,
            label,
            descriptor.get("sha256"),
        )
    return len(descriptors)


def freeze_decision(
    *,
    instance_id: str,
    review_path: Path,
    expected_review_file_sha256: str,
    expected_internal_review_sha256: str,
    decision: str,
    checks: Mapping[str, bool],
    caveats: Sequence[str],
    notes: str,
    user_explicit_decision: str,
    user_explicit_review_file_sha256: str,
    output_path: Path,
) -> Path:
    """Authenticate and atomically freeze one research-only human approval."""

    expected_internal_review_sha256 = stable._require_sha256(
        expected_internal_review_sha256,
        "expected internal review SHA-256",
    )
    expected_review_file_sha256 = stable._require_sha256(
        expected_review_file_sha256,
        "expected review file SHA-256",
    )
    user_explicit_review_file_sha256 = stable._require_sha256(
        user_explicit_review_file_sha256,
        "user-explicit review file SHA-256",
    )
    if (
        decision != APPROVED
        or user_explicit_decision != APPROVED
        or user_explicit_decision != decision
    ):
        raise contracts.ContractError(
            "the frozen approval must match the user's explicit approval"
        )
    if user_explicit_review_file_sha256 != expected_review_file_sha256:
        raise contracts.ContractError(
            "the user's explicit approval is not bound to the expected review"
        )
    if (
        not isinstance(checks, Mapping)
        or set(checks) != CHECK_FIELDS
        or any(not isinstance(value, bool) for value in checks.values())
    ):
        raise contracts.ContractError(
            "all six derived-static human checks must be explicit booleans"
        )
    if not all(checks.values()):
        raise contracts.ContractError(
            "research approval requires all six human checks to pass"
        )
    if (
        not isinstance(notes, str)
        or not notes.strip()
        or not isinstance(caveats, Sequence)
        or isinstance(caveats, (str, bytes))
        or any(not isinstance(value, str) or not value for value in caveats)
        or len(caveats) != len(set(caveats))
    ):
        raise contracts.ContractError("decision notes/caveats are invalid")

    review_path, review_bytes, review_record = stable._authenticated_snapshot(
        review_path,
        expected_review_file_sha256,
        "derived-static review JSON",
    )
    review = review_contract.validate_review(
        stable._json_object(review_bytes, "derived-static review JSON")
    )
    if (
        review["instance_identity"]["instance_id"] != instance_id
        or review["review_sha256"] != expected_internal_review_sha256
    ):
        raise contracts.ContractError(
            "derived-static review instance or internal SHA-256 changed"
        )
    if (
        review["status"] != review_contract.REVIEW_STATUS
        or review["formal_dataset_registration_authorized"] is not False
        or review["human_review"]["status"] != "pending"
        or review["human_review"]["decision"] is not None
    ):
        raise contracts.ContractError(
            "human approval requires a still-pending derived-static review"
        )
    artifact_count = _authenticate_review_artifacts(review, review_path.parent)
    raw_static_decision = stable._validate_raw_static_decision(
        review, review_path.parent
    )

    payload: dict[str, Any] = {
        "schema": DECISION_SCHEMA,
        "instance_id": instance_id,
        "decision": APPROVED,
        "checks": {name: checks[name] for name in sorted(CHECK_FIELDS)},
        "attribute_evidence": _expected_attribute_evidence(review),
        "caveats": list(caveats),
        "notes": notes,
        "review_binding": {
            "review_file": review_record,
            "expected_external_review_file_sha256": expected_review_file_sha256,
            "internal_review_sha256": review["review_sha256"],
            "expected_internal_review_sha256": expected_internal_review_sha256,
            "review_status": review["status"],
            "human_review_status_at_freeze": review["human_review"]["status"],
            "human_review_decision_at_freeze": review["human_review"]["decision"],
        },
        "raw_static_decision": raw_static_decision,
        "user_instruction_binding": {
            "decision": APPROVED,
            "review_file_sha256": user_explicit_review_file_sha256,
            "internal_review_sha256": expected_internal_review_sha256,
            "all_six_checks_explicit": True,
        },
        "user_instruction_authority": copy.deepcopy(USER_INSTRUCTION_AUTHORITY),
        "authority": copy.deepcopy(AUTHORITY),
        "authenticated_review_artifact_count": artifact_count,
        "effect": copy.deepcopy(EFFECT),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": NEXT_GATE,
        "automatic_checks": copy.deepcopy(AUTOMATIC_CHECKS),
        "producer": {
            "tool": stable._producer_record(
                Path(__file__),
                "derived-static human decision producer",
            )
        },
        "decision_sha256": "",
    }
    payload["decision_sha256"] = review_contract.hash_without(
        payload, "decision_sha256"
    )
    payload = validate_decision(payload)
    encoded = (contracts.canonical_json(payload) + "\n").encode("utf-8")
    return stable._publish_atomic_no_replace(output_path, encoded)


def _explicit_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise argparse.ArgumentTypeError("expected exactly True or False")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--expected-review-file-sha256", required=True)
    parser.add_argument("--expected-internal-review-sha256", required=True)
    parser.add_argument("--decision", choices=(APPROVED,), required=True)
    parser.add_argument(
        "--user-explicit-decision",
        choices=(APPROVED,),
        required=True,
    )
    parser.add_argument("--user-explicit-review-file-sha256", required=True)
    for name in sorted(CHECK_FIELDS):
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=_explicit_bool,
            choices=(True, False),
            required=True,
        )
    parser.add_argument("--notes", required=True)
    parser.add_argument("--caveat", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        output = freeze_decision(
            instance_id=args.instance_id,
            review_path=args.review,
            expected_review_file_sha256=args.expected_review_file_sha256,
            expected_internal_review_sha256=args.expected_internal_review_sha256,
            decision=args.decision,
            checks={name: getattr(args, name) for name in CHECK_FIELDS},
            caveats=args.caveat,
            notes=args.notes,
            user_explicit_decision=args.user_explicit_decision,
            user_explicit_review_file_sha256=(
                args.user_explicit_review_file_sha256
            ),
            output_path=args.output,
        )
        payload = validate_decision(
            stable._json_object(
                output.read_bytes(),
                "published derived-static human decision",
            )
        )
    except (contracts.ContractError, OSError) as error:
        print(
            f"CONTROLLED_ANIMAL_DERIVED_STATIC_HUMAN_DECISION_FAILED {error}",
            file=sys.stderr,
        )
        return 2
    print(
        "CONTROLLED_ANIMAL_DERIVED_STATIC_HUMAN_DECISION_OK "
        f"decision={payload['decision']} output={output} "
        f"decision_sha256={payload['decision_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
