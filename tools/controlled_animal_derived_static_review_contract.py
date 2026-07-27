"""Strict contract for a repaired controlled-animal static review.

The contract deliberately stops before any approval or source registration.
It can describe authenticated automatic evidence and inherited manual-review
claims, but its human decision fields must remain null.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from tools import controlled_source_asset_schema as contracts

REVIEW_SCHEMA = "avengine_controlled_animal_derived_static_review_v1"
REVIEW_STATUS = "rendered_pending_human_derived_static_review"
NEXT_GATE = "explicit_human_derived_static_review_decision"
VIEWS = ("front", "back", "side", "top", "quarter")
FRONT_AXES = frozenset({"negative-x", "positive-x", "negative-y", "positive-y"})
HUMAN_CHECK_FIELDS = frozenset(
    {
        "breed_and_body_shape_coherent",
        "coat_and_pbr_appearance_coherent",
        "four_complete_riggable_limbs",
        "no_large_holes_or_detached_geometry",
        "no_rear_whisker_bar_or_stray_geometry",
        "one_tail_without_duplicate_tail",
    }
)
AUTOMATIC_CHECK_FIELDS = frozenset(
    {
        "frozen_request_reauthenticated",
        "pixal_attempt_reauthenticated",
        "raw_static_rejection_preserved",
        "raw_rejection_not_overwritten_or_upgraded",
        "bounded_same_source_repair_lineage_reauthenticated",
        "repair_geometry_closure_reauthenticated",
        "repaired_glb_matched_geometry_closure",
        "automatic_topology_gates_reauthenticated",
        "inherited_manual_geometry_claims_preserved_as_non_authoritative",
        "clay_five_view_evidence_reauthenticated",
        "pbr_container_present",
        "pbr_five_view_rendered",
        "pbr_fidelity_pending_human",
        "all_human_checks_pending",
        "no_source_asset_or_registry_published",
        "no_ue_or_native_execution_performed",
        "overall",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class DerivedStaticReviewContractError(ValueError):
    """Fail-closed derived-static review contract error."""


def _canonical(value: Any) -> str:
    try:
        return contracts.canonical_json(value)
    except contracts.ContractError as error:
        raise DerivedStaticReviewContractError(str(error)) from error


def hash_without(value: Mapping[str, Any], field: str) -> str:
    payload = {
        name: copy.deepcopy(item) for name, item in value.items() if name != field
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DerivedStaticReviewContractError(f"{label} must be an object")
    return value


def _exact(
    value: Any, fields: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    result = _mapping(value, label)
    if set(result) != set(fields):
        missing = sorted(set(fields) - set(result))
        extra = sorted(set(result) - set(fields))
        raise DerivedStaticReviewContractError(
            f"{label} fields are invalid: missing={missing} extra={extra}"
        )
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DerivedStaticReviewContractError(f"{label} must be non-empty text")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _text(value, label)
    if not _ID_RE.fullmatch(value):
        raise DerivedStaticReviewContractError(f"{label} is not a canonical identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise DerivedStaticReviewContractError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DerivedStaticReviewContractError(f"{label} must be a positive integer")
    return value


def _file_record(value: Any, label: str, *, absolute: bool) -> dict[str, Any]:
    record = _exact(value, {"path", "sha256", "size_bytes"}, label)
    path = _text(record["path"], f"{label}.path")
    pure = PurePosixPath(path)
    if absolute:
        if not Path(path).is_absolute():
            raise DerivedStaticReviewContractError(f"{label}.path must be absolute")
    elif pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise DerivedStaticReviewContractError(
            f"{label}.path must stay relative to the review root"
        )
    _sha256(record["sha256"], f"{label}.sha256")
    _positive_integer(record["size_bytes"], f"{label}.size_bytes")
    return copy.deepcopy(dict(record))


def _finite_json(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DerivedStaticReviewContractError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise DerivedStaticReviewContractError(
                    f"{label} contains a non-string key"
                )
            _finite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_json(child, f"{label}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise DerivedStaticReviewContractError(f"{label} is not JSON-compatible")


def _view_evidence(
    value: Any,
    label: str,
    *,
    internal: bool,
    execution_log: bool,
) -> dict[str, Any]:
    fields = {
        "front_axis",
        "resolution",
        "material_mode",
        "render_manifest",
        "views",
        "contact_sheet",
    }
    if execution_log:
        fields.add("execution_log")
    evidence = _exact(value, fields, label)
    if evidence["front_axis"] not in FRONT_AXES:
        raise DerivedStaticReviewContractError(f"{label}.front_axis is invalid")
    if evidence["resolution"] != [480, 480]:
        raise DerivedStaticReviewContractError(f"{label}.resolution must be [480, 480]")
    _text(evidence["material_mode"], f"{label}.material_mode")
    record_absolute = not internal
    _file_record(
        evidence["render_manifest"],
        f"{label}.render_manifest",
        absolute=record_absolute,
    )
    views = _exact(evidence["views"], set(VIEWS), f"{label}.views")
    for view_name in VIEWS:
        _file_record(
            views[view_name],
            f"{label}.views.{view_name}",
            absolute=record_absolute,
        )
    _file_record(
        evidence["contact_sheet"],
        f"{label}.contact_sheet",
        absolute=record_absolute,
    )
    if execution_log:
        _file_record(
            evidence["execution_log"],
            f"{label}.execution_log",
            absolute=False,
        )
    return copy.deepcopy(dict(evidence))


def validate_review(value: Any) -> dict[str, Any]:
    """Validate a pending-human review without performing filesystem I/O."""

    fields = {
        "schema",
        "created_at",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "claim_boundary",
        "instance_identity",
        "source_authorities",
        "derived_geometry",
        "evidence",
        "producer",
        "automatic_checks",
        "human_review",
        "next_gate",
        "review_sha256",
    }
    review = _exact(value, fields, "derived static review")
    _finite_json(review, "derived static review")
    if review["schema"] != REVIEW_SCHEMA:
        raise DerivedStaticReviewContractError("derived static review schema changed")
    _text(review["created_at"], "created_at")
    if review["status"] != REVIEW_STATUS:
        raise DerivedStaticReviewContractError("derived static review status changed")
    if review["state_classification"] != "research_candidate":
        raise DerivedStaticReviewContractError(
            "derived static review must remain a research candidate"
        )
    if review["formal_dataset_registration_authorized"] is not False:
        raise DerivedStaticReviewContractError(
            "derived static review cannot authorize formal registration"
        )
    _text(review["claim_boundary"], "claim_boundary")

    identity = _exact(
        review["instance_identity"],
        {
            "instance_id",
            "profile_schema_id",
            "profile_sha256",
            "request_sha256",
            "taxonomy",
            "fixed_attributes",
            "sampled_attributes",
            "target_physical_profile",
        },
        "instance_identity",
    )
    _identifier(identity["instance_id"], "instance_identity.instance_id")
    _identifier(identity["profile_schema_id"], "instance_identity.profile_schema_id")
    _sha256(identity["profile_sha256"], "instance_identity.profile_sha256")
    _sha256(identity["request_sha256"], "instance_identity.request_sha256")
    for name in (
        "taxonomy",
        "fixed_attributes",
        "sampled_attributes",
        "target_physical_profile",
    ):
        _mapping(identity[name], f"instance_identity.{name}")

    authorities = _exact(
        review["source_authorities"],
        {
            "frozen_preflight",
            "pixal_batch",
            "raw_static_decision_batch",
            "raw_static_decision",
            "raw_pixal_glb",
            "reference_2d",
        },
        "source_authorities",
    )
    preflight = _exact(
        authorities["frozen_preflight"],
        {"file", "preflight_sha256", "validation_mode"},
        "source_authorities.frozen_preflight",
    )
    _file_record(preflight["file"], "frozen preflight", absolute=True)
    _sha256(preflight["preflight_sha256"], "frozen preflight internal SHA-256")
    if preflight["validation_mode"] != "frozen_historical_preflight_v1":
        raise DerivedStaticReviewContractError(
            "derived review requires frozen historical preflight validation"
        )
    pixal = _exact(
        authorities["pixal_batch"],
        {"file", "batch_sha256"},
        "source_authorities.pixal_batch",
    )
    _file_record(pixal["file"], "Pixal batch", absolute=True)
    _sha256(pixal["batch_sha256"], "Pixal batch internal SHA-256")
    decision_batch = _exact(
        authorities["raw_static_decision_batch"],
        {"file", "decision_batch_sha256"},
        "source_authorities.raw_static_decision_batch",
    )
    _file_record(decision_batch["file"], "raw static decision batch", absolute=True)
    _sha256(
        decision_batch["decision_batch_sha256"],
        "raw static decision batch internal SHA-256",
    )
    decision = _exact(
        authorities["raw_static_decision"],
        {
            "file",
            "decision_sha256",
            "decision",
            "state_classification",
            "formal_dataset_registration_authorized",
        },
        "source_authorities.raw_static_decision",
    )
    _file_record(decision["file"], "raw static decision", absolute=True)
    _sha256(decision["decision_sha256"], "raw static decision internal SHA-256")
    if (
        decision["decision"] != "rejected"
        or decision["state_classification"] != "rejected"
        or decision["formal_dataset_registration_authorized"] is not False
    ):
        raise DerivedStaticReviewContractError(
            "the raw static rejection must be preserved exactly"
        )
    _file_record(authorities["raw_pixal_glb"], "raw Pixal GLB", absolute=True)
    _file_record(authorities["reference_2d"], "2D reference", absolute=True)

    geometry = _exact(
        review["derived_geometry"],
        {
            "geometry_closure",
            "repaired_glb",
            "repair_manifest",
            "independent_geometry_audit",
            "repair_method",
            "lineage_kind",
            "automatic_gate_statuses",
            "inherited_manual_review_statuses",
            "pbr_container_readback",
        },
        "derived_geometry",
    )
    for name in (
        "geometry_closure",
        "repaired_glb",
        "repair_manifest",
        "independent_geometry_audit",
    ):
        _file_record(geometry[name], f"derived_geometry.{name}", absolute=True)
    _text(geometry["repair_method"], "derived_geometry.repair_method")
    if geometry["lineage_kind"] != "bounded_same_pixal_mesh_repair":
        raise DerivedStaticReviewContractError("derived geometry lineage kind changed")
    automatic_statuses = _exact(
        geometry["automatic_gate_statuses"],
        {
            "four_independent_leg_chains",
            "no_low_cross_limb_membrane",
            "nonmanifold",
            "watertight",
        },
        "derived_geometry.automatic_gate_statuses",
    )
    if any(value != "passed" for value in automatic_statuses.values()):
        raise DerivedStaticReviewContractError(
            "derived geometry automatic gate did not pass"
        )
    inherited_statuses = _exact(
        geometry["inherited_manual_review_statuses"],
        {"single_breed_valid_tail", "centerline", "clay_five_view"},
        "derived_geometry.inherited_manual_review_statuses",
    )
    expected_inherited = {
        "single_breed_valid_tail": "passed_manual_multiview",
        "centerline": "passed_manual_top_view_review",
        "clay_five_view": "passed_manual_geometry_review",
    }
    if dict(inherited_statuses) != expected_inherited:
        raise DerivedStaticReviewContractError(
            "inherited manual geometry statuses changed"
        )
    pbr = _exact(
        geometry["pbr_container_readback"],
        {
            "material_count",
            "texture_count",
            "image_count",
            "pbr_fidelity_qualified",
        },
        "derived_geometry.pbr_container_readback",
    )
    for name in ("material_count", "texture_count", "image_count"):
        _positive_integer(pbr[name], f"derived_geometry.pbr_container_readback.{name}")
    if pbr["pbr_fidelity_qualified"] is not False:
        raise DerivedStaticReviewContractError(
            "PBR fidelity must remain pending human review"
        )

    evidence = _exact(
        review["evidence"],
        {"clay_five_view", "pbr_five_view"},
        "evidence",
    )
    clay = _view_evidence(
        evidence["clay_five_view"],
        "evidence.clay_five_view",
        internal=False,
        execution_log=False,
    )
    pbr_evidence = _view_evidence(
        evidence["pbr_five_view"],
        "evidence.pbr_five_view",
        internal=True,
        execution_log=True,
    )
    if clay["material_mode"] != "neutral_clay_geometry_qa_v1":
        raise DerivedStaticReviewContractError("clay evidence material mode changed")
    if pbr_evidence["material_mode"] != "ue_animal_nonmetallic_roughness_preview_v1":
        raise DerivedStaticReviewContractError("PBR evidence material mode changed")
    if pbr_evidence["front_axis"] != clay["front_axis"]:
        raise DerivedStaticReviewContractError("clay and PBR review front axes differ")

    producer = _exact(
        review["producer"],
        {"tool", "renderer", "blender"},
        "producer",
    )
    _file_record(producer["tool"], "producer.tool", absolute=True)
    _file_record(producer["renderer"], "producer.renderer", absolute=True)
    blender = _exact(
        producer["blender"],
        {"path", "version", "build_hash"},
        "producer.blender",
    )
    if not Path(_text(blender["path"], "producer.blender.path")).is_absolute():
        raise DerivedStaticReviewContractError("producer.blender.path must be absolute")
    _text(blender["version"], "producer.blender.version")
    _text(blender["build_hash"], "producer.blender.build_hash")

    automatic = _exact(
        review["automatic_checks"],
        AUTOMATIC_CHECK_FIELDS,
        "automatic_checks",
    )
    if any(value is not True for value in automatic.values()):
        raise DerivedStaticReviewContractError(
            "all derived review automatic checks must be true"
        )
    human = _exact(
        review["human_review"],
        {
            "status",
            "decision",
            "checks",
            "review_authority_required",
            "decision_artifact",
        },
        "human_review",
    )
    if (
        human["status"] != "pending"
        or human["decision"] is not None
        or human["decision_artifact"] is not None
        or human["review_authority_required"]
        != "explicit_project_owner_decision_bound_to_review_file_sha256"
    ):
        raise DerivedStaticReviewContractError(
            "derived static human review must remain explicitly pending"
        )
    checks = _exact(human["checks"], HUMAN_CHECK_FIELDS, "human_review.checks")
    if any(value is not None for value in checks.values()):
        raise DerivedStaticReviewContractError(
            "derived static human checks cannot be inferred automatically"
        )
    if review["next_gate"] != NEXT_GATE:
        raise DerivedStaticReviewContractError("derived static next gate changed")
    _sha256(review["review_sha256"], "review_sha256")
    if review["review_sha256"] != hash_without(review, "review_sha256"):
        raise DerivedStaticReviewContractError(
            "derived static review self-hash changed"
        )
    return copy.deepcopy(dict(review))
