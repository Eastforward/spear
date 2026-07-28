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
REPAIR_MANIFEST_SCHEMA = "avengine_pixal_same_mesh_mirrored_limb_repair_v1"
REPAIR_IMPLEMENTATION_CONTRACT = "bounded_local_surface_identity_v2"
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
REPAIR_CHECK_FIELDS = frozenset(
    {
        "authenticated_same_pixal_mesh_only",
        "near_side_front_and_hind_donor_masks_passed",
        "height_independent_source_tail_identified",
        "tail_region_never_selected_for_replacement_or_mirroring",
        "source_topology_outside_corridors_already_closed",
        "four_independent_low_limb_chains",
        "no_low_cross_limb_membrane",
        "one_connected_output_component",
        "watertight_manifold_topology",
        "output_within_authenticated_source_envelope",
        "outside_corridor_position_index_uv_material_preserved",
        "authenticated_source_tail_surface_preserved",
        "embedded_textures_and_pbr_unchanged",
    }
)


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


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DerivedStaticReviewContractError(
            f"{label} must be a nonnegative integer"
        )
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


def _surface_identity_audit(value: Any, label: str) -> Mapping[str, Any]:
    audit = _exact(
        value,
        {
            "method",
            "expected_triangle_count",
            "actual_matching_triangle_count",
            "missing_triangle_count",
            "unexpected_duplicate_triangle_count",
            "expected_signature_sha256",
            "missing_signature_sha256",
            "unexpected_duplicate_signature_sha256",
            "passed",
        },
        label,
    )
    if (
        audit["method"]
        != "exact_float32_position_uv_material_triangle_multiset_with_winding_preserved"
    ):
        raise DerivedStaticReviewContractError(
            f"{label} surface-identity method changed"
        )
    expected = _positive_integer(
        audit["expected_triangle_count"],
        f"{label}.expected_triangle_count",
    )
    actual = _positive_integer(
        audit["actual_matching_triangle_count"],
        f"{label}.actual_matching_triangle_count",
    )
    missing = _nonnegative_integer(
        audit["missing_triangle_count"],
        f"{label}.missing_triangle_count",
    )
    duplicates = _nonnegative_integer(
        audit["unexpected_duplicate_triangle_count"],
        f"{label}.unexpected_duplicate_triangle_count",
    )
    for name in (
        "expected_signature_sha256",
        "missing_signature_sha256",
        "unexpected_duplicate_signature_sha256",
    ):
        _sha256(audit[name], f"{label}.{name}")
    if (
        audit["passed"] is not True
        or actual != expected
        or missing != 0
        or duplicates != 0
    ):
        raise DerivedStaticReviewContractError(
            f"{label} did not preserve the complete source-surface identity"
        )
    return audit


def _closed_topology_audit(value: Any, label: str) -> Mapping[str, Any]:
    audit = _mapping(value, label)
    if (
        audit.get("method")
        != "exact_position_logical_edge_incidence_outside_authorized_corridors"
        or audit.get("passed") is not True
        or audit.get("failure_meaning") is not None
        or audit.get("immutable_component_count") != 1
        or audit.get("outside_corridor_boundary_edges") != 0
        or audit.get("outside_corridor_edges_over_two_faces") != 0
        or audit.get("outside_corridor_orientation_mismatch_edges") != 0
        or audit.get("degenerate_triangle_count") != 0
    ):
        raise DerivedStaticReviewContractError(
            f"{label} does not prove closed immutable topology"
        )
    for name in (
        "raw_vertex_count",
        "logical_position_count",
        "triangle_count",
        "logical_edge_count",
        "logical_component_count",
        "immutable_component_count",
    ):
        _positive_integer(audit.get(name), f"{label}.{name}")
    _nonnegative_integer(
        audit.get("wholly_mutable_component_count"),
        f"{label}.wholly_mutable_component_count",
    )
    return audit


def validate_bounded_repair_manifest(value: Any) -> dict[str, Any]:
    """Validate the current bounded local repair result without filesystem I/O.

    The original v1 producer used the same schema string while globally voxel
    remeshing, smoothing, and decimating the animal.  The implementation
    contract and export readback are therefore mandatory parts of the schema
    identity; a schema-only consumer is unsafe.
    """

    manifest = _exact(
        value,
        {
            "schema",
            "implementation_contract",
            "created_at",
            "lineage",
            "repair_spec",
            "coordinate_contract",
            "source_pbr_contract",
            "tail_source_surface",
            "mask_audit",
            "face_scope",
            "source_topology_preflight",
            "mutation",
            "topology",
            "low_slice_dynamic_geometry_gate",
            "output_envelope",
            "export_readback",
            "checks",
            "decision",
            "output",
            "formal_dataset_registration_authorized",
        },
        "bounded geometry repair manifest",
    )
    _finite_json(manifest, "bounded geometry repair manifest")
    if (
        manifest["schema"] != REPAIR_MANIFEST_SCHEMA
        or manifest["implementation_contract"] != REPAIR_IMPLEMENTATION_CONTRACT
    ):
        raise DerivedStaticReviewContractError(
            "bounded geometry repair schema/implementation contract changed"
        )
    _text(manifest["created_at"], "bounded repair created_at")
    if manifest["formal_dataset_registration_authorized"] is not False:
        raise DerivedStaticReviewContractError(
            "bounded repair cannot authorize formal registration"
        )

    lineage = _exact(
        manifest["lineage"],
        {
            "instance_id",
            "approved_reference",
            "owner_review",
            "owner_review_decision",
            "owner_single_tail_gate",
            "pixal_manifest",
            "pixal_source",
            "static_decision",
            "static_decision_state",
            "raw_four_limbs_usable",
            "raw_pose_riggable",
        },
        "bounded repair lineage",
    )
    _identifier(lineage["instance_id"], "bounded repair lineage.instance_id")
    for name in (
        "approved_reference",
        "owner_review",
        "pixal_manifest",
        "pixal_source",
        "static_decision",
    ):
        _file_record(
            lineage[name],
            f"bounded repair lineage.{name}",
            absolute=True,
        )
    if (
        lineage["owner_review_decision"] != "approved_for_pixal3d"
        or lineage["owner_single_tail_gate"] != "passed"
        or lineage["static_decision_state"] != "rejected"
        or lineage["raw_four_limbs_usable"] is not False
        or lineage["raw_pose_riggable"] is not False
    ):
        raise DerivedStaticReviewContractError(
            "bounded repair lineage changed the frozen raw decisions"
        )

    repair_spec = _exact(
        manifest["repair_spec"],
        {
            "source_side",
            "head_direction",
            "front_foot_x_fraction",
            "front_attachment_x_fraction",
            "hind_foot_x_fraction",
            "hind_attachment_x_fraction",
            "attachment_height_fraction",
            "foot_half_width_fraction",
            "attachment_half_width_fraction",
            "source_side_guard_fraction",
            "central_attachment_bridge_start_fraction",
            "mirrored_attachment_taper_start_fraction",
            "mirrored_attachment_top_lateral_scale",
            "tail_protection_x_fraction",
            "tail_protection_height_fraction",
            "low_slice_height_fraction",
        },
        "bounded repair spec",
    )
    if repair_spec["source_side"] not in {"positive-y", "negative-y"}:
        raise DerivedStaticReviewContractError("bounded repair source side changed")
    if repair_spec["head_direction"] not in {"positive-x", "negative-x"}:
        raise DerivedStaticReviewContractError("bounded repair head direction changed")

    source_pbr = _exact(
        manifest["source_pbr_contract"],
        {"payload_sha256", "embedded_image_bytes_compared"},
        "bounded repair source PBR contract",
    )
    _sha256(source_pbr["payload_sha256"], "bounded repair source PBR payload")
    if source_pbr["embedded_image_bytes_compared"] is not True:
        raise DerivedStaticReviewContractError(
            "bounded repair did not compare embedded source images"
        )

    tail = _mapping(manifest["tail_source_surface"], "bounded repair tail surface")
    if (
        tail.get("method")
        != "most_posterior_authenticated_source_surface_component_height_independent"
        or tail.get("height_used_for_selection") is not False
        or tail.get("passed") is not True
        or tail.get("rejection_reasons") != []
    ):
        raise DerivedStaticReviewContractError(
            "bounded repair lacks a passed height-independent source-tail proof"
        )
    _positive_integer(tail.get("vertex_count"), "bounded repair tail vertex_count")

    mask = _mapping(manifest["mask_audit"], "bounded repair mask audit")
    if (
        mask.get("passed") is not True
        or mask.get("tail_protected_donor_overlap_vertex_count") != 0
        or mask.get("tail_protected_replacement_overlap_vertex_count") != 0
    ):
        raise DerivedStaticReviewContractError(
            "bounded repair mask escaped the authenticated limb corridors"
        )

    face_scope = _exact(
        manifest["face_scope"],
        {
            "source_triangle_count",
            "mutable_triangle_count",
            "donor_triangle_count",
            "immutable_triangle_count",
            "tail_triangle_count",
            "mutation_rule",
        },
        "bounded repair face scope",
    )
    source_triangles = _positive_integer(
        face_scope["source_triangle_count"],
        "bounded repair source_triangle_count",
    )
    mutable_triangles = _positive_integer(
        face_scope["mutable_triangle_count"],
        "bounded repair mutable_triangle_count",
    )
    immutable_triangles = _positive_integer(
        face_scope["immutable_triangle_count"],
        "bounded repair immutable_triangle_count",
    )
    if (
        mutable_triangles + immutable_triangles != source_triangles
        or face_scope["mutation_rule"]
        != "only_triangles_wholly_inside_limb_corridor_and_not_on_authenticated_tail_surface"
    ):
        raise DerivedStaticReviewContractError(
            "bounded repair face-scope accounting changed"
        )

    _closed_topology_audit(
        manifest["source_topology_preflight"],
        "bounded repair source topology preflight",
    )
    mutation = _exact(
        manifest["mutation"],
        {
            "mirrored_geometry_source",
            "external_geometry_inputs",
            "external_skeleton_inputs",
            "external_weight_inputs",
            "external_material_inputs",
            "external_texture_inputs",
            "animation_inputs",
            "tail_geometry_selected_for_mirroring",
            "whole_animal_voxel_remesh",
            "whole_animal_smoothing",
            "whole_animal_decimation",
            "base_edit",
            "donor_edit",
            "mirrored_attachment",
            "local_weld",
        },
        "bounded repair mutation",
    )
    if (
        mutation["mirrored_geometry_source"]
        != "same_authenticated_pixal_mesh_only"
        or mutation["tail_geometry_selected_for_mirroring"] is not False
        or mutation["whole_animal_voxel_remesh"] is not False
        or mutation["whole_animal_smoothing"] is not False
        or mutation["whole_animal_decimation"] is not False
        or any(
            mutation[name] != []
            for name in (
                "external_geometry_inputs",
                "external_skeleton_inputs",
                "external_weight_inputs",
                "external_material_inputs",
                "external_texture_inputs",
                "animation_inputs",
            )
        )
    ):
        raise DerivedStaticReviewContractError(
            "bounded repair mutation widened beyond the local Pixal surface"
        )

    topology = _exact(
        manifest["topology"],
        {
            "raw_blender_after_local_weld",
            "exact_position_logical_after_local_weld",
        },
        "bounded repair output topology",
    )
    _mapping(
        topology["raw_blender_after_local_weld"],
        "bounded repair Blender topology",
    )
    _closed_topology_audit(
        topology["exact_position_logical_after_local_weld"],
        "bounded repair output logical topology",
    )

    low_slice = _mapping(
        manifest["low_slice_dynamic_geometry_gate"],
        "bounded repair low-slice gate",
    )
    low_slice_checks = _mapping(
        low_slice.get("checks"),
        "bounded repair low-slice checks",
    )
    if (
        low_slice.get("method")
        != "four_disconnected_floor_to_attachment_induced_components"
        or low_slice.get("passed") is not True
        or low_slice.get("rejection_reasons") != []
        or not low_slice_checks
        or any(value is not True for value in low_slice_checks.values())
    ):
        raise DerivedStaticReviewContractError(
            "bounded repair low-slice dynamic gate did not pass"
        )

    envelope = _mapping(
        manifest["output_envelope"],
        "bounded repair output envelope",
    )
    if envelope.get("passed") is not True or envelope.get("allowed_range") != [
        -0.06,
        1.06,
    ]:
        raise DerivedStaticReviewContractError(
            "bounded repair output escaped the authenticated source envelope"
        )

    export = _exact(
        manifest["export_readback"],
        {"immutable_outside_corridor_surface", "tail_surface", "pbr"},
        "bounded repair export readback",
    )
    _surface_identity_audit(
        export["immutable_outside_corridor_surface"],
        "bounded repair immutable surface readback",
    )
    _surface_identity_audit(
        export["tail_surface"],
        "bounded repair tail surface readback",
    )
    pbr = _exact(
        export["pbr"],
        {
            "method",
            "source_payload_sha256",
            "output_payload_sha256",
            "embedded_image_sha256s",
            "passed",
        },
        "bounded repair PBR readback",
    )
    source_payload = _sha256(
        pbr["source_payload_sha256"],
        "bounded repair PBR source payload",
    )
    output_payload = _sha256(
        pbr["output_payload_sha256"],
        "bounded repair PBR output payload",
    )
    embedded_images = pbr["embedded_image_sha256s"]
    if not isinstance(embedded_images, list):
        raise DerivedStaticReviewContractError(
            "bounded repair embedded image hashes must be a list"
        )
    for index, digest in enumerate(embedded_images):
        _sha256(digest, f"bounded repair embedded image {index}")
    if (
        pbr["method"]
        != "source_glb_pbr_bindings_and_embedded_bytes_restored_exactly"
        or pbr["passed"] is not True
        or source_payload != output_payload
        or source_payload != source_pbr["payload_sha256"]
    ):
        raise DerivedStaticReviewContractError(
            "bounded repair PBR export readback changed"
        )

    checks = _exact(
        manifest["checks"],
        REPAIR_CHECK_FIELDS,
        "bounded repair checks",
    )
    if any(value is not True for value in checks.values()):
        raise DerivedStaticReviewContractError(
            "all bounded repair checks must pass"
        )
    decision = _exact(
        manifest["decision"],
        {
            "status",
            "rejection_reasons",
            "cat_semantic_retarget_authorized",
            "next_gate",
        },
        "bounded repair decision",
    )
    if (
        decision["status"]
        != "passed_automatic_geometry_gate_pending_multiview_review"
        or decision["rejection_reasons"] != []
        or decision["cat_semantic_retarget_authorized"] is not False
        or decision["next_gate"]
        != "multiview_one_tail_four_limb_no_stray_visual_review"
    ):
        raise DerivedStaticReviewContractError(
            "bounded repair decision is not a passed pending-review result"
        )
    _file_record(manifest["output"], "bounded repair output", absolute=True)
    return copy.deepcopy(dict(manifest))


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
    if geometry["repair_method"] != REPAIR_IMPLEMENTATION_CONTRACT:
        raise DerivedStaticReviewContractError(
            "derived geometry repair implementation contract changed"
        )
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
