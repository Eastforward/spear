from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from tools import audit_quadruped_i23d_geometry as geometry_audit
from tools import build_controlled_animal_derived_static_review as review_builder
from tools import controlled_animal_derived_static_review_contract as review_contract
from tools import generated_animal_tokenrig_closure as tokenrig_closure
from tools import publish_generated_animal_geometry_closure as publisher
from tools import register_controlled_animal_source_assets as source_registry

_SUPPORT_PATH = Path(__file__).with_name("test_repair_pixal_oriented_sheet_indices.py")
_SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "_oriented_repair_test_support",
    _SUPPORT_PATH,
)
assert _SUPPORT_SPEC is not None and _SUPPORT_SPEC.loader is not None
_SUPPORT = importlib.util.module_from_spec(_SUPPORT_SPEC)
_SUPPORT_SPEC.loader.exec_module(_SUPPORT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _audit(path: Path) -> dict[str, object]:
    topology = {
        "imported_vertices": 5,
        "position_unique_vertices": 4,
        "imported_triangles": 4,
        "position_indexed_triangles": 4,
        "degenerate_triangles_after_position_indexing": 0,
        "boundary_edges": 0,
        "manifold_two_face_edges": 6,
        "two_face_orientation_mismatch_edges": 0,
        "nonmanifold_edges_over_two_faces": 0,
        "balanced_oriented_multicover_edges_over_two_faces": 0,
        "unbalanced_edges_over_two_faces": 0,
        "unpaired_oriented_edges": 0,
        "unpaired_oriented_edge_occurrences": 0,
        "paired_oriented_sheet_edge_occurrences": 12,
        "maximum_edge_face_multiplicity": 2,
        "nonmanifold_edge_ratio_per_triangle": 0.0,
        "unpaired_oriented_edge_ratio_per_triangle": 0.0,
        "topology_acceptance_semantics": (
            "every_exact_position_directed_edge_occurrence_has_one_"
            "oppositely_oriented_partner"
        ),
    }
    primary = {
        "coordinate_frame": ("gltf_positive_x_forward_positive_y_up_positive_z_side"),
        "central_longitudinal_percentiles": [25.0, 75.0],
        "torso_floor_fraction_of_robust_height": 0.35,
        "surface_side_percentiles": [5.0, 95.0],
        "section_count": 8,
        "selected_vertex_count": 5,
        "side_slope_per_forward_unit": 0.0,
        "yaw_degrees": 0.0,
        "global_axis_yaw_degrees": 0.0,
        "global_axis_semantics": (
            "rigid_unsigned_orientation_evidence_only_not_a_shape_defect"
        ),
        "centerline_bend_p95_degrees": 0.0,
        "centerline_bend_max_degrees": 0.0,
        "centerline_lateral_rms_ratio": 0.0,
        "centerline_lateral_peak_ratio": 0.0,
        "centerline_curve_degree": 1,
        "centerline_shape_semantics": (
            "local_tangent_deviation_after_removing_rigid_pca_axis"
        ),
        "fit_r_squared": 1.0,
    }
    return {
        "schema": geometry_audit.SCHEMA,
        "created_at": "2026-07-28T00:00:00+00:00",
        "purpose": "prebind_geometry_measurement_without_direction_inference",
        "records": [
            {
                "label": "fixture",
                "mesh": {
                    "absolute_path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                },
                "topology": topology,
                "torso_midline": {
                    "primary": primary,
                    "sensitivity_yaw_degrees": [0.0, 0.0, 0.0],
                    "sensitivity_global_axis_yaw_degrees": [0.0, 0.0, 0.0],
                    "sensitivity_centerline_bend_p95_degrees": [0.0, 0.0, 0.0],
                    "sensitivity_central_percentiles": [
                        [25.0, 75.0],
                        [30.0, 70.0],
                        [35.0, 65.0],
                    ],
                },
                "decision": geometry_audit.decision(topology, primary),
            }
        ],
    }


def _fixture(
    tmp_path: Path,
    *,
    degenerate: bool = False,
) -> dict[str, object]:
    repair_root = tmp_path / "repair"
    repair_root.mkdir()
    raw, repaired, repair_manifest, _ = _SUPPORT._run(
        repair_root,
        degenerate=degenerate,
    )
    pixal_manifest = repair_root / "pixal.manifest.json"
    decision_batch = (
        repair_root / "static_decision" / "static_decision_batch_manifest.json"
    )
    decision = (
        repair_root / "static_decision" / _SUPPORT.INSTANCE_ID / "static_decision.json"
    )
    reference = repair_root / "static_review" / "reference_rgba.png"

    evidence_root = tmp_path / "geometry_evidence"
    audit_path = _write_json(
        evidence_root / "geometry_audit.json",
        _audit(repaired),
    )
    views_root = evidence_root / "clay" / "views"
    views_root.mkdir(parents=True)
    views: dict[str, Path] = {}
    cameras = {
        "front": [-2.0, 0.0, 0.0],
        "back": [2.0, 0.0, 0.0],
        "side": [0.0, 0.0, 2.0],
        "top": [0.0, 2.0, 0.0],
        "quarter": [-2.0, 1.0, 2.0],
    }
    for index, name in enumerate(publisher.VIEWS):
        view = views_root / f"{name}.png"
        Image.new(
            "RGB",
            (480, 480),
            (40 + index * 20, 60, 80),
        ).save(view)
        views[name] = view
    render_manifest = _write_json(
        views_root / "render_manifest.json",
        {
            "input": str(repaired.resolve()),
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [1.0, 1.0, 1.0],
            "extent": [1.0, 1.0, 1.0],
            "front_axis": "negative-x",
            "views": cameras,
            "resolution": [480, 480],
            "samples": 16,
            "material_preview": {
                "mode": "neutral_clay_geometry_qa_v1",
                "principled_nodes_changed": 0,
                "metallic_links_removed": 0,
                "roughness_links_removed": 0,
            },
            "lighting": {
                "area_light_scale": 1.0,
                "world_strength": 0.5,
                "exposure": 0.0,
            },
        },
    )
    contact = evidence_root / "clay" / "contact_sheet.png"
    Image.new("RGB", (960, 640), (30, 30, 30)).save(contact)
    output = tmp_path / "geometry_closure.json"
    arguments = {
        "instance_id": _SUPPORT.INSTANCE_ID,
        "raw_pixal_glb_path": raw,
        "expected_raw_pixal_glb_sha256": _sha256(raw),
        "pixal_manifest_path": pixal_manifest,
        "expected_pixal_manifest_sha256": _sha256(pixal_manifest),
        "source_reference_path": reference,
        "expected_source_reference_sha256": _sha256(reference),
        "raw_static_decision_batch_path": decision_batch,
        "expected_raw_static_decision_batch_sha256": _sha256(decision_batch),
        "raw_static_decision_path": decision,
        "expected_raw_static_decision_sha256": _sha256(decision),
        "repair_manifest_path": repair_manifest,
        "expected_repair_manifest_sha256": _sha256(repair_manifest),
        "repaired_glb_path": repaired,
        "expected_repaired_glb_sha256": _sha256(repaired),
        "geometry_audit_path": audit_path,
        "expected_geometry_audit_sha256": _sha256(audit_path),
        "clay_render_manifest_path": render_manifest,
        "expected_clay_render_manifest_sha256": _sha256(render_manifest),
        "clay_views": views,
        "expected_clay_view_sha256s": {
            name: _sha256(path) for name, path in views.items()
        },
        "clay_contact_sheet_path": contact,
        "expected_clay_contact_sheet_sha256": _sha256(contact),
        "output_path": output,
    }
    return {
        "arguments": arguments,
        "raw": raw,
        "repaired": repaired,
        "repair_manifest": repair_manifest,
        "pixal_manifest": pixal_manifest,
        "decision_batch": decision_batch,
        "decision": decision,
        "reference": reference,
        "audit": audit_path,
        "views": views,
        "output": output,
    }


def _publish(case: dict[str, object]) -> Path:
    return publisher.publish_geometry_closure(**case["arguments"])


def _load_with_all_bindings(case: dict[str, object], path: Path) -> dict[str, object]:
    return publisher.load_geometry_closure_v2(
        path,
        expected_manifest_sha256=_sha256(path),
        expected_instance_id=_SUPPORT.INSTANCE_ID,
        expected_raw_pixal_glb=case["raw"],
        expected_pixal_manifest=case["pixal_manifest"],
        expected_source_reference=case["reference"],
        expected_raw_static_decision_batch=case["decision_batch"],
        expected_raw_static_decision=case["decision"],
        expected_repair_manifest=case["repair_manifest"],
        expected_repaired_glb=case["repaired"],
        expected_geometry_audit=case["audit"],
    )


def test_publishes_generic_strict_geometry_closure_v2(tmp_path):
    case = _fixture(tmp_path)
    output = _publish(case)
    replay = _load_with_all_bindings(case, output)
    manifest = replay["manifest"]

    assert manifest["schema"] == publisher.SCHEMA
    assert "owner_approved_flux_reference" not in json.dumps(manifest)
    assert manifest["state_classification"] == "research_candidate"
    assert manifest["formal_dataset_registration_authorized"] is False
    assert manifest["inherited_static_judgment"]["new_human_approval_created"] is False
    assert manifest["clay_readback"]["human_approval_claimed"] is False
    assert manifest["bounded_repair"]["output_byte_identical_to_raw"] is True
    assert manifest["downstream"] == {
        "tokenrig_entry_authorized": True,
        "tokenrig_execution_performed": False,
        "quaternius_rig_swap_authorized": False,
        "animation_execution_performed": False,
        "ue_import_executed": False,
        "native_change_executed": False,
        "emitter_measurement_executed": False,
        "formal_dataset_registration_authorized": False,
    }


def test_dispatch_reader_retains_explicit_legacy_v1_path(tmp_path):
    legacy = _write_json(
        tmp_path / "legacy-v1.json",
        {
            "schema": publisher.LEGACY_SCHEMA,
            "status": "pass_geometry_only",
        },
    )

    schema, payload = publisher.load_geometry_closure(
        legacy,
        expected_manifest_sha256=_sha256(legacy),
    )

    assert schema == publisher.LEGACY_SCHEMA
    assert payload["status"] == "pass_geometry_only"


def test_publishes_only_the_bounded_zero_area_filter_branch(tmp_path):
    case = _fixture(tmp_path, degenerate=True)
    output = _publish(case)
    manifest = _load_with_all_bindings(case, output)["manifest"]

    assert (
        manifest["bounded_repair"]["removed_exact_position_degenerate_triangle_count"]
        == 1
    )
    assert manifest["bounded_repair"]["output_byte_identical_to_raw"] is False
    assert manifest["bounded_repair"]["mutation_class"] == publisher.MUTATION_CLASS


def test_derived_review_consumer_strictly_replays_v2(tmp_path):
    case = _fixture(tmp_path)
    output = _publish(case)

    geometry = review_builder._validate_geometry_closure(
        closure_path=output,
        repaired_glb=case["repaired"],
        source={
            "request": {"instance_id": _SUPPORT.INSTANCE_ID},
            "raw_glb": case["raw"],
            "raw_attempt_manifest": case["pixal_manifest"],
            "pixal_input": case["reference"],
            "decision_batch_path": case["decision_batch"],
            "decision_path": case["decision"],
        },
    )

    assert (
        geometry["automatic_statuses"]
        == review_contract.ORIENTED_V2_AUTOMATIC_GATE_STATUSES
    )
    assert (
        geometry["inherited_statuses"]
        == review_contract.ORIENTED_V2_INHERITED_MANUAL_REVIEW_STATUSES
    )
    assert geometry["repair_manifest_path"] == case["repair_manifest"].resolve()
    assert geometry["geometry_audit_path"] == case["audit"].resolve()


def test_registry_consumer_strictly_replays_v2_without_new_approval(tmp_path):
    case = _fixture(tmp_path)
    output = _publish(case)
    review = {
        "instance_identity": {"instance_id": _SUPPORT.INSTANCE_ID},
        "derived_geometry": {
            "repair_method": review_contract.ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT,
            "automatic_gate_statuses": dict(
                review_contract.ORIENTED_V2_AUTOMATIC_GATE_STATUSES
            ),
            "inherited_manual_review_statuses": dict(
                review_contract.ORIENTED_V2_INHERITED_MANUAL_REVIEW_STATUSES
            ),
        },
    }

    source_registry._reauthenticate_bounded_derived_geometry(
        review=review,
        raw_pixal_path=case["raw"],
        reviewed_reference_path=case["reference"],
        raw_decision_path=case["decision"],
        raw_attempt_manifest_path=case["pixal_manifest"],
        repaired_glb_path=case["repaired"],
        geometry_closure_path=output,
        repair_manifest_path=case["repair_manifest"],
        geometry_audit_path=case["audit"],
        raw_decision_batch_path=case["decision_batch"],
        pixal_input_path=case["reference"],
    )


def test_tokenrig_consumer_strictly_replays_v2_with_external_batch(tmp_path):
    case = _fixture(tmp_path)
    output = _publish(case)

    raw, kind, extra = tokenrig_closure.validate_upstream_lineage(
        case["pixal_manifest"],
        output,
        case["repaired"],
        raw_static_decision_batch_path=case["decision_batch"],
        expected_raw_static_decision_batch_sha256=_sha256(case["decision_batch"]),
        require_oriented_batch_external_authority=True,
    )

    assert raw == case["raw"].resolve()
    assert kind == "bounded_geometry_closure"
    assert extra == [
        case["repair_manifest"].resolve(),
        case["audit"].resolve(),
        case["decision_batch"].resolve(),
    ]


def test_publish_is_create_only_and_preserves_existing_output(tmp_path):
    case = _fixture(tmp_path)
    output = _publish(case)
    original = output.read_bytes()

    with pytest.raises(publisher.GeometryClosureError, match="refusing to replace"):
        _publish(case)

    assert output.read_bytes() == original


def test_publish_rejects_leaf_symlink_reference(tmp_path):
    case = _fixture(tmp_path)
    reference_link = tmp_path / "reference-link.png"
    reference_link.symlink_to(case["reference"])
    arguments = dict(case["arguments"])
    arguments["source_reference_path"] = reference_link

    with pytest.raises(publisher.GeometryClosureError, match="unsafe symlink"):
        publisher.publish_geometry_closure(**arguments)

    assert not case["output"].exists()


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda payload: payload.__setitem__("unsupported_claim", True), "fields"),
        (
            lambda payload: payload["container_readback"].__setitem__(
                "material_count",
                2,
            ),
            "container readback changed",
        ),
    ),
)
def test_strict_reader_rejects_resealed_unknown_or_readback_claim(
    tmp_path,
    mutate,
    match,
):
    case = _fixture(tmp_path)
    published = _publish(case)
    payload = json.loads(published.read_text(encoding="utf-8"))
    mutate(payload)
    payload["manifest_sha256"] = publisher._hash_without(
        payload,
        "manifest_sha256",
    )
    tampered = _write_json(tmp_path / "resealed.json", payload)

    with pytest.raises(publisher.GeometryClosureError, match=match):
        publisher.load_geometry_closure_v2(
            tampered,
            expected_manifest_sha256=_sha256(tampered),
        )


def test_strict_reader_rejects_same_byte_alternate_raw_path(tmp_path):
    case = _fixture(tmp_path)
    published = _publish(case)
    alternate = tmp_path / "alternate-source.glb"
    alternate.write_bytes(case["raw"].read_bytes())
    payload = json.loads(published.read_text(encoding="utf-8"))
    payload["source_authorities"]["raw_pixal_glb"] = {
        "path": str(alternate.resolve()),
        "sha256": _sha256(alternate),
        "size_bytes": alternate.stat().st_size,
    }
    payload["manifest_sha256"] = publisher._hash_without(
        payload,
        "manifest_sha256",
    )
    resealed = _write_json(tmp_path / "alternate-path.json", payload)

    with pytest.raises(publisher.GeometryClosureError, match="alternate path"):
        _load_with_all_bindings(case, resealed)


def test_strict_reader_rejects_same_byte_alternate_view_path(tmp_path):
    case = _fixture(tmp_path)
    published = _publish(case)
    alternate = tmp_path / "alternate-front.png"
    alternate.write_bytes(case["views"]["front"].read_bytes())
    payload = json.loads(published.read_text(encoding="utf-8"))
    payload["clay_readback"]["views"]["front"] = {
        "path": str(alternate.resolve()),
        "sha256": _sha256(alternate),
        "size_bytes": alternate.stat().st_size,
    }
    payload["manifest_sha256"] = publisher._hash_without(
        payload,
        "manifest_sha256",
    )
    resealed = _write_json(tmp_path / "alternate-view.json", payload)

    with pytest.raises(publisher.GeometryClosureError, match="alternate path"):
        _load_with_all_bindings(case, resealed)


def test_strict_reader_recomputes_resealed_oriented_repair_claims(tmp_path):
    case = _fixture(tmp_path)
    published = _publish(case)
    repair = json.loads(case["repair_manifest"].read_text(encoding="utf-8"))
    repair["source_contract"]["position_accessor_index"] += 1
    resealed_repair = _write_json(tmp_path / "resealed-repair.json", repair)
    payload = json.loads(published.read_text(encoding="utf-8"))
    payload["bounded_repair"]["manifest"] = {
        "path": str(resealed_repair.resolve()),
        "sha256": _sha256(resealed_repair),
        "size_bytes": resealed_repair.stat().st_size,
    }
    payload["manifest_sha256"] = publisher._hash_without(
        payload,
        "manifest_sha256",
    )
    resealed = _write_json(tmp_path / "resealed-repair-closure.json", payload)

    with pytest.raises(
        publisher.GeometryClosureError,
        match="does not replay from the actual GLB bytes",
    ):
        publisher.load_geometry_closure_v2(
            resealed,
            expected_manifest_sha256=_sha256(resealed),
        )
