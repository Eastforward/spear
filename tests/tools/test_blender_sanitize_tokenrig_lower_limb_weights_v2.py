"""CPU contracts for generic canonical-world lower-limb sanitation v2."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

import pytest

from tools import blender_sanitize_tokenrig_lower_limb_weights_v2 as sanitizer


CHAINS = {
    "axial": ["root", "spine_a", "spine_b", "head"],
    "left_arm": ["l_clavicle", "l_upper_arm", "l_forearm", "l_hand"],
    "right_arm": ["r_clavicle", "r_upper_arm", "r_forearm", "r_hand"],
    "left_leg": ["l_thigh", "l_calf", "l_foot", "l_toe"],
    "right_leg": ["r_thigh", "r_calf", "r_foot", "r_toe"],
}

SEMANTIC = {
    "method": "fixture_semantics_v1",
    "chains": CHAINS,
    "semantic_bones": {
        "pelvis": "root",
        "spine": ["spine_a", "spine_b"],
        "neck": "spine_b",
        "head": "head",
        "left_thigh": "l_thigh",
        "left_calf": "l_calf",
        "left_foot": "l_foot",
        "left_toe": "l_toe",
        "right_thigh": "r_thigh",
        "right_calf": "r_calf",
        "right_foot": "r_foot",
        "right_toe": "r_toe",
    },
    "side_basis": {"left": "positive-x", "right": "negative-x"},
}


def _problem_fixture():
    positions = (
        (0.12, 0.0, 0.01),
        (-0.12, 0.0, 0.01),
        (0.079197824, 0.0019, 0.0129),
        (0.079197824, 0.0019, 0.0129),  # UV seam duplicate
        (-0.079197824, 0.0019, 0.0129),
        (0.0, 0.0, 0.8),
        (0.001, 0.0, 0.1),
        (1.0, 0.0, 1.4),  # arm width must not affect the lower-body gate
        (-1.0, 0.0, 1.4),
    )
    weights = (
        {"l_toe": 1.0},
        {"r_toe": 1.0},
        {"l_foot": 0.00027, "l_toe": 0.9106, "r_toe": 0.08913},
        {"l_foot": 0.00027, "l_toe": 0.9106, "r_toe": 0.08913},
        {"r_foot": 0.00027, "r_toe": 0.9106, "l_toe": 0.08913},
        {"root": 0.6, "l_thigh": 0.2, "r_thigh": 0.2},
        {"l_toe": 0.6, "r_toe": 0.4},
        {"l_hand": 1.0},
        {"r_hand": 1.0},
    )
    return positions, weights


def _run_problem():
    positions, weights = _problem_fixture()
    return (
        positions,
        weights,
        *sanitizer.sanitize_lower_limb_weight_maps(
            canonical_world_positions=positions,
            vertex_weights=weights,
            chains=CHAINS,
        ),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_readonly(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o444)
    return path


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
        "mode": "0444",
    }


def test_near_midline_shoe_vertex_moves_opposite_toe_mass_on_canonical_side():
    _, _, sanitized, report, changes = _run_problem()

    assert sanitized[2] == pytest.approx({"l_foot": 0.00027, "l_toe": 0.99973})
    assert sanitized[3] == sanitized[2]
    assert sanitized[4] == pytest.approx({"r_foot": 0.00027, "r_toe": 0.99973})
    assert report["gate"]["coordinate_space"] == "canonical_world"
    assert report["gate"]["transition_half_width_m"] < 0.079197824
    assert report["maximum_opposite_leg_mass_before"] == pytest.approx(0.08913)
    assert report["maximum_opposite_leg_mass_after"] == 0.0
    assert report["transferred_mass_by_bone_pair"] == pytest.approx(
        {"l_toe->r_toe": 0.08913, "r_toe->l_toe": 0.17826}
    )
    assert {record["vertex_index"] for record in changes} == {2, 3, 4}
    assert all(len(weights) <= 4 for weights in sanitized)
    assert report["seam_validation"]["maximum_weight_l1_error"] == 0.0


def test_report_records_before_and_after_contamination_by_side_region_bone_and_position():
    _, _, _, report, _ = _run_problem()

    before = report["contamination_statistics"]["before"]
    after = report["contamination_statistics"]["after"]
    assert before["contaminated_vertex_count"] == 3
    assert before["total_opposite_leg_mass"] == pytest.approx(0.26739)
    assert before["by_side"]["left"]["contaminated_vertex_count"] == 2
    assert before["by_side"]["right"]["contaminated_vertex_count"] == 1
    assert before["by_region"]["toe"]["contaminated_vertex_count"] == 3
    assert before["by_region"]["toe"]["total_opposite_leg_mass"] == pytest.approx(
        0.26739
    )
    assert before["by_bone"]["r_toe"]["total_opposite_leg_mass"] == pytest.approx(
        0.17826
    )
    assert before["by_bone"]["l_toe"]["total_opposite_leg_mass"] == pytest.approx(
        0.08913
    )
    assert before["contaminated_position_bounds_m"] == {
        "minimum": pytest.approx([-0.079197824, 0.0019, 0.0129]),
        "maximum": pytest.approx([0.079197824, 0.0019, 0.0129]),
    }
    assert after["contaminated_vertex_count"] == 0
    assert after["total_opposite_leg_mass"] == 0.0
    assert after["contaminated_position_bounds_m"] is None


def test_extracted_vertex_positions_are_kept_in_world_space_without_double_transform():
    class Rotation180:
        def __matmul__(self, point):
            return (-float(point[0]), -float(point[1]), float(point[2]))

    class Vertex:
        def __init__(self, coordinates):
            self.co = coordinates

    class Data:
        vertices = (Vertex((1.0, 2.0, 3.0)), Vertex((-4.0, 5.0, 6.0)))

    class Mesh:
        matrix_world = Rotation180()
        data = Data()

    extracted_world = ((-1.0, -2.0, 3.0), (4.0, -5.0, 6.0))
    assert sanitizer.validate_extracted_world_positions(Mesh(), extracted_world) == (
        extracted_world
    )

    with pytest.raises(sanitizer.LowerLimbSanitationError, match="not world-space"):
        sanitizer.validate_extracted_world_positions(
            Mesh(), ((1.0, 2.0, 3.0), (-4.0, 5.0, 6.0))
        )


def test_center_pelvis_and_true_center_leg_transition_are_unchanged():
    _, original, sanitized, report, changes = _run_problem()

    assert sanitized[5] == original[5]
    assert sanitized[6] == original[6]
    assert {record["vertex_index"] for record in changes}.isdisjoint({5, 6})
    assert report["preserved_center_transition_vertex_count"] >= 1
    assert report["maximum_non_leg_weight_error"] == 0.0


def test_mirror_symmetry_uses_semantic_pairs_not_bone_name_patterns():
    _, _, sanitized, _, _ = _run_problem()

    assert sanitized[2]["l_toe"] == pytest.approx(sanitized[4]["r_toe"])
    assert sanitized[2]["l_foot"] == pytest.approx(sanitized[4]["r_foot"])
    transfer = sanitizer.build_leg_transfer_maps(CHAINS)
    assert transfer["pairs"] == [
        ["l_thigh", "r_thigh"],
        ["l_calf", "r_calf"],
        ["l_foot", "r_foot"],
        ["l_toe", "r_toe"],
    ]
    assert transfer["to_left"]["r_toe"] == "l_toe"
    assert transfer["to_right"]["l_toe"] == "r_toe"


def test_sanitation_is_idempotent_and_deterministic():
    positions, _, first, first_report, first_changes = _run_problem()
    second, second_report, second_changes = sanitizer.sanitize_lower_limb_weight_maps(
        canonical_world_positions=positions,
        vertex_weights=first,
        chains=CHAINS,
    )

    assert second == first
    assert second_changes == ()
    assert second_report["changed_vertex_count"] == 0
    assert first_report["idempotence"] == {
        "passed": True,
        "second_pass_changed_vertex_count": 0,
    }
    assert [record["vertex_index"] for record in first_changes] == sorted(
        record["vertex_index"] for record in first_changes
    )


def test_lower_body_gate_ignores_full_arm_width_and_requires_both_leg_sides():
    positions, weights = _problem_fixture()
    gate = sanitizer.derive_lower_body_gate(
        canonical_world_positions=positions,
        vertex_weights=weights,
        chains=CHAINS,
    )

    assert gate["positive_extent_m"] <= 0.12
    assert gate["negative_extent_m"] <= 0.12
    assert gate["symmetric_half_width_m"] < 0.13
    assert gate["transition_half_width_m"] < 0.01

    one_sided = tuple({"l_toe": 1.0} for _ in positions)
    with pytest.raises(sanitizer.LowerLimbSanitationError, match="both"):
        sanitizer.derive_lower_body_gate(
            canonical_world_positions=positions,
            vertex_weights=one_sided,
            chains=CHAINS,
        )


def test_fail_closed_for_bad_chains_weights_seams_and_nonfinite_positions():
    positions, weights = _problem_fixture()
    with pytest.raises(sanitizer.LowerLimbSanitationError, match="disjoint"):
        sanitizer.sanitize_lower_limb_weight_maps(
            canonical_world_positions=positions,
            vertex_weights=weights,
            chains={**CHAINS, "right_leg": CHAINS["left_leg"]},
        )
    bad_weights = list(weights)
    bad_weights[2] = {"l_toe": 0.4}
    with pytest.raises(sanitizer.LowerLimbSanitationError, match="input skin"):
        sanitizer.sanitize_lower_limb_weight_maps(
            canonical_world_positions=positions,
            vertex_weights=bad_weights,
            chains=CHAINS,
        )
    bad_seam = list(weights)
    bad_seam[3] = {"l_toe": 1.0}
    with pytest.raises(sanitizer.LowerLimbSanitationError, match="seam"):
        sanitizer.sanitize_lower_limb_weight_maps(
            canonical_world_positions=positions,
            vertex_weights=bad_seam,
            chains=CHAINS,
        )
    bad_positions = list(positions)
    bad_positions[0] = (math.nan, 0.0, 0.0)
    with pytest.raises(sanitizer.LowerLimbSanitationError, match="finite"):
        sanitizer.derive_lower_body_gate(
            canonical_world_positions=bad_positions,
            vertex_weights=weights,
            chains=CHAINS,
        )


def test_change_jsonl_is_sorted_complete_and_empty_safe():
    _, _, _, _, changes = _run_problem()

    payload = sanitizer.serialize_change_records(tuple(reversed(changes)))
    decoded = [json.loads(line) for line in payload.decode().splitlines()]
    assert [row["vertex_index"] for row in decoded] == [2, 3, 4]
    assert all(
        set(row)
        == {
            "vertex_index",
            "canonical_world_position_m",
            "classified_side",
            "leg_mass",
            "opposite_leg_mass_before",
            "opposite_leg_mass_after",
            "transferred_mass",
            "transferred_mass_by_bone_pair",
            "before",
            "after",
            "l1_before_after",
        }
        for row in decoded
    )
    assert sanitizer.serialize_change_records(()) == b""


def test_surface_corner_skin_gate_matches_every_corner_without_relaxing_tolerance():
    Surface = sanitizer.static_audit.SurfaceReference
    expected = Surface(
        polygon_loop_counts=[3],
        polygon_material_indices=[0],
        corner_unique_indices=[0, 1, 2],
        corner_positions=[0.0] * 9,
        corner_normals=[0.0] * 9,
        uv_layers=(),
        unique_positions=[0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
        bounds=((0.0, 0.0, 0.0), (0.1, 0.1, 0.0)),
        surface_area_m2=0.005,
    )
    actual = copy.deepcopy(expected)
    actual = Surface(
        **{
            **actual.__dict__,
            "unique_positions": [
                0.2e-6,
                0.0,
                0.0,
                0.1,
                0.2e-6,
                0.0,
                0.0,
                0.1,
                0.2e-6,
            ],
        }
    )
    expected_weights = ({"l_toe": 1.0}, {"l_toe": 1.0}, {"root": 1.0})
    actual_weights = (
        {"l_toe": 1.0},
        {"l_toe": 1.0},
        {"root": 0.9999999, "l_toe": 0.0000001},
    )

    result = sanitizer.compare_surface_corner_skin_weights(
        expected_surface=expected,
        expected_corner_vertex_indices=[0, 1, 2],
        expected_weights=expected_weights,
        actual_surface=actual,
        actual_corner_vertex_indices=[0, 1, 2],
        actual_weights=actual_weights,
    )

    assert result["polygon_corner_count"] == result["matched_polygon_corner_count"] == 3
    assert result["missing_representative_count"] == 0
    assert result["nearest_position_error_m"]["maximum"] < 2.0e-6
    assert result["weight_l1_error"]["maximum"] < 1.0e-6
    bad = list(actual_weights)
    bad[2] = {"root": 0.9, "l_toe": 0.1}
    with pytest.raises(sanitizer.LowerLimbSanitationError, match="weights changed"):
        sanitizer.compare_surface_corner_skin_weights(
            expected_surface=expected,
            expected_corner_vertex_indices=[0, 1, 2],
            expected_weights=expected_weights,
            actual_surface=actual,
            actual_corner_vertex_indices=[0, 1, 2],
            actual_weights=bad,
        )


def test_authentication_binds_glb_producer_manifest_and_static_semantics(
    tmp_path, monkeypatch
):
    glb = _write_readonly(tmp_path / "bind_pose.glb", b"glTFfixture")
    producer_payload = {
        "schema": "fixture_producer_v1",
        "asset_id": "person_female_01",
        "input": {
            "source_glb": {
                "path": str(glb.resolve()),
                "sha256": _sha(glb),
                "size_bytes": glb.stat().st_size,
            }
        },
        "output": {
            "path": str(glb.resolve()),
            "sha256": _sha(glb),
            "size_bytes": glb.stat().st_size,
        },
    }
    producer = _write_readonly(
        tmp_path / "producer.json",
        (json.dumps(producer_payload) + "\n").encode(),
    )
    static_payload = {
        "schema": "tokenrig_human_static_qa_v1",
        "asset_id": "person_female_01",
        "readiness_bundle_published": True,
        "decision": "automatic_static_checks_passed",
        "authenticated": {
            "tokenrig_manifest_sha256": _sha(producer),
            "tokenrig_glb_sha256": _sha(glb),
            "source_glb_sha256": _sha(glb),
        },
        "artifacts": {
            "bind_pose.glb": {
                "filename": glb.name,
                "sha256": _sha(glb),
                "size_bytes": glb.stat().st_size,
            }
        },
        "checks": {
            "automatic_static_checks": "passed",
            "semantic_mapping": SEMANTIC,
        },
    }
    static_qa = _write_readonly(
        tmp_path / "static_qa.json",
        (json.dumps(static_payload) + "\n").encode(),
    )
    monkeypatch.setattr(sanitizer.static_audit, "read_glb", lambda path: object())
    monkeypatch.setattr(
        sanitizer.static_audit,
        "extract_inverse_bind_contract",
        lambda parsed: {"joint_names": [f"joint_{index}" for index in range(19)]},
    )
    monkeypatch.setattr(
        sanitizer.static_audit,
        "pbr_payload_contract",
        lambda parsed: {"passed": True},
    )
    monkeypatch.setattr(
        sanitizer.static_audit,
        "compare_pbr_payloads",
        lambda first, second: {"passed": True},
    )

    result = sanitizer.authenticate_inputs(
        mode="passed_static_owner",
        asset_id="person_female_01",
        source_glb=glb,
        input_glb=glb,
        input_manifest=producer,
        static_qa_json=static_qa,
        prior_failures=(),
    )
    assert result["semantic_evidence"]["chains"]["left_leg"][-1] == "l_toe"
    assert result["input_glb"]["sha256"] == _sha(glb)
    assert result["input_manifest"]["sha256"] == _sha(producer)

    static_qa.chmod(0o644)
    with pytest.raises(sanitizer.LowerLimbSanitationError, match="mode 0444"):
        sanitizer.authenticate_inputs(
            mode="passed_static_owner",
            asset_id="person_female_01",
            source_glb=glb,
            input_glb=glb,
            input_manifest=producer,
            static_qa_json=static_qa,
            prior_failures=(),
        )


def test_pre_static_repair_authenticates_source_output_and_immutable_failure_chain(
    tmp_path, monkeypatch
):
    source = _write_readonly(tmp_path / "source.glb", b"glTFsource")
    tokenrig = _write_readonly(tmp_path / "tokenrig_transfer.glb", b"glTFtokenrig")
    failure_payload = {
        "schema": "tokenrig_human_static_failure_v1",
        "asset_id": "person_female_01",
        "decision": "rejected",
        "readiness_bundle_published": False,
        "failure": {
            "type": "StaticAuditError",
            "message": "opposite-limb contamination on distal vertices",
        },
    }
    failure = _write_readonly(
        tmp_path / "static_audit_v1.failed.fixture.json",
        (json.dumps(failure_payload) + "\n").encode(),
    )
    producer_payload = {
        "schema": "pixal_tokenrig_fitted_skeleton_v1",
        "asset_id": "person_female_01",
        "input": {
            "original_source_glb": {
                "path": str(source.resolve()),
                "sha256": _sha(source),
                "size_bytes": source.stat().st_size,
            },
            "static_failures": [
                {
                    "path": str(failure.resolve()),
                    "sha256": _sha(failure),
                    "size_bytes": failure.stat().st_size,
                }
            ],
        },
        "output": {
            "path": str(tokenrig.resolve()),
            "sha256": _sha(tokenrig),
            "size_bytes": tokenrig.stat().st_size,
        },
    }
    producer = _write_readonly(
        tmp_path / "tokenrig_manifest.json",
        (json.dumps(producer_payload) + "\n").encode(),
    )
    monkeypatch.setattr(sanitizer.static_audit, "read_glb", lambda path: object())
    monkeypatch.setattr(
        sanitizer.static_audit,
        "extract_inverse_bind_contract",
        lambda parsed: {"joint_names": [f"joint_{index}" for index in range(19)]},
    )
    monkeypatch.setattr(
        sanitizer.static_audit,
        "pbr_payload_contract",
        lambda parsed: {"fixture": True},
    )
    monkeypatch.setattr(
        sanitizer.static_audit,
        "compare_pbr_payloads",
        lambda first, second: {"passed": True},
    )

    result = sanitizer.authenticate_inputs(
        mode="pre_static_repair",
        asset_id="person_female_01",
        source_glb=source,
        input_glb=tokenrig,
        input_manifest=producer,
        static_qa_json=None,
        prior_failures=(failure,),
    )

    assert result["mode"] == "pre_static_repair"
    assert result["semantic_evidence"] is None
    assert result["static_qa"] is None
    assert result["prior_failures"] == [_record(failure)]
    assert result["input_pbr"] == {"passed": True}

    with pytest.raises(sanitizer.LowerLimbSanitationError, match="requires failures"):
        sanitizer.authenticate_inputs(
            mode="pre_static_repair",
            asset_id="person_female_01",
            source_glb=source,
            input_glb=tokenrig,
            input_manifest=producer,
            static_qa_json=None,
            prior_failures=(),
        )


def _passed_validation():
    passed = {"passed": True}
    return {
        "in_scene_mesh": passed,
        "in_scene_surface": passed,
        "in_scene_rest": passed,
        "in_scene_full_rest": passed,
        "inverse_bind": passed,
        "output_pbr": passed,
        "roundtrip": passed,
        "roundtrip_full_rest": passed,
        "removed_gltf_import_helpers": [],
        "removed_proven_orphans": [],
    }


def test_immutable_manifest_reauthenticates_inputs_code_output_and_change_log(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "sanitized_weights_v2"
    input_dir.mkdir()
    output_dir.mkdir()
    input_glb = _write_readonly(input_dir / "bind_pose.glb", b"glTFinput")
    producer = _write_readonly(input_dir / "producer.json", b'{"fixture":true}\n')
    static_qa = _write_readonly(input_dir / "static_qa.json", b'{"fixture":true}\n')
    output_glb = _write_readonly(output_dir / sanitizer.OUTPUT_GLB_NAME, b"glTFoutput")
    change = {
        "vertex_index": 7,
        "canonical_world_position_m": [0.08, 0.0, 0.01],
        "classified_side": "left",
        "leg_mass": 1.0,
        "opposite_leg_mass_before": 0.1,
        "opposite_leg_mass_after": 0.0,
        "transferred_mass": 0.1,
        "transferred_mass_by_bone_pair": {"r_toe->l_toe": 0.1},
        "before": {"l_toe": 0.9, "r_toe": 0.1},
        "after": {"l_toe": 1.0},
        "l1_before_after": 0.2,
    }
    change_log = _write_readonly(
        output_dir / sanitizer.CHANGE_LOG_NAME,
        (json.dumps(change, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    diagnostic = _write_readonly(
        output_dir / sanitizer.ROUNDTRIP_DIAGNOSTIC_NAME,
        b'{"schema":"tokenrig_surface_skin_roundtrip_diagnostic_v2","status":"passed"}\n',
    )
    authenticated = {
        "mode": "passed_static_owner",
        "source_glb": _record(input_glb),
        "input_glb": _record(input_glb),
        "input_manifest": _record(producer),
        "static_qa": _record(static_qa),
        "prior_failures": [],
        "semantic_evidence": SEMANTIC,
    }
    report = {
        "algorithm_version": sanitizer.ALGORITHM_VERSION,
        "idempotence": {"passed": True, "second_pass_changed_vertex_count": 0},
        "maximum_opposite_leg_mass_after": 0.0,
        "maximum_influences": 4,
        "changed_vertex_count": 1,
    }
    payload = sanitizer._manifest_payload(
        asset_id="person_male_01",
        authenticated=authenticated,
        code=sanitizer._code_snapshot(),
        report=report,
        validation=_passed_validation(),
        output_record=_record(output_glb),
        change_record=_record(change_log),
        diagnostic_record=_record(diagnostic),
    )
    manifest = _write_readonly(
        output_dir / sanitizer.OUTPUT_MANIFEST_NAME,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )

    assert sanitizer.validate_published_manifest(manifest)["asset_id"] == (
        "person_male_01"
    )
    output_glb.chmod(0o644)
    output_glb.write_bytes(b"glTFtampered")
    output_glb.chmod(0o444)
    with pytest.raises(sanitizer.LowerLimbSanitationError, match="stale"):
        sanitizer.validate_published_manifest(manifest)


def test_manifest_rejects_missing_change_rows_and_user_approval(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    inputs = [
        _write_readonly(input_dir / name, b"fixture")
        for name in ("bind_pose.glb", "producer.json", "static_qa.json")
    ]
    output_glb = _write_readonly(output_dir / sanitizer.OUTPUT_GLB_NAME, b"glTFoutput")
    empty_log = _write_readonly(output_dir / sanitizer.CHANGE_LOG_NAME, b"")
    diagnostic = _write_readonly(
        output_dir / sanitizer.ROUNDTRIP_DIAGNOSTIC_NAME,
        b'{"schema":"tokenrig_surface_skin_roundtrip_diagnostic_v2","status":"passed"}\n',
    )
    payload = sanitizer._manifest_payload(
        asset_id="person_male_01",
        authenticated={
            "mode": "passed_static_owner",
            "source_glb": _record(inputs[0]),
            "input_glb": _record(inputs[0]),
            "input_manifest": _record(inputs[1]),
            "static_qa": _record(inputs[2]),
            "prior_failures": [],
            "semantic_evidence": SEMANTIC,
        },
        code=sanitizer._code_snapshot(),
        report={
            "algorithm_version": sanitizer.ALGORITHM_VERSION,
            "idempotence": {"passed": True, "second_pass_changed_vertex_count": 0},
            "maximum_opposite_leg_mass_after": 0.0,
            "maximum_influences": 4,
            "changed_vertex_count": 1,
        },
        validation=_passed_validation(),
        output_record=_record(output_glb),
        change_record=_record(empty_log),
        diagnostic_record=_record(diagnostic),
    )
    payload["user_approval"] = "user_approved"
    manifest = _write_readonly(
        output_dir / sanitizer.OUTPUT_MANIFEST_NAME,
        (json.dumps(payload, sort_keys=True) + "\n").encode(),
    )
    with pytest.raises(sanitizer.LowerLimbSanitationError, match="unexpected"):
        sanitizer.validate_published_manifest(manifest)


def test_failure_bundle_preserves_glb_change_log_and_structured_diagnostic(tmp_path):
    output_dir = tmp_path / "sanitized_weights_v2"
    staging = tmp_path / ".sanitized_weights_v2.fixture.staging"
    staging.mkdir()
    (staging / sanitizer.OUTPUT_GLB_NAME).write_bytes(b"glTFfailed-output")
    (staging / sanitizer.CHANGE_LOG_NAME).write_bytes(b'{"vertex_index":7}\n')
    (staging / sanitizer.ROUNDTRIP_DIAGNOSTIC_NAME).write_bytes(
        b'{"status":"failed"}\n'
    )

    manifest = sanitizer._preserve_failed_staging(
        staging=staging,
        output_dir=output_dir,
        asset_id="person_male_01",
        error=RuntimeError("injected roundtrip rejection"),
    )

    bundle = manifest.parent
    payload = json.loads(manifest.read_text())
    assert not staging.exists()
    assert bundle.name.startswith("sanitized_weights_v2.failed.")
    assert bundle.stat().st_mode & 0o777 == 0o555
    assert set(payload["preserved_artifacts"]) == {
        sanitizer.OUTPUT_GLB_NAME,
        sanitizer.CHANGE_LOG_NAME,
        sanitizer.ROUNDTRIP_DIAGNOSTIC_NAME,
    }
    assert payload["external_inventory_descriptor"]["artifact_count"] == 3
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in bundle.iterdir())


def test_cli_is_generic_and_source_has_no_bone_number_or_inference_backend():
    args = sanitizer.parse_args(
        [
            "--asset-id",
            "person_female_01",
            "--mode",
            "passed_static_owner",
            "--source-glb",
            "/input/source.glb",
            "--input-glb",
            "/input/bind.glb",
            "--input-manifest",
            "/input/manifest.json",
            "--static-qa-json",
            "/input/static_qa.json",
            "--output-dir",
            "/output/sanitized_v2",
        ]
    )
    assert args.asset_id == "person_female_01"
    assert args.mode == "passed_static_owner"
    assert args.output_dir == Path("/output/sanitized_v2")

    source = (
        Path(__file__).resolve().parents[2]
        / "tools/blender_sanitize_tokenrig_lower_limb_weights_v2.py"
    ).read_text(encoding="utf-8")
    assert re.search(r"[\"']bone_\d+", source) is None
    assert "torch" not in source
    assert "subprocess" not in source
    assert "validate_extracted_world_positions" in source
    assert "def _world_positions" not in source
    assert "export_animations=False" not in source  # delegated pinned bind exporter
