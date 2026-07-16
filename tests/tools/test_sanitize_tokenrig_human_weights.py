"""CPU contracts for deterministic, no-inference TokenRig weight sanitation."""

from __future__ import annotations

import importlib
import ast
import json
import math
from pathlib import Path

import pytest


sanitizer = importlib.import_module("tools.blender_sanitize_tokenrig_human_weights")
static = importlib.import_module("tools.blender_tokenrig_human_static_audit")


CHAINS = {
    "axial": ("pelvis", "spine", "neck", "head"),
    "left_arm": ("left_clavicle", "left_upper", "left_fore", "left_hand"),
    "right_arm": ("right_clavicle", "right_upper", "right_fore", "right_hand"),
    "left_leg": ("left_thigh", "left_calf", "left_foot", "left_toe"),
    "right_leg": ("right_thigh", "right_calf", "right_foot", "right_toe"),
}

PARENTS = {
    "pelvis": None,
    "spine": "pelvis",
    "neck": "spine",
    "head": "neck",
    "left_clavicle": "spine",
    "left_upper": "left_clavicle",
    "left_fore": "left_upper",
    "left_hand": "left_fore",
    "left_finger": "left_hand",
    "right_clavicle": "spine",
    "right_upper": "right_clavicle",
    "right_fore": "right_upper",
    "right_hand": "right_fore",
    "right_finger": "right_hand",
    "left_thigh": "pelvis",
    "left_calf": "left_thigh",
    "left_foot": "left_calf",
    "left_toe": "left_foot",
    "right_thigh": "pelvis",
    "right_calf": "right_thigh",
    "right_foot": "right_calf",
    "right_toe": "right_foot",
}


def test_transfer_map_pairs_core_chains_in_topological_order_and_folds_fingers():
    result = sanitizer.build_bilateral_transfer_maps(CHAINS, PARENTS)

    assert result["to_left"]["right_clavicle"] == "left_clavicle"
    assert result["to_left"]["right_toe"] == "left_toe"
    assert result["to_right"]["left_fore"] == "right_fore"
    assert result["to_left"]["right_finger"] == "left_hand"
    assert result["to_right"]["left_finger"] == "right_hand"
    assert result["core_pairs"] == [
        ["left_clavicle", "right_clavicle"],
        ["left_upper", "right_upper"],
        ["left_fore", "right_fore"],
        ["left_hand", "right_hand"],
        ["left_thigh", "right_thigh"],
        ["left_calf", "right_calf"],
        ["left_foot", "right_foot"],
        ["left_toe", "right_toe"],
    ]


def test_contract_pins_fitted_candidate_failures_and_nested_output():
    contract = sanitizer.PINNED_CONTRACT

    assert contract.asset_id == "rocketbox_male_adult_01"
    assert contract.fitted_glb.sha256 == (
        "eb9566f091b6de5357375dee750e66a48bcf4b12ba97a87615c26bed4cf77017"
    )
    assert contract.fitted_manifest.sha256 == (
        "f2be8c719ea5049b76efc77220af5ae686e72c50913acbe85b7555276a506e56"
    )
    assert [record.sha256 for record in contract.fitted_failures] == [
        "39a5b61542c9355cb8f584637a692d801fc4ba6e2e5c33b757b39f1351b7d9ee",
        "1b3a11c0708ffe2b70f2c363d7617ca4437deab0503cbe165cac9e7c1d0366e4",
    ]
    assert [record.sha256 for record in contract.sanitation_failures] == [
        "dc4aafc914d8ebbd521dcf0c14320d1f2fdf93666be1787e0b73c2a24c6ae4e4",
        "841a350f1b9178f76b09f72a65f9873e46b3745326d9a2a15eb84d6e4549021c",
        "a1796a7e9b159bbe66bc3218a02327322253b0e2a9bf8f02e8d720103210228e",
    ]
    assert contract.output_dir.name == "sanitized_weights_v1"
    assert contract.output_dir.parent.name == "fitted_skeleton_v1"


def test_real_fitted_glb_has_exact_52_by_16_inverse_bind_contract():
    parsed = static.read_glb(sanitizer.PINNED_CONTRACT.fitted_glb.path)

    contract = static.extract_inverse_bind_contract(parsed)

    assert len(contract["joint_names"]) == 52
    assert len(set(contract["joint_names"])) == 52
    assert len(contract["matrices"]) == 52
    assert {len(matrix) for matrix in contract["matrices"]} == {16}


def test_fitted_failure_payloads_prove_quantization_history_and_real_seam_rejection():
    result = sanitizer.validate_fitted_failure_payloads(
        (
            {
                "decision": "rejected",
                "readiness_bundle_published": False,
                "failure": {"message": "surface unique position count changed"},
            },
            {
                "decision": "rejected",
                "readiness_bundle_published": False,
                "failure": {
                    "message": "UV seam duplicate vertex 6041 has inconsistent skin weight"
                },
            },
        )
    )

    assert result == {
        "obsolete_exact_tuple_import_gate": "rejected",
        "ordered_fitted_skin_gate": "rejected_at_seam",
        "animation_authorized": False,
    }
    with pytest.raises(sanitizer.SanitationError, match="seam"):
        sanitizer.validate_fitted_failure_payloads(
            (
                {
                    "decision": "rejected",
                    "readiness_bundle_published": False,
                    "failure": {"message": "surface unique position count changed"},
                },
                {
                    "decision": "rejected",
                    "readiness_bundle_published": False,
                    "failure": {"message": "different"},
                },
            )
        )


def test_sanitation_transfers_opposite_mass_then_makes_seams_identical():
    positions = (
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (0.8, 0.1, 1.0),
        (0.8, 0.1, 1.0),
        (0.0, 0.0, 1.0),
    )
    weights = (
        {"left_fore": 0.8, "right_fore": 0.2},
        {"right_toe": 0.7, "left_toe": 0.2, "left_finger": 0.1},
        {"left_hand": 0.8, "right_hand": 0.2},
        {"left_hand": 0.6, "spine": 0.4},
        {"right_fore": 0.6, "spine": 0.4},
    )

    sanitized, report, changes = sanitizer.sanitize_weight_maps(
        positions=positions,
        vertex_weights=weights,
        chains=CHAINS,
        parents=PARENTS,
    )

    assert sanitized[0] == {"left_fore": 1.0}
    assert sanitized[1] == pytest.approx({"right_hand": 0.1, "right_toe": 0.9})
    assert sanitized[2] == sanitized[3] == {"left_hand": 0.8, "spine": 0.2}
    assert sanitized[4] == weights[4]  # central vertices are intentionally untouched
    assert report["algorithm_version"] == "tokenrig_side_transfer_seam_hybrid_export_floor_v3"
    assert report["changed_vertex_count"] == 4
    assert report["changed_vertex_ratio"] == pytest.approx(0.8)
    assert report["per_vertex_l1_accounting"] == {
        "vertex_count": 5,
        "explicit_changed_record_count": 4,
        "implicit_unchanged_vertex_count": 1,
        "implicit_unchanged_l1_before_after": 0.0,
    }
    assert report["total_transferred_mass"] == pytest.approx(0.7)
    assert report["bilateral_validation"]["contaminated_vertex_count"] == 0
    assert report["seam_validation"]["maximum_weight_l1_error"] == 0.0
    assert report["seam_reconciliation_method_counts"]["weighted_average"] == 1
    assert report["seam_group_records"][0]["method"] == "weighted_average"
    assert report["total_truncated_mass"] == 0.0
    assert report["l1_all_vertices"]["maximum"] > 0.0
    assert set(report["l1_all_vertices"]) == {"p50", "p95", "p99", "maximum"}
    assert [record["vertex_index"] for record in changes] == [0, 1, 2, 3]
    assert all(record["l1_before_after"] > 0.0 for record in changes)
    assert all("before" in record and "after" in record for record in changes)
    static.validate_seam_weights(positions, sanitized)
    static.validate_bilateral_contamination(positions, sanitized, CHAINS)


def test_seam_top_four_is_deterministic_for_equal_average_weights():
    positions = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    weights = (
        {"a": 0.4, "b": 0.2, "c": 0.2, "d": 0.2},
        {"b": 0.2, "c": 0.2, "d": 0.2, "e": 0.4},
        {"a": 1.0},
    )
    parents = {**PARENTS, "a": None, "b": None, "c": None, "d": None, "e": None}

    sanitized, report, _ = sanitizer.sanitize_weight_maps(
        positions=positions,
        vertex_weights=weights,
        chains=CHAINS,
        parents=parents,
    )

    assert sanitized[0] == sanitized[1]
    assert sanitized[0] == weights[0]
    assert math.isclose(sum(sanitized[0].values()), 1.0)
    assert report["total_truncated_mass"] == 0.0
    assert report["total_proposed_average_truncated_mass"] == pytest.approx(0.2)
    assert report["maximum_proposed_average_truncated_mass"] == pytest.approx(0.2)
    assert report["seam_reconciliation_method_counts"]["l1_medoid"] == 1
    group = report["seam_group_records"][0]
    assert group["method"] == "l1_medoid"
    assert group["medoid_vertex_index"] == 0
    assert group["applied_truncated_mass"] == 0.0


def test_export_floor_preserves_support_across_blender_hard_cutoff():
    tiny = 9.99997864710167e-05
    weights = {"large": 1.0 - tiny, "tiny": tiny}

    projected, record = sanitizer.project_export_safe_weights(weights)

    assert sanitizer.BLENDER_EXPORT_MIN_INFLUENCE == 0.0001
    assert sanitizer.BLENDER_EXPORT_SAFE_FLOOR == pytest.approx(
        0.00010000000474974513, rel=0.0, abs=0.0
    )
    assert projected["tiny"] == sanitizer.BLENDER_EXPORT_SAFE_FLOOR
    assert math.isclose(sum(projected.values()), 1.0, abs_tol=1e-15)
    assert record["component_count"] == 1
    assert record["added_mass"] == pytest.approx(
        sanitizer.BLENDER_EXPORT_SAFE_FLOOR - tiny
    )
    assert record["l1"] == pytest.approx(2.0 * record["added_mass"])

    exported = {
        name: value
        for name, value in projected.items()
        if value > sanitizer.BLENDER_EXPORT_MIN_INFLUENCE
    }
    total = sum(exported.values())
    exported = {name: value / total for name, value in exported.items()}
    assert sanitizer.static_audit._weight_l1(projected, exported) <= 1e-15


def test_export_floor_rejects_non_micro_support_projection():
    with pytest.raises(sanitizer.SanitationError, match="export-floor.*budget"):
        sanitizer.project_export_safe_weights({"large": 0.999999, "tiny": 0.000001})


def test_pinned_blender_exporter_source_proves_the_hard_cutoff():
    source = Path(
        "/data/jzy/blender/blender-4.2.1-linux-x64/4.2/scripts/addons_core/"
        "io_scene_gltf2/blender/exp/gltf2_blender_gather_primitives_extract.py"
    ).read_text(encoding="utf-8")

    assert "min_influence = 0.0001" in source
    assert "if weight <= min_influence:" in source


def test_transfer_map_rejects_non_disjoint_or_short_core_chains():
    with pytest.raises(sanitizer.SanitationError, match="four core bones"):
        sanitizer.build_bilateral_transfer_maps(
            {**CHAINS, "left_arm": ("left_upper", "left_fore", "left_hand")},
            PARENTS,
        )
    with pytest.raises(sanitizer.SanitationError, match="disjoint"):
        sanitizer.build_bilateral_transfer_maps(
            {**CHAINS, "right_leg": ("right_thigh", "right_calf", "right_foot", "left_toe")},
            PARENTS,
        )


def test_change_jsonl_is_sorted_and_complete_for_each_changed_vertex():
    records = (
        {
            "vertex_index": 4,
            "before": {"b": 1.0},
            "after": {"a": 1.0},
            "l1_before_after": 2.0,
            "transferred_mass": 1.0,
        },
        {
            "vertex_index": 2,
            "before": {"a": 0.5, "b": 0.5},
            "after": {"a": 1.0},
            "l1_before_after": 1.0,
            "transferred_mass": 0.5,
        },
    )

    payload = sanitizer.serialize_change_records(records)
    parsed = [json.loads(line) for line in payload.decode("utf-8").splitlines()]

    assert [record["vertex_index"] for record in parsed] == [2, 4]
    assert payload.endswith(b"\n")


def test_atomic_immutable_failure_write_never_exposes_partial_json(tmp_path, monkeypatch):
    target = tmp_path / "sanitized.failed.json"
    sanitizer.atomic_write_immutable_noreplace(target, b'{"decision":"rejected"}\n')

    assert target.read_bytes() == b'{"decision":"rejected"}\n'
    assert target.stat().st_mode & 0o777 == 0o444
    with pytest.raises(sanitizer.SanitationError, match="already exists"):
        sanitizer.atomic_write_immutable_noreplace(target, b"tamper\n")
    assert target.read_bytes() == b'{"decision":"rejected"}\n'

    interrupted = tmp_path / "interrupted.failed.json"

    def fail_rename(source, destination):
        raise sanitizer.SanitationError("injected rename failure")

    monkeypatch.setattr(sanitizer, "_rename_file_noreplace", fail_rename)
    with pytest.raises(sanitizer.SanitationError, match="injected"):
        sanitizer.atomic_write_immutable_noreplace(interrupted, b"partial\n")
    assert not interrupted.exists()
    assert not list(tmp_path.glob(f".{interrupted.name}.*.tmp"))


def test_code_snapshot_detects_shared_worktree_mutation(tmp_path):
    sanitizer_code = tmp_path / "sanitizer.py"
    static_code = tmp_path / "static.py"
    sanitizer_code.write_text("version = 1\n", encoding="utf-8")
    static_code.write_text("version = 1\n", encoding="utf-8")
    snapshot = sanitizer.capture_code_snapshot(
        {"sanitizer": sanitizer_code, "static_audit": static_code}
    )

    sanitizer.verify_code_snapshot(snapshot)
    static_code.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(sanitizer.SanitationError, match="code changed"):
        sanitizer.verify_code_snapshot(snapshot)


def test_cli_is_fixed_to_the_authenticated_fitted_input_and_has_no_animation_option():
    args = sanitizer.parse_args([])

    assert args.asset_id == "rocketbox_male_adult_01"
    assert args.output_dir.name == "sanitized_weights_v1"
    assert not hasattr(args, "animation")


def test_runner_source_has_no_inference_subprocess_or_rocketbox_weight_path():
    source = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "blender_sanitize_tokenrig_human_weights.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "subprocess" not in imported
    assert "torch" not in imported
    assert "Rocketbox" not in source
    assert "export_animations=False" in source


def test_runner_wires_full_rest_and_inverse_bind_before_publication():
    source = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "blender_sanitize_tokenrig_human_weights.py"
    ).read_text(encoding="utf-8")

    assert source.count("capture_blender_full_rest_contract(") >= 3
    assert source.count("compare_full_rest_contracts(") >= 2
    assert source.count("extract_inverse_bind_contract(") >= 2
    assert "compare_inverse_bind_contracts(" in source
    assert source.index("compare_inverse_bind_contracts(") < source.index(
        "rename_directory_noreplace("
    )
