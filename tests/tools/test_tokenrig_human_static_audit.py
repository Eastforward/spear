"""CPU and source-contract tests for the Route 2 TokenRig static gate."""

from __future__ import annotations

import hashlib
import importlib
import ast
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest


audit = importlib.import_module("tools.blender_tokenrig_human_static_audit")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _glb_bytes(
    *,
    image_payload: bytes = b"pixal-pbr",
    image_name: str = "pbr",
    material_name: str | None = "PixalPBR",
    webp_extension: bool = False,
) -> bytes:
    binary = image_payload + b"\x00" * ((-len(image_payload)) % 4)
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(image_payload)}],
        "images": [
            {
                "name": image_name,
                "mimeType": "image/png",
                "bufferView": 0,
            }
        ],
        "textures": [
            {"extensions": {"EXT_texture_webp": {"source": 0}}}
            if webp_extension
            else {"source": 0}
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 0},
                },
                "normalTexture": {"index": 0},
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2,
                            "JOINTS_0": 3,
                            "WEIGHTS_0": 4,
                        },
                        "material": 0,
                    }
                ]
            }
        ],
        "nodes": [{"name": "Body", "mesh": 0, "skin": 0}],
        "skins": [{"joints": [1]}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if material_name is not None:
        document["materials"][0]["name"] = material_name
    if webp_extension:
        document["extensionsUsed"] = ["EXT_texture_webp"]
        document["extensionsRequired"] = ["EXT_texture_webp"]
        document["images"][0]["mimeType"] = "image/webp"
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), audit.GLB_JSON_CHUNK)
        + encoded
        + struct.pack("<II", len(binary), audit.GLB_BIN_CHUNK)
        + binary
    )


def _write_task3_fixture(tmp_path: Path):
    source = tmp_path / "canary.glb"
    source_manifest = tmp_path / "canary.manifest.json"
    tokenrig = tmp_path / "tokenrig_transfer.glb"
    source.write_bytes(_glb_bytes())
    source_manifest.write_text('{"backend":"pixal3d"}\n', encoding="utf-8")
    tokenrig.write_bytes(_glb_bytes())
    manifest = tmp_path / "tokenrig_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "pixal_tokenrig_canary_v1",
                "asset_id": "rocketbox_male_adult_01",
                "source_front": "positive-y",
                "canonical_front": "negative-y",
                "attempt": "direct_transfer",
                "input": {
                    "glb": {
                        "path": str(source.resolve()),
                        "sha256": _sha256(source),
                        "bytes": source.stat().st_size,
                    },
                    "manifest": {
                        "path": str(source_manifest.resolve()),
                        "sha256": _sha256(source_manifest),
                        "bytes": source_manifest.stat().st_size,
                    },
                },
                "output": {
                    "path": str(tokenrig.resolve()),
                    "sha256": _sha256(tokenrig),
                    "bytes": tokenrig.stat().st_size,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source, source_manifest, tokenrig, manifest


def test_authenticates_task3_source_output_and_axis_contract(tmp_path):
    source, source_manifest, tokenrig, manifest = _write_task3_fixture(tmp_path)

    authenticated = audit.authenticate_task3_inputs(
        asset_id="rocketbox_male_adult_01",
        source_glb=source,
        tokenrig_glb=tokenrig,
        tokenrig_manifest=manifest,
    )

    assert authenticated["source_glb_sha256"] == _sha256(source)
    assert authenticated["source_manifest_sha256"] == _sha256(source_manifest)
    assert authenticated["tokenrig_glb_sha256"] == _sha256(tokenrig)
    assert authenticated["tokenrig_manifest_sha256"] == _sha256(manifest)
    assert authenticated["attempt"] == "direct_transfer"


def test_authentication_rejects_stale_output_without_publishing(tmp_path):
    source, _, tokenrig, manifest = _write_task3_fixture(tmp_path)
    tokenrig.write_bytes(tokenrig.read_bytes() + b"tamper")

    with pytest.raises(audit.StaticAuditError, match="TokenRig output.*SHA-256"):
        audit.authenticate_task3_inputs(
            asset_id="rocketbox_male_adult_01",
            source_glb=source,
            tokenrig_glb=tokenrig,
            tokenrig_manifest=manifest,
        )


def test_authentication_rejects_a_stale_original_pixal_manifest(tmp_path):
    source, source_manifest, tokenrig, manifest = _write_task3_fixture(tmp_path)
    source_manifest.write_text('{"backend":"tampered"}\n', encoding="utf-8")

    with pytest.raises(audit.StaticAuditError, match="Pixal.*manifest.*SHA-256"):
        audit.authenticate_task3_inputs(
            asset_id="rocketbox_male_adult_01",
            source_glb=source,
            tokenrig_glb=tokenrig,
            tokenrig_manifest=manifest,
        )


def test_authenticates_honest_failed_gate_recovery_without_calling_task3_passed(tmp_path):
    source, _, tokenrig, manifest = _write_task3_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    output_record = payload["output"]
    payload.update(
        {
            "schema": "pixal_tokenrig_recovery_v1",
            "state_classification": "research_candidate_recovered_from_hygiene_assertion",
            "task3_gate_status": "failed",
            "pbr_validation_status": "pending_static_audit",
            "recovery": {
                "task3_passed": False,
                "returncode": 0,
                "failure_stage": "output_validation",
                "error": {"type": "CanaryError", "message": "hygiene assertion"},
                "failed_evidence": {"files": {"tokenrig_transfer.glb": output_record}},
                "upstream_clean_bpy": {"bpyparser_load_calls_clean_before_import": True},
            },
        }
    )
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    authenticated = audit.authenticate_task3_inputs(
        asset_id="rocketbox_male_adult_01",
        source_glb=source,
        tokenrig_glb=tokenrig,
        tokenrig_manifest=manifest,
    )

    assert authenticated["manifest_schema"] == "pixal_tokenrig_recovery_v1"
    assert authenticated["task3_gate_status"] == "failed"
    assert authenticated["recovered_candidate"] is True


def _write_fitted_fixture(tmp_path: Path) -> dict[str, object]:
    source, source_manifest, conditioning, _ = _write_task3_fixture(tmp_path)
    fitted = tmp_path / "fitted_skeleton.glb"
    fitted.write_bytes(conditioning.read_bytes())
    recovery = tmp_path / "recovery.json"
    recovery.write_text('{"schema":"pixal_tokenrig_recovery_v1"}\n', encoding="utf-8")
    failures = []
    for index, message in enumerate(
        (
            "raw GLB triangle count changed: source=10 output=9",
            "opposite-limb contamination on distal vertices: count=2 maximum=0.1",
        )
    ):
        path = tmp_path / f"static.failed.{index}.json"
        path.write_text(
            json.dumps(
                {
                    "decision": "rejected",
                    "readiness_bundle_published": False,
                    "failure": {"message": message},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o444)
        failures.append(_record(path))
    conditioning_record = _record(conditioning)

    wrapper = tmp_path / "tokenrig_human_fitted_skeleton_fallback.py"
    wrapper.write_text("from tools import tokenrig_human_canary as base\n", encoding="utf-8")
    delegated = tmp_path / "tokenrig_human_canary.py"
    delegated.write_text("SCHEMA = 'pixal_tokenrig_canary_v1'\n", encoding="utf-8")
    command = [
        "/skintokens/python",
        "-c",
        "seed bootstrap",
        "42",
        "/skintokens/demo.py",
        "--input",
        str(conditioning.resolve()),
        "--output",
        str((tmp_path / ".staging/tokenrig_transfer.glb").resolve()),
        "--use_transfer",
        "--use_skeleton",
    ]
    ledger = tmp_path / "fitted_skeleton_v1.tokenrig_attempt.json"
    ledger.write_text(
        json.dumps(
            {
                "schema": "pixal_tokenrig_attempt_v1",
                "status": "succeeded",
                "returncode": 0,
                "command": command,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime_patch"
    markers = runtime / "markers"
    markers.mkdir(parents=True)
    sitecustomize = runtime / "sitecustomize.py"
    sitecustomize.write_text("# deterministic clean-before-load patch\n", encoding="utf-8")
    load_audit = runtime / "load_audit.jsonl"
    phases = ("before_clean", "after_clean", "after_import")
    load_audit.write_text(
        "".join(
            json.dumps(
                {
                    "sequence": sequence,
                    "phase": phase,
                    "filepath": str(conditioning.resolve()),
                    "inventory": {
                        "objects": [],
                        "mesh_count": 0,
                        "material_count": 0,
                        "image_count": 0,
                    },
                },
                sort_keys=True,
            )
            + "\n"
            for sequence in (1, 2)
            for phase in phases
        ),
        encoding="utf-8",
    )
    processes = []
    for pid, role in ((101, "demo"), (202, "bpy_server")):
        marker = markers / f"{pid}.json"
        marker.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "seed": 42,
                    "patch_sha256": _sha256(sitecustomize),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        processes.append(
            {
                "pid": pid,
                "role": role,
                "marker": {
                    "path": str(marker.relative_to(tmp_path)),
                    "sha256": _sha256(marker),
                    "size_bytes": marker.stat().st_size,
                },
            }
        )
    manifest = tmp_path / "fitted_manifest.json"
    payload = {
        "schema": "pixal_tokenrig_fitted_skeleton_v1",
        "base_runner_schema": "pixal_tokenrig_canary_v1",
        "asset_id": "rocketbox_male_adult_01",
        "attempt": "fitted_skeleton_transfer",
        "source_front": "positive-y",
        "canonical_front": "negative-y",
        "task3_direct_gate_status": "failed",
        "static_audit_status": "pending_fitted_static_audit",
        "pbr_validation_status": "pending_static_audit",
        "animation_authorized": False,
        "inference_parameters": {"use_skeleton": True, "use_transfer": True},
        "random_parameters": {"seed": 42},
        "command": command,
        "attempt_ledger": _record(ledger),
        "orchestrator": {
            "provenance_schema": "pixal_tokenrig_canary_v1",
            "runner": _record(wrapper),
            "delegated_runner": _record(delegated),
        },
        "server_hygiene": {
            "cleans_before_every_bpyparser_load": True,
            "mechanism": "injected_sitecustomize_v1",
            "relative_path": str(sitecustomize.relative_to(tmp_path)),
            "sha256": _sha256(sitecustomize),
            "load_audit": {
                "relative_path": str(load_audit.relative_to(tmp_path)),
                "sha256": _sha256(load_audit),
                "size_bytes": load_audit.stat().st_size,
            },
            "loads": [
                {
                    "filepath": str(conditioning.resolve()),
                    "role": "source",
                    "sequence": 1,
                },
                {
                    "filepath": str(conditioning.resolve()),
                    "role": "transfer_target",
                    "sequence": 2,
                },
            ],
            "processes": processes,
        },
        "input": {
            "glb": conditioning_record,
            "manifest": _record(recovery),
            "fallback_provenance": {
                "original_source_glb": _record(source),
                "original_source_manifest": _record(source_manifest),
                "static_failures": failures,
                "animation_authorized": False,
            },
        },
        "fitted_skeleton": {
            "use_skeleton_input": True,
            "conditioning_source": conditioning_record,
        },
        "output": _record(fitted),
    }
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "source": source,
        "fitted": fitted,
        "manifest": manifest,
        "ledger": ledger,
        "load_audit": load_audit,
        "wrapper": wrapper,
        "delegated": delegated,
        "payload": payload,
    }


def test_authenticates_fitted_manifest_only_with_forced_skeleton_and_two_failures(tmp_path):
    fixture = _write_fitted_fixture(tmp_path)

    authenticated = audit.authenticate_task3_inputs(
        asset_id="rocketbox_male_adult_01",
        source_glb=fixture["source"],
        tokenrig_glb=fixture["fitted"],
        tokenrig_manifest=fixture["manifest"],
    )

    assert authenticated["manifest_schema"] == "pixal_tokenrig_fitted_skeleton_v1"
    assert authenticated["fitted_candidate"] is True
    assert authenticated["task3_gate_status"] == "failed"
    assert authenticated["attempt_ledger_sha256"] == _sha256(fixture["ledger"])
    assert authenticated["orchestrator_runner_sha256"] == _sha256(fixture["wrapper"])
    assert authenticated["delegated_runner_sha256"] == _sha256(fixture["delegated"])
    assert authenticated["server_hygiene_load_event_count"] == 6


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("ledger_status", "ledger.*succeeded"),
        ("command", "command"),
        ("load_audit", "six.*load-audit"),
        ("processes", "two.*process"),
        ("wrapper", "orchestrator runner.*SHA-256"),
        ("delegated", "delegated runner.*SHA-256"),
    ),
)
def test_fitted_auth_rejects_incomplete_execution_and_code_provenance(
    tmp_path, mutation, message
):
    fixture = _write_fitted_fixture(tmp_path)
    payload = fixture["payload"]
    if mutation == "ledger_status":
        ledger_payload = json.loads(fixture["ledger"].read_text(encoding="utf-8"))
        ledger_payload["status"] = "failed"
        fixture["ledger"].write_text(json.dumps(ledger_payload) + "\n", encoding="utf-8")
        payload["attempt_ledger"] = _record(fixture["ledger"])
    elif mutation == "command":
        payload["command"] = [value for value in payload["command"] if value != "--use_skeleton"]
        ledger_payload = json.loads(fixture["ledger"].read_text(encoding="utf-8"))
        ledger_payload["command"] = payload["command"]
        fixture["ledger"].write_text(json.dumps(ledger_payload) + "\n", encoding="utf-8")
        payload["attempt_ledger"] = _record(fixture["ledger"])
    elif mutation == "load_audit":
        lines = fixture["load_audit"].read_text(encoding="utf-8").splitlines()
        fixture["load_audit"].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        payload["server_hygiene"]["load_audit"] = {
            "relative_path": str(fixture["load_audit"].relative_to(tmp_path)),
            "sha256": _sha256(fixture["load_audit"]),
            "size_bytes": fixture["load_audit"].stat().st_size,
        }
    elif mutation == "processes":
        payload["server_hygiene"]["processes"] = payload["server_hygiene"]["processes"][:1]
    elif mutation == "wrapper":
        fixture["wrapper"].write_text("tampered wrapper\n", encoding="utf-8")
    elif mutation == "delegated":
        fixture["delegated"].write_text("tampered delegated runner\n", encoding="utf-8")
    fixture["manifest"].write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(audit.StaticAuditError, match=message):
        audit.authenticate_task3_inputs(
            asset_id="rocketbox_male_adult_01",
            source_glb=fixture["source"],
            tokenrig_glb=fixture["fitted"],
            tokenrig_manifest=fixture["manifest"],
        )


def _write_sanitized_fixture(tmp_path: Path) -> dict[str, object]:
    fitted = _write_fitted_fixture(tmp_path)
    output_dir = tmp_path / "sanitized_weights_v1"
    output_dir.mkdir()
    output = output_dir / "tokenrig_transfer.glb"
    output.write_bytes(fitted["fitted"].read_bytes())
    changes = output_dir / "weight_changes.jsonl"
    changes.write_text(
        '{"after":{"left":1.0},"before":{"right":1.0},'
        '"export_floor_added_mass":2e-10,"export_floor_component_count":1,'
        '"l1_before_after":2.0,'
        '"transferred_mass":1.0,"vertex_index":0}\n',
        encoding="utf-8",
    )
    changes.chmod(0o444)
    seam_groups = output_dir / "seam_groups.jsonl"
    seam_groups.write_text(
        '{"applied_truncated_mass":0.0,"group_index":0,'
        '"maximum_member_l1_to_reconciled":0.1,"method":"l1_medoid",'
        '"method_reason":"influence_union_exceeds_four",'
        '"medoid_vertex_index":0,'
        '"proposed_average_truncated_mass":0.01,'
        '"representative_vertex_index":0,"total_member_l1_to_reconciled":0.2,'
        '"union_influence_count":5,"vertex_count":2}\n',
        encoding="utf-8",
    )
    seam_groups.chmod(0o444)
    fitted_failures = []
    for index, message in enumerate(
        (
            "surface unique position count changed",
            "UV seam duplicate vertex 6041 has inconsistent skin weight",
        )
    ):
        path = tmp_path / f"fitted.static.failed.{index}.json"
        path.write_text(
            json.dumps(
                {
                    "decision": "rejected",
                    "readiness_bundle_published": False,
                    "failure": {"message": message},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o444)
        fitted_failures.append(_record(path))
    prior_sanitation_failures = []
    for index, message in enumerate(
        (
            "surface unique position coverage changed",
            "roundtrip skin position coverage changed: missing=2 extra=2",
            "roundtrip skin weights changed at 2: L1=0.0002",
        )
    ):
        path = tmp_path / f"sanitized_weights_v1.failed.prior.{index}.json"
        path.write_text(
            json.dumps(
                {
                    "decision": "rejected",
                    "readiness_bundle_published": False,
                    "failure": {"message": message},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o444)
        prior_sanitation_failures.append(_record(path))
    sanitizer_code = tmp_path / "blender_sanitize_tokenrig_human_weights.py"
    sanitizer_code.write_text("ALGORITHM_VERSION = 'v1'\n", encoding="utf-8")
    static_code = tmp_path / "blender_tokenrig_human_static_audit.py"
    static_code.write_text("SCHEMA = 'static'\n", encoding="utf-8")
    fitted_authentication = audit.authenticate_task3_inputs(
        asset_id="rocketbox_male_adult_01",
        source_glb=fitted["source"],
        tokenrig_glb=fitted["fitted"],
        tokenrig_manifest=fitted["manifest"],
    )
    fitted_payload = fitted["payload"]
    payload = {
        "schema": "pixal_tokenrig_sanitized_weights_v1",
        "asset_id": "rocketbox_male_adult_01",
        "attempt": "deterministic_learned_weight_sanitation",
        "algorithm_version": "tokenrig_side_transfer_seam_hybrid_export_floor_v3",
        "source_front": "positive-y",
        "canonical_front": "negative-y",
        "inference_used": False,
        "rocketbox_mesh_used": False,
        "rocketbox_weights_used": False,
        "animation_authorized": False,
        "static_audit_status": "pending_sanitized_static_audit",
        "publication": {
            "directory_mode": "0755",
            "artifact_mode": "0444",
            "no_replace": True,
            "directory_mode_reason": (
                "owner write permission is required only to create the nested "
                "static_audit_v1 readiness bundle"
            ),
        },
        "input": {
            "original_source_glb": _record(fitted["source"]),
            "original_source_manifest": fitted_payload["input"]["fallback_provenance"][
                "original_source_manifest"
            ],
            "direct_glb": fitted_payload["input"]["glb"],
            "recovery_manifest": fitted_payload["input"]["manifest"],
            "direct_failures": fitted_payload["input"]["fallback_provenance"][
                "static_failures"
            ],
            "fitted_glb": _record(fitted["fitted"]),
            "fitted_manifest": _record(fitted["manifest"]),
            "fitted_failures": fitted_failures,
            "prior_sanitation_failures": prior_sanitation_failures,
            "fitted_failure_summary": {
                "obsolete_exact_tuple_import_gate": "rejected",
                "ordered_fitted_skin_gate": "rejected_at_seam",
                "animation_authorized": False,
            },
            "fitted_authentication": fitted_authentication,
        },
        "code": {
            "sanitizer": _record(sanitizer_code),
            "static_audit": _record(static_code),
            "fitted_wrapper": _record(fitted["wrapper"]),
            "delegated_base_runner": _record(fitted["delegated"]),
        },
        "pre_sanitation": {
            "seam_measurement": {"violating_group_count": 1},
            "seam_rejection": "UV seam duplicate vertex 6041 has inconsistent skin weight",
            "bilateral_measurement": {
                "contaminated_vertex_count": 3,
                "maximum_opposite_limb_weight": 0.2,
            },
            "bilateral_rejection": (
                "opposite-limb contamination on distal vertices: count=3 maximum=0.2"
            ),
        },
        "sanitation": {
            "algorithm_version": "tokenrig_side_transfer_seam_hybrid_export_floor_v3",
            "inference_used": False,
            "vertex_count": 10,
            "changed_vertex_count": 1,
            "changed_vertex_ratio": 0.1,
            "per_vertex_l1_accounting": {
                "vertex_count": 10,
                "explicit_changed_record_count": 1,
                "implicit_unchanged_vertex_count": 9,
                "implicit_unchanged_l1_before_after": 0.0,
            },
            "total_transferred_mass": 1.0,
            "export_floor_projection": {
                "policy": "raise_droppable_support_to_next_float32_and_debit_largest_v1",
                "blender_min_influence": 0.0001,
                "safe_floor": 0.00010000000474974513,
                "maximum_added_mass_per_vertex_budget": 1e-8,
                "projected_vertex_count": 1,
                "projected_component_count": 1,
                "total_added_mass": 2e-10,
                "maximum_added_mass": 2e-10,
                "l1_all_vertices": {
                    "p50": 0.0,
                    "p95": 2.2e-10,
                    "p99": 3.64e-10,
                    "maximum": 4e-10,
                },
                "minimum_output_weight": 0.00010000000474974513,
                "minimum_applied_blender_weight": 0.00010000000474974513,
            },
            "total_truncated_mass": 0.0,
            "maximum_truncated_mass": 0.0,
            "total_proposed_average_truncated_mass": 0.01,
            "maximum_proposed_average_truncated_mass": 0.01,
            "seam_duplicate_group_count": 1,
            "seam_reconciliation_method_counts": {
                "weighted_average": 0,
                "l1_medoid": 1,
            },
            "transferred_mass_by_bone_pair": {"right->left": 1.0},
            "l1_all_vertices": {"p50": 0.0, "p95": 1.1, "p99": 1.82, "maximum": 2.0},
            "l1_changed_vertices": {"p50": 2.0, "p95": 2.0, "p99": 2.0, "maximum": 2.0},
            "weight_validation": {"vertex_count": 10, "maximum_influences": 4},
            "seam_validation": {
                "maximum_weight_l1_error": 0.0,
                "weight_l1_tolerance": 1e-6,
            },
            "bilateral_validation": {
                "contaminated_vertex_count": 0,
                "maximum_opposite_limb_weight": 0.0,
                "tolerance": 1e-4,
            },
        },
        "validation": {
            "input_pbr": {"passed": True},
            "input_raw_surface": {"passed": True},
            "in_scene_mesh": {"passed": True},
            "in_scene_surface": {"passed": True},
            "in_scene_rest": {"bone_count": 52},
            "in_scene_full_rest": {
                "passed": True,
                "bone_count": 52,
                "maximum_object_matrix_element_error": 0.0,
                "maximum_head_error_m": 0.0,
                "maximum_tail_error_m": 0.0,
                "maximum_roll_axis_error": 0.0,
                "maximum_roll_error_radians": 0.0,
                "maximum_matrix_element_error": 0.0,
                "tolerance": 2e-6,
            },
            "roundtrip_full_rest": {
                "passed": True,
                "bone_count": 52,
                "maximum_object_matrix_element_error": 0.0,
                "maximum_head_error_m": 0.0,
                "maximum_tail_error_m": 0.0,
                "maximum_roll_axis_error": 0.0,
                "maximum_roll_error_radians": 0.0,
                "maximum_matrix_element_error": 0.0,
                "tolerance": 2e-6,
            },
            "inverse_bind": {
                "passed": True,
                "joint_count": 52,
                "joint_order_unchanged": True,
                "exact_matrices_unchanged": False,
                "maximum_matrix_element_error": 0.0,
                "tolerance": 2e-6,
            },
            "restored_root_matrix_maximum_error": 0.0,
            "output_pbr": {"passed": True},
            "output_raw_surface": {"passed": True},
            "roundtrip": {"passed": True},
        },
        "artifacts": {
            "weight_changes": _record(changes),
            "seam_groups": _record(seam_groups),
        },
        "output": _record(output),
    }
    manifest = output_dir / "tokenrig_manifest.json"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **fitted,
        "sanitized": output,
        "sanitized_manifest": manifest,
        "sanitized_payload": payload,
        "changes": changes,
        "seam_groups": seam_groups,
        "sanitizer_code": sanitizer_code,
    }


def test_authenticates_sanitized_manifest_recursively_and_records_code(tmp_path):
    fixture = _write_sanitized_fixture(tmp_path)

    authenticated = audit.authenticate_task3_inputs(
        asset_id="rocketbox_male_adult_01",
        source_glb=fixture["source"],
        tokenrig_glb=fixture["sanitized"],
        tokenrig_manifest=fixture["sanitized_manifest"],
    )

    assert authenticated["manifest_schema"] == "pixal_tokenrig_sanitized_weights_v1"
    assert authenticated["sanitized_candidate"] is True
    assert authenticated["fitted_candidate"] is False
    assert authenticated["conditioning_glb_sha256"] == _sha256(fixture["fitted"])
    assert authenticated["sanitizer_runner_sha256"] == _sha256(
        fixture["sanitizer_code"]
    )
    assert authenticated["weight_changes_sha256"] == _sha256(fixture["changes"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("inference", "inference"),
        ("seam", "seam"),
        ("bilateral", "bilateral"),
        ("changes", "weight changes.*SHA-256"),
        ("sanitizer", "sanitizer runner.*SHA-256"),
        ("fitted_manifest", "fitted.*manifest.*SHA-256"),
        ("trunc_nan", "truncation"),
        ("seam_method", "seam-group"),
        ("changes_payload", "weight changes"),
        ("full_rest", "full-rest"),
        ("floor_policy", "export-floor"),
        ("floor_row", "export-floor"),
    ),
)
def test_sanitized_auth_rejects_mutated_algorithm_or_provenance(
    tmp_path, mutation, message
):
    fixture = _write_sanitized_fixture(tmp_path)
    payload = fixture["sanitized_payload"]
    if mutation == "inference":
        payload["inference_used"] = True
    elif mutation == "seam":
        payload["sanitation"]["seam_validation"]["maximum_weight_l1_error"] = 0.1
    elif mutation == "bilateral":
        payload["sanitation"]["bilateral_validation"]["contaminated_vertex_count"] = 1
    elif mutation == "changes":
        fixture["changes"].chmod(0o600)
        fixture["changes"].write_text("tampered\n", encoding="utf-8")
    elif mutation == "sanitizer":
        fixture["sanitizer_code"].write_text("tampered\n", encoding="utf-8")
    elif mutation == "fitted_manifest":
        fixture["manifest"].write_text("tampered\n", encoding="utf-8")
    elif mutation == "trunc_nan":
        payload["sanitation"]["total_proposed_average_truncated_mass"] = math.nan
    elif mutation == "seam_method":
        fixture["seam_groups"].chmod(0o600)
        group = json.loads(fixture["seam_groups"].read_text(encoding="utf-8"))
        group["method"] = "weighted_average"
        group["medoid_vertex_index"] = None
        fixture["seam_groups"].write_text(json.dumps(group) + "\n", encoding="utf-8")
        fixture["seam_groups"].chmod(0o444)
        payload["artifacts"]["seam_groups"] = _record(fixture["seam_groups"])
        payload["sanitation"]["seam_reconciliation_method_counts"] = {
            "weighted_average": 1,
            "l1_medoid": 0,
        }
    elif mutation == "changes_payload":
        fixture["changes"].chmod(0o600)
        change = json.loads(fixture["changes"].read_text(encoding="utf-8"))
        change["l1_before_after"] = 0.5
        fixture["changes"].write_text(json.dumps(change) + "\n", encoding="utf-8")
        fixture["changes"].chmod(0o444)
        payload["artifacts"]["weight_changes"] = _record(fixture["changes"])
    elif mutation == "full_rest":
        payload["validation"]["roundtrip_full_rest"].update(
            {
                "maximum_tail_error_m": math.inf,
                "tolerance": 2e-6,
            }
        )
    elif mutation == "floor_policy":
        payload["sanitation"]["export_floor_projection"][
            "safe_floor"
        ] = 0.0001
    elif mutation == "floor_row":
        fixture["changes"].chmod(0o600)
        change = json.loads(fixture["changes"].read_text(encoding="utf-8"))
        change["export_floor_component_count"] = 0
        fixture["changes"].write_text(json.dumps(change) + "\n", encoding="utf-8")
        fixture["changes"].chmod(0o444)
        payload["artifacts"]["weight_changes"] = _record(fixture["changes"])
    fixture["sanitized_manifest"].write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(audit.StaticAuditError, match=message):
        audit.authenticate_task3_inputs(
            asset_id="rocketbox_male_adult_01",
            source_glb=fixture["source"],
            tokenrig_glb=fixture["sanitized"],
            tokenrig_manifest=fixture["sanitized_manifest"],
        )


def test_glb_parser_hashes_embedded_pbr_payload_by_material_role(tmp_path):
    glb = tmp_path / "candidate.glb"
    glb.write_bytes(_glb_bytes(image_payload=b"exact-packed-pbr"))

    parsed = audit.read_glb(glb)
    contract = audit.pbr_payload_contract(parsed)

    expected = hashlib.sha256(b"exact-packed-pbr").hexdigest()
    assert contract == {
        "material_slot_0:base_color": {
            "material_name": "PixalPBR",
            "image_name": "pbr",
            "mime_type": "image/png",
            "size_bytes": len(b"exact-packed-pbr"),
            "sha256": expected,
        },
        "material_slot_0:metallic_roughness": {
            "material_name": "PixalPBR",
            "image_name": "pbr",
            "mime_type": "image/png",
            "size_bytes": len(b"exact-packed-pbr"),
            "sha256": expected,
        },
        "material_slot_0:normal": {
            "material_name": "PixalPBR",
            "image_name": "pbr",
            "mime_type": "image/png",
            "size_bytes": len(b"exact-packed-pbr"),
            "sha256": expected,
        },
    }


def test_pbr_contract_supports_required_ext_texture_webp_and_anonymous_material(tmp_path):
    payload = b"RIFF\x1a\x00\x00\x00WEBPVP8 pixal-packed"
    glb = tmp_path / "pixal-source.glb"
    glb.write_bytes(
        _glb_bytes(
            image_payload=payload,
            image_name="pixal_base_color",
            material_name=None,
            webp_extension=True,
        )
    )

    contract = audit.pbr_payload_contract(audit.read_glb(glb))

    expected = hashlib.sha256(payload).hexdigest()
    assert contract["material_slot_0:base_color"] == {
        "material_name": None,
        "image_name": "pixal_base_color",
        "mime_type": "image/webp",
        "size_bytes": len(payload),
        "sha256": expected,
    }
    assert set(contract) == {
        "material_slot_0:base_color",
        "material_slot_0:metallic_roughness",
        "material_slot_0:normal",
    }


def test_pbr_contract_rejects_external_or_missing_images(tmp_path):
    glb = tmp_path / "candidate.glb"
    glb.write_bytes(_glb_bytes())
    parsed = audit.read_glb(glb)
    parsed.document["images"][0] = {"uri": "texture.png"}

    with pytest.raises(audit.StaticAuditError, match="embedded bufferView"):
        audit.pbr_payload_contract(parsed)


def test_pbr_comparison_is_slot_role_based_across_anonymous_and_blender_names(tmp_path):
    source = tmp_path / "source.glb"
    output = tmp_path / "output.glb"
    source.write_bytes(_glb_bytes(material_name=None, image_name="packed"))
    output.write_bytes(
        _glb_bytes(material_name="PixalPBR.001", image_name="packed.001")
    )

    comparison = audit.compare_pbr_payloads(
        audit.pbr_payload_contract(audit.read_glb(source)),
        audit.pbr_payload_contract(audit.read_glb(output)),
    )

    assert comparison["passed"] is True
    assert comparison["roles"] == [
        "material_slot_0:base_color",
        "material_slot_0:metallic_roughness",
        "material_slot_0:normal",
    ]


def test_bounded_serialization_equivalence_is_explicitly_not_exact_topology_or_normals():
    metrics = {
        "source_triangle_count": 976970,
        "output_triangle_count": 976951,
        "removed_triangle_count": 19,
        "triangle_loss_ratio": 19 / 976970,
        "removed_faces_are_reverse_coincident": True,
        "unique_undirected_face_sets_equal": True,
        "removed_area_ratio": 8.73e-7,
        "maximum_position_error_m": 0.0,
        "maximum_uv_error": 0.0,
        "normal_error_p99": 2.70e-4,
        "maximum_normal_error": 0.01275,
        "backface_cull_risk": True,
    }

    result = audit.validate_serialization_equivalence_metrics(metrics)

    assert result["passed"] is True
    assert result["exact_topology_unchanged"] is False
    assert result["exact_normals_unchanged"] is False
    with pytest.raises(audit.StaticAuditError, match="loss ratio"):
        audit.validate_serialization_equivalence_metrics(
            dict(metrics, triangle_loss_ratio=2.1e-5)
        )
    with pytest.raises(audit.StaticAuditError, match="reverse-coincident"):
        audit.validate_serialization_equivalence_metrics(
            dict(metrics, removed_faces_are_reverse_coincident=False)
        )
    with pytest.raises(audit.StaticAuditError, match="normal p99"):
        audit.validate_serialization_equivalence_metrics(
            dict(metrics, normal_error_p99=1.1e-3)
        )


def test_canonical_axis_contract_is_exactly_one_proper_yaw():
    contract = audit.canonical_axis_contract(
        source_front="positive-y", prior_transform_count=0
    )

    matrix = contract["matrix"]
    assert matrix == (
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert contract["yaw_radians"] == math.pi
    assert contract["transform_count"] == 1
    assert contract["canonical_front_vector"] == (0.0, -1.0, 0.0)
    assert contract["canonical_up_vector"] == (0.0, 0.0, 1.0)
    assert contract["determinant"] == 1.0

    with pytest.raises(audit.StaticAuditError, match="exactly once"):
        audit.canonical_axis_contract(
            source_front="positive-y", prior_transform_count=1
        )


def _human_bones():
    values = [
        ("bone_0", None, (0.0, 0.0, 1.00)),
        ("bone_1", "bone_0", (0.0, 0.0, 1.20)),
        ("bone_2", "bone_1", (0.0, 0.0, 1.42)),
        ("bone_3", "bone_2", (0.0, 0.0, 1.58)),
        ("bone_4", "bone_3", (0.0, 0.0, 1.78)),
        ("bone_5", "bone_2", (0.16, 0.0, 1.43)),
        ("bone_6", "bone_5", (0.38, 0.0, 1.39)),
        ("bone_7", "bone_6", (0.62, 0.0, 1.34)),
        ("bone_8", "bone_7", (0.82, 0.0, 1.31)),
        ("bone_9", "bone_2", (-0.16, 0.0, 1.43)),
        ("bone_10", "bone_9", (-0.38, 0.0, 1.39)),
        ("bone_11", "bone_10", (-0.62, 0.0, 1.34)),
        ("bone_12", "bone_11", (-0.82, 0.0, 1.31)),
        ("bone_13", "bone_0", (0.13, 0.0, 0.92)),
        ("bone_14", "bone_13", (0.13, 0.0, 0.52)),
        ("bone_15", "bone_14", (0.13, -0.02, 0.12)),
        ("bone_16", "bone_15", (0.13, -0.22, 0.02)),
        ("bone_17", "bone_0", (-0.13, 0.0, 0.92)),
        ("bone_18", "bone_17", (-0.13, 0.0, 0.52)),
        ("bone_19", "bone_18", (-0.13, -0.02, 0.12)),
        ("bone_20", "bone_19", (-0.13, -0.22, 0.02)),
    ]
    return tuple(audit.BoneRecord(name=name, parent=parent, head=head) for name, parent, head in values)


def test_ground_contract_moves_runtime_closure_to_zero_exactly_once():
    contract = audit.ground_bind_contract(
        source_floor_z=-0.454006, prior_transform_count=0
    )

    assert contract == {
        "source_floor_z": -0.454006,
        "ground_translation_z": 0.454006,
        "post_floor_z": 0.0,
        "canonical_floor_z": 0.0,
        "transform_count": 1,
    }
    with pytest.raises(audit.StaticAuditError, match="exactly once"):
        audit.ground_bind_contract(
            source_floor_z=-0.454006, prior_transform_count=1
        )


def test_validates_finite_connected_single_root_hierarchy():
    result = audit.validate_hierarchy(_human_bones())

    assert result["root"] == "bone_0"
    assert result["bone_count"] == 21
    assert result["connected"] is True
    assert result["parent_first"] is True


def _full_rest_fixture(*, tail_x=0.0, roll=0.0, matrix_delta=0.0):
    matrix = [value for row in _identity_matrix() for value in row]
    matrix[0] += matrix_delta
    return {
        "armature_object_matrix_world": [
            value for row in _identity_matrix() for value in row
        ],
        "bones": [
            {
                "name": "bone_0",
                "parent": None,
                "head_local": [0.0, 0.0, 0.0],
                "tail_local": [tail_x, 0.0, 1.0],
                "roll_axis": [0.0, 0.0, 1.0],
                "roll_radians": roll,
                "matrix_local": matrix,
                "use_connect": False,
                "use_deform": True,
                "inherit_scale": "FULL",
            }
        ],
    }


def test_full_rest_contract_covers_tail_roll_and_every_matrix_element():
    expected = _full_rest_fixture()
    actual = _full_rest_fixture(tail_x=2e-7, roll=2e-7, matrix_delta=2e-7)

    result = audit.compare_full_rest_contracts(expected, actual)

    assert result["passed"] is True
    assert result["bone_count"] == 1
    assert result["maximum_tail_error_m"] == pytest.approx(2e-7)
    assert result["maximum_roll_error_radians"] == pytest.approx(2e-7)
    assert result["maximum_matrix_element_error"] == pytest.approx(2e-7)
    with pytest.raises(audit.StaticAuditError, match="tail"):
        audit.compare_full_rest_contracts(
            expected, _full_rest_fixture(tail_x=3e-6)
        )


def test_inverse_bind_contract_requires_joint_order_and_bounded_full_matrices():
    expected = {
        "joint_names": ["bone_0", "bone_1"],
        "matrices": [[1.0] * 16, [2.0] * 16],
    }
    actual = {
        "joint_names": ["bone_0", "bone_1"],
        "matrices": [[1.0 + 2e-7] + [1.0] * 15, [2.0] * 16],
    }

    result = audit.compare_inverse_bind_contracts(expected, actual)

    assert result["passed"] is True
    assert result["joint_count"] == 2
    assert result["maximum_matrix_element_error"] == pytest.approx(2e-7)
    with pytest.raises(audit.StaticAuditError, match="joint order"):
        audit.compare_inverse_bind_contracts(
            expected, {**actual, "joint_names": ["bone_1", "bone_0"]}
        )
    changed = json.loads(json.dumps(actual))
    changed["matrices"][0][0] = 1.01
    with pytest.raises(audit.StaticAuditError, match="inverse-bind"):
        audit.compare_inverse_bind_contracts(expected, changed)


def test_hierarchy_rejects_cycles_missing_parents_and_non_finite_heads():
    bones = list(_human_bones())
    bones[0] = audit.BoneRecord("bone_0", "bone_1", bones[0].head)
    with pytest.raises(audit.StaticAuditError, match="root|cycle"):
        audit.validate_hierarchy(tuple(bones))

    bones = list(_human_bones())
    bones[3] = audit.BoneRecord("bone_3", "missing", bones[3].head)
    with pytest.raises(audit.StaticAuditError, match="missing parent"):
        audit.validate_hierarchy(tuple(bones))

    bones = list(_human_bones())
    bones[4] = audit.BoneRecord("bone_4", "bone_3", (0.0, 0.0, math.nan))
    with pytest.raises(audit.StaticAuditError, match="non-finite"):
        audit.validate_hierarchy(tuple(bones))


def test_resolves_unique_five_semantic_chains_from_topology_and_position():
    resolved = audit.resolve_five_semantic_chains(_human_bones())

    assert resolved["chains"] == {
        "axial": ["bone_0", "bone_1", "bone_2", "bone_3", "bone_4"],
        "left_arm": ["bone_5", "bone_6", "bone_7", "bone_8"],
        "right_arm": ["bone_9", "bone_10", "bone_11", "bone_12"],
        "left_leg": ["bone_13", "bone_14", "bone_15", "bone_16"],
        "right_leg": ["bone_17", "bone_18", "bone_19", "bone_20"],
    }
    assert resolved["semantic_bones"]["pelvis"] == "bone_0"
    assert resolved["semantic_bones"]["head"] == "bone_4"
    assert resolved["semantic_bones"]["left_hand"] == "bone_8"
    assert resolved["semantic_bones"]["right_toe"] == "bone_20"
    assert resolved["side_basis"] == {
        "left": "positive-x",
        "right": "negative-x",
    }


def test_semantic_mapping_rejects_an_ambiguous_second_head_chain():
    bones = _human_bones() + (
        audit.BoneRecord("bone_21", "bone_3", (0.01, 0.0, 1.781)),
    )

    with pytest.raises(audit.StaticAuditError, match="ambiguous.*axial|axial.*ambiguous"):
        audit.resolve_five_semantic_chains(bones)


def test_semantic_mapping_clusters_proven_finger_descendants_below_each_hand():
    bones = list(_human_bones())
    for side, hand, sign in (("left", "bone_8", 1.0), ("right", "bone_12", -1.0)):
        hand_head = next(bone.head for bone in bones if bone.name == hand)
        for finger in range(5):
            bones.append(
                audit.BoneRecord(
                    f"{side}_finger_{finger}",
                    hand,
                    (
                        hand_head[0] + sign * (0.03 + 0.005 * finger),
                        hand_head[1] - 0.01 * finger,
                        hand_head[2] + 0.005 * (finger - 2),
                    ),
                )
            )

    resolved = audit.resolve_five_semantic_chains(tuple(bones))

    assert resolved["chains"]["left_arm"] == ["bone_5", "bone_6", "bone_7", "bone_8"]
    assert resolved["chains"]["right_arm"] == ["bone_9", "bone_10", "bone_11", "bone_12"]
    assert resolved["semantic_bones"]["left_hand"] == "bone_8"
    assert set(resolved["ignored_proven_distal_descendants"]) == {
        *(f"left_finger_{index}" for index in range(5)),
        *(f"right_finger_{index}" for index in range(5)),
    }


def test_semantic_mapping_ignores_only_short_descendants_of_a_top_head_anchor():
    bones = _human_bones() + (
        audit.BoneRecord("face_left", "bone_4", (0.03, -0.02, 1.79)),
        audit.BoneRecord("face_right", "bone_4", (-0.03, -0.02, 1.79)),
    )

    resolved = audit.resolve_five_semantic_chains(bones)

    assert resolved["chains"]["axial"][-1] == "bone_4"
    assert resolved["semantic_bones"]["head"] == "bone_4"
    assert set(resolved["ignored_proven_head_descendants"]) == {
        "face_left",
        "face_right",
    }


def _soft_t_tokenrig_bones():
    bones = [
        audit.BoneRecord(f"bone_{index}", None if index == 0 else f"bone_{index - 1}", head)
        for index, head in enumerate(
            (
                (0.0, 0.0, 0.47),
                (0.0, 0.0, 0.52),
                (0.0, 0.0, 0.59),
                (0.0, 0.0, 0.66),
                (0.0, 0.0, 0.74),
                (0.0, 0.0, 0.78),
            )
        )
    ]
    for start, sign in ((6, 1.0), (25, -1.0)):
        bones.extend(
            (
                audit.BoneRecord(f"bone_{start}", "bone_3", (0.04 * sign, 0.0, 0.73)),
                audit.BoneRecord(f"bone_{start + 1}", f"bone_{start}", (0.10 * sign, 0.0, 0.70)),
                audit.BoneRecord(f"bone_{start + 2}", f"bone_{start + 1}", (0.18 * sign, 0.0, 0.62)),
                audit.BoneRecord(f"bone_{start + 3}", f"bone_{start + 2}", (0.27 * sign, 0.0, 0.51)),
            )
        )
        for finger, offset in enumerate((4, 7, 10, 13, 16)):
            first = start + offset
            bones.extend(
                (
                    audit.BoneRecord(f"bone_{first}", f"bone_{start + 3}", (0.29 * sign, 0.01 * finger, 0.49)),
                    audit.BoneRecord(f"bone_{first + 1}", f"bone_{first}", (0.30 * sign, 0.01 * finger, 0.47)),
                    audit.BoneRecord(f"bone_{first + 2}", f"bone_{first + 1}", (0.31 * sign, 0.01 * finger, 0.45)),
                )
            )
    bones.extend(
        (
            audit.BoneRecord("bone_44", "bone_0", (0.05, 0.0, 0.44)),
            audit.BoneRecord("bone_45", "bone_44", (0.07, 0.0, 0.25)),
            audit.BoneRecord("bone_46", "bone_45", (0.07, -0.05, 0.07)),
            audit.BoneRecord("bone_47", "bone_46", (0.09, -0.12, 0.01)),
            audit.BoneRecord("bone_48", "bone_0", (-0.05, 0.0, 0.44)),
            audit.BoneRecord("bone_49", "bone_48", (-0.07, 0.0, 0.25)),
            audit.BoneRecord("bone_50", "bone_49", (-0.07, -0.05, 0.07)),
            audit.BoneRecord("bone_51", "bone_50", (-0.09, -0.12, 0.01)),
        )
    )
    return tuple(bones)


def test_semantic_mapping_classifies_soft_t_fingers_by_upper_axial_divergence():
    resolved = audit.resolve_five_semantic_chains(_soft_t_tokenrig_bones())

    assert resolved["chains"]["axial"] == [f"bone_{index}" for index in range(6)]
    assert resolved["chains"]["left_arm"] == [f"bone_{index}" for index in range(6, 10)]
    assert resolved["chains"]["right_arm"] == [f"bone_{index}" for index in range(25, 29)]
    assert resolved["semantic_bones"]["left_hand"] == "bone_9"
    assert resolved["semantic_bones"]["right_hand"] == "bone_28"
    assert len(resolved["ignored_proven_distal_descendants"]) == 30


def _valid_weights():
    return (
        {"bone_0": 1.0},
        {"bone_5": 0.5, "bone_6": 0.5},
        {"bone_13": 0.1, "bone_14": 0.2, "bone_15": 0.3, "bone_16": 0.4},
    )


def test_validates_one_to_four_finite_normalized_influences():
    result = audit.validate_vertex_weights(
        _valid_weights(), bone_names={bone.name for bone in _human_bones()}
    )

    assert result["vertex_count"] == 3
    assert result["maximum_influences"] == 4
    assert result["maximum_weight_sum_error"] <= audit.WEIGHT_SUM_TOLERANCE


@pytest.mark.parametrize(
    ("weights", "message"),
    (
        (({},), "zero-weight"),
        (({"bone_0": 0.2, "bone_1": 0.2, "bone_2": 0.2, "bone_3": 0.2, "bone_4": 0.2},), "more than four"),
        (({"bone_0": 0.7, "bone_1": 0.2},), "normalized"),
        (({"bone_0": math.nan},), "non-finite"),
        (({"missing": 1.0},), "unknown bone"),
    ),
)
def test_weight_gate_rejects_invalid_influences(weights, message):
    with pytest.raises(audit.StaticAuditError, match=message):
        audit.validate_vertex_weights(
            weights, bone_names={bone.name for bone in _human_bones()}
        )


def test_uv_seam_duplicates_must_have_identical_normalized_weights():
    positions = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    valid = ({"bone_0": 1.0}, {"bone_0": 1.0}, {"bone_1": 1.0})
    result = audit.validate_seam_weights(positions, valid)
    assert result["duplicate_position_group_count"] == 1
    assert result["maximum_weight_l1_error"] == 0.0

    invalid = ({"bone_0": 1.0}, {"bone_1": 1.0}, {"bone_1": 1.0})
    with pytest.raises(audit.StaticAuditError, match="seam.*weight"):
        audit.validate_seam_weights(positions, invalid)


def test_mesh_contract_comparison_rejects_uv_or_polygon_material_changes():
    expected = {
        "vertex_count": 4,
        "polygon_count": 2,
        "loop_count": 6,
        "material_slot_count": 1,
        "position_sha256": "a" * 64,
        "topology_sha256": "b" * 64,
        "uv_sha256": "c" * 64,
        "polygon_material_sha256": "d" * 64,
        "material_names": ["PixalPBR"],
    }
    assert audit.compare_mesh_contracts(expected, dict(expected))["passed"] is True

    renamed_only = dict(expected, material_names=["Material_0"])
    assert audit.compare_mesh_contracts(expected, renamed_only)["passed"] is True

    changed_uv = dict(expected, uv_sha256="e" * 64)
    with pytest.raises(audit.StaticAuditError, match="mesh/UV/material contract changed"):
        audit.compare_mesh_contracts(expected, changed_uv)

    changed_material = dict(expected, polygon_material_sha256="f" * 64)
    with pytest.raises(audit.StaticAuditError, match="mesh/UV/material contract changed"):
        audit.compare_mesh_contracts(expected, changed_material)


def test_mesh_contract_allows_only_proven_glb_vertex_serialization_splits():
    expected = {
        "vertex_count": 699120,
        "polygon_count": 976951,
        "loop_count": 2930853,
        "uv_layer_count": 1,
        "material_slot_count": 1,
        "position_sha256": "a" * 64,
        "topology_sha256": "b" * 64,
        "corner_position_sha256": "c" * 64,
        "corner_normal_sha256": "3" * 64,
        "unique_position_count": 690000,
        "unique_position_sha256": "4" * 64,
        "bounds_quantized": ((-10, -20, -30), (10, 20, 30)),
        "surface_area_m2": 1.23456789,
        "uv_sha256": "d" * 64,
        "polygon_material_sha256": "e" * 64,
    }
    serialized = dict(
        expected,
        vertex_count=706964,
        position_sha256="f" * 64,
        topology_sha256="1" * 64,
    )

    result = audit.compare_mesh_contracts(
        expected, serialized, allow_serialization_splits=True
    )

    assert result["passed"] is True
    assert result["serialized_vertex_count_change"] == 7844
    changed_area = dict(serialized, surface_area_m2=1.3)
    with pytest.raises(audit.StaticAuditError, match="surface area"):
        audit.compare_mesh_contracts(
            expected, changed_area, allow_serialization_splits=True
        )


def _surface_reference(*, corner_x: float = 1.0, normal_z: float = 1.0):
    return audit.SurfaceReference(
        polygon_loop_counts=(3,),
        polygon_material_indices=(0,),
        corner_unique_indices=(0, 1, 2),
        corner_positions=(0.0, 0.0, 0.0, corner_x, 0.0, 0.0, 0.0, 1.0, 0.0),
        corner_normals=(0.0, 0.0, normal_z) * 3,
        uv_layers=((0.0, 0.0, 1.0, 0.0, 0.0, 1.0),),
        unique_positions=(0.0, 0.0, 0.0, corner_x, 0.0, 0.0, 0.0, 1.0, 0.0),
        bounds=((0.0, 0.0, 0.0), (corner_x, 1.0, 0.0)),
        surface_area_m2=0.5 * corner_x,
    )


def test_surface_reference_is_tolerance_aware_but_rejects_geometry_or_normal_changes():
    expected = _surface_reference()
    serialized = _surface_reference(corner_x=1.0 + 2.0e-7, normal_z=1.0 - 2.0e-7)

    result = audit.compare_surface_references(expected, serialized)

    assert result["passed"] is True
    assert result["maximum_corner_position_error_m"] <= 2.0e-6
    with pytest.raises(audit.StaticAuditError, match="position"):
        audit.compare_surface_references(expected, _surface_reference(corner_x=1.001))
    with pytest.raises(audit.StaticAuditError, match="corner normal"):
        audit.compare_surface_references(expected, _surface_reference(normal_z=0.9))


def test_surface_reference_compares_duplicate_face_multisets_without_unique_key_assumption():
    single = _surface_reference()
    duplicated = audit.SurfaceReference(
        polygon_loop_counts=(3, 3),
        polygon_material_indices=(0, 0),
        corner_unique_indices=(0, 1, 2, 0, 1, 2),
        corner_positions=single.corner_positions * 2,
        corner_normals=single.corner_normals * 2,
        uv_layers=(single.uv_layers[0] * 2,),
        unique_positions=single.unique_positions,
        bounds=single.bounds,
        surface_area_m2=single.surface_area_m2 * 2,
    )

    result = audit.compare_surface_references(duplicated, duplicated)

    assert result["passed"] is True
    assert result["polygon_count"] == 2


def test_surface_reference_matches_reversed_duplicate_choice_by_corner_identity():
    expected = _surface_reference()
    actual = audit.SurfaceReference(
        polygon_loop_counts=(3,),
        polygon_material_indices=(0,),
        corner_unique_indices=(0, 2, 1),
        corner_positions=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        corner_normals=(0.0, 0.0, 1.0) * 3,
        uv_layers=((0.0, 0.0, 0.0, 1.0, 1.0, 0.0),),
        unique_positions=expected.unique_positions,
        bounds=expected.bounds,
        surface_area_m2=expected.surface_area_m2,
    )

    assert audit.compare_surface_references(expected, actual)["passed"] is True


def test_surface_reference_allows_near_duplicate_import_splits_with_full_coverage():
    expected = _surface_reference()
    actual = audit.SurfaceReference(
        polygon_loop_counts=(3,),
        polygon_material_indices=(0,),
        corner_unique_indices=(0, 3, 2),
        corner_positions=(0.0, 0.0, 0.0, 1.0 + 2.0e-7, 0.0, 0.0, 0.0, 1.0, 0.0),
        corner_normals=expected.corner_normals,
        uv_layers=expected.uv_layers,
        unique_positions=(
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0 + 2.0e-7,
            0.0,
            0.0,
        ),
        bounds=((0.0, 0.0, 0.0), (1.0 + 2.0e-7, 1.0, 0.0)),
        surface_area_m2=0.5 + 1.0e-7,
    )

    result = audit.compare_surface_references(expected, actual)

    assert result["passed"] is True
    assert result["serialized_unique_position_count_change"] == 1


def test_surface_reference_allows_near_duplicate_reimport_merges_with_full_coverage():
    actual = _surface_reference()
    expected = audit.SurfaceReference(
        polygon_loop_counts=(3,),
        polygon_material_indices=(0,),
        corner_unique_indices=(0, 3, 2),
        corner_positions=(0.0, 0.0, 0.0, 1.0 + 2.0e-7, 0.0, 0.0, 0.0, 1.0, 0.0),
        corner_normals=actual.corner_normals,
        uv_layers=actual.uv_layers,
        unique_positions=(
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0 + 2.0e-7,
            0.0,
            0.0,
        ),
        bounds=((0.0, 0.0, 0.0), (1.0 + 2.0e-7, 1.0, 0.0)),
        surface_area_m2=0.5 + 1.0e-7,
    )

    result = audit.compare_surface_references(expected, actual)

    assert result["passed"] is True
    assert result["serialized_unique_position_count_change"] == -1


def test_position_cluster_match_rejects_ambiguous_multi_match():
    with pytest.raises(audit.StaticAuditError, match="ambiguous"):
        audit._match_unique_positions(
            (0.0, 0.0, 0.0, 3.0e-6, 0.0, 0.0),
            (1.5e-6, 0.0, 0.0, 3.0e-6, 0.0, 0.0),
        )


def test_distal_vertices_reject_opposite_limb_contamination():
    chains = audit.resolve_five_semantic_chains(_human_bones())["chains"]
    positions = ((0.8, 0.0, 1.3), (-0.8, 0.0, 1.3), (0.0, 0.0, 1.2))
    valid = (
        {"bone_8": 1.0},
        {"bone_12": 1.0},
        {"bone_1": 1.0},
    )
    result = audit.validate_bilateral_contamination(positions, valid, chains)
    assert result["contaminated_vertex_count"] == 0

    invalid = (
        {"bone_8": 0.9, "bone_12": 0.1},
        {"bone_12": 1.0},
        {"bone_1": 1.0},
    )
    with pytest.raises(audit.StaticAuditError, match="opposite-limb contamination"):
        audit.validate_bilateral_contamination(positions, invalid, chains)


def test_roundtrip_skin_comparison_is_position_keyed_and_seam_safe():
    expected_positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    expected_weights = ({"bone_0": 1.0}, {"bone_1": 1.0})
    serialized_positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    serialized_weights = (
        {"bone_0": 1.0},
        {"bone_1": 1.0},
        {"bone_1": 1.0},
    )

    result = audit.compare_skin_by_position(
        expected_positions,
        expected_weights,
        serialized_positions,
        serialized_weights,
    )
    assert result["passed"] is True
    assert result["serialized_vertex_count_change"] == 1

    changed = list(serialized_weights)
    changed[1] = {"bone_0": 1.0}
    with pytest.raises(audit.StaticAuditError, match="roundtrip skin weights changed"):
        audit.compare_skin_by_position(
            expected_positions,
            expected_weights,
            serialized_positions,
            tuple(changed),
        )


def test_roundtrip_skin_comparison_uses_tolerance_clusters_not_rounding_cells():
    expected_positions = ((0.9e-6, 0.0, 0.0), (1.0, 0.0, 0.0))
    actual_positions = ((1.1e-6, 0.0, 0.0), (1.0, 0.0, 0.0))
    weights = ({"bone_0": 1.0}, {"bone_1": 1.0})

    result = audit.compare_skin_by_position(
        expected_positions, weights, actual_positions, weights
    )

    assert result["passed"] is True
    assert result["maximum_position_error_m"] == pytest.approx(0.2e-6)


def test_roundtrip_skin_rejects_inconsistent_weights_inside_tolerance_cluster():
    expected_positions = ((0.0, 0.0, 0.0), (1.0e-6, 0.0, 0.0))
    expected_weights = ({"bone_0": 1.0}, {"bone_1": 1.0})

    with pytest.raises(audit.StaticAuditError, match="tolerance cluster"):
        audit.compare_skin_by_position(
            expected_positions,
            expected_weights,
            ((0.5e-6, 0.0, 0.0),),
            ({"bone_0": 1.0},),
        )


def _identity_matrix():
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_only_proven_identity_orphan_empties_may_be_removed_from_runtime_scene():
    world = SimpleNamespace(
        name="world",
        type="EMPTY",
        parent=None,
        children=(),
        data=None,
        matrix_world=_identity_matrix(),
    )

    assert audit.validate_proven_runtime_orphans((world,)) == (
        {
            "name": "world",
            "type": "EMPTY",
            "reason": "finite_identity_childless_dataless_root",
        },
    )

    nonidentity = SimpleNamespace(**vars(world))
    nonidentity.matrix_world = tuple(
        tuple(2.0 if row == column == 0 else value for column, value in enumerate(values))
        for row, values in enumerate(_identity_matrix())
    )
    with pytest.raises(audit.StaticAuditError, match="identity"):
        audit.validate_proven_runtime_orphans((nonidentity,))

    parented = SimpleNamespace(**vars(world))
    parented.parent = object()
    with pytest.raises(audit.StaticAuditError, match="root|parent"):
        audit.validate_proven_runtime_orphans((parented,))

    camera = SimpleNamespace(**vars(world))
    camera.type = "CAMERA"
    with pytest.raises(audit.StaticAuditError, match="non-runtime scene object"):
        audit.validate_proven_runtime_orphans((camera,))


def test_only_the_exact_hidden_gltf_icosphere_helper_collection_may_be_removed():
    collection = SimpleNamespace(
        name="glTF_not_exported",
        hide_render=True,
        hide_viewport=True,
    )
    helper = SimpleNamespace(
        name="Icosphere",
        type="MESH",
        parent=None,
        children=(),
        matrix_world=_identity_matrix(),
        data=SimpleNamespace(vertices=range(42), polygons=range(80)),
        users_collection=(collection,),
    )
    collection.objects = (helper,)

    assert audit.validate_gltf_import_helper_collection(collection) == {
        "collection": "glTF_not_exported",
        "object": "Icosphere",
        "vertex_count": 42,
        "polygon_count": 80,
        "reason": "blender_gltf_generated_nonexported_joint_shape",
    }

    collection.hide_render = False
    with pytest.raises(audit.StaticAuditError, match="hidden"):
        audit.validate_gltf_import_helper_collection(collection)
    collection.hide_render = True
    helper.data = SimpleNamespace(vertices=range(43), polygons=range(80))
    with pytest.raises(audit.StaticAuditError, match="42.*80|shape"):
        audit.validate_gltf_import_helper_collection(collection)


def test_atomic_directory_publication_is_no_replace(tmp_path):
    staged = tmp_path / ".static_audit.staging"
    destination = tmp_path / "static_audit_v1"
    staged.mkdir()
    (staged / "static_qa.json").write_text("{}\n", encoding="utf-8")

    audit.rename_directory_noreplace(staged, destination)
    assert destination.is_dir()
    assert not staged.exists()

    second = tmp_path / ".second.staging"
    second.mkdir()
    with pytest.raises(audit.StaticAuditError, match="already exists|no-replace"):
        audit.rename_directory_noreplace(second, destination)
    assert second.is_dir()


def test_failed_attempt_writes_immutable_evidence_without_readiness_bundle(tmp_path):
    output_dir = tmp_path / "static_audit_v1"
    evidence = audit.write_failure_evidence(
        output_dir=output_dir,
        asset_id="rocketbox_male_adult_01",
        error=audit.StaticAuditError("ambiguous arm chains"),
        authenticated={"tokenrig_glb_sha256": "a" * 64},
    )

    assert not output_dir.exists()
    assert evidence.parent == tmp_path
    assert evidence.name.startswith("static_audit_v1.failed.")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["schema"] == "tokenrig_human_static_attempt_v1"
    assert payload["decision"] == "rejected"
    assert payload["readiness_bundle_published"] is False
    assert payload["agent_qa_status"] == "rejected"
    assert payload["failure"]["message"] == "ambiguous arm chains"
    assert evidence.stat().st_mode & 0o222 == 0


def _module_source() -> str:
    return (Path(__file__).resolve().parents[2] / "tools" / "blender_tokenrig_human_static_audit.py").read_text(
        encoding="utf-8"
    )


def _function_source(name: str) -> str:
    source = _module_source()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            value = ast.get_source_segment(source, node)
            assert value is not None
            return value
    raise AssertionError(f"missing function {name}")


def test_module_is_import_safe_and_exposes_the_pinned_task4_cli():
    tree = ast.parse(_module_source())
    top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all(
        not (
            isinstance(node, ast.Import) and any(alias.name == "bpy" for alias in node.names)
        )
        for node in top_level_imports
    )
    parser = _function_source("parse_args")
    for option in (
        "--asset-id",
        "--source-glb",
        "--tokenrig-glb",
        "--tokenrig-manifest",
        "--output-dir",
    ):
        assert option in parser
    assert "static_audit_v1" in _module_source()


def test_blender_path_clears_scene_and_transforms_runtime_closure_roots_once():
    source = _function_source("run_blender_audit")

    assert source.count("bpy.ops.wm.read_factory_settings(use_empty=True)") >= 2
    assert "bpy.ops.import_scene.gltf" in source
    assert "identify_exact_runtime" in source
    assert "remove_gltf_import_helpers" in source
    assert "remove_proven_runtime_orphans" in source
    assert "runtime_roots" in source
    assert "Matrix.Rotation(math.pi, 4, \"Z\")" in source
    assert "Matrix.Translation((0.0, 0.0, grounding[\"ground_translation_z\"]))" in source
    assert "root.matrix_world = canonical_ground @ root.matrix_world" in source
    assert "mesh.matrix_world = canonical_ground" not in source
    assert "post_floor_z" in source


def test_blender_path_requires_complete_evidence_and_skin_uv_roundtrip():
    assert audit.REQUIRED_BUNDLE_FILES == (
        "bind_pose.glb",
        "bind_front.png",
        "bind_back.png",
        "bind_side.png",
        "bind_top.png",
        "skeleton_overlay.png",
        "weights_contact.png",
        "texture_compare.png",
        "joint_hierarchy.txt",
        "static_qa.json",
    )
    export = _function_source("export_bind_pose_glb")
    for value in (
        'export_format="GLB"',
        "use_selection=True",
        "export_animations=False",
        "export_skins=True",
        "export_texcoords=True",
        "export_normals=True",
    ):
        assert value in export
    roundtrip = _function_source("roundtrip_validate_bind")
    assert "read_factory_settings(use_empty=True)" in roundtrip
    assert "import_scene.gltf" in roundtrip
    for helper in (
        "compare_mesh_contracts",
        "compare_pbr_payloads",
        "validate_vertex_weights",
        "validate_seam_weights",
        "resolve_five_semantic_chains",
    ):
        assert helper in roundtrip
    assert "allow_serialization_splits=True" in roundtrip


def test_publication_is_staged_and_never_claims_user_approval():
    source = _module_source()
    runner = _function_source("run_static_audit")
    assert "tempfile.mkdtemp" in runner
    assert "rename_directory_noreplace" in runner
    assert "write_failure_evidence" in runner
    assert "shutil.rmtree" in runner
    assert "tokenrig_failed_attempt" in runner
    assert "pending_agent_visual_qa" in source
    assert "bounded_serialization_equivalence_not_exact" in source
    assert "prior_strict_failure_evidence" in source
    assert "backface_cull_risk" in source
    assert "user_approved" not in source
