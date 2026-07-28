from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import tempfile

import pytest

from tools import audit_quadruped_i23d_geometry as geometry_audit
from tools import generated_animal_tokenrig_closure as closure
from tools import publish_generated_animal_geometry_closure as geometry_closures
from tools import register_controlled_animal_source_assets as source_registry


def test_closure_validation_remains_compatible_with_python_39() -> None:
    source = Path(closure.__file__).read_text(encoding="utf-8")

    assert "zip(extra_records, extra_upstream, strict=True)" not in source


def descriptor(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": closure.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_bytes(closure.canonical_bytes(payload) + b"\n")


def write_production_proxy_glb(path: Path, *, shift: float = 0.0) -> None:
    vertices = [
        (0.0 + shift, 0.0, 0.0),
        (1.0 + shift, 0.0, 0.0),
        (0.0 + shift, 1.0, 0.0),
        (0.0 + shift, 0.0, 1.0),
    ]
    normals = [(0.0, 1.0, 0.0)] * 4
    uvs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    indices = (0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3)
    chunks = [
        struct.pack("<12f", *(item for vertex in vertices for item in vertex)),
        struct.pack("<12f", *(item for normal in normals for item in normal)),
        struct.pack("<8f", *(item for uv in uvs for item in uv)),
        struct.pack("<12H", *indices),
        b"\x89PNG\r\n\x1a\n",
        b"\x89PNG\r\n\x1a\n",
    ]
    offsets = []
    binary = b""
    for chunk in chunks:
        offsets.append(len(binary))
        binary += chunk
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": offsets[0],
                "byteLength": len(chunks[0]),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": offsets[1],
                "byteLength": len(chunks[1]),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": offsets[2],
                "byteLength": len(chunks[2]),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": offsets[3],
                "byteLength": len(chunks[3]),
                "target": 34963,
            },
            {
                "buffer": 0,
                "byteOffset": offsets[4],
                "byteLength": len(chunks[4]),
            },
            {
                "buffer": 0,
                "byteOffset": offsets[5],
                "byteLength": len(chunks[5]),
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [shift, 0.0, 0.0],
                "max": [1.0 + shift, 1.0, 1.0],
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 4,
                "type": "VEC2",
            },
            {
                "bufferView": 3,
                "componentType": 5123,
                "count": 12,
                "type": "SCALAR",
            },
        ],
        "images": [
            {
                "bufferView": 4,
                "mimeType": "image/png",
                "name": "Watertight_BaseColor",
            },
            {
                "bufferView": 5,
                "mimeType": "image/png",
                "name": "Watertight_Roughness",
            },
        ],
        "samplers": [{"magFilter": 9729, "minFilter": 9987}],
        "textures": [
            {"sampler": 0, "source": 0},
            {"sampler": 0, "source": 1},
        ],
        "materials": [
            {
                "doubleSided": True,
                "name": "Watertight_Baked_PBR",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0,
                    "metallicRoughnessTexture": {"index": 1},
                },
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
                        },
                        "indices": 3,
                        "material": 0,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0, "name": "Watertight_Runtime_Proxy"}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    encoded = json.dumps(
        document,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        b"".join(
            (
                struct.pack("<4sII", b"glTF", 2, total),
                struct.pack("<II", len(encoded), 0x4E4F534A),
                encoded,
                struct.pack("<II", len(binary), 0x004E4942),
                binary,
            )
        )
    )


def proxy_geometry_audit(path: Path) -> dict:
    topology = {
        "imported_vertices": 4,
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
        "coordinate_frame": "gltf_positive_x_forward_positive_y_up_positive_z_side",
        "central_longitudinal_percentiles": [25.0, 75.0],
        "torso_floor_fraction_of_robust_height": 0.35,
        "surface_side_percentiles": [5.0, 95.0],
        "section_count": 8,
        "selected_vertex_count": 4,
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
                "label": "fixture_production_proxy",
                "mesh": {
                    "absolute_path": str(path.resolve()),
                    "sha256": closure.sha256_file(path),
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


def valid_readback(path: Path, source: Path, rig: Path) -> dict:
    payload = {
        "schema": closure.READBACK_SCHEMA,
        "tokenrig_input": descriptor(source),
        "tokenrig_output": descriptor(rig),
        "automatic_checks": {
            "overall": "passed",
            "world_geometry_unchanged": True,
            "logical_vertices_bijective": True,
            "triangles_bijective": True,
            "uvs_unchanged": True,
            "pbr_unchanged": True,
            "exactly_one_output_skin": True,
            "exactly_one_output_armature": True,
            "no_animation": True,
        },
        "formal_dataset_registration_authorized": False,
    }
    payload["manifest_sha256"] = closure.hash_without(payload, "manifest_sha256")
    write_json(path, payload)
    return payload


def test_same_workspace_same_bytes_cannot_replace_expected_glb(tmp_path):
    expected = tmp_path / "expected.glb"
    replacement = tmp_path / "replacement.glb"
    expected.write_bytes(b"same GLB bytes")
    replacement.write_bytes(expected.read_bytes())
    record = descriptor(replacement)

    with pytest.raises(closure.ClosureError, match="different file"):
        closure.require_descriptor_matches(record, expected, "expected target rig GLB")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sha256", "0" * 64, "SHA-256 mismatch"),
        ("size_bytes", 1, "size mismatch"),
    ],
)
def test_input_output_descriptor_hash_and_size_mutations_fail(
    tmp_path, field, value, match
):
    artifact = tmp_path / "artifact.glb"
    artifact.write_bytes(b"authenticated GLB")
    record = descriptor(artifact)
    record[field] = value

    with pytest.raises(closure.ClosureError, match=match):
        closure.descriptor_path(record, "TokenRig input/output")


def execution_pair():
    argv = [
        "demo.py",
        "--input",
        "/asset/source.glb",
        "--output",
        "/asset/rig.glb",
    ]
    model = {
        "invocation_path": "/model/checkpoint",
        "symlink_chain": [],
        "snapshot_revision": "1" * 40,
        "resolved_payload": {
            "path": "/model/blob",
            "sha256": "2" * 64,
            "size_bytes": 100,
        },
    }
    reconstructed = {
        "exact_argv": argv,
        "exact_argv_sha256": closure.sha256_bytes(closure.canonical_bytes(argv)),
        "environment": {"TOKENRIG_CANARY_SEED": "42"},
        "patch_sha256": "3" * 64,
        "model_checkpoint": model,
    }
    execution = {
        "seed": 42,
        "exact_argv": deepcopy(argv),
        "exact_argv_sha256": reconstructed["exact_argv_sha256"],
        "environment": deepcopy(reconstructed["environment"]),
        "runtime_patch_sha256": reconstructed["patch_sha256"],
        "model_checkpoint": deepcopy(model),
    }
    return execution, reconstructed


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value.__setitem__("seed", 43),
            "seed is not exactly 42",
        ),
        (
            lambda value: value["exact_argv"].__setitem__(-1, "/asset/other.glb"),
            "argv",
        ),
        (
            lambda value: value["model_checkpoint"]["resolved_payload"].__setitem__(
                "sha256", "4" * 64
            ),
            "model identity",
        ),
        (
            lambda value: value.__setitem__("runtime_patch_sha256", "5" * 64),
            "runtime patch",
        ),
    ],
)
def test_seed_argv_model_and_patch_mutations_fail_closed(mutate, match):
    execution, reconstructed = execution_pair()
    mutate(execution)

    with pytest.raises(closure.ClosureError, match=match):
        closure.validate_execution_identity_fields(execution, reconstructed)


def test_upstream_raw_pixal_link_cannot_be_redirected_to_same_bytes(tmp_path):
    raw = tmp_path / "pixal_raw.glb"
    raw.write_bytes(b"raw pixal")
    raw_clone = tmp_path / "pixal_raw_clone.glb"
    raw_clone.write_bytes(raw.read_bytes())
    tokenrig_input = tmp_path / "watertight.glb"
    tokenrig_input.write_bytes(b"watertight")
    raw_manifest = tmp_path / "raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(raw),
                "bytes": raw.stat().st_size,
            },
        },
    )
    upstream_manifest = tmp_path / "watertight.manifest.json"
    upstream = {
        "schema": "avengine_watertight_textured_runtime_proxy_v1",
        "input": descriptor(raw),
        "output": descriptor(tokenrig_input),
        "authority_contract": {
            "approved_skeleton_or_animation_touched": False,
        },
    }
    write_json(upstream_manifest, upstream)
    observed_raw, kind, extra = closure.validate_upstream_lineage(
        raw_manifest, upstream_manifest, tokenrig_input
    )
    assert os.path.samefile(observed_raw, raw)
    assert kind == "watertight_runtime_proxy"
    assert extra == []

    upstream["input"] = descriptor(raw_clone)
    write_json(upstream_manifest, upstream)
    with pytest.raises(closure.ClosureError, match="different file"):
        closure.validate_upstream_lineage(
            raw_manifest, upstream_manifest, tokenrig_input
        )


def _bounded_upstream_fixture(tmp_path):
    raw = tmp_path / "pixal_raw.glb"
    raw.write_bytes(b"raw Pixel3D")
    tokenrig_input = tmp_path / "bounded.glb"
    tokenrig_input.write_bytes(b"bounded Pixel3D")
    raw_manifest = tmp_path / "raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(raw),
                "bytes": raw.stat().st_size,
            },
        },
    )
    repair_path = tmp_path / "repair.json"
    write_json(repair_path, {"fixture": True})
    upstream_path = tmp_path / "geometry_closure.json"
    write_json(
        upstream_path,
        {
            "schema": "avengine_generated_animal_geometry_closure_v1",
            "status": "pass_geometry_only",
            "candidate": {"source_pixal_glb": descriptor(raw)},
            "output": {
                "glb": descriptor(tokenrig_input),
                "repair_manifest": descriptor(repair_path),
            },
            "downstream": {"tokenrig_entry_authorized": True},
        },
    )
    return raw, raw_manifest, tokenrig_input, repair_path, upstream_path


def _direct_copy_v2_fixture(tmp_path, monkeypatch):
    historical_raw = tmp_path / "original" / "pixal_raw.glb"
    historical_raw.parent.mkdir()
    historical_raw.write_bytes(b"direct-adopted Pixel3D")
    adopted_raw = tmp_path / "adopted" / "pixal_raw.glb"
    adopted_raw.parent.mkdir()
    adopted_raw.write_bytes(historical_raw.read_bytes())
    tokenrig_input = tmp_path / "bounded.glb"
    write_production_proxy_glb(tokenrig_input)
    raw_manifest = tmp_path / "adopted" / "pixal_raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(historical_raw),
                "bytes": historical_raw.stat().st_size,
            },
        },
    )
    upstream = tmp_path / "geometry_closure_v2.json"
    geometry_payload = {
        "schema": "avengine_generated_animal_geometry_closure_v2"
    }
    geometry_payload["manifest_sha256"] = closure.hash_without(
        geometry_payload,
        "manifest_sha256",
    )
    write_json(upstream, geometry_payload)
    repair = tmp_path / "repair.json"
    repair.write_bytes(b"repair")
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"audit")
    decision_batch = tmp_path / "raw_decision_batch.json"
    decision_batch.write_bytes(b"decision batch")
    observed: dict[str, object] = {}

    def load_geometry_closure(path, **expected):
        observed["path"] = path
        observed["expected"] = expected
        return {
            "paths": {
                "raw_pixal_glb": adopted_raw,
                "repair_manifest": repair,
                "geometry_audit": audit,
                "raw_static_decision_batch": decision_batch,
            }
        }

    monkeypatch.setattr(
        geometry_closures,
        "load_geometry_closure_v2",
        load_geometry_closure,
    )
    return {
        "historical_raw": historical_raw,
        "adopted_raw": adopted_raw,
        "raw_manifest": raw_manifest,
        "tokenrig_input": tokenrig_input,
        "upstream": upstream,
        "geometry_manifest_sha256": geometry_payload["manifest_sha256"],
        "repair": repair,
        "audit": audit,
        "decision_batch": decision_batch,
        "observed": observed,
    }


def _bounded_watertight_v2_fixture(tmp_path, monkeypatch):
    case = _direct_copy_v2_fixture(tmp_path, monkeypatch)
    repaired = case["tokenrig_input"]
    proxy = tmp_path / "watertight_proxy.glb"
    write_production_proxy_glb(proxy, shift=0.0001)
    proxy_manifest = tmp_path / "watertight_proxy.manifest.json"
    write_json(
        proxy_manifest,
        {
            "schema": "avengine_watertight_textured_runtime_proxy_v1",
            "created_at": "2026-07-28T00:00:00+00:00",
            "input": descriptor(repaired),
            "attribute_input": {
                **descriptor(repaired),
                "same_as_geometry_input": True,
            },
            "output": descriptor(proxy),
            "parameters": {
                "voxel_resolution": 200,
                "voxel_size": 0.01,
                "target_faces": 100000,
                "smooth_iterations": 1,
                "shrinkwrap_strength": 0.0,
                "post_shrinkwrap_smooth_iterations": 2,
                "torso_fold_repair_iterations": 6,
                "double_sided": False,
                "attribute_transfer_backend": "bake",
                "bake_resolution": 2048,
                "base_color_encoding_policy": "preserve-bake",
                "base_color_gain": [1.0, 1.0, 1.0],
            },
            "topology": {
                "source": {
                    "vertices": 4,
                    "edges": 6,
                    "faces": 4,
                    "boundary_edges": 0,
                    "wire_edges": 0,
                    "nonmanifold_edges_over_two_faces": 0,
                    "noncontiguous_two_face_edges": 0,
                },
                "after_voxel_remesh": {
                    "vertices": 4,
                    "edges": 4,
                    "faces": 2,
                    "boundary_edges": 0,
                    "wire_edges": 0,
                    "nonmanifold_edges_over_two_faces": 0,
                    "noncontiguous_two_face_edges": 0,
                },
                "final": {
                    "vertices": 4,
                    "edges": 4,
                    "faces": 2,
                    "boundary_edges": 0,
                    "wire_edges": 0,
                    "nonmanifold_edges_over_two_faces": 0,
                    "noncontiguous_two_face_edges": 0,
                },
            },
            "surface_attributes": {
                "backend": "bake",
                "bake_resolution": 2048,
                "bake_device": "CPU",
                "ray_distance": 0.03,
                "cage_extrusion": 0.005,
                "uv_layers": ["UVMap"],
                "baked_images": [
                    "Watertight_BaseColor",
                    "Watertight_Roughness",
                ],
                "base_color_bake_type": (
                    "EMIT_FROM_PRINCIPLED_BASE_COLOR"
                ),
                "color_attributes": [],
                "material_slots": ["Watertight_Baked_PBR"],
                "metallic_policy": (
                    "constant_zero_for_nonmetallic_animal_surface"
                ),
                "base_color_encoding_policy": "preserve-bake",
                "base_color_gain": [1.0, 1.0, 1.0],
            },
            "torso_fold_repair": {
                "iterations": 6,
                "selected_vertices": 2,
                "longitudinal_axis": 0,
                "normalized_longitudinal_range": [0.25, 0.7],
                "normalized_vertical_range": [0.34, 0.72],
                "fade": 0.08,
                "lambda_factor": 0.18,
                "policy": "weighted_mid_torso_only_preserve_volume",
            },
            "authority_contract": {
                "attribute_source_pbr_material_reused": False,
                "attribute_source_uvs_transferred_by_nearest_surface": False,
                "attribute_source_pbr_baked_to_new_uv_atlas": True,
                "full_resolution_source_remains_geometry_authority": True,
                "source_geometry_replaced": True,
                "approved_skeleton_or_animation_touched": False,
            },
            "actual_faces": 2,
            "status": "research_candidate_pending_static_and_animation_qa",
            "formal_dataset_registration_authorized": False,
        },
    )
    proxy_audit = tmp_path / "proxy_geometry_audit.json"
    write_json(proxy_audit, proxy_geometry_audit(proxy))

    def load_geometry_closure(path, **expected):
        case["observed"]["path"] = path
        case["observed"]["expected"] = expected
        try:
            repaired_matches = os.path.samefile(
                expected["expected_repaired_glb"], repaired
            )
            batch_matches = os.path.samefile(
                expected["expected_raw_static_decision_batch"],
                case["decision_batch"],
            )
        except OSError:
            repaired_matches = False
            batch_matches = False
        if not repaired_matches:
            raise geometry_closures.GeometryClosureError(
                "expected repaired GLB identity changed"
            )
        if not batch_matches:
            raise geometry_closures.GeometryClosureError(
                "expected raw decision batch identity changed"
            )
        return {
            "manifest": closure.load_json(
                case["upstream"], "fixture geometry closure"
            ),
            "paths": {
                "raw_pixal_glb": case["adopted_raw"],
                "repair_manifest": case["repair"],
                "geometry_audit": case["audit"],
                "raw_static_decision_batch": case["decision_batch"],
            }
        }

    monkeypatch.setattr(
        geometry_closures,
        "load_geometry_closure_v2",
        load_geometry_closure,
    )
    case.update(
        {
            "repaired": repaired,
            "proxy": proxy,
            "proxy_manifest": proxy_manifest,
            "proxy_audit": proxy_audit,
        }
    )
    return case


def _validate_bounded_watertight(case, **overrides):
    arguments = {
        "bounded_geometry_closure_path": case["upstream"],
        "expected_bounded_geometry_closure_sha256": closure.sha256_file(
            case["upstream"]
        ),
        "expected_bounded_geometry_closure_manifest_sha256": case[
            "geometry_manifest_sha256"
        ],
        "expected_watertight_proxy_manifest_sha256": closure.sha256_file(
            case["proxy_manifest"]
        ),
        "watertight_proxy_geometry_audit_path": case["proxy_audit"],
        "expected_watertight_proxy_geometry_audit_sha256": closure.sha256_file(
            case["proxy_audit"]
        ),
        "raw_static_decision_batch_path": case["decision_batch"],
        "expected_raw_static_decision_batch_sha256": closure.sha256_file(
            case["decision_batch"]
        ),
        "require_oriented_batch_external_authority": True,
    }
    arguments.update(overrides)
    return closure.validate_upstream_lineage(
        case["raw_manifest"],
        case["proxy_manifest"],
        case["proxy"],
        **arguments,
    )


def test_bounded_geometry_v2_can_strictly_feed_watertight_proxy_then_tokenrig(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)
    case["historical_raw"].unlink()

    observed_raw, kind, extra = _validate_bounded_watertight(case)

    assert os.path.samefile(observed_raw, case["adopted_raw"])
    assert kind == "bounded_watertight_runtime_proxy"
    assert extra == [
        case["upstream"].resolve(),
        case["proxy_audit"].resolve(),
        case["repair"],
        case["audit"],
        case["decision_batch"],
    ]
    assert case["observed"]["path"] == case["upstream"].resolve()
    expected = case["observed"]["expected"]
    assert expected["expected_manifest_sha256"] == closure.sha256_file(
        case["upstream"]
    )
    assert (
        case["geometry_manifest_sha256"]
        != closure.sha256_file(case["upstream"])
    )
    assert expected["expected_pixal_manifest"] == case["raw_manifest"]
    assert os.path.samefile(expected["expected_repaired_glb"], case["repaired"])
    assert os.path.samefile(
        expected["expected_raw_static_decision_batch"],
        case["decision_batch"],
    )


def test_bounded_watertight_rejects_proxy_from_wrong_repaired_glb(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)
    wrong_repaired = tmp_path / "wrong_repaired.glb"
    wrong_repaired.write_bytes(case["repaired"].read_bytes())
    proxy_manifest = closure.load_json(
        case["proxy_manifest"], "watertight proxy manifest"
    )
    proxy_manifest["input"] = descriptor(wrong_repaired)
    proxy_manifest["attribute_input"] = {
        **descriptor(wrong_repaired),
        "same_as_geometry_input": True,
    }
    write_json(case["proxy_manifest"], proxy_manifest)

    with pytest.raises(
        closure.ClosureError,
        match="strict replay failed.*repaired GLB identity changed",
    ):
        _validate_bounded_watertight(case)


def test_bounded_watertight_rejects_wrong_raw_decision_batch(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)
    wrong_batch = tmp_path / "wrong_raw_decision_batch.json"
    wrong_batch.write_bytes(case["decision_batch"].read_bytes())

    with pytest.raises(
        closure.ClosureError,
        match="strict replay failed.*decision batch identity changed",
    ):
        _validate_bounded_watertight(
            case,
            raw_static_decision_batch_path=wrong_batch,
            expected_raw_static_decision_batch_sha256=closure.sha256_file(
                wrong_batch
            ),
        )


def test_bounded_watertight_rejects_arbitrary_tokenrig_proxy_same_bytes(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)
    arbitrary_proxy = tmp_path / "arbitrary_proxy.glb"
    arbitrary_proxy.write_bytes(case["proxy"].read_bytes())

    with pytest.raises(closure.ClosureError, match="different file"):
        closure.validate_upstream_lineage(
            case["raw_manifest"],
            case["proxy_manifest"],
            arbitrary_proxy,
            bounded_geometry_closure_path=case["upstream"],
            expected_bounded_geometry_closure_sha256=closure.sha256_file(
                case["upstream"]
            ),
            expected_bounded_geometry_closure_manifest_sha256=case[
                "geometry_manifest_sha256"
            ],
            expected_watertight_proxy_manifest_sha256=closure.sha256_file(
                case["proxy_manifest"]
            ),
            watertight_proxy_geometry_audit_path=case["proxy_audit"],
            expected_watertight_proxy_geometry_audit_sha256=closure.sha256_file(
                case["proxy_audit"]
            ),
            raw_static_decision_batch_path=case["decision_batch"],
            expected_raw_static_decision_batch_sha256=closure.sha256_file(
                case["decision_batch"]
            ),
            require_oriented_batch_external_authority=True,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["input"].__setitem__("sha256", "0" * 64),
            "SHA-256 mismatch",
        ),
        (
            lambda payload: payload["output"].__setitem__("size_bytes", 1),
            "size mismatch",
        ),
        (
            lambda payload: payload["authority_contract"].__setitem__(
                "approved_skeleton_or_animation_touched", True
            ),
            "touched skeleton or animation",
        ),
    ],
)
def test_bounded_watertight_proxy_manifest_tamper_fails_closed(
    tmp_path,
    monkeypatch,
    mutate,
    match,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)
    proxy_manifest = closure.load_json(
        case["proxy_manifest"], "watertight proxy manifest"
    )
    mutate(proxy_manifest)
    write_json(case["proxy_manifest"], proxy_manifest)

    with pytest.raises(closure.ClosureError, match=match):
        _validate_bounded_watertight(case)


def test_bounded_watertight_requires_external_proxy_manifest_hash(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)

    with pytest.raises(
        closure.ClosureError,
        match="proxy manifest changed from external authority",
    ):
        _validate_bounded_watertight(
            case,
            expected_watertight_proxy_manifest_sha256="0" * 64,
        )


def test_bounded_watertight_rejects_proxy_audit_external_hash_tamper(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)
    retained_sha256 = closure.sha256_file(case["proxy_audit"])
    payload = closure.load_json(
        case["proxy_audit"],
        "watertight proxy geometry audit",
    )
    payload["records"][0]["created_by_untrusted_producer"] = True
    write_json(case["proxy_audit"], payload)

    with pytest.raises(
        closure.ClosureError,
        match="proxy geometry audit changed from external authority",
    ):
        _validate_bounded_watertight(
            case,
            expected_watertight_proxy_geometry_audit_sha256=retained_sha256,
        )


def test_bounded_watertight_rejects_self_consistent_arbitrary_proxy(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)
    arbitrary_proxy = tmp_path / "self_consistent_arbitrary_proxy.glb"
    write_production_proxy_glb(arbitrary_proxy, shift=1.0)
    arbitrary_manifest = tmp_path / "self_consistent_proxy.manifest.json"
    manifest_payload = closure.load_json(
        case["proxy_manifest"],
        "watertight proxy manifest",
    )
    manifest_payload["output"] = descriptor(arbitrary_proxy)
    write_json(arbitrary_manifest, manifest_payload)
    arbitrary_audit = tmp_path / "self_consistent_proxy.audit.json"
    write_json(arbitrary_audit, proxy_geometry_audit(arbitrary_proxy))

    with pytest.raises(
        closure.ClosureError,
        match="not independently correspondent",
    ):
        closure.validate_upstream_lineage(
            case["raw_manifest"],
            arbitrary_manifest,
            arbitrary_proxy,
            bounded_geometry_closure_path=case["upstream"],
            expected_bounded_geometry_closure_sha256=closure.sha256_file(
                case["upstream"]
            ),
            expected_bounded_geometry_closure_manifest_sha256=case[
                "geometry_manifest_sha256"
            ],
            expected_watertight_proxy_manifest_sha256=closure.sha256_file(
                arbitrary_manifest
            ),
            watertight_proxy_geometry_audit_path=arbitrary_audit,
            expected_watertight_proxy_geometry_audit_sha256=closure.sha256_file(
                arbitrary_audit
            ),
            raw_static_decision_batch_path=case["decision_batch"],
            expected_raw_static_decision_batch_sha256=closure.sha256_file(
                case["decision_batch"]
            ),
            require_oriented_batch_external_authority=True,
        )


def test_bounded_watertight_rejects_retained_correspondence_tamper(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)
    observed: dict[str, object] = {}
    _validate_bounded_watertight(
        case,
        observed_watertight_proxy_correspondence=observed,
    )
    retained = deepcopy(observed)
    retained["proxy_to_repaired"]["max_ratio"] = 0.0

    with pytest.raises(
        closure.ClosureError,
        match="correspondence changed from closure authority",
    ):
        _validate_bounded_watertight(
            case,
            expected_watertight_proxy_correspondence=retained,
        )


def test_bounded_watertight_requires_external_geometry_closure_hash(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)

    with pytest.raises(
        closure.ClosureError,
        match="bounded geometry closure changed from external authority",
    ):
        _validate_bounded_watertight(
            case,
            expected_bounded_geometry_closure_sha256="0" * 64,
        )


def test_bounded_watertight_rejects_file_and_internal_hash_confusion(
    tmp_path,
    monkeypatch,
):
    case = _bounded_watertight_v2_fixture(tmp_path, monkeypatch)
    file_sha256 = closure.sha256_file(case["upstream"])
    manifest_sha256 = case["geometry_manifest_sha256"]
    assert file_sha256 != manifest_sha256

    with pytest.raises(
        closure.ClosureError,
        match="changed from external authority",
    ):
        _validate_bounded_watertight(
            case,
            expected_bounded_geometry_closure_sha256=manifest_sha256,
        )

    with pytest.raises(
        closure.ClosureError,
        match="internal manifest hash changed",
    ):
        _validate_bounded_watertight(
            case,
            expected_bounded_geometry_closure_manifest_sha256=file_sha256,
        )


def _write_load_audit(path, tokenrig_input, empty_objects):
    common = {
        "filepath": str(tokenrig_input),
        "patch_sha256": "a" * 64,
        "pid": 101,
        "generation": 7,
    }
    events = []
    for sequence in (1, 2):
        events.extend(
            (
                {
                    **common,
                    "sequence": sequence,
                    "phase": "before_clean",
                    "inventory": {"objects": [{"name": "old", "type": "MESH"}]},
                },
                {
                    **common,
                    "sequence": sequence,
                    "phase": "after_clean",
                    "inventory": {
                        "objects": [],
                        "mesh_count": 0,
                        "material_count": 0,
                        "image_count": 0,
                    },
                },
                {
                    **common,
                    "sequence": sequence,
                    "phase": "after_import",
                    "inventory": {
                        "objects": [
                            {"name": "geometry_0", "type": "MESH"},
                            *deepcopy(empty_objects),
                        ],
                        "mesh_count": 1,
                    },
                },
            )
        )
    path.write_bytes(
        b"\n".join(closure.canonical_bytes(event) for event in events) + b"\n"
    )
    return common


@pytest.mark.parametrize(
    "empty_objects",
    [
        [],
        [{"name": "world", "type": "EMPTY"}],
    ],
)
def test_load_audit_accepts_mesh_with_optional_exact_world_root(
    tmp_path,
    empty_objects,
):
    tokenrig_input = tmp_path / "tokenrig_input.glb"
    tokenrig_input.write_bytes(b"static Pixel3D input")
    audit = tmp_path / "load_audit.jsonl"
    common = _write_load_audit(audit, tokenrig_input, empty_objects)

    events = closure.validate_load_audit(
        audit,
        tokenrig_input,
        {
            "patch_sha256": common["patch_sha256"],
            "pid": common["pid"],
            "generation": common["generation"],
        },
    )

    assert len(events) == 6


@pytest.mark.parametrize(
    "empty_objects",
    [
        [{"name": "other", "type": "EMPTY"}],
        [
            {"name": "world", "type": "EMPTY"},
            {"name": "other", "type": "EMPTY"},
        ],
    ],
)
def test_load_audit_rejects_other_or_multiple_empty_roots(
    tmp_path,
    empty_objects,
):
    tokenrig_input = tmp_path / "tokenrig_input.glb"
    tokenrig_input.write_bytes(b"static Pixel3D input")
    audit = tmp_path / "load_audit.jsonl"
    common = _write_load_audit(audit, tokenrig_input, empty_objects)

    with pytest.raises(closure.ClosureError, match="inventory is contaminated"):
        closure.validate_load_audit(
            audit,
            tokenrig_input,
            {
                "patch_sha256": common["patch_sha256"],
                "pid": common["pid"],
                "generation": common["generation"],
            },
        )


def test_geometry_closure_v2_selects_hash_equivalent_direct_adopted_raw(
    tmp_path,
    monkeypatch,
):
    case = _direct_copy_v2_fixture(tmp_path, monkeypatch)
    case["historical_raw"].unlink()

    observed_raw, kind, extra = closure.validate_upstream_lineage(
        case["raw_manifest"],
        case["upstream"],
        case["tokenrig_input"],
    )

    assert os.path.samefile(observed_raw, case["adopted_raw"])
    assert kind == "bounded_geometry_closure"
    assert extra == [case["repair"], case["audit"], case["decision_batch"]]
    assert case["observed"]["path"] == case["upstream"]
    expected = case["observed"]["expected"]
    assert "expected_raw_pixal_glb" not in expected
    assert expected["expected_pixal_manifest"] == case["raw_manifest"]
    assert expected["expected_repaired_glb"] == case["tokenrig_input"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sha256", "0" * 64, "SHA-256 mismatch"),
        ("bytes", 1, "size mismatch"),
    ],
)
def test_geometry_closure_v2_direct_copy_still_requires_raw_content_binding(
    tmp_path,
    monkeypatch,
    field,
    value,
    match,
):
    case = _direct_copy_v2_fixture(tmp_path, monkeypatch)
    manifest = closure.load_json(case["raw_manifest"], "raw Pixal manifest")
    manifest["output"][field] = value
    write_json(case["raw_manifest"], manifest)

    with pytest.raises(closure.ClosureError, match=match):
        closure.validate_upstream_lineage(
            case["raw_manifest"],
            case["upstream"],
            case["tokenrig_input"],
        )


def test_tokenrig_lineage_reader_keeps_strict_mirror_v2_path(
    tmp_path,
    monkeypatch,
):
    raw, raw_manifest, tokenrig_input, repair_path, upstream = (
        _bounded_upstream_fixture(tmp_path)
    )
    mirror = {
        "implementation_contract": (
            closure.derived_review_contract.REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "pixal_source": descriptor(raw),
            "pixal_manifest": descriptor(raw_manifest),
        },
        "output": descriptor(tokenrig_input),
    }
    monkeypatch.setattr(
        closure.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: deepcopy(mirror),
    )

    observed_raw, kind, extra = closure.validate_upstream_lineage(
        raw_manifest,
        upstream,
        tokenrig_input,
    )

    assert observed_raw == raw
    assert kind == "bounded_geometry_closure"
    assert extra == [repair_path]


def test_tokenrig_build_mode_keeps_legacy_watertight_without_batch(tmp_path):
    raw = tmp_path / "pixal_raw.glb"
    raw.write_bytes(b"raw pixal")
    tokenrig_input = tmp_path / "watertight.glb"
    tokenrig_input.write_bytes(b"watertight")
    raw_manifest = tmp_path / "raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(raw),
                "bytes": raw.stat().st_size,
            },
        },
    )
    upstream = tmp_path / "watertight.json"
    write_json(
        upstream,
        {
            "schema": "avengine_watertight_textured_runtime_proxy_v1",
            "input": descriptor(raw),
            "output": descriptor(tokenrig_input),
            "authority_contract": {
                "approved_skeleton_or_animation_touched": False
            },
        },
    )
    _raw, kind, extra = closure.validate_upstream_lineage(
        raw_manifest,
        upstream,
        tokenrig_input,
        require_oriented_batch_external_authority=True,
    )

    assert kind == "watertight_runtime_proxy"
    assert extra == []


def test_tokenrig_lineage_reader_rejects_oriented_batch_on_mirror_v2(
    tmp_path,
    monkeypatch,
):
    raw, raw_manifest, tokenrig_input, _repair_path, upstream = (
        _bounded_upstream_fixture(tmp_path)
    )
    batch = tmp_path / "raw_decision_batch.json"
    batch.write_bytes(b"oriented authority")
    mirror = {
        "implementation_contract": (
            closure.derived_review_contract.REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "pixal_source": descriptor(raw),
            "pixal_manifest": descriptor(raw_manifest),
        },
        "output": descriptor(tokenrig_input),
    }
    monkeypatch.setattr(
        closure.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: deepcopy(mirror),
    )

    with pytest.raises(
        closure.ClosureError,
        match="mirror-v2 repair cannot consume oriented",
    ):
        closure.validate_upstream_lineage(
            raw_manifest,
            upstream,
            tokenrig_input,
            raw_static_decision_batch_path=batch,
            expected_raw_static_decision_batch_sha256=(
                closure.sha256_file(batch)
            ),
            require_oriented_batch_external_authority=True,
        )


def test_tokenrig_lineage_reader_accepts_oriented_canonical_batch(
    tmp_path,
    monkeypatch,
):
    raw, raw_manifest, tokenrig_input, repair_path, upstream = (
        _bounded_upstream_fixture(tmp_path)
    )
    decision = tmp_path / "raw_decision.json"
    write_json(decision, {"fixture": True})
    batch = tmp_path / "raw_decision_batch.json"
    write_json(batch, {"fixture": True})
    instance_id = "dog_fixture_oriented_sheet_v1"
    oriented = {
        "implementation_contract": (
            closure.derived_review_contract
            .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "instance_id": instance_id,
            "pixal_source": descriptor(raw),
            "pixal_manifest": descriptor(raw_manifest),
            "static_decision_batch": descriptor(batch),
            "static_decision": descriptor(decision),
        },
        "output": descriptor(tokenrig_input),
    }
    approved = {
        "decision": "approved_for_lod_and_binding",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": "lod_then_species_rig_binding",
        "checks": {
            name: True
            for name in closure.derived_review_contract.RAW_STATIC_CHECK_FIELDS
        },
    }
    monkeypatch.setattr(
        closure.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: deepcopy(oriented),
    )
    monkeypatch.setattr(
        source_registry,
        "load_decision_batch",
        lambda _path: (
            batch,
            {},
            {
                instance_id: {
                    "path": decision,
                    "payload": deepcopy(approved),
                }
            },
        ),
    )

    _raw, kind, extra = closure.validate_upstream_lineage(
        raw_manifest,
        upstream,
        tokenrig_input,
        raw_static_decision_batch_path=batch,
        expected_raw_static_decision_batch_sha256=closure.sha256_file(batch),
        require_oriented_batch_external_authority=True,
    )

    assert kind == "bounded_geometry_closure"
    assert extra == [repair_path, batch]


def test_tokenrig_lineage_reader_rejects_oriented_batch_rebinding(
    tmp_path,
    monkeypatch,
):
    raw, raw_manifest, tokenrig_input, _repair_path, upstream = (
        _bounded_upstream_fixture(tmp_path)
    )
    decision = tmp_path / "raw_decision.json"
    decision.write_bytes(b"decision")
    rebound = tmp_path / "rebound_decision.json"
    rebound.write_bytes(decision.read_bytes())
    batch = tmp_path / "raw_decision_batch.json"
    batch.write_bytes(b"batch")
    instance_id = "dog_fixture_oriented_sheet_v1"
    oriented = {
        "implementation_contract": (
            closure.derived_review_contract
            .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "instance_id": instance_id,
            "pixal_source": descriptor(raw),
            "pixal_manifest": descriptor(raw_manifest),
            "static_decision_batch": descriptor(batch),
            "static_decision": descriptor(decision),
        },
        "output": descriptor(tokenrig_input),
    }
    monkeypatch.setattr(
        closure.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: deepcopy(oriented),
    )
    monkeypatch.setattr(
        source_registry,
        "load_decision_batch",
        lambda _path: (
            batch,
            {},
            {
                instance_id: {
                    "path": rebound,
                    "payload": {
                        "decision": "approved_for_lod_and_binding",
                    },
                }
            },
        ),
    )

    with pytest.raises(
        closure.ClosureError,
        match="canonical approved raw static decision",
    ):
        closure.validate_upstream_lineage(
            raw_manifest,
            upstream,
            tokenrig_input,
            raw_static_decision_batch_path=batch,
            expected_raw_static_decision_batch_sha256=(
                closure.sha256_file(batch)
            ),
            require_oriented_batch_external_authority=True,
        )


def test_tokenrig_lineage_reader_rejects_raw_fallback_with_derived_authority(
    tmp_path,
):
    raw = tmp_path / "pixal_raw.glb"
    raw.write_bytes(b"raw pixal")
    tokenrig_input = tmp_path / "watertight.glb"
    tokenrig_input.write_bytes(b"watertight")
    raw_manifest = tmp_path / "raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(raw),
                "bytes": raw.stat().st_size,
            },
        },
    )
    upstream = tmp_path / "watertight.json"
    write_json(
        upstream,
        {
            "schema": "avengine_watertight_textured_runtime_proxy_v1",
            "input": descriptor(raw),
            "output": descriptor(tokenrig_input),
            "authority_contract": {
                "approved_skeleton_or_animation_touched": False
            },
        },
    )
    batch = tmp_path / "decision_batch.json"
    batch.write_bytes(b"batch")

    with pytest.raises(closure.ClosureError, match="raw/watertight fallback"):
        closure.validate_upstream_lineage(
            raw_manifest,
            upstream,
            tokenrig_input,
            raw_static_decision_batch_path=batch,
            expected_raw_static_decision_batch_sha256=(
                closure.sha256_file(batch)
            ),
            require_oriented_batch_external_authority=True,
        )


def test_geometry_readback_changed_gate_fails_even_after_self_rehash(tmp_path):
    source = tmp_path / "source.glb"
    rig = tmp_path / "rig.glb"
    source.write_bytes(b"source")
    rig.write_bytes(b"rig")
    readback = tmp_path / "readback.json"
    payload = valid_readback(readback, source, rig)
    closure.validate_readback(readback, source, rig)

    payload["automatic_checks"]["world_geometry_unchanged"] = False
    payload["manifest_sha256"] = closure.hash_without(payload, "manifest_sha256")
    write_json(readback, payload)
    with pytest.raises(closure.ClosureError, match="did not pass every gate"):
        closure.validate_readback(readback, source, rig)


def test_recomputed_embedded_hash_cannot_replace_external_manifest_authority(
    tmp_path,
):
    root = tmp_path / "closure"
    root.mkdir()
    manifest = root / "manifest.json"
    original = {
        "schema": closure.SCHEMA,
        "asset_id": "original",
        "status": "passed_source_to_tokenrig_closure",
        "formal_dataset_registration_authorized": False,
    }
    original["manifest_sha256"] = closure.hash_without(original, "manifest_sha256")
    write_json(manifest, original)
    external_hash = closure.sha256_file(manifest)

    mutated = deepcopy(original)
    mutated["asset_id"] = "mutated"
    mutated["manifest_sha256"] = closure.hash_without(mutated, "manifest_sha256")
    write_json(manifest, mutated)
    with pytest.raises(closure.ClosureError, match="external authority"):
        closure.validate_closure(root, expected_manifest_sha256=external_hash)


def test_unknown_evidence_mode_fails_before_any_downstream_use(tmp_path):
    root = tmp_path / "closure"
    root.mkdir()
    payload = {
        "schema": closure.SCHEMA,
        "asset_id": "asset",
        "status": "passed_source_to_tokenrig_closure",
        "formal_dataset_registration_authorized": False,
        "evidence_mode": {
            "mode": "invented_mode",
            "no_missing_evidence_was_fabricated": True,
        },
    }
    payload["manifest_sha256"] = closure.hash_without(payload, "manifest_sha256")
    manifest = root / "manifest.json"
    write_json(manifest, payload)

    with pytest.raises(closure.ClosureError, match="unsupported.*evidence mode"):
        closure.validate_closure(
            root,
            expected_manifest_sha256=closure.sha256_file(manifest),
        )


def test_leaf_and_arbitrary_parent_symlinks_are_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    artifact = real / "asset.glb"
    artifact.write_bytes(b"asset")
    leaf = tmp_path / "leaf.glb"
    leaf.symlink_to(artifact)
    parent = tmp_path / "parent"
    parent.symlink_to(real, target_is_directory=True)

    with pytest.raises(closure.ClosureError, match="unsafe symlink"):
        closure.require_regular_file(leaf, "leaf")
    with pytest.raises(closure.ClosureError, match="unsafe symlink"):
        closure.require_regular_file(parent / "asset.glb", "parent")


def test_known_logical_tmp_path_resolves_to_physical_identity():
    if not closure.LOGICAL_TMP_ROOT.is_symlink():
        pytest.skip("workspace compatibility tmp link is not configured")
    with tempfile.TemporaryDirectory(
        prefix="tokenrig-closure-path-test-",
        dir=closure.PHYSICAL_TMP_ROOT,
    ) as physical_directory:
        physical = Path(physical_directory) / "asset.glb"
        physical.write_bytes(b"identity")
        relative = physical.relative_to(closure.PHYSICAL_TMP_ROOT)
        logical = closure.LOGICAL_TMP_ROOT / relative

        logical_result = closure.require_regular_file(logical, "logical asset")
        physical_result = closure.require_regular_file(physical, "physical asset")
        assert os.path.samefile(logical_result, physical_result)


def test_nonfinite_json_is_rejected():
    with pytest.raises(closure.ClosureError, match="non-finite"):
        closure.recursive_finite({"value": float("nan")})
