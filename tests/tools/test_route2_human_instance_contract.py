from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


MODULES = (
    "tools.route2_human_instance_contract",
    "tools.route2_human_static_decision",
    "tools.route2_human_qualified_candidate",
)


def _modules():
    return tuple(importlib.import_module(name) for name in MODULES)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    return _write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _record(path: Path, *, filename: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }
    if filename is not None:
        result["filename"] = filename
    return result


def _fixture_skeleton() -> list[tuple[str, str | None]]:
    bones: list[tuple[str, str | None]] = [
        ("bone_pelvis", None),
        ("bone_spine_0", "bone_pelvis"),
        ("bone_spine_1", "bone_spine_0"),
        ("bone_spine_2", "bone_spine_1"),
        ("bone_head", "bone_spine_2"),
        ("bone_left_upper_arm", "bone_spine_2"),
        ("bone_left_forearm", "bone_left_upper_arm"),
        ("bone_left_hand", "bone_left_forearm"),
        ("bone_right_upper_arm", "bone_spine_2"),
        ("bone_right_forearm", "bone_right_upper_arm"),
        ("bone_right_hand", "bone_right_forearm"),
        ("bone_left_thigh", "bone_pelvis"),
        ("bone_left_calf", "bone_left_thigh"),
        ("bone_left_foot", "bone_left_calf"),
        ("bone_left_toe", "bone_left_foot"),
        ("bone_right_thigh", "bone_pelvis"),
        ("bone_right_calf", "bone_right_thigh"),
        ("bone_right_foot", "bone_right_calf"),
        ("bone_right_toe", "bone_right_foot"),
        ("bone_hat", "bone_head"),
    ]
    bones.extend(
        (f"bone_extra_{index:02d}", "bone_head") for index in range(32)
    )
    assert len(bones) == 52
    return bones


def _fixture_hierarchy_bytes() -> bytes:
    rows = [
        "TokenRig static joint hierarchy",
        "root=bone_pelvis",
        "bone_count=52",
        "format: bone<TAB>parent<TAB>world_rest_head_xyz<TAB>semantic_labels",
    ]
    rows.extend(
        f"{name}\t{parent or '-'}\t0.000000000,0.000000000,0.000000000\t-"
        for name, parent in _fixture_skeleton()
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


def _write_glb(
    path: Path,
    *,
    action: str | None = None,
    include_skin: bool = True,
    joint_count: int = 52,
    vertex_joint_name: str = "bone_head",
    include_pbr: bool = True,
    pbr_image_uri: str | None = None,
    include_binary_chunk: bool = True,
    corrupt_pbr_buffer_view: bool = False,
) -> Path:
    vertex_count = 128
    positions = b"\x00" * (vertex_count * 3 * 4)
    skeleton = _fixture_skeleton()[:joint_count] if include_skin else []
    skin_names = [name for name, _ in skeleton]
    joint_index = (
        skin_names.index(vertex_joint_name)
        if vertex_joint_name in skin_names
        else 0
    )
    joints = b"".join(
        struct.pack("<4H", joint_index, 0, 0, 0) for _ in range(vertex_count)
    )
    weights = b"".join(
        struct.pack("<4f", 1.0, 0.0, 0.0, 0.0) for _ in range(vertex_count)
    )
    texcoords = b"\x00" * (vertex_count * 2 * 4)
    base_color_stream = io.BytesIO()
    Image.new("RGB", (1, 1), (64, 128, 192)).save(
        base_color_stream, format="PNG"
    )
    base_color_image = base_color_stream.getvalue()
    metallic_roughness_stream = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 128, 0)).save(
        metallic_roughness_stream, format="PNG"
    )
    metallic_roughness_image = metallic_roughness_stream.getvalue()
    binary = positions + joints + weights
    if include_pbr:
        binary += texcoords + base_color_image + metallic_roughness_image
    document: dict[str, Any] = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(joints),
            },
            {
                "buffer": 0,
                "byteOffset": len(positions) + len(joints),
                "byteLength": len(weights),
            },
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": vertex_count, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": vertex_count, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": vertex_count, "type": "VEC4"},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "JOINTS_0": 1,
                            "WEIGHTS_0": 2,
                        }
                    }
                ]
            }
        ],
    }
    if include_pbr:
        texcoord_offset = len(positions) + len(joints) + len(weights)
        base_color_offset = texcoord_offset + len(texcoords)
        metallic_roughness_offset = base_color_offset + len(base_color_image)
        document["bufferViews"].extend(
            [
                {
                    "buffer": 0,
                    "byteOffset": texcoord_offset,
                    "byteLength": len(texcoords),
                },
                {
                    "buffer": 0,
                    "byteOffset": base_color_offset,
                    "byteLength": len(base_color_image),
                },
                {
                    "buffer": 0,
                    "byteOffset": metallic_roughness_offset,
                    "byteLength": len(metallic_roughness_image),
                },
            ]
        )
        document["accessors"].append(
            {
                "bufferView": 3,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC2",
            }
        )
        primitive = document["meshes"][0]["primitives"][0]
        primitive["attributes"]["TEXCOORD_0"] = 3
        primitive["material"] = 0
        document["materials"] = [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 1.0,
                    "roughnessFactor": 1.0,
                }
            }
        ]
        document["textures"] = [{"source": 0}, {"source": 1}]
        document["images"] = (
            [{"uri": pbr_image_uri}, {"uri": pbr_image_uri}]
            if pbr_image_uri is not None
            else [
                {"bufferView": 4, "mimeType": "image/png"},
                {"bufferView": 5, "mimeType": "image/png"},
            ]
        )
        if corrupt_pbr_buffer_view and pbr_image_uri is None:
            document["bufferViews"][4]["byteOffset"] = len(binary) + 4096
    if include_skin:
        positions = {name: index for index, (name, _) in enumerate(skeleton)}
        nodes: list[dict[str, Any]] = [{"name": name} for name, _ in skeleton]
        for child_index, (_, parent) in enumerate(skeleton):
            if parent is not None:
                nodes[positions[parent]].setdefault("children", []).append(child_index)
        document["nodes"] = nodes
        document["skins"] = [{"joints": list(range(len(nodes)))}]
    if action is not None:
        document["animations"] = [{"name": action}]
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    chunks = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    if include_binary_chunk:
        chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
    payload = b"glTF" + struct.pack("<II", 2, 12 + len(chunks)) + chunks
    return _write(path, payload)


def _write_png(path: Path) -> Path:
    return _write(path, b"\x89PNG\r\n\x1a\nroute2 fixture")


def _write_review_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (2, 2), (80, 100, 120, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))
    image.save(path)
    return path


def _complete_static_checks() -> dict[str, Any]:
    semantic = {
        name: ([f"bone_{name}_0", f"bone_{name}_1"] if name == "spine" else f"bone_{name}")
        for name in (
            "pelvis",
            "spine",
            "head",
            "left_upper_arm",
            "left_forearm",
            "left_hand",
            "right_upper_arm",
            "right_forearm",
            "right_hand",
            "left_thigh",
            "left_calf",
            "left_foot",
            "left_toe",
            "right_thigh",
            "right_calf",
            "right_foot",
            "right_toe",
        )
    }
    return {
        "automatic_static_checks": "passed",
        "raw_triangle_contract": {"passed": True},
        "input_pbr": {"passed": True},
        "input_mesh": {"passed": True},
        "input_surface": {"passed": True},
        "source_mesh_contract": {"vertex_count": 4},
        "source_removed_gltf_import_helpers": [],
        "tokenrig_mesh_contract_before_canonical_transform": {"vertex_count": 4},
        "tokenrig_removed_gltf_import_helpers": [],
        "removed_proven_orphans": [],
        "axis_canonicalization": {
            "canonical_front": "negative-y",
            "canonical_up_vector": [0.0, 0.0, 1.0],
            "transform_count": 1,
        },
        "grounding": {
            "canonical_floor_z": 0.0,
            "post_floor_z": 0.0,
            "transform_count": 1,
        },
        "canonical_mesh_contract": {"vertex_count": 4},
        "hierarchy": {
            "root": "bone_pelvis",
            "connected": True,
            "parent_first": True,
            "bone_count": 52,
        },
        "semantic_mapping": {
            "semantic_bones": semantic,
            "ignored_proven_head_descendants": ["bone_hat"],
        },
        "weights": {
            "vertex_count": 4,
            "maximum_influences": 4,
            "maximum_weight_sum_error": 0.0,
            "weight_sum_tolerance": 1.0e-6,
        },
        "seams": {
            "maximum_weight_l1_error": 0.0,
            "weight_l1_tolerance": 1.0e-6,
        },
        "bilateral_contamination": {
            "contaminated_vertex_count": 0,
            "maximum_opposite_limb_weight": 0.0,
            "tolerance": 1.0e-4,
        },
        "exported_pbr": {"passed": True},
        "glb_roundtrip": {
            "passed": True,
            "mesh": {"passed": True},
            "surface": {"passed": True},
            "pbr": {"passed": True},
            "skin": {"passed": True},
            "full_rest": {
                "passed": True,
                "bone_count": 52,
                "maximum_matrix_element_error": 0.0,
                "maximum_tail_error_m": 0.0,
                "maximum_roll_error_radians": 0.0,
                "tolerance": 2.0e-6,
            },
            "inverse_bind": {
                "passed": True,
                "joint_count": 52,
                "joint_order_unchanged": True,
                "maximum_matrix_element_error": 0.0,
                "tolerance": 2.0e-6,
            },
        },
        "renders": {"passed": True},
    }


def _readonly_bundle(root: Path) -> None:
    for path in root.iterdir():
        if path.is_file():
            path.chmod(0o444)
    root.chmod(0o555)


def _base_lineage(
    source_root: Path,
    asset_id: str,
    *,
    include_pbr: bool = True,
    pbr_image_uri: str | None = None,
    include_binary_chunk: bool = True,
    corrupt_pbr_buffer_view: bool = False,
) -> tuple[dict[str, Any], Path, Path]:
    from tools import route2_human_instance_contract as route2_contract

    source_image = _write_png(source_root / "source.png")
    flux_candidate = _write_png(source_root / "candidate.png")
    flux_manifest = _write_json(
        source_root / "candidate_manifest.json",
        {
            "schema_version": "human_reference_candidate_v1",
            "asset_id": asset_id,
            "model_revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            "input_sha256": _sha(source_image),
            "output_sha256": _sha(flux_candidate),
        },
    )
    source_review = _write_json(
        source_root / "reference_review.json",
        {
            "schema_version": "human_reference_review_v1",
            "asset_id": asset_id,
            "decision": "approved",
            "source_sha256": _sha(source_image),
            "candidate_sha256": _sha(flux_candidate),
            "candidate_manifest_sha256": _sha(flux_manifest),
        },
    )
    pixal_input = _write_png(source_root / "input_rgba.png")
    pixal_snapshot = route2_contract.MODEL_ROOT / route2_contract.PIXAL3D_SNAPSHOT_RELATIVE
    dino_snapshot = route2_contract.MODEL_ROOT / route2_contract.DINO_SNAPSHOT_RELATIVE
    pixal_glb = _write_glb(
        source_root / "pixal.glb",
        include_pbr=include_pbr,
        pbr_image_uri=pbr_image_uri,
        include_binary_chunk=include_binary_chunk,
        corrupt_pbr_buffer_view=corrupt_pbr_buffer_view,
    )
    pixal_manifest = _write_json(
        source_root / "pixal.manifest.json",
        {
            "backend": "pixal3d",
            "model": {
                "snapshot": str(pixal_snapshot),
                "revision": "0b31f9160aa400719af409098bff7936a932f726",
            },
            "dino": {
                "snapshot": str(dino_snapshot),
                "revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881",
            },
            "input": {"path": str(pixal_input), "sha256": _sha(pixal_input)},
            "output": {
                "path": str(pixal_glb),
                "sha256": _sha(pixal_glb),
                "bytes": pixal_glb.stat().st_size,
            },
            "parameters": {
                "low_vram": True,
                "manual_fov": 0.2,
                "resolution": 1024,
                "seed": 42,
            },
        },
    )
    paths = {
        "source_image": ("approved_soft_t_source", source_image),
        "flux_candidate": ("flux2_candidate_image", flux_candidate),
        "flux_manifest": ("flux2_candidate_manifest", flux_manifest),
        "source_review": ("source_reference_review", source_review),
        "pixal_input_rgba": ("pixal_input_rgba", pixal_input),
        "pixal_manifest": ("pixal_manifest", pixal_manifest),
        "pixal_pbr_glb": ("pixal_pbr_glb", pixal_glb),
    }
    return (
        {
            key: {"role": role, "path": path, "root": source_root}
            for key, (role, path) in paths.items()
        },
        pixal_glb,
        pixal_manifest,
    )


def _attribute_lineage(
    source_root: Path,
    *,
    base_qualified: Path | None = None,
    case_id: str = "tall_man",
    base_avatar_id: str = "rocketbox_male_adult_01",
) -> dict[str, Any]:
    from tools import route2_human_qualified_candidate as qualified_candidate
    from tools import route2_human_instance_contract as route2_contract
    from tools import human_attribute_pixal_contract
    from tools.spike_rlr import human_attribute_review

    asset_id = f"route2_{case_id}_v1"
    asset_root = source_root / asset_id
    asset_root.mkdir()
    base_review_dir = (
        base_qualified.parent / "dynamic_review_v1"
        if base_qualified is not None
        else source_root / base_avatar_id / "dynamic_review_v1"
    )
    if base_qualified is not None:
        qualified_payload = qualified_candidate.validate_qualified_candidate(base_qualified)
        base_qualification = {
            "asset_id": base_avatar_id,
            "status": human_attribute_review.PASS_STATUS,
            "qualified_candidate": _record(base_qualified),
            "final_branch": dict(qualified_payload["final_branch"]),
            "review_dir": qualified_payload["dynamic"]["review_dir"],
        }
    else:
        base_qualification = {
            "asset_id": base_avatar_id,
            "status": human_attribute_review.PASS_STATUS,
            "qualified_candidate": {
                "path": str(source_root / "missing_qualified_candidate_v1.json"),
                "sha256": "0" * 64,
                "size_bytes": 1,
            },
            "final_branch": {
                "branch_id": "direct",
                "path": str(source_root / base_avatar_id),
                "relative_root": ".",
            },
            "review_dir": str(base_review_dir),
        }
    candidate_root = source_root / "attribute_bundle"
    candidate_root.mkdir()
    artifacts: dict[str, Any] = {}
    for name in sorted(human_attribute_review.CANDIDATE_ARTIFACTS):
        path = candidate_root / name
        if name.endswith(".png"):
            _write_review_png(path)
        else:
            _write_json(path, {"schema": "attribute_bundle_aux_v1"})
        artifacts[name] = _record(path)
    candidate_manifest = _write_json(
        candidate_root / "candidate_manifest.json",
        {
            "schema": "flux2_human_attribute_candidate_v2",
            "case_id": case_id,
            "base_asset_id": base_avatar_id,
            "downstream_asset_id": asset_id,
            "state_classification": "research_candidate",
            "bundle_status": "generated_pending_agent_2d_visual_qa",
            "agent_qa_status": "pending_agent_2d_visual_qa",
            "user_acceptance": "pending_user_review",
            "base_route2_qualification": base_qualification,
            "quantitative_snapshot": {
                "automatic_checks": "passed",
                "pixel_proof": {
                    "outside_changed_pixels": 0,
                    "outside_max_abs_channel_delta": 0,
                    "transition_is_feathered": True,
                },
                "alpha_proof": {"outside_changed_pixels": 0},
                "case_metrics": {
                    "passed": True,
                    "checks": {
                        "target_attribute_changed": True,
                        "non_target_regions_preserved": True,
                    },
                    "metrics": {"fixture_target_delta": 1.0},
                },
            },
            "artifacts": artifacts,
        },
    )
    _readonly_bundle(candidate_root)
    decision = human_attribute_review.record_agent_2d_visual_qa(
        candidate_root,
        status=human_attribute_review.PASS_STATUS,
        reviewer="codex-route2-attribute-review",
        notes="Fixture passes the complete owner 2D review contract.",
        checks={name: True for name in human_attribute_review.AGENT_2D_VISUAL_CHECKS},
    )
    rgba = candidate_root / "candidate_rgba.png"
    wrapper = Path(human_attribute_pixal_contract.PIXAL_WRAPPER_PATH)
    executor_code = Path(human_attribute_pixal_contract.__file__).resolve()
    pixal_snapshot = route2_contract.MODEL_ROOT / route2_contract.PIXAL3D_SNAPSHOT_RELATIVE
    dino_snapshot = route2_contract.MODEL_ROOT / route2_contract.DINO_SNAPSHOT_RELATIVE
    pixal_glb = _write_glb(asset_root / "canary_1024_seed42.glb")
    pixal_manifest = _write_json(
        asset_root / "canary_1024_seed42.manifest.json",
        {
            "backend": "pixal3d",
            "asset_id": asset_id,
            "case_id": case_id,
            "base_avatar_id": base_avatar_id,
            "input": {
                "path": str(rgba),
                "sha256": _sha(rgba),
                "mode": "RGBA",
                "size": [2, 2],
                "alpha_min": 0,
                "alpha_max": 255,
            },
            "model": {
                "snapshot": str(pixal_snapshot),
                "revision": "0b31f9160aa400719af409098bff7936a932f726",
            },
            "dino": {
                "snapshot": str(dino_snapshot),
                "revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881",
            },
            "output": {
                "path": str(pixal_glb),
                "sha256": _sha(pixal_glb),
                "bytes": pixal_glb.stat().st_size,
            },
            "parameters": {
                "low_vram": True,
                "manual_fov": 0.2,
                "resolution": 1024,
                "seed": 42,
            },
        },
    )
    argv = [
        str(wrapper),
        "--backend",
        "pixal3d",
        "--image",
        str(rgba),
        "--output",
        str(pixal_glb),
        "--gpu",
        "3",
        "--seed",
        "42",
        "--resolution",
        "1024",
        "--manual-fov",
        "0.2",
        "--low-vram",
    ]
    pixal_job = _write_json(
        source_root / f"{asset_id}.pixal_job.json",
        {
            "schema": "pixal3d_human_attribute_job_v1",
            "case_id": case_id,
            "asset_id": asset_id,
            "base_asset_id": base_avatar_id,
            "state_classification": "research_candidate",
            "model_revision": "0b31f9160aa400719af409098bff7936a932f726",
            "dino_revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881",
            "candidate_manifest": _record(candidate_manifest),
            "agent_2d_decision": {
                **_record(decision),
                "status": human_attribute_review.PASS_STATUS,
            },
            "input_rgba": {
                **_record(rgba),
                "mode": "RGBA",
                "size": [2, 2],
                "alpha_min": 0,
                "alpha_max": 255,
            },
            "parameters": {
                "seed": 42,
                "manual_fov": 0.2,
                "resolution": 1024,
                "low_vram": True,
            },
            "wrapper": _record(wrapper),
            "output_glb": str(pixal_glb),
            "output_manifest": str(pixal_manifest),
            "output_policy": "atomic_no_replace",
            "executor": {
                "kind": "atomic_pixal3d_executor_v1",
                "argv": argv,
                "execution_authorized": True,
                "atomic_no_replace": True,
                **_record(executor_code),
            },
        },
    )
    pixal_job.chmod(0o444)
    attempt_id = "attempt_001"
    staging_path = source_root / f".{asset_id}.{attempt_id}.fixture.staging"
    started_at = "2026-07-12T00:00:00Z"
    executor_record = _record(executor_code)
    execution_guard_before = route2_contract.pixal_execution_guard_evidence()
    start_ledger = _write_json(
        source_root
        / ".attempts"
        / asset_id
        / f"{attempt_id}.started.json",
        {
            "schema": "pixal3d_human_attribute_attempt_start_v1",
            "attempt_id": attempt_id,
            "status": "started",
            "case_id": case_id,
            "asset_id": asset_id,
            "base_avatar_id": base_avatar_id,
            "job": _record(pixal_job),
            "executor": executor_record,
            "execution_guard_before": execution_guard_before,
            "argv": argv,
            "started_at_utc": started_at,
            "staging": {"path": str(staging_path), "created": True},
            "publication_policy": "atomic_no_replace",
        },
    )
    start_ledger.chmod(0o444)
    success_sentinel = str(staging_path / pixal_manifest.name)
    execution_log = _write_json(
        asset_root / "execution.log",
        {
            "schema": "pixal3d_human_attribute_execution_log_v1",
            "attempt_id": attempt_id,
            "returncode": 0,
            "logical_argv": argv,
            "staged_command": [
                str(route2_contract.PIXAL_PYTHON_EXECUTABLE),
                *[
                    str(staging_path / pixal_glb.name)
                    if value == str(pixal_glb)
                    else value
                    for value in argv
                ],
            ],
            "stdout": f"fixture Pixal output\n{success_sentinel}\n",
            "stderr": "",
            "success_sentinel": success_sentinel,
        },
    )
    execution_log.chmod(0o444)
    pixal_attempt = _write_json(
        asset_root / "pixal_attempt.json",
        {
            "schema": "pixal3d_human_attribute_attempt_v1",
            "attempt_id": attempt_id,
            "status": "succeeded",
            "case_id": case_id,
            "asset_id": asset_id,
            "base_avatar_id": base_avatar_id,
            "job": _record(pixal_job),
            "executor": executor_record,
            "execution_guard": {
                "before": execution_guard_before,
                "after": execution_guard_before,
                "unchanged": True,
            },
            "start_ledger": _record(start_ledger),
            "execution_log": _record(execution_log),
            "argv": argv,
            "environment": {
                "python_executable": str(route2_contract.PIXAL_PYTHON_EXECUTABLE),
                "python_executable_record": {
                    **_record(route2_contract.PIXAL_PYTHON_EXECUTABLE),
                    "mode": format(
                        route2_contract.PIXAL_PYTHON_EXECUTABLE.stat().st_mode
                        & 0o777,
                        "04o",
                    ),
                },
                "python_version": "3.10.20",
                "torch_version": "2.6.0+cu124",
                "cuda_version": "12.4",
                "cuda_visible_devices": "3",
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_device_name": "fixture-gpu",
                "cuda_device_uuid": "fixture-gpu-uuid",
                "attention_backend": "sdpa",
                "hf_hub_cache": "/data/models/hub",
                "hf_hub_offline": "1",
                "transformers_offline": "1",
                "torch_home": "/data/models/torch",
                "opencv_io_enable_openexr": "1",
                "pytorch_cuda_alloc_conf": "expandable_segments:True",
            },
            "wrapper": _record(wrapper),
            "started_at_utc": started_at,
            "finished_at_utc": "2026-07-12T00:10:00Z",
            "returncode": 0,
            "preflight_reauthenticated": True,
            "postflight_reauthenticated": True,
            "staging": {
                "path": str(staging_path),
                "created": True,
                "preserved_after_success": False,
            },
            "publication": {
                "policy": "atomic_no_replace",
                "glb_published": True,
                "manifest_published": True,
            },
            "model_inventory": {
                "pixal_revision": "0b31f9160aa400719af409098bff7936a932f726",
                "dino_revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881",
                "pixal_snapshot_inventory_sha256": route2_contract._validate_model_snapshot(
                    str(pixal_snapshot),
                    route2_contract.PIXAL3D_REVISION,
                    "fixture Pixal snapshot",
                )["inventory_sha256"],
                "dino_snapshot_inventory_sha256": route2_contract._validate_model_snapshot(
                    str(dino_snapshot),
                    route2_contract.DINO_REVISION,
                    "fixture DINO snapshot",
                )["inventory_sha256"],
            },
            "licenses": {
                "pixal_license_sha256": route2_contract.MODEL_SNAPSHOT_CONTRACTS[
                    route2_contract.PIXAL3D_REVISION
                ]["license_sha256"],
                "dino_license_sha256": route2_contract.MODEL_SNAPSHOT_CONTRACTS[
                    route2_contract.DINO_REVISION
                ]["license_sha256"],
            },
            "output_glb": _record(pixal_glb),
            "output_manifest": _record(pixal_manifest),
            "failure_evidence": [],
        },
    )
    for path in (pixal_glb, pixal_manifest, pixal_attempt):
        path.chmod(0o444)
    if base_qualified is None:
        base_qualified = _write_json(
            source_root / "qualified_candidate_v1.json", {"schema": "fixture"}
        )
    descriptors = {
        "base_qualified_candidate": (
            "base_qualified_candidate",
            base_qualified,
            base_qualified.parent,
        ),
        "attribute_candidate_manifest": (
            "attribute_candidate_manifest",
            candidate_manifest,
            candidate_root,
        ),
        "attribute_agent_decision": (
            "attribute_agent_decision",
            decision,
            source_root,
        ),
        "candidate_rgba": ("agent_accepted_rgba", rgba, candidate_root),
        "pixal_job": ("pixal_attribute_job", pixal_job, source_root),
        "pixal_attempt": ("pixal_attempt_ledger", pixal_attempt, source_root),
        "pixal_manifest": ("pixal_manifest", pixal_manifest, source_root),
        "pixal_pbr_glb": ("pixal_pbr_glb", pixal_glb, source_root),
    }
    return {
        key: {"role": role, "path": path, "root": root}
        for key, (role, path, root) in descriptors.items()
    }


@pytest.fixture
def instance(tmp_path: Path) -> dict[str, Any]:
    instance_contract, _, _ = _modules()
    return _build_base_instance(tmp_path, instance_contract)


def _build_base_instance(
    tmp_path: Path,
    instance_contract: Any,
    *,
    asset_id: str = "rocketbox_male_adult_01",
) -> dict[str, Any]:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    lineage, source_glb, source_manifest = _base_lineage(
        source_root, asset_id
    )
    output_parent = tmp_path / "route2"
    output_parent.mkdir()
    output_root = output_parent / asset_id
    output_root.mkdir()
    payload = instance_contract.build_instance_contract(
        asset_id=asset_id,
        base_avatar_id=asset_id,
        case_id=asset_id,
        case_kind="base_avatar",
        output_root=output_root,
        source_lineage=lineage,
    )
    contract_path = instance_contract.publish_instance_contract(payload)
    return {
        "contract": contract_path,
        "output_root": output_root,
        "source_root": source_root,
        "source_glb": source_glb,
        "source_glb_bytes": source_glb.read_bytes(),
        "source_manifest": source_manifest,
        "payload": payload,
    }


@pytest.fixture(autouse=True)
def _adapt_synthetic_retarget_metrics(monkeypatch, tmp_path):
    """The real deep metrics validator has its own exhaustive suite.

    Contract-layer fixtures retain tiny JSON while still proving that qualification
    calls the owner; the dedicated regression overrides this adapter with a rejection.
    """

    from tools import blender_retarget_rocketbox_to_tokenrig as retarget_owner
    from tools import route2_human_instance_contract as instance_contract

    model_root = tmp_path / "models"
    model_root.mkdir()
    monkeypatch.setattr(instance_contract, "MODEL_ROOT", model_root)
    pixal_snapshot = model_root / instance_contract.PIXAL3D_SNAPSHOT_RELATIVE
    dino_snapshot = model_root / instance_contract.DINO_SNAPSHOT_RELATIVE
    _write(pixal_snapshot / "LICENSE", b"fixture Pixal license\n")
    _write(pixal_snapshot / "pipeline.json", b"{}\n")
    _write(
        pixal_snapshot / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        b"fixture Pixal weights",
    )
    _write(dino_snapshot / "LICENSE.md", b"fixture DINO license\n")
    _write(dino_snapshot / "config.json", b"{}\n")
    _write(dino_snapshot / "model.safetensors", b"fixture DINO weights")
    _write(dino_snapshot / "preprocessor_config.json", b"{}\n")
    monkeypatch.setitem(
        instance_contract.MODEL_SNAPSHOT_CONTRACTS[instance_contract.PIXAL3D_REVISION],
        "license_sha256",
        _sha(pixal_snapshot / "LICENSE"),
    )
    monkeypatch.setitem(
        instance_contract.MODEL_SNAPSHOT_CONTRACTS[instance_contract.DINO_REVISION],
        "license_sha256",
        _sha(dino_snapshot / "LICENSE.md"),
    )
    runtime_executable = _write(
        tmp_path / "pixal_runtime/python3.10",
        b"#!/bin/sh\nexit 0\n",
    )
    runtime_executable.chmod(0o755)
    monkeypatch.setattr(
        instance_contract, "PIXAL_PYTHON_EXECUTABLE", runtime_executable
    )
    monkeypatch.setattr(
        instance_contract,
        "_probe_pixal_python_runtime",
        lambda _executable, _environment: {
            "python_version": "3.10.20",
            "torch_version": "2.6.0+cu124",
            "cuda_version": "12.4",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_name": "fixture-gpu",
            "cuda_device_uuid": "fixture-gpu-uuid",
        },
    )

    def validate(value):
        action_name = value.get("action_name") if isinstance(value, dict) else None
        if action_name not in retarget_owner.ACTION_NAMES.values():
            raise retarget_owner.RetargetError("synthetic action name is invalid")
        return {"status": "passed", "action_name": action_name}

    def build(
        *,
        asset_id,
        base_avatar_id,
        authenticated,
        metrics,
        artifacts,
        command,
        blender_version,
    ):
        del command
        return {
            "schema": "tokenrig_rocketbox_retarget_v1",
            "asset_id": asset_id,
            "base_avatar_id": base_avatar_id,
            "state_classification": "research_candidate",
            "canonical_front": "negative-y",
            "canonical_up": "positive-z",
            "automatic_checks": "passed",
            "user_acceptance": "pending_user_review",
            "environment": {"blender_version": blender_version, "fps": 30},
            "authenticated_inputs": dict(authenticated),
            "actions": {
                name: {"status": "passed", "action_name": name}
                for name in metrics["actions"]
            },
            "artifacts": dict(artifacts),
        }

    monkeypatch.setattr(retarget_owner, "validate_action_metrics", validate)
    monkeypatch.setattr(retarget_owner, "build_retarget_manifest", build)


@pytest.fixture
def accessory_instance(instance: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    instance_contract, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)
    base_qualified = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )
    source_root = tmp_path / "glasses_sources"
    source_root.mkdir()
    lineage = _attribute_lineage(
        source_root,
        base_qualified=base_qualified,
        case_id="glasses",
        base_avatar_id="rocketbox_male_adult_01",
    )
    output_root = tmp_path / "route2_glasses_v1"
    output_root.mkdir()
    payload = instance_contract.build_instance_contract(
        asset_id="route2_glasses_v1",
        base_avatar_id="rocketbox_male_adult_01",
        case_id="glasses",
        case_kind="attribute_instance",
        output_root=output_root,
        source_lineage=lineage,
    )
    contract_path = instance_contract.publish_instance_contract(payload)
    return {
        "contract": contract_path,
        "output_root": output_root,
        "source_root": source_root,
        "source_glb": Path(lineage["pixal_pbr_glb"]["path"]),
        "source_glb_bytes": Path(lineage["pixal_pbr_glb"]["path"]).read_bytes(),
        "source_manifest": Path(lineage["pixal_manifest"]["path"]),
        "payload": payload,
    }


def _make_static_bundle(
    instance: dict[str, Any],
    *,
    branch: str = "direct",
    include_accessory: bool = False,
    accessory_non_head_weight: float = 0.0,
    accessory_descendant: str = "bone_hat",
    direct_female_lineage: dict[str, Any] | None = None,
    direct_female_lineage_field: str = "direct_female_lineage",
) -> Path:
    instance_contract, static_decision, _ = _modules()
    from tools import blender_tokenrig_human_static_audit as static_audit

    contract = instance_contract.validate_instance_contract(instance["contract"])
    branch_root = instance_contract.resolve_branch_root(contract, branch)
    branch_root.mkdir(parents=True, exist_ok=True)
    tokenrig_glb = _write_glb(branch_root / "tokenrig_transfer.glb")
    attempt = _write_json(
        branch_root / "tokenrig_attempt.json",
        {
            "schema": "pixal_tokenrig_attempt_v1",
            "asset_id": contract["asset_id"],
            "status": "succeeded",
        },
    )
    manifest_payload = {
        "schema": {
            "direct": "pixal_tokenrig_canary_v1",
            "fitted_skeleton": "pixal_tokenrig_fitted_skeleton_v1",
            "sanitized_weights": "pixal_tokenrig_sanitized_weights_v1",
        }[branch],
        "asset_id": contract["asset_id"],
        "attempt": "direct_transfer",
        "source_front": "positive-y",
        "canonical_front": "negative-y",
        "input": {
            "glb": _record(instance["source_glb"]),
            "manifest": _record(instance["source_manifest"]),
        },
        "output": _record(tokenrig_glb),
        "attempt_ledger": _record(attempt),
    }
    if direct_female_lineage is not None:
        manifest_payload[direct_female_lineage_field] = direct_female_lineage
    tokenrig_manifest = _write_json(
        branch_root / "tokenrig_manifest.json",
        manifest_payload,
    )
    for path in (tokenrig_glb, attempt, tokenrig_manifest):
        path.chmod(0o444)
    authenticated = static_audit.authenticate_task3_inputs(
        asset_id=contract["asset_id"],
        source_glb=instance["source_glb"],
        tokenrig_glb=tokenrig_glb,
        tokenrig_manifest=tokenrig_manifest,
    )
    static_root = branch_root / static_decision.STATIC_BUNDLE_DIRNAME
    static_root.mkdir()
    artifacts: dict[str, Any] = {}
    for name in static_decision.STATIC_ARTIFACTS:
        if name == "bind_pose.glb":
            path = _write_glb(static_root / name)
        elif name.endswith(".png"):
            path = _write_review_png(static_root / name)
        elif name == "joint_hierarchy.txt":
            path = _write(static_root / name, _fixture_hierarchy_bytes())
        else:  # pragma: no cover - the contract tuple is intentionally exhaustive.
            raise AssertionError(name)
        artifacts[name] = {
            "filename": name,
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }
    if include_accessory:
        overlay = _write_review_png(static_root / "accessory_head_binding.png")
        artifacts[overlay.name] = {
            "filename": overlay.name,
            "sha256": _sha(overlay),
            "size_bytes": overlay.stat().st_size,
        }
        contract_lineage = contract["source_lineage"]
        binding = _write_json(
            static_root / "accessory_head_binding.json",
            {
                "schema": "route2_accessory_head_binding_v1",
                "asset_id": contract["asset_id"],
                "base_avatar_id": contract["base_avatar_id"],
                "case_id": contract["case"]["case_id"],
                "reviewed_candidate_rgba_sha256": contract_lineage["candidate_rgba"]["sha256"],
                "reviewed_candidate_manifest_sha256": contract_lineage[
                    "attribute_candidate_manifest"
                ]["sha256"],
                "head_bone": "bone_head",
                "head_descendant_bones": [accessory_descendant],
                "allowed_bones": ["bone_head", accessory_descendant],
                "accessory_vertex_count": 24,
                "minimum_head_descendant_weight": 1.0 - accessory_non_head_weight,
                "maximum_non_head_descendant_weight": accessory_non_head_weight,
                "vertex_weights": [
                    {
                        "vertex_index": index,
                        "weights": [
                            {
                                "bone": "bone_head",
                                "weight": 1.0,
                            }
                        ],
                    }
                    for index in range(24)
                ],
                "pixel_to_vertex_mapping": {
                    "reviewed_pixel_count": 96,
                    "mapped_vertex_count": 24,
                    "coverage": 1.0,
                    "method": "reviewed_pixels_to_surface_vertices_v1",
                },
                "rigid_binding_passed": accessory_non_head_weight <= 1.0e-6,
                "overlay": artifacts[overlay.name],
                "user_acceptance": "pending_user_review",
            },
        )
        artifacts[binding.name] = {
            "filename": binding.name,
            "sha256": _sha(binding),
            "size_bytes": binding.stat().st_size,
        }
    _write_json(
        static_root / "static_qa.json",
        {
            "schema": "tokenrig_human_static_qa_v1",
            "asset_id": contract["asset_id"],
            "decision": "automatic_static_checks_passed",
            "agent_qa_status": "pending_agent_visual_qa",
            "user_acceptance": "pending_user_review",
            "readiness_bundle_published": True,
            "authenticated": authenticated,
            "checks": _complete_static_checks(),
            "artifacts": artifacts,
        },
    )
    _readonly_bundle(static_root)
    return static_root


def _static_checks(value: bool = True) -> dict[str, bool]:
    _, static_decision, _ = _modules()
    return {name: value for name in static_decision.STATIC_VISUAL_CHECKS}


def _record_static_pass(instance: dict[str, Any], *, branch: str = "direct") -> Path:
    _, static_decision, _ = _modules()
    return static_decision.record_static_agent_visual_decision(
        instance["contract"],
        branch_id=branch,
        status="agent_qa_passed_pending_user_acceptance",
        reviewer="codex-route2-static-review",
        notes="All hash-locked static views are anatomically and materially reasonable.",
        checks=_static_checks(),
    )


def _make_dynamic_bundle(instance: dict[str, Any], *, branch: str = "direct") -> Path:
    instance_contract, static_decision, _ = _modules()
    from tools.spike_rlr import tokenrig_human_review

    contract = instance_contract.validate_instance_contract(instance["contract"])
    branch_root = instance_contract.resolve_branch_root(contract, branch)
    static_root = branch_root / static_decision.STATIC_BUNDLE_DIRNAME
    bind_pose = static_root / "bind_pose.glb"
    static_qa = static_root / "static_qa.json"

    retarget_root = branch_root / "retarget_v1"
    retarget_root.mkdir()
    animated = _write(retarget_root / "animated.blend", b"animated blend")
    walking = _write_glb(retarget_root / "walking.glb", action="Walking")
    standing_idle = _write_glb(
        retarget_root / "standing_idle.glb", action="Standing_Idle"
    )
    metrics = _write_json(
        retarget_root / "retarget_metrics.json",
        {
            "schema": "tokenrig_rocketbox_retarget_metrics_v1",
            "automatic_checks": "passed",
            "actions": {
                "Walking": {"action_name": "Walking"},
                "Standing_Idle": {"action_name": "Standing_Idle"},
            },
        },
    )
    retarget_manifest = _write_json(
        retarget_root / "retarget_manifest.json",
        {
            "schema": "tokenrig_rocketbox_retarget_v1",
            "asset_id": contract["asset_id"],
            "base_avatar_id": contract["base_avatar_id"],
            "state_classification": "research_candidate",
            "canonical_front": "negative-y",
            "canonical_up": "positive-z",
            "automatic_checks": "passed",
            "user_acceptance": "pending_user_review",
            "environment": {"blender_version": "4.2.1", "fps": 30},
            "authenticated_inputs": {
                "static": {
                    "bind_pose": _record(bind_pose),
                    "static_qa": _record(static_qa),
                    "floor_z_m": 0.0,
                    "axis_map_3x3": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            },
            "actions": {
                "Walking": {"status": "passed", "action_name": "Walking"},
                "Standing_Idle": {
                    "status": "passed",
                    "action_name": "Standing_Idle",
                },
            },
            "artifacts": {
                "animated.blend": {
                    **_record(animated),
                    "path": "animated.blend",
                },
                "walking.glb": {**_record(walking), "path": "walking.glb"},
                "standing_idle.glb": {
                    **_record(standing_idle),
                    "path": "standing_idle.glb",
                },
                "retarget_metrics.json": {
                    **_record(metrics),
                    "path": "retarget_metrics.json",
                },
            },
        },
    )
    _readonly_bundle(retarget_root)

    execution_root = instance["source_root"] / "execution"
    execution_root.mkdir()
    renderer = _write(execution_root / "renderer.py", b"# renderer")
    ffmpeg = _write(execution_root / "ffmpeg", b"ffmpeg executable")
    ffprobe = _write(execution_root / "ffprobe", b"ffprobe executable")
    execution = {
        "renderer": _record(renderer),
        "ffmpeg": {**_record(ffmpeg), "version": "ffmpeg version fixture"},
        "ffprobe": {**_record(ffprobe), "version": "ffprobe version fixture"},
    }

    review_root = branch_root / "dynamic_review_v1"
    review_root.mkdir()
    actions: dict[str, Any] = {}
    qa_actions: dict[str, Any] = {}
    for motion, action_name in tokenrig_human_review.MOTIONS.items():
        views: dict[str, Any] = {}
        qa_views: dict[str, Any] = {}
        for view in tokenrig_human_review.VIEWS:
            png = _write(
                review_root / f"{motion}_{view}.png",
                f"{motion} {view} png".encode(),
            )
            mp4 = _write(
                review_root / f"{motion}_{view}.mp4",
                f"{motion} {view} mp4".encode(),
            )
            views[view] = {
                "png": _record(png, filename=png.name),
                "mp4": _record(mp4, filename=mp4.name),
            }
            qa_views[view] = {"passed": True}
        actions[motion] = {
            "action_name": action_name,
            "fps": 30,
            "views": views,
        }
        qa_actions[motion] = qa_views
    media_qa = _write_json(
        review_root / "media_qa.json",
        {
            "schema": "tokenrig_human_media_qa_v1",
            "asset_id": contract["asset_id"],
            "actions": qa_actions,
            "automatic_checks": "passed",
        },
    )
    static_evidence = {
        name: _record(static_root / name)
        for name in tokenrig_human_review.STATIC_EVIDENCE
    }
    _write_json(
        review_root / "review_manifest.json",
        {
            "schema": "tokenrig_human_dynamic_review_v1",
            "asset_id": contract["asset_id"],
            "display_label": "Route-2 fixture",
            "instance_kind": contract["case"]["kind"],
            "state_classification": "research_candidate",
            "canonical_front": "negative-y",
            "canonical_up": "positive-z",
            "fixed_floor_z_m": 0.0,
            "upstream": {
                "asset_id": contract["asset_id"],
                "static_qa": _record(static_qa),
                "bind_pose": _record(bind_pose),
                "retarget_manifest": _record(retarget_manifest),
                "retarget_metrics": _record(metrics),
                "glbs": {
                    "walking": _record(walking),
                    "standing_idle": _record(standing_idle),
                },
                "static_evidence": static_evidence,
            },
            "actions": actions,
            "media_qa": _record(media_qa, filename="media_qa.json"),
            "execution": execution,
            "automatic_checks": "passed",
            "agent_visual_qa": "pending_agent_visual_qa",
            "user_acceptance": "pending_user_review",
        },
    )
    _readonly_bundle(review_root)
    decision = tokenrig_human_review.record_agent_visual_qa(
        review_root,
        status=tokenrig_human_review.PASS_STATUS,
        reviewer="codex-route2-dynamic-review",
        notes="Walking and Standing Idle media are visually reasonable.",
        checks={name: True for name in tokenrig_human_review.AGENT_VISUAL_CHECKS},
    )
    assert decision.stat().st_mode & 0o777 == 0o444
    return review_root


def test_contract_modules_exist_before_behavior_is_exercised():
    missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    assert missing == []


def test_instance_contract_is_canonical_hash_locked_and_immutable(instance):
    instance_contract, _, _ = _modules()
    payload = instance_contract.validate_instance_contract(instance["contract"])

    assert payload["schema"] == "route2_human_instance_contract_v1"
    assert payload["asset_id"] == "rocketbox_male_adult_01"
    assert payload["base_avatar_id"] == "rocketbox_male_adult_01"
    assert payload["case"] == {
        "case_id": "rocketbox_male_adult_01",
        "kind": "base_avatar",
    }
    assert payload["canonical_output_root"] == str(instance["output_root"].resolve())
    assert payload["allowed_branch_dag"] == instance_contract.DEFAULT_BRANCH_DAG
    assert set(payload["source_lineage"]) == set(instance_contract.BASE_LINEAGE_ROLES)
    assert payload["source_lineage"]["pixal_pbr_glb"]["sha256"] == _sha(
        instance["source_glb"]
    )
    assert instance["contract"].name == "route2_human_instance_contract_v1.json"
    assert instance["contract"].stat().st_mode & 0o777 == 0o444
    assert "user_approved" not in instance["contract"].read_text()

    with pytest.raises(instance_contract.InstanceContractError, match="already exists"):
        instance_contract.publish_instance_contract(instance["payload"])


def test_instance_contract_rejects_readable_pixal_glb_without_pbr_materials(tmp_path):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    lineage, _, _ = _base_lineage(
        source_root,
        "rocketbox_male_adult_01",
        include_pbr=False,
    )
    output_root = tmp_path / "rocketbox_male_adult_01"
    output_root.mkdir()

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="PBR|material|texture|image",
    ):
        instance_contract.build_instance_contract(
            asset_id="rocketbox_male_adult_01",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="rocketbox_male_adult_01",
            case_kind="base_avatar",
            output_root=output_root,
            source_lineage=lineage,
        )


def test_instance_contract_rejects_pixal_glb_with_external_texture_uri(tmp_path):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    lineage, _, _ = _base_lineage(
        source_root,
        "rocketbox_male_adult_01",
        pbr_image_uri="textures/not_published.png",
    )
    output_root = tmp_path / "rocketbox_male_adult_01"
    output_root.mkdir()

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="packed|data URI|image|PBR",
    ):
        instance_contract.build_instance_contract(
            asset_id="rocketbox_male_adult_01",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="rocketbox_male_adult_01",
            case_kind="base_avatar",
            output_root=output_root,
            source_lineage=lineage,
        )


@pytest.mark.parametrize(
    ("lineage_options", "message"),
    [
        ({"include_binary_chunk": False}, "BIN|binary"),
        ({"corrupt_pbr_buffer_view": True}, "buffer|range|image"),
        (
            {"pbr_image_uri": "data:image/png;base64,not-valid-base64!"},
            "base64|image|data URI",
        ),
    ],
)
def test_instance_contract_rejects_forged_pbr_payload_bytes(
    tmp_path,
    lineage_options,
    message,
):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    lineage, _, _ = _base_lineage(
        source_root,
        "rocketbox_male_adult_01",
        **lineage_options,
    )
    output_root = tmp_path / "rocketbox_male_adult_01"
    output_root.mkdir()

    with pytest.raises(instance_contract.InstanceContractError, match=message):
        instance_contract.build_instance_contract(
            asset_id="rocketbox_male_adult_01",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="rocketbox_male_adult_01",
            case_kind="base_avatar",
            output_root=output_root,
            source_lineage=lineage,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_id", "../person"),
        ("base_avatar_id", "Person-01"),
        ("case_id", "bad/case"),
        ("case_kind", "user_approved"),
    ],
)
def test_instance_contract_rejects_noncanonical_ids(tmp_path, field, value):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = _write(source_root / "source.glb", b"source")
    output_root = tmp_path / "rocketbox_male_adult_01"
    output_root.mkdir()
    arguments = {
        "asset_id": "rocketbox_male_adult_01",
        "base_avatar_id": "rocketbox_male_adult_01",
        "case_id": "rocketbox_male_adult_01",
        "case_kind": "base_avatar",
        "output_root": output_root,
        "source_lineage": {
            "source": {"role": "pixal_pbr_glb", "path": source, "root": source_root}
        },
    }
    arguments[field] = value

    with pytest.raises(instance_contract.InstanceContractError, match=field):
        instance_contract.build_instance_contract(**arguments)


def test_instance_contract_rejects_output_or_lineage_escape(tmp_path):
    instance_contract, _, _ = _modules()
    output_root = tmp_path / "wrong_root"
    output_root.mkdir()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    outside = _write(tmp_path / "outside.glb", b"outside")

    with pytest.raises(instance_contract.InstanceContractError, match="output root.*asset_id"):
        instance_contract.build_instance_contract(
            asset_id="rocketbox_male_adult_01",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="rocketbox_male_adult_01",
            case_kind="base_avatar",
            output_root=output_root,
            source_lineage={
                "source": {
                    "role": "pixal_pbr_glb",
                    "path": outside,
                    "root": source_root,
                }
            },
        )

    canonical_root = tmp_path / "rocketbox_male_adult_01"
    canonical_root.mkdir()
    lineage, _, _ = _base_lineage(source_root, "rocketbox_male_adult_01")
    lineage["pixal_pbr_glb"] = {
        "role": "pixal_pbr_glb",
        "path": outside,
        "root": source_root,
    }
    with pytest.raises(instance_contract.InstanceContractError, match="outside.*root"):
        instance_contract.build_instance_contract(
            asset_id="rocketbox_male_adult_01",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="rocketbox_male_adult_01",
            case_kind="base_avatar",
            output_root=canonical_root,
            source_lineage=lineage,
        )


def test_instance_contract_rejects_changed_source_bytes_and_mutable_contract(instance):
    instance_contract, _, _ = _modules()
    instance["source_glb"].write_bytes(b"changed")
    with pytest.raises(instance_contract.InstanceContractError, match="source lineage.*SHA-256"):
        instance_contract.validate_instance_contract(instance["contract"])

    instance["source_glb"].write_bytes(instance["source_glb_bytes"])
    instance["contract"].chmod(0o644)
    with pytest.raises(instance_contract.InstanceContractError, match="mode 0444"):
        instance_contract.validate_instance_contract(instance["contract"])


def test_branch_dag_rejects_escape_unknown_edges_and_cycles():
    instance_contract, _, _ = _modules()
    escape = json.loads(json.dumps(instance_contract.DEFAULT_BRANCH_DAG))
    escape["nodes"][1]["relative_root"] = "../escape"
    with pytest.raises(instance_contract.InstanceContractError, match="relative_root"):
        instance_contract.validate_branch_dag(escape)

    unknown = json.loads(json.dumps(instance_contract.DEFAULT_BRANCH_DAG))
    unknown["edges"].append(["direct", "unknown"])
    with pytest.raises(instance_contract.InstanceContractError, match="unknown branch"):
        instance_contract.validate_branch_dag(unknown)

    cycle = json.loads(json.dumps(instance_contract.DEFAULT_BRANCH_DAG))
    cycle["edges"].append(["sanitized_weights", "direct"])
    with pytest.raises(instance_contract.InstanceContractError, match="cycle"):
        instance_contract.validate_branch_dag(cycle)


def test_static_decision_hash_locks_every_evidence_byte(instance):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(instance)
    decision_path = _record_static_pass(instance)
    payload = static_decision.validate_static_agent_visual_decision(
        instance["contract"], branch_id="direct", require_pass=True
    )

    assert decision_path.name == "static_audit_v1.agent_visual_qa.json"
    assert decision_path.stat().st_mode & 0o777 == 0o444
    assert payload["schema"] == "route2_human_static_agent_visual_qa_v1"
    assert payload["status"] == "agent_qa_passed_pending_user_acceptance"
    assert set(payload["snapshot"]["artifacts"]) == {
        "static_qa.json",
        *static_decision.STATIC_ARTIFACTS,
    }
    assert payload["snapshot"]["artifacts"]["bind_pose.glb"]["sha256"] == _sha(
        static_root / "bind_pose.glb"
    )
    assert "user_approved" not in decision_path.read_text()

    with pytest.raises(static_decision.StaticDecisionError, match="already exists"):
        _record_static_pass(instance)


def test_static_decision_rejects_invalid_status_or_incomplete_pass(instance):
    _, static_decision, _ = _modules()
    _make_static_bundle(instance)

    with pytest.raises(static_decision.StaticDecisionError, match="status"):
        static_decision.record_static_agent_visual_decision(
            instance["contract"],
            branch_id="direct",
            status="user_approved",
            reviewer="agent",
            notes="invalid status",
            checks=_static_checks(),
        )

    checks = _static_checks()
    checks[next(iter(checks))] = False
    with pytest.raises(static_decision.StaticDecisionError, match="all visual checks"):
        static_decision.record_static_agent_visual_decision(
            instance["contract"],
            branch_id="direct",
            status="agent_qa_passed_pending_user_acceptance",
            reviewer="agent",
            notes="one failed check",
            checks=checks,
        )


def test_static_rejection_is_immutable_but_not_qualifiable(instance):
    _, static_decision, _ = _modules()
    _make_static_bundle(instance)
    checks = _static_checks()
    checks[next(iter(checks))] = False
    static_decision.record_static_agent_visual_decision(
        instance["contract"],
        branch_id="direct",
        status="rejected",
        reviewer="agent",
        notes="The left foot visibly hovers.",
        checks=checks,
    )

    payload = static_decision.validate_static_agent_visual_decision(
        instance["contract"], branch_id="direct"
    )
    assert payload["status"] == "rejected"
    with pytest.raises(static_decision.StaticDecisionNotAccepted, match="rejected"):
        static_decision.validate_static_agent_visual_decision(
            instance["contract"], branch_id="direct", require_pass=True
        )


def test_static_decision_detects_tamper_and_noncanonical_modes(instance):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(instance)
    _record_static_pass(instance)
    evidence = static_root / "bind_side.png"
    evidence.chmod(0o644)
    evidence.write_bytes(b"tampered side")

    with pytest.raises(static_decision.StaticDecisionError, match="snapshot changed|mode 0444"):
        static_decision.validate_static_agent_visual_decision(
            instance["contract"], branch_id="direct", require_pass=True
        )


def test_qualified_candidate_points_to_actual_branch_without_copying(instance):
    instance_contract, static_decision, qualified = _modules()
    static_root = _make_static_bundle(instance)
    static_path = _record_static_pass(instance)
    review_root = _make_dynamic_bundle(instance)
    from tools.spike_rlr import tokenrig_human_review

    dynamic_decision = tokenrig_human_review.agent_decision_path(review_root)
    before = {
        path: (path.stat().st_ino, _sha(path))
        for path in (static_root / "bind_pose.glb", static_path, dynamic_decision)
    }
    destination = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )
    payload = qualified.validate_qualified_candidate(destination)

    assert destination == instance["output_root"] / "qualified_candidate_v1.json"
    assert destination.stat().st_mode & 0o777 == 0o444
    assert payload["schema"] == "route2_human_qualified_candidate_v1"
    assert payload["status"] == "agent_qa_passed_pending_user_acceptance"
    assert payload["final_branch"] == {
        "branch_id": "direct",
        "path": str(instance["output_root"].resolve()),
        "relative_root": ".",
    }
    assert payload["static"]["decision"]["path"] == str(static_path.resolve())
    assert payload["dynamic"]["review_dir"] == str(review_root.resolve())
    assert payload["dynamic"]["decision"]["path"] == str(dynamic_decision.resolve())
    assert {item["relative_path"] for item in payload["inventory"]} == {
        "route2_human_instance_contract_v1.json",
        "static_audit_v1.agent_visual_qa.json",
        "dynamic_review_v1.agent_visual_qa.json",
        *{f"static_audit_v1/{name}" for name in ("static_qa.json", *static_decision.STATIC_ARTIFACTS)},
        *{
            f"retarget_v1/{name}"
            for name in (
                "animated.blend",
                "walking.glb",
                "standing_idle.glb",
                "retarget_metrics.json",
                "retarget_manifest.json",
            )
        },
        *{
            f"dynamic_review_v1/{name}"
            for name in (
                "review_manifest.json",
                "media_qa.json",
                *(
                    f"{motion}_{view}.{kind}"
                    for motion in ("walking", "standing_idle")
                    for view in ("front", "side", "top", "feet", "skeleton")
                    for kind in ("png", "mp4")
                ),
            )
        },
    }
    assert all(item["mode"] == "0444" for item in payload["inventory"])
    assert "user_approved" not in destination.read_text()
    for path, state in before.items():
        assert (path.stat().st_ino, _sha(path)) == state

    assert instance_contract.resolve_branch_root(
        instance_contract.validate_instance_contract(instance["contract"]), "direct"
    ) == instance["output_root"].resolve()


def test_qualified_candidate_is_no_replace_and_revalidates_every_hash(instance):
    _, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    review_root = _make_dynamic_bundle(instance)
    destination = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )

    with pytest.raises(qualified.QualificationError, match="already exists"):
        qualified.publish_qualified_candidate(instance["contract"], branch_id="direct")

    media = review_root / "walking_feet.mp4"
    media.chmod(0o644)
    media.write_bytes(b"tampered media")
    with pytest.raises(qualified.QualificationError, match="snapshot changed|mode 0444|review"):
        qualified.validate_qualified_candidate(destination)


def test_qualification_rejects_wrong_branch_or_nonpassed_dynamic_decision(instance):
    _, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)

    with pytest.raises(qualified.QualificationError, match="retarget|dynamic review"):
        qualified.publish_qualified_candidate(instance["contract"], branch_id="direct")

    with pytest.raises(qualified.QualificationError, match="branch"):
        qualified.publish_qualified_candidate(
            instance["contract"], branch_id="sanitized_weights"
        )


def test_qualification_rejects_symlinked_or_mutable_bundle_entry(instance):
    _, _, qualified = _modules()
    static_root = _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)

    static_root.chmod(0o755)
    bind = static_root / "bind_pose.glb"
    bind.chmod(0o644)
    real = static_root / "bind_pose.real.glb"
    bind.rename(real)
    bind.symlink_to(real.name)
    static_root.chmod(0o555)
    with pytest.raises(qualified.QualificationError, match="symlink|regular file|snapshot"):
        qualified.publish_qualified_candidate(instance["contract"], branch_id="direct")


def test_attribute_instance_contract_retains_base_identity(instance, tmp_path):
    instance_contract, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)
    base_qualified = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    lineage = _attribute_lineage(source_root, base_qualified=base_qualified)
    output_root = tmp_path / "route2_tall_man_v1"
    output_root.mkdir()

    payload = instance_contract.build_instance_contract(
        asset_id="route2_tall_man_v1",
        base_avatar_id="rocketbox_male_adult_01",
        case_id="tall_man",
        case_kind="attribute_instance",
        output_root=output_root,
        source_lineage=lineage,
    )
    path = instance_contract.publish_instance_contract(payload)
    validated = instance_contract.validate_instance_contract(path)

    assert validated["asset_id"] == "route2_tall_man_v1"
    assert validated["base_avatar_id"] == "rocketbox_male_adult_01"
    assert validated["case"] == {
        "case_id": "tall_man",
        "kind": "attribute_instance",
    }


def test_instance_contract_carries_the_pinned_coordinate_frame(instance):
    instance_contract, _, _ = _modules()
    payload = instance_contract.validate_instance_contract(instance["contract"])

    assert payload["coordinate_frame"] == {
        "front": "negative-y",
        "up": "positive-z",
        "floor_z_m": 0.0,
    }


def test_instance_contract_rejects_unpinned_custom_branch_dag(tmp_path):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = _write(source_root / "source.glb", b"source")
    output_root = tmp_path / "rocketbox_male_adult_01"
    output_root.mkdir()
    custom = json.loads(json.dumps(instance_contract.DEFAULT_BRANCH_DAG))
    custom["nodes"].append({"branch_id": "rocketbox_swap", "relative_root": "rocketbox_swap"})
    custom["edges"].append(["direct", "rocketbox_swap"])
    custom["finalizable_branches"].append("rocketbox_swap")

    with pytest.raises(instance_contract.InstanceContractError, match="pinned branch DAG"):
        instance_contract.build_instance_contract(
            asset_id="rocketbox_male_adult_01",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="rocketbox_male_adult_01",
            case_kind="base_avatar",
            output_root=output_root,
            source_lineage={
                "source": {"role": "pixal_pbr_glb", "path": source, "root": source_root}
            },
            allowed_branch_dag=custom,
        )


@pytest.mark.parametrize(
    ("case_id", "expected_base"),
    [
        ("tall_man", "rocketbox_male_adult_01"),
        ("short_woman", "rocketbox_female_adult_01"),
        ("glasses", "rocketbox_male_adult_01"),
        ("hat", "rocketbox_female_adult_01"),
        ("short_sleeve_color", "rocketbox_male_adult_01"),
        ("trousers", "rocketbox_female_adult_01"),
        ("shoes", "rocketbox_male_adult_01"),
    ],
)
def test_attribute_case_profile_rejects_wrong_base(tmp_path, case_id, expected_base):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _write(source_root / "candidate_rgba.png", b"RGBA")
    asset_id = f"route2_{case_id}_v1"
    output_root = tmp_path / asset_id
    output_root.mkdir()
    wrong_base = (
        "rocketbox_female_adult_01"
        if expected_base == "rocketbox_male_adult_01"
        else "rocketbox_male_adult_01"
    )

    with pytest.raises(instance_contract.InstanceContractError, match="base_avatar_id"):
        instance_contract.build_instance_contract(
            asset_id=asset_id,
            base_avatar_id=wrong_base,
            case_id=case_id,
            case_kind="attribute_instance",
            output_root=output_root,
            source_lineage={
                "candidate_rgba": {
                    "role": "agent_accepted_rgba",
                    "path": source,
                    "root": source_root,
                }
            },
        )


def test_attribute_contract_rejects_incomplete_lineage(tmp_path):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = _write(source_root / "candidate_rgba.png", b"RGBA")
    output_root = tmp_path / "route2_tall_man_v1"
    output_root.mkdir()

    with pytest.raises(instance_contract.InstanceContractError, match="source_lineage roles"):
        instance_contract.build_instance_contract(
            asset_id="route2_tall_man_v1",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="tall_man",
            case_kind="attribute_instance",
            output_root=output_root,
            source_lineage={
                "candidate_rgba": {
                    "role": "agent_accepted_rgba",
                    "path": source,
                    "root": source_root,
                }
            },
        )


def test_qualification_rejects_dynamic_review_that_reuses_idle_as_walk(instance):
    _, _, qualified = _modules()
    from tools.spike_rlr import tokenrig_human_review

    _make_static_bundle(instance)
    _record_static_pass(instance)
    review_root = _make_dynamic_bundle(instance)
    decision = tokenrig_human_review.agent_decision_path(review_root)
    decision.unlink()
    manifest_path = review_root / "review_manifest.json"
    review_root.chmod(0o755)
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["upstream"]["glbs"]["walking"] = dict(
        manifest["upstream"]["glbs"]["standing_idle"]
    )
    _write_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    review_root.chmod(0o555)
    tokenrig_human_review.record_agent_visual_qa(
        review_root,
        status=tokenrig_human_review.PASS_STATUS,
        reviewer="codex-route2-dynamic-review",
        notes="Deliberately stale motion binding for the contract regression.",
        checks={name: True for name in tokenrig_human_review.AGENT_VISUAL_CHECKS},
    )

    with pytest.raises(qualified.QualificationError, match="walking.glb|Walking|distinct"):
        qualified.build_qualified_candidate(instance["contract"], branch_id="direct")


def test_static_decision_rejects_malformed_glb_or_png(instance):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(instance)
    (static_root / "bind_front.png").chmod(0o644)
    (static_root / "bind_front.png").write_bytes(b"not a PNG")
    (static_root / "bind_front.png").chmod(0o444)
    static_qa_path = static_root / "static_qa.json"
    static_root.chmod(0o755)
    static_qa_path.chmod(0o644)
    payload = json.loads(static_qa_path.read_text())
    payload["artifacts"]["bind_front.png"] = {
        "filename": "bind_front.png",
        "sha256": _sha(static_root / "bind_front.png"),
        "size_bytes": (static_root / "bind_front.png").stat().st_size,
    }
    _write_json(static_qa_path, payload)
    static_qa_path.chmod(0o444)
    static_root.chmod(0o555)

    with pytest.raises(static_decision.StaticDecisionError, match="PNG|static artifact"):
        _record_static_pass(instance)


def test_static_decision_rejects_bind_glb_without_skin(instance):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(instance)
    bind = static_root / "bind_pose.glb"
    static_qa_path = static_root / "static_qa.json"
    static_root.chmod(0o755)
    bind.chmod(0o644)
    _write_glb(bind, include_skin=False)
    bind.chmod(0o444)
    static_qa_path.chmod(0o644)
    payload = json.loads(static_qa_path.read_text())
    payload["artifacts"]["bind_pose.glb"] = {
        "filename": "bind_pose.glb",
        "sha256": _sha(bind),
        "size_bytes": bind.stat().st_size,
    }
    _write_json(static_qa_path, payload)
    static_qa_path.chmod(0o444)
    static_root.chmod(0o555)

    with pytest.raises(static_decision.StaticDecisionError, match="skin|bind_pose"):
        _record_static_pass(instance)


def test_static_decision_requires_every_automatic_static_check(instance):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(instance)
    static_qa_path = static_root / "static_qa.json"
    static_root.chmod(0o755)
    static_qa_path.chmod(0o644)
    payload = json.loads(static_qa_path.read_text())
    payload["checks"].pop("weights", None)
    _write_json(static_qa_path, payload)
    static_qa_path.chmod(0o444)
    static_root.chmod(0o555)

    with pytest.raises(static_decision.StaticDecisionError, match="static checks|weights"):
        _record_static_pass(instance)


def test_static_decision_rejects_present_but_inconsistent_checks(instance):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(instance)
    static_qa_path = static_root / "static_qa.json"
    original = json.loads(static_qa_path.read_text())
    mutations = (
        ("zero_mesh", "mesh contract"),
        ("inconsistent_removal", "removal evidence"),
        ("empty_head", "semantic bone"),
        ("swapped_hand_foot", "semantic hierarchy"),
        ("arm_chain_on_leg", "semantic hierarchy"),
        ("nan_weight", "finite number"),
    )
    for mutation, message in mutations:
        payload = json.loads(json.dumps(original))
        if mutation == "zero_mesh":
            payload["checks"]["canonical_mesh_contract"]["vertex_count"] = 0
        elif mutation == "inconsistent_removal":
            payload["checks"]["source_removed_gltf_import_helpers"] = {
                "passed": True
            }
        elif mutation == "empty_head":
            payload["checks"]["semantic_mapping"]["semantic_bones"]["head"] = ""
        elif mutation == "swapped_hand_foot":
            payload["checks"]["semantic_mapping"]["semantic_bones"][
                "left_hand"
            ] = payload["checks"]["semantic_mapping"]["semantic_bones"][
                "left_foot"
            ]
        elif mutation == "arm_chain_on_leg":
            semantics = payload["checks"]["semantic_mapping"]["semantic_bones"]
            semantics["left_upper_arm"] = semantics["left_thigh"]
            semantics["left_forearm"] = semantics["left_calf"]
            semantics["left_hand"] = semantics["left_foot"]
        else:
            payload["checks"]["weights"]["maximum_weight_sum_error"] = float("nan")
        static_root.chmod(0o755)
        static_qa_path.chmod(0o644)
        _write_json(static_qa_path, payload)
        static_qa_path.chmod(0o444)
        static_root.chmod(0o555)

        with pytest.raises(static_decision.StaticDecisionError, match=message):
            static_decision.validated_static_snapshot(
                instance["contract"], branch_id="direct"
            )


def test_static_decision_requires_full_rest_and_inverse_bind_roundtrip(instance):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(instance)
    static_qa_path = static_root / "static_qa.json"
    static_root.chmod(0o755)
    static_qa_path.chmod(0o644)
    payload = json.loads(static_qa_path.read_text())
    payload["checks"]["glb_roundtrip"].pop("full_rest", None)
    payload["checks"]["glb_roundtrip"].pop("inverse_bind", None)
    _write_json(static_qa_path, payload)
    static_qa_path.chmod(0o444)
    static_root.chmod(0o555)

    with pytest.raises(static_decision.StaticDecisionError, match="full_rest|inverse_bind"):
        _record_static_pass(instance)


def test_static_decision_revalidates_branch_owner_before_animation(instance):
    _, static_decision, _ = _modules()
    _make_static_bundle(instance)
    (instance["output_root"] / "tokenrig_transfer.glb").unlink()

    with pytest.raises(static_decision.StaticDecisionError, match="producer|TokenRig|branch"):
        _record_static_pass(instance)


def test_static_decision_requires_readonly_branch_producer(instance):
    _, static_decision, _ = _modules()
    _make_static_bundle(instance)
    (instance["output_root"] / "tokenrig_transfer.glb").chmod(0o644)

    with pytest.raises(static_decision.StaticDecisionError, match="mode 0444|readonly"):
        _record_static_pass(instance)


def test_qualification_rejects_retarget_static_hash_mismatch(instance):
    _, _, qualified = _modules()
    from tools.spike_rlr import tokenrig_human_review

    _make_static_bundle(instance)
    _record_static_pass(instance)
    review_root = _make_dynamic_bundle(instance)
    decision = tokenrig_human_review.agent_decision_path(review_root)
    decision.unlink()
    retarget_manifest_path = review_root.parent / "retarget_v1" / "retarget_manifest.json"
    retarget_root = retarget_manifest_path.parent
    retarget_root.chmod(0o755)
    retarget_manifest_path.chmod(0o644)
    retarget_manifest = json.loads(retarget_manifest_path.read_text())
    retarget_manifest["authenticated_inputs"]["static"]["static_qa"]["sha256"] = "0" * 64
    _write_json(retarget_manifest_path, retarget_manifest)
    retarget_manifest_path.chmod(0o444)
    retarget_root.chmod(0o555)

    review_manifest_path = review_root / "review_manifest.json"
    review_root.chmod(0o755)
    review_manifest_path.chmod(0o644)
    review_manifest = json.loads(review_manifest_path.read_text())
    review_manifest["upstream"]["retarget_manifest"] = _record(retarget_manifest_path)
    _write_json(review_manifest_path, review_manifest)
    review_manifest_path.chmod(0o444)
    review_root.chmod(0o555)
    tokenrig_human_review.record_agent_visual_qa(
        review_root,
        status=tokenrig_human_review.PASS_STATUS,
        reviewer="codex-route2-dynamic-review",
        notes="Deliberately wrong static lineage for the contract regression.",
        checks={name: True for name in tokenrig_human_review.AGENT_VISUAL_CHECKS},
    )

    with pytest.raises(qualified.QualificationError, match="retarget static QA hash|static"):
        qualified.build_qualified_candidate(instance["contract"], branch_id="direct")


def test_qualification_calls_full_retarget_action_metrics_validator(instance, monkeypatch):
    _, _, qualified = _modules()
    from tools import blender_retarget_rocketbox_to_tokenrig as retarget_owner

    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)

    def reject_metrics(_value):
        raise retarget_owner.RetargetError("deep action metrics gate invoked")

    monkeypatch.setattr(retarget_owner, "validate_action_metrics", reject_metrics)
    with pytest.raises(qualified.QualificationError, match="deep action metrics gate"):
        qualified.build_qualified_candidate(instance["contract"], branch_id="direct")


def test_qualification_rebuilds_the_canonical_retarget_manifest(instance, monkeypatch):
    _, _, qualified = _modules()
    from tools import blender_retarget_rocketbox_to_tokenrig as retarget_owner

    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)

    def reject_manifest(**_kwargs):
        raise retarget_owner.RetargetError("canonical retarget manifest builder invoked")

    monkeypatch.setattr(retarget_owner, "build_retarget_manifest", reject_manifest)
    with pytest.raises(qualified.QualificationError, match="canonical retarget manifest builder"):
        qualified.build_qualified_candidate(instance["contract"], branch_id="direct")


def test_base_lineage_rejects_non_pixal_manifest(tmp_path):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    lineage, _, pixal_manifest = _base_lineage(
        source_root, "rocketbox_male_adult_01"
    )
    payload = json.loads(pixal_manifest.read_text())
    payload["backend"] = "hunyuan3d"
    _write_json(pixal_manifest, payload)
    output_root = tmp_path / "rocketbox_male_adult_01"
    output_root.mkdir()

    with pytest.raises(instance_contract.InstanceContractError, match="Pixal|pixal"):
        instance_contract.build_instance_contract(
            asset_id="rocketbox_male_adult_01",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="rocketbox_male_adult_01",
            case_kind="base_avatar",
            output_root=output_root,
            source_lineage=lineage,
        )


def test_model_snapshots_must_remain_below_the_canonical_model_root(tmp_path):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    lineage, _, pixal_manifest = _base_lineage(
        source_root, "rocketbox_male_adult_01"
    )
    outside = (
        tmp_path.parent
        / f"{tmp_path.name}_outside_models"
        / "0b31f9160aa400719af409098bff7936a932f726"
    )
    outside.mkdir(parents=True)
    payload = json.loads(pixal_manifest.read_text())
    payload["model"]["snapshot"] = str(outside)
    _write_json(pixal_manifest, payload)
    output_root = tmp_path / "rocketbox_male_adult_01"
    output_root.mkdir()

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="model root|/data/models|canonical fixed snapshot",
    ):
        instance_contract.build_instance_contract(
            asset_id="rocketbox_male_adult_01",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="rocketbox_male_adult_01",
            case_kind="base_avatar",
            output_root=output_root,
            source_lineage=lineage,
        )


def test_attribute_lineage_requires_current_base_qualification(tmp_path):
    instance_contract, _, qualified = _modules()
    _, base_qualified = _qualified_base_fixture(
        tmp_path / "base",
        instance_contract,
        qualified,
        asset_id="rocketbox_male_adult_01",
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    lineage = _attribute_lineage(source_root, base_qualified=base_qualified)
    base_qualified.chmod(0o644)
    _write_json(base_qualified, {"schema": "changed_base_qualification"})
    base_qualified.chmod(0o444)
    output_root = tmp_path / "route2_tall_man_v1"
    output_root.mkdir()

    with pytest.raises(instance_contract.InstanceContractError, match="base qualified"):
        instance_contract.build_instance_contract(
            asset_id="route2_tall_man_v1",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="tall_man",
            case_kind="attribute_instance",
            output_root=output_root,
            source_lineage=lineage,
        )


def test_attribute_pixal_attempt_requires_complete_execution_provenance(instance, tmp_path):
    instance_contract, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)
    base_qualified = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )
    source_root = tmp_path / "attribute_attempt_sources"
    source_root.mkdir()
    lineage = _attribute_lineage(source_root, base_qualified=base_qualified)
    attempt = Path(lineage["pixal_attempt"]["path"])
    payload = json.loads(attempt.read_text())
    payload.pop("environment", None)
    attempt.chmod(0o644)
    _write_json(attempt, payload)
    attempt.chmod(0o444)
    output_root = tmp_path / "route2_tall_man_v1"
    output_root.mkdir()

    with pytest.raises(instance_contract.InstanceContractError, match="attempt.*provenance|environment"):
        instance_contract.build_instance_contract(
            asset_id="route2_tall_man_v1",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="tall_man",
            case_kind="attribute_instance",
            output_root=output_root,
            source_lineage=lineage,
        )


def test_attribute_pixal_job_rejects_a_consistent_but_unpinned_wrapper(
    instance, tmp_path
):
    instance_contract, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)
    base_qualified = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )
    source_root = tmp_path / "attribute_wrapper_sources"
    source_root.mkdir()
    lineage = _attribute_lineage(source_root, base_qualified=base_qualified)
    replacement = _write(
        source_root / "untrusted_pixal_wrapper.py",
        b"#!/usr/bin/env python3\n# provenance-complete but not pinned\n",
    )
    job = Path(lineage["pixal_job"]["path"])
    job_payload = json.loads(job.read_text())
    job_payload["wrapper"] = _record(replacement)
    job_payload["executor"]["argv"][0] = str(replacement)
    job.chmod(0o644)
    _write_json(job, job_payload)
    job.chmod(0o444)
    attempt = Path(lineage["pixal_attempt"]["path"])
    attempt_payload = json.loads(attempt.read_text())
    attempt_payload["job"] = _record(job)
    attempt_payload["wrapper"] = _record(replacement)
    attempt_payload["argv"][0] = str(replacement)
    attempt.chmod(0o644)
    _write_json(attempt, attempt_payload)
    attempt.chmod(0o444)
    output_root = tmp_path / "route2_tall_man_v1"
    output_root.mkdir()

    with pytest.raises(instance_contract.InstanceContractError, match="pinned executor"):
        instance_contract.build_instance_contract(
            asset_id="route2_tall_man_v1",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="tall_man",
            case_kind="attribute_instance",
            output_root=output_root,
            source_lineage=lineage,
        )


def test_lineage_json_may_not_claim_user_approved(tmp_path):
    instance_contract, _, _ = _modules()
    source_root = tmp_path / "sources"
    source_root.mkdir()
    lineage, _, _ = _base_lineage(source_root, "rocketbox_male_adult_01")
    review = Path(lineage["source_review"]["path"])
    _write_json(review, {"schema_version": "human_reference_review_v1", "user_approved": True})
    output_root = tmp_path / "rocketbox_male_adult_01"
    output_root.mkdir()

    with pytest.raises(instance_contract.InstanceContractError, match="user_approved"):
        instance_contract.build_instance_contract(
            asset_id="rocketbox_male_adult_01",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="rocketbox_male_adult_01",
            case_kind="base_avatar",
            output_root=output_root,
            source_lineage=lineage,
        )


def test_qualified_candidate_inventories_branch_producer_and_attempt(instance):
    _, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)

    payload = qualified.build_qualified_candidate(instance["contract"], branch_id="direct")

    paths = {record["path"] for record in payload["branch_provenance_inventory"]}
    assert str(instance["output_root"] / "tokenrig_manifest.json") in paths
    assert str(instance["output_root"] / "tokenrig_transfer.glb") in paths
    assert str(instance["output_root"] / "tokenrig_attempt.json") in paths


def test_branch_producer_tamper_stales_qualified_pointer(instance):
    _, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)
    pointer = qualified.publish_qualified_candidate(instance["contract"], branch_id="direct")
    producer = instance["output_root"] / "tokenrig_transfer.glb"
    assert producer.is_file(), "qualification must require the canonical branch producer"
    producer.chmod(0o644)
    producer.write_bytes(b"tampered producer")

    with pytest.raises(qualified.QualificationError, match="branch|producer|snapshot changed"):
        qualified.validate_qualified_candidate(pointer)


def test_qualification_rejects_mutable_recursive_producer_provenance(instance):
    _, _, qualified = _modules()
    from tools import blender_tokenrig_human_static_audit as static_audit

    static_root = _make_static_bundle(instance)
    branch_root = instance["output_root"]
    manifest_path = branch_root / "tokenrig_manifest.json"
    extra = _write(branch_root / "mutable_provenance.json", b"mutable evidence")
    extra.chmod(0o644)
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["nested_producer_provenance"] = {"record": _record(extra)}
    _write_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    qa_path = static_root / "static_qa.json"
    static_root.chmod(0o755)
    qa_path.chmod(0o644)
    qa = json.loads(qa_path.read_text())
    qa["authenticated"] = static_audit.authenticate_task3_inputs(
        asset_id="rocketbox_male_adult_01",
        source_glb=instance["source_glb"],
        tokenrig_glb=branch_root / "tokenrig_transfer.glb",
        tokenrig_manifest=manifest_path,
    )
    _write_json(qa_path, qa)
    qa_path.chmod(0o444)
    static_root.chmod(0o555)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)

    with pytest.raises(
        qualified.QualificationError,
        match="mode 0444|mutable|provenance",
    ):
        qualified.build_qualified_candidate(instance["contract"], branch_id="direct")


def _qualified_base_fixture(
    root: Path,
    instance_contract: Any,
    qualified: Any,
    *,
    asset_id: str,
) -> tuple[dict[str, Any], Path]:
    root.mkdir()
    candidate = _build_base_instance(root, instance_contract, asset_id=asset_id)
    _make_static_bundle(candidate)
    _record_static_pass(candidate)
    _make_dynamic_bundle(candidate)
    pointer = qualified.publish_qualified_candidate(
        candidate["contract"], branch_id="direct"
    )
    return candidate, pointer


def _authorize_female_fixture(
    monkeypatch: pytest.MonkeyPatch,
    female: dict[str, Any],
    male_pointer: Path,
) -> tuple[Any, Path]:
    from tools import tokenrig_human_female_canary as female_owner

    monkeypatch.setattr(female_owner, "_OUTPUT_DIR", female["output_root"])
    monkeypatch.setattr(
        female_owner, "CANONICAL_MALE_QUALIFIED_CANDIDATE", male_pointer
    )
    gate = female_owner.authenticate_male_gate(male_pointer)
    gate_record = female_owner.publish_female_gate_record(gate=gate)
    authorization = female_owner.publish_female_authorization_manifest(
        gate=gate,
        gate_record=gate_record,
        tokenrig_manifest=female["output_root"] / "tokenrig_manifest.json",
    )
    return female_owner, authorization


def test_female_qualification_requires_owner_authorization(tmp_path):
    instance_contract, _, qualified = _modules()
    female_root = tmp_path / "female"
    female_root.mkdir()
    female = _build_base_instance(
        female_root,
        instance_contract,
        asset_id="rocketbox_female_adult_01",
    )
    _make_static_bundle(female)
    _record_static_pass(female)
    _make_dynamic_bundle(female)

    with pytest.raises(
        qualified.QualificationError,
        match="female.*authorization|male.*qualified.*gate",
    ):
        qualified.build_qualified_candidate(female["contract"], branch_id="direct")


def test_female_direct_qualification_accepts_owner_authorized_male_gate(
    tmp_path, monkeypatch
):
    instance_contract, _, qualified = _modules()
    _, male_pointer = _qualified_base_fixture(
        tmp_path / "male",
        instance_contract,
        qualified,
        asset_id="rocketbox_male_adult_01",
    )
    female_root = tmp_path / "female"
    female_root.mkdir()
    female = _build_base_instance(
        female_root,
        instance_contract,
        asset_id="rocketbox_female_adult_01",
    )
    _make_static_bundle(female)
    _, authorization = _authorize_female_fixture(
        monkeypatch, female, male_pointer
    )
    _record_static_pass(female)
    _make_dynamic_bundle(female)

    payload = qualified.build_qualified_candidate(
        female["contract"], branch_id="direct"
    )

    provenance_paths = {
        record["path"] for record in payload["branch_provenance_inventory"]
    }
    assert str(authorization) in provenance_paths
    assert str(male_pointer) in provenance_paths
    assert str(female["output_root"] / "tokenrig_attempt.json") in provenance_paths


def _fixture_static_owner_authentication(
    *, asset_id: str, source_glb: Path, tokenrig_glb: Path, tokenrig_manifest: Path
) -> dict[str, Any]:
    return {
        "fixture_owner": "nested_female_branch",
        "asset_id": asset_id,
        "source_glb_sha256": _sha(source_glb),
        "tokenrig_glb_sha256": _sha(tokenrig_glb),
        "tokenrig_manifest_sha256": _sha(tokenrig_manifest),
    }


def _female_nested_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    branch_id: str,
    include_direct_lineage: bool,
    direct_female_lineage_field: str = "direct_female_lineage",
) -> tuple[Any, dict[str, Any], Path]:
    instance_contract, _, qualified = _modules()
    from tools import blender_tokenrig_human_static_audit as static_owner

    _, male_pointer = _qualified_base_fixture(
        tmp_path / "male",
        instance_contract,
        qualified,
        asset_id="rocketbox_male_adult_01",
    )
    female_root = tmp_path / "female"
    female_root.mkdir()
    female = _build_base_instance(
        female_root,
        instance_contract,
        asset_id="rocketbox_female_adult_01",
    )
    _make_static_bundle(female)
    _, authorization = _authorize_female_fixture(
        monkeypatch, female, male_pointer
    )
    direct_lineage = None
    if include_direct_lineage:
        direct_lineage = {
            "manifest": _record(female["output_root"] / "tokenrig_manifest.json"),
            "attempt": _record(female["output_root"] / "tokenrig_attempt.json"),
            "authorization": _record(authorization),
        }
    real_static_owner = static_owner.authenticate_task3_inputs

    def conditional_static_owner(**kwargs):
        if (
            kwargs["asset_id"] == "rocketbox_female_adult_01"
            and Path(kwargs["tokenrig_manifest"]).parent != female["output_root"]
        ):
            return _fixture_static_owner_authentication(**kwargs)
        return real_static_owner(**kwargs)

    monkeypatch.setattr(
        static_owner,
        "authenticate_task3_inputs",
        conditional_static_owner,
    )
    _make_static_bundle(
        female,
        branch=branch_id,
        direct_female_lineage=direct_lineage,
        direct_female_lineage_field=direct_female_lineage_field,
    )
    _record_static_pass(female, branch=branch_id)
    _make_dynamic_bundle(female, branch=branch_id)
    return qualified, female, authorization


def test_female_nested_qualification_rejects_missing_direct_lineage(
    tmp_path, monkeypatch
):
    qualified, female, _ = _female_nested_fixture(
        tmp_path,
        monkeypatch,
        branch_id="fitted_skeleton",
        include_direct_lineage=False,
    )

    with pytest.raises(
        qualified.QualificationError,
        match="nested female.*direct manifest.*attempt.*authorization",
    ):
        qualified.build_qualified_candidate(
            female["contract"], branch_id="fitted_skeleton"
        )


def test_female_nested_qualification_rejects_direct_records_in_unrelated_field(
    tmp_path, monkeypatch
):
    qualified, female, _ = _female_nested_fixture(
        tmp_path,
        monkeypatch,
        branch_id="fitted_skeleton",
        include_direct_lineage=True,
        direct_female_lineage_field="unrelated_provenance",
    )

    with pytest.raises(
        qualified.QualificationError,
        match="direct_female_lineage|nested female",
    ):
        qualified.build_qualified_candidate(
            female["contract"], branch_id="fitted_skeleton"
        )


@pytest.mark.parametrize("branch_id", ["fitted_skeleton", "sanitized_weights"])
def test_female_nested_qualification_accepts_complete_direct_lineage(
    tmp_path, monkeypatch, branch_id
):
    qualified, female, authorization = _female_nested_fixture(
        tmp_path,
        monkeypatch,
        branch_id=branch_id,
        include_direct_lineage=True,
    )

    payload = qualified.build_qualified_candidate(
        female["contract"], branch_id=branch_id
    )

    assert payload["final_branch"]["branch_id"] == branch_id
    provenance_paths = {
        record["path"] for record in payload["branch_provenance_inventory"]
    }
    assert str(authorization) in provenance_paths
    assert str(female["output_root"] / "tokenrig_manifest.json") in provenance_paths
    assert str(female["output_root"] / "tokenrig_attempt.json") in provenance_paths


def test_female_qualified_pointer_rejects_changed_authorization(
    tmp_path, monkeypatch
):
    instance_contract, _, qualified = _modules()
    _, male_pointer = _qualified_base_fixture(
        tmp_path / "male",
        instance_contract,
        qualified,
        asset_id="rocketbox_male_adult_01",
    )
    female_root = tmp_path / "female"
    female_root.mkdir()
    female = _build_base_instance(
        female_root,
        instance_contract,
        asset_id="rocketbox_female_adult_01",
    )
    _make_static_bundle(female)
    _, authorization = _authorize_female_fixture(
        monkeypatch, female, male_pointer
    )
    _record_static_pass(female)
    _make_dynamic_bundle(female)
    pointer = qualified.publish_qualified_candidate(
        female["contract"], branch_id="direct"
    )
    authorization.chmod(0o644)
    payload = json.loads(authorization.read_text())
    payload["male_gate"]["inventory_sha256"] = "0" * 64
    _write_json(authorization, payload)
    authorization.chmod(0o444)

    with pytest.raises(
        qualified.QualificationError,
        match="snapshot changed|female authorization|male qualified gate",
    ):
        qualified.validate_qualified_candidate(pointer)


def test_accessory_static_decision_requires_rigid_head_binding_evidence(accessory_instance):
    _, static_decision, _ = _modules()
    _make_static_bundle(accessory_instance)

    with pytest.raises(static_decision.StaticDecisionError, match="accessory|Head|head"):
        _record_static_pass(accessory_instance)


def test_accessory_static_decision_accepts_only_rigid_head_weights(accessory_instance):
    _, static_decision, _ = _modules()
    _make_static_bundle(accessory_instance, include_accessory=True)
    decision = _record_static_pass(accessory_instance)
    assert static_decision.validate_static_agent_visual_decision(
        accessory_instance["contract"], branch_id="direct", require_pass=True
    )["status"] == static_decision.PASS_STATUS
    assert decision.stat().st_mode & 0o777 == 0o444


def test_accessory_static_decision_rejects_non_head_weight(accessory_instance):
    _, static_decision, _ = _modules()
    _make_static_bundle(
        accessory_instance,
        include_accessory=True,
        accessory_non_head_weight=0.01,
    )

    with pytest.raises(static_decision.StaticDecisionError, match="Head|head|rigid"):
        _record_static_pass(accessory_instance)


def test_accessory_static_decision_rejects_claimed_non_head_descendant(accessory_instance):
    _, static_decision, _ = _modules()
    _make_static_bundle(
        accessory_instance,
        include_accessory=True,
        accessory_descendant="bone_left_foot",
    )

    with pytest.raises(static_decision.StaticDecisionError, match="descendant|Head|head"):
        _record_static_pass(accessory_instance)


def test_fd_snapshot_rejects_file_swap_to_symlink(tmp_path, monkeypatch):
    from tools import route2_human_contract_common as common

    root = tmp_path / "root"
    root.mkdir()
    target = _write(root / "evidence.bin", b"trusted")
    outside = _write(tmp_path / "outside.bin", b"trusted")
    saved = root / "evidence.saved.bin"
    real_open = common.os.open
    swapped = False

    def racing_open(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        if path == target.name and dir_fd is not None and not swapped:
            swapped = True
            target.rename(saved)
            target.symlink_to(outside)
        return real_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(common.os, "open", racing_open)
    with pytest.raises(OSError):
        common.read_file_snapshot(
            target,
            root=root,
            description="race evidence",
            error_type=OSError,
        )
    assert swapped is True


def test_fd_snapshot_rejects_authenticated_parent_rename(tmp_path, monkeypatch):
    from tools import route2_human_contract_common as common

    root = tmp_path / "root"
    root.mkdir()
    target = _write(root / "evidence.bin", b"trusted")
    moved = tmp_path / "root.original"
    real_open = common.os.open
    swapped = False

    def racing_open(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        if path == target.name and dir_fd is not None and not swapped:
            root.rename(moved)
            root.mkdir()
            _write(root / target.name, b"trusted")
            swapped = True
        return real_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(common.os, "open", racing_open)
    with pytest.raises(OSError, match="changed|directory"):
        common.read_file_snapshot(
            target,
            root=root,
            description="race evidence",
            error_type=OSError,
        )
    assert swapped is True


def test_authenticated_json_is_parsed_from_the_hashed_fd(tmp_path, monkeypatch):
    from tools import route2_human_contract_common as common

    root = tmp_path / "root"
    root.mkdir()
    path = _write_json(root / "value.json", {"value": 1})
    expected_sha = _sha(path)

    def forbidden_path_open(*_args, **_kwargs):
        raise AssertionError("authenticated JSON must not reopen through pathlib")

    monkeypatch.setattr(Path, "open", forbidden_path_open)
    payload, record = common.load_json_mapping_record(
        path,
        root=root,
        description="value",
        error_type=RuntimeError,
    )
    assert payload == {"value": 1}
    assert record["sha256"] == expected_sha


def test_stable_mapping_snapshot_requires_two_identical_reads():
    from tools import route2_human_contract_common as common

    values = iter([{"generation": 1}, {"generation": 2}, {"generation": 2}, {"generation": 2}])
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        return next(values)

    assert common.stable_mapping_snapshot(
        reader, RuntimeError, "fixture snapshot", maximum_attempts=2
    ) == {"generation": 2}
    assert calls == 4


def test_source_lineage_mode_is_snapshotted_and_revalidated(instance):
    instance_contract, _, _ = _modules()
    payload = instance_contract.validate_instance_contract(instance["contract"])
    assert payload["source_lineage"]["pixal_pbr_glb"]["mode"] == format(
        instance["source_glb"].stat().st_mode & 0o777, "04o"
    )
    instance["source_glb"].chmod(0o444)

    with pytest.raises(instance_contract.InstanceContractError, match="mode"):
        instance_contract.validate_instance_contract(instance["contract"])


def test_noreplace_publication_rejects_parent_directory_swap(tmp_path, monkeypatch):
    from tools import route2_human_contract_common as common

    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "parent.original"
    destination = parent / "record.json"
    real_link = common.os.link
    swapped = False

    def racing_link(src, dst, *args, **kwargs):
        nonlocal swapped
        parent.rename(moved)
        parent.mkdir()
        swapped = True
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(common.os, "link", racing_link)
    with pytest.raises(RuntimeError, match="parent changed"):
        common.write_json_immutable_noreplace(
            destination,
            {"schema": "race_probe_v1"},
            RuntimeError,
            "race probe",
        )
    assert swapped is True
    assert not destination.exists()
    assert not (moved / destination.name).exists()


def test_qualification_revalidates_snapshot_immediately_before_link(instance, monkeypatch):
    _, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    review_root = _make_dynamic_bundle(instance)
    media = review_root / "walking_feet.mp4"
    real_write = qualified.common.write_json_immutable_noreplace
    changed = False

    def mutate_then_write(destination, value, error_type, description, **kwargs):
        nonlocal changed
        media.chmod(0o644)
        media.write_bytes(b"changed between build and link")
        media.chmod(0o444)
        changed = True
        return real_write(destination, value, error_type, description, **kwargs)

    monkeypatch.setattr(
        qualified.common, "write_json_immutable_noreplace", mutate_then_write
    )
    with pytest.raises(qualified.QualificationError, match="changed|pre-publication|snapshot"):
        qualified.publish_qualified_candidate(instance["contract"], branch_id="direct")
    assert changed is True
    assert not (instance["output_root"] / "qualified_candidate_v1.json").exists()


def test_attribute_contract_rejects_spliced_qualified_base_branch(
    instance, tmp_path
):
    instance_contract, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)
    qualified_a = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )

    other_root = tmp_path / "other_base"
    other_root.mkdir()
    other = _build_base_instance(other_root, instance_contract)
    _make_static_bundle(other)
    _record_static_pass(other)
    _make_dynamic_bundle(other)
    qualified_b = qualified.publish_qualified_candidate(
        other["contract"], branch_id="direct"
    )

    source_root = tmp_path / "spliced_attribute"
    source_root.mkdir()
    lineage = _attribute_lineage(source_root, base_qualified=qualified_a)
    lineage["base_qualified_candidate"] = {
        "role": "base_qualified_candidate",
        "path": qualified_b,
        "root": qualified_b.parent,
    }
    output_root = tmp_path / "route2_tall_man_v1"
    output_root.mkdir()

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="base.*qualification|qualified.*candidate|candidate.*qualified|branch|authenticated lineage",
    ):
        instance_contract.build_instance_contract(
            asset_id="route2_tall_man_v1",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="tall_man",
            case_kind="attribute_instance",
            output_root=output_root,
            source_lineage=lineage,
        )


@pytest.mark.parametrize(
    "failure_kind",
    ["inventory", "runtime", "gpu", "executable_record", "execution_guard"],
)
def test_attribute_contract_rejects_mismatched_inventory_and_runtime(
    instance, tmp_path, failure_kind
):
    instance_contract, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)
    base_qualified = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )
    source_root = tmp_path / "mismatched_pixal_provenance"
    source_root.mkdir()
    lineage = _attribute_lineage(source_root, base_qualified=base_qualified)
    attempt_path = Path(lineage["pixal_attempt"]["path"])
    attempt_path.chmod(0o644)
    attempt = json.loads(attempt_path.read_text())
    if failure_kind == "inventory":
        attempt["model_inventory"]["pixal_snapshot_inventory_sha256"] = "0" * 64
        attempt["model_inventory"]["dino_snapshot_inventory_sha256"] = "f" * 64
    elif failure_kind == "runtime":
        attempt["environment"].update(
            {
                "python_executable": "/definitely/not/a/python",
                "python_version": "not-python",
                "torch_version": "not-torch",
                "cuda_version": "not-cuda",
            }
        )
    elif failure_kind == "gpu":
        attempt["environment"]["cuda_device_name"] = "unrelated device"
    elif failure_kind == "executable_record":
        attempt["environment"]["python_executable_record"]["sha256"] = "0" * 64
    else:
        attempt["execution_guard"]["after"]["guard_sha256"] = "0" * 64
    _write_json(attempt_path, attempt)
    attempt_path.chmod(0o444)
    output_root = tmp_path / "route2_tall_man_v1"
    output_root.mkdir()

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="inventory|Python|runtime|executable|Torch|CUDA|guard|execution-critical",
    ):
        instance_contract.build_instance_contract(
            asset_id="route2_tall_man_v1",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="tall_man",
            case_kind="attribute_instance",
            output_root=output_root,
            source_lineage=lineage,
        )


def test_attribute_contract_rejects_success_from_unauthorized_scaffold(
    instance, tmp_path
):
    instance_contract, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)
    base_qualified = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )
    source_root = tmp_path / "unauthorized_pixal_scaffold"
    source_root.mkdir()
    lineage = _attribute_lineage(source_root, base_qualified=base_qualified)
    job_path = Path(lineage["pixal_job"]["path"])
    attempt_path = Path(lineage["pixal_attempt"]["path"])
    job_path.chmod(0o644)
    attempt_path.chmod(0o644)
    job = json.loads(job_path.read_text())
    job["executor"] = {
        "kind": "python_cli_scaffold",
        "argv": job["executor"]["argv"],
        "execution_authorized": False,
    }
    _write_json(job_path, job)
    job_path.chmod(0o444)
    attempt = json.loads(attempt_path.read_text())
    attempt["job"] = _record(job_path)
    _write_json(attempt_path, attempt)
    attempt_path.chmod(0o444)
    output_root = tmp_path / "route2_tall_man_v1"
    output_root.mkdir()

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="authorized|atomic.*executor|scaffold|executor fields",
    ):
        instance_contract.build_instance_contract(
            asset_id="route2_tall_man_v1",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="tall_man",
            case_kind="attribute_instance",
            output_root=output_root,
            source_lineage=lineage,
        )


@pytest.mark.parametrize(
    "mutation", ["job_executor", "attempt_executor", "start_ledger", "execution_log"]
)
def test_attribute_contract_crosschecks_executor_start_and_log(
    instance, tmp_path, mutation
):
    instance_contract, _, qualified = _modules()
    _make_static_bundle(instance)
    _record_static_pass(instance)
    _make_dynamic_bundle(instance)
    base_qualified = qualified.publish_qualified_candidate(
        instance["contract"], branch_id="direct"
    )
    source_root = tmp_path / f"pixal_{mutation}"
    source_root.mkdir()
    lineage = _attribute_lineage(source_root, base_qualified=base_qualified)
    job_path = Path(lineage["pixal_job"]["path"])
    attempt_path = Path(lineage["pixal_attempt"]["path"])
    attempt = json.loads(attempt_path.read_text())
    if mutation == "job_executor":
        job_path.chmod(0o644)
        job = json.loads(job_path.read_text())
        job["executor"].update(_record(Path(job["wrapper"]["path"])))
        _write_json(job_path, job)
        job_path.chmod(0o444)
    elif mutation == "attempt_executor":
        attempt["executor"]["sha256"] = "0" * 64
    elif mutation == "start_ledger":
        start_path = Path(attempt["start_ledger"]["path"])
        start_path.chmod(0o644)
        start = json.loads(start_path.read_text())
        start["started_at_utc"] = "2026-07-12T00:00:01Z"
        _write_json(start_path, start)
        start_path.chmod(0o444)
        attempt["start_ledger"] = _record(start_path)
    else:
        log_path = Path(attempt["execution_log"]["path"])
        log_path.chmod(0o644)
        log = json.loads(log_path.read_text())
        log["stdout"] += f"{log['success_sentinel']}\n"
        _write_json(log_path, log)
        log_path.chmod(0o444)
        attempt["execution_log"] = _record(log_path)
    if mutation != "job_executor":
        attempt_path.chmod(0o644)
        _write_json(attempt_path, attempt)
        attempt_path.chmod(0o444)
    output_root = tmp_path / "route2_tall_man_v1"
    output_root.mkdir()

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="executor|start ledger|started|execution log|sentinel",
    ):
        instance_contract.build_instance_contract(
            asset_id="route2_tall_man_v1",
            base_avatar_id="rocketbox_male_adult_01",
            case_id="tall_man",
            case_kind="attribute_instance",
            output_root=output_root,
            source_lineage=lineage,
        )


def test_model_snapshot_rejects_empty_revision_directory(tmp_path):
    instance_contract, _, _ = _modules()
    snapshot = tmp_path / instance_contract.PIXAL3D_REVISION
    snapshot.mkdir()

    with pytest.raises(
        instance_contract.InstanceContractError, match="inventory|empty|snapshot"
    ):
        instance_contract._validate_model_snapshot(
            str(snapshot), instance_contract.PIXAL3D_REVISION, "empty model snapshot"
        )


def test_model_snapshot_rechecks_content_addressed_blob_bytes(tmp_path):
    instance_contract, _, _ = _modules()
    snapshot = instance_contract.MODEL_ROOT / instance_contract.PIXAL3D_SNAPSHOT_RELATIVE
    weight = snapshot / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
    original = weight.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    repository_root = snapshot.parents[1]
    blob = _write(repository_root / "blobs" / digest, original)
    weight.unlink()
    weight.symlink_to(os.path.relpath(blob, weight.parent))
    instance_contract._validate_model_snapshot(
        str(snapshot), instance_contract.PIXAL3D_REVISION, "fixture Pixal snapshot"
    )
    blob.write_bytes(b"x" * len(original))

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="content|SHA|blob|inventory",
    ):
        instance_contract._validate_model_snapshot(
            str(snapshot),
            instance_contract.PIXAL3D_REVISION,
            "changed Pixal snapshot",
        )


def test_model_snapshot_cache_requires_exact_entry_metadata(tmp_path):
    instance_contract, _, _ = _modules()
    snapshot = instance_contract.MODEL_ROOT / instance_contract.PIXAL3D_SNAPSHOT_RELATIVE
    first = instance_contract.model_snapshot_evidence(
        instance_contract.PIXAL3D_REVISION
    )
    second = instance_contract.model_snapshot_evidence(
        instance_contract.PIXAL3D_REVISION
    )
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["inventory_sha256"] == second["inventory_sha256"]

    pipeline = snapshot / "pipeline.json"
    pipeline.chmod(0o444)
    after_mode = instance_contract.model_snapshot_evidence(
        instance_contract.PIXAL3D_REVISION
    )
    assert after_mode["cache_hit"] is False
    assert after_mode["inventory_sha256"] != first["inventory_sha256"]

    extra = _write(snapshot / "extra_provenance.txt", b"extra")
    after_add = instance_contract.model_snapshot_evidence(
        instance_contract.PIXAL3D_REVISION
    )
    assert after_add["cache_hit"] is False
    assert after_add["inventory_sha256"] != after_mode["inventory_sha256"]
    assert instance_contract.model_snapshot_evidence(
        instance_contract.PIXAL3D_REVISION
    )["cache_hit"] is True
    extra.unlink()
    after_remove = instance_contract.model_snapshot_evidence(
        instance_contract.PIXAL3D_REVISION
    )
    assert after_remove["cache_hit"] is False

    weight = snapshot / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
    repository_root = snapshot.parents[1]
    first_blob_bytes = weight.read_bytes()
    first_blob = _write(
        repository_root / "blobs" / hashlib.sha256(first_blob_bytes).hexdigest(),
        first_blob_bytes,
    )
    weight.unlink()
    weight.symlink_to(os.path.relpath(first_blob, weight.parent))
    after_symlink = instance_contract.model_snapshot_evidence(
        instance_contract.PIXAL3D_REVISION
    )
    assert after_symlink["cache_hit"] is False
    second_blob_bytes = b"replacement Pixal weights"
    second_blob = _write(
        repository_root / "blobs" / hashlib.sha256(second_blob_bytes).hexdigest(),
        second_blob_bytes,
    )
    weight.unlink()
    weight.symlink_to(os.path.relpath(second_blob, weight.parent))
    after_retarget = instance_contract.model_snapshot_evidence(
        instance_contract.PIXAL3D_REVISION
    )
    assert after_retarget["cache_hit"] is False
    assert after_retarget["inventory_sha256"] != after_symlink["inventory_sha256"]


def test_pixal_execution_guard_detects_transient_swap_restore(
    tmp_path,
    monkeypatch,
):
    from tools import human_attribute_pixal_contract as pixal_contract
    from tools import route2_human_instance_contract as instance_contract

    guard_root = tmp_path / "execution_guard"
    guard_root.mkdir()
    wrapper = _write(guard_root / "wrapper.py", b"print('pinned wrapper')\n")
    executor = _write(guard_root / "executor.py", b"print('pinned executor')\n")
    monkeypatch.setattr(pixal_contract, "PIXAL_WRAPPER_PATH", wrapper)
    monkeypatch.setattr(pixal_contract, "EXECUTOR_PATH", executor)

    before = instance_contract.pixal_execution_guard_evidence()
    original_wrapper_sha = _sha(wrapper)
    saved = guard_root / "wrapper.saved.py"
    wrapper.rename(saved)
    _write(wrapper, b"print('transient untrusted wrapper')\n")
    wrapper.unlink()
    saved.rename(wrapper)
    assert _sha(wrapper) == original_wrapper_sha

    after = instance_contract.pixal_execution_guard_evidence()
    assert before["files"]["wrapper"]["sha256"] == after["files"]["wrapper"]["sha256"]
    assert before["guard_sha256"] != after["guard_sha256"]

    model_before = after
    model_file = (
        instance_contract.MODEL_ROOT
        / instance_contract.PIXAL3D_SNAPSHOT_RELATIVE
        / "pipeline.json"
    )
    original_model_sha = _sha(model_file)
    saved_model = model_file.with_name("pipeline.saved.json")
    model_file.rename(saved_model)
    _write(model_file, b'{"transient":"untrusted"}\n')
    model_file.unlink()
    saved_model.rename(model_file)
    assert _sha(model_file) == original_model_sha

    model_after = instance_contract.pixal_execution_guard_evidence()
    assert (
        model_before["models"]["pixal"]["metadata_sha256"]
        != model_after["models"]["pixal"]["metadata_sha256"]
    )


def test_static_decision_crosschecks_claimed_bones_with_bind_glb(instance):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(instance)
    bind_path = static_root / "bind_pose.glb"
    qa_path = static_root / "static_qa.json"
    static_root.chmod(0o755)
    bind_path.chmod(0o644)
    qa_path.chmod(0o644)
    _write_glb(bind_path, joint_count=1)
    bind_path.chmod(0o444)
    qa = json.loads(qa_path.read_text())
    qa["artifacts"][bind_path.name] = {
        "filename": bind_path.name,
        "sha256": _sha(bind_path),
        "size_bytes": bind_path.stat().st_size,
    }
    _write_json(qa_path, qa)
    qa_path.chmod(0o444)
    static_root.chmod(0o555)

    with pytest.raises(
        static_decision.StaticDecisionError,
        match="joint|bone|hierarchy|semantic",
    ):
        static_decision.validated_static_snapshot(
            instance["contract"], branch_id="direct"
        )


def test_accessory_static_decision_rejects_nonfinite_or_boolean_weight_proof(
    accessory_instance
):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(accessory_instance, include_accessory=True)
    binding_path = static_root / "accessory_head_binding.json"
    qa_path = static_root / "static_qa.json"
    static_root.chmod(0o755)
    binding_path.chmod(0o644)
    qa_path.chmod(0o644)
    binding = json.loads(binding_path.read_text())
    binding["minimum_head_descendant_weight"] = "NaN"
    binding["maximum_non_head_descendant_weight"] = "NaN"
    binding["accessory_vertex_count"] = True
    binding["pixel_to_vertex_mapping"]["reviewed_pixel_count"] = True
    binding["pixel_to_vertex_mapping"]["mapped_vertex_count"] = True
    binding["pixel_to_vertex_mapping"]["coverage"] = True
    _write_json(binding_path, binding)
    binding_path.chmod(0o444)
    qa = json.loads(qa_path.read_text())
    qa["artifacts"][binding_path.name] = {
        "filename": binding_path.name,
        "sha256": _sha(binding_path),
        "size_bytes": binding_path.stat().st_size,
    }
    _write_json(qa_path, qa)
    qa_path.chmod(0o444)
    static_root.chmod(0o555)

    with pytest.raises(
        static_decision.StaticDecisionError,
        match="finite|numeric|integer|weight|coverage",
    ):
        _record_static_pass(accessory_instance)


def test_accessory_static_decision_requires_per_vertex_weight_evidence(
    accessory_instance
):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(accessory_instance, include_accessory=True)
    binding_path = static_root / "accessory_head_binding.json"
    qa_path = static_root / "static_qa.json"
    static_root.chmod(0o755)
    binding_path.chmod(0o644)
    qa_path.chmod(0o644)
    binding = json.loads(binding_path.read_text())
    binding.pop("vertex_weights")
    _write_json(binding_path, binding)
    binding_path.chmod(0o444)
    qa = json.loads(qa_path.read_text())
    qa["artifacts"][binding_path.name] = {
        "filename": binding_path.name,
        "sha256": _sha(binding_path),
        "size_bytes": binding_path.stat().st_size,
    }
    _write_json(qa_path, qa)
    qa_path.chmod(0o444)
    static_root.chmod(0o555)

    with pytest.raises(
        static_decision.StaticDecisionError,
        match="vertex.*weight|Head.*binding|accessory",
    ):
        _record_static_pass(accessory_instance)


def test_accessory_static_decision_crosschecks_glb_vertex_weights(
    accessory_instance
):
    _, static_decision, _ = _modules()
    static_root = _make_static_bundle(accessory_instance, include_accessory=True)
    bind_path = static_root / "bind_pose.glb"
    qa_path = static_root / "static_qa.json"
    static_root.chmod(0o755)
    bind_path.chmod(0o644)
    qa_path.chmod(0o644)
    _write_glb(bind_path, vertex_joint_name="bone_left_foot")
    bind_path.chmod(0o444)
    qa = json.loads(qa_path.read_text())
    qa["artifacts"][bind_path.name] = {
        "filename": bind_path.name,
        "sha256": _sha(bind_path),
        "size_bytes": bind_path.stat().st_size,
    }
    _write_json(qa_path, qa)
    qa_path.chmod(0o444)
    static_root.chmod(0o555)

    with pytest.raises(
        static_decision.StaticDecisionError,
        match="GLB.*weight|vertex.*weight|Head",
    ):
        _record_static_pass(accessory_instance)


def test_fd_snapshot_rejects_swap_after_pathname_inode_check(tmp_path, monkeypatch):
    from tools import route2_human_contract_common as common

    root = tmp_path / "late_swap"
    root.mkdir()
    target = _write(root / "evidence.bin", b"trusted")
    saved = root / "evidence.saved.bin"
    real_open_chain = common._open_directory_chain
    calls = 0

    def racing_open_chain(path, description, error_type):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.rename(saved)
            target.write_bytes(b"replacement")
        return real_open_chain(path, description, error_type)

    monkeypatch.setattr(common, "_open_directory_chain", racing_open_chain)
    with pytest.raises(RuntimeError, match="changed|pathname"):
        common.read_file_snapshot(
            target,
            root=root,
            description="late-swap evidence",
            error_type=RuntimeError,
        )


def test_instance_publication_rejects_upstream_swap_at_link(tmp_path, monkeypatch):
    from tools import route2_human_contract_common as common

    instance_contract, _, _ = _modules()
    source_root = tmp_path / "publication_sources"
    source_root.mkdir()
    lineage, _, _ = _base_lineage(source_root, "rocketbox_male_adult_01")
    output_root = tmp_path / "publication" / "rocketbox_male_adult_01"
    output_root.mkdir(parents=True)
    payload = instance_contract.build_instance_contract(
        asset_id="rocketbox_male_adult_01",
        base_avatar_id="rocketbox_male_adult_01",
        case_id="rocketbox_male_adult_01",
        case_kind="base_avatar",
        output_root=output_root,
        source_lineage=lineage,
    )
    victim = Path(lineage["flux_candidate"]["path"])
    real_link = common.os.link
    swapped = False

    def racing_link(source, destination, *args, **kwargs):
        nonlocal swapped
        victim.write_bytes(b"changed after final pre-link validation")
        swapped = True
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(common.os, "link", racing_link)
    with pytest.raises(
        instance_contract.InstanceContractError, match="changed|publication|lineage"
    ):
        instance_contract.publish_instance_contract(payload)
    assert swapped is True
    assert not (output_root / instance_contract.FILENAME).exists()


def test_noreplace_publication_removes_link_before_reraising_interrupt(tmp_path):
    from tools import route2_human_contract_common as common

    destination = tmp_path / "interrupted.json"

    def interrupt_after_link() -> None:
        raise KeyboardInterrupt("fixture interruption after link")

    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        common.write_json_immutable_noreplace(
            destination,
            {"schema": "interrupted_publication_fixture_v1"},
            RuntimeError,
            "interrupted publication fixture",
            postlink_validator=interrupt_after_link,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".interrupted.json.*.staging"))


def _failure_bundle_fixture(
    tmp_path: Path,
    *,
    job_mutation: str | None = None,
    spliced_staging: bool = False,
) -> Path:
    from tools import human_attribute_pixal_contract
    from tools import route2_human_instance_contract as route2_contract
    from tools.spike_rlr import human_attribute_review

    asset_id = "route2_tall_man_v1"
    attempt_id = "attempt_failure_001"
    executor = Path(human_attribute_pixal_contract.__file__).resolve()
    executor_record = _record(executor)
    wrapper = Path(human_attribute_pixal_contract.PIXAL_WRAPPER_PATH)
    candidate_root = tmp_path / "attribute_bundle"
    candidate_root.mkdir()
    candidate_manifest = _write_json(
        candidate_root / "candidate_manifest.json",
        {"schema": "flux2_human_attribute_candidate_v2"},
    )
    decision = _write_json(
        human_attribute_review.decision_path(candidate_root),
        {"status": human_attribute_review.PASS_STATUS},
    )
    rgba = _write_review_png(candidate_root / "candidate_rgba.png")
    output_glb = tmp_path / asset_id / "canary_1024_seed42.glb"
    argv = [
        str(wrapper),
        "--backend",
        "pixal3d",
        "--image",
        str(rgba),
        "--output",
        str(output_glb),
        "--gpu",
        "3",
        "--seed",
        "42",
        "--resolution",
        "1024",
        "--manual-fov",
        "0.2",
        "--low-vram",
    ]
    job_payload = {
        "schema": "pixal3d_human_attribute_job_v1",
        "case_id": "tall_man",
        "asset_id": asset_id,
        "base_asset_id": "rocketbox_male_adult_01",
        "state_classification": "research_candidate",
        "input_rgba": {
            **_record(rgba),
            "mode": "RGBA",
            "size": [2, 2],
            "alpha_min": 0,
            "alpha_max": 255,
        },
        "candidate_manifest": _record(candidate_manifest),
        "agent_2d_decision": {
            **_record(decision),
            "status": human_attribute_review.PASS_STATUS,
        },
        "model_revision": "0b31f9160aa400719af409098bff7936a932f726",
        "dino_revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881",
        "parameters": {
            "seed": 42,
            "manual_fov": 0.2,
            "resolution": 1024,
            "low_vram": True,
        },
        "wrapper": _record(wrapper),
        "output_glb": str(output_glb),
        "output_manifest": str(output_glb.with_suffix(".manifest.json")),
        "output_policy": "atomic_no_replace",
        "executor": {
            "kind": "atomic_pixal3d_executor_v1",
            "argv": argv,
            "execution_authorized": True,
            "atomic_no_replace": True,
            **executor_record,
        },
    }
    if job_mutation == "parameters":
        job_payload["parameters"]["resolution"] = 512
    elif job_mutation == "output_path":
        job_payload["output_glb"] = str(tmp_path / "spliced.glb")
    elif job_mutation == "wrapper":
        fake_wrapper = _write(tmp_path / "fake_wrapper.py", b"print('fake')\n")
        job_payload["wrapper"] = _record(fake_wrapper)
    elif job_mutation == "argv":
        job_payload["executor"]["argv"] = [
            "0" if value == "3" else value for value in argv
        ]
        argv = list(job_payload["executor"]["argv"])
    elif job_mutation == "input_path":
        job_payload["input_rgba"]["path"] = str(candidate_manifest)
    elif job_mutation is not None:  # pragma: no cover - test helper contract.
        raise AssertionError(job_mutation)
    job_path = _write_json(
        tmp_path / f"{asset_id}.pixal_job.json",
        job_payload,
    )
    job_path.chmod(0o444)
    staging_path = (
        tmp_path / "spliced.staging"
        if spliced_staging
        else tmp_path / f".{asset_id}.{attempt_id}.fixture.staging"
    )
    execution_guard_before = route2_contract.pixal_execution_guard_evidence()
    start_path = _write_json(
        tmp_path / ".attempts" / asset_id / f"{attempt_id}.started.json",
        {
            "schema": "pixal3d_human_attribute_attempt_start_v1",
            "attempt_id": attempt_id,
            "status": "started",
            "case_id": "tall_man",
            "asset_id": asset_id,
            "base_avatar_id": "rocketbox_male_adult_01",
            "job": _record(job_path),
            "executor": executor_record,
            "execution_guard_before": execution_guard_before,
            "argv": argv,
            "started_at_utc": "2026-07-12T00:00:00Z",
            "staging": {"path": str(staging_path), "created": True},
            "publication_policy": "atomic_no_replace",
        },
    )
    start_path.chmod(0o444)
    bundle = tmp_path / ".failed_attempts" / asset_id / attempt_id
    bundle.mkdir(parents=True)
    partial = _write(bundle / "partial.stdout", b"partial Pixal evidence\n")
    partial.chmod(0o444)
    failure_manifest = _write_json(
        bundle / "failure_manifest.json",
        {
            "schema": "pixal3d_human_attribute_failure_bundle_v1",
            "attempt_id": attempt_id,
            "status": "failed",
            "case_id": "tall_man",
            "asset_id": asset_id,
            "base_avatar_id": "rocketbox_male_adult_01",
            "job": _record(job_path),
            "start_ledger": _record(start_path),
            "failure_stage": "subprocess",
            "error": {"type": "RuntimeError", "message": "fixture failure"},
            "returncode": 1,
            "artifacts": [
                {
                    "relative_path": partial.relative_to(bundle).as_posix(),
                    "sha256": _sha(partial),
                    "size_bytes": partial.stat().st_size,
                    "mode": "0444",
                }
            ],
        },
    )
    failure_manifest.chmod(0o444)
    bundle.chmod(0o555)
    return bundle


def test_pixal_failure_bundle_requires_readonly_exact_inventory(tmp_path):
    instance_contract, _, _ = _modules()
    bundle = _failure_bundle_fixture(tmp_path)

    validated = instance_contract.validate_pixal_attribute_failure_bundle(bundle)
    assert validated["status"] == "failed"
    partial = bundle / "partial.stdout"
    partial.chmod(0o644)

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="failure bundle|mode 0444|inventory",
    ):
        instance_contract.validate_pixal_attribute_failure_bundle(bundle)
    partial.write_bytes(b"changed preserved evidence\n")
    partial.chmod(0o444)

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="failure bundle|inventory",
    ):
        instance_contract.validate_pixal_attribute_failure_bundle(bundle)


@pytest.mark.parametrize(
    "job_mutation",
    ["parameters", "output_path", "wrapper", "argv", "input_path"],
)
def test_pixal_failure_bundle_rejects_spliced_job_semantics(
    tmp_path,
    job_mutation,
):
    instance_contract, _, _ = _modules()
    bundle = _failure_bundle_fixture(tmp_path, job_mutation=job_mutation)

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="Pixal|job|wrapper|argv|input|output|parameter",
    ):
        instance_contract.validate_pixal_attribute_failure_bundle(bundle)


def test_pixal_failure_bundle_rejects_start_ledger_with_spliced_staging_path(
    tmp_path,
):
    instance_contract, _, _ = _modules()
    bundle = _failure_bundle_fixture(tmp_path, spliced_staging=True)

    with pytest.raises(
        instance_contract.InstanceContractError,
        match="staging|attempt",
    ):
        instance_contract.validate_pixal_attribute_failure_bundle(bundle)
