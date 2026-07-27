#!/usr/bin/env python3
"""Finalize one approved generated static object in the AVEngine local frame.

The operation is rigid/data-driven except for one uniform physical scale:
reviewed source yaw is mapped to +X forward, target height comes from the
authenticated profile request, and the mesh minimum is grounded at zero.
No object class, category, or product name changes the implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import bpy
from mathutils import Matrix, Vector


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import generated_asset_emitter_contract as contract  # noqa: E402


WATERTIGHT_SCHEMA = "avengine_watertight_textured_runtime_proxy_v1"
STATIC_DECISION_SCHEMA = "avengine_controlled_static_object_decision_v1"
HEADING_EVIDENCE_SCHEMA = "avengine_static_heading_review_v1"
GROUND_TOLERANCE_M = 1.0e-5


def parse_argv() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--watertight-manifest", type=Path, required=True)
    parser.add_argument("--static-decision", type=Path, required=True)
    parser.add_argument("--heading-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def require_new_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise contract.EmitterContractError(f"refusing to replace {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_watertight_manifest(
    path: Path,
    input_glb: Path,
) -> dict[str, Any]:
    payload = contract.load_json_object(path, "watertight manifest")
    if (
        payload.get("schema") != WATERTIGHT_SCHEMA
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("status")
        != "research_candidate_pending_static_and_animation_qa"
    ):
        raise contract.EmitterContractError("watertight manifest contract is invalid")
    contract.validate_file_record(
        payload.get("output"),
        label="watertight output",
        expected_path=input_glb,
    )
    contract.validate_file_record(
        payload.get("input"),
        label="raw Pixal geometry input",
    )
    topology = payload.get("topology", {}).get("final")
    if (
        not isinstance(topology, Mapping)
        or any(
            topology.get(field) != 0
            for field in (
                "boundary_edges",
                "wire_edges",
                "nonmanifold_edges_over_two_faces",
            )
        )
    ):
        raise contract.EmitterContractError("watertight topology gate is not passed")
    authority = payload.get("authority_contract")
    if (
        not isinstance(authority, Mapping)
        or authority.get("approved_skeleton_or_animation_touched") is not False
    ):
        raise contract.EmitterContractError("watertight authority contract changed")
    return payload


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return contract.json_sha256(
        {name: item for name, item in value.items() if name != key}
    )


def validate_static_decision(
    path: Path,
    watertight: Mapping[str, Any],
) -> dict[str, Any]:
    payload = contract.load_json_object(path, "static-object decision")
    required = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "target_physical_profile",
        "pixal_output",
        "decision",
        "next_gate",
        "decision_sha256",
        "formal_dataset_registration_authorized",
    }
    if not required.issubset(payload):
        raise contract.EmitterContractError("static-object decision fields are invalid")
    if (
        payload["schema"] != STATIC_DECISION_SCHEMA
        or payload["decision"] != "approved_for_watertight_finalization"
        or payload["next_gate"] != "watertight_then_static_finalization"
        or payload["formal_dataset_registration_authorized"] is not False
        or payload["decision_sha256"] != _hash_without(payload, "decision_sha256")
    ):
        raise contract.EmitterContractError(
            "static-object decision is not approved/authenticated"
        )
    contract.require_identifier(payload["instance_id"], "instance_id")
    contract.require_sha256(payload["request_sha256"], "request_sha256")
    contract.require_sha256(payload["profile_sha256"], "profile_sha256")
    pixal = contract.validate_file_record(
        payload["pixal_output"],
        label="approved raw Pixal GLB",
    )
    raw = Path(watertight["input"]["path"]).resolve()
    if (
        pixal != raw
        or payload["pixal_output"]["sha256"] != watertight["input"]["sha256"]
        or payload["pixal_output"]["size_bytes"] != watertight["input"]["size_bytes"]
    ):
        raise contract.EmitterContractError(
            "static decision/Pixal/watertight lineage changed"
        )
    physical = payload["target_physical_profile"]
    if (
        not isinstance(physical, Mapping)
        or physical.get("control_attribute") is not None
        or physical.get("measurement") != "height_cm"
    ):
        raise contract.EmitterContractError(
            "static physical profile must declare an absolute height"
        )
    target = contract.require_finite_vector(
        [physical.get("target_value_cm")], 1, "target height"
    )[0]
    tolerance = contract.require_finite_vector(
        [physical.get("tolerance_cm")], 1, "height tolerance"
    )[0]
    if not 0.1 <= target <= 1000.0 or not 0.0 < tolerance <= target:
        raise contract.EmitterContractError("static physical height range is invalid")
    return payload


def validate_heading_evidence(
    path: Path,
    *,
    decision: Mapping[str, Any],
    input_glb: Path,
) -> dict[str, Any]:
    payload = contract.load_json_object(path, "static heading review evidence")
    required = {
        "schema",
        "instance_id",
        "request_sha256",
        "profile_sha256",
        "input_glb_sha256",
        "review_artifact",
        "reviewed_source_front_yaw_deg",
        "target_front_axis",
        "decision",
        "formal_dataset_registration_authorized",
    }
    if set(payload) != required:
        raise contract.EmitterContractError("static heading evidence fields are invalid")
    if (
        payload["schema"] != HEADING_EVIDENCE_SCHEMA
        or payload["target_front_axis"] != "positive-x"
        or payload["decision"] != "approved_for_positive_x_normalization"
        or payload["formal_dataset_registration_authorized"] is not False
    ):
        raise contract.EmitterContractError("static heading evidence is not approved")
    for field in ("instance_id", "request_sha256", "profile_sha256"):
        if payload[field] != decision[field]:
            raise contract.EmitterContractError(f"static heading {field} changed")
    if contract.require_sha256(
        payload["input_glb_sha256"], "heading input GLB hash"
    ) != contract.sha256_file(input_glb):
        raise contract.EmitterContractError("static heading input GLB hash changed")
    contract.validate_file_record(
        payload["review_artifact"],
        label="static heading review artifact",
    )
    yaw = contract.require_finite_vector(
        [payload["reviewed_source_front_yaw_deg"]],
        1,
        "reviewed source front yaw",
    )[0]
    if not -180.0 <= yaw <= 180.0:
        raise contract.EmitterContractError(
            "reviewed source front yaw must be in [-180, 180]"
        )
    return payload


def real_meshes() -> list[Any]:
    meshes = sorted(
        (item for item in bpy.context.scene.objects if item.type == "MESH"),
        key=lambda item: item.name,
    )
    if len(meshes) != 1:
        raise contract.EmitterContractError(
            f"expected one watertight static mesh, got {[item.name for item in meshes]}"
        )
    mesh = meshes[0]
    if (
        mesh.vertex_groups
        or any(modifier.type == "ARMATURE" for modifier in mesh.modifiers)
        or any(item.type == "ARMATURE" for item in bpy.context.scene.objects)
        or bpy.data.actions
    ):
        raise contract.EmitterContractError(
            "static finalization input contains rig or animation data"
        )
    return meshes


def scene_bounds(meshes: list[Any]) -> tuple[Vector, Vector]:
    corners = [
        mesh.matrix_world @ Vector(corner)
        for mesh in meshes
        for corner in mesh.bound_box
    ]
    minimum = Vector(
        tuple(min(point[axis] for point in corners) for axis in range(3))
    )
    maximum = Vector(
        tuple(max(point[axis] for point in corners) for axis in range(3))
    )
    extent = maximum - minimum
    if any(not math.isfinite(value) or value <= 0.0 for value in extent):
        raise contract.EmitterContractError("static finalization bounds are degenerate")
    return minimum, maximum


def scene_summary(meshes: list[Any]) -> dict[str, Any]:
    return {
        "mesh_count": len(meshes),
        "skin_count": sum(
            any(modifier.type == "ARMATURE" for modifier in mesh.modifiers)
            for mesh in meshes
        ),
        "armature_count": sum(
            item.type == "ARMATURE" for item in bpy.context.scene.objects
        ),
        "animation_count": len(bpy.data.actions),
        "vertex_count": sum(len(mesh.data.vertices) for mesh in meshes),
        "face_count": sum(len(mesh.data.polygons) for mesh in meshes),
        "material_count": len(bpy.data.materials),
        "image_count": len(bpy.data.images),
    }


def transform_roots(matrix: Matrix) -> None:
    roots = [item for item in bpy.context.scene.objects if item.parent is None]
    if not roots:
        raise contract.EmitterContractError("static scene has no root objects")
    for root in roots:
        root.matrix_world = matrix @ root.matrix_world
    bpy.context.view_layer.update()


def export_glb(output: Path) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        use_selection=True,
        export_animations=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_all_vertex_colors=True,
        export_vertex_color="ACTIVE",
    )


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    args = parse_argv()
    input_glb = args.input_glb.resolve()
    watertight_path = args.watertight_manifest.resolve()
    decision_path = args.static_decision.resolve()
    heading_path = args.heading_evidence.resolve()
    output = require_new_file(args.output, "finalized static GLB")
    manifest_path = require_new_file(args.manifest, "static finalization manifest")

    watertight = validate_watertight_manifest(watertight_path, input_glb)
    decision = validate_static_decision(decision_path, watertight)
    heading = validate_heading_evidence(
        heading_path,
        decision=decision,
        input_glb=input_glb,
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    meshes = real_meshes()
    before_summary = scene_summary(meshes)
    source_yaw = float(heading["reviewed_source_front_yaw_deg"])
    delta_yaw = -source_yaw
    transform_roots(Matrix.Rotation(math.radians(delta_yaw), 4, "Z"))
    rotated_minimum, rotated_maximum = scene_bounds(meshes)
    height_before = float(rotated_maximum.z - rotated_minimum.z)
    physical = decision["target_physical_profile"]
    target_height_m = float(physical["target_value_cm"]) / 100.0
    tolerance_m = float(physical["tolerance_cm"]) / 100.0
    uniform_scale = target_height_m / height_before
    if not math.isfinite(uniform_scale) or not 1.0e-4 <= uniform_scale <= 1.0e4:
        raise contract.EmitterContractError("static uniform scale is unsafe")
    transform_roots(Matrix.Scale(uniform_scale, 4))
    scaled_minimum, _scaled_maximum = scene_bounds(meshes)
    transform_roots(Matrix.Translation((0.0, 0.0, -float(scaled_minimum.z))))
    grounded_minimum, grounded_maximum = scene_bounds(meshes)
    if abs(float(grounded_minimum.z)) > GROUND_TOLERANCE_M:
        raise contract.EmitterContractError("static grounding gate failed before export")
    export_glb(output)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(output))
    readback_meshes = real_meshes()
    after_summary = scene_summary(readback_meshes)
    readback_minimum, readback_maximum = scene_bounds(readback_meshes)
    readback_height = float(readback_maximum.z - readback_minimum.z)
    height_error = abs(readback_height - target_height_m)
    ground_error = abs(float(readback_minimum.z))
    if height_error > tolerance_m:
        raise contract.EmitterContractError(
            "static physical-height readback exceeded profile tolerance"
        )
    if ground_error > GROUND_TOLERANCE_M:
        raise contract.EmitterContractError(
            "static grounding readback exceeded tolerance"
        )
    for field in (
        "mesh_count",
        "skin_count",
        "armature_count",
        "animation_count",
        "vertex_count",
        "face_count",
        "material_count",
        "image_count",
    ):
        if after_summary[field] != before_summary[field]:
            raise contract.EmitterContractError(
                f"static finalization changed protected scene field: {field}"
            )

    payload = {
        "schema": contract.STATIC_FINALIZATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_final_scaled_grounded_canonical_glb",
        "asset_class": "static_object",
        "instance_id": decision["instance_id"],
        "request_sha256": decision["request_sha256"],
        "profile_sha256": decision["profile_sha256"],
        "input": {
            "path": str(input_glb),
            "sha256": contract.sha256_file(input_glb),
            "size_bytes": input_glb.stat().st_size,
        },
        "output": {
            "path": str(output),
            "sha256": contract.sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
        "coordinate_system": contract.COORDINATE_SYSTEM,
        "heading": {
            "passed": True,
            "reviewed_source_front_yaw_deg": source_yaw,
            "target_front_axis": "positive-x",
            "applied_world_z_yaw_deg": delta_yaw,
            "evidence": {
                "path": str(heading_path),
                "sha256": contract.sha256_file(heading_path),
                "size_bytes": heading_path.stat().st_size,
            },
        },
        "physical_scale": {
            "passed": True,
            "measurement": "height_m",
            "height_before_m": height_before,
            "target_height_m": target_height_m,
            "tolerance_m": tolerance_m,
            "uniform_scale": uniform_scale,
            "readback_height_m": readback_height,
            "absolute_error_m": height_error,
        },
        "grounding": {
            "passed": True,
            "method": "mesh_minimum_up_to_asset_root_zero_v1",
            "minimum_up_before_translation_m": float(scaled_minimum.z),
            "minimum_up_after_export_readback_m": float(readback_minimum.z),
            "tolerance_m": GROUND_TOLERANCE_M,
        },
        "scene_readback": {
            "before": before_summary,
            "after": after_summary,
            "bounds_minimum_blender_xyz_m": list(readback_minimum),
            "bounds_maximum_blender_xyz_m": list(readback_maximum),
            "protected_scene_counts_preserved": True,
            "no_rig_or_animation": True,
        },
        "formal_dataset_registration_authorized": False,
    }
    write_json_exclusive(manifest_path, payload)
    print(
        "GENERATED_STATIC_FINALIZATION_OK "
        f"height_m={readback_height:.6f} ground_m={readback_minimum.z:.8f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
